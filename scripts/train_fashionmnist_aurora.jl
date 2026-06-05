#=
scripts/train_fashionmnist_aurora.jl

Aurora-specific copy of train_fashionmnist.jl that routes GPU work through
the oneAPI backend instead of CUDA. Run from repository root like:

  julia --project=. scripts/train_fashionmnist_aurora.jl --lr 0.001 \
        --epochs 5 --optimizer rmsprop --batchsize 128 --use_gpu true

Changes vs scripts/train_fashionmnist.jl:
 - `using oneAPI` (triggers PhasorNetworksOneAPIExt: select_device(:oneapi)
   and NNlib._batched_gemm! for oneArray).
 - `--use_cuda` CLI flag renamed to `--use_gpu` — the semantics are still
   "put model + data on the device", but the device is whichever the
   project selects via `Args(backend=:oneapi)` on this script.
 - GPU detection / params-move guard switched from `CUDA.functional()`
   to `on_gpu(...)` (backend-agnostic via AbstractGPUArray).
 - `cpuify` recursively unwraps any `AbstractGPUArray`, not just CuArray.
=#

using Pkg
function find_repo_root(start_dir::String = pwd())
    dir = start_dir
    while !(isfile(joinpath(dir, "Project.toml")) && isdir(joinpath(dir, ".git")))
        parent = dirname(dir)
        if parent == dir
            error("Repository root not found from $(start_dir)")
        end
        dir = parent
    end
    return dir
end

repo_root = find_repo_root(@__DIR__)
cd(repo_root)
Pkg.activate(repo_root)

# Load PhasorNetworks as the registered package (NOT via `include` +
# `using .PhasorNetworks`). Package extensions like
# `PhasorNetworksOneAPIExt` are keyed to the registered package UUID;
# `include`-loaded modules do not trigger them, so the extension's
# `select_device(::Val{:oneapi})` method never registers and we hit
# `backend.jl:49`'s fallback error.
using PhasorNetworks

using Lux, MLUtils, OneHotArrays, Statistics, Random, Zygote, Optimisers, ComponentArrays
using GPUArraysCore: AbstractGPUArray
using ArgParse, JLD2, Dates
using Random: Xoshiro

# Load oneAPI — triggers PhasorNetworksOneAPIExt which registers
# select_device(:oneapi) and the NNlib batched-gemm hook for oneArray.
# Must come before `Args(backend=:oneapi)` to avoid the
# "oneAPI backend requires `using oneAPI`" error from select_device's
# fallback method.
using oneAPI

function build_parser()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--lr"
            help = "Learning rate"
            arg_type = Float64
            default = 0.001
        "--epochs"
            help = "Number of training epochs"
            arg_type = Int
            default = 5
        "--batchsize"
            help = "Batch size"
            arg_type = Int
            default = 128
        "--seed"
            help = "RNG seed"
            arg_type = Int
            default = 42
        "--optimizer"
            help = "Optimizer name (rmsprop, adam, sgd)"
            default = "rmsprop"
        "--use_gpu"
            help = "Use oneAPI GPU if available"
            arg_type = Bool
            default = true
        "--out"
            help = "Output directory to save run artifacts"
            default = "runs"
    end
    return s
end

parsed = parse_args(build_parser())
lr = parsed["lr"]
epochs = parsed["epochs"]
seed = parsed["seed"]
batchsize = parsed["batchsize"]
optimizer_name = lowercase(parsed["optimizer"])
use_gpu = parsed["use_gpu"]
outdir = parsed["out"]

# Pick backend: :oneapi if requested AND oneAPI hardware is present;
# otherwise :cpu. The functional check matches what select_device's :cuda
# branch does (warn-and-fall-back), but applied to oneAPI here.
backend = if use_gpu
    if oneAPI.functional()
        :oneapi
    else
        @warn "oneAPI requested but not functional, falling back to CPU"
        :cpu
    end
else
    :cpu
end

args = Args(batchsize = batchsize,
            epochs = epochs,
            lr = lr,
            rng = Xoshiro(seed),
            backend = backend)

println("Settings: lr=$(lr), epochs=$(epochs), batchsize=$(batchsize), optimizer=$(optimizer_name), backend=$(backend), out=$(outdir)")

cdev = cpu_device()
gdev = select_device(backend)   # routes through PhasorNetworksOneAPIExt for :oneapi
dev = backend == :cpu ? cdev : gdev

function get_optimizer(name::String, lr::Float64)
    name = lowercase(name)
    if name in ("rmsprop", "rmsprop()")
        return Optimisers.RMSProp
    elseif name in ("adam", "adam()")
        return Optimisers.Adam
    elseif name in ("sgd", "sgd()")
        return Optimisers.SGD
    else
        error("Unknown optimizer: $name. Supported: rmsprop, adam, sgd")
    end
end

optimiser = get_optimizer(optimizer_name, lr)

println("Loading FashionMNIST...")
train_data = fashion_mnist_data(:train)
test_data = fashion_mnist_data(:test)
train_loader = DataLoader(train_data, batchsize=batchsize)
test_loader = DataLoader(test_data, batchsize=batchsize)

construct_model = n -> Chain(FlattenLayer(),
                        LayerNorm((28^2,)),
                        Dense(28^2 => n, relu),
                        Dense(n => 10),
                        softmax)

function loss_function(x, y, model, ps, st, dev=dev)
    x = x |> dev
    y = y |> dev
    y_pred, _ = Lux.apply(model, x, ps, st)
    y_onehot = onehotbatch(y, 0:9)
    return CrossEntropyLoss(;logits=false, dims=1)(y_pred, y_onehot)
end

function test(model, data_loader, ps, st, dev=dev)
    total_correct = 0
    total_samples = 0
    for (x, y) in data_loader
        x = x |> dev
        y_pred, _ = Lux.apply(model, x, ps, st)
        pred_labels = onecold(cdev(y_pred))
        total_correct += sum(pred_labels .== y .+ 1)
        total_samples += length(y)
    end
    return total_correct / total_samples
end

function run_training(model, ps, st, train_loader, loss_fn, args)
    if isdefined(Main, :train)
        return train(model, ps, st, train_loader, loss_fn, args; optimiser=optimiser)
    else
        error("No train function found in environment. Please use the project's train implementation.")
    end
end

println("\n=== Conventional network ===")
model = construct_model(128)
ps, st = Lux.setup(args.rng, model)
ps = ps |> dev
st = st |> dev

println("Initial loss (first batch): ", loss_function(first(train_loader)..., model, ps, st))

losses, pst, stt = run_training(model, ps, st, train_loader, loss_function, args)
println("Conventional final loss: ", losses[end])
acc = test(model, test_loader, pst, stt)
println("Conventional test accuracy: ", acc)

println("\n=== Phasor network ===")
import PhasorNetworks: default_bias, Codebook
p_model = Chain(FlattenLayer(),
                LayerNorm((28^2,)),
                x -> Phase.(tanh.(x)),
                x -> x,
                PhasorDense(28^2 => 128, normalize_to_unit_circle, init_bias=default_bias),
                PhasorDense(128 => 16, normalize_to_unit_circle, init_bias=default_bias),
                Codebook(16 => 10))

psp, stp = Lux.setup(args.rng, p_model)

# Move phasor model to GPU if we have one. Backend-agnostic: gdev is a
# oneAPIDevice (or CUDADevice, or cpu_device) depending on what select_device
# returned. The original CUDA-only check `use_cuda && CUDA.functional()`
# wouldn't move anything on Aurora.
if backend != :cpu
    psp = psp |> gdev
    stp = stp |> gdev
end

function phasor_loss_function(x, y, model, ps, st, dev=dev)
    x = x |> dev
    y = y |> dev
    y_pred, _ = Lux.apply(model, x, ps, st)
    y_onehot = onehotbatch(y, 0:9)
    loss = evaluate_loss(y_pred, y_onehot, :similarity)
    loss = mean(loss)
    return loss
end

function test_phasor(model, data_loader, ps, st, dev=dev)
    total_correct = 0
    total_samples = 0
    for (x, y) in data_loader
        x = x |> dev

        y_pred, _ = Lux.apply(model, x, ps, st)
        pred_labels = predict(cdev(y_pred), :similarity)

        total_correct += sum(pred_labels .== y .+ 1)
        total_samples += length(y)
    end

    acc = total_correct / total_samples
end

println("Initial phasor loss (first batch): ", phasor_loss_function(first(train_loader)..., p_model, psp, stp))

losses_f, ps_train_f, st_train_f = run_training(p_model, psp, stp, train_loader, phasor_loss_function, args)
println("Phasor final loss: ", losses_f[end])
acc_p = test_phasor(p_model, test_loader, ps_train_f, st_train_f)
println("Phasor test accuracy (codebook prediction): ", acc_p)

println("\nDone.")


# --- Save run artifacts (loss histories, trained params, args) using JLD2

# Backend-agnostic: AbstractGPUArray covers CuArray, oneArray, and any
# other GPUArrays-rooted device array. Replaces the CUDA.CuArray check
# in the original script.
function cpuify(x)
    if isa(x, AbstractGPUArray)
        return Array(x)
    elseif isa(x, AbstractArray)
        return map(cpuify, x)
    elseif isa(x, Dict)
        out = Dict()
        for (k,v) in x
            out[k] = cpuify(v)
        end
        return out
    elseif isa(x, NamedTuple)
        return NamedTuple{keys(x)}(cpuify.(values(x)))
    else
        return x
    end
end

function save_run(outdir, info)
    if !isdir(outdir)
        mkpath(outdir)
    end
    ts = Dates.format(now(), "yyyy-mm-dd_HHMMSS")
    fname = joinpath(outdir, "fmnist_aurora_run_$(ts).jld2")
    @info "Saving run to $fname"
    @save fname info
    return fname
end

info = Dict(
    :args => Dict(:lr=>lr, :epochs=>epochs, :batchsize=>batchsize, :optimizer=>optimizer_name, :backend=>backend, :seed=>seed),
    :conventional => Dict(:losses => cpuify(losses), :params => cpuify(pst), :state => cpuify(stt), :test_accuracy => acc),
    :phasor => Dict(:losses => cpuify(losses_f), :params => cpuify(ps_train_f), :state => cpuify(st_train_f), :test_accuracy => acc_p)
)

save_path = save_run(outdir, info)
println("Saved run artifacts to: ", save_path)
