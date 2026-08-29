#!/usr/bin/env julia
#
# scripts/ep_backprop_transfer.jl — A3: Does a backprop-trained chain survive EP settle?
#
# Train a 784→256→64 PhasorDense chain with `train()` (feedforward, Zygote),
# then evaluate it with `ep_predict` (settle-based). Report the accuracy delta.
#
# Usage:
#   julia --project=. scripts/ep_backprop_transfer.jl
#   julia --project=. -t 4 scripts/ep_backprop_transfer.jl

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

using PhasorNetworks, Lux, MLUtils, OneHotArrays, Statistics, Random, Zygote, Optimisers, CUDA
using Dates, LinearAlgebra
using Random: Xoshiro

const BATCHSIZE = 128
const EPOCHS = 5
const LR = 0.001
const SEED = 42
const USE_CUDA = CUDA.functional()
const HID = 256
const DOUT = 64
const SCALE = 0.4f0

cdev = cpu_device()
gdev = gpu_device()
dev = USE_CUDA ? gdev : cdev

args = Args(batchsize = BATCHSIZE,
            epochs = EPOCHS,
            lr = LR,
            rng = Xoshiro(SEED),
            use_cuda = USE_CUDA)

println("Settings: lr=$LR, epochs=$EPOCHS, batchsize=$BATCHSIZE, use_cuda=$USE_CUDA")

# ---- encode_phase (matching ep_fashionmnist.jl) ----
function encode_phase(imgs::AbstractArray{Float32,3})
    N = size(imgs, 3)
    flat = reshape(imgs, :, N)                       # (784, N)
    μ = mean(flat; dims=1)
    σ = std(flat; dims=1) .+ 1f-6
    return Phase.(0.5f0 .* tanh.((flat .- μ) ./ σ))
end

# Load FashionMNIST
println("Loading FashionMNIST...")
tr = fashion_mnist_data(:train)
te = fashion_mnist_data(:test)
ntr = min(60000, length(tr.targets))
nte = min(10000, length(te.targets))
Xtr = encode_phase(Float32.(tr.features[:, :, 1:ntr]))
Xte = encode_phase(Float32.(te.features[:, :, 1:nte]))
ytr = Int.(tr.targets[1:ntr]) .+ 1
yte = Int.(te.targets[1:nte]) .+ 1

# DataLoaders for backprop training
train_loader = DataLoader((Xtr, ytr), batchsize=BATCHSIZE, shuffle=true)
test_loader = DataLoader((Xte, yte), batchsize=BATCHSIZE, shuffle=false)

# ---- Build EP chain (PhasorDense only) ----
import PhasorNetworks: default_bias, normalize_to_unit_circle

function build_chain(rng)
    chain = Chain(
        PhasorDense(784 => HID, normalize_to_unit_circle, use_bias=true),
        PhasorDense(HID => DOUT, normalize_to_unit_circle, use_bias=true)
    )
    ps, st = Lux.setup(rng, chain)
    ps = (layer_1 = merge(ps.layer_1, (weight = SCALE .* ps.layer_1.weight,)),
          layer_2 = merge(ps.layer_2, (weight = SCALE .* ps.layer_2.weight,)))
    return chain, ps, st
end

function make_codes(rng)
    return ComplexF32.(angle_to_complex(orthogonal_codes(rng, DOUT, 10)))
end

rng = Xoshiro(SEED)
chain, ps, st = build_chain(rng)
cb_codes = make_codes(rng)

if USE_CUDA
    ps = ps |> gdev
    st = st |> gdev
    cb_codes = cb_codes |> gdev
end

# Loss function for backprop training (feedforward)
function bp_loss(x, y, model, ps, st, codebook, dev=dev)
    x = x |> dev
    y = y |> dev
    z_out, _ = Lux.apply(model, x, ps, st)  # (64, B) Phase
    # Convert to Complex for similarity
    z_out_c = ComplexF32.(angle_to_complex(z_out))
    codes_dev = codebook |> dev
    logits = similarity_outer(z_out_c, codes_dev)  # (10, B)
    y_onehot = onehotbatch(y .- 1, 0:9)
    # Manual cross-entropy for logits (no softmax)
    log_probs = logits .- log.(sum(exp.(logits); dims=1))
    loss = -mean(sum(y_onehot .* log_probs; dims=1))
    return loss
end

# Train with backprop
println("\n=== Training with backprop (Zygote) ===")
initial_loss = bp_loss(first(train_loader)..., chain, ps, st, cb_codes, dev)
println("Initial loss: $initial_loss")

function train_bp(chain, ps, st, train_loader, args, codebook, dev; epochs=EPOCHS, lr=LR)
    optimiser = Optimisers.Adam(lr)
    opt_state = Optimisers.setup(optimiser, ps)
    losses = Float64[]
    for epoch in 1:epochs
        epoch_losses = Float64[]
        for (x, y) in train_loader
            x = x |> dev
            y = y |> dev
            lf = p -> bp_loss(x, y, chain, p, st, codebook, dev)
            lossval, gs = withgradient(lf, ps)
            push!(epoch_losses, lossval)
            if args.weight_decay > 0
                PhasorNetworks._apply_weight_decay(gs[1], ps, args.weight_decay)
            end
            opt_state, ps = Optimisers.update(opt_state, ps, gs[1])
        end
        push!(losses, mean(epoch_losses))
        println("Epoch $epoch: loss = $(losses[end])")
    end
    return ps, losses
end

ps, losses = train_bp(chain, ps, st, train_loader, args, cb_codes, dev)

# --- Evaluate with feedforward (backprop) ---
println("\n=== Evaluating with feedforward (backprop) ===")
function bp_accuracy(model, data_loader, ps, st, codebook, dev=dev)
    total_correct = 0
    total_samples = 0
    for (x, y) in data_loader
        x = x |> dev
        y = y |> dev
        z_out, _ = Lux.apply(model, x, ps, st)
        z_out_c = ComplexF32.(angle_to_complex(z_out))
        logits = similarity_outer(z_out_c, codebook |> dev)
        # Move logits to CPU for argmax to avoid scalar indexing issues on GPU
        logits_cpu = logits |> cdev
        pred_indices = argmax(logits_cpu; dims=1)
        # pred_indices is (1, B) matrix, need to flatten to get 1D vector
        pred_labels = [idx[1] for idx in vec(pred_indices)]
        y_cpu = y |> cdev
        batch_correct = sum(pred_labels .== y_cpu)
        total_correct += batch_correct
        total_samples += length(y_cpu)
    end
    return total_correct / total_samples
end

bp_acc = bp_accuracy(chain, test_loader, ps, st, cb_codes, dev)
println("Backprop (feedforward) test accuracy: $bp_acc")

# --- Evaluate with EP settle ---
println("\n=== Evaluating with EP settle (ep_predict) ===")

function ep_accuracy(model, X, y, ps, st, codebook, dev=dev; T=200, dt=0.5f0, K_mode=:zero)
    correct = 0
    for i in 1:512:length(y)
        e = min(i + 512 - 1, length(y))
        x_batch = X[:, i:e] |> dev
        logits = ep_predict(model, ps, st, x_batch, codebook; T=T, dt=dt, K_mode=K_mode)
        pred = [argmax(view(logits, :, b))[1] for b in 1:(e - i + 1)]
        correct += sum(pred .== y[i:e])
    end
    return correct / length(y)
end

ep_acc = ep_accuracy(chain, Xte, yte, ps, st, cb_codes, dev)
println("EP settle test accuracy: $ep_acc")

# --- Summary ---
println("\n=== Summary ===")
println("Backprop (feedforward) accuracy: $bp_acc")
println("EP settle accuracy:             $ep_acc")
println("Difference (EP - BP):           $(ep_acc - bp_acc)")
println("Relative drop:                  $(100 * (bp_acc - ep_acc) / bp_acc)%")

# Also test gradient fidelity
println("\n=== Gradient fidelity check (LockinEP vs centered StaticEP) ===")
# Use CPU to avoid scalar indexing issues on GPU
x_test = Xte[:, 1:1] |> cdev
y_test = yte[1:1] |> cdev
y_target = cb_codes[:, y_test[1]] |> cdev

# Move params and state to CPU for this check
ps_cpu = ps |> cdev
st_cpu = st |> cdev

using PhasorNetworks: SimilarityCost, LockinEP, StaticEP, ep_gradient

# Local cosine function
cosine(a, b) = real(dot(vec(a), vec(b)) / (norm(vec(a)) * norm(vec(b)) + 1e-30))

cost = SimilarityCost(y_target)

# Centered StaticEP as oracle
static_method = StaticEP(β=0.005f0, T_free=200, T_nudge=100, dt=0.5f0, centered=true)
g_static, _ = ep_gradient(static_method, chain, ps_cpu, st_cpu, x_test, cost)

# LockinEP
lockin_method = LockinEP(ε=0.05f0, ω_p=0.05f0, n_cycles=4, T_warmup_cycles=2, T_free=100, dt=0.1f0)
g_lockin, _ = ep_gradient(lockin_method, chain, ps_cpu, st_cpu, x_test, cost)

for key in (:layer_1, :layer_2)
    if haskey(g_static, key) && haskey(g_lockin, key)
        g_s = g_static[key].weight
        g_l = g_lockin[key].weight
        re = norm(g_l - g_s) / norm(g_s)
        cs = cosine(g_l, g_s)
        println("$key: cos=$cs, rel-err=$re")
    end
end

println("\nDone.")