#=
scripts/ssm_hyperparam.jl

Hyperparameter exploration script for PhasorSSM models on FashionMNIST.
Designed to be called by an external optimizer (e.g. Optuna via PyJulia):

    accuracy = run_trial(; lr=1e-3, epochs=10, activation=:hard, readout_frac=0.25)

Or from the command line for a single trial:

    julia --project=. scripts/ssm_hyperparam.jl \
        --lr 0.001 --epochs 10 --activation hard \
        --r_lo 0.1 --r_hi 0.6 --readout_frac 0.25

Hyperparameters explored:
  - lr:            Learning rate
  - epochs:        Number of training epochs
  - activation:    "hard" (normalize_to_unit_circle) or "soft" (soft_normalize_to_unit_circle)
  - r_lo, r_hi:    Soft activation thresholds (only used when activation=soft)
  - readout_frac:  Fraction of final time steps averaged by SSMReadout

Fixed choices (per user spec):
  - init = :uniform (SSM weight initializer)
  - encoding = PSK (constant phase-encoded inputs, not impulse)
=#

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using PhasorNetworks
using Lux, Random, Optimisers, Zygote, Statistics
using MLDatasets, MLUtils, OneHotArrays
using CUDA, LuxCUDA
using ArgParse

# ================================================================
# Data Loading (cached so repeated trials don't reload)
# ================================================================

const DATA_CACHE = Ref{Any}(nothing)

function get_data(; batchsize::Int=128)
    if DATA_CACHE[] === nothing
        println("Loading FashionMNIST...")
        train_data = FashionMNIST(split=:train)
        test_data  = FashionMNIST(split=:test)
        x_train = Float32.(train_data.features)                   # 28 × 28 × 60000
        y_train = Float32.(onehotbatch(train_data.targets, 0:9))  # 10 × 60000
        x_test  = Float32.(test_data.features)
        y_test  = Float32.(onehotbatch(test_data.targets, 0:9))
        DATA_CACHE[] = (x_train, y_train, x_test, y_test)
    end
    x_train, y_train, x_test, y_test = DATA_CACHE[]
    train_loader = DataLoader((x_train, y_train); batchsize, shuffle=true)
    test_loader  = DataLoader((x_test,  y_test);  batchsize)
    return train_loader, test_loader
end

# ================================================================
# Model Construction
# ================================================================

function build_activation(mode::Symbol; r_lo::Float32=0.1f0, r_hi::Float32=0.6f0)
    if mode == :hard
        return normalize_to_unit_circle
    elseif mode == :soft
        return z -> soft_normalize_to_unit_circle(z; r_lo, r_hi)
    else
        error("Unknown activation mode: $mode. Use :hard or :soft.")
    end
end

function create_ssm_model(; D_hidden::Int=128, n_classes::Int=10, C_in::Int=28,
                            activation::Symbol=:hard,
                            r_lo::Float32=0.1f0, r_hi::Float32=0.6f0,
                            readout_frac::Float32=0.25f0)
    act = build_activation(activation; r_lo, r_hi)
    model = Chain(
        PhasorSSM(C_in => D_hidden, act; init=:uniform),
        PhasorSSM(D_hidden => D_hidden, identity; init=:uniform),
        SSMReadout(D_hidden => n_classes; readout_frac),
    )
    return model
end

# ================================================================
# Loss & Evaluation
# ================================================================

function ssm_loss(x, y, model, ps, st)
    x_enc = psk_encode(x)
    sims, _ = model(x_enc, ps, st)
    return mean(similarity_loss(sims, y))
end

function evaluate(model, ps, st, loader, device)
    correct = 0
    total = 0
    for (x, y) in loader
        x_dev = x |> device
        y_dev = y |> device
        x_enc = psk_encode(x_dev)
        sims, _ = model(x_enc, ps, st)
        preds = argmax(sims; dims=1)
        truth = argmax(y_dev; dims=1)
        correct += sum(preds .== truth)
        total += size(x, 3)
    end
    return correct / total
end

# ================================================================
# Single Trial
# ================================================================

"""
    run_trial(; kwargs...) -> Float64

Train a PhasorSSM model with the given hyperparameters and return test accuracy.
This is the entry point for external optimizers (e.g. Optuna).

# Keyword Arguments
- `lr::Float64=3e-4`           — Learning rate
- `epochs::Int=10`             — Number of training epochs
- `activation::Symbol=:hard`   — `:hard` or `:soft`
- `r_lo::Float32=0.1f0`        — Soft activation lower threshold (ignored if hard)
- `r_hi::Float32=0.6f0`        — Soft activation upper threshold (ignored if hard)
- `readout_frac::Float32=0.25f0` — Fraction of final time steps for SSMReadout
- `hidden::Int=128`            — Hidden dimension
- `batchsize::Int=128`         — Training batch size
- `seed::Int=42`               — Random seed
- `use_cuda::Bool=true`        — Use GPU if available
- `verbose::Bool=false`        — Print per-epoch progress
"""
function run_trial(; lr::Float64=3e-4,
                     epochs::Int=10,
                     activation::Symbol=:hard,
                     r_lo::Float32=0.1f0,
                     r_hi::Float32=0.6f0,
                     readout_frac::Float32=0.25f0,
                     hidden::Int=128,
                     batchsize::Int=128,
                     seed::Int=42,
                     use_cuda::Bool=true,
                     verbose::Bool=false)

    rng = Xoshiro(seed)

    train_loader, test_loader = get_data(; batchsize)

    model = create_ssm_model(; D_hidden=hidden, activation, r_lo, r_hi, readout_frac)
    ps, st = Lux.setup(rng, model)

    cuda_available = use_cuda && CUDA.functional()
    device = cuda_available ? gpu_device() : cpu_device()
    ps = ps |> device
    st = st |> device

    args = Args(epochs=1, batchsize=batchsize, lr=lr, use_cuda=cuda_available)

    for epoch in 1:epochs
        losses, ps, st = train(model, ps, st, train_loader, ssm_loss, args)
        if verbose
            avg_loss = mean(losses)
            acc = evaluate(model, ps, st, test_loader, device)
            println("  Epoch $epoch/$epochs  loss=$(round(avg_loss; digits=4))  acc=$(round(acc; digits=4))")
        end
    end

    acc = evaluate(model, ps, st, test_loader, device)
    if verbose
        println("Final test accuracy: $(round(acc; digits=4))")
    end
    return Float64(acc)
end

# ================================================================
# CLI
# ================================================================

function parse_cli()
    s = ArgParseSettings(description="PhasorSSM hyperparameter trial")
    @add_arg_table! s begin
        "--lr"
            help = "learning rate"
            arg_type = Float64
            default = 3e-4
        "--epochs"
            help = "number of training epochs"
            arg_type = Int
            default = 10
        "--activation"
            help = "activation mode: hard or soft"
            arg_type = String
            default = "hard"
        "--r_lo"
            help = "soft activation lower threshold"
            arg_type = Float64
            default = 0.1
        "--r_hi"
            help = "soft activation upper threshold"
            arg_type = Float64
            default = 0.6
        "--readout_frac"
            help = "fraction of final time steps for SSMReadout"
            arg_type = Float64
            default = 0.25
        "--hidden"
            help = "hidden dimension"
            arg_type = Int
            default = 128
        "--batchsize"
            help = "training batch size"
            arg_type = Int
            default = 128
        "--seed"
            help = "random seed"
            arg_type = Int
            default = 42
        "--no-cuda"
            help = "disable CUDA"
            action = :store_true
        "--verbose"
            help = "print per-epoch progress"
            action = :store_true
    end
    return ArgParse.parse_args(s)
end

function main()
    parsed = parse_cli()

    acc = run_trial(
        lr           = parsed["lr"],
        epochs       = parsed["epochs"],
        activation   = Symbol(parsed["activation"]),
        r_lo         = Float32(parsed["r_lo"]),
        r_hi         = Float32(parsed["r_hi"]),
        readout_frac = Float32(parsed["readout_frac"]),
        hidden       = parsed["hidden"],
        batchsize    = parsed["batchsize"],
        seed         = parsed["seed"],
        use_cuda     = !parsed["no-cuda"],
        verbose      = parsed["verbose"],
    )

    println("ACCURACY=$(acc)")
    return acc
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
