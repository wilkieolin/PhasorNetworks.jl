#!/usr/bin/env julia
# scripts/e2_impaired_only_pilot.jl — Quick impaired-accuracy sweep to find severity ranges
# Measures ONLY impaired accuracy (fast ep_predict) to find clean->chance transition
# No fine-tuning, so runs in minutes not hours

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
Pkg.activate(joinpath(repo_root, "scripts"))

using PhasorNetworks, Lux, MLUtils, OneHotArrays, Statistics, Random, Optimisers, LinearAlgebra, CSV, DataFrames, Printf, Zygote
using Random: Xoshiro

# Force CPU
const USE_CUDA = false
const DEVICE = cpu_device()

# ============================================================
# CONFIGURATION
# ============================================================
const OUT = joinpath(repo_root, "results", "ep_analog_finetune")
mkpath(OUT)

const N_TRAIN = 60000
const N_TEST  = 10000
const BATCH   = 128
const HID     = 256
const DOUT    = 64
const SEED    = 0x42

const BP_EPOCHS     = 5
const BP_LR         = 0.001

# Provenance
const GITREV = try
    rev = strip(read(`git -C $(repo_root) rev-parse --short HEAD`, String))
    d   = read(`git -C $(repo_root) diff HEAD -- src`, String)
    isempty(strip(d)) ? rev : rev * "-d" * string(hash(d), base = 16)[1:8]
catch
    "unknown"
end

# ============================================================
# WIDE IMPAIRMENT RANGES FOR CALIBRATION
# ============================================================
const CALIBRATION_IMPAIRMENTS = [
    (:lognormal,   [0.01f0, 0.03f0, 0.1f0, 0.3f0, 0.5f0, 0.7f0, 1.0f0, 1.5f0, 2.0f0]),
    (:gaussian,    [0.01f0, 0.03f0, 0.1f0, 0.3f0, 0.5f0, 0.7f0, 1.0f0]),
    (:stuck_zero,  [0.01f0, 0.03f0, 0.1f0, 0.3f0, 0.5f0, 0.7f0, 0.9f0]),
    (:stuck_sat,   [0.01f0, 0.03f0, 0.1f0, 0.3f0, 0.5f0, 0.7f0]),
]

const REPS = 3

# ============================================================
# HELPERS
# ============================================================

function build_chain(rng::Xoshiro; scale=0.4f0)
    chain = Chain(
        PhasorDense(784 => HID, normalize_to_unit_circle, use_bias=true),
        PhasorDense(HID => DOUT, normalize_to_unit_circle, use_bias=true)
    )
    ps, st = Lux.setup(rng, chain)
    ps = (layer_1 = merge(ps.layer_1, (weight = scale .* ps.layer_1.weight,)),
          layer_2 = merge(ps.layer_2, (weight = scale .* ps.layer_2.weight,)))
    return chain, ps, st
end

function make_codes(rng::Xoshiro)
    return ComplexF32.(angle_to_complex(orthogonal_codes(rng, DOUT, 10)))
end

function encode_phase(imgs::AbstractArray{Float32,3})
    N = size(imgs, 3)
    flat = reshape(imgs, :, N)
    μ = mean(flat; dims=1)
    σ = std(flat; dims=1) .+ 1f-6
    return Phase.(0.5f0 .* tanh.((flat .- μ) ./ σ))
end

function load_fashionmnist()
    tr = fashion_mnist_data(:train)
    te = fashion_mnist_data(:test)
    ntr = min(N_TRAIN, length(tr.targets))
    nte = min(N_TEST,  length(te.targets))
    Xtr = encode_phase(Float32.(tr.features[:, :, 1:ntr]))
    Xte = encode_phase(Float32.(te.features[:, :, 1:nte]))
    ytr = Int.(tr.targets[1:ntr]) .+ 1
    yte = Int.(te.targets[1:nte]) .+ 1
    return (Xtr, ytr), (Xte, yte)
end

function bp_loss(x, y, model, ps, st, codebook)
    z_out, _ = Lux.apply(model, x, ps, st)
    z_out_c = ComplexF32.(angle_to_complex(z_out))
    logits = similarity_outer(z_out_c, codebook)
    y_onehot = onehotbatch(y .- 1, 0:9)
    log_probs = logits .- log.(sum(exp.(logits); dims=1))
    return -mean(sum(y_onehot .* log_probs; dims=1))
end

function train_bp(chain, ps, st, train_loader, epochs, lr, codebook)
    optimiser = Optimisers.Adam(lr)
    opt_state = Optimisers.setup(optimiser, ps)
    for epoch in 1:epochs
        epoch_losses = Float64[]
        for (x, y) in train_loader
            lf = p -> bp_loss(x, y, chain, p, st, codebook)
            lossval, gs = Zygote.withgradient(lf, ps)
            push!(epoch_losses, lossval)
            opt_state, ps = Optimisers.update(opt_state, ps, gs[1])
        end
        @printf("  BP epoch %d/%d: loss = %.4f\n", epoch, epochs, mean(epoch_losses))
    end
    return ps
end

function ep_accuracy(model, X, y, ps, st, codebook; batch=512, T=200, dt=0.5f0, K_mode=:zero)
    correct = 0
    for i in 1:batch:length(y)
        e = min(i + batch - 1, length(y))
        x_batch = X[:, i:e]
        logits = ep_predict(model, ps, st, x_batch, codebook; T=T, dt=dt, K_mode=K_mode)
        pred = [argmax(view(logits, :, b))[1] for b in 1:(e - i + 1)]
        correct += sum(pred .== y[i:e])
    end
    return correct / length(y)
end

# --- Impairment functions ---
function apply_lognormal(ps, σ, rng)
    W1 = ps.layer_1.weight .* (1f0 .+ exp.(σ .* randn(rng, Float32, size(ps.layer_1.weight))) .- 1f0)
    W2 = ps.layer_2.weight .* (1f0 .+ exp.(σ .* randn(rng, Float32, size(ps.layer_2.weight))) .- 1f0)
    return merge(ps, (layer_1 = merge(ps.layer_1, (weight = W1,)),
                        layer_2 = merge(ps.layer_2, (weight = W2,)))), nothing
end

function apply_gaussian(ps, σ, rng)
    W1 = ps.layer_1.weight .+ σ .* randn(rng, Float32, size(ps.layer_1.weight))
    W2 = ps.layer_2.weight .+ σ .* randn(rng, Float32, size(ps.layer_2.weight))
    return merge(ps, (layer_1 = merge(ps.layer_1, (weight = W1,)),
                        layer_2 = merge(ps.layer_2, (weight = W2,)))), nothing
end

function apply_stuck_zero(ps, frac, rng)
    mask1 = rand(rng, Float32, size(ps.layer_1.weight)) .>= frac
    mask2 = rand(rng, Float32, size(ps.layer_2.weight)) .>= frac
    W1 = ps.layer_1.weight .* Float32.(mask1)
    W2 = ps.layer_2.weight .* Float32.(mask2)
    return merge(ps, (layer_1 = merge(ps.layer_1, (weight = W1,)),
                        layer_2 = merge(ps.layer_2, (weight = W2,)))), nothing
end

function apply_stuck_sat(ps, frac, rng)
    max1 = maximum(abs.(ps.layer_1.weight))
    max2 = maximum(abs.(ps.layer_2.weight))
    mask1 = rand(rng, Float32, size(ps.layer_1.weight)) .>= frac
    mask2 = rand(rng, Float32, size(ps.layer_2.weight)) .>= frac
    signs1 = rand(rng, [1f0, -1f0], size(ps.layer_1.weight))
    signs2 = rand(rng, [1f0, -1f0], size(ps.layer_2.weight))
    W1 = ps.layer_1.weight .* Float32.(mask1) .+ (1f0 .- Float32.(mask1)) .* signs1 .* max1
    W2 = ps.layer_2.weight .* Float32.(mask2) .+ (1f0 .- Float32.(mask2)) .* signs2 .* max2
    return merge(ps, (layer_1 = merge(ps.layer_1, (weight = W1,)),
                        layer_2 = merge(ps.layer_2, (weight = W2,)))), nothing
end

# ============================================================
# MAIN
# ============================================================

println("=== E2 Impaired-Only Calibration Pilot ===")
@printf("HID=%d, DOUT=%d, REPS=%d, USE_CUDA=%s\n", HID, DOUT, REPS, USE_CUDA)
println("Output: $OUT")

# Load data
println("\nLoading FashionMNIST...")
(Xtr, ytr), (Xte, yte) = load_fashionmnist()
println("Train: $(size(Xtr,2)), Test: $(size(Xte,2))")

# Build chain
rng = Xoshiro(SEED)
chain, ps_init, st = build_chain(rng)
cb_codes = make_codes(rng)

train_loader = DataLoader((Xtr, ytr); batchsize=BATCH, shuffle=true)

# CSV output for pilot
csv_file = joinpath(OUT, "e2_impaired_only_pilot_$(GITREV).csv")
isfile(csv_file) || CSV.write(csv_file, DataFrame(
    gitrev=String[],
    pretrain=String[],
    impairment_type=String[],
    param=Float32[],
    rep=Int[],
    acc_clean=Float32[],
    acc_impaired=Float32[],
    width=Int[],
    depth=Int[]
))

# Pretrain: Backprop only
println("\n=== Pretraining: Backprop (5 epochs) ===")
ps_bp = train_bp(chain, ps_init, st, train_loader, BP_EPOCHS, BP_LR, cb_codes)
acc_clean = ep_accuracy(chain, Xte, yte, ps_bp, st, cb_codes)
println("Clean accuracy (ep_predict): $acc_clean")

# ============================================================
# IMPAIRED-ONLY SWEEP
# ============================================================

for (imp_type, params) in CALIBRATION_IMPAIRMENTS
    println("\n=== Impairment: $imp_type ===")
    for param in params
        @printf("  param = %.3f\n", param)
        for rep in 1:REPS
            rng_rep = Xoshiro(hash((imp_type, param, rep)) % UInt64)
            
            # Apply weight impairment
            if imp_type == :lognormal
                ps_imp, _ = apply_lognormal(ps_bp, param, rng_rep)
            elseif imp_type == :gaussian
                ps_imp, _ = apply_gaussian(ps_bp, param, rng_rep)
            elseif imp_type == :stuck_zero
                ps_imp, _ = apply_stuck_zero(ps_bp, param, rng_rep)
            elseif imp_type == :stuck_sat
                ps_imp, _ = apply_stuck_sat(ps_bp, param, rng_rep)
            end
            
            # Evaluate impaired (FAST - just ep_predict)
            acc_impaired = ep_accuracy(chain, Xte, yte, ps_imp, st, cb_codes)
            
            row = DataFrame(
                gitrev=GITREV,
                pretrain="backprop",
                impairment_type=string(imp_type),
                param=param,
                rep=rep,
                acc_clean=Float32(acc_clean),
                acc_impaired=Float32(acc_impaired),
                width=HID,
                depth=2
            )
            CSV.write(csv_file, row, append=true)
            
            @printf("    rep=%d: clean=%.4f impaired=%.4f\n",
                rep, acc_clean, acc_impaired)
        end
    end
end

println("\n=== Impaired-only calibration pilot complete. Results in $csv_file ===")