#!/usr/bin/env julia
# scripts/ep_analog_finetune.jl — A4: Analog-impairment fine-tuning harness
#
# Pretrain → impair → fine-tune with LockinEP through the impairment.
# Reports recovery fraction vs. impairment severity.
#
# Usage:
#   julia --project=. scripts/ep_analog_finetune.jl
#   PRETRAIN=both julia --project=. scripts/ep_analog_finetune.jl
#   PRETRAIN=backprop STATIC_EPOCHS=0 julia --project=. scripts/ep_analog_finetune.jl
#   RUN_GPU=0 julia --project=. scripts/ep_analog_finetune.jl  # force CPU

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

using PhasorNetworks, Lux, MLUtils, OneHotArrays, Statistics, Random, Optimisers, LinearAlgebra, CSV, DataFrames, Printf, Zygote
using Random: Xoshiro
using CUDA

# ============================================================
# CONFIGURATION (env-overridable)
# ============================================================

const OUT = get(ENV, "EPS_OUT", joinpath(repo_root, "results", "ep_analog_finetune"))
mkpath(OUT)

const PRETRAIN = Symbol(get(ENV, "PRETRAIN", "both"))  # :backprop, :staticep, :both
const N_TRAIN = parse(Int, get(ENV, "N_TRAIN", "60000"))
const N_TEST  = parse(Int, get(ENV, "N_TEST", "10000"))
const BATCH   = parse(Int, get(ENV, "BATCH", "128"))
const HID     = parse(Int, get(ENV, "HID", "256"))
const DOUT    = parse(Int, get(ENV, "DOUT", "64"))
const SEED    = parse(UInt, get(ENV, "SEED", "42"))

const BP_EPOCHS     = parse(Int, get(ENV, "BP_EPOCHS", "5"))
const BP_LR         = parse(Float64, get(ENV, "BP_LR", "0.001"))
const STATIC_EPOCHS = parse(Int, get(ENV, "STATIC_EPOCHS", "20"))
const STATIC_BETA   = parse(Float32, get(ENV, "STATIC_BETA", "0.1"))
const FT_EPOCHS     = parse(Int, get(ENV, "FT_EPOCHS", "5"))

# LockinEP knobs (from A5 optimal)
const LOCKIN_EPS     = parse(Float32, get(ENV, "LOCKIN_EPS", "0.03"))
const LOCKIN_WP      = parse(Float32, get(ENV, "LOCKIN_WP", "0.02"))
const LOCKIN_CYCLES  = parse(Int, get(ENV, "LOCKIN_CYCLES", "4"))
const LOCKIN_TFREE   = parse(Int, get(ENV, "LOCKIN_TFREE", "200"))
const LOCKIN_DT      = parse(Float32, get(ENV, "LOCKIN_DT", "0.5"))

# Weight decay
const WEIGHT_DECAY  = parse(Float64, get(ENV, "WEIGHT_DECAY", "0.0001"))

# Device
const RUN_GPU = get(ENV, "RUN_GPU", "auto")  # "1", "0", "auto"
const USE_CUDA = if RUN_GPU == "auto"
    CUDA.functional()
elseif RUN_GPU == "1"
    CUDA.functional() || error("CUDA requested but not functional")
else
    false
end
const DEVICE = USE_CUDA ? gpu_device() : cpu_device()

# Provenance
const GITREV = try
    rev = strip(read(`git -C $(repo_root) rev-parse --short HEAD`, String))
    d   = read(`git -C $(repo_root) diff HEAD -- src`, String)
    isempty(strip(d)) ? rev : rev * "-d" * string(hash(d), base = 16)[1:8]
catch
    "unknown"
end

# ============================================================
# IMPAIRMENT CONFIGS
# ============================================================

# Weight impairments: (name, param_values, apply_fn)
const WEIGHT_IMPAIRMENTS = [
    (:lognormal,   [0.01f0, 0.03f0, 0.1f0, 0.3f0]),
    (:gaussian,    [0.01f0, 0.03f0, 0.1f0, 0.3f0]),
    (:stuck_zero,  [0.01f0, 0.03f0, 0.1f0, 0.3f0]),
    (:stuck_sat,   [0.01f0, 0.03f0, 0.1f0, 0.3f0]),
]

# Update impairments: applied during fine-tuning via gradient post-processing
const UPDATE_IMPAIRMENTS = [
    (:asym_plasticity, [0.5f0, 0.8f0, 1.2f0, 2.0f0]),
    (:granular_dw,     [0.001f0, 0.01f0, 0.1f0]),
]

const REPS = 3

# ============================================================
# HELPERS
# ============================================================

cosine(a, b) = real(dot(vec(a), vec(b)) / (norm(vec(a)) * norm(vec(b)) + 1e-30))

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

function to_device(x)
    return USE_CUDA ? x |> DEVICE : x
end

# --- Backprop training (feedforward) ---
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

# --- EP evaluation ---
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

# --- StaticEP training ---
function train_staticep(chain, ps, st, train_loader, epochs, beta, codes)
    args = Args(lr=0.001, epochs=epochs, weight_decay=WEIGHT_DECAY, rng=Xoshiro(SEED + 1))
    method = StaticEP(β=beta, T_free=200, T_nudge=100, dt=0.5f0, centered=true)
    losses, ps_out, st_out = ep_train(chain, ps, st, train_loader, args;
                                       method=method,
                                       cost_fn=yb -> CodebookCost(codes, yb),
                                       optimiser=Optimisers.Adam)
    return ps_out, st_out
end

# --- LockinEP fine-tuning ---
function train_lockinep(chain, ps, st, train_loader, epochs, codes; weight_mask=nothing)
    args = Args(lr=0.001, epochs=epochs, weight_decay=WEIGHT_DECAY, rng=Xoshiro(SEED + 2))
    lockin = LockinEP(ε=LOCKIN_EPS, ω_p=LOCKIN_WP, n_cycles=LOCKIN_CYCLES,
                      T_warmup_cycles=2, T_free=LOCKIN_TFREE, dt=LOCKIN_DT)
    losses, ps_out, st_out = ep_train(chain, ps, st, train_loader, args;
                                       method=lockin,
                                       cost_fn=yb -> CodebookCost(codes, yb),
                                       optimiser=Optimisers.Adam,
                                       weight_mask=weight_mask)
    return ps_out, st_out
end

# --- Backprop fine-tuning (ceiling baseline) ---
function train_bp_finetune(chain, ps, st, train_loader, epochs, lr, codebook)
    optimiser = Optimisers.Adam(lr)
    opt_state = Optimisers.setup(optimiser, ps)
    for epoch in 1:epochs
        for (x, y) in train_loader
            lf = p -> bp_loss(x, y, chain, p, st, codebook)
            lossval, gs = Zygote.withgradient(lf, ps)
            opt_state, ps = Optimisers.update(opt_state, ps, gs[1])
        end
    end
    return ps
end

# --- Readout-only retraining (cheap baseline) ---
function train_readout_only(chain, ps, st, train_loader, epochs, lr, codebook)
    # Freeze layer_1, only train layer_2 (readout)
    frozen_ps = merge(ps, (layer_1 = ps.layer_1,))
    optimiser = Optimisers.Adam(lr)
    opt_state = Optimisers.setup(optimiser, frozen_ps)
    for epoch in 1:epochs
        for (x, y) in train_loader
            lf = p -> bp_loss(x, y, chain, p, st, codebook)
            lossval, gs = Zygote.withgradient(lf, frozen_ps)
            # gs is a tuple with one element (the gradient NamedTuple)
            g = gs[1]
            # Zero out layer_1 gradients
            g = (layer_1 = merge(g.layer_1, (weight = zero(g.layer_1.weight),)),
                 layer_2 = g.layer_2)
            if haskey(g.layer_1, :bias_real)
                g = merge(g, (layer_1 = merge(g.layer_1,
                    (bias_real = zero(g.layer_1.bias_real),
                     bias_imag = zero(g.layer_1.bias_imag))),))
            end
            opt_state, frozen_ps = Optimisers.update(opt_state, frozen_ps, g)
        end
    end
    return frozen_ps
end

# --- Impairment functions ---

# Weight impairments (return impaired ps and optionally a mask for stuck synapses)
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
    # Build weight_mask (0 for stuck, 1 for free)
    wmask = (layer_1 = (weight = Float32.(mask1), bias_real = ones(Float32, size(ps.layer_1.bias_real)), bias_imag = ones(Float32, size(ps.layer_1.bias_imag))),
             layer_2 = (weight = Float32.(mask2), bias_real = ones(Float32, size(ps.layer_2.bias_real)), bias_imag = ones(Float32, size(ps.layer_2.bias_imag))))
    return merge(ps, (layer_1 = merge(ps.layer_1, (weight = W1,)),
                        layer_2 = merge(ps.layer_2, (weight = W2,)))), wmask
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
    # Build weight_mask (0 for stuck, 1 for free)
    wmask = (layer_1 = (weight = Float32.(mask1), bias_real = ones(Float32, size(ps.layer_1.bias_real)), bias_imag = ones(Float32, size(ps.layer_1.bias_imag))),
             layer_2 = (weight = Float32.(mask2), bias_real = ones(Float32, size(ps.layer_2.bias_real)), bias_imag = ones(Float32, size(ps.layer_2.bias_imag))))
    return merge(ps, (layer_1 = merge(ps.layer_1, (weight = W1,)),
                        layer_2 = merge(ps.layer_2, (weight = W2,)))), wmask
end

# Update impairments (applied via callback in ep_train)
# We'll implement these as gradient post-processing in the callback
struct UpdateImpairment
    type::Symbol
    param::Float32
end

function apply_update_impairment!(grads, imp::UpdateImpairment)
    if imp.type == :asym_plasticity
        # η₊/η₋ asymmetry
        η₊ = 1.0f0
        η₋ = imp.param  # η₋/η₊ ratio
        for key in keys(grads)
            if haskey(grads[key], :weight)
                g = grads[key].weight
                grads[key] = merge(grads[key], (weight = (η₊ .* (g .> 0) .+ η₋ .* (g .< 0)) .* g,))
                if haskey(grads[key], :bias_real)
                    gr = grads[key].bias_real
                    gi = grads[key].bias_imag
                    grads[key] = merge(grads[key], (
                        bias_real = (η₊ .* (gr .> 0) .+ η₋ .* (gr .< 0)) .* gr,
                        bias_imag = (η₊ .* (gi .> 0) .+ η₋ .* (gi .< 0)) .* gi,
                    ))
                end
            end
        end
    elseif imp.type == :granular_dw
        # Granular Δw: round to nearest step
        step = imp.param
        for key in keys(grads)
            if haskey(grads[key], :weight)
                g = grads[key].weight
                grads[key] = merge(grads[key], (weight = round.(g ./ step) .* step,))
                if haskey(grads[key], :bias_real)
                    gr = grads[key].bias_real
                    gi = grads[key].bias_imag
                    grads[key] = merge(grads[key], (
                        bias_real = round.(gr ./ step) .* step,
                        bias_imag = round.(gi ./ step) .* step,
                    ))
                end
            end
        end
    end
    return grads
end

# ============================================================
# MAIN
# ============================================================

println("=== A4: Analog-Impairment Fine-Tuning Harness ===")
@printf("HID=%d, DOUT=%d, PRETRAIN=%s, USE_CUDA=%s\n", HID, DOUT, PRETRAIN, USE_CUDA)
println("Output: $OUT")

# Load data
println("\nLoading FashionMNIST...")
(Xtr, ytr), (Xte, yte) = load_fashionmnist()
Xtr, Xte = to_device(Xtr), to_device(Xte)
ytr, yte = to_device(ytr), to_device(yte)
println("Train: $(size(Xtr,2)), Test: $(size(Xte,2))")

# Build chain
rng = Xoshiro(SEED)
chain, ps_init, st = build_chain(rng)
cb_codes = make_codes(rng)
chain, ps_init, st = to_device(chain), to_device(ps_init), to_device(st)
cb_codes = to_device(cb_codes)

train_loader = DataLoader((Xtr, ytr); batchsize=BATCH, shuffle=true)

# CSV output
csv_file = joinpath(OUT, "analog_finetune_$(GITREV).csv")
isfile(csv_file) || CSV.write(csv_file, DataFrame(
    gitrev=String[],
    pretrain=String[],
    impairment_type=String[],
    param=Float32[],
    rep=Int[],
    acc_clean=Float32[],
    acc_impaired=Float32[],
    acc_tuned=Float32[],
    acc_bp_ft=Float32[],
    acc_readout_only=Float32[],
    recovery_frac=Float32[],
    bp_recovery_frac=Float32[],
    readout_recovery_frac=Float32[],
    width=Int[],
    depth=Int[]
))

# Pretrain paths
pretrain_results = Dict{Symbol, Any}()

# --- Pretrain: Backprop ---
if PRETRAIN in (:backprop, :both)
    println("\n=== Pretraining: Backprop (5 epochs) ===")
    ps_bp = train_bp(chain, ps_init, st, train_loader, BP_EPOCHS, BP_LR, cb_codes)
    acc_clean = ep_accuracy(chain, Xte, yte, ps_bp, st, cb_codes)
    println("Clean accuracy (ep_predict): $acc_clean")
    pretrain_results[:backprop] = (ps=ps_bp, st=st, acc_clean=acc_clean)
end

# --- Pretrain: StaticEP ---
if PRETRAIN in (:staticep, :both)
    println("\n=== Pretraining: StaticEP (20 epochs) ===")
    ps_ep, st_ep = train_staticep(chain, ps_init, st, train_loader, STATIC_EPOCHS, STATIC_BETA, cb_codes)
    acc_clean = ep_accuracy(chain, Xte, yte, ps_ep, st_ep, cb_codes)
    println("Clean accuracy (ep_predict): $acc_clean")
    pretrain_results[:staticep] = (ps=ps_ep, st=st_ep, acc_clean=acc_clean)
end

# ============================================================
# IMPAIRMENT SWEEP
# ============================================================

for (pretrain_name, pretrain_data) in pretrain_results
    ps_clean = pretrain_data.ps
    st_clean = pretrain_data.st
    local acc_clean = pretrain_data.acc_clean

    # Weight impairments
    for (imp_type, params) in WEIGHT_IMPAIRMENTS
        println("\n=== Impairment: $imp_type ===")
        for param in params
            @printf("  param = %.3f\n", param)
            for rep in 1:REPS
                rng_rep = Xoshiro(hash((imp_type, param, rep, pretrain_name)) % UInt64)
                
                # Apply weight impairment
                if imp_type == :lognormal
                    ps_imp, wmask = apply_lognormal(ps_clean, param, rng_rep)
                elseif imp_type == :gaussian
                    ps_imp, wmask = apply_gaussian(ps_clean, param, rng_rep)
                elseif imp_type == :stuck_zero
                    ps_imp, wmask = apply_stuck_zero(ps_clean, param, rng_rep)
                elseif imp_type == :stuck_sat
                    ps_imp, wmask = apply_stuck_sat(ps_clean, param, rng_rep)
                end
                
                # Evaluate impaired
                acc_impaired = ep_accuracy(chain, Xte, yte, ps_imp, st_clean, cb_codes)
                
                # Fine-tune with LockinEP
                ps_tuned, st_tuned = train_lockinep(chain, ps_imp, st_clean, train_loader, FT_EPOCHS, cb_codes; weight_mask=wmask)
                acc_tuned = ep_accuracy(chain, Xte, yte, ps_tuned, st_tuned, cb_codes)
                
                # Baseline: Backprop fine-tune
                ps_bp_ft = train_bp_finetune(chain, ps_imp, st_clean, train_loader, FT_EPOCHS, BP_LR, cb_codes)
                acc_bp_ft = ep_accuracy(chain, Xte, yte, ps_bp_ft, st_clean, cb_codes)
                
                # Baseline: Readout-only
                ps_ro = train_readout_only(chain, ps_imp, st_clean, train_loader, FT_EPOCHS, BP_LR, cb_codes)
                acc_ro = ep_accuracy(chain, Xte, yte, ps_ro, st_clean, cb_codes)
                
                # Recovery fractions
                denom = acc_clean - acc_impaired
                if denom > 0
                    rec_frac = clamp((acc_tuned - acc_impaired) / denom, -1.0, 2.0)
                    bp_rec = clamp((acc_bp_ft - acc_impaired) / denom, -1.0, 2.0)
                    ro_rec = clamp((acc_ro - acc_impaired) / denom, -1.0, 2.0)
                else
                    rec_frac = bp_rec = ro_rec = NaN
                end
                
                row = DataFrame(
                    gitrev=GITREV,
                    pretrain=string(pretrain_name),
                    impairment_type=string(imp_type),
                    param=param,
                    rep=rep,
                    acc_clean=Float32(acc_clean),
                    acc_impaired=Float32(acc_impaired),
                    acc_tuned=Float32(acc_tuned),
                    acc_bp_ft=Float32(acc_bp_ft),
                    acc_readout_only=Float32(acc_ro),
                    recovery_frac=Float32(rec_frac),
                    bp_recovery_frac=Float32(bp_rec),
                    readout_recovery_frac=Float32(ro_rec),
                    width=HID,
                    depth=2
                )
                CSV.write(csv_file, row, append=true)
                
                @printf("    rep=%d: clean=%.4f impaired=%.4f tuned=%.4f bp_ft=%.4f ro=%.4f rec=%.3f\n",
                    rep, acc_clean, acc_impaired, acc_tuned, acc_bp_ft, acc_ro, rec_frac)
            end
        end
    end
    
    # Update impairments (only for lognormal σ=0.03 as representative)
    println("\n=== Update Impairments (on lognormal σ=0.03) ===")
    for (imp_type, params) in UPDATE_IMPAIRMENTS
        for param in params
            @printf("  param = %.3f\n", param)
            for rep in 1:REPS
                rng_rep = Xoshiro(hash((imp_type, param, rep, pretrain_name)) % UInt64)
                
                # Apply base weight impairment (lognormal σ=0.03)
                ps_imp, wmask = apply_lognormal(ps_clean, 0.03f0, rng_rep)
                acc_impaired = ep_accuracy(chain, Xte, yte, ps_imp, st_clean, cb_codes)
                
                # Fine-tune with LockinEP + update impairment via callback
                # We'll use a custom callback to post-process gradients
                # For now, skip update impairments as they need callback support in ep_train
                # TODO: implement callback-based update impairment
                println("    TODO: update impairments need callback support")
            end
        end
    end
end

println("\n=== Sweep complete. Results in $csv_file ===")