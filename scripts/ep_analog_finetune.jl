#!/usr/bin/env julia
#
# scripts/ep_analog_finetune.jl — A4: Analog-impairment fine-tuning harness
#
# This is the headline experiment: can EP fine-tune through impaired weights?
#
# Structure:
# 1. Pretrain to a good checkpoint (backprop, per A3)
# 2. Apply impairment model to ps — menu, each independently switchable:
#    - multiplicative lognormal on W (σ sweep)
#    - additive Gaussian noise
#    - fraction of stuck-at-zero synapses
#    - fraction stuck at saturation
# 3. Record accuracy drop
# 4. Fine-tune through impaired network with LockinEP for k epochs
#    (stuck synapses stay stuck: mask both weight AND its update)
# 5. Optionally impair the update too: asymmetric plasticity η₊ ≠ η₋;
#    granular Δw with minimum representable step
# 6. Compare against three references:
#    - backprop fine-tuning (ceiling)
#    - no fine-tuning (floor)
#    - readout-only retraining (cheap baseline)
#
# Report: recovery fraction (acc_tuned − acc_impaired) / (acc_clean − acc_impaired)
# against impairment severity.
#
# Usage:
#   julia --project=. scripts/ep_analog_finetune.jl
#   julia --project=. -t 4 scripts/ep_analog_finetune.jl
#   julia --project=. scripts/ep_analog_finetune.jl --impairment lognormal --sigma 0.1

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
using Dates, LinearAlgebra, Printf
using Random: Xoshiro

# Parse command line args
const IMPAIRMENT_TYPE = get(ENV, "IMPAIRMENT", "lognormal")  # lognormal, gaussian, stuck_zero, stuck_sat
const SIGMA = parse(Float32, get(ENV, "SIGMA", "0.1"))
const FRAC = parse(Float32, get(ENV, "FRAC", "0.1"))
const FINETUNE_EPOCHS = parse(Int, get(ENV, "FINETUNE_EPOCHS", "10"))
const PRETRAIN_EPOCHS = parse(Int, get(ENV, "PRETRAIN_EPOCHS", "5"))
const LR = parse(Float64, get(ENV, "LR", "0.001"))
const FINETUNE_LR = parse(Float64, get(ENV, "FINETUNE_LR", "0.001"))
const SEED = parse(Int, get(ENV, "SEED", "42"))
const USE_CUDA = CUDA.functional()
const BATCHSIZE = 128
const HID = 256
const DOUT = 64
const SCALE = 0.4f0

cdev = cpu_device()
gdev = gpu_device()
dev = USE_CUDA ? gdev : cdev

args = Args(batchsize = BATCHSIZE,
            epochs = PRETRAIN_EPOCHS,
            lr = LR,
            rng = Xoshiro(SEED),
            use_cuda = USE_CUDA)

finetune_args = Args(batchsize = BATCHSIZE,
                     epochs = FINETUNE_EPOCHS,
                     lr = FINETUNE_LR,
                     rng = Xoshiro(SEED + 1000),
                     use_cuda = USE_CUDA)

println("=== A4: Analog Fine-tuning Harness ===")
println("Impairment: $IMPAIRMENT_TYPE")
println("Params: sigma=$SIGMA, frac=$FRAC, finetune_epochs=$FINETUNE_EPOCHS, pretrain_epochs=$PRETRAIN_EPOCHS")
println("LR: pretrain=$LR, finetune=$FINETUNE_LR, seed=$SEED, use_cuda=$USE_CUDA")

# ---- encode_phase (matching ep_fashionmnist.jl) ----
function encode_phase(imgs::AbstractArray{Float32,3})
    N = size(imgs, 3)
    flat = reshape(imgs, :, N)
    μ = mean(flat; dims=1)
    σ = std(flat; dims=1) .+ 1f-6
    return Phase.(0.5f0 .* tanh.((flat .- μ) ./ σ))
end

# Load FashionMNIST
println("\nLoading FashionMNIST...")
tr = fashion_mnist_data(:train)
te = fashion_mnist_data(:test)
ntr = min(60000, length(tr.targets))
nte = min(10000, length(te.targets))
Xtr = encode_phase(Float32.(tr.features[:, :, 1:ntr]))
Xte = encode_phase(Float32.(te.features[:, :, 1:nte]))
ytr = Int.(tr.targets[1:ntr]) .+ 1
yte = Int.(te.targets[1:nte]) .+ 1

train_loader = DataLoader((Xtr, ytr), batchsize=BATCHSIZE, shuffle=true)
test_loader = DataLoader((Xte, yte), batchsize=BATCHSIZE, shuffle=false)

# ---- Build EP chain ----
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

# ---- Loss/accuracy functions ----
function bp_loss(x, y, model, ps, st, codebook, dev=dev)
    x = x |> dev
    y = y |> dev
    z_out, _ = Lux.apply(model, x, ps, st)
    z_out_c = ComplexF32.(angle_to_complex(z_out))
    codes_dev = codebook |> dev
    logits = similarity_outer(z_out_c, codes_dev)
    y_onehot = onehotbatch(y .- 1, 0:9)
    log_probs = logits .- log.(sum(exp.(logits); dims=1))
    loss = -mean(sum(y_onehot .* log_probs; dims=1))
    return loss
end

function bp_accuracy(model, data_loader, ps, st, codebook, dev=dev)
    total_correct = 0
    total_samples = 0
    for (x, y) in data_loader
        x = x |> dev
        y = y |> dev
        z_out, _ = Lux.apply(model, x, ps, st)
        z_out_c = ComplexF32.(angle_to_complex(z_out))
        logits = similarity_outer(z_out_c, codebook |> dev)
        logits_cpu = logits |> cdev
        pred_indices = argmax(logits_cpu; dims=1)
        pred_labels = [idx[1] for idx in vec(pred_indices)]
        y_cpu = y |> cdev
        total_correct += sum(pred_labels .== y_cpu)
        total_samples += length(y_cpu)
    end
    return total_correct / total_samples
end

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

# ---- Impairment functions ----
function apply_lognormal_impairment(ps, σ, rng)
    W1 = ps.layer_1.weight
    W2 = ps.layer_2.weight
    # Generate noise on same device as weights
    noise1 = exp.(σ .* randn(rng, Float32, size(W1) |> cdev)) .- 1f0
    noise2 = exp.(σ .* randn(rng, Float32, size(W2) |> cdev)) .- 1f0
    noise1 = noise1 |> typeof(W1)
    noise2 = noise2 |> typeof(W2)
    return (layer_1 = merge(ps.layer_1, (weight = W1 .* (1f0 .+ noise1),)),
            layer_2 = merge(ps.layer_2, (weight = W2 .* (1f0 .+ noise2),)))
end

function apply_gaussian_impairment(ps, σ, rng)
    W1 = ps.layer_1.weight
    W2 = ps.layer_2.weight
    noise1 = σ .* randn(rng, Float32, size(W1) |> cdev)
    noise2 = σ .* randn(rng, Float32, size(W2) |> cdev)
    noise1 = noise1 |> typeof(W1)
    noise2 = noise2 |> typeof(W2)
    return (layer_1 = merge(ps.layer_1, (weight = W1 .+ noise1,)),
            layer_2 = merge(ps.layer_2, (weight = W2 .+ noise2,)))
end

function apply_stuck_zero_impairment(ps, frac, rng)
    W1 = ps.layer_1.weight
    W2 = ps.layer_2.weight
    mask1 = rand(rng, Float32, size(W1) |> cdev) .< frac
    mask2 = rand(rng, Float32, size(W2) |> cdev) .< frac
    mask1 = mask1 |> typeof(W1)
    mask2 = mask2 |> typeof(W2)
    W1_imp = W1 .* (1f0 .- mask1)
    W2_imp = W2 .* (1f0 .- mask2)
    # Also create update masks (same stuck synapses stay stuck)
    update_mask1 = 1f0 .- mask1
    update_mask2 = 1f0 .- mask2
    ps_imp = (layer_1 = merge(ps.layer_1, (weight = W1_imp,)),
              layer_2 = merge(ps.layer_2, (weight = W2_imp,)))
    update_masks = (layer_1 = (weight = update_mask1,),
                    layer_2 = (weight = update_mask2,))
    return ps_imp, update_masks
end

function apply_stuck_saturation_impairment(ps, frac, rng)
    W1 = ps.layer_1.weight
    W2 = ps.layer_2.weight
    # Stuck at max magnitude (saturation)
    max1 = maximum(abs, W1 |> cdev)
    max2 = maximum(abs, W2 |> cdev)
    mask1 = rand(rng, Float32, size(W1) |> cdev) .< frac
    mask2 = rand(rng, Float32, size(W2) |> cdev) .< frac
    mask1 = mask1 |> typeof(W1)
    mask2 = mask2 |> typeof(W2)
    sign1 = sign.(W1)
    sign2 = sign.(W2)
    W1_imp = W1 .* (1f0 .- mask1) .+ mask1 .* sign1 .* max1
    W2_imp = W2 .* (1f0 .- mask2) .+ mask2 .* sign2 .* max2
    update_mask1 = 1f0 .- mask1
    update_mask2 = 1f0 .- mask2
    ps_imp = (layer_1 = merge(ps.layer_1, (weight = W1_imp,)),
              layer_2 = merge(ps.layer_2, (weight = W2_imp,)))
    update_masks = (layer_1 = (weight = update_mask1,),
                    layer_2 = (weight = update_mask2,))
    return ps_imp, update_masks
end

# ---- Training functions ----
function train_bp(chain, ps, st, train_loader, args, codebook, dev; epochs=args.epochs, lr=args.lr)
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
        println("  Epoch $epoch: loss = $(losses[end])")
    end
    return ps, losses
end

function train_ep(chain, ps, st, train_loader, args, codebook, dev; epochs=args.epochs, lr=args.lr, update_masks=nothing)
    method = LockinEP(ε=0.05f0, ω_p=0.05f0, n_cycles=4, T_warmup_cycles=2, T_free=100, dt=0.1f0)
    optimiser = Optimisers.Adam(lr)
    opt_state = Optimisers.setup(optimiser, ps)
    losses = Float64[]
    for epoch in 1:epochs
        epoch_losses = Float64[]
        for (x, y) in train_loader
            x = x |> dev
            y = y |> dev
            x_phase = Phase.(x)
            # Use CodebookCost with batch labels (1-based) - move y to CPU first
            y_cpu = y |> cdev
            cost = CodebookCost(codebook, y_cpu)
            g, _ = ep_gradient(method, chain, ps, st, x_phase, cost)
            # Apply update masks if provided (for stuck synapses)
            if update_masks !== nothing
                if haskey(g, :layer_1) && haskey(update_masks, :layer_1)
                    g = merge(g, (layer_1 = merge(g.layer_1, (weight = g.layer_1.weight .* update_masks.layer_1.weight,)),))
                end
                if haskey(g, :layer_2) && haskey(update_masks, :layer_2)
                    g = merge(g, (layer_2 = merge(g.layer_2, (weight = g.layer_2.weight .* update_masks.layer_2.weight,)),))
                end
            end
            if args.weight_decay > 0
                PhasorNetworks._apply_weight_decay(g, ps, args.weight_decay)
            end
            opt_state, ps = Optimisers.update(opt_state, ps, g)
            lossval = ep_loss(cost, PhasorNetworks.phasor_settle(chain, ps, st, x_phase, cost, 0f0;
                              T=100, dt=0.1f0, K_mode=:zero)[end])
            push!(epoch_losses, lossval)
        end
        push!(losses, mean(epoch_losses))
        println("  EP Epoch $epoch: loss = $(losses[end])")
    end
    return ps, losses
end

function train_readout_only(chain, ps, st, train_loader, args, codebook, dev; epochs=args.epochs, lr=args.lr)
    """Retrain only the output layer (layer_2) - cheap baseline."""
    # Freeze layer_1
    ps_frozen = (layer_1 = ps.layer_1,
                 layer_2 = ps.layer_2)
    optimiser = Optimisers.Adam(lr)
    opt_state = Optimisers.setup(optimiser, ps_frozen)
    losses = Float64[]
    for epoch in 1:epochs
        epoch_losses = Float64[]
        for (x, y) in train_loader
            x = x |> dev
            y = y |> dev
            # Only compute gradient for layer_2
            lf = p -> bp_loss(x, y, chain, p, st, codebook, dev)
            lossval, gs = withgradient(lf, ps_frozen)
            # Zero out layer_1 gradients
            gs = (layer_1 = (weight = zero(gs[1].layer_1.weight),
                            log_neg_lambda = zero(gs[1].layer_1.log_neg_lambda),
                            bias_real = zero(gs[1].layer_1.bias_real),
                            bias_imag = zero(gs[1].layer_1.bias_imag)),
                  layer_2 = gs[1].layer_2)
            push!(epoch_losses, lossval)
            if args.weight_decay > 0
                PhasorNetworks._apply_weight_decay(gs[1], ps_frozen, args.weight_decay)
            end
            opt_state, ps_frozen = Optimisers.update(opt_state, ps_frozen, gs[1])
        end
        push!(losses, mean(epoch_losses))
        println("  Readout-only Epoch $epoch: loss = $(losses[end])")
    end
    return ps_frozen, losses
end

# ============================================================
# MAIN EXPERIMENT
# ============================================================

# 1. PRETRAIN (backprop)
println("\n=== 1. PRETRAIN (backprop, $PRETRAIN_EPOCHS epochs) ===")
initial_loss = bp_loss(first(train_loader)..., chain, ps, st, cb_codes, dev)
println("Initial loss: $initial_loss")
ps, pretrain_losses = train_bp(chain, ps, st, train_loader, args, cb_codes, dev)

acc_clean_bp = bp_accuracy(chain, test_loader, ps, st, cb_codes, dev)
acc_clean_ep = ep_accuracy(chain, Xte, yte, ps, st, cb_codes, dev)
println("Clean BP accuracy: $acc_clean_bp")
println("Clean EP accuracy: $acc_clean_ep")

# Save clean checkpoint
ps_clean = deepcopy(ps)
st_clean = deepcopy(st)

# 2. APPLY IMPAIRMENT
println("\n=== 2. APPLY IMPAIRMENT: $IMPAIRMENT_TYPE ===")
rng_imp = Xoshiro(SEED + 2000)

if IMPAIRMENT_TYPE == "lognormal"
    ps_imp = apply_lognormal_impairment(ps_clean, SIGMA, rng_imp)
    update_masks = nothing
elseif IMPAIRMENT_TYPE == "gaussian"
    ps_imp = apply_gaussian_impairment(ps_clean, SIGMA, rng_imp)
    update_masks = nothing
elseif IMPAIRMENT_TYPE == "stuck_zero"
    ps_imp, update_masks = apply_stuck_zero_impairment(ps_clean, FRAC, rng_imp)
elseif IMPAIRMENT_TYPE == "stuck_sat"
    ps_imp, update_masks = apply_stuck_saturation_impairment(ps_clean, FRAC, rng_imp)
else
    error("Unknown impairment type: $IMPAIRMENT_TYPE")
end

# Evaluate impaired
acc_imp_bp = bp_accuracy(chain, test_loader, ps_imp, st_clean, cb_codes, dev)
acc_imp_ep = ep_accuracy(chain, Xte, yte, ps_imp, st_clean, cb_codes, dev)
println("Impaired BP accuracy: $acc_imp_bp (drop: $(acc_clean_bp - acc_imp_bp))")
println("Impaired EP accuracy: $acc_imp_ep (drop: $(acc_clean_ep - acc_imp_ep))")

# 3. FINE-TUNE with LockinEP through impaired network
println("\n=== 3. FINE-TUNE with LockinEP ($FINETUNE_EPOCHS epochs) ===")
ps_finetuned, ft_losses = train_ep(chain, ps_imp, st_clean, train_loader, finetune_args, cb_codes, dev; update_masks=update_masks)

acc_ft_bp = bp_accuracy(chain, test_loader, ps_finetuned, st_clean, cb_codes, dev)
acc_ft_ep = ep_accuracy(chain, Xte, yte, ps_finetuned, st_clean, cb_codes, dev)
println("Finetuned BP accuracy: $acc_ft_bp")
println("Finetuned EP accuracy: $acc_ft_ep")

# 4. BASELINES
println("\n=== 4. BASELINES ===")

# No fine-tuning (floor)
acc_floor_bp = acc_imp_bp
acc_floor_ep = acc_imp_ep
println("Floor (no finetune) BP: $acc_floor_bp")
println("Floor (no finetune) EP: $acc_floor_ep")

# Backprop fine-tuning (ceiling)
println("\n--- Backprop fine-tuning (ceiling) ---")
ps_bp_ft, _ = train_bp(chain, ps_imp, st_clean, train_loader, finetune_args, cb_codes, dev)
acc_bp_ceiling = bp_accuracy(chain, test_loader, ps_bp_ft, st_clean, cb_codes, dev)
println("BP ceiling accuracy: $acc_bp_ceiling")

# Readout-only retraining (cheap baseline)
println("\n--- Readout-only retraining ---")
ps_ro_ft, _ = train_readout_only(chain, ps_imp, st_clean, train_loader, finetune_args, cb_codes, dev)
acc_ro_bp = bp_accuracy(chain, test_loader, ps_ro_ft, st_clean, cb_codes, dev)
acc_ro_ep = ep_accuracy(chain, Xte, yte, ps_ro_ft, st_clean, cb_codes, dev)
println("Readout-only BP accuracy: $acc_ro_bp")
println("Readout-only EP accuracy: $acc_ro_ep")

# 5. COMPUTE RECOVERY FRACTION
println("\n=== 5. RECOVERY FRACTION ===")
# Recovery = (acc_tuned - acc_impaired) / (acc_clean - acc_impaired)
# Using BP accuracy as primary metric
rec_bp = (acc_ft_bp - acc_imp_bp) / (acc_clean_bp - acc_imp_bp)
rec_bp_ceiling = (acc_bp_ceiling - acc_imp_bp) / (acc_clean_bp - acc_imp_bp)
rec_ro = (acc_ro_bp - acc_imp_bp) / (acc_clean_bp - acc_imp_bp)

# Using EP accuracy
rec_ep = (acc_ft_ep - acc_imp_ep) / (acc_clean_ep - acc_imp_ep)
rec_ro_ep = (acc_ro_ep - acc_imp_ep) / (acc_clean_ep - acc_imp_ep)

println("LockinEP recovery (BP metric): $(round(rec_bp*100, digits=1))%")
println("BP ceiling recovery (BP metric): $(round(rec_bp_ceiling*100, digits=1))%")
println("Readout-only recovery (BP metric): $(round(rec_ro*100, digits=1))%")
println()
println("LockinEP recovery (EP metric): $(round(rec_ep*100, digits=1))%")
println("Readout-only recovery (EP metric): $(round(rec_ro_ep*100, digits=1))%")

# 6. SUMMARY
println("\n=== SUMMARY ===")
@printf("Impairment: %s (σ=%.3f, frac=%.3f)\n", IMPAIRMENT_TYPE, SIGMA, FRAC)
@printf("Clean BP:     %.4f\n", acc_clean_bp)
@printf("Impaired BP:  %.4f  (drop %.4f)\n", acc_imp_bp, acc_clean_bp - acc_imp_bp)
@printf("Floor BP:     %.4f\n", acc_floor_bp)
@printf("LockinEP FT:  %.4f  (recovery %.1f%%)\n", acc_ft_bp, rec_bp*100)
@printf("BP ceiling:   %.4f  (recovery %.1f%%)\n", acc_bp_ceiling, rec_bp_ceiling*100)
@printf("Readout-only: %.4f  (recovery %.1f%%)\n", acc_ro_bp, rec_ro*100)
println()
@printf("Clean EP:     %.4f\n", acc_clean_ep)
@printf("Impaired EP:  %.4f  (drop %.4f)\n", acc_imp_ep, acc_clean_ep - acc_imp_ep)
@printf("LockinEP FT:  %.4f  (recovery %.1f%%)\n", acc_ft_ep, rec_ep*100)
@printf("Readout-only: %.4f  (recovery %.1f%%)\n", acc_ro_ep, rec_ro_ep*100)

# Save results
results = Dict(
    "impairment" => IMPAIRMENT_TYPE,
    "sigma" => SIGMA,
    "frac" => FRAC,
    "finetune_epochs" => FINETUNE_EPOCHS,
    "acc_clean_bp" => acc_clean_bp,
    "acc_imp_bp" => acc_imp_bp,
    "acc_ft_bp" => acc_ft_bp,
    "acc_bp_ceiling" => acc_bp_ceiling,
    "acc_ro_bp" => acc_ro_bp,
    "recovery_bp" => rec_bp,
    "recovery_bp_ceiling" => rec_bp_ceiling,
    "recovery_ro" => rec_ro,
    "acc_clean_ep" => acc_clean_ep,
    "acc_imp_ep" => acc_imp_ep,
    "acc_ft_ep" => acc_ft_ep,
    "acc_ro_ep" => acc_ro_ep,
    "recovery_ep" => rec_ep,
    "recovery_ro_ep" => rec_ro_ep,
    "gitrev" => readchomp(`git rev-parse HEAD`),
    "timestamp" => Dates.format(now(), "yyyy-mm-dd HH:MM:SS")
)

outdir = joinpath("results", "ep_analog_finetune")
mkpath(outdir)
fname = joinpath(outdir, "finetune_$(IMPAIRMENT_TYPE)_s$(SIGMA)_f$(FRAC)_$(Dates.format(now(), "yyyy-mm-dd_HHMMSS")).jld2")
using JLD2
@save fname results
println("\nResults saved to: $fname")

println("\nDone.")