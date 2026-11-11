#!/usr/bin/env julia
#
# scripts/attractor_sweep.jl  —  diagnostic sweep for AttractorPhasorSSM
#
# A small synthetic phasor classification task built to isolate the
# attractor mechanism from any encoder confounds. Each input is a
# length-L sequence of noisy versions of one of K random phasor
# prototypes; the model is AttractorPhasorSSM(D=>D, K) + LastStepDense
# and is asked to identify which prototype each sequence came from.
#
# The first sfmnist+attractor run mode-collapsed (effective_codes = 1.0
# across the first four epochs at default init α=0.4, β=8). This sweep
# scans (init_α, init_β, init_codes) to find a regime where the K
# codes actually differentiate during training rather than one code
# dominating from the first batch.
#
# What "good" looks like:
#   - effective_codes climbs toward K over epochs (codes diversify)
#   - test_acc climbs above chance (1/K = 0.10 for K=10)
#   - mean_max_sim climbs (states concentrate near codes)
#
# What "mode collapse" looks like:
#   - effective_codes ≈ 1 throughout (one code wins everything)
#   - test_acc stuck near chance
#   - mean_max_sim plateaus low (states stuck near one code's basin)
#
# ---------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------
#   julia --project=scripts scripts/attractor_sweep.jl                      # default sweep
#   julia --project=scripts -e 'include("scripts/attractor_sweep.jl"); attractor_sweep()'

using PhasorNetworks, Lux, Zygote, Optimisers, CUDA, LuxCUDA
using OneHotArrays: onehotbatch
using Random: Xoshiro, AbstractRNG
using Statistics: mean, std
using LinearAlgebra: I

# ---------------------------------------------------------------------
# Synthetic task
# ---------------------------------------------------------------------

"""
    generate_phasor_task(rng; D, K, L, n_samples, noise) -> (X, y, prototypes)

K random phasor prototypes (D, K). For each sample n: pick a class
c_n ∈ 1:K, build a length-L sequence of noisy versions of prototype c_n
(IID phase noise per step). Returns:

- `X :: (D, L, n_samples)` Phase  — input sequences
- `y :: (n_samples,) Int`         — class labels in 1..K
- `prototypes :: (D, K)` Phase    — ground-truth class prototypes
"""
function generate_phasor_task(rng::AbstractRNG;
                              D::Int, K::Int, L::Int, n_samples::Int,
                              noise::Float32 = 0.15f0)
    prototypes = random_symbols(rng, (D, K))                            # (D, K) Phase
    y = rand(rng, 1:K, n_samples)
    X = zeros(Phase, D, L, n_samples)
    for n in 1:n_samples
        proto_f = Float32.(prototypes[:, y[n]])                          # (D,) Float32
        for t in 1:L
            noisy = PhasorNetworks.remap_phase.(proto_f .+ noise .* randn(rng, Float32, D))
            X[:, t, n] .= noisy
        end
    end
    return X, y, prototypes
end

function batch_synth(X::Array{Phase, 3}, y::Vector{Int}, B::Int)
    N = length(y)
    out = Vector{Tuple{Array{Phase, 3}, Vector{Int}}}()
    for i in 1:B:N
        e = min(i + B - 1, N)
        push!(out, (X[:, :, i:e], y[i:e]))
    end
    return out
end

# ---------------------------------------------------------------------
# Model: bare attractor + readout (no encoder)
# ---------------------------------------------------------------------

"""
    LastStepDense(in_dims, out_dims) <: Lux.AbstractLuxLayer

Same as in scripts/sfmnist.jl — final-timestep linear readout.
"""
struct LastStepDense <: Lux.AbstractLuxLayer
    in_dims::Int
    out_dims::Int
end

function Lux.initialparameters(rng::AbstractRNG, l::LastStepDense)
    scale = 1f0 / sqrt(Float32(l.in_dims))
    return (W = scale .* randn(rng, Float32, l.out_dims, l.in_dims),
            b = zeros(Float32, l.out_dims))
end
Lux.initialstates(::AbstractRNG, ::LastStepDense) = NamedTuple()

function (l::LastStepDense)(x::AbstractArray{<:Phase, 3}, ps, st)
    last_step = Float32.(x[:, end, :])
    return ps.W * last_step .+ ps.b, st
end

function build_attractor_only_model(D::Int, K::Int;
                                     init_log_alpha::Real,
                                     init_log_beta::Real,
                                     init_codes::Symbol,
                                     attractor_log_neg_lambda::Real = log(0.1))
    return Chain(
        AttractorPhasorSSM(D => D, K;
                            init_log_alpha = init_log_alpha,
                            init_log_beta  = init_log_beta,
                            init_codes     = init_codes,
                            init_log_neg_lambda = attractor_log_neg_lambda,
                            trainable_codes = true),
        LastStepDense(D, K)
    )
end

# ---------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------

function ce_loss(logits, y_oh)
    m = maximum(logits; dims = 1)
    log_sm = logits .- m .- log.(sum(exp.(logits .- m); dims = 1))
    return -mean(vec(sum(y_oh .* log_sm; dims = 1)))
end

function accuracy(logits, labels)
    preds = vec(getindex.(argmax(Array(logits); dims = 1), 1))
    return mean(preds .== labels)
end

function train_epoch!(model, ps, st, opt_state, batches, K; gdev)
    losses = Float32[]
    for (x, y) in batches
        x_d  = x |> gdev
        y_oh = Float32.(onehotbatch(y, 1:K)) |> gdev
        loss_val, back = Zygote.pullback(ps) do p
            logits, _ = model(x_d, p, st)
            ce_loss(logits, y_oh)
        end
        grads = back(one(loss_val))[1]
        opt_state, ps = Optimisers.update(opt_state, ps, grads)
        push!(losses, loss_val)
    end
    return ps, opt_state, losses
end

function evaluate(model, ps, st, batches; gdev)
    accs = Float32[]
    for (x, y) in batches
        x_d = x |> gdev
        logits, _ = model(x_d, ps, st)
        push!(accs, accuracy(logits, y))
    end
    return mean(accs)
end

# ---------------------------------------------------------------------
# Attractor diagnostic (mirrors scripts/sfmnist.jl, but skips the
# encoder since the attractor is now layer 1)
# ---------------------------------------------------------------------

function attractor_diag_bare(model, ps, st, x_b)
    att = model.layers[1]
    @assert att isa AttractorPhasorSSM "layer 1 must be AttractorPhasorSSM"
    out, _ = att(x_b, ps.layer_1, st.layer_1)
    codes_phase = att.trainable_codes ? ps.layer_1.codes : st.layer_1.codes
    codes_h = Array(codes_phase); out_h = Array(out)
    D, L, B = size(out_h); K = size(codes_h, 2)

    states = reshape(out_h, D, L * B)
    sims   = similarity_outer(states, codes_h; dims = 2)
    mean_max_sim = mean(vec(maximum(sims; dims = 1)))

    final_states = out_h[:, end, :]
    final_sims   = similarity_outer(final_states, codes_h; dims = 2)
    winners      = vec(getindex.(argmax(final_sims; dims = 1), 1))
    counts       = [count(==(k), winners) for k in 1:K]
    p_safe       = filter(>(0), counts ./ B)
    eff_codes    = isempty(p_safe) ? 0.0 : exp(-sum(pi -> pi * log(pi), p_safe; init = 0.0))

    α = inv(one(Float32) + exp(-Array(ps.layer_1.log_alpha)[1]))
    β = exp(Array(ps.layer_1.log_beta)[1])
    return (mean_max_sim = Float32(mean_max_sim),
            effective_codes = Float32(eff_codes),
            alpha = Float32(α), beta = Float32(β),
            n_codes = K)
end

# ---------------------------------------------------------------------
# Single training run — used by the sweep
# ---------------------------------------------------------------------

"""
    train_one(; init_log_alpha, init_log_beta, init_codes, ...) -> NamedTuple

Train one configuration of the bare-attractor model on the synthetic
phasor task. Returns a NamedTuple of per-epoch test_acc and the final
attractor diagnostic.
"""
function train_one(; init_log_alpha::Real,
                    init_log_beta::Real,
                    init_codes::Symbol,
                    D::Int = 128,
                    K::Int = 10,
                    L::Int = 50,
                    B::Int = 64,
                    n_train::Int = 2000,
                    n_test::Int  = 500,
                    n_epochs::Int = 20,
                    noise::Float32 = 0.15f0,
                    lr::Real = 1f-3,
                    use_cuda::Bool = true,
                    seed::Int = 0,
                    verbose::Bool = false)
    rng = Xoshiro(seed)
    gdev = (use_cuda && CUDA.functional()) ? gpu_device() : cpu_device()

    Xtr, ytr, _ = generate_phasor_task(rng; D, K, L, n_samples = n_train, noise)
    Xte, yte, _ = generate_phasor_task(rng; D, K, L, n_samples = n_test,  noise)
    train_batches = batch_synth(Xtr, ytr, B)
    test_batches  = batch_synth(Xte, yte, B)

    model = build_attractor_only_model(D, K;
                                        init_log_alpha = init_log_alpha,
                                        init_log_beta  = init_log_beta,
                                        init_codes     = init_codes)
    ps_cpu, st_cpu = Lux.setup(rng, model)
    ps = ps_cpu |> gdev
    st = st_cpu |> gdev
    opt_state = Optimisers.setup(Optimisers.Adam(Float32(lr)), ps)
    diag_probe = test_batches[1][1] |> gdev

    accs       = Float32[]
    eff_codes  = Float32[]
    losses_per = Float32[]

    for epoch in 1:n_epochs
        ps, opt_state, losses = train_epoch!(model, ps, st, opt_state, train_batches, K; gdev)
        test_acc = evaluate(model, ps, st, test_batches; gdev)
        diag = attractor_diag_bare(model, ps, st, diag_probe)
        push!(accs, test_acc)
        push!(eff_codes, diag.effective_codes)
        push!(losses_per, mean(losses))
        if verbose
            @info "epoch" epoch loss = round(mean(losses), digits = 3) test_acc = round(test_acc, digits = 3) eff_codes = round(diag.effective_codes, digits = 2) α = round(diag.alpha, digits = 3) β = round(diag.beta, digits = 3)
        end
    end

    final_diag = attractor_diag_bare(model, ps, st, diag_probe)
    return (accs = accs, eff_codes = eff_codes, losses = losses_per,
            final_alpha = final_diag.alpha, final_beta = final_diag.beta,
            final_eff_codes = final_diag.effective_codes,
            final_max_sim = final_diag.mean_max_sim)
end

# ---------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------

"""
    attractor_sweep(; D=128, K=10, ...) -> Vector{NamedTuple}

Sweep init_α × init_β × init_codes. Prints a results table and returns
the row data for downstream inspection.
"""
function attractor_sweep(; D::Int = 128,
                          K::Int = 10,
                          L::Int = 50,
                          B::Int = 64,
                          n_train::Int = 2000,
                          n_test::Int  = 500,
                          n_epochs::Int = 20,
                          noise::Float32 = 0.15f0,
                          lr::Real = 1f-3,
                          seed::Int = 0)
    α_grid     = (log(0.05), log(0.1), log(0.4))
    β_grid     = (log(0.5), log(2.0), log(8.0))
    codes_grid = (:random, :orthogonal)

    results = NamedTuple[]
    println("\n=== AttractorPhasorSSM hyperparameter sweep ===")
    println("  D=$D, K=$K, L=$L, B=$B, n_train=$n_train, n_epochs=$n_epochs, noise=$noise, lr=$lr")
    println("  chance test_acc = $(round(1/K, digits=3))")
    println()
    println(rpad("init_codes", 12), rpad("init_α", 10), rpad("init_β", 10),
            rpad("final_acc", 12), rpad("peak_acc", 12),
            rpad("eff_codes", 12), rpad("max_sim", 10),
            rpad("trained_α", 12), rpad("trained_β", 12))
    println("-" ^ 110)

    for codes_init in codes_grid, log_α in α_grid, log_β in β_grid
        r = train_one(; init_log_alpha = log_α,
                       init_log_beta  = log_β,
                       init_codes     = codes_init,
                       D, K, L, B, n_train, n_test, n_epochs, noise, lr, seed)
        row = (codes_init = codes_init,
               init_alpha = exp(log_α), init_beta = exp(log_β),
               final_acc = r.accs[end], peak_acc = maximum(r.accs),
               final_eff_codes = r.final_eff_codes,
               final_max_sim = r.final_max_sim,
               final_alpha = r.final_alpha, final_beta = r.final_beta)
        push!(results, row)
        println(rpad(string(codes_init), 12),
                rpad(round(exp(log_α), digits=2), 10),
                rpad(round(exp(log_β), digits=2), 10),
                rpad(round(row.final_acc, digits=3), 12),
                rpad(round(row.peak_acc,  digits=3), 12),
                rpad(round(row.final_eff_codes, digits=2), 12),
                rpad(round(row.final_max_sim, digits=3), 10),
                rpad(round(row.final_alpha, digits=3), 12),
                rpad(round(row.final_beta,  digits=3), 12))
    end

    println()
    by_acc = sort(results, by = r -> -r.peak_acc)
    println("Top 5 by peak accuracy:")
    for r in by_acc[1:min(5, length(by_acc))]
        println("  codes=:$(r.codes_init), init_α=$(round(r.init_alpha; digits=3)), init_β=$(round(r.init_beta; digits=3)) → ",
                "peak_acc=$(round(r.peak_acc; digits=3)), eff_codes=$(round(r.final_eff_codes; digits=2))")
    end

    return results
end

if abspath(PROGRAM_FILE) == @__FILE__
    attractor_sweep()
end
