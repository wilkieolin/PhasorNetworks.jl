#!/usr/bin/env julia
#
# scripts/long_range_ssm.jl  —  long-range memory benchmark for PhasorResonant
#
# Synthetic "first-element recall" task: a single class-label phasor is
# presented at t=1, followed by L-1 zero (or low-noise) timesteps. The
# model must classify the original label from the resonator state at the
# end of the sequence.
#
# This task isolates long-range memory: the model has exactly one bit of
# information per sample, separated by L−1 timesteps from the readout.
# As L grows, the resonator must retain the t=1 impulse without it being
# washed out by decay.
#
# Compares two PhasorResonant initializations:
#
#   :uniform — ω linearly spread in [0.2, 2.5]; all channels share a
#              single decay rate (default `log(0.1)` ⇒ λ = −0.1). At
#              L≫10, exp(λ·L) underflows everywhere — the impulse is
#              lost. Should perform near-chance.
#
#   :hippo   — (λ, ω) from `hippo_legs_diagonal(D)`. The HiPPO-LegS
#              eigenvalue spectrum spans many decades of decay rates;
#              the slowest channels (|λ| ≪ 1) retain the impulse across
#              thousands of timesteps. Should learn the task.
#
# ---------------------------------------------------------------------
# Memory characteristics (forward + backward, FFT path, ComplexF32)
# ---------------------------------------------------------------------
# Measured GPU allocations on this dev box (NVIDIA GB10), Lux + Zygote:
#
#   D=32,  L=1024, B=64  →   620 MiB
#   D=64,  L=1024, B=64  →  1.2 GiB
#   D=64,  L=2048, B=64  →  2.4 GiB    (default — comfortably under 8 GiB)
#   D=128, L=4096, B=32  →  4.9 GiB
#   D=64,  L=8192, B=16  →  2.5 GiB    (long-L regime)
#
# Allocations scale linearly with D · L · B. The FFT pads to 2L, so peak
# transient is roughly k · D · 2L · B · 8 bytes (k≈10 across the chain).
# For modest networks (D ≤ 128, L ≤ 4096, B ≤ 32) the script fits in
# under 5 GiB.
#
# ---------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------
# From the package root:
#   julia --project=. scripts/long_range_ssm.jl
#
# Or interactively, after loading globals from the script:
#   include("scripts/long_range_ssm.jl")
#   main(L=4096, D=128, n_epochs=10)
#

using PhasorNetworks, Lux, Zygote, Optimisers, CUDA, LuxCUDA
using OneHotArrays: onehotbatch
using Random: Xoshiro, AbstractRNG
using Statistics: mean

# ---------------------------------------------------------------------
# Synthetic task
# ---------------------------------------------------------------------

"""
    first_element_recall(rng, n_classes, L, B; noise=0f0) -> (x, y)

Generate `B` examples of the first-element-recall task.

Each input is a length-`L` complex sequence with a class-label phasor at
t=1 (`exp(2πi · class/n_classes)`) and zeros (plus optional Gaussian
complex noise) at t=2..L.

Returns:
- `x :: Array{ComplexF32, 3}` of shape `(1, L, B)`.
- `y :: Vector{Int}` of length `B`, class labels in `1:n_classes`.
"""
function first_element_recall(rng::AbstractRNG, n_classes::Int, L::Int, B::Int;
                               noise::Real = 0f0)
    labels = rand(rng, 1:n_classes, B)
    x = if noise > 0
        ComplexF32.(noise) .* (randn(rng, ComplexF32, 1, L, B))
    else
        zeros(ComplexF32, 1, L, B)
    end
    for b in 1:B
        θ = 2f0 * Float32(π) * Float32(labels[b] - 1) / Float32(n_classes)
        x[1, 1, b] = exp(1im * θ)
    end
    return x, labels
end

# ---------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------

"""
    LastStepDense(in_dims, out_dims) <: Lux.AbstractLuxLayer

Trainable classification head: takes the last timestep of a phase-valued
3-D sequence `(C, L, B)` and applies a dense map to `(out_dims, B)`
real-valued logits. Lets the model fit the task without relying on
fixed random codes (as `Codebook` / `SSMReadout` do).
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
    last_step = Float32.(x[:, end, :])           # (C, B)
    return ps.W * last_step .+ ps.b, st
end

"""
    build_model(mode, in_dims, D, n_classes; omega_lo, omega_hi)

Build a Chain `PhasorResonant → LastStepDense` configured for the chosen
init `mode` (`:uniform` or `:hippo`).
"""
function build_model(mode::Symbol, in_dims::Int, D::Int, n_classes::Int;
                     omega_lo::Real = 0.2f0,
                     omega_hi::Real = 2.5f0)
    encoder = if mode === :uniform
        PhasorResonant(in_dims => D; omega_lo = omega_lo, omega_hi = omega_hi)
    elseif mode === :hippo
        λ_h, ω_h = hippo_legs_diagonal(D)
        PhasorResonant(in_dims => D;
                       omega = ω_h,
                       init_log_neg_lambda = log.(-λ_h))
    else
        error("mode must be :uniform or :hippo, got :$mode")
    end
    return Chain(encoder, LastStepDense(D, n_classes))
end

# ---------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------

"""
Cross-entropy on similarity logits in [-1, 1]. Temperature scales the
logits so the softmax has useful dynamic range. Vectorized via a
one-hot mask so it runs without scalar indexing on GPU.
"""
function ce_loss(logits::AbstractArray, y_onehot::AbstractMatrix;
                 temperature::Real = 5f0)
    scaled = Float32(temperature) .* logits
    m = maximum(scaled; dims=1)                                    # numerical-safety
    log_softmax = scaled .- m .- log.(sum(exp.(scaled .- m); dims=1))
    # Sum the correct-class log-prob per batch element via the one-hot mask.
    per_batch_logp = vec(sum(y_onehot .* log_softmax; dims=1))
    return -mean(per_batch_logp)
end

function accuracy(logits::AbstractArray, labels::AbstractVector{<:Integer})
    # Run on CPU — argmax + scalar comparison is fine off-device.
    preds = vec(getindex.(argmax(Array(logits); dims=1), 1))
    return mean(preds .== labels)
end

function train_epoch!(model, ps, st, opt_state, batches, n_classes; gdev)
    losses = Float32[]
    for (x, y) in batches
        x_d = x |> gdev
        # Build the one-hot target on the same device as the logits.
        y_oh = Float32.(onehotbatch(y, 1:n_classes)) |> gdev
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
# Main
# ---------------------------------------------------------------------

"""
    main(; n_classes=4, L=2048, D=64, B=32, n_train_batches=50,
          n_test_batches=10, n_epochs=5, lr=3e-3, use_cuda=true, seed=0)

Train uniform-init and HiPPO-init PhasorResonant on the first-element
recall task at sequence length `L`. Logs train loss and test accuracy
per epoch for each init mode.
"""
function main(; n_classes::Int = 4,
              L::Int = 2048,
              D::Int = 64,
              B::Int = 32,
              n_train_batches::Int = 50,
              n_test_batches::Int = 10,
              n_epochs::Int = 5,
              lr::Real = 3e-3,
              use_cuda::Bool = true,
              seed::Int = 0)
    rng  = Xoshiro(seed)
    gdev = (use_cuda && CUDA.functional()) ? gpu_device() : cpu_device()

    @info "config" n_classes L D B n_train_batches n_test_batches n_epochs lr device=string(gdev)

    train_data = [first_element_recall(rng, n_classes, L, B) for _ in 1:n_train_batches]
    test_data  = [first_element_recall(rng, n_classes, L, B) for _ in 1:n_test_batches]

    chance = 1.0 / n_classes
    @info "task" task = "first-element recall (one-shot impulse at t=1, zero rest)" chance

    results = Dict{Symbol, Vector{Float32}}()
    for mode in (:uniform, :hippo)
        @info "training" mode
        model = build_model(mode, 1, D, n_classes)
        ps_cpu, st_cpu = Lux.setup(rng, model)
        ps = ps_cpu |> gdev
        st = st_cpu |> gdev
        opt_state = Optimisers.setup(Optimisers.Adam(Float32(lr)), ps)

        accs = Float32[]
        for epoch in 1:n_epochs
            ps, opt_state, losses = train_epoch!(model, ps, st, opt_state, train_data, n_classes; gdev)
            test_acc = evaluate(model, ps, st, test_data; gdev)
            push!(accs, test_acc)
            @info "epoch" mode epoch mean_loss = mean(losses) test_acc
        end
        results[mode] = accs
    end

    println()
    @info "summary"
    println("  L = $L, D = $D, B = $B, n_classes = $n_classes")
    println("  chance = $(round(chance, digits=3))")
    for mode in (:uniform, :hippo)
        accs = results[mode]
        println("  $(rpad(string(mode), 8)) final test_acc = $(round(accs[end], digits=3))" *
                "    (per-epoch: $(round.(accs, digits=3)))")
    end
    return results
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
