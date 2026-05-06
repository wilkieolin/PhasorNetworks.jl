#!/usr/bin/env julia
#
# scripts/sfmnist.jl  —  sequential FashionMNIST benchmark for PhasorResonant
#
# Each 28×28 image is read pixel-by-pixel in row-major order, giving a
# length-784 sequence of complex unit-modulus phasors:
#
#     pixel value v ∈ [0, 1]   →   phase = 2v − 1   →   exp(iπ · phase)
#
# The model must classify the image based on the resonator state at the
# end of the sequence. This is the canonical "sequential MNIST"
# benchmark adapted to FashionMNIST (slightly harder, 10 classes, more
# texture).
#
# Compares two PhasorResonant configurations — both at shared carrier
# ω = 2π (per-channel ω rule):
#
#   :flat   — single uniform decay across all D channels.
#   :hippo  — log-spaced HiPPO-LegS λ spectrum across D channels;
#             slowest channel retains early pixels, fast channels
#             integrate recent context.
#
# Both modes scale `init_log_neg_lambda` so the slowest channel actually
# survives 784 timesteps (default `λ = -0.1` underflows past L≈870 in
# Float32 — see the long_range_ssm.jl notes for the underflow analysis).
#
# ---------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------
#   julia --project=scripts scripts/sfmnist.jl                 # default config
#   julia --project=scripts -e 'include("scripts/sfmnist.jl"); main_sfmnist(D=128, n_epochs=10)'
#
# First-time setup of the scripts environment:
#   julia --project=scripts -e 'using Pkg; Pkg.instantiate()'
#
# Memory: D=64, B=64 ≈ 0.7 GiB peak GPU; D=128, B=64 ≈ 2.7 GiB.

using PhasorNetworks, Lux, Zygote, Optimisers, CUDA, LuxCUDA
using MLDatasets: FashionMNIST
using OneHotArrays: onehotbatch
using Random: Xoshiro, AbstractRNG
using Statistics: mean

# ---------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------

"""
    pixel_to_phasor_batch(X) -> Array{ComplexF32, 3}

Convert an `H × W × B` image batch into a length-`H·W` complex sequence
per sample, suitable for `PhasorResonant`.

Pixel values are mapped `v ↦ exp(iπ · (2v − 1))` so v=0 ⇒ phase 1+0im
(no information / "off"), v=1 ⇒ phase −1+0im (opposite phase). The
ordering walks columns-then-rows ("row-major" within each image).
"""
function pixel_to_phasor_batch(X::AbstractArray{<:Real, 3})
    H, W, B = size(X)
    L = H * W
    # Permute (H, W, B) → (W, H, B) so vec along (W, H) gives row-major.
    Xp = permutedims(X, (2, 1, 3))
    flat = reshape(Xp, L, B)
    phases = 2f0 .* Float32.(flat) .- 1f0          # [-1, 1]
    cmplx  = exp.(1.0f0im .* Float32(π) .* phases) # unit modulus
    return reshape(cmplx, 1, L, B)                  # (C_in=1, L, B)
end

"""
    load_fashionmnist(; n_train, n_test) -> ((Xtr, ytr), (Xte, yte))

Load FashionMNIST (downloads on first call). `Xtr/Xte` are
`28 × 28 × N` Float32 in [0, 1]; labels are `Int` in `0..9`.
"""
function load_fashionmnist(; n_train::Int = 10_000, n_test::Int = 2_000)
    train = FashionMNIST(:train)
    test  = FashionMNIST(:test)
    n_train_avail = length(train.targets)
    n_test_avail  = length(test.targets)
    n_train = min(n_train, n_train_avail)
    n_test  = min(n_test, n_test_avail)
    Xtr = Float32.(train.features[:, :, 1:n_train])
    ytr = Int.(train.targets[1:n_train])
    Xte = Float32.(test.features[:, :, 1:n_test])
    yte = Int.(test.targets[1:n_test])
    return (Xtr, ytr), (Xte, yte)
end

function batch_iterator(X::Array{Float32, 3}, y::Vector{Int}, B::Int)
    N = length(y)
    out = Vector{Tuple{Array{ComplexF32, 3}, Vector{Int}}}()
    for i in 1:B:N
        e = min(i + B - 1, N)
        x_b = pixel_to_phasor_batch(X[:, :, i:e])
        y_b = y[i:e] .+ 1   # FashionMNIST labels are 0..9 → use 1..10
        push!(out, (x_b, y_b))
    end
    return out
end

# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------

"""
    LastStepDense(in_dims, out_dims) <: Lux.AbstractLuxLayer

Trainable classification head: takes the last timestep of a phase-valued
3-D sequence `(C, L, B)` and applies a dense map to `(out_dims, B)`
real-valued logits.
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
    build_sfmnist_model(mode, D; L=784, use_attention=false)

Build a chain `PhasorResonant → [SSMSelfAttention] → LastStepDense` for
the chosen λ layout (`:flat` or `:hippo`). Both share ω = 2π.

`init_log_neg_lambda` is chosen so the slowest channel survives the
full 784-step sequence (`λ_min · 784 > log(eps(Float32)) ≈ −87`).

When `use_attention = true`, an `SSMSelfAttention(D => D)` layer sits
between the encoder and the readout — adds a phase-domain mixing step
across the L=784 timesteps before the final-step pooling. After the
attention-layer refactor, this layer wraps `SingleHeadAttention` with
`PhasorDense` Q/K/V projections; the chain stays in the phase domain
end-to-end.
"""
function build_sfmnist_model(mode::Symbol, D::Int;
                              L::Int = 784,
                              use_attention::Bool = false)
    encoder = if mode === :flat
        # Single timescale tuned to retain the impulse over L steps.
        # α = 5/L gives exp(−5) ≈ 0.0067 retention at the readout.
        α = 5f0 / Float32(L)
        PhasorResonant(1 => D; init_log_neg_lambda = log(α))
    elseif mode === :hippo
        # HiPPO-LegS λ spectrum, rescaled so the slowest channel has
        # decay timescale ~L (retains the impulse) and the fastest ~5 steps.
        λ_h, _ = hippo_legs_diagonal(D)
        α_min = 1f0 / Float32(L)
        α_max = 5f0
        scale_min = α_min / minimum(abs.(λ_h))
        scale_max = α_max / maximum(abs.(λ_h))
        # Interpolate per-channel scale on log axis from min→max scaling.
        log_scales = range(log(scale_min), log(scale_max); length = D)
        αs = Float32.(abs.(λ_h) .* exp.(log_scales))
        PhasorResonant(1 => D; init_log_neg_lambda = log.(αs))
    else
        error("mode must be :flat or :hippo, got :$mode")
    end
    if use_attention
        return Chain(encoder,
                     SSMSelfAttention(D => D, normalize_to_unit_circle),
                     LastStepDense(D, 10))
    else
        return Chain(encoder, LastStepDense(D, 10))
    end
end

# ---------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------

function ce_loss(logits::AbstractArray, y_onehot::AbstractMatrix;
                 temperature::Real = 1f0)
    scaled = Float32(temperature) .* logits
    m = maximum(scaled; dims=1)
    log_softmax = scaled .- m .- log.(sum(exp.(scaled .- m); dims=1))
    per_batch_logp = vec(sum(y_onehot .* log_softmax; dims=1))
    return -mean(per_batch_logp)
end

function accuracy(logits::AbstractArray, labels::AbstractVector{<:Integer})
    preds = vec(getindex.(argmax(Array(logits); dims=1), 1))
    return mean(preds .== labels)
end

function train_epoch!(model, ps, st, opt_state, batches, n_classes; gdev)
    losses = Float32[]
    for (x, y) in batches
        x_d  = x |> gdev
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
    main_sfmnist(; D=128, B=64, n_epochs=10, lr=3e-3,
                  n_train=10000, n_test=2000, use_cuda=true, seed=0)

Train flat-init and HiPPO-init PhasorResonant on sequential FashionMNIST
and report per-epoch test accuracy for each mode.
"""
function main_sfmnist(; D::Int        = 128,
                       B::Int        = 64,
                       n_epochs::Int = 10,
                       lr::Real      = 3f-3,
                       n_train::Int  = 10_000,
                       n_test::Int   = 2_000,
                       use_attention::Bool = false,
                       use_cuda::Bool = true,
                       seed::Int     = 0)
    rng  = Xoshiro(seed)
    gdev = (use_cuda && CUDA.functional()) ? gpu_device() : cpu_device()

    @info "config" D B n_epochs lr n_train n_test use_attention device = string(gdev)

    @info "loading FashionMNIST..."
    (Xtr, ytr), (Xte, yte) = load_fashionmnist(; n_train, n_test)
    train_batches = batch_iterator(Xtr, ytr, B)
    test_batches  = batch_iterator(Xte, yte, B)
    @info "data" n_train_used = length(ytr) n_test_used = length(yte) L = 28*28 n_classes = 10

    results = Dict{Symbol, Vector{Float32}}()
    for mode in (:flat, :hippo)
        @info "training" mode
        model = build_sfmnist_model(mode, D; use_attention = use_attention)
        ps_cpu, st_cpu = Lux.setup(rng, model)
        ps = ps_cpu |> gdev
        st = st_cpu |> gdev
        opt_state = Optimisers.setup(Optimisers.Adam(Float32(lr)), ps)

        accs = Float32[]
        for epoch in 1:n_epochs
            ps, opt_state, losses = train_epoch!(model, ps, st, opt_state, train_batches, 10; gdev)
            test_acc = evaluate(model, ps, st, test_batches; gdev)
            push!(accs, test_acc)
            @info "epoch" mode epoch mean_loss = round(mean(losses), digits = 4) test_acc = round(test_acc, digits = 4)
        end
        results[mode] = accs
    end

    println()
    @info "summary"
    println("  D = $D, B = $B, L = 784, n_train = $(length(ytr)), n_test = $(length(yte))")
    println("  chance accuracy = 0.1")
    for mode in (:flat, :hippo)
        accs = results[mode]
        println("  $(rpad(string(mode), 8)) final test_acc = $(round(accs[end], digits=3))" *
                "    (per-epoch: $(round.(accs, digits=3)))")
    end
    return results
end

if abspath(PROGRAM_FILE) == @__FILE__
    main_sfmnist()
end
