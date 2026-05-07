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
using Statistics: mean, std

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

"""
    pixel_to_phase_batch(X) -> Array{Phase, 3}

Direct phase encoding (no ZOH SSM): map pixel `v ∈ [0, 1]` to a Phase
value `v/2 ∈ [0, 0.5]` (units of π, i.e. angles in `[0, π/2]`). Output
shape is `(C_in=1, L=H·W, B)`, suitable for the Phase 3D dispatch on
`PhasorDense`. This skips the per-encoder PhasorResonant layer entirely
— the network goes straight from pixel-as-phase into a learned dense
SSM via `causal_conv_dirac`.
"""
function pixel_to_phase_batch(X::AbstractArray{<:Real, 3})
    H, W, B = size(X)
    L = H * W
    Xp = permutedims(X, (2, 1, 3))
    flat = reshape(Xp, L, B)
    return reshape(Phase.(Float32.(flat) ./ 2f0), 1, L, B)
end

function batch_iterator_phase(X::Array{Float32, 3}, y::Vector{Int}, B::Int)
    N = length(y)
    out = Vector{Tuple{Array{Phase, 3}, Vector{Int}}}()
    for i in 1:B:N
        e = min(i + B - 1, N)
        push!(out, (pixel_to_phase_batch(X[:, :, i:e]), y[i:e] .+ 1))
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
    _encoder_lnl(mode, D; L=784) -> Vector{Float32}

Compute the per-channel `init_log_neg_lambda` vector that the encoder
layer should be initialized with so the slowest channel survives the
full L-step sequence. Shared between the ZOH (`PhasorResonant`) and
direct-phase (`PhasorDense`) encoders.

- `:flat`  — single-timescale `α = 5/L` (≈0.0067 retention at L).
- `:hippo` — HiPPO-LegS spectrum rescaled so the slowest channel has
  α_min = 1/L and the fastest α_max = 5 (log-interpolated per channel).
"""
function _encoder_lnl(mode::Symbol, D::Int; L::Int = 784)
    if mode === :flat
        α = 5f0 / Float32(L)
        return fill(Float32(log(α)), D)
    elseif mode === :hippo
        λ_h, _ = hippo_legs_diagonal(D)
        α_min = 1f0 / Float32(L)
        α_max = 5f0
        scale_min = α_min / minimum(abs.(λ_h))
        scale_max = α_max / maximum(abs.(λ_h))
        log_scales = range(log(scale_min), log(scale_max); length = D)
        αs = Float32.(abs.(λ_h) .* exp.(log_scales))
        return Float32.(log.(αs))
    else
        error("mode must be :flat or :hippo, got :$mode")
    end
end

"""
    build_sfmnist_model(mode, D; L=784, use_attention=false,
                         encoding=:zoh, init_scale=3f0)

Build a chain `Encoder → [SSMSelfAttention] → LastStepDense` for the
chosen λ layout (`:flat` or `:hippo`). All layers share ω = 2π.

`encoding` selects the input front-end:
- `:zoh`   — pixel `v ∈ [0,1]` → `exp(iπ·(2v-1))` complex sample, then
  `PhasorResonant(1 => D)` does ZOH SSM. (Original baseline.)
- `:phase` — pixel `v` → Phase `v/2 ∈ [0, 0.5]` directly, then
  `PhasorDense(1 => D)` integrates as a Dirac SSM (no separate
  encoder layer; uses `causal_conv_dirac` internally). Cheaper and
  treats the pixel sequence as spike-time arrivals into a single
  learnable SSM bank.

`init_scale` controls `score_scale` inside the optional
`SSMSelfAttention(D => D)` block: `exp(scale · sim)/d_k` where
`sim ∈ [-1, 1]` is the normalized phasor interference. Too low ⇒
all queries weight all keys nearly the same (no selection). Too
high ⇒ a tiny similarity gap dominates the output. Defaults to 3,
which previously worked better than 1 in this regime — track scores
during training to verify.
"""
function build_sfmnist_model(mode::Symbol, D::Int;
                              L::Int = 784,
                              use_attention::Bool = false,
                              encoding::Symbol = :zoh,
                              init_scale::Real = 3f0)
    lnl = _encoder_lnl(mode, D; L = L)
    encoder = if encoding === :zoh
        PhasorResonant(1 => D; init_log_neg_lambda = lnl)
    elseif encoding === :phase
        # init_mode is irrelevant when init_log_neg_lambda is supplied
        # (it overrides the mode default).
        PhasorDense(1 => D; init_log_neg_lambda = lnl)
    else
        error("encoding must be :zoh or :phase, got :$encoding")
    end
    if use_attention
        return Chain(encoder,
                     SSMSelfAttention(D => D, normalize_to_unit_circle;
                                      init_scale = init_scale),
                     LastStepDense(D, 10))
    else
        return Chain(encoder, LastStepDense(D, 10))
    end
end

# ---------------------------------------------------------------------
# Diagnostics: attention score statistics
# ---------------------------------------------------------------------

"""
    attention_score_stats(model, ps, st, x_batch) -> NamedTuple

Snapshot the SSMSelfAttention block's score statistics on a single
batch. Reaches into `model.layers[2].inner` (`SingleHeadAttention`),
manually replays Q/K projections, computes the raw similarity tensor
`(L_q, L_k, B)` and the post-`score_scale` weights, and returns
summary statistics — used to detect score collapse during training.

Returns:
- `raw_*`  — stats over the raw `similarity_outer` output (∈ ≈ [-1, 1]).
- `scaled_*` — stats over `exp(scale * sim) / d_k`.
- `per_query_std` — mean std-across-keys per query timestep. If this
  is near 0, every query is treating every key the same: collapse.
- `effective_keys` — `exp(H)` where `H = -Σ p log p` is the entropy of
  the per-query softmax-normalized scores; values close to `L_k`
  indicate uniform attention, values close to 1 indicate single-key
  selection.
"""
function attention_score_stats(model, ps, st, x_b)
    @assert length(model.layers) >= 2 "model must include encoder + attention layer"
    enc = model.layers[1]
    att = model.layers[2]
    @assert att isa SSMSelfAttention "layer 2 must be SSMSelfAttention; got $(typeof(att))"

    # Run encoder forward (no AD) to get phase tensor going into attention.
    enc_out, _ = enc(x_b, ps.layer_1, st.layer_1)

    # Reach into the SingleHeadAttention container for q/k/v projections.
    inner = att.inner
    inner_ps = ps.layer_2.inner
    inner_st = st.layer_2.inner
    q, _ = inner.q_proj(enc_out, inner_ps.q_proj, inner_st.q_proj)
    k, _ = inner.k_proj(enc_out, inner_ps.k_proj, inner_st.k_proj)

    # Raw similarity (Real) and post-scale weights.
    raw    = PhasorNetworks.similarity_outer(q, k, dims = 2)              # (L_q, L_k, B)
    scale  = Array(inner_ps.attention.scale)                                # length 1
    scaled = exp.(scale .* raw) ./ Float32(size(raw, 1))                   # same shape

    raw_h = Array(raw); sc_h = Array(scaled)

    # per-query std across keys (averaged over batch)
    per_q_std = mean(std(raw_h; dims = 2))

    # Effective number of keys per query via softmax entropy.
    # softmax across the key axis (dim=2): P[q,k,b] = exp(raw)/Σ_k exp(raw)
    # Use raw (not scaled) to avoid double-scaling — this measures
    # selectivity intrinsic to the projection geometry.
    m = maximum(raw_h; dims = 2)
    expr = exp.(raw_h .- m)
    P    = expr ./ sum(expr; dims = 2)
    H    = -dropdims(sum(P .* log.(P .+ 1f-30); dims = 2); dims = 2)        # (L_q, B)
    eff_keys = mean(exp.(H))

    return (
        raw_min     = minimum(raw_h),  raw_max  = maximum(raw_h),
        raw_mean    = mean(raw_h),     raw_std  = std(raw_h),
        scaled_min  = minimum(sc_h),   scaled_max = maximum(sc_h),
        scaled_mean = mean(sc_h),      scaled_std = std(sc_h),
        per_query_std = Float32(per_q_std),
        effective_keys = Float32(eff_keys),
        scale         = Float32(scale[1]),
    )
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
                       encoding::Symbol    = :zoh,
                       init_scale::Real    = 3f0,
                       track_scores::Bool  = false,
                       use_cuda::Bool = true,
                       seed::Int     = 0)
    rng  = Xoshiro(seed)
    gdev = (use_cuda && CUDA.functional()) ? gpu_device() : cpu_device()

    @info "config" D B n_epochs lr n_train n_test use_attention encoding init_scale device = string(gdev)

    @info "loading FashionMNIST..."
    (Xtr, ytr), (Xte, yte) = load_fashionmnist(; n_train, n_test)
    iter = encoding === :zoh ? batch_iterator : batch_iterator_phase
    train_batches = iter(Xtr, ytr, B)
    test_batches  = iter(Xte, yte, B)
    @info "data" n_train_used = length(ytr) n_test_used = length(yte) L = 28*28 n_classes = 10

    results = Dict{Symbol, Vector{Float32}}()
    for mode in (:flat, :hippo)
        @info "training" mode
        model = build_sfmnist_model(mode, D;
                                     use_attention = use_attention,
                                     encoding = encoding,
                                     init_scale = init_scale)
        ps_cpu, st_cpu = Lux.setup(rng, model)
        ps = ps_cpu |> gdev
        st = st_cpu |> gdev
        opt_state = Optimisers.setup(Optimisers.Adam(Float32(lr)), ps)

        # Held-out batch for score-stats snapshots (constant across epochs).
        score_probe = (use_attention && track_scores) ?
            (test_batches[1][1] |> gdev) : nothing

        accs = Float32[]
        for epoch in 1:n_epochs
            ps, opt_state, losses = train_epoch!(model, ps, st, opt_state, train_batches, 10; gdev)
            test_acc = evaluate(model, ps, st, test_batches; gdev)
            push!(accs, test_acc)
            @info "epoch" mode epoch mean_loss = round(mean(losses), digits = 4) test_acc = round(test_acc, digits = 4)

            if score_probe !== nothing
                stats = attention_score_stats(model, ps, st, score_probe)
                @info "scores" mode epoch scale=stats.scale raw_min=round(stats.raw_min, digits=3) raw_max=round(stats.raw_max, digits=3) raw_mean=round(stats.raw_mean, digits=3) raw_std=round(stats.raw_std, digits=4) per_q_std=round(stats.per_query_std, digits=4) eff_keys=round(stats.effective_keys, digits=2) scaled_min=round(stats.scaled_min, digits=5) scaled_max=round(stats.scaled_max, digits=5)
            end
        end
        results[mode] = accs
    end

    println()
    @info "summary"
    println("  D = $D, B = $B, L = 784, n_train = $(length(ytr)), n_test = $(length(yte))")
    println("  encoding = $encoding, use_attention = $use_attention, init_scale = $init_scale, lr = $lr")
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
