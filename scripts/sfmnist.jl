#!/usr/bin/env julia
#
# scripts/sfmnist.jl  —  sequential FashionMNIST benchmark for phasor SSMs
#
# Each 28×28 image is read pixel-by-pixel in row-major order, giving a
# length-784 sequence. The model must classify the image based on the
# state at the final timestep.
#
# Two encoder front-ends:
#   :phase  (default) — pixel v ∈ [0,1] → Phase v/2 ∈ [0, 0.5] → PhasorDense
#                       Dirac SSM. Treats pixels as spike-time arrivals.
#                       Outperformed :zoh in the May-2026 ablation.
#   :zoh              — pixel v → exp(iπ·(2v-1)) → PhasorResonant ZOH SSM.
#
# Two λ initialization modes (shared carrier ω = 2π per the per-channel ω rule):
#   :flat   — single uniform decay across all D channels (`α = 5/L`).
#   :hippo  — log-spaced HiPPO-LegS λ spectrum across D channels, rescaled
#             so slowest channel α_min = 1/L survives 784 steps.
#
# Optional Hopfield-style attractor mid-layer (use_attractor=true):
#   AttractorPhasorSSM(D=>D, n_codes) sits between encoder and readout. Each
#   timestep of the encoder's phase output is mixed by a linear SSM step
#   and then nudged toward the nearest of n_codes learnable code prototypes
#   via softmax-weighted retrieval. Replaces a previous SSMSelfAttention
#   experiment (transformer-style attention) that suffered score collapse
#   in this regime — the recurrent attractor formulation preserves phase
#   semantics and adds content-addressable persistence inside the SSM.
#
# ---------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------
#   julia --project=scripts scripts/sfmnist.jl                                         # default config
#   julia --project=scripts -e 'include("scripts/sfmnist.jl"); main_sfmnist(use_attractor=true, track_attractor=true)'
#
# First-time setup of the scripts environment:
#   julia --project=scripts -e 'using Pkg; Pkg.instantiate()'
#
# Memory (rough, GPU): no-attractor D=96, B=64 ≈ 1 GiB peak; with attractor
# D=96, B=16 ≈ 8 GiB peak (the per-step Dirac kick `enc :: (D, D, B)` is
# pinned across all L=784 timesteps in the recurrent loop). For larger B
# the attractor-mid-layer needs gradient checkpointing — TODO.

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
    build_sfmnist_model(mode, D; L=784, use_attractor=false,
                         encoding=:phase, n_codes=10,
                         attractor_log_neg_lambda=log(0.1))

Build a chain `Encoder → [AttractorPhasorSSM] → LastStepDense` for the
chosen λ layout (`:flat` or `:hippo`). All layers share ω = 2π.

`encoding` selects the input front-end:
- `:zoh`   — pixel `v ∈ [0,1]` → `exp(iπ·(2v-1))` complex sample, then
  `PhasorResonant(1 => D)` does ZOH SSM.
- `:phase` (default) — pixel `v` → Phase `v/2 ∈ [0, 0.5]` directly,
  then `PhasorDense(1 => D)` integrates as a Dirac SSM (no separate
  encoder layer; treats the pixel sequence as spike-time arrivals
  into a single learnable SSM bank). Phase encoding consistently
  outperformed ZOH in the May-2026 ablation (hippo final test_acc
  0.347 vs 0.328 with much faster loss decay).

`use_attractor = true` inserts an `AttractorPhasorSSM(D => D, n_codes)`
between encoder and readout. Each timestep of the encoder's phase
output is mixed by the linear SSM half then nudged toward the nearest
of `n_codes` learnable phasor codes via a Hopfield-style soft retrieval
(the pull strength α and softmax sharpness β are also trainable). This
provides selective, content-addressable persistence — the network can
learn to lock the state onto code prototypes that correspond to class
features rather than relying solely on the linear decay kernel. With
`use_attractor = false` the chain is `Encoder → LastStepDense` only.

`attractor_log_neg_lambda` sets the mid-layer's per-channel decay rate
(default `log(0.1)` = α=0.1, mild). The mid-layer doesn't need the
slow `1/L` decay the encoder uses — it's processing the encoder's
already-time-integrated output.
"""
function build_sfmnist_model(mode::Symbol, D::Int;
                              L::Int = 784,
                              use_attractor::Bool = false,
                              encoding::Symbol = :phase,
                              n_codes::Int = 10,
                              attractor_log_neg_lambda::Real = log(0.1))
    lnl = _encoder_lnl(mode, D; L = L)
    encoder = if encoding === :zoh
        PhasorResonant(1 => D; init_log_neg_lambda = lnl)
    elseif encoding === :phase
        PhasorDense(1 => D; init_log_neg_lambda = lnl)
    else
        error("encoding must be :zoh or :phase, got :$encoding")
    end
    if use_attractor
        return Chain(encoder,
                     AttractorPhasorSSM(D => D, n_codes;
                                         init_log_neg_lambda = attractor_log_neg_lambda,
                                         trainable_codes = true),
                     LastStepDense(D, 10))
    else
        return Chain(encoder, LastStepDense(D, 10))
    end
end

# ---------------------------------------------------------------------
# Diagnostics: attractor code utilization
# ---------------------------------------------------------------------

"""
    attractor_diagnostics(model, ps, st, x_batch) -> NamedTuple

Snapshot the AttractorPhasorSSM mid-layer's behavior on a held-out
batch. Runs the encoder forward, then *manually* runs the attractor
mid-layer to inspect:

- `mean_max_sim` — average (over batch × timesteps) of each state's
  highest similarity to any code. Higher ⇒ states are concentrated near
  codes. Random-output baseline ≈ 0.0; perfect lock-on = 1.0.
- `effective_codes` — `exp(H)` where `H` is the entropy of code-utilization
  across the batch's final-timestep states. Values near `n_codes` ⇒
  every code gets used; values near 1 ⇒ only one code wins everything
  (mode collapse). For a class-balanced batch we'd want this near
  `n_codes` (or `min(n_codes, n_classes)`).
- `alpha`, `beta` — current pull strength and softmax sharpness.
"""
function attractor_diagnostics(model, ps, st, x_b)
    @assert length(model.layers) >= 2 "model must include encoder + attractor"
    enc = model.layers[1]
    att = model.layers[2]
    @assert att isa AttractorPhasorSSM "layer 2 must be AttractorPhasorSSM; got $(typeof(att))"

    enc_out, _ = enc(x_b, ps.layer_1, st.layer_1)        # (D, L, B) Phase
    out, _     = att(enc_out, ps.layer_2, st.layer_2)    # (D, L, B) Phase

    codes_phase = att.trainable_codes ? ps.layer_2.codes : st.layer_2.codes
    codes_h     = Array(codes_phase)                      # (D, K) Phase on host
    out_h       = Array(out)                              # (D, L, B) Phase on host

    D, L, B = size(out_h)
    K       = size(codes_h, 2)

    # Flatten time × batch into a single "sample" axis for stats.
    states = reshape(out_h, D, L * B)                     # (D, L*B) Phase
    sims   = similarity_outer(states, codes_h; dims = 2)  # (K, L*B) Real
    max_sim = vec(maximum(sims; dims = 1))                # (L*B,)
    mean_max_sim = mean(max_sim)

    # Code utilization on FINAL timestep only (this is what readout sees).
    final_states = out_h[:, end, :]                       # (D, B)
    final_sims   = similarity_outer(final_states, codes_h; dims = 2)  # (K, B)
    winners      = vec(getindex.(argmax(final_sims; dims = 1), 1))    # (B,)
    counts       = [count(==(k), winners) for k in 1:K]
    p            = counts ./ B
    p_safe       = filter(>(0), p)
    H_codes      = -sum(pi -> pi * log(pi), p_safe; init = 0.0)
    eff_codes    = exp(H_codes)

    α = inv(one(Float32) + exp(-Array(ps.layer_2.log_alpha)[1]))
    β = exp(Array(ps.layer_2.log_beta)[1])

    return (mean_max_sim = Float32(mean_max_sim),
            effective_codes = Float32(eff_codes),
            alpha = Float32(α),
            beta  = Float32(β),
            n_codes = K)
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
    main_sfmnist(; D=96, B=16, n_epochs=10, lr=1e-3, n_train=10000,
                 n_test=2000, use_attractor=false, n_codes=10,
                 attractor_log_neg_lambda=log(0.1), encoding=:phase,
                 track_attractor=false, use_cuda=true, seed=0)

Train flat-init and HiPPO-init phasor SSMs on sequential FashionMNIST
and report per-epoch test accuracy for each mode.

Defaults reflect what worked best in the May-2026 ablations:
- `encoding = :phase` (PhasorDense Dirac path beats the ZOH PhasorResonant)
- `lr = 1e-3` (less aggressive than the prior 3e-3, more stable trajectory)
- `B = 16` (small enough that the optional AttractorPhasorSSM mid-layer's
  per-step `enc` tape — `O(D·in·L·B)` — fits in a 35 GiB hard cap; bump
  D=128, B=64 only after gradient checkpointing is added).

`use_attractor = true` adds an `AttractorPhasorSSM(D=>D, n_codes)` between
encoder and readout — selective recurrent persistence via Hopfield-style
soft retrieval. See `build_sfmnist_model` for details.

`track_attractor = true` snapshots `attractor_diagnostics` each epoch:
mean max-similarity to any code (concentration), effective number of
distinct codes used (mode collapse detector), and the live α / β values.
"""
function main_sfmnist(; D::Int        = 96,
                       B::Int        = 16,
                       n_epochs::Int = 10,
                       lr::Real      = 1f-3,
                       n_train::Int  = 10_000,
                       n_test::Int   = 2_000,
                       use_attractor::Bool = false,
                       n_codes::Int        = 10,
                       attractor_log_neg_lambda::Real = log(0.1),
                       encoding::Symbol    = :phase,
                       track_attractor::Bool = false,
                       use_cuda::Bool = true,
                       seed::Int     = 0)
    rng  = Xoshiro(seed)
    gdev = (use_cuda && CUDA.functional()) ? gpu_device() : cpu_device()

    @info "config" D B n_epochs lr n_train n_test use_attractor n_codes encoding device = string(gdev)

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
                                     use_attractor = use_attractor,
                                     encoding = encoding,
                                     n_codes = n_codes,
                                     attractor_log_neg_lambda = attractor_log_neg_lambda)
        ps_cpu, st_cpu = Lux.setup(rng, model)
        ps = ps_cpu |> gdev
        st = st_cpu |> gdev
        opt_state = Optimisers.setup(Optimisers.Adam(Float32(lr)), ps)

        # Held-out batch for attractor diagnostics (constant across epochs).
        diag_probe = (use_attractor && track_attractor) ?
            (test_batches[1][1] |> gdev) : nothing

        accs = Float32[]
        for epoch in 1:n_epochs
            ps, opt_state, losses = train_epoch!(model, ps, st, opt_state, train_batches, 10; gdev)
            test_acc = evaluate(model, ps, st, test_batches; gdev)
            push!(accs, test_acc)
            @info "epoch" mode epoch mean_loss = round(mean(losses), digits = 4) test_acc = round(test_acc, digits = 4)

            if diag_probe !== nothing
                stats = attractor_diagnostics(model, ps, st, diag_probe)
                @info "attractor" mode epoch alpha = round(stats.alpha, digits=3) beta = round(stats.beta, digits=3) mean_max_sim = round(stats.mean_max_sim, digits=3) effective_codes = round(stats.effective_codes, digits=2) n_codes = stats.n_codes
            end
        end
        results[mode] = accs
    end

    println()
    @info "summary"
    println("  D = $D, B = $B, L = 784, n_train = $(length(ytr)), n_test = $(length(yte))")
    println("  encoding = $encoding, use_attractor = $use_attractor, n_codes = $n_codes, lr = $lr")
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
