# ================================================================
# SSM Support: Attention, Encoding, and Spiking Helpers
# ================================================================
#
# Kernel math (phasor_kernel, causal_conv, hippo_legs_diagonal) is in kernels.jl.
# The former PhasorSSM struct has been unified into PhasorDense (network.jl);
# there is NO PhasorSSM constructor. Use
#   PhasorDense(in => out, act; init_mode=:default|:hippo, use_bias=false)
# instead. This file keeps: SSMReadout, attention layers, encoding, and
# spiking helpers.


# ================================================================
# 2. SSM Readout Layer (Codebook-First)
# ================================================================

"""
    SSMReadout(hidden_dims => n_classes; readout_frac=0.25)

Temporal readout layer that applies codebook similarity at each timestep
before averaging, avoiding the phase-cancellation problem of averaging
rotating complex vectors.

The complex membrane potentials rotate at each oscillator's angular frequency
ω.  Averaging these rotating phasors directly causes destructive interference
(the mean tends toward zero when the readout window spans full rotations).
Instead, this layer:

1. Normalizes to the unit circle and extracts phase at each timestep
2. Computes cosine similarity against codebook prototypes at each timestep
   (similarity is a rotation-invariant scalar)
3. Averages the resulting scalar logits over the readout window

This is equivalent to asking "at every moment in time, how well does the
current phase pattern match each class?" and averaging that confidence.

Input:  (C × L × B) complex  (membrane potentials over time)
Output: (n_classes × B) Float32  (averaged similarity logits)

# Arguments
- `hidden_dims => n_classes` — Hidden dimension (must match SSM output) and
  number of classification targets.
- `readout_frac` — Fraction of final time steps to average over (`:mean`
  pooling only). Default 0.25.
- `pool` — Temporal pooling mode. `:mean` (default) averages the cosine
  similarity over the last `readout_frac` window. `:logsumexp` takes a smooth
  max over the **whole clip** — `(1/κ)·(logsumexp_t(κ·s) − log L)` — the
  keyword-spotting readout ("is the class present at *any* timestep"). Ports the
  PyTorch Tier-1 readout `phasor_torch/layers/ssm_readout.py:83`.
- `lse_kappa` — Sharpness κ for `:logsumexp` (→ max as κ→∞, → mean as κ→0).
  Default 10.0. Ignored for `:mean`.
"""
struct SSMReadout <: Lux.AbstractLuxLayer
    hidden_dims::Int
    n_classes::Int
    readout_frac::Float32
    pool::Symbol
    lse_kappa::Float32
end

function SSMReadout(dims::Pair{Int,Int}; readout_frac::Float32=0.25f0,
                    pool::Symbol=:mean, lse_kappa::Float32=10.0f0)
    pool in (:mean, :logsumexp) ||
        throw(ArgumentError("pool must be :mean or :logsumexp, got :$pool"))
    return SSMReadout(dims.first, dims.second, readout_frac, pool, lse_kappa)
end

Lux.initialparameters(::AbstractRNG, ::SSMReadout) = NamedTuple()

function Lux.initialstates(rng::AbstractRNG, l::SSMReadout)
    return (codes = random_symbols(rng, (l.hidden_dims, l.n_classes)),)
end

# Pool per-timestep code similarities into (n_classes × B) logits, branching on
# `l.pool`. Shared by the Complex-3D and Phase-3D dispatches (both first map to a
# (C × L × B) Phase array). cos(π·(p−c)) is 2-periodic, so working in raw Float32
# (no phase wrap) is identical to Phase subtraction and matches the PyTorch port.
function _readout_pool(l::SSMReadout, phases::AbstractArray{<:Phase, 3}, codes)
    C, L, B = size(phases)
    n_cls = size(codes, 2)
    c = reshape(Float32.(codes), C, n_cls, 1, 1)         # C × n_cls × 1 × 1

    if l.pool == :logsumexp
        # Smooth max over the WHOLE clip (keyword-spotting readout).
        p = reshape(Float32.(phases), C, 1, L, B)        # C × 1 × L × B
        cos_diff = cos.(pi_f32 .* (p .- c))              # C × n_cls × L × B
        sims_per_step = dropdims(mean(cos_diff; dims=1); dims=1)  # n_cls × L × B
        k = l.lse_kappa
        ks = k .* sims_per_step
        m = maximum(ks; dims=2)                          # n_cls × 1 × B (stable)
        lse = dropdims(m .+ log.(sum(exp.(ks .- m); dims=2)); dims=2)  # n_cls × B
        return (lse .- log(Float32(L))) ./ k
    end

    # :mean — average over the last `readout_frac` window.
    t0 = max(1, L - max(1, round(Int, L * l.readout_frac)) + 1)
    W = L - t0 + 1
    p = reshape(Float32.(phases[:, t0:L, :]), C, 1, W, B)  # C × 1 × W × B
    cos_diff = cos.(pi_f32 .* (p .- c))                    # C × n_cls × W × B
    sims_per_step = mean(cos_diff; dims=1)                 # 1 × n_cls × W × B
    sims_avg = mean(sims_per_step; dims=3)                 # 1 × n_cls × 1 × B
    return dropdims(sims_avg; dims=(1, 3))                 # n_cls × B
end

function (l::SSMReadout)(z::AbstractArray{<:Complex, 3}, ps::LuxParams, st::NamedTuple)
    # Extract phase at each timestep, then pool. (normalize is per-element, so
    # normalizing the whole clip and windowing inside _readout_pool is identical
    # to the old window-then-normalize for :mean.)
    phases = complex_to_angle(normalize_to_unit_circle(z))  # C × L × B  Phase
    return _readout_pool(l, phases, st.codes), st          # n_classes × B
end

function (l::SSMReadout)(x::AbstractArray{<:Phase, 3}, ps::LuxParams, st::NamedTuple)
    # Phase input: already normalized, skip normalize_to_unit_circle.
    return _readout_pool(l, x, st.codes), st
end

# ================================================================
# 3. PSK Encoding
# ================================================================

"""
    psk_encode(images; n_repeats=1) -> ComplexF32 array (C × L × B)

Phase-shift-key encode grayscale images as complex time series.
Columns = channels (C=W), rows = time steps (L=H×n_repeats).
Pixel value v∈[0,1] → phase θ = 2v-1 ∈ [-1,1] (π-radians) → exp(iπθ).

# Arguments
- `images::AbstractArray{<:Real, 3}` — (H × W × B) grayscale images in [0,1].
- `n_repeats::Int` — Number of times to repeat the time dimension. Default 1.

# Returns
ComplexF32 array (W × H*n_repeats × B).
"""
function psk_encode(images::AbstractArray{<:Real, 3}; n_repeats::Int=1)
    H, W, B = size(images)
    phases = 2f0 .* images .- 1f0                       # [0,1] → [-1,1]
    phases_ct = permutedims(phases, (2, 1, 3))           # channels × time × batch
    if n_repeats > 1
        phases_ct = repeat(phases_ct, 1, n_repeats, 1)
    end
    return angle_to_complex(phases_ct)
end

# ================================================================
# 4. Impulse Encoding
# ================================================================

"""
    impulse_encode(images; substeps=4) -> ComplexF32 array (C × L × B)

Encode pixel values as temporally-shifted real-valued impulse currents.

Each row of the image gets `substeps` discrete time steps (one "period").
The pixel's phase determines WHERE within those substeps the impulse fires:
  pixel v ∈ [0,1] → phase θ = 2v-1 → spike time t = (θ+1)/2 · T

A von-Mises-shaped pulse centered at the spike time produces a real current.
Total sequence length L = H × substeps (e.g. 28 × 4 = 112).

# Arguments
- `images::AbstractArray{<:Real, 3}` — (H × W × B) grayscale images in [0,1].
- `substeps::Int` — Number of substeps per row. Default 4.

# Returns
ComplexF32 array (W × H*substeps × B).
"""
function impulse_encode(images::AbstractArray{<:Real, 3}; substeps::Int=4)
    return ignore_derivatives() do
        _impulse_encode_impl(images; substeps)
    end
end

function _impulse_encode_impl(images::AbstractArray{<:Real, 3}; substeps::Int=4)
    H, W, B = size(images)
    C = W
    T = Float32(substeps)
    t_sigma = T * 0.1f0
    kappa = 1f0 / (1f0 - cos(2f0 * Float32(π) * t_sigma / T))

    phases = 2f0 .* images .- 1f0                     # H × W × B → [-1,1]
    phases_ct = permutedims(phases, (2, 1, 3))          # C × H × B

    # spike_times[c, row, b] = position in [0, T) within the row's period
    spike_times = (phases_ct .+ 1f0) ./ 2f0 .* T       # C × H × B

    # Build substep time indices on the same device as input
    ts_dev = similar(phases, Float32, substeps)
    copyto!(ts_dev, Float32.(1:substeps))
    ts_sub = reshape(ts_dev, 1, substeps, 1)              # 1 × substeps × 1

    slices = map(1:H) do r
        t_spike = spike_times[:, r:r, :]                # C × 1 × B
        dt = ts_sub .- t_spike                          # C × substeps × B
        exp.(kappa .* (cos.(2f0 * Float32(π) .* dt ./ T) .- 1f0))
    end
    signal = cat(slices...; dims=2)                     # C × L × B

    return complex.(signal, zero(signal))
end

# ================================================================
# 5. SSM Cross-Attention Layer
# ================================================================

"""
    SSMCrossAttention(in_dims => d_model, n_keys, activation; init_scale=3f0)

Cross-attention layer that pools a length-`L` phase sequence onto
`n_keys` learned key prototypes. Q and V projections go through
[`PhasorDense`](@ref) (Phase 3D dispatch ⇒ Dirac SSM dynamics — Q and V
each evolve under per-channel oscillator dynamics on the way in); the
attention compute itself is delegated to [`attend`](@ref).

This layer is a thin composition of `PhasorDense` × 2 (Q, V projections)
+ stored learnable `Phase` keys + a trainable scalar scale, applied via
the shared [`attend`](@ref) primitive — the same one [`PhasorAttention`]
(@ref) uses. There's no special attention math here.

**Note:** The temporal dimension changes from L to `n_keys`. Set
`n_keys = L` to preserve the temporal dimension, or pick a different
value as a bottleneck / pooling mechanism.

# Arguments
- `in_dims => d_model` — Input channel dimension and output channel dim.
- `n_keys::Int` — Number of stored key prototypes.
- `activation` — Applied after attention. Default `normalize_to_unit_circle`.
- `init_scale::Real` — Initial value of the trainable attention scale.

# Trainable parameters
- `q_proj`, `v_proj` — `PhasorDense` parameter trees (`weight`,
  `log_neg_lambda`; bias-free).
- `keys` (Phase, d_model × n_keys) — Stored key prototypes.
- `scale` (Float32, length 1) — Exponential score-scaling factor.

# Data flow
```
Input (C_in × L × B) Phase
  → Q = q_proj(x)         (d_model × L × B) Phase  (Dirac SSM)
  → V = v_proj(x)         (d_model × L × B) Phase  (Dirac SSM)
  → K = ps.keys broadcast to (d_model × n_keys × B)
  → out = attend(Q, K, V; scale)  (d_model × n_keys × B) Phase
  → activation
```
"""
struct SSMCrossAttention <: Lux.AbstractLuxLayer
    in_dims::Int
    d_model::Int
    n_keys::Int
    activation::Function
    q_proj::PhasorDense
    v_proj::PhasorDense
    init_scale::Float32
end

function SSMCrossAttention(dims::Pair{Int,Int}, n_keys::Int,
                           act = normalize_to_unit_circle;
                           init_scale::Real = 3f0)
    in_dims, d_model = dims.first, dims.second
    q_proj = PhasorDense(in_dims => d_model; use_bias = false)
    v_proj = PhasorDense(in_dims => d_model; use_bias = false)
    return SSMCrossAttention(in_dims, d_model, n_keys, act,
                             q_proj, v_proj, Float32(init_scale))
end

function Lux.initialparameters(rng::AbstractRNG, l::SSMCrossAttention)
    keys  = Phase.(2f0 .* rand(rng, Float32, l.d_model, l.n_keys) .- 1f0)
    scale = Float32[l.init_scale]
    return (q_proj = Lux.initialparameters(rng, l.q_proj),
            v_proj = Lux.initialparameters(rng, l.v_proj),
            keys   = keys,
            scale  = scale)
end

function Lux.initialstates(rng::AbstractRNG, l::SSMCrossAttention)
    return (q_proj = Lux.initialstates(rng, l.q_proj),
            v_proj = Lux.initialstates(rng, l.v_proj))
end

function Lux.parameterlength(l::SSMCrossAttention)
    return Lux.parameterlength(l.q_proj) +
           Lux.parameterlength(l.v_proj) +
           l.d_model * l.n_keys +    # keys
           1                          # scale
end

function (l::SSMCrossAttention)(x::AbstractArray{<:Phase, 3}, ps::LuxParams, st::NamedTuple)
    Q, _ = l.q_proj(x, ps.q_proj, st.q_proj)              # (d_model, L, B) Phase
    V, _ = l.v_proj(x, ps.v_proj, st.v_proj)              # (d_model, L, B) Phase
    B    = size(x, 3)
    # Expand stored keys to (d_model, n_keys, B) for similarity_outer.
    K    = repeat(reshape(ps.keys, l.d_model, l.n_keys, 1), 1, 1, B)
    out, _ = attend(Q, K, V; scale = ps.scale)            # Phase 3D, attention pooled
    return _apply_phase_activation(l.activation, out), st
end

# Complex 3D back-compat: trampoline through the Phase 3D path.
function (l::SSMCrossAttention)(x::AbstractArray{<:Complex, 3}, ps::LuxParams, st::NamedTuple)
    y_phase, st_new = l(complex_to_angle(x), ps, st)
    return angle_to_complex(y_phase), st_new
end

# ================================================================
# 6. SSM Self-Attention Layer
# ================================================================

"""
    SSMSelfAttention(in_dims => d_model, activation; init_scale=3f0)

Self-attention layer that projects an input phase sequence into queries,
keys, and values via three [`PhasorDense`](@ref) layers and runs the
standard scaled dot-product attention via [`attend`](@ref).

This is a thin wrapper over [`SingleHeadAttention`](@ref) configured with
`PhasorDense` projections and an identity output projection. The
projections use Phase 3D dispatch ⇒ Q, K, V each evolve under per-channel
oscillator dynamics (Dirac SSM) on the way in. The attention compute
itself is the same `attend` function used by every other phasor
attention path.

# Arguments
- `in_dims => d_model` — Input and output channel dimensions.
- `activation` — Applied after attention. Default `normalize_to_unit_circle`.
- `init_scale::Real` — Initial value of the trainable attention scale.

# Trainable parameters
- `inner` — Container holding the inner `SingleHeadAttention`'s
  parameter tree:
  - `inner.q_proj`, `inner.k_proj`, `inner.v_proj` — `PhasorDense`
    parameter trees (`weight`, `log_neg_lambda`; bias-free).
  - `inner.attention.scale` — trainable Float32 vector of length 1.
  - `inner.out_proj` — empty (identity).

# Data flow
```
Input (C_in × L × B) Phase
  → Q = q_proj(x), K = k_proj(x), V = v_proj(x)   (d_model × L × B) Phase
  → out = attend(Q, K, V; scale)                  (d_model × L × B) Phase
  → activation
```
"""
struct SSMSelfAttention <: Lux.AbstractLuxLayer
    in_dims::Int
    d_model::Int
    activation::Function
    inner::SingleHeadAttention
end

function SSMSelfAttention(dims::Pair{Int,Int}, act = normalize_to_unit_circle;
                          init_scale::Real = 3f0)
    in_dims, d_model = dims.first, dims.second
    inner = SingleHeadAttention(in_dims, d_model;
                                q_proj  = PhasorDense(in_dims => d_model; use_bias = false),
                                k_proj  = PhasorDense(in_dims => d_model; use_bias = false),
                                v_proj  = PhasorDense(in_dims => d_model; use_bias = false),
                                out_proj = identity_layer,
                                scale   = Float32(init_scale))
    return SSMSelfAttention(in_dims, d_model, act, inner)
end

Lux.initialparameters(rng::AbstractRNG, l::SSMSelfAttention) =
    (inner = Lux.initialparameters(rng, l.inner),)
Lux.initialstates(rng::AbstractRNG, l::SSMSelfAttention) =
    (inner = Lux.initialstates(rng, l.inner),)
Lux.parameterlength(l::SSMSelfAttention) = Lux.parameterlength(l.inner)

function (l::SSMSelfAttention)(x::AbstractArray{<:Phase, 3}, ps::LuxParams, st::NamedTuple)
    y, _ = l.inner(x, x, ps.inner, st.inner)        # Phase 3D output (out_proj=identity)
    return _apply_phase_activation(l.activation, y), st
end

# Complex 3D back-compat: trampoline through the Phase 3D path.
function (l::SSMSelfAttention)(x::AbstractArray{<:Complex, 3}, ps::LuxParams, st::NamedTuple)
    y_phase, st_new = l(complex_to_angle(x), ps, st)
    return angle_to_complex(y_phase), st_new
end

# Apply an activation to a Phase-typed output. For the angle-preserving
# default (`normalize_to_unit_circle`) and `identity`, no work is
# required because the input is already on the unit circle. Other
# activations are lifted to complex via `angle_to_complex`, applied,
# and lowered back to phase.
function _apply_phase_activation(activation, y::AbstractArray{<:Phase, 3})
    if activation === normalize_to_unit_circle || activation === identity
        return y
    else
        return complex_to_angle(activation(angle_to_complex(y)))
    end
end

# ================================================================
# 6b. Phasor Local Self-Attention (LSA)
# ================================================================

"""
    PhasorLSA(in_dims => d_model, n_heads, activation; init_scale=3f0, init_mode=:hippo, spk_args=SpikingArgs())

Local Self-Attention layer for the Phasor SSM. Computes the attention
score *across heads* (axis `H`) instead of *across time* (axis `L`), so
the operation is pointwise in `L` and inherits the three-mode
(discrete / continuous / parallel) equivalence of the surrounding
state-space layers. See `docs/local_attention_derivation.tex` for the
formal definitions and the equivalence proof.

Input is projected through three `PhasorDense` layers (Q, K, V), then
reshaped from `(D, L, B)` to `(D_h, H, L, B)`. A head-axis Fourier-HRR
score `(H, H, L, B)` is computed via
[`similarity_outer_heads`](@ref), scaled exponentially, and used to mix
the V tensor in the complex domain. Output shape: `(D, L, B)` Phase.

# Arguments
- `in_dims => d_model` — Input and output channel widths. `d_model`
  must be divisible by `n_heads`.
- `n_heads::Int` — Number of attention heads.
- `activation` — Phase-space activation applied to the output. Default
  `normalize_to_unit_circle`.

# Keyword arguments
- `init_scale::Real = 3f0` — Initial value of the trainable scalar
  `β` (the exponential's inverse temperature).
- `init_mode::Symbol = :default` — `PhasorDense` λ-init mode for the Q/K/V
  projections. Defaults to `:default` (uniform, single-timescale) so the read
  heads stay sharp for content routing; the MQAR ablation
  (`results/xform_mqar/`) shows `:hippo` here *hurts* long-range recall.
- `spk_args::SpikingArgs = SpikingArgs()` — Shared spiking dynamics
  for the projections.

# Trainable parameters
- `q_proj`, `k_proj`, `v_proj` — bias-free `PhasorDense` parameter
  trees (`weight`, `log_neg_lambda`).
- `scale` — 1-element `Vector{Float32}`.
"""
struct PhasorLSA <: Lux.AbstractLuxLayer
    in_dims::Int
    d_model::Int
    n_heads::Int
    activation::Function
    q_proj::PhasorDense
    k_proj::PhasorDense
    v_proj::PhasorDense
    init_scale::Float32
    full_d_heads::Bool   # false: heads are Dh=d_model/H slices (default). true: each
                         # head is a full d_model projection (proj d_model*H), heads
                         # combined by VSA bundling — the "don't slice the symbol" variant.
end

function PhasorLSA(dims::Pair{Int,Int}, n_heads::Int,
                   act = normalize_to_unit_circle;
                   init_scale::Real = 3f0,
                   init_mode::Symbol = :default,
                   full_d_heads::Bool = false,
                   spk_args::SpikingArgs = SpikingArgs())
    in_dims, d_model = dims.first, dims.second
    @assert d_model % n_heads == 0 "d_model ($d_model) must be divisible by n_heads ($n_heads)"
    proj_out = full_d_heads ? d_model * n_heads : d_model
    q = PhasorDense(in_dims => proj_out; use_bias = false, init_mode = init_mode, spk_args = spk_args)
    k = PhasorDense(in_dims => proj_out; use_bias = false, init_mode = init_mode, spk_args = spk_args)
    v = PhasorDense(in_dims => proj_out; use_bias = false, init_mode = init_mode, spk_args = spk_args)
    return PhasorLSA(in_dims, d_model, n_heads, act, q, k, v, Float32(init_scale), full_d_heads)
end

function Lux.initialparameters(rng::AbstractRNG, l::PhasorLSA)
    return (q_proj = Lux.initialparameters(rng, l.q_proj),
            k_proj = Lux.initialparameters(rng, l.k_proj),
            v_proj = Lux.initialparameters(rng, l.v_proj),
            scale  = Float32[l.init_scale])
end

function Lux.initialstates(rng::AbstractRNG, l::PhasorLSA)
    return (q_proj = Lux.initialstates(rng, l.q_proj),
            k_proj = Lux.initialstates(rng, l.k_proj),
            v_proj = Lux.initialstates(rng, l.v_proj))
end

function Lux.parameterlength(l::PhasorLSA)
    return Lux.parameterlength(l.q_proj) +
           Lux.parameterlength(l.k_proj) +
           Lux.parameterlength(l.v_proj) + 1
end

# Internal helper: bundle V over heads using the head-similarity weights.
#   Vc       :: (Dh, H, L, B) Complex
#   weights  :: (H, H, L, B) Real,  weights[h, h', l, b] = w_{h h'}^{(l,b)}
# Returns Y :: (Dh, H, L, B) Complex with
#   Y[:, h, l, b] = Σ_{h'} weights[h, h', l, b] · Vc[:, h', l, b].
function _lsa_head_mix(Vc::AbstractArray{<:Complex, 4}, weights::AbstractArray{<:Real, 4})
    Dh, H, L, B = size(Vc)
    Vc_r = reshape(Vc, Dh, H, L * B)                              # (Dh, H, L*B)
    # batched_mul on the last axis: (Dh, H) * (H, H) → (Dh, H) per (l, b).
    # The Vc factor's second axis is h'; weights' first axis (after transposing
    # along (1,2)) is h' → align so the contraction sums over h'.
    W_r  = reshape(permutedims(weights, (2, 1, 3, 4)), H, H, L * B)
    Y_r  = batched_mul(Vc_r, W_r)                                 # (Dh, H, L*B)
    return reshape(Y_r, Dh, H, L, B)
end

# (i) 3D Phase — the workhorse path.
function (l::PhasorLSA)(x::AbstractArray{<:Phase, 3}, ps::LuxParams, st::NamedTuple)
    Q, _ = l.q_proj(x, ps.q_proj, st.q_proj)             # (P, L, B) Phase, P=D or D*H
    K, _ = l.k_proj(x, ps.k_proj, st.k_proj)
    V, _ = l.v_proj(x, ps.v_proj, st.v_proj)

    _, L, B = size(Q)
    H  = l.n_heads
    D  = l.d_model
    # Per-head width: full-D (each head sees the whole symbol) or Dh=D/H slice.
    Dh = l.full_d_heads ? D : D ÷ H
    Qh = reshape(Q, Dh, H, L, B)
    Kh = reshape(K, Dh, H, L, B)
    Vh = reshape(V, Dh, H, L, B)

    scores  = similarity_outer_heads(Qh, Kh)             # (H, H, L, B) Float32
    weights = exp.(ps.scale .* scores) ./ Float32(H)      # (H, H, L, B)

    Vc = angle_to_complex(Vh)                             # (Dh, H, L, B) Complex
    Y  = _lsa_head_mix(Vc, weights)                       # (Dh, H, L, B) Complex
    # Combine heads: full-D → VSA bundle (sum) over heads → (D,L,B);
    #                sliced → concat heads (reshape) → (D,L,B).
    Y  = l.full_d_heads ? dropdims(sum(Y, dims = 2); dims = 2) : reshape(Y, D, L, B)
    Y_phase = complex_to_angle(Y)                         # (D, L, B) Phase

    return _apply_phase_activation(l.activation, Y_phase), st
end

# (ii) 2D Phase — single-slice; wrap to 3D with L=1.
function (l::PhasorLSA)(x::AbstractArray{<:Phase, 2}, ps::LuxParams, st::NamedTuple)
    x3 = reshape(x, size(x, 1), 1, size(x, 2))
    y3, st2 = l(x3, ps, st)
    return dropdims(y3, dims=2), st2
end

# (iii) Complex 3D back-compat — trampoline through Phase 3D.
function (l::PhasorLSA)(x::AbstractArray{<:Complex, 3}, ps::LuxParams, st::NamedTuple)
    y_phase, st2 = l(complex_to_angle(x), ps, st)
    return angle_to_complex(y_phase), st2
end

# (iv) SpikingCall — trampoline to CurrentCall.
function (l::PhasorLSA)(x::SpikingCall, ps::LuxParams, st::NamedTuple)
    return l(CurrentCall(x), ps, st)
end

# (v) CurrentCall — reconstruct 3D phase from the ODE solution, then 3D path.
function (l::PhasorLSA)(x::CurrentCall, ps::LuxParams, st::NamedTuple)
    L = round(Int, (x.t_span[2] - x.t_span[1]) / x.spk_args.t_period)
    z_3d = reconstruct_from_current(x, L, x.spk_args)
    return l(z_3d, ps, st)
end

# ================================================================
# 6c. Phasor Local Cross-Attention (LCA)
# ================================================================

"""
    PhasorLCA(in_dims => d_model, n_heads, n_anchors, activation; init_scale=3f0, init_mode=:hippo, spk_args=SpikingArgs())

Local Cross-Attention layer with a trainable phase anchor bank. Computes
a Hopfield-style content-addressable lookup against the anchors and
applies the retrieved bundle as a *binding rotation* to an input-derived
value. The score is computed across heads at each `(l, b)` slice — like
[`PhasorLSA`](@ref), the operation is pointwise in `L` and inherits the
three-mode equivalence of the surrounding state-space layers.

# Forward (Phase 3D)

1. `K = k_proj(x)` and `V = v_proj(x)`, both `(D, L, B)` Phase, via
   bias-free [`PhasorDense`](@ref) projections.
2. Reshape `K`, `V` → `(D_h, H, L, B)`; reshape the trainable anchor
   bank `(D, A) → (D_h, H, A)`.
3. Score `(A, H, L, B)` via [`similarity_outer_heads`](@ref):
   `s[a, h, l, b] = sim(anchor[:, h, a], K[:, h, l, b])`.
4. Weights `w = exp(β · s) / A` (per-anchor; the per-head divisor is
   implicit in the bundle).
5. Per `(h, l, b)`, bundle the anchors in the complex domain:
   `Bundle[:, h, l, b] = Σ_a w[a, h, l, b] · exp(iπ · anchor[:, h, a])`.
6. **Bind V with the anchor bundle**:
   `Y_complex[:, h, l, b] = exp(iπ · V[:, h, l, b]) ⊙ Bundle[:, h, l, b]`
   (element-wise complex multiplication = phase addition; the canonical
   VSA binding operation).
7. Reshape `(D_h, H, L, B)` → `(D, L, B)`, extract phase, apply
   activation.

# Design notes

This binding form is non-degenerate (distinct `(l, b)` produce distinct
output phases) because the anchor bundle is itself a complex-domain
weighted superposition over `a`. Setting V to a zero phase recovers the
pure Hopfield-retrieval form of `docs/local_attention_derivation.tex`,
Proposition 3. A future ablation may add (i) a separate trainable V
anchor bank, or (ii) a trainable `(A, H, H)` head-mix tensor `M` for
richer cross-head interaction.

# Arguments
- `in_dims => d_model` — Input and output channel widths. `d_model`
  must be divisible by `n_heads`.
- `n_heads::Int` — Number of attention heads.
- `n_anchors::Int` — Size of the stored anchor bank `A`.
- `activation` — Phase-space activation applied to the output. Default
  `normalize_to_unit_circle`.

# Keyword arguments
- `init_scale::Real = 3f0` — Initial value of the trainable scalar `β`.
- `init_mode::Symbol = :default` — `PhasorDense` λ-init mode for the K/V
  projections; uniform read heads (see `PhasorLSA` / `results/xform_mqar/`).
- `spk_args::SpikingArgs = SpikingArgs()` — Shared spiking dynamics.

# Trainable parameters
- `k_proj`, `v_proj` — bias-free `PhasorDense` parameter trees.
- `anchors` — `(D, A)` Phase.
- `scale` — 1-element `Vector{Float32}`.
"""
struct PhasorLCA <: Lux.AbstractLuxLayer
    in_dims::Int
    d_model::Int
    n_heads::Int
    n_anchors::Int
    activation::Function
    k_proj::PhasorDense
    v_proj::PhasorDense
    init_scale::Float32
end

function PhasorLCA(dims::Pair{Int,Int}, n_heads::Int, n_anchors::Int,
                   act = normalize_to_unit_circle;
                   init_scale::Real = 3f0,
                   init_mode::Symbol = :default,
                   spk_args::SpikingArgs = SpikingArgs())
    in_dims, d_model = dims.first, dims.second
    @assert d_model % n_heads == 0 "d_model ($d_model) must be divisible by n_heads ($n_heads)"
    k = PhasorDense(in_dims => d_model; use_bias = false, init_mode = init_mode, spk_args = spk_args)
    v = PhasorDense(in_dims => d_model; use_bias = false, init_mode = init_mode, spk_args = spk_args)
    return PhasorLCA(in_dims, d_model, n_heads, n_anchors, act, k, v, Float32(init_scale))
end

function Lux.initialparameters(rng::AbstractRNG, l::PhasorLCA)
    anchors = Phase.(2f0 .* rand(rng, Float32, l.d_model, l.n_anchors) .- 1f0)
    return (k_proj  = Lux.initialparameters(rng, l.k_proj),
            v_proj  = Lux.initialparameters(rng, l.v_proj),
            anchors = anchors,
            scale   = Float32[l.init_scale])
end

function Lux.initialstates(rng::AbstractRNG, l::PhasorLCA)
    return (k_proj = Lux.initialstates(rng, l.k_proj),
            v_proj = Lux.initialstates(rng, l.v_proj))
end

function Lux.parameterlength(l::PhasorLCA)
    return Lux.parameterlength(l.k_proj) +
           Lux.parameterlength(l.v_proj) +
           l.d_model * l.n_anchors + 1
end

# Internal helper: bundle anchors per head, per (l, b), weighted by attention scores.
#   Ac :: (Dh, H, A)        Complex anchor bank
#   w  :: (A, H, L, B)      Real weights
# Returns B :: (Dh, H, L, B) Complex with
#   B[:, h, l, b] = Σ_a w[a, h, l, b] · Ac[:, h, a].
#
# Implementation: arrange head as the batched dim of NNlib.batched_mul, so each
# head computes its own (Dh, A) × (A, L*B) → (Dh, L*B) contraction.
function _lca_anchor_mix(Ac::AbstractArray{<:Complex, 3}, w::AbstractArray{<:Real, 4})
    Dh, H, A = size(Ac)
    L, B = size(w, 3), size(w, 4)
    Ac_b = permutedims(Ac, (1, 3, 2))                            # (Dh, A, H)
    w_b  = reshape(permutedims(w, (1, 3, 4, 2)), A, L * B, H)    # (A,  L*B, H)
    Y_b  = batched_mul(Ac_b, w_b)                                # (Dh, L*B, H)
    Y    = reshape(Y_b, Dh, L, B, H)
    return permutedims(Y, (1, 4, 2, 3))                          # (Dh, H, L, B)
end

# (i) 3D Phase — the workhorse path.
function (l::PhasorLCA)(x::AbstractArray{<:Phase, 3}, ps::LuxParams, st::NamedTuple)
    K, _ = l.k_proj(x, ps.k_proj, st.k_proj)              # (D, L, B) Phase
    V, _ = l.v_proj(x, ps.v_proj, st.v_proj)

    D, L, B = size(K)
    H  = l.n_heads
    Dh = l.d_model ÷ H
    A  = l.n_anchors

    Kh        = reshape(K, Dh, H, L, B)
    Vh        = reshape(V, Dh, H, L, B)
    Anchors_h = reshape(ps.anchors, Dh, H, A)

    scores  = similarity_outer_heads(Anchors_h, Kh)       # (A, H, L, B)
    weights = exp.(ps.scale .* scores) ./ Float32(A)      # (A, H, L, B)

    Ac     = angle_to_complex(Anchors_h)                  # (Dh, H, A) Complex
    Bundle = _lca_anchor_mix(Ac, weights)                 # (Dh, H, L, B) Complex
    Vc     = angle_to_complex(Vh)                         # (Dh, H, L, B) Complex
    Y      = Vc .* Bundle                                 # element-wise binding
    Y      = reshape(Y, D, L, B)
    Y_phase = complex_to_angle(Y)

    return _apply_phase_activation(l.activation, Y_phase), st
end

# (ii) 2D Phase — single-slice; wrap to 3D with L=1.
function (l::PhasorLCA)(x::AbstractArray{<:Phase, 2}, ps::LuxParams, st::NamedTuple)
    x3 = reshape(x, size(x, 1), 1, size(x, 2))
    y3, st2 = l(x3, ps, st)
    return dropdims(y3, dims=2), st2
end

# (iii) Complex 3D back-compat — trampoline through Phase 3D.
function (l::PhasorLCA)(x::AbstractArray{<:Complex, 3}, ps::LuxParams, st::NamedTuple)
    y_phase, st2 = l(complex_to_angle(x), ps, st)
    return angle_to_complex(y_phase), st2
end

# (iv) SpikingCall — trampoline to CurrentCall.
function (l::PhasorLCA)(x::SpikingCall, ps::LuxParams, st::NamedTuple)
    return l(CurrentCall(x), ps, st)
end

# (v) CurrentCall — reconstruct 3D phase from the ODE solution, then 3D path.
function (l::PhasorLCA)(x::CurrentCall, ps::LuxParams, st::NamedTuple)
    L = round(Int, (x.t_span[2] - x.t_span[1]) / x.spk_args.t_period)
    z_3d = reconstruct_from_current(x, L, x.spk_args)
    return l(z_3d, ps, st)
end

# ================================================================
# 7. SSM Spiking Infrastructure
# ================================================================

# ---- Temporal Encoding Helpers ----

"""
    ssm_phases_to_train(phases::AbstractArray{<:Phase, 3}; spk_args::SpikingArgs) -> SpikeTrain

Encode a 3D phase array (C × L × B) as a SpikeTrain for SSM spiking mode.

Unlike `phase_to_train` (which repeats the same phase each period), this function
maps each time step `l` to a separate oscillation period, with each channel firing
at a time determined by the phase at that step.

# Arguments
- `phases`: (C × L × B) Phase array — channels × time steps × batch
- `spk_args::SpikingArgs`: Spiking parameters (uses `t_period` for temporal mapping)

# Returns
SpikeTrain with `shape=(C, B)` containing `C*L*B` spikes total.
Time step `l` maps to period `[(l-1)*t_period, l*t_period)`.
"""
function ssm_phases_to_train(phases::AbstractArray{<:Phase, 3}; spk_args::SpikingArgs)
    C, L, B = size(phases)
    shape = (C, B)
    period = spk_args.t_period

    # Preallocate for all spikes: C channels × L time steps × B batch
    n_total = C * L * B
    all_indices = Vector{CartesianIndex{2}}(undef, n_total)
    all_times = Vector{Float32}(undef, n_total)

    spatial_indices = vec(CartesianIndices((C, B)))
    idx = 0
    for l in 1:L
        offset = Float32(l - 1) * period
        # phase_to_time returns times in [0, period) due to internal mod
        # Add offset afterward to place spikes in the correct period
        step_times = vec(phase_to_time(phases[:, l, :], period)) .+ offset
        for j in 1:(C * B)
            idx += 1
            all_indices[idx] = spatial_indices[j]
            all_times[idx] = step_times[j]
        end
    end

    return SpikeTrain(all_indices, all_times, shape, 0.0f0)
end

"""
    MakeSpikingSSM <: Lux.AbstractLuxLayer

Chain-compatible layer that converts a 3D complex SSM input (C × L × B) into a
SpikingCall for downstream spiking SSM layers.

Extracts phases from the complex input via `complex_to_angle(normalize_to_unit_circle(x))`,
then encodes them as a SpikeTrain with L oscillation periods using `ssm_phases_to_train`.

# Fields
- `spk_args::SpikingArgs`: Spiking parameters for temporal encoding
"""
struct MakeSpikingSSM <: Lux.AbstractLuxLayer
    spk_args::SpikingArgs
end

Lux.initialparameters(::AbstractRNG, ::MakeSpikingSSM) = NamedTuple()
Lux.initialstates(::AbstractRNG, ::MakeSpikingSSM) = NamedTuple()

function (m::MakeSpikingSSM)(x::AbstractArray{<:Complex, 3}, ps::LuxParams, st::NamedTuple)
    C, L, B = size(x)
    phases = complex_to_angle(normalize_to_unit_circle(x))
    train = ssm_phases_to_train(phases, spk_args=m.spk_args)
    tspan = (0.0f0, Float32(L) * m.spk_args.t_period)
    call = SpikingCall(train, m.spk_args, tspan)
    return call, st
end

# ---- ODE Output Extraction ----

"""
    sample_phases_at_periods(sol, L::Int, spk_args::SpikingArgs;
                              activation = identity,
                              unrotate::Bool = false,
                              offset::Real = 0.0f0) -> AbstractArray{<:Phase}

Interpolate an ODE solution (or any callable returning the per-time
membrane potential) at the L period boundaries
`Float32(n) * spk_args.t_period + offset`, optionally apply
[`unrotate_solution`](@ref) to put the samples in the static phase
frame, then apply `activation` and [`complex_to_angle`](@ref) to
return a Phase tensor.

This is the recommended way to extract per-period phases from a
`PhasorDense` (or `PhasorConv`) layer running with
`return_type = SolutionType(:potential)`. The layer's `:phase`
return type is intentionally the **dense per-save-point trajectory**
of the ODE solver — it carries sub-period information that
period-boundary sampling at the layer would discard. When you do
want per-period phases (e.g. to compare against the discrete Dirac
output, or to feed a downstream consumer that expects an
`(C_out, L, B)` Phase tensor), pull the raw `ODESolution` via
`:potential` and call this helper.

# Arguments
- `sol`: ODE solution (interpolatable at arbitrary times — typically
  an `ODESolution` from `DifferentialEquations.solve`, but any
  callable `t -> potential` works).
- `L::Int`: Number of period boundaries to sample.
- `spk_args::SpikingArgs`: Provides `t_period`.

# Keyword arguments
- `activation = identity`: Applied to the sampled complex potentials
  before phase extraction. Pass `normalize_to_unit_circle` to match
  what `PhasorDense`'s `:phase` dispatch does internally.
- `unrotate::Bool = false`: When `true`, applies
  [`unrotate_solution`](@ref) so the resulting phases live in the
  **static phase frame** — matching the 2D Phase MLP, the
  ODE-via-`unrotate_solution` pair, and the post-§4.1 3D Phase Dirac
  dispatch (`PhasorDense._forward_3d_dirac`). Use `true` for direct
  comparison against the layer's 3D Phase output. When `false`
  (default), phases live in the **rotating frame at the sample
  time** — useful for inspecting the ODE state without applying the
  derotation step.
- `offset::Real = 0.0f0`: Time offset added to the sample times.

# Returns
A Phase tensor of shape `(C_out, L, B)` for 2D per-time potentials
(the typical batched case), or `(C_out, L)` for 1D potentials.

# Example
```julia
layer = PhasorDense(C_in => C_out, normalize_to_unit_circle;
                    return_type = SolutionType(:potential))
ps, st = Lux.setup(rng, layer)
sol, _ = layer(spiking_call, ps, st)
phases = sample_phases_at_periods(sol, L, spk_args;
                                  activation = normalize_to_unit_circle,
                                  unrotate = true)
# `phases` is (C_out, L, B) Phase, in the static frame — directly
# comparable to a (post-§4.1) `PhasorDense` 3D Phase Dirac output.
```

See also: [`reconstruct_from_current`](@ref) (re-solves a bare
oscillator and additionally deconvolves causal accumulation — used by
SSM attention spiking dispatch).
"""
function sample_phases_at_periods(sol, L::Int, spk_args::SpikingArgs;
                                   activation = identity,
                                   unrotate::Bool = false,
                                   offset::Real = 0.0f0)
    T = spk_args.t_period
    sample_ts = Float32[Float32(n) * T + Float32(offset) for n in 1:L]

    samples = [sol(t) for t in sample_ts]

    if unrotate
        samples = unrotate_solution(samples, sample_ts;
                                    spk_args = spk_args, offset = offset)
    end

    # Stack into (C_out, L, B) for 2D per-time potentials, or
    # (C_out, L) for 1D.
    if ndims(samples[1]) == 1
        Z = reduce(hcat, [reshape(s, :, 1) for s in samples])
    else
        Z = cat([reshape(s, size(s, 1), 1, size(s, 2)) for s in samples]...; dims = 2)
    end

    Y = activation(Z)
    return complex_to_angle(Y)
end

# ---- Reconstruct 3D complex from CurrentCall ----

"""
    reconstruct_from_current(x::CurrentCall, L::Int, spk_args::SpikingArgs)

Solve a bare oscillator ODE driven by the current in `x` and sample at L period
boundaries to reconstruct a 3D complex tensor representing the encoded input at
each time step.

Uses three steps to faithfully recover per-period phases from the continuous ODE:

1. **ODE integration** at global `k₀ = leakage + i·2π/t_period`: accumulates spike
   contributions across all L periods into a single trajectory.
2. **Unrotation**: removes the global oscillator rotation so that each sampled
   potential's angle reflects the input phase (not the oscillator's natural phase).
3. **Deconvolution**: the ODE state at period `l` includes decayed residual from
   all previous periods (`z[l] = decay·z[l-1] + response[l]`).  A backward
   difference with `decay = exp(leakage·t_period)` removes this accumulation,
   isolating the single-period spike response whose phase matches the original
   input `exp(iπθ)`.

# Returns
Complex array (C × L × B), normalized to the unit circle.
"""
function reconstruct_from_current(x::CurrentCall, L::Int, spk_args::SpikingArgs)
    k = neuron_constant(spk_args)
    T = spk_args.t_period
    sample_I = x.current.current_fn(x.t_span[1])
    u0 = zeros(ComplexF32, size(sample_I))

    dzdt(u, p, t) = k .* u .+ x.current.current_fn(t)
    sol = spiking_solve(dzdt, u0, x.t_span, spk_args)

    # Sample at period boundaries
    sample_times = Float32.([l * T for l in 1:L])
    samples = [sol(t) for t in sample_times]

    # Unrotate: remove global oscillator rotation to recover encoded phases
    unrotated = unrotate_solution(samples, sample_times, spk_args=spk_args)

    # Stack into 3D array (C × L × B)
    if ndims(unrotated[1]) >= 2
        Z = cat([reshape(s, size(s, 1), 1, size(s, 2)) for s in unrotated]...; dims=2)
    else
        Z = reduce(hcat, [reshape(s, :, 1) for s in unrotated])
    end

    # Deconvolve: remove causal accumulation from global dynamics
    # z_unrot[l] = decay * z_unrot[l-1] + spike_response[l]
    # => spike_response[l] = z_unrot[l] - decay * z_unrot[l-1]
    decay = Float32(exp(spk_args.leakage * T))
    Z_prev = cat(zero(Z[:, 1:1, :]), Z[:, 1:end-1, :]; dims=2)
    Z_deconv = Z .- decay .* Z_prev

    return normalize_to_unit_circle(Z_deconv)
end

# ---- SSMSelfAttention Spiking Dispatch ----

function (l::SSMSelfAttention)(x::SpikingCall, ps::LuxParams, st::NamedTuple)
    current_call = CurrentCall(x)
    return l(current_call, ps, st)
end

function (l::SSMSelfAttention)(x::CurrentCall, ps::LuxParams, st::NamedTuple)
    L = round(Int, (x.t_span[2] - x.t_span[1]) / x.spk_args.t_period)
    z_3d = reconstruct_from_current(x, L, x.spk_args)
    return l(z_3d, ps, st)
end

# ---- SSMCrossAttention Spiking Dispatch ----

function (l::SSMCrossAttention)(x::SpikingCall, ps::LuxParams, st::NamedTuple)
    current_call = CurrentCall(x)
    return l(current_call, ps, st)
end

function (l::SSMCrossAttention)(x::CurrentCall, ps::LuxParams, st::NamedTuple)
    L = round(Int, (x.t_span[2] - x.t_span[1]) / x.spk_args.t_period)
    z_3d = reconstruct_from_current(x, L, x.spk_args)
    return l(z_3d, ps, st)
end

# ---- SSMReadout Spiking Dispatch ----

function (l::SSMReadout)(x::SpikingCall, ps::LuxParams, st::NamedTuple)
    current_call = CurrentCall(x)
    return l(current_call, ps, st)
end

function (l::SSMReadout)(x::CurrentCall, ps::LuxParams, st::NamedTuple)
    L = round(Int, (x.t_span[2] - x.t_span[1]) / x.spk_args.t_period)
    z_3d = reconstruct_from_current(x, L, x.spk_args)
    return l(z_3d, ps, st)
end

# ================================================================
# 8. Phase-domain pre-norm, residual wrapper, and transformer block
# ================================================================
#
# Building blocks for *stacking* the local-attention layers (PhasorLSA /
# PhasorLCA) into deep transformer towers. PhasorLSA/PhasorLCA are bare
# attention layers (no skip connection); a residual wrapper + a phase-domain
# pre-norm are what make them stackable at depth without the representation
# scrambling into a random walk (see the identity-at-init analysis behind
# `ResidualBlock`, and `results/lsa_lca_residual/`).

"""
    PhaseRecenter() <: Lux.AbstractLuxLayer

Phase-domain pre-norm: subtract the per-sample circular mean across the
channel axis (dim 1), pulling the representation back toward phase 0. The
phase analog of LayerNorm's centering step; parameter-free.

Operates on `(C, …)` Phase arrays of any rank (2D `(C, B)` or 3D
`(C, L, B)`) and returns a `Phase` array of the same shape. The circular
mean is computed in the complex domain (`angle_to_complex` → sum over
channels → `complex_to_angle`) so it is well-defined under wraparound.
"""
struct PhaseRecenter <: Lux.AbstractLuxLayer end
Lux.initialparameters(::AbstractRNG, ::PhaseRecenter) = NamedTuple()
Lux.initialstates(::AbstractRNG, ::PhaseRecenter) = NamedTuple()
function (::PhaseRecenter)(x::AbstractArray{<:Phase}, ps::LuxParams, st::NamedTuple)
    z  = angle_to_complex(x)
    mθ = complex_to_angle(sum(z, dims = 1))   # circular mean angle over channels
    return v_bind(x, .-mθ), st
end

"""
    PhasorResidual(layer; gate = :none, alpha0 = 0.1f0) <: Lux.AbstractLuxLayer

Identity-at-init residual wrapper around an arbitrary **shape-preserving**
phase layer: `y = v_bind(x, g · layer(x))`. `v_bind` is phase-domain
addition (its identity element is a branch output of 0, and it passes a
straight-through gradient of 1 to *both* the skip and the branch), and `g`
is the gate:

- `:none` — `g = 1`. Identity-at-init then depends on `layer` itself
  emitting ≈ 0 output phase at init (e.g. a down-scaled `PhasorDense`
  branch).
- `:rezero` — `g = α`, a single learnable scalar initialized to `alpha0`.
  With `alpha0 = 0` the block is **exactly** identity at init (`dy/dx = I`)
  regardless of what `layer` computes. This is the right identity-at-init
  mechanism for attention sublayers, whose output is a head-mix / binding
  rotation that weight-downscaling cannot cleanly drive to zero phase.

Unlike [`ResidualBlock`](@ref) — which builds and wraps its *own*
`PhasorDense` chain — `PhasorResidual` wraps **any** pre-built layer
(`PhasorLSA`, `PhasorLCA`, `SSMSelfAttention`, a `Chain`, …), so it is the
residual unit used by [`PhasorTransformerBlock`](@ref). `layer` must map
`(C, …)` → `(C, …)` (in_dims == out_dims) for the skip to be well-typed.

# Parameters
- `layer` — the wrapped layer's parameter tree.
- `alpha` — `Float32[alpha0]` (only when `gate = :rezero`).
"""
struct PhasorResidual{L} <: Lux.AbstractLuxLayer
    layer::L
    gate::Symbol
    alpha0::Float32
end

function PhasorResidual(layer; gate::Symbol = :none, alpha0::Real = 0.1f0)
    gate in (:none, :rezero) ||
        throw(ArgumentError("gate must be :none or :rezero, got :$gate"))
    return PhasorResidual(layer, gate, Float32(alpha0))
end

function Lux.initialparameters(rng::AbstractRNG, r::PhasorResidual)
    ps = (layer = Lux.initialparameters(rng, r.layer),)
    return r.gate === :rezero ? merge(ps, (alpha = Float32[r.alpha0],)) : ps
end
Lux.initialstates(rng::AbstractRNG, r::PhasorResidual) =
    (layer = Lux.initialstates(rng, r.layer),)
Lux.parameterlength(r::PhasorResidual) =
    Lux.parameterlength(r.layer) + (r.gate === :rezero ? 1 : 0)

function (r::PhasorResidual)(x, ps::LuxParams, st::NamedTuple)
    branch, st_layer = r.layer(x, ps.layer, st.layer)
    y = r.gate === :rezero ? v_bind(x, ps.alpha .* branch) : v_bind(x, branch)
    return y, (layer = st_layer,)
end

"""
    PhasorTransformerBlock(d_model, attn; d_ff = d_model,
                           activation = normalize_to_unit_circle,
                           gate = :rezero, alpha0 = 0.1f0,
                           branch_init_scale = 0.1f0, recenter = false)

Pre-norm phasor transformer block:

```
x → [recenter] → PhasorResidual(attn) → [recenter] → PhasorResidual(FFN) → y
```

with `v_bind` (phase-addition) skip connections. `attn` is any
shape-preserving `d_model ⇒ d_model` phase attention layer
([`PhasorLSA`](@ref), [`PhasorLCA`](@ref), [`SSMSelfAttention`](@ref)),
constructed by the caller; `FFN` is a two-layer `PhasorDense` MLP
(`d_model → d_ff → d_model`). When `recenter = true` a [`PhaseRecenter`]
(@ref) sits at the head of each residual *branch* (true pre-norm — the
skip path is left untouched).

!!! note "recenter defaults to false"
    `PhaseRecenter` computes `complex_to_angle(sum(z))` over channels, which is
    ill-conditioned when channel phasors cancel (`|sum| → 0`) — a
    gradient-blow-up source. The MQAR ablation (`results/xform_recenter/`) found
    it both *hurts* trainability (recenter=false solved 5/5 seeds vs 2/5 stuck
    with recenter=true) and *amplifies* near-origin gradients (~3× larger
    `max|dz|`), so it is **off by default**. It may still help very deep stacks
    (standard pre-norm rationale); re-enable with `recenter = true` and verify.

The residual treatment is fully configurable, so one struct expresses both
the pre-identity-at-init regime (`gate = :none, branch_init_scale = 1,
recenter = false`) and the identity-at-init regime (`gate = :rezero`
and/or `branch_init_scale < 1`):

- `branch_init_scale` down-scales the **FFN** `PhasorDense` weight init
  toward a near-identity branch.
- the **attention** sublayer is brought to identity-at-init by the ReZero
  gate (`gate = :rezero`, `alpha0 → 0`), since down-scaling Q/K/V does not
  cleanly zero the attention output.
- `ffn_init_mode` sets the per-channel `λ` init of **both FFN `PhasorDense`
  layers** (`:hippo` multi-timescale tape — the default — or `:default`
  uniform λ=−0.2). The residual-stream FFN defaults to `:hippo` because the
  MQAR ablation (`results/xform_mqar/`) shows the multi-timescale memory tape
  belongs here, while the attention projections should stay uniform (their
  `init_mode` defaults to `:default`). Note `λ` only shapes dynamics in the
  3D SSM / ODE path — it is a no-op in 2D static.

Modes: operates on `(d_model, L, B)` (or `(d_model, B)`) Phase arrays in
discrete dispatch, and end-to-end on spike trains when given a
`SpikingCall` (spike train in, spike train out). In spiking mode the skip
combines are [`spike_phase_bind`](@ref) (phase addition as a spike delay; the
ReZero α scales the branch spike's lead/lag), the optional pre-norm is
[`spike_phase_recenter`](@ref), the FFN `PhasorDense` layers run their ODE and
emit spikes (`return_type = :spiking`, the default), and the attention layer
runs its `SpikingCall` dispatch (for `PhasorLSA`/`PhasorLCA`: reconstruct a
per-cycle phase field from the incoming spikes, attend discretely, re-emit
one spike per channel per cycle). Stacks of blocks: [`ScanStack`](@ref).

# Fields
- `attn_res::PhasorResidual` — residual-wrapped attention (+ optional pre-norm).
- `ffn_res::PhasorResidual` — residual-wrapped feed-forward MLP (+ optional pre-norm).

See also: [`PhasorResidual`](@ref), [`PhaseRecenter`](@ref),
[`ResidualBlock`](@ref), [`PhasorLSA`](@ref), [`PhasorLCA`](@ref).
"""
struct PhasorTransformerBlock{A, F} <: LuxCore.AbstractLuxContainerLayer{(:attn_res, :ffn_res)}
    attn_res::A
    ffn_res::F
end

function PhasorTransformerBlock(d_model::Int, attn;
                               d_ff::Int = d_model,
                               activation = normalize_to_unit_circle,
                               gate::Symbol = :rezero,
                               alpha0::Real = 0.1f0,
                               branch_init_scale::Real = 0.1f0,
                               ffn_init_mode::Symbol = :hippo,
                               ffn_n_modes::Int = 1,
                               ffn_hippo_tau_max::Union{Real, Nothing} = nothing,
                               ffn_hippo_tau_min::Union{Real, Nothing} = nothing,
                               recenter::Bool = false)
    iw = (rng, dims...) -> Float32(branch_init_scale) .* glorot_uniform(rng, dims...)
    # FFN sublayers: plain PhasorDense (ffn_n_modes=1) or MultiModePhasorDense
    # (ffn_n_modes>1 — the SSM state-expansion knob). Both consume the λ-range
    # knobs (ffn_hippo_tau_max/min).
    mk_ffn(a, b) = ffn_n_modes > 1 ?
        MultiModePhasorDense(a => b, ffn_n_modes, activation; use_bias = true,
                             init_weight = iw, init_mode = ffn_init_mode,
                             hippo_tau_max = ffn_hippo_tau_max, hippo_tau_min = ffn_hippo_tau_min) :
        PhasorDense(a => b, activation; use_bias = true, init_weight = iw,
                    init_mode = ffn_init_mode,
                    hippo_tau_max = ffn_hippo_tau_max, hippo_tau_min = ffn_hippo_tau_min)
    ffn = Chain(mk_ffn(d_model, d_ff), mk_ffn(d_ff, d_model))
    attn_branch = recenter ? Chain(PhaseRecenter(), attn) : attn
    ffn_branch  = recenter ? Chain(PhaseRecenter(), ffn) : ffn
    attn_res = PhasorResidual(attn_branch; gate = gate, alpha0 = alpha0)
    ffn_res  = PhasorResidual(ffn_branch;  gate = gate, alpha0 = alpha0)
    return PhasorTransformerBlock(attn_res, ffn_res)
end

function (b::PhasorTransformerBlock)(x, ps::LuxParams, st::NamedTuple)
    h, st_a = b.attn_res(x, ps.attn_res, st.attn_res)
    y, st_f = b.ffn_res(h, ps.ffn_res, st.ffn_res)
    return y, (attn_res = st_a, ffn_res = st_f)
end

"""
    ScanStack(block, depth; checkpoint = false) <: Lux.AbstractLuxLayer

Apply `depth` independent copies of `block` (same shape, distinct params) in
sequence via a runtime loop instead of a length-`depth` `Lux.Chain`.

A `Chain` of N layers is a length-N tuple, so every distinct depth forces a new
`applychain` specialization; a loop over a homogeneous parameter container
compiles **one** block body and reuses it for every depth. Per-step FLOPs are
unchanged; the win is compile-once + (optionally) checkpointed memory.

Mode-agnostic: the loop just threads `x` through `block`, so the stack runs in
whatever mode `block` supports — discrete 3D Phase, or spiking (`SpikingCall`
in, `SpikingCall` out) for [`PhasorTransformerBlock`](@ref),
[`PhasorResidual`](@ref) and [`ResidualBlock`](@ref).

# Parameters / state
- Params: `(blocks = [p_1, …, p_depth],)` — a Vector of the block's own param
  NamedTuples (distinct random init per copy).
- State: `(block = st,)` — the block's (shared, param-free) state.

`checkpoint = true` wraps each step in `Zygote.checkpointed` to recompute
activations in the backward pass (O(1) tape instead of O(depth)).
"""
struct ScanStack{B} <: Lux.AbstractLuxLayer
    block::B
    depth::Int
    checkpoint::Bool
end
ScanStack(block, depth::Int; checkpoint::Bool = false) = ScanStack(block, depth, checkpoint)

function Lux.initialparameters(rng::AbstractRNG, s::ScanStack)
    return (blocks = [Lux.initialparameters(rng, s.block) for _ in 1:s.depth],)
end
Lux.initialstates(rng::AbstractRNG, s::ScanStack) = (block = Lux.initialstates(rng, s.block),)
# Default parameterlength doesn't recurse the `blocks` Vector → undercounts.
Lux.parameterlength(s::ScanStack) = s.depth * Lux.parameterlength(s.block)

# One block application (top-level so Zygote.checkpointed can target it).
_scan_apply_block(block, x, p, bst) = first(block(x, p, bst))

function (s::ScanStack)(x, ps::LuxParams, st::NamedTuple)
    bst = st.block
    for i in 1:s.depth
        p = ps.blocks[i]
        x = s.checkpoint ? checkpointed(_scan_apply_block, s.block, x, p, bst) :
                           _scan_apply_block(s.block, x, p, bst)
    end
    return x, st
end

# ================================================================
# 9. Spike-domain residual combine (SpikingCall dispatch for the
#    residual / stacking layers)
# ================================================================
#
# Ground rule for the spiking path: the only thing that crosses between
# layers is spike *timing* — at most one event per neuron per carrier cycle,
# whose time relative to the cycle's reference encodes the phase (the
# `ssm_phases_to_train` / `solution_to_train` convention: cycle `l` covers
# `[offset + (l-1)T, offset + lT)`, and phase φ sits at
# `offset + (l-1)T + T·(φ/2 + 1/2)`, so phase 0 is the mid-cycle reference
# time `t_ref(l) = offset + (l-1)T + T/2`). No amplitude crosses.
#
# In the phase domain the residual combine is `v_bind(x, g·b) =
# remap_phase(φ_x + g·φ_b)`: phase addition. In the time domain phase
# addition is a *delay*: the skip spike is delayed by the branch spike's lead/
# lag relative to the reference, `Δ_b = t_b − t_ref`, scaled by the gate,
#
#     t_out = start(l) + mod(t_x − start(l) + g·Δ_b, T)
#
# which is exactly `phase_to_time(remap_phase(φ_x + g·φ_b))` (the `mod T` is
# the oscillator's natural phase wrap — the straight-through wrap of
# `remap_phase`).
#
# Why the ReZero gate is a time-scaling, not an amplitude gain: `α` scales the
# branch's *phase offset*, so it is realised as a gain on a time interval — the
# interval between the reference tick and the branch spike — e.g. a ramp that
# charges at rate α during [t_ref, t_b] and is read out as a delay on the skip
# spike. That ramp is internal to the single combine unit of each channel; the
# only signals entering or leaving it are the skip spike, the branch spike and
# the reference clock, and the only signal it emits is one spike per cycle.
# No branch amplitude ever reaches another neuron, and α = 0 makes the unit a
# pure relay (exact identity), α = 1 recovers plain `v_bind`.
#
# ⚠ Timing idealisation (cost: zero phase error, one cycle of latency in a
# causal realisation). A combine unit can only emit its delayed spike after
# *both* its inputs have arrived; when g·Δ_b < 0 (or the branch spike arrives
# after t_x) the ideal output time precedes an input. We place the output in
# the same cycle `l` as its inputs — the same convention `solution_to_train`
# already uses for every `PhasorDense` (it samples the potential at the *end* of
# cycle `l` and places the spike *inside* cycle `l`). A causal circuit would
# emit the identical phase one cycle later; because every channel and every
# residual would carry the same one-cycle pipeline delay, phases are unchanged
# and only the cycle index shifts. This is why the equivalence tests can demand
# agreement with the discrete path cycle-by-cycle.
#
# Event-driven implementation: the helpers below tabulate spike *times* per
# (neuron, cycle) and do the time arithmetic above on them. This is an exact
# simulation of the delay units, not a phase-domain shortcut: the tables hold
# times, not potentials, and `NaN` marks a silent neuron in that cycle.

# Number of carrier cycles covered by a call's time span.
_n_cycles(call::SpikingCall) =
    round(Int, (call.t_span[2] - call.t_span[1]) / call.spk_args.t_period)

# (N, L) table of spike times, N = prod(shape), NaN where a neuron is silent in
# a cycle. At most one event per neuron per cycle is the spiking convention; if a
# neuron nevertheless fires more than once in a cycle the last event wins.
function _spike_time_table(train::SpikeTrain, L::Int, T::Real)
    T = Float32(T)
    shape = train.shape
    lin = LinearIndices(shape)
    tab = fill(Float32(NaN), prod(shape), L)
    for (idx, t) in zip(train.indices, train.times)
        l = floor(Int, (t - train.offset) / T) + 1
        (1 <= l <= L) || continue
        tab[lin[idx], l] = t
    end
    return tab
end
_spike_time_table(train::SpikeTrainGPU, L::Int, T::Real) =
    _spike_time_table(SpikeTrain(train), L, T)

# Inverse of `_spike_time_table`: drop silent entries, rebuild a train on the
# same device as `like`.
function _table_to_train(tab::AbstractMatrix{Float32}, shape::Tuple, offset::Real, like)
    cart = CartesianIndices(shape)
    keep = findall(!isnan, tab)
    inds = [cart[k[1]] for k in keep]
    times = Float32[tab[k] for k in keep]
    train = SpikeTrain(inds, times, shape, offset)
    return like isa SpikeTrainGPU ? SpikeTrainGPU(train) : train
end

# Cycle start times, shaped (1, L) to broadcast against an (N, L) table.
_cycle_starts(offset::Real, L::Int, T::Real) =
    reshape(Float32(offset) .+ Float32(T) .* Float32.(0:L-1), 1, L)

function _check_compatible(x::SpikingCall, b::SpikingCall)
    x.train.shape == b.train.shape ||
        throw(DimensionMismatch("skip train shape $(x.train.shape) ≠ branch train shape $(b.train.shape); the residual branch must be shape-preserving"))
    x.t_span == b.t_span ||
        throw(ArgumentError("skip t_span $(x.t_span) ≠ branch t_span $(b.t_span). The spike-domain residual combine needs both trains on the same cycles; SpikingArgs(warmup_periods > 0) inside a residual branch is not supported."))
    isapprox(x.train.offset, b.train.offset; atol = 1f-6) ||
        throw(ArgumentError("skip and branch spike trains have different reference offsets ($(x.train.offset) vs $(b.train.offset))"))
    return nothing
end

"""
    spike_phase_bind(x::SpikingCall, b::SpikingCall; gain = 1f0) -> SpikingCall

Spike-domain residual combine: the spiking counterpart of
`v_bind(φ_x, gain · φ_b)` (phase addition with wrap), computed purely from
spike times. In every cycle, each channel's skip spike `t_x` is delayed by
`gain · (t_b − t_ref)`, the branch spike's lead/lag relative to the cycle's
phase-0 reference, and wrapped back into the cycle (`mod T`). `gain` scales a
*time interval*, so the ReZero gate α is realised without any amplitude
crossing between neurons (see the section comment above for the circuit
reading and the ⚠ timing idealisation).

Silent neurons: a silent branch contributes no shift (the `v_bind` identity
element, phase 0); a silent skip neuron stays silent.

Inference only — not differentiable (the event-driven table is built
outside the AD graph).
"""
function spike_phase_bind(x::SpikingCall, b::SpikingCall; gain::Real = 1f0)
    _check_compatible(x, b)
    T = x.spk_args.t_period
    L = _n_cycles(x)
    off = x.train.offset
    tx = _spike_time_table(x.train, L, T)
    tb = _spike_time_table(b.train, L, T)
    start = _cycle_starts(off, L, T)
    t_ref = start .+ Float32(T) / 2f0
    Δb = ifelse.(isnan.(tb), 0f0, tb .- t_ref)          # branch lead/lag vs reference
    t_out = start .+ mod.(tx .- start .+ Float32(gain) .* Δb, Float32(T))
    train = _table_to_train(t_out, x.train.shape, off, x.train)
    return SpikingCall(train, x.spk_args, x.t_span)
end

"""
    spike_phase_recenter(x::SpikingCall) -> SpikingCall

Spiking counterpart of [`PhaseRecenter`](@ref): per cycle and per batch
column, subtract the circular mean phase across channels (dim 1 of the train
shape).

Circuit reading: one *mean unit* per batch column receives every channel's
spike in the cycle and integrates them as a non-leaky resonator that resets at
the cycle boundary; its phase at cycle end is `angle(Σ_c exp(iπφ_c))`, the
circular mean, which it emits as a single spike. Each channel's spike is then
advanced by the mean unit's lead/lag relative to the reference — the same
delay unit as [`spike_phase_bind`](@ref) with gain −1. Only spike times cross
between units (the resonator's amplitude stays inside the mean unit). Silent
channels do not contribute to the mean. Same ⚠ timing idealisation as
`spike_phase_bind`.
"""
function spike_phase_recenter(x::SpikingCall)
    T = Float32(x.spk_args.t_period)
    L = _n_cycles(x)
    off = x.train.offset
    shape = x.train.shape
    tx = _spike_time_table(x.train, L, T)                # (N, L)
    start = _cycle_starts(off, L, T)
    t_ref = start .+ T / 2f0
    # phase of each spike relative to its cycle's reference, in units of π
    φ = 2f0 .* (tx .- t_ref) ./ T
    zc = ifelse.(isnan.(φ), zero(ComplexF32), cis.(Float32(π) .* φ))
    C = shape[1]
    rest = prod(shape[2:end]; init = 1)
    zc3 = reshape(zc, C, rest, L)
    m = sum(zc3; dims = 1)                               # (1, rest, L) mean-unit state
    t_mean = reshape(T / 2f0 .* angle.(m) ./ Float32(π), 1, rest, L)   # mean unit lead/lag
    tx3 = reshape(tx, C, rest, L)
    st3 = reshape(start, 1, 1, L)
    t_out = st3 .+ mod.(tx3 .- st3 .- t_mean, T)
    train = _table_to_train(reshape(t_out, :, L), shape, off, x.train)
    return SpikingCall(train, x.spk_args, x.t_span)
end

"""
    ssm_train_to_phases(call::SpikingCall) -> Array{Phase}

Inverse of [`ssm_phases_to_train`](@ref): decode a spike train with one event
per neuron per carrier cycle into a `(C, L, B)` Phase array (or `(C, L)` for a
1-D train shape), reading each spike's time relative to its cycle's reference.
Silent neurons decode to `NaN`. Pure timing readout — no ODE.
"""
function ssm_train_to_phases(call::SpikingCall)
    T = Float32(call.spk_args.t_period)
    L = _n_cycles(call)
    tr = call.train
    tab = _spike_time_table(tr, L, T)
    start = _cycle_starts(tr.offset, L, T)
    φ = Phase.(2f0 .* (tab .- start) ./ T .- 1f0)        # (N, L)
    shape = tr.shape
    if length(shape) == 1
        return φ
    end
    C = shape[1]
    rest = shape[2:end]
    φ3 = reshape(φ, C, prod(rest), L)
    return reshape(permutedims(φ3, (1, 3, 2)), C, L, rest...)
end

# Bring a branch output back onto spikes, as the branch's output neurons
# emitting one spike per cycle at their phase. Layers whose spiking dispatch
# already emits a train pass through; layers whose spiking dispatch returns a
# per-cycle phase field (PhasorLSA / PhasorLCA / SSMSelfAttention return
# Complex 3D or Phase 3D after `reconstruct_from_current`) are re-encoded with
# the canonical `ssm_phases_to_train` timing.
_as_spike_call(y::SpikingCall, ref::SpikingCall) = y
_as_spike_call(y::AbstractArray{<:Complex, 3}, ref::SpikingCall) =
    _as_spike_call(complex_to_angle(y), ref)
function _as_spike_call(y::AbstractArray{<:Phase, 3}, ref::SpikingCall)
    L = _n_cycles(ref)
    size(y, 2) == L ||
        throw(DimensionMismatch("branch returned $(size(y, 2)) cycles, expected $L"))
    train = ssm_phases_to_train(Array(y); spk_args = ref.spk_args)
    off = ref.train.offset
    if off != 0f0
        train = SpikeTrain(train.indices, train.times .+ off, train.shape, off)
    end
    train = ref.train isa SpikeTrainGPU ? SpikeTrainGPU(train) : train
    return SpikingCall(train, ref.spk_args, ref.t_span)
end
_as_spike_call(y, ref::SpikingCall) =
    throw(ArgumentError("residual branch returned $(typeof(y)) for a SpikingCall input; " *
                        "it must return a SpikingCall (e.g. PhasorDense with " *
                        "return_type = SolutionType(:spiking)) or a per-cycle (C, L, B) phase field"))

_gate_value(alpha) = Float32(only(Array(alpha)))

# ---- PhaseRecenter ----

function (::PhaseRecenter)(x::SpikingCall, ps::LuxParams, st::NamedTuple)
    return spike_phase_recenter(x), st
end

# ---- PhasorResidual ----

function (r::PhasorResidual)(x::SpikingCall, ps::LuxParams, st::NamedTuple)
    branch, st_layer = r.layer(x, ps.layer, st.layer)
    b = _as_spike_call(branch, x)
    g = r.gate === :rezero ? _gate_value(ps.alpha) : 1f0
    return spike_phase_bind(x, b; gain = g), (layer = st_layer,)
end

const _RESIDUAL_CURRENT_MSG =
    "The residual skip must carry spike timing (one event per neuron per cycle), " *
    "so the spike-domain combine has no CurrentCall method: a continuous current " *
    "is an amplitude signal and cannot be combined without passing amplitude " *
    "between neurons. Feed a SpikingCall (e.g. emit with " *
    "return_type = SolutionType(:spiking) upstream)."

(r::PhasorResidual)(x::CurrentCall, ps::LuxParams, st::NamedTuple) =
    throw(ArgumentError("PhasorResidual: " * _RESIDUAL_CURRENT_MSG))

# ---- ResidualBlock (defined in network.jl) ----

function (rb::ResidualBlock)(x::SpikingCall, ps::LuxParams, st::NamedTuple)
    ff_out, st_ff = rb.ff(x, ps.ff, st.ff)
    b = _as_spike_call(ff_out, x)
    g = rb.gate === :rezero ? _gate_value(ps.alpha) : 1f0
    return spike_phase_bind(x, b; gain = g), (ff = st_ff,)
end

(rb::ResidualBlock)(x::CurrentCall, ps::LuxParams, st::NamedTuple) =
    throw(ArgumentError("ResidualBlock: " * _RESIDUAL_CURRENT_MSG))

# PhasorTransformerBlock and ScanStack need no extra methods: their forward
# passes are mode-agnostic compositions of PhasorResidual / PhaseRecenter /
# the wrapped block, so a SpikingCall flows through them spike-in, spike-out.
