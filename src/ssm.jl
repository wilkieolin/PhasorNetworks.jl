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
- `readout_frac` — Fraction of final time steps to average over. Default 0.25.
"""
struct SSMReadout <: Lux.AbstractLuxLayer
    hidden_dims::Int
    n_classes::Int
    readout_frac::Float32
end

function SSMReadout(dims::Pair{Int,Int}; readout_frac::Float32=0.25f0)
    return SSMReadout(dims.first, dims.second, readout_frac)
end

Lux.initialparameters(::AbstractRNG, ::SSMReadout) = NamedTuple()

function Lux.initialstates(rng::AbstractRNG, l::SSMReadout)
    return (codes = random_symbols(rng, (l.hidden_dims, l.n_classes)),)
end

function (l::SSMReadout)(z::AbstractArray{<:Complex, 3}, ps::LuxParams, st::NamedTuple)
    C, L, B = size(z)
    t0 = max(1, L - max(1, round(Int, L * l.readout_frac)) + 1)
    W = L - t0 + 1

    # Extract phase at each timestep in the readout window
    z_window = z[:, t0:L, :]                         # C × W × B
    z_norm = normalize_to_unit_circle(z_window)
    phases = complex_to_angle(z_norm)                 # C × W × B  Phase

    # Broadcast similarity: cos(π·(phase - code)) averaged over features
    codes = st.codes                                  # C × n_classes  Phase
    n_cls = size(codes, 2)
    p = reshape(phases, C, 1, W, B)                   # C × 1 × W × B
    c = reshape(codes, C, n_cls, 1, 1)                # C × n_classes × 1 × 1
    cos_diff = cos.(pi_f32 .* (p .- c))               # C × n_classes × W × B
    sims_per_step = mean(cos_diff; dims=1)            # 1 × n_classes × W × B

    # Average logits over the readout window
    sims_avg = mean(sims_per_step; dims=3)            # 1 × n_classes × 1 × B

    return dropdims(sims_avg; dims=(1, 3)), st        # n_classes × B
end

function (l::SSMReadout)(x::AbstractArray{<:Phase, 3}, ps::LuxParams, st::NamedTuple)
    # Phase input: already normalized, skip normalize_to_unit_circle
    C, L, B = size(x)
    t0 = max(1, L - max(1, round(Int, L * l.readout_frac)) + 1)
    W = L - t0 + 1

    phases = x[:, t0:L, :]                               # C × W × B  Phase

    codes = st.codes                                      # C × n_classes  Phase
    n_cls = size(codes, 2)
    p = reshape(phases, C, 1, W, B)
    c = reshape(codes, C, n_cls, 1, 1)
    cos_diff = cos.(pi_f32 .* (Float32.(p) .- Float32.(c)))
    sims_per_step = mean(cos_diff; dims=1)
    sims_avg = mean(sims_per_step; dims=3)

    return dropdims(sims_avg; dims=(1, 3)), st
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
- `init_mode::Symbol = :hippo` — `PhasorDense` λ-initialization mode.
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
end

function PhasorLSA(dims::Pair{Int,Int}, n_heads::Int,
                   act = normalize_to_unit_circle;
                   init_scale::Real = 3f0,
                   init_mode::Symbol = :hippo,
                   spk_args::SpikingArgs = SpikingArgs())
    in_dims, d_model = dims.first, dims.second
    @assert d_model % n_heads == 0 "d_model ($d_model) must be divisible by n_heads ($n_heads)"
    q = PhasorDense(in_dims => d_model; use_bias = false, init_mode = init_mode, spk_args = spk_args)
    k = PhasorDense(in_dims => d_model; use_bias = false, init_mode = init_mode, spk_args = spk_args)
    v = PhasorDense(in_dims => d_model; use_bias = false, init_mode = init_mode, spk_args = spk_args)
    return PhasorLSA(in_dims, d_model, n_heads, act, q, k, v, Float32(init_scale))
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
    Q, _ = l.q_proj(x, ps.q_proj, st.q_proj)             # (D, L, B) Phase
    K, _ = l.k_proj(x, ps.k_proj, st.k_proj)
    V, _ = l.v_proj(x, ps.v_proj, st.v_proj)

    D, L, B = size(Q)
    H  = l.n_heads
    Dh = l.d_model ÷ H
    Qh = reshape(Q, Dh, H, L, B)
    Kh = reshape(K, Dh, H, L, B)
    Vh = reshape(V, Dh, H, L, B)

    scores  = similarity_outer_heads(Qh, Kh)             # (H, H, L, B) Float32
    weights = exp.(ps.scale .* scores) ./ Float32(H)      # (H, H, L, B)

    Vc = angle_to_complex(Vh)                             # (Dh, H, L, B) Complex
    Y  = _lsa_head_mix(Vc, weights)                       # (Dh, H, L, B) Complex
    Y  = reshape(Y, D, L, B)
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
- `init_mode::Symbol = :hippo` — `PhasorDense` λ-init mode.
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
                   init_mode::Symbol = :hippo,
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
    ssm_extract_phases(sol, L::Int, t_period::Float32, activation::Function)

Sample an ODE solution at L period boundaries and return a 3D complex array.

# Arguments
- `sol`: ODE solution (interpolatable at arbitrary times)
- `L::Int`: Number of time steps to sample
- `t_period::Float32`: Duration of each oscillation period
- `activation::Function`: Applied after sampling (e.g. `normalize_to_unit_circle`)

# Returns
Complex array (out_dims × L × B) — the activated membrane potentials at each period.
No per-channel derotation is applied; both discrete SSM and ODE include the same
rotation from omega, so phases are directly comparable.
"""
function ssm_extract_phases(sol, L::Int, t_period::Float32, activation::Function)
    samples = [sol(Float32(l) * t_period) for l in 1:L]

    # Stack into 3D: each sample is (out_dims,) or (out_dims, B)
    if ndims(samples[1]) == 1
        # (out_dims,) → (out_dims, L)
        Z = reduce(hcat, [reshape(s, :, 1) for s in samples])
    else
        # (out_dims, B) → (out_dims, L, B)
        Z = cat([reshape(s, size(s, 1), 1, size(s, 2)) for s in samples]...; dims=2)
    end

    return activation(Z)
end

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

See also: [`ssm_extract_phases`](@ref) (returns complex without phase
conversion or unrotation), [`reconstruct_from_current`](@ref)
(re-solves a bare oscillator and additionally deconvolves causal
accumulation — used by SSM attention spiking dispatch).
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
    # (C_out, L) for 1D — same convention as ssm_extract_phases.
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
    prob = ODEProblem(dzdt, u0, x.t_span)
    sol = solve(prob, spk_args.solver; spk_args.solver_args...)

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
