# ================================================================
# SSM Support: Attention, Encoding, and Spiking Helpers
# ================================================================
#
# Kernel math (phasor_kernel, causal_conv, hippo_legs_diagonal) is in kernels.jl.
# PhasorSSM struct has been unified into PhasorDense (network.jl).
# This file keeps: SSMReadout, attention layers, encoding, spiking helpers,
# and a backward-compatible PhasorSSM(...) constructor function.

# ================================================================
# 1. PhasorSSM Backward-Compatible Constructor
# ================================================================

"""
    PhasorSSM(in_dims => out_dims, activation; init_omega_range, init, return_type)

Backward-compatible constructor that returns a `PhasorDense` with SSM-appropriate defaults.

This function exists for backward compatibility. New code should use `PhasorDense` directly
with `init_mode=:uniform` or `init_mode=:hippo`.

# Arguments
- `dims::Pair{Int,Int}` — Channel dimensions (same as `PhasorDense(in => out)`).
- `activation` — Applied after temporal integration. Default `normalize_to_unit_circle`.
- `init_omega_range` — Initial spread of angular frequencies. Default `(0.2, 2.5)`.
- `init` — Parameter initialization: `:uniform` or `:hippo`.
- `return_type` — Output format. Default `SolutionType(:phase)`.

# Returns
A `PhasorDense` layer configured for SSM use (no bias, uniform/hippo init).
"""
function PhasorSSM(dims::Pair{Int,Int}, act=normalize_to_unit_circle;
                   init_omega_range=(0.2f0, 2.5f0), init=:uniform,
                   return_type::SolutionType=SolutionType(:phase))
    @assert init in (:uniform, :hippo) "init must be :uniform or :hippo"
    Base.depwarn("PhasorSSM is deprecated, use PhasorDense with init_mode instead", :PhasorSSM)
    return PhasorDense(dims, act;
                       init_mode=init,
                       omega_lo=Float32(init_omega_range[1]),
                       omega_hi=Float32(init_omega_range[2]),
                       use_bias=false,
                       return_type=return_type)
end

# Parameterlength for backward compat (PhasorDense doesn't define this)
function Lux.parameterlength(l::PhasorDense)
    n = l.out_dims * l.in_dims + l.out_dims  # weight + log_neg_lambda
    if l.use_bias
        n += 2 * l.out_dims  # bias_real + bias_imag
    end
    if l.trainable_omega
        n += l.out_dims
    end
    return n
end

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
    SSMCrossAttention(in_dims => d_model, n_keys, activation; kwargs...)

Cross-attention layer for the phasor SSM framework.  Accepts complex-valued
3D input and produces attention-weighted complex output using stored key
prototypes.

Two dense sub-networks project the input into complex queries (Q) and values
(V).  Keys (K) are trainable Phase parameters — learned prototypes that the
sequence attends to.  QK similarity scores (via `similarity_outer`) are
exponentially scaled (via `score_scale`) and used to weight-multiply the
values.  The result is renormalized to the unit circle.

**Note:** The temporal dimension changes from L to `n_keys`.  Set
`n_keys = L` to preserve the temporal dimension, or use a different value
as a bottleneck / pooling mechanism.

# Arguments
- `in_dims => d_model` — Input channel dimension and output channel dimension.
- `n_keys::Int` — Number of stored key prototypes.
- `activation` — Applied after attention.  Default `normalize_to_unit_circle`.

# Trainable parameters
- `weight_q` (Float32, d_model × in_dims) — Query projection matrix.
- `weight_v` (Float32, d_model × in_dims) — Value projection matrix.
- `keys` (Phase, d_model × n_keys) — Stored key prototypes.
- `scale` (Float32, length 1) — Exponential score scaling factor.

# Data flow
```
Input (C_in × L × B) complex
  → Q = W_q · x  (d_model × L × B) complex
  → V = W_v · x  (d_model × L × B) complex
  → Q_phase = complex_to_angle(normalize(Q))
  → scores = score_scale(similarity_outer(Q_phase, K_phase))  (L × N_keys × B)
  → output = batched_mul(V, scores)  (d_model × N_keys × B) complex
  → activation(output)
```
"""
struct SSMCrossAttention <: Lux.AbstractLuxLayer
    in_dims::Int
    d_model::Int
    n_keys::Int
    activation::Function
end

function SSMCrossAttention(dims::Pair{Int,Int}, n_keys::Int,
                           act=normalize_to_unit_circle)
    return SSMCrossAttention(dims.first, dims.second, n_keys, act)
end

function Lux.initialparameters(rng::AbstractRNG, l::SSMCrossAttention)
    inv_sqrt = 1f0 / sqrt(Float32(l.in_dims))
    W_q = Float32.(randn(rng, l.d_model, l.in_dims)) .* inv_sqrt
    W_v = Float32.(randn(rng, l.d_model, l.in_dims)) .* inv_sqrt
    keys = Phase.(2f0 .* rand(rng, Float32, l.d_model, l.n_keys) .- 1f0)
    scale = [3.0f0]
    return (weight_q=W_q, weight_v=W_v, keys=keys, scale=scale)
end

Lux.initialstates(::AbstractRNG, ::SSMCrossAttention) = NamedTuple()
Lux.parameterlength(l::SSMCrossAttention) = l.d_model * l.in_dims * 2 + l.d_model * l.n_keys + 1

function (l::SSMCrossAttention)(x::AbstractArray{<:Complex, 3}, ps::LuxParams, st::NamedTuple)
    C_in, L, B = size(x)

    # Project input to Q and V via real weight matrices on complex input
    xr = reshape(x, C_in, L * B)
    Q = reshape(complex.(ps.weight_q * real.(xr), ps.weight_q * imag.(xr)),
                l.d_model, L, B)
    V = reshape(complex.(ps.weight_v * real.(xr), ps.weight_v * imag.(xr)),
                l.d_model, L, B)

    # Convert Q to phase for similarity computation
    Q_phase = complex_to_angle(normalize_to_unit_circle(Q))  # d_model × L × B Phase

    # Expand stored keys to 3D for similarity_outer
    K_phase = repeat(reshape(ps.keys, l.d_model, l.n_keys, 1), 1, 1, B)

    # Compute attention scores and weight values
    scores = score_scale(similarity_outer(Q_phase, K_phase, dims=2),
                         scale=ps.scale)          # L × n_keys × B
    output = batched_mul(V, scores)                # d_model × n_keys × B

    return l.activation(output), st
end

# ================================================================
# 6. SSM Self-Attention Layer
# ================================================================

"""
    SSMSelfAttention(in_dims => d_model, activation)

Self-attention layer for the phasor SSM framework.  Projects the complex
input into queries (Q), keys (K), and values (V) via three dense
sub-networks.  QK similarity scores weight the values, and the result is
renormalized to the unit circle.

Unlike `SSMCrossAttention`, keys are derived from the input rather than
stored as parameters.  Since Q and K have the same temporal extent L, the
output preserves the temporal dimension — making this a drop-in layer in
a `Chain`.

# Arguments
- `in_dims => d_model` — Input and output channel dimensions.
- `activation` — Applied after attention.  Default `normalize_to_unit_circle`.

# Trainable parameters
- `weight_q` (Float32, d_model × in_dims) — Query projection.
- `weight_k` (Float32, d_model × in_dims) — Key projection.
- `weight_v` (Float32, d_model × in_dims) — Value projection.
- `scale` (Float32, length 1) — Exponential score scaling factor.

# Data flow
```
Input (C_in × L × B) complex
  → Q = W_q · x,  K = W_k · x,  V = W_v · x   (d_model × L × B)
  → scores = score_scale(similarity_outer(Q_phase, K_phase))  (L × L × B)
  → output = batched_mul(V, scores)  (d_model × L × B) complex
  → activation(output)
```
"""
struct SSMSelfAttention <: Lux.AbstractLuxLayer
    in_dims::Int
    d_model::Int
    activation::Function
end

function SSMSelfAttention(dims::Pair{Int,Int}, act=normalize_to_unit_circle)
    return SSMSelfAttention(dims.first, dims.second, act)
end

function Lux.initialparameters(rng::AbstractRNG, l::SSMSelfAttention)
    inv_sqrt = 1f0 / sqrt(Float32(l.in_dims))
    W_q = Float32.(randn(rng, l.d_model, l.in_dims)) .* inv_sqrt
    W_k = Float32.(randn(rng, l.d_model, l.in_dims)) .* inv_sqrt
    W_v = Float32.(randn(rng, l.d_model, l.in_dims)) .* inv_sqrt
    scale = [3.0f0]
    return (weight_q=W_q, weight_k=W_k, weight_v=W_v, scale=scale)
end

Lux.initialstates(::AbstractRNG, ::SSMSelfAttention) = NamedTuple()
Lux.parameterlength(l::SSMSelfAttention) = l.d_model * l.in_dims * 3 + 1

function (l::SSMSelfAttention)(x::AbstractArray{<:Complex, 3}, ps::LuxParams, st::NamedTuple)
    C_in, L, B = size(x)

    # Project input to Q, K, V
    xr = reshape(x, C_in, L * B)
    Q = reshape(complex.(ps.weight_q * real.(xr), ps.weight_q * imag.(xr)),
                l.d_model, L, B)
    K = reshape(complex.(ps.weight_k * real.(xr), ps.weight_k * imag.(xr)),
                l.d_model, L, B)
    V = reshape(complex.(ps.weight_v * real.(xr), ps.weight_v * imag.(xr)),
                l.d_model, L, B)

    # Convert Q, K to phase for similarity
    Q_phase = complex_to_angle(normalize_to_unit_circle(Q))
    K_phase = complex_to_angle(normalize_to_unit_circle(K))

    # Compute attention scores and weight values
    scores = score_scale(similarity_outer(Q_phase, K_phase, dims=2),
                         scale=ps.scale)          # L × L × B
    output = batched_mul(V, scores)                # d_model × L × B

    return l.activation(output), st
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
