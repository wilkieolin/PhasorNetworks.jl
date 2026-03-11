# ================================================================
# Discrete State Space Model for Phasor Networks
# ================================================================
#
# Discretizes the Resonate-and-Fire neuron ODE (dz/dt = kz + I(t))
# into a linear recurrence:  z[n+1] = A·z[n] + B·I[n]
# where A = exp(k·Δt), B = (A-1)/k, k = λ + iω.
#
# Unrolling gives a causal convolution: z[n] = Σⱼ K[n-j]·I[j],
# K[n] = Aⁿ·B, computable in parallel via matrix multiply.

# ================================================================
# 1. Discrete Phasor Kernel
# ================================================================

"""
    phasor_kernel(λ, ω, Δt, L) -> ComplexF32 matrix (C × L)

Compute the causal impulse-response kernel for C damped oscillators over L
discrete time steps.  This is the discrete equivalent of what `oscillator_bank`
computes continuously: given a single unit impulse at time 0, how does the
oscillator ring down over subsequent steps?

# Arguments
- `λ::AbstractVector{Float32}` — **Decay rates** (length C, one per oscillator).
  Same physical meaning as `spk_args.leakage` in the spiking system: controls
  how fast the membrane potential spirals inward.  Must be negative for
  stability (e.g. -0.1).  More negative = faster decay = shorter memory.
  In the continuous ODE this appears as:  dz/dt = (λ + iω)z + I(t).

- `ω::AbstractVector{Float32}` — **Angular frequencies** (length C, rad/step).
  Same role as `2π / spk_args.t_period` in the spiking system: controls how
  fast the oscillator rotates in the complex plane.  In the continuous ODE
  the neuron constant is k = λ + iω; here ω is angular frequency directly
  rather than derived from a period.

- `Δt::Real` — **Time step** between consecutive input samples.  Analogous to
  the `dt` solver argument in `oscillator_bank`.  We set Δt=1 here because
  our "time" axis is just sample indices (row 1, row 2, ..., row 28 of the
  image).  If your samples were 0.02s apart (like the ODE demo), you'd set
  Δt=0.02.

- `L::Int` — **Sequence length** (number of time steps).  For FashionMNIST
  with rows-as-time, L=28.

# Returns
A `C × L` complex matrix where K[c, n] is the response of oscillator c at
lag n.  Entry K[c, 0] (column 1) is the immediate response to the current
input; K[c, 27] (column 28) is the lingering contribution from 27 steps ago.

# Connection to the continuous ODE
In `oscillator_bank`, the neuron constant is:

    k = neuron_constant(leakage, t_period) = leakage + i·(2π / t_period)

The ODE `dz/dt = k·z + I(t)` has exact solution over one step Δt:

    z[n+1] = exp(k·Δt) · z[n] + B · I[n]

where `A = exp(k·Δt)` and `B = (A - 1) / k` (zero-order-hold discretization).
Unrolling this recurrence gives z[n] = Σⱼ Aⁿ⁻ʲ · B · I[j], which is a
causal convolution with kernel K[n] = Aⁿ · B.

So this function precomputes that entire kernel in one shot, replacing the
sequential ODE solve with a single matrix that can be applied in parallel.
"""
function phasor_kernel(λ::AbstractVector, ω::AbstractVector, Δt::Real, L::Int)
    k = ComplexF32.(λ .+ im .* ω)            # C eigenvalues
    # Build range on same device as k (GPU-safe)
    ns_cpu = Float32.(0:L-1)
    ns = reshape(typeof(real.(k))(ns_cpu), 1, L) # 1 × L, same device as params
    A_powers = exp.(k .* Δt .* ns)            # C × L:  A^n
    B_gain = (exp.(k .* Δt) .- 1f0) ./ k     # C × 1:  input gain
    return A_powers .* B_gain                  # C × L
end

# ================================================================
# 2. Causal Convolution via Lower-Triangular Toeplitz Matrix
# ================================================================

"""
    causal_conv(K, H) -> ComplexF32 array (C × L × B)

Apply the precomputed impulse-response kernel to a batch of input signals.
This replaces the role of `DifferentialEquations.solve()` in the spiking
system — instead of stepping through time sequentially, we compute all
time steps at once via matrix multiplication.

# Arguments
- `K::AbstractMatrix{<:Complex}` — **Kernel matrix** (C × L), as returned by
  `phasor_kernel`.  Each row is one oscillator's impulse response over L lags.

- `H::AbstractArray{<:Complex, 3}` — **Input signal** (C × L × B).
  C = number of output channels (oscillators), L = time steps, B = batch size.

# Returns
`Z` (C × L × B): the complex membrane potential at each time step, for each
oscillator, for each batch element.

# Implementation
The causal convolution Z[c,t] = Σⱼ₌₀ᵗ K[c, t-j] · H[c, j] is equivalent to
multiplying H by a lower-triangular Toeplitz matrix.  Uses `NNlib.batched_mul`
for GPU-friendly parallel computation.  Cost is O(C·L²·B).
"""
function causal_conv(K::AbstractMatrix{<:Complex}, H::AbstractArray{<:Complex, 3})
    C, L, B = size(H)

    # Pad kernel with one zero column so out-of-bounds indices map to 0
    K_pad = cat(K, zero(K[:, 1:1]); dims=2)  # C × (L+1)

    # Lower-triangular index matrix: T[i,j] = K[:, i-j+1] when i≥j, else K[:, L+1]=0
    idx = [i >= j ? i - j + 1 : L + 1 for i in 1:L, j in 1:L]  # L × L (constant, CPU)
    T = K_pad[:, idx]                                      # C × L × L

    # batched_mul wants (M,K,batch) × (K,N,batch) → (M,N,batch), batch = C
    T_perm = permutedims(T, (2, 3, 1))  # L × L × C
    H_perm = permutedims(H, (2, 3, 1))  # L × B × C
    Z_perm = batched_mul(T_perm, H_perm) # L × B × C
    return permutedims(Z_perm, (3, 1, 2)) # C × L × B
end

# ================================================================
# 3. HiPPO-LegS Initialization
# ================================================================

"""
    hippo_legs_diagonal(N; clip_decay=nothing) -> (λ, ω)

HiPPO-LegS (Legendre Scaled) diagonal initialization from S4D.

The HiPPO framework defines optimal state matrices for online function
approximation.  The LegS variant projects input history onto a scaled
Legendre polynomial basis, giving a principled multi-timescale memory.

Diagonalizing the N×N HiPPO-LegS matrix yields N complex eigenvalues:

    k_n = -(n + 1/2) + iπ(n + 1/2),   n = 0, 1, ..., N-1

# Arguments
- `N::Int` — Number of oscillators (state dimension).

# Keyword arguments
- `clip_decay`: Maximum magnitude of λ.  When `nothing` (default), eigenvalues
  are log-spaced from 0.5 to N-0.5 across N channels, preserving the HiPPO
  multi-timescale structure while keeping all channels in a trainable range.

# Returns
Tuple `(λ, ω)` of Float32 vectors of length N.  Caller must map to the
log-parameterization (`log_neg_lambda = log.(-λ)`).
"""
function hippo_legs_diagonal(N::Int; clip_decay::Union{Nothing, Real}=nothing)
    if clip_decay === nothing
        # Log-spaced variant: span from λ=-0.5 to λ=-N+0.5 on a log scale.
        λ_mag = Float32.(exp.(range(log(0.5), log(N - 0.5); length=N)))
    else
        # Linear HiPPO with hard clip
        ns = Float32.(0:N-1)
        λ_mag = min.(ns .+ 0.5f0, Float32(clip_decay))
    end
    λ = -λ_mag
    # Frequency paired to decay: one oscillation per memory window
    ω = Float32(π) .* λ_mag
    return λ, ω
end

# ================================================================
# 4. PhasorSSM Layer (Lux-compatible)
# ================================================================

"""
    PhasorSSM(in_dims => out_dims, activation; init_omega_range=(0.2, 2.5), init=:uniform)

Discrete phasor state-space layer — the SSM equivalent of `PhasorDense`.

Where `PhasorDense` feeds weighted input into `oscillator_bank` (a
continuous ODE solver), `PhasorSSM` does the same integration as a
precomputed convolution kernel applied via matrix multiply.  The two
produce equivalent dynamics; the difference is computational:
- ODE: sequential, expensive, but handles continuous/irregular time
- SSM: parallel, fast, but requires uniformly-sampled discrete input

# Arguments
- `in_dims => out_dims` — Channel dimensions (same as `PhasorDense(in => out)`).
- `activation` — Applied after temporal integration.  Typically
  `normalize_to_unit_circle` between layers, or `identity` for the final layer.
- `init_omega_range` — Initial spread of angular frequencies. Default `(0.2, 2.5)`.
- `init` — Parameter initialization: `:uniform` or `:hippo`.

# Trainable parameters
- `weight` (Float32, out × in) — Channel mixing matrix.
- `log_neg_lambda` (Float32, out) — Log-space decay rates. λ = -exp(log_neg_lambda).
- `omega` (Float32, out) — Angular frequencies in radians per time step.
"""
struct PhasorSSM <: Lux.AbstractLuxLayer
    in_dims::Int
    out_dims::Int
    activation::Function
    omega_lo::Float32
    omega_hi::Float32
    init_mode::Symbol   # :uniform or :hippo
end

function PhasorSSM(dims::Pair{Int,Int}, act=normalize_to_unit_circle;
                   init_omega_range=(0.2f0, 2.5f0), init=:uniform)
    @assert init in (:uniform, :hippo) "init must be :uniform or :hippo"
    return PhasorSSM(dims.first, dims.second, act,
                     Float32(init_omega_range[1]), Float32(init_omega_range[2]),
                     init)
end

function Lux.initialparameters(rng::AbstractRNG, l::PhasorSSM)
    W = Float32.(randn(rng, l.out_dims, l.in_dims)) ./ sqrt(Float32(l.in_dims))

    if l.init_mode == :hippo
        λ_init, ω_init = hippo_legs_diagonal(l.out_dims)
        log_neg_lambda = log.(-λ_init)
        omega = ω_init
    else  # :uniform
        log_neg_lambda = fill(Float32(log(0.1)), l.out_dims)
        omega = Float32.(collect(range(l.omega_lo, l.omega_hi; length=l.out_dims)))
    end
    return (weight=W, log_neg_lambda=log_neg_lambda, omega=omega)
end

Lux.initialstates(::AbstractRNG, ::PhasorSSM) = NamedTuple()
Lux.parameterlength(l::PhasorSSM) = l.out_dims * l.in_dims + 2 * l.out_dims

function (l::PhasorSSM)(x::AbstractArray{<:Complex, 3}, ps::LuxParams, st::NamedTuple)
    C_in, L, B = size(x)

    # Build impulse-response kernel from oscillator parameters
    λ = -exp.(ps.log_neg_lambda)                # (out,)  leakage (always negative)
    ω = ps.omega                                 # (out,)  angular frequency
    K = phasor_kernel(λ, ω, 1f0, L)             # out × L

    # Apply weight matrix to mix input channels (all time steps at once)
    xr = reshape(x, C_in, L * B)
    Hr = complex.(ps.weight * real.(xr), ps.weight * imag.(xr))
    H  = reshape(Hr, l.out_dims, L, B)          # out × L × B

    # Temporal integration via causal convolution
    Z = causal_conv(K, H)                        # out × L × B

    # Activation (e.g. project to unit circle)
    Y = l.activation(Z)
    return Y, st
end

# ================================================================
# 5. SSM Readout Layer
# ================================================================

"""
    SSMReadout(readout_frac=0.25)

Bridges the 3D complex SSM output to 2D Phase output for the Codebook layer.
Averages the last `readout_frac` fraction of time steps, normalizes to the
unit circle, and converts to phase angles in [-1, 1].

Input:  (C × L × B) complex  (membrane potentials over time)
Output: (C × B)     Phase    (phase angles, ready for Codebook)
"""
struct SSMReadout <: Lux.AbstractLuxLayer
    readout_frac::Float32
end

SSMReadout() = SSMReadout(0.25f0)

Lux.initialparameters(::AbstractRNG, ::SSMReadout) = NamedTuple()
Lux.initialstates(::AbstractRNG, ::SSMReadout) = NamedTuple()

function (l::SSMReadout)(z::AbstractArray{<:Complex, 3}, ps::LuxParams, st::NamedTuple)
    L = size(z, 2)
    t0 = max(1, L - max(1, round(Int, L * l.readout_frac)) + 1)
    z_avg  = mean(z[:, t0:L, :]; dims=2)[:, 1, :]   # C × B
    z_norm = normalize_to_unit_circle(z_avg)
    return complex_to_angle(z_norm), st               # C × B Phase
end

# ================================================================
# 6. PSK Encoding
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
# 7. Impulse Encoding
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
