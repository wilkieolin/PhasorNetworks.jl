# ================================================================
# Discrete Phasor Kernels for State Space Models
# ================================================================
#
# Pure math functions for causal convolution with damped oscillator
# kernels.  No layer or Lux dependencies.
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
# 2. Causal Convolution
# ================================================================

"""
    causal_conv(K, H) -> ComplexF32 array (C × L × B)

Apply the precomputed impulse-response kernel to a batch of input signals.
This replaces the role of `DifferentialEquations.solve()` in the spiking
system — instead of stepping through time sequentially, we compute all
time steps at once.

Automatically selects the best implementation:
- **FFT-based** (L > 64): O(C·L·log(L)·B) via frequency-domain multiplication.
  Enables long sequences (L=784 for sequential MNIST).
- **Toeplitz matrix** (L ≤ 64): O(C·L²·B) via lower-triangular matrix multiply.
  Lower overhead for short sequences.

# Arguments
- `K::AbstractMatrix{<:Complex}` — **Kernel matrix** (C × L), as returned by
  `phasor_kernel`.  Each row is one oscillator's impulse response over L lags.

- `H::AbstractArray{<:Complex, 3}` — **Input signal** (C × L × B).
  C = number of output channels (oscillators), L = time steps, B = batch size.

# Returns
`Z` (C × L × B): the complex membrane potential at each time step, for each
oscillator, for each batch element.
"""
function causal_conv(K::AbstractMatrix{<:Complex}, H::AbstractArray{<:Complex, 3})
    C, L, B = size(H)
    if L > 64
        return causal_conv_fft(K, H)
    else
        return _causal_conv_toeplitz(K, H)
    end
end

# ---- Toeplitz implementation (O(C·L²·B), efficient for short sequences) ----

"""
    _causal_conv_toeplitz(K, H) -> ComplexF32 array (C × L × B)

Causal convolution via lower-triangular Toeplitz matrix multiplication.
Uses `NNlib.batched_mul` for GPU-friendly parallel computation.
Cost is O(C·L²·B) — prefer `causal_conv_fft` for L > 64.
"""
function _causal_conv_toeplitz(K::AbstractMatrix{<:Complex}, H::AbstractArray{<:Complex, 3})
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

# ---- FFT implementation (O(C·L·log(L)·B), efficient for long sequences) ----

"""
    causal_conv_fft(K, H) -> ComplexF32 array (C × L × B)

Causal convolution via FFT.  Zero-pads kernel and input to length 2L,
multiplies in the frequency domain, then truncates the IFFT output to
the first L elements (the causal part).

Cost is O(C·L·log(L)·B) — dramatically faster than the Toeplitz approach
for long sequences.  Works on both CPU and GPU (CuArrays dispatch to cuFFT
automatically via AbstractFFTs).

Zygote-compatible: `fft`/`ifft` have AD rules via ChainRules.jl.

# Arguments
- `K::AbstractMatrix{<:Complex}` — Kernel (C × L) from `phasor_kernel`.
- `H::AbstractArray{<:Complex, 3}` — Input signal (C × L × B).

# Returns
`Z` (C × L × B): causal convolution result, equivalent to `_causal_conv_toeplitz`.
"""
function causal_conv_fft(K::AbstractMatrix{<:Complex}, H::AbstractArray{<:Complex, 3})
    C, L, B = size(H)
    N = 2 * L  # pad to avoid circular aliasing

    # Zero-pad kernel and input along time axis (non-mutating for Zygote)
    z_K = zero(K)                                          # C × L zeros, same device
    z_H = zero(H)                                          # C × L × B zeros, same device
    K_pad = cat(K, z_K; dims=2)                            # C × N
    H_pad = cat(H, z_H; dims=2)                            # C × N × B

    # FFT along time dimension (dim=2)
    K_f = fft(K_pad, 2)                   # C × N
    H_f = fft(H_pad, 2)                   # C × N × B

    # Pointwise multiply in frequency domain (broadcast K over batch)
    Z_f = reshape(K_f, C, N, 1) .* H_f   # C × N × B

    # IFFT back to time domain and keep first L elements (causal part)
    Z_full = ifft(Z_f, 2)                 # C × N × B
    Z = Z_full[:, 1:L, :]                 # C × L × B

    # Ensure output stays ComplexF32 (FFT may promote to ComplexF64)
    return ComplexF32.(Z)
end

# ================================================================
# 3. Dirac Discretization for Phase Inputs
# ================================================================

"""
    dirac_encode(phases, λ, ω, T; k₀) -> ComplexF32 array (C_out × C_in × L × B)

Encode phase inputs as coupled two-stage Dirac spike responses.

For a spike at phase θ in period n, the spike fires at absolute time
`t_s = n·T + (θ/2 + 0.5)·T`.  The coupled two-stage ODE response at the
end of that period (time `(n+1)·T`) is:

```
G(k_c, k₀, dt) = (exp(k_c·dt) - exp(k₀·dt)) / (k_c - k₀)
```

where `dt = T - t_within = T·(0.5 - θ/2)`.  This encoding captures the
continuous coupling between input oscillators (at k₀) and output oscillators
(at per-channel k_c).

# Arguments
- `phases::AbstractArray{<:Real, 3}` — (C_in, L, B) phases in [-1, 1]
- `λ::AbstractVector` — (C_out,) output channel decay rates
- `ω::AbstractVector` — (C_out,) output channel angular frequencies
- `T::Real` — Oscillation period
- `k₀::Complex` — Global neuron constant for input oscillators

# Returns
(C_out, C_in, L, B) complex array — per-channel Dirac-encoded input.
"""
function dirac_encode(phases::AbstractArray{<:Real, 3},
                      λ::AbstractVector, ω::AbstractVector, T::Real;
                      k₀::Complex=ComplexF32(-0.2f0 + im * Float32(2π)))
    k_c = ComplexF32.(λ .+ im .* ω)                            # (C_out,)
    dt = Float32(T) .* (0.5f0 .- Float32.(phases) ./ 2f0)     # (C_in, L, B)

    k_c_r = reshape(k_c, :, 1, 1, 1)                           # (C_out, 1, 1, 1)
    dt_r = reshape(dt, 1, size(dt)...)                          # (1, C_in, L, B)
    k₀_f = ComplexF32(k₀)

    exp_kc = exp.(k_c_r .* dt_r)                                # (C_out, C_in, L, B)
    exp_k0 = exp.(k₀_f .* dt_r)                                # (1, C_in, L, B)
    dk = k_c_r .- k₀_f                                         # (C_out, 1, 1, 1)
    return (exp_kc .- exp_k0) ./ dk                             # (C_out, C_in, L, B)
end

"""
    causal_conv_dirac(phases, W, λ, ω, T; k₀, spike_energy) -> ComplexF32 (C_out × L × B)

Exact causal convolution with Dirac discretization for phase-valued inputs.

Computes the exact elapsed time from each input spike to each output sample
point and evaluates the coupled two-stage integral analytically.  The key
factoring splits `exp(k·Δt_elapsed)` into a lag-dependent part (causal
convolution kernel) and a phase-dependent part (input encoding):

```
exp(k · [(m-n)·T - t_within]) = exp(k·(m-n)·T) · exp(-k·t_within)
                                 ^^^^^^^^^^^^^^^^   ^^^^^^^^^^^^^^^^^^
                                 lag kernel          phase encoding
```

The coupled integral `G = (exp(k_c·Δt) - exp(k₀·Δt))/(k_c - k₀)` then
decomposes into **two independent causal convolutions**:

```
z_c[m] = E/(k_c-k₀) · { conv(A_c^n, H_c^kc)[m] - conv(A₀^n, H_c^k₀)[m] }
```

- **Term 1**: per-channel encoding `exp(-k_c·t_within)` + per-channel kernel `A_c^n`
- **Term 2**: shared encoding `exp(-k₀·t_within)` + shared kernel `A₀^n`

Both use FFT convolution for O(L·log(L)) per channel.  Total cost is
O(C_out·C_in·L·B) for encoding + O(C_out·L·log(L)·B) for two convolutions.

# Arguments
- `phases::AbstractArray{<:Real, 3}` — (C_in, L, B) phases in [-1, 1]
- `W::AbstractMatrix{<:Real}` — (C_out, C_in) weight matrix
- `λ::AbstractVector` — (C_out,) per-channel decay rates
- `ω::AbstractVector` — (C_out,) per-channel angular frequencies
- `T::Real` — Oscillation period
- `k₀::Complex` — Global neuron constant for input oscillators.
  Default: `-0.2 + i·2π` (standard SpikingArgs defaults)
- `spike_energy::Float32` — Integral of the spike kernel (≈ 2·t_window).
  Default: `0.002` (matches SpikingArgs t_window=0.001)
"""
function causal_conv_dirac(phases::AbstractArray{<:Real, 3},
                           W::AbstractMatrix{<:Real},
                           λ::AbstractVector, ω::AbstractVector, T::Real;
                           k₀::Complex=ComplexF32(-0.2f0 + im * Float32(2π)),
                           spike_energy::Float32=0.002f0)
    C_in, L, B = size(phases)
    C_out = length(λ)
    k_c = ComplexF32.(λ .+ im .* ω)                        # (C_out,)
    k₀_f = ComplexF32(k₀)
    T_f = Float32(T)

    # Time remaining from spike to end of same period: dt = T·(0.5 - θ/2)
    # Spike at step n fires at absolute time (n-1)·T + (θ/2+0.5)·T
    # ODE samples at m·T, so elapsed = m·T - spike_time = (m-n)·T + dt
    # where dt = T·(0.5 - θ/2) is the within-period remainder
    dt = T_f .* (0.5f0 .- Float32.(phases) ./ 2f0)        # (C_in, L, B)

    # GPU-safe lag indices
    ns_cpu = Float32.(0:L-1)
    ns = reshape(typeof(real.(k_c))(ns_cpu), 1, L)         # (1, L) on same device as params

    # === Term 1: per-channel encoding + per-channel kernel ===
    # p_c[c,j,n] = exp(k_c[c] · dt[j,n])
    p_c = exp.(reshape(k_c, :, 1, 1, 1) .* reshape(dt, 1, C_in, L, B))
    # Weight contraction: H^(1)_c[n] = Σ_j W[c,j] · p_c[c,j,n]
    W_r = reshape(ComplexF32.(W), C_out, C_in, 1, 1)
    H1 = dropdims(sum(W_r .* p_c; dims=2); dims=2)        # (C_out, L, B)
    # Per-channel kernel: K_c[n] = exp(k_c · n · T)
    K_c = exp.(k_c .* T_f .* ns)                           # (C_out, L)
    Z1 = causal_conv(K_c, H1)                              # (C_out, L, B)

    # === Term 2: shared encoding + shared kernel ===
    # p₀[j,n] = exp(k₀ · dt[j,n])  — channel-independent
    p₀ = exp.(k₀_f .* dt)                                  # (C_in, L, B)
    # Standard weight multiply (shared across output channels)
    p₀_r = reshape(p₀, C_in, L * B)
    H2 = reshape(complex.(W * real.(p₀_r), W * imag.(p₀_r)), C_out, L, B)
    # Shared kernel: K₀[n] = exp(k₀ · n · T), replicated for causal_conv
    K₀_row = exp.(k₀_f .* T_f .* ns)                       # (1, L)
    K₀ = repeat(K₀_row, C_out, 1)                          # (C_out, L)
    Z2 = causal_conv(K₀, H2)                               # (C_out, L, B)

    # === Combine: z = E · (Z1 - Z2) / (k_c - k₀) ===
    dk = reshape(k_c .- k₀_f, :, 1, 1)                    # (C_out, 1, 1)
    return spike_energy .* (Z1 .- Z2) ./ dk
end

# ================================================================
# 4. HiPPO-LegS Initialization
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
