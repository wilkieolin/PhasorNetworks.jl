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

    # Only cast if FFT promoted away from ComplexF32 (e.g. from mixed precision)
    return eltype(Z) === ComplexF32 ? Z : ComplexF32.(Z)
end

# ================================================================
# 3. Dirac Discretization for Phase Inputs
# ================================================================

"""
    dirac_encode(phases, λ, ω, T) -> ComplexF32 array (C_out × C_in × L × B)

Encode phase inputs as single-oscillator Dirac spike responses.

Each R&F neuron integrates incoming spikes directly.  A spike at phase θ
arrives at time `t_s = (θ/2 + 0.5)·T` within the period, leaving
`dt = T·(0.5 - θ/2)` until the next sample point.  The neuron with
eigenvalue k_c responds as `exp(k_c · dt)`.

All neurons in a layer share the same resonant frequency — spikes carry
phase information via timing, and the receiving neuron's response depends
only on its own dynamics k_c and the elapsed time dt.

# Arguments
- `phases::AbstractArray{<:Real, 3}` — (C_in, L, B) phases in [-1, 1]
- `λ::AbstractVector` — (C_out,) per-channel decay rates
- `ω::AbstractVector` — (C_out,) per-channel angular frequencies
- `T::Real` — Oscillation period

# Returns
(C_out, C_in, L, B) complex array — per-channel Dirac-encoded input.
"""
function dirac_encode(phases::AbstractArray{<:Real, 3},
                      λ::AbstractVector, ω::AbstractVector, T::Real)
    k_c = ComplexF32.(λ .+ im .* ω)                            # (C_out,)
    dt = Float32(T) .* (0.5f0 .- Float32.(phases) ./ 2f0)     # (C_in, L, B)

    k_c_r = reshape(k_c, :, 1, 1, 1)                           # (C_out, 1, 1, 1)
    dt_r = reshape(dt, 1, size(dt)...)                          # (1, C_in, L, B)
    return exp.(k_c_r .* dt_r)                                  # (C_out, C_in, L, B)
end

"""
    _exp_kdt(k, dt) -> exp.(k .* dt)

Element-wise `exp(k * dt)` with broadcasted shapes — used inside
`causal_conv_dirac`'s per-group encoding. Behaviorally identical to
`exp.(k .* dt)`, but the closed-form `rrule` below bypasses Zygote's
generic `broadcast_forward` adjoint, which lifts each output to
`Complex{ForwardDiff.Dual{Float32, 2}}` (24 bytes/element) and balloons
the saved tape ~3× over `ComplexF32` (8 bytes/element). At
`(C_out=64, C_in=64, L=784, B=64)` with the default `group_size=8`,
this saves ~10 GiB across the 3 Q/K/V Phase-3D PhasorDense layers in
SSMSelfAttention.

`k` may be complex or real; `dt` is real. Result is complex when `k` is
complex.
"""
_exp_kdt(k::AbstractArray, dt::AbstractArray) = exp.(k .* dt)

"""
    rrule(_exp_kdt, k, dt)

Closed-form pullback for `enc = exp.(k .* dt)`:

    ∂enc[i,j,k] / ∂k[i]      = dt[j,k] * enc[i,j,k]
    ∂enc[i,j,k] / ∂dt[j,k]   = k[i]    * enc[i,j,k]

For complex output cotangent `ḡ`, using ChainRules conventions
(holomorphic in `k`, real-input projection for `dt`):

    k̄[i]      = Σ_jk dt[j,k] · conj(enc[i,j,k]) · ḡ[i,j,k]
    dt̄[j,k]   = Re( Σ_i conj(k[i]) · conj(enc[i,j,k]) · ḡ[i,j,k] )

Sums are taken over whichever broadcast axes `k` and `dt` were spread
across, so the returned cotangents match each input's original shape.
The only `(g, C_in, L*B)` tensor saved on the tape is `enc` itself
(ComplexF32, not Dual), giving the 3× memory reduction.
"""
function ChainRulesCore.rrule(::typeof(_exp_kdt),
                              k::AbstractArray, dt::AbstractArray)
    enc = _exp_kdt(k, dt)
    function _exp_kdt_pullback(ḡ_)
        ḡ = unthunk(ḡ_)
        # Common factor: ḡ * conj(enc) (same shape as enc)
        contrib = ḡ .* conj.(enc)

        # k̄: sum over dims where k was broadcasted (i.e. dims of size 1 in k)
        k_grad_full = contrib .* dt
        k_grad = _sum_broadcast_dims(k_grad_full, size(k))

        # dt̄: sum over dims where dt was broadcasted, then take real
        dt_grad_full = real.(conj.(k) .* contrib)
        dt_grad = _sum_broadcast_dims(dt_grad_full, size(dt))

        return (NoTangent(), k_grad, dt_grad)
    end
    return enc, _exp_kdt_pullback
end

"""
    _sum_broadcast_dims(x, target_shape) -> Array

Sum `x` over every axis where `target_shape` has size 1 but `x` has size
> 1. The returned array has `ndims(x)` dims (singleton along reduced
axes), then is reshaped to `target_shape`. Used by the `_exp_kdt`
pullback to collapse broadcasted axes back to each input's shape.
"""
function _sum_broadcast_dims(x::AbstractArray, target_shape::NTuple{N,Int}) where {N}
    @assert ndims(x) == N "ndims mismatch: $(ndims(x)) vs target $N"
    reduce_dims = ntuple(i -> size(x, i) != target_shape[i], N)
    if any(reduce_dims)
        dims_to_sum = Tuple(i for i in 1:N if reduce_dims[i])
        x = sum(x; dims=dims_to_sum)
    end
    return reshape(x, target_shape)
end

"""
    causal_conv_dirac(phases, W, λ, ω, T) -> ComplexF32 (C_out × L × B)

Causal convolution with Dirac discretization for phase-valued inputs.

Models a single R&F neuron per output channel directly integrating incoming
spikes.  A spike at phase θ arrives at time `t_s = (θ/2 + 0.5)·T` within
the period.  The neuron at eigenvalue k_c responds with `exp(k_c · dt)` where
`dt = T·(0.5 - θ/2)` is the time remaining until the next sample point.

The computation factors into a phase-dependent encoding and a lag-dependent
causal convolution kernel:

```
exp(k_c · Δt_elapsed) = exp(k_c · lag·T) · exp(k_c · dt)
                         ^^^^^^^^^^^^^^^^   ^^^^^^^^^^^^^^^
                         kernel (FFT-able)  phase encoding
```

So: `z_c[m] = Σ_n K_c[m-n] · H_c[n]`  where  `K_c[n] = exp(k_c·n·T)`
and  `H_c[n] = Σ_j W[c,j] · exp(k_c · dt_j[n])`.

# Arguments
- `phases::AbstractArray{<:Phase, 3}` — (C_in, L, B) phases in [-1, 1]
- `W::AbstractMatrix{<:Real}` — (C_out, C_in) weight matrix
- `λ::AbstractVector` — (C_out,) per-channel decay rates
- `ω::AbstractVector` — (C_out,) per-channel angular frequencies
- `T::Real` — Oscillation period
"""
function causal_conv_dirac(phases::AbstractArray{<:Phase, 3},
                           W::AbstractMatrix{<:Real},
                           λ::AbstractVector, ω::AbstractVector, T::Real;
                           group_size::Int = 8)
    C_in, L, B = size(phases)
    C_out = length(λ)
    k_c = ComplexF32.(λ .+ im .* ω)                        # (C_out,)
    T_f = Float32(T)

    # Time remaining from spike to end of same period: dt = T·(0.5 - θ/2).
    # Phase / Float32 promotes to Float32 elementwise (per the Phase
    # type's promotion rules), so no explicit cast is needed.
    dt = T_f .* (0.5f0 .- phases ./ 2f0)                   # (C_in, L, B) Float32

    # Grouped diagonal encoding: compute H_c[n] = Σ_j W[c,j] · exp(k_c · dt_j[n])
    # Processes G output channels at once to reduce GPU kernel launch overhead
    # while avoiding the full (C_out, C_in, L, B) tensor.
    # Uses map (not a mutating loop) for Zygote compatibility.
    # Avoids scalar indexing for GPU compatibility — uses range slicing.
    W_c = ComplexF32.(W)                                    # (C_out, C_in)
    dt_flat = reshape(dt, 1, C_in, L * B)                  # (1, C_in, L*B)
    G = min(group_size, C_out)

    H_slices = map(1:G:C_out) do c_start
        c_end = min(c_start + G - 1, C_out)
        k_group = reshape(k_c[c_start:c_end], :, 1, 1)     # (g, 1, 1)
        enc = _exp_kdt(k_group, dt_flat)                     # (g, C_in, L*B); custom rrule avoids Dual blowup
        w_group = reshape(W_c[c_start:c_end, :], :, C_in, 1) # (g, C_in, 1)
        h = sum(w_group .* enc; dims=2)                      # (g, 1, L*B)
        reshape(dropdims(h; dims=2), :, L, B)                # (g, L, B)
    end
    H = reduce(vcat, H_slices)                              # (C_out, L, B)

    # Causal convolution kernel: K_c[n] = exp(k_c · n · T)
    ns_cpu = Float32.(0:L-1)
    ns = reshape(typeof(real.(k_c))(ns_cpu), 1, L)         # (1, L) GPU-safe
    K = exp.(k_c .* T_f .* ns)                             # (C_out, L)

    return causal_conv(K, H)
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

!!! note "Per-channel ω rule"
    The returned `ω` carries a HiPPO-style per-channel frequency spread
    that is meaningful in *frequency-decomposition* contexts (i.e.
    [`ResonantSTFT`](@ref)). Phase-locked layers — `PhasorDense`,
    `PhasorConv`, `PhasorFixed`, `PhasorResonant` — discard it and use
    a single shared `ω = 2π` so output phases remain commensurable for
    HD-VSA downstream operations. Only `λ` from this function is used
    by those layers' `:hippo` init mode.
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
