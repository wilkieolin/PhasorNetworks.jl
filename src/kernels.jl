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

"""
    bias_kernel_accumulation(λ, ω, T, L) -> AbstractMatrix{ComplexF32}

Period-by-period accumulation factor for a constant per-period drive
through the resonant kernel. Returns `(C, L)` complex matrix where

    G[c, m] = Σ_{n=0}^{m} exp(k_c · n · T)
            = (1 - exp(k_c · (m+1) · T)) / (1 - exp(k_c · T))

with `k_c = λ_c + i·ω_c`. Used to compute the contribution of a constant
per-period bias `b[c]` to the SSM output as `bias_contrib[c, m] = b[c] · G[c, m]`
— the discrete equivalent of injecting `bias_current` continuously in
the ODE rather than adding `bias` once to the conv output. See
`PhasorDense._forward_3d_dirac` for the use site.

The closed form is well-defined for `λ < 0` (so |exp(k·T)| < 1, denominator nonzero).
At |A| = 1 (purely oscillatory, λ = 0) the formula degenerates to `m+1` in the
limit but is implemented as the literal ratio, which is fine for `λ_c < 0`.
"""
function bias_kernel_accumulation(λ::AbstractVector, ω::AbstractVector,
                                  T::Real, L::Int)
    k   = ComplexF32.(λ .+ im .* ω)                                    # (C,)
    T_f = Float32(T)
    A   = exp.(k .* T_f)                                                # (C,) per-period factor
    ns_cpu = Float32.(1:L)                                              # (L,) — m+1 for m=0..L-1
    ns = reshape(typeof(real.(k))(ns_cpu), 1, L)                        # (1, L) on-device
    A_pow = exp.(k .* T_f .* ns)                                        # (C, L) = A^(m+1)
    return (1f0 .- A_pow) ./ reshape(1f0 .- A, :, 1)                    # (C, L)
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
    _spike_dt(phases, T) -> Float32 array

Time from a spike at phase θ to the end-of-slot sample point. The spike sits at
`t_s = mod((θ/2 + ½)·T, T)` — the same arithmetic as [`phase_to_time`](@ref) — so
`dt = T − t_s`. Both θ = +1 and θ = −1 map to the **start** of the slot
(`dt = T`), exactly as the spiking encoder places them, so the discrete (Dirac)
and spiking paths agree on the slot edge.

The edge itself is a genuine discontinuity of the slotted representation: just
below θ = +1 the spike is at the end of the slot (`dt → 0`, full weight); at
θ = ±1 it is at the start (`dt = T`, weight `e^{λT}`). The contribution's phase
is continuous (ωT = 2π); its magnitude jumps by `e^{λT}`. Real weights and
real biases keep the real axis invariant, so inputs on {0, ±1} (e.g. blank
rows) can park units exactly on the edge — see [`near_wrap_fraction`](@ref) and
[`default_bias`](@ref).

Gradient: `∂dt/∂θ = −T/2` (almost everywhere), as before.
"""
function _spike_dt(phases, T::Real)
    T_f = Float32(T)
    return T_f .- mod.((Float32.(phases) ./ 2f0 .+ 0.5f0) .* T_f, T_f)
end

function ChainRulesCore.rrule(::typeof(_spike_dt), phases, T::Real)
    y = _spike_dt(phases, T)
    half = Float32(T) / 2f0
    function _spike_dt_pullback(ȳ_)
        ȳ = unthunk(ȳ_)
        ȳ isa ChainRulesCore.AbstractZero && return (NoTangent(), ȳ, NoTangent())
        return (NoTangent(), -half .* ȳ, NoTangent())
    end
    return y, _spike_dt_pullback
end

"""
    dirac_encode(phases, λ, ω, T) -> ComplexF32 array (C_out × C_in × L × B)

Encode phase inputs as single-oscillator Dirac spike responses.

Each R&F neuron integrates incoming spikes directly.  A spike at phase θ
arrives at time `t_s = (θ/2 + 0.5)·T` within the period, leaving
`dt = T·(0.5 - θ/2)` until the next sample point (θ = ±1 both mean the slot
start, `dt = T`; see [`_spike_dt`](@ref)).  The neuron with
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
    dt = _spike_dt(phases, T)                                  # (C_in, L, B)

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
    _sum_trailing(x, nkeep) -> Array of size `size(x)[1:nkeep]`

Sum `x` over all axes after the first `nkeep`, as one matrix–vector product
(`reshape(x, m, :) * ones`). Equivalent to `sum(x; dims=nkeep+1:ndims(x))`,
but a reduction that keeps axis 1 is nearly serial on oneAPI's Aurora LTS
stack (`ONEAPI_LTS=1` reduces it with one work-item per output element),
while a GEMV is parallel on every backend.
"""
function _sum_trailing(x::AbstractArray, nkeep::Int)
    X = reshape(x, prod(size(x)[1:nkeep]), :)
    return reshape(X * fill!(similar(X, size(X, 2)), one(eltype(X))), size(x)[1:nkeep])
end

"""
    _dirac_encode_sum(k, W, dt, q, G) -> ComplexF32 (C_out × N)

`H[c, n] = Σ_j W[c,j] · q[j,n] · exp(k[c] · dt[j,n])` (`q === nothing` = unit
charge), computed `G` output channels at a time so only a `(G, C_in, N)`
encoding exists at once. `k` is `(C_out,)` complex, `W` `(C_out, C_in)` real,
`dt` and `q` `(C_in, N)` real.

The rrule recomputes each group's encoding in the pullback instead of saving
it. Under plain Zygote every group's `(G, C_in, N)` tensor stayed on the tape,
so a gradient step held O(C_out · C_in · N) memory: ~63 GiB for one
PhasorLCA 256→256 at L 500, B 16. With the rrule the tape holds only the
inputs, and the backward pass peaks at a few `(G, C_in, N)` temporaries.
"""
function _dirac_encode_sum(k::AbstractVector, W::AbstractMatrix{<:Real},
                           dt::AbstractMatrix, q, G::Int)
    C_out, C_in = size(W)
    N = size(dt, 2)
    W_c = ComplexF32.(W)
    dt_r = reshape(dt, 1, C_in, N)
    q_r = q === nothing ? nothing : reshape(q, 1, C_in, N)
    H_slices = map(1:G:C_out) do c_start
        cs = c_start:min(c_start + G - 1, C_out)
        enc = _exp_kdt(reshape(k[cs], :, 1, 1), dt_r)       # (g, C_in, N)
        q_r === nothing || (enc = enc .* q_r)
        dropdims(sum(reshape(W_c[cs, :], :, C_in, 1) .* enc; dims=2); dims=2)  # (g, N)
    end
    return reduce(vcat, H_slices)
end

function ChainRulesCore.rrule(::typeof(_dirac_encode_sum), k::AbstractVector,
                              W::AbstractMatrix{<:Real}, dt::AbstractMatrix, q, G::Int)
    H = _dirac_encode_sum(k, W, dt, q, G)
    function _dirac_encode_sum_pullback(H̄_)
        H̄ = unthunk(H̄_)
        H̄ isa ChainRulesCore.AbstractZero &&
            return (NoTangent(), ZeroTangent(), ZeroTangent(), ZeroTangent(), ZeroTangent(), NoTangent())
        C_out, C_in = size(W)
        N = size(dt, 2)
        dt_r = reshape(dt, 1, C_in, N)
        q_r = q === nothing ? nothing : reshape(q, 1, C_in, N)
        Wf = Float32.(W)
        k̄ = similar(k, ComplexF32, C_out)
        W̄ = similar(Wf, Float32, C_out, C_in)
        dt̄ = fill!(similar(dt, Float32, C_in, N), 0f0)
        q̄ = q === nothing ? NoTangent() : fill!(similar(q, Float32, C_in, N), 0f0)
        for c_start in 1:G:C_out
            cs = c_start:min(c_start + G - 1, C_out)
            kg = reshape(k[cs], :, 1, 1)                          # (g, 1, 1)
            Wg = reshape(Wf[cs, :], :, C_in, 1)                   # (g, C_in, 1)
            Bc = conj.(_exp_kdt(kg, dt_r)) .* reshape(H̄[cs, :], :, 1, N)   # conj(e)·H̄, (g, C_in, N)
            if q_r !== nothing
                # q̄[j,n] = Re Σ_c W[c,j] conj(e) H̄
                q̄ .+= dropdims(real.(sum(Wg .* Bc; dims=1)); dims=1)
                Bc = Bc .* q_r                                    # conj(q e)·H̄
            end
            # W real: W̄[c,j] = Re Σ_n conj(q e) H̄
            # (_sum_trailing: these two keep axis 1, few outputs; see its docstring)
            W̄[cs, :] .= real.(_sum_trailing(Bc, 2))
            # k̄[c] = Σ_{j,n} W dt conj(q e) H̄   (holomorphic in k)
            k̄[cs] .= _sum_trailing(Wg .* dt_r .* Bc, 1)
            # dt̄[j,n] = Re Σ_c W conj(k) conj(q e) H̄
            dt̄ .+= dropdims(real.(sum(Wg .* conj.(kg) .* Bc; dims=1)); dims=1)
        end
        return (NoTangent(), k̄, W̄, dt̄, q̄, NoTangent())
    end
    return H, _dirac_encode_sum_pullback
end

"""
    causal_conv_dirac(phases, W, λ, ω, T) -> ComplexF32 (C_out × L × B)

Causal convolution with Dirac discretization for phase-valued inputs.

Models a single R&F neuron per output channel directly integrating incoming
spikes.  A spike at phase θ arrives at time `t_s = (θ/2 + 0.5)·T` within
the period.  The neuron at eigenvalue k_c responds with `exp(k_c · dt)` where
`dt = T·(0.5 - θ/2)` is the time remaining until the next sample point
(θ = ±1 both mean the slot start, `dt = T`; see [`_spike_dt`](@ref)).

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

# Keyword arguments
- `charges = nothing` — optional `(C_in, L, B)` per-spike charge (Hill
  emission): each input spike enters with amplitude `q` instead of 1, i.e.
  `H_c[n] = Σ_j W[c,j] · q_j[n] · exp(k_c · dt_j[n])`. `nothing` = unit
  charge (unchanged path). A [`Charged`](@ref) input supplies them itself.
"""
function causal_conv_dirac(phases::AbstractArray{<:Phase, 3},
                           W::AbstractMatrix{<:Real},
                           λ::AbstractVector, ω::AbstractVector, T::Real;
                           group_size::Int = 8,
                           charges::Union{Nothing, AbstractArray{<:Real, 3}} = nothing)
    C_in, L, B = size(phases)
    C_out = length(λ)
    k_c = ComplexF32.(λ .+ im .* ω)                        # (C_out,)
    T_f = Float32(T)

    # Time remaining from spike to end of same period: dt = T·(0.5 - θ/2).
    # Phase / Float32 promotes to Float32 elementwise (per the Phase
    # type's promotion rules), so no explicit cast is needed.
    dt = _spike_dt(phases, T_f)                            # (C_in, L, B) Float32

    # Grouped diagonal encoding: H_c[n] = Σ_j W[c,j] · q_j[n] · exp(k_c · dt_j[n]),
    # G output channels at a time (see `_dirac_encode_sum`, whose rrule
    # recomputes each group's encoding in the backward pass instead of keeping
    # it on the tape).
    G = min(group_size, C_out)
    charges === nothing || size(charges) == size(phases) ||
        throw(DimensionMismatch("charges $(size(charges)) must match phases $(size(phases))"))
    dt_flat = reshape(dt, C_in, L * B)                      # (C_in, L*B)
    q_flat = charges === nothing ? nothing : reshape(charges, C_in, L * B)
    H = reshape(_dirac_encode_sum(k_c, W, dt_flat, q_flat, G), C_out, L, B)

    # Causal convolution kernel: K_c[n] = exp(k_c · n · T)
    ns_cpu = Float32.(0:L-1)
    ns = reshape(typeof(real.(k_c))(ns_cpu), 1, L)         # (1, L) GPU-safe
    K = exp.(k_c .* T_f .* ns)                             # (C_out, L)

    return causal_conv(K, H)
end

causal_conv_dirac(x::Charged, W::AbstractMatrix{<:Real}, λ::AbstractVector,
                  ω::AbstractVector, T::Real; kwargs...) =
    causal_conv_dirac(x.phase, W, λ, ω, T; charges = x.q, kwargs...)

# ================================================================
# 3b. Hill emission (per-spike charge)
# ================================================================

"""
    hill_charge(r, log_theta, n) -> q

Hill emission charge `q = rⁿ/(rⁿ + θⁿ)` with `θ = exp(log_theta)`, evaluated
as `σ(n·(log r − log θ))` (identical value; finite gradients for every `r > 0`
and every `n`). `log_theta` broadcasts against `r` (a 1-element vector = one
θ per layer; a `(C,)` vector = one per channel). `n = 1` gives
`q = r/(r + θ)`.

Used wherever a unit's complex state `z` is emitted onto the wire with
`emit_n < Inf`: the wire carries charge `q(|z|)` with the phase of `z`
(see [`Charged`](@ref), and the per-spike `charges` of [`SpikeTrain`](@ref)).
"""
hill_charge(r::AbstractArray{<:Real}, log_theta, n::Real) =
    sigmoid.(Float32(n) .* (log.(r) .- log_theta))

# |z| regularised at the origin (r ≥ 1e-6): keeps log r finite and the
# gradient d r / d z = z / r at z = 0 equal to 0.
_emit_r(z::AbstractArray{<:Complex}) = sqrt.(abs2.(real.(z)) .+ abs2.(imag.(z)) .+ 1f-12)

# Emit a complex state field Z (C, L, B) with Hill emission: Charged(phase(Y), q(|Z|)),
# Y = activation(Z) (the activation only shapes the phase, as without emission).
# θ is frozen (no gradient) unless `learn_theta`.
function _hill_emit(Z::AbstractArray{<:Complex, 3}, log_theta, n::Real, learn_theta::Bool;
                    phase = complex_to_angle(Z))
    lt = learn_theta ? log_theta : ignore_derivatives(log_theta)
    return Charged(phase, hill_charge(_emit_r(Z), lt, n))
end

# Dirac ↔ ODE amplitude map for emission in the spiking path. One spike
# delivers the current spk_scale·k(t − t_s) with the raised-cosine kernel k of
# half-width 2·t_window, whose integral is 2·t_window; the discrete (Dirac)
# path models the same spike as a unit impulse. The resonator is linear and
# the kernel is short against the period, so the cycle-sample ODE amplitude
# is |u| ≈ g·|Z_dirac| with g = 2·t_window·spk_scale (verified in
# test_hill_emission.jl). The bias current carries no spk_scale, so with
# spk_scale ≠ 1 the ODE and Dirac paths already disagree on the signal/bias
# balance; spk_scale = 1 is the parity case.
_emit_gain(spk_args::SpikingArgs) = 2f0 * spk_args.t_window * spk_args.spk_scale

# charge_fn for `solution_to_train`: ODE amplitude |u| → q(|u|/g); θ is in Dirac units.
function _spike_charge_fn(log_theta, n::Real, spk_args::SpikingArgs)
    g = _emit_gain(spk_args)
    lt = ignore_derivatives(log_theta)
    return r -> hill_charge(max.(r ./ g, 1f-6), lt, n)
end

_check_emit_n(n::Real) = (n > 0 || throw(ArgumentError("emit_n must be > 0 (Inf = off), got $n")); Float32(n))

# ================================================================
# 4. HiPPO-LegS Initialization
# ================================================================

# Time-constant span (in discrete steps, Δt=1) for the log-spaced :hippo
# spectrum. τ = 1/|λ|, so these bound how far back each channel remembers.
#   HIPPO_TAU_MIN — fastest channel: a near-instant "read what's here now" tap.
#   HIPPO_TAU_MAX — slowest channel: a long-memory integrator (a genuine tape).
# Rationale: the earlier default (λ log-spaced 0.5 … N-0.5) capped the SLOWEST
# channel at τ=2 steps — i.e. it was a spread of *short* timescales, not a tape,
# and its extreme fast end (λ≈-63.5) drove projection outputs to the origin
# (complex_to_angle gradient blow-up). Anchoring the slow end at τ=HIPPO_TAU_MAX
# makes :hippo a real short-AND-long multi-timescale basis and gentles the fast
# end. λ only shapes dynamics in the 3D SSM / ODE path (no-op in 2D static).
const HIPPO_TAU_MIN = 0.5f0
const HIPPO_TAU_MAX = 64.0f0

"""
    hippo_legs_diagonal(N; tau_max=nothing, clip_decay=nothing) -> (λ, ω)

HiPPO-LegS (Legendre Scaled) inspired diagonal initialization.

The HiPPO framework defines state matrices for online function approximation;
the LegS variant projects input history onto a scaled Legendre basis, giving a
principled multi-timescale memory. This routine returns a diagonal (per-channel)
λ spectrum log-spaced across time-constants so a layer holds a spread of memory
horizons simultaneously — short (recent detail) through long (a memory tape).

# Arguments
- `N::Int` — Number of oscillators (state dimension).

# Keyword arguments
- `tau_max`: Longest time-constant (in steps) of the slowest channel. When
  `nothing` (default), uses `HIPPO_TAU_MAX`. λ magnitudes are log-spaced from
  `1/tau_max` (slow, long memory) up to `1/HIPPO_TAU_MIN` (fast, near-instant),
  giving τ ∈ [`HIPPO_TAU_MIN`, `tau_max`] across the N channels.
- `clip_decay`: Legacy linear-HiPPO path. When set, λ magnitudes are
  `min(n + 1/2, clip_decay)` for n = 0…N-1 (overrides `tau_max`).

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
function hippo_legs_diagonal(N::Int; tau_max::Union{Nothing, Real}=nothing,
                             tau_min::Union{Nothing, Real}=nothing,
                             clip_decay::Union{Nothing, Real}=nothing)
    if clip_decay !== nothing
        # Legacy linear HiPPO with hard clip.
        ns = Float32.(0:N-1)
        λ_mag = min.(ns .+ 0.5f0, Float32(clip_decay))
    else
        # Log-spaced time-constants τ ∈ [tau_min, tau_max]: the slow end is a
        # long-memory integrator, the fast end a near-instant read tap. Widening
        # this range (esp. tau_max) is the SSM's "timescale-range" knob.
        τmax   = tau_max === nothing ? HIPPO_TAU_MAX : Float32(tau_max)
        τmin   = tau_min === nothing ? HIPPO_TAU_MIN : Float32(tau_min)
        λ_slow = 1f0 / τmax            # small |λ| → long memory
        λ_fast = 1f0 / τmin            # large |λ| → short memory
        λ_mag  = Float32.(exp.(range(log(λ_slow), log(λ_fast); length=N)))
    end
    λ = -λ_mag
    # Frequency paired to decay: one oscillation per memory window
    ω = Float32(π) .* λ_mag
    return λ, ω
end
