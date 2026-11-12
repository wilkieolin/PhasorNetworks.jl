# Device specs: the dimensionless simulation description of one hardware
# platform, built from a DeviceCard (physical units, tagged values) by
# `device_spec`, or by hand from the sub-spec keyword constructors. Every
# sub-spec defaults to the shared-ω ideal (docs/devices.md §1), so
# `DeviceSpec()` is the ideal device and a spec only describes departures
# from it. Units follow docs/devices.md C1: times in nominal periods T,
# rates as rate·T, frequencies relative to ω₀ = 2π/T.
#
# Nothing here simulates; per-unit draws and the layers that consume a spec
# come later (on_device, ResonatorUnit, LandauUnit).

# ---------------------------------------------------------------- periodic functions of the orbit angle

"""
    PeriodicFn

A function of the orbit angle `θ ∈ (−π, π]`, periodic in `2π`: the orbit
speed `S(θ)` (real) or the kick response `G(θ)` (complex) of an
[`OrbitSpec`](@ref). Concrete types: [`FourierFn`](@ref) (differentiable,
GPU-friendly, the default) and [`TableFn`](@ref) (periodic linear
interpolation of a table, for exact parity with table-driven experiments).
"""
abstract type PeriodicFn end

"""
    FourierFn{T, N}(c::NTuple{N, ComplexF32})
    FourierFn(x::Real)  /  FourierFn(x::Complex)

A truncated Fourier series `f(θ) = Σ_{k=−K}^{K} c_k e^{ikθ}` with
`N = 2K + 1` coefficients stored lowest harmonic first. `T` is the output
type: `Float32` returns the real part, `ComplexF32` the complex value. The
coefficients are a tuple, so the struct is `isbits` and broadcasts on the
GPU. `FourierFn(x)` is the constant `x`. Build one with
[`fourier_fit`](@ref), [`speed_distortion`](@ref) or
[`kick_distortion`](@ref).
"""
struct FourierFn{T <: Union{Float32, ComplexF32}, N} <: PeriodicFn
    c::NTuple{N, ComplexF32}
    function FourierFn{T, N}(c::NTuple{N, ComplexF32}) where {T, N}
        isodd(N) || throw(ArgumentError("FourierFn needs 2K + 1 coefficients, got $N"))
        return new{T, N}(c)
    end
end
FourierFn{T}(c::NTuple{N, <:Number}) where {T, N} = FourierFn{T, N}(ComplexF32.(c))
FourierFn(x::Real) = FourierFn{Float32}((ComplexF32(x),))
FourierFn(x::Complex) = FourierFn{ComplexF32}((ComplexF32(x),))

function (f::FourierFn{T, N})(θ) where {T, N}
    K = (N - 1) ÷ 2
    s = sum(ntuple(j -> f.c[j] * cis((j - 1 - K) * θ), Val(N)))
    return T === Float32 ? real(s) : s
end

"""Number of harmonics `K` of a [`FourierFn`](@ref)."""
nharmonics(::FourierFn{T, N}) where {T, N} = (N - 1) ÷ 2

"""
    TableFn(v::AbstractVector)

Periodic linear interpolation of the samples `v` on the uniform grid
`θ_j = −π + 2π(j − 1)/n`, `j = 1…n`. Real samples give a real (`Float32`)
function, complex samples a complex (`ComplexF32`) one. This is the rule of
the SD1 orbit-distortion experiments, kept for exact parity; prefer
[`FourierFn`](@ref) elsewhere (it is smooth, differentiable in `θ`, and runs
on the GPU).
"""
struct TableFn{T <: Union{Float32, ComplexF32}, V <: AbstractVector} <: PeriodicFn
    v::V
end
TableFn(v::AbstractVector{<:Real}) = TableFn{Float32, Vector{Float64}}(Float64.(collect(v)))
TableFn(v::AbstractVector{<:Complex}) = TableFn{ComplexF32, Vector{ComplexF64}}(ComplexF64.(collect(v)))

function (p::TableFn{T})(θ) where {T}
    n = length(p.v); x = (mod(θ + π, 2π)) / (2π) * n
    i = floor(Int, x); f = x - i; i = mod(i, n)
    return T((1 - f) * p.v[i + 1] + f * p.v[mod(i + 1, n) + 1])
end

_is_const_one(f::FourierFn) = length(f.c) == 1 && f.c[1] == 1
_is_const_one(f::PeriodicFn) = false

_theta_grid(n) = [-π + 2π * (j - 1) / n for j in 1:n]

"""
    fourier_fit(v::AbstractVector; nharm = 8) -> (f::FourierFn, residual)

Fit the samples `v` on the grid `θ_j = −π + 2π(j − 1)/n` with a Fourier
series of `nharm` harmonics (`nharm ≤ (n − 1) ÷ 2`), by the discrete Fourier
transform. Real samples give a real series. `residual` is the largest
difference between the series and the [`TableFn`](@ref) of the same samples
on a grid 16× finer, relative to `max |v|`: the cost of replacing table
interpolation by the truncated series.
"""
function fourier_fit(v::AbstractVector; nharm::Integer = 8)
    n = length(v)
    n >= 3 || throw(ArgumentError("fourier_fit needs at least 3 samples"))
    K = min(Int(nharm), (n - 1) ÷ 2)
    K >= 0 || throw(ArgumentError("nharm must be ≥ 0"))
    θ = _theta_grid(n)
    c = ntuple(j -> ComplexF32(sum(v[i] * cis(-(j - 1 - K) * θ[i]) for i in 1:n) / n), 2K + 1)
    T = eltype(v) <: Real ? Float32 : ComplexF32
    f = FourierFn{T, 2K + 1}(c)
    tab = TableFn(v)
    fine = range(-π, π; length = 16n + 1)
    scale = max(maximum(abs, v), eps(Float64))
    residual = maximum(abs(f(t) - tab(t)) for t in fine) / scale
    return f, Float64(residual)
end

"""
    speed_distortion(a; k = 1, phase = 0) -> FourierFn{Float32}

The orbit speed `S(θ) = (1 + a·cos(kθ + phase)) / √(1 − a²)`, normalised so
that `⟨1/S⟩_θ = 1` (docs/devices.md C4: the period is unchanged). `|a| < 1`.
`a = 0` is the ideal `S ≡ 1`. This is SD1's phenomenological `P` axis.
"""
function speed_distortion(a::Real; k::Integer = 1, phase::Real = 0)
    abs(a) < 1 || throw(ArgumentError("speed distortion needs |a| < 1, got $a"))
    a == 0 && return FourierFn(1f0)
    k >= 1 || throw(ArgumentError("harmonic k must be ≥ 1"))
    nP = 1 / sqrt(1 - a^2)
    c = zeros(ComplexF64, 2k + 1)
    c[k + 1] = nP
    c[1] += nP * a / 2 * cis(-phase)      # e^{−ikθ}
    c[end] += nP * a / 2 * cis(phase)     # e^{+ikθ}
    return FourierFn{Float32}(Tuple(c))
end

"""
    kick_distortion(a; k = 1, phase = 0) -> FourierFn{ComplexF32}

The kick response `G(θ) = 1 + a·cos(kθ + phase)`: how strongly an input
moves the unit at each point of its orbit. `a = 0` is the ideal `G ≡ 1`.
This is SD1's phenomenological `W` axis. Device tables (complex) come
through [`fourier_fit`](@ref) or [`TableFn`](@ref) instead.
"""
function kick_distortion(a::Real; k::Integer = 1, phase::Real = 0)
    a == 0 && return FourierFn(1f0 + 0f0im)
    k >= 1 || throw(ArgumentError("harmonic k must be ≥ 1"))
    c = zeros(ComplexF64, 2k + 1)
    c[k + 1] = 1
    c[1] += a / 2 * cis(-phase)
    c[end] += a / 2 * cis(phase)
    return FourierFn{ComplexF32}(Tuple(c))
end

"""
    mean_inverse_speed(S::PeriodicFn; n = 4096)

`⟨1/S⟩_θ`, the average over the orbit angle. Convention C4 requires it to be
1; [`device_spec`](@ref) rescales card tables to meet it.
"""
mean_inverse_speed(S; n::Integer = 4096) = sum(1 / S(-π + 2π * (j - 0.5) / n) for j in 1:n) / n

_scale(f::FourierFn{T, N}, s) where {T, N} = FourierFn{T, N}(ComplexF32.(f.c .* s))
_scale(f::TableFn, s) = TableFn(f.v .* s)

# ---------------------------------------------------------------- amplitude dependence of frequency (C2)

"""
    AmpFn

The non-isochrony factor `F` of convention C2: a unit at amplitude `r`
rotates at `ω_c·F(ρ)`, `ρ = r / r_ref`. Called on `x = ln ρ`. Every shape is
parameterised by the same `β = dF/d ln ρ` at `ρ = 1`. Build one with
`AmpFn(shape; beta, L, table)`; shapes:

| shape | `F` | concrete type |
|---|---|---|
| `:none` | `1` | `IsoAmp` |
| `:loglinear` | `1 + β·clamp(ln ρ, ±L)`, `L = ln 3` | `LogLinearAmp` |
| `:linear` | `1 + β(ρ − 1)` | `LinearAmp` |
| `:quadratic` | `1 + (β/2)(ρ² − 1)` (Stuart–Landau) | `QuadraticAmp` |
| `:table` | linear interpolation of `(x, F)`, flat outside | `TableAmp` |
"""
abstract type AmpFn end
struct IsoAmp <: AmpFn end
struct LogLinearAmp <: AmpFn
    β::Float32
    L::Float32
end
struct LinearAmp <: AmpFn
    β::Float32
end
struct QuadraticAmp <: AmpFn
    β::Float32
end
struct TableAmp <: AmpFn
    x::Vector{Float64}
    y::Vector{Float64}
    function TableAmp(x::AbstractVector, y::AbstractVector)
        length(x) == length(y) >= 2 || throw(ArgumentError("F table needs ≥ 2 (x, F) pairs of equal length"))
        issorted(x) || throw(ArgumentError("F table x = ln(r/r_ref) must be sorted"))
        return new(Float64.(collect(x)), Float64.(collect(y)))
    end
end

(::IsoAmp)(x) = one(x)
(f::LogLinearAmp)(x) = 1 + f.β * clamp(x, -f.L, f.L)
(f::LinearAmp)(x) = 1 + f.β * (exp(x) - 1)
(f::QuadraticAmp)(x) = 1 + (f.β / 2) * (exp(2x) - 1)
function _interp(l::TableAmp, x)
    x <= l.x[1] && return l.y[1]; x >= l.x[end] && return l.y[end]
    j = searchsortedlast(l.x, x); f = (x - l.x[j]) / (l.x[j + 1] - l.x[j])
    return (1 - f) * l.y[j] + f * l.y[j + 1]
end
(l::TableAmp)(x) = Float32(_interp(l, x))

function AmpFn(shape::Symbol; beta::Real = 0, L::Real = log(3), table = nothing)
    shape === :none && return IsoAmp()
    shape === :loglinear && return LogLinearAmp(beta, L)
    shape === :linear && return LinearAmp(beta)
    shape === :quadratic && return QuadraticAmp(beta)
    if shape === :table
        table === nothing && throw(ArgumentError("AmpFn(:table) needs `table = (x, F)`"))
        return TableAmp(table...)
    end
    throw(ArgumentError("unknown F shape :$shape; expected :none, :loglinear, :linear, :quadratic or :table"))
end

"""
    amp_beta(F::AmpFn) -> Float32

`β = dF/d ln ρ` at `ρ = 1` (for a table, the central finite difference at
`x = 0`).
"""
amp_beta(::IsoAmp) = 0f0
amp_beta(f::Union{LogLinearAmp, LinearAmp, QuadraticAmp}) = f.β
amp_beta(f::TableAmp) = Float32((_interp(f, 1e-6) - _interp(f, -1e-6)) / 2e-6)

_is_iso(f::AmpFn) = f isa IsoAmp || (!(f isa TableAmp) && amp_beta(f) == 0)

# ---------------------------------------------------------------- β converters (C2)

"""
    beta_from_nu(ν, muT) -> Float64

Non-isochrony `β` of a Stuart–Landau unit with shear `ν`:
`ż = μ(1 − ρ²)z + iω₀[1 − (νμ/ω₀)(ρ² − 1)]z`, `ρ = |z|/r₀`, so
`β = −2νμ/ω₀ = −ν·μT/π` (`muT = μ·T`). In the unscaled form
`ż = (μ + iω)z − |z|²z − iν(|z|² − r₀²)z` (where `r₀² = μ`) this is
`−2νr₀²/ω₀`. The matching `F` shape is `:quadratic`.
"""
beta_from_nu(ν::Real, muT::Real) = -Float64(ν) * Float64(muT) / π

"""
    beta_from_e4(β_e4; kappa = 10π, lambdaT = -0.2) -> Float64

Convert the E4 experiments' non-isochrony (an extra rotation
`κ·β_e4·ln ρ` per unit of `1/|λ|`) to the C2 `β`:
`β = β_e4·κ·|λT|/(2π)`. With E4's `κ = ω₀/|λ| = 10π` and `λT = −0.2` the two
coincide.
"""
beta_from_e4(β::Real; kappa::Real = 10π, lambdaT::Real = -0.2) = Float64(β) * kappa * abs(lambdaT) / (2π)

"""
    beta_from_iso(iso, omega; r_ref = 1) -> Float64

Convert the lock-in EP experiments' non-isochrony (a rotation rate
`iso·(|g| − r_ref)`, `|g|` the drive amplitude, `omega` the carrier in the
same time units) to the C2 `β` of the `:linear` shape:
`β = iso·r_ref/ω`.
"""
beta_from_iso(iso::Real, omega::Real; r_ref::Real = 1) = Float64(iso) * r_ref / omega

# ---------------------------------------------------------------- phase-noise converters (C1)

"""
    diffusion_from_linewidth(Δf_Hz, f0_Hz) -> Float64

Phase-diffusion constant `D·T` (dimensionless, `Var φ(t) = 2Dt`) of an
oscillator with Lorentzian full linewidth `Δf`: `D = πΔf`, `T = 1/f₀`.
"""
diffusion_from_linewidth(Δf::Real, f0::Real) = π * Float64(Δf) / f0

"""
    diffusion_from_phase_noise(L_dBcHz, offset_Hz, f0_Hz) -> Float64

`D·T` from white-FM phase noise `L(Δf)` in dBc/Hz at offset `Δf`:
`L = D / (2π²Δf²)`, so `D = 2π²Δf²·10^(L/10)`.
"""
diffusion_from_phase_noise(L::Real, offset::Real, f0::Real) = 2π^2 * Float64(offset)^2 * 10^(L / 10) / f0

"""
    diffusion_from_jitter(σ_rel) -> Float64

`D·T` from the cycle-to-cycle period jitter `σ_t/T`: one period accumulates
phase variance `2D·T = (2π·σ_t/T)²`.
"""
diffusion_from_jitter(σ::Real) = (2π * Float64(σ))^2 / 2

# ---------------------------------------------------------------- sub-specs

_check_sym(name, v, allowed) =
    v in allowed || throw(ArgumentError("$name must be one of $(join(map(repr, allowed), ", ")), got $(repr(v))"))
_check_nonneg(name, v) = v >= 0 || throw(ArgumentError("$name must be ≥ 0, got $v"))

"""
    UnitSpec(; class = :R, omega_rel = 1, lambdaT = nothing, muT = 0, r0 = 1, input_tauT = 0)

The phase unit's own dynamics, dimensionless (C1).
- `class`: `:R` driven resonator (`ż = (λ + iω)z + …`, decays to rest) or
  `:L` limit-cycle oscillator (Stuart–Landau, self-sustained).
- `omega_rel`: systematic frequency offset `ω_c/ω₀` shared by every unit.
- `lambdaT`: decay rate `λ·T` (class R, negative); `nothing` keeps each
  layer's own trained `λ`.
- `muT`, `r0`: class L growth rate `μ·T` and limit-cycle radius.
- `input_tauT`: input low-pass time constant in periods (0 = none).
"""
Base.@kwdef struct UnitSpec
    class::Symbol = :R
    omega_rel::Float32 = 1f0
    lambdaT::Union{Nothing, Float32} = nothing
    muT::Float32 = 0f0
    r0::Float32 = 1f0
    input_tauT::Float32 = 0f0
    function UnitSpec(class, omega_rel, lambdaT, muT, r0, input_tauT)
        _check_sym("UnitSpec class", class, (:R, :L))
        lambdaT === nothing || lambdaT < 0 || throw(ArgumentError("lambdaT must be negative, got $lambdaT"))
        _check_nonneg("muT", muT); _check_nonneg("input_tauT", input_tauT)
        r0 > 0 || throw(ArgumentError("r0 must be positive, got $r0"))
        class === :L && muT == 0 && throw(ArgumentError("a class L unit needs muT > 0"))
        return new(class, omega_rel, lambdaT, muT, r0, input_tauT)
    end
end

"""
    SpreadSpec(; sigma_omega = 0, sigma_lambda = 0, lambda_clamp = 0.05, sigma_mu = 0,
                 sigma_amp = 0, sigma_threshold = 0, threshold_to_f_gain = 0,
                 sigma_delay = 0, sigma_orbit = 0)

Static device-to-device spread, drawn once per unit (relative values;
`sigma_delay` in periods). `sigma_omega` is `σ_ω/ω₀` before any trim;
`sigma_lambda` multiplies `|λ|` by `max(1 + σ_λ ξ, lambda_clamp)`;
`threshold_to_f_gain` couples a threshold draw into frequency
(`d ln f₀ / d ln V_th`, relaxation oscillators).
"""
Base.@kwdef struct SpreadSpec
    sigma_omega::Float32 = 0f0
    sigma_lambda::Float32 = 0f0
    lambda_clamp::Float32 = 0.05f0
    sigma_mu::Float32 = 0f0
    sigma_amp::Float32 = 0f0
    sigma_threshold::Float32 = 0f0
    threshold_to_f_gain::Float32 = 0f0
    sigma_delay::Float32 = 0f0
    sigma_orbit::Float32 = 0f0
    function SpreadSpec(args...)
        for (n, v) in zip(fieldnames(SpreadSpec), args)
            n === :threshold_to_f_gain || _check_nonneg(String(n), v)
        end
        return new(args...)
    end
end

"""
    OrbitSpec(; S = FourierFn(1f0), G = FourierFn(1f0 + 0f0im), F = IsoAmp(),
                r_ref = nothing, amp_clip = Inf, clip_gammaT = 20, isf = nothing)

The shape of the unit's orbit (identical for every unit of a layer; the
per-unit spread is `SpreadSpec.sigma_orbit`).
- `S(θ)`: orbit speed, real [`PeriodicFn`](@ref), `⟨1/S⟩ = 1` (C4).
- `G(θ)`: kick response, complex [`PeriodicFn`](@ref).
- `F`: non-isochrony [`AmpFn`](@ref) (C2), evaluated at `ln(r/r_ref)`.
- `r_ref`: reference amplitude; `nothing` means calibrate it on data (the
  per-unit median `|z|` at cycle ends).
- `amp_clip`, `clip_gammaT`: soft amplitude clip `−γ·max(|z| − A, 0)·ẑ`
  with `γ·T = clip_gammaT`.
- `isf`: class L impulse-sensitivity function `Γ(φ)` (real
  [`PeriodicFn`](@ref)), from which a class L unit derives `G`.
"""
Base.@kwdef struct OrbitSpec
    S::PeriodicFn = FourierFn(1f0)
    G::PeriodicFn = FourierFn(1f0 + 0f0im)
    F::AmpFn = IsoAmp()
    r_ref::Union{Nothing, Float32} = nothing
    amp_clip::Float32 = Inf32
    clip_gammaT::Float32 = 20f0
    isf::Union{Nothing, PeriodicFn} = nothing
end

"""
    NoiseSpec(; diffusionT = 0, flicker_rel = 0, flicker_tauT = 0.5, flicker_n = 7,
                drift_rel = 0, common_rel = 0, clock_walk = 0, amp_rel = 0,
                current_sigma = 0)

Dynamic noise, redrawn every run from the noise seed (C6).
- `diffusionT`: white-FM phase diffusion `D·T` (`Var φ = 2Dt`); see
  [`diffusion_from_linewidth`](@ref) and its siblings.
- `flicker_rel`: RMS 1/f frequency wander relative to `ω₀`, as a sum of
  `flicker_n` Ornstein–Uhlenbeck processes with `τ = flicker_tauT·4^j`
  periods.
- `drift_rel`: linear frequency drift, `Δω/ω₀` per period.
- `common_rel`: common-mode (temperature, supply) frequency wander shared by
  every unit; `clock_walk`: common rhythm walk (rad).
- `amp_rel`: cycle-to-cycle amplitude noise; `current_sigma`: white input
  current noise.
"""
Base.@kwdef struct NoiseSpec
    diffusionT::Float32 = 0f0
    flicker_rel::Float32 = 0f0
    flicker_tauT::Float32 = 0.5f0
    flicker_n::Int = 7
    drift_rel::Float32 = 0f0
    common_rel::Float32 = 0f0
    clock_walk::Float32 = 0f0
    amp_rel::Float32 = 0f0
    current_sigma::Float32 = 0f0
    function NoiseSpec(diffusionT, flicker_rel, flicker_tauT, flicker_n, drift_rel, common_rel, clock_walk,
                       amp_rel, current_sigma)
        for (n, v) in (("diffusionT", diffusionT), ("flicker_rel", flicker_rel), ("common_rel", common_rel),
                       ("clock_walk", clock_walk), ("amp_rel", amp_rel), ("current_sigma", current_sigma))
            _check_nonneg(n, v)
        end
        flicker_tauT > 0 || throw(ArgumentError("flicker_tauT must be positive"))
        flicker_n >= 1 || throw(ArgumentError("flicker_n must be ≥ 1"))
        return new(diffusionT, flicker_rel, flicker_tauT, flicker_n, drift_rel, common_rel, clock_walk, amp_rel,
                   current_sigma)
    end
end

"""
    EmitSpec(; rule = :sampled, event_phase = 0, threshold = nothing, hill_n = Inf,
               warp_a = 0, warp_c = 0, snap_m = 0, snap_offset_sigma = 0, burst = 1,
               jitter = 0)

How a unit turns its state into spikes.
- `rule`: `:sampled` (the library's cycle-end sample) or `:event` (a spike
  each time the orbit crosses `event_phase`; class L).
- `threshold`, `hill_n`: Hill emission in Dirac units (C3); `threshold =
  nothing` keeps the library rule, `hill_n = Inf` gives unit charges.
- `warp_a`, `warp_c`: spike-time warp `u → u + (a/2π)·sin(2πu + c)` of the
  within-cycle time `u` (EM1/SD1 `P_emit`).
- `snap_m`, `snap_offset_sigma`: snap spike times to a `Z_m` grid of the
  global clock, with per-unit grid offsets (levels) of this spread.
- `burst`: spikes per emission; `jitter`: fresh per-spike timing jitter
  `σ_t/T`.
"""
Base.@kwdef struct EmitSpec
    rule::Symbol = :sampled
    event_phase::Float32 = 0f0
    threshold::Union{Nothing, Float32} = nothing
    hill_n::Float32 = Inf32
    warp_a::Float32 = 0f0
    warp_c::Float32 = 0f0
    snap_m::Int = 0
    snap_offset_sigma::Float32 = 0f0
    burst::Int = 1
    jitter::Float32 = 0f0
    function EmitSpec(rule, event_phase, threshold, hill_n, warp_a, warp_c, snap_m, snap_offset_sigma, burst, jitter)
        _check_sym("EmitSpec rule", rule, (:sampled, :event))
        threshold === nothing || _check_nonneg("threshold", threshold)
        hill_n > 0 || throw(ArgumentError("hill_n must be positive"))
        _check_nonneg("snap_m", snap_m); _check_nonneg("snap_offset_sigma", snap_offset_sigma)
        _check_nonneg("jitter", jitter)
        burst >= 1 || throw(ArgumentError("burst must be ≥ 1"))
        return new(rule, event_phase, threshold, hill_n, warp_a, warp_c, snap_m, snap_offset_sigma, burst, jitter)
    end
end

"""
    SynapseSpec(; weight_bits = 0, weight_per = :tensor, sigma_w = 0, sigma_w_dynamic = 0,
                  dead_frac = 0, stuck_frac = 0, weight_drift = 0, weight_drift_nu = 0,
                  gain_drift = 0, K_rel = 0, coupling_phase = 0, complex_coupling = false,
                  bias_through_G = true)

The synapses and the coupling into a unit.
- `weight_bits`: symmetric uniform quantisation of `W` (0 = off), scaled per
  `:tensor` or per `:row`.
- `sigma_w`: static multiplicative mismatch `W(1 + σξ)`;
  `sigma_w_dynamic`: the same, redrawn every step (read noise).
- `dead_frac`, `stuck_frac`: fractions of dead (zero) and stuck units.
- `weight_drift`: linear drift `W(1 + drift·t·ξ)` per period;
  `weight_drift_nu`: conductance relaxation `G ∝ t^−ν`; `gain_drift`:
  emission-gain drift per period.
- `K_rel`: class L coupling strength `K/ω₀`; `coupling_phase` (rad) rotates
  the coupling; `complex_coupling` multiplies it by `1 + iν`.
- `bias_through_G`: the bias current enters through `G(θ)` like the input.
"""
Base.@kwdef struct SynapseSpec
    weight_bits::Int = 0
    weight_per::Symbol = :tensor
    sigma_w::Float32 = 0f0
    sigma_w_dynamic::Float32 = 0f0
    dead_frac::Float32 = 0f0
    stuck_frac::Float32 = 0f0
    weight_drift::Float32 = 0f0
    weight_drift_nu::Float32 = 0f0
    gain_drift::Float32 = 0f0
    K_rel::Float32 = 0f0
    coupling_phase::Float32 = 0f0
    complex_coupling::Bool = false
    bias_through_G::Bool = true
    function SynapseSpec(weight_bits, weight_per, sigma_w, sigma_w_dynamic, dead_frac, stuck_frac, weight_drift,
                         weight_drift_nu, gain_drift, K_rel, coupling_phase, complex_coupling, bias_through_G)
        _check_nonneg("weight_bits", weight_bits)
        _check_sym("weight_per", weight_per, (:tensor, :row))
        for (n, v) in (("sigma_w", sigma_w), ("sigma_w_dynamic", sigma_w_dynamic), ("K_rel", K_rel),
                       ("weight_drift_nu", weight_drift_nu))
            _check_nonneg(n, v)
        end
        0 <= dead_frac <= 1 && 0 <= stuck_frac <= 1 && dead_frac + stuck_frac <= 1 ||
            throw(ArgumentError("dead_frac and stuck_frac must lie in [0, 1] and sum to ≤ 1"))
        return new(weight_bits, weight_per, sigma_w, sigma_w_dynamic, dead_frac, stuck_frac, weight_drift,
                   weight_drift_nu, gain_drift, K_rel, coupling_phase, complex_coupling, bias_through_G)
    end
end

"""
    WireSpec(; distortion = 0, delay = 0, sigma_delay = 0, jitter = 0)

What happens to a spike train between layers, in periods: a within-cycle
phase distortion `u → u + (a/2π)·sin 2πu` (`distortion = a`), a mean link
delay, a static per-link delay spread and fresh per-spike jitter (H1's
`pd`, `σd`, `σt`, applied in that order).
"""
Base.@kwdef struct WireSpec
    distortion::Float32 = 0f0
    delay::Float32 = 0f0
    sigma_delay::Float32 = 0f0
    jitter::Float32 = 0f0
    function WireSpec(distortion, delay, sigma_delay, jitter)
        _check_nonneg("delay", delay); _check_nonneg("sigma_delay", sigma_delay); _check_nonneg("jitter", jitter)
        return new(distortion, delay, sigma_delay, jitter)
    end
end

"""
    TrimSpec(; freq = :none, freq_bits = 0, freq_range_sigma = 3, delay = :none, delay_bits = 0,
               phase = :none, fll = :none, fll_gainT = 0, homeostat_eta = 0, pin_K_rel = 0,
               lock_m = 0, lock_K_rel = 0)

Calibration and regulation that work against the spread.
- `freq`: `:none`, `:quantised` (a `freq_bits` trim over `±freq_range_sigma·σ_ω`,
  the H1r rule) or `:exact`.
- `delay`: `:none`, `:oracle` (measured against the ideal twin) or
  `:quantised` (`delay_bits`); `phase`: `:none`, `:phase` (static rotation)
  or `:full` (complex factor).
- `fll`: frequency-locked loop `:none`, `:push` (mixer on the drive) or
  `:nat` (ideal natural-rate sensor), gain `g_L·T = fll_gainT`.
- `homeostat_eta`: threshold homeostat rate across calls.
- `pin_K_rel`: reference-pin injection strength `K/ω₀`.
- `lock_m`, `lock_K_rel`: `m`-fold locking torque towards a `Z_m` alphabet.
"""
Base.@kwdef struct TrimSpec
    freq::Symbol = :none
    freq_bits::Int = 0
    freq_range_sigma::Float32 = 3f0
    delay::Symbol = :none
    delay_bits::Int = 0
    phase::Symbol = :none
    fll::Symbol = :none
    fll_gainT::Float32 = 0f0
    homeostat_eta::Float32 = 0f0
    pin_K_rel::Float32 = 0f0
    lock_m::Int = 0
    lock_K_rel::Float32 = 0f0
    function TrimSpec(freq, freq_bits, freq_range_sigma, delay, delay_bits, phase, fll, fll_gainT, homeostat_eta,
                      pin_K_rel, lock_m, lock_K_rel)
        _check_sym("TrimSpec freq", freq, (:none, :quantised, :exact))
        _check_sym("TrimSpec delay", delay, (:none, :oracle, :quantised))
        _check_sym("TrimSpec phase", phase, (:none, :phase, :full))
        _check_sym("TrimSpec fll", fll, (:none, :push, :nat))
        freq === :quantised && freq_bits < 1 && throw(ArgumentError("a :quantised frequency trim needs freq_bits ≥ 1"))
        delay === :quantised && delay_bits < 1 && throw(ArgumentError("a :quantised delay trim needs delay_bits ≥ 1"))
        freq_range_sigma > 0 || throw(ArgumentError("freq_range_sigma must be positive"))
        for (n, v) in (("fll_gainT", fll_gainT), ("homeostat_eta", homeostat_eta), ("pin_K_rel", pin_K_rel),
                       ("lock_m", lock_m), ("lock_K_rel", lock_K_rel))
            _check_nonneg(n, v)
        end
        return new(freq, freq_bits, freq_range_sigma, delay, delay_bits, phase, fll, fll_gainT, homeostat_eta,
                   pin_K_rel, lock_m, lock_K_rel)
    end
end

"""
    ClockSpec(; decode = :nominal, error_rel = 0, jitter = 0, offset = 0, ref_units = 0)

The clock that decodes spike times into phases where something samples or
reads (C7). `decode`: `:nominal` (the library's clock, today's behaviour),
`:reader` (a reader clock with frequency error `error_rel = δ`, sampling
jitter `jitter` (rad) and a fixed `offset` (rad)), `:own` (each unit's own
clock, an oracle) or `:local` (a local reference of `ref_units` units in the
same cycle).
"""
Base.@kwdef struct ClockSpec
    decode::Symbol = :nominal
    error_rel::Float32 = 0f0
    jitter::Float32 = 0f0
    offset::Float32 = 0f0
    ref_units::Int = 0
    function ClockSpec(decode, error_rel, jitter, offset, ref_units)
        _check_sym("ClockSpec decode", decode, (:nominal, :reader, :own, :local))
        _check_nonneg("jitter", jitter); _check_nonneg("ref_units", ref_units)
        decode === :local && ref_units < 1 && throw(ArgumentError("a :local clock needs ref_units ≥ 1"))
        return new(decode, error_rel, jitter, offset, ref_units)
    end
end

"""
    ReaderSpec(; hebbian_sigma = 0, offset_in = 0, offset_prod = 0, adc_bits = 0, ac_tauT = 0,
                 readout_quant = 0, readout_jitter = 0)

The analog reader of lock-in EP and the phase readout: white noise on the
Hebbian accumulators, input and product offsets of the multiplier, ADC
bits (0 = ideal), AC-coupling time constant in periods (0 = DC), and the
phase readout quantiser step and jitter (turns).
"""
Base.@kwdef struct ReaderSpec
    hebbian_sigma::Float32 = 0f0
    offset_in::Float32 = 0f0
    offset_prod::Float32 = 0f0
    adc_bits::Int = 0
    ac_tauT::Float32 = 0f0
    readout_quant::Float32 = 0f0
    readout_jitter::Float32 = 0f0
    function ReaderSpec(hebbian_sigma, offset_in, offset_prod, adc_bits, ac_tauT, readout_quant, readout_jitter)
        for (n, v) in (("hebbian_sigma", hebbian_sigma), ("adc_bits", adc_bits), ("ac_tauT", ac_tauT),
                       ("readout_quant", readout_quant), ("readout_jitter", readout_jitter))
            _check_nonneg(n, v)
        end
        return new(hebbian_sigma, offset_in, offset_prod, adc_bits, ac_tauT, readout_quant, readout_jitter)
    end
end

"""
    CostSpec(; energy_per_event_J = NaN, energy_per_cycle_J = NaN, area_um2 = NaN,
               eta_dB = NaN, trim_area_um2 = NaN)

Bookkeeping only: physical costs carried with the spec for reporting
(`NaN` = not given). No simulation reads them.
"""
Base.@kwdef struct CostSpec
    energy_per_event_J::Float64 = NaN
    energy_per_cycle_J::Float64 = NaN
    area_um2::Float64 = NaN
    eta_dB::Float64 = NaN
    trim_area_um2::Float64 = NaN
end

# ---------------------------------------------------------------- the device spec

"""
    DeviceSpec(; name = "ideal", unit, spread, orbit, noise, emit, synapse, wire, trim,
                 clock, reader, cost, source = "", provenance = [], unknown = [], notes = [])

The complete dimensionless description of one device, the input of the
device layers. Every sub-spec ([`UnitSpec`](@ref), [`SpreadSpec`](@ref),
[`OrbitSpec`](@ref), [`NoiseSpec`](@ref), [`EmitSpec`](@ref),
[`SynapseSpec`](@ref), [`WireSpec`](@ref), [`TrimSpec`](@ref),
[`ClockSpec`](@ref), [`ReaderSpec`](@ref), [`CostSpec`](@ref)) defaults to
the ideal, so `DeviceSpec()` is the ideal device ([`IdealDevice`](@ref)).
Build one from a card with [`device_spec`](@ref), or by hand:

```julia
spec = DeviceSpec(name = "2 % spread", spread = SpreadSpec(sigma_omega = 0.02),
                  trim = TrimSpec(freq = :quantised, freq_bits = 4))
```

`source` names the card file; `provenance` lists each card field used with
its value type (`"spread.sigma_omega_rel" => :meas`); `unknown` lists the
card fields marked unknown that were replaced by the ideal; `notes` records
conversions worth knowing (table fits and their residuals, rescalings).
"""
Base.@kwdef struct DeviceSpec
    name::String = "ideal"
    unit::UnitSpec = UnitSpec()
    spread::SpreadSpec = SpreadSpec()
    orbit::OrbitSpec = OrbitSpec()
    noise::NoiseSpec = NoiseSpec()
    emit::EmitSpec = EmitSpec()
    synapse::SynapseSpec = SynapseSpec()
    wire::WireSpec = WireSpec()
    trim::TrimSpec = TrimSpec()
    clock::ClockSpec = ClockSpec()
    reader::ReaderSpec = ReaderSpec()
    cost::CostSpec = CostSpec()
    source::String = ""
    provenance::Vector{Pair{String, Symbol}} = Pair{String, Symbol}[]
    unknown::Vector{String} = String[]
    notes::Vector{String} = String[]
end

"""
    IdealDevice() -> DeviceSpec

The ideal device: every sub-spec at its default. A network wrapped with it
runs exactly the shared-ω layers (docs/devices.md §1).
"""
IdealDevice() = DeviceSpec(name = "ideal")

"""
    is_ideal(spec) -> Bool

Whether a [`DeviceSpec`](@ref), or one of its sub-specs, departs from the
shared-ω ideal in any way that changes a simulation. `CostSpec`, `name`,
`source`, `provenance` and notes are ignored.
"""
is_ideal(u::UnitSpec) = u.class === :R && u.omega_rel == 1 && u.lambdaT === nothing && u.input_tauT == 0
is_ideal(s::SpreadSpec) = s.sigma_omega == 0 && s.sigma_lambda == 0 && s.sigma_mu == 0 && s.sigma_amp == 0 &&
                          s.sigma_threshold == 0 && s.sigma_delay == 0 && s.sigma_orbit == 0
is_ideal(o::OrbitSpec) = _is_const_one(o.S) && _is_const_one(o.G) && _is_iso(o.F) && isinf(o.amp_clip) &&
                         o.isf === nothing
is_ideal(n::NoiseSpec) = n.diffusionT == 0 && n.flicker_rel == 0 && n.drift_rel == 0 && n.common_rel == 0 &&
                         n.clock_walk == 0 && n.amp_rel == 0 && n.current_sigma == 0
is_ideal(e::EmitSpec) = e.rule === :sampled && e.threshold === nothing && e.warp_a == 0 && e.snap_m == 0 &&
                        e.burst == 1 && e.jitter == 0
is_ideal(s::SynapseSpec) = s.weight_bits == 0 && s.sigma_w == 0 && s.sigma_w_dynamic == 0 && s.dead_frac == 0 &&
                           s.stuck_frac == 0 && s.weight_drift == 0 && s.weight_drift_nu == 0 && s.gain_drift == 0 &&
                           s.K_rel == 0 && s.coupling_phase == 0 && !s.complex_coupling
is_ideal(w::WireSpec) = w.distortion == 0 && w.delay == 0 && w.sigma_delay == 0 && w.jitter == 0
is_ideal(t::TrimSpec) = t.freq === :none && t.delay === :none && t.phase === :none && t.fll === :none &&
                        t.homeostat_eta == 0 && t.pin_K_rel == 0 && t.lock_m == 0
is_ideal(c::ClockSpec) = c.decode === :nominal
is_ideal(r::ReaderSpec) = r.hebbian_sigma == 0 && r.offset_in == 0 && r.offset_prod == 0 && r.adc_bits == 0 &&
                          r.ac_tauT == 0 && r.readout_quant == 0 && r.readout_jitter == 0
is_ideal(::CostSpec) = true
is_ideal(d::DeviceSpec) = is_ideal(d.unit) && is_ideal(d.spread) && is_ideal(d.orbit) && is_ideal(d.noise) &&
                          is_ideal(d.emit) && is_ideal(d.synapse) && is_ideal(d.wire) && is_ideal(d.trim) &&
                          is_ideal(d.clock) && is_ideal(d.reader)

const _SPEC_PARTS = (:unit, :spread, :orbit, :noise, :emit, :synapse, :wire, :trim, :clock, :reader)

function Base.show(io::IO, d::DeviceSpec)
    off = [String(p) for p in _SPEC_PARTS if !is_ideal(getfield(d, p))]
    print(io, "DeviceSpec(\"", d.name, "\", class ", d.unit.class, ", ",
          isempty(off) ? "ideal" : "non-ideal: " * join(off, ", "), ")")
end

# ---------------------------------------------------------------- card → spec

function _read_csv_table(path::AbstractString)
    lines = filter(l -> !isempty(strip(l)) && !startswith(strip(l), "#"), readlines(path))
    isempty(lines) && throw(ArgumentError("empty table: $path"))
    header = strip.(split(lines[1], ','))
    rows = [parse.(Float64, strip.(split(l, ','))) for l in lines[2:end]]
    all(r -> length(r) == length(header), rows) ||
        throw(ArgumentError("table $path: every row needs $(length(header)) columns"))
    return header, reduce(hcat, rows)'           # (rows, columns)
end

function _periodic_table(path, ncols_value; field)
    header, M = _read_csv_table(path)
    size(M, 2) == 1 + ncols_value ||
        throw(ArgumentError("$field: expected $(1 + ncols_value) columns (theta + values), got $(join(header, ", "))"))
    θ = M[:, 1]; n = length(θ)
    maximum(abs.(θ .- _theta_grid(n))) < 1e-4 ||
        throw(ArgumentError("$field: theta must be the uniform grid −π + 2π(j − 1)/n (n = $n)"))
    return ncols_value == 1 ? M[:, 2] : complex.(M[:, 2], M[:, 3])
end

function _orbit_fn(v, tables, nharm, field, notes)
    tables === :table && return TableFn(v)
    f, res = fourier_fit(v; nharm)
    push!(notes, "$field: Fourier fit, $(nharmonics(f)) harmonics, residual $(round(res, sigdigits = 3)) of max|v|")
    return f
end

"""
    device_spec(card::DeviceCard, spk_args::SpikingArgs; op = nothing, on_unknown = :ideal,
                set = Dict(), tables = :fourier, nharm = 8, normalize_speed = true,
                lambda = :card) -> DeviceSpec

Turn a [`DeviceCard`](@ref) (physical units) into a [`DeviceSpec`](@ref)
(dimensionless, C1) at operating point `op`. The card's own period
`1/f₀` sets the unit of time, so a spec is independent of the simulation's
`spk_args.t_period`; `spk_args` is used only to express a `raw` Hill
threshold in the library's Dirac units (C3).

- `on_unknown`: a field marked `unknown` is replaced by the ideal and listed
  in `spec.unknown` (`:ideal`), or raises (`:error`).
- `set`: overrides, `"section.field" => value` in the card's physical
  units, applied before conversion and recorded with type `:set`; use it to
  sweep a field the card does not know.
- `tables`: orbit tables (`S_table`, `G_table`, `isf_table`) as
  [`FourierFn`](@ref) with `nharm` harmonics (`:fourier`, the residual goes
  into `spec.notes`) or as exact [`TableFn`](@ref) (`:table`).
- `normalize_speed`: rescale a speed table so `⟨1/S⟩ = 1` (C4).
- `lambda`: a class R card's decay (`unit.lambda_per_s`, else `unit.Q`)
  sets `lambdaT` (`:card`), or each layer keeps its trained `λ` (`:layer`).

The card is not validated here; run [`validate_device_card`](@ref) first.
Errors are raised for values the spec cannot represent.
"""
function device_spec(card::DeviceCard, spk_args::SpikingArgs; op = nothing, on_unknown::Symbol = :ideal,
                     set::AbstractDict = Dict{String, Any}(), tables::Symbol = :fourier, nharm::Integer = 8,
                     normalize_speed::Bool = true, lambda::Symbol = :card)
    _check_sym("on_unknown", on_unknown, (:ideal, :error))
    _check_sym("lambda", lambda, (:card, :layer))
    _check_sym("tables", tables, (:fourier, :table))
    provenance = Pair{String, Symbol}[]; unknown = String[]; notes = String[]
    setd = Dict{String, Any}(string(k) => v for (k, v) in set)

    # Value of a field, or `nothing` when absent or unknown (recorded).
    function val(field)
        if haskey(setd, field)
            push!(provenance, field => :set)
            return setd[field]
        end
        v = card_value(card, field; op, default = missing, on_unknown = :value)
        v === missing && return nothing
        if v.kind === :unknown
            on_unknown === :error && throw(ArgumentError("device card \"$(card.name)\": $field is marked unknown"))
            push!(unknown, field)
            return nothing
        end
        push!(provenance, field => v.kind)
        return v.file === nothing ? v.value : v.file
    end
    num(field) = (x = val(field); x === nothing ? nothing :
                  x isa Real && !(x isa Bool) ? Float64(x) :
                  throw(ArgumentError("device card \"$(card.name)\": $field must be a number, got $(repr(x))")))
    str(field) = (x = val(field); x === nothing ? nothing : lowercase(string(x)))
    numor(field, d) = something(num(field), d)

    cls_v = val("identity.class")
    cls_v === nothing && throw(ArgumentError("device card \"$(card.name)\" does not state its class (R or L)"))
    cls = Symbol(uppercase(string(cls_v)))
    cls in (:R, :L) || throw(ArgumentError("device card \"$(card.name)\": class must be R or L, got $cls"))

    f0 = num("unit.f0_Hz")
    needs_f0(field) = f0 === nothing &&
        throw(ArgumentError("device card \"$(card.name)\": $field needs unit.f0_Hz to convert to periods"))
    perT(field) = (x = num(field); x === nothing ? nothing : (needs_f0(field); x / f0))   # rate → rate·T
    inT(field) = (x = num(field); x === nothing ? nothing : (needs_f0(field); x * f0))    # time → periods

    # unit: class R decay from λ, else from Q (λT = −π/Q, C1)
    lambdaT = nothing
    if cls === :R && lambda === :card
        lam = perT("unit.lambda_per_s")
        lambdaT = lam !== nothing ? lam : (Q = num("unit.Q"); Q === nothing ? nothing : -π / Q)
    elseif cls === :R
        push!(notes, "unit: lambda = :layer, each layer keeps its trained λ; the card's decay is not used")
    end
    muT = cls === :L ? perT("unit.mu_per_s") : nothing
    cls === :L && muT === nothing &&
        throw(ArgumentError("device card \"$(card.name)\": a class L card needs unit.mu_per_s"))
    unit = UnitSpec(class = cls, lambdaT = lambdaT, muT = something(muT, 0.0),
                    r0 = cls === :L ? numor("unit.r0", 1.0) : 1.0,
                    input_tauT = something(inT("unit.input_tau_s"), 0.0))

    # spread
    spread = SpreadSpec(sigma_omega = numor("spread.sigma_omega_rel", 0.0),
                        sigma_lambda = cls === :R ? numor("spread.sigma_lambda_rel", 0.0) : 0.0,
                        sigma_mu = cls === :L ? numor("spread.sigma_mu_rel", 0.0) : 0.0,
                        sigma_amp = numor("spread.sigma_amp_rel", 0.0),
                        sigma_threshold = numor("spread.sigma_threshold_rel", 0.0),
                        threshold_to_f_gain = numor("spread.threshold_to_f_gain", 0.0),
                        sigma_delay = something(inT("spread.sigma_delay_s"), 0.0),
                        sigma_orbit = numor("spread.sigma_orbit_rel", 0.0))

    # noise: one diffusion source, in order of directness
    lw = num("noise.linewidth_Hz"); pn = num("noise.phase_noise_dBcHz"); jit = num("noise.jitter_rel")
    diffusionT = if lw !== nothing
        needs_f0("noise.linewidth_Hz"); diffusion_from_linewidth(lw, f0)
    elseif pn !== nothing
        off = num("noise.offset_Hz")
        off === nothing && throw(ArgumentError("device card \"$(card.name)\": noise.phase_noise_dBcHz needs noise.offset_Hz"))
        needs_f0("noise.phase_noise_dBcHz"); diffusion_from_phase_noise(pn, off, f0)
    elseif jit !== nothing
        diffusion_from_jitter(jit)
    else
        0.0
    end
    count(!isnothing, (lw, pn, jit)) > 1 &&
        push!(notes, "noise: phase diffusion from " * (lw !== nothing ? "linewidth_Hz" : "phase_noise_dBcHz") *
                     "; the other noise.* diffusion fields were not used")
    noise = NoiseSpec(diffusionT = diffusionT,
                      flicker_rel = numor("noise.flicker_rel", 0.0),
                      drift_rel = something(perT("noise.drift_rel_per_s"), 0.0),
                      clock_walk = numor("noise.clock_walk_rad", 0.0),
                      amp_rel = numor("noise.amp_noise_rel", 0.0))

    # orbit
    shape = str("orbit.F_shape")
    shape = shape === nothing ? (cls === :L ? :quadratic : :loglinear) : Symbol(shape)
    β = num("orbit.beta")
    ν = cls === :L ? num("orbit.nu") : nothing
    if β === nothing && ν !== nothing
        β = beta_from_nu(ν, muT)
        push!(notes, "orbit.beta from orbit.nu: β = −ν·μT/π = $(round(β, sigdigits = 4))")
    end
    Ftab = val("orbit.F_table")
    F = if shape === :table || (Ftab !== nothing && β === nothing)
        Ftab === nothing && throw(ArgumentError("device card \"$(card.name)\": F_shape = table needs orbit.F_table"))
        h, M = _read_csv_table(Ftab)
        size(M, 2) == 2 || throw(ArgumentError("orbit.F_table: expected columns (x, F), got $(join(h, ", "))"))
        TableAmp(M[:, 1], M[:, 2])
    else
        β === nothing || β == 0 ? IsoAmp() : AmpFn(shape; beta = β)
    end
    Stab = val("orbit.S_table")
    S = if Stab === nothing
        FourierFn(1f0)
    else
        s = _orbit_fn(_periodic_table(Stab, 1; field = "orbit.S_table"), tables, nharm, "orbit.S_table", notes)
        m = mean_inverse_speed(s)
        if normalize_speed && abs(m - 1) > 1e-6
            push!(notes, "orbit.S_table: rescaled by ⟨1/S⟩ = $(round(m, sigdigits = 6)) so the period is kept (C4)")
            s = _scale(s, m)
        end
        s
    end
    Gtab = val("orbit.G_table")
    G = Gtab === nothing ? FourierFn(1f0 + 0f0im) :
        _orbit_fn(_periodic_table(Gtab, 2; field = "orbit.G_table"), tables, nharm, "orbit.G_table", notes)
    isftab = cls === :L ? val("orbit.isf_table") : nothing
    isf = isftab === nothing ? nothing :
        _orbit_fn(_periodic_table(isftab, 1; field = "orbit.isf_table"), tables, nharm, "orbit.isf_table", notes)
    orbit = OrbitSpec(S = S, G = G, F = F, r_ref = num("orbit.r_ref"), isf = isf)

    # emission
    rule = str("emission.rule")
    rule_sym = rule === nothing ? (cls === :L ? :event : :sampled) :
               rule in ("sampled", "hill") ? :sampled : rule == "event" ? :event :
               throw(ArgumentError("device card \"$(card.name)\": emission.rule \"$rule\" is not supported " *
                                   "(sampled, hill, event)"))
    θ = num("emission.threshold")
    if θ !== nothing
        units = something(str("emission.threshold_units"), "dirac")
        units in ("raw", "dirac") || throw(ArgumentError("emission.threshold_units must be raw or dirac"))
        if units == "raw"
            θ = θ / _emit_gain(spk_args)
            push!(notes, "emission.threshold: raw → Dirac units, ÷ 2·t_window·spk_scale = $(_emit_gain(spk_args)) (C3)")
        end
    end
    emit = EmitSpec(rule = rule_sym, event_phase = numor("emission.event_phase_rad", 0.0), threshold = θ,
                    hill_n = numor("emission.hill_n", Inf), burst = Int(numor("emission.burst", 1.0)),
                    snap_m = Int(numor("clock.zm_m", 0.0)))

    # synapses and coupling
    K = num("coupling.K_rel")
    if K === nothing
        KHz = num("coupling.K_Hz")
        KHz === nothing || (needs_f0("coupling.K_Hz"); K = KHz / f0)
    end
    per = str("coupling.weight_per")
    synapse = SynapseSpec(weight_bits = Int(numor("coupling.weight_bits", 0.0)),
                          weight_per = per === nothing ? :tensor : Symbol(per),
                          sigma_w = numor("coupling.sigma_weight_rel", 0.0),
                          dead_frac = numor("coupling.dead_frac", 0.0),
                          weight_drift_nu = numor("coupling.weight_drift_nu", 0.0),
                          gain_drift = something(perT("coupling.gain_drift_per_s"), 0.0),
                          K_rel = cls === :L ? something(K, 0.0) : 0.0,
                          coupling_phase = numor("coupling.coupling_phase_rad", 0.0))
    wire = WireSpec(delay = something(inT("coupling.link_delay_s"), 0.0),
                    jitter = something(inT("coupling.sigma_link_delay_s"), 0.0))

    # trims
    fb = num("trim.freq_bits")
    pt = str("trim.phase_trim"); fll = str("trim.fll"); db = num("trim.delay_bits")
    trim = TrimSpec(freq = fb === nothing || fb == 0 ? :none : isinf(fb) ? :exact : :quantised,
                    freq_bits = fb === nothing || isinf(fb) ? 0 : Int(fb),
                    freq_range_sigma = numor("trim.freq_range_sigma", 3.0),
                    delay = db === nothing || db == 0 ? :none : :quantised,
                    delay_bits = db === nothing ? 0 : Int(db),
                    phase = pt === nothing ? :none : Symbol(pt),
                    fll = fll === nothing ? :none : Symbol(fll),
                    fll_gainT = numor("trim.fll_gain_rel", 0.0),
                    homeostat_eta = numor("trim.homeostat_eta", 0.0),
                    pin_K_rel = cls === :L ? numor("clock.shil_K_rel", 0.0) : 0.0,
                    lock_m = Int(numor("trim.lock_m", 0.0)),
                    lock_K_rel = numor("trim.lock_K_rel", 0.0))

    # clock and reader
    δ = num("clock.reader_clock_error_rel"); rj = num("clock.reader_jitter_rad")
    clock = δ === nothing && rj === nothing ? ClockSpec() :
            ClockSpec(decode = :reader, error_rel = something(δ, 0.0), jitter = something(rj, 0.0))
    reader = ReaderSpec(hebbian_sigma = numor("reader.hebbian_sigma", 0.0),
                        offset_in = numor("reader.offset_in", 0.0),
                        offset_prod = numor("reader.offset_prod", 0.0),
                        adc_bits = Int(numor("reader.adc_bits", 0.0)),
                        ac_tauT = something(inT("reader.ac_tau_s"), 0.0),
                        readout_quant = numor("reader.readout_quant_turn", 0.0))

    cost = CostSpec(energy_per_event_J = numor("cost.energy_per_event_J", NaN),
                    energy_per_cycle_J = numor("cost.energy_per_cycle_J", NaN),
                    area_um2 = numor("cost.area_um2", NaN), eta_dB = numor("cost.eta_dB", NaN),
                    trim_area_um2 = numor("cost.trim_area_um2", NaN))

    unused = setdiff(keys(setd), first.(provenance))
    isempty(unused) || throw(ArgumentError("device_spec: `set` fields not used by the spec: " *
                                           join(sort(collect(unused)), ", ")))
    name = op === nothing ? card.name : "$(card.name) @ $op"
    return DeviceSpec(; name, unit, spread, orbit, noise, emit, synapse, wire, trim, clock, reader, cost,
                      source = card.path, provenance = unique(provenance), unknown = unique(unknown), notes)
end

device_spec(path::AbstractString, spk_args::SpikingArgs; kw...) = device_spec(load_device_card(path), spk_args; kw...)
