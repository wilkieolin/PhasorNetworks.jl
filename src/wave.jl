# ================================================================
# PhasorWaveSheet — recurrent resonate-and-fire sheet (traveling waves)
# ================================================================
#
# A 2-D lattice of complex resonate-and-fire units coupled by a
# translation-invariant, delayed difference-of-Gaussians (Mexican-hat)
# kernel. This is the Tier-1 *discrete phase-SSM* realization of the
# biophysical spec in docs/rf_wave_network_plan.html; see the companion
# docs/rf_wave_network_implementation.md for the full derivation.
#
# Per-step update (fork of the AttractorPhasorSSM Buffer loop, with the
# attractor pull replaced by spatial coupling + conduction delays):
#
#     a[t] = ρ_a·a[t-1] + δ_a·|z[t-1]|                    (optional adaptation)
#     z[t] = A·z[t-1] + g·ifft2(Ŵ ⊙ fft2(z[t-1]))        per-channel decay + DoG coupling
#            + drive[t] − i·a[t]                          external drive; adaptation → voltage
#     z[t] = normalize_to_unit_circle(z[t])              (if saturating) phase-only self-limit
#
# Three insights make this fit the codebase (see companion doc):
#   1. Conduction delay τ becomes a complex phase factor e^{-iωτ} on the
#      shared carrier ω — no delay-ODE solver, stays linear & differentiable.
#   2. Translation-invariant coupling diagonalizes under the spatial FFT,
#      so per-step coupling is a circular convolution (ifft2 ∘ ·Ŵ· ∘ fft2),
#      O(HW·log HW), cuFFT-native and Zygote-differentiable. The per-mode
#      step multiplier M(q)=A+g·Ŵ(q) yields dispersion and criticality
#      (spectral radius) in closed form — see `dispersion`.
#   3. A phase-only wave (project to the unit circle each step) is
#      self-limiting for free — linear SSMs cannot be.
#
# The coupling kernel is parameterized by a few interpretable trainable
# scalars, so the plan's ablations are one-liners: B_inh = 0 removes
# lateral inhibition; δ_a = 0 removes adaptation; scaling g detunes
# criticality.
#
# See also: AttractorPhasorSSM (the recurrent template), causal_conv_fft
# (the same FFT-convolution idea over time), normalize_to_unit_circle.

"""
    PhasorWaveSheet(H, W; kwargs...)

A recurrent resonate-and-fire **sheet** of `H × W` complex units coupled by
a delayed difference-of-Gaussians kernel — a trainable, GPU-native,
phase-SSM realization of self-sustaining sparse traveling waves.

The sheet maps a flattened phase field `(H*W, L, B)` to its state
trajectory `(H*W, L, B)` (Lux forward, for training), and additionally
exposes [`wave_simulate`](@ref) (pure-forward autonomous evolution from an
initial state) and [`dispersion`](@ref) (closed-form per-mode growth &
speed) for analysis and demos.

# Keyword arguments

- `saturating::Bool = true` — project state onto the unit circle each step
  (phase-only self-limiting). Set `false` for the *linear* medium, whose
  dispersion matches [`dispersion`](@ref) exactly.
- `use_adaptation::Bool = false` — enable the slow negative-feedback state
  `a` (spike-triggered adaptation surrogate) that destabilizes standing
  bumps into traveling waves.
- `init_log_neg_lambda::Real = log(0.15)` — `log(-λ)`; subthreshold damping.
- `init_log_g::Real = log(1.0)` — `log` of the recurrent gain `g` (the
  criticality knob; spectral radius scales with `g`). Retune to criticality
  per sheet via [`dispersion`](@ref); see `demos/wave_dispersion.jl`.
- `init_A_exc::Real = 1.0`, `init_log_sigma_exc::Real = log(1.5)` —
  excitatory (center) amplitude and spatial width.
- `init_B_inh::Real = 0.25`, `init_log_sigma_inh::Real = log(3.0)` —
  inhibitory (surround) amplitude and width (`σ_I > σ_E` ⇒ Mexican hat).
  Defaults are near **spatial balance** `A_exc·σ_E² ≈ B_inh·σ_I²` (∫W ≈ 0,
  per the plan), which is what selects a nonzero wavelength (a ring) rather
  than a uniform front.
- `init_log_speed::Real = log(40.0)` — conduction speed `c` in pixels per
  period; delay `τ(r) = r / c`, phase factor `e^{-iω·r/c}`. Moderate delay
  lets a selected-wavelength pattern *propagate*; very small `c` washes out
  spatial selection into a uniform temporal oscillation.
- `init_log_rho_a::Real = log(0.9)`, `init_log_delta_a::Real = log(0.2)` —
  adaptation retention (`ρ_a`, kept in (0,1) via sigmoid) and increment
  (`δ_a`); used only when `use_adaptation`.
- `spk_args::SpikingArgs = SpikingArgs()` — supplies the shared carrier
  `ω = period_to_angfreq(t_period)` and step `T = t_period`.

# Parameter / state layout

| name | shape | location |
|------|-------|----------|
| `log_neg_lambda`, `log_g` | `(1,)` Float32 | params |
| `A_exc`, `log_sigma_exc`, `B_inh`, `log_sigma_inh`, `log_speed` | `(1,)` Float32 | params |
| `log_rho_a`, `log_delta_a` | `(1,)` Float32 | params (if `use_adaptation`) |
| `rgrid` | `(H, W)` Float32 | state (constant wrapped-distance grid) |

`ω` and `T` derive from `spk_args.t_period`.
"""
struct PhasorWaveSheet <: Lux.AbstractLuxLayer
    grid_h::Int
    grid_w::Int
    saturating::Bool
    use_adaptation::Bool
    init_log_neg_lambda::Float32
    init_log_g::Float32
    init_A_exc::Float32
    init_log_sigma_exc::Float32
    init_B_inh::Float32
    init_log_sigma_inh::Float32
    init_log_speed::Float32
    init_log_rho_a::Float32
    init_log_delta_a::Float32
    spk_args::SpikingArgs
end

function PhasorWaveSheet(H::Integer, W::Integer;
                         saturating::Bool = true,
                         use_adaptation::Bool = false,
                         init_log_neg_lambda::Real = log(0.15),
                         init_log_g::Real = log(1.0),
                         init_A_exc::Real = 1.0,
                         init_log_sigma_exc::Real = log(1.5),
                         init_B_inh::Real = 0.25,
                         init_log_sigma_inh::Real = log(3.0),
                         init_log_speed::Real = log(40.0),
                         init_log_rho_a::Real = log(0.9),
                         init_log_delta_a::Real = log(0.2),
                         spk_args::SpikingArgs = SpikingArgs())
    return PhasorWaveSheet(Int(H), Int(W), saturating, use_adaptation,
                           Float32(init_log_neg_lambda), Float32(init_log_g),
                           Float32(init_A_exc), Float32(init_log_sigma_exc),
                           Float32(init_B_inh), Float32(init_log_sigma_inh),
                           Float32(init_log_speed),
                           Float32(init_log_rho_a), Float32(init_log_delta_a),
                           spk_args)
end

function Base.show(io::IO, l::PhasorWaveSheet)
    print(io, "PhasorWaveSheet($(l.grid_h)×$(l.grid_w); ")
    print(io, "saturating=$(l.saturating), use_adaptation=$(l.use_adaptation), ")
    print(io, "t_period=$(l.spk_args.t_period))")
end

# ---- Lux interface -----------------------------------------------------

"""
    _wrapped_distance_grid(H, W) -> Matrix{Float32}

Toroidal (periodic) distance from the origin `(1,1)` for an `H×W` sheet.
Entry `[i,j]` is the Euclidean distance to `(1,1)` using the shorter of the
two wrap-around offsets in each axis — the right convention for a circular
(FFT) convolution kernel centered at the origin.
"""
function _wrapped_distance_grid(H::Int, W::Int)
    rg = Matrix{Float32}(undef, H, W)
    for j in 1:W, i in 1:H
        di = i - 1;  di = di > H ÷ 2 ? di - H : di
        dj = j - 1;  dj = dj > W ÷ 2 ? dj - W : dj
        rg[i, j] = sqrt(Float32(di)^2 + Float32(dj)^2)
    end
    return rg
end

function Lux.initialparameters(rng::AbstractRNG, l::PhasorWaveSheet)
    base = (log_neg_lambda = Float32[l.init_log_neg_lambda],
            log_g          = Float32[l.init_log_g],
            A_exc          = Float32[l.init_A_exc],
            log_sigma_exc  = Float32[l.init_log_sigma_exc],
            B_inh          = Float32[l.init_B_inh],
            log_sigma_inh  = Float32[l.init_log_sigma_inh],
            log_speed      = Float32[l.init_log_speed])
    if l.use_adaptation
        base = merge(base, (log_rho_a   = Float32[l.init_log_rho_a],
                            log_delta_a = Float32[l.init_log_delta_a]))
    end
    return base
end

Lux.initialstates(::AbstractRNG, l::PhasorWaveSheet) =
    (rgrid = _wrapped_distance_grid(l.grid_h, l.grid_w),)

# ---- Coupling kernel ---------------------------------------------------

"""
    _build_coupling(l, ps, rgrid, ω) -> (A_step, g, W_hat)

Resolve the trainable scalars into the objects the per-step recurrence
needs, all on the parameter device and fully differentiable:

- `A_step :: (1,)` complex — per-channel full-step decay `exp(k·T)`,
  `k = -exp(log_neg_lambda) + iω`.
- `g :: (1,)` real — recurrent gain.
- `W_hat :: (H,W)` complex — the spatial FFT of the delayed difference-of-
  Gaussians coupling kernel, so a circular convolution is `ifft2(W_hat ⊙
  fft2(z))`. The DoG magnitude is
  `A_exc·G(r;σ_E) − B_inh·G(r;σ_I)`, the delay factor `e^{-iω·r/c}`, and
  the self term (`r = 0`) is zeroed (self-dynamics live in `A_step`).
"""
function _build_coupling(l::PhasorWaveSheet, ps, rgrid, ω)
    T = Float32(l.spk_args.t_period)
    λ      = -exp.(ps.log_neg_lambda)                 # (1,)
    k      = ComplexF32.(λ .+ 1im .* ω)               # (1,)
    A_step = exp.(k .* T)                             # (1,)
    g      = exp.(ps.log_g)                           # (1,)

    σe = exp.(ps.log_sigma_exc)                       # (1,)
    σi = exp.(ps.log_sigma_inh)                       # (1,)
    c  = exp.(ps.log_speed)                           # (1,)
    r2 = rgrid .^ 2                                    # (H,W)

    # Difference-of-Gaussians magnitude (Mexican hat when σ_I > σ_E).
    mag = ps.A_exc .* exp.(-r2 ./ (2f0 .* σe .^ 2)) .-
          ps.B_inh .* exp.(-r2 ./ (2f0 .* σi .^ 2))  # (H,W)

    # Conduction delay as a phase factor on the shared carrier ω.
    delay_phase = exp.(-1im .* (ω .* rgrid ./ c))     # (H,W) complex

    self_mask = rgrid .> 0f0                          # constant; zero self-coupling
    W_full = ComplexF32.(mag .* self_mask) .* delay_phase   # (H,W)
    W_hat  = fft(W_full)                              # (H,W)
    return A_step, g, W_hat
end

# ---- Core recurrence ---------------------------------------------------

# One evolution step in the spatial-Fourier-coupled complex plane.
# `z, drive_t, a` are (H,W,B); scalars A_step, g, ρ_a, δ_a are (1,).
# Returns (z_next, a_next).
function _wave_step(z, drive_t, a, A_step, g, W_hat, ρ_a, δ_a,
                    saturating::Bool, use_adaptation::Bool)
    H, W, B = size(z)
    coupled = ifft(reshape(W_hat, H, W, 1) .* fft(z, (1, 2)), (1, 2))    # (H,W,B)
    z_lin = reshape(A_step, 1, 1, 1) .* z .+
            reshape(g, 1, 1, 1) .* coupled .+ drive_t
    a_next = a
    if use_adaptation
        a_next = reshape(ρ_a, 1, 1, 1) .* a .+ reshape(δ_a, 1, 1, 1) .* abs.(z)
        z_lin = z_lin .- 1im .* a_next
    end
    z_next = saturating ? normalize_to_unit_circle(ComplexF32.(z_lin)) : ComplexF32.(z_lin)
    return z_next, a_next
end

# Roll the recurrence for L steps from initial state z0 (H,W,B), returning
# the complex trajectory (H,W,L,B). `drive` is nothing (autonomous) or a
# (H,W,L,B) complex drive. Uses Zygote.Buffer (the AttractorPhasorSSM
# pattern) so it is AD-safe.
function _wave_rollout(l::PhasorWaveSheet, ps, rgrid, z0, drive, L::Int)
    ω = period_to_angfreq(l.spk_args.t_period)
    A_step, g, W_hat = _build_coupling(l, ps, rgrid, ω)
    ρ_a = l.use_adaptation ? _sigmoid.(ps.log_rho_a) : nothing
    δ_a = l.use_adaptation ? exp.(ps.log_delta_a)    : nothing

    H, W, B = size(z0)
    a0 = ignore_derivatives() do
        a = similar(rgrid, Float32, H, W, B); a .= 0f0; return a
    end

    Y = Buffer(similar(z0, ComplexF32, H, W, L, B))
    z = ComplexF32.(z0)
    a = a0
    for t in 1:L
        drive_t = drive === nothing ?
            ignore_derivatives() do
                d = similar(z, ComplexF32, H, W, B); d .= 0f0; return d
            end : drive[:, :, t, :]
        z, a = _wave_step(z, drive_t, a, A_step, g, W_hat, ρ_a, δ_a,
                          l.saturating, l.use_adaptation)
        Y[:, :, t, :] = z
    end
    return copy(Y)                                    # (H,W,L,B) complex
end

# ---- Lux forward: Phase 3D --------------------------------------------
#
# x :: (H*W, L, B) Phase. Each timestep's phase field is injected as a
# unit-magnitude complex drive; the sheet state trajectory is returned as
# (H*W, L, B) Phase. Drop-in for a Lux chain (feed to SSMReadout/Codebook).

function (l::PhasorWaveSheet)(x::AbstractArray{<:Phase, 3},
                              ps::LuxParams, st::NamedTuple)
    H, W = l.grid_h, l.grid_w
    N, L, B = size(x)
    @assert N == H * W "input channels $(N) ≠ grid $(H)×$(W) = $(H*W)"

    drive = reshape(angle_to_complex(x), H, W, L, B)  # (H,W,L,B) unit phasors
    z0 = ignore_derivatives() do
        z = similar(st.rgrid, ComplexF32, H, W, B); z .= 0f0; return z
    end
    Y = _wave_rollout(l, ps, st.rgrid, z0, drive, L)   # (H,W,L,B)
    return complex_to_angle(reshape(Y, H * W, L, B)), st
end

# ---- Pure-forward analysis interface ----------------------------------

"""
    wave_simulate(l::PhasorWaveSheet, ps, st; z0, L, drive=nothing) -> Array{ComplexF32}

Evolve the sheet for `L` steps from initial complex state `z0` and return
the full complex trajectory. Autonomous when `drive === nothing`; otherwise
`drive` is a per-step complex drive.

`z0` may be `(H,W)` or `(H,W,B)`; the return is `(H,W,L)` or `(H,W,L,B)`
correspondingly. `drive`, if given, must be `(H,W,L)` / `(H,W,L,B)` to
match. This is the interface for the dispersion / ablation demos (seed a
localized pulse and watch it travel).
"""
function wave_simulate(l::PhasorWaveSheet, ps, st;
                       z0::AbstractArray, L::Integer, drive=nothing)
    H, W = l.grid_h, l.grid_w
    had_batch = ndims(z0) == 3
    z0_3 = had_batch ? ComplexF32.(z0) : reshape(ComplexF32.(z0), H, W, 1)
    @assert size(z0_3, 1) == H && size(z0_3, 2) == W "z0 must be $(H)×$(W)[×B]"
    drive_4 = drive === nothing ? nothing :
        (ndims(drive) == 4 ? ComplexF32.(drive) :
         reshape(ComplexF32.(drive), H, W, Int(L), 1))
    Y = _wave_rollout(l, ps, st.rgrid, z0_3, drive_4, Int(L))    # (H,W,L,B)
    return had_batch ? Y : dropdims(Y; dims=4)
end

"""
    dispersion(l::PhasorWaveSheet, ps, st) -> NamedTuple

Closed-form linear dispersion of the sheet (§2.2 of the companion doc).
Because a translation-invariant coupling diagonalizes under the spatial
FFT, each wavevector `q` evolves independently with per-step multiplier

    M(q) = A + g · Ŵ(q),     A = exp(k·T)

Returns `(; M, W_hat, spectral_radius, k_eff, growth_rate)`:

- `M :: (H,W)` complex — per-mode step multiplier.
- `W_hat :: (H,W)` complex — FFT of the coupling kernel.
- `spectral_radius :: Float32` — `max_q |M(q)|`. `< 1` extinguishes,
  `≈ 1` self-sustaining (critical), `> 1` saturating. This is the
  phase-SSM analogue of the plan's branching ratio σ.
- `k_eff :: (H,W)` complex — continuous effective eigenvalue `log(M)/T`.
- `growth_rate :: (H,W)` Float32 — `real(k_eff) = log|M|/T` per mode.

Exact for the linear medium (`saturating=false`, `use_adaptation=false`);
a leading-order guide otherwise.
"""
function dispersion(l::PhasorWaveSheet, ps, st)
    ω = period_to_angfreq(l.spk_args.t_period)
    T = Float32(l.spk_args.t_period)
    A_step, g, W_hat = _build_coupling(l, ps, st.rgrid, ω)
    M = reshape(A_step, 1, 1) .+ reshape(g, 1, 1) .* W_hat          # (H,W)
    spectral_radius = maximum(abs.(M))
    k_eff = log.(M) ./ T
    return (; M, W_hat, spectral_radius, k_eff, growth_rate = real.(k_eff))
end
