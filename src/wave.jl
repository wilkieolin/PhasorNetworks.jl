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

- `coupling::Symbol = :dog` — how the recurrent coupling kernel is
  parameterized. `:dog` is the parametric delayed difference-of-Gaussians
  (≈9 interpretable scalars). `:stencil` is a **free, learnable** local
  complex kernel of radius `stencil_radius` (`(2R+1)²` complex entries),
  giving the sheet real trainable capacity while staying translation-
  invariant and FFT-diagonal; it is seeded from the DoG so it starts in the
  same regime, then trains freely.
- `stencil_radius::Integer = 2` — radius `R` of the learnable stencil (used
  only when `coupling = :stencil`); the kernel is `(2R+1)×(2R+1)`.
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
| `A_exc`, `log_sigma_exc`, `B_inh`, `log_sigma_inh`, `log_speed` | `(1,)` Float32 | params (`coupling = :dog`) |
| `stencil_re`, `stencil_im` | `(2R+1, 2R+1)` Float32 | params (`coupling = :stencil`) |
| `log_rho_a`, `log_delta_a` | `(1,)` Float32 | params (if `use_adaptation`) |
| `rgrid` | `(H, W)` Float32 | state (constant wrapped-distance grid) |
| `place` | `(H*W, (2R+1)²)` Float32 | state (`coupling = :stencil`; constant scatter matrix) |

`ω` and `T` derive from `spk_args.t_period`.
"""
struct PhasorWaveSheet <: Lux.AbstractLuxLayer
    grid_h::Int
    grid_w::Int
    coupling::Symbol          # :dog (parametric difference-of-Gaussians) | :stencil (free learnable)
    stencil_radius::Int       # R: the learnable stencil is (2R+1)×(2R+1); used when coupling=:stencil
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
                         coupling::Symbol = :dog,
                         stencil_radius::Integer = 2,
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
    coupling in (:dog, :stencil) ||
        throw(ArgumentError("coupling must be :dog or :stencil, got :$coupling"))
    coupling === :stencil && (2 * stencil_radius + 1 > min(H, W)) &&
        throw(ArgumentError("stencil_radius=$stencil_radius too large for $(H)×$(W) sheet"))
    return PhasorWaveSheet(Int(H), Int(W), coupling, Int(stencil_radius),
                           saturating, use_adaptation,
                           Float32(init_log_neg_lambda), Float32(init_log_g),
                           Float32(init_A_exc), Float32(init_log_sigma_exc),
                           Float32(init_B_inh), Float32(init_log_sigma_inh),
                           Float32(init_log_speed),
                           Float32(init_log_rho_a), Float32(init_log_delta_a),
                           spk_args)
end

function Base.show(io::IO, l::PhasorWaveSheet)
    print(io, "PhasorWaveSheet($(l.grid_h)×$(l.grid_w); coupling=:$(l.coupling)")
    l.coupling === :stencil && print(io, "(R=$(l.stencil_radius))")
    print(io, ", saturating=$(l.saturating), use_adaptation=$(l.use_adaptation), ")
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

# Initial values for a free (2R+1)² complex stencil, seeded from the parametric
# DoG (+ conduction-delay phase) so :stencil starts in the same sensible,
# wavelength-selecting regime the :dog default lives in, then trains freely.
# Center (self, offset (0,0)) is zeroed — self-dynamics live in A_step.
function _dog_stencil_init(l::PhasorWaveSheet)
    R = l.stencil_radius
    ω = period_to_angfreq(l.spk_args.t_period)
    σe = exp(l.init_log_sigma_exc); σi = exp(l.init_log_sigma_inh)
    c  = exp(l.init_log_speed)
    n = 2R + 1
    sre = zeros(Float32, n, n); sim = zeros(Float32, n, n)
    for j in 1:n, i in 1:n
        di = i - 1 - R; dj = j - 1 - R
        (di == 0 && dj == 0) && continue
        r = sqrt(Float32(di)^2 + Float32(dj)^2)
        mag = l.init_A_exc * exp(-r^2 / (2f0 * σe^2)) - l.init_B_inh * exp(-r^2 / (2f0 * σi^2))
        val = ComplexF32(mag) * exp(-1im * ω * r / c)
        sre[i, j] = real(val); sim[i, j] = imag(val)
    end
    return sre, sim
end

function Lux.initialparameters(rng::AbstractRNG, l::PhasorWaveSheet)
    base = (log_neg_lambda = Float32[l.init_log_neg_lambda],
            log_g          = Float32[l.init_log_g])
    if l.coupling === :stencil
        sre, sim = _dog_stencil_init(l)
        base = merge(base, (stencil_re = sre, stencil_im = sim))
    else  # :dog — parametric difference-of-Gaussians
        base = merge(base, (A_exc         = Float32[l.init_A_exc],
                            log_sigma_exc = Float32[l.init_log_sigma_exc],
                            B_inh         = Float32[l.init_B_inh],
                            log_sigma_inh = Float32[l.init_log_sigma_inh],
                            log_speed     = Float32[l.init_log_speed]))
    end
    if l.use_adaptation
        base = merge(base, (log_rho_a   = Float32[l.init_log_rho_a],
                            log_delta_a = Float32[l.init_log_delta_a]))
    end
    return base
end

# Constant (H*W, K) placement matrix mapping each of the K=(2R+1)² stencil
# entries to its (wrap-around) position on the sheet, so the full coupling
# kernel is `reshape(place * stencil_vec, H, W)` — a differentiable, linear
# scatter with no mutation. The self column (offset (0,0)) is left all-zero.
function _stencil_placement(H::Int, W::Int, R::Int)
    n = 2R + 1; K = n * n
    place = zeros(Float32, H * W, K)
    o = 0
    for j in 1:n, i in 1:n           # column-major, matches reshape(stencil, K)
        o += 1
        di = i - 1 - R; dj = j - 1 - R
        (di == 0 && dj == 0) && continue          # zero self-coupling
        gi = mod(di, H) + 1; gj = mod(dj, W) + 1
        place[gi + (gj - 1) * H, o] = 1f0
    end
    return place
end

function Lux.initialstates(::AbstractRNG, l::PhasorWaveSheet)
    st = (rgrid = _wrapped_distance_grid(l.grid_h, l.grid_w),)
    if l.coupling === :stencil
        st = merge(st, (place = _stencil_placement(l.grid_h, l.grid_w, l.stencil_radius),))
    end
    return st
end

# ---- Coupling kernel ---------------------------------------------------

"""
    _build_coupling(l, ps, st, ω) -> (A_step, g, W_hat)

Resolve the trainable parameters into the objects the per-step recurrence
needs, all on the parameter device and fully differentiable:

- `A_step :: (1,)` complex — per-channel full-step decay `exp(k·T)`,
  `k = -exp(log_neg_lambda) + iω`.
- `g :: (1,)` real — recurrent gain.
- `W_hat :: (H,W)` complex — the spatial FFT of the coupling kernel, so a
  circular convolution is `ifft2(W_hat ⊙ fft2(z))`.

Two coupling parameterizations (`l.coupling`):
- `:dog` — the delayed difference-of-Gaussians `A_exc·G(r;σ_E) −
  B_inh·G(r;σ_I)` × `e^{-iω·r/c}` (≈9 interpretable scalars); self term
  (`r=0`) zeroed.
- `:stencil` — a free, learnable local complex kernel of radius `R`
  (`(2R+1)²` complex entries). Built as `reshape(place · stencil_vec, H, W)`
  — a differentiable linear scatter of the small stencil onto the sheet
  (the `place` matrix in state); self column zeroed. Gives the sheet real
  trainable capacity while staying translation-invariant and FFT-diagonal.
"""
function _build_coupling(l::PhasorWaveSheet, ps, st, ω)
    T = Float32(l.spk_args.t_period)
    λ      = -exp.(ps.log_neg_lambda)                 # (1,)
    k      = ComplexF32.(λ .+ 1im .* ω)               # (1,)
    A_step = exp.(k .* T)                             # (1,)
    g      = exp.(ps.log_g)                           # (1,)

    if l.coupling === :stencil
        H, W = l.grid_h, l.grid_w
        sv = reshape(ComplexF32.(ps.stencil_re) .+ 1im .* ComplexF32.(ps.stencil_im), :)  # (K,)
        W_full = reshape(st.place * sv, H, W)         # (H,W) differentiable linear scatter
        return A_step, g, fft(W_full)
    else  # :dog
        rgrid = st.rgrid
        σe = exp.(ps.log_sigma_exc)                   # (1,)
        σi = exp.(ps.log_sigma_inh)                   # (1,)
        c  = exp.(ps.log_speed)                       # (1,)
        r2 = rgrid .^ 2                                # (H,W)
        # Difference-of-Gaussians magnitude (Mexican hat when σ_I > σ_E).
        mag = ps.A_exc .* exp.(-r2 ./ (2f0 .* σe .^ 2)) .-
              ps.B_inh .* exp.(-r2 ./ (2f0 .* σi .^ 2))
        # Conduction delay as a phase factor on the shared carrier ω.
        delay_phase = exp.(-1im .* (ω .* rgrid ./ c))
        self_mask = rgrid .> 0f0                       # constant; zero self-coupling
        W_full = ComplexF32.(mag .* self_mask) .* delay_phase
        return A_step, g, fft(W_full)
    end
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
function _wave_rollout(l::PhasorWaveSheet, ps, st, z0, drive, L::Int)
    ω = period_to_angfreq(l.spk_args.t_period)
    A_step, g, W_hat = _build_coupling(l, ps, st, ω)
    ρ_a = l.use_adaptation ? _sigmoid.(ps.log_rho_a) : nothing
    δ_a = l.use_adaptation ? exp.(ps.log_delta_a)    : nothing

    H, W, B = size(z0)
    a0 = ignore_derivatives() do
        a = similar(st.rgrid, Float32, H, W, B); a .= 0f0; return a
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
    Y = _wave_rollout(l, ps, st, z0, drive, L)         # (H,W,L,B)
    return complex_to_angle(reshape(Y, H * W, L, B)), st
end

# ---- Pure-forward analysis interface ----------------------------------

"""
    wave_simulate(l::PhasorWaveSheet, ps, st; z0, L, drive=nothing, mode=:discrete) -> Array{ComplexF32}

Evolve the sheet for `L` steps from initial complex state `z0` and return
the full complex trajectory. Autonomous when `drive === nothing`; otherwise
`drive` is a per-step complex drive.

`z0` may be `(H,W)` or `(H,W,B)`; the return is `(H,W,L)` or `(H,W,L,B)`
correspondingly. `drive`, if given, must be `(H,W,L)` / `(H,W,L,B)` to
match. This is the interface for the dispersion / ablation demos (seed a
localized pulse and watch it travel).

`mode`:
- `:discrete` (default) — the Tier-1 operator-split recurrence (fast, and
  the mode the discrete Lux forward uses).
- `:ode` — the Tier-2 continuous ODE `dz/dt = k·z + g·(coupling)` integrated
  by `spk_args.solver` and sampled at each period. Autonomous only (`drive`
  must be `nothing`); the ground-truth continuous dynamics the discrete
  recurrence approximates. See [`dispersion`](@ref)`(...; mode=:continuous)`.
"""
function wave_simulate(l::PhasorWaveSheet, ps, st;
                       z0::AbstractArray, L::Integer, drive=nothing,
                       mode::Symbol = :discrete)
    H, W = l.grid_h, l.grid_w
    had_batch = ndims(z0) == 3
    z0_3 = had_batch ? ComplexF32.(z0) : reshape(ComplexF32.(z0), H, W, 1)
    @assert size(z0_3, 1) == H && size(z0_3, 2) == W "z0 must be $(H)×$(W)[×B]"
    if mode === :ode
        drive === nothing || throw(ArgumentError("wave_simulate mode=:ode is autonomous; drive must be nothing"))
        Y = _wave_rollout_ode(l, ps, st, z0_3, Int(L))           # (H,W,L,B)
    elseif mode === :discrete
        drive_4 = drive === nothing ? nothing :
            (ndims(drive) == 4 ? ComplexF32.(drive) :
             reshape(ComplexF32.(drive), H, W, Int(L), 1))
        Y = _wave_rollout(l, ps, st, z0_3, drive_4, Int(L))      # (H,W,L,B)
    else
        throw(ArgumentError("wave_simulate mode must be :discrete or :ode, got :$mode"))
    end
    return had_batch ? Y : dropdims(Y; dims=4)
end

# Tier-2 continuous rollout: integrate dz/dt = k·z + g·(FFT-coupling) with the
# layer's ODE solver (Tsit5 + BacksolveAdjoint by default) and sample at each
# period. Closes over the coupling built from `ps` — used for pure-forward
# simulation/validation (the trainable AD path is the CurrentCall dispatch).
function _wave_rollout_ode(l::PhasorWaveSheet, ps, st, z0, L::Int)
    ω = period_to_angfreq(l.spk_args.t_period)
    T = Float32(l.spk_args.t_period)
    _, g, W_hat = _build_coupling(l, ps, st, ω)
    λ = -exp.(ps.log_neg_lambda)
    k = ComplexF32.(λ .+ 1im .* ω)
    H, W, B = size(z0)
    kr  = reshape(k, 1, 1, 1)
    gr  = reshape(g, 1, 1, 1)
    Whr = reshape(W_hat, H, W, 1)

    dzdt(u, p, t) = kr .* u .+ gr .* ifft(Whr .* fft(u, (1, 2)), (1, 2))
    tspan = (0.0f0, Float32(L) * T)
    sol = oscillator_bank(ComplexF32.(z0), dzdt; tspan = tspan, spk_args = l.spk_args)

    samples = [ComplexF32.(sol(Float32(j) * T)) for j in 1:L]     # each (H,W,B)
    return cat([reshape(s, H, W, 1, B) for s in samples]...; dims = 3)  # (H,W,L,B)
end

"""
    dispersion(l::PhasorWaveSheet, ps, st; mode = :discrete) -> NamedTuple

Closed-form linear dispersion of the sheet (§2.2 of the companion doc).
Because a translation-invariant coupling diagonalizes under the spatial
FFT, each wavevector `q` evolves independently with a per-step multiplier
`M(q)`. Two conventions, selected by `mode`:

- `:discrete` (default) — the Tier-1 operator-split recurrence's per-step
  multiplier `M(q) = A + g·Ŵ(q)`, `A = exp(k·T)`. Matches
  [`wave_simulate`](@ref)`(...; mode = :discrete)` exactly.
- `:continuous` — the Tier-2 ODE's exact linear propagator over one period,
  `M(q) = exp(k_eff(q)·T)` with the true operator eigenvalue
  `k_eff(q) = k + g·Ŵ(q)`. Matches [`wave_simulate`](@ref)`(...; mode = :ode)`.

The two agree to first order in `g·T` (operator-splitting error), so they
converge as the coupling weakens or the step shrinks.

Returns `(; M, W_hat, spectral_radius, k_eff, growth_rate)`:

- `M :: (H,W)` complex — per-mode step multiplier.
- `W_hat :: (H,W)` complex — FFT of the coupling kernel.
- `spectral_radius :: Float32` — `max_q |M(q)|`. `< 1` extinguishes,
  `≈ 1` self-sustaining (critical), `> 1` saturating. This is the
  phase-SSM analogue of the plan's branching ratio σ.
- `k_eff :: (H,W)` complex — continuous effective eigenvalue.
- `growth_rate :: (H,W)` Float32 — `real(k_eff)` per mode.

Exact for the linear medium (`saturating=false`, `use_adaptation=false`);
a leading-order guide otherwise.
"""
function dispersion(l::PhasorWaveSheet, ps, st; mode::Symbol = :discrete)
    ω = period_to_angfreq(l.spk_args.t_period)
    T = Float32(l.spk_args.t_period)
    A_step, g, W_hat = _build_coupling(l, ps, st, ω)
    if mode === :continuous
        λ = -exp.(ps.log_neg_lambda)
        k = ComplexF32.(λ .+ 1im .* ω)
        k_eff = reshape(k, 1, 1) .+ reshape(g, 1, 1) .* W_hat      # (H,W) exact operator eigenvalue
        M = exp.(k_eff .* T)
        return (; M, W_hat, spectral_radius = maximum(abs.(M)), k_eff,
                  growth_rate = real.(k_eff))
    elseif mode === :discrete
        M = reshape(A_step, 1, 1) .+ reshape(g, 1, 1) .* W_hat     # (H,W)
        k_eff = log.(M) ./ T
        return (; M, W_hat, spectral_radius = maximum(abs.(M)), k_eff,
                  growth_rate = real.(k_eff))
    else
        throw(ArgumentError("dispersion mode must be :discrete or :continuous, got :$mode"))
    end
end

# ---- Tier-2 continuous dispatch: CurrentCall --------------------------
#
# The ODE mode of the same defining equation, mirroring PhasorDense /
# AttractorPhasorSSM's CurrentCall path: build the dzdt closure with the
# recurrent FFT-coupling term and hand it to DifferentialEquations.jl via
# the shared oscillator_bank machinery (Tsit5 + BacksolveAdjoint/ZygoteVJP).
#
#     dz/dt = k·z + g·ifft2(Ŵ ⊙ fft2(z)) + I(t)
#
# The coupling kernel Ŵ is rebuilt from `p` inside dzdt so gradients flow to
# the coupling parameters through the ODE adjoint. Input current `I(t)` is
# reshaped from the CurrentCall's `(H*W, …)` current to the `(H,W,B)` sheet.
# Output is sampled at each period and returned as `(H*W, L, B)` Phase,
# matching the discrete Lux forward's interface.

function (l::PhasorWaveSheet)(x::CurrentCall, ps::LuxParams, st::NamedTuple)
    spk_args = x.spk_args
    tspan    = x.t_span
    H, W     = l.grid_h, l.grid_w
    ω_val    = period_to_angfreq(l.spk_args.t_period)
    T        = Float32(spk_args.t_period)

    sample_I = x.current.current_fn(Float32(tspan[1]))
    B = ndims(sample_I) >= 2 ? size(sample_I, 2) : 1
    @assert size(sample_I, 1) == H * W "CurrentCall channels $(size(sample_I,1)) ≠ grid $(H*W)"

    u0 = ignore_derivatives() do
        u = similar(sample_I, ComplexF32, H, W, B); u .= zero(ComplexF32); return u
    end

    function dzdt(u, p, t)
        _, g, W_hat = _build_coupling(l, p, st, ω_val)            # rebuilt for AD
        λ = -exp.(p.log_neg_lambda)
        k = ComplexF32.(λ .+ 1im .* ω_val)
        coupled = ifft(reshape(W_hat, H, W, 1) .* fft(u, (1, 2)), (1, 2))
        drive   = reshape(ComplexF32.(x.current.current_fn(t)), H, W, B)
        return reshape(k, 1, 1, 1) .* u .+ reshape(g, 1, 1, 1) .* coupled .+ drive
    end

    # Sample at each period via `saveat` (NOT sol(t) interpolation, which is
    # disabled under the adjoint) so the CurrentCall path is differentiable
    # end-to-end with BacksolveAdjoint. offset t=0 is dropped via save_start=false.
    L = round(Int, (tspan[2] - tspan[1]) / spk_args.t_period)
    save_args = ignore_derivatives() do          # solver bookkeeping — not differentiable
        merge(spk_args.solver_args,
              Dict{Symbol,Any}(:saveat => Float32.(collect(T:T:(L * T))),
                               :save_start => false))
    end
    prob = ODEProblem(dzdt, u0, tspan, ps)
    sol  = solve(prob, spk_args.solver, p = ps; save_args...)

    Z = cat([reshape(ComplexF32.(u), H * W, 1, B) for u in sol.u]...; dims = 2)  # (H*W, L, B)
    return complex_to_angle(Z), st
end

function (l::PhasorWaveSheet)(x::SpikingCall, ps::LuxParams, st::NamedTuple)
    return l(CurrentCall(x), ps, st)
end
