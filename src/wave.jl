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
- `transmit::Symbol = :spike` (**default**) — what a neuron sends to its
  neighbours. `:spike` transmits a **unit-magnitude** event `z/|z|` — a
  fixed-size spike whose phase carries the info. The coupling drive is then
  hard-bounded (`|Σ W·s| ≤ Σ|W|`), so the leaky integrator is BIBO-stable with
  **no state snap** — and the state magnitude `|z|` survives to encode local
  phase coherence / interference intensity. Sub-threshold neurons emit ≈0 (a
  natural firing threshold via the ε-safe normalize). This is the principled,
  physically-faithful self-limiting mechanism and the library default.
  `:potential` transmits the full complex state `z` (linear diffusive coupling;
  can run away, so it needs the legacy state-level `saturating` snap). Use
  `:potential` for the *linear* medium whose dispersion matches
  [`dispersion`](@ref) exactly.
- `init_log_theta::Union{Real,Nothing} = nothing` — `log θ`, the **emission
  threshold** (units of `|z|`); the spike is `z/√(|z|²+θ²)`, i.e. the `ε` of
  [`normalize_to_unit_circle`](@ref) with `ε = θ²`. **The default is derived**:
  `θ = init_theta_frac · g·max_q|Ŵ(q)|`, see [`emission_threshold`](@ref). `θ`
  continuously interpolates between the two transmission modes: for `|z| ≫ θ`
  the emit is `z/|z|` (hard spike), for `|z| ≪ θ` it is `z/θ` — *linear*, i.e.
  `:potential` coupling at effective gain `g/θ`. The legacy hardcoded value was
  `√1e-8 = 1e-4`, five orders of magnitude below the derived reference, which
  put the sheet in a regime where every site above `10⁻⁴` emitted a full spike
  (an ignition cascade, not a wave). Only used when `transmit = :spike`.
- `init_theta_frac::Real = 1.4` — multiplier on the derived reference. The
  reference `g·max|Ŵ|` is the right *scale* but sits just **below** the
  flood boundary (measured at `1.16 × ` reference for `:dog` at `λ = 0.15`), so
  using it unmultiplied puts the sheet in the saturating basin: it floods,
  homogenizes (`std|z| → 10⁻⁴`), and stops being a medium. `1.4` is above that
  boundary and is also where the `:global` homeostat settles on its own
  (measured `θ/θ_ref = 1.54–1.56` across `g ∈ [0.3, 3]`).
- `homeostasis::Symbol = :none` — activity regulation of `θ`. `:none` holds it
  fixed. `:global` adapts a scalar `θ_g` toward a target emission fraction;
  this tracks `g` automatically, which matters because the usable `θ` band is
  only ≈1.3× wide and moves as `θ* ∝ g`, so a constant `θ` falls out of regime
  as soon as `g` trains. `:local` adds a **slow, zero-sum per-site** term
  `θ_l` on top of the global one — refractoriness, which is what converts the
  standing localized blob that `:global` alone produces into a *traveling*
  packet. A local-only mode is deliberately not offered: on its own the
  per-site term can only hit its target by silencing whoever just fired, which
  gives rate-regulated flicker (measured step-to-step overlap of the active set
  is exactly 0 at every rate and target). See §3-quater of the derivation.

  !!! warning "Validated at the default damping only"
      The homeostat regulates the firing rate to target at **any** `λ` and `g`
      — but at `λ ≥ 0.5` it does so with a *synchronously bursting* sheet
      rather than a structured one. Everything below was measured at the
      default `λ = 0.15`, across `g ∈ [0.3, 3]` and `N ∈ [48, 96]`. Hitting the
      target rate is necessary, not sufficient; check `std|z|` too.
- `init_log_eta_g::Real = log(0.15)`, `init_log_eta_l::Real = log(0.02)` —
  homeostatic rates (global, local). `η_l` has a narrow usable window
  `≈[0.01, 0.02]`, over which net drift roughly triples while step-to-step
  overlap of the active set stays ≈0.7; by `η_l = 0.05` overlap has fallen to
  0.36 and the packet has decohered back into flicker.
- `init_logit_target::Real = log(0.02/0.98)` — target firing fraction, logit-
  parameterized (so it stays in `(0,1)` under training). **2%, not 5%**: at
  `target = 0.10, η_g = 0.15` the controller drives `θ` below the flood boundary
  and the sheet collapses to a uniform state. `0.01`–`0.05` are all stable at
  either `η_g`.
- `init_log_theta_beta::Real = log(0.05)` — width of the sigmoid used as the
  *backward* pass of the fire indicator; the forward pass is the hard
  `|z| > θ`, via a straight-through estimator (the same idiom as
  [`moe_gate`](@ref)). Physically this is threshold jitter / trial-to-trial
  firing variability; numerically it sets the gradient scale reaching the
  homeostat parameters. The hard forward value is load-bearing — with a plain
  sigmoid the homeostat can hit its target with a *uniform subthreshold* sheet
  instead of a sparse active one (measured: target met exactly, zero sites
  firing). See the `_fire_indicator` comment.
- `saturating::Bool = false` (**default**) — the *legacy* self-limiting snap:
  project state onto the unit circle each step (phase-only). Off by default
  because spike transmission already self-limits **and** preserves magnitude,
  whereas the snap discards it. Only meaningful with `transmit = :potential`.
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
- `init_log_speed::Union{Real,Nothing} = nothing` — conduction speed `c` in
  sites per period; delay `τ(r) = r / c`, phase factor `e^{-iω·r/c}`. **The
  default is derived, not fixed**: `c = 2σ_I/T` from
  [`matched_conduction_speed`](@ref), which sets the delay phase across the
  inhibitory surround to `φ = ω σ_I/c = π`. This is the regime in which the
  band-selected mode also carries group velocity, so an impulse gives an
  *expanding ring*. Much larger `c` (weak delay) forces `v_g(q*) → 0` and an
  impulse gives **standing** concentric rings instead — the previous default of
  `40` sat there (`φ ≈ 0.15π`, transport ratio 0.03). Much smaller `c` pushes
  the selected wavelength toward the grid scale. Check the resulting regime with
  [`wave_transport`](@ref). Deriving the default keeps the sheet matched when
  `σ_I` or `t_period` change.
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
| `log_theta` | `(1,)` Float32 | params (if `transmit = :spike`) |
| `log_eta_g`, `logit_target`, `log_theta_beta` | `(1,)` Float32 | params (if `homeostasis ≠ :none`) |
| `log_eta_l` | `(1,)` Float32 | params (if `homeostasis = :local`) |
| `rgrid` | `(H, W)` Float32 | state (constant wrapped-distance grid) |
| `place` | `(H*W, (2R+1)²)` Float32 | state (`coupling = :stencil`; constant scatter matrix) |

`θ_g :: (1,)` and `θ_l :: (H,W)` are *rollout-local* homeostatic variables: they
are initialized fresh at the start of every rollout (like the adaptation state
`a`) rather than persisting in `st`, so a forward pass is a pure function of its
inputs and AD stays clean.

`ω` and `T` derive from `spk_args.t_period`.
"""
struct PhasorWaveSheet <: Lux.AbstractLuxLayer
    grid_h::Int
    grid_w::Int
    coupling::Symbol          # :dog (parametric DoG) | :stencil (free learnable) | :aniso (DoG + directed advection)
    stencil_radius::Int       # R: the learnable stencil is (2R+1)×(2R+1); used when coupling=:stencil
    transmit::Symbol          # :potential (send full z) | :spike (send unit-magnitude z/|z|)
    saturating::Bool
    use_adaptation::Bool
    homeostasis::Symbol       # :none | :global (scalar θ_g) | :local (θ_g × per-site θ_l)
    init_log_neg_lambda::Float32
    init_log_g::Float32
    init_A_exc::Float32
    init_log_sigma_exc::Float32
    init_B_inh::Float32
    init_log_sigma_inh::Float32
    init_log_speed::Float32
    init_log_rho_a::Float32
    init_log_delta_a::Float32
    init_beta_h::Float32       # advection drift along axis-1 (rows); used when coupling=:aniso
    init_beta_w::Float32       # advection drift along axis-2 (cols); used when coupling=:aniso
    init_shift_h::Float32      # directed shift (sites/period) along axis-1; used when coupling=:shift
    init_shift_w::Float32      # directed shift along axis-2; used when coupling=:shift
    init_log_theta::Union{Float32,Nothing}   # nothing ⇒ DERIVED θ = frac·g·max|Ŵ| in initialparameters
    init_theta_frac::Float32   # multiplier on the derived reference (see emission_threshold)
    init_log_eta_g::Float32    # global homeostatic rate
    init_log_eta_l::Float32    # per-site homeostatic rate
    init_logit_target::Float32 # target mean soft-fire fraction (logit-parameterized)
    init_log_theta_beta::Float32  # sigmoid width in the fire indicator's STE backward pass
    spk_args::SpikingArgs
end

function PhasorWaveSheet(H::Integer, W::Integer;
                         coupling::Symbol = :dog,
                         stencil_radius::Integer = 2,
                         transmit::Symbol = :spike,
                         saturating::Bool = false,
                         use_adaptation::Bool = false,
                         homeostasis::Symbol = :none,
                         init_log_neg_lambda::Real = log(0.15),
                         init_log_g::Real = log(1.0),
                         init_A_exc::Real = 1.0,
                         init_log_sigma_exc::Real = log(1.5),
                         init_B_inh::Real = 0.25,
                         init_log_sigma_inh::Real = log(3.0),
                         init_log_speed::Union{Real,Nothing} = nothing,
                         init_log_rho_a::Real = log(0.9),
                         init_log_delta_a::Real = log(0.2),
                         init_beta_h::Real = 0.5,
                         init_beta_w::Real = 0.0,
                         init_shift_h::Real = 1.0,
                         init_shift_w::Real = 0.0,
                         init_log_theta::Union{Real,Nothing} = nothing,
                         init_theta_frac::Real = 1.4,
                         init_log_eta_g::Real = log(0.15),
                         init_log_eta_l::Real = log(0.02),
                         init_logit_target::Real = log(0.02 / 0.98),
                         init_log_theta_beta::Real = log(0.05),
                         spk_args::SpikingArgs = SpikingArgs())
    coupling in (:dog, :stencil, :aniso, :shift) ||
        throw(ArgumentError("coupling must be :dog, :stencil, :aniso or :shift, got :$coupling"))
    transmit in (:potential, :spike) ||
        throw(ArgumentError("transmit must be :potential or :spike, got :$transmit"))
    homeostasis in (:none, :global, :local) ||
        throw(ArgumentError("homeostasis must be :none, :global or :local, got :$homeostasis"))
    homeostasis !== :none && transmit !== :spike &&
        throw(ArgumentError("homeostasis=:$homeostasis regulates the emission threshold, " *
              "which only exists under transmit=:spike (got transmit=:$transmit)"))
    coupling === :stencil && (2 * stencil_radius + 1 > min(H, W)) &&
        throw(ArgumentError("stencil_radius=$stencil_radius too large for $(H)×$(W) sheet"))
    # Default conduction speed is DERIVED, not a constant: c = 2σ_I/T puts the
    # delay phase across the surround at φ = ω σ_I/c = π, the traveling-wave
    # regime. Deriving it keeps the sheet matched when σ_I or t_period change —
    # a fixed default silently drifts into the standing-ring regime instead.
    init_log_speed = init_log_speed === nothing ?
        log(matched_conduction_speed(exp(init_log_sigma_inh), spk_args.t_period)) :
        init_log_speed
    return PhasorWaveSheet(Int(H), Int(W), coupling, Int(stencil_radius), transmit,
                           saturating, use_adaptation, homeostasis,
                           Float32(init_log_neg_lambda), Float32(init_log_g),
                           Float32(init_A_exc), Float32(init_log_sigma_exc),
                           Float32(init_B_inh), Float32(init_log_sigma_inh),
                           Float32(init_log_speed),
                           Float32(init_log_rho_a), Float32(init_log_delta_a),
                           Float32(init_beta_h), Float32(init_beta_w),
                           Float32(init_shift_h), Float32(init_shift_w),
                           init_log_theta === nothing ? nothing : Float32(init_log_theta),
                           Float32(init_theta_frac),
                           Float32(init_log_eta_g), Float32(init_log_eta_l),
                           Float32(init_logit_target), Float32(init_log_theta_beta),
                           spk_args)
end

function Base.show(io::IO, l::PhasorWaveSheet)
    print(io, "PhasorWaveSheet($(l.grid_h)×$(l.grid_w); coupling=:$(l.coupling)")
    l.coupling === :stencil && print(io, "(R=$(l.stencil_radius))")
    print(io, ", transmit=:$(l.transmit), saturating=$(l.saturating), use_adaptation=$(l.use_adaptation), ")
    l.transmit === :spike &&
        print(io, "theta=$(l.init_log_theta === nothing ? "derived" : exp(l.init_log_theta)), ")
    l.homeostasis !== :none && print(io, "homeostasis=:$(l.homeostasis), ")
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
    elseif l.coupling === :shift    # pure directed shift (conveyor): trainable shift vector
        base = merge(base, (shift_h = Float32[l.init_shift_h],
                            shift_w = Float32[l.init_shift_w]))
    else  # :dog / :aniso — parametric difference-of-Gaussians
        base = merge(base, (A_exc         = Float32[l.init_A_exc],
                            log_sigma_exc = Float32[l.init_log_sigma_exc],
                            B_inh         = Float32[l.init_B_inh],
                            log_sigma_inh = Float32[l.init_log_sigma_inh],
                            log_speed     = Float32[l.init_log_speed]))
        if l.coupling === :aniso     # directed advection drift (breaks reflection symmetry)
            base = merge(base, (beta_h = Float32[l.init_beta_h],
                                beta_w = Float32[l.init_beta_w]))
        end
    end
    if l.use_adaptation
        base = merge(base, (log_rho_a   = Float32[l.init_log_rho_a],
                            log_delta_a = Float32[l.init_log_delta_a]))
    end
    if l.transmit === :spike
        # Emission threshold. Derived by default (θ = g·max|Ŵ|) rather than a
        # constant, because θ* ∝ g exactly and the usable band is only ≈1.3×
        # wide — a fixed value silently leaves the regime once g moves.
        base = merge(base, (log_theta = Float32[log(_configured_theta(l))],))
    end
    if l.homeostasis !== :none
        base = merge(base, (log_eta_g      = Float32[l.init_log_eta_g],
                            logit_target   = Float32[l.init_logit_target],
                            log_theta_beta = Float32[l.init_log_theta_beta]))
        l.homeostasis === :local &&
            (base = merge(base, (log_eta_l = Float32[l.init_log_eta_l],)))
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
    elseif l.coupling === :aniso
        # Precomputed spatial-frequency sines for the advection term
        # Ŵ_adv(q) = −i(β_h·sin q_h + β_w·sin q_w). q along each FFT axis.
        H, W = l.grid_h, l.grid_w
        sin_qh = reshape(Float32.(sin.(2f0 .* Float32(pi) .* (0:H-1) ./ H)), H, 1)
        sin_qw = reshape(Float32.(sin.(2f0 .* Float32(pi) .* (0:W-1) ./ W)), 1, W)
        st = merge(st, (sin_qh = sin_qh, sin_qw = sin_qw))
    elseif l.coupling === :shift
        # Full FFT frequency grids for the shift ramp Ŵ_shift(q) = e^{−i(s_h q_h + s_w q_w)}.
        H, W = l.grid_h, l.grid_w
        qh = reshape(Float32.(2f0 .* Float32(pi) .* (0:H-1) ./ H), H, 1)
        qw = reshape(Float32.(2f0 .* Float32(pi) .* (0:W-1) ./ W), 1, W)
        st = merge(st, (qh = qh, qw = qw))
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

Four coupling parameterizations (`l.coupling`):
- `:shift` — a pure directed shift `Ŵ_shift(q) = e^{−i(s_h q_h + s_w q_w)}` (circular
  translation by the trainable vector `(s_h,s_w)`). Unit gain, linear phase ⇒ constant
  group velocity `s` and zero dispersion; with a strong leak (`A≈0`) the sheet is a
  ballistic conveyor at any depth. Two scalars `shift_h, shift_w`.
- `:dog` — the delayed difference-of-Gaussians `A_exc·G(r;σ_E) −
  B_inh·G(r;σ_I)` × `e^{-iω·r/c}` (≈9 interpretable scalars); self term
  (`r=0`) zeroed. Reflection-symmetric ⇒ waves spread symmetrically (no net
  transport).
- `:aniso` — `:dog` plus a directed **advection** term added in Fourier,
  `Ŵ_adv(q) = −i(β_h·sin q_h + β_w·sin q_w)` (a real antisymmetric stencil).
  Breaks reflection symmetry and gives a net group velocity `v = g·β` — a
  drifting packet — while staying FFT-diagonal (parallel-scan trainable).
  Two extra scalars `β_h, β_w`.
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
    elseif l.coupling === :shift
        # Pure directed shift (conveyor): Ŵ_shift(q) = e^{−i(s_h q_h + s_w q_w)} — a phase
        # ramp = circular translation by the (trainable) shift vector (s_h,s_w). Unit gain
        # |Ŵ|=1, linear phase ⇒ constant group velocity s and ZERO dispersion; with a strong
        # leak (A≈0, log_neg_lambda large) the sheet is a ballistic delay-line that carries a
        # packet coherently at any depth — breaking the drift↔dispersion trade-off of :aniso.
        W_hat = exp.((-1im) .* (ps.shift_h .* st.qh .+ ps.shift_w .* st.qw))   # (H,W)
        return A_step, g, W_hat
    else  # :dog / :aniso
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
        W_hat = fft(W_full)
        if l.coupling === :aniso
            # Directed advection: a real, ANTISYMMETRIC stencil whose transform is
            # purely imaginary and ODD in q — Ŵ_adv(q) = −i(β_h·sin q_h + β_w·sin q_w).
            # This is the only term that adds a *net* group velocity v = g·β (a drifting
            # packet); it breaks the reflection symmetry W(d)=W(−d) while staying a
            # function of the displacement d alone, so the coupling is still FFT-diagonal
            # (parallel-scan trainable). Added directly in Fourier — no mutation, AD-safe.
            adv = (-1im) .* (ps.beta_h .* st.sin_qh .+ ps.beta_w .* st.sin_qw)  # (H,W)
            W_hat = W_hat .+ adv
        end
        return A_step, g, W_hat
    end
end

# ---- Dispersion diagnostics (packet-spreading analysis) ---------------
#
# Tools to attack the low-dispersion directed-transport problem analytically:
# read the along-drift-axis dispersion straight off the actual coupling W_hat and
# report the carrier-mode quantities that govern whether a packet holds together.
#
# Two spreading mechanisms (see the derivation): (i) NARROWING — the q-dependence
# of the growth/damping real part (gain_curv = Γ''(q*) / Re Λ''); (ii) GVD — the
# curvature of the phase band (gvd = Ω''(q*) / d²ImΛ). Directed transport wants
# both ≈ 0 at the carrier q* while v_g(q*) ≠ 0 (advection).

"""
    dispersion_diagnostics(l::PhasorWaveSheet, ps, st; axis=:h, mode=l.transmit) -> NamedTuple

Along-drift-axis (`:h` = rows, the `β_h` advection axis; `:w` = cols) dispersion of
the sheet, computed from the real coupling `W_hat` at the transverse-DC slice.

`mode`:
- `:potential` — per-mode map `M(q)=A+g·Ŵ(q)`; `growth = log|M|` (per step, >0
  unstable), phase `Θ=arg M`. Exact for the linear sheet.
- `:spike` — Kuramoto phase-perturbation generator `Λ(q)=(g/R)[RW(q)−RW(0)]`,
  `RW = FT{Re W(d)} = ½[Ŵ(q)+conj Ŵ(−q)]`; `growth = Re Λ` (damping, ≤0 stable),
  phase `= Im Λ`. Up to the positive scale `g/R` (unknown mean-field `R`), which
  does not move `q*`, the GVD zero, or curvature signs.

Group velocity / GVD use branch-safe derivatives `M'/M` (no phase unwrap).

!!! warning "Not valid for a non-integer `:shift`"
    The derivatives are `circshift` finite differences, which treat the sampled
    band as exactly 2π-periodic. For `coupling = :shift` that holds only when the
    shift vector is an **integer** number of sites: at `s = 0.7` the ramp
    `e^{-isq}` jumps by `2πs` across the Brillouin-zone edge, the wrapped stencil
    straddles a discontinuity, and the second difference blows up — measured
    `gvd_star = 88.4` and `v_g_star = 3.83` for a medium whose true group velocity
    is 0.7 and whose true GVD is 0. Downstream, [`transport_forecast`](@ref) then
    predicts `t½ ≈ 1.6` for a translation that is provably lossless. Use integer
    shifts when reading these diagnostics, or read the shift vector directly.

# Returns
`(; q, growth, freq, v_g, gvd, gain_curv, q_star, growth_star, v_g_star,
   gvd_star, gain_curv_star, mode, axis)` — per-`q` vectors plus carrier-mode
(`q_star = argmax growth`) scalars. `v_g` in sites/period; `gvd`/`gain_curv` are
the phase / growth band curvatures at the carrier (both →0 = dispersionless).
"""
function dispersion_diagnostics(l::PhasorWaveSheet, ps::LuxParams, st::NamedTuple;
                                axis::Symbol = :h, mode::Symbol = l.transmit)
    axis in (:h, :w) || throw(ArgumentError("axis must be :h or :w, got :$axis"))
    mode in (:potential, :spike) || throw(ArgumentError("mode must be :potential or :spike"))
    ω = period_to_angfreq(l.spk_args.t_period)
    A_step, g, W_hat = _build_coupling(l, ps, st, ω)
    H, W = size(W_hat)
    A = ComplexF32(A_step[1]); gg = Float32(g[1])
    N = axis === :h ? H : W
    Wq = ComplexF32.(collect(axis === :h ? W_hat[:, 1] : W_hat[1, :]))    # transverse-DC slice
    dq = 2f0 * Float32(pi) / N
    qwrap = Float32.([i <= N ÷ 2 ? i : i - N for i in 0:N-1]) .* dq        # (−π,π]
    d1(v) = (circshift(v, -1) .- circshift(v, 1)) ./ (2f0 * dq)
    d2(v) = (circshift(v, -1) .- 2f0 .* v .+ circshift(v, 1)) ./ dq^2

    local growth, freq, Fp, Fpp
    if mode === :potential
        M = A .+ gg .* Wq                                                 # per-mode map
        growth = log.(abs.(M)); freq = angle.(M)
        Fp = d1(M) ./ M                                                   # d(log M)/dq
        Fpp = d2(M) ./ M .- Fp .^ 2                                       # d²(log M)/dq²
    else  # :spike
        Wneg = Wq[mod.(-(0:N-1), N) .+ 1]                                 # Ŵ(−q)
        RW = 0.5f0 .* (Wq .+ conj.(Wneg))                                 # FT{Re W(d)}
        Λ = gg .* (RW .- RW[1])                                           # RW(0) at index 1
        growth = real.(Λ); freq = imag.(Λ)
        Fp = d1(Λ); Fpp = d2(Λ)
    end
    v_g = .-imag.(Fp)                                                     # group velocity
    gvd = imag.(Fpp)                                                      # phase-band curvature (GVD)
    gain_curv = real.(Fpp)                                                # growth/damping curvature (narrowing)

    ks = argmax(growth)
    return (; q = qwrap, growth, freq, v_g, gvd, gain_curv,
              q_star = qwrap[ks], growth_star = growth[ks], v_g_star = v_g[ks],
              gvd_star = gvd[ks], gain_curv_star = gain_curv[ks], mode, axis)
end

# ---- The symbol's spectrum sets its transport (report §4.6) ------------
#
# `dispersion_diagnostics` characterises the medium one mode at a time. But a
# symbol is not a mode — it is a spatial pattern, and the wave that arrives after
# propagation is, mode by mode, M(q)·ĉ(q). So what the medium does to a symbol is
# fixed by the OVERLAP of the symbol's spectral occupancy ĉ(q) with the flat
# regions of the band (flat gain Γ, flat/linear phase Ω).
#
# Fourier's uncertainty principle turns that into a statement about the code's
# spatial extent: a CONCENTRATED code is broadband — it samples the whole band, so
# its components are attenuated unequally (gain narrowing) and drift at different
# speeds (GVD) — while an EXTENDED code tuned to q* is narrowband and translates
# almost rigidly.
#
# These two functions put that reasoning in code. Report §4.6 states it and draws
# it (Figure 10) from hand-computed numbers; nothing computed it on demand, and
# nothing checked the prediction against a measured survival distance. The
# companion measurement is `demos/wave_transport_fidelity.jl`.

"""
    spectral_occupancy(code; axis=:h) -> NamedTuple

The spatial-frequency content of a code laid out on the sheet, reduced to the
transport axis — the `ĉ(q)` of report §4.6.

`code` is an `(H, W, …)` field: `Complex` (used directly), `Phase` (mapped through
[`angle_to_complex`](@ref)), or `Real` (taken as a real amplitude field). Trailing
dimensions are summed in **power**, so a batch of codes gives their mean spectrum.

The reduction to one axis is a *marginal*, not a slice: `p(q_h) = Σ_{q_w} |ĉ|²`,
normalised to sum 1. A slice at `q_w = 0` would be wrong — a code concentrated in
`x` has spectral energy at every `q_w`, and all of it drifts along `q_h`.

# Returns
`(; q, p, q_dom, q_width, q_rms, w_x, axis)`

- `q` — wrapped Brillouin-zone coordinates in unshifted FFT order, **the same grid
  [`dispersion_diagnostics`](@ref) returns**, so `p` and `v_g` align elementwise.
- `p` — normalised power marginal along `axis`.
- `q_dom` — dominant `|q|` (the folded spectrum's peak).
- `q_width` — power-weighted RMS spread of `|q|` about `q_dom`. This is the
  narrowband/broadband number: small = tuned, large = spans the band.
- `q_rms` — `√(Σ p·q²)`, the second moment about zero.
- `w_x` — the code's spatial *envelope* width along `axis` in sites, as a
  **wrap-aware** circular standard deviation (`√(−2 ln|R|)` on the intensity
  marginal), floored at one site. A plain variance would read a packet
  straddling the torus seam as sheet-wide.
- `ell` — the code's **coherence length** along `axis`: the first lag at which its
  autocorrelation `ℱ⁻¹|ĉ|²` falls below `1/e`. This, not `w_x`, is the scale that
  dispersion has to shear across before the symbol stops being recoverable, and
  the two come apart exactly where it matters: a random FHRR payload spread over
  the whole sheet has `w_x` ≈ sheet-wide (uniform intensity) but `ell` = 1 (phases
  decorrelate site to site). Using `w_x` predicts such a code survives forever; it
  does not.

See also [`transport_forecast`](@ref), [`dispersion_diagnostics`](@ref).
"""
function spectral_occupancy(code::AbstractArray; axis::Symbol = :h)
    axis in (:h, :w) || throw(ArgumentError("axis must be :h or :w, got :$axis"))
    ndims(code) >= 2 || throw(ArgumentError("code must have at least 2 spatial dims"))
    H, W = size(code, 1), size(code, 2)
    c = if eltype(code) <: Complex
        ComplexF32.(code)
    elseif eltype(code) <: Phase
        angle_to_complex(code)
    else
        ComplexF32.(code)
    end
    cf = reshape(c, H, W, :)                                          # (H,W,rest)

    # Power spectrum, summed over trailing dims, then marginalised onto `axis`.
    P = sum(abs2.(fft(cf, (1, 2))); dims = 3)[:, :, 1]                # (H,W)
    ell = _coherence_length(P, axis)
    pmarg = axis === :h ? vec(sum(P; dims = 2)) : vec(sum(P; dims = 1))
    p = pmarg ./ (sum(pmarg) + 1f-20)
    N = length(p)
    dq = 2f0 * Float32(pi) / N
    q = Float32.([i <= N ÷ 2 ? i : i - N for i in 0:N-1]) .* dq       # matches dispersion_diagnostics

    # Fold onto |q| to talk about tuning width without ±q cancelling.
    aq = abs.(q)
    q_dom = aq[argmax(p)]
    q_width = sqrt(max(0f0, sum(p .* (aq .- q_dom) .^ 2)))
    q_rms = sqrt(max(0f0, sum(p .* q .^ 2)))

    # Spatial width along `axis`, wrap-aware (circular std of the intensity marginal).
    I2 = abs2.(cf)
    imarg = axis === :h ? vec(sum(I2; dims = (2, 3))) : vec(sum(I2; dims = (1, 3)))
    M = length(imarg)
    tot = sum(imarg) + 1f-20
    Rv = sum(imarg .* cis.(2f0 * Float32(pi) .* (0:M-1) ./ M)) / tot
    # Clamped at both ends: below, because the lattice cannot resolve a code
    # narrower than a site (and an unfloored single-site delta returns exactly 0,
    # which would collapse any width-over-velocity estimate); above, because
    # |R| → 0 for a perfectly uniform marginal and the unclamped formula returns
    # `Inf` rather than "as wide as the sheet".
    Rm = clamp(abs(Rv), 1f-7, 1f0 - 1f-7)
    w_x = max(1f0, sqrt(max(0f0, -2f0 * log(Rm))) * Float32(M) / (2f0 * Float32(pi)))

    return (; q, p, q_dom, q_width, q_rms, w_x, ell, axis)
end

# Coherence length along `axis` from a power spectrum `P (H,W)`: the first lag at
# which the normalised autocorrelation ℱ⁻¹|ĉ|² (Wiener–Khinchin) falls below 1/e.
#
# A 1/e crossing rather than a moment. A finite random-phase code has an
# autocorrelation that is a spike at zero lag sitting on a noise pedestal of height
# ~1/√M across every other lag; that pedestal carries M·(1/√M) = √M of the total
# mass, so ANY moment-based width is dominated by it and reports a sheet-scale
# coherence length for a code that decorrelates in one site (measured ℓ = 16.2 on a
# 48-wide random payload, true answer 1). The crossing is immune: the pedestal sits
# at 0.14, well under 1/e = 0.368.
#
# Read at ZERO lag of the other axis, not summed over it — summing mixes the two
# axes' decay into one number, which is the thing `axis` exists to separate.
function _coherence_length(P::AbstractMatrix, axis::Symbol)
    ac = ifft(ComplexF32.(P))
    m = abs.(axis === :h ? ac[:, 1] : ac[1, :])
    M = length(m)
    peak = m[1] + 1f-20
    thr = peak / Float32(exp(1))
    for k in 1:(M ÷ 2)
        (m[k + 1] < thr && m[M - k + 1] < thr) && return Float32(k)
    end
    return Float32(M ÷ 2)
end

# Full 2-D band fields: per-step log-gain and the group-velocity vector field,
# both branch-safe. The `:spike` branch mirrors `dispersion_diagnostics`: the
# Kuramoto generator Λ(q) = g[RW(q) − RW(0)] with RW = ℱ{Re W(d)}, whose
# derivative is taken directly rather than as a log-derivative.
function _band_fields_2d(l::PhasorWaveSheet, ps::LuxParams, st::NamedTuple, mode::Symbol)
    ω = period_to_angfreq(l.spk_args.t_period)
    A_step, g, W_hat = _build_coupling(l, ps, st, ω)
    H, W = size(W_hat)
    hh = 2f0 * Float32(pi) / H
    hw = 2f0 * Float32(pi) / W
    dh(X) = (circshift(X, (-1, 0)) .- circshift(X, (1, 0))) ./ (2f0 * hh)
    dw(X) = (circshift(X, (0, -1)) .- circshift(X, (0, 1))) ./ (2f0 * hw)
    if mode === :potential
        M = ComplexF32(A_step[1]) .+ Float32(g[1]) .* W_hat
        return log.(abs.(M)), .-imag.(dh(M) ./ M), .-imag.(dw(M) ./ M)
    else  # :spike
        Wneg = W_hat[mod.(-(0:H-1), H) .+ 1, mod.(-(0:W-1), W) .+ 1]
        RW = 0.5f0 .* (W_hat .+ conj.(Wneg))
        Λ = Float32(g[1]) .* (RW .- RW[1, 1])
        return real.(Λ), .-imag.(dh(Λ)), .-imag.(dw(Λ))
    end
end

"""
    transport_forecast(l::PhasorWaveSheet, ps, st, code; axis=:both, mode=l.transmit) -> NamedTuple

Predict how far a given code survives in a given medium, by evaluating the band
against the code's own [`spectral_occupancy`](@ref). This is report §4.6 made
computable.

Two spreading channels are scored, and the shorter one wins:

- **GVD** — the code's components drift at different speeds, so it shears apart
  once the differential drift has carried them past the code's own coherence
  length: `t_half_gvd = ℓ / Δv_g`.
- **Gain narrowing** — the band-pass gain reweights the components, so the code is
  reshaped toward `q*` and stops being the symbol that was injected:
  `t_half_gain = ln2 / ΔΓ`.

`t_half_pred = min(...)` is in **periods** (one discrete step = one carrier
period), and `d_half_pred = |v̄_g|·t_half_pred` in sites.

!!! warning "`axis = :both` is the default, and a single axis is usually wrong"
    Report §4.6 reasons along one axis, and that is incomplete for any real code:
    a symbol decoheres through **whichever axis is broadband**, so scoring only
    the transport axis flatters codes whose payload varies across it. With
    `:both` the two axes are combined as `min(ℓ_h/Δv_h, ℓ_w/Δv_w)` — the first one
    to shear wins. Keep `:h`/`:w` only when you deliberately want the 1-D slice
    that matches [`dispersion_diagnostics`](@ref).

!!! note "How well it actually predicts (measured)"
    Against end-to-end symbol survival in `demos/wave_transport_fidelity.jl`
    (36 diffusive arms, 48², L=100): pooled Pearson **+0.59**, Spearman +0.53.
    It works on the **isotropic** media (`:dog` matched-c Spearman +0.79, `:dog`
    c=40 +0.45) and carries **no signal on the advective ones** (`:aniso` sharp
    −0.15, gentle-DC −0.03). It is also badly wrong for a non-integer `:shift`
    (predicts t½ ≈ 1.6 for a provably lossless translation) — see the
    Brillouin-zone warning on [`dispersion_diagnostics`](@ref). Use it to rank
    codes within an isotropic medium; do not use it as a survival guarantee.

!!! note "`d_half_pred` is zero for a reflection-symmetric medium, and that is correct"
    An isotropic `:dog` kernel has an even band, so `v̄_g = 0` exactly and the
    predicted transport *distance* is 0 — a stone-in-a-pond sheet carries a symbol
    nowhere, however long it stays coherent (report §10). The coherence **time**
    `t_half_pred` is still meaningful there, and is the quantity to compare across
    media. See [`radial_band`](@ref) for the radial speed of such a sheet.

!!! warning "`:spike` gain is known only up to a positive scale"
    In `:spike` mode the Kuramoto growth carries the unknown mean-field factor
    `g/R` (see [`dispersion_diagnostics`](@ref)), so `t_half_gain` — and therefore
    `t_half_pred` when gain is the limiter — is comparable *between codes* in one
    medium but not *between* media. `limiter` reports which channel bound it.

# Returns
`(; q_dom, q_width, w_x, ell, vg_mean, vg_spread, gamma_mean, gamma_spread,
   t_half_gvd, t_half_gain, t_half_pred, d_half_pred, limiter, mode, axis)`

Under `axis = :both`, `vg_mean` is the drift along the dominant axis and `ell`
the coherence length on the axis that binds; `limiter` is `:gvd`, `:gain`, or
`:none` (neither channel active — a rigid conveyor, where both spreads vanish
and the prediction is `Inf`).
"""
function transport_forecast(l::PhasorWaveSheet, ps::LuxParams, st::NamedTuple,
                            code::AbstractArray; axis::Symbol = :both,
                            mode::Symbol = l.transmit)
    axis in (:h, :w, :both) ||
        throw(ArgumentError("axis must be :h, :w or :both, got :$axis"))
    (size(code, 1) == l.grid_h && size(code, 2) == l.grid_w) ||
        throw(DimensionMismatch("code is $(size(code, 1))×$(size(code, 2)) but the sheet " *
                                "is $(l.grid_h)×$(l.grid_w); the code must live on the sheet"))

    sh = spectral_occupancy(code; axis = :h)
    sw = spectral_occupancy(code; axis = :w)

    if axis === :both
        # 2-D occupancy against the 2-D band: each axis shears at its own rate
        # over its own coherence length, and the FIRST one to shear kills the
        # symbol — hence `min`, not an average.
        cf = eltype(code) <: Phase ? angle_to_complex(code) : ComplexF32.(code)
        P = sum(abs2.(fft(reshape(cf, size(code, 1), size(code, 2), :), (1, 2))); dims = 3)[:, :, 1]
        p2 = P ./ (sum(P) + 1f-20)
        growth, vgh, vgw = _band_fields_2d(l, ps, st, mode)

        vh_mean = sum(p2 .* vgh); vw_mean = sum(p2 .* vgw)
        vh_sp = sqrt(max(0f0, sum(p2 .* (vgh .- vh_mean) .^ 2)))
        vw_sp = sqrt(max(0f0, sum(p2 .* (vgw .- vw_mean) .^ 2)))
        gamma_mean = sum(p2 .* growth)
        gamma_spread = sqrt(max(0f0, sum(p2 .* (growth .- gamma_mean) .^ 2)))

        th = vh_sp > 1f-8 ? sh.ell / vh_sp : Inf32
        tw = vw_sp > 1f-8 ? sw.ell / vw_sp : Inf32
        t_half_gvd = min(th, tw)
        binds_h = th <= tw
        vg_mean = abs(vh_mean) >= abs(vw_mean) ? vh_mean : vw_mean
        vg_spread = sqrt(vh_sp^2 + vw_sp^2)
        s = binds_h ? sh : sw
    else
        d = dispersion_diagnostics(l, ps, st; axis = axis, mode = mode)
        s = axis === :h ? sh : sw
        p = s.p
        vg_mean = sum(p .* d.v_g)
        vg_spread = sqrt(max(0f0, sum(p .* (d.v_g .- vg_mean) .^ 2)))
        gamma_mean = sum(p .* d.growth)
        gamma_spread = sqrt(max(0f0, sum(p .* (d.growth .- gamma_mean) .^ 2)))
        t_half_gvd = vg_spread > 1f-8 ? s.ell / vg_spread : Inf32
    end

    t_half_gain = gamma_spread > 1f-8 ? log(2f0) / gamma_spread : Inf32
    t_half_pred = min(t_half_gvd, t_half_gain)
    limiter = isfinite(t_half_pred) ? (t_half_gvd <= t_half_gain ? :gvd : :gain) : :none
    d_half_pred = isfinite(t_half_pred) ? abs(vg_mean) * t_half_pred : Inf32

    return (; q_dom = s.q_dom, q_width = s.q_width, w_x = s.w_x, ell = s.ell,
              vg_mean, vg_spread, gamma_mean, gamma_spread,
              t_half_gvd, t_half_gain, t_half_pred, d_half_pred, limiter,
              mode = mode, axis = axis)
end

"""
    zero_gvd_speed(l, ps, st; axis=:h, mode=l.transmit, crange=(2,200), n=64) -> NamedTuple

Solve condition (C): find the conduction speed `c` (pixels/period) at which the
carrier-mode group-velocity dispersion vanishes, `β₂(q*) = 0` — the zero-dispersion
transport regime. Scans `c ∈ crange` (log-spaced) for a sign change of
`gvd_star(c)` and geometric-bisects. Needs `:dog`/`:aniso` (has `log_speed`).

# Returns
`(; c, v_g_star, gain_curv_star, q_star, found)` at the root (`found=false` and
`c=nothing` if no sign change in `crange`).
"""
function zero_gvd_speed(l::PhasorWaveSheet, ps::LuxParams, st::NamedTuple;
                        axis::Symbol = :h, mode::Symbol = l.transmit,
                        crange = (2f0, 200f0), n::Int = 64)
    l.coupling in (:dog, :aniso) ||
        throw(ArgumentError("zero_gvd_speed needs coupling :dog or :aniso (has log_speed)"))
    gvd_at(c) = dispersion_diagnostics(l, merge(ps, (; log_speed = Float32[log(c)])), st;
                                       axis = axis, mode = mode).gvd_star
    cs = exp.(range(log(Float32(crange[1])), log(Float32(crange[2])); length = n))
    gs = [gvd_at(Float32(c)) for c in cs]
    idx = findfirst(i -> gs[i] * gs[i+1] < 0, 1:length(cs)-1)
    idx === nothing && return (; c = nothing, v_g_star = NaN32, gain_curv_star = NaN32,
                                 q_star = NaN32, found = false)
    a, b, ga = Float32(cs[idx]), Float32(cs[idx+1]), gs[idx]
    for _ in 1:44
        m = sqrt(a * b); gm = gvd_at(m)
        (ga * gm < 0) ? (b = m) : (a = m; ga = gm)
    end
    c = sqrt(a * b)
    d = dispersion_diagnostics(l, merge(ps, (; log_speed = Float32[log(c)])), st;
                               axis = axis, mode = mode)
    return (; c = c, v_g_star = d.v_g_star, gain_curv_star = d.gain_curv_star,
              q_star = d.q_star, found = true)
end

# ---- Traveling vs standing: the transport diagnostic -------------------
#
# A reflection-symmetric kernel amplifies +q and −q identically, so a localized
# impulse ALWAYS excites counter-propagating pairs at the selected wavenumber.
# Whether the result travels or stands is decided by a single number: the group
# velocity at the band-selected mode.
#
#   v_g(q*) = 0  → the pair never separates, interferes → STANDING pattern
#   v_g(q*) ≠ 0  → the two packets walk apart          → TRAVELING (expanding ring)
#
# With weak conduction delay this is not a free choice. To first order in ω·r/c,
#     Γ(q) ≈ λ + g·D̂(q),        Ω(q) ≈ ω − g(ω/c)·(rD)^(q)
# are cosine transforms of nearly the same band-pass radial profile, so argmax Γ
# lands on an extremum of Ω and v_g(q*) → 0 *identically* — standing rings are
# the generic outcome, not a parameter accident. Escaping that regime needs
# enough delay to decouple the two transforms, which happens once the conduction
# phase across the kernel's own width approaches half a cycle:
#
#     φ = ω·σ_I/c ≈ π    ⟺    c ≈ 2σ_I/T        (`matched_conduction_speed`)
#
# Measured across σ_I ∈ [1.5, 9]: the c maximizing |v_g(q*)| sits at
# φ/π = 1.17, 0.99, 1.15, 1.10, 1.09, 1.08 — the rule holds over a 6× range, and
# the selected wavelength tracks λ* ≈ 3σ_I.
#
# Biological check: Muller, Chavane, Reynolds & Sejnowski (Nat Rev Neurosci 19:255,
# 2018) identify distance-dependent conduction delay as "the key factor governing
# the generation and propagation" of mesoscopic cortical waves, and report
# unmyelinated horizontal fibres at 0.1–0.8 m/s. At ~0.5 mm/site and a 25 ms
# carrier those bounds map to c ≈ 5–40 sites/period: the matched value sits at the
# slow (typical) end, and c = 40 sits at the very fastest — i.e. in the weak-delay
# corner where waves cannot form.

"""
    matched_conduction_speed(sigma_inh, t_period = 1) -> Float32

Conduction speed `c` (sites per period) that places the sheet in the
traveling-wave regime: the delay phase across the inhibitory surround reaches
half a cycle, `φ = ω·σ_I/c = π`, giving `c = 2σ_I/T`.

Below this (larger `c`, weaker delay) the gain peak and the phase-band extremum
coincide, `v_g(q*) → 0`, and an impulse produces **standing** concentric rings.
Near it, the band-selected mode is also the fastest-transporting one and an
impulse produces an **expanding ring**. This is the default used by
[`PhasorWaveSheet`](@ref) when `init_log_speed` is not given.

See [`wave_transport`](@ref) to check the resulting regime, and
[`radial_band`](@ref) for the ring-averaged bands.
"""
matched_conduction_speed(sigma_inh::Real, t_period::Real = 1) =
    Float32(2 * sigma_inh / t_period)

# ---- Emission threshold -----------------------------------------------
#
# The `:spike` emit is  s = z/√(|z|² + θ²)  (`normalize_to_unit_circle` with
# ε = θ²). θ is not a numerical safety constant — it is the sheet's FIRING
# THRESHOLD, and it interpolates continuously between the two transmission
# modes:
#
#     |z| ≫ θ :  s ≈ z/|z|      hard spike, amplitude discarded
#     |z| ≪ θ :  s ≈ z/θ        LINEAR — :potential coupling at gain g/θ
#
# So the subthreshold small-signal map of a spike sheet is exactly
# M(q) = A + (g/θ)·Ŵ(q): the same object §2 derives for :potential, with a
# computable effective gain in place of the "unknown mean-field r" of the §3
# Kuramoto linearization. This is what makes `dispersion`, `radial_band` and
# `wave_transport` meaningful for a spike sheet at all.
#
# WHERE THE THRESHOLD HAS TO SIT. Bisecting the flood/extinct boundary θ* on a
# seeded impulse (N=64, :dog, matched c) gives θ* ∝ g EXACTLY —
#
#     λ      θ*/g at g=0.3 / 1 / 3        θ*/(g·max|Ŵ|)
#     0.15   8.750  8.797  8.815          1.164
#     0.50   6.178  6.179  6.179          0.818
#     1.50   5.337  5.338  5.338          0.706
#
# — with only a weak λ-dependence. So `g·max_q|Ŵ(q)|` is the right SCALE, and
# it is also the right physical statement: it is the drive a site receives when
# its entire neighbourhood emits unit spikes in phase, so θ = g·max|Ŵ| means
# "fire only on near-maximal local coherence."
#
# It is stable in grid size (θ*/θ_ref = 1.10 at N=16, then 1.13–1.17 for
# N ≥ 24 — max|Ŵ| has converged to 4 s.f. by N=32) and holds for the other
# band-pass couplings (:aniso 1.28, :stencil 1.34).
#
# WHY A CONSTANT IS NOT ENOUGH. The band of θ giving partial (1%–99%) activity
# is only ≈1.3× wide, and it MOVES as θ* ∝ g. A fixed θ is therefore wrong the
# same way the old fixed `c = 40` default was wrong: correct at one operating
# point and silently out of regime once `g` trains. That is the argument for
# `homeostasis`, not a preference for biological realism.

"""
    emission_threshold(l::PhasorWaveSheet, ps, st) -> Float32

Derived reference emission threshold `θ_ref = g·max_q|Ŵ(q)|` for a `:spike`
sheet — the coupling drive a site receives when its whole neighbourhood emits
unit spikes in phase. Setting `θ` here makes firing require near-maximal local
coherence.

[`PhasorWaveSheet`](@ref) defaults `θ` to `init_theta_frac` (1.4) times this,
because the reference itself sits just *below* the flood boundary. The
flood/extinct boundary is at `1.10–1.17 × θ_ref` for `:dog` at `λ = 0.15`
(`1.28` for `:aniso`, `1.34` for `:stencil`); `θ_ref` scales **exactly** with
`g`, which is the property that matters — the usable band is only ≈1.3× wide,
so the *scaling* is what keeps the sheet in regime as `g` trains.

!!! warning "Not a criticality reference for `:shift`"
    `coupling = :shift` is a unit-gain phase ramp (`|Ŵ| ≡ 1`), so it has no
    amplifying band and never floods at any `θ`. `θ_ref = g` is still a sane
    amplitude scale there, but it does not mark a transition.

!!! note "Calibrated at the default damping"
    `θ*/θ_ref` falls from 1.16 to 0.71 as `λ` goes 0.15 → 1.5, so the `1.4`
    multiplier lands on the *extinct* side for `λ ≳ 0.5`. If you move `λ` far
    from the default, either re-measure the boundary or use
    `homeostasis = :global`, which finds the operating point itself (measured
    `θ/θ_ref` → 1.55, 0.88, 0.76 at `λ` = 0.15, 0.5, 1.5, bracketing the
    directly-bisected boundary in each case).

See also [`matched_conduction_speed`](@ref) — the other derived default.
"""
function emission_threshold(l::PhasorWaveSheet, ps::LuxParams, st::NamedTuple)
    ω = period_to_angfreq(l.spk_args.t_period)
    _, g, W_hat = _build_coupling(l, ps, st, ω)
    return Float32(only(g) * maximum(abs.(W_hat)))
end

# Same quantity from the `init_*` fields alone, for `initialparameters` (which
# has no `ps` yet). Mirrors `_build_coupling` for each coupling family.
function _init_emission_threshold(l::PhasorWaveSheet)
    H, W = l.grid_h, l.grid_w
    ω = Float32(period_to_angfreq(l.spk_args.t_period))
    g = exp(l.init_log_g)
    if l.coupling === :shift
        return Float32(g)                              # |Ŵ| ≡ 1 for a pure phase ramp
    end
    rgrid = _wrapped_distance_grid(H, W)
    σe = exp(l.init_log_sigma_exc); σi = exp(l.init_log_sigma_inh)
    c  = exp(l.init_log_speed)
    mag = l.init_A_exc .* exp.(-rgrid .^ 2 ./ (2f0 * σe^2)) .-
          l.init_B_inh .* exp.(-rgrid .^ 2 ./ (2f0 * σi^2))
    W_full = ComplexF32.(mag .* (rgrid .> 0f0)) .* exp.(-1im .* (ω .* rgrid ./ c))
    if l.coupling === :stencil
        # The stencil is seeded from the same DoG but TRUNCATED to radius R, so
        # its Ŵ differs; build it from the seed rather than from the full DoG.
        sre, sim = _dog_stencil_init(l)
        place = _stencil_placement(H, W, l.stencil_radius)
        W_full = reshape(place * reshape(ComplexF32.(sre) .+ 1im .* ComplexF32.(sim), :), H, W)
    end
    W_hat = fft(W_full)
    if l.coupling === :aniso
        sin_qh = reshape(Float32.(sin.(2f0 * pi .* (0:H-1) ./ H)), H, 1)
        sin_qw = reshape(Float32.(sin.(2f0 * pi .* (0:W-1) ./ W)), 1, W)
        W_hat = W_hat .+ (-1im) .* (l.init_beta_h .* sin_qh .+ l.init_beta_w .* sin_qw)
    end
    return Float32(g * maximum(abs.(W_hat)))
end

# The layer's configured θ: explicit if given, else frac × the derived reference.
_configured_theta(l::PhasorWaveSheet) =
    l.init_log_theta === nothing ?
        Float32(l.init_theta_frac * _init_emission_threshold(l)) :
        Float32(exp(l.init_log_theta))

# Wrapped Brillouin-zone coordinate grids in standard (unshifted) FFT order.
function _bz_grids(H::Int, W::Int)
    qh = Float32[2f0 * pi * ((i - 1) > H ÷ 2 ? (i - 1 - H) : (i - 1)) / H for i in 1:H]
    qw = Float32[2f0 * pi * ((j - 1) > W ÷ 2 ? (j - 1 - W) : (j - 1)) / W for j in 1:W]
    QH = repeat(reshape(qh, H, 1), 1, W)
    QW = repeat(reshape(qw, 1, W), H, 1)
    return QH, QW, sqrt.(QH .^ 2 .+ QW .^ 2)
end

# Radial group speed field v_r = ⟨v_g · q̂⟩, branch-safe via d(log M)/dq = M'/M
# (no phase unwrapping even if arg M approaches ±π). circshift is the correct
# Brillouin-zone wrap: M is 2π-periodic in each q component.
function _radial_group_speed(M::AbstractMatrix, QH, QW, qmag)
    H, W = size(M)
    hh = 2f0 * pi / H
    hw = 2f0 * pi / W
    Fh = (circshift(M, (-1, 0)) .- circshift(M, (1, 0))) ./ (2f0 * hh) ./ M
    Fw = (circshift(M, (0, -1)) .- circshift(M, (0, 1))) ./ (2f0 * hw) ./ M
    vgh = .-imag.(Fh)                                   # v_g = -∇_q Θ
    vgw = .-imag.(Fw)
    VR  = (vgh .* QH .+ vgw .* QW) ./ max.(qmag, 1f-6)  # project THEN average
    return VR, vgh, vgw
end

"""
    radial_band(l::PhasorWaveSheet, ps, st; dq = 0.08, mode = :discrete) -> NamedTuple

Ring-averaged band structure vs `|q|` for the isotropic (`:dog`) sheet — the
2-D analogue of [`dispersion_diagnostics`](@ref), which only reads a
transverse-DC axis slice and therefore **misses the true critical mode whenever
it sits off-axis** (a diagonal lattice direction).

# The velocity metric

The group velocity is a vector, `v_g = -∇_q Ω`. For an isotropic band it points
radially, `v_g = v_r(|q|)·q̂`, so averaging the *vector* over an annulus gives
**exactly zero** — every outward mode is cancelled by its opposite-travelling
partner. That zero is a property of the concentric geometry, not of the wave,
and it is the trap this function exists to avoid. It is returned as `v_vec` so
the null is visible rather than silent.

The quantity that is not zero, and that is the rate the wavefront expands, is
the radial projection taken **before** averaging:

    v_r(|q|) = ⟨v_g · q̂⟩_annulus = -dΩ/d|q|

Sign convention follows `docs/wave_dispersion_derivation.md` (`e^{+iΩt}`), so
positive `v_r` means outward.

# Returns

`(; q, gain, v_r, v_vec)` — per-annulus `|q|`, growth rate `Γ = Re κ`, radial
group speed (sites/period), and the vector-average magnitude (≡ 0).
"""
function radial_band(l::PhasorWaveSheet, ps::LuxParams, st::NamedTuple;
                     dq::Real = 0.08, mode::Symbol = :discrete)
    d = dispersion(l, ps, st; mode = mode)
    H, W = l.grid_h, l.grid_w
    QH, QW, qmag = _bz_grids(H, W)
    VR, vgh, vgw = _radial_group_speed(d.M, QH, QW, qmag)
    gamma = real.(d.k_eff)

    dqf = Float32(dq)
    qmid = Float32[]; gain = Float32[]; v_r = Float32[]; v_vec = Float32[]
    for b in 0f0:dqf:(maximum(qmag) - dqf)
        mask = (qmag .>= b) .& (qmag .< b + dqf)
        sum(mask) == 0 && continue
        push!(qmid,  mean(qmag[mask]))
        push!(gain,  mean(gamma[mask]))
        push!(v_r,   mean(VR[mask]))
        push!(v_vec, sqrt(mean(vgh[mask])^2 + mean(vgw[mask])^2))
    end
    return (; q = qmid, gain = gain, v_r = v_r, v_vec = v_vec)
end

"""
    wave_transport(l::PhasorWaveSheet, ps, st; mode = :discrete) -> NamedTuple

Does an impulse on this sheet produce a **traveling** wave or a **standing**
pattern? Answers with one scalar, `transport_ratio ∈ [0,1]`:

    transport_ratio = |v_r(q*)| / max_q |v_r(q)|

i.e. how much of the band's available transport the band-*selected* mode
actually has. `≈ 0` means the amplified mode is the one that does not move
(standing rings); `≈ 1` means the amplified mode is also the fastest-transporting
one (expanding ring). See the comment block above for why the weak-delay limit
forces this to zero.

Unlike [`dispersion_diagnostics`](@ref), the critical mode is located over the
**full 2-D Brillouin zone**, so an off-axis (diagonal) critical mode is found
correctly rather than being read off a slice that does not contain it.

# Returns

- `q_star`, `q_star_vec`, `lambda_star` — the most-unstable mode: `|q|`, its
  `(q_h, q_w)`, and wavelength `2π/|q|` in sites.
- `on_axis` — whether that mode lies on a lattice axis (vs diagonal).
- `v_r_star` — radial group speed there (sites/period; + = outward).
- `v_r_max`, `q_at_v_r_max` — the band's best available transport, and where.
- `transport_ratio` — the verdict number above.
- `delay_phase` — `φ = ω·σ_I/c` in radians; `≈ π` is the matched regime
  (`:dog`/`:aniso` only, `NaN` otherwise). See [`matched_conduction_speed`](@ref).
- `spectral_radius`, `growth_star` — criticality at the selected mode.
- `verdict` — `:traveling` (ratio ≥ 0.5), `:standing` (≤ 0.15), else `:mixed`.

!!! warning "Evaluate at your operating gain"
    The verdict depends on `g`, not just the kernel, because `g` moves which mode
    is most unstable. Well above criticality the DC mode wins outright, so
    `q_star → 0`, `v_r → 0`, and **any** sheet reports `:standing` — including one
    whose kernel is perfectly matched. The layer's `init_log_g = log(1.0)` default
    is ~14× critical for the matched `:dog` sheet and lands squarely there. Tune to
    criticality first (bisect on `dispersion(...).spectral_radius`), then ask.
"""
function wave_transport(l::PhasorWaveSheet, ps::LuxParams, st::NamedTuple;
                        mode::Symbol = :discrete)
    d = dispersion(l, ps, st; mode = mode)
    H, W = l.grid_h, l.grid_w
    QH, QW, qmag = _bz_grids(H, W)
    VR, _, _ = _radial_group_speed(d.M, QH, QW, qmag)

    # True 2-D critical mode: argmax over the whole BZ, not an axis slice.
    idx  = argmax(abs.(d.M))
    qh_s = QH[idx]; qw_s = QW[idx]
    q_s  = qmag[idx]
    on_axis = (abs(qh_s) < 1f-6) || (abs(qw_s) < 1f-6)

    # Take the band's best transport from the FULL v_r field, not the binned
    # profile: annulus averaging smooths the peak down, which can push the ratio
    # slightly above 1 for a mode that sits right on it.
    jmax = argmax(abs.(VR))
    v_r_max = VR[jmax]
    ratio = abs(VR[idx]) / max(abs(v_r_max), 1f-9)

    phi = (l.coupling === :dog || l.coupling === :aniso) ?
        Float32(period_to_angfreq(l.spk_args.t_period) * exp(ps.log_sigma_inh[1]) /
                exp(ps.log_speed[1])) : NaN32

    verdict = ratio >= 0.5f0 ? :traveling : (ratio <= 0.15f0 ? :standing : :mixed)
    return (; q_star = q_s, q_star_vec = (qh_s, qw_s),
              lambda_star = q_s > 1f-6 ? Float32(2pi) / q_s : Inf32,
              on_axis = on_axis,
              v_r_star = VR[idx], v_r_max = v_r_max, q_at_v_r_max = qmag[jmax],
              transport_ratio = ratio, delay_phase = phi,
              spectral_radius = d.spectral_radius,
              growth_star = real(d.k_eff[idx]), verdict = verdict)
end

# ---- Core recurrence ---------------------------------------------------

# What a neuron transmits to its neighbors:
#   :potential — its full complex state z (linear diffusive coupling; can run
#                away, hence the state-level `saturating` snap).
#   :spike     — z/√(|z|² + θ²): an event of magnitude ≈1 once |z| clears the
#                emission threshold θ, and ≈ z/θ below it. The coupling drive is
#                then bounded (|Σ W·s| ≤ Σ|W|), so the leaky-integrator state is
#                BIBO-stable with NO state snap — and |z| survives to encode
#                local phase coherence / interference intensity.
#
# θ is the sheet's firing threshold, NOT a numerical guard. It used to be the
# hardcoded ε = 1e-8 default of `normalize_to_unit_circle` (θ = 1e-4), which is
# ~5 orders of magnitude below the derived reference g·max|Ŵ| ≈ 7.6, so every
# site above 1e-4 emitted a FULL spike and an impulse became an ignition cascade
# rather than a wave. See `emission_threshold`.
_transmit(l::PhasorWaveSheet, z, θ) =
    l.transmit === :spike ? normalize_to_unit_circle(z; ε = θ .^ 2) : z

_transmit(l::PhasorWaveSheet, z) = _transmit(l, z, _static_theta(l))

# θ for call sites with no `ps` in hand (the Tier-2 ODE closure): the layer's
# configured threshold. Homeostasis is a discrete-time recurrence and is not
# available on that path, so the ODE mode holds θ fixed.
_static_theta(l::PhasorWaveSheet) =
    l.transmit === :spike ? _configured_theta(l) : 1f0

# Fire indicator driving the homeostat: hard `|z| > θ` forward, sigmoid
# `σ((|z|−θ)/(βθ))` backward — the same straight-through estimator `moe_gate`
# uses for its top-1 selection.
#
# The STE is not a convenience here, it is required for correctness. ANY smooth
# pointwise indicator makes the homeostat's target degenerate: a mean cannot
# distinguish "5% of sites at 1" from "100% of sites at 0.05", so the controller
# can satisfy the target by driving the sheet to a UNIFORM SUBTHRESHOLD state
# instead of a sparse active one. That is not hypothetical — measured with a
# plain sigmoid at β = 0.05 and θ₀ = θ_ref, the sheet converges to
#
#     θ = 9.402   soft-fire = 0.0500 (on target)   HARD fire = 0.0000
#     mean|z| = 8.017   std|z| = 0.033  (uniform, no structure, no transport)
#
# — a uniform field sitting 2.94 sigmoid-widths below θ, whose tail integrates
# to exactly the target. Shrinking β moves that attractor closer to θ but never
# removes it. The hard forward value has no tail, so the only way to reach the
# target is genuine sparse firing.
#
# β then only sets the gradient scale (and reads physically as threshold jitter
# / trial-to-trial firing variability); it no longer has to carry the dynamics.
function _fire_indicator(z, θ, β)
    r = abs.(z)
    soft = _sigmoid.((r .- θ) ./ (β .* θ))
    hard = ignore_derivatives() do
        Float32.(r .> θ)
    end
    return hard .+ (soft .- ignore_derivatives(soft))     # STE: fwd=hard, bwd=∂soft
end

# One evolution step in the spatial-Fourier-coupled complex plane.
# `z, drive_t, a` are (H,W,B); scalars A_step, g, ρ_a, δ_a are (1,).
# `θg` is (1,) or (1,1,B); `θl` is (H,W,B) or nothing. Returns (z, a, θg, θl).
function _wave_step(l::PhasorWaveSheet, z, drive_t, a, A_step, g, W_hat, ρ_a, δ_a,
                    saturating::Bool, use_adaptation::Bool,
                    θg = nothing, θl = nothing, η_g = nothing, η_l = nothing,
                    target = nothing, β = nothing)
    H, W, B = size(z)
    θ = θg === nothing ? _static_theta(l) :
        (θl === nothing ? reshape(θg, 1, 1, :) : reshape(θg, 1, 1, :) .* θl)
    src = _transmit(l, z, θ)                                             # spike or potential
    coupled = ifft(reshape(W_hat, H, W, 1) .* fft(src, (1, 2)), (1, 2))  # (H,W,B)
    z_lin = reshape(A_step, 1, 1, 1) .* z .+
            reshape(g, 1, 1, 1) .* coupled .+ drive_t
    a_next = a
    if use_adaptation
        a_next = reshape(ρ_a, 1, 1, 1) .* a .+ reshape(δ_a, 1, 1, 1) .* abs.(z)
        z_lin = z_lin .- 1im .* a_next
    end
    z_next = saturating ? normalize_to_unit_circle(ComplexF32.(z_lin)) : ComplexF32.(z_lin)

    # Homeostasis, evaluated on the NEW state so the threshold responds to the
    # activity it will gate next step. Per-batch-element (the sheets in a batch
    # are independent media, so they must regulate independently).
    θg_next = θg; θl_next = θl
    if η_g !== nothing                       # homeostasis on (θg alone = fixed θ)
        fire = _fire_indicator(z_next, θ, β)                             # (H,W,B)
        f̄ = reshape(sum(fire; dims = (1, 2)) ./ Float32(H * W), :)       # (B,)
        θg_next = θg .* exp.(η_g .* (f̄ .- target))
        if θl !== nothing
            # ZERO-SUM per-site term: it carries only the spatial *pattern* of
            # excitability, leaving the overall level entirely to θg. Renormalizing
            # the geometric mean to 1 each step is what keeps the two timescales
            # from fighting over the same degree of freedom.
            θl_raw = θl .* exp.(η_l .* (fire .- reshape(f̄, 1, 1, :)))
            gm = exp.(sum(log.(θl_raw); dims = (1, 2)) ./ Float32(H * W))  # (1,1,B)
            θl_next = θl_raw ./ gm
        end
    end
    return z_next, a_next, θg_next, θl_next
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

    # Emission threshold and (optionally) its homeostatic state. θg/θl are
    # rollout-local, initialized fresh each forward pass so the layer stays a
    # pure function of (ps, st, input).
    θg = η_g = η_l = target = β = θl = nothing
    if l.transmit === :spike && l.homeostasis !== :none
        ones_B = ignore_derivatives() do
            o = similar(st.rgrid, Float32, B); o .= 1f0; return o
        end
        θg     = exp.(ps.log_theta) .* ones_B                 # (B,)
        η_g    = exp.(ps.log_eta_g)
        target = _sigmoid.(ps.logit_target)
        β      = exp.(ps.log_theta_beta)
        if l.homeostasis === :local
            η_l = exp.(ps.log_eta_l)
            θl = ignore_derivatives() do
                t = similar(st.rgrid, Float32, H, W, B); t .= 1f0; return t
            end
        end
    elseif l.transmit === :spike
        θg = ignore_derivatives() do
            o = similar(st.rgrid, Float32, B); o .= 1f0; return o
        end .* exp.(ps.log_theta)                              # fixed θ, still per-batch
    end

    Y = Buffer(similar(z0, ComplexF32, H, W, L, B))
    z = ComplexF32.(z0)
    a = a0
    for t in 1:L
        drive_t = drive === nothing ?
            ignore_derivatives() do
                d = similar(z, ComplexF32, H, W, B); d .= 0f0; return d
            end : drive[:, :, t, :]
        z, a, θg, θl = _wave_step(l, z, drive_t, a, A_step, g, W_hat, ρ_a, δ_a,
                                  l.saturating, l.use_adaptation,
                                  θg, θl, η_g, η_l, target, β)
        Y[:, :, t, :] = z
    end
    return copy(Y)                                    # (H,W,L,B) complex
end

# Parallel FFT-in-time forward for the LINEAR sheet (§5.8 #1 of the sparse-experts
# design note, docs/wavesheet_experts_design.md). Valid only in the linear regime
# — transmit=:potential, no saturation, no adaptation — where the recurrence is
#     z[n+1] = A·z[n] + g·(W ⊛ z[n]) + drive[n].
# The spatial FFT diagonalizes it per mode q into a scalar linear SSM with step
# multiplier M(q) = A + g·Ŵ(q) (the discrete `dispersion` M), whose rollout has
# the closed form
#     ẑ_q[n] = M_q^n·ẑ0_q  +  Σ_{m=1}^n M_q^{n-m}·d̂_q[m],
# a homogeneous term plus a causal time-convolution of the drive with the
# geometric kernel K_q[m] = M_q^{m-1} (reused via `causal_conv`). No sequential
# recurrence and no ODE adjoint — time-parallel via FFT. Equivalent to
# `_wave_rollout` in this regime (see `test_wave_scan_equivalence`).
function _wave_rollout_scan(l::PhasorWaveSheet, ps, st, z0, drive, L::Int)
    (l.transmit === :potential && !l.saturating && !l.use_adaptation) ||
        throw(ArgumentError("_wave_rollout_scan needs the linear sheet " *
              "(transmit=:potential, saturating=false, use_adaptation=false)"))
    ω = period_to_angfreq(l.spk_args.t_period)
    A_step, g, W_hat = _build_coupling(l, ps, st, ω)
    H, W, B = size(z0)
    C = H * W

    # Per-mode step multiplier M(q) = A + g·Ŵ(q), flattened to channels.
    Mv = reshape(reshape(A_step, 1, 1) .+ reshape(g, 1, 1) .* W_hat, C)   # (C,)

    # Geometric kernel K[c,m] = M_c^{m-1} = exp((m-1)·log M_c). Built with exp∘log
    # (as `phasor_kernel` builds exp(k·Δt·n)) so it is AD-differentiable —
    # cumprod's `dims` adjoint is unsupported — and branch-cut-safe here because
    # the exponents are integers: both the forward value and its M-gradient are
    # branch-independent away from the measure-zero negative-real axis.
    ns = ignore_derivatives() do
        n = similar(W_hat, Float32, 1, L); n .= reshape(Float32.(0:L-1), 1, L); return n
    end
    K = exp.(reshape(log.(Mv), C, 1) .* ns)                              # (C,L): M^0..M^{L-1}

    # Homogeneous term M_c^n·ẑ0, n=1..L. (M .* K)[c,n] = M_c^n.
    ẑ0 = reshape(fft(z0, (1, 2)), C, B)                                   # (C,B)
    Mpow_n = reshape(Mv, C, 1) .* K                                       # (C,L) = M^1..M^L
    homog = reshape(Mpow_n, C, L, 1) .* reshape(ẑ0, C, 1, B)             # (C,L,B)

    Ŷ = if drive === nothing
        homog
    else
        d̂ = reshape(fft(ComplexF32.(drive), (1, 2)), C, L, B)           # (C,L,B)
        homog .+ causal_conv(K, d̂)                                       # (C,L,B)
    end

    return ifft(reshape(Ŷ, H, W, L, B), (1, 2))                          # (H,W,L,B)
end

# ---- Sparse experts: gate + input-conditioned bind (§5.3/§5.4) --------
#
# The input-conditioned expert path (design note §5.4 regime 1, §5.8 #2/#3/#5):
# a sparsely-gated router picks experts from the *input*, each expert stamps a
# learned bind phasor onto the injected drive inside its patch, and the resulting
# precomputed drive feeds `_wave_rollout_scan` unchanged — so the whole augmented
# sheet stays one parallel scan. State-conditioned experts (reading the live
# wave) need the event-split of §5.4 regime 2 and are not implemented here.

"""
    moe_gate(logits, bias; hard=true, τ=1f0) -> gate

Sparsely-gated top-1 router (§5.4/§5.8 of `docs/wavesheet_experts_design.md`).

# Arguments
- `logits :: (E, …)` — routing scores over `E` experts along dim 1.
- `bias :: (E,)` — the DeepSeek *loss-free* per-expert bias. It steers only the
  **selection** (`argmax` over `logits .+ bias`), never the gate-value gradient
  (which is `softmax` of the **unbiased** logits) — so balancing the load does
  not inject interference gradients.

# Keywords
- `hard=true` — straight-through one-hot gate: hard `0/1` in the forward pass,
  `softmax` gradient in the backward pass. `hard=false` returns plain `softmax`
  weights (all experts active).
- `τ` — softmax temperature.

# Returns
Gate weights the same shape as `logits`.

# Implementation
Straight-through estimator `gate = sel + (soft − stop(soft))`: forward value is
the hard selection `sel`, gradient flows through `soft`. Load balancing follows
DeepSeek's auxiliary-loss-free scheme — update `bias` outside the graph via
[`update_moe_bias`](@ref).
"""
function moe_gate(logits::AbstractArray, bias::AbstractVector;
                  hard::Bool = true, τ::Real = 1f0)
    E = size(logits, 1)
    soft = softmax(logits ./ Float32(τ); dims = 1)              # (E,…) differentiable
    hard || return soft
    bshape = reshape(Float32.(bias), E, ntuple(_ -> 1, ndims(logits) - 1)...)
    sel = ignore_derivatives() do
        biased = logits .+ bshape
        mx = maximum(biased; dims = 1)
        oh = Float32.(biased .>= mx)                            # one-hot (argmax; ties→shared)
        oh ./ sum(oh; dims = 1)                                 # normalise rare ties
    end
    return sel .+ (soft .- ignore_derivatives(soft))           # STE: fwd=sel, bwd=∂soft
end

"""
    update_moe_bias(bias, gate; target=nothing, rate=1f-3) -> new_bias

DeepSeek auxiliary-loss-free load-balancing update (call **outside** the gradient
tape). Increases the bias of under-used experts and decreases that of over-used
ones by `rate · sign(target − load)`, where `load[e]` is expert `e`'s mean gate
occupancy over the batch and `target` defaults to uniform `1/E`.
"""
function update_moe_bias(bias::AbstractVector, gate::AbstractArray;
                         target = nothing, rate::Real = 1f-3)
    E = size(gate, 1)
    load = vec(sum(gate; dims = Tuple(2:ndims(gate)))) ./ prod(size(gate)[2:end])  # (E,)
    tgt = target === nothing ? fill!(similar(load), 1f0 / E) : Float32.(target)
    return bias .+ Float32(rate) .* sign.(tgt .- load)
end

# Modulate the input drive with input-conditioned experts. Where expert `e`
# fires (gate `g_e`), it rotates the drive by its unit bind phasor `φ_e` inside
# its patch mask: factor = 1 + Σ_e mask_e·g_e·(φ_e − 1), drive′ = drive ⊙ factor
# (so g_e=0 ⇒ unchanged, g_e=1 inside patch ⇒ ×φ_e — a true phasor bind). The
# result is a precomputed drive for `_wave_rollout_scan`.
#   drive (H,W,L,B) · masks (H,W,E) real · phis (E,) complex · gate (E,L,B) real
function _apply_wave_experts(drive::AbstractArray, masks::AbstractMatrix,
                             phis::AbstractVector, gate::AbstractArray)
    H, W, L, B = size(drive)
    E = length(phis)
    dphi = reshape(ComplexF32.(phis) .- 1f0, 1, E)                        # (1,E)
    md = reshape(ComplexF32.(masks), H * W, E) .* dphi                    # (HW,E)
    gLB = reshape(ComplexF32.(gate), E, L * B)                            # (E,L*B)
    contrib = reshape(md * gLB, H, W, L, B)                               # (H,W,L,B)
    return drive .* (1f0 .+ contrib)
end

# ---- State-conditioned experts: event-based operator split (§5.4 r.2) --
#
# For experts that read the *live propagating wave* (not the precomputed drive),
# the gate/bind depends on the state, so a single global scan no longer applies.
# We operator-split (design note §5.4 regime 2.1, §5.8 #4): partition the L cycles
# into chunks, run the linear sheet as one parallel scan (`_wave_rollout_scan`)
# *within* each chunk, and at each chunk boundary read the evolved state and apply
# a state-conditioned event. Sequential cost = number of chunks, so sparse/rare
# events stay cheap; with an identity event the chunked rollout equals the full
# scan exactly (linear transport composes across boundaries). No ODE adjoint.

# Masked pooled complex read of the state at each expert's patch: r_e = Σ_site
# mask_e·z / Σ_site mask_e. `z (H,W,B)` · `masks (HW,E)` real → `(E,B)` complex.
function _state_read(z::AbstractArray, masks::AbstractMatrix)
    H, W, B = size(z)
    zf = reshape(ComplexF32.(z), H * W, B)                                # (HW,B)
    num = transpose(ComplexF32.(masks)) * zf                              # (E,B)
    den = ignore_derivatives() do
        reshape(sum(masks; dims = 1), :, 1) .+ 1f-8                       # (E,1)
    end
    return num ./ den
end

"""
    _wave_rollout_chunked(l, ps, st, z0, drive, L; chunk, interact) -> (H,W,L,B)

Event-based operator split for state-conditioned experts. `interact(z, c)` maps
the chunk-`c` entry state `(H,W,B)` to a post-event state — e.g. a
state-conditioned gate+bind — and is applied at chunk boundaries `c ≥ 2` (the
first chunk propagates the initial state as-is, since "modify the passing wave"
needs a wave first). Within each chunk the linear sheet runs as one parallel
scan. `interact = (z, c) -> z` recovers the plain scan exactly.
"""
function _wave_rollout_chunked(l::PhasorWaveSheet, ps, st, z0, drive, L::Int;
                               chunk::Int, interact)
    H, W, B = size(z0)
    Y = Buffer(similar(z0, ComplexF32, H, W, L, B))
    z = ComplexF32.(z0)
    pos = 1
    c = 1
    while pos <= L
        ℓ = min(chunk, L - pos + 1)
        z = c > 1 ? interact(z, c) : z                                    # boundary event
        dchunk = drive === nothing ? nothing : drive[:, :, pos:(pos + ℓ - 1), :]
        Ychunk = _wave_rollout_scan(l, ps, st, z, dchunk, ℓ)             # (H,W,ℓ,B) parallel
        Y[:, :, pos:(pos + ℓ - 1), :] = Ychunk
        z = Ychunk[:, :, ℓ, :]                                            # carry to next chunk
        pos += ℓ
        c += 1
    end
    return copy(Y)
end

# ---- Spike mode as a phase-domain SSM: DEQ fixed point (§5.6) ----------
#
# In `:spike` mode each neuron emits a fixed-magnitude spike `s[n] = emit(z[n])`
# and the coupling acts on that spike, so the state evolves *linearly given the
# spike train*:  z[n] = Aⁿ·z0 + Σ_{j≤n} A^{n-j}·(g·(W ⊛ s[j-1]) + drive[j]).
# Transport is the scalar `A` (coupling is on the emitted spike, not the state),
# so — given `s` — the whole trajectory is a parallel `causal_conv` in time. The
# only nonlinearity is re-emitting, making `:spike` a fixed point
#   s* = emit( linear_response(s*, drive) )
# solved by iterating parallel sweeps with a cheap pointwise `emit` between them
# (design note §5.6b). The map is strictly causal, so `n_sweeps == L` reproduces
# the sequential `_wave_rollout` exactly; fewer sweeps give a settling/DEQ
# approximation (differentiate by unrolling, or by the implicit-function theorem).

# Dirac-consistent emission (§5.6a): a spike at phase θ = arg(z)/π arrives at
# t_s = (θ/2+0.5)·T, leaving dt = T·(0.5 − θ/2) to the sample point, so the
# unit-cycle response is exp(k·dt) — exactly `dirac_encode`'s core, reused here so
# the sheet and `PhasorDense` share one phase-SSM emission. Differs from the
# `:unit` emit `z/|z|` only by the sub-cycle leak magnitude `exp(λ·dt)`.
_emit_dirac(z, k, T) =
    exp.(k .* (Float32(T) .* (0.5f0 .- Float32.(complex_to_angle(z)) ./ 2f0)))

"""
    _wave_rollout_deq(l, ps, st, z0, drive, L; n_sweeps, emit_mode) -> (H,W,L,B)

Parallel spike-mode trainer via the DEQ fixed point (§5.6b). `emit_mode` is
`:unit` (`z/|z|`, the sheet default) or `:dirac` (`exp(k·dt)`, `PhasorDense`-
consistent; §5.6a). `n_sweeps == L` reproduces the sequential spike rollout;
fewer sweeps settle toward it.
"""
function _wave_rollout_deq(l::PhasorWaveSheet, ps, st, z0, drive, L::Int;
                           n_sweeps::Int, emit_mode::Symbol = :unit)
    # The DEQ solves for the whole spike train at once, so a threshold that
    # itself depends on the (still-settling) activity is not a fixed point of
    # the same map: the "n_sweeps == L reproduces the sequential rollout
    # exactly" guarantee would silently become an approximation. Refuse rather
    # than return a number that looks exact and is not.
    l.homeostasis === :none ||
        throw(ArgumentError("mode=:deq needs homeostasis=:none — a homeostatic " *
              "threshold is sweep-dependent, so the DEQ fixed point no longer " *
              "reproduces the sequential spike rollout. Use mode=:discrete."))
    ω = period_to_angfreq(l.spk_args.t_period)
    T = Float32(l.spk_args.t_period)
    A_step, g, W_hat = _build_coupling(l, ps, st, ω)
    λ = -exp.(ps.log_neg_lambda)
    kk = ComplexF32(λ[1] + 1im * ω)                                       # scalar eigenvalue
    H, W, B = size(z0)
    C = H * W

    logA = log(ComplexF32(A_step[1]))
    ns = ignore_derivatives() do
        n = similar(W_hat, Float32, 1, L); n .= reshape(Float32.(0:L-1), 1, L); return n
    end
    onesC = ignore_derivatives() do
        o = similar(W_hat, ComplexF32, C, 1); o .= 1f0; return o
    end
    Ak = onesC .* exp.(logA .* ns)                                        # (C,L): A^0..A^{L-1}
    An = exp.(logA .* (ns .+ 1f0))                                        # (1,L): A^1..A^L
    z0r = reshape(ComplexF32.(z0), H, W, 1, B)
    homog = reshape(An, 1, 1, L, 1) .* z0r                               # (H,W,L,B)

    drive_ = drive === nothing ? ignore_derivatives() do
        d = similar(W_hat, ComplexF32, H, W, L, B); d .= 0f0; return d
    end : ComplexF32.(drive)

    θ_deq = exp.(ps.log_theta)                       # fixed (homeostasis :none)
    emit = emit_mode === :dirac ? (zz -> _emit_dirac(zz, kk, T)) :
                                  (zz -> normalize_to_unit_circle(zz; ε = θ_deq .^ 2))

    z = homog
    for _ in 1:n_sweeps
        zshift = cat(z0r, z[:, :, 1:(L - 1), :]; dims = 3)               # z[0..L-1]
        sp = emit(zshift)                                                 # (H,W,L,B)
        coupled = ifft(reshape(W_hat, H, W, 1) .* fft(sp, (1, 2)), (1, 2))
        u = reshape(g, 1, 1, 1) .* coupled .+ drive_                     # (H,W,L,B)
        part = reshape(causal_conv(Ak, reshape(u, C, L, B)), H, W, L, B)
        z = homog .+ part
    end
    return z
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
- `:scan` — the parallel FFT-in-time forward for the **linear** sheet
  (`transmit=:potential`, no saturation/adaptation). Mathematically identical
  to `:discrete` in that regime (`ẑ_q[n] = M_q^n·ẑ0_q + Σ_m M_q^{n-m}·d̂_q[m]`)
  but computed as a homogeneous power term + a causal time-convolution rather
  than a sequential recurrence — no ODE adjoint, time-parallel. Errors on the
  nonlinear regimes. See §5.8 of `docs/wavesheet_experts_design.md`.
- `:deq` — the **spike**-mode fixed-point solver (§5.6b): iterates `n_sweeps`
  parallel sweeps of `s ← emit(linear_response(s))`. `n_sweeps == L` reproduces
  the sequential spike rollout exactly; fewer sweeps settle toward it. `emit` is
  `:unit` (`z/|z|`) or `:dirac` (`exp(k·dt)`, `PhasorDense`-consistent; §5.6a).
- `:ode` — the Tier-2 continuous ODE `dz/dt = k·z + g·(coupling)` integrated
  by `spk_args.solver` and sampled at each period. Autonomous only (`drive`
  must be `nothing`); the ground-truth continuous dynamics the discrete
  recurrence approximates. See [`dispersion`](@ref)`(...; mode=:continuous)`.
"""
function wave_simulate(l::PhasorWaveSheet, ps, st;
                       z0::AbstractArray, L::Integer, drive=nothing,
                       mode::Symbol = :discrete,
                       n_sweeps::Integer = L, emit::Symbol = :unit)
    H, W = l.grid_h, l.grid_w
    had_batch = ndims(z0) == 3
    z0_3 = had_batch ? ComplexF32.(z0) : reshape(ComplexF32.(z0), H, W, 1)
    @assert size(z0_3, 1) == H && size(z0_3, 2) == W "z0 must be $(H)×$(W)[×B]"
    if mode === :ode
        drive === nothing || throw(ArgumentError("wave_simulate mode=:ode is autonomous; drive must be nothing"))
        Y = _wave_rollout_ode(l, ps, st, z0_3, Int(L))           # (H,W,L,B)
    elseif mode === :discrete || mode === :scan
        drive_4 = drive === nothing ? nothing :
            (ndims(drive) == 4 ? ComplexF32.(drive) :
             reshape(ComplexF32.(drive), H, W, Int(L), 1))
        Y = mode === :scan ?
            _wave_rollout_scan(l, ps, st, z0_3, drive_4, Int(L)) :   # (H,W,L,B)
            _wave_rollout(l, ps, st, z0_3, drive_4, Int(L))          # (H,W,L,B)
    elseif mode === :deq
        drive_4 = drive === nothing ? nothing :
            (ndims(drive) == 4 ? ComplexF32.(drive) :
             reshape(ComplexF32.(drive), H, W, Int(L), 1))
        Y = _wave_rollout_deq(l, ps, st, z0_3, drive_4, Int(L);
                              n_sweeps = Int(n_sweeps), emit_mode = emit)  # (H,W,L,B)
    else
        throw(ArgumentError("wave_simulate mode must be :discrete, :scan, :deq or :ode, got :$mode"))
    end
    return had_batch ? Y : dropdims(Y; dims=4)
end

"""
    wave_homeostat_trace(l, ps, st; z0, L, keep_fields = false) -> NamedTuple

Run the sheet for `L` steps and return the **homeostatic threshold trajectory**
alongside the activity it is regulating. `θ` is rollout-local (it does not live
in `st`), so without this the mechanism is invisible: a homeostat that has
locked onto the wrong branch looks identical, from the returned `z`, to one that
is working.

With `keep_fields = true` the full `z` and `θ` field trajectories come back too,
as `(H,W,L)` arrays. This is the only way to draw the *firing set* of a `:local`
sheet: there `θ = θ_g · θ_l` varies per site, so thresholding the state against
the scalar `theta_g` is wrong (measured `θ_l` spread reaches 1.75×). Costs
`L·H·W·12` bytes — 200 MB at `256²×250`, so pass it deliberately.

Returns `(; theta_g, fire, mean_abs, std_abs, theta_l_spread)`, each a length-`L`
vector for a single (unbatched) sheet (plus `z`, `theta` when `keep_fields`):

- `theta_g` — the global threshold per step. Should settle; a sawtooth means the
  controller is relaxation-oscillating against the flood boundary rather than
  regulating (see `init_logit_target`).
- `fire` — fraction of sites above threshold. Hitting the target is **necessary,
  not sufficient**: check `std_abs` too, because a uniform sheet sliced at the
  right level also reports the target rate.
- `std_abs` — spatial spread of `|z|`. Collapse toward 0 is the failure mode.
- `theta_l_spread` — `max θ_l / min θ_l`, i.e. how much per-site refractoriness
  has built up (`1.0` when `homeostasis ≠ :local`).

Only meaningful for `transmit = :spike`.
"""
function wave_homeostat_trace(l::PhasorWaveSheet, ps::LuxParams, st::NamedTuple;
                              z0::AbstractArray, L::Integer,
                              keep_fields::Bool = false)
    l.transmit === :spike ||
        throw(ArgumentError("wave_homeostat_trace needs transmit=:spike (no threshold otherwise)"))
    ω = period_to_angfreq(l.spk_args.t_period)
    A_step, g, W_hat = _build_coupling(l, ps, st, ω)
    H, W = l.grid_h, l.grid_w
    z = reshape(ComplexF32.(z0), H, W, 1)
    a = zeros(Float32, H, W, 1)
    ρ_a = l.use_adaptation ? _sigmoid.(ps.log_rho_a) : nothing
    δ_a = l.use_adaptation ? exp.(ps.log_delta_a)    : nothing
    d0  = zeros(ComplexF32, H, W, 1)

    on = l.homeostasis !== :none
    θg = Float32[exp(only(ps.log_theta))]
    θl = l.homeostasis === :local ? ones(Float32, H, W, 1) : nothing
    η_g = on ? exp.(ps.log_eta_g) : nothing
    η_l = l.homeostasis === :local ? exp.(ps.log_eta_l) : nothing
    tgt = on ? _sigmoid.(ps.logit_target) : nothing
    β   = on ? exp.(ps.log_theta_beta) : nothing

    tg = Float32[]; fr = Float32[]; ma = Float32[]; sa = Float32[]; ls = Float32[]
    Zf = keep_fields ? Array{ComplexF32}(undef, H, W, L) : nothing
    Tf = keep_fields ? Array{Float32}(undef, H, W, L)    : nothing
    for t in 1:L
        θnow = θl === nothing ? reshape(θg, 1, 1, :) : reshape(θg, 1, 1, :) .* θl
        z, a, θg, θl = _wave_step(l, z, d0, a, A_step, g, W_hat, ρ_a, δ_a,
                                  l.saturating, l.use_adaptation,
                                  θg, θl, η_g, η_l, tgt, β)
        r = abs.(z)
        push!(tg, only(θg)); push!(fr, mean(r .> θnow))
        push!(ma, mean(r));  push!(sa, std(r))
        push!(ls, θl === nothing ? 1f0 : maximum(θl) / minimum(θl))
        if keep_fields
            Zf[:, :, t] = view(z, :, :, 1)
            # The θ that gated THIS step's emission, broadcast to a full field so
            # callers can threshold without knowing which homeostasis mode ran.
            Tf[:, :, t] = θl === nothing ? fill(only(θnow), H, W) : view(θnow, :, :, 1)
        end
    end
    base = (; theta_g = tg, fire = fr, mean_abs = ma, std_abs = sa,
              theta_l_spread = ls)
    return keep_fields ? merge(base, (; z = Zf, theta = Tf)) : base
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

    dzdt(u, p, t) = kr .* u .+ gr .* ifft(Whr .* fft(_transmit(l, u), (1, 2)), (1, 2))
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

# Spike sheets: the subthreshold effective gain

For `transmit = :spike` the emit is `z/√(|z|²+θ²)`, which for `|z| ≪ θ` is
**linear**: `≈ z/θ`. So below threshold the sheet is exactly the `:potential`
medium at gain `g_eff = g/θ`, and that is what this function reports — the
returned `M`, `spectral_radius`, and `k_eff` all use `g_eff`. Verified: sweeping
`g/(θ·g_crit)` through 1 reproduces the criticality knife edge
(`0.5 → 5.8e-6`, `1.0 → 4.2e-3`, `2.0 → 23.5` peak `|z|` at t=80).

This replaces the `g/r` Kuramoto linearization of §3, whose scale factor
involved an unknown mean-field amplitude `r`, with a computable one — and it is
what makes [`radial_band`](@ref) and [`wave_transport`](@ref) meaningful on a
spike sheet.

!!! warning "Only valid below threshold"
    Once the sheet is firing (`|z| ≳ θ`) the emit saturates and the linearization
    stops applying — the true dynamics are then the amplitude-blind regime of §3.
    A sheet whose `θ` is far below `emission_threshold` never has a subthreshold
    regime to speak of. Compare `exp(ps.log_theta)` against
    [`emission_threshold`](@ref)`(l, ps, st)` before trusting these numbers.
"""
function dispersion(l::PhasorWaveSheet, ps, st; mode::Symbol = :discrete)
    ω = period_to_angfreq(l.spk_args.t_period)
    T = Float32(l.spk_args.t_period)
    A_step, g, W_hat = _build_coupling(l, ps, st, ω)
    # Subthreshold linearization of the spike emit: s ≈ z/θ ⇒ g_eff = g/θ.
    l.transmit === :spike && (g = g ./ exp.(ps.log_theta))
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
        coupled = ifft(reshape(W_hat, H, W, 1) .* fft(_transmit(l, u), (1, 2)), (1, 2))
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

# ======================================================================
# §6 prototype: WaveExpertSheet — read patch → gate → re-bind, discrete SSM
# ======================================================================
#
# The prototype from §6 of docs/wavesheet_experts_design.md. Composes a
# `PhasorWaveSheet` substrate with a bank of sparse experts: each expert owns a
# patch of the sheet, a top-1 router (`moe_gate`, DeepSeek loss-free per-expert
# bias) picks which fire, and each fired expert stamps a learned unit bind phasor
# onto the wave inside its patch. Trained end-to-end on the DISCRETE SSM — no ODE
# adjoint. Two routing modes:
#   :input — gate/bind read the input drive; folds into the parallel scan
#            (`_wave_rollout_scan`) or spike DEQ (`_wave_rollout_deq`). §5.4 r.1.
#   :state — gate/bind read the live wave at chunk boundaries via the event-based
#            operator split (`_wave_rollout_chunked`); needs a :potential sheet. §5.4 r.2.
# `route_stats` returns the gate entropy / per-expert load for the go/no-go probe
# (does the gate collapse to one always-on expert?).
#
# The router summarises each patch with a *phase-domain* feature (`route_feature`):
#   :coherence — |pooled read|: amplitude-weighted phase coherence (legacy default).
#   :dispersion — 1 − order parameter of the unit phasors: phase *diversity*
#                 ("a patch carrying a variety of phases holds more information").
#   :matched   — Re(unbind-by-key read): a VSA "is this my key?" similarity. Each
#                expert carries a learned per-site phase key `bind_key`; the read
#                unbinds the patch by that key and takes the real part, so the
#                expert fires when the patch phase pattern matches its key. This is
#                the read-side dual of the write-side bind `bind_phase` — key/value.
# All three collapse to a single real logit per expert (then mixed by `Wr`), so the
# discrete top-1 route stays compatible with the R&F neuron readout.

struct WaveExpertSheet{S <: PhasorWaveSheet} <: Lux.AbstractLuxLayer
    sheet::S
    n_experts::Int
    routing::Symbol       # :input or :state
    route_feature::Symbol # :coherence | :dispersion | :matched
    chunk::Int            # chunk size for :state routing
    n_sweeps::Int         # spike-DEQ sweeps for :input on a :spike sheet (0 → L)
    tau::Float32          # gate temperature
    hard::Bool            # straight-through hard gate
    balance::Bool         # online DeepSeek bias update in the forward pass
    bias_rate::Float32
end

function WaveExpertSheet(H::Integer, W::Integer; n_experts::Integer = 4,
                         routing::Symbol = :input, route_feature::Symbol = :coherence,
                         chunk::Integer = 4, n_sweeps::Integer = 0, tau = 1f0,
                         hard::Bool = true, balance::Bool = true, bias_rate = 1f-2,
                         sheet_kwargs...)
    routing in (:input, :state) ||
        throw(ArgumentError("routing must be :input or :state, got :$routing"))
    route_feature in (:coherence, :dispersion, :matched) ||
        throw(ArgumentError("route_feature must be :coherence, :dispersion or " *
                            ":matched, got :$route_feature"))
    n_experts <= H ||
        throw(ArgumentError("n_experts ($(n_experts)) must be ≤ grid height ($(H)) " *
                            "for the row-band patch tiling"))
    sheet = PhasorWaveSheet(Int(H), Int(W); sheet_kwargs...)
    routing === :state && sheet.transmit !== :potential &&
        throw(ArgumentError("state routing needs a :potential (linear) substrate " *
                            "for the chunked within-chunk scan"))
    return WaveExpertSheet(sheet, Int(n_experts), routing, route_feature, Int(chunk),
                           Int(n_sweeps), Float32(tau), hard, balance, Float32(bias_rate))
end

# Spatially-local patches: split the grid into E horizontal row-bands.
function _tile_masks(H::Int, W::Int, E::Int)
    masks = zeros(Float32, H, W, E)
    for i in 1:H, j in 1:W
        e = min(E, 1 + ((i - 1) * E) ÷ H)
        masks[i, j, e] = 1f0
    end
    return reshape(masks, H * W, E)
end

function Lux.initialparameters(rng::AbstractRNG, l::WaveExpertSheet)
    sheet_ps = Lux.initialparameters(rng, l.sheet)
    E = l.n_experts
    Wr = glorot_uniform(rng, E, E)                        # content router
    bind_phase = 0.1f0 .* randn(rng, Float32, E)          # start near φ=1, slight spread
    base = (; sheet = sheet_ps, Wr = Wr, bind_phase = bind_phase)
    if l.route_feature === :matched
        # Per-site phase key ψ_{e,x} (units of π). Random per expert so the experts
        # start with distinct preferred phase patterns (symmetry-breaking); only the
        # sites inside each expert's patch mask are ever consulted.
        HW = l.sheet.grid_h * l.sheet.grid_w
        bind_key = 2f0 .* rand(rng, Float32, HW, E) .- 1f0
        return merge(base, (; bind_key = bind_key))
    end
    return base
end

function Lux.initialstates(rng::AbstractRNG, l::WaveExpertSheet)
    sheet_st = Lux.initialstates(rng, l.sheet)
    masks = _tile_masks(l.sheet.grid_h, l.sheet.grid_w, l.n_experts)
    route_bias = zeros(Float32, l.n_experts)             # DeepSeek loss-free bias
    return (; sheet = sheet_st, masks = masks, route_bias = route_bias)
end

# Masked pooled complex read at each patch: field (H,W,rest...) → (E, rest...).
function _patch_read(field::AbstractArray, masks::AbstractMatrix)
    H, W = size(field, 1), size(field, 2)
    rest = size(field)[3:end]
    ff = reshape(ComplexF32.(field), H * W, :)                        # (HW, prod(rest))
    num = transpose(ComplexF32.(masks)) * ff                          # (E, prod(rest))
    den = ignore_derivatives() do
        reshape(sum(masks; dims = 1), :, 1) .+ 1f-8                   # (E,1)
    end
    return reshape(num ./ den, size(masks, 2), rest...)
end

# Router logits from patch reads: content feature |read| mixed by Wr (E,E).
function _router_logits(Wr::AbstractMatrix, reads::AbstractArray)
    E = size(Wr, 1)
    rest = size(reads)[2:end]
    feat = abs.(reshape(reads, E, :))                                 # (E, prod(rest))
    return reshape(Wr * feat, E, rest...)
end

# Keyed mask-pooled complex read: ρ_e = Σ_x mask_{e,x}·e^{-iπ·key_{e,x}}·field_x / Σ_x mask.
# `key_phase === nothing` reduces to the plain pooled read (`_patch_read`). The read
# is amplitude-weighted (raw field, not unit-normalised): a good phase match is
# usually also a strong-signal site, so amplitude acts as confidence. AD-safe.
function _patch_read_keyed(field::AbstractArray, masks::AbstractMatrix,
                           key_phase::Union{Nothing,AbstractMatrix})
    H, W = size(field, 1), size(field, 2)
    rest = size(field)[3:end]
    ff = reshape(ComplexF32.(field), H * W, :)                        # (HW, prod(rest))
    unbind = key_phase === nothing ? ComplexF32.(masks) :
             ComplexF32.(masks) .* cis.(-pi_f32 .* key_phase)         # (HW,E) mask·conj(key)
    num = transpose(unbind) * ff                                      # (E, prod(rest))
    den = ignore_derivatives() do
        reshape(sum(masks; dims = 1), :, 1) .+ 1f-8                   # (E,1)
    end
    return reshape(num ./ den, size(masks, 2), rest...)               # (E, rest...)
end

# Phase-domain routing feature → logits (mixed by Wr). See the WaveExpertSheet
# header for the three `route_feature` modes. `field` is (H,W,rest...) complex.
function _route_logits(l::WaveExpertSheet, ps::LuxParams, st::NamedTuple,
                       field::AbstractArray)
    E = l.n_experts
    feat = if l.route_feature === :matched
        real.(_patch_read_keyed(field, st.masks, ps.bind_key))        # (E,rest) "is this my key?"
    elseif l.route_feature === :dispersion
        u = field ./ (abs.(field) .+ 1f-6)                            # phase-only unit phasors
        1f0 .- abs.(_patch_read_keyed(u, st.masks, nothing))         # 1 − order parameter ∈[0,1]
    else  # :coherence (legacy): |pooled read|
        abs.(_patch_read_keyed(field, st.masks, nothing))
    end
    rest = size(feat)[2:end]
    return reshape(ps.Wr * reshape(feat, E, :), E, rest...)           # (E, rest...)
end

function (l::WaveExpertSheet)(x::AbstractArray{<:Phase, 3}, ps::LuxParams, st::NamedTuple)
    H, W = l.sheet.grid_h, l.sheet.grid_w
    N, L, B = size(x)
    @assert N == H * W "input channels $(N) ≠ grid $(H)×$(W)"
    E = l.n_experts
    drive = reshape(angle_to_complex(x), H, W, L, B)
    phis = cis.(pi_f32 .* ps.bind_phase)                              # (E,) unit bind phasors
    z0 = ignore_derivatives() do
        z = similar(st.masks, ComplexF32, H, W, B); z .= 0f0; return z
    end

    local Y, gate_input
    if l.routing === :input
        gate = moe_gate(_route_logits(l, ps, st, drive), st.route_bias;
                        hard = l.hard, τ = l.tau)                    # (E,L,B)
        drive2 = _apply_wave_experts(drive, st.masks, phis, gate)
        nsw = l.n_sweeps > 0 ? l.n_sweeps : L
        Y = l.sheet.transmit === :spike ?
            _wave_rollout_deq(l.sheet, ps.sheet, st.sheet, z0, drive2, L;
                              n_sweeps = nsw, emit_mode = :unit) :
            _wave_rollout_scan(l.sheet, ps.sheet, st.sheet, z0, drive2, L)
        gate_input = gate
    else  # :state — gate/bind read the live wave at chunk boundaries
        interact = (z, c) -> begin
            g = moe_gate(_route_logits(l, ps, st, z),
                         st.route_bias; hard = l.hard, τ = l.tau)    # (E,B)
            zb = _apply_wave_experts(reshape(z, H, W, 1, B), st.masks, phis,
                                     reshape(g, E, 1, B))
            reshape(zb, H, W, B)
        end
        Y = _wave_rollout_chunked(l.sheet, ps.sheet, st.sheet, z0, drive, L;
                                  chunk = l.chunk, interact = interact)
        gate_input = nothing                                          # per-boundary; see route_stats
    end

    # Online DeepSeek loss-free load balancing (input routing only; out of graph).
    new_bias = (l.balance && gate_input !== nothing) ?
        ignore_derivatives() do
            update_moe_bias(st.route_bias, gate_input; rate = l.bias_rate)
        end : st.route_bias
    st_out = merge(st, (route_bias = new_bias,))
    return complex_to_angle(reshape(Y, N, L, B)), st_out
end

"""
    route_stats(l::WaveExpertSheet, ps, st, x) -> (; load, entropy, gate)

Go/no-go probe for §6: `load` is the per-expert soft occupancy (`(E,)`, sums to 1)
and `entropy` is the mean routing entropy (→ `0` if the gate collapses to one
always-on expert, → `log(E)` if balanced). Computed from the input-routing gate.
"""
function route_stats(l::WaveExpertSheet, ps::LuxParams, st::NamedTuple,
                     x::AbstractArray{<:Phase, 3})
    H, W = l.sheet.grid_h, l.sheet.grid_w
    N, L, B = size(x)
    drive = reshape(angle_to_complex(x), H, W, L, B)
    logits = _route_logits(l, ps, st, drive)                         # (E,L,B)
    p = softmax(logits; dims = 1)
    load = vec(sum(p; dims = (2, 3))) ./ Float32(L * B)              # (E,)
    ent = -sum(p .* log.(p .+ 1f-12); dims = 1)                      # (1,L,B)
    gate = moe_gate(logits, st.route_bias; hard = l.hard, τ = l.tau)
    return (; load = load, entropy = mean(ent), gate = gate)
end

# =======================================================================
# SolitonWaveSheet — conservative (unitary) nonlinear wave sheet
# =======================================================================
#
# Backwards-design layer (demos/wave_soliton_c1.jl screen). The goal is
# SOLITONIC transport: a wave packet that carries a phase code over long
# distance WITHOUT spreading. The C-1 screen showed two things:
#
#   (i)  the linear PhasorWaveSheet map M(q)=A+g·Ŵ(q) is DISSIPATIVE and
#        modal-filtering — bolting an amplitude-dependent ω onto it just
#        bleeds power into lossy modes (no self-trapping);
#   (ii) once the linear step is made CONSERVATIVE (unit-modulus per mode),
#        the same amplitude-dependent ω (Kerr / nonlinear-Schrödinger self-
#        phase-modulation) DOES arrest dispersion — a soliton balance.
#
# So this layer realizes the medium the analysis prescribed:
#
#   linear step (Fourier):   ẑ(q) ← e^{iθ(q)}·ẑ(q)          (|·|=1, UNITARY)
#     θ(q) = ωT + D·(cos q_h + cos q_w) − (s_h q_h + s_w q_w)
#            └carrier┘ └─ tight-binding dispersion ─┘ └── directed drift ──┘
#   Kerr step (pointwise):   z ← z·exp(i·β·|z|²)             (|·|=1, UNITARY)
#
# Both steps preserve |z| pointwise / by Parseval ⇒ the whole recurrence is
# NORM-CONSERVING (an isometry) ⇒ UNCONDITIONALLY STABLE: no spectral-radius
# tuning, no leak, no runaway. This is the key structural difference from
# PhasorWaveSheet (which needs ρ<1). The conservative linear step is the
# Hamiltonian (λ→0) limit of the R&F sheet with a discrete-Laplacian
# (nearest-neighbour) imaginary coupling; D = exp(log_D) is the hopping
# strength (dispersion curvature θ''(0) = −D), and β>0 focuses to balance it.
#
# Trainable params: log_D (dispersion), beta (Kerr), drift_h, drift_w.
# Everything else (carrier ω, FFT frequency grids) is fixed state.

struct SolitonWaveSheet <: Lux.AbstractLuxLayer
    grid_h::Int
    grid_w::Int
    dispersion::Symbol         # :tight_binding — θ(q)=D(cos q_h+cos q_w)
    init_log_D::Float32        # dispersion curvature D=exp(log_D)   (trainable)
    init_beta::Float32         # Kerr self-phase-modulation strength (trainable)
    init_drift_h::Float32      # directed drift (sites/step) along axis-1 (rows)
    init_drift_w::Float32      # directed drift along axis-2 (cols)
    spk_args::SpikingArgs
end

function SolitonWaveSheet(H::Integer, W::Integer;
                          dispersion::Symbol = :tight_binding,
                          init_log_D::Real = log(1.0),
                          init_beta::Real  = 0.0,
                          init_drift_h::Real = 0.0,
                          init_drift_w::Real = 0.0,
                          spk_args::SpikingArgs = SpikingArgs())
    dispersion === :tight_binding ||
        throw(ArgumentError("dispersion must be :tight_binding, got :$dispersion"))
    return SolitonWaveSheet(Int(H), Int(W), dispersion,
                            Float32(init_log_D), Float32(init_beta),
                            Float32(init_drift_h), Float32(init_drift_w), spk_args)
end

function Base.show(io::IO, l::SolitonWaveSheet)
    print(io, "SolitonWaveSheet($(l.grid_h)×$(l.grid_w); dispersion=:$(l.dispersion), ",
          "D=$(round(exp(l.init_log_D),digits=3)), β=$(round(l.init_beta,digits=3)), ",
          "drift=($(l.init_drift_h),$(l.init_drift_w)))")
end

function Lux.initialparameters(::AbstractRNG, l::SolitonWaveSheet)
    return (log_D    = Float32[l.init_log_D],
            beta     = Float32[l.init_beta],
            drift_h  = Float32[l.init_drift_h],
            drift_w  = Float32[l.init_drift_w])
end

function Lux.initialstates(::AbstractRNG, l::SolitonWaveSheet)
    H, W = l.grid_h, l.grid_w
    qh = reshape(Float32.(2f0 .* Float32(pi) .* (0:H-1) ./ H), H, 1)   # (H,1) FFT freqs
    qw = reshape(Float32.(2f0 .* Float32(pi) .* (0:W-1) ./ W), 1, W)   # (1,W)
    return (qh = qh, qw = qw, cos_qh = cos.(qh), cos_qw = cos.(qw))
end

# Unit-modulus per-mode propagator e^{iθ(q)} — the conservative linear step.
function _soliton_prop(l::SolitonWaveSheet, ps, st)
    ω = period_to_angfreq(l.spk_args.t_period)
    T = Float32(l.spk_args.t_period)
    D = exp.(ps.log_D)                                                 # (1,)
    θ = (ω * T) .+ D .* (st.cos_qh .+ st.cos_qw) .-
        (ps.drift_h .* st.qh .+ ps.drift_w .* st.qw)                   # (H,W)
    return cis.(θ)
end

# Autonomous (or driven) rollout for L steps from z0 (H,W,B) → (H,W,L,B).
# AD-safe (Buffer, no mutation of tracked arrays). Norm-conserving when
# drive === nothing.
function _soliton_rollout(l::SolitonWaveSheet, ps, st, z0, drive, L::Int)
    prop = _soliton_prop(l, ps, st)                                    # (H,W)
    β = ps.beta[1]
    H, W, B = size(z0)
    P = reshape(prop, H, W, 1)
    Y = Buffer(similar(z0, ComplexF32, H, W, L, B))
    z = ComplexF32.(z0)
    for t in 1:L
        z_lin = ifft(P .* fft(z, (1, 2)), (1, 2))                     # unitary propagate
        if drive !== nothing
            z_lin = z_lin .+ drive[:, :, t, :]
        end
        z = z_lin .* cis.(β .* abs2.(z_lin))                          # Kerr (norm-preserving)
        Y[:, :, t, :] = z
    end
    return copy(Y)                                                     # (H,W,L,B)
end

"""
    soliton_simulate(l::SolitonWaveSheet, ps, st; z0, L, drive=nothing) -> Array{ComplexF32}

Evolve the conservative nonlinear sheet for `L` steps from the initial complex
field `z0`, returning the full trajectory. Autonomous when `drive === nothing`
(then `‖z‖` is conserved to machine precision — the recurrence is unitary).

`z0` may be `(H,W)` or `(H,W,B)`; the return is `(H,W,L)` or `(H,W,L,B)`.
`drive`, if given, is `(H,W,L)` / `(H,W,L,B)`. This is the pure-forward
interface for the soliton / VSA-transport demos: seed a localized packet whose
per-site phases carry a code, and watch it travel coherently.
"""
function soliton_simulate(l::SolitonWaveSheet, ps::LuxParams, st::NamedTuple;
                          z0::AbstractArray, L::Integer, drive = nothing)
    H, W = l.grid_h, l.grid_w
    had_batch = ndims(z0) == 3
    z0_3 = had_batch ? ComplexF32.(z0) : reshape(ComplexF32.(z0), H, W, 1)
    @assert size(z0_3, 1) == H && size(z0_3, 2) == W "z0 must be $(H)×$(W)[×B]"
    drive_4 = drive === nothing ? nothing :
        (ndims(drive) == 4 ? ComplexF32.(drive) :
         reshape(ComplexF32.(drive), H, W, Int(L), 1))
    Y = _soliton_rollout(l, ps, st, z0_3, drive_4, Int(L))
    return had_batch ? Y : dropdims(Y; dims = 4)
end

# Lux forward on Phase 3D input (N=H*W, L, B): inject the encoded input as a
# per-step complex drive from a zero initial field (mirrors PhasorWaveSheet),
# propagate, and decode back to phase. For pure autonomous soliton transport
# use `soliton_simulate` directly.
function (l::SolitonWaveSheet)(x::AbstractArray{<:Phase, 3},
                               ps::LuxParams, st::NamedTuple)
    H, W = l.grid_h, l.grid_w
    N, L, B = size(x)
    @assert N == H * W "input channels $(N) ≠ grid $(H)×$(W) = $(H*W)"
    drive = reshape(angle_to_complex(x), H, W, L, B)
    z0 = ignore_derivatives() do
        z = similar(st.qh, ComplexF32, H, W, B); z .= 0f0; return z
    end
    Y = _soliton_rollout(l, ps, st, z0, drive, L)
    return complex_to_angle(reshape(Y, H * W, L, B)), st
end
