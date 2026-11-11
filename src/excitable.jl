# ================================================================
# ExcitableWaveSheet — a genuinely excitable fork of PhasorWaveSheet
# ================================================================
#
# WHY THIS EXISTS. `PhasorWaveSheet(transmit=:spike)` is *not* an excitable
# medium, and cannot be made one by tuning. Measured 2026-08-13 at N=128,
# default λ/g: suprathreshold activity decays to exactly zero within ~30 steps
# from ANY seed — a single site, or a 3209-site disc covering 20% of the sheet,
# at amplitude 20θ, with `homeostasis` ∈ {:none, :global, :local}. There is no
# critical nucleus because there is nothing to nucleate.
#
# The reason is the emit function itself:
#
#     |emit| = |z| / √(|z|² + θ²)          bounded by 1, and 0.998 already at |z| = 20θ
#     gain   = |emit|/|z| = 1/√(|z|²+θ²)   MAXIMAL AT REST, decreasing as the site activates
#
# That is a *limiter*, not an amplifier. Crossing threshold opens no new energy
# source — it closes the one you had. A chain reaction is impossible. This is not
# an accident: `:spike` was chosen precisely because it hard-bounds the drive
# (`|Σ W·s| ≤ Σ|W|`) and so gives BIBO stability for free. Self-limiting and
# excitable are opposite requirements.
#
# The numbers, at the `:dog` defaults (see `excitable_regime`):
#
#     θ                    = 10.581      (frac 1.4)
#     g·Σ|W|               = 12.598      coherent upper bound on ignition drive
#       ratio to θ         =  1.191      > 1, so geometry ALLOWS propagation
#     actual max drive     =  7.738      with 3209 sites firing at |emit| = 0.998
#       fraction of bound  =  0.614      phase-coherence loss (inhibitory lobe +
#                                        distance-dependent delay phases e^{-iωr/c})
#       fraction of θ      =  0.731      ==> short by 27%
#
# Geometry leaves 19% of headroom; coherence loss eats it and 27% more. And the
# two boundaries cannot be separated by tuning: propagation needs frac < 1.024,
# quiescence needs frac ≳ 1.25. Nor by kernel design — for a purely excitatory
# delay-free kernel Σ|W| = Ŵ(0) = max|Ŵ|, so the bound gets *worse* (frac < 1.0).
#
# WHAT THIS LAYER ADDS. The two ingredients an excitable medium needs and the
# base sheet lacks, and nothing else — the kernel, the FFT-diagonal coupling and
# the carrier convention are all inherited from a wrapped `PhasorWaveSheet`:
#
#   (a) REGENERATIVE CURRENT — a threshold-localised boost on the emitted spike
#
#           emit = z/√(|z|²+θe²) · (1 + α·σ((|z| − θe)/(β·θe)))
#
#       Below θe the boost factor is ≈1 and the sheet is EXACTLY the old linear
#       subthreshold medium at gain g/θ; above it, the site emits (1+α) rather
#       than 1. This is the whole trick: because the boost is *threshold-
#       localised*, it raises the suprathreshold drive without touching ρ_sub.
#       That decouples the propagation boundary from the flood boundary, which
#       is what no amount of tuning the base sheet could do. Verified: firing
#       from a noise seed stays at 0 for every α tested, at every frac ≥ 1.8.
#
#   (b) FAST REFRACTORY STATE — a per-site variable that elevates the local
#       threshold after firing, giving the two-variable (fast activator / slow
#       inhibitor) structure every excitable model has:
#
#           u[t]  = ρ_u·u[t−1] + (1−ρ_u)·1[|z| > θe]
#           θe[x] = θ·(1 + κ_r·u[x])
#
#       Without it, (a) alone gives a front that propagates but never recovers:
#       measured interior fill 0.75 → 0.92, a filling disc rather than a pulse.
#       With it, interior fill drops to 0.002 and the firing count scales with
#       ring *circumference* instead of area.
#
#       NOTE this is NOT the existing `use_adaptation` term. That one is driven
#       by |z| (not by firing) and enters as an imaginary component −i·a (a phase
#       shift). This is fire-driven and enters as a threshold elevation.
#
# α = 0 and κ_r = 0 reproduce `PhasorWaveSheet` bit-for-bit — verified against
# `wave_simulate` at 4.9e-7 relative (float32 roundoff). The defaults here do NOT
# reproduce it; they are the excitable operating point below.
#
# CALIBRATION (N=160, `:dog`, λ = 0.15, g = 1, frac = 2.6 ⇒ ρ_sub = 0.953,
# ρ_u = 0.98, seed disc R=6 at 20θ). "amp-inv" is the spread of the front speed
# over a 167× stimulus range (3θ → 500θ) — the defining excitability test.
# "fill" is the fraction of the pulse interior still firing at t = 30/50/70:
# 0 is a clean annulus, 1 is a filling disc.
#
#     α    κ_r   rise   speed  amp-inv  fill @30/50/70   torus annihilation
#     4.5  0.75  0.02   0.79     1%     0.98/1.00/1.00   floods
#     4.5  0.75  0.08   0.93     2%     0.50/0.32/0.35   floods
#     4.5  1.50  0.02   0.86     6%     0.77/0.50/0.02   RETURNS TO REST
#     4.5  1.50  0.08     —    323%     0.00/0.00/0.00   front collapses inward
#     6.0  0.75  0.02   0.82     1%     1.00/1.00/1.00   floods
#     6.0  1.50  0.02   0.82     2%     1.00/1.00/1.00   floods
#     6.0  1.50  0.08   1.13     1%     0.00/0.00/0.00   RETURNS TO REST  <-- DEFAULT
#     6.0  1.50  0.20     —    291%     0.06/0.00/0.00   front collapses inward
#
# The default is the only row that passes all four criteria at once. Note how
# little slack surrounds it: rise 0.02 floods, rise 0.20 collapses, and κ_r 0.75
# floods at every rise. Retune one knob at a time and re-run all four checks —
# amplitude invariance alone is a misleading score, since the 1% rows include
# both the default and two configurations that flood.
#
# THE KNOBS:
#   - α has a hard floor (nothing propagates below ≈3.5) and, above it, the
#     speed is nearly flat in α — which is exactly what makes the speed a
#     *medium* property rather than a stimulus one.
#   - κ_r < 1.5 floods on collision at every rise tested. Refractory depth is
#     what suppresses the re-entry that turns a collision into turbulence.
#   - rise is the refractory ON rate and is the knob that resolved the
#     invariance-versus-annihilation trade-off; see the `_excitable_step`
#     comment for why it has to be separate from ρ_u and why neither extreme
#     works.
#   - ρ_u sets the refractory wavelength ≈ speed/(1−ρ_u). At ρ_u = 0.8 that is
#     ~5 sites, short enough for re-entry, and amplitude invariance degrades to
#     34%. 0.98 gives ~57 sites. Longer refractory ⇒ better invariance.
#
# WHY THE DEFAULTS ARE NOT THE PROTOTYPE'S. An earlier prototype drove `u` from
# the *pre-update* state and calibrated to α=3.0, κ_r=0.75. The shipped step
# drives it from `z_next` (one step earlier in the feedback, matching the base
# sheet's homeostat convention — "evaluated on the NEW state so the threshold
# responds to the activity it will gate next step"). That one-step shift in the
# inhibitor is enough to move the window: the prototype's point scores 46%
# invariance and a slower front here. If you port numbers from any earlier
# notes, re-measure them.
#
# Memory: state is (H,W,B) complex + (H,W,B) real for u — 1.5× the base sheet.

"""
    ExcitableWaveSheet(H, W; kwargs...)

An excitable fork of [`PhasorWaveSheet`](@ref): a quiescent medium that
propagates a self-sustaining annular pulse whose speed is a property of the
medium rather than of the stimulus.

Wraps a `PhasorWaveSheet` (which supplies the grid, the DoG kernel, the carrier
convention and all of the FFT-diagonal machinery) and adds a regenerative
emission current and a fast per-site refractory variable. See the file header
for why the base sheet cannot do this and for the full calibration table.

Unlike the base sheet, this one satisfies all four excitability criteria
(defaults): quiescent (0 sites firing over 150 steps from a noise seed); has a
critical nucleus (a single-site seed extinguishes, a larger one propagates);
front speed independent of stimulus amplitude (**1% spread over a 167× range**,
3θ → 500θ, measured 1.12/1.13/1.13/1.13); and colliding fronts annihilate,
returning the sheet to complete rest.

The speed does still depend on seed *size* near the nucleus, but that is the
ordinary curvature correction of an excitable front (`v = v_plane − D·κ`) — a
function of the front's own radius rather than of the stimulus, converging as
the front flattens. Measure after the front has passed ~20 sites if you want the
plane-wave value.

# Keyword arguments
- `theta_frac = 2.6` — emission threshold as a multiple of `g·max|Ŵ|`, passed
  through to the base sheet. **Much higher than the base default of 1.4**, and
  deliberately: the excitable regime needs `ρ_sub < 1` (a genuinely stable rest
  state), which does not happen until `frac ≳ 1.85`. At 1.8 the sheet still
  floods from its own subthreshold tail over ~360 steps.
- `alpha = 6.0` — regenerative strength `α`. The emitted spike is `(1+α)` rather
  than unit magnitude. `α = 0` recovers the base sheet exactly. There is a hard
  floor below which nothing propagates at any other setting; above the ceiling
  the sheet floods on collision. See the header table before changing it.
- `kappa_r = 1.5` — refractory depth: a fully-refractory site has its threshold
  raised to `θ(1+κ_r)`. Too small and the pulse fills in behind (interior fill
  0.19 at 0.5) and collisions degenerate into turbulence; too large and it
  extinguishes.
- `rho_u = 0.99` — refractory recovery (OFF) rate per step; the recovery time is
  `1/(1−ρ_u)` and the refractory wavelength `≈ speed/(1−ρ_u)`, here ≈122 sites.
  **Keep the wavelength above the sheet's half-width.** At `ρ_u = 0.98` (λ ≈ 61)
  a pulse on a 192² torus does not annihilate cleanly on self-collision: the
  fragments have room to seed re-entrant spirals and the sheet settles into
  sustained turbulence. 0.99 fixes exactly that, at both 192² and 256².
- `rise = 0.08` — refractory ON rate, deliberately decoupled from `rho_u`:
  `u ← min(1, ρ_u·u + rise·fire)`. Real refractoriness is fast-on/slow-off, and
  a single constant for both cannot express that. Both extremes fail — see the
  `_excitable_step` comment. Narrow: 0.02 floods on collision, 0.20 collapses
  the front inward.
- `beta_e = 0.15` — width of the regenerative transfer's sigmoid, in units of
  `θe`. Sharper than this makes the ignition gradient stiff; broader leaks
  regeneration into the subthreshold band and lifts `ρ_sub`.
- remaining kwargs are forwarded to the wrapped [`PhasorWaveSheet`](@ref);
  `transmit` is forced to `:spike` and `homeostasis` to `:none` (the homeostat
  regulates the same `θ` this layer modulates, and the two fight).
"""
struct ExcitableWaveSheet <: Lux.AbstractLuxLayer
    base::PhasorWaveSheet
    init_log_alpha::Float32
    init_log_kappa_r::Float32
    init_logit_rho_u::Float32
    init_logit_rise::Float32
    init_log_beta_e::Float32
end

function ExcitableWaveSheet(H::Integer, W::Integer;
                            theta_frac::Real = 2.6,
                            alpha::Real      = 6.0,
                            kappa_r::Real    = 1.5,
                            rho_u::Real      = 0.99,
                            rise::Real       = 0.08,
                            beta_e::Real     = 0.15,
                            kwargs...)
    0 < rho_u < 1 || throw(ArgumentError("rho_u must be in (0,1), got $rho_u"))
    0 < rise <= 1 || throw(ArgumentError("rise must be in (0,1], got $rise"))
    alpha   >= 0 || throw(ArgumentError("alpha must be ≥ 0, got $alpha"))
    kappa_r >= 0 || throw(ArgumentError("kappa_r must be ≥ 0, got $kappa_r"))
    beta_e  >  0 || throw(ArgumentError("beta_e must be > 0, got $beta_e"))
    base = PhasorWaveSheet(H, W; transmit = :spike, homeostasis = :none,
                           init_theta_frac = theta_frac, kwargs...)
    # α is log-parameterised, so α = 0 needs a sentinel rather than log(0).
    la = alpha == 0 ? -30f0 : Float32(log(alpha))
    lk = kappa_r == 0 ? -30f0 : Float32(log(kappa_r))
    lr = rise == 1 ? 30f0 : Float32(log(rise / (1 - rise)))
    return ExcitableWaveSheet(base, la, lk,
                              Float32(log(rho_u / (1 - rho_u))), lr, Float32(log(beta_e)))
end

function Base.show(io::IO, l::ExcitableWaveSheet)
    print(io, "ExcitableWaveSheet($(l.base.grid_h)×$(l.base.grid_w); ",
          "α=", round(exp(l.init_log_alpha); digits = 2),
          ", κ_r=", round(exp(l.init_log_kappa_r); digits = 2),
          ", ρ_u=", round(_sigmoid(l.init_logit_rho_u); digits = 3), ")")
end

function Lux.initialparameters(rng::AbstractRNG, l::ExcitableWaveSheet)
    base_ps = Lux.initialparameters(rng, l.base)
    return (; base_ps...,
            log_alpha   = Float32[l.init_log_alpha],
            log_kappa_r = Float32[l.init_log_kappa_r],
            logit_rho_u = Float32[l.init_logit_rho_u],
            logit_rise  = Float32[l.init_logit_rise],
            log_beta_e  = Float32[l.init_log_beta_e])
end

Lux.initialstates(rng::AbstractRNG, l::ExcitableWaveSheet) = Lux.initialstates(rng, l.base)

# ---- Emission ----------------------------------------------------------

"""
    excitable_emit(z, θe, α, β) -> complex array

The regenerative emission (a). Reduces to the base sheet's
`normalize_to_unit_circle(z; ε = θe²)` at `α = 0`.

The boost is *threshold-localised* by construction: for `|z| ≪ θe` the sigmoid
is ≈0 and this is exactly `z/θe`, the linear subthreshold medium at gain `g/θ`
that `dispersion` describes. That is the property that lets the propagation
boundary move without dragging the flood boundary with it.
"""
function excitable_emit(z, θe, α, β)
    r = abs.(z)
    base = z ./ sqrt.(r .^ 2 .+ θe .^ 2)
    return base .* (1 .+ α .* _sigmoid.((r .- θe) ./ (β .* θe)))
end

# ---- Step / rollout ----------------------------------------------------

function _excitable_step(z, u, A_step, g, W_hat, θ, α, β, κ_r, ρ_u, rise, drive_t)
    H, W, B = size(z)
    θe   = θ .* (1 .+ κ_r .* u)                                    # (H,W,B)
    emit = excitable_emit(z, θe, α, β)
    coupled = ifft(reshape(W_hat, H, W, 1) .* fft(emit, (1, 2)), (1, 2))
    z_next  = reshape(A_step, 1, 1, 1) .* z .+ reshape(g, 1, 1, 1) .* coupled
    drive_t === nothing || (z_next = z_next .+ drive_t)
    # Fire indicator on the NEW state, hard forward / sigmoid backward — the same
    # straight-through estimator the base homeostat uses, and load-bearing for the
    # same reason: a smooth indicator lets the refractory variable be satisfied by
    # a uniform subthreshold sheet instead of a sparse firing one.
    fire   = _fire_indicator(z_next, θe, β)
    # FAST-ON, SLOW-OFF, and the ON rate is its own parameter for a reason. A
    # plain leaky integrator `ρ_u·u + (1−ρ_u)·fire` charges on the SAME constant
    # it discharges on, so at ρ_u = 0.98 a site needs ~35 steps of firing before
    # u can shut it down: the pulse travels as a filled disc and only hollows out
    # by t≈75 (measured interior fill 1.00 → 0.77 → 0.63 → 0.17 → 0.00). That is
    # adaptation, not refractoriness.
    #
    # But the opposite extreme (`u ← max(ρ_u·u, fire)`, u → 1 the instant a site
    # fires) is worse: θe gates this site's OWN emission as well as its
    # excitability, so instant refractoriness cuts the spike off before it can
    # drive anyone. Measured: every (frac, α, κ_r) in the calibration grid went
    # to a NEGATIVE front speed — the pulse collapses inward — except one, which
    # flooded. In a real neuron the spike is emitted and *then* inactivation sets
    # in; `rise` is that delay.
    u_next = min.(1f0, ρ_u .* u .+ rise .* fire)
    return ComplexF32.(z_next), u_next, fire
end

"""
    excitable_simulate(l, ps, st; z0, L, drive=nothing, keep_fields=true)

Evolve the excitable sheet for `L` steps. Returns
`(; z, fire, u, theta, speed_hint)` where `z`, `fire` and `u` are `(H,W,L[,B])`
(omitted when `keep_fields = false`, leaving the scalar summaries).

`z0` may be `(H,W)` or `(H,W,B)`. Unlike [`wave_simulate`](@ref) there is no
`:scan` / `:deq` mode: the regenerative emit is not a fixed linear operator, so
there is no per-mode `M` and no parallel scan. See the header of
`velocity_bank.jl` for the same distinction drawn the other way.
"""
function excitable_simulate(l::ExcitableWaveSheet, ps, st;
                            z0::AbstractArray, L::Integer,
                            drive = nothing, keep_fields::Bool = true)
    H, W = l.base.grid_h, l.base.grid_w
    had_batch = ndims(z0) == 3
    z = had_batch ? ComplexF32.(z0) : reshape(ComplexF32.(z0), H, W, 1)
    size(z, 1) == H && size(z, 2) == W || throw(DimensionMismatch("z0 must be $(H)×$(W)[×B]"))
    B = size(z, 3)
    ω = period_to_angfreq(l.base.spk_args.t_period)
    A_step, g, W_hat = _build_coupling(l.base, ps, st, ω)
    θ = exp(ps.log_theta[1])
    α = exp(ps.log_alpha[1]); β = exp(ps.log_beta_e[1])
    κ_r = exp(ps.log_kappa_r[1]); ρ_u = _sigmoid(ps.logit_rho_u[1])
    rise = _sigmoid(ps.logit_rise[1])

    drive_4 = drive === nothing ? nothing :
              (ndims(drive) == 3 ? reshape(ComplexF32.(drive), H, W, L, 1) : ComplexF32.(drive))
    u = zeros(Float32, H, W, B)
    Zs = keep_fields ? Array{ComplexF32}(undef, H, W, L, B) : nothing
    Fs = keep_fields ? Array{Float32}(undef, H, W, L, B) : nothing
    Us = keep_fields ? Array{Float32}(undef, H, W, L, B) : nothing
    rate = Vector{Float32}(undef, L)
    for n in 1:L
        dt = drive_4 === nothing ? nothing : drive_4[:, :, n, :]
        z, u, f = _excitable_step(z, u, A_step, g, W_hat, θ, α, β, κ_r, ρ_u, rise, dt)
        rate[n] = sum(f) / Float32(H * W * B)
        if keep_fields
            Zs[:, :, n, :] = z; Fs[:, :, n, :] = f; Us[:, :, n, :] = u
        end
    end
    sq(x) = had_batch ? x : (x === nothing ? nothing : reshape(x, H, W, L))
    return (; z = sq(Zs), fire = sq(Fs), u = sq(Us), theta = θ, rate)
end

"""
    excitable_regime(l, ps, st) -> NamedTuple

Closed-form diagnostics for whether the layer is in the excitable window. All
four fields must hold; the base sheet fails `regenerative_margin` at every
setting, which is why it needs this fork at all.

!!! note "rho_sub is a slight underestimate"
    The regenerative sigmoid has a floor, so the boost is `1 + α·σ(−1/β)` rather
    than exactly 1 far below threshold — ≈1.0076 at the defaults. `rho_sub` is
    computed from the unboosted kernel and so understates the true subthreshold
    spectral radius by roughly that factor. It matters only if you run within
    ~1% of `rho_sub = 1`; the default 0.953 has ample margin.

- `rho_sub` — spectral radius of the subthreshold linearisation `A + (g/θ)Ŵ`.
  **Must be < 1** or the sheet floods from its own tail regardless of the
  excitable terms (this is what makes `theta_frac = 1.8` unusable despite it
  giving the cleanest amplitude invariance).
- `coherent_bound` — `g·Σ|W|·(1+α)`, the ignition drive from a perfectly
  phase-aligned, fully-firing neighbourhood.
- `regenerative_margin` — `coherent_bound / θ`. Must exceed 1 for propagation to
  be possible at all; because real coherence is ≈0.61 of the bound, it wants to
  be ≳ 1.6 in practice.
- `refractory_wavelength` — `≈ speed/(1−ρ_u)` in sites, using `speed_estimate`.
  Short wavelengths admit re-entry and degrade amplitude invariance.
"""
function excitable_regime(l::ExcitableWaveSheet, ps, st)
    ω = period_to_angfreq(l.base.spk_args.t_period)
    A_step, g, W_hat = _build_coupling(l.base, ps, st, ω)
    θ = exp(ps.log_theta[1]); α = exp(ps.log_alpha[1]); ρ_u = _sigmoid(ps.logit_rho_u[1])
    Wr = ifft(W_hat)
    sumW = sum(abs.(Wr))
    bound = only(g) * sumW * (1 + α)
    ρ_sub = maximum(abs.(only(A_step) .+ (only(g) / θ) .* W_hat))
    return (; rho_sub = Float32(ρ_sub),
              theta = Float32(θ),
              coherent_bound = Float32(bound),
              regenerative_margin = Float32(bound / θ),
              refractory_recovery = Float32(1 / (1 - ρ_u)),
              quiescent = ρ_sub < 1,
              can_propagate = bound / θ > 1)
end

"""
    front_speed(fire; center=nothing) -> (speed, radii)

Least-squares front speed in sites/period from a `(H,W,L)` fire trajectory,
measured as the slope of the mean radius of the firing set. Returns `NaN` when
fewer than five frames carry a resolvable front.

This is the diagnostic that distinguishes a real excitable front from a level
set of a growing linear mode: for the former the returned speed is invariant to
the seed amplitude, for the latter it is not. Sweep the seed and compare.
"""
function front_speed(fire::AbstractArray{<:Real,3}; center = nothing)
    H, W, L = size(fire)
    cy, cx = center === nothing ? (H ÷ 2, W ÷ 2) : center
    rg = Float32[sqrt(Float32((i - cy)^2 + (j - cx)^2)) for i in 1:H, j in 1:W]
    ts = Float32[]; rs = Float32[]
    for n in 1:L
        m = @view(fire[:, :, n]) .> 0f0
        count(m) > 4 || continue
        push!(ts, Float32(n)); push!(rs, mean(rg[m]))
    end
    length(rs) < 5 && return (NaN32, rs)
    μt, μr = mean(ts), mean(rs)
    return (sum((ts .- μt) .* (rs .- μr)) / sum((ts .- μt) .^ 2), rs)
end
