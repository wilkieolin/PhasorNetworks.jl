# velocity_bank_hw.jl — physical units and device imperfections for the bank
#
# `velocity_bank.jl` works in lattice units: κ in rad/site, v in sites/period,
# T = 1. This file is the dictionary to a real device — SNSPD sensing elements
# feeding a kinetic-inductance coupling lattice in a sub-4 K environment — and
# the noise models needed to ask whether such a device would work.
#
# The whole mapping is one line. Because ω·T = 2π exactly, a particle exciting
# successive sites Δt apart deposits a phase ramp
#
#     κ = 2π·Δt/T   rad/site,        v = T/Δt   sites/period
#
# and the channel that matches it is the one whose coupling delay satisfies
# τ_c = Δt. Note what is *absent*: ω cancels out of the match condition, so a
# channel is a delay line whose delay equals the transit time, independent of
# carrier frequency. That is Jeffress' coincidence principle, and it means one
# delay layout serves any carrier. The carrier sets resolution (δΔt = δκ·T/2π)
# and ambiguity (Δt < T/2), not matching.
#
# At the design point T = 100 ps (10 GHz), Δt = 10 ps is κ = 0.628 = π/5, i.e.
# v = 10 sites/period — the middle of the range the bank already validates.
#
# THREE IMPERFECTIONS, and why the obvious model of each is wrong:
#
#   detector jitter — an SNSPD fires ONCE per event, with ONE timing error,
#     frozen for that event. Resampling jitter per substep averages it down by
#     √(substeps/v) and overstates performance by ~2.5× at the design point.
#
#   photon dropout — likewise frozen: a dead or unlit site is dead for the
#     whole transit, not independently dead at each substep. A sparse aperture,
#     which the matched filter handles gracefully (85% tolerable).
#
#   delay scatter — fabrication spread on the coupling delays. The honest model
#     is PER LINK, which breaks translation invariance and so cannot be applied
#     in the Fourier path at all. `velocity_bank_run_disordered` pays for a
#     real-space stencil convolution to model it properly; this is affordable
#     only for a truncated stencil (4 taps/site at R = 1, 12 at R = 2).
#
# Systematic delay error needs no model: it maps κ → (1+ε)κ, a pure rescale
# absorbed by one calibration constant.
#
# See docs/wavesheet_hardware_ladder.md for the measured device numbers these
# models are meant to be driven with, and for provenance of every figure.

# ---- Units -------------------------------------------------------------

"""
    kappa_from_dt(dt_ps, t_period_ps) -> κ  (rad/site)
    dt_from_kappa(κ, t_period_ps)     -> Δt (ps)
    speed_from_dt(dt_ps, t_period_ps) -> v  (sites/period)
    dt_from_speed(v, t_period_ps)     -> Δt (ps)

The physical-units dictionary for the bank. `κ = 2π·Δt/T` and `v = T/Δt`.

Unambiguous only for `Δt < T/2` (κ < π); beyond that the ramp aliases and a
second carrier is needed to disambiguate.
"""
kappa_from_dt(dt_ps::Real, t_period_ps::Real) = 2π * dt_ps / t_period_ps
dt_from_kappa(κ::Real, t_period_ps::Real)     = κ * t_period_ps / 2π
speed_from_dt(dt_ps::Real, t_period_ps::Real) = t_period_ps / dt_ps
dt_from_speed(v::Real, t_period_ps::Real)     = t_period_ps / v

"""
    nyquist_dt(t_period_ps) -> Δt_max (ps)

Largest site-to-site step a single carrier can resolve unambiguously, `T/2`.
"""
nyquist_dt(t_period_ps::Real) = t_period_ps / 2

# ---- Bank construction in physical units -------------------------------

"""
    hardware_bank(H, W; dt_lo_ps, dt_hi_ps, t_period_ps, n_speeds, n_angles, ...)

A [`PhasorVelocityBank`](@ref) whose channels tile a **Δt range in picoseconds**
on a grid uniform in `κ` (not in speed — uniform speed spacing bunches the
channels at the fast end, where the response is narrowest).

`stencil_radius` defaults to 1 — the 4-tap von Neumann neighbourhood, which
measures 0.17% speed error against ~1300 taps/site for the full Gaussian. Raise
it to 2 for the clean optimum (12 taps, 0.02%) or 3–4 if detector jitter rather
than delay-line area is the limiting resource; see the ladder doc §2.4.

Throws if the requested range violates the `Δt < T/2` Nyquist bound.

# Example
```julia
l = hardware_bank(64, 64; dt_lo_ps = 8, dt_hi_ps = 40, t_period_ps = 100)
```
"""
function hardware_bank(H::Integer, W::Integer;
                       dt_lo_ps::Real = 8.0, dt_hi_ps::Real = 40.0,
                       t_period_ps::Real = 100.0,
                       n_speeds::Integer = 24, n_angles::Integer = 16,
                       init_log_sigma::Real = log(6.0),
                       init_log_neg_lambda::Real = log(0.15),
                       margin::Real = 0.9, stencil_radius::Real = 1.0,
                       stencil_aspect::Real = 1.0)
    0 < dt_lo_ps < dt_hi_ps || throw(ArgumentError("need 0 < dt_lo_ps < dt_hi_ps"))
    dt_hi_ps < nyquist_dt(t_period_ps) ||
        throw(ArgumentError("dt_hi_ps = $dt_hi_ps ps exceeds the Nyquist bound " *
                            "T/2 = $(nyquist_dt(t_period_ps)) ps for T = $t_period_ps ps; " *
                            "raise t_period_ps or add a second carrier"))
    κlo = kappa_from_dt(dt_lo_ps, t_period_ps)
    κhi = kappa_from_dt(dt_hi_ps, t_period_ps)
    κs  = range(κlo, κhi; length = n_speeds)
    speeds = [2π / κ for κ in κs]                      # v = ω/κ with ω = 2π (T ≡ 1)
    angles = range(0, 2π; length = n_angles + 1)[1:n_angles]
    return PhasorVelocityBank(H, W; speeds, angles, init_log_sigma,
                              init_log_neg_lambda, margin, stencil_radius,
                              stencil_aspect)
end

# ---- Detector imperfections --------------------------------------------

"""
    detector_noise(rng, H, W; jitter_fwhm_ps, t_period_ps, dropout_frac)
        -> (site_jitter, site_alive)

Frozen per-event detector imperfections, to hand to [`moving_drive`](@ref).

- `site_jitter` — `(H, W)` of timing offsets in **periods**, drawn once per
  site from a Gaussian of the given FWHM. Frozen because a detector fires once.
- `site_alive` — `(H, W)` Bool; `dropout_frac` of sites register nothing for
  the whole transit.

Measured tolerances at `T = 100 ps`, `N = 128`, `R = 1`, log-parabola readout:
25 ps FWHM → 0.63%, 35 ps → 1.5%, 50 ps → 6.4%; dropout is benign to 85% and
fails at 95%. Reported SNSPD jitter runs 15–18 ps (high-I_c NbN) through 26 ps
(MoSi) to 50–62 ps (arrays, SNAP) and 191 ps (WSi at 2.5 K), so the material
choice is a first-order design decision rather than a detail.
"""
function detector_noise(rng::AbstractRNG, H::Integer, W::Integer;
                        jitter_fwhm_ps::Real = 0.0, t_period_ps::Real = 100.0,
                        dropout_frac::Real = 0.0)
    0 <= dropout_frac < 1 || throw(ArgumentError("dropout_frac must lie in [0,1)"))
    σ = jitter_fwhm_ps / 2.355 / t_period_ps            # rms, in periods
    jit = σ > 0 ? Float32.(randn(rng, H, W) .* σ) : zeros(Float32, H, W)
    alive = dropout_frac > 0 ? (rand(rng, H, W) .> dropout_frac) : trues(H, W)
    return jit, alive
end

detector_noise(H::Integer, W::Integer; kwargs...) =
    detector_noise(Random.default_rng(), H, W; kwargs...)

# ---- Disordered (per-link) rollout -------------------------------------

# All non-zero offsets inside a hard radius. `(0,0)` is excluded: a site does
# not couple to itself.
function _stencil_offsets(R::Real)
    r = floor(Int, R)
    offs = Tuple{Int,Int}[]
    for dj in -r:r, di in -r:r
        d = sqrt(di^2 + dj^2)
        (d > 0 && d <= R) && push!(offs, (di, dj))
    end
    return offs
end

"""
    velocity_bank_run_disordered(l, ps, st, drive; delay_scatter, rng) -> (H,W,C,B)

Rollout with **per-link** fabrication scatter on the coupling delays: the delay
of every (site, offset, channel) link is independently wrong by a fractional
`delay_scatter` (rms, Gaussian).

This cannot be done in the Fourier path. Per-link disorder destroys translation
invariance, so `Ŵ(q)` no longer exists and the coupling must be evaluated as a
real-space stencil convolution — `O(H·W·C·|stencil|)` per step instead of one
shared FFT. That is affordable only because `R = 1` suffices, which is exactly
what this function exists to check. `l.stencil_radius` must be finite.

Three independent fabrication defects, all drawn per device:

- `delay_scatter` — fractional rms error on every link's delay. Since
  delay ∝ √L_s ∝ 1/√thickness, x% delay scatter is ~2x% film-thickness spread.
- `loss_spread` — fractional rms on each site's resonator loss `λ`, i.e. Q
  non-uniformity across the sheet.
- `gain_error` — one draw per device on the coupling gain, standing in for
  coupler/loss mismatch: the device not sitting exactly at the intended
  `margin`. This is why `margin = 0.9` rather than 0.99 (see the ladder doc
  §2.2) — at 0.99 a few per cent of gain error walks the sheet into instability.

Systematic delay error is deliberately not modelled: it rescales κ uniformly and
one calibration constant removes it.

Set `delay_scatter = 0` to get a disorder-free real-space rollout, which is a
useful cross-check on the FFT path in [`velocity_bank_run`](@ref).
"""
function velocity_bank_run_disordered(l::PhasorVelocityBank, ps, st,
                                      drive::AbstractArray{<:Complex,4};
                                      delay_scatter::Real = 0.0,
                                      loss_spread::Real = 0.0,
                                      gain_error::Real = 0.0,
                                      rng::AbstractRNG = Random.default_rng())
    H, W, L, B = size(drive)
    (H == l.grid_h && W == l.grid_w) ||
        throw(DimensionMismatch("drive is $(H)×$(W), sheet is $(l.grid_h)×$(l.grid_w)"))
    isfinite(l.stencil_radius) ||
        throw(ArgumentError("velocity_bank_run_disordered needs a finite stencil_radius; " *
                            "the full Gaussian would make the real-space path O(H²W²C)"))

    T = Float32(l.spk_args.t_period)
    λ = -exp(ps.log_neg_lambda[1])
    ω = period_to_angfreq(T)
    A = ComplexF32(exp((λ + 1im * ω) * T))               # ωT = 2π ⇒ real positive
    σ = exp(ps.log_sigma[1])
    C = l.n_channels
    R = Float32(l.stencil_radius); asp = Float32(l.stencil_aspect)

    # Offsets are the enclosing DISC; an anisotropic channel simply zeroes the
    # ones outside its own ellipse. Wasteful by |disc|/|ellipse| ≈ 1/aspect, but
    # it keeps one offset list shared across channels.
    offs = _stencil_offsets(R)
    nO = length(offs)
    nO > 0 || throw(ArgumentError("stencil_radius = $R encloses no sites"))

    # Per-channel envelope in that channel's own frame, and its own g_crit.
    Genv = zeros(Float32, nO, C)
    gvec = Vector{Float32}(undef, C)
    for c in 1:C
        kh, kw = ps.kappa[1, c], ps.kappa[2, c]
        km = sqrt(kh^2 + kw^2) + 1f-12
        uh, uw = kh / km, kw / km
        for o in 1:nO
            di, dj = offs[o]
            dpar  =  di * uh + dj * uw
            dperp = -di * uw + dj * uh
            if (dpar / R)^2 + (dperp / (asp * R))^2 <= 1 + 1f-4   # see velocity_coupling
                Genv[o, c] = exp(-dpar^2 / (2f0 * σ^2) - dperp^2 / (2f0 * (asp * σ)^2))
            end
        end
        gvec[c] = Float32(l.margin) * (1f0 - real(A)) / max(sum(@view Genv[:, c]), 1f-12)
    end

    # Per-link weights. Memory is H·W·nO·C·8 bytes — 6 MB at 64×64, R=1, C=24.
    Wl = Array{ComplexF32}(undef, H, W, nO, C)
    for c in 1:C, o in 1:nO
        di, dj = offs[o]
        base = ps.kappa[1, c] * di + ps.kappa[2, c] * dj
        gw = Genv[o, c]
        @inbounds for j in 1:W, i in 1:H
            e = delay_scatter > 0 ? Float32(randn(rng) * delay_scatter) : 0f0
            Wl[i, j, o, c] = gw * cis(base * (1f0 + e))
        end
    end

    # Site-to-site resonator loss spread (Q non-uniformity), and a single
    # per-device coupling-gain error standing in for coupler/loss mismatch —
    # i.e. the device not sitting exactly at the intended `margin`.
    Asite = fill(A, H, W)
    if loss_spread > 0
        for j in 1:W, i in 1:H
            λij = λ * (1 + Float32(randn(rng) * loss_spread))
            Asite[i, j] = ComplexF32(exp((λij + 1im * ω) * T))
        end
    end
    gscale = gain_error > 0 ? 1f0 + Float32(randn(rng) * gain_error) : 1f0
    gvec .*= gscale

    Z = zeros(ComplexF32, H, W, C, B)
    acc = Array{ComplexF32}(undef, H, W, C, B)
    for n in 1:L
        fill!(acc, zero(ComplexF32))
        # (W ⊛ z)[p] = Σ_m W[m]·z[p − m], matching the Fourier path's convention
        for b in 1:B, c in 1:C, o in 1:nO
            di, dj = offs[o]
            @inbounds for j in 1:W
                jm = mod1(j - dj, W)
                for i in 1:H
                    im = mod1(i - di, H)
                    acc[i, j, c, b] += Wl[i, j, o, c] * Z[im, jm, c, b]
                end
            end
        end
        @inbounds for b in 1:B, c in 1:C, j in 1:W, i in 1:H
            Z[i, j, c, b] = Asite[i, j] * Z[i, j, c, b] +
                            gvec[c] * acc[i, j, c, b] + drive[i, j, n, b]
        end
    end
    return Z
end

# ---- Layout helper -----------------------------------------------------

"""
    delay_line_length(dt_ps; sheet_inductance_pH_sq, dielectric_nm, eps_r)
        -> (length_um, ps_per_mm, v_over_c, Z0_ohm_per_um_width)

Physical length of kinetic-inductance microstrip needed to realise a delay of
`dt_ps`, from `v = √(d / (L_s·ε₀·ε_r))` — the line width cancels out of the
phase velocity and appears only in the impedance.

Defaults are the NbTiN value measured for kinetic-inductance travelling-wave
amplifiers (8.5 pH/sq) with a 50 nm dielectric, giving ~78 ps/mm.

Reported sheet inductances span roughly 3–30 pH/sq: 3 pH/sq for a planarized
NbN foundry layer, 8 for Mo₂N, 8.5 for NbN HKIL and NbTiN, tens for 40 nm MgB₂.
Earlier back-of-envelope figures of 50–1000 pH/sq were 3–10× optimistic and
would understate every meander by the same factor.

Consequences worth remembering: a full 100 ps period is 1.3–1.8 mm of line, an
`R = 1` stencil costs 1.0–5.0 mm per site per channel, and fitting `C = 24`
channels needs a lattice pitch of ~500 µm.
"""
function delay_line_length(dt_ps::Real; sheet_inductance_pH_sq::Real = 8.5,
                           dielectric_nm::Real = 50.0, eps_r::Real = 4.0,
                           width_um::Real = 1.0)
    ε0 = 8.8541878128e-12
    Ls = sheet_inductance_pH_sq * 1e-12
    d  = dielectric_nm * 1e-9
    v  = sqrt(d / (Ls * ε0 * eps_r))                     # m/s
    return (length_um   = dt_ps * 1e-12 * v * 1e6,
            ps_per_mm   = 1e12 / (v * 1e3),
            v_over_c    = v / 299792458.0,
            Z0_ohm      = sqrt(Ls * d / (ε0 * eps_r)) / (width_um * 1e-6))
end
