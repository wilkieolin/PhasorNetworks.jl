# velocity_bank.jl — velocity-tuned delay banks on the phasor sheet
#
# The obvious way to read a moving excitation off a PhasorWaveSheet is from the
# *wake* it leaves behind, which needs the sheet to be a good wave medium. This
# file takes the other route — matched filtering. The two reasons originally
# given for that choice are now out of date; both are corrected here, and the
# choice still stands, for a better reason.
#
# WHAT THE ORIGINAL RATIONALE SAID, AND WHAT IS TRUE NOW (re-measured 2026-08-13,
# after the matched-conduction-speed and emission-threshold work):
#
#   "reduced phase velocity of the default DoG sits at ~0.08 sites/period"
#       Stale — it predates the derived c = 2σ_I/T default. At the matched speed,
#       evaluated AT CRITICALITY, v_p(q*) = −Θ(q*)/q* = 0.747: ~9× higher, and
#       above the 0.39 of the `:aniso β_w=20` sheet that DID support usable wake
#       velocimetry (<3% over Mach 1.5–8). Caveat: v_p is strongly g-dependent —
#       at the layer's default g (≈14× critical) the DC mode wins, q* → 0 and
#       v_p → 0, which is the regime the old figure describes. Under this
#       definition I also could not reproduce 0.08 at c = 40 (I measure 0.51),
#       so the original convention likely differed; pin it before relying on
#       either number.
#
#   "in :spike transmission the delay-induced band vanishes identically, Im Λ ≡ 0"
#       True only of the SATURATED branch. The Kuramoto generator does have
#       Im Λ ≡ 0 (measured 6e-8, roundoff) because it sees only FT{Re W(d)},
#       which is real for an isotropic delayed kernel. But `:spike` now carries
#       an emission threshold θ, and BELOW it the sheet is the linear medium at
#       gain g/θ — a full complex band: max|Im M| = 0.71, v_p = +1.42,
#       v_g = −0.83, `wave_transport` → :traveling (ratio 0.928). The
#       impossibility result is real but scoped to amplitude-blind operation,
#       not to spike transmission as such.
#
# WHY THE BANK IS STILL THE RIGHT TOOL, independently of all that: a matched
# filter has no Mach floor (a wake needs v_source > v_p; the bank's 2.5–13
# sites/period is Mach 3.3–17.4 against v_p = 0.747, and the `:aniso` wake
# experiment broke by Mach 13); it does not require holding the sheet at
# criticality; and — decisively — it returns a DENSE PER-SITE velocity field, so
# it tracks curved paths that a single global wake FFT cannot represent at all.
# See the curved-track section of demos/wave_velocity_bank.jl.
#
# The mechanism. Because ω·T = 2π exactly, a particle crossing
# site x at continuous time t(x) leaves the neuron there at sub-cycle phase
# θ = 2π·frac(t(x)/T). For straight motion x(t) = x₀ + v·t·û that is
#
#     z(x) = exp( i·(2π/v)·(x − x₀)·û )
#
# — a plane wave whose wavevector IS the velocity, κ = (2π/v)·û. So velocity
# estimation is a matched-filter problem in κ, and §3.1's "a conduction delay is
# a complex weight" gives the filter for free: a kernel whose delay is linear in
# displacement,
#
#     W_c(d) = G(|d|)·exp(i·κ_c·d)      ⟹     Ŵ_c(q) = Ĝ(q − κ_c)
#
# is a rigid translation of the envelope's transform in q-space. Channel c is
# then a coherent integrator in the frame co-moving at velocity 2π·κ_c/|κ_c|²:
# a matched target is de-rotated to DC and accumulates, everything else rotates
# and cancels.
#
# Two properties make this cheap. W_c is still a function of displacement alone,
# so the bank stays translation-invariant, stays FFT-diagonal, and stays
# parallel-scan trainable — none of the closed-form machinery of §3.2/§5 is
# forfeited, unlike the patchy connectivity the experts overlay needs (§8). And
# Ĝ is real and positive for a low-pass envelope, so M_c(q) = A + g·Ĝ(q − κ_c)
# is real positive: arg M ≡ 0, no band, no dispersion, and stability is the
# closed form g < (1 − A)/ΣG with no bisection.
#
# NOTE the envelope must be LOW-PASS (a plain Gaussian), not the sheet's
# Mexican hat. A band-pass Ĝ peaks on a ring |q| = q*, so the match condition
# |κ_true − κ_c| = q* is a ring rather than a point and the channel cannot
# localise velocity.
#
# Memory: the bank state is (H, W, C, B) complex — H·W·C·B·8 bytes. A 128×128
# sheet with 176 channels and B=1 is 23 MB; scale C and B with that in mind.

# ---- Layer -------------------------------------------------------------

"""
    PhasorVelocityBank(H, W; speeds, angles, kwargs...)

A bank of `C = length(speeds)·length(angles)` velocity-tuned channels on an
`H×W` toroidal sheet of resonate-and-fire neurons. Each channel carries a
delay-ramped low-pass kernel and integrates coherently in its own co-moving
phase frame, so the channel whose preferred velocity matches a transiting
excitation accumulates while the rest cancel.

Preferred velocities are stored as the wavevectors `κ_c` (units rad/site) and
are trainable; recover the velocity a channel is tuned to with
[`channel_velocities`](@ref) (`v = 2π·κ/|κ|²`).

# Keyword arguments
- `speeds`, `angles` — grid used to initialise `κ`; sites/period and radians.
- `init_log_sigma` — log of the Gaussian envelope width (sites). Sets velocity
  selectivity from the *spatial* side; the temporal side is set by the leak.
- `init_log_neg_lambda` — log of `−λ`. The leak sets the integration window
  `1/(1−e^λ)` periods, which trades velocity resolution (`δv/v ≈ 1/2L`) against
  agility on a manoeuvring target.
- `margin` — coupling gain as a fraction of the closed-form critical gain
  `g_crit = (1 − A)/ΣG`. As `margin → 1` the matched mode integrates losslessly
  while mismatched modes saturate at `1/(1−A)`, so contrast is set by track
  length, not by margin: at `L = 45` periods, 0.99 buys only 30% more contrast
  than 0.9 but demands coupler/loss matching at the 1% rather than the 10%
  level. The default is 0.9 for that reason — see `docs/wavesheet_hardware_ladder.md` §2.2.
- `stencil_radius` — hard truncation of the coupling in sites; `Inf` keeps the
  full Gaussian, which at the default σ = 6 is ~1300 taps/site and unwireable.
  Truncation is measured in Euclidean distance, so `R = 1` is the 4-tap von
  Neumann stencil (the diagonals sit at √2) and `R = 1.5` the 8-tap Moore one.
  In 2D the clean optimum is `R = 2` (12 taps, 0.02% speed error) with `R = 1`
  close behind at 0.17%; under detector jitter the optimum moves out to
  `R = 3–4`. Note that with `R` small the envelope is nearly flat across the
  stencil, so truncation, not σ, sets the κ-response width.
- `stencil_aspect` — cross-track extent as a fraction of `R`, measured in each
  channel's *own* frame: the stencil is an ellipse with semi-axis `R` along
  `κ_c` and `aspect·R` across it, and the Gaussian is elongated to match.
  `aspect = 1` is the isotropic disc and reduces exactly to the previous
  behaviour. Smaller values buy along-track reach at a fraction of the taps,
  which matters because a 1D track embedded in a 2D lattice otherwise couples
  mostly to off-track neighbours carrying no signal.
"""
struct PhasorVelocityBank <: Lux.AbstractLuxLayer
    grid_h::Int
    grid_w::Int
    n_channels::Int
    n_speeds::Int                    # channel grid shape, for separable readout
    n_angles::Int                    #   refinement; κ is stored angle-fastest
    init_kappa::Matrix{Float32}      # (2, C) — rows are (κ_h, κ_w)
    init_log_sigma::Float32
    init_log_neg_lambda::Float32
    margin::Float32
    stencil_radius::Float32
    stencil_aspect::Float32
    spk_args::SpikingArgs
end

function PhasorVelocityBank(H::Integer, W::Integer;
                            speeds::AbstractVector{<:Real} = Float32[2, 3, 4, 6, 8, 12],
                            angles::AbstractVector{<:Real} = range(0, 2π; length = 17)[1:16],
                            init_log_sigma::Real = log(6.0),
                            init_log_neg_lambda::Real = log(0.15),
                            margin::Real = 0.9,
                            stencil_radius::Real = Inf,
                            stencil_aspect::Real = 1.0,
                            spk_args::SpikingArgs = SpikingArgs())
    ω = period_to_angfreq(spk_args.t_period)
    ns, na = length(speeds), length(angles)
    C = ns * na
    κ = Matrix{Float32}(undef, 2, C)
    i = 1
    for v in speeds, a in angles          # a varies fastest ⇒ i = (iv−1)·na + ia
        κ[1, i] = Float32(ω / v * cos(a))
        κ[2, i] = Float32(ω / v * sin(a))
        i += 1
    end
    0 < margin < 1 || throw(ArgumentError("margin must lie in (0,1), got $margin"))
    stencil_radius > 0 || throw(ArgumentError("stencil_radius must be > 0, got $stencil_radius"))
    0 < stencil_aspect <= 1 ||
        throw(ArgumentError("stencil_aspect must lie in (0,1], got $stencil_aspect"))
    stencil_aspect == 1 || isfinite(stencil_radius) ||
        throw(ArgumentError("an anisotropic stencil needs a finite stencil_radius"))
    return PhasorVelocityBank(Int(H), Int(W), C, ns, na, κ, Float32(init_log_sigma),
                              Float32(init_log_neg_lambda), Float32(margin),
                              Float32(stencil_radius), Float32(stencil_aspect), spk_args)
end

function Base.show(io::IO, l::PhasorVelocityBank)
    r = isfinite(l.stencil_radius) ? ", R=$(l.stencil_radius)" : ""
    r *= l.stencil_aspect == 1 ? "" : "×$(l.stencil_aspect)"
    print(io, "PhasorVelocityBank($(l.grid_h)×$(l.grid_w); C=$(l.n_channels)",
              "=$(l.n_speeds)×$(l.n_angles), σ=$(round(exp(l.init_log_sigma); digits = 2))",
              r, ", margin=$(l.margin), t_period=$(l.spk_args.t_period))")
end

# Signed toroidal offsets from the origin (1,1) — the phase ramp needs the
# *vector* displacement, not the distance the DoG kernel uses.
function _wrapped_offset_grids(H::Int, W::Int)
    dh = Matrix{Float32}(undef, H, W)
    dw = Matrix{Float32}(undef, H, W)
    for j in 1:W, i in 1:H
        a = i - 1;  a = a > H ÷ 2 ? a - H : a
        b = j - 1;  b = b > W ÷ 2 ? b - W : b
        dh[i, j] = Float32(a);  dw[i, j] = Float32(b)
    end
    return dh, dw
end

function Lux.initialparameters(::AbstractRNG, l::PhasorVelocityBank)
    return (kappa          = copy(l.init_kappa),
            log_sigma      = Float32[l.init_log_sigma],
            log_neg_lambda = Float32[l.init_log_neg_lambda])
end

function Lux.initialstates(::AbstractRNG, l::PhasorVelocityBank)
    dh, dw = _wrapped_offset_grids(l.grid_h, l.grid_w)
    return (dh = dh, dw = dw, rgrid = _wrapped_distance_grid(l.grid_h, l.grid_w),
            margin = l.margin, stencil_radius = l.stencil_radius,
            stencil_aspect = l.stencil_aspect)
end

# ---- Coupling ----------------------------------------------------------

"""
    velocity_coupling(l, ps, st) -> (A_step, g, M)

Build the per-channel transition `M[:,:,c] = A + g_c·Ŵ_c(q)` with
`Ŵ_c(q) = Ĝ_c(q − κ_c)`. `Ĝ_c` is real and positive, so every `M` is real
positive (no band, no dispersion) and `g_crit = (1 − A)/ΣG_c` is exact — the
spectral radius is `A + g_c·ΣG_c` with no search.

`g` is returned as a `(1, 1, C)` array: with an anisotropic stencil each
channel carries its own envelope and therefore its own critical gain, so that
every channel sits at the same `margin` rather than the same absolute gain.

The envelope is evaluated in each channel's own frame, rotated to `κ_c`:

    d∥ = d·û_c,   d⊥ = d − (d·û_c)û_c,   û_c = κ_c/|κ_c|
    G_c(d) = exp(−d∥²/2σ² − d⊥²/2(aσ)²)   on   (d∥/R)² + (d⊥/aR)² ≤ 1

With `a = stencil_aspect = 1` this is exactly the isotropic disc of radius `R`.
"""
function velocity_coupling(l::PhasorVelocityBank, ps, st)
    T  = Float32(l.spk_args.t_period)
    λ  = -exp.(ps.log_neg_lambda)
    ω  = period_to_angfreq(T)
    A  = ComplexF32(exp((λ[1] + 1im * ω) * T))       # ωT = 2π ⇒ real positive
    σ  = exp(ps.log_sigma[1])
    R  = Float32(get(st, :stencil_radius, Inf32))    # along-track half-length
    a  = Float32(get(st, :stencil_aspect, 1f0))      # cross-track fraction of R
    H, W, C = l.grid_h, l.grid_w, l.n_channels

    dh = reshape(st.dh, H, W, 1);  dw = reshape(st.dw, H, W, 1)
    kh = reshape(ps.kappa[1, :], 1, 1, C);  kw = reshape(ps.kappa[2, :], 1, 1, C)
    km = sqrt.(kh .^ 2 .+ kw .^ 2) .+ 1f-12
    uh = kh ./ km;  uw = kw ./ km                    # unit vector along κ_c
    dpar  =  dh .* uh .+ dw .* uw
    dperp = -dh .* uw .+ dw .* uh

    G = exp.(-(dpar .^ 2) ./ (2f0 * σ^2) .- (dperp .^ 2) ./ (2f0 * (a * σ)^2))
    # Tolerance is load-bearing. d∥,d⊥ are a rotation of integer offsets, so a
    # lattice point sitting exactly on the boundary lands at 1±ulp depending on
    # the channel's direction. Without slack the disc keeps 3.7 of its 4 taps at
    # R=1, and *which* taps it drops varies per channel — the stencil stops
    # being isotropic, every channel is perturbed differently, and the readout's
    # interpolation across channels degrades by an order of magnitude.
    inside = ((dpar ./ R) .^ 2 .+ (dperp ./ (a * R)) .^ 2) .<= 1f0 + 1f-4
    G = G .* inside .* (reshape(st.rgrid, H, W, 1) .> 0f0)   # no self-coupling
    g = Float32(st.margin) .* (1f0 - real(A)) ./ max.(sum(G; dims = (1, 2)), 1f-12)

    ramp = kh .* dh .+ kw .* dw
    Wc = ComplexF32.(G) .* exp.(1im .* ComplexF32.(ramp))
    M  = A .+ ComplexF32.(g) .* fft(Wc, (1, 2))
    return A, g, M
end

"""
    channel_velocities(l, ps) -> (2, C) matrix

The velocity each channel is tuned to, `v = 2π·κ/|κ|²`, in sites/period.
"""
function channel_velocities(l::PhasorVelocityBank, ps)
    ω = period_to_angfreq(l.spk_args.t_period)
    k2 = sum(abs2, ps.kappa; dims = 1)
    return ω .* ps.kappa ./ max.(k2, 1f-12)
end

# ---- Rollout -----------------------------------------------------------

"""
    velocity_bank_run(l, ps, st, drive) -> (H, W, C, B) complex

Integrate the bank over the drive `(H, W, L, B)`, returning the final
per-channel field. One shared FFT of the drive per step feeds all `C` channels;
the channels never mix, so `M` is diagonal and the rollout is the same
geometric-kernel causal convolution `K_c,q[m] = M_c(q)^{m−1}` the linear sheet
uses in §5 — swap the loop for a parallel scan when `L` is large.
"""
function velocity_bank_run(l::PhasorVelocityBank, ps, st, drive::AbstractArray{<:Complex,4})
    H, W, L, B = size(drive)
    (H == l.grid_h && W == l.grid_w) ||
        throw(DimensionMismatch("drive is $(H)×$(W), sheet is $(l.grid_h)×$(l.grid_w)"))
    _, _, M = velocity_coupling(l, ps, st)
    C = l.n_channels
    Mx = reshape(M, H, W, C, 1)
    Z  = zeros(ComplexF32, H, W, C, B)
    for n in 1:L
        u = reshape(fft(drive[:, :, n, :], (1, 2)), H, W, 1, B)
        Z = Mx .* Z .+ u
    end
    return ifft(Z, (1, 2))
end

# Vertex of the parabola through three (x, y) points with arbitrary abscissae.
# Falls back to x2 when the fit is degenerate (collinear, flat, or a minimum).
function _parabola_vertex(x1::Real, y1::Real, x2::Real, y2::Real, x3::Real, y3::Real)
    d1 = float(x2) - float(x1)
    d3 = float(x2) - float(x3)
    num = d1^2 * (y2 - y3) - d3^2 * (y2 - y1)
    den = d1 * (y2 - y3) - d3 * (y2 - y1)
    (isfinite(num) && isfinite(den) && abs(den) > 1e-20) || return float(x2)
    return float(x2) - 0.5 * num / den
end

"""
    decode_velocity(l, ps, Z; frac = 0.97, interpolate = true) -> (speed, angle, mask)

Per-site velocity read-off. At each site the peak channel is located, then —
unless `interpolate = false` — refined by a **3-point log-parabola** fit run
separably over the two channel axes: over heading (which wraps) and over `|κ|`
(which does not, so edge channels are left unrefined).

Interpolating is not a nicety. Bare argmax is quantised to the channel grid and
floors out at the bin width: with `C = 24` spanning `κ ∈ [0.25, 2.2]` that is a
13.5% speed error, while the Cramér–Rao bound at 3 ps of per-site timing jitter
is 0.03% — a 150–300× gap that costs nothing in hardware to close. Measured on
the 1D bank, refinement takes 0.86–7.26% down to 0.02–0.35%.

The fit is **2D and Cartesian**, over the 3×3 channel neighbourhood of the peak,
and that matters. Refining `|κ|` and heading separably on the *polar* channel
grid carries a geometric bias: the channel best matching a target at angular
offset `Δθ` sits at radius `κ_true·cos Δθ`, not `κ_true`, so a heading falling
mid-bin foreshortens the speed estimate. The bias is a clean function of
position within the angle bin — zero at bin centres, ~2% at the half-bin point
with 8 headings — and no amount of separable refinement removes it, because it
is the polar grid rather than the fit that introduces it. Fitting a quadratic
in `(κ_h, κ_w)` and taking its vertex does remove it.

The fit is run on `log|Z|`, so it is invariant to whether amplitude or energy
is used. It falls back to separable refinement, and then to bare argmax, when
the neighbourhood is unavailable (edge channel) or the fitted Hessian is not
negative definite. Unlike the separable path it does *not* assume `κ` lies on
the product grid, so it degrades gracefully under training.
`interpolate = false` recovers bare argmax.

`mask` marks sites whose peak amplitude exceeds the `frac` quantile — use a
quantile, not a fraction of the global maximum, or a fast target that exits the
sheet early is swamped by a slow one that dwells.
"""
function decode_velocity(l::PhasorVelocityBank, ps, Z::AbstractArray{<:Complex,4};
                         frac::Real = 0.97, interpolate::Bool = true)
    H, W  = l.grid_h, l.grid_w
    ns, na = l.n_speeds, l.n_angles
    ω = period_to_angfreq(l.spk_args.t_period)
    amp  = dropdims(maximum(abs.(Z); dims = 4); dims = 4)        # (H,W,C)
    peak = dropdims(maximum(amp; dims = 3); dims = 3)

    # Grid coordinates read back from ps, so a trained κ is tracked rather than
    # assumed. Channels are stored angle-fastest: c = (iv−1)·na + ia.
    kmag = [sqrt(ps.kappa[1, (iv - 1) * na + 1]^2 + ps.kappa[2, (iv - 1) * na + 1]^2)
            for iv in 1:ns]
    angs = [atan(ps.kappa[2, ia], ps.kappa[1, ia]) for ia in 1:na]
    dθ   = 2π / na

    A4    = reshape(amp, H, W, na, ns)
    speed = Matrix{Float32}(undef, H, W)
    angle = Matrix{Float32}(undef, H, W)
    lg(x) = log(max(float(x), 1e-30))

    # Cartesian κ of every channel, and normal-equation buffers reused per site.
    cx = [kmag[s] * cos(angs[a]) for a in 1:na, s in 1:ns]
    cy = [kmag[s] * sin(angs[a]) for a in 1:na, s in 1:ns]
    Amat = zeros(Float64, 6, 6); bvec = zeros(Float64, 6); bas = zeros(Float64, 6)

    for j in 1:W, i in 1:H
        best = -Inf32; ia = 1; iv = 1
        for s in 1:ns, a in 1:na
            @inbounds v = A4[i, j, a, s]
            if v > best; best = v; ia = a; iv = s; end
        end
        θ, κm = float(angs[ia]), float(kmag[iv])
        done = false

        # --- 2D quadratic in Cartesian κ over the 3×3 channel neighbourhood ---
        if interpolate && na >= 3 && ns >= 3 && 1 < iv < ns
            x0, y0 = cx[ia, iv], cy[ia, iv]
            fill!(Amat, 0.0); fill!(bvec, 0.0)
            for ds in -1:1, da in -1:1
                a = mod1(ia + da, na); s = iv + ds        # heading wraps, radius does not
                dx = cx[a, s] - x0; dy = cy[a, s] - y0
                z  = lg(A4[i, j, a, s])
                bas[1] = 1.0; bas[2] = dx; bas[3] = dy
                bas[4] = dx * dx; bas[5] = dy * dy; bas[6] = dx * dy
                @inbounds for p in 1:6
                    bvec[p] += bas[p] * z
                    for q in 1:6; Amat[p, q] += bas[p] * bas[q]; end
                end
            end
            coef = try Amat \ bvec catch; nothing end
            if coef !== nothing && all(isfinite, coef)
                b_, c_, d_, e_, f_ = coef[2], coef[3], coef[4], coef[5], coef[6]
                det = 4 * d_ * e_ - f_ * f_
                if det > 0 && d_ < 0                      # negative definite ⇒ a maximum
                    sx = (-2 * e_ * b_ + f_ * c_) / det
                    sy = ( f_ * b_ - 2 * d_ * c_) / det
                    # a vertex further than one grid cell means the fit is not local
                    lim = max(abs(kmag[iv + 1] - kmag[iv]), abs(kmag[iv] - kmag[iv - 1]),
                              kmag[iv] * dθ)
                    if isfinite(sx) && isfinite(sy) && hypot(sx, sy) <= lim
                        κm = hypot(x0 + sx, y0 + sy)
                        θ  = atan(y0 + sy, x0 + sx)
                        done = true
                    end
                end
            end
        end

        # --- separable fallback: edge channels, or a fit that was not a maximum ---
        if interpolate && !done
            if na >= 3
                am, ap = mod1(ia - 1, na), mod1(ia + 1, na)
                δ = _parabola_vertex(-dθ, lg(A4[i, j, am, iv]),
                                     0.0, lg(best),
                                      dθ, lg(A4[i, j, ap, iv]))
                θ += clamp(δ, -dθ, dθ)
            end
            if ns >= 3 && 1 < iv < ns
                κm = _parabola_vertex(kmag[iv - 1], lg(A4[i, j, ia, iv - 1]),
                                      kmag[iv],     lg(best),
                                      kmag[iv + 1], lg(A4[i, j, ia, iv + 1]))
                lo, hi = minmax(float(kmag[iv - 1]), float(kmag[iv + 1]))
                κm = clamp(κm, lo, hi)
            end
        end

        speed[i, j] = Float32(ω / max(κm, 1e-12))
        angle[i, j] = Float32(θ)
    end

    sorted = sort(vec(peak))
    thr = sorted[clamp(ceil(Int, frac * length(sorted)), 1, length(sorted))]
    return speed, angle, peak .> thr
end

# ---- Stimulus ----------------------------------------------------------

"""
    moving_drive(H, W, L; v, angle, x0, width, curvature, substeps) -> (H,W,L,B=1)

A particle transiting the sheet, depositing its *sub-cycle* arrival phase.
A site crossed at continuous time `t` receives `exp(2πi·frac(t/T))`, which is
what makes the deposited field a plane wave at `κ = (2π/v)·û`.

`width` is the excitation footprint. It must be well under `v` sites: a
footprint of width `w` low-passes the ramp by `exp(−κ²w²/2)`, so a blunt
particle erases the phase ramp of a slow one. With `w = 0.8` the ramp survives
from `v ≈ 2.5` up; below the lattice Nyquist bound `v = 2` the wavevector
aliases and no readout can recover it.

`substeps` resolves the sub-cycle arrival time and therefore bounds the timing
detail the drive can represent: `T/substeps`. At the hardware design point
`T = 100 ps`, the old default of 16 resolved only 6.25 ps — the same order as
the 10 ps step being measured. It is now 64 (1.6 ps).

`site_jitter` and `site_alive` are `(H, W)` arrays carrying detector
imperfections, and both are *frozen per event*: a detector fires once, with one
timing error, not a fresh error every substep. Build them with
[`detector_noise`](@ref); resampling them per substep understates the error by
`√(substeps/v)`.

`curvature` is 1/radius in sites, and the arc starts along `angle`. Keep the
path length `v·L` **below one circumference** `2π/curvature`: past that the
particle laps its own track and re-deposits at a different carrier phase, which
destroys the ramp for any stencil and looks exactly like a curvature limit while
being nothing of the kind.
"""
function moving_drive(H::Integer, W::Integer, L::Integer;
                      v::Real, angle::Real = 0.0, x0 = (H ÷ 5, W ÷ 5),
                      width::Real = 0.8, curvature::Real = 0.0,
                      substeps::Integer = 64, t_period::Real = 1.0,
                      site_jitter::Union{Nothing,AbstractMatrix} = nothing,
                      site_alive::Union{Nothing,AbstractMatrix} = nothing)
    D = zeros(ComplexF32, H, W, L, 1)
    u = (cos(angle), sin(angle))
    for n in 0:(L - 1), k in 0:(substeps - 1)
        t = n + k / substeps
        s = v * t
        # The arc is built in a frame whose x-axis is the initial heading, then
        # rotated by `angle`. Previously the curved branch ignored `angle` and
        # always launched along +h, so every curved test silently ran at heading
        # 0 — which is a channel-grid bin *centre*, the most favourable case,
        # while straight-track tests were deliberately run mid-bin.
        if curvature == 0
            p = (x0[1] + s * u[1], x0[2] + s * u[2])
        else
            ah = sin(curvature * s) / curvature          # along initial heading
            aw = (1 - cos(curvature * s)) / curvature    # perpendicular to it
            p = (x0[1] + u[1] * ah - u[2] * aw,
                 x0[2] + u[2] * ah + u[1] * aw)
        end
        (0 <= p[1] < H && 0 <= p[2] < W) || continue
        i0, j0 = round(Int, p[1]), round(Int, p[2])
        rad = ceil(Int, 3 * width)                       # scale the splat with the footprint
        for jj in (j0 - rad):(j0 + rad), ii in (i0 - rad):(i0 + rad)
            (1 <= ii + 1 <= H && 1 <= jj + 1 <= W) || continue
            site_alive === nothing || site_alive[ii + 1, jj + 1] || continue
            w2 = ((ii - p[1])^2 + (jj - p[2])^2) / (2 * width^2)
            w2 > 12 && continue
            # the arrival phase is per-site, because the jitter is
            tj = site_jitter === nothing ? t : t + site_jitter[ii + 1, jj + 1]
            ph = ComplexF32(cis(2π * (tj - n) * (1.0 / t_period))) / substeps
            D[ii + 1, jj + 1, n + 1, 1] += ph * ComplexF32(exp(-w2))
        end
    end
    return D
end
