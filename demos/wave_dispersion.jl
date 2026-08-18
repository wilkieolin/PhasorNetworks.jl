# wave_dispersion.jl — dispersion, wave-speed, and ablation demo for PhasorWaveSheet
#
# The first milestone from docs/rf_wave_network_implementation.md §5: a
# *pure-forward* demonstration (no training) that the recurrent
# resonate-and-fire sheet
#   (1) has a closed-form dispersion relation and a computable criticality
#       point (spectral radius ≈ 1) — the phase-SSM analogue of branching
#       ratio σ ≈ 1 from rf_wave_network_plan.html;
#   (2) propagates a seeded pulse as a traveling wave at a measurable speed;
#   (3) reproduces the plan's decisive ablations:
#         • remove lateral inhibition → spreading/saturating front, no ring;
#         • detune g below/above criticality → extinguish / saturate;
#         • add adaptation → standing bump destabilizes and travels.
#
# Run:  julia --project=. demos/wave_dispersion.jl
# Outputs PNG/GIF into demos/wave_out/ and prints diagnostics to stdout.

ENV["GKSwstype"] = "100"                     # headless GR (no display needed)

using PhasorNetworks, Lux, Random, Printf, Statistics
using Plots
using FFTW: fftshift

const OUTDIR = joinpath(@__DIR__, "wave_out")
isdir(OUTDIR) || mkpath(OUTDIR)

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

"RMS radius of the energy distribution |z|² about the seed at (1,1). This is a
*bulk-spread* statistic (second moment of the whole sheet), NOT a front speed:
it is dominated by where the energy already is, and it bends over as the pulse
fills the torus, so its slope depends on the fit window. Kept for the spread
plot; use `front_speed` for the wave speed."
function rms_radius(frame::AbstractMatrix{<:Complex}, rgrid::AbstractMatrix)
    e = abs2.(frame)
    s = sum(e)
    s < 1f-20 && return 0f0
    return sqrt(sum(e .* rgrid .^ 2) / s)
end

"Leading-edge radius: the largest wrapped distance at which |z| still reaches
`frac` of that frame's peak. Normalizing by the per-frame peak makes this
amplitude-blind, so it measures the *wavefront position* identically in the
decaying, marginal, and growing regimes."
function front_radius(frame::AbstractMatrix{<:Complex}, rgrid::AbstractMatrix;
                      frac::Real = 0.01)
    a = abs.(frame)
    m = maximum(a)
    m < 1f-20 && return 0f0
    return maximum(rgrid[a .>= Float32(frac) * m])
end

"Front speed in px/step: least-squares slope of the leading-edge radius vs step
over `win`. `win` must end before the front wraps the torus (max wrapped radius
is √2·N/2 ≈ 34 px for a 48×48 sheet), otherwise the edge saturates and the slope
is biased low.

CAVEAT — this is a *level-set* speed and it depends on `frac`: at criticality it
runs 0.65 px/step at frac=0.001 and 0.09 at frac=0.3. The packet does not lock
into a rigid travelling front (the medium is marginal, so there is no
amplification to pull one), it spreads dispersively — every level set moves at
its own speed. There is therefore no single scalar 'wave speed' here; the
honest answer is the whole band `v_r(|q|)` from `radial_band`, validated by
`packet_velocity`. Reported only as a coarse 'how far has the disturbance got'."
function front_speed(traj::AbstractArray{<:Complex,3}, rgrid::AbstractMatrix;
                     win = 5:35, frac::Real = 0.01)
    r = Float32[front_radius(traj[:, :, t], rgrid; frac = frac) for t in win]
    x = Float32.(collect(win))
    x̄ = mean(x); r̄ = mean(r)
    return sum((x .- x̄) .* (r .- r̄)) / sum((x .- x̄) .^ 2)
end

"""
    packet_velocity(layer, ps, st, q0; w_env, win) -> (v, r2)

Direct simulation measurement of the group velocity at wavenumber `q0`: launch a
Gaussian-enveloped plane wave `e^{i q0 x}` and least-squares fit the energy
centroid `⟨x⟩` vs step. This is the honest check on the `v_r(|q|)` band — the
band is a claim about how fast a packet at `q0` moves, so move one and look.

Two known biases, both physical and both convergent:
  • finite bandwidth `Δq ≈ 1/w_env` averages `v_r` over a window of `q`, which
    pulls the result toward the mean near a peak of the band;
  • the band-pass gain reshapes the packet spectrum toward `q*` as it runs,
    pulling the measured velocity toward `v_r(q*)`.
Widening `w_env` shrinks the first; lowering `g` shrinks the second.

The centroid is **circular** (energy-weighted phasor mean, then unwrapped over
time). A plain `⟨x⟩` over signed offsets silently breaks on the torus as soon as
the packet has appreciable mass near the wrap seam — which it does immediately
at the matched conduction speed, where the wave is ~15× faster and the envelope
spans a good fraction of a small sheet. Run this on a sheet several wavelengths
wider than the distance the packet covers in `win`.
"""
function packet_velocity(layer, ps, st, q0::Real; w_env::Real = 7f0, win = 2:16)
    H, W = layer.grid_h, layer.grid_w
    dx = Float32[(j - 1) > W ÷ 2 ? (j - 1 - W) : (j - 1) for i in 1:H, j in 1:W]
    dy = Float32[(i - 1) > H ÷ 2 ? (i - 1 - H) : (i - 1) for i in 1:H, j in 1:W]
    x0 = -Float32(W) / 4f0                     # start a quarter-sheet back
    env = exp.(-(((dx .- x0) .^ 2 .+ dy .^ 2) ./ (2f0 * Float32(w_env)^2)))
    z0  = ComplexF32.(env) .* exp.(1im .* Float32(q0) .* dx)
    traj = wave_simulate(layer, ps, st; z0 = z0, L = maximum(win) + 1)

    # circular centroid: angle of the energy-weighted phasor, in sites
    cx = Float32[]
    for t in win
        e = abs2.(traj[:, :, t])
        ph = sum(e .* cis.(2f0 * Float32(pi) .* dx ./ W)) / (sum(e) + 1f-20)
        push!(cx, Float32(angle(ph)) * W / (2f0 * Float32(pi)))
    end
    for i in 2:length(cx)                      # unwrap across the seam
        while cx[i] - cx[i-1] >  W / 2; cx[i] -= W; end
        while cx[i] - cx[i-1] < -W / 2; cx[i] += W; end
    end

    x = Float32.(collect(win))
    x̄ = mean(x); c̄ = mean(cx)
    v = sum((x .- x̄) .* (cx .- c̄)) / sum((x .- x̄) .^ 2)
    r2 = 1 - sum((cx .- (c̄ .+ v .* (x .- x̄))) .^ 2) / max(sum((cx .- c̄) .^ 2), 1f-20)
    return (v, r2)
end

"Bisection for the recurrent gain g giving a target spectral radius.
Returns g (a Float32). Uses that spectral_radius is monotone increasing in
g once g is large enough that the coupling dominates the fixed decay A."
function critical_gain(layer, ps, st; target = 1.0f0, lo = 1f-3, hi = 1f2, iters = 40)
    setg(g) = merge(ps, (log_g = Float32[log(g)],))
    sr(g) = dispersion(layer, setg(g), st).spectral_radius
    for _ in 1:iters
        mid = sqrt(lo * hi)                  # geometric bisection (g spans orders of magnitude)
        sr(mid) < target ? (lo = mid) : (hi = mid)
    end
    return Float32(sqrt(lo * hi))
end

set_param(ps, name, val) = merge(ps, NamedTuple{(name,)}((Float32[val],)))

# ---------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------

rng = Xoshiro(1)
H = W = 48
L = 60

# Linear medium (saturating=false) so the dispersion readout is exact.
# Difference-of-Gaussians tuned near spatial balance (A_exc·σ_E² ≈ B_inh·σ_I²,
# so ∫W ≈ 0) — this selects a nonzero wavelength (a ring).
#
# `init_log_speed` is deliberately OMITTED so the layer derives the matched
# conduction speed c = 2σ_I/T = 6 (delay phase φ = ω σ_I/c = π). That is what
# makes the selected ring actually *propagate*. The previous value here was 40,
# which is the weak-delay corner: there the gain peak coincides with the phase-band
# extremum, v_g(q*) → 0, and an impulse produces a STANDING ring pattern whose
# crests never move. See `wave_transport` and the note in src/wave.jl.
layer = PhasorWaveSheet(H, W;
                        transmit            = :potential,  # linear-wave dispersion demo (spike is the library default)
                        saturating          = false,
                        init_log_neg_lambda = log(0.15),   # light subthreshold damping
                        init_A_exc          = 1.0,
                        init_log_sigma_exc  = log(1.5),     # tight excitatory center
                        init_B_inh          = 0.25,         # balanced surround (∫W ≈ 0)
                        init_log_sigma_inh  = log(3.0))     # broad inhibitory surround (Mexican hat)
ps, st = Lux.setup(rng, layer)
rgrid  = st.rgrid

# A localized complex seed at the sheet origin (1,1), matching rgrid's center.
seed = zeros(ComplexF32, H, W)
seed[1, 1] = 1.0f0 + 0.0f0im

# Signed wrapped displacement grids — the wavepacket lives on the same torus.
const DX_GRID = Float32[(j - 1) > W ÷ 2 ? (j - 1 - W) : (j - 1) for i in 1:H, j in 1:W]
const DY_GRID = Float32[(i - 1) > H ÷ 2 ? (i - 1 - H) : (i - 1) for i in 1:H, j in 1:W]

println("="^68)
println("PhasorWaveSheet dispersion & wave demo   ($(H)×$(W), L=$(L))")
println("="^68)

# ---------------------------------------------------------------------
# 1. Dispersion + criticality
# ---------------------------------------------------------------------

g_crit = critical_gain(layer, ps, st; target = 1.0f0)
@printf("\n[1] Critical gain (spectral radius = 1):  g_crit = %.4f\n", g_crit)
for f in (0.85f0, 1.0f0, 1.10f0)
    d = dispersion(layer, set_param(ps, :log_g, log(f * g_crit)), st)
    @printf("     g = %.3f·g_crit  →  spectral radius = %.4f   max growth/step = %+.4f\n",
            f, d.spectral_radius, log(d.spectral_radius))
end

# Transport regime — the guard. A sheet can be perfectly critical and still be
# incapable of propagating anything, if the mode the gain selects is the one mode
# with no group velocity. That is invisible in the spectral radius.
tr = wave_transport(layer, set_param(ps, :log_g, log(g_crit)), st)
@printf("\n[1b] Transport regime:  %s  (transport ratio %.3f)\n", uppercase(string(tr.verdict)),
        tr.transport_ratio)
@printf("     conduction speed c = %.2f sites/period  →  delay phase φ = %.2f rad = %.2f·π\n",
        exp(ps.log_speed[1]), tr.delay_phase, tr.delay_phase / pi)
@printf("     selected mode q* = %.3f rad/px (λ = %.1f px), %s;  v_r(q*) = %+.3f px/period\n",
        tr.q_star, tr.lambda_star, tr.on_axis ? "on-axis" : "off-axis (diagonal)", tr.v_r_star)
@printf("     band's best transport: v_r = %+.3f at |q| = %.3f\n", tr.v_r_max, tr.q_at_v_r_max)
if tr.verdict !== :traveling
    println("     ⚠ the amplified mode carries little of the band's transport — an impulse")
    println("       will give standing rings. Try c = matched_conduction_speed(σ_I, T).")
end

# Growth-rate map over spatial frequency (fftshifted so DC is centered).
d_crit = dispersion(layer, set_param(ps, :log_g, log(g_crit)), st)
gr = fftshift(real.(d_crit.k_eff))
hm = heatmap(gr; title = "growth rate Re k_eff(q) at g_crit  (bright = marginal)",
             xlabel = "qₓ", ylabel = "q_y", aspect_ratio = 1, c = :viridis)
savefig(hm, joinpath(OUTDIR, "dispersion_growth.png"))
println("     saved dispersion_growth.png")

# ---------------------------------------------------------------------
# 1b. Band diagram: gain and radial group speed vs |q|
# ---------------------------------------------------------------------
#
# `radial_band` and `wave_transport` now live in the library (src/wave.jl) —
# they were prototyped here, but the transport ratio is the guard that catches a
# sheet sitting in the standing-ring regime, so it belongs where every caller
# can reach it.

# Multi-regime comparison centered on the critical point.
#
# NOTE the `g_crit` argument. This used to be called with the layer's *init*
# params (log_g = 0 ⇒ g = 1.0 ≈ 38·g_crit). At that gain the linear medium
# overflows Float32 within ~30 steps, every measured speed came back NaN, and
# the bottom panel drew nothing — while the top panel was labelled "critical"
# but actually showed a grid-scale (near-Nyquist) instability.
function plot_band_diagram(layer, ps_base, st, H, W, g_crit; g_factors=[0.85f0, 1.0f0, 1.1f0])
    colors = [:blue, :black, :red]
    labels = ["subcritical 0.85·g_crit", "critical g_crit", "supercritical 1.10·g_crit"]

    # Everything interesting lives below |q| ≈ 2; beyond that both bands are flat
    # (Γ saturates at -|λ|, v_r → 0) — clipping the tail is what makes this readable.
    qmax = 2.1f0
    bp_gain = plot(xlabel="|q| (rad/px)", ylabel="gain Γ = Re κ (1/period)",
                   title="Gain band Γ(|q|) — the sheet selects a ring",
                   legend=:topright, xlims=(0, qmax))
    bp_vg   = plot(xlabel="|q| (rad/px)", ylabel="radial group speed (px/period)",
                   title="Propagation band v_r(|q|) = -dΩ/d|q|  (+ = outward)",
                   legend=:bottomright, xlims=(0, qmax))

    rbs = Dict{Float32,Any}()
    for (f, c, lab) in zip(g_factors, colors, labels)
        psg = set_param(ps_base, :log_g, log(f * g_crit))
        rb  = radial_band(layer, psg, st; dq = 0.08f0)
        rbs[f] = rb
        plot!(bp_gain, rb.q, rb.gain; lw=2, label=lab, c=c)
        plot!(bp_vg,   rb.q, rb.v_r;  lw=2, label=lab, c=c)
    end

    rb_c = rbs[1.0f0]
    q_peak = rb_c.q[argmax(rb_c.gain)]
    for p in (bp_gain, bp_vg)
        vline!(p, [q_peak]; c=:gray, ls=:dash, lw=1,
               label=@sprintf("q* = %.2f (λ ≈ %.1f px)", q_peak, 2π/q_peak))
    end
    hline!(bp_gain, [0f0]; c=:black, ls=:dot, lw=1, label="marginal (Γ = 0)")
    hline!(bp_vg,   [0f0]; c=:black, lw=1, label="")

    # The null metric, drawn LAST so it sits on top of the zero line: averaging
    # the group-velocity VECTOR over each annulus gives exactly 0 at every |q| —
    # the concentric front has no preferred direction, so the vector cancels
    # ring-wise. That zero is what the old bottom panel was chasing.
    plot!(bp_vg, rb_c.q, rb_c.v_vec; lw=3, ls=:dot, c=:gray,
          label="|⟨v_g⟩| vector-averaged ≡ 0  (the wrong metric)")

    # Simulation check: launch a narrow-band packet at each on-grid q0 and fit its
    # centroid velocity. These points are a measurement, not a fit.
    #
    # Run on a DEDICATED wider sheet. At the matched conduction speed the packet
    # covers ~0.7 px/period, so over the fit window it crosses a large fraction of
    # a 48-px torus and both the centroid and the band sampling degrade. Same
    # kernel, more room; the band is grid-independent up to q-resolution, so we
    # compare against the probe sheet's own band.
    PH = PWv = 160
    probe = PhasorWaveSheet(PH, PWv;
                            transmit = :potential, saturating = false,
                            init_log_neg_lambda = log(0.15), init_A_exc = 1.0,
                            init_log_sigma_exc = log(1.5), init_B_inh = 0.25,
                            init_log_sigma_inh = log(3.0),
                            init_log_speed = log(exp(ps_base.log_speed[1])))
    pp, sp = Lux.setup(Xoshiro(1), probe)
    gp = critical_gain(probe, pp, sp; target = 1.0f0)
    rb_probe = radial_band(probe, set_param(pp, :log_g, log(gp)), sp; dq = 0.08f0)
    vr_at(q) = rb_probe.v_r[argmin(abs.(rb_probe.q .- q))]

    qs = Float32[]; vs = Float32[]; errs = Float32[]
    for n in 4:2:20
        q0 = 2f0 * Float32(pi) * n / PWv
        q0 > 1.6f0 && continue
        v, _ = packet_velocity(probe, set_param(pp, :log_g, log(gp)), sp, q0; win = 2:12)
        push!(qs, q0); push!(vs, v); push!(errs, v - vr_at(q0))
    end
    scatter!(bp_vg, qs, vs; c=:black, ms=5, markershape=:diamond,
             label="measured wavepacket centroid")

    band_plot = plot(bp_gain, bp_vg, layout=(2,1), size=(900, 680), leftmargin=5Plots.mm)
    savefig(band_plot, joinpath(OUTDIR, "band_diagram_critical_regime.png"))
    println("     saved band_diagram_critical_regime.png (gain band + radial group-speed band)")

    # Everything below is COMPUTED, not asserted — this block previously carried
    # hard-coded commentary that silently went stale when the operating point moved.
    imax = argmax(abs.(rb_c.v_r))              # abs: v_r is negative here (backward wave)
    vr_star = rb_c.v_r[argmax(rb_c.gain)]
    ratio = abs(vr_star) / max(abs(rb_c.v_r[imax]), 1f-9)
    @printf("       q* = %.3f rad/px (λ ≈ %.1f px);  |v_r| peaks at %.3f px/period at |q| = %.2f\n",
            q_peak, 2π/q_peak, abs(rb_c.v_r[imax]), rb_c.q[imax])
    @printf("       v_r at q* = %+.3f  →  transport ratio %.3f  (%s)\n", vr_star, ratio,
            ratio > 0.5 ? "selected mode carries the transport: TRAVELING" :
                          "selected mode barely moves: STANDING rings")
    @printf("       vector-averaged |⟨v_g⟩| over all annuli: max = %.2e  ← the metric that reads zero\n",
            maximum(rb_c.v_vec))
    @printf("       packet vs band (probe %d×%d, %d points): RMS deviation = %.4f px/period (%.1f%% of |v_r| peak)\n",
            PH, PWv, length(errs), sqrt(mean(errs .^ 2)),
            100 * sqrt(mean(errs .^ 2)) / abs(rb_c.v_r[imax]))
    return band_plot
end

band_plot_crit = plot_band_diagram(layer, ps, st, H, W, g_crit)

# ---------------------------------------------------------------------
# 2. Wave propagation + speed, at three operating points
# ---------------------------------------------------------------------

println("\n[2] Seeded-pulse propagation (linear medium):")
println("     A concentric pulse has NO net vector velocity — these are radial-extent")
println("     statistics, and neither is 'the' wave speed (see the band diagram for that).")
println("     front(1%) = leading-edge level set — depends on the threshold (0.65 px/step")
println("                 at 0.1%, 0.09 at 30%): the packet spreads, it does not lock a front")
println("     bulk      = RMS-radius slope — a second moment; bends over as the sheet fills")
speeds = Dict{String,Float32}()
radius_curves = Dict{String, Vector{Float32}}()
front_curves  = Dict{String, Vector{Float32}}()
for (label, f) in (("subcritical_0.85", 0.85f0),
                   ("critical_1.00",    1.00f0),
                   ("supercritical_1.10", 1.10f0))
    psg = set_param(ps, :log_g, log(f * g_crit))
    traj = wave_simulate(layer, psg, st; z0 = seed, L = L)      # (H,W,L)
    radii = Float32[rms_radius(traj[:, :, t], rgrid) for t in 1:L]
    fronts = Float32[front_radius(traj[:, :, t], rgrid) for t in 1:L]
    radius_curves[label] = radii
    front_curves[label]  = fronts
    v_front = front_speed(traj, rgrid; win = 5:min(L, 35))
    win = 5:min(L, 30)
    v_bulk = (radii[last(win)] - radii[first(win)]) / (last(win) - first(win))
    speeds[label] = v_front
    @printf("     %-20s  front(1%%) %.3f   bulk %.3f   (px/step)   final |z|ₘₐₓ = %.3e\n",
            label, v_front, v_bulk, maximum(abs.(traj[:, :, end])))
end

rc = plot(title = "radial extent of activity vs step", xlabel = "step",
          ylabel = "radius (px)", legend = :topleft)
for (label, fronts) in sort(collect(front_curves))
    plot!(rc, 1:L, fronts; label = "1% level set — $label", lw = 2)
end
for (label, radii) in sort(collect(radius_curves))
    plot!(rc, 1:L, radii; label = "RMS radius — $label", lw = 1, ls = :dash, alpha = 0.7)
end
savefig(rc, joinpath(OUTDIR, "wave_spread.png"))
println("     saved wave_spread.png")
println("     the level set climbs in steps: the wave is ringed, so it advances lobe by lobe —")
println("     each jump is a whole new outer annulus crossing 1% at once (count +~100 px in one step),")
println("     with slow creep in between. Another reason no single scalar summarises this well.")

# A GIF of the critical-gain traveling wave (phase-colored, magnitude-gated).
psg = set_param(ps, :log_g, log(g_crit))
traj = wave_simulate(layer, psg, st; z0 = seed, L = L)
anim = @animate for t in 1:L
    frame = fftshift(traj[:, :, t])
    mag = abs.(frame); mag ./= (maximum(mag) + 1f-12)
    heatmap(angle.(frame) .* mag; c = :twilight, clims = (-π, π),
            title = @sprintf("traveling wave  step %02d/%d", t, L),
            aspect_ratio = 1, colorbar = false, axis = false)
end
gif(anim, joinpath(OUTDIR, "traveling_wave.gif"); fps = 12)
println("     saved traveling_wave.gif")

# ---------------------------------------------------------------------
# 3. Ablations
# ---------------------------------------------------------------------

println("\n[3] Decisive ablations:")

# 3a. Remove lateral inhibition (B_inh = 0). At WEAK delay the Mexican hat is the
#     only thing selecting a wavelength, so deleting it collapses the peak to DC —
#     a uniform front instead of a ring. At the MATCHED conduction speed that is
#     no longer true: the delay phase e^{-iωr/c} varies by ~π across the kernel and
#     selects a wavelength on its own, so the ring survives the ablation. Both are
#     computed below rather than asserted — the old hard-coded "→ 0 = uniform
#     front" reading was only ever valid in the weak-delay regime.
function peak_q(l, p, s)                       # |q| of the fastest-growing mode, rad/px
    hh, ww = l.grid_h, l.grid_w
    idx = argmax(real.(dispersion(l, p, s).k_eff))
    di = idx[1] - 1; di = di > hh ÷ 2 ? di - hh : di
    dj = idx[2] - 1; dj = dj > ww ÷ 2 ? dj - ww : dj
    return sqrt((2pi*di/hh)^2 + (2pi*dj/ww)^2)
end
weak = PhasorWaveSheet(H, W; transmit = :potential, saturating = false,
                       init_log_neg_lambda = log(0.15), init_A_exc = 1.0,
                       init_log_sigma_exc = log(1.5), init_B_inh = 0.25,
                       init_log_sigma_inh = log(3.0), init_log_speed = log(40.0))
pw, sw = Lux.setup(Xoshiro(1), weak)
gw = critical_gain(weak, pw, sw; target = 1.0f0)
for (lab, l, p, s, gc) in (("matched c=$(round(exp(ps.log_speed[1]), digits=1))", layer, ps, st, g_crit),
                           ("weak    c=40.0", weak, pw, sw, gw))
    q_full  = peak_q(l, set_param(p, :log_g, log(gc)), s)
    q_noinh = peak_q(l, set_param(set_param(p, :B_inh, 0.0f0), :log_g, log(gc)), s)
    @printf("     %s:  peak |q| with inhibition = %.3f,  without = %.3f  →  %s\n",
            lab, q_full, q_noinh,
            q_noinh < 0.15 ? "collapses to DC (uniform front)" :
                             "ring survives (delay also selects a wavelength)")
end

# 3b. Criticality detune already shown in [2]: subcritical decays, supercritical saturates.
@printf("     detune g:  |z|ₘₐₓ(final)  sub=%.2e  crit=%.2e  super=%.2e  (extinguish → sustain → saturate)\n",
        maximum(abs.(wave_simulate(layer, set_param(ps,:log_g,log(0.85f0*g_crit)), st; z0=seed, L=L)[:,:,end])),
        maximum(abs.(wave_simulate(layer, set_param(ps,:log_g,log(1.00f0*g_crit)), st; z0=seed, L=L)[:,:,end])),
        maximum(abs.(wave_simulate(layer, set_param(ps,:log_g,log(1.10f0*g_crit)), st; z0=seed, L=L)[:,:,end])))

# 3c. Self-limiting (insight #3 of the companion doc). A *linear* wave medium
#     cannot self-limit: above criticality its amplitude grows geometrically
#     (|z|ₘₐₓ ~ spectral_radiusᵗ). Projecting onto the unit circle each step
#     (saturating=true, phase-only) bounds |z| ≡ 1 by construction — the
#     wave regulates its own amplitude while keeping all its phase content.
g_super = 1.6f0 * g_crit
# This section contrasts the two legacy potential-coupling regimes (linear vs
# the phase-only snap); spike transmission is shown as the default elsewhere.
layer_lin = PhasorWaveSheet(H, W; transmit = :potential, saturating = false,
                            init_log_neg_lambda = log(0.15), init_A_exc = 1.0,
                            init_log_sigma_exc = log(1.5), init_B_inh = 0.25,
                            init_log_sigma_inh = log(3.0), init_log_speed = log(40.0))
layer_sat = PhasorWaveSheet(H, W; transmit = :potential, saturating = true,
                            init_log_neg_lambda = log(0.15), init_A_exc = 1.0,
                            init_log_sigma_exc = log(1.5), init_B_inh = 0.25,
                            init_log_sigma_inh = log(3.0), init_log_speed = log(40.0))
ps_l, st_l = Lux.setup(rng, layer_lin)
ps_s, st_s = Lux.setup(rng, layer_sat)
ps_l = set_param(ps_l, :log_g, log(g_super))
ps_s = set_param(ps_s, :log_g, log(g_super))
traj_lin = wave_simulate(layer_lin, ps_l, st_l; z0 = seed, L = L)
traj_sat = wave_simulate(layer_sat, ps_s, st_s; z0 = seed, L = L)
amp_lin = Float32[maximum(abs.(traj_lin[:, :, t])) for t in 1:L]
amp_sat = Float32[maximum(abs.(traj_sat[:, :, t])) for t in 1:L]
@printf("     self-limiting @ g=1.6·g_crit:  linear |z|ₘₐₓ %.3f → %.3f (unbounded),  saturating %.3f → %.3f (≡1)\n",
        amp_lin[1], amp_lin[end], amp_sat[1], amp_sat[end])
sl = plot(1:L, amp_lin; label = "linear (grows geometrically)", lw = 2, yscale = :log10,
          title = "phase-only saturation self-limits amplitude", xlabel = "step",
          ylabel = "|z|ₘₐₓ (log)", legend = :topleft)
plot!(sl, 1:L, amp_sat; label = "saturating (phase-only, ≡ 1)", lw = 2)
savefig(sl, joinpath(OUTDIR, "self_limiting.png"))
println("     saved self_limiting.png")

# ---------------------------------------------------------------------
# 4. Tier-2 equivalence: discrete recurrence ≈ continuous ODE
# ---------------------------------------------------------------------
#
# The discrete phase-SSM recurrence (Tier 1) and the continuous ODE (Tier 2,
# integrated by DifferentialEquations.jl) integrate the *same* defining equation
# dz/dt = k·z + g·(coupling) — the SSM/ODE duality (K[n]=Aⁿ·B) the codebase is
# built on.
#
# They are NOT numerically interchangeable here, and the traveling regime makes
# that worse. Tier 1 operator-splits (M = A + gŴ) where Tier 2 exponentiates
# (M = e^{(k+gŴ)T}); since k and Ŵ commute the exact per-period map is A·e^{gŴT},
# so the two differ by ≈ (1−A)·gŴ per step — first order in g, as the `dispersion`
# docstring states. The matched conduction speed needs g_crit ≈ 0.069 instead of
# ≈ 0.026, and that ~2.7× gain is enough to turn a negligible per-step gap into a
# visible one over 60 steps. Measured field similarity at t=60, g = 0.85·g_crit:
#
#     c = 40 (g_crit 0.026) → 0.9997      c = 12 (g_crit 0.016) → 0.9995
#     c =  6 (g_crit 0.069) → 0.56
#
# It tracks g_crit, not c. Use `mode = :continuous` in `dispersion` when reasoning
# about the ODE tier at this operating point, and do not assume long discrete and
# ODE rollouts stay aligned.

println("\n[4] Tier-1 vs Tier-2 (discrete recurrence vs continuous ODE):")
@printf("     NOTE: g_crit = %.4f here; the O(g·T) splitting gap is no longer negligible.\n", g_crit)
g_sub = 0.85f0 * g_crit
psg = set_param(ps, :log_g, log(g_sub))
traj_d = wave_simulate(layer, psg, st; z0 = seed, L = L, mode = :discrete)
traj_o = wave_simulate(layer, psg, st; z0 = seed, L = L, mode = :ode)
fieldsim(a, b) = abs(sum(vec(a) .* conj.(vec(b)))) /
                 (sqrt(sum(abs2, vec(a))) * sqrt(sum(abs2, vec(b))) + 1f-20)
for t in (5, 20, 40, L)
    @printf("     step %2d  discrete↔ODE field similarity = %.4f\n", t,
            fieldsim(traj_d[:, :, t], traj_o[:, :, t]))
end
cmp = plot(layout = (1, 2), size = (720, 340))
for (i, (lab, tr)) in enumerate(("discrete (Tier 1)" => traj_d, "ODE (Tier 2)" => traj_o))
    fr = fftshift(tr[:, :, L]); m = abs.(fr); m ./= (maximum(m) + 1f-12)
    heatmap!(cmp[i], angle.(fr) .* m; c = :twilight, clims = (-π, π),
             title = lab, aspect_ratio = 1, colorbar = false, axis = false)
end
savefig(cmp, joinpath(OUTDIR, "tier_equivalence.png"))
println("     saved tier_equivalence.png")

# ---------------------------------------------------------------------
# 5. Emission threshold θ and homeostatic regulation (:spike mode)
# ---------------------------------------------------------------------
#
# Everything above is the :potential (linear) medium. :spike transmits
# z/√(|z|²+θ²), and θ — long a hardcoded 1e-4 "numerical guard" — is really the
# firing threshold. This section shows the three regimes it selects and what
# homeostasis buys. See §3-quater of docs/wave_dispersion_derivation.md.

println("\n[5] Emission threshold and homeostasis (:spike):")

const NS = 64
spike0 = PhasorWaveSheet(NS, NS)                       # transmit=:spike default
ps_s, st_s = Lux.setup(Xoshiro(1), spike0)
θ_ref = emission_threshold(spike0, ps_s, st_s)
@printf("     theta_ref = g*max|W_hat| = %.4f   (layer default = 1.4x = %.4f)\n",
        θ_ref, exp(only(ps_s.log_theta)))
@printf("     legacy hardcoded value    = %.6f   (%.0fx smaller)\n",
        1f-4, θ_ref / 1f-4)

seed_s = zeros(ComplexF32, NS, NS); seed_s[1, 1] = 1f0

# (a) θ sweep: flood → structured → extinct. `fire` and `std|z| / mean|z|` are
#     reported together because they fail independently — a uniform sheet sliced
#     at the right level reports the target rate with no structure at all.
θs = Float32[0.5, 1, 2, 4, 6, 8, 9, 10.58, 12, 15, 20, 30] .* 1f0
fire_v = Float32[]; rel_v = Float32[]; amp_v = Float32[]
for θ in θs
    lθ = PhasorWaveSheet(NS, NS; init_log_theta = log(θ))
    pθ, sθ = Lux.setup(Xoshiro(1), lθ)
    Z = wave_simulate(lθ, pθ, sθ; z0 = seed_s, L = 300)
    a = abs.(Z[:, :, 300])
    push!(fire_v, mean(a .> θ))
    push!(rel_v, std(a) / (mean(a) + 1f-12))
    push!(amp_v, mean(a) / θ)                       # mean |z| relative to threshold
end
println("     theta      fire%    std/mean   mean|z|/θ   regime")
for (θ, f, r, u) in zip(θs, fire_v, rel_v, amp_v)
    # Four distinct outcomes, and the labels must not be collapsed: a sheet with
    # 0% firing may be genuinely dead OR alive-but-subthreshold (still carrying a
    # structured field, just never crossing θ), and a sheet at 100% firing is not
    # "active" — it is saturated and spatially uniform, which is the same failure
    # as being dead for anything downstream.
    reg = f > 0.9         ? "flooded (uniform)" :
          u < 0.05        ? "extinct" :
          f < 1f-4        ? "subthreshold (alive, silent)" :
          r > 0.2         ? "STRUCTURED" : "uniform"
    @printf("     %6.2f   %6.2f%%   %8.4f   %9.4f   %s\n", θ, 100f0 * f, r, u, reg)
end

# (b) homeostasis: does it find the operating point on its own, across gains?
println("     homeostasis (target 2%):")
println("       mode      g     theta_final   theta/g    fire%    std/mean")
for hm in (:global, :local), gg in Float32[0.1, 1.0, 10.0]
    lh = PhasorWaveSheet(NS, NS; homeostasis = hm, init_log_g = log(gg))
    ph, sh = Lux.setup(Xoshiro(5), lh)
    local trh = wave_homeostat_trace(lh, ph, sh; z0 = seed_s, L = 500)
    local w = 401:500
    @printf("       %-8s %5.2f  %11.4f  %8.3f  %6.2f%%  %9.4f\n",
            hm, gg, trh.theta_g[end], trh.theta_g[end] / gg,
            100f0 * mean(trh.fire[w]), mean(trh.std_abs[w]) / mean(trh.mean_abs[w]))
end

tr_none = wave_homeostat_trace(PhasorWaveSheet(NS, NS),
              Lux.setup(Xoshiro(5), PhasorWaveSheet(NS, NS))...;
              z0 = seed_s, L = 500)
lgl = PhasorWaveSheet(NS, NS; homeostasis = :global)
tr_gl = wave_homeostat_trace(lgl, Lux.setup(Xoshiro(5), lgl)...; z0 = seed_s, L = 500)
lloc = PhasorWaveSheet(NS, NS; homeostasis = :local)
tr_lo = wave_homeostat_trace(lloc, Lux.setup(Xoshiro(5), lloc)...; z0 = seed_s, L = 500)

thr = plot(layout = (2, 2), size = (1000, 700), legend = :topright,
           titlefontsize = 9, guidefontsize = 8, legendfontsize = 7,
           left_margin = 5Plots.mm, bottom_margin = 4Plots.mm)
# Top-left: the θ sweep. std/mean is masked where the sheet is DEAD — on a field
# that has decayed to ~0 it is a ratio of two tiny numbers and reads as spurious
# "structure" (0.49 at θ=15) in exactly the region where there is none.
alive = amp_v .> 0.05f0
plot!(thr[1], θs, 100f0 .* fire_v; xscale = :log10, lw = 2, marker = :circle,
      label = "firing %", xlabel = "emission threshold θ", ylabel = "%",
      title = "θ selects the regime")
plot!(thr[1], θs[alive], 100f0 .* rel_v[alive]; lw = 2, ls = :dash, marker = :square,
      label = "std|z| / mean|z| (×100), live sheet only")
vline!(thr[1], [θ_ref]; lw = 2, c = :gray, ls = :dot, label = "θ_ref = g·max|Ŵ|")
vline!(thr[1], [1.4f0 * θ_ref]; lw = 2, c = :black, label = "default (1.4 θ_ref)")
# Top-right: θ trajectories.
plot!(thr[2], tr_none.theta_g; lw = 2, label = ":none (fixed)",
      xlabel = "step", ylabel = "θ_g", title = "homeostat finds the operating point")
plot!(thr[2], tr_gl.theta_g; lw = 2, label = ":global")
plot!(thr[2], tr_lo.theta_g; lw = 2, label = ":local")
# Bottom-left: firing rate vs the target.
for (lab, trp) in ((":none", tr_none), (":global", tr_gl), (":local", tr_lo))
    plot!(thr[3], 100f0 .* trp.fire; lw = 1.5, label = lab,
          xlabel = "step", ylabel = "% firing", title = "regulated to the 2% target")
end
hline!(thr[3], [2.0]; lw = 2, c = :black, ls = :dash, label = "target")
# Bottom-right: structure — the check a rate-only plot cannot make. Drop the
# first 60 steps and clip y: the charging transient peaks near 27 and would
# otherwise flatten the entire informative range into the axis.
for (lab, trp) in ((":none", tr_none), (":global", tr_gl), (":local", tr_lo))
    plot!(thr[4], 61:500, (trp.std_abs./(trp.mean_abs .+ 1f-12))[61:500];
          lw = 1.5, label = lab, ylims = (0, 1),
          xlabel = "step (charging transient dropped)", ylabel = "std|z| / mean|z|",
          title = "structure: →0 is the uniform-sheet failure")
end
savefig(thr, joinpath(OUTDIR, "emission_threshold.png"))
println("     saved emission_threshold.png")

println("\nDone. Figures in $(OUTDIR)")
