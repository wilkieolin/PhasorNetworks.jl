# wave_mach_cone.jl — a supersonic particle's Mach cone on the excitable sheet.
#
# A particle crosses the sheet in a straight line at v = 3 sites/period. The
# medium's own front speed is u ≈ 1.25, so the particle is supersonic (Mach 2.4)
# and the fronts it lights along its track cannot keep up: their envelope is a
# Mach cone of half-angle α = asin(u/v) trailing the particle.
#
# WHY THIS NEEDS `ExcitableWaveSheet` AND NOT `PhasorWaveSheet`. The base sheet
# is not an excitable medium — measured, suprathreshold activity decays to zero
# from any seed, because its emit function `z/√(|z|²+θ²)` is a limiter (gain is
# *maximal at rest* and falls as a site activates) so there is no chain reaction
# to sustain a front. See the header of src/excitable.jl. Running this script
# against the base sheet gives a track that lights up and goes out.
#
# TWO THINGS THAT ARE EASY TO GET WRONG HERE, both of which produce a
# convincing-looking animation of the wrong thing:
#
#   1. THE TRACK MUST BE WIDER THAN THE CRITICAL NUCLEUS. At the default
#      `moving_drive` footprint (width = 0.8) the particle deposits a line one
#      site across. One site is below the nucleus (a single-site seed
#      extinguishes; ~5 sites propagate), so the track fires but launches no
#      lateral front — measured off-axis extent 1 site at mid-track versus 66
#      at width 2.0. What you see instead is the circle from the *entry* point,
#      where the drive dwells long enough to nucleate, expanding and wrapping
#      the torus. It looks like a wave. It is not the cone.
#
#   2. THE CONE IS IN THE INSTANTANEOUS WAVEFRONT, NOT THE EVER-FIRED SET. Each
#      point of the track emits a circle; the ever-fired region is the *union*
#      of those discs, which is dominated by the largest one and is not conical
#      at all (a plane fit to the first-spike-time field over that union returns
#      v = 5.9 against a true 3.0). Refractoriness is what rescues the picture:
#      overlapping fronts annihilate, so only the outer envelope survives to
#      propagate — and that envelope is the cone. Hence an animation rather than
#      a still.
#
# The geometry is checked numerically below, not just drawn: the front half-width
# measured at increasing distance behind the particle converges on tan α, and is
# printed next to the value predicted from an independently measured u.
#
# Cost: keep_fields holds L·N²·(8+4+4) bytes plus a L·N²·8 drive — at 512²×160
# that is 0.67 GB + 0.34 GB ≈ 1.0 GB. CPU only (FFTW); no GPU allocation.
#
# Writes demos/wave_out/mach_cone.gif and mach_cone_geometry.png

ENV["GKSwstype"] = "100"                     # headless GR

using PhasorNetworks, Lux, Random, Printf, Statistics
using Plots

const OUTDIR = joinpath(@__DIR__, "wave_out")
isdir(OUTDIR) || mkpath(OUTDIR)

# N bounds the run, and the binding constraint is NOT the one you would guess.
# The track must not wrap (v·L < N) and the widest front must not (u·L < N/2) —
# but the real limit is that the entry point emits a full circle, and its
# BACKWARD half wraps the torus almost immediately (after ROW0/u ≈ 13 steps),
# then races around to meet the particle head-on. That happens at
#     t* = (N + ROW0 − ROW0) / (v + u) ≈ N/(v+u)
# — about t=120 here — after which the frame shows a collision, not a cone.
# L is set below t* on purpose. Raise MACH_L past it and the last third of the
# animation is the wrapped entry front annihilating against the cone; that is
# real behaviour (and a nice annihilation demo) but it is not what this shows.
const N      = parse(Int, get(ENV, "MACH_N", "640"))
const L      = parse(Int, get(ENV, "MACH_L", "145"))
const STRIDE = parse(Int, get(ENV, "MACH_STRIDE", "2"))
const VP     = parse(Float32, get(ENV, "MACH_V", "3.0"))    # particle speed
const AMP    = 150f0                                        # drive amplitude
const WIDTH  = 2.0                                          # > critical nucleus; see (1) above
const ROW0   = 16                                           # entry row

@info "sheet $(N)×$(N), $L steps, particle v=$VP" GB = round(L * N^2 * 24 / 2^30; digits = 2)
VP * L < N || @warn "track wraps: v·L = $(VP*L) ≥ N = $N"
let tstar = N / (VP + 1.27)
    L > tstar && @warn "L = $L exceeds the wrapped-entry-front collision at t ≈ $(round(tstar)); " *
                       "the tail of the animation shows a collision rather than a clean cone"
end

rng = Xoshiro(21)
sheet = ExcitableWaveSheet(N, N)
ps, st = Lux.setup(rng, sheet)
reg = excitable_regime(sheet, ps, st)
θ = reg.theta
@printf("θ = %.2f   ρ_sub = %.4f (%s)   regen margin = %.2f\n",
        θ, reg.rho_sub, reg.quiescent ? "quiescent" : "UNSTABLE", reg.regenerative_margin)

# ---------------------------------------------------------------------------
# [1] Measure the medium's front speed u INDEPENDENTLY of the particle, so the
#     predicted cone angle is not fitted from the same data it is checked against.
# ---------------------------------------------------------------------------
println("\n[1] medium front speed from a radial seed (no particle) ...")
let cen = N ÷ 2
    rg = Float32[sqrt(Float32((i-cen)^2 + (j-cen)^2)) for i in 1:N, j in 1:N]
    z0 = ComplexF32[rg[i,j] <= 6 ? 20f0*θ : 0 for i in 1:N, j in 1:N]
    u, _ = front_speed(excitable_simulate(sheet, ps, st; z0 = z0, L = 90).fire)
    global U_MEDIUM = u
end
const MACH   = VP / U_MEDIUM
const ALPHA  = asin(clamp(U_MEDIUM / VP, 0, 1))
@printf("    u = %.3f sites/period   ⇒  Mach = v/u = %.2f,  half-angle α = asin(u/v) = %.1f°  (tan α = %.3f)\n",
        U_MEDIUM, MACH, rad2deg(ALPHA), tan(ALPHA))
MACH > 1 || error("particle is subsonic (Mach $MACH) — no cone exists; raise MACH_V")

# ---------------------------------------------------------------------------
# [2] Run the particle across the sheet.
# ---------------------------------------------------------------------------
println("\n[2] traversing ...")
drive = AMP .* moving_drive(N, N, L; v = VP, angle = 0.0,
                            x0 = (ROW0, N ÷ 2), width = WIDTH)[:, :, :, 1]
tr = excitable_simulate(sheet, ps, st; z0 = zeros(ComplexF32, N, N), L = L, drive = drive)
fire, Z = tr.fire, tr.z
@printf("    fired at some point: %.1f%% of sites;  peak simultaneous front: %d sites\n",
        100 * count(any(fire .> 0f0; dims = 3)) / N^2,
        maximum(Int(sum(fire[:, :, t])) for t in 1:L))

particle_row(t) = ROW0 + VP * t

# ---------------------------------------------------------------------------
# [3] The animation. Left: the field, log |z|/θ, which shows the subthreshold
#     precursor as well. Right: the firing set — the wavefront itself — with the
#     cone predicted from [1] overlaid as dashed lines. The lines are a
#     PREDICTION drawn on top, not a fit to what is displayed.
# ---------------------------------------------------------------------------
println("\n[3] rendering mach_cone.gif ...")
const LO = -3.0f0
const HI =  1.6f0
cone_d = collect(0:4:Float64(N))
anim = @animate for t in 1:STRIDE:L
    pr = particle_row(t)
    rel = log10.(clamp.(abs.(@view Z[:, :, t]) ./ θ, 1f-12, Inf32))

    p1 = heatmap(clamp.(rel, LO, HI); c = :inferno, clims = (LO, HI),
                 aspect_ratio = 1, axis = false, colorbar = false, yflip = true,
                 title = "field   log₁₀|z|/θ")
    p2 = heatmap(Float32.(@view fire[:, :, t]); c = cgrad([:black, :cyan]),
                 clims = (0, 1), aspect_ratio = 1, axis = false, colorbar = false,
                 yflip = true, title = "firing front + predicted cone")
    # Cone edges: from the particle, running BACKWARD at α to the track.
    ys = pr .- cone_d
    keep = ys .>= 1
    for s in (+1, -1)
        xs = (N ÷ 2) .+ s .* cone_d .* tan(ALPHA)
        ok = keep .& (xs .>= 1) .& (xs .<= N)
        plot!(p2, xs[ok], ys[ok]; lc = :orange, ls = :dash, lw = 1.5, label = "")
    end
    scatter!(p2, [N ÷ 2], [pr]; mc = :white, ms = 3, msw = 0, label = "")

    plot(p1, p2; layout = (1, 2), size = (940, 500),
         plot_title = @sprintf("t = %3d   particle row %3.0f   v = %.1f,  u = %.2f  ⇒  Mach %.2f,  α = %.1f°",
                               t, pr, VP, U_MEDIUM, MACH, rad2deg(ALPHA)))
end
gif(anim, joinpath(OUTDIR, "mach_cone.gif"); fps = 12)
println("    saved mach_cone.gif")

# ---------------------------------------------------------------------------
# [4] The numbers behind the animation, so the gif is not the only evidence.
#     For a Mach cone the instantaneous front half-width must grow LINEARLY with
#     distance behind the particle, at slope tan α.
#
#     THE COMPARISON IS ASYMPTOTIC, and this is physics rather than bookkeeping.
#     Measured/predicted half-width converges to 1 FROM BELOW — 0.61, 0.81, 0.89,
#     0.98, 1.03, 1.02 at d = 160…400. The near cone really is narrower than
#     asin(u/v) predicts, because the circles emitted most recently are small and
#     strongly curved, and an excitable front obeys v = v_plane − D·κ: they have
#     been expanding at less than the plane-wave speed the prediction uses. Only
#     far behind the particle, where the defining circles are large and nearly
#     flat, does the envelope reach the textbook angle.
#
#     Two consequences for measurement. (i) Compare at LARGE d only; a
#     through-origin fit over the whole range is biased low by the curved near
#     field and will report a too-narrow cone. (ii) The cone is only fully
#     developed within d_max = v·L·(1 − sin²α): the envelope at distance d is set
#     by the circle emitted x = d/(1−sin²α) behind (maximise √(x²sin²α − (x−d)²)
#     over x), so past d_max the track has not existed long enough to supply it,
#     and the width falls off as missing data rather than as geometry.
#
# ---------------------------------------------------------------------------
println("\n[4] cone geometry, measured on the final frame")
t_end = L
pr_end = particle_row(t_end)
d_max = VP * L * (1 - sin(ALPHA)^2)          # cone fully developed only within this
ds = Float64[]; hw = Float64[]
for d in 20:10:Int(min(pr_end - 2, N - 2))
    i = round(Int, pr_end - d); (1 <= i <= N) || continue
    js = [j for j in 1:N if fire[i, j, t_end] > 0f0]
    isempty(js) && continue
    push!(ds, d); push!(hw, max(maximum(js) - N ÷ 2, N ÷ 2 - minimum(js)))
end
pred = tan(ALPHA) .* ds
println("      d    measured   d·tanα   ratio")
for k in eachindex(ds)
    (ds[k] % 40 == 0) || continue
    @printf("    %5.0f  %8.0f  %8.1f   %.3f%s\n", ds[k], hw[k], pred[k], hw[k]/pred[k],
            ds[k] > d_max ? "   (beyond d_max — track too short here)" : "")
end
# Asymptotic comparison: the top quarter of the fully-developed range.
win = (ds .>= 0.75 * d_max) .& (ds .<= d_max)
ratio = mean(hw[win] ./ pred[win])
slope = sum(ds[win] .* hw[win]) / sum(ds[win] .^ 2)
@printf("\n    predicted tan α = %.3f  (α = %.1f°, from u measured independently in [1])\n",
        tan(ALPHA), rad2deg(ALPHA))
@printf("    asymptotic window d ∈ [%.0f, %.0f] (%d points, d_max = v·L·(1−sin²α))\n",
        0.75*d_max, d_max, count(win))
@printf("    measured  tan α = %.3f  (α = %.1f°)   mean measured/predicted = %.3f  (%+.1f%%)\n",
        slope, rad2deg(atan(slope)), ratio, 100*(ratio-1))
@printf("    implied medium speed from the cone alone: u = v·sin α = %.3f  (direct measurement %.3f)\n",
        VP * sin(atan(slope)), U_MEDIUM)

plt = plot(ds, hw; seriestype = :scatter, ms = 3, mc = :steelblue, label = "measured front half-width",
           xlabel = "distance behind particle (sites)", ylabel = "front half-width (sites)",
           title = @sprintf("Mach cone geometry — v=%.1f, u=%.2f, Mach %.2f", VP, U_MEDIUM, MACH),
           legend = :topleft, size = (760, 480), left_margin = 5Plots.mm, bottom_margin = 5Plots.mm)
plot!(plt, ds, tan(ALPHA) .* ds; lc = :orange, lw = 2, ls = :dash,
      label = @sprintf("predicted  tan α = %.3f (α=%.1f°)", tan(ALPHA), rad2deg(ALPHA)))
plot!(plt, ds, slope .* ds; lc = :crimson, lw = 2,
      label = @sprintf("asymptotic fit  tan α = %.3f (α=%.1f°)", slope, rad2deg(atan(slope))))
vline!(plt, [0.75*d_max, d_max]; lc = :gray, ls = :dot, lw = 1, label = "asymptotic window")
savefig(plt, joinpath(OUTDIR, "mach_cone_geometry.png"))
println("    saved mach_cone_geometry.png")
