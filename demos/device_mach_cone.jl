# device_mach_cone.jl — a supersonic particle's Mach cone on a sheet of
# superconducting resonate-and-fire transponders coupled by delay lines.
#
# The device-level counterpart of `wave_mach_cone.jl`. That demo runs on
# ExcitableWaveSheet: an abstract phasor medium stepped once per carrier
# period, with conduction delay as a carrier phase. This one runs on
# `DeviceSheet`: every site is an `RFTransponder` (10 GHz LC tank, Q = 21,
# switching nanowire), every coupling is a kinetic-inductance line with a real
# delay, and time is in picoseconds. The particle is the hardware ladder's
# design case: it excites successive sites 10 ps apart (0.1 sites/ps).
#
#     julia -t 16 --project=scripts demos/device_mach_cone.jl
#
# PHASOR_BACKEND=cpu forces the CPU; otherwise the GPU is used if present
# (`scripts/gpu_backend.jl`, which caps CUDA memory at 16 GiB).
#
# WHAT IS DIFFERENT FROM THE PHASOR DEMO, all of it measured:
#
#   1. THE MACH NUMBER IS A LAYOUT CHOICE. The medium's front speed is
#      u ≈ 1/(ps_per_site + ℓ), with ℓ ≈ 11 ps/site of hop latency (the tank's
#      rise to threshold); see the calibration table in src/device_sheet.jl.
#      At the default 10 ps of line per pitch, u ≈ 0.048 sites/ps and a
#      10 ps/site particle is Mach 2.1. Lengthen the lines (MACH_PPS) and the
#      cone narrows — the delay lines are the knob, not a device parameter.
#
#   2. A ONE-SITE TRACK IS ENOUGH. The phasor demo needed width 2.0 because a
#      one-site line sat below its critical nucleus and launched nothing
#      sideways. Here each hit cell fires a full pulse into ~28 lines and the
#      neighbours along a straight track receive several of them nearly in
#      phase; widths 1, 2 and 3 give the same cone angle to 0.1° (256² probe).
#
#   3. THE FIRST-SPIKE MAP IS THE CONE. In the phasor sheet the ever-fired set
#      was a union of discs and a plane fit to it returned v = 5.9 against a
#      true 3.0, so that demo had to work from instantaneous frames. Here every
#      site fires exactly once and colliding fronts annihilate, so a site's
#      first spike is the first arrival over all circles emitted along the
#      track — the Huygens envelope — and behind the particle it obeys
#
#          t_first(s, ρ) = s/v + β·ρ,     β = √(1/u² − 1/v²) = cot α / v
#
#      (s along the track, ρ off it), valid where s ≥ ρ·tan α (the tangent
#      point lies on the track rather than before its start). A plane fit to
#      t_first therefore measures α directly, from every site in the cone at
#      once, and is checked against α = asin(u/v) with u measured
#      independently from a radial seed. The frame half-width check of the
#      phasor demo is kept as a second, independent measurement.
#
# Cost at the default 512²: link arrays 3·H·W·K·4 B = 88 MB, history ≈ 35 MB,
# frames F·H·W·4 B ≈ 0.26 GB (F ≈ 245) — well under 1 GB total. ~1.8 ms/step
# on the GB10 GPU (~10 s per run), ~19 ms/step on 16 CPU threads (~95 s).
#
# Writes demos/wave_out/device_mach_cone.gif, device_mach_cone_geometry.png.

ENV["GKSwstype"] = "100"                     # headless GR

using PhasorNetworks, Random, Printf, Statistics
using Plots
include(joinpath(@__DIR__, "..", "scripts", "gpu_backend.jl"))

const OUTDIR = joinpath(@__DIR__, "wave_out")
isdir(OUTDIR) || mkpath(OUTDIR)

const N      = parse(Int, get(ENV, "MACH_N", "512"))
const PPS    = parse(Float64, get(ENV, "MACH_PPS", "10"))      # line delay per pitch, ps
const DTSITE = parse(Float64, get(ENV, "MACH_DTSITE", "10"))   # particle: ps per site
const WIDTH  = parse(Float64, get(ENV, "MACH_WIDTH", "1"))     # track width, sites
const FRAME  = parse(Int, get(ENV, "MACH_FRAME", "20"))        # ps per animation frame
const JITTER = parse(Float64, get(ENV, "MACH_JITTER", "25"))   # robustness run: SNSPD FWHM, ps
const DROP   = parse(Float64, get(ENV, "MACH_DROPOUT", "0.5")) # robustness run: dead fraction
const ROW0   = 12                                              # entry row
const VP     = 1 / DTSITE                                      # sites/ps
const DT     = 1.0                                             # ps per step

backend, dev = load_gpu_backend()
cell  = RFTransponder()
sheet = DeviceSheet(cell, N, N; dt = DT, ps_per_site = PPS)
nsteps = round(Int, (N - ROW0 - 8) * DTSITE / DT)              # particle stops 8 rows short
F = nsteps ÷ round(Int, FRAME / DT)
@info "device sheet on $backend" sheet steps = nsteps frames = F GB = round(
    (3 * N^2 * n_links(sheet) * 4 + N^2 * sheet.NB * 4 + F * N^2 * 4) / 2^30; digits = 2)

# ---------------------------------------------------------------------------
# [0] The physical dictionary — what this sheet would be on a chip.
# ---------------------------------------------------------------------------
pe   = phasor_equivalent(cell)
line = delay_line_length(PPS)                    # NbTiN, 8.5 pH/sq, 50 nm dielectric
pitch_um = line.length_um                        # a straight line spanning one pitch
ll   = link_lengths(sheet)
println("\n[0] device and layout")
println("    ", cell)
@printf("    phasor equivalent: T = %.2f ps, leakage λT = %.4f (the phasor layers' default is 0.15)\n",
        pe.t_period, pe.leakage)
@printf("    %g ps of line per pitch at %.0f ps/mm ⇒ pitch %.0f µm, sheet %.1f × %.1f mm\n",
        PPS, line.ps_per_mm, pitch_um, N * pitch_um / 1000, N * pitch_um / 1000)
@printf("    %d lines/site, %.1f mm of line per interior site, %.1f m in total\n",
        n_links(sheet), maximum(ll.per_site_mm), ll.total_m)
@printf("    particle: %g ps/site ⇒ %.2e m/s (%.3f c)\n", DTSITE, pitch_um * 1e-6 / (DTSITE * 1e-12),
        pitch_um * 1e-6 / (DTSITE * 1e-12) / 299792458)

# ---------------------------------------------------------------------------
# [1] Medium front speed, measured INDEPENDENTLY of the particle so the
#     predicted cone is not fitted from the data it is checked against.
# ---------------------------------------------------------------------------
println("\n[1] medium front speed from a radial seed (no particle) ...")
cen = (N ÷ 2, N ÷ 2)
rad = sheet_simulate(sheet, round(Int, 0.45N / 0.04 / DT); drive = seed_drive(N, N; radius = 2),
                     device = dev)
const U = arrival_speed(rad.t_first; center = cen, rmin = 15, rmax = 0.4N)[1]
an = front_anisotropy(rad.t_first; center = cen, rmin = 15, rmax = 0.4N)
const MACH  = VP / U
const ALPHA = asin(clamp(U / VP, 0, 1))
@printf("    u = %.4f sites/ps (diagonal/axis %.3f)  ⇒  Mach %.2f,  α = asin(u/v) = %.1f°,  β = %.2f ps/site\n",
        U, an.ratio, MACH, rad2deg(ALPHA), sqrt(1 / U^2 - 1 / VP^2))
MACH > 1 || error("particle is subsonic (Mach $MACH) — no cone; lower MACH_DTSITE or raise MACH_PPS")

# ---------------------------------------------------------------------------
# [2] The particle.
# ---------------------------------------------------------------------------
println("\n[2] traversing ...")
drive = particle_drive(N, N; dt_site = DTSITE, x0 = (ROW0, N ÷ 2), width = WIDTH)
t2 = @elapsed tr = sheet_simulate(sheet, nsteps; drive, device = dev,
                                  frame_every = round(Int, FRAME / DT))
@printf("    %.1f s;  fired: %.1f%% of sites;  max spikes per site: %d\n",
        t2, 100 * count(isfinite, tr.t_first) / N^2, Int(maximum(tr.n_spikes)))
particle_row(t) = ROW0 + VP * t

# ---------------------------------------------------------------------------
# [3] Cone geometry from the first-spike map: fit t_first − s/v = t₀ + β·ρ
#     over every site behind the particle whose tangent point lies on the track.
# ---------------------------------------------------------------------------
function cone_fit(tf; αref = ALPHA, smin = 30, ρmin = 10)
    xs = Float64[]; ys = Float64[]
    s_end = nsteps * DT * VP
    for x in 1:N, y in 1:N
        t = tf[y, x]; isfinite(t) || continue
        s = y - ROW0; ρ = abs(x - N ÷ 2)
        (smin <= s <= s_end - 20 && ρ >= ρmin && s - ρ * tan(αref) >= 10) || continue
        push!(xs, ρ); push!(ys, t - s / VP)
    end
    μx, μy = mean(xs), mean(ys)
    β = sum((xs .- μx) .* (ys .- μy)) / sum((xs .- μx) .^ 2)
    res = ys .- (μy .+ β .* (xs .- μx))
    return (; β, α = atan(1 / (VP * β)), n = length(xs), rms = sqrt(mean(res .^ 2)))
end
println("\n[3] cone geometry")
fit = cone_fit(tr.t_first)
@printf("    first-spike plane fit over %d sites: β = %.3f ps/site (rms residual %.2f ps)\n",
        fit.n, fit.β, fit.rms)
@printf("    measured α = %.2f°   predicted α = %.2f°   ⇒  implied u = v·sin α = %.4f  (direct %.4f, %+.1f%%)\n",
        rad2deg(fit.α), rad2deg(ALPHA), VP * sin(fit.α), U, 100 * (VP * sin(fit.α) / U - 1))

# Independent check: instantaneous front half-width on the last frame against
# d·tan α. Compared only well inside the developed cone (d ≤ v·T·cos²α), where
# the envelope at distance d is supplied by track that has existed long enough.
fr = tr.fire_frames[:, :, end]
pr_end = particle_row(tr.frame_t[end])
d_max = VP * tr.frame_t[end] * cos(ALPHA)^2
ds = Float64[]; hw = Float64[]
for d in 20:5:Int(floor(min(pr_end - ROW0 - 2, d_max)))
    i = round(Int, pr_end - d); (1 <= i <= N) || continue
    js = findall(>(0), @view fr[i, :]); isempty(js) && continue
    push!(ds, d); push!(hw, max(maximum(js) - N ÷ 2, N ÷ 2 - minimum(js)))
end
win = ds .>= 0.5 * d_max
slope_fr = sum(ds[win] .* hw[win]) / sum(ds[win] .^ 2)
@printf("    last-frame half-width, d ∈ [%.0f, %.0f]: tan α = %.3f (α = %.1f°) vs predicted %.3f\n",
        0.5d_max, d_max, slope_fr, rad2deg(atan(slope_fr)), tan(ALPHA))

# ---------------------------------------------------------------------------
# [4] Robustness: the same particle seen through real detectors — frozen
#     per-site timing jitter and dead pixels (hardware ladder §2.4).
# ---------------------------------------------------------------------------
println("\n[4] detector jitter $(JITTER) ps FWHM, $(round(Int, 100DROP))% dropout ...")
dn = particle_drive(N, N; dt_site = DTSITE, x0 = (ROW0, N ÷ 2), width = max(WIDTH, 3),
                    jitter_fwhm_ps = JITTER, dropout_frac = DROP, rng = Xoshiro(7))
trn = sheet_simulate(sheet, nsteps; drive = dn, device = dev)
fn = cone_fit(trn.t_first)
@printf("    width %g track, %d of its sites alive: α = %.2f° (clean %.2f°, predicted %.2f°), rms %.2f ps\n",
        max(WIDTH, 3), count(isfinite, dn.t_hit), rad2deg(fn.α), rad2deg(fit.α), rad2deg(ALPHA), fn.rms)

# ---------------------------------------------------------------------------
# [5] Figures. Left: the refractory wake — time since each cell's spike, over
#     its t_ref recovery. That band, u·t_ref ≈ 14 sites deep, is what makes
#     colliding fronts annihilate. (There is no subthreshold halo to show:
#     cells drive their lines only when they fire.) Right: the firing front,
#     with the cone PREDICTED from [1] drawn on top — a prediction, not a fit.
#     Frames are max-pooled for display so one-site fronts survive downscaling.
# ---------------------------------------------------------------------------
println("\n[5] rendering ...")
const POOL = max(1, ceil(Int, N / 400))
pool(A, op) = [op(A[(i-1)*POOL+1:min(i*POOL, end), (j-1)*POOL+1:min(j*POOL, end)])
               for i in 1:cld(size(A, 1), POOL), j in 1:cld(size(A, 2), POOL)]
nanmin(a) = (b = filter(!isnan, a); isempty(b) ? NaN32 : minimum(b))
cone_d = collect(0:4:Float64(N))
anim = @animate for k in 1:2:F
    t = tr.frame_t[k]; pr = particle_row(t)
    age = t .- tr.t_first
    wake = ifelse.((age .>= 0) .& (age .<= cell.t_ref), age, NaN32)
    p1 = heatmap(pool(wake, nanmin); c = :magma, clims = (0, cell.t_ref), aspect_ratio = 1,
                 axis = false, colorbar = false, yflip = true, background_color_inside = :black,
                 title = "refractory wake   t − t_spike  (0–$(round(Int, cell.t_ref)) ps)")
    p2 = heatmap(pool(@view(tr.fire_frames[:, :, k]), maximum); c = cgrad([:black, :cyan]),
                 clims = (0, 1), aspect_ratio = 1, axis = false, colorbar = false, yflip = true,
                 title = "firing front + predicted cone")
    ys = (pr .- cone_d) ./ POOL
    for sgn in (+1, -1)
        xs = ((N ÷ 2) .+ sgn .* cone_d .* tan(ALPHA)) ./ POOL
        ok = (ys .>= ROW0 / POOL) .& (xs .>= 0.5) .& (xs .<= N / POOL)
        plot!(p2, xs[ok], ys[ok]; lc = :orange, ls = :dash, lw = 1.5, label = "")
    end
    scatter!(p2, [(N ÷ 2) / POOL], [min(pr, N) / POOL]; mc = :white, ms = 3, msw = 0, label = "")
    plot(p1, p2; layout = (1, 2), size = (940, 500),
         plot_title = @sprintf("t = %4.0f ps   v = %.2f sites/ps,  u = %.3f  ⇒  Mach %.2f,  α = %.1f°",
                               t, VP, U, MACH, rad2deg(ALPHA)))
end
gif(anim, joinpath(OUTDIR, "device_mach_cone.gif"); fps = 12)
println("    saved device_mach_cone.gif")

tf_plot = copy(tr.t_first); tf_plot[.!isfinite.(tf_plot)] .= NaN
p1 = heatmap(tf_plot; c = :viridis, aspect_ratio = 1, yflip = true, axis = false,
             colorbar_title = "first spike (ps)", title = "first-spike map")
contour!(p1, tf_plot; levels = 0:500:Float64(nsteps), c = :white, lw = 0.6, colorbar_entry = false)
for sgn in (+1, -1)
    xs = (N ÷ 2) .+ sgn .* cone_d .* tan(ALPHA); ys = pr_end .- cone_d
    ok = (ys .>= ROW0) .& (xs .>= 1) .& (xs .<= N)
    plot!(p1, xs[ok], ys[ok]; lc = :orange, ls = :dash, lw = 1.5, label = "")
end
p2 = scatter(ds, hw; ms = 3, mc = :steelblue, label = "last-frame half-width",
             xlabel = "distance behind particle (sites)", ylabel = "front half-width (sites)",
             legend = :topleft)
plot!(p2, ds, tan(ALPHA) .* ds; lc = :orange, lw = 2, ls = :dash,
      label = @sprintf("predicted, u from [1]: α = %.1f°", rad2deg(ALPHA)))
plot!(p2, ds, tan(fit.α) .* ds; lc = :crimson, lw = 2,
      label = @sprintf("first-spike plane fit: α = %.1f°", rad2deg(fit.α)))
vline!(p2, [0.5d_max, d_max]; lc = :gray, ls = :dot, label = "comparison window")
plt = plot(p1, p2; layout = (1, 2), size = (1100, 480), left_margin = 4Plots.mm, bottom_margin = 5Plots.mm,
           plot_title = @sprintf("Mach cone on an R&F transponder sheet — %g ps/site lines, Mach %.2f", PPS, MACH))
savefig(plt, joinpath(OUTDIR, "device_mach_cone_geometry.png"))
println("    saved device_mach_cone_geometry.png")
