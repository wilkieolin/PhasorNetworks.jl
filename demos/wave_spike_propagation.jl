# wave_spike_propagation.jl — animate what `transmit = :spike` actually does.
#
# The point of this demo is that spike-mode propagation has TWO phases, and a
# single |z| heatmap conflates them:
#
#   1. Subthreshold. Nothing has fired. |z| ≪ θ, so the emit is ≈ z/θ and the
#      sheet is exactly the LINEAR medium at gain g/θ. An impulse gives a clean
#      concentric ring expanding at ≈0.55 sites/period — identical, to 4 s.f.,
#      to a `:potential` sheet at the same effective gain. This is a real
#      traveling wave but it is not a spiking one.
#   2. Suprathreshold. Sites start crossing θ and the emit saturates. The firing
#      set is a CLOSED, THIN, CONSTANT-SPEED RING: 72/72 angular sectors lit and
#      radial sd/mean 0.04–0.08, sustained from t=100 to the wrap at t=177 while
#      the radius marches 104.9 → 236.2. That is the pebble-in-pond behaviour, in
#      the firing activity and not just the field.
#
#      It travels at +1.844 sites/period — 2.1× the subthreshold ring. The two
#      are different objects: phase 1 is the linear group velocity, phase 2 is a
#      nonlinear ignition front. Do not quote one as "the" spike wave speed.
#
# So the animation draws the field and the firing set side by side, with a
# radial profile that shows |z| climbing toward θ. Without the firing overlay
# you would read phase 1's ring as "spike mode makes traveling waves" and miss
# that nothing has spiked yet.
#
# SHEET SIZE IS LOAD-BEARING. On a 128² torus the ring reaches the wrap radius
# (√2·64 ≈ 90) at about the same time it starts firing, so the firing set is
# measured *after* it has self-intersected and reads as disconnected arcs
# (17–20 of 36 sectors) — an artifact that looks exactly like a real negative
# result. At 256² the wrap is at t≈110 and there is a clean window. If you
# shrink N, re-derive the window before trusting any ring statistic.
#
# Run:  julia --project=. demos/wave_spike_propagation.jl
# Writes demos/wave_out/spike_propagation.gif and spike_regime_compare.gif

ENV["GKSwstype"] = "100"                     # headless GR

using PhasorNetworks, Lux, Random, Printf, Statistics
using Plots
using FFTW: fftshift

const OUTDIR = joinpath(@__DIR__, "wave_out")
isdir(OUTDIR) || mkpath(OUTDIR)

# Sheet size is the binding constraint on how long you can watch, not the
# dynamics. The firing ring self-intersects once its radius passes N/2, so with
# a ring speed of ≈1 site/period and first spike at t≈83 the clean window is
# roughly  t ∈ [83, 83 + (N/2 − 70)]:
#
#     N = 128 →  ~0 steps   (wraps as it starts firing; reads as broken arcs)
#     N = 256 →  ~30 steps
#     N = 512 →  ~190 steps
#
# Cost is L·N²·12 bytes per regime with keep_fields (CPU; FFTW, no GPU):
# 512²×320 = 1.0 GB each, 3.1 GB for all three.
const N      = parse(Int, get(ENV, "SHEET_N", "512"))
const L      = parse(Int, get(ENV, "SHEET_L", "320"))
const STRIDE = parse(Int, get(ENV, "SHEET_STRIDE", "3"))
@info "sheet $(N)×$(N), $L steps" keep_fields_GB = round(3 * L * N^2 * 12 / 2^30; digits = 2)

seed_z() = (z = zeros(ComplexF32, N, N); z[1, 1] = 1f0 + 0im; z)

# Wrapped radius, matching the sheet's own rgrid convention (origin at (1,1)).
const RG = let l = PhasorWaveSheet(N, N); Lux.setup(Xoshiro(1), l)[2].rgrid end
const RGS = fftshift(RG)

"Ring-averaged |z| in unit-width annuli."
function radial_profile(a::AbstractMatrix, rmax::Int)
    p = zeros(Float32, rmax); c = zeros(Int, rmax)
    @inbounds for j in 1:N, i in 1:N
        b = floor(Int, RG[i, j]) + 1
        b <= rmax || continue
        p[b] += a[i, j]; c[b] += 1
    end
    return p ./ max.(c, 1)
end

# ---------------------------------------------------------------------
# Run the three regimes, keeping the θ field so the firing set is exact.
# For :local, θ = θ_g · θ_l varies per site (spread reaches ~1.75×), so
# thresholding against the scalar θ_g would draw the wrong set.
# ---------------------------------------------------------------------
println("Running $(N)×$(N) for $L steps, three regimes...")
runs = Dict{Symbol,Any}()
for hm in (:none, :global, :local)
    l = PhasorWaveSheet(N, N; homeostasis = hm)
    ps, st = Lux.setup(Xoshiro(1), l)
    @time runs[hm] = wave_homeostat_trace(l, ps, st;
                                          z0 = seed_z(), L = L, keep_fields = true)
    tr = runs[hm]
    first_fire = findfirst(>(0f0), tr.fire)
    @printf("  %-8s theta %.2f→%.2f   first spike at t=%s   peak fire %.2f%%\n",
            hm, tr.theta_g[1], tr.theta_g[end],
            first_fire === nothing ? "never" : string(first_fire),
            100 * maximum(tr.fire))
end

# Max wrapped radius on an N² torus is √2·N/2 ≈ 181, NOT N/2. Capping the
# profile at 128 truncates the corners and makes the peak stall there.
const RMAX  = ceil(Int, sqrt(2) * N / 2) + 1

# Floor for the log colour scale, in decades below threshold. The seeded impulse
# starts ~5 decades down, so anything shallower clips the first half of the run.
const LO = -5.0f0

# ---------------------------------------------------------------------
# Is the firing ring SUSTAINED? Three things have to hold together, and
# each fails differently:
#   closure   — all angular sectors lit (else it has broken into arcs)
#   thinness  — small radial sd/mean (else it has smeared into a disc)
#   constant speed — radius linear in t (else it is decelerating/stalling)
# A ring can stay closed while smearing, or stay thin while fragmenting, so
# reporting one of the three would not settle the question.
# ---------------------------------------------------------------------
const ANG = [atan(Float32((i-1) > N ÷ 2 ? i-1-N : i-1),
                  Float32((j-1) > N ÷ 2 ? j-1-N : j-1)) for i in 1:N, j in 1:N]
const NSECT = 72

function ring_quality(z, θf)
    a = abs.(z); act = a .> θf
    n = count(act)
    n == 0 && return (; fire = 0f0, r = NaN32, rel_sd = NaN32, sectors = 0)
    rs = RG[act]
    r̄ = mean(rs)
    sect = length(unique(min.(NSECT,
        1 .+ floor.(Int, (ANG[act] .+ Float32(pi)) ./ (2Float32(pi) / NSECT)))))
    return (; fire = Float32(n) / N^2, r = r̄, rel_sd = std(rs) / r̄, sectors = sect)
end

# Wrap must be derived from whichever ring reaches r = N/2 FIRST. Once the sheet
# ignites, the firing front outruns the |z|-profile peak (nonlinear ignition
# front vs linear group velocity), so keying the window off the |z| peak alone
# labels ~15 steps of already-wrapped data as clean.
const T_WRAP = let
    rz(t) = argmax(radial_profile(abs.(view(runs[:local].z, :, :, t)), RMAX)) - 1
    rf(t) = ring_quality(view(runs[:local].z, :, :, t),
                         view(runs[:local].theta, :, :, t)).r
    wz = something(findfirst(t -> rz(t) >= N ÷ 2, 1:L), L)
    wf = something(findfirst(t -> (r = rf(t); !isnan(r) && r >= N ÷ 2), 1:L), L)
    min(wz, wf)
end
@info "torus wrap at t = $T_WRAP of $L (earliest of |z|-peak and firing front)"

println("\nRing quality over time (firing set, :local).  [wrap at t=$T_WRAP]")
println("    t   fire%   mean r   sd/mean   sectors/72   verdict")
qs = NamedTuple[]
for t in 60:10:L
    q = ring_quality(view(runs[:local].z, :, :, t), view(runs[:local].theta, :, :, t))
    push!(qs, merge(q, (; t)))
    if q.sectors == 0
        @printf("  %3d   silent%s\n", t, t > T_WRAP ? "   (post-wrap)" : "")
    else
        v = q.sectors >= 0.9NSECT && q.rel_sd < 0.15 ? "CLOSED THIN RING" :
            q.sectors >= 0.5NSECT ? "closed but smeared" : "fragmented"
        @printf("  %3d  %6.2f%%  %7.1f  %8.3f  %11d   %s%s\n",
                t, 100q.fire, q.r, q.rel_sd, q.sectors, v,
                t > T_WRAP ? "   (post-wrap)" : "")
    end
end

# Speed of the FIRING ring specifically (a level set, so it need not equal the
# |z|-peak speed), fit only inside the clean window.
let cl = [q for q in qs if q.sectors > 0 && q.t <= T_WRAP]
    if length(cl) >= 3
        ts = Float32[q.t for q in cl]; rs = Float32[q.r for q in cl]
        sl = sum((ts .- mean(ts)) .* (rs .- mean(rs))) / sum((ts .- mean(ts)) .^ 2)
        res = rs .- (mean(rs) .+ sl .* (ts .- mean(ts)))
        @printf("  firing-ring speed (t=%d..%d): %+.3f sites/period, fit residual %.2f sites\n",
                Int(first(ts)), Int(last(ts)), sl, maximum(abs, res))
    end
end

# ---------------------------------------------------------------------
# Main animation: field | firing set | radial profile, for :local
# ---------------------------------------------------------------------
println("Rendering spike_propagation.gif ...")
tr = runs[:local]
amax = maximum(abs.(tr.z))

anim = @animate for t in 1:STRIDE:L
    zf   = view(tr.z, :, :, t)
    θf   = view(tr.theta, :, :, t)
    a    = abs.(zf)
    fire = a .> θf
    nfire = count(fire)

    # (1) the field, as log10(|z|/θ). LOG is not optional here: the ring grows
    # from ~1e-5 of threshold to threshold over ~100 steps, so on a linear scale
    # the entire subthreshold phase — which is most of the propagation — renders
    # as a black screen. 0.0 on this scale is exactly at threshold.
    rel = log10.(fftshift(a ./ θf) .+ 1f-12)
    p1 = heatmap(clamp.(rel, LO, 0.3f0); c = :inferno, clims = (LO, 0.3),
                 aspect_ratio = 1, axis = false, colorbar = true,
                 title = "log₁₀(|z| / θ)    0 = at threshold")

    # (2) the firing set — the thing the ring picture gets wrong
    p2 = heatmap(Float32.(fftshift(fire)); c = cgrad([:black, :cyan]), clims = (0, 1),
                 aspect_ratio = 1, axis = false, colorbar = false,
                 title = @sprintf("firing sites: %d (%.2f%%)", nfire, 100 * nfire / N^2))

    # (3) radial profile vs threshold, same log scale
    pr = log10.(radial_profile(a, RMAX) ./ mean(θf) .+ 1f-12)
    p3 = plot(0:RMAX-1, clamp.(pr, LO, 1f0); lw = 2, label = "log₁₀(⟨|z|⟩/θ)",
              xlabel = "radius (sites)", ylabel = "log₁₀(⟨|z|⟩ / θ)",
              ylims = (LO, 1.0), legend = :bottomright,
              title = "ring position vs threshold")
    hline!(p3, [0.0]; lw = 2, ls = :dash, c = :red, label = "threshold")
    pk = argmax(radial_profile(a, RMAX))
    vline!(p3, [pk - 1]; lw = 1, ls = :dot, c = :gray, label = "peak r = $(pk-1)")
    t > T_WRAP && vline!(p3, [Float64(N ÷ 2)]; lw = 1, c = :orange, label = "torus wrap")

    phase = nfire == 0 ? "SUBTHRESHOLD (nothing has fired — linear medium at g/θ)" :
            t <= T_WRAP ? "FIRING" : "FIRING · ring has wrapped the torus"
    plot(p1, p2, p3; layout = (1, 3), size = (1320, 430),
         plot_title = @sprintf("transmit=:spike  homeostasis=:local   step %3d/%d   %s", t, L, phase),
         plot_titlefontsize = 11, titlefontsize = 9, guidefontsize = 8,
         legendfontsize = 7, bottom_margin = 6Plots.mm, left_margin = 4Plots.mm)
end

gif(anim, joinpath(OUTDIR, "spike_propagation.gif"); fps = 12)
println("  saved spike_propagation.gif")

# ---------------------------------------------------------------------
# Comparison: the same impulse under all three regimes, firing sets only.
# This is where :none (drifts to ~11% firing, scattered) separates from
# :global (regulated but the patch stands) and :local (regulated + travels).
# ---------------------------------------------------------------------
println("Rendering spike_regime_compare.gif ...")
anim2 = @animate for t in 1:STRIDE:L
    ps_ = map((:none, :global, :local)) do hm
        r = runs[hm]
        a = abs.(view(r.z, :, :, t)); θf = view(r.theta, :, :, t)
        fire = a .> θf
        heatmap(clamp.(log10.(fftshift(a ./ θf) .+ 1f-12), LO, 0.3f0);
                c = :inferno, clims = (LO, 0.3),
                aspect_ratio = 1, axis = false, colorbar = false,
                title = @sprintf(":%s   fire %.2f%%   θ̄ %.1f",
                                 hm, 100 * count(fire) / N^2, mean(θf)))
    end
    plot(ps_...; layout = (1, 3), size = (1200, 430),
         plot_title = @sprintf("log₁₀(|z|/θ)  —  same impulse, three regulation modes   step %3d/%d", t, L),
         plot_titlefontsize = 11, titlefontsize = 10)
end
gif(anim2, joinpath(OUTDIR, "spike_regime_compare.gif"); fps = 12)
println("  saved spike_regime_compare.gif")

# ---------------------------------------------------------------------
# The numbers behind the animation, so the gif is not the only evidence.
# ---------------------------------------------------------------------
println("\nRing peak radius (ring-averaged |z|):   [| = torus wrap at t≈$(T_WRAP)]")
for hm in (:none, :global, :local)
    r_of(t) = argmax(radial_profile(abs.(view(runs[hm].z, :, :, t)), RMAX)) - 1
    @printf("  %-8s", hm)
    for t in 20:20:L
        @printf("%s%4d", t == 120 ? " |" : " ", r_of(t))
    end
    ts = Float32.(20:10:T_WRAP)
    rs = Float32[r_of(Int(t)) for t in ts]
    sl = sum((ts .- mean(ts)) .* (rs .- mean(rs))) / sum((ts .- mean(ts)) .^ 2)
    @printf("   →  %+.3f sites/period (t=20..%d)\n", sl, T_WRAP)
end

# The band's own prediction, for comparison. This is only meaningful because θ
# puts the sheet in its subthreshold regime, where dispersion() is exact.
let l = PhasorWaveSheet(N, N), (ps, st) = Lux.setup(Xoshiro(1), PhasorWaveSheet(N, N))
    wt = wave_transport(l, ps, st)
    @printf("\n  band prediction  |v_r(q*)| = %.3f sites/period   verdict = %s (ratio %.3f)\n",
            abs(wt.v_r_star), wt.verdict, wt.transport_ratio)
end
println("  first spike:  :none t=$(findfirst(>(0f0), runs[:none].fire))  " *
        ":global t=$(findfirst(>(0f0), runs[:global].fire))  " *
        ":local t=$(findfirst(>(0f0), runs[:local].fire))")
println("  → the ring starts expanding long before anything fires: up to the first")
println("    spike it is the subthreshold LINEAR medium at gain g/θ, which is why its")
println("    speed matches the band prediction. The firing ring that follows is the")
println("    genuinely spiking wave — closed (72/72 sectors), thin (sd/mean < 0.1).")

# ---------------------------------------------------------------------
# Spiking activity only — no |z| anywhere on this figure.
#
# The field plots answer "where is the energy"; these answer "who spiked, and
# when", which is the question the sheet is actually built to answer and the
# one a magnitude heatmap cannot. In particular the radial raster puts the wave
# speed on the page as a readable slope, and the spike-count map settles whether
# each site fires ONCE as the front passes (a solitary wave) or repeatedly
# (re-entrant / oscillatory) — indistinguishable in any single frame.
# ---------------------------------------------------------------------
println("\nBuilding spike-activity figure ...")

"Boolean spike raster (H,W,L) for one run."
spikes_of(r) = abs.(r.z) .> r.theta

const RBIN = 2                                  # annulus width, sites
const NRB  = cld(RMAX, RBIN)
const RIDX = clamp.(floor.(Int, RG ./ RBIN) .+ 1, 1, NRB)
const RCNT = [count(==(b), RIDX) for b in 1:NRB]

"Fraction of each annulus firing, per step → (NRB, L). The wave is a diagonal."
function radial_raster(S)
    R = zeros(Float32, NRB, size(S, 3))
    @inbounds for t in axes(S, 3), j in 1:N, i in 1:N
        S[i, j, t] && (R[RIDX[i, j], t] += 1f0)
    end
    return R ./ max.(RCNT, 1)
end

"Step at which each site first fires (0 = never)."
function first_spike(S)
    F = zeros(Int32, N, N)
    @inbounds for t in axes(S, 3), j in 1:N, i in 1:N
        (F[i, j] == 0 && S[i, j, t]) && (F[i, j] = t)
    end
    return F
end

S_loc  = spikes_of(runs[:local])
raster = radial_raster(S_loc)
fs     = first_spike(S_loc)
counts = dropdims(sum(S_loc; dims = 3); dims = 3)
# Counts restricted to the CLEAN window. Over the full run the map is dominated
# by post-wrap turbulence (corner sites reach 31 spikes), which buries the
# question actually worth answering: how many times does a site fire as the
# front passes it, once?
counts_clean = dropdims(sum(view(S_loc, :, :, 1:T_WRAP); dims = 3); dims = 3)

let never = counts .== 0, inner = RG .< 20
    @printf("  ever spiked: %.1f%%   median spikes/site (clean window): %d   max (full): %d\n",
            100 * count(>(0), counts) / N^2,
            Int(median(counts_clean[counts_clean .> 0])), maximum(counts))
    @printf("  central disc r<20: %.0f%% NEVER fire — held at |z|/θ ≈ 0.7–0.98 by the\n",
            100 * mean(never[inner]))
    @printf("  surround while the ring reaches 1.03–1.44. Not a threshold effect:\n")
    @printf("  :local's zero-sum term LOWERS θ where nothing fires, and it still never crosses.\n")
end

# (1) radial raster: r vs t. A traveling wave is a straight diagonal band and
#     its slope IS the speed, read straight off the axes.
p1 = heatmap((1:L), (0:NRB-1) .* RBIN, raster;
             c = :magma, clims = (0, 0.6), xlabel = "step",
             ylabel = "radius (sites)", colorbar_title = "fraction of annulus firing",
             title = "radial spike raster — slope = wave speed", legend = :bottomright)
plot!(p1, [90, 170], [80.5, 228.0]; lw = 2, ls = :dash, c = :cyan,
      label = "+1.84 sites/period")
hline!(p1, [N ÷ 2]; lw = 1.5, c = :orange, ls = :dot, label = "torus wrap (r = N/2)")

# (2) first-spike latency: concentric bands ⇒ the wave is radial, and the band
#     spacing is the speed. Never-fired sites are WHITE (NaN), which is how the
#     silent central disc shows up — it is a result, not missing data.
lat = Float32.(fftshift(fs)); lat[lat .== 0] .= NaN32
p2 = heatmap(lat; c = :turbo, aspect_ratio = 1, axis = false,
             colorbar_title = "step of first spike",
             title = "first-spike latency  (white = never fired)")
annotate!(p2, N ÷ 2, N ÷ 2 - 34,
          text("silent core\n(r<20, 100% never fire)", 7, :white, :center))

# (3) population rate — ignition, then a regulated plateau
p3 = plot(1:L, 100 .* runs[:none].fire;   lw = 1.5, label = ":none",
          xlabel = "step", ylabel = "% of sheet spiking",
          title = "population firing rate", legend = :topright)
plot!(p3, 1:L, 100 .* runs[:global].fire; lw = 1.5, label = ":global")
plot!(p3, 1:L, 100 .* runs[:local].fire;  lw = 1.5, label = ":local")
hline!(p3, [2.0]; lw = 2, ls = :dash, c = :black, label = "target 2%")
vline!(p3, [T_WRAP]; lw = 1.5, ls = :dot, c = :orange, label = "wrap")

# (4) spikes per site while the front passes — once, or repeatedly?
p4 = heatmap(Float32.(fftshift(counts_clean)); c = :viridis, aspect_ratio = 1,
             axis = false, colorbar_title = "spikes in t ≤ $T_WRAP",
             title = "spike count per site (clean window only)")

plt = plot(p1, p2, p3, p4; layout = (2, 2), size = (1250, 900),
           titlefontsize = 10, guidefontsize = 8, legendfontsize = 7,
           left_margin = 6Plots.mm, bottom_margin = 6Plots.mm,
           plot_title = "transmit=:spike, homeostasis=:local — spiking activity only ($(N)², $L steps)",
           plot_titlefontsize = 12)
savefig(plt, joinpath(OUTDIR, "spike_activity.png"))
println("  saved spike_activity.png")

# Companion: the firing set alone, full frame, nothing else.
println("Rendering spike_raster.gif ...")
anim3 = @animate for t in 1:STRIDE:L
    n = count(view(S_loc, :, :, t))
    heatmap(Float32.(fftshift(view(S_loc, :, :, t)));
            c = cgrad([:black, :cyan]), clims = (0, 1), aspect_ratio = 1,
            axis = false, colorbar = false, size = (620, 660),
            title = @sprintf("spikes only — step %3d/%d\n%d sites (%.2f%%)%s",
                             t, L, n, 100n / N^2, t > T_WRAP ? "  · wrapped" : ""),
            titlefontsize = 11)
end
gif(anim3, joinpath(OUTDIR, "spike_raster.gif"); fps = 12)
println("  saved spike_raster.gif")

println("\nDone. Figures and GIFs in $(OUTDIR)")
