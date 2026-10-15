# E2 — how velocity resolution scales with aperture.
#
# Square sheets N = 128 … 800 (aperture L = N − 25 sites along the track), a
# 3-wide track down the centre column, 25 ps FWHM jitter. For each, σv/v of
# the three readouts (device_velocity_readout.jl): two-point TOF over L
# (analytic, ∝ L⁻¹), the fully instrumented track (∝ L^{-3/2}), and the
# two side edges (channel count 2N, lattice bias calibrated at zero jitter).
# The 800² point is re-run here so every point uses the same code.

include("common.jl")
using Plots
backend, dev = load_gpu_backend()

const ROW0 = 13
const JIT = 25.0; const σT = JIT / 2.355
cell = RFTransponder()

function one_run(sheet, N, base, v, seed, jit, u)
    d = particle_drive(N, N; dt_site = 1 / v, x0 = (ROW0, N ÷ 2), width = 3, jitter_fwhm_ps = jit, rng = Xoshiro(seed))
    d.t_hit[ROW0+base+1:end, :] .= Inf32
    s_h = Float64[]; t_h = Float64[]
    for x in 1:N, y in 1:N
        t = d.t_hit[y, x]; isfinite(t) && (push!(s_h, y - ROW0); push!(t_h, t))
    end
    nst = round(Int, base / v + (N ÷ 2) * sqrt(1 / u^2 - 1 / max(v, 1.01u)^2) + 1500)
    tr = sheet_simulate(sheet, nst; drive = d, device = dev)
    return 1 / lin(s_h, t_h)[1], edge_speed_sides(tr.t_first, u; row0 = ROW0, base)
end

rows = NamedTuple[]
for N in (128, 256, 512, 800)
    sheet = DeviceSheet(cell, N, N)
    u = medium_speed(sheet, dev; nsteps = round(Int, 0.45N / 0.045) + 500, rmax = 0.4N)
    base = N - 25
    seeds = N <= 256 ? 16 : (N == 512 ? 10 : 6)
    for f in (0.043, 1.0)
        v = f * C_SITES
        _, cal = one_run(sheet, N, base, v, 0, 0.0, u)
        b = cal / v - 1
        vt = Float64[]; ve = Float64[]
        for s in 1:seeds
            a, e = one_run(sheet, N, base, v, s, JIT, u)
            push!(vt, a); push!(ve, e / (1 + b))
        end
        ve = filter(isfinite, ve)
        r = (; N, L_sites = base, L_mm = base * P_UM / 1000, v_over_c = f, seeds = length(ve),
             tof_pct = 100 * sqrt(2) * σT / (base / v), track_pct = 100std(vt) / v, edge_pct = 100std(ve) / v,
             edge_bias_raw_pct = 100b, edge_resid_bias_pct = 100(mean(ve) / v - 1))
        @printf("N %4d  L %5.1f mm  v %.3fc | TOF %6.3f%%  track %6.3f%%  edges %6.3f%% (raw bias %+.2f%%, resid %+.3f%%)\n",
                N, r.L_mm, f, r.tof_pct, r.track_pct, r.edge_pct, r.edge_bias_raw_pct, r.edge_resid_bias_pct)
        flush(stdout)
        push!(rows, r)
    end
end
write_tsv("e2_aperture_scaling.tsv", rows)

println("\nfitted exponents σ ∝ L^k")
fits = NamedTuple[]
for f in (0.043, 1.0)
    r = filter(x -> x.v_over_c == f, rows)
    lx = log.([x.L_mm for x in r])
    k(sym) = lin(lx, log.([getfield(x, sym) for x in r]))[1]
    q = (; v_over_c = f, k_tof = k(:tof_pct), k_track = k(:track_pct), k_edge = k(:edge_pct))
    @printf("    v = %.3fc: TOF %.2f, track %.2f, edges %.2f\n", f, q.k_tof, q.k_track, q.k_edge)
    push!(fits, q)
end
write_tsv("e2_exponents.tsv", fits)

plt = plot(; xscale = :log10, yscale = :log10, xlabel = "aperture L (mm)", ylabel = "σv/v (%)", legend = :bottomleft,
           title = "velocity resolution vs aperture (25 ps FWHM jitter)")
for (f, m) in ((0.043, :circle), (1.0, :diamond))
    r = filter(x -> x.v_over_c == f, rows); L = [x.L_mm for x in r]
    plot!(plt, L, [x.tof_pct for x in r]; lc = :gray, m, mc = :gray, label = "2-pt TOF, v=$(f)c")
    plot!(plt, L, [x.track_pct for x in r]; lc = :seagreen, m, mc = :seagreen, label = "track array (N² TDC), v=$(f)c")
    plot!(plt, L, [x.edge_pct for x in r]; lc = :crimson, lw = 2, m, mc = :crimson, label = "sheet edges (2N), v=$(f)c")
end
savefig(plot(plt; size = (760, 520), left_margin = 4Plots.mm, bottom_margin = 4Plots.mm), joinpath(STUDY_OUT, "e2_aperture_scaling.png"))
println("    saved e2_aperture_scaling.png")
