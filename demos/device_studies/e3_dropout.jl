# E3 — sensitivity to dead pixels.
#
# 512² sheet, particle at 0.1c (Mach 4.9), 25 ps FWHM jitter, side-edge
# readout. Grid: track width {1, 3, 5} × pixel dropout {0, 25, 50, 75}% ×
# line gain {0.30, 0.45}. Dropout is frozen per event (a dead pixel never
# reports); the lattice cells themselves are all alive here — only the sensing
# fails. Bias is relative to the zero-jitter, zero-dropout run of the same gain.

include("common.jl")
using Plots
backend, dev = load_gpu_backend()

const N = 512; const ROW0 = 13; const BASE = N - 25; const SEEDS = 4
cell = RFTransponder()
v = 0.1C_SITES

rows = NamedTuple[]
for G in (0.30, 0.45)
    sheet = DeviceSheet(cell, N, N; gain = G)
    u = medium_speed(sheet, dev; nsteps = 6000, rmax = 0.4N)
    nst = round(Int, BASE / v + (N ÷ 2) * sqrt(1 / u^2 - 1 / v^2) + 1500)
    run(w, drop, seed, jit) = begin
        d = particle_drive(N, N; dt_site = 1 / v, x0 = (ROW0, N ÷ 2), width = w, jitter_fwhm_ps = jit,
                           dropout_frac = drop, rng = Xoshiro(seed))
        d.t_hit[ROW0+BASE+1:end, :] .= Inf32
        s_h = Float64[]; t_h = Float64[]
        for x in 1:N, y in 1:N
            t = d.t_hit[y, x]; isfinite(t) && (push!(s_h, y - ROW0); push!(t_h, t))
        end
        tr = sheet_simulate(sheet, nst; drive = d, device = dev)
        (1 / lin(s_h, t_h)[1], edge_speed_sides(tr.t_first, u; row0 = ROW0, base = BASE),
         count(isfinite, tr.t_first) / N^2)
    end
    _, cal, _ = run(3, 0.0, 0, 0.0); b = cal / v - 1
    for w in (1, 3, 5), drop in (0.0, 0.25, 0.5, 0.75)
        vt = Float64[]; ve = Float64[]; fr = Float64[]
        for s in 1:SEEDS
            a, e, f = run(w, drop, s, 25.0)
            push!(vt, a); push!(ve, e / (1 + b)); push!(fr, f)
        end
        ok = filter(isfinite, ve)
        r = (; gain = G, width = w, dropout = drop, n_ok = length(ok), track_pct = 100std(vt) / v,
             edge_pct = length(ok) >= 2 ? 100std(ok) / v : NaN,
             edge_bias_pct = isempty(ok) ? NaN : 100(mean(ok) / v - 1), fired_frac = mean(fr))
        @printf("G %.2f  width %d  dropout %3.0f%% | track %.3f%% | edges σ %6.3f%%  bias %+7.3f%%  ok %d/%d  fired %.2f\n",
                G, w, 100drop, r.track_pct, r.edge_pct, r.edge_bias_pct, r.n_ok, SEEDS, r.fired_frac)
        flush(stdout)
        push!(rows, r)
    end
end
write_tsv("e3_dropout.tsv", rows)

plt = plot(; xlabel = "pixel dropout (%)", ylabel = "edge σv/v (%)", yscale = :log10, legend = :topleft,
           title = "dead pixels: edge readout at 0.1c, 25 ps jitter")
for G in (0.30, 0.45), (w, col) in ((1, :crimson), (3, :steelblue), (5, :seagreen))
    r = filter(x -> x.gain == G && x.width == w, rows)
    plot!(plt, [100x.dropout for x in r], [x.edge_pct for x in r]; lc = col, ls = G == 0.30 ? :solid : :dash,
          m = :circle, mc = col, label = "width $w, G=$G")
end
savefig(plot(plt; size = (760, 500), left_margin = 4Plots.mm), joinpath(STUDY_OUT, "e3_dropout.png"))
println("    saved e3_dropout.png")
