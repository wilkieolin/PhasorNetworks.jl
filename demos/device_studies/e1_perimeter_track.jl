# E1 — speed, heading and crossing point from the perimeter alone.
#
# Tracks enter through the top edge at different positions and headings, at
# Mach 2 to Mach 49, with 25 ps FWHM detector jitter. Only the 4N−4 perimeter
# sites are read. `perimeter_fit` fits the exact first-arrival model (cone
# plus entry circle, clamped to the track inside the sheet) for entry point,
# heading and speed, with the medium speed u calibrated separately.

include("common.jl")
using Plots
backend, dev = load_gpu_backend()

const N = parse(Int, get(ENV, "E1_N", "512"))
const SEEDS = parse(Int, get(ENV, "E1_SEEDS", "3"))
const JIT = 25.0
cell = RFTransponder()
sheet = DeviceSheet(cell, N, N)
u = medium_speed(sheet, dev; nsteps = 7000, rmax = 0.4N)
@printf("medium u = %.5f sites/ps (%.4fc); sheet %d² = %.1f mm\n", u, u / C_SITES, N, N * P_UM / 1000)

rows = NamedTuple[]
@printf("%6s %6s %7s | %8s %9s %9s %7s\n", "v/c", "θ°", "x_entry", "dv/v %", "dθ (deg)", "d_entry", "rms ps")
for f in (0.043, 0.3, 1.0), θd in (0.0, 15.0, 30.0, 45.0)
    v = f * C_SITES; θ = deg2rad(θd)
    for seed in 1:SEEDS
        rng = Xoshiro(100seed + round(Int, θd))
        x0 = (1, round(Int, N * (0.3 + 0.2rand(rng))))
        d = particle_drive(N, N; dt_site = 1 / v, angle = θ, x0, width = 3, jitter_fwhm_ps = JIT, rng)
        se = exit_length(Float64.(x0), θ, N, N)
        nst = round(Int, se / v + 1.25 * N / u + 500)
        tr = sheet_simulate(sheet, nst; drive = d, device = dev)
        fit = perimeter_fit(tr.t_first, u)
        dv = 100 * (fit.v / v - 1)
        dθ = rad2deg(rem2pi(fit.θ - θ, RoundNearest))
        # crossing error: perpendicular distance from the true entry to the fitted track line
        ry, rx = x0[1] - fit.e[1], x0[2] - fit.e[2]
        de = abs(-ry * sin(fit.θ) + rx * cos(fit.θ))
        @printf("%6.3f %6.1f %7d | %+8.2f %+9.3f %9.2f %7.2f\n", f, θd, x0[2], dv, dθ, de, fit.rms)
        push!(rows, (; v_over_c = f, theta_deg = θd, x_entry = x0[2], seed, dv_pct = dv, dtheta_deg = dθ,
                       d_entry_sites = de, rms_ps = fit.rms, n_perim = fit.n))
        flush(stdout)
    end
end
write_tsv("e1_perimeter_track.tsv", rows)

println("\nsummary (rms over headings and seeds)")
summ = NamedTuple[]
for f in (0.043, 0.3, 1.0)
    r = filter(x -> x.v_over_c == f, rows)
    s = (; v_over_c = f, sigma_v_pct = sqrt(mean(x.dv_pct^2 for x in r)), sigma_theta_deg = sqrt(mean(x.dtheta_deg^2 for x in r)),
         sigma_entry_um = P_UM * sqrt(mean(x.d_entry_sites^2 for x in r)), worst_theta_deg = maximum(abs(x.dtheta_deg) for x in r))
    @printf("    v = %.3fc: σv/v = %.2f%%, σθ = %.3f° (worst %.3f°), σ_crossing = %.0f µm\n",
            f, s.sigma_v_pct, s.sigma_theta_deg, s.worst_theta_deg, s.sigma_entry_um)
    push!(summ, s)
end
write_tsv("e1_summary.tsv", summ)

# figure: one example — first-spike map with the true and fitted tracks
v = 0.3C_SITES; θ = deg2rad(30); x0 = (1, round(Int, 0.35N))
d = particle_drive(N, N; dt_site = 1 / v, angle = θ, x0, width = 3, jitter_fwhm_ps = JIT, rng = Xoshiro(5))
tr = sheet_simulate(sheet, round(Int, exit_length(Float64.(x0), θ, N, N) / v + 1.25N / u + 500); drive = d, device = dev)
fit = perimeter_fit(tr.t_first, u)
tf = copy(tr.t_first); tf[.!isfinite.(tf)] .= NaN
p = heatmap(tf; c = :viridis, yflip = true, aspect_ratio = 1, axis = false, colorbar_title = "first spike (ps)",
            title = @sprintf("perimeter-only fit: v %.4f vs %.4f, θ %.2f° vs 30°", fit.v / C_SITES, v / C_SITES, rad2deg(fit.θ)))
for (e, th, col, ls, lab) in ((Float64.(x0), θ, :white, :solid, "true track"), (fit.e, fit.θ, :red, :dash, "fit (perimeter only)"))
    L = exit_length(e, th, N, N)
    plot!(p, [e[2], e[2] + L * sin(th)], [e[1], e[1] + L * cos(th)]; lc = col, ls, lw = 2, label = lab)
end
P = perimeter(N, N)
scatter!(p, [q[2] for q in P], [q[1] for q in P]; ms = 1, mc = :orange, msw = 0, label = "read-out sites")
savefig(plot(p; size = (640, 600)), joinpath(STUDY_OUT, "e1_perimeter_example.png"))
println("    saved e1_perimeter_example.png")
