# E4 — the wake a particle leaves, by Mach number.
#
# 384² sheet, default medium (u ≈ 0.048 sites/ps). One frame per Mach number,
# taken when the particle (or, subsonic, the medium's front) has crossed ~80%
# of the sheet: the refractory wake
# (time since each cell's spike), with the particle and the predicted cone.
#   Mach 0.5 — subsonic: the medium outruns the particle; the first-spike map
#              is set by the entry circle and carries no information about v.
#   Mach 1.5, 3, 10 — cones of half-angle asin(1/M).

include("common.jl")
using Plots
backend, dev = load_gpu_backend()

const N = 384; const ROW0 = 10
cell = RFTransponder()
sheet = DeviceSheet(cell, N, N)
u = medium_speed(sheet, dev; nsteps = 5000, rmax = 0.4N)
panels = []; rows = NamedTuple[]
for M in (0.5, 1.5, 3.0, 10.0)
    v = M * u
    # subsonic: the medium's own front outruns the particle, so snapshot while it is still on the sheet
    t_end = 0.8 * (N - ROW0) / max(v, u)
    nst = round(Int, t_end)
    d = particle_drive(N, N; dt_site = 1 / v, x0 = (ROW0, N ÷ 2), width = 3)
    tr = sheet_simulate(sheet, nst; drive = d, device = dev)
    age = Float32(nst) .- tr.t_first
    wake = ifelse.((age .>= 0) .& (age .<= cell.t_ref), age, NaN32)
    pr = ROW0 + v * nst
    p = heatmap(wake; c = :magma, clims = (0, cell.t_ref), yflip = true, aspect_ratio = 1, axis = false,
                colorbar = false, background_color_inside = :black,
                title = @sprintf("Mach %.1f  (v = %.3fc)", M, v / C_SITES))
    if M > 1
        α = asin(1 / M); dd = 0:4:N
        for s in (1, -1)
            xs = (N ÷ 2) .+ s .* dd .* tan(α); ys = pr .- dd
            ok = (ys .>= ROW0) .& (xs .>= 1) .& (xs .<= N)
            plot!(p, xs[ok], ys[ok]; lc = :cyan, ls = :dash, lw = 1, label = "")
        end
    end
    scatter!(p, [N ÷ 2], [pr]; mc = :white, ms = 4, msw = 0, label = "")
    push!(panels, p)
    lead = maximum(y for y in 1:N, x in 1:N if isfinite(tr.t_first[y, x])) - pr
    push!(rows, (; mach = M, v_over_c = v / C_SITES, fired_frac = count(isfinite, tr.t_first) / N^2,
                   front_lead_sites = lead, half_angle_deg = M > 1 ? rad2deg(asin(1 / M)) : NaN))
    @printf("Mach %4.1f: fired %.1f%%, foremost spike %+.0f sites ahead of the particle\n", M,
            100rows[end].fired_frac, lead)
end
write_tsv("e4_wakes.tsv", rows)
savefig(plot(panels...; layout = (1, 4), size = (1500, 420)), joinpath(STUDY_OUT, "e4_wakes.png"))
println("    saved e4_wakes.png")
