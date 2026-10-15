# E6 — shaping wave propagation with the per-link delays.
#
# (a) Anisotropic medium: delays D = √((d_y·dy)² + (d_x·dx)²) with 10 ps per
#     pitch along rows and 20 along columns. A radial seed grows an ellipse;
#     its axis ratio is read off the first-spike map and compared with the
#     plane-front theory of E5 for the two directions.
# (b) Refraction: the left half of the sheet has 10 ps/pitch lines, the right
#     half 25 ps/pitch (D set by the RECEIVER's half). A planar front launched
#     in the left half at incidence θ₁ crosses the interface; its refracted
#     direction θ₂ (plane fit to first-spike times) is compared with Snell's
#     law sin θ₂ / sin θ₁ = u₂ / u₁, u measured on uniform sheets.

include("common.jl")
using Plots
backend, dev = load_gpu_backend()

const N = 320
cell = RFTransponder()
offs = stencil_offsets(3); K = length(offs)

println("[a] anisotropic delays")
Da = Float32[sqrt((10o[1])^2 + (20o[2])^2) for y in 1:N, x in 1:N, o in offs]
sha = DeviceSheet(cell, N, N; D = Da)
tra = sheet_simulate(sha, 9000; drive = seed_drive(N, N; radius = 3), device = dev)
cen = (N ÷ 2, N ÷ 2)
u_row = arrival_speed(tra.t_first; center = cen, rmin = 20, rmax = 120, sector = (0.0, π / 12))[1]   # along rows (dy)
u_col = arrival_speed(tra.t_first; center = cen, rmin = 20, rmax = 120, sector = (π / 2, π / 12))[1] # along columns (dx)
u10 = medium_speed(DeviceSheet(cell, N, N; ps_per_site = 10), dev; nsteps = 6000, rmax = 120)
u20 = medium_speed(DeviceSheet(cell, N, N; ps_per_site = 20), dev; nsteps = 9000, rmax = 120)
@printf("    speed along rows %.4f, along columns %.4f (ratio %.3f); uniform 10 ps %.4f, uniform 20 ps %.4f (ratio %.3f)\n",
        u_row, u_col, u_row / u_col, u10, u20, u10 / u20)

println("[b] refraction at a delay interface")
right(x) = x > N ÷ 2
Db = Float32[(right(x) ? 25 : 10) * sqrt(o[1]^2 + o[2]^2) for y in 1:N, x in 1:N, o in offs]
shb = DeviceSheet(cell, N, N; D = Db)
u1 = u10
u2 = medium_speed(DeviceSheet(cell, N, N; ps_per_site = 25), dev; nsteps = 12000, rmax = 120)
rows = NamedTuple[]
panels = []
for θ1d in (20.0, 35.0, 50.0)
    θ1 = deg2rad(θ1d)
    n = (sin(θ1), cos(θ1))                      # propagation direction in (row, col); col = across the interface
    P0 = (N ÷ 2 - round(Int, 90 * sin(θ1)), N ÷ 4)
    th = Float32[abs((y - P0[1]) * n[1] + (x - P0[2]) * n[2]) <= 0.7 &&
                 abs((y - P0[1]) * n[2] - (x - P0[2]) * n[1]) <= 70 && !right(x + 12) ? 0f0 : Inf32
                 for y in 1:N, x in 1:N]
    tr = sheet_simulate(shb, 14000; drive = EventDrive(th, 40), device = dev)
    tf = tr.t_first
    # where the central ray meets the interface
    yh = P0[1] + (N ÷ 2 - P0[2]) * n[1] / n[2]
    θ2p = asin(clamp(u2 / u1 * sin(θ1), -1, 1))
    fitθ(xr, yc) = begin
        A = Float64[]; Y = Float64[]; X = Float64[]
        for x in xr, y in round(Int, yc - 25):round(Int, yc + 25)
            (1 <= y <= N && isfinite(tf[y, x])) || continue
            push!(Y, y); push!(X, x); push!(A, tf[y, x])
        end
        M = hcat(ones(length(A)), Y, X); c = M \ A
        atan(c[2], c[3]), 1 / hypot(c[2], c[3])
    end
    θin, uin = fitθ(N ÷ 2 - 50:N ÷ 2 - 10, yh - 30 * tan(θ1))
    θ2, uout = fitθ(N ÷ 2 + 10:N ÷ 2 + 50, yh + 30 * tan(θ2p))
    @printf("    θ₁ set %.0f°, measured %.1f° (u %.4f) → θ₂ measured %.1f° (u %.4f), Snell predicts %.1f°\n",
            θ1d, rad2deg(θin), uin, rad2deg(θ2), uout, rad2deg(asin(clamp(u2 / u1 * sin(θin), -1, 1))))
    push!(rows, (; theta1_set_deg = θ1d, theta1_meas_deg = rad2deg(θin), theta2_meas_deg = rad2deg(θ2),
                   theta2_snell_deg = rad2deg(asin(clamp(u2 / u1 * sin(θin), -1, 1))), u1, u2, u_in = uin, u_out = uout))
    t = copy(tf); t[.!isfinite.(t)] .= NaN
    p = heatmap(t; c = :viridis, yflip = true, aspect_ratio = 1, axis = false, colorbar = false,
                title = @sprintf("θ₁ %.0f° → θ₂ %.1f° (Snell %.1f°)", rad2deg(θin), rad2deg(θ2), rows[end].theta2_snell_deg))
    contour!(p, t; levels = 0:300:14000, c = :white, lw = 0.5, colorbar_entry = false)
    vline!(p, [N ÷ 2]; lc = :orange, lw = 1.5, label = "")
    push!(panels, p)
end
write_tsv("e6_refraction.tsv", rows)
write_tsv("e6_anisotropy.tsv", [(; u_row, u_col, ratio = u_row / u_col, u10, u20, ratio_uniform = u10 / u20)])
ta = copy(tra.t_first); ta[.!isfinite.(ta)] .= NaN
pa = heatmap(ta; c = :viridis, yflip = true, aspect_ratio = 1, axis = false, colorbar = false,
             title = @sprintf("anisotropic delays: speed ratio %.2f", u_row / u_col))
contour!(pa, ta; levels = 0:400:9000, c = :white, lw = 0.5, colorbar_entry = false)
savefig(plot(pa, panels...; layout = (1, 4), size = (1500, 400)), joinpath(STUDY_OUT, "e6_shaping.png"))
println("    saved e6_shaping.png")
