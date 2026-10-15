# E5 — front speed from first principles: a self-consistent plane front.
#
# A planar front crosses the lattice at speed u along unit normal n. A cell at
# the origin fires at t = 0; a neighbour at offset o fired at t = (n·o)/u, and
# its output pulse arrives through its line at τ_o = (n·o)/u + D_o. Below
# threshold the tank is linear, so the loop current is a superposition of
# impulse responses h(t) (the tank's response to one output pulse, computed
# from the device model):
#
#     i(t) = Σ_o G·h(t − τ_o(u))
#
# and the front is self-consistent when |i(t)| first reaches I_th exactly at
# t = 0. Solving for u needs nothing measured from a sheet: only the cell, the
# stencil, the gains and the delays. Where no u satisfies it, the model
# predicts that nothing propagates.
#
# Predicts: u against line delay and gain, the gain threshold of propagation,
# and the coherence failure at long delays (inputs spread over a carrier
# period cancel in h). Compared with every sheet measurement made so far.

include("common.jl")
using Plots

c0 = RFTransponder()

# Impulse response of the loop current to ONE output pulse of `c` (gain 1),
# from the device model with the threshold disabled.
function impulse_response(c; dt = 0.05, T = 3000.0)
    lin = RFTransponder(c.L, c.C, c.R, 1f9, c.t_ref, c.I_out, c.tau_out)
    τ = c.tau_out
    drive(t) = (s = t / τ; s > 0 ? Float32(c.I_out * s * exp(1 - s)) : 0f0)
    tr = cell_simulate(lin, drive, dt, round(Int, T / dt))
    return vcat(0.0, tr.t), vcat(0.0, Float64.(tr.i))
end

function first_crossing(h_t, h_i, τs, G, Ith; tmax = 50.0, dt = 0.25)
    t0 = minimum(τs)
    for t in t0:dt:tmax
        i = 0.0
        for τ in τs
            x = t - τ
            x <= 0 && continue
            k = clamp(searchsortedlast(h_t, x), 1, length(h_t) - 1)
            w = (x - h_t[k]) / (h_t[k+1] - h_t[k])
            i += G * ((1 - w) * h_i[k] + w * h_i[k+1])
        end
        abs(i) >= Ith && return t
    end
    return Inf
end

"""
    plane_speed(c, offs, D, G; n) -> u (sites/ps) or NaN

Largest self-consistent plane-front speed along normal `n` (the fastest
solution is the one a sheet realises; slower roots are unstable).
"""
function plane_speed(c, offs, D, G; n = (0.0, 1.0), ugrid = exp.(range(log(0.003), log(0.3), length = 160)))
    h_t, h_i = impulse_response(c)
    behind = [k for k in eachindex(offs) if n[1] * offs[k][1] + n[2] * offs[k][2] < 0]
    tc(u) = first_crossing(h_t, h_i, [(n[1] * offs[k][1] + n[2] * offs[k][2]) / u + D[k] for k in behind], G, c.I_th)
    vals = [tc(u) for u in ugrid]
    # t_c increases with u through 0 at the solution; take the largest bracket
    for j in length(ugrid)-1:-1:1
        if isfinite(vals[j]) && isfinite(vals[j+1]) && vals[j] <= 0 <= vals[j+1]
            lo, hi = ugrid[j], ugrid[j+1]
            for _ in 1:40
                mid = sqrt(lo * hi); tc(mid) <= 0 ? (lo = mid) : (hi = mid)
            end
            return lo
        end
    end
    return NaN
end

dirs = ((0.0, 1.0), (sqrt(0.5), sqrt(0.5)))
predict(c, R, pps, G; even_only = false) = begin
    offs = stencil_offsets(R)
    even_only && (offs = [o for o in offs if iseven(o[1] + o[2])])
    D = [pps * sqrt(o[1]^2 + o[2]^2) for o in offs]
    us = [plane_speed(c, offs, D, G; n) for n in dirs]
    (axis = us[1], diag = us[2], mean = mean(us))
end

# Every radial-seed measurement made so far (sites/ps; NaN = did not propagate).
meas = [
    (R = 3, pps = 10, G = 0.10, u = NaN,    src = "calib1 96²"),
    (R = 3, pps = 10, G = 0.15, u = NaN,    src = "calib1 96²"),
    (R = 3, pps = 10, G = 0.20, u = 0.0416, src = "calib1 96²"),
    (R = 3, pps = 10, G = 0.30, u = 0.0475, src = "calib2 128²"),
    (R = 3, pps = 10, G = 0.45, u = 0.0522, src = "calib1 96²"),
    (R = 3, pps = 10, G = 0.70, u = 0.0569, src = "calib1 96²"),
    (R = 2, pps = 10, G = 0.30, u = NaN,    src = "calib1 96²"),
    (R = 2, pps = 10, G = 0.45, u = 0.0339, src = "calib1 96²"),
    (R = 2, pps = 10, G = 0.70, u = 0.0393, src = "calib1 96²"),
    (R = 3, pps = 2,  G = 0.30, u = 0.0859, src = "calib2 128²"),
    (R = 3, pps = 5,  G = 0.30, u = 0.0656, src = "calib2 128²"),
    (R = 3, pps = 20, G = 0.30, u = 0.0309, src = "calib2 128²"),
    (R = 3, pps = 40, G = 0.30, u = 0.0184, src = "calib2 128²"),
]
sub = [   # checkerboard sublattice (even offsets only), G = 0.45
    (pps = 10, u = 0.04125, src = "multi 800²"),
    (pps = 50, u = NaN,     src = "calib_sub 192²"),
    (pps = 70, u = NaN,     src = "calib_sub 192² (60% fired, re-fires)"),
]

println("[1] full stencil")
rows = NamedTuple[]
@printf("    %2s %4s %5s | %9s | %9s %9s %9s | %7s\n", "R", "pps", "G", "measured", "axis", "diag", "mean", "err")
for m in meas
    p = predict(c0, m.R, m.pps, m.G)
    err = isfinite(m.u) && isfinite(p.mean) ? 100 * (p.mean / m.u - 1) : NaN
    @printf("    %2d %4d %5.2f | %9.4f | %9.4f %9.4f %9.4f | %+6.1f%%\n", m.R, m.pps, m.G, m.u, p.axis, p.diag, p.mean, err)
    push!(rows, (; kind = "full", m.R, m.pps, m.G, measured = m.u, pred_axis = p.axis, pred_diag = p.diag, pred_mean = p.mean, err_pct = err))
end
println("[2] checkerboard sublattice (even offsets), G = 0.45")
for m in sub
    p = predict(c0, 3, m.pps, 0.45; even_only = true)
    err = isfinite(m.u) && isfinite(p.mean) ? 100 * (p.mean / m.u - 1) : NaN
    @printf("    pps %3d | measured %8.4f | predicted %8.4f (axis %.4f, diag %.4f) | %+6.1f%%   [%s]\n",
            m.pps, m.u, p.mean, p.axis, p.diag, err, m.src)
    push!(rows, (; kind = "sublattice", R = 3, m.pps, G = 0.45, measured = m.u, pred_axis = p.axis, pred_diag = p.diag, pred_mean = p.mean, err_pct = err))
end
write_tsv("e5_front_theory.tsv", rows)

println("[3] predicted maps: u(pps) and the gain threshold")
ppsg = [1, 2, 3, 5, 7, 10, 14, 20, 28, 40, 56, 80]
curves = Dict(G => [predict(c0, 3, p, G).mean for p in ppsg] for G in (0.2, 0.3, 0.45))
sub_curve = [predict(c0, 3, p, 0.45; even_only = true).mean for p in ppsg]
Gs = range(0.08, 0.3; length = 23)
gth = [predict(c0, 3, 10, g).mean for g in Gs]
i_on = findfirst(isfinite, gth)
@printf("    gain threshold (R=3, 10 ps/pitch): propagation from G ≈ %.3f\n", i_on === nothing ? NaN : Gs[i_on])
write_tsv("e5_maps.tsv", [(; pps = ppsg[j], u_G020 = curves[0.2][j], u_G030 = curves[0.3][j], u_G045 = curves[0.45][j],
                             u_sub_G045 = sub_curve[j]) for j in eachindex(ppsg)])

p1 = plot(; xscale = :log10, yscale = :log10, xlabel = "line delay per pitch (ps)", ylabel = "front speed u (sites/ps)",
          title = "plane-front theory vs sheet", legend = :bottomleft)
for (G, col) in ((0.2, :orange), (0.3, :steelblue), (0.45, :seagreen))
    ok = isfinite.(curves[G])
    plot!(p1, ppsg[ok], curves[G][ok]; lw = 2, lc = col, label = "theory G=$G")
end
okS = isfinite.(sub_curve)
plot!(p1, ppsg[okS], sub_curve[okS]; lw = 2, ls = :dash, lc = :purple, label = "theory, checkerboard G=0.45")
plot!(p1, ppsg, 1 ./ ppsg; lc = :gray, ls = :dot, label = "line-only bound 1/pps")
mm = filter(m -> m.R == 3 && m.G == 0.3 && isfinite(m.u), meas)
scatter!(p1, [m.pps for m in mm], [m.u for m in mm]; mc = :steelblue, ms = 5, label = "measured G=0.3")
scatter!(p1, [10], [0.04125]; mc = :purple, ms = 5, label = "measured checkerboard")
p2 = plot(Gs, gth; lw = 2, lc = :steelblue, xlabel = "line gain G", ylabel = "u (sites/ps)", label = "theory (R=3, 10 ps)",
          title = "gain threshold")
mg = filter(m -> m.R == 3 && m.pps == 10, meas)
scatter!(p2, [m.G for m in mg if isfinite(m.u)], [m.u for m in mg if isfinite(m.u)]; mc = :black, ms = 5, label = "measured")
scatter!(p2, [m.G for m in mg if !isfinite(m.u)], zeros(count(m -> !isfinite(m.u), mg)); m = :x, mc = :red, ms = 6,
         label = "measured: no propagation")
savefig(plot(p1, p2; layout = (1, 2), size = (1150, 440), left_margin = 5Plots.mm, bottom_margin = 6Plots.mm),
        joinpath(STUDY_OUT, "e5_front_theory.png"))
println("    saved e5_front_theory.png")
