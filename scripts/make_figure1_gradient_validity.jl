#!/usr/bin/env julia
# Figure 1 (redraw): "Does the estimator recover the gradient, and when does it stop?"
#
# Supersedes make_figure1_fashionmnist.jl panels (b) and (c). The old 1c plotted
# cos vs ‖W₁‖ over three points from randomly-rescaled matrices and read as a
# weight-norm threshold. results/ep_trained_vs_rescaled/ shows that threshold is a
# property of the random-matrix construction, not of the norm, so the panel is
# rebuilt around the trained-vs-rescaled contrast at matched norm.
#
# Deliberately avoids figure_style.jl's Lx/Ly (they force math mode, killing
# spaces) and its hline!/vline! (they read ax.finallimits[] before layout).
# Uses Makie's own hlines!/vlines! and rich() titles.

using Pkg
Pkg.activate(@__DIR__)

using CairoMakie, DataFrames, CSV, Statistics
include("figure_style.jl")

const RES  = "results"
const E1   = joinpath(RES, "ep_trained_vs_rescaled")
const FM   = joinpath(RES, "ep_fashionmnist")

calib = CSV.read(joinpath(FM, "lockin_calibration.csv"), DataFrame)
e1    = CSV.read(joinpath(E1, "trained_vs_rescaled.csv"), DataFrame)
conv  = CSV.read(joinpath(E1, "settle_convergence.csv"), DataFrame)

# Colour roles held constant across panels, and across the whole figure set:
# these are the Section 2 TikZ palette (sn-article.tex:64-70) re-exported by
# figure_style.jl, so the drawn schematics and the plotted data agree.
const C_TRAINED  = C_HOLO       # trained structure
const C_TR_RESC  = C_PROBE      # trained structure, rescaled
const C_RESCALED = C_ANTIHOLO   # rescaled random control
const C_GUIDE    = C_AXIS

sel(df; kw...) = filter(r -> all(getproperty(r, k) == v for (k, v) in kw), df)

# Each panel is a function of a GridPosition so the same code produces both the
# combined figure and the standalone per-panel files the manuscript includes.

# ---------------------------------------------------------------- (a) lock-in
function panel_a!(gp)
ax_a = Axis(gp;
    xlabel = "probe frequency  ω_p",
    ylabel = "relative error vs StaticEP",
    title  = panel_title("a", "Lock-in costs steps, not fidelity"),
    xscale = log10, yscale = log10,
    xticks = ([0.005, 0.01, 0.02, 0.05], ["0.005", "0.01", "0.02", "0.05"]),
    yticks = ([0.01, 0.1, 1.0], ["0.01", "0.1", "1"]))

for (i, ε) in enumerate(sort(unique(calib.eps)))
    sub = sort(sel(calib; eps = ε), :omega_p)
    col = [C_ANTIHOLO, C_HOLO, C_PROBE][i]
    lines!(ax_a, sub.omega_p, sub.relerr_L1; color = col, label = "ε = $ε")
    scatter!(ax_a, sub.omega_p, sub.relerr_L1; color = col)
end
vlines!(ax_a, [0.02]; color = C_GUIDE, linestyle = :dot, linewidth = 1)
text!(ax_a, 0.021, 0.016; text = "operating point\n3968 steps/grad",
      fontsize = 6, color = C_GUIDE, align = (:left, :bottom))
safe_axislegend!(ax_a; position = :lt, framevisible = false, padding = (2, 2, 2, 2))

    ax_a.titlealign = :left; ax_a.titlesize = FONT_SIZE_NORMAL
    return ax_a
end

# --------------------------------------------------------- (b) the 1/β signature
function panel_b!(gp)
# Matched norm ‖W₁‖ = 32.1 (nodecay, epoch 2): the published probe's 31.4.
ax_b = Axis(gp;
    xlabel = "nudge amplitude  β",
    ylabel = "relative error vs FD oracle",
    title  = panel_title("b", "Only the random control hops"),
    xscale = log10, yscale = log10,
    xticks = ([0.003, 0.01, 0.03, 0.1, 0.3], ["0.003", "0.01", "0.03", "0.1", "0.3"]))

# slope -1 guide first, offset above the data so it reads as a reference, not a fit
let βs = [0.0032, 0.28], a = 12.0
    lines!(ax_b, βs, a ./ βs; color = C_GUIDE, linestyle = :dot, linewidth = 1)
    text!(ax_b, 0.014, 12.0 / 0.014 * 1.3; text = "slope −1", fontsize = 6,
          color = C_GUIDE, align = (:left, :bottom))
end
for (tag, col) in ((:trained, C_TRAINED), (:rescaled, C_RESCALED))
    for (est, ls, mk, lbl) in (("static_onesided", :solid, :circle, "one-sided"),
                               ("static_centered", :dash, :utriangle, "centered"))
        sub = sort(sel(e1; run = "nodecay", epoch = 2, tag = String(tag), estimator = est), :beta)
        lines!(ax_b, sub.beta, sub.relerr_fd; color = col, linestyle = ls,
               label = "$(tag), $lbl")
        scatter!(ax_b, sub.beta, sub.relerr_fd; color = col, marker = mk)
    end
end
safe_axislegend!(ax_b; position = :rt, framevisible = false, padding = (1, 1, 1, 1),
           patchsize = (10, 5), rowgap = -2, labelsize = FONT_SIZE_SMALL - 2)

    ax_b.titlealign = :left; ax_b.titlesize = FONT_SIZE_NORMAL
    return ax_b
end

# ------------------------------------------------- (c) fidelity across the run
function panel_c!(gp)
# Two training trajectories are pooled here, so points are NOT connected: the
# wd=1e-4 run sits entirely at ‖W₁‖ ≈ 21-26 while the no-decay run spans 32-89,
# and joining them by norm draws crossings that are not trajectories.
ax_c = Axis(gp;
    xlabel = "weight norm  ‖W₁‖",
    ylabel = "cos(EP, FD oracle),  β = 0.1",
    title  = panel_title("c", "Structure, not scale, is what EP needs"))

hlines!(ax_c, [0.0]; color = (:black, 0.22), linewidth = 0.6)
vlines!(ax_c, [15.0]; color = C_GUIDE, linestyle = :dot, linewidth = 1)

# trained + centered underneath, as the mitigation
tc = sel(e1; tag = "trained", estimator = "static_centered", beta = 0.1)
scatter!(ax_c, tc.w1_norm, tc.cos_fd; color = (:gray72, 0.95), markersize = 11,
         strokewidth = 0, label = "trained, centered")

for (tag, col, lbl) in ((:trained_rescaled, C_TR_RESC, "trained structure, rescaled"),
                        (:rescaled,         C_RESCALED, "random init, rescaled"),
                        (:trained,          C_TRAINED,  "trained, one-sided"))
    sub = sel(e1; tag = String(tag), estimator = "static_onesided", beta = 0.1)
    tr  = sel(sub; fd_trusted = true)
    un  = sel(sub; fd_trusted = false)
    scatter!(ax_c, tr.w1_norm, tr.cos_fd; color = col, label = lbl)
    isempty(un) || scatter!(ax_c, un.w1_norm, un.cos_fd; color = :transparent,
                            strokecolor = col, strokewidth = 1)
end

text!(ax_c, 93, 1.22; text = "open marker: FD oracle untrusted",
      fontsize = 6, color = :gray45, align = (:right, :bottom))
ylims!(ax_c, -1.45, 1.45)
xlims!(ax_c, 12, 95)
axislegend(ax_c; position = :lb, orientation = :vertical, nbanks = 1,
       framevisible = false, labelsize = FONT_SIZE_SMALL - 2, patchsize = (7, 5),
       padding = (1, 1, 1, 1), rowgap = 0)

    ax_c.titlealign = :left; ax_c.titlesize = FONT_SIZE_NORMAL
    return ax_c
end

# ----------------------------------------------- (d) the convergence diagnostic
function panel_d!(gp)
ax_d = Axis(gp;
    xlabel = "settle drift  ‖z(2T) − z(T)‖/√N",
    ylabel = "cos(EP, FD oracle)",
    title  = panel_title("d", "Drift predicts it"),
    xscale = log10,
    xticks = ([1e-9, 1e-7, 1e-5, 1e-3, 1e-1], ["1e-9", "1e-7", "1e-5", "1e-3", "1e-1"]))

# exact zeros cannot be drawn on a log axis; floor them at the axis edge and say so
dr = max.(Float64.(conv.settle_drift), 1e-9)
hlines!(ax_d, [0.99]; color = (:black, 0.22), linewidth = 0.6)
vlines!(ax_d, [1e-3]; color = C_RESCALED, linestyle = :dash, linewidth = 1)
scatter!(ax_d, dr, conv.cos_onesided; color = C_TRAINED, label = "one-sided")
scatter!(ax_d, dr, conv.cos_centered; color = C_TR_RESC, marker = :utriangle,
         markersize = 5, label = "centered")
text!(ax_d, 7e-4, 0.55; text = "no cell right of 1e-3\nreaches cos 0.99",
      fontsize = 6, color = C_RESCALED, align = (:right, :bottom))
ylims!(ax_d, -0.35, 1.14)
xlims!(ax_d, 4e-10, 1.0)
safe_axislegend!(ax_d; position = :lb, framevisible = false, padding = (2, 2, 2, 2),
           labelsize = FONT_SIZE_SMALL)

    ax_d.titlealign = :left; ax_d.titlesize = FONT_SIZE_NORMAL
    return ax_d
end

# ---------------------------------------------------------------- assemble
const PANELS = [("a", panel_a!), ("b", panel_b!), ("c", panel_c!), ("d", panel_d!)]

fig = Figure(size = (FULL_W, 300))
panel_a!(fig[1, 1]); panel_b!(fig[1, 2]); panel_c!(fig[2, 1]); panel_d!(fig[2, 2])
colgap!(fig.layout, 14)
rowgap!(fig.layout, 12)
save_fig(fig, "figure1_gradient_validity")

# Standalone panels, same names the manuscript's \includegraphics uses. The old
# fig1{a,b,c}_*.png from make_figure1_fashionmnist.jl are NOT overwritten -- they
# are kept under their own names and are superseded; see the README note written
# beside them.
for (letter, build!) in PANELS
    g = Figure(size = (HALF_W, HALF_W * 0.80))
    build!(g[1, 1])
    save_fig(g, "fig1$(letter)_gradient_validity")
end

open(joinpath("figures", "SUPERSEDED.txt"), "w") do io
    println(io, """
    fig1b_calibration.{pdf,png}  and  fig1c_weight_norm.{pdf,png}
    are produced by scripts/make_figure1_fashionmnist.jl and are SUPERSEDED.

    fig1c_weight_norm in particular plots cos vs |W1| over three points measured on
    randomly-rescaled matrices, and reads as a weight-norm threshold. E1
    (results/ep_trained_vs_rescaled/FINDINGS.md) shows that threshold is a property
    of the random-matrix construction, not of the norm. Do not put that panel in the
    paper.

    Use instead, from scripts/make_figure1_gradient_validity.jl:
      fig1a_gradient_validity  lock-in cost/fidelity trade
      fig1b_gradient_validity  the 1/beta basin-hop signature at matched |W1|
      fig1c_gradient_validity  three-arm contrast: structure, not scale   <-- replaces fig1c_weight_norm
      fig1d_gradient_validity  the T-vs-2T drift diagnostic
      figure1_gradient_validity  all four combined

    fig1a_training_curves is not superseded but needs E3 (seeds + a backprop line)
    before it is publishable.
    """)
end

println("wrote figures/figure1_gradient_validity.{pdf,png} + 4 standalone panels")
println("wrote figures/SUPERSEDED.txt")
