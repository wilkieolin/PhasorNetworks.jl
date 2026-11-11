#!/usr/bin/env julia
# Figure 3 -- the spike-timing readout floor.
#
# Data: results/ep_readout_floor/grid.csv (3072 runs)
#   eps in {0.003, 0.03, 0.3} x omega_p in {0.02, 0.05} x n_cycles in {2, 8}
#   x readout quantum delta in {0, 0.005, 0.01, 0.02} turns
#   x jitter in {0, 0.25, 1, 4} (MULTIPLES of delta) x project in {hard, soft}
#
# The Introduction argues phase networks are attractive because a 1-bit
# spike-time detector suffices to read them (sn-article.tex:179). This is the
# figure that checks whether the estimator survives that readout.
#
# Two things to keep straight:
#
#  * jitter is a multiple of the quantum, so a setting means the same thing at
#    every delta. At delta = 0 there is no quantum to scale against and the
#    value is an ABSOLUTE phase noise in turns -- jitter 0.25 there is 90 deg,
#    which is why that row collapses. It says nothing about dithering, so
#    delta = 0 appears in panel (c) only as the exact-readout ceiling.
#  * hard vs soft projection makes no difference here (0.409 vs 0.395 at
#    delta = 0.005): the soft projection addresses the large-amplitude phase
#    flip, which is a different failure from the quantiser dead zone.

using Pkg; Pkg.activate(@__DIR__)
using CairoMakie, DataFrames, CSV, Statistics, Printf
include("figure_style.jl")

df = CSV.read(joinpath(@__DIR__, "..", "results", "ep_readout_floor", "grid.csv"), DataFrame)
df = df[df.project .== "hard", :]

deltas = sort(unique(df.readout))
epss   = sort(unique(df.eps))
DCOL   = Dict(0.0 => C_HOLO, 0.005 => C_PROBE, 0.01 => C_HEBB, 0.02 => C_BAD)

med(x) = median(collect(skipmissing(x)))
fig = Figure(size = (FULL_W, 158))

# ---------------------------------------------------------------- (b)
axb = Axis(fig[1, 1]; title = panel_title("b", "A phase quantum closes the window"),
    xlabel = "probe amplitude  ε", ylabel = "median cos",
    xscale = log10, titlealign = :left, xticks = (epss, string.(epss)))
for d in deltas
    s = df[(df.readout .== d) .& (df.jitter .== 0), :]
    m = combine(groupby(s, :eps), :cos_l1 => med => :c); sort!(m, :eps)
    lines!(axb, m.eps, m.c; color = DCOL[d], linewidth = LINE_WIDTH,
           linestyle = d == 0 ? :dash : :solid,
           label = d == 0 ? "δ = 0 (exact)" : @sprintf("δ = %.3f turns", d))
    scatter!(axb, m.eps, m.c; color = DCOL[d], markersize = MARKER_SIZE - 2)
end
ylims!(axb, -0.05, 1.05)
axislegend(axb; position = :rb, framevisible = false, labelsize = FONT_SIZE_SMALL - 1,
           padding = (1, 1, 1, 1), rowgap = -1, patchsize = (11, 6), nbanks = 2)

# ---------------------------------------------------------------- (c)
axc = Axis(fig[1, 2]; title = panel_title("c", "Dither reopens it"),
    xlabel = "spike-time jitter  (multiples of δ)", ylabel = "median cos",
    titlealign = :left)
jit = sort(unique(df.jitter))
best0 = maximum(med(df[(df.readout .== 0) .& (df.jitter .== 0) .& (df.eps .== e), :cos_l1]) for e in epss)
hlines!(axc, [best0]; color = C_HOLO, linestyle = :dash, linewidth = 1.2)
text!(axc, 0.05, best0; text = "exact readout (δ = 0)", fontsize = FONT_SIZE_SMALL - 1,
      color = C_HOLO, align = (:left, :top), offset = (0, -2))
for d in deltas
    d == 0 && continue
    s = df[(df.readout .== d) .& (df.eps .== 0.3), :]
    m = combine(groupby(s, :jitter), :cos_l1 => med => :c,
                :cos_l1 => (v -> quantile(v, 0.25)) => :lo,
                :cos_l1 => (v -> quantile(v, 0.75)) => :hi); sort!(m, :jitter)
    band!(axc, m.jitter, m.lo, m.hi; color = (DCOL[d], 0.14))
    lines!(axc, m.jitter, m.c; color = DCOL[d], linewidth = LINE_WIDTH,
           label = @sprintf("δ = %.3f", d))
    scatter!(axc, m.jitter, m.c; color = DCOL[d], markersize = MARKER_SIZE - 2)
end
peak = 0.25
vlines!(axc, [peak]; color = C_AXIS, linestyle = :dot, linewidth = 1.0)
text!(axc, peak, 0.02; text = " optimum at 0.25 δ", fontsize = FONT_SIZE_SMALL - 1,
      color = C_AXIS, align = (:left, :bottom))
ylims!(axc, -0.05, 1.05)
axislegend(axc; position = :rt, framevisible = false, labelsize = FONT_SIZE_SMALL - 1,
           padding = (1, 1, 1, 1), rowgap = -1, patchsize = (11, 6))

colgap!(fig.layout, 12)
save_fig(fig, "figure3_readout")

for d in deltas
    b = maximum(med(df[(df.readout .== d) .& (df.jitter .== 0) .& (df.eps .== e), :cos_l1]) for e in epss)
    bj = maximum(med(df[(df.readout .== d) .& (df.jitter .== j) .& (df.eps .== 0.3), :cos_l1]) for j in jit)
    @printf("delta=%.3f  best-over-eps (no dither) %.3f   best-over-jitter %.3f\n", d, b, bj)
end
