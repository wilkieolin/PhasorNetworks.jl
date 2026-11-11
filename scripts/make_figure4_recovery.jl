#!/usr/bin/env julia
# Figure 4 -- repair of impaired analog parameters.
#
# Data:
#   results/ep_analog_finetune/analog_finetune_ddbaa3d.csv   (E2 sweep, 377 rows)
#   results/ep_analog_finetune/lockin_paired_*.csv           (E2b, paired arm)
#
# The main text uses the pretrain == "backprop" arm: a network trained
# digitally, transferred to the analog substrate, damaged, then repaired in
# situ. The staticep arm is supplementary.
#
# Panels (a)-(d) plot ACCURACY, not recovery_frac. recovery_frac is clamped to
# [-1, 2] at scripts/e2_full_sweep.jl and is NaN wherever the damage is
# negligible, so it saturates exactly where the interesting cells are.

using Pkg; Pkg.activate(@__DIR__)
using CairoMakie, DataFrames, CSV, Statistics, Printf
include("figure_style.jl")

const RES = joinpath(@__DIR__, "..", "results", "ep_analog_finetune")

df = CSV.read(joinpath(RES, "analog_finetune_ddbaa3d.csv"), DataFrame)
df = df[df.pretrain .== "backprop", :]

# Human-facing names and the severity parameter each type is swept in.
const TYPES = [
    ("gaussian",   "Additive Gaussian",     "σ  (weight units)"),
    ("lognormal",  "Multiplicative lognormal", "σ  (log units)"),
    ("stuck_zero", "Stuck at zero",         "stuck fraction"),
    ("stuck_sat",  "Stuck at rail",         "stuck fraction"),
]

# series = (column, colour, label, linestyle)
# Backprop is dashed because it lands almost exactly on top of EP at every
# severity -- which is the result, but two solid lines there read as one.
const SERIES = [
    (:acc_bp_ft,         C_BP,       "Backprop fine-tune (ceiling)", :dash),
    (:acc_tuned,         C_STATIC,   "EP fine-tune",                 :solid),
    (:acc_readout_only,  C_READOUT,  "Readout retune only",          :solid),
    (:acc_impaired,      C_IMPAIRED, "Impaired (floor)",             :solid),
]

# One colour per impairment type, used in panels (e) and (f).
const TYPE_COLOUR = Dict("gaussian" => C_HOLO, "lognormal" => C_PROBE,
                         "stuck_zero" => C_HEBB, "stuck_sat" => C_ANTIHOLO)

med(v) = median(skipmissing(v))
q(v, p) = quantile(collect(skipmissing(v)), p)

fig = Figure(size = (FULL_W, 318))

# ---------------------------------------------------------------- (a)-(d)
axs = Axis[]
for (i, (tkey, tname, xlab)) in enumerate(TYPES)
    r, c = fldmod1(i, 2)
    ax = Axis(fig[r, c];
        title = panel_title(string('b' + i - 1), tname),
        xlabel = xlab, ylabel = "Test accuracy",
        xscale = log10, titlealign = :left)
    push!(axs, ax)

    sub = df[df.impairment_type .== tkey, :]
    g = combine(groupby(sub, :param),
        [c => med => c for c in [:acc_clean, :acc_impaired, :acc_tuned, :acc_bp_ft, :acc_readout_only]]...,
        [c => (v -> q(v, 0.25)) => Symbol(c, :_lo) for c in [:acc_impaired, :acc_tuned, :acc_bp_ft, :acc_readout_only]]...,
        [c => (v -> q(v, 0.75)) => Symbol(c, :_hi) for c in [:acc_impaired, :acc_tuned, :acc_bp_ft, :acc_readout_only]]...,
        nrow => :n)
    sort!(g, :param)
    ax.xticks = (collect(g.param), [replace(@sprintf("%g", v), "0." => ".") for v in g.param])

    # clean ceiling: constant across severities by construction
    hlines!(ax, [med(sub.acc_clean)]; color = C_REFERENCE, linestyle = :dash, linewidth = 1.0,
            label = i == 1 ? "Undamaged (ceiling)" : nothing)

    for (col, colour, lab, ls) in SERIES
        band!(ax, g.param, g[!, Symbol(col, :_lo)], g[!, Symbol(col, :_hi)];
              color = (colour, 0.18))
        lines!(ax, g.param, g[!, col]; color = colour, linestyle = ls,
               linewidth = LINE_WIDTH, label = lab)
        scatter!(ax, g.param, g[!, col]; color = colour, markersize = MARKER_SIZE - 2)
    end
    ylims!(ax, 0.05, 0.95)
    iseven(i) && (ax.ylabel = "")
end
linkyaxes!(axs...)

# ---------------------------------------------------------------- (e)
axe = Axis(fig[3, 1];
    title = panel_title("f", "Recovery vs damage"),
    xlabel = "Impaired accuracy", ylabel = "After EP fine-tune",
    titlealign = :left)
mk = Dict("gaussian" => :circle, "lognormal" => :utriangle,
          "stuck_zero" => :rect, "stuck_sat" => :diamond)
for (tkey, tname, _) in TYPES
    sub = df[df.impairment_type .== tkey, :]
    scatter!(axe, sub.acc_impaired, sub.acc_tuned;
             color = (TYPE_COLOUR[tkey], 0.65), marker = mk[tkey],
             markersize = MARKER_SIZE - 1.5, label = tname, strokewidth = 0)
end
lines!(axe, [0.05, 0.95], [0.05, 0.95]; color = C_REFERENCE, linestyle = :dot, linewidth = 1.0)
hlines!(axe, [med(df.acc_clean)]; color = C_REFERENCE, linestyle = :dash, linewidth = 1.0)
text!(axe, 0.95, 0.05; space = :relative, text = "no repair (y = x)",
      fontsize = FONT_SIZE_SMALL - 2, color = C_REFERENCE, align = (:right, :bottom))
xlims!(axe, 0.05, 0.92); ylims!(axe, 0.60, 0.92)

# ---------------------------------------------------------------- (f)
axf = Axis(fig[3, 2];
    title = panel_title("g", "Lock-in vs static EP"),
    xlabel = "Static EP", ylabel = "Lock-in EP",
    titlealign = :left)
paired = filter(f -> startswith(basename(f), "lockin_paired_") && endswith(f, ".csv"),
                readdir(RES; join = true))
if isempty(paired)
    text!(axf, 0.5, 0.5; text = "awaiting scripts/e2_lockin_paired.jl",
          fontsize = FONT_SIZE_SMALL, color = C_REFERENCE,
          align = (:center, :center), space = :relative)
    hidedecorations!(axf); hidespines!(axf)
else
    pf = CSV.read(last(sort(paired)), DataFrame)
    for (tkey, tname, _) in TYPES
        s = pf[pf.impairment_type .== tkey, :]
        isempty(s) && continue
        scatter!(axf, s.acc_static, s.acc_lockin; color = (TYPE_COLOUR[tkey], 0.75),
                 marker = mk[tkey], markersize = MARKER_SIZE - 1.5, strokewidth = 0)
    end
    lo = min(minimum(pf.acc_static), minimum(pf.acc_lockin)) - 0.02
    hi = max(maximum(pf.acc_static), maximum(pf.acc_lockin)) + 0.02
    lines!(axf, [lo, hi], [lo, hi]; color = C_REFERENCE, linestyle = :dot, linewidth = 1.0)
    d = pf.acc_lockin .- pf.acc_static
    se = std(d) / sqrt(length(d))
    text!(axf, 0.04, 0.93; space = :relative, align = (:left, :top),
          fontsize = FONT_SIZE_SMALL - 1, color = C_AXIS,
          text = @sprintf("mean Δ = %+.4f ± %.4f (n = %d)", mean(d), 1.96se, length(d)))
    xlims!(axf, lo, hi); ylims!(axf, lo, hi)
end

# ---------------------------------------------------------------- legend
Legend(fig[4, 1:2], axs[1]; orientation = :horizontal, framevisible = false,
       nbanks = 2, patchsize = (14, 8), padding = (0, 0, 0, 0),
       labelsize = FONT_SIZE_SMALL - 1, colgap = 8, rowgap = 0)
Legend(fig[5, 1:2], axe; orientation = :horizontal, framevisible = false,
       nbanks = 1, patchsize = (14, 8), padding = (0, 0, 0, 0),
       labelsize = FONT_SIZE_SMALL - 1, colgap = 8)

rowgap!(fig.layout, 6); colgap!(fig.layout, 10)
rowsize!(fig.layout, 1, Relative(0.30)); rowsize!(fig.layout, 2, Relative(0.30))
rowsize!(fig.layout, 3, Relative(0.30))

save_fig(fig, "figure4_recovery")
println("figure4_recovery written")
