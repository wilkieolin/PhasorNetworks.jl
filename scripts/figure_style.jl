# Shared styling for publication figures
# Usage: include("scripts/figure_style.jl") then use `fig_style()`, `save_fig(fig, "name")`

using CairoMakie, ColorSchemes, LaTeXStrings, Colors

# Color palettes (colorblind-safe)
const OKABE_ITO = [
    RGB(0.0, 0.0, 0.0),       # black
    RGB(230/255, 159/255, 0), # orange
    RGB(86/255, 180/255, 233/255), # skyblue
    RGB(0, 158/255, 115/255), # bluish green
    RGB(240/255, 228/255, 66/255), # yellow
    RGB(0, 114/255, 178/255), # blue
    RGB(213/255, 94/255, 0),  # vermillion
    RGB(204/255, 121/255, 167/255), # reddish purple
]

const VIRIDIS = ColorSchemes.viridis
const CIVIDIS = ColorSchemes.cividis

# ---------------------------------------------------------------------------
# Figure dimensions
# ---------------------------------------------------------------------------
#
# The destination is the one-column Springer `sn-jnl` class used by
# ICRC-LIEP-manuscript/sn-article.tex. `sn-jnl.cls` sets `text={31pc,194.25mm}`
# for the single-column design, so
#
#     \textwidth  = 31 pc = 372 TeX pt = 130.75 mm
#     \textheight = 194.25 mm
#     body font   = 10 bp on 12 bp
#
# Mind the two kinds of point. TeX's pt/pc are 1/72.27 in; PDF (and Makie, and
# Springer's `bp`) use 1/72 in. So \textwidth = 372 TeX pt = 370.6 PDF pt, and
# the constants below are in PDF points because that is what Makie writes.
#
# `save_fig` writes PDFs with `pt_per_unit = 1`, so Makie `size` units ARE PDF
# points, 1:1. Size a figure to FULL_W and place it with
# `\includegraphics[width=\textwidth]{...}` for a scale factor of exactly 1.000,
# which is what makes the 7/8/9 pt fonts below render at their nominal size.
#
# The old SINGLE_COL/DOUBLE_COL were inches used as `Figure(size = (COL*100, …))`,
# i.e. a 340 pt / 700 pt canvas. A 700 pt figure dropped into a 372 pt text block
# is scaled 0.53x, which renders those same 7/8/9 pt fonts at 3.7/4.2/4.8 pt.
# They are kept only so unconverted scripts still run.

const FULL_W = 370.6    # PDF pt, == \textwidth (31 pc) exactly
const HALF_W = 180.3    # PDF pt, (FULL_W - 10 pt gutter) / 2
const TEXT_H = 550.6    # PDF pt, == \textheight (194.25 mm)

# Suggested heights (pt). A [t] float also needs room for a 5-8 line caption.
const H_ROW1 = 150.0    # single row of panels
const H_ROW2 = 300.0    # two rows / 2x2
const H_ROW3 = 330.0    # three rows

# Deprecated: inches, and used with a `* 100` idiom that produced the wrong page
# size. Use FULL_W / HALF_W.
const SINGLE_COL = 3.4
const DOUBLE_COL = 7.0
const FULL_PAGE = 7.0

# ---------------------------------------------------------------------------
# Manuscript palette
# ---------------------------------------------------------------------------
#
# These are the colours already \definecolor'd in sn-article.tex:64-70 and used
# by the hand-written TikZ figures of Section 2. Reusing them here is what makes
# the TikZ schematics and the Makie data panels read as one visual system.

const C_HOLO     = RGB(30/255, 110/255, 190/255)    # holo
const C_ANTIHOLO = RGB(200/255,  80/255,  30/255)   # antiholo
const C_PROBE    = RGB(40/255, 140/255,  90/255)    # probeC
const C_BASIN    = RGB(90/255, 150/255, 200/255)    # basinC
const C_BAD      = RGB(200/255, 120/255, 120/255)   # badC
const C_AXIS     = RGB(110/255, 110/255, 110/255)   # axisgray
const C_HEBB     = RGB(120/255,  60/255, 160/255)   # hebbC

# Semantic assignments. Hold these across every figure -- a reader should only
# have to learn the colours once.
const C_LIEP        = C_HOLO       # lock-in EP, the method
const C_BP          = C_ANTIHOLO   # backprop, the ceiling
const C_STATIC      = C_HEBB       # static EP
const C_READOUT     = C_PROBE      # readout-only retune / the probe signal
const C_IMPAIRED    = C_BAD        # perturbed floor, failure
const C_REFERENCE   = C_AXIS       # chance, y = x, ceilings, guides
const C_INSIDE      = C_BASIN      # inside the adiabatic basin

# Font settings
const FONT_REGULAR = "TeX Gyre Heros"
const FONT_MONO = "TeX Gyre Cursor"
const FONT_SIZE_SMALL = 7
const FONT_SIZE_NORMAL = 8
const FONT_SIZE_LARGE = 9

# Line/marker settings
const LINE_WIDTH = 1.5
const MARKER_SIZE = 6
const MARKER_STROKE = 0.5

function fig_style(; 
    fontsize = FONT_SIZE_NORMAL,
    linewidth = LINE_WIDTH,
    markersize = MARKER_SIZE,
    font = FONT_REGULAR
)
    theme = Theme(
        fontsize = fontsize,
        font = font,
        linewidth = linewidth,
        Axis = (
            xlabelsize = fontsize + 1,
            ylabelsize = fontsize + 1,
            xticklabelsize = fontsize - 1,
            yticklabelsize = fontsize - 1,
            xgridstyle = :dash,
            ygridstyle = :dash,
            xgridwidth = 0.5,
            ygridwidth = 0.5,
            xgridcolor = (:gray, 0.3),
            ygridcolor = (:gray, 0.3),
            spinewidth = 0.8,
        ),
        Legend = (
            fontsize = fontsize - 1,
            framewidth = 0.5,
            patchsize = (12, 8),
            rowgap = 2,
            colgap = 6,
        ),
        Lines = (
            linewidth = linewidth,
        ),
        Scatter = (
            markersize = markersize,
            strokewidth = MARKER_STROKE,
            strokecolor = :black,
        ),
        Heatmap = (
            colormap = VIRIDIS,
        ),
        Colorbar = (
            labelsize = fontsize - 1,
            ticklabelsize = fontsize - 2,
            width = 12,
            height = Relative(0.6),
        ),
    )
    return theme
end

# Apply theme
set_theme!(fig_style())

# Figure save helpers
function save_fig(fig, name::String;
    dir = "figures",
    formats = [".pdf", ".png"],
    px_per_unit = 4      # 372 pt * 4 = 1488 px across == ~289 dpi at \textwidth
)
    mkpath(dir)
    for fmt in formats
        filepath = joinpath(dir, "$name$fmt")
        if fmt == ".pdf"
            # pt_per_unit = 1 makes Makie `size` units literal PDF points, so a
            # figure built at FULL_W is exactly \textwidth and needs no scaling.
            save(filepath, fig; pt_per_unit = 1)
        else
            # CairoMakie rasterises by px_per_unit, not dpi; the old `dpi = 300`
            # kwarg was not one CairoMakie reads, so PNGs came out at 1 px/pt.
            save(filepath, fig; px_per_unit = px_per_unit)
        end
        @info "Saved $filepath"
    end
end

# ---------------------------------------------------------------------------
# Panel titles
# ---------------------------------------------------------------------------
#
# Do NOT write titles as L"\textbf{(a)} Some Words". LaTeXStrings puts the whole
# string in math mode, where inter-word spaces are discarded and every letter is
# set in italic math -- "AdiabaticZone : Mediancos(L1)". That was the cause of
# the run-together italic titles in all 17 panels, not the Lx/Ly helpers (which
# no figure script ever called; they have been removed).
#
# Use `panel_title` instead. It returns a Makie `rich` string: a bold "(a) "
# prefix and upright body text. Pass real maths as Unicode -- Makie's text
# renderer handles ω, Δ, ≈, →, ‖ directly, and `sub`/`sup` cover subscripts.
#
#   Axis(f[1,1]; title = panel_title("a", "Adiabatic zone: median cos(L1)"))
#   Axis(f[1,2]; title = panel_title("b", "Failure rate vs ", sub_label("ω_p / R", "relax")))
#
panel_title(letter, body...) = rich(rich("($letter) "; font = :bold), body...)

# Bare rich helpers for labels that mix words with a subscript or superscript.
sub_label(base, sub)  = rich(base, subscript(sub))
sup_label(base, sup)  = rich(base, superscript(sup))

# ---------------------------------------------------------------------------
# Reference lines
# ---------------------------------------------------------------------------
#
# The previous implementations read `ax.finallimits[]` at call time to work out
# how far to draw the segment. finallimits is only correct after the axis has
# been laid out and its limits finalised, which has not happened while the plot
# is still being built -- so lines landed at the wrong place or off-frame, and
# they also froze the limits against data added afterwards. Makie's own
# hlines!/vlines! span the axis at render time and track later data. These wrap
# them, keeping the old singular names and signatures so existing call sites
# work unchanged.
#
# `label` is passed through only when non-empty: an empty-string label still
# registers a legend entry, which is where the bare dashes in the Fig 2b, 4b and
# 5a legends came from.
_labelled(kw, label) = isempty(String(label)) ? kw : (; kw..., label = label)

function hline!(ax, y; color = :gray, linestyle = :dash, linewidth = 0.8, label = "")
    hlines!(ax, y isa Number ? [y] : collect(y);
            _labelled((; color, linestyle, linewidth), label)...)
end

function vline!(ax, x; color = :gray, linestyle = :dash, linewidth = 0.8, label = "")
    vlines!(ax, x isa Number ? [x] : collect(x);
            _labelled((; color, linestyle, linewidth), label)...)
end

# ---------------------------------------------------------------------------
# Axis configuration
# ---------------------------------------------------------------------------
#
# `ax.xlimits` / `ax.ylimits` are not Axis attributes; setting them silently did
# nothing (Makie ≥0.20 errors). Limits go through xlims!/ylims!.
function config_axis!(ax;
    xlabel = nothing, ylabel = nothing, title = nothing,
    xlim = nothing, ylim = nothing,
    xticks = nothing, yticks = nothing,
    xscale = nothing, yscale = nothing
)
    xlabel  === nothing || (ax.xlabel  = xlabel)
    ylabel  === nothing || (ax.ylabel  = ylabel)
    title   === nothing || (ax.title   = title)
    xticks  === nothing || (ax.xticks  = xticks)
    yticks  === nothing || (ax.yticks  = yticks)
    xscale  === nothing || (ax.xscale  = xscale)
    yscale  === nothing || (ax.yscale  = yscale)
    xlim    === nothing || xlims!(ax, xlim...)
    ylim    === nothing || ylims!(ax, ylim...)
    return ax
end

"""
    fit_ylims!(ax, values...; pad = 0.06, log = false, include = ())

Set y limits from the data rather than from a guessed constant, so nothing is
clipped. This is the fix for panels where a dip, a bar or a whole series fell
outside a hardcoded `ylims!` and was invisible (Fig 1a's StaticEP dips, Fig 3c's
1.06% "Drop" bar, Fig 5b/5c series leaving the frame).

`include` forces extra values into range -- pass any reference line you draw, or
0.0 for a bar chart, so the baseline is guaranteed visible. With `log = true`
the padding is applied multiplicatively and non-positive values are ignored.
"""
function fit_ylims!(ax, values...; pad = 0.06, log = false, include = ())
    vals = Float64[]
    for v in values, x in (v isa Number ? (v,) : v)
        isfinite(x) && push!(vals, Float64(x))
    end
    for x in (include isa Number ? (include,) : include)
        isfinite(x) && push!(vals, Float64(x))
    end
    isempty(vals) && return ax
    if log
        pos = filter(>(0), vals)
        isempty(pos) && return ax
        lo, hi = extrema(pos)
        f = (hi / lo) ^ pad
        ylims!(ax, lo / f, hi * f)
    else
        lo, hi = extrema(vals)
        m = hi - lo
        m = m == 0 ? (abs(hi) == 0 ? 1.0 : abs(hi) * 0.1) : m
        ylims!(ax, lo - pad * m, hi + pad * m)
    end
    return ax
end

"""
    categorical_x!(ax, categories; rotation = 0.0)

Place categorical values at integer positions 1:n with their names as ticks, and
return a `name -> position` lookup for plotting.

Two panels currently draw categories on a continuous axis, which misrepresents
them: Fig 5b puts `sample_every ∈ {1, 628}` on a linear axis, so every series is
a straight line between two points and the spacing implies 627 unmeasured
values; Fig 5a shares one log axis between `scaling` (a gain centred on 1.0) and
lognormal/gaussian/signflip (fractional σ), so a gain of 1.0 and a σ of 1.0 land
on the same tick although they are different quantities. A gain and a σ should
not share an axis at all -- use separate panels, and use this helper wherever
the x variable is genuinely a set of labels.
"""
function categorical_x!(ax, categories; rotation = 0.0)
    names = string.(collect(categories))
    ax.xticks = (collect(1:length(names)), names)
    rotation == 0.0 || (ax.xticklabelrotation = rotation)
    xlims!(ax, 0.5, length(names) + 0.5)
    return Dict(n => i for (i, n) in enumerate(names))
end

# ---------------------------------------------------------------------------
# Legends
# ---------------------------------------------------------------------------
"""
    legend_outside!(fig, ax, cell; orientation = :vertical, kwargs...)

Put the legend in its own layout cell instead of on top of the data. `axislegend`
draws inside the axis, which is why the Fig 3a, 3b, 4b, 5a and 5b legend boxes
obscure their plots -- in Fig 3a it hides three of the four impairment groups.

    legend_outside!(fig, ax, fig[1, 2])                       # column beside
    legend_outside!(fig, ax, fig[2, 1:2]; orientation = :horizontal)

Entries with empty labels are dropped, so a `label = ""` on a reference line
cannot produce a bare dash in the legend.
"""
function legend_outside!(fig, ax, cell; orientation = :vertical,
                         framevisible = false, kwargs...)
    elems, labels = Makie.get_labeled_plots(ax; unique = false, merge = false)
    keep = [i for (i, l) in enumerate(labels) if !isempty(string(l))]
    isempty(keep) && return nothing
    return Legend(cell, elems[keep], string.(labels[keep]);
                  orientation, framevisible, kwargs...)
end

"""
    safe_axislegend!(ax; kwargs...)

`axislegend` with two guards. It returns `nothing` instead of throwing when the
axis has no labelled plots, and it drops entries whose label is the empty string.

Both cases are real here. Once `hline!`/`vline!` stopped registering empty
labels, `axislegend` on Fig 4b -- whose only labelled artists *were* two
reference lines carrying `label = ""` -- began erroring instead of drawing a box
of bare dashes. A legend with nothing in it should be absent, not fatal and not
a row of dashes.

Prefer `legend_outside!` where the legend would otherwise sit on the data.
"""
function safe_axislegend!(ax; kwargs...)
    elems, labels = Makie.get_labeled_plots(ax; unique = false, merge = false)
    keep = [i for (i, l) in enumerate(labels) if !isempty(string(l))]
    isempty(keep) && return nothing
    return axislegend(ax, elems[keep], string.(labels[keep]); kwargs...)
end

# ---------------------------------------------------------------------------
# Grouped bars
# ---------------------------------------------------------------------------
#
# Note the tick trap this helper does not fix for you: if you offset groups by
# some stride (Fig 3a uses 5.0) you must set `xticks` to the *offset* centres.
# Fig 3a sets `xticks = 1:4` while its groups sit at 1, 6, 11, 16, so only the
# first group is labelled. `group_centres` below returns the right positions.
function grouped_barplot!(ax, x_positions, values, labels, colors;
    bar_width = 0.8 / length(values),
    offset = 0,
    label_prefix = ""
)
    n_bars = length(values)
    for (i, (vals, label, color)) in enumerate(zip(values, labels, colors))
        positions = x_positions .+ (i - 1 - (n_bars - 1)/2) * bar_width .+ offset
        lbl = isempty(string(label)) ? "" : "$label_prefix$label"
        barplot!(ax, positions, vals;
            _labelled((; width = bar_width, color = color,
                         strokewidth = 0.5, strokecolor = :black), lbl)...)
    end
end

"""
    group_centres(n_groups, stride)

Centre positions for `n_groups` groups laid out with `stride` between them, for
use as `xticks`. Pass these rather than `1:n_groups`.
"""
group_centres(n_groups, stride) = [1 + (g - 1) * stride for g in 1:n_groups]

# Makie exports `errorbars!` already, and it handles log axes and grouping
# correctly; the hand-rolled version that used to live here shadowed it and drew
# whiskers in data units (so they changed width with the axis). Use Makie's.

export fig_style, save_fig, panel_title, sub_label, sup_label
export config_axis!, hline!, vline!, fit_ylims!, categorical_x!
export legend_outside!, safe_axislegend!, grouped_barplot!, group_centres
export OKABE_ITO, VIRIDIS, CIVIDIS, SINGLE_COL, DOUBLE_COL, FULL_PAGE
export FULL_W, HALF_W, TEXT_H, H_ROW1, H_ROW2, H_ROW3
export C_HOLO, C_ANTIHOLO, C_PROBE, C_BASIN, C_BAD, C_AXIS, C_HEBB
export C_LIEP, C_BP, C_STATIC, C_READOUT, C_IMPAIRED, C_REFERENCE, C_INSIDE
export FONT_SIZE_SMALL, FONT_SIZE_NORMAL, FONT_SIZE_LARGE
