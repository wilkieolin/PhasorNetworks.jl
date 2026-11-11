#!/usr/bin/env julia
# Figure 2 -- the adiabatic operating regime.
#
# Data:
#   results/ep_adiabatic/grid.csv          960 runs: eps x omega_p x n_cycles x 8 draws
#   results/ep_adiabatic/variance_paired.csv  32 draws at the operating point
#   results/e6_basin_training/e6_basin_*.csv  training at 5 points on that map
#
# Two things about grid.csv drive the design and must not be smoothed over.
#
# 1. `rep` is an INPUT DRAW, not a seed. ep_adiabatic_sweep.jl:470 builds
#    `inputs = [(make_input(100 + r), make_cost(codes, 100 + r)) for r in 1:REPS]`
#    and rep r indexes into it. So a k/8 "failure rate" pools a fixed per-input
#    effect with sampling variation, and one input dominates: draw 6 fails in 19
#    of the 20 (eps, n_cycles) cells at omega_p = 0.02, and 82/120 grid-wide
#    against ~53 for draws 1-5 and 7. Panel (c) therefore shows the per-draw
#    spread rather than a bare rate.
#
# 2. Failure here is DECORRELATION, not sign inversion. Median cos_l1 falls
#    0.999 -> 0.997 -> 0.989 -> 0.885 -> 0.254 -> 0.110 across the omega_p
#    ladder; only 14 of 960 runs go negative and the most negative is -0.039.
#    The anticorrelated phase-flip failure that Section 2 describes is real but
#    lives elsewhere -- variance_paired.csv, 1 draw in 32 at cos = -0.120 -- and
#    is a large-amplitude/projection effect, not the frequency wall. Do not draw
#    them as one mechanism.

using Pkg; Pkg.activate(@__DIR__)
using CairoMakie, DataFrames, CSV, Statistics, Printf
include("figure_style.jl")

const RES = joinpath(@__DIR__, "..", "results")
const FAIL_COS = 0.9          # same threshold the previous figure used
const NC_MAIN  = 4
const WTICKS = [0.005, 0.02, 0.05, 0.2]            # operating default; other n_cycles go to supplementary

grid = CSV.read(joinpath(RES, "ep_adiabatic", "grid.csv"), DataFrame)
grid.fail = grid.cos_l1 .< FAIL_COS

# ---------------------------------------------------------------- boundary fit
# Logistic regression of failure on log10(omega_p), restricted to eps >= 0.01.
# Restricted because the eps dependence is NOT monotone: eps >= 0.01 behaves
# identically (the curves lie on top of one another) while eps = 0.003 fails
# from below, at the readout floor. A two-variable logistic would impose a
# monotone eps trend that the data contradicts. The lower eps bound is reported
# separately, in panel (b).
function logistic_fit(x, y; iters = 60)
    b = [0.0, 0.0]                       # intercept, slope
    for _ in 1:iters
        eta = b[1] .+ b[2] .* x
        p = 1 ./ (1 .+ exp.(-eta))
        w = max.(p .* (1 .- p), 1e-9)
        r = y .- p
        # 2x2 weighted normal equations
        s0 = sum(w); s1 = sum(w .* x); s2 = sum(w .* (x .^ 2))
        g0 = sum(r);  g1 = sum(r .* x)
        det = s0 * s2 - s1^2
        abs(det) < 1e-12 && break
        db = [( s2 * g0 - s1 * g1) / det, (-s1 * g0 + s0 * g1) / det]
        b .+= db
        maximum(abs, db) < 1e-10 && break
    end
    return b
end

sub = grid[grid.eps .>= 0.01, :]
bcoef = logistic_fit(log10.(Float64.(sub.omega_p)), Float64.(sub.fail))
omega_star = 10.0^(-bcoef[1] / bcoef[2])
@printf("boundary: p_fail = 0.5 at omega_p = %.4f  (slope %.2f per decade, n = %d, eps >= 0.01)\n",
        omega_star, bcoef[2], nrow(sub))

wilson(k, n; z = 1.96) = begin
    p = k / n; d = 1 + z^2 / n
    c = (p + z^2 / (2n)) / d
    h = z * sqrt(p * (1 - p) / n + z^2 / (4n^2)) / d
    (max(0.0, c - h), min(1.0, c + h))
end

fig = Figure(size = (FULL_W, 330))

# ---------------------------------------------------------------- (a) basin map
g4 = grid[grid.n_cycles .== NC_MAIN, :]
epsv = sort(unique(g4.eps)); omv = sort(unique(g4.omega_p))
Z = [median(g4[(g4.eps .== e) .& (g4.omega_p .== w), :cos_l1]) for w in omv, e in epsv]
# geometric cell edges so a log axis shows equal-width cells
edges(v) = vcat(v[1] / sqrt(v[2] / v[1]), sqrt.(v[1:end-1] .* v[2:end]), v[end] * sqrt(v[end] / v[end-1]))

axa = Axis(fig[1, 1]; title = panel_title("a", "Gradient fidelity vs (ε, ω_p)"),
    xlabel = "probe frequency  ω_p", ylabel = "probe amplitude  ε",
    xscale = log10, yscale = log10, titlealign = :left,
    xticks = (WTICKS, string.(WTICKS)), yticks = (epsv, string.(epsv)))
hm = heatmap!(axa, edges(omv), edges(epsv), Z; colormap = :viridis, colorrange = (0, 1))
Colorbar(fig[1, 2], hm; label = "median cos", width = 7, ticklabelsize = FONT_SIZE_SMALL - 1,
         labelsize = FONT_SIZE_SMALL - 1)
vlines!(axa, [omega_star]; color = :white, linestyle = :dash, linewidth = 1.4)
text!(axa, omega_star, epsv[end]; text = " boundary", color = :white,
      fontsize = FONT_SIZE_SMALL - 1, align = (:left, :top))

# E6 operating points, if the run has produced anything yet
e6files = filter(f -> startswith(basename(f), "e6_basin_") && endswith(f, ".csv"),
                 isdir(joinpath(RES, "e6_basin_training")) ?
                 readdir(joinpath(RES, "e6_basin_training"); join = true) : String[])
e6 = isempty(e6files) ? nothing : CSV.read(last(sort(e6files)), DataFrame)
if e6 !== nothing
    pts = unique(select(e6, [:point, :eps, :omega_p]))
    scatter!(axa, pts.omega_p, pts.eps; marker = :star5, markersize = 11,
             color = :white, strokecolor = :black, strokewidth = 0.6)
    for r in eachrow(pts)
        text!(axa, r.omega_p, r.eps; text = r.point, fontsize = FONT_SIZE_SMALL - 1,
              color = :white, align = (:center, :center), offset = (0, -11), font = :bold)
    end
end

# ---------------------------------------------------------------- (b) fidelity vs omega_p
axb = Axis(fig[1, 3]; title = panel_title("b", "The wall is in frequency"),
    xlabel = "probe frequency  ω_p", ylabel = "median cos",
    xscale = log10, titlealign = :left, xticks = (WTICKS, string.(WTICKS)))
for (i, e) in enumerate(epsv)
    s = grid[grid.eps .== e, :]
    m = combine(groupby(s, :omega_p), :cos_l1 => median => :med,
                :cos_l1 => (v -> quantile(v, 0.1)) => :lo,
                :cos_l1 => (v -> quantile(v, 0.9)) => :hi)
    sort!(m, :omega_p)
    below = e < 0.01
    # below the wall the four eps >= 0.01 curves are indistinguishable; above it
    # they fan out, larger eps retaining more correlation. One legend entry for
    # the family keeps the panel about omega_p, which is where the wall is.
    lab = below ? "ε = 0.003 (response too small)" :
          (e == 0.01 ? "ε = 0.01 – 0.3 (four values)" : "")
    lines!(axb, m.omega_p, m.med; color = below ? C_BAD : (C_HOLO, 0.30 + 0.17i),
           linewidth = below ? 2.0 : 1.3, linestyle = below ? :dash : :solid,
           label = isempty(lab) ? nothing : lab)
end
hlines!(axb, [FAIL_COS]; color = C_REFERENCE, linestyle = :dot, linewidth = 1.0)
vlines!(axb, [omega_star]; color = C_ANTIHOLO, linestyle = :dash, linewidth = 1.2)
text!(axb, omega_star, 1.05; text = @sprintf(" ω_p*=%.3f", omega_star),
      fontsize = FONT_SIZE_SMALL - 1, color = C_ANTIHOLO, align = (:left, :top))
axislegend(axb; position = :lb, framevisible = false, labelsize = FONT_SIZE_SMALL - 1,
           padding = (1, 1, 1, 1), rowgap = 0, patchsize = (11, 6))
ylims!(axb, -0.1, 1.08)

# ---------------------------------------------------------------- (c) failure rate
axc = Axis(fig[2, 1]; title = panel_title("c", "Failure rate, 8 input draws per cell"),
    xlabel = "probe frequency  ω_p", ylabel = "fraction of draws failing",
    xscale = log10, titlealign = :left, xticks = (WTICKS, string.(WTICKS)))
for (i, e) in enumerate(epsv)
    s = grid[grid.eps .== e, :]
    m = combine(groupby(s, :omega_p), :fail => sum => :k, nrow => :n)
    sort!(m, :omega_p)
    lo = [wilson(r.k, r.n)[1] for r in eachrow(m)]
    hi = [wilson(r.k, r.n)[2] for r in eachrow(m)]
    col = e < 0.01 ? C_BAD : (C_HOLO, 0.25 + 0.19i)
    band!(axc, m.omega_p, lo, hi; color = (e < 0.01 ? C_BAD : C_HOLO, 0.10))
    lines!(axc, m.omega_p, m.k ./ m.n; color = col, linewidth = e < 0.01 ? 2.0 : 1.3,
           linestyle = e < 0.01 ? :dash : :solid)
end
vlines!(axc, [omega_star]; color = C_ANTIHOLO, linestyle = :dash, linewidth = 1.2)
ylims!(axc, -0.05, 1.05)
text!(axc, 0.03, 0.95; space = :relative, align = (:left, :top), fontsize = FONT_SIZE_SMALL - 1,
      color = C_AXIS, text = "bands: Wilson 95%")

# ---------------------------------------------------------------- (d) training
axd = Axis(fig[2, 3]; title = panel_title("d", "Training at points A–E"),
    xlabel = "epoch", ylabel = "test accuracy", titlealign = :left)
if e6 === nothing || nrow(e6) == 0
    text!(axd, 0.5, 0.5; text = "awaiting scripts/e6_basin_training.jl", space = :relative,
          fontsize = FONT_SIZE_SMALL, color = C_REFERENCE, align = (:center, :center))
    hidedecorations!(axd); hidespines!(axd)
else
    POINT_COL = Dict("A" => C_HOLO, "B" => C_BASIN, "C" => C_PROBE,
                     "D" => C_ANTIHOLO, "E" => C_BAD)
    for p in sort(unique(e6.point))
        s = e6[e6.point .== p, :]
        m = combine(groupby(s, :epoch), :test_acc => median => :med,
                    :test_acc => minimum => :lo, :test_acc => maximum => :hi, nrow => :n)
        sort!(m, :epoch)
        reg = first(s.region); w = first(s.omega_p); ee = first(s.eps)
        band!(axd, m.epoch, m.lo, m.hi; color = (POINT_COL[p], 0.15))
        # eps is 0.03 for A-D and only differs at E, so name the knob that varies
        # eps is 0.03 for A-D and only differs at E, so name the knob that varies.
        # E is labelled by its amplitude rather than by a verdict -- whether small eps
        # merely slows training or stops it is what this panel is measuring.
        disp = Dict("inside" => "inside", "operating" => "operating point",
                    "boundary" => "at the boundary", "outside" => "outside",
                    "below_eps_floor" => "small amplitude")
        lab = p == "E" ? @sprintf("E  ε = %g", ee) : @sprintf("%s  ω_p = %g", p, w)
        lines!(axd, m.epoch, m.med; color = POINT_COL[p], linewidth = LINE_WIDTH,
               label = lab * "  " * get(disp, reg, reg))
    end
    axislegend(axd; position = :rc, framevisible = false, labelsize = FONT_SIZE_SMALL - 1,
               padding = (1, 1, 1, 1), rowgap = -1, patchsize = (10, 5))
end

colsize!(fig.layout, 1, Relative(0.46))
colgap!(fig.layout, 6); rowgap!(fig.layout, 8)
save_fig(fig, "figure2_basin")
println("figure2_basin written")
