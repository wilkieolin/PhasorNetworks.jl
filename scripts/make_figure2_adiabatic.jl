#!/usr/bin/env julia
# Figure 2: Adiabatic Basin — Operating Zone & Failure Rates

using Pkg
Pkg.activate(@__DIR__)

using CairoMakie, DataFrames, CSV, Statistics, LaTeXStrings, KernelDensity
include("figure_style.jl")

# Load adiabatic grid data
grid_df = CSV.read("results/ep_adiabatic/grid.csv", DataFrame)

# Compute median cos_l1 per (ε, ω_p) cell
agg_df = combine(groupby(grid_df, [:eps, :omega_p]),
    :cos_l1 => median => :cos_l1_median,
    :cos_l1 => (x -> mean(x .< 0.9)) => :fail_rate,
    :cos_l1 => (x -> mean(x .>= 0.9)) => :pass_rate,
    :relerr_l1 => median => :relerr_median,
    nrow => :n_samples
)

# ============================================================
# Figure 2A: Adiabatic Heatmap - use heatmap with linear scale, manual log ticks
# ============================================================
fig2a = Figure(size = (SINGLE_COL * 100, SINGLE_COL * 100))
ax2a = Axis(fig2a[1, 1];
    xlabel = "Probe frequency ω_p", ylabel = "Probe amplitude ε",
    title = panel_title("a", "Adiabatic Zone: Median cos(L1)"),
)

# Create heatmap data matrix
eps_vals = sort(unique(agg_df.eps))
omega_vals = sort(unique(agg_df.omega_p))
heatmap_data = Matrix{Float32}(undef, length(eps_vals), length(omega_vals))
for (i, eps) in enumerate(eps_vals), (j, omega) in enumerate(omega_vals)
    row = filter(r -> r.eps ≈ eps && r.omega_p ≈ omega, agg_df)
    heatmap_data[i, j] = isempty(row) ? NaN : row.cos_l1_median[1]
end

# Use heatmap on linear indices, then set custom tick labels
hm = heatmap!(ax2a, 1:length(omega_vals), 1:length(eps_vals), heatmap_data'; 
    colormap = VIRIDIS, colorrange = (0.0, 1.0))
Colorbar(fig2a[1, 2], hm; label = "Median cos(L1)", width = 12)

# Set custom tick labels to show log-scale values
ax2a.xticks = (1:length(omega_vals), [string(v) for v in omega_vals])
ax2a.yticks = (1:length(eps_vals), [string(v) for v in eps_vals])

# Contour lines
contour!(ax2a, 1:length(omega_vals), 1:length(eps_vals), heatmap_data'; 
    levels = [0.9, 0.95, 0.99],
    color = :white, linewidth = 1.5, linestyle = :dash)

# Highlight operating point (find indices)
op_eps_idx = findfirst(≈(0.03f0), eps_vals)
op_omega_idx = findfirst(≈(0.02f0), omega_vals)
if op_eps_idx !== nothing && op_omega_idx !== nothing
    scatter!(ax2a, [op_omega_idx], [op_eps_idx]; color = :red, markersize = 12, marker = :star5)
    text!(ax2a, op_omega_idx, op_eps_idx + 0.3, text = "Operating\npoint", fontsize = 6, color = :red, align = (:center, :bottom))
end

# ============================================================
# Figure 2B: Failure Rate vs ω_p/R_relax
# ============================================================
R_relax = 0.1f0
agg_df.omega_over_R = agg_df.omega_p ./ R_relax

fig2b = Figure(size = (SINGLE_COL * 100, SINGLE_COL * 100))
ax2b = Axis(fig2b[1, 1];
    xlabel = sub_label("ω_p / R", "relax"), ylabel = "Failure rate (cos < 0.9)",
    title = panel_title("b", "Failure Rate vs Normalized Probe Frequency"),
    xscale = log10, yscale = log10
)

eps_colors = [OKABE_ITO[3], OKABE_ITO[2], OKABE_ITO[4], OKABE_ITO[6], OKABE_ITO[7]]
for (eps, color) in zip(sort(unique(agg_df.eps)), eps_colors)
    sub = filter(r -> r.eps ≈ eps, agg_df)
    sort!(sub, :omega_over_R)
    lines!(ax2b, sub.omega_over_R, sub.fail_rate; color, label = "ε=$eps")
    scatter!(ax2b, sub.omega_over_R, sub.fail_rate; color)
end

vline!(ax2b, 1.0; color = :red, linestyle = :dot, label = "ω_p = R_relax")
text!(ax2b, 1.0, 0.5, text = "ω_p = R_relax", fontsize = 6, color = :red, rotation = π/2)

safe_axislegend!(ax2b; position = :lt)
ylims!(ax2b, 1e-3, 1.1)

# ============================================================
# Figure 2C: Bimodal Distribution at Problematic Cell
# ============================================================
problem_cell = filter(r -> r.eps ≈ 0.03f0 && r.omega_p ≈ 0.02f0, grid_df)

fig2c = Figure(size = (SINGLE_COL * 100, SINGLE_COL * 100))
ax2c = Axis(fig2c[1, 1];
    xlabel = "Cosine similarity (cos_l1)", ylabel = "Density",
    title = panel_title("c", "Bimodal Failure at ε=0.03, ω_p=0.02")
)

hist!(ax2c, problem_cell.cos_l1; 
    bins = 30, color = (OKABE_ITO[2], 0.7), strokewidth = 0.5, strokecolor = :black,
    normalization = :pdf
)
kde = KernelDensity.kde(problem_cell.cos_l1)
lines!(ax2c, kde.x, kde.density; color = OKABE_ITO[2], linewidth = 2)

median_cos = median(problem_cell.cos_l1)
vline!(ax2c, median_cos; color = :red, linestyle = :dash, linewidth = 1.5)
text!(ax2c, median_cos, maximum(kde.density) * 0.9, 
    text = "Median = $(round(median_cos, digits=3))", fontsize = 7, color = :red, align = (:center, :top))

text!(ax2c, 0.5, maximum(kde.density) * 0.5, 
    text = "Bimodal:\nbasin hopping", fontsize = 7, color = :black, align = (:center, :center))

# ============================================================
# Combined Figure 2
# ============================================================
fig2 = Figure(size = (DOUBLE_COL * 100, SINGLE_COL * 100 * 1.2))

# Top row: 2A full width
ax2a_comb = Axis(fig2[1, 1:2];
    xlabel = "Probe frequency ω_p", ylabel = "Probe amplitude ε",
    title = panel_title("a", "Adiabatic Zone: Median cos(L1)")
)
hm2 = heatmap!(ax2a_comb, 1:length(omega_vals), 1:length(eps_vals), heatmap_data'; colormap = VIRIDIS, colorrange = (0.0, 1.0))
ax2a_comb.xticks = (1:length(omega_vals), [string(v) for v in omega_vals])
ax2a_comb.yticks = (1:length(eps_vals), [string(v) for v in eps_vals])
Colorbar(fig2[1, 3], hm2; label = "Median cos(L1)", width = 12)
contour!(ax2a_comb, 1:length(omega_vals), 1:length(eps_vals), heatmap_data'; levels = [0.9, 0.95, 0.99], color = :white, linewidth = 1.5, linestyle = :dash)
if op_eps_idx !== nothing && op_omega_idx !== nothing
    scatter!(ax2a_comb, [op_omega_idx], [op_eps_idx]; color = :red, markersize = 12, marker = :star5)
end

# Bottom left: 2B
ax2b_comb = Axis(fig2[2, 1];
    xlabel = sub_label("ω_p / R", "relax"), ylabel = "Failure rate (cos < 0.9)",
    title = panel_title("b", "Failure Rate vs Normalized Probe"),
    xscale = log10, yscale = log10
)
for (eps, color) in zip(sort(unique(agg_df.eps)), eps_colors)
    sub = filter(r -> r.eps ≈ eps, agg_df)
    sort!(sub, :omega_over_R)
    lines!(ax2b_comb, sub.omega_over_R, sub.fail_rate; color, label = "ε=$eps")
    scatter!(ax2b_comb, sub.omega_over_R, sub.fail_rate; color)
end
vline!(ax2b_comb, 1.0; color = :red, linestyle = :dot)
safe_axislegend!(ax2b_comb; position = :lt)
ylims!(ax2b_comb, 1e-3, 1.1)

# Bottom right: 2C
ax2c_comb = Axis(fig2[2, 2];
    xlabel = "Cosine similarity (cos_l1)", ylabel = "Density",
    title = panel_title("c", "Bimodal Failure at ε=0.03, ω_p=0.02")
)
hist!(ax2c_comb, problem_cell.cos_l1; bins = 30, color = (OKABE_ITO[2], 0.7), 
      strokewidth = 0.5, strokecolor = :black, normalization = :pdf)
kde = KernelDensity.kde(problem_cell.cos_l1)
lines!(ax2c_comb, kde.x, kde.density; color = OKABE_ITO[2], linewidth = 2)
vline!(ax2c_comb, median_cos; color = :red, linestyle = :dash, linewidth = 1.5)

rowgap!(fig2.layout, 10)
colgap!(fig2.layout, 10)

save_fig(fig2a, "fig2a_heatmap")
save_fig(fig2b, "fig2b_failure_rate")
save_fig(fig2c, "fig2c_bimodal")
save_fig(fig2, "figure2_adiabatic_basin")

println("Figure 2 saved to figures/")
