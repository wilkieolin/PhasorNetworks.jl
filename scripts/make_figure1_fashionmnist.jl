#!/usr/bin/env julia
# Figure 1: LockinEP on FashionMNIST — Core Result

using Pkg
Pkg.activate(@__DIR__)

using CairoMakie, DataFrames, CSV, Statistics, LaTeXStrings
include("figure_style.jl")

# Load data
epoch_df = CSV.read("results/ep_fashionmnist/epoch_curves.csv", DataFrame)
calib_df = CSV.read("results/ep_fashionmnist/lockin_calibration.csv", DataFrame)
wnorm_df = CSV.read("results/ep_fashionmnist/gradient_fidelity_vs_weightnorm.csv", DataFrame)

# ============================================================
# Figure 1A: Training Curves
# ============================================================
fig1a = Figure(size = (SINGLE_COL * 100, SINGLE_COL * 100 * 0.7))
ax1a = Axis(fig1a[1, 1]; 
    xlabel = "Epoch", ylabel = "Test Accuracy",
    title = "FashionMNIST 784→256→64",
    xticks = 0:5:20
)

# StaticEP (20 epochs)
static_df = filter(:method => ==("StaticEP"), epoch_df)
lines!(ax1a, static_df.epoch, static_df.test_acc; 
    color = OKABE_ITO[3], label = "StaticEP (centered, β=0.1)")
scatter!(ax1a, static_df.epoch, static_df.test_acc; color = OKABE_ITO[3])

# LockinEP (10 epochs)
lockin_df = filter(:method => ==("LockinEP"), epoch_df)
lines!(ax1a, lockin_df.epoch, lockin_df.test_acc; 
    color = OKABE_ITO[2], label = "LockinEP (ε=0.03, ω_p=0.02)")
scatter!(ax1a, lockin_df.epoch, lockin_df.test_acc; color = OKABE_ITO[2])

safe_axislegend!(ax1a; position = :rb)
ylims!(ax1a, 0.75, 0.88)

# Peak accuracy annotation
text!(ax1a, 10, 0.843, text = "0.8423", color = OKABE_ITO[2], fontsize = 7, align = (:center, :bottom))
text!(ax1a, 8, 0.837, text = "0.8361", color = OKABE_ITO[3], fontsize = 7, align = (:center, :bottom))

# ============================================================
# Figure 1B: Gradient Fidelity Calibration
# ============================================================
fig1b = Figure(size = (SINGLE_COL * 100, SINGLE_COL * 100 * 0.7))
ax1b = Axis(fig1b[1, 1];
    xlabel = "Probe frequency ω_p", ylabel = "Cosine similarity (vs StaticEP)",
    title = "LockinEP Gradient Calibration (ε=0.03)",
    xscale = log10,
    xticks = ([0.005, 0.01, 0.02, 0.05, 0.1], ["0.005", "0.01", "0.02", "0.05", "0.1"])
)

lines!(ax1b, calib_df.omega_p, calib_df.cos_L1; color = OKABE_ITO[1], label = "Layer 1")
scatter!(ax1b, calib_df.omega_p, calib_df.cos_L1; color = OKABE_ITO[1])
lines!(ax1b, calib_df.omega_p, calib_df.cos_L2; color = OKABE_ITO[4], label = "Layer 2")
scatter!(ax1b, calib_df.omega_p, calib_df.cos_L2; color = OKABE_ITO[4])

# Highlight operating point
op_idx = findfirst(==(0.02f0), calib_df.omega_p)
if op_idx !== nothing
    scatter!(ax1b, [calib_df.omega_p[op_idx]], [calib_df.cos_L1[op_idx]]; 
        color = :red, markersize = 10, marker = :star5)
    text!(ax1b, calib_df.omega_p[op_idx], calib_df.cos_L1[op_idx] + 0.01, 
        text = "Operating point\n(cos=0.998)", fontsize = 6, color = :red, align = (:center, :bottom))
end

safe_axislegend!(ax1b; position = :lb)
ylims!(ax1b, 0.9, 1.01)

# Twin axis: steps/gradient
ax1b_twin = Axis(fig1b[1, 1], 
    yaxisposition = :right,
    ylabel = "Steps/gradient",
    yscale = log10
)
hidedecorations!(ax1b_twin; label = false, ticklabels = false, grid = false)
lines!(ax1b_twin, calib_df.omega_p, calib_df.steps_per_grad; color = :gray, linestyle = :dash)

# ============================================================
# Figure 1C: Weight Norm Effect (reshape long format)
# ============================================================
fig1c = Figure(size = (SINGLE_COL * 100, SINGLE_COL * 100 * 0.7))
ax1c = Axis(fig1c[1, 1];
    xlabel = "Weight norm ‖W₁‖", ylabel = "Cosine similarity (vs centered StaticEP)",
    title = "Basin Hopping: Gradient Fidelity vs ‖W‖"
)

# One-sided (centered=false, beta=0.1)
onesided = filter(r -> r.centered == false && r.beta == 0.1, wnorm_df)
sort!(onesided, :w1_norm)
lines!(ax1c, onesided.w1_norm, onesided.cos_L1; 
    color = OKABE_ITO[7], label = "One-sided β=0.1", linestyle = :dash)
scatter!(ax1c, onesided.w1_norm, onesided.cos_L1; color = OKABE_ITO[7])

# Centered (centered=true)
centered = filter(r -> r.centered == true, wnorm_df)
for beta in unique(centered.beta)
    sub = filter(r -> r.beta == beta, centered)
    sort!(sub, :w1_norm)
    col = beta == 0.1 ? OKABE_ITO[3] : (beta == 0.03 ? OKABE_ITO[2] : OKABE_ITO[6])
    lines!(ax1c, sub.w1_norm, sub.cos_L1; color = col, label = "Centered β=$beta")
    scatter!(ax1c, sub.w1_norm, sub.cos_L1; color = col)
end

# Vertical line at ‖W‖=15 (basin hop onset)
vline!(ax1c, 15; color = :red, linestyle = :dot, linewidth = 1)
text!(ax1c, 15, 0.5, text = "‖W‖≈15", fontsize = 6, color = :red, rotation = π/2, align = (:center, :center))

safe_axislegend!(ax1c; position = :lb)
ylims!(ax1c, -0.05, 1.05)

# ============================================================
# Combine into Figure 1
# ============================================================
fig1 = Figure(size = (DOUBLE_COL * 100, SINGLE_COL * 100 * 1.1))

# Layout: 1A full width top, 1B and 1C bottom row
ga = fig1[1, 1:2] = GridLayout()
gb = fig1[2, 1] = GridLayout()
gc = fig1[2, 2] = GridLayout()

# Panel (a) - Training curves
ax_combined_1a = Axis(ga[1, 1]; 
    xlabel = "Epoch", ylabel = "Test Accuracy",
    title = panel_title("a", "FashionMNIST 784→ 256→ 64"),
    xticks = 0:5:20
)
lines!(ax_combined_1a, static_df.epoch, static_df.test_acc; color = OKABE_ITO[3], label = "StaticEP")
scatter!(ax_combined_1a, static_df.epoch, static_df.test_acc; color = OKABE_ITO[3])
lines!(ax_combined_1a, lockin_df.epoch, lockin_df.test_acc; color = OKABE_ITO[2], label = "LockinEP")
scatter!(ax_combined_1a, lockin_df.epoch, lockin_df.test_acc; color = OKABE_ITO[2])
safe_axislegend!(ax_combined_1a; position = :rb)
ylims!(ax_combined_1a, 0.75, 0.88)
text!(ax_combined_1a, 10, 0.843, text = "0.8423", color = OKABE_ITO[2], fontsize = 7, align = (:center, :bottom))
text!(ax_combined_1a, 8, 0.837, text = "0.8361", color = OKABE_ITO[3], fontsize = 7, align = (:center, :bottom))

# Panel (b) - Calibration
ax_combined_1b = Axis(gb[1, 1];
    xlabel = "Probe frequency ω_p", ylabel = "Cosine similarity",
    title = panel_title("b", "Gradient Calibration"),
    xscale = log10,
    xticks = ([0.005, 0.01, 0.02, 0.05, 0.1], ["0.005", "0.01", "0.02", "0.05", "0.1"])
)
lines!(ax_combined_1b, calib_df.omega_p, calib_df.cos_L1; color = OKABE_ITO[1], label = "Layer 1")
scatter!(ax_combined_1b, calib_df.omega_p, calib_df.cos_L1; color = OKABE_ITO[1])
lines!(ax_combined_1b, calib_df.omega_p, calib_df.cos_L2; color = OKABE_ITO[4], label = "Layer 2")
scatter!(ax_combined_1b, calib_df.omega_p, calib_df.cos_L2; color = OKABE_ITO[4])
safe_axislegend!(ax_combined_1b; position = :lb)
ylims!(ax_combined_1b, 0.9, 1.01)

# Panel (c) - Weight norm
ax_combined_1c = Axis(gc[1, 1];
    xlabel = "Weight norm ‖W₁‖", ylabel = "Cosine similarity",
    title = panel_title("c", "Basin Hopping")
)
lines!(ax_combined_1c, onesided.w1_norm, onesided.cos_L1; 
    color = OKABE_ITO[7], label = "One-sided β=0.1", linestyle = :dash)
scatter!(ax_combined_1c, onesided.w1_norm, onesided.cos_L1; color = OKABE_ITO[7])
for beta in unique(centered.beta)
    sub = filter(r -> r.beta == beta, centered)
    sort!(sub, :w1_norm)
    col = beta == 0.1 ? OKABE_ITO[3] : (beta == 0.03 ? OKABE_ITO[2] : OKABE_ITO[6])
    lines!(ax_combined_1c, sub.w1_norm, sub.cos_L1; color = col, label = "Centered β=$beta")
    scatter!(ax_combined_1c, sub.w1_norm, sub.cos_L1; color = col)
end
vline!(ax_combined_1c, 15; color = :red, linestyle = :dot)
safe_axislegend!(ax_combined_1c; position = :lb)
ylims!(ax_combined_1c, -0.05, 1.05)

rowgap!(fig1.layout, 10)
colgap!(fig1.layout, 10)

# Save
save_fig(fig1a, "fig1a_training_curves")
save_fig(fig1b, "fig1b_calibration")
save_fig(fig1c, "fig1c_weight_norm")
save_fig(fig1, "figure1_fashionmnist_core")

println("Figure 1 panels saved to figures/")
