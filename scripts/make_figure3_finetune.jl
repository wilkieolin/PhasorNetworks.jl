#!/usr/bin/env julia
# Figure 3: Analog Defect Repair — Fine-Tuning Recovery
# New format: damage-recovery curves (x = impaired accuracy, y = recovered accuracy)

using Pkg
Pkg.activate(@__DIR__)

using CairoMakie, DataFrames, CSV, Statistics, LaTeXStrings
include("figure_style.jl")

# ============================================================
# Load data
# ============================================================
# Main E2 sweep (StaticEP fine-tuning, REPS=8)
ft_df = CSV.read("results/ep_analog_finetune/analog_finetune_ddbaa3d.csv", DataFrame)

# Backprop transfer (for Fig 3c)
bp_transfer = CSV.read("results/ep_backprop_transfer/backprop_transfer_ddbaa3d.csv", DataFrame)

# Filter: only rows with meaningful degradation (denom > 0.01)
ft_clean = filter(r -> !isnan(r.recovery_frac) || r.recovery_frac > 0, ft_df)

# Impairment types and colors
impairment_order = ["stuck_sat", "gaussian", "stuck_zero", "lognormal"]
impairment_labels = Dict(
    "stuck_sat" => "Stuck-at-sat",
    "gaussian" => "Gaussian noise",
    "stuck_zero" => "Stuck-at-zero",
    "lognormal" => "Lognormal mult."
)
impairment_colors = Dict(
    "stuck_sat" => OKABE_ITO[2],
    "gaussian" => OKABE_ITO[3],
    "stuck_zero" => OKABE_ITO[4],
    "lognormal" => OKABE_ITO[6]
)
pretrain_labels = Dict("backprop" => "BP pretrain", "staticep" => "StaticEP pretrain")
pretrain_markers = Dict("backprop" => :circle, "staticep" => :diamond)

# ============================================================
# Figure 3A: Damage-Recovery Curves (one panel per impairment)
# ============================================================
fig3a = Figure(size = (DOUBLE_COL * 100, DOUBLE_COL * 100 * 0.8))

# 2x2 grid of panels
for (imp_idx, impairment) in enumerate(impairment_order)
    row_idx = (imp_idx - 1) ÷ 2 + 1
    col_idx = (imp_idx - 1) % 2 + 1
    
    ax = Axis(fig3a[row_idx, col_idx];
        xlabel = "Impaired accuracy",
        ylabel = "Recovered accuracy",
        title = panel_title(string('a' + imp_idx - 1), impairment_labels[impairment]),
        aspect = 1,
        xgridstyle = :dash, ygridstyle = :dash
    )
    
    # Diagonal reference line (y = x)
    x_diag = 0:0.01:0.9
    lines!(ax, x_diag, x_diag; color = :gray, linestyle = :dash, linewidth = 0.8, label = "y = x")
    
    sub = filter(r -> r.impairment_type == impairment, ft_clean)
    
    # Plot each pretrain method
    for (pretrain, marker) in [("backprop", :circle), ("staticep", :diamond)]
        color = impairment_colors[impairment]
        label_prefix = pretrain_labels[pretrain]
        
        sub_pre = filter(r -> r.pretrain == pretrain, sub)
        
        # Aggregate by impaired accuracy (median over reps at each param)
        agg = combine(groupby(sub_pre, :param),
            :acc_impaired => median => :acc_impaired_median,
            :acc_tuned => median => :acc_tuned_median,
            :acc_bp_ft => median => :acc_bp_ft_median,
            :acc_readout_only => median => :acc_readout_only_median,
            :acc_impaired => (x -> std(x)/sqrt(length(x))) => :acc_impaired_sem,
            :acc_tuned => (x -> std(x)/sqrt(length(x))) => :acc_tuned_sem,
            :acc_bp_ft => (x -> std(x)/sqrt(length(x))) => :acc_bp_ft_sem,
            :acc_readout_only => (x -> std(x)/sqrt(length(x))) => :acc_readout_only_sem,
            nrow => :n_reps
        )
        
        # Sort by impaired accuracy for line continuity
        sort!(agg, :acc_impaired_median)
        
        # LIEP (StaticEP fine-tuning proxy) - solid line
        lines!(ax, agg.acc_impaired_median, agg.acc_tuned_median; 
            color = color, linewidth = 2, linestyle = :solid,
            label = imp_idx == 1 && pretrain == "backprop" ? "$label_prefix (LIEP)" : "")
        band!(ax, agg.acc_impaired_median, 
              agg.acc_tuned_median .- agg.acc_tuned_sem,
              agg.acc_tuned_median .+ agg.acc_tuned_sem;
            color = (color, 0.2), label = "")
        scatter!(ax, agg.acc_impaired_median, agg.acc_tuned_median;
            color = color, marker = marker, markersize = 8,
            strokewidth = 0.5, strokecolor = :black)
        
        # Backprop FT (ceiling) - dashed line
        lines!(ax, agg.acc_impaired_median, agg.acc_bp_ft_median;
            color = OKABE_ITO[4], linewidth = 2, linestyle = :dash,
            label = imp_idx == 1 && pretrain == "backprop" ? "Backprop FT (ceiling)" : "")
        band!(ax, agg.acc_impaired_median,
              agg.acc_bp_ft_median .- agg.acc_bp_ft_sem,
              agg.acc_bp_ft_median .+ agg.acc_bp_ft_sem;
            color = (OKABE_ITO[4], 0.2), label = "")
        scatter!(ax, agg.acc_impaired_median, agg.acc_bp_ft_median;
            color = OKABE_ITO[4], marker = marker, markersize = 6,
            strokewidth = 0.5, strokecolor = :black)
        
        # Readout-only (floor) - dotted line
        lines!(ax, agg.acc_impaired_median, agg.acc_readout_only_median;
            color = OKABE_ITO[7], linewidth = 2, linestyle = :dot,
            label = imp_idx == 1 && pretrain == "backprop" ? "Readout-only (floor)" : "")
        band!(ax, agg.acc_impaired_median,
              agg.acc_readout_only_median .- agg.acc_readout_only_sem,
              agg.acc_readout_only_median .+ agg.acc_readout_only_sem;
            color = (OKABE_ITO[7], 0.2), label = "")
        scatter!(ax, agg.acc_impaired_median, agg.acc_readout_only_median;
            color = OKABE_ITO[7], marker = marker, markersize = 6,
            strokewidth = 0.5, strokecolor = :black)
    end
    
    xlims!(ax, 0, 0.9)
    ylims!(ax, 0, 0.9)
    
    # Legend only on first panel
    if imp_idx == 1
        safe_axislegend!(ax; position = :lb, nbanks = 2, framevisible = true)
    end
end

rowgap!(fig3a.layout, 15)
colgap!(fig3a.layout, 15)

# ============================================================
# Figure 3B: Impaired vs Recovered Scatter (all data combined)
# ============================================================
fig3b = Figure(size = (SINGLE_COL * 100, SINGLE_COL * 100))
ax3b = Axis(fig3b[1, 1];
    xlabel = "Impaired accuracy", ylabel = "Recovered accuracy",
    title = panel_title("b", "Impaired vs. Recovered Accuracy"),
    aspect = 1
)

x_diag = 0:0.01:0.9
lines!(ax3b, x_diag, x_diag; color = :gray, linestyle = :dash, linewidth = 0.8)

for row in eachrow(ft_clean)
    scatter!(ax3b, [row.acc_impaired], [row.acc_tuned];
        color = impairment_colors[row.impairment_type],
        marker = pretrain_markers[row.pretrain],
        markersize = 6,
        strokewidth = 0.5,
        strokecolor = :black,
        alpha = 0.7
    )
end

text!(ax3b, 0.05, 0.85, text = "○ BP pretrain  ◆ StaticEP pretrain", fontsize = 7, align = (:left, :center))
text!(ax3b, 0.05, 0.78, text = "■ Stuck-sat  ■ Gaussian", fontsize = 7, align = (:left, :center), color = OKABE_ITO[2])
text!(ax3b, 0.05, 0.71, text = "■ Stuck-zero  ■ Lognormal", fontsize = 7, align = (:left, :center), color = OKABE_ITO[4])

xlims!(ax3b, 0, 0.9)
ylims!(ax3b, 0, 0.9)

# ============================================================
# Figure 3C: Backprop → EP Transfer (from CSV)
# ============================================================
bp_acc = bp_transfer.bp_acc[1]
ep_acc = bp_transfer.ep_acc[1]
drop = bp_transfer.drop[1]

fig3c = Figure(size = (SINGLE_COL * 100, SINGLE_COL * 100))
ax3c = Axis(fig3c[1, 1];
    ylabel = "Test Accuracy",
    title = panel_title("c", "Backprop → EP Transfer"),
    xticks = (1:3, ["Backprop\n(feedforward)", "EP\nsettle", "Drop"]),
    yticks = 0.8:0.01:0.88
)

barplot!(ax3c, [1, 2], [bp_acc, ep_acc]; 
    color = [OKABE_ITO[3], OKABE_ITO[2]], width = 0.6,
    strokewidth = 0.5, strokecolor = :black)
barplot!(ax3c, [3], [drop]; 
    color = OKABE_ITO[7], width = 0.6, strokewidth = 0.5, strokecolor = :black)

text!(ax3c, 1, bp_acc + 0.002, text = "$(round(bp_acc*100, digits=1))%", fontsize = 8, align = (:center, :bottom), font = :bold)
text!(ax3c, 2, ep_acc + 0.002, text = "$(round(ep_acc*100, digits=1))%", fontsize = 8, align = (:center, :bottom), font = :bold)
text!(ax3c, 3, drop + 0.001, text = "$(round(drop*100, digits=2))%", fontsize = 8, align = (:center, :bottom), color = OKABE_ITO[7])

ylims!(ax3c, 0.80, 0.88)

# ============================================================
# Combined Figure 3
# ============================================================
fig3 = Figure(size = (DOUBLE_COL * 100, SINGLE_COL * 100 * 1.8))

# Top: 3A (2x2) full width
for (imp_idx, impairment) in enumerate(impairment_order)
    row_idx = (imp_idx - 1) ÷ 2 + 1
    col_idx = (imp_idx - 1) % 2 + 1
    
    ax = Axis(fig3[row_idx, col_idx];
        xlabel = "Impaired accuracy",
        ylabel = "Recovered accuracy",
        title = panel_title(string('a' + imp_idx - 1), impairment_labels[impairment]),
        aspect = 1,
        xgridstyle = :dash, ygridstyle = :dash
    )
    
    x_diag = 0:0.01:0.9
    lines!(ax, x_diag, x_diag; color = :gray, linestyle = :dash, linewidth = 0.8)
    
    sub = filter(r -> r.impairment_type == impairment, ft_clean)
    
    for (pretrain, marker) in [("backprop", :circle), ("staticep", :diamond)]
        color = impairment_colors[impairment]
        
        sub_pre = filter(r -> r.pretrain == pretrain, sub)
        
        agg = combine(groupby(sub_pre, :param),
            :acc_impaired => median => :acc_impaired_median,
            :acc_tuned => median => :acc_tuned_median,
            :acc_bp_ft => median => :acc_bp_ft_median,
            :acc_readout_only => median => :acc_readout_only_median,
            :acc_impaired => (x -> std(x)/sqrt(length(x))) => :acc_impaired_sem,
            :acc_tuned => (x -> std(x)/sqrt(length(x))) => :acc_tuned_sem,
            :acc_bp_ft => (x -> std(x)/sqrt(length(x))) => :acc_bp_ft_sem,
            :acc_readout_only => (x -> std(x)/sqrt(length(x))) => :acc_readout_only_sem,
            nrow => :n_reps
        )
        
        sort!(agg, :acc_impaired_median)
        
        # LIEP
        lines!(ax, agg.acc_impaired_median, agg.acc_tuned_median; 
            color = color, linewidth = 2, linestyle = :solid,
            label = imp_idx == 1 && pretrain == "backprop" ? "$(pretrain_labels[pretrain]) (LIEP)" : "")
        band!(ax, agg.acc_impaired_median, 
              agg.acc_tuned_median .- agg.acc_tuned_sem,
              agg.acc_tuned_median .+ agg.acc_tuned_sem;
            color = (color, 0.2), label = "")
        scatter!(ax, agg.acc_impaired_median, agg.acc_tuned_median;
            color = color, marker = marker, markersize = 8,
            strokewidth = 0.5, strokecolor = :black)
        
        # Backprop FT
        lines!(ax, agg.acc_impaired_median, agg.acc_bp_ft_median;
            color = OKABE_ITO[4], linewidth = 2, linestyle = :dash,
            label = imp_idx == 1 && pretrain == "backprop" ? "Backprop FT (ceiling)" : "")
        band!(ax, agg.acc_impaired_median,
              agg.acc_bp_ft_median .- agg.acc_bp_ft_sem,
              agg.acc_bp_ft_median .+ agg.acc_bp_ft_sem;
            color = (OKABE_ITO[4], 0.2), label = "")
        scatter!(ax, agg.acc_impaired_median, agg.acc_bp_ft_median;
            color = OKABE_ITO[4], marker = marker, markersize = 6,
            strokewidth = 0.5, strokecolor = :black)
        
        # Readout-only
        lines!(ax, agg.acc_impaired_median, agg.acc_readout_only_median;
            color = OKABE_ITO[7], linewidth = 2, linestyle = :dot,
            label = imp_idx == 1 && pretrain == "backprop" ? "Readout-only (floor)" : "")
        band!(ax, agg.acc_impaired_median,
              agg.acc_readout_only_median .- agg.acc_readout_only_sem,
              agg.acc_readout_only_median .+ agg.acc_readout_only_sem;
            color = (OKABE_ITO[7], 0.2), label = "")
        scatter!(ax, agg.acc_impaired_median, agg.acc_readout_only_median;
            color = OKABE_ITO[7], marker = marker, markersize = 6,
            strokewidth = 0.5, strokecolor = :black)
    end
    
    xlims!(ax, 0, 0.9)
    ylims!(ax, 0, 0.9)
    
    if imp_idx == 1
        safe_axislegend!(ax; position = :lb, nbanks = 2, framevisible = true)
    end
end

# Bottom left: 3B
ax3b_comb = Axis(fig3[3, 1];
    xlabel = "Impaired accuracy", ylabel = "Recovered accuracy",
    title = panel_title("b", "Impaired vs. Recovered"),
    aspect = 1
)
lines!(ax3b_comb, 0:0.01:0.9, 0:0.01:0.9; color = :gray, linestyle = :dash, linewidth = 0.8)
for row in eachrow(ft_clean)
    scatter!(ax3b_comb, [row.acc_impaired], [row.acc_tuned];
        color = impairment_colors[row.impairment_type],
        marker = pretrain_markers[row.pretrain],
        markersize = 6, strokewidth = 0.5, strokecolor = :black, alpha = 0.7
    )
end
text!(ax3b_comb, 0.05, 0.85, text = "○ BP pretrain  ◆ StaticEP pretrain", fontsize = 7, align = (:left, :center))
text!(ax3b_comb, 0.05, 0.78, text = "■ Stuck-sat  ■ Gaussian", fontsize = 7, align = (:left, :center), color = OKABE_ITO[2])
text!(ax3b_comb, 0.05, 0.71, text = "■ Stuck-zero  ■ Lognormal", fontsize = 7, align = (:left, :center), color = OKABE_ITO[4])
xlims!(ax3b_comb, 0, 0.9)
ylims!(ax3b_comb, 0, 0.9)

# Bottom right: 3C
ax3c_comb = Axis(fig3[3, 2];
    ylabel = "Test Accuracy",
    title = panel_title("c", "Backprop → EP Transfer"),
    xticks = (1:3, ["Backprop\n(feedforward)", "EP\nsettle", "Drop"]),
    yticks = 0.8:0.01:0.88
)
barplot!(ax3c_comb, [1, 2], [bp_acc, ep_acc]; color = [OKABE_ITO[3], OKABE_ITO[2]], width = 0.6)
barplot!(ax3c_comb, [3], [drop]; color = OKABE_ITO[7], width = 0.6)
text!(ax3c_comb, 1, bp_acc + 0.002, text = "$(round(bp_acc*100, digits=1))%", fontsize = 8, align = (:center, :bottom), font = :bold)
text!(ax3c_comb, 2, ep_acc + 0.002, text = "$(round(ep_acc*100, digits=1))%", fontsize = 8, align = (:center, :bottom), font = :bold)
text!(ax3c_comb, 3, drop + 0.001, text = "$(round(drop*100, digits=2))%", fontsize = 8, align = (:center, :bottom), color = OKABE_ITO[7])
ylims!(ax3c_comb, 0.80, 0.88)

rowgap!(fig3.layout, 15)
colgap!(fig3.layout, 15)

# Save
save_fig(fig3a, "fig3a_damage_recovery_curves")
save_fig(fig3b, "fig3b_scatter")
save_fig(fig3c, "fig3c_bptransfer")
save_fig(fig3, "figure3_finetune_recovery")

println("Figure 3 saved to figures/")