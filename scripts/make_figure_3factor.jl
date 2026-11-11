# Run with: julia --project=scripts scripts/make_figure_3factor.jl
using CSV, DataFrames, Statistics, CairoMakie, LaTeXStrings, Printf

df = CSV.read("results/e4_three_factor_scale/e4_three_factor_ddbaa3d.csv", DataFrame)

fig = Figure(size=(1200, 600), fontsize=14)

# 6 param groups: layer_1.weight, layer_1.bias_real, layer_1.bias_imag, layer_2.weight, layer_2.bias_real, layer_2.bias_imag
param_groups = ["layer_1.weight", "layer_1.bias_real", "layer_1.bias_imag", "layer_2.weight", "layer_2.bias_real", "layer_2.bias_imag"]
df.group = df.layer .* "." .* df.param

# Panel A: Cosine similarity by group
ax1 = Axis(fig[1,1], title="Cosine Similarity: LockinEP vs 3-Factor",
           ylabel=L"\cos(\theta)", xlabel="Parameter group",
           ytickformat=vals -> [@sprintf("%.6f", v) for v in vals],
           xticklabelrotation=pi/4)
bar_width = 0.4
x_pos = 1:length(param_groups)
for (i, g) in enumerate(param_groups)
    sub = filter(r -> r.group == g, df)
    barplot!(ax1, [i], [mean(sub.cos_lk_3f)], color=i<=3 ? :steelblue : :coral, width=bar_width)
    errorbars!(ax1, [i], [mean(sub.cos_lk_3f)], [std(sub.cos_lk_3f)], color=:black, whiskerwidth=4)
end
ax1.xticks = (x_pos, param_groups)
hlines!(ax1, [1.0], color=:gray, linestyle=:dash, linewidth=1)
# Legend
leg = Legend(fig[1,2], [PolyElement(polycolor=:steelblue), PolyElement(polycolor=:coral)], ["layer_1", "layer_2"], "Layer")

# Panel B: Relative error by group
ax2 = Axis(fig[2,1], title="Relative Error: LockinEP vs 3-Factor",
           ylabel=L"\|g_{3f} - g_{le}\| / \|g_{le}\|", xlabel="Parameter group",
           yscale=log10, xticklabelrotation=pi/4)
for (i, g) in enumerate(param_groups)
    sub = filter(r -> r.group == g, df)
    barplot!(ax2, [i], [mean(sub.rel_err_lk_3f)], color=i<=3 ? :steelblue : :coral, width=bar_width)
    errorbars!(ax2, [i], [mean(sub.rel_err_lk_3f)], [std(sub.rel_err_lk_3f)], color=:black, whiskerwidth=4)
end
ax2.xticks = (x_pos, param_groups)

# Panel C: Heatmap - cosine across all 100 inputs × 6 groups
ax3 = Axis(fig[1,3], title="Cosine Similarity Heatmap (100 inputs × 6 groups)",
           xlabel="Input sample", ylabel="Parameter group")
heatmap_data = zeros(100, 6)
for (j, g) in enumerate(param_groups)
    for k in 1:100
        row = filter(r -> r.group == g && r.input_idx == k, df)
        heatmap_data[k, j] = row.cos_lk_3f[1]
    end
end
hm = heatmap!(ax3, 1:100, 1:6, heatmap_data, colormap=:viridis, colorrange=(0.99999, 1.0))
Colorbar(fig[1,4], hm, label=L"\cos(\theta)", height=Relative(0.8))
ax3.yticks = (1:6, param_groups)

# Panel D: Rel-error heatmap
ax4 = Axis(fig[2,3], title="Relative Error Heatmap (100 inputs × 6 groups)",
           xlabel="Input sample", ylabel="Parameter group")
rel_data = zeros(100, 6)
for (j, g) in enumerate(param_groups)
    for k in 1:100
        row = filter(r -> r.group == g && r.input_idx == k, df)
        rel_data[k, j] = row.rel_err_lk_3f[1]
    end
end
hm2 = heatmap!(ax4, 1:100, 1:6, rel_data, colormap=:viridis, colorrange=(1e-4, 1e-2))
Colorbar(fig[2,4], hm2, label=L"\|g_{3f} - g_{le}\| / \|g_{le}\|", height=Relative(0.8))
ax4.yticks = (1:6, param_groups)

# Panel E: Summary stats table
ax5 = Axis(fig[2,2], title="Summary Statistics", aspect=DataAspect())
hidedecorations!(ax5); hidespines!(ax5)
stats_text = ""
for g in param_groups
    sub = filter(r -> r.group == g, df)
    global stats_text *= "$(g): cos=$(mean(sub.cos_lk_3f)) ± $(std(sub.cos_lk_3f)), rel_err=$(mean(sub.rel_err_lk_3f)) ± $(std(sub.rel_err_lk_3f))\n"
end
global stats_text *= "\nMin cosine: $(minimum(df.cos_lk_3f))\nMax rel-err: $(maximum(df.rel_err_lk_3f))"
text!(ax5, 0.05, 0.95, text=stats_text, align=(:left, :top), fontsize=10)

save("figures/figure_3factor_equivalence.pdf", fig)
save("figures/figure_3factor_equivalence.png", fig)
println("Saved figures/figure_3factor_equivalence.pdf/.png")