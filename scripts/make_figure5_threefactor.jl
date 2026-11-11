#!/usr/bin/env julia
# Figure 5, data panel -- the three-factor rule reproduces lock-in EP.
#
# Data: results/e4_three_factor_scale/e4_three_factor_ddbaa3d.csv
#   100 random inputs x {layer_1, layer_2} x {weight, bias_real, bias_imag},
#   at the FashionMNIST architecture (784 -> 256 -> 64).
#
# SCOPE. Every cos_fd_* / rel_err_fd_* column in that run is NaN: the
# finite-difference oracle was not computed at this scale. So this panel makes
# an IDENTITY claim -- the local three-factor accumulation and the lock-in
# estimator are the same computation -- and says nothing about fidelity to the
# true loss gradient. Fidelity is Figure 1's job, where a directional-FD oracle
# exists.
#
# The reference line is what makes the size of the identity error meaningful.
# It is NOT machine epsilon: 1e-4..3e-3 is Float32 accumulation over T_lockin
# steps, not roundoff, and a machine-epsilon reference would frame an honest
# result as a near-miss. The meaningful comparison is the error that lock-in EP
# itself carries against the finite-difference gradient, from E1
# (results/ep_trained_vs_rescaled/trained_vs_rescaled.csv, trained arm,
# FD-trusted, estimator == "lockin"): median 0.166. The three-factor rule
# reproduces the estimator two orders of magnitude more tightly than the
# estimator reproduces the gradient.

using Pkg; Pkg.activate(@__DIR__)
using CairoMakie, DataFrames, CSV, Statistics, Random, Printf
include("figure_style.jl")

const RES = joinpath(@__DIR__, "..", "results")
df = CSV.read(joinpath(RES, "e4_three_factor_scale", "e4_three_factor_ddbaa3d.csv"), DataFrame)

# LIEP's own error against the FD oracle, for the reference line
e1 = CSV.read(joinpath(RES, "ep_trained_vs_rescaled", "trained_vs_rescaled.csv"), DataFrame)
e1 = e1[(e1.tag .== "trained") .& (e1.estimator .== "lockin") .& (e1.fd_trusted .== true), :]
liep_err = median(skipmissing(e1.relerr_fd))

const GROUPS = [
    ("layer_1", "weight",    "W₁"),
    ("layer_1", "bias_real", "Re b₁"),
    ("layer_1", "bias_imag", "Im b₁"),
    ("layer_2", "weight",    "W₂"),
    ("layer_2", "bias_real", "Re b₂"),
    ("layer_2", "bias_imag", "Im b₂"),
]

fig = Figure(size = (FULL_W, 132))
ax = Axis(fig[1, 1];
    title = panel_title("e", "Three-factor rule vs lock-in EP, 100 inputs at 784→256→64"),
    xlabel = "Parameter tensor", ylabel = "Relative error",
    yscale = log10, titlealign = :left)

rng = MersenneTwister(20260908)
positions = Float64[]; labels = String[]
for (i, (lay, par, lab)) in enumerate(GROUPS)
    v = df[(df.layer .== lay) .& (df.param .== par), :rel_err_lk_3f]
    colour = lay == "layer_1" ? C_HEBB : C_HOLO
    scatter!(ax, i .+ 0.26 .* (rand(rng, length(v)) .- 0.5), v;
             color = (colour, 0.45), markersize = MARKER_SIZE - 3, strokewidth = 0)
    lines!(ax, [i - 0.22, i + 0.22], fill(median(v), 2);
           color = colour, linewidth = 2.0)
    push!(positions, i); push!(labels, lab)
end
ax.xticks = (positions, labels)
xlims!(ax, 0.4, 6.6)

hlines!(ax, [liep_err]; color = C_ANTIHOLO, linestyle = :dash, linewidth = 1.2)
text!(ax, 6.5, liep_err; text = @sprintf("lock-in EP's own error vs finite-difference gradient (%.2f)", liep_err),
      fontsize = FONT_SIZE_SMALL - 1, color = C_ANTIHOLO, align = (:right, :bottom), offset = (0, 2))

allv = df.rel_err_lk_3f
ylims!(ax, 10.0^floor(log10(minimum(allv))), 10.0^(ceil(log10(liep_err)) + 0.6))
text!(ax, 0.02, 0.06; space = :relative, align = (:left, :bottom),
      fontsize = FONT_SIZE_SMALL - 1, color = C_AXIS,
      text = @sprintf("all 600 runs: cos ≥ %.6f", minimum(df.cos_lk_3f)))

save_fig(fig, "figure5_threefactor")
@printf("figure5_threefactor written; relerr %.2e..%.2e, LIEP-vs-FD ref %.3f\n",
        minimum(allv), maximum(allv), liep_err)
