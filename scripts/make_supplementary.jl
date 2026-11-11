#!/usr/bin/env julia
# Figure 5: Supplementary — Weight Symmetry, Readout Frame, Detuning

using Pkg
Pkg.activate(@__DIR__)

using CairoMakie, DataFrames, CSV, Statistics, LaTeXStrings, LinearAlgebra
include("figure_style.jl")

# ============================================================
# Load data
# ============================================================
# Weight symmetry (R4 - feedback asymmetry)
fb_df = CSV.read("results/ep_feedback_asymmetry/feedback_asymmetry_unknown.csv", DataFrame)

# A1: Readout frame
readout_df = CSV.read("results/ep_readout_grid_v2/grid.csv", DataFrame)

# A8: Detuning (run script to get data)
using PhasorNetworks, Lux, Random, Optimisers
using PhasorNetworks: LockinEP, StaticEP, SimilarityCost, ep_gradient
using PhasorNetworks: phasor_settle, ep_loss, _replace_param, _zero_other_params

# Inline FD gradient computation (avoids method invalidation issues)
function fd_gradient_inline(chain, ps, st, x, cost; ε=1e-5, T=200, dt=0.5f0, K_mode=:zero, omega_override=nothing, project=:hard)
    ε_f  = Float32(ε)

    function loss_at(ps_perturbed)
        s = phasor_settle(chain, ps_perturbed, st, x, cost, 0f0;
                          T=T, dt=dt, K_mode=K_mode,
                          omega_override=omega_override, project=project)
        return ep_loss(cost, s[end])
    end

    base = loss_at(ps)

    pairs = Pair{Symbol,Any}[]
    for key in keys(ps)
        layer_ps = ps[key]
        if haskey(layer_ps, :ff) && haskey(layer_ps, :alpha)
            # ResidualBlock not used here
        else
            filled = NamedTuple()
            for pname in (:weight, :bias_real, :bias_imag)
                haskey(layer_ps, pname) || continue
                P = layer_ps[pname]
                gP = zeros(Float32, size(P))
                for i in eachindex(P)
                    Pp = copy(P)
                    Pp[i] += ε_f
                    ps_perturbed = _replace_param(ps, key, pname, Pp)
                    gP[i] = (loss_at(ps_perturbed) - base) / ε_f
                end
                filled = merge(filled, NamedTuple{(pname,)}((gP,)))
            end
            filled = _zero_other_params(layer_ps, filled)
            push!(pairs, key => filled)
        end
    end
    return NamedTuple(pairs)
end

# Run detuning experiment
function run_detuning()
    rng = Xoshiro(42)
    chain = Chain(
        PhasorDense(4 => 8, PhasorNetworks.normalize_to_unit_circle, use_bias=true),
        PhasorDense(8 => 2, PhasorNetworks.normalize_to_unit_circle, use_bias=true)
    )
    ps, st = Lux.setup(rng, chain)
    ps = (layer_1 = merge(ps.layer_1, (weight = 0.4f0 .* ps.layer_1.weight,)),
          layer_2 = merge(ps.layer_2, (weight = 0.4f0 .* ps.layer_2.weight,)))
    
    x = Phase.(2f0 .* rand(rng, Float32, 4) .- 1f0)
    y = ComplexF32.(exp.(im .* π .* (2f0 .* rand(rng, Float32, 2) .- 1f0)))
    cost = SimilarityCost(y)
    
    # FD ground truth - use inline version
    g_fd = fd_gradient_inline(chain, ps, st, x, cost; ε=1e-5, T=200, dt=0.5f0)
    
    Δω_vals = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
    results = DataFrame(
        Δω = Float32[],
        layer = String[],
        param = String[],
        cos = Float32[],
        rel_err = Float32[]
    )
    
    for Δω in Δω_vals
        carriers = [0.0f0, Float32(Δω)]
        m_lockin = LockinEP(ε=0.05f0, ω_p=0.05f0, n_cycles=4, T_warmup_cycles=2,
                            T_free=100, dt=0.1f0, K_mode=:zero, carrier=carriers)
        g_lockin, _ = ep_gradient(m_lockin, chain, ps, st, x, cost)
        
        for key in (:layer_1, :layer_2)
            for pname in (:weight, :bias_real, :bias_imag)
                haskey(g_fd[key], pname) || continue
                haskey(g_lockin[key], pname) || continue
                fd = g_fd[key][pname]
                lk = g_lockin[key][pname]
                size(fd) == size(lk) || continue
                cos_val = real(dot(vec(lk), vec(fd)) / (norm(vec(lk)) * norm(vec(fd)) + 1e-30))
                re_val = norm(lk - fd) / (norm(fd) + 1e-30)
                push!(results, (Δω = Float32(Δω), layer = string(key), param = string(pname), cos = cos_val, rel_err = re_val))
            end
        end
    end
    return results
end

println("Running detuning experiment...")
detune_df = run_detuning()
CSV.write("figures/detuning_results.csv", detune_df)

# ============================================================
# Figure 5A: Weight Symmetry Tolerance (R4)
# ============================================================
fig5a = Figure(size = (DOUBLE_COL * 100, SINGLE_COL * 100))
ax5a = Axis(fig5a[1, 1];
    ylabel = "Cosine similarity (vs symmetric)", title = panel_title("a", "Feedback Weight Asymmetry Tolerance"),
    xscale = log10,
    limits = (0.005, 5.0, -1.1, 1.1)
)

asym_types = unique(fb_df.asymmetry_type)
colors = [OKABE_ITO[2], OKABE_ITO[3], OKABE_ITO[4], OKABE_ITO[6], OKABE_ITO[7]]

for (i, asym) in enumerate(asym_types)
    sub = filter(r -> r.asymmetry_type == asym, fb_df)
    layer1 = filter(r -> r.layer == "layer_1", sub)
    layer2 = filter(r -> r.layer == "layer_2", sub)
    
    sort!(layer1, :param)
    sort!(layer2, :param)
    
    lines!(ax5a, layer1.param, layer1.cos; color = colors[i], label = "$asym (L1)", linestyle = :solid, linewidth = 1.5)
    scatter!(ax5a, layer1.param, layer1.cos; color = colors[i], markersize = 6)
    lines!(ax5a, layer2.param, layer2.cos; color = colors[i], label = "$asym (L2)", linestyle = :dash, linewidth = 1.5)
    scatter!(ax5a, layer2.param, layer2.cos; color = colors[i], marker = :diamond, markersize = 6)
end

hline!(ax5a, 0.99; color = :gray, linestyle = :dot, linewidth = 0.5, label = "cos=0.99")
safe_axislegend!(ax5a; position = :lb, nbanks = 2)

# ============================================================
# Figure 5B: Readout Frame Analysis (A1)
# ============================================================
fig5b = Figure(size = (DOUBLE_COL * 100, SINGLE_COL * 100))
ax5b = Axis(fig5b[1, 1];
    xlabel = "Sampling (1=every, 628=per-period)", ylabel = "Cosine (layer 1)",
    title = panel_title("b", "Readout Frame × Sampling Rate (δ=0.005 turns)"),
    xscale = log10, xticks = ([1, 628], ["1 (every)", "628 (per-period)"])
)

carrier_labels = Dict(
    "co_rotating" => "Co-rotating (no carrier)",
    "lab_2pi" => "Lab ω=2π (commensurate)",
    "lab_1p7" => "Lab ω=1.7 (incommensurate)"
)
carrier_colors = Dict(
    "co_rotating" => OKABE_ITO[1],
    "lab_2pi" => OKABE_ITO[2],
    "lab_1p7" => OKABE_ITO[3]
)
marker_styles = Dict(
    "co_rotating" => :circle,
    "lab_2pi" => :square,
    "lab_1p7" => :diamond
)

for carrier in ["co_rotating", "lab_2pi", "lab_1p7"]
    for frame in ["co_rotating", "lab"]
        sub = filter(r -> r.carrier == carrier && r.frame == frame, readout_df)
        isempty(sub) && continue
        sort!(sub, :sample_every)
        lbl = "$(carrier_labels[carrier]), $(frame == "co_rotating" ? "co-rot" : "lab") frame"
        lines!(ax5b, sub.sample_every, sub.cos_l1; color = carrier_colors[carrier], 
               linestyle = frame == "co_rotating" ? :solid : :dash, linewidth = 1.5, 
               label = sub.sample_every[1] == 1 ? lbl : "")
        scatter!(ax5b, sub.sample_every, sub.cos_l1; color = carrier_colors[carrier],
                 marker = marker_styles[carrier], markersize = 8)
    end
end

safe_axislegend!(ax5b; position = :rb, nbanks = 2)
ylims!(ax5b, -0.1, 1.1)

# ============================================================
# Figure 5C: Detuning / Adler Threshold (A8)
# ============================================================
K_est = 0.15

fig5c = Figure(size = (SINGLE_COL * 100, SINGLE_COL * 100))
ax5c = Axis(fig5c[1, 1];
    xlabel = "Δω / K", ylabel = "Cosine similarity (vs Δω=0)",
    title = panel_title("c", "Adler Threshold: Phase-Locking Breaks at |Δω| ≈ K"),
    xscale = log10,
    limits = (0.05, 500.0, -0.1, 1.1)
)

layer2_weight = filter(r -> r.layer == "layer_2" && r.param == "weight", detune_df)
sort!(layer2_weight, :Δω)

cos0 = layer2_weight[1, :cos]
normalized_cos = layer2_weight.cos ./ cos0

lines!(ax5c, layer2_weight.Δω ./ K_est, normalized_cos; color = OKABE_ITO[2], linewidth = 2)
scatter!(ax5c, layer2_weight.Δω ./ K_est, normalized_cos; color = OKABE_ITO[2], markersize = 8)

vline!(ax5c, 1.0; color = :red, linestyle = :dash, linewidth = 1.5, label = "Adler threshold |Δω| = K")
text!(ax5c, 1.0, 0.5, text = "|Δω| = K", fontsize = 7, color = :red, rotation = π/2, align = (:center, :center))

text!(ax5c, 0.3, 0.9, text = "Locked", fontsize = 8, color = :green, align = (:center, :center), font = :bold)
text!(ax5c, 3.0, 0.3, text = "Unlocked\n(drift)", fontsize = 8, color = :red, align = (:center, :center), font = :bold)

safe_axislegend!(ax5c; position = :rb)

# ============================================================
# Combined Supplementary Figure
# ============================================================
fig5 = Figure(size = (DOUBLE_COL * 100, DOUBLE_COL * 100 * 0.9))

# 5A full width top
ax5a_comb = Axis(fig5[1, 1:2];
    ylabel = "Cosine similarity", title = panel_title("a", "Feedback Weight Asymmetry Tolerance (R4)"),
    xscale = log10,
    limits = (0.005, 5.0, -1.1, 1.1)
)

for (i, asym) in enumerate(asym_types)
    sub = filter(r -> r.asymmetry_type == asym, fb_df)
    layer1 = filter(r -> r.layer == "layer_1", sub)
    layer2 = filter(r -> r.layer == "layer_2", sub)
    sort!(layer1, :param); sort!(layer2, :param)
    lines!(ax5a_comb, layer1.param, layer1.cos; color = colors[i], label = "$asym (L1)", linestyle = :solid)
    scatter!(ax5a_comb, layer1.param, layer1.cos; color = colors[i])
    lines!(ax5a_comb, layer2.param, layer2.cos; color = colors[i], label = "$asym (L2)", linestyle = :dash)
    scatter!(ax5a_comb, layer2.param, layer2.cos; color = colors[i], marker = :diamond)
end
hline!(ax5a_comb, 0.99; color = :gray, linestyle = :dot)
safe_axislegend!(ax5a_comb; position = :lb, nbanks = 2)

# 5B bottom left
ax5b_comb = Axis(fig5[2, 1];
    xlabel = "Sample every (1=every, 628=per-period)", ylabel = "Cosine (layer 1)",
    title = panel_title("b", "Readout Frame × Sampling (A1)"),
    xscale = log10, xticks = ([1, 628], ["1", "628"])
)
for carrier in ["co_rotating", "lab_2pi", "lab_1p7"]
    for frame in ["co_rotating", "lab"]
        sub = filter(r -> r.carrier == carrier && r.frame == frame, readout_df)
        isempty(sub) && continue
        sort!(sub, :sample_every)
        lbl = "$(carrier_labels[carrier]), $(frame == "co_rotating" ? "co-rot" : "lab")"
        lines!(ax5b_comb, sub.sample_every, sub.cos_l1; color = carrier_colors[carrier], 
               linestyle = frame == "co_rotating" ? :solid : :dash, 
               label = sub.sample_every[1] == 1 ? lbl : "")
        scatter!(ax5b_comb, sub.sample_every, sub.cos_l1; color = carrier_colors[carrier],
                 marker = marker_styles[carrier], markersize = 8)
    end
end
safe_axislegend!(ax5b_comb; position = :rb, nbanks = 2)
ylims!(ax5b_comb, -0.1, 1.1)

# 5C bottom right
ax5c_comb = Axis(fig5[2, 2];
    xlabel = "Δω / K", ylabel = "Cosine (normalized)",
    title = panel_title("c", "Adler Threshold (A8)"),
    xscale = log10,
    limits = (0.05, 500.0, -0.1, 1.1)
)
lines!(ax5c_comb, layer2_weight.Δω ./ K_est, normalized_cos; color = OKABE_ITO[2], linewidth = 2)
scatter!(ax5c_comb, layer2_weight.Δω ./ K_est, normalized_cos; color = OKABE_ITO[2], markersize = 8)
vline!(ax5c_comb, 1.0; color = :red, linestyle = :dash, linewidth = 1.5)
text!(ax5c_comb, 1.0, 0.5, text = "|Δω| = K", fontsize = 7, color = :red, rotation = π/2)
text!(ax5c_comb, 0.3, 0.9, text = "Locked", fontsize = 8, color = :green, font = :bold)
text!(ax5c_comb, 3.0, 0.3, text = "Unlocked", fontsize = 8, color = :red, font = :bold)

rowgap!(fig5.layout, 15)
colgap!(fig5.layout, 15)

save_fig(fig5a, "fig5a_weight_symmetry")
save_fig(fig5b, "fig5b_readout_frame")
save_fig(fig5c, "fig5c_adler_threshold")
save_fig(fig5, "figure5_supplementary")

println("Supplementary figures saved to figures/")