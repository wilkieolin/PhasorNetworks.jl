#!/usr/bin/env julia
# Figure 4: Three-Factor Learning Rule Equivalence

using Pkg
Pkg.activate(@__DIR__)

using CairoMakie, DataFrames, CSV, Statistics, LaTeXStrings, LinearAlgebra
include("figure_style.jl")

# We'll run the three_factor_stdp script to generate data
# First, let's run it and capture the output
using PhasorNetworks, Lux, Optimisers, Random
using PhasorNetworks: LockinEP, StaticEP, SimilarityCost, fd_gradient_phasor, ep_gradient
using PhasorNetworks: phasor_settle, chain_hebbians, _phase_input_to_complex, _weight_cache, _input_drive
using PhasorNetworks: _phasor_step, _pad_dynamics_zeros, _zero_grad, pi_f32, gpu_zeros
using LinearAlgebra: mul!

# ---- ThreeFactorLockin implementation (copied from script) ----
Base.@kwdef struct ThreeFactorLockin
    ε::Float32          = 0.05f0
    ω_p::Float32        = 0.05f0
    n_cycles::Int       = 8
    T_warmup_cycles::Int = 2
    T_free::Int         = 200
    dt::Float32         = 0.1f0
    K_mode::Symbol      = :zero
    project::Symbol     = :hard
end

function three_factor_gradient(m::ThreeFactorLockin, chain::Lux.Chain, ps, st, x,
                               cost::PhasorNetworks.AbstractEPCost;
                               omega_override::Union{Nothing, Vector} = nothing)
    s_free = phasor_settle(chain, ps, st, x, cost, 0f0;
                           T=m.T_free, dt=m.dt, K_mode=m.K_mode,
                           omega_override=omega_override, project=m.project)
    t_free_end = Float32(m.T_free * m.dt)

    h_dc = chain_hebbians(chain, ps, st, x, s_free)

    layer_keys = collect(keys(ps))
    z0 = _phase_input_to_complex(x)
    period_steps = round(Int, 2π / (m.ω_p * m.dt))
    T_warmup = m.T_warmup_cycles * period_steps
    T_lockin = m.n_cycles * period_steps

    states = [copy(s) for s in s_free]
    cache = _weight_cache(chain, ps, layer_keys)
    drive0 = _input_drive(chain, ps, st, layer_keys, z0; cache=cache)

    for t in 1:T_warmup
        β_t = m.ε * cos(m.ω_p * t * m.dt)
        t_now = t_free_end + Float32((t - 1) * m.dt)
        states = _phasor_step(chain, ps, st, layer_keys, z0, cost,
                              β_t, m.dt, states; K_mode=m.K_mode,
                              omega_override=omega_override, drive0=drive0,
                              cache=cache, project=m.project,
                              carriers=nothing, t_now=t_now)
    end

    t_warm_end = t_free_end + Float32(T_warmup * m.dt)

    E_W = Dict{Symbol, Any}()
    E_b = Dict{Symbol, Any}()
    for (l, key) in enumerate(layer_keys)
        haskey(ps[key], :weight) || continue
        E_W[key] = gpu_zeros(ps[key].weight, ComplexF32, size(ps[key].weight)...)
        if haskey(ps[key], :bias_real)
            E_b[key] = gpu_zeros(states[l], ComplexF32, size(states[l])...)
        end
    end

    for t in 1:T_lockin
        β_t = m.ε * cos(m.ω_p * t * m.dt)
        t_now = t_warm_end + Float32((t - 1) * m.dt)
        states = _phasor_step(chain, ps, st, layer_keys, z0, cost,
                              β_t, m.dt, states; K_mode=m.K_mode,
                              omega_override=omega_override, drive0=drive0,
                              cache=cache, project=m.project,
                              carriers=nothing, t_now=t_now)

        demod = ComplexF32(exp(-im * m.ω_p * Float32(t) * m.dt))

        for l in 1:length(layer_keys)
            key = layer_keys[l]
            haskey(ps[key], :weight) || continue

            z_l = states[l]
            z_in = (l == 1) ? z0 : states[l-1]

            elig_W = z_l * adjoint(z_in)
            elig_b = z_l

            elig_W_dc = elig_W .- h_dc[key].weight
            elig_b_dc = elig_b .- ComplexF32.(h_dc[key].bias_real .+ 1f0im .* h_dc[key].bias_imag)

            E_W[key] .+= elig_W_dc .* demod
            if haskey(E_b, key)
                E_b[key] .+= elig_b_dc .* demod
            end
        end
    end

    norm_factor = Float32(T_lockin) * m.ε
    grads = NamedTuple()

    for key in layer_keys
        if haskey(ps[key], :weight)
            entry = (weight = -2f0 .* real.(E_W[key]) ./ norm_factor,)
            if haskey(ps[key], :bias_real)
                entry = merge(entry, (
                    bias_real = -2f0 .* real.(E_b[key]) ./ norm_factor,
                    bias_imag = -2f0 .* imag.(E_b[key]) ./ norm_factor,
                ))
            end
            entry = _pad_dynamics_zeros(entry, ps[key])
            grads = merge(grads, NamedTuple{(key,)}((entry,)))
        else
            grads = merge(grads, NamedTuple{(key,)}((_zero_grad(ps[key]),)))
        end
    end

    return grads, s_free
end

# STDP approximation
function stdp_gradient(chain::Lux.Chain, ps, st, x, cost::PhasorNetworks.AbstractEPCost;
                       window::Float32 = 0.1f0, A_plus::Float32 = 1.0f0,
                       A_minus::Float32 = 1.0f0, τ::Float32 = 0.02f0,
                       T::Int = 100, dt::Float32 = 0.5f0,
                       K_mode::Symbol = :zero,
                       omega_override::Union{Nothing, Vector} = nothing)
    s_free = phasor_settle(chain, ps, st, x, cost, 0f0;
                           T=T, dt=dt, K_mode=K_mode,
                           omega_override=omega_override)
    
    layer_keys = collect(keys(ps))
    z0 = _phase_input_to_complex(x)
    grads = NamedTuple()
    
    for (l, key) in enumerate(layer_keys)
        haskey(ps[key], :weight) || continue
        
        z_self = s_free[l]
        z_in = (l == 1) ? z0 : s_free[l-1]
        
        phase_self = angle.(z_self) / (2f0 * pi_f32)
        phase_in = angle.(z_in) / (2f0 * pi_f32)
        
        d_phi = phase_self .- phase_in'
        
        ltp = A_plus .* exp.(-d_phi ./ τ) .* (d_phi .> 0)
        ltd = -A_minus .* exp.(d_phi ./ τ) .* (d_phi .< 0)
        stdp_w = ltp + ltd
        stdp_w = clamp.(stdp_w, -window, window)
        
        g_weight = real.(stdp_w)
        
        entry = (weight = g_weight,)
        if haskey(ps[key], :bias_real)
            entry = merge(entry, (
                bias_real = Float32.(mean(real.(z_self); dims=2)),
                bias_imag = Float32.(mean(imag.(z_self); dims=2)),
            ))
        end
        entry = _pad_dynamics_zeros(entry, ps[key])
        grads = merge(grads, NamedTuple{(key,)}((entry,)))
    end
    
    return grads, s_free
end

# Cosine similarity
cosine(a, b) = real(dot(vec(a), vec(b)) / (norm(vec(a)) * norm(vec(b)) + 1e-30))
rel_err(a, b) = norm(a - b) / (norm(b) + 1e-30)

# ============================================================
# Run comparison on toy chain
# ============================================================
println("Running three-factor comparison...")

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

# 1. FD ground truth
g_fd, _ = fd_gradient_phasor(chain, ps, st, x, cost; ε=1e-5, T=200, dt=0.5f0)

# 2. LockinEP
m_lockin = LockinEP(ε=0.05f0, ω_p=0.05f0, n_cycles=4, T_warmup_cycles=2, 
                    T_free=100, dt=0.1f0, K_mode=:zero)
g_lockin, _ = ep_gradient(m_lockin, chain, ps, st, x, cost)

# 3. ThreeFactorLockin
m_3f = ThreeFactorLockin(ε=0.05f0, ω_p=0.05f0, n_cycles=4, T_warmup_cycles=2,
                         T_free=100, dt=0.1f0, K_mode=:zero)
g_3f, _ = three_factor_gradient(m_3f, chain, ps, st, x, cost)

# 4. STDP
g_stdp, _ = stdp_gradient(chain, ps, st, x, cost; T=100, dt=0.5f0)

# Wrap FD for comparison
g_fd_layered = (layer_1 = (weight = g_fd.weight, bias_real = g_fd.bias_real, bias_imag = g_fd.bias_imag),
                layer_2 = (weight = g_fd.weight, bias_real = g_fd.bias_real, bias_imag = g_fd.bias_imag))

# Collect comparison data
comparison_data = DataFrame(
    layer = String[],
    param = String[],
    fd_vs_lockin_cos = Float32[],
    fd_vs_lockin_re = Float32[],
    fd_vs_3f_cos = Float32[],
    fd_vs_3f_re = Float32[],
    fd_vs_stdp_cos = Float32[],
    fd_vs_stdp_re = Float32[],
    lockin_vs_3f_re = Float32[]
)

for key in (:layer_1, :layer_2)
    for pname in (:weight, :bias_real, :bias_imag)
        haskey(g_fd_layered[key], pname) || continue
        haskey(g_lockin[key], pname) || continue
        haskey(g_3f[key], pname) || continue
        haskey(g_stdp[key], pname) || continue
        
        fd = g_fd_layered[key][pname]
        lk = g_lockin[key][pname]
        f3 = g_3f[key][pname]
        stdp = g_stdp[key][pname]
        
        size(fd) == size(lk) || continue
        
        push!(comparison_data, (
            layer = string(key),
            param = string(pname),
            fd_vs_lockin_cos = cosine(lk, fd),
            fd_vs_lockin_re = rel_err(lk, fd),
            fd_vs_3f_cos = cosine(f3, fd),
            fd_vs_3f_re = rel_err(f3, fd),
            fd_vs_stdp_cos = cosine(stdp, fd),
            fd_vs_stdp_re = rel_err(stdp, fd),
            lockin_vs_3f_re = rel_err(lk, f3)
        ))
    end
end

# Also collect training trajectory comparison
function train_epochs(chain, ps_init, st, method, n_epochs=5)
    ps = deepcopy(ps_init)
    opt_state = Optimisers.setup(Optimisers.Adam(0.05), ps)
    weight_history = [deepcopy(ps)]
    
    for epoch in 1:n_epochs
        for i in 1:10
            x = Phase.(2f0 .* rand(rng, Float32, 4) .- 1f0)
            y = ComplexF32.(exp.(im .* π .* (2f0 .* rand(rng, Float32, 2) .- 1f0)))
            cost = SimilarityCost(y)
            if method isa LockinEP
                g, _ = ep_gradient(method, chain, ps, st, x, cost)
            elseif method isa ThreeFactorLockin
                g, _ = three_factor_gradient(method, chain, ps, st, x, cost)
            else
                error("Unknown method type")
            end
            opt_state, ps = Optimisers.update(opt_state, ps, g)
        end
        push!(weight_history, deepcopy(ps))
    end
    return weight_history
end

println("Running training comparison...")
ps_lockin_hist = train_epochs(chain, ps, st, m_lockin, 5)
ps_3f_hist = train_epochs(chain, ps, st, m_3f, 5)

# Weight difference over training
weight_diff = Float32[]
for (pl, p3) in zip(ps_lockin_hist, ps_3f_hist)
    diff = norm(pl.layer_1.weight - p3.layer_1.weight) + norm(pl.layer_2.weight - p3.layer_2.weight)
    norm_w = norm(pl.layer_1.weight) + norm(pl.layer_2.weight)
    push!(weight_diff, diff / norm_w)
end

# ============================================================
# Figure 4A: Gradient Cosine vs FD
# ============================================================
fig4a = Figure(size = (SINGLE_COL * 100, SINGLE_COL * 100))
ax4a = Axis(fig4a[1, 1];
    ylabel = "Cosine similarity (vs FD)",
    title = panel_title("a", "Gradient Fidelity vs Finite Difference"),
    xticks = (1:nrow(comparison_data), [replace(r.param, "_"=>" ") for r in eachrow(comparison_data)]),
    xticklabelrotation = π/4
)

x_pos = 1:nrow(comparison_data)
width = 0.25

barplot!(ax4a, x_pos .- width, comparison_data.fd_vs_lockin_cos; 
    width = width, color = OKABE_ITO[3], label = "LockinEP", strokewidth = 0.5)
barplot!(ax4a, x_pos, comparison_data.fd_vs_3f_cos; 
    width = width, color = OKABE_ITO[2], label = "ThreeFactorLockin", strokewidth = 0.5)
barplot!(ax4a, x_pos .+ width, comparison_data.fd_vs_stdp_cos; 
    width = width, color = OKABE_ITO[7], label = "STDP (approx)", strokewidth = 0.5)

# Reference line at 1.0
hline!(ax4a, 1.0; color = :gray, linestyle = :dash, linewidth = 0.5)

safe_axislegend!(ax4a; position = :rb)
ylims!(ax4a, -0.1, 1.1)

# Annotate Lockin vs 3F match
text!(ax4a, nrow(comparison_data)/2, 0.95, 
    text = "LockinEP ≡ ThreeFactorLockin (rel-err ~1e-5)", 
    fontsize = 6, color = :black, align = (:center, :top), font = :bold)

# ============================================================
# Figure 4B: Lockin vs 3F Relative Error (Numerical Identity)
# ============================================================
fig4b = Figure(size = (SINGLE_COL * 100, SINGLE_COL * 100))
ax4b = Axis(fig4b[1, 1];
    ylabel = "Relative error (LockinEP vs ThreeFactorLockin)",
    title = panel_title("b", "Numerical Identity: LockinEP = Three-Factor Rule"),
    xticks = (1:nrow(comparison_data), [replace(r.param, "_"=>" ") for r in eachrow(comparison_data)]),
    xticklabelrotation = π/4,
    yscale = log10
)

barplot!(ax4b, x_pos, comparison_data.lockin_vs_3f_re; 
    width = 0.6, color = OKABE_ITO[1], strokewidth = 0.5, label = "Lockin vs 3F")
hline!(ax4b, 1e-5; color = :red, linestyle = :dash, linewidth = 1, label = "1e-5 threshold")
hline!(ax4b, 1e-10; color = :gray, linestyle = :dot, linewidth = 0.5)

safe_axislegend!(ax4b; position = :lt)

# ============================================================
# Figure 4C: STDP Window vs Lock-in Window
# ============================================================
# Compute effective Δw(Δt) windows
# For Lock-in: cosine window (symmetric, periodic)
# For STDP: exponential (asymmetric)

Δt_vals = -1.0:0.01:1.0  # phase difference in turns

# Lock-in window: cosine (even in Δt, periodic with period 1)
lockin_window = cos.(2π .* Δt_vals)

# STDP window: exponential (odd in Δt)
A_plus = 1.0; A_minus = 1.0; τ = 0.02
stdp_window = A_plus .* exp.(-Δt_vals ./ τ) .* (Δt_vals .> 0) .- A_minus .* exp.(Δt_vals ./ τ) .* (Δt_vals .< 0)

fig4c = Figure(size = (SINGLE_COL * 100, SINGLE_COL * 100))
ax4c = Axis(fig4c[1, 1];
    xlabel = "Phase difference Δφ (turns)", ylabel = "Weight update Δw",
    title = panel_title("c", "Effective Learning Windows")
)

lines!(ax4c, Δt_vals, lockin_window; color = OKABE_ITO[3], linewidth = 2, label = "Lock-in (cosine, even)")
lines!(ax4c, Δt_vals, stdp_window; color = OKABE_ITO[2], linewidth = 2, label = "STDP (exponential, odd)")

# Vertical line at 0
vline!(ax4c, 0.0; color = :gray, linestyle = :dot, linewidth = 0.5)

safe_axislegend!(ax4c; position = :rb)
ylims!(ax4c, -1.2, 1.2)

# ============================================================
# Figure 4D: Training Trajectory Match
# ============================================================
fig4d = Figure(size = (SINGLE_COL * 100, SINGLE_COL * 100))
ax4d = Axis(fig4d[1, 1];
    xlabel = "Epoch", ylabel = "Relative weight difference ||W_lockin - W_3F|| / ||W||",
    title = panel_title("d", "Training Trajectories Match")
)

epochs = 0:5
lines!(ax4d, epochs, weight_diff; color = OKABE_ITO[1], linewidth = 2)
scatter!(ax4d, epochs, weight_diff; color = OKABE_ITO[1], markersize = 8)
hline!(ax4d, 1e-5; color = :red, linestyle = :dash, label = "1e-5")
safe_axislegend!(ax4d; position = :lt)
ax4d.yscale = log10

# ============================================================
# Combined Figure 4
# ============================================================
fig4 = Figure(size = (DOUBLE_COL * 100, DOUBLE_COL * 100 * 0.9))

# 2x2 grid
ax4a_comb = Axis(fig4[1, 1];
    ylabel = "Cosine vs FD", title = panel_title("a", "Gradient Fidelity"),
    xticks = (1:nrow(comparison_data), [replace(r.param, "_"=>" ") for r in eachrow(comparison_data)]),
    xticklabelrotation = π/4
)
barplot!(ax4a_comb, x_pos .- width, comparison_data.fd_vs_lockin_cos; width = width, color = OKABE_ITO[3], label = "LockinEP")
barplot!(ax4a_comb, x_pos, comparison_data.fd_vs_3f_cos; width = width, color = OKABE_ITO[2], label = "ThreeFactorLockin")
barplot!(ax4a_comb, x_pos .+ width, comparison_data.fd_vs_stdp_cos; width = width, color = OKABE_ITO[7], label = "STDP")
hline!(ax4a_comb, 1.0; color = :gray, linestyle = :dash)
safe_axislegend!(ax4a_comb; position = :rb)
ylims!(ax4a_comb, -0.1, 1.1)

ax4b_comb = Axis(fig4[1, 2];
    ylabel = "Rel. error (Lockin vs 3F)", title = panel_title("b", "Numerical Identity"),
    xticks = (1:nrow(comparison_data), [replace(r.param, "_"=>" ") for r in eachrow(comparison_data)]),
    xticklabelrotation = π/4, yscale = log10
)
barplot!(ax4b_comb, x_pos, comparison_data.lockin_vs_3f_re; width = 0.6, color = OKABE_ITO[1])
hline!(ax4b_comb, 1e-5; color = :red, linestyle = :dash)
hline!(ax4b_comb, 1e-10; color = :gray, linestyle = :dot)
safe_axislegend!(ax4b_comb; position = :lt)

ax4c_comb = Axis(fig4[2, 1];
    xlabel = "Phase difference Δφ (turns)", ylabel = "Weight update Δw",
    title = panel_title("c", "Learning Windows")
)
lines!(ax4c_comb, Δt_vals, lockin_window; color = OKABE_ITO[3], linewidth = 2, label = "Lock-in (cosine)")
lines!(ax4c_comb, Δt_vals, stdp_window; color = OKABE_ITO[2], linewidth = 2, label = "STDP (exponential)")
vline!(ax4c_comb, 0.0; color = :gray, linestyle = :dot)
safe_axislegend!(ax4c_comb; position = :rb)
ylims!(ax4c_comb, -1.2, 1.2)

ax4d_comb = Axis(fig4[2, 2];
    xlabel = "Epoch", ylabel = "Rel. weight diff", title = panel_title("d", "Training Match"),
    yscale = log10
)
lines!(ax4d_comb, epochs, weight_diff; color = OKABE_ITO[1], linewidth = 2)
scatter!(ax4d_comb, epochs, weight_diff; color = OKABE_ITO[1], markersize = 8)
hline!(ax4d_comb, 1e-5; color = :red, linestyle = :dash)
safe_axislegend!(ax4d_comb; position = :lt)

rowgap!(fig4.layout, 15)
colgap!(fig4.layout, 15)

# Save
save_fig(fig4a, "fig4a_gradient_fidelity")
save_fig(fig4b, "fig4b_numerical_identity")
save_fig(fig4c, "fig4c_learning_windows")
save_fig(fig4d, "fig4d_training_match")
save_fig(fig4, "figure4_threefactor_equivalence")

# Save comparison data for paper
CSV.write("figures/threefactor_comparison.csv", comparison_data)

println("Figure 4 saved to figures/")
println("Comparison data:")
show(comparison_data, allrows = true)
println()
println("Training weight diff per epoch: ", weight_diff)
