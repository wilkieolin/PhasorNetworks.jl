#!/usr/bin/env julia
#
# scripts/ep_detuning.jl — A8: Detuning/Adler threshold
#
# Theory: Phase locking between layers with frequency detuning Δω
# The Adler equation: dφ/dt = Δω - K·sin(φ)
# Phase locking requires |Δω| < K (Adler threshold)
# For our phasor networks, K is the coupling strength (weight magnitude)

using Pkg
function find_repo_root(start_dir::String = pwd())
    dir = start_dir
    while !(isfile(joinpath(dir, "Project.toml")) && isdir(joinpath(dir, ".git")))
        parent = dirname(dir)
        if parent == dir
            error("Repository root not found from $(start_dir)")
        end
        dir = parent
    end
    return dir
end

repo_root = find_repo_root(@__DIR__)
cd(repo_root)
Pkg.activate(repo_root)

using PhasorNetworks, Lux, LinearAlgebra, Statistics, Random, Optimisers
using Random: Xoshiro
using PhasorNetworks: LockinEP, phasor_settle, ep_gradient, SimilarityCost,
                      fd_gradient_phasor, normalize_to_unit_circle, Phase, ComplexF32

# ============================================================
# Adler threshold theory
# ============================================================

"""
    adler_threshold(K::Real) -> Float32

Adler threshold for phase locking: |Δω| < K
where K is the coupling strength (≈ mean weight magnitude for dense layer).
"""
adler_threshold(K::Real) = Float32(K)

"""
    detuning_locked(Δω::Real, K::Real) -> Bool

Check if detuning is within Adler locking range.
"""
detuning_locked(Δω::Real, K::Real) = abs(Δω) < adler_threshold(K)

# ============================================================
# Modified phasor_settle with per-layer carrier (detuning support)
# ============================================================

"""
    phasor_settle_detuned(chain, ps, st, x, cost, β; 
                          carriers::Union{Nothing, Vector{<:Real}} = nothing,
                          kwargs...)

Run phasor_settle with per-layer carrier frequencies for detuning experiments.
If `carriers` is provided, layer `l` uses carrier `carriers[l]` (in rad/s).
The default shared carrier is `carriers[1]` (or 2π).
"""
function phasor_settle_detuned(chain::Lux.Chain, ps, st, x, cost::PhasorNetworks.AbstractEPCost, β::Real;
                               T::Int = 100, dt::Real = 0.5f0,
                               init::Union{Nothing, Vector} = nothing,
                               K_mode::Symbol = :zero,
                               omega_override::Union{Nothing, Vector} = nothing,
                               carriers::Union{Nothing, Vector{<:Real}} = nothing,
                               t0::Real = 0,
                               project::Symbol = :hard,
                               soft_ε::Real = 0.1f0)
    layer_keys = collect(keys(ps))
    n_layers = length(layer_keys)
    
    # Default: all layers share the same carrier (first layer's or 2π)
    if carriers === nothing
        carriers = fill(2f0 * π, n_layers)
    else
        @assert length(carriers) == n_layers "carriers must match number of layers ($n_layers)"
    end
    
    # Convert to Float32
    carriers_f = Float32.(carriers)
    dt_f = Float32(dt)
    β_f = Float32(β)
    t_f = Float32(t0)
    
    z0 = PhasorNetworks._phase_input_to_complex(x)
    
    states = init === nothing ?
        PhasorNetworks._init_states(chain, layer_keys, z0) :
        [ComplexF32.(s) for s in init]
    
    cache = PhasorNetworks._weight_cache(chain, ps, layer_keys)
    drive0 = PhasorNetworks._input_drive(chain, ps, st, layer_keys, z0; cache=cache)
    
    for step in 1:T
        states = PhasorNetworks._phasor_step(chain, ps, st, layer_keys, z0, cost,
                                              β_f, dt_f, states; 
                                              K_mode=K_mode,
                                              omega_override=omega_override,
                                              drive0=drive0,
                                              cache=cache,
                                              carrier=nothing,  # we handle carriers manually
                                              t_now=t_f,
                                              project=project,
                                              soft_ε=Float32(soft_ε))
        t_f += dt_f
    end
    return states
end

# ============================================================
# Experiment: Measure EP gradient fidelity vs detuning
# ============================================================

function run_detuning_experiment()
    println("=== A8: Detuning/Adler Threshold ===\n")
    
    rng = Xoshiro(1234)
    
    # Two-layer network: 16 -> 8 -> 4
    in_dim = 16
    hidden = 8
    out_dim = 4
    
    chain = Chain(
        PhasorDense(in_dim => hidden, normalize_to_unit_circle, use_bias=true),
        PhasorDense(hidden => out_dim, normalize_to_unit_circle, use_bias=true)
    )
    
    ps, st = Lux.setup(rng, chain)
    
    # Scale weights to reasonable range
    scale = 0.5f0
    ps = (layer_1 = merge(ps.layer_1, (weight = scale .* ps.layer_1.weight,)),
          layer_2 = merge(ps.layer_2, (weight = scale .* ps.layer_2.weight,)))
    
    # Test sample
    x = Phase.(2f0 .* rand(rng, Float32, in_dim) .- 1f0)
    y = ComplexF32.(exp.(im .* π .* (2f0 .* rand(rng, Float32, out_dim) .- 1f0)))
    cost = SimilarityCost(y)
    
    # Measure coupling strength K (mean weight magnitude for layer 2)
    K = mean(abs.(ps.layer_2.weight))
    println("Layer 2 weight stats: mean=$(mean(abs.(ps.layer_2.weight))), std=$(std(abs.(ps.layer_2.weight)))")
    println("Adler threshold K = $K")
    
    # Baseline: no detuning (shared carrier)
    println("\n--- Baseline (no detuning) ---")
    m_lockin = LockinEP(ε=0.05f0, ω_p=0.05f0, n_cycles=4, T_warmup_cycles=2, 
                        T_free=100, dt=0.1f0, K_mode=:zero)
    g_lockin, _ = ep_gradient(m_lockin, chain, ps, st, x, cost)
    
    # FD gradient
    g_fd, _ = fd_gradient_phasor(chain, ps, st, x, cost; ε=1e-5, T=200, dt=0.5f0)
    
    cosine(a, b) = real(dot(vec(a), vec(b)) / (norm(vec(a)) * norm(vec(b)) + 1e-30))
    rel_err(a, b) = norm(a - b) / (norm(b) + 1e-30)
    
    # Layer 1 comparison
    for pname in keys(ps.layer_1)
        haskey(g_fd, pname) && haskey(g_lockin.layer_1, pname) || continue
        fd = g_fd[pname]
        lk = g_lockin.layer_1[pname]
        size(fd) == size(lk) || continue
        println("  layer_1 $pname: cos=$(cosine(lk, fd)), rel-err=$(rel_err(lk, fd))")
    end
    
    # Layer 2 comparison
    for pname in keys(ps.layer_2)
        haskey(g_fd, pname) && haskey(g_lockin.layer_2, pname) || continue
        fd = g_fd[pname]
        lk = g_lockin.layer_2[pname]
        size(fd) == size(lk) || continue
        println("  layer_2 $pname: cos=$(cosine(lk, fd)), rel-err=$(rel_err(lk, fd))")
    end
    
    # Sweep detuning
    println("\n--- Detuning sweep ---")
    detunings = Float32[0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
    
    for Δω in detunings
        # Carrier for layer 1: 2π, Carrier for layer 2: 2π + Δω
        carriers = [2f0*π, 2f0*π + Δω]
        
        # Create LockinEP with per-layer carriers
        m_detuned = LockinEP(ε=0.05f0, ω_p=0.05f0, n_cycles=4, T_warmup_cycles=2, 
                            T_free=100, dt=0.1f0, K_mode=:zero, carrier=carriers)
        
        # Compute EP gradient with detuning
        g_detuned, _ = ep_gradient(m_detuned, chain, ps, st, x, cost)
        
        # Compare to baseline (no detuning)
        for pname in keys(ps.layer_2)
            haskey(g_lockin.layer_2, pname) && haskey(g_detuned.layer_2, pname) || continue
            base = g_lockin.layer_2[pname]
            det = g_detuned.layer_2[pname]
            size(base) == size(det) || continue
            c = cosine(det, base)
            re = rel_err(det, base)
            locked = detuning_locked(Δω, K) ? "LOCKED" : "UNLOCKED"
            println("  Δω=$Δω (K=$K, $locked): layer_2 $pname: cos=$c, rel-err=$re")
        end
    end
    
    println("\n=== A8 Detuning Analysis Complete ===")
end

run_detuning_experiment()