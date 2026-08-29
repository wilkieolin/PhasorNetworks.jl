#!/usr/bin/env julia
#
# scripts/ep_binding_layer.jl — A7: Binding as EP layer
#
# Tests the PhasorBind layer (fixed-key binding) with EP gradients.

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

using PhasorNetworks, Lux, LinearAlgebra, Statistics, Random, Optimisers, Zygote, ChainRulesCore
using Random: Xoshiro

function run_binding_test()
    println("=== A7: Binding as EP Layer ===\n")
    
    rng = Xoshiro(42)
    
    # Create bind layer with random key (using package's PhasorBind)
    in_dims = 8
    key = 2f0 .* rand(rng, Float32, in_dims) .- 1f0
    bind_layer = PhasorNetworks.PhasorBind(in_dims, key, use_bias=true)
    
    # Create a small chain: PhasorDense → PhasorBind → PhasorDense
    chain = Chain(
        PhasorDense(4 => in_dims, normalize_to_unit_circle, use_bias=true),
        bind_layer,
        PhasorDense(in_dims => 2, normalize_to_unit_circle, use_bias=true)
    )
    
    ps, st = Lux.setup(rng, chain)
    
    # Scale weights
    ps = (layer_1 = merge(ps.layer_1, (weight = 0.4f0 .* ps.layer_1.weight,)),
          layer_2 = merge(ps.layer_2, (key = ps.layer_2.key,)),  # bind key stays as-is
          layer_3 = merge(ps.layer_3, (weight = 0.4f0 .* ps.layer_3.weight,)))
    
    # Test sample
    x = Phase.(2f0 .* rand(rng, Float32, 4) .- 1f0)
    y = ComplexF32.(exp.(im .* π .* (2f0 .* rand(rng, Float32, 2) .- 1f0)))
    cost = SimilarityCost(y)
    
    # 1. FD gradient
    println("1. Finite-difference gradient...")
    g_fd, _ = fd_gradient_phasor(chain, ps, st, x, cost; ε=1e-5, T=200, dt=0.5f0)
    
    # 2. LockinEP gradient
    println("2. LockinEP gradient...")
    m_lockin = LockinEP(ε=0.05f0, ω_p=0.05f0, n_cycles=4, T_warmup_cycles=2, 
                        T_free=100, dt=0.1f0, K_mode=:zero)
    g_lockin, _ = ep_gradient(m_lockin, chain, ps, st, x, cost)
    
    # 3. StaticEP gradient (centered)
    println("3. StaticEP gradient (centered)...")
    m_static = StaticEP(β=0.005f0, T_free=200, T_nudge=100, dt=0.5f0, centered=true)
    g_static, _ = ep_gradient(m_static, chain, ps, st, x, cost)
    
    # Compare - Lockin vs Static which both have per-layer structure
    cosine(a, b) = real(dot(vec(a), vec(b)) / (norm(vec(a)) * norm(vec(b)) + 1e-30))
    rel_err(a, b) = norm(a - b) / (norm(b) + 1e-30)
    
    println("\n=== Gradient Comparison (Lockin vs Static) ===")
    for key_name in keys(ps)
        println("\n$key_name:")
        for pname in keys(ps[key_name])
            haskey(g_lockin[key_name], pname) || continue
            haskey(g_static[key_name], pname) || continue
            
            lk = g_lockin[key_name][pname]
            st_g = g_static[key_name][pname]
            
            size(lk) == size(st_g) || continue
            
            cos_lk_st = cosine(lk, st_g)
            re_lk_st = rel_err(lk, st_g)
            
            println("  $pname:")
            println("    Lockin vs Static: cos=$cos_lk_st, rel-err=$re_lk_st")
        end
    end
    
    # Also compare FD for layer_1 (first layer only, since FD is flat)
    println("\n=== FD vs Lockin (layer_1 only) ===")
    if haskey(g_lockin, :layer_1)
        for pname in keys(ps.layer_1)
            haskey(g_fd, pname) || continue
            haskey(g_lockin.layer_1, pname) || continue
            
            fd = g_fd[pname]
            lk = g_lockin.layer_1[pname]
            
            size(fd) == size(lk) || continue
            
            cos_fd_lk = cosine(lk, fd)
            re_fd_lk = rel_err(lk, fd)
            
            println("  layer_1 $pname:")
            println("    FD vs Lockin:  cos=$cos_fd_lk, rel-err=$re_fd_lk")
        end
    end
    
    # 4. Test settle equivalence
    println("\n=== Settle Test ===")
    s_free = phasor_settle(chain, ps, st, x, cost, 0f0; T=100, dt=0.5f0)
    s_nudge = phasor_settle(chain, ps, st, x, cost, 0.1f0; T=50, dt=0.5f0, init=s_free)
    
    println("Free settle output phase: ", angle.(s_free[end]))
    println("Nudged settle output phase: ", angle.(s_nudge[end]))
    
    # 5. Test training step
    println("\n=== Training Step Test ===")
    opt_state = Optimisers.setup(Optimisers.Adam(0.01), ps)
    g, _ = ep_gradient(m_lockin, chain, ps, st, x, cost)
    opt_state, ps_new = Optimisers.update(opt_state, ps, g)
    println("Training step completed successfully")
    
    # 6. Test forward pass
    println("\n=== Forward Pass Test ===")
    y_phase, _ = chain(x, ps, st)
    println("Output phase shape: ", size(y_phase))
    println("Output phase: ", y_phase)
    
    # 7. Test with 3D phase input (time dimension)
    println("\n=== 3D Phase Input Test ===")
    L = 5  # time steps
    x3d = Phase.(2f0 .* rand(rng, Float32, 4, L) .- 1f0)
    y3d, _ = chain(x3d, ps, st)
    println("3D output shape: ", size(y3d))
    println("3D output: ", y3d)
    
    println("\n=== A7 Binding Layer Validated ===")
end

run_binding_test()