#!/usr/bin/env julia
# StaticEP vs FD validation for ResidualBlock
# Tests gradient fidelity of EP methods on toy ResidualBlock chains

using Pkg
Pkg.activate(".")

using PhasorNetworks
using Lux
using Random
using LinearAlgebra
using Statistics

# Test configuration
const TEST_DEPTHS = [1, 2, 3]
const TEST_WIDTHS = [8, 16]
const N_SAMPLES = 4
const BATCH_SIZE = 2
const SEED = 1234

function build_test_chain(input_dim, hidden_dim, output_dim, n_residual; gate=:rezero, alpha0=0.1f0)
    layers = []
    push!(layers, PhasorDense(input_dim => hidden_dim, normalize_to_unit_circle; use_bias=true))
    
    for i in 1:n_residual
        push!(layers, ResidualBlock(
            (hidden_dim, hidden_dim),
            normalize_to_unit_circle;
            gate=gate,
            alpha0=alpha0,
            branch_init_scale=0.1f0,
            use_bias=true
        ))
    end
    
    push!(layers, PhasorDense(hidden_dim => output_dim, normalize_to_unit_circle; use_bias=true))
    
    return Lux.Chain(layers...)
end

function cosine_similarity(a, b)
    # Flatten named tuples of arrays and compute cosine similarity
    function flatten(x)
        if x isa NamedTuple
            return vcat([flatten(v) for v in values(x)]...)
        elseif x isa AbstractArray
            return vec(x)
        else
            return Float32[]
        end
    end
    flat_a = flatten(a)
    flat_b = flatten(b)
    isempty(flat_a) && return 1f0
    dot(flat_a, flat_b) / (norm(flat_a) * norm(flat_b) + 1f-10)
end

function test_gradient_fidelity(chain, ps, st, x, y, rng; β=0.01f0, T_free=200, T_nudge=100, dt=0.5f0)
    # Create codebook for CodebookCost
    n_classes = 10
    output_dim = 2
    codebook = randn(rng, ComplexF32, output_dim, n_classes)
    codebook = PhasorNetworks.normalize_to_unit_circle(codebook)
    cost = PhasorNetworks.CodebookCost(codebook, y)  # y is vector of class indices
    
    # StaticEP gradient
    g_static, _ = PhasorNetworks.ep_gradient(
        PhasorNetworks.StaticEP(β=β, T_free=T_free, T_nudge=T_nudge, dt=dt, centered=true),
        chain, ps, st, x, cost
    )
    
    # FD gradient
    g_fd = PhasorNetworks.fd_gradient_phasor(
        chain, ps, st, x, cost; ε=1f-4
    )
    
    # Compute cosine similarity per layer
    results = Dict()
    for key in keys(ps)
        println("  Comparing $key")
        println("    g_static[$key] type: $(typeof(g_static[key]))")
        println("    g_fd[$key] type: $(typeof(g_fd[key]))")
        println("    g_static[$key] keys: $(keys(g_static[key]))")
        println("    g_fd[$key] keys: $(keys(g_fd[key]))")
        cos = cosine_similarity(g_static[key], g_fd[key])
        results[key] = cos
    end
    
    return results, g_static, g_fd
end

function run_test()
    rng = Xoshiro(SEED)
    
    println("="^60)
    println("ResidualBlock StaticEP vs FD Gradient Validation")
    println("="^60)
    
    all_results = []
    
    for n_residual in TEST_DEPTHS
        for hidden_dim in TEST_WIDTHS
            println("\nTesting: n_residual=$n_residual, hidden_dim=$hidden_dim")
            
            # Build chain
            chain = build_test_chain(4, hidden_dim, 2, n_residual)
            
            # Initialize
            ps, st = Lux.setup(rng, chain)
            st = Lux.initialstates(rng, chain)
            
            # Create test data
            x = randn(rng, ComplexF32, 4, BATCH_SIZE)
            x = PhasorNetworks.normalize_to_unit_circle(x)
            y = rand(rng, 1:2, BATCH_SIZE)
            
            # Test
            results, g_static, g_fd = test_gradient_fidelity(chain, ps, st, x, y, rng)
            
            println("  Layer cosines:")
            for (key, cos) in results
                println("    $key: cos = $(cos)")
            end
            
            min_cos = minimum(values(results))
            mean_cos = mean(values(results))
            println("  Min cos: $min_cos, Mean cos: $mean_cos")
            
            push!(all_results, (n_residual=n_residual, hidden_dim=hidden_dim, 
                               min_cos=min_cos, mean_cos=mean_cos, results=results))
            
            if min_cos < 0.9
                println("  ⚠️  LOW FIDELITY: min cos < 0.9")
            end
        end
    end
    
    println("\n" * "="^60)
    println("SUMMARY")
    println("="^60)
    for r in all_results
        status = r.min_cos >= 0.95 ? "✅ PASS" : (r.min_cos >= 0.9 ? "⚠️  MARGINAL" : "❌ FAIL")
        println("$status: depth=$(r.n_residual), width=$(r.hidden_dim), min_cos=$(round(r.min_cos, digits=4)), mean_cos=$(round(r.mean_cos, digits=4))")
    end
    
    return all_results
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_test()
end