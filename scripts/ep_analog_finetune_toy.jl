#!/usr/bin/env julia
#
# scripts/ep_analog_finetune_toy.jl — A4: Analog-impairment fine-tuning harness (toy version)
#
# Quick validation of the A4 pipeline on a toy chain (4→8→2).
# This runs in seconds, not minutes.

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

const SEED = 42

# Toy chain
function toy_chain(rng::Xoshiro; scale=0.4f0)
    chain = Chain(
        PhasorDense(4 => 8, normalize_to_unit_circle, use_bias=true),
        PhasorDense(8 => 2, normalize_to_unit_circle, use_bias=true)
    )
    ps, st = Lux.setup(rng, chain)
    ps = (layer_1 = merge(ps.layer_1, (weight = scale .* ps.layer_1.weight,)),
          layer_2 = merge(ps.layer_2, (weight = scale .* ps.layer_2.weight,)))
    return chain, ps, st
end

function toy_input(rng::Xoshiro)
    v = 2f0 .* rand(rng, Float32, 4) .- 1f0
    return Phase.(vec(v))
end

function toy_cost(rng::Xoshiro)
    y = ComplexF32.(exp.(im .* π .* (2f0 .* rand(rng, Float32, 2) .- 1f0)))
    return SimilarityCost(y)
end

# Impairment functions
function apply_lognormal_impairment(ps, σ, rng)
    W1 = ps.layer_1.weight
    W2 = ps.layer_2.weight
    noise1 = exp.(σ .* randn(rng, Float32, size(W1))) .- 1f0
    noise2 = exp.(σ .* randn(rng, Float32, size(W2))) .- 1f0
    return (layer_1 = merge(ps.layer_1, (weight = W1 .* (1f0 .+ noise1),)),
            layer_2 = merge(ps.layer_2, (weight = W2 .* (1f0 .+ noise2),)))
end

function apply_stuck_zero_impairment(ps, frac, rng)
    W1 = ps.layer_1.weight
    W2 = ps.layer_2.weight
    mask1 = rand(rng, Float32, size(W1)) .< frac
    mask2 = rand(rng, Float32, size(W2)) .< frac
    W1_imp = W1 .* (1f0 .- mask1)
    W2_imp = W2 .* (1f0 .- mask2)
    update_mask1 = 1f0 .- mask1
    update_mask2 = 1f0 .- mask2
    ps_imp = (layer_1 = merge(ps.layer_1, (weight = W1_imp,)),
              layer_2 = merge(ps.layer_2, (weight = W2_imp,)))
    update_masks = (layer_1 = (weight = update_mask1,),
                    layer_2 = (weight = update_mask2,))
    return ps_imp, update_masks
end

# Gradient cosine
cosine(a, b) = real(dot(vec(a), vec(b)) / (norm(vec(a)) * norm(vec(b)) + 1e-30))

# Training functions
function train_staticep(chain, ps, st, method, n_epochs, n_steps_per_epoch, rng)
    opt_state = Optimisers.setup(Optimisers.Adam(0.05), ps)
    for epoch in 1:n_epochs
        for i in 1:n_steps_per_epoch
            x = toy_input(rng)
            y_cost = toy_cost(rng)
            g, _ = ep_gradient(method, chain, ps, st, x, y_cost)
            opt_state, ps = Optimisers.update(opt_state, ps, g)
        end
        println("  Epoch $epoch done")
    end
    return ps
end

function train_lockinep(chain, ps, st, method, n_steps, rng)
    opt_state = Optimisers.setup(Optimisers.Adam(0.01), ps)
    for i in 1:n_steps
        x = toy_input(rng)
        y_cost = toy_cost(rng)
        g, _ = ep_gradient(method, chain, ps, st, x, y_cost)
        opt_state, ps = Optimisers.update(opt_state, ps, g)
        println("  LockinEP step $i done")
    end
    return ps
end

# Gradient fidelity
function check_gradient_fidelity(chain, ps, st, x, cost)
    static_method = StaticEP(β=0.005f0, T_free=100, T_nudge=50, dt=0.5f0, centered=true)
    g_static, _ = ep_gradient(static_method, chain, ps, st, x, cost)
    lockin_method = LockinEP(ε=0.05f0, ω_p=0.05f0, n_cycles=2, T_warmup_cycles=1, T_free=50, dt=0.1f0)
    g_lockin, _ = ep_gradient(lockin_method, chain, ps, st, x, cost)
    
    for key in (:layer_1, :layer_2)
        if haskey(g_static, key) && haskey(g_lockin, key)
            g_s = g_static[key].weight
            g_l = g_lockin[key].weight
            re = norm(g_l - g_s) / norm(g_s)
            cs = cosine(g_l, g_s)
            println("$key: cos=$cs, rel-err=$re")
        end
    end
end

# ============================================================
# MAIN EXPERIMENT (TOY)
# ============================================================

rng = Xoshiro(SEED)
chain, ps, st = toy_chain(rng)

# Test sample for fidelity checks
x_test = toy_input(rng)
cost_test = toy_cost(rng)

# 1. PRETRAIN (StaticEP)
println("=== 1. PRETRAIN (StaticEP, 10 epochs × 10 steps) ===")
method = StaticEP(β=0.05f0, T_free=100, T_nudge=50, dt=0.5f0, centered=true)
ps = train_staticep(chain, ps, st, method, 10, 10, rng)

println("\nClean gradient fidelity:")
check_gradient_fidelity(chain, ps, st, x_test, cost_test)

ps_clean = deepcopy(ps)
st_clean = deepcopy(st)

# 2. APPLY IMPAIRMENT - check gradient fidelity
println("\n=== 2. APPLY IMPAIRMENT - Gradient Fidelity ===")
for (name, impair_fn) in [("lognormal σ=0.1", ps -> apply_lognormal_impairment(ps, 0.1f0, Xoshiro(123))),
                           ("lognormal σ=0.3", ps -> apply_lognormal_impairment(ps, 0.3f0, Xoshiro(124))),
                           ("stuck_zero 10%", ps -> apply_stuck_zero_impairment(ps, 0.1f0, Xoshiro(125))[1])]
    ps_imp = impair_fn(ps_clean)
    println("\n$name gradient fidelity:")
    check_gradient_fidelity(chain, ps_imp, st_clean, x_test, cost_test)
end

# 3. FINE-TUNE through impairment (StaticEP)
println("\n=== 3. FINE-TUNE (StaticEP, 5 epochs × 10 steps) ===")
method_ft = StaticEP(β=0.05f0, T_free=100, T_nudge=50, dt=0.5f0, centered=true)

for (name, impair_fn) in [("lognormal σ=0.1", ps -> apply_lognormal_impairment(ps, 0.1f0, Xoshiro(123))),
                           ("lognormal σ=0.3", ps -> apply_lognormal_impairment(ps, 0.3f0, Xoshiro(124))),
                           ("stuck_zero 10%", ps -> apply_stuck_zero_impairment(ps, 0.1f0, Xoshiro(125))[1])]
    ps_imp = impair_fn(ps_clean)
    ps_imp = train_staticep(chain, ps_imp, st_clean, method_ft, 5, 10, rng)
    println("\n$name after fine-tune gradient fidelity:")
    check_gradient_fidelity(chain, ps_imp, st_clean, x_test, cost_test)
end

# 4. Test LockinEP fine-tune (5 steps - slow)
println("\n=== 4. LOCKIN EP FINE-TUNE (5 steps, σ=0.1) ===")
ps_imp = apply_lognormal_impairment(ps_clean, 0.1f0, Xoshiro(123))
lockin_method = LockinEP(ε=0.05f0, ω_p=0.05f0, n_cycles=2, T_warmup_cycles=1, T_free=50, dt=0.1f0)
ps_imp = train_lockinep(chain, ps_imp, st_clean, lockin_method, 5, rng)

println("\nLockinEP after fine-tune gradient fidelity:")
check_gradient_fidelity(chain, ps_imp, st_clean, x_test, cost_test)

println("\n=== TOY A4 PIPELINE VALIDATED ===")
println("The A4 pipeline structure works. Full FashionMNIST runs need HPC.")