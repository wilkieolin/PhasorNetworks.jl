#!/usr/bin/env julia
#
# scripts/ep_depth_width_scaling.jl — A5: Depth and width scaling of the operating zone
#
# Tests whether ω_p / R_relax is the invariant that collapses the failure contour
# across architectures. If it is, the whole sweep collapses to one cheap R_relax
# measurement per network, which makes the method deployable rather than hand-tuned.

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

using PhasorNetworks, Lux, LinearAlgebra, Statistics, Random, Printf, CSV, DataFrames
using Random: Xoshiro

# ============================================================
# CONFIGURATION
# ============================================================

const OUT = get(ENV, "EPS_OUT",
                joinpath(repo_root, "results", "ep_depth_width_scaling"))
mkpath(OUT)

_envi(k, d) = parse(Int, get(ENV, k, string(d)))
_envf(k, d) = parse(Float32, get(ENV, k, string(d)))
_envs(k, d) = get(ENV, k, string(d))

# Grid axes (env-overridable)
_envfs_or(k, default) = haskey(ENV, k) ?
    Tuple(parse(Float32, strip(x)) for x in split(ENV[k], ",")) : default

const HID_LIST   = _envs("EPS_HID_LIST",    "64,256,1024")  # hidden widths to test
const DEPTH_LIST = _envs("EPS_DEPTH_LIST",  "2,3,4")        # number of hidden layers
const DOUT       = _envi("EPS_DOUT",        10)              # output dim (10 for MNIST)
const B          = _envi("EPS_B",           32)
const REPS       = _envi("EPS_REPS",        8)
const DT         = _envf("EPS_DT",          0.5)
const SCALE      = _envf("EPS_SCALE",       0.4)
const T_FREE     = _envi("EPS_TFREE",       200)
const WARMUP     = _envi("EPS_WARMUP",      1)
const THRESH     = parse(Float64, get(ENV, "EPS_THRESH", "0.99"))

# LockinEP params (same as adiabatic sweep defaults)
const EPS_VALS   = _envfs_or("EPS_EPS",     (0.01f0, 0.03f0, 0.1f0, 0.3f0))
const OMEGA_VALS = _envfs_or("EPS_OMEGA",   (0.005f0, 0.01f0, 0.02f0, 0.05f0, 0.1f0, 0.2f0))
const NCYC_VALS  = parse.(Int, split(_envs("EPS_NCYC", "4,8,16"), ","))  # n_cycles values

# Provenance
const GITREV = try
    rev = strip(read(`git -C $(repo_root) rev-parse --short HEAD`, String))
    d   = read(`git -C $(repo_root) diff HEAD -- ../src`, String)
    isempty(strip(d)) ? rev : rev * "-d" * string(hash(d), base = 16)[1:8]
catch
    "unknown"
end

# ============================================================
# HELPER FUNCTIONS
# ============================================================

function _make_chain(din::Int, hid::Int, depth::Int, dout::Int, rng::Xoshiro)
    layers = []
    push!(layers, PhasorDense(din => hid, normalize_to_unit_circle, use_bias=true))
    for _ in 2:depth
        push!(layers, PhasorDense(hid => hid, normalize_to_unit_circle, use_bias=true))
    end
    push!(layers, PhasorDense(hid => dout, normalize_to_unit_circle, use_bias=true))
    chain = Chain(layers...)
    ps, st = Lux.setup(rng, chain)
    # Scale weights
    new_ps = NamedTuple()
    for key in keys(ps)
        p = ps[key]
        if haskey(p, :weight)
            new_ps = merge(new_ps, NamedTuple{(key,)}((merge(p, (weight = SCALE .* p.weight,)),)))
        else
            new_ps = merge(new_ps, NamedTuple{(key,)}((p,)))
        end
    end
    return chain, new_ps, st
end

function _random_input(din::Int, rng::Xoshiro)
    return Phase.(2f0 .* rand(rng, Float32, din) .- 1f0)
end

function _random_target(dout::Int, rng::Xoshiro)
    return ComplexF32.(exp.(im .* π .* (2f0 .* rand(rng, Float32, dout) .- 1f0)))
end

# Measure R_relax using phasor_settle with monitoring
# R_relax ≈ 1 / (T_settle * dt) where T_settle is steps to reach steady state
function _measure_R_relax(chain, ps, st, x, cost;
                           T_max=500, dt=0.5f0, thresh=1e-4)
    # Run settle and track residual at each step
    # We'll use a modified version that returns trajectory
    layer_keys = collect(keys(ps))
    z0 = _phase_input_to_complex(x)
    states = [zeros(ComplexF32, chain.layers[k].out_dims) for k in layer_keys]
    cache = _weight_cache(chain, ps, layer_keys)
    drive0 = _input_drive(chain, ps, st, layer_keys, z0; cache=cache)
    
    residual_history = Float32[]
    
    for t in 1:T_max
        # One step
        new_states = _phasor_step(chain, ps, st, layer_keys, z0, cost,
                                  0f0, dt, states; K_mode=:zero, drive0=drive0, cache=cache)
        
        # Compute residual
        res = 0f0
        for (l, key) in enumerate(layer_keys)
            z = new_states[l]
            z_prev = (l == 1) ? z0 : new_states[l-1]
            
            # drive = W_l * z_{l-1} + bias
            grad = _cached_drive(cache, l, chain, key, ps[key], st[key], z_prev)
            if l < length(layer_keys)
                key_n = layer_keys[l+1]
                grad = grad .+ _cached_feedback(cache, l+1, chain, key_n,
                                                ps[key_n], st[key_n], new_states[l+1])
            end
            # residual = ||z - project(grad)||
            # Use the internal projection (hard, ε=0)
            proj = PhasorNetworks._project_damp.(z, grad, dt, 1.0f-10)
            res += sqrt(sum(abs2.(z .- proj)))
        end
        push!(residual_history, res)
        states = new_states
        if res < thresh
            break
        end
    end
    T_settle = length(residual_history)
    return Float32(1.0 / (T_settle * dt))
end

# Run LockinEP gradient and measure fidelity vs centered StaticEP oracle
function _fidelity_at(chain, ps, st, x, cost, ε, ω_p, n_cycles;
                       T_free=T_FREE, T_warmup_cycles=WARMUP, dt=DT)
    # Reference: centered StaticEP
    ref_method = StaticEP(β=0.005f0, T_free=200, T_nudge=100, dt=0.5f0,
                          centered=true, K_mode=:zero)
    g_ref, _ = ep_gradient(ref_method, chain, ps, st, x, cost)
    
    # LockinEP
    m = LockinEP(ε=ε, ω_p=ω_p, n_cycles=n_cycles,
                 T_warmup_cycles=T_warmup_cycles,
                 T_free=T_free, dt=dt, K_mode=:zero)
    g_lockin, _ = ep_gradient(m, chain, ps, st, x, cost)
    
    # Cosine similarity on weights (only)
    cos_vals = Float32[]
    for key in keys(ps)
        haskey(ps[key], :weight) || continue
        g1 = g_ref[key].weight
        g2 = g_lockin[key].weight
        c = real(dot(vec(g1), vec(g2)) / (norm(vec(g1)) * norm(vec(g2)) + 1e-30))
        push!(cos_vals, Float32(c))
    end
    return isempty(cos_vals) ? 0f0 : minimum(cos_vals)  # min over layers
end

# ============================================================
# MAIN SWEEP
# ============================================================

function run_sweep()
    println("=== A5: Depth/Width Scaling Sweep ===")
    println("HID_LIST: $HID_LIST")
    println("DEPTH_LIST: $DEPTH_LIST")
    println("EPS_VALS: $EPS_VALS")
    println("OMEGA_VALS: $OMEGA_VALS")
    println("NCYC_VALS: $NCYC_VALS")
    println("Output: $OUT")
    
    hid_vals = parse.(Int, split(HID_LIST, ","))
    depth_vals = parse.(Int, split(DEPTH_LIST, ","))
    
    # CSV output
    csv_file = joinpath(OUT, "scaling_$(GITREV).csv")
    isfile(csv_file) || CSV.write(csv_file, DataFrame(
        gitrev=String[], hid=Int[], depth=Int[], din=Int[],
        R_relax=Float32[],
        ε=Float32[], ω_p=Float32[], n_cycles=Int[],
        cos_min=Float32[], pass=Bool[],
        reps=Int[]
    ))
    
    rng_base = Xoshiro(42)
    
    for hid in hid_vals
        for depth in depth_vals
            println("\n=== Architecture: hid=$hid, depth=$depth ===")
            
            # Fixed architecture for this config
            chain, ps, st = _make_chain(784, hid, depth, DOUT, rng_base)
            
            # Measure R_relax (once per architecture)
            println("  Measuring R_relax...")
            x_test = _random_input(784, rng_base)
            y_test = _random_target(DOUT, rng_base)
            cost_test = SimilarityCost(y_test)
            
            R_relax = _measure_R_relax(chain, ps, st, x_test, cost_test)
            println("  R_relax = $R_relax")
            
            # Sweep over ε, ω_p, n_cycles
            for ε in EPS_VALS
                for ω_p in OMEGA_VALS
                    for n_cycles in NCYC_VALS
                        # Skip combinations that are clearly too slow
                        period_steps = round(Int, 2π / (ω_p * DT))
                        total_steps = T_FREE + (WARMUP + n_cycles) * period_steps
                        if total_steps > 5000  # too slow
                            println("    ε=$ε, ω_p=$ω_p, n_cycles=$n_cycles: SKIP (steps=$total_steps)")
                            continue
                        end
                        
                        println("    ε=$ε, ω_p=$ω_p, n_cycles=$n_cycles (steps=$total_steps)")
                        
                        # Run REPS replicates
                        cos_reps = Float32[]
                        for rep in 1:REPS
                            rng_rep = Xoshiro(hash((hid, depth, ε, ω_p, n_cycles, rep)) % UInt64)
                            x = _random_input(784, rng_rep)
                            y = _random_target(DOUT, rng_rep)
                            cost = SimilarityCost(y)
                            
                            c = _fidelity_at(chain, ps, st, x, cost, ε, ω_p, n_cycles)
                            push!(cos_reps, c)
                        end
                        
                        cos_min = minimum(cos_reps)
                        pass = cos_min >= THRESH
                        println("      cos_min=$cos_min, pass=$pass")
                        
                        # Append to CSV
                        row = DataFrame(
                            gitrev=GITREV,
                            hid=hid,
                            depth=depth,
                            din=784,
                            R_relax=R_relax,
                            ε=ε,
                            ω_p=ω_p,
                            n_cycles=n_cycles,
                            cos_min=cos_min,
                            pass=pass,
                            reps=REPS
                        )
                        CSV.write(csv_file, row, append=true)
                    end
                end
            end
        end
    end
    
    println("\n=== Sweep complete. Results in $csv_file ===")
    
    # Analysis: check if ω_p / R_relax collapses the data
    println("\n=== Analysis: ω_p / R_relax collapse ===")
    df = CSV.read(csv_file, DataFrame)
    df[!, :omega_over_R] = df.ω_p ./ df.R_relax
    
    # Group by ω_p/R and show pass rates
    if nrow(df) > 0
        collapse = combine(groupby(df, [:omega_over_R, :hid, :depth]),
                          :pass => mean => :pass_rate,
                          :cos_min => minimum => :cos_min,
                          nrow => :n)
        println(collapse)
    end
    
    return df
end

# Import needed internal functions
using PhasorNetworks: _phase_input_to_complex, _weight_cache, _input_drive,
                      _phasor_step, _cached_drive, _cached_feedback,
                      StaticEP, LockinEP, SimilarityCost, ep_gradient,
                      _project_damp

run_sweep()