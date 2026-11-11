#!/usr/bin/env julia
# Component-wise update pairs for Figure 3's correspondence panel.
#
# results/e4_three_factor_scale/e4_three_factor_*.csv stores only per-tensor
# SUMMARIES (cosine, relative error). To show that the three-factor rule and the
# lock-in estimator produce the same update -- rather than merely asserting a
# summary statistic -- the figure needs the updates themselves, plotted against
# one another on the identity line. This script dumps a random sample of
# component pairs per parameter tensor.
#
# Configuration mirrors scripts/e4_three_factor_scale.jl exactly (same seed,
# architecture, method settings and input construction) so the pairs come from
# the same experiment the summary CSV describes. The rule itself is included
# from scripts/three_factor_rule.jl, which both scripts share.
#
#   julia --project=scripts scripts/e4_export_update_pairs.jl
#
# Env: E4P_INPUTS (default 8), E4P_PER_TENSOR (samples kept per tensor, 250).

using Pkg
function find_repo_root(start_dir::String = pwd())
    dir = start_dir
    while !(isfile(joinpath(dir, "Project.toml")) && isdir(joinpath(dir, ".git")))
        parent = dirname(dir); parent == dir && error("no repo root"); dir = parent
    end
    return dir
end
repo_root = find_repo_root(@__DIR__); cd(repo_root); Pkg.activate(joinpath(repo_root, "scripts"))

using PhasorNetworks, Lux, LinearAlgebra, Statistics, Random, CSV, DataFrames, Printf, CUDA
using Random: Xoshiro
using PhasorNetworks: AbstractEPMethod, LockinEP, SimilarityCost, AbstractEPCost,
                      ep_gradient, phasor_settle, chain_hebbians, _phase_input_to_complex,
                      _weight_cache, _input_drive, _phasor_step, _pad_dynamics_zeros,
                      _zero_grad, gpu_zeros, normalize_to_unit_circle

include(joinpath(@__DIR__, "three_factor_rule.jl"))

_envi(k, d) = parse(Int, get(ENV, k, string(d)))

# ---- identical to scripts/e4_three_factor_scale.jl ----
const SEED  = 42
const HID   = 256
const DOUT  = 64
const SCALE = 0.4f0
const N_INPUTS    = _envi("E4P_INPUTS", 8)
const PER_TENSOR  = _envi("E4P_PER_TENSOR", 250)
const USE_CUDA = CUDA.functional()

const OUT = joinpath(repo_root, "results", "e4_three_factor_scale"); mkpath(OUT)
const GITREV = try
    rev = strip(read(`git -C $(repo_root) rev-parse --short HEAD`, String))
    d   = read(`git -C $(repo_root) diff HEAD -- src`, String)
    isempty(strip(d)) ? rev : rev * "-d" * string(hash(d), base = 16)[1:8]
catch; "unknown" end

cdev = cpu_device(); gdev = gpu_device(); dev = USE_CUDA ? gdev : cdev

rng = Xoshiro(SEED)
chain = Chain(PhasorDense(784 => HID, normalize_to_unit_circle, use_bias = true),
              PhasorDense(HID => DOUT, normalize_to_unit_circle, use_bias = true))
ps, st = Lux.setup(rng, chain)
ps = (layer_1 = merge(ps.layer_1, (weight = SCALE .* ps.layer_1.weight,)),
      layer_2 = merge(ps.layer_2, (weight = SCALE .* ps.layer_2.weight,)))
if USE_CUDA
    chain = chain |> gdev; ps = ps |> gdev; st = st |> gdev
end

m_lockin = LockinEP(ε = 0.05f0, ω_p = 0.05f0, n_cycles = 4, T_warmup_cycles = 2,
                    T_free = 100, dt = 0.1f0, K_mode = :zero)
m_3f = ThreeFactorLockin(ε = 0.05f0, ω_p = 0.05f0, n_cycles = 4, T_warmup_cycles = 2,
                         T_free = 100, dt = 0.1f0, K_mode = :zero)

# Same construction as the E4 sweep, so these are the same inputs it used.
inputs  = [Phase.(rand(rng, Float32, 784) .* 2 .- 1) for _ in 1:100]
targets = [ComplexF32.(exp.(im .* π .* (rand(rng, Float32, DOUT) .* 2 .- 1))) for _ in 1:100]

const TENSORS = [(:layer_1, :weight, "W1"), (:layer_1, :bias_real, "Reb1"),
                 (:layer_1, :bias_imag, "Imb1"), (:layer_2, :weight, "W2"),
                 (:layer_2, :bias_real, "Reb2"), (:layer_2, :bias_imag, "Imb2")]
tid = Dict(t[3] => i for (i, t) in enumerate(TENSORS))

cosine(a, b) = real(dot(vec(a), vec(b)) / (norm(vec(a)) * norm(vec(b)) + 1e-30))

rows = DataFrame(glk = Float32[], gtf = Float32[], tid = Int[], input_idx = Int[])
srng = Xoshiro(SEED + 99)

println("=== component-wise update pairs, $N_INPUTS inputs x $(length(TENSORS)) tensors ===")
worst = 1.0
for idx in 1:N_INPUTS
    x = inputs[idx] |> dev
    cost = SimilarityCost(targets[idx] |> dev)
    g_lk, _ = ep_gradient(m_lockin, chain, ps, st, x, cost)
    g_3f, _ = three_factor_gradient(m_3f, chain, ps, st, x, cost)
    for (lay, par, name) in TENSORS
        a = vec(Array(getproperty(g_lk[lay], par)))
        b = vec(Array(getproperty(g_3f[lay], par)))
        c = cosine(a, b); global worst = min(worst, c)
        k = min(PER_TENSOR, length(a))
        for i in randperm(srng, length(a))[1:k]
            push!(rows, (a[i], b[i], tid[name], idx))
        end
    end
    @printf("  input %2d/%d  worst tensor cos so far %.7f\n", idx, N_INPUTS, worst)
end

# Self-check: these must reproduce the identity the E4 summary reports. If this
# drifts, the rule in three_factor_rule.jl has diverged from the one that
# produced results/e4_three_factor_scale/e4_three_factor_*.csv.
@printf("\nworst per-tensor cosine over all inputs: %.7f\n", worst)
worst < 0.9999 && @warn "cosine below the 0.99999 the E4 sweep reports -- check three_factor_rule.jl"

f = joinpath(OUT, "update_pairs_$(GITREV).csv")
CSV.write(f, rows)
@printf("wrote %s (%d pairs; tid %s)\n", f, nrow(rows),
        join(("$i=$n" for (n, i) in sort(collect(tid), by = last)), ", "))
