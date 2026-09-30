#!/usr/bin/env julia
# E7: measure kappa, the slowest relaxation rate of the free settle.
#
# WHY
# ---
# Hypothesis (H4) of the LIEP derivation is stated as omega_p << kappa, where
# kappa is the slowest relaxation rate of the linearised settling dynamics about
# the free equilibrium. The adiabatic sweep, however, reports its boundary in
# raw omega_p, which is a setting rather than a physical quantity and does not
# transfer to another width. This script measures kappa so the basin map can be
# drawn against omega_p/kappa.
#
# The whole grid was measured on ONE network -- `stage_grid` calls
# `build_chain(7)` once (scripts/ep_adiabatic_sweep.jl:464) and varies only the
# input across its eight draws -- so a single kappa rescales the entire map
# exactly. Everything below therefore mirrors that script's constants and its
# `make_input(100 + r)` draws; if those drift, this measurement is void.
#
# WHY NOT `_measure_R_relax`
# --------------------------
# scripts/ep_depth_width_scaling.jl:106 already has an estimator, but it returns
# 1/(T_settle*dt) for the first step at which the residual crosses 1e-4. That is
# a threshold crossing, not a rate. Decaying from O(1) to 1e-4 takes about
# ln(1e4) = 9.2 time constants, so it understates kappa by roughly an order of
# magnitude -- and taken at face value it would place the measured basin
# boundary at omega_p/kappa ~ 2, i.e. the method surviving well past omega_p =
# kappa, which contradicts (H4).
#
# Here kappa is fitted from the exponential tail of ||z(t) - z*||: linearising
# the projected update about z*, the error contracts by a fixed factor per step,
# so log||z(t) - z*|| is affine in t with slope -kappa*dt. Both numbers are
# recorded so the discrepancy is documented rather than silent.
#
#   julia --project=scripts scripts/measure_kappa.jl
#
# Env: EK_TLONG (steps used to define z*), EK_TTRACE (steps traced), EK_REPS.

using Pkg
function find_repo_root(start_dir::String = pwd())
    dir = start_dir
    while !(isfile(joinpath(dir, "Project.toml")) && isdir(joinpath(dir, ".git")))
        parent = dirname(dir); parent == dir && error("no repo root"); dir = parent
    end
    return dir
end
repo_root = find_repo_root(@__DIR__); cd(repo_root); Pkg.activate(joinpath(repo_root, "scripts"))

using PhasorNetworks, Lux, LinearAlgebra, Statistics, Random, Printf, CSV, DataFrames
using Random: Xoshiro
using PhasorNetworks: _phase_input_to_complex, _weight_cache, _input_drive,
                      _phasor_step, _init_states, normalize_to_unit_circle

_envi(k, d) = parse(Int, get(ENV, k, string(d)))

# ---- constants copied from scripts/ep_adiabatic_sweep.jl; keep matched ----
const HID    = 256
const DOUT   = 64
const NCLS   = 10
const B      = 32
const DT     = 0.5f0
const SCALE  = 0.4f0
const REPS   = _envi("EK_REPS", 8)
const CHAIN_SEED = 7      # stage_grid's build_chain(7)
const CODE_SEED  = 8      # stage_grid's make_codes(8)
const INPUT_SEED0 = 100   # stage_grid's make_input(100 + r)

const T_LONG  = _envi("EK_TLONG", 6000)   # long settle defining z*
const T_TRACE = _envi("EK_TTRACE", 1200)  # steps traced for the fit

const OUT = joinpath(repo_root, "results", "e7_kappa"); mkpath(OUT)
const GITREV = try
    rev = strip(read(`git -C $(repo_root) rev-parse --short HEAD`, String))
    d   = read(`git -C $(repo_root) diff HEAD -- src`, String)
    isempty(strip(d)) ? rev : rev * "-d" * string(hash(d), base = 16)[1:8]
catch; "unknown" end

function build_chain(seed::Int)
    chain = Chain(PhasorDense(784 => HID,  normalize_to_unit_circle, use_bias = true),
                  PhasorDense(HID  => DOUT, normalize_to_unit_circle, use_bias = true))
    ps, st = Lux.setup(Xoshiro(seed), chain)
    ps = (layer_1 = merge(ps.layer_1, (weight = SCALE .* ps.layer_1.weight,)),
          layer_2 = merge(ps.layer_2, (weight = SCALE .* ps.layer_2.weight,)))
    return chain, ps, st
end
make_codes(seed::Int) =
    ComplexF32.(angle_to_complex(orthogonal_codes(Xoshiro(seed), DOUT, NCLS)))
function make_input(seed::Int, b::Int = B)
    rng = Xoshiro(seed)
    pix = rand(rng, Float32, 784, b)
    pix[rand(rng, Float32, 784, b) .< 0.8f0] .= 0f0
    mu = mean(pix; dims = 1); sd = std(pix; dims = 1) .+ 1f-6
    return Phase.(0.5f0 .* tanh.((pix .- mu) ./ sd))
end
make_cost(codes, seed::Int, b::Int = B) =
    CodebookCost(codes, rand(Xoshiro(seed + 5_000), 1:NCLS, b))

"""
Trace the free settle and return, per step, the distance to the converged state
and the step-to-step residual. `beta = 0` throughout: kappa is a property of the
free dynamics, and the nudge is what it has to be slow compared with.
"""
function settle_trace(chain, ps, st, x, cost; T_long = T_LONG, T_trace = T_TRACE)
    layer_keys = collect(keys(ps))
    z0 = _phase_input_to_complex(x)
    cache  = _weight_cache(chain, ps, layer_keys)
    drive0 = _input_drive(chain, ps, st, layer_keys, z0; cache = cache)
    step(s) = _phasor_step(chain, ps, st, layer_keys, z0, cost, 0f0, DT, s;
                           K_mode = :zero, drive0 = drive0, cache = cache,
                           project = :hard)

    # z*: settle far past anything the estimator uses
    zstar = _init_states(chain, layer_keys, z0)
    for _ in 1:T_long; zstar = step(zstar); end
    nrm(a) = sqrt(sum(sum(abs2, v) for v in a))
    scale = nrm(zstar) + 1f-30

    states = _init_states(chain, layer_keys, z0)
    err  = Float64[]; resid = Float64[]
    for _ in 1:T_trace
        prev = states
        states = step(states)
        push!(err,   sqrt(sum(sum(abs2, s .- z) for (s, z) in zip(states, zstar))) / scale)
        push!(resid, sqrt(sum(sum(abs2, s .- p) for (s, p) in zip(states, prev))) / scale)
    end
    return err, resid
end

"""
Fit log(err) affine in time over the cleanest decade of the tail, and return
kappa = -slope. The window starts after the initial transient and stops before
the Float32 noise floor, both located from the data rather than hardcoded.
"""
function fit_kappa(err; dt = DT)
    lo_v, hi_v = 1e-6, 1e-1                    # trust only this band
    idx = findall(e -> lo_v < e < hi_v && isfinite(e), err)
    length(idx) < 20 && return (NaN, NaN, 0, 0, 0)
    # drop the first 20% of the usable range: still transient, several modes live
    i0 = idx[max(1, round(Int, 0.20 * length(idx)))]
    i1 = idx[end]
    ii = [i for i in i0:i1 if lo_v < err[i] < hi_v]
    length(ii) < 20 && return (NaN, NaN, 0, 0, 0)
    t = Float64.(ii) .* dt
    y = log.(err[ii])
    tb = mean(t); yb = mean(y)
    slope = sum((t .- tb) .* (y .- yb)) / sum((t .- tb) .^ 2)
    yhat = yb .+ slope .* (t .- tb)
    ss_res = sum((y .- yhat) .^ 2); ss_tot = sum((y .- yb) .^ 2)
    r2 = ss_tot > 0 ? 1 - ss_res / ss_tot : NaN
    return (-slope, r2, first(ii), last(ii), length(ii))
end

"The repo's existing convention, for cross-reference: 1/(T_settle*dt) at 1e-4."
function r_relax(resid; dt = DT, thresh = 1e-4)
    k = findfirst(<(thresh), resid)
    k === nothing ? NaN : 1 / (k * dt)
end

println("=== E7: relaxation rate of the free settle ===")
println("network build_chain($CHAIN_SEED), inputs make_input($INPUT_SEED0 + r), dt = $DT")
chain, ps, st = build_chain(CHAIN_SEED)
codes = make_codes(CODE_SEED)

rows = DataFrame(gitrev = String[], rep = Int[], kappa = Float64[], r2 = Float64[],
                 fit_lo = Int[], fit_hi = Int[], fit_n = Int[],
                 R_relax = Float64[], ratio = Float64[],
                 dt = Float32[], hidden = Int[], output = Int[], scale = Float32[],
                 chain_seed = Int[], T_long = Int[], T_trace = Int[])

@printf("%4s %12s %8s %14s %12s %8s\n", "rep", "kappa", "R^2", "fit window", "R_relax", "ratio")
for r in 1:REPS
    x    = make_input(INPUT_SEED0 + r)
    cost = make_cost(codes, INPUT_SEED0 + r)
    err, resid = settle_trace(chain, ps, st, x, cost)
    k, r2, lo, hi, n = fit_kappa(err)
    rr = r_relax(resid)
    @printf("%4d %12.5f %8.4f %6d-%-6d %12.5f %8.2f\n", r, k, r2, lo, hi, rr, k / rr)
    push!(rows, (GITREV, r, k, r2, lo, hi, n, rr, k / rr,
                 DT, HID, DOUT, SCALE, CHAIN_SEED, T_LONG, T_TRACE))
end

ks = filter(isfinite, rows.kappa)
kmed = median(ks)
@printf("\nkappa  median %.5f   min %.5f   max %.5f   (n = %d)\n",
        kmed, minimum(ks), maximum(ks), length(ks))
@printf("R_relax median %.5f  -- the repo's 1/(T_settle*dt) convention, %.1fx smaller\n",
        median(filter(isfinite, rows.R_relax)), kmed / median(filter(isfinite, rows.R_relax)))
@printf("\nbasin boundary omega_p* = 0.0425 becomes omega_p*/kappa = %.3f\n", 0.0425 / kmed)
for w in (0.005, 0.01, 0.02, 0.05, 0.1, 0.2)
    @printf("  omega_p = %-6g ->  omega_p/kappa = %.4f\n", w, w / kmed)
end

f = joinpath(OUT, "kappa_$(GITREV).csv")
CSV.write(f, rows)
println("\nwrote $f")
