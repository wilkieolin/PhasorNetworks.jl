#!/usr/bin/env julia
#
# scripts/ep_readout_frame_check.jl — does the carrier reduction reach the
# readout?
#
# WHY THIS FILE EXISTS
# --------------------
# `docs/phasor_lockin_derivation.tex` §Rotating Substrate proves that for one
# shared carrier ω the lab and co-rotating frames give the SAME settle, exactly,
# at any dt. That proof is about the state. It says nothing about how the state
# is observed, and a spiking substrate does not report a complex number — it
# reports a spike time quantized against a clock.
#
# `_quantize_phase` is the one non-U(1)-equivariant operation in the pipeline:
# rounding onto a grid of spacing δ turns commutes with a rotation of θ turns
# iff θ/δ ∈ ℤ. So with `readout_δ > 0` the frame stops being a change of
# variables and becomes a modelling choice about what the readout clock is
# locked to.
#
# This script measures the size of that choice. It is the evidence behind the
# scope caveat in `results/ep_readout_floor/FINDINGS.md` and §0 of
# `docs/ep_rotating_followups.md`, and it is checked in so those numbers have a
# reproducible source rather than a remembered one.
#
# Gate F in `scripts/ep_rotating_gates.jl` asserts the equivariance boundary
# (part 1 here) as a blocking test. Part 2 is a measurement, not a gate — it
# reports, it does not fail.
#
#   julia --project=. scripts/ep_readout_frame_check.jl
#
using PhasorNetworks, Lux, LinearAlgebra, Printf, Statistics
using Random: Xoshiro
const P = PhasorNetworks

const DELTA = 0.005f0          # 0.005 turns = 1.8° = t_window/t_period at 0.005

function toy_chain(seed = 42; scale = 0.4f0)
    ch = Chain(PhasorDense(4 => 8, normalize_to_unit_circle, use_bias=true),
               PhasorDense(8 => 2, normalize_to_unit_circle, use_bias=true))
    ps, st = Lux.setup(Xoshiro(seed), ch)
    # Same downscale as test/test_ep.jl and the gates: the default glorot is
    # wide enough that some initial drives land on `_project_damp`'s
    # discontinuity, which is not what this script is measuring.
    ps = (layer_1 = merge(ps.layer_1, (weight = scale .* ps.layer_1.weight,)),
          layer_2 = merge(ps.layer_2, (weight = scale .* ps.layer_2.weight,)))
    return ch, ps, st
end

cosine(a, b) = real(dot(vec(a), vec(b)) / (norm(vec(a)) * norm(vec(b)) + 1e-30))

# ---------------------------------------------------------------------
# Part 1 — the quantizer is equivariant only on grid multiples.
# ---------------------------------------------------------------------
function part1()
    println("1. Quantizer equivariance:  ‖Q(e^{iθ}z)·e^{-iθ} − Q(z)‖∞,  δ = $(DELTA) turns")
    z = ComplexF32.(cis.(2π .* rand(Xoshiro(1), Float32, 4000)))
    for θt in (0.5f0, DELTA, 0.1353f0, 0.0017f0)
        θ = 2f0 * Float32(π) * θt
        d = maximum(abs.(P._readout(z .* cis(θ), DELTA, 0f0, nothing) .* conj(cis(θ)) .-
                         P._readout(z, DELTA, 0f0, nothing)))
        @printf("   rotation = %.4f turns = %8.3f bins  ->  max|Δ| = %.3e  %s\n",
                θt, θt / DELTA, d,
                abs(θt / DELTA - round(θt / DELTA)) < 1e-6 ? "(commensurate)" : "")
    end
end

# ---------------------------------------------------------------------
# Part 2 — the same lock-in estimator with the grid in different frames.
# ---------------------------------------------------------------------
# `LockinEP` has no `carrier` field (the co-rotating choice is baked in), so the
# loop is driven by hand here. The ONLY difference between arms is where the
# readout grid sits: `obs` quantizes the LAB-frame value and then demodulates,
# so `carrier=nothing` reproduces the estimator as `LockinEP` runs it today.
#
# Note the one-step time convention: `_phasor_step(…, t_now=t)` returns the
# state at `t + dt` (see `phasor_settle`'s loop). Demodulating at `t` instead of
# `t + dt` costs an exact sign flip at ω = 2π, dt = 0.5.
function lockin(ch, ps, st, x, cost; ε, ω_p, n_cycles, T_warm, T_free, dt,
                δ, jitter, seed, carrier = nothing)
    keys_ = collect(keys(ps)); n = length(keys_)
    z0 = P._phase_input_to_complex(x)
    ph(t) = carrier === nothing ? ComplexF32(1) :
            ComplexF32(cis(mod(Float64(carrier) * Float64(t), 2π)))
    rng = Xoshiro(seed)
    nz(a) = jitter <= 0f0 ? nothing : jitter .* randn(rng, Float32, size(a))
    obs(z_lab, t) = P._readout(z_lab, δ, jitter, nz(z_lab)) .* conj(ph(t))

    sf = phasor_settle(ch, ps, st, x, cost, 0f0; T=T_free, dt=dt,
                       carrier=carrier, t0=0f0)
    tf = Float32(T_free * dt)
    hdc = chain_hebbians(ch, ps, st, obs(z0 .* ph(tf), tf), [obs(s, tf) for s in sf])

    cache  = P._weight_cache(ch, ps, keys_)
    drive0 = P._input_drive(ch, ps, st, keys_, z0; cache=cache)
    states = [copy(s) for s in sf]
    pst = round(Int, 2π / (ω_p * dt))

    for t in 1:(T_warm * pst)
        states = P._phasor_step(ch, ps, st, keys_, z0, cost,
                                Float32(ε * cos(ω_p * t * dt)), Float32(dt), states;
                                drive0=drive0, cache=cache,
                                carrier=carrier, t_now=Float32(tf + (t - 1) * dt))
    end

    tw = tf + T_warm * pst * dt
    T  = n_cycles * pst
    HW = [zeros(ComplexF32, size(ps[k].weight)) for k in keys_]
    c  = zero(ComplexF32)
    for t in 1:T
        tn = Float32(tw + t * dt)                       # time of the NEW state
        states = P._phasor_step(ch, ps, st, keys_, z0, cost,
                                Float32(ε * cos(ω_p * t * dt)), Float32(dt), states;
                                drive0=drive0, cache=cache,
                                carrier=carrier, t_now=Float32(tw + (t - 1) * dt))
        d = ComplexF32(exp(-im * ω_p * t * dt)); c += d
        o   = [obs(s, tn) for s in states]
        inp = obs(z0 .* ph(tn), tn)
        for l in 1:n
            HW[l] .+= d .* (o[l] * adjoint(l == 1 ? inp : o[l - 1]))
        end
    end

    g = Dict{Symbol,Any}()
    for (l, k) in enumerate(keys_)
        invB = 1f0 / Float32(P._batch_size(sf[l]))
        H = HW[l] .* invB .- c .* ComplexF32.(hdc[k].weight)
        g[k] = -2f0 .* real.(H) ./ (Float32(T) * Float32(ε))
    end
    return g
end

const ARMS = [(nothing,       "co-rotating (LockinEP as it stands)"),
              (Float32(2π),   "lab ω=2π   (0.25 turn/step = 50 bins)"),
              (1.7f0,         "lab ω=1.7  (27.06 bins/step)"),
              (2.9f0,         "lab ω=2.9  (46.16 bins/step)")]

function part2(; δ = DELTA, jitter = 0f0, draws = 8)
    kw = (ε=0.02f0, ω_p=0.02f0, n_cycles=4, T_warm=2, T_free=200, dt=0.5f0)
    @printf("\n2. Lock-in gradient vs centered StaticEP — %d draws, δ = %.4f turns, jitter = %.5f\n",
            draws, δ, jitter)
    println("   median cos (layer_1 / layer_2), grid fixed in the stated frame")
    res = Dict(a[2] => (Float64[], Float64[]) for a in ARMS)
    for d in 1:draws
        ch, ps, st = toy_chain(40 + d)
        x = Phase.(2f0 .* rand(Xoshiro(100 + d), Float32, 4, 3) .- 1f0)
        y = ComplexF32.(exp.(im .* π .* (2f0 .* rand(Xoshiro(200 + d), Float32, 2, 3) .- 1f0)))
        cost = SimilarityCost(y)
        gref, _ = ep_gradient(StaticEP(β=0.005f0, T_free=200, T_nudge=100,
                                       dt=0.5f0, centered=true), ch, ps, st, x, cost)
        for (car, lab) in ARMS
            g = lockin(ch, ps, st, x, cost; kw..., δ=δ, jitter=jitter, seed=99, carrier=car)
            push!(res[lab][1], cosine(g[:layer_1], gref.layer_1.weight))
            push!(res[lab][2], cosine(g[:layer_2], gref.layer_2.weight))
        end
    end
    for (_, lab) in ARMS
        a, b = res[lab]
        @printf("   %-40s  %+.4f / %+.4f   (min %+.3f / %+.3f)\n",
                lab, median(a), median(b), minimum(a), minimum(b))
    end
    return res
end

function main()
    println("Readout-frame check: does the carrier reduction reach the quantizer?")
    println("====================================================================")
    part1()
    part2(δ = 0f0)          # control: analog readout, frames must agree
    part2(δ = DELTA)        # the result
    part2(δ = DELTA, jitter = 0.25f0 * DELTA)   # vs. the dither optimum
    println("""
    Reading these: with δ = 0 every arm agrees, which is Theorem (carrier
    reduction) holding. With δ > 0 the incommensurate lab arms recover almost
    completely while the commensurate one (ω=2π at dt=0.5 is exactly 50 bins)
    collapses onto the co-rotating arm — so the effect is commensurability, not
    the lab frame being privileged. Whether a real device is commensurate is a
    hardware question; see docs/ep_rotating_followups.md §0.""")
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main()
