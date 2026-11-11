#!/usr/bin/env julia
# E9: can the lock-in demodulation reference be recovered locally?
#
# The lock-in update is local except for one thing: every synapse must know
# cos(ω_p t) to demodulate against. This script asks whether a neuron could
# regenerate that reference from what it can already sense -- its own ringing,
# or an ambient population signal -- instead of being handed it on a wire.
#
# The key observation is that the estimator ALREADY computes the per-neuron
# response and discards it. In ep_gradient(::LockinEP, ...) the accumulator
# `Zhat[l] .+= obs[l] .* demod` (src/ep.jl:1791) is a complex Fourier
# coefficient at +ω_p per neuron; `_sum_batch` at src/ep.jl:1832 collapses it to
# a bias gradient. That coefficient is the per-neuron susceptibility χ_i. So the
# reference variants below are post-hoc ROTATIONS of one probed settle:
#
#     g_true  = -2 Re(      Ĥ_ij ) / (T ε)
#     g_ref   = -2 Re( e^{-iθ_i} Ĥ_ij ) / (T ε)
#
# with θ_i the phase of whatever reference neuron i used. No re-simulation.
#
# The loop is forked from scripts/three_factor_rule.jl (itself extracted from
# e4_three_factor_scale.jl) so it cannot drift from the estimator it explains;
# the only changes are keeping the per-neuron accumulators instead of reducing
# them, adding a second demodulator at 2ω_p, and optionally driving a two-tone
# probe.
#
#   JULIA_CUDA_HARD_MEMORY_LIMIT=8GiB julia --project=scripts scripts/e9_local_reference.jl
#
# Env: E9_INPUTS (default 8), E9_C2 (two-tone amplitude, 0.5), E9_PSI (phase, pi/2).

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

_envi(k, d) = parse(Int, get(ENV, k, string(d)))
_envf(k, d) = parse(Float64, get(ENV, k, string(d)))

# ---- identical to e4_three_factor_scale.jl / e4_export_update_pairs.jl ----
const SEED  = 42
const HID   = 256
const DOUT  = 64
const SCALE = 0.4f0
const N_INPUTS = _envi("E9_INPUTS", 8)
const C2  = Float32(_envf("E9_C2", 0.5))
const PSI = Float32(_envf("E9_PSI", pi / 2))
const USE_CUDA = CUDA.functional()

const OUT = joinpath(repo_root, "results", "e9_local_reference"); mkpath(OUT)
const GITREV = try
    rev = strip(read(`git -C $(repo_root) rev-parse --short HEAD`, String))
    d   = read(`git -C $(repo_root) diff HEAD -- src`, String)
    isempty(strip(d)) ? rev : rev * "-d" * string(hash(d), base = 16)[1:8]
catch; "unknown" end

cdev = cpu_device(); gdev = gpu_device(); dev = USE_CUDA ? gdev : cdev

Base.@kwdef struct ProbeCfg
    ε::Float32           = 0.05f0
    ω_p::Float32         = 0.05f0
    n_cycles::Int        = 4
    T_warmup_cycles::Int = 2
    T_free::Int          = 100
    dt::Float32          = 0.1f0
    K_mode::Symbol       = :zero
    project::Symbol      = :hard
    two_tone::Bool       = false
end

"""
Drive the probe and keep everything the estimator normally throws away.

Returns per layer the demodulated per-neuron quantities at ω_p and 2ω_p, and per
parameter the demodulated Hebbian Ĥ. The four field proxies are all per-neuron
SCALAR time series, so each is demodulated per neuron here and summed over a
pool afterwards -- pooling is linear, so no proxy needs its own pass.
"""
function probe_run(m::ProbeCfg, chain, ps, st, x, cost)
    s_free = phasor_settle(chain, ps, st, x, cost, 0f0;
                           T = m.T_free, dt = m.dt, K_mode = m.K_mode, project = m.project)
    t_free_end = Float32(m.T_free * m.dt)
    h_dc = chain_hebbians(chain, ps, st, x, s_free)

    layer_keys = collect(keys(ps))
    z0 = _phase_input_to_complex(x)
    period_steps = round(Int, 2π / (m.ω_p * m.dt))
    T_warmup = m.T_warmup_cycles * period_steps
    T_lockin = m.n_cycles * period_steps

    states = [copy(s) for s in s_free]
    zfree  = [copy(s) for s in s_free]
    cache  = _weight_cache(chain, ps, layer_keys)
    drive0 = _input_drive(chain, ps, st, layer_keys, z0; cache = cache)

    # β(t). The two-tone form breaks the half-period antisymmetry of the pure
    # cosine, which is what makes the sign of a neuron's response locally
    # observable at all -- for a pure tone, negating the drive is exactly a
    # half-period time shift, so no time-invariant observer can tell them apart.
    βof(t) = m.two_tone ?
        m.ε * (cos(m.ω_p * t * m.dt) + C2 * cos(2 * m.ω_p * t * m.dt + PSI)) :
        m.ε *  cos(m.ω_p * t * m.dt)

    for t in 1:T_warmup
        states = _phasor_step(chain, ps, st, layer_keys, z0, cost, βof(t), m.dt, states;
                              K_mode = m.K_mode, drive0 = drive0, cache = cache,
                              carriers = nothing,
                              t_now = t_free_end + Float32((t - 1) * m.dt),
                              project = m.project, soft_ε = 0.1f0)
    end
    t_warm_end = t_free_end + Float32(T_warmup * m.dt)

    nL = length(layer_keys)
    zc(l)  = gpu_zeros(states[l], ComplexF32, size(states[l])...)
    Zhat  = [zc(l) for l in 1:nL]      # per-neuron state response, = χ_i
    Zhat2 = [zc(l) for l in 1:nL]      # ... at 2ω_p
    Disp  = [zc(l) for l in 1:nL]      # Im(z_i conj(z_i^free))       signed displacement
    Pop   = [zc(l) for l in 1:nL]      # Re(z_i)                      population phasor sum
    Rect  = [zc(l) for l in 1:nL]      # σ((|g_i|-θ)/s)               rectified drive
    Enrg  = [zc(l) for l in 1:nL]      # Re(conj(z_i) g_i)            layer energy
    EW = Dict{Symbol,Any}(); Eb = Dict{Symbol,Any}()
    for (l, key) in enumerate(layer_keys)
        haskey(ps[key], :weight) || continue
        EW[key] = gpu_zeros(ps[key].weight, ComplexF32, size(ps[key].weight)...)
        haskey(ps[key], :bias_real) && (Eb[key] = zc(l))
    end

    # Rectification threshold for the rate proxy: the median drive magnitude at
    # the free equilibrium. Inside the settle every state is on the unit circle
    # (_project_damp normalises), so there is no amplitude to threshold -- the
    # pre-projection drive g = W z_in + b is the only membrane-potential-like
    # quantity, and this is the honest place to put a spike threshold.
    gfree = Vector{Any}(undef, nL)
    for (l, key) in enumerate(layer_keys)
        z_in = (l == 1) ? z0 : zfree[l-1]
        gfree[l] = ps[key].weight * z_in .+
                   ComplexF32.(ps[key].bias_real .+ 1f0im .* ps[key].bias_imag)
    end
    θrect = [Float32(median(abs.(Array(gfree[l])))) for l in 1:nL]
    srect = [max(1f-6, Float32(std(abs.(Array(gfree[l]))))) for l in 1:nL]
    sig(u) = 1f0 / (1f0 + exp(-u))

    # Free-state value of every proxy, for DC subtraction. n_cycles*period_steps
    # is not an exact integer number of periods (1257 vs 1256.64), so Σ_t e^{-iωt}
    # is ~1.5 rather than 0 and the O(1) DC term leaks into the bin. Small, but
    # the real estimator carries the same correction via its `c` accumulator.
    dcZ = [copy(zfree[l]) for l in 1:nL]
    dcP = [ComplexF32.(real.(zfree[l])) for l in 1:nL]
    dcR = [ComplexF32.(sig.((abs.(gfree[l]) .- θrect[l]) ./ srect[l])) for l in 1:nL]
    dcE = [ComplexF32.(real.(conj.(zfree[l]) .* gfree[l])) for l in 1:nL]
    c1 = ComplexF32(0); c2 = ComplexF32(0)
    # largest phase excursion per neuron -- the direct check on linear response
    maxdphi = [gpu_zeros(zfree[l], Float32, size(zfree[l])...) for l in 1:nL]

    for t in 1:T_lockin
        states = _phasor_step(chain, ps, st, layer_keys, z0, cost, βof(t), m.dt, states;
                              K_mode = m.K_mode, drive0 = drive0, cache = cache,
                              carriers = nothing,
                              t_now = t_warm_end + Float32((t - 1) * m.dt),
                              project = m.project, soft_ε = 0.1f0)
        d1 = ComplexF32(exp(-im * m.ω_p * Float32(t) * m.dt))
        d2 = ComplexF32(exp(-2im * m.ω_p * Float32(t) * m.dt))
        c1 += d1; c2 += d2

        for (l, key) in enumerate(layer_keys)
            haskey(ps[key], :weight) || continue
            z_l  = states[l]
            z_in = (l == 1) ? z0 : states[l-1]
            g_l  = ps[key].weight * z_in .+
                   ComplexF32.(ps[key].bias_real .+ 1f0im .* ps[key].bias_imag)

            maxdphi[l] .= max.(maxdphi[l], abs.(angle.(z_l .* conj.(zfree[l]))))
            Zhat[l]  .+= z_l .* d1
            Zhat2[l] .+= z_l .* d2
            Disp[l]  .+= ComplexF32.(imag.(z_l .* conj.(zfree[l]))) .* d1
            Pop[l]   .+= ComplexF32.(real.(z_l)) .* d1
            Rect[l]  .+= ComplexF32.(sig.((abs.(g_l) .- θrect[l]) ./ srect[l])) .* d1
            Enrg[l]  .+= ComplexF32.(real.(conj.(z_l) .* g_l)) .* d1

            EW[key] .+= (z_l * adjoint(z_in) .- h_dc[key].weight) .* d1
            haskey(Eb, key) && (Eb[key] .+= (z_l .-
                ComplexF32.(h_dc[key].bias_real .+ 1f0im .* h_dc[key].bias_imag)) .* d1)
        end
    end

    for l in 1:nL
        Zhat[l]  .-= c1 .* dcZ[l]
        Zhat2[l] .-= c2 .* dcZ[l]
        Pop[l]   .-= c1 .* dcP[l]
        Rect[l]  .-= c1 .* dcR[l]
        Enrg[l]  .-= c1 .* dcE[l]
        # Disp is identically 0 at the free state, so needs no correction
    end
    return (; layer_keys, T_lockin, Zhat, Zhat2, Disp, Pop, Rect, Enrg, EW, Eb, s_free,
            maxdphi = [Array(m) for m in maxdphi])
end

cosine(a, b) = real(dot(vec(a), vec(b)) / (norm(vec(a)) * norm(vec(b)) + 1e-30))

"Gradient for layer `key`, demodulated against a per-output-neuron reference phase θ."
function grad_with_reference(EWk, θ, T_lockin, ε)
    rot = exp.(-1im .* θ)                       # (out,)
    return -2f0 .* real.(rot .* EWk) ./ (Float32(T_lockin) * ε)
end

# =====================================================================
# Driver
# =====================================================================
rng = Xoshiro(SEED)
chain = Chain(PhasorDense(784 => HID, normalize_to_unit_circle, use_bias = true),
              PhasorDense(HID => DOUT, normalize_to_unit_circle, use_bias = true))
ps, st = Lux.setup(rng, chain)
ps = (layer_1 = merge(ps.layer_1, (weight = SCALE .* ps.layer_1.weight,)),
      layer_2 = merge(ps.layer_2, (weight = SCALE .* ps.layer_2.weight,)))
if USE_CUDA
    chain = chain |> gdev; ps = ps |> gdev; st = st |> gdev
end

inputs  = [Phase.(rand(rng, Float32, 784) .* 2 .- 1) for _ in 1:100]
targets = [ComplexF32.(exp.(im .* π .* (rand(rng, Float32, DOUT) .* 2 .- 1))) for _ in 1:100]

const CFG  = ProbeCfg()
const CFG2 = ProbeCfg(two_tone = true)
const M_REF = LockinEP(ε = CFG.ε, ω_p = CFG.ω_p, n_cycles = CFG.n_cycles,
                       T_warmup_cycles = CFG.T_warmup_cycles, T_free = CFG.T_free,
                       dt = CFG.dt, K_mode = CFG.K_mode)
const KS = [1, 8, 32, 128, 0]                     # 0 = whole layer
const PROXIES = [:state, :disp, :pop, :rect, :energy]

neuron_rows = DataFrame(gitrev = String[], input = Int[], layer = Int[], neuron = Int[],
                        chi_abs = Float64[], chi_arg = Float64[],
                        h2_ratio = Float64[], sign_ok = Union{Missing,Bool}[])
sweep_rows  = DataFrame(gitrev = String[], input = Int[], layer = Int[], proxy = String[],
                        K = Int[], cos_to_true = Float64[], sign_agree = Float64[],
                        coherence = Float64[])
summary_rows = DataFrame(gitrev = String[], input = Int[], layer = Int[],
                         cos_sanity = Float64[], settle_resid = Float64[],
                         frac_tiny_chi = Float64[], arg_spread = Float64[],
                         second_harm = Float64[], sign_recovery = Float64[],
                         max_dphi = Float64[])

@printf("=== E9 local reference, %d inputs, ε=%.3g ω_p=%.3g (two-tone c=%.2g ψ=%.2gπ) ===\n",
        N_INPUTS, CFG.ε, CFG.ω_p, C2, PSI / π)

for idx in 1:N_INPUTS
    x = inputs[idx] |> dev
    cost = SimilarityCost(targets[idx] |> dev)

    r1 = probe_run(CFG,  chain, ps, st, x, cost)
    r2 = probe_run(CFG2, chain, ps, st, x, cost)
    gref, _ = ep_gradient(M_REF, chain, ps, st, x, cost)

    # settle residual: one more free step from the free equilibrium
    s1 = phasor_settle(chain, ps, st, x, cost, 0f0; T = 1, dt = CFG.dt,
                       K_mode = CFG.K_mode, project = CFG.project, init = r1.s_free)
    resid = maximum(maximum(abs.(Array(a) .- Array(b))) for (a, b) in zip(s1, r1.s_free))

    for (l, key) in enumerate(r1.layer_keys)
        haskey(ps[key], :weight) || continue
        EWk   = Array(r1.EW[key])
        χ     = Array(r1.Zhat[l])
        χ2    = Array(r1.Zhat2[l])
        n     = length(χ)
        gtrue = -2f0 .* real.(EWk) ./ (Float32(r1.T_lockin) * CFG.ε)

        cos_sanity = cosine(gtrue, Array(gref[key].weight))
        tiny  = count(abs.(χ) .< 0.01 * median(abs.(χ))) / n
        # circular spread of arg χ, folded to the half-line since ±χ are the
        # same reference up to the sign bit we are chasing
        args  = angle.(χ)
        spread = std(mod.(args .+ π/2, π) .- π/2)
        h2    = median(abs.(χ2)) / (median(abs.(χ)) + 1e-30)
        dphi  = median(r1.maxdphi[l])

        # ---- two-tone sign recovery, with the neuron's clock deliberately unknown
        A1 = Array(r2.Zhat[l]); A2 = Array(r2.Zhat2[l])
        srng = Xoshiro(SEED + 1000l + idx)
        τ = rand(srng, Float32, n) .* Float32(2π / CFG.ω_p)          # unknown per-neuron delay
        A1p = A1 .* exp.(-1im .* CFG.ω_p .* τ)
        A2p = A2 .* exp.(-2im .* CFG.ω_p .* τ)
        â   = C2 .* (A1p .^ 2) ./ (A2p .+ 1e-30) .* exp(1im * PSI)
        ok  = sign.(real.(â)) .== sign.(real.(χ))
        sign_rec = count(ok) / n

        for i in 1:n
            push!(neuron_rows, (GITREV, idx, l, i, abs(χ[i]), angle(χ[i]),
                                abs(χ2[i]) / (abs(χ[i]) + 1e-30), ok[i]))
        end

        # ---- neighbourhood sweep over the field proxies
        field = Dict(:state => χ, :disp => Array(r1.Disp[l]), :pop => Array(r1.Pop[l]),
                     :rect => Array(r1.Rect[l]), :energy => Array(r1.Enrg[l]))
        for pname in PROXIES
            f = field[pname]
            # No mean subtraction across neurons: that would force sum(f) = 0,
            # which is the very pooled reference being measured. DC in TIME is
            # already removed by the demodulator and the c-correction above.
            θglob = angle(sum(f))
            coh   = abs(sum(f)) / (sum(abs.(f)) + 1e-30)
            for K in KS
                prng = Xoshiro(SEED + 7919l + 13idx + K)
                θ = Vector{Float32}(undef, n)
                for i in 1:n
                    pool = K == 0 ? (1:n) : (K == 1 ? (i:i) : rand(prng, 1:n, K))
                    θ[i] = angle(sum(@view f[collect(pool)]))
                end
                g = grad_with_reference(EWk, θ, r1.T_lockin, CFG.ε)
                agree = count(cos.(θ .- θglob) .> 0) / n
                push!(sweep_rows, (GITREV, idx, l, string(pname), K,
                                   cosine(g, gtrue), agree, coh))
            end
        end

        push!(summary_rows, (GITREV, idx, l, cos_sanity, Float64(resid), tiny,
                             Float64(spread), Float64(h2), sign_rec, Float64(dphi)))
        @printf("  input %d layer %d: sanity %.5f | tiny χ %.1f%% | arg spread %.2f | 2ω/ω %.3f | max δφ %.4f rad | sign rec %.1f%%\n",
                idx, l, cos_sanity, 100tiny, spread, h2, dphi, 100sign_rec)
    end
end

CSV.write(joinpath(OUT, "neurons_$(GITREV).csv"), neuron_rows)
CSV.write(joinpath(OUT, "sweep_$(GITREV).csv"), sweep_rows)
CSV.write(joinpath(OUT, "summary_$(GITREV).csv"), summary_rows)

println("\n=== VERDICT ===")
@printf("sanity  cos(g_true, ep_gradient) = %.6f (min over layers/inputs)\n",
        minimum(summary_rows.cos_sanity))
@printf("settle residual %.3g | tiny-χ %.1f%% | arg spread %.3f rad | 2ω/ω %.3f | median max δφ %.4f rad\n",
        maximum(summary_rows.settle_resid), 100mean(summary_rows.frac_tiny_chi),
        mean(summary_rows.arg_spread), mean(summary_rows.second_harm),
        mean(summary_rows.max_dphi))
@printf("two-tone sign recovery: %.1f%%\n", 100mean(summary_rows.sign_recovery))
println("\ngradient fidelity vs neighbourhood size (mean over inputs and layers):")
@printf("  %-8s %8s %8s %8s %8s %8s   %s\n", "proxy", "K=1", "K=8", "K=32", "K=128", "K=all", "coherence")
for pname in PROXIES
    s = sweep_rows[sweep_rows.proxy .== string(pname), :]
    vals = [mean(s[s.K .== K, :cos_to_true]) for K in KS]
    @printf("  %-8s %8.3f %8.3f %8.3f %8.3f %8.3f   %.3f\n", pname, vals..., mean(s.coherence))
end
println("\nwrote $OUT")
