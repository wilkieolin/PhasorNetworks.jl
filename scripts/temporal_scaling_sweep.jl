#!/usr/bin/env julia
#
# scripts/temporal_scaling_sweep.jl
#   Scaling study for stacked PhasorLSA / PhasorLCA blocks, aimed at building
#   confidence for scaling up an AUDIO-classification network.
#
# Why a new task (and not cross-position MQAR)
# --------------------------------------------
# PhasorLSA / PhasorLCA are *pointwise in the sequence axis L*: their attention
# scores are computed per (l, b) slice (head×head for LSA, anchor×key for LCA)
# and mix values only WITHIN a position. No layer routes information ACROSS
# sequence positions. The only cross-timestep transport is the causal λ-conv
# SSM memory inside the q/k/v/FFN PhasorDense projections. So "multi-hop where
# depth = #position-to-position hops" is architecturally unsupported here, and
# cross-position MQAR only ever exercised λ-transport + a per-position readout.
#
# The audio-relevant capability is exactly what the λ memory *does* provide:
# extract evidence that is spread across many timesteps, integrate/retain it,
# and transform it into a decision at readout. This script builds a task that
# structurally requires that, and uses it as the substrate for the scaling
# knobs (depth, FFN presence/width, depth-vs-width).
#
# Task: Temporal Integration Recall (TIR)
# ---------------------------------------
# A length-L sequence of D-dim phasors. One target value v* ∈ 1:n_vals (the
# class — the "growing set" difficulty knob) is planted at `m_signal` evidence
# timesteps, each corrupted by phase noise. `n_distract` other timesteps each
# carry a DIFFERENT, one-off random value (also noisy). A CONSTANT cue phasor
# sits at the read position L and never contains v*. Readout = similarity of the
# post-stack representation at L to the value codebook → predict v*.
#
# Because v* is the only value that recurs coherently (m copies) while every
# distractor appears once, NO single frame separates signal from distractor:
# the readout MUST accumulate across timesteps. That is the cross-timestep
# integration we need to demonstrate for audio. Difficulty knobs: n_vals,
# m_signal, n_distract, noise, sig_max_frac (how far the evidence sits from the
# query → stresses the λ memory / hippo tape), L.
#
# Experiments
# -----------
#   calibrate_tir   Exp A — find a non-saturating regime (depth-1 acc has headroom)
#   exp_depth_tir   Exp B — depth as a scaling knob (does stacking BUY accuracy?)
#   exp_ffn_tir     Exp C — FFN role: off vs widths {D/2,D,2D,4D}, × depth
#   exp_width_tir   Exp D — depth vs width at matched parameter count
#   exp_integration_tir  Exp E — is integration real? acc vs m_signal, and
#                          long-range (sig_max_frac) × λ-init (hippo vs default)
#   exp_depth_mqar  contrast — depth on the existing routing MQAR (reuses infra)
#
# Reuses xform_init_ablation.jl (train infra, gen_mqar_*, build_model, write_csv,
# onehot_dev, boost_alpha_lr!, similarity_outer, cdev, …) via include().
#
# Usage
# -----
#   julia --project=. -e 'include("scripts/temporal_scaling_sweep.jl"); smoke()'          # CPU sanity
#   julia --project=. -e 'include("scripts/temporal_scaling_sweep.jl"); calibrate_tir()'  # Exp A
#   julia --project=. -e 'include("scripts/temporal_scaling_sweep.jl"); run_all()'        # full study (GPU)

include(joinpath(@__DIR__, "xform_init_ablation.jl"))

using Statistics: mean, std
using Random: Xoshiro, AbstractRNG, randperm

const _OUT = joinpath(@__DIR__, "..", "results", "temporal_scaling")

# wrap a Float32 phase back into [-1, 1] (units of π) after additive noise.
_wrap(x) = mod.(x .+ 1f0, 2f0) .- 1f0

# ---------------------------------------------------------------------
# Incremental, resumable CSV I/O — each trial is appended the moment it
# finishes, so a killed/timed-out run always leaves partial results on disk
# and can be resumed (skip rows whose key tuple is already present).
# ---------------------------------------------------------------------

"Append one NamedTuple row to `file` (writing the header if the file is new)."
function append_row(file::String, row::NamedTuple)
    isdir(dirname(file)) || mkpath(dirname(file))
    newfile = !isfile(file) || filesize(file) == 0
    open(file, "a") do io
        newfile && println(io, join(String.(keys(row)), ","))
        println(io, join((string(getproperty(row, k)) for k in keys(row)), ","))
    end
end

"Set of key tuples already recorded in `file` (by the named `keyfields`)."
function done_keys(file::String, keyfields::Tuple)
    done = Set{Tuple}()
    isfile(file) || return done
    lines = readlines(file)
    length(lines) <= 1 && return done
    hdr = split(lines[1], ',')
    idx = [findfirst(==(String(k)), hdr) for k in keyfields]
    any(isnothing, idx) && return done
    for ln in lines[2:end]
        f = split(ln, ',')
        length(f) < maximum(idx) && continue
        push!(done, Tuple(String(f[i]) for i in idx))
    end
    return done
end

# key tuple as strings (so it matches CSV round-trip)
_key(vals...) = Tuple(string(v) for v in vals)

# =====================================================================
# Task: Temporal Integration Recall (TIR)
# =====================================================================

"Value codebook (classes) + a constant read cue phasor."
function setup_tir_task(rng; D, n_vals)
    Vfloat = 2f0 .* rand(rng, Float32, D, n_vals) .- 1f0
    cue    = 2f0 .* rand(rng, Float32, D) .- 1f0
    return Vfloat, cue
end

"""
    gen_tir_batch(rng, Vfloat, cue; D, L, n_vals, B, m_signal, n_distract, noise, sig_max_frac)

One TIR batch. Target value v* planted (noisily) at `m_signal` positions,
`n_distract` one-off random values at other positions, constant `cue` at L.
Signal positions are drawn from `1 : round(sig_max_frac*(L-1))` (set < 1 to push
the evidence far from the query, stressing the λ memory). Returns
`(x, tgt, qpos)`; readout is at `qpos = L`.
"""
function gen_tir_batch(rng::AbstractRNG, Vfloat::Matrix{Float32}, cue::Vector{Float32};
                       D::Int, L::Int, n_vals::Int, B::Int,
                       m_signal::Int, n_distract::Int, noise::Float32,
                       sig_max_frac::Float64 = 1.0)
    qpos = L
    sig_hi = max(m_signal, round(Int, sig_max_frac * (L - 1)))
    @assert m_signal + n_distract ≤ L - 1 "too many tokens for L=$L"
    @assert m_signal ≤ sig_hi ≤ L - 1

    X   = zeros(Float32, D, L, B)
    tgt = Vector{Int}(undef, B)

    for b in 1:B
        vstar = rand(rng, 1:n_vals); tgt[b] = vstar
        # signal positions from the early window; distractors from the rest.
        sig_pos = randperm(rng, sig_hi)[1:m_signal]
        taken   = Set(sig_pos)
        dis_pos = Int[]
        while length(dis_pos) < n_distract
            p = rand(rng, 1:L-1)
            (p in taken) && continue
            push!(dis_pos, p); push!(taken, p)
        end
        for p in sig_pos
            X[:, p, b] = _wrap(@view(Vfloat[:, vstar]) .+ noise .* (2f0 .* rand(rng, Float32, D) .- 1f0))
        end
        for p in dis_pos
            u = rand(rng, 1:n_vals)
            X[:, p, b] = _wrap(@view(Vfloat[:, u]) .+ noise .* (2f0 .* rand(rng, Float32, D) .- 1f0))
        end
        X[:, qpos, b] = cue
    end
    return (x = Phase.(X), tgt = tgt, qpos = qpos)
end

function gen_tir_batches(rng, Vfloat, cue; D, L, n_vals, B, m_signal, n_distract,
                         noise, sig_max_frac, n_batches)
    return [gen_tir_batch(rng, Vfloat, cue; D, L, n_vals, B, m_signal, n_distract,
                          noise, sig_max_frac) for _ in 1:n_batches]
end

# =====================================================================
# Model: stack of blocks with FFN on/off + width control
# =====================================================================

"""
    build_stack(; D, n_heads, n_blocks, attn_kind, n_anchors, attn_mode, ffn,
                d_ff, ffn_mode, gate, alpha0) -> Chain

`n_blocks` residual blocks over (D,L,B) Phase input. `ffn = :on` uses a full
PhasorTransformerBlock (attention + FFN residual); `ffn = :off` uses a bare
`PhasorResidual(attn)` (attention-only block, no FFN) — this is exactly the
attn_res half of the transformer block, so the FFN ablation needs no src change.
`d_ff` sets the FFN hidden width (expansion ratio) when `ffn = :on`.
"""
function build_stack(; D::Int, n_heads::Int, n_blocks::Int,
                     attn_kind::Symbol = :lsa, n_anchors::Int = 8,
                     attn_mode::Symbol = :default, ffn::Symbol = :on,
                     d_ff::Int = D, ffn_mode::Symbol = :hippo,
                     ffn_n_modes::Int = 1,
                     ffn_hippo_tau_max::Union{Real, Nothing} = nothing,
                     ffn_hippo_tau_min::Union{Real, Nothing} = nothing,
                     full_d_heads::Bool = false, input_embed::Bool = false,
                     gate::Symbol = :rezero, alpha0::Float32 = 0.1f0)
    mkattn() = attn_kind === :lca ?
        PhasorLCA(D => D, n_heads, n_anchors; init_mode = attn_mode) :
        PhasorLSA(D => D, n_heads; init_mode = attn_mode, full_d_heads = full_d_heads)
    blocks = ntuple(n_blocks) do _
        if ffn === :off
            PhasorResidual(mkattn(); gate = gate, alpha0 = alpha0)
        else
            PhasorTransformerBlock(D, mkattn(); d_ff = d_ff, ffn_init_mode = ffn_mode,
                                   ffn_n_modes = ffn_n_modes,
                                   ffn_hippo_tau_max = ffn_hippo_tau_max,
                                   ffn_hippo_tau_min = ffn_hippo_tau_min,
                                   gate = gate, alpha0 = alpha0)
        end
    end
    # Optional leading input-embedding PhasorDense (uniform λ + bias), mirroring
    # the phasor_torch / audio pipeline which TIR normally lacks.
    if input_embed
        emb = PhasorDense(D => D, normalize_to_unit_circle; use_bias = true, init_mode = :default)
        return Chain(emb, blocks...)
    end
    return Chain(blocks...)
end

nparams(model) = Lux.parameterlength(model)

# =====================================================================
# TIR readout / loss / eval / train
# =====================================================================

# Readout. pool_frac == 0 → single-position (read at qpos, the original TIR
# readout). pool_frac > 0 → average the per-position similarity over the last
# `round(pool_frac·L)` positions, mirroring the audio SSMReadout pooling — the
# temporal integration the linchpin showed makes the FFN redundant.
function tir_readout(x, model, ps, st, qpos, Vc; pool_frac::Float64 = 0.0)
    y, _ = model(x, ps, st)                        # (D, L, B) Phase
    if pool_frac <= 0
        return similarity_outer(angle_to_complex(y[:, qpos, :]), Vc)   # (n_vals, B)
    end
    L = size(y, 2)
    W = max(1, round(Int, pool_frac * L))
    t0 = L - W + 1
    sims = map(t0:L) do t
        similarity_outer(angle_to_complex(y[:, t, :]), Vc)
    end
    return sum(sims) ./ Float32(length(sims))       # (n_vals, B)
end

tir_loss(x, model, ps, st, qpos, Vc, yoh; pool_frac::Float64 = 0.0) =
    mean(evaluate_loss(tir_readout(x, model, ps, st, qpos, Vc; pool_frac), yoh, :similarity))

function tir_eval(model, ps, st, batches, Vc, n_vals, dev; pool_frac::Float64 = 0.0)
    c = 0; tot = 0
    for bt in batches
        s = tir_readout(bt.x |> dev, model, ps, st, bt.qpos, Vc; pool_frac)
        p = predict(cdev(s), :similarity)
        c += sum(p .== bt.tgt); tot += length(bt.tgt)
    end
    return c / tot
end

function tir_train!(model, ps, st, batches, Vc, n_vals, epochs, lr, dev; alpha_lr_mult = 5f0,
                    pool_frac::Float64 = 0.0, rho::Float64 = 0.9)
    opt = Optimisers.setup(Optimisers.RMSProp(Float32(lr), Float32(rho)), ps)
    alpha_lr_mult != 1 && boost_alpha_lr!(opt, ps, Float32(lr) * Float32(alpha_lr_mult))
    losses = Float32[]
    for _ in 1:epochs
        el = 0f0; nb = 0
        for bt in batches
            xd  = bt.x |> dev
            yoh = onehot_dev(bt.tgt, n_vals, dev)
            l, back = Zygote.pullback(p -> tir_loss(xd, model, p, st, bt.qpos, Vc, yoh; pool_frac), ps)
            g = back(one(l))[1]
            opt, ps = Optimisers.update(opt, ps, g)
            el += Float32(l); nb += 1
        end
        push!(losses, el / nb)
    end
    return ps, losses
end

# One train+eval trial. Returns a NamedTuple row (caller adds extra fields).
function tir_trial(; D, L, n_vals, n_heads, n_blocks, attn_kind, attn_mode, ffn,
                   d_ff, ffn_mode, m_signal, n_distract, noise, sig_max_frac,
                   B, n_train_batches, n_eval_batches, epochs, lr, seed, dev,
                   n_anchors::Int = 8, ffn_n_modes::Int = 1,
                   ffn_hippo_tau_max::Union{Real, Nothing} = nothing,
                   ffn_hippo_tau_min::Union{Real, Nothing} = nothing,
                   full_d_heads::Bool = false, input_embed::Bool = false,
                   pool_frac::Float64 = 0.0, rho::Float64 = 0.9)
    Vfloat, cue = setup_tir_task(Xoshiro(777); D, n_vals)
    Vc = angle_to_complex(Phase.(Vfloat)) |> dev
    train_b = gen_tir_batches(Xoshiro(seed),   Vfloat, cue; D, L, n_vals, B, m_signal,
                              n_distract, noise, sig_max_frac, n_batches = n_train_batches)
    eval_b  = gen_tir_batches(Xoshiro(9_999),  Vfloat, cue; D, L, n_vals, B, m_signal,
                              n_distract, noise, sig_max_frac, n_batches = n_eval_batches)

    model = build_stack(; D, n_heads, n_blocks, attn_kind, attn_mode, ffn, d_ff, ffn_mode,
                        n_anchors, ffn_n_modes, ffn_hippo_tau_max, ffn_hippo_tau_min, full_d_heads,
                        input_embed)
    ps, st = Lux.setup(Xoshiro(seed + 1), model)
    ps = ps |> dev; st = st |> dev
    np = nparams(model)

    ps, losses = tir_train!(model, ps, st, train_b, Vc, n_vals, epochs, lr, dev; pool_frac, rho)
    acc = tir_eval(model, ps, st, eval_b, Vc, n_vals, dev; pool_frac)
    GC.gc(); dev !== cdev && CUDA.reclaim()
    return (; acc = Float32(acc), n_params = np,
            init_loss = losses[1], final_loss = losses[end])
end

_dev(use_cuda) = (use_cuda && CUDA.functional()) ? gpu_device() : cdev

# =====================================================================
# Exp A — difficulty calibration (depth 1, 1 seed): find headroom
# =====================================================================

"""
    calibrate_tir(; ...) -> rows

Sweep the difficulty knobs at depth 1 to locate a regime where a single block is
well below ceiling (target acc ≈ 0.4–0.6) so that added depth/width has room to
help. Writes `<outdir>/calibrate.csv`.
"""
function calibrate_tir(; D::Int = 64, L::Int = 48, n_heads::Int = 4,
                       n_vals_list = (8, 16, 32),
                       m_signal_list = (2, 4),
                       n_distract_list = (8, 16),
                       noise_list = (0.15f0, 0.30f0),
                       sig_max_frac::Float64 = 1.0,
                       B::Int = 64, n_train_batches::Int = 30, n_eval_batches::Int = 10,
                       epochs::Int = 20, lr::Real = 1f-3,
                       attn_kind::Symbol = :lsa, use_cuda::Bool = true,
                       outdir::String = _OUT)
    dev = _dev(use_cuda); mkpath(outdir)
    @info "TIR calibration" device=string(dev) D L attn_kind
    rows = NamedTuple[]
    for n_vals in n_vals_list, m_signal in m_signal_list,
        n_distract in n_distract_list, noise in noise_list
        r = tir_trial(; D, L, n_vals, n_heads, n_blocks = 1, attn_kind,
                      attn_mode = :default, ffn = :on, d_ff = D, ffn_mode = :hippo,
                      m_signal, n_distract, noise, sig_max_frac, B,
                      n_train_batches, n_eval_batches, epochs, lr, seed = 1, dev)
        row = (; n_vals, m_signal, n_distract, noise, chance = 1f0 / n_vals,
               acc = r.acc, n_params = r.n_params, final_loss = r.final_loss)
        push!(rows, row)
        @info "calib" n_vals m_signal n_distract noise acc=round(r.acc, digits=3) chance=round(1/n_vals, digits=3)
    end
    write_csv(joinpath(outdir, "calibrate.csv"), rows)
    return rows
end

# A single "hard" TIR config chosen from calibration (edit after Exp A).
# Placeholder until calibrate_tir reports; kept here so the other drivers share it.
# Chosen from calibrate_tir / the shrunk-scale probe: depth-1 ≈ 0.43 (chance
# 0.0625) at the shrunk scale (D=48, L=32), i.e. clear headroom for depth/width
# to show an effect. cfg A (nv=12) saturated a single block (~0.78, depth-flat);
# cfg B (this one) leaves room.
const HARD = (n_vals = 16, m_signal = 3, n_distract = 16, noise = 0.35f0, L = 32, sig_max_frac = 1.0)

# =====================================================================
# Exp B — depth as a scaling knob
# =====================================================================

function exp_depth_tir(; D::Int = 48, n_heads::Int = 4, depths = (1, 2, 3, 4),
                       seeds = 1:2, attn_kinds = (:lsa, :lca),
                       B::Int = 48, n_train_batches::Int = 16, n_eval_batches::Int = 8,
                       epochs::Int = 40, lr::Real = 1f-3, use_cuda::Bool = true,
                       hard = HARD, outdir::String = _OUT)
    dev = _dev(use_cuda); mkpath(outdir)
    file = joinpath(outdir, "depth.csv")
    done = done_keys(file, (:attn_kind, :depth, :seed))
    @info "TIR depth-as-knob" device=string(dev) depths=collect(depths) seeds=collect(seeds) hard done=length(done)
    for attn_kind in attn_kinds, depth in depths, seed in seeds
        _key(attn_kind, depth, seed) in done && (@info "skip (done)" attn_kind depth seed; continue)
        r = tir_trial(; D, L = hard.L, n_vals = hard.n_vals, n_heads, n_blocks = depth,
                      attn_kind, attn_mode = :default, ffn = :on, d_ff = D, ffn_mode = :hippo,
                      m_signal = hard.m_signal, n_distract = hard.n_distract,
                      noise = hard.noise, sig_max_frac = hard.sig_max_frac, B,
                      n_train_batches, n_eval_batches, epochs, lr, seed, dev)
        append_row(file, (; attn_kind, depth, seed, acc = r.acc, n_params = r.n_params,
                          final_loss = r.final_loss))
        @info "depth" attn_kind depth seed acc=round(r.acc, digits=3) n_params=r.n_params
    end
    @info "depth.csv complete" file
    return file
end

# =====================================================================
# Exp C — FFN role: off vs widths, crossed with depth
# =====================================================================

function exp_ffn_tir(; D::Int = 48, n_heads::Int = 4, depths = (2, 4),
                     ffn_specs = ((:off, 0), (:on, D ÷ 2), (:on, D), (:on, 2D)),
                     seeds = 1:2, attn_kind::Symbol = :lsa,
                     B::Int = 48, n_train_batches::Int = 16, n_eval_batches::Int = 8,
                     epochs::Int = 40, lr::Real = 1f-3, use_cuda::Bool = true,
                     hard = HARD, outdir::String = _OUT)
    dev = _dev(use_cuda); mkpath(outdir)
    file = joinpath(outdir, "ffn.csv")
    done = done_keys(file, (:depth, :ffn, :d_ff, :seed))
    @info "TIR FFN role" device=string(dev) depths=collect(depths) ffn_specs done=length(done)
    for depth in depths, (ffn, d_ff) in ffn_specs, seed in seeds
        dff = ffn === :off ? 0 : d_ff
        _key(depth, ffn, dff, seed) in done && (@info "skip (done)" depth ffn dff seed; continue)
        r = tir_trial(; D, L = hard.L, n_vals = hard.n_vals, n_heads, n_blocks = depth,
                      attn_kind, attn_mode = :default, ffn, d_ff = max(dff, 1), ffn_mode = :hippo,
                      m_signal = hard.m_signal, n_distract = hard.n_distract,
                      noise = hard.noise, sig_max_frac = hard.sig_max_frac, B,
                      n_train_batches, n_eval_batches, epochs, lr, seed, dev)
        append_row(file, (; depth, ffn = String(ffn), d_ff = dff, seed,
                          acc = r.acc, n_params = r.n_params, final_loss = r.final_loss))
        @info "ffn" depth ffn d_ff=dff seed acc=round(r.acc, digits=3) n_params=r.n_params
    end
    @info "ffn.csv complete" file
    return file
end

# =====================================================================
# Exp D — depth vs width at matched parameter count
# =====================================================================

"Pick D values whose depth-2 stack param count ≈ each depth-arm target."
function _match_widths(depth_targets::Vector{Int}; n_heads, attn_kind, base_D = 64,
                       ffn = :on, ffn_mode = :hippo)
    # candidate widths (multiples of n_heads); measure params of a depth-2 stack.
    cand = filter(d -> d % n_heads == 0, 48:4:320)
    pcount = Dict(d => nparams(build_stack(; D = d, n_heads, n_blocks = 2, attn_kind,
                                           ffn, d_ff = d, ffn_mode)) for d in cand)
    Ds = Int[]
    for tgt in depth_targets
        best = argmin(d -> abs(pcount[d] - tgt), cand)
        push!(Ds, best)
    end
    return Ds, pcount
end

function exp_width_tir(; base_D::Int = 48, n_heads::Int = 4,
                       depth_arm = (1, 2, 3, 4), seeds = 1:2, attn_kind::Symbol = :lsa,
                       B::Int = 48, n_train_batches::Int = 16, n_eval_batches::Int = 8,
                       epochs::Int = 40, lr::Real = 1f-3, use_cuda::Bool = true,
                       hard = HARD, outdir::String = _OUT)
    dev = _dev(use_cuda); mkpath(outdir)
    file = joinpath(outdir, "width.csv")
    done = done_keys(file, (:arm, :knob, :seed))
    # depth-arm param targets (at base_D), then matched widths at depth 2.
    depth_params = [nparams(build_stack(; D = base_D, n_heads, n_blocks = d, attn_kind,
                                        ffn = :on, d_ff = base_D, ffn_mode = :hippo))
                    for d in depth_arm]
    width_Ds, _ = _match_widths(depth_params; n_heads, attn_kind, base_D)
    @info "TIR depth-vs-width" device=string(dev) depth_arm=collect(depth_arm) depth_params width_Ds done=length(done)

    for (d, _t) in zip(depth_arm, depth_params), seed in seeds         # depth arm
        _key("depth", d, seed) in done && continue
        r = tir_trial(; D = base_D, L = hard.L, n_vals = hard.n_vals, n_heads,
                      n_blocks = d, attn_kind, attn_mode = :default, ffn = :on, d_ff = base_D,
                      ffn_mode = :hippo, m_signal = hard.m_signal, n_distract = hard.n_distract,
                      noise = hard.noise, sig_max_frac = hard.sig_max_frac, B,
                      n_train_batches, n_eval_batches, epochs, lr, seed, dev)
        append_row(file, (; arm = "depth", knob = d, D = base_D, depth = d, seed,
                          acc = r.acc, n_params = r.n_params, final_loss = r.final_loss))
        @info "width-study(depth)" depth=d seed acc=round(r.acc, digits=3) n_params=r.n_params
    end
    for (Dw, _d) in zip(width_Ds, depth_arm), seed in seeds            # width arm (depth 2)
        _key("width", Dw, seed) in done && continue
        r = tir_trial(; D = Dw, L = hard.L, n_vals = hard.n_vals, n_heads,
                      n_blocks = 2, attn_kind, attn_mode = :default, ffn = :on, d_ff = Dw,
                      ffn_mode = :hippo, m_signal = hard.m_signal, n_distract = hard.n_distract,
                      noise = hard.noise, sig_max_frac = hard.sig_max_frac, B,
                      n_train_batches, n_eval_batches, epochs, lr, seed, dev)
        append_row(file, (; arm = "width", knob = Dw, D = Dw, depth = 2, seed,
                          acc = r.acc, n_params = r.n_params, final_loss = r.final_loss))
        @info "width-study(width)" D=Dw seed acc=round(r.acc, digits=3) n_params=r.n_params
    end
    @info "width.csv complete" file
    return file
end

# =====================================================================
# Exp E — integration reality: acc vs #evidence, and long-range × λ-init
# =====================================================================

function exp_integration_tir(; D::Int = 48, n_heads::Int = 4, n_blocks::Int = 3,
                             m_list = (1, 2, 4, 8), seeds = 1:2, attn_kind::Symbol = :lsa,
                             B::Int = 48, n_train_batches::Int = 16, n_eval_batches::Int = 8,
                             epochs::Int = 40, lr::Real = 1f-3, use_cuda::Bool = true,
                             hard = HARD, outdir::String = _OUT)
    dev = _dev(use_cuda); mkpath(outdir)
    file = joinpath(outdir, "integration.csv")
    done = done_keys(file, (:probe, :m_signal, :sig_max_frac, :ffn_mode, :seed))
    @info "TIR integration reality" device=string(dev) m_list depth=n_blocks done=length(done)
    # (i) accumulation: more coherent evidence frames → higher accuracy.
    for m_signal in m_list, seed in seeds
        _key("accumulate", m_signal, hard.sig_max_frac, "hippo", seed) in done && continue
        r = tir_trial(; D, L = hard.L, n_vals = hard.n_vals, n_heads, n_blocks,
                      attn_kind, attn_mode = :default, ffn = :on, d_ff = D, ffn_mode = :hippo,
                      m_signal, n_distract = hard.n_distract, noise = hard.noise,
                      sig_max_frac = hard.sig_max_frac, B, n_train_batches, n_eval_batches,
                      epochs, lr, seed, dev)
        append_row(file, (; probe = "accumulate", m_signal, sig_max_frac = hard.sig_max_frac,
                          ffn_mode = "hippo", seed, acc = r.acc, final_loss = r.final_loss))
        @info "accumulate" m_signal seed acc=round(r.acc, digits=3)
    end
    # (ii) long-range: evidence pushed to the front (far from the query) — does a
    #      long λ tape (hippo) beat a single-timescale (default) FFN?
    for frac in (1.0, 0.34), ffn_mode in (:hippo, :default), seed in seeds
        _key("longrange", hard.m_signal, frac, String(ffn_mode), seed) in done && continue
        r = tir_trial(; D, L = hard.L, n_vals = hard.n_vals, n_heads, n_blocks,
                      attn_kind, attn_mode = :default, ffn = :on, d_ff = D, ffn_mode,
                      m_signal = hard.m_signal, n_distract = hard.n_distract, noise = hard.noise,
                      sig_max_frac = frac, B, n_train_batches, n_eval_batches, epochs, lr, seed, dev)
        append_row(file, (; probe = "longrange", m_signal = hard.m_signal, sig_max_frac = frac,
                          ffn_mode = String(ffn_mode), seed, acc = r.acc, final_loss = r.final_loss))
        @info "longrange" frac ffn_mode seed acc=round(r.acc, digits=3)
    end
    @info "integration.csv complete" file
    return file
end

# =====================================================================
# Contrast — depth on the existing routing MQAR (reuses xform infra)
# =====================================================================

function exp_depth_mqar(; D::Int = 64, L::Int = 48, n_heads::Int = 4,
                        depths = (1, 2, 4, 8), seeds = 1:3, attn_kind::Symbol = :lsa,
                        n_keys::Int = 16, n_vals::Int = 16, near_gap::Int = 4,
                        density::Float64 = 0.3, B::Int = 64,
                        n_train_batches::Int = 40, n_eval_batches::Int = 12,
                        epochs::Int = 30, lr::Real = 1f-3, use_cuda::Bool = true,
                        outdir::String = _OUT)
    dev = _dev(use_cuda); mkpath(outdir)
    Kfloat, Vfloat = setup_mqar_task(Xoshiro(777); D, n_keys, n_vals)
    Vc = angle_to_complex(Phase.(Vfloat)) |> dev
    eval_b = gen_mqar_batches(Xoshiro(9999), Kfloat, Vfloat;
                              D, L, n_keys, n_vals, B, near_gap, density, n_batches = n_eval_batches)
    file = joinpath(outdir, "depth_mqar.csv")
    done = done_keys(file, (:depth, :seed))
    @info "MQAR depth contrast" device=string(dev) depths=collect(depths) n_keys n_vals density done=length(done)
    for depth in depths, seed in seeds
        _key(depth, seed) in done && continue
        train_b = gen_mqar_batches(Xoshiro(seed), Kfloat, Vfloat;
                                   D, L, n_keys, n_vals, B, near_gap, density, n_batches = n_train_batches)
        model = build_model(; D, n_heads, n_blocks = depth, attn_mode = :default,
                            ffn_mode = :hippo, attn_kind = (attn_kind === :lca ? :lca : :lsa),
                            recenter = false)
        ps, st = Lux.setup(Xoshiro(seed + 1), model); ps = ps |> dev; st = st |> dev
        ps, losses = train!(model, ps, st, train_b, Vc, n_vals, epochs, lr, dev)
        far, near = eval_accuracy(model, ps, st, eval_b, Vc, n_vals, dev)
        append_row(file, (; depth, seed, far_acc = Float32(far), near_acc = Float32(near),
                          n_params = nparams(model), final_loss = losses[end]))
        @info "mqar-depth" depth seed far=round(far, digits=3) near=round(near, digits=3)
        GC.gc(); dev !== cdev && CUDA.reclaim()
    end
    @info "depth_mqar.csv complete" file
    return file
end

# =====================================================================
# Knob 4 — capacity: width D × FFN expansion d_ff
# =====================================================================

function exp_capacity_tir(; Ds = (48, 64, 96, 128), ff_mults = (1, 2, 4),
                          n_heads::Int = 4, n_blocks::Int = 2, seeds = 1:2,
                          attn_kind::Symbol = :lsa, B::Int = 48,
                          n_train_batches::Int = 16, n_eval_batches::Int = 8,
                          epochs::Int = 40, lr::Real = 1f-3, use_cuda::Bool = true,
                          input_embed::Bool = false, pool_frac::Float64 = 0.0,
                          hard = HARD, outdir::String = _OUT)
    dev = _dev(use_cuda); mkpath(outdir)
    file = joinpath(outdir, "capacity.csv")
    done = done_keys(file, (:D, :d_ff, :seed))
    @info "TIR capacity (width × d_ff)" device=string(dev) Ds ff_mults done=length(done)
    for Dw in Ds, mult in ff_mults, seed in seeds
        dff = Dw * mult
        _key(Dw, dff, seed) in done && continue
        r = tir_trial(; D = Dw, L = hard.L, n_vals = hard.n_vals, n_heads, n_blocks,
                      attn_kind, attn_mode = :default, ffn = :on, d_ff = dff, ffn_mode = :hippo,
                      m_signal = hard.m_signal, n_distract = hard.n_distract,
                      noise = hard.noise, sig_max_frac = hard.sig_max_frac, B,
                      n_train_batches, n_eval_batches, epochs, lr, seed, dev, input_embed, pool_frac)
        append_row(file, (; D = Dw, d_ff = dff, ff_mult = mult, seed,
                          acc = r.acc, n_params = r.n_params, final_loss = r.final_loss))
        @info "capacity" D=Dw d_ff=dff seed acc=round(r.acc, digits=3) n_params=r.n_params
    end
    @info "capacity.csv complete" file
    return file
end

# =====================================================================
# Knob 5 — LCA associative-memory size (n_anchors)
# =====================================================================

function exp_anchors_tir(; D::Int = 48, n_heads::Int = 4, n_blocks::Int = 2,
                         anchors = (4, 8, 16, 32, 64), seeds = 1:2, B::Int = 48,
                         n_train_batches::Int = 16, n_eval_batches::Int = 8,
                         epochs::Int = 40, lr::Real = 1f-3, use_cuda::Bool = true,
                         input_embed::Bool = false, pool_frac::Float64 = 0.0,
                         hard = HARD, outdir::String = _OUT)
    dev = _dev(use_cuda); mkpath(outdir)
    file = joinpath(outdir, "anchors.csv")
    done = done_keys(file, (:n_anchors, :seed))
    @info "TIR anchors (LCA memory size)" device=string(dev) anchors done=length(done)
    for na in anchors, seed in seeds
        _key(na, seed) in done && continue
        r = tir_trial(; D, L = hard.L, n_vals = hard.n_vals, n_heads, n_blocks,
                      attn_kind = :lca, n_anchors = na, attn_mode = :default,
                      ffn = :on, d_ff = D, ffn_mode = :hippo,
                      m_signal = hard.m_signal, n_distract = hard.n_distract,
                      noise = hard.noise, sig_max_frac = hard.sig_max_frac, B,
                      n_train_batches, n_eval_batches, epochs, lr, seed, dev, input_embed, pool_frac)
        append_row(file, (; n_anchors = na, seed, acc = r.acc, n_params = r.n_params,
                          final_loss = r.final_loss))
        @info "anchors" n_anchors=na seed acc=round(r.acc, digits=3) n_params=r.n_params
    end
    @info "anchors.csv complete" file
    return file
end

# =====================================================================
# Knob 3 — FFN λ-timescale range (hippo_tau_max), on the long-range probe
# =====================================================================

function exp_tau_tir(; D::Int = 48, n_heads::Int = 4, n_blocks::Int = 3,
                     tau_maxes = (16f0, 64f0, 256f0, 1024f0), fracs = (1.0, 0.34),
                     seeds = 1:2, attn_kind::Symbol = :lsa, B::Int = 48,
                     n_train_batches::Int = 16, n_eval_batches::Int = 8,
                     epochs::Int = 40, lr::Real = 1f-3, use_cuda::Bool = true,
                     input_embed::Bool = false, pool_frac::Float64 = 0.0,
                     hard = HARD, outdir::String = _OUT)
    dev = _dev(use_cuda); mkpath(outdir)
    file = joinpath(outdir, "tau.csv")
    done = done_keys(file, (:tau_max, :sig_max_frac, :seed))
    @info "TIR λ-range (hippo_tau_max)" device=string(dev) tau_maxes fracs done=length(done)
    for tmax in tau_maxes, frac in fracs, seed in seeds
        _key(tmax, frac, seed) in done && continue
        r = tir_trial(; D, L = hard.L, n_vals = hard.n_vals, n_heads, n_blocks,
                      attn_kind, attn_mode = :default, ffn = :on, d_ff = D, ffn_mode = :hippo,
                      ffn_hippo_tau_max = tmax,
                      m_signal = hard.m_signal, n_distract = hard.n_distract,
                      noise = hard.noise, sig_max_frac = frac, B,
                      n_train_batches, n_eval_batches, epochs, lr, seed, dev, input_embed, pool_frac)
        append_row(file, (; tau_max = tmax, sig_max_frac = frac, seed,
                          acc = r.acc, n_params = r.n_params, final_loss = r.final_loss))
        @info "tau" tau_max=tmax frac seed acc=round(r.acc, digits=3)
    end
    @info "tau.csv complete" file
    return file
end

# =====================================================================
# Knob 1 — SSM state expansion: FFN modes per channel (ffn_n_modes)
# =====================================================================

function exp_modes_tir(; D::Int = 48, n_heads::Int = 4, n_blocks::Int = 3,
                       modes = (1, 2, 4, 8), fracs = (1.0, 0.34), seeds = 1:2,
                       attn_kind::Symbol = :lsa, ffn_hippo_tau_max = 256f0, B::Int = 48,
                       n_train_batches::Int = 16, n_eval_batches::Int = 8,
                       epochs::Int = 40, lr::Real = 1f-3, use_cuda::Bool = true,
                       input_embed::Bool = false, pool_frac::Float64 = 0.0,
                       hard = HARD, outdir::String = _OUT)
    dev = _dev(use_cuda); mkpath(outdir)
    file = joinpath(outdir, "modes.csv")
    done = done_keys(file, (:n_modes, :sig_max_frac, :seed))
    @info "TIR modes-per-channel (SSM state expansion)" device=string(dev) modes fracs done=length(done)
    for M in modes, frac in fracs, seed in seeds
        _key(M, frac, seed) in done && continue
        r = tir_trial(; D, L = hard.L, n_vals = hard.n_vals, n_heads, n_blocks,
                      attn_kind, attn_mode = :default, ffn = :on, d_ff = D, ffn_mode = :hippo,
                      ffn_n_modes = M, ffn_hippo_tau_max = ffn_hippo_tau_max,
                      m_signal = hard.m_signal, n_distract = hard.n_distract,
                      noise = hard.noise, sig_max_frac = frac, B,
                      n_train_batches, n_eval_batches, epochs, lr, seed, dev, input_embed, pool_frac)
        append_row(file, (; n_modes = M, sig_max_frac = frac, seed,
                          acc = r.acc, n_params = r.n_params, final_loss = r.final_loss))
        @info "modes" n_modes=M frac seed acc=round(r.acc, digits=3) n_params=r.n_params
    end
    @info "modes.csv complete" file
    return file
end

# =====================================================================
# Head-count probe — does slicing the D-symbol into H heads of Dh=D/H
# degrade the VSA similarity (holographic-distribution hypothesis)?
# Prediction under "slicing limits fidelity": acc FALLS as H grows / Dh shrinks.
# =====================================================================

function exp_heads_tir(; D::Int = 48, n_blocks::Int = 3,
                       heads = (1, 2, 4, 8, 16), fracs = (0.34, 1.0),
                       kinds = (:lsa, :lca), seeds = 1:2, B::Int = 48,
                       n_train_batches::Int = 16, n_eval_batches::Int = 8,
                       epochs::Int = 40, lr::Real = 1f-3, use_cuda::Bool = true,
                       hard = HARD, outdir::String = _OUT)
    dev = _dev(use_cuda); mkpath(outdir)
    file = joinpath(outdir, "heads.csv")
    done = done_keys(file, (:attn_kind, :n_heads, :sig_max_frac, :seed))
    @info "TIR head-count sweep (Dh=D/H)" device=string(dev) D heads fracs kinds done=length(done)
    for kind in kinds, H in heads, frac in fracs, seed in seeds
        D % H == 0 || (@warn "skip: D not divisible by H" D H; continue)
        _key(kind, H, frac, seed) in done && continue
        r = tir_trial(; D, L = hard.L, n_vals = hard.n_vals, n_heads = H, n_blocks,
                      attn_kind = kind, attn_mode = :default, ffn = :on, d_ff = D, ffn_mode = :hippo,
                      m_signal = hard.m_signal, n_distract = hard.n_distract,
                      noise = hard.noise, sig_max_frac = frac, B,
                      n_train_batches, n_eval_batches, epochs, lr, seed, dev)
        append_row(file, (; attn_kind = kind, n_heads = H, Dh = D ÷ H, sig_max_frac = frac,
                          seed, acc = r.acc, n_params = r.n_params, final_loss = r.final_loss))
        @info "heads" kind n_heads=H Dh=D÷H frac seed acc=round(r.acc, digits=3)
    end
    @info "heads.csv complete" file
    return file
end

# =====================================================================
# Direct test — full-D heads (each head sees the whole symbol, D→D·H, bundled)
# vs Dh-slice heads, both DIRECTLY (same D) and at MATCHED params (sliced widened).
# Tests the "don't slice the holographic symbol" hypothesis controlling for params.
# =====================================================================

function exp_fulldhead_tir(; base_D::Int = 48, n_heads::Int = 4, n_blocks::Int = 3,
                           fracs = (0.34, 1.0), seeds = 1:2, B::Int = 48,
                           n_train_batches::Int = 16, n_eval_batches::Int = 8,
                           epochs::Int = 40, lr::Real = 1f-3, use_cuda::Bool = true,
                           hard = HARD, outdir::String = _OUT)
    dev = _dev(use_cuda); mkpath(outdir)
    file = joinpath(outdir, "fulldhead.csv")
    done = done_keys(file, (:variant, :D, :sig_max_frac, :seed))

    # params of the full-D-head model at base_D → matched sliced width.
    p_full = nparams(build_stack(; D = base_D, n_heads, n_blocks, attn_kind = :lsa,
                                 ffn = :on, d_ff = base_D, ffn_mode = :hippo, full_d_heads = true))
    cand = filter(d -> d % n_heads == 0, base_D:4:4*base_D)
    D_match = argmin(d -> abs(nparams(build_stack(; D = d, n_heads, n_blocks, attn_kind = :lsa,
                              ffn = :on, d_ff = d, ffn_mode = :hippo)) - p_full), cand)
    configs = [(:sliced, base_D, false), (:full_d, base_D, true), (:sliced_matched, D_match, false)]
    @info "TIR full-D-head test" device=string(dev) base_D n_heads p_full D_match done=length(done)

    for (variant, Dw, fdh) in configs, frac in fracs, seed in seeds
        _key(variant, Dw, frac, seed) in done && continue
        r = tir_trial(; D = Dw, L = hard.L, n_vals = hard.n_vals, n_heads, n_blocks,
                      attn_kind = :lsa, attn_mode = :default, ffn = :on, d_ff = Dw, ffn_mode = :hippo,
                      full_d_heads = fdh, m_signal = hard.m_signal, n_distract = hard.n_distract,
                      noise = hard.noise, sig_max_frac = frac, B,
                      n_train_batches, n_eval_batches, epochs, lr, seed, dev)
        append_row(file, (; variant = String(variant), D = Dw, sig_max_frac = frac, seed,
                          acc = r.acc, n_params = r.n_params, final_loss = r.final_loss))
        @info "fulldhead" variant D=Dw frac seed acc=round(r.acc, digits=3) n_params=r.n_params
    end
    @info "fulldhead.csv complete" file
    return file
end

# =====================================================================
# Tier-1 readout ablation — contrastive softmax-CE + learnable codes +
# logsumexp-over-time pooling (+ learnable temperature). Understands which
# readout upgrade shifts the needle / the knob disparities most.
# Uses the smooth complex-similarity path (no complex_to_angle at readout).
# =====================================================================

const _PI32 = Float32(pi)

# sims (n_vals, B) from model output y (D,L,B) Phase and codes (D,n_vals) Float32-phase.
function _ablation_sims(y, codes; readout_pool::Symbol, pool_frac::Float64, lse_kappa::Float32 = 10f0)
    Vc = cis.(_PI32 .* codes)                                  # (D, n_vals) ComplexF32 (differentiable)
    L = size(y, 2)
    if readout_pool === :single
        return similarity_outer(angle_to_complex(y[:, L, :]), Vc)          # (n_vals, B)
    elseif readout_pool === :mean
        W = max(1, round(Int, pool_frac * L)); t0 = L - W + 1
        s = map(t -> similarity_outer(angle_to_complex(y[:, t, :]), Vc), t0:L)
        return sum(s) ./ Float32(length(s))
    elseif readout_pool === :logsumexp
        s = map(t -> similarity_outer(angle_to_complex(y[:, t, :]), Vc), 1:L)
        sumexp = sum(map(st -> exp.(lse_kappa .* st), s))                   # (n_vals, B)
        return log.(sumexp ./ Float32(L)) ./ lse_kappa                     # smooth max over time
    else
        error("unknown readout_pool :$readout_pool")
    end
end

function _ablation_loss(sims, yoh; loss_type::Symbol, beta)
    if loss_type === :similarity
        return mean(evaluate_loss(sims, yoh, :similarity))
    else  # :softmax_ce — contrastive, temperature β
        logits = beta .* sims
        m = maximum(logits, dims = 1)
        logZ = m .+ log.(sum(exp.(logits .- m), dims = 1))
        return -mean(sum(yoh .* (logits .- logZ), dims = 1))
    end
end

"""
    ablation_trial(...) -> acc

One TIR train+eval with a configurable Tier-1 readout: `readout_pool`
(:single/:mean/:logsumexp), `loss_type` (:similarity/:softmax_ce), and optional
learnable `codes` / temperature `beta` (jointly optimized with the model).
"""
function ablation_trial(; D = 48, L = 32, n_vals = 16, n_heads = 4, n_blocks = 2,
                        attn_kind = :lca, n_anchors = 8, ffn = :on, d_ff = 48, ffn_mode = :hippo,
                        ffn_n_modes = 1, ffn_hippo_tau_max = nothing,
                        m_signal = 3, n_distract = 16, noise = 0.35f0, sig_max_frac = 1.0,
                        B = 48, n_train_batches = 16, n_eval_batches = 8, epochs = 40,
                        lr = 1f-3, rho = 0.9, seed = 1, dev = cdev, input_embed = true,
                        readout_pool = :mean, pool_frac = 0.25, loss_type = :softmax_ce,
                        beta0 = 8f0, learn_codes = false, learn_beta = false)
    Vfloat, cue = setup_tir_task(Xoshiro(777); D, n_vals)
    train_b = gen_tir_batches(Xoshiro(seed),  Vfloat, cue; D, L, n_vals, B, m_signal, n_distract, noise, sig_max_frac, n_batches = n_train_batches)
    eval_b  = gen_tir_batches(Xoshiro(9_999), Vfloat, cue; D, L, n_vals, B, m_signal, n_distract, noise, sig_max_frac, n_batches = n_eval_batches)
    model = build_stack(; D, n_heads, n_blocks, attn_kind, attn_mode = :default, ffn, d_ff, ffn_mode,
                        n_anchors, ffn_n_modes, ffn_hippo_tau_max, input_embed)
    ps, st = Lux.setup(Xoshiro(seed + 1), model); ps = ps |> dev; st = st |> dev
    codes0 = (Float32.(Vfloat)) |> dev                        # (D, n_vals)

    P = (; model = ps)
    learn_codes && (P = merge(P, (codes = codes0,)))
    (learn_beta && loss_type === :softmax_ce) && (P = merge(P, (logbeta = [log(Float32(beta0))] |> dev,)))

    opt = Optimisers.setup(Optimisers.RMSProp(Float32(lr), Float32(rho)), P)
    boost_alpha_lr!(opt, P, Float32(lr) * 5f0)

    lossfn(p, xd, yoh) = begin
        cds = haskey(p, :codes) ? p.codes : codes0
        # exp.(logbeta) (broadcast, not [1] scalar-index) — scalar-indexing a GPU
        # array is disallowed; a length-1 array broadcasts cleanly against sims.
        bta = haskey(p, :logbeta) ? exp.(p.logbeta) : Float32(beta0)
        y, _ = model(xd, p.model, st)
        _ablation_loss(_ablation_sims(y, cds; readout_pool, pool_frac), yoh; loss_type, beta = bta)
    end
    for _ in 1:epochs, bt in train_b
        xd = bt.x |> dev; yoh = onehot_dev(bt.tgt, n_vals, dev)
        l, back = Zygote.pullback(p -> lossfn(p, xd, yoh), P)
        g = back(one(l))[1]
        opt, P = Optimisers.update(opt, P, g)
    end

    cds = haskey(P, :codes) ? P.codes : codes0
    c = 0; tot = 0
    for bt in eval_b
        y, _ = model(bt.x |> dev, P.model, st)
        sims = _ablation_sims(y, cds; readout_pool, pool_frac)
        p = predict(cdev(sims), :similarity)
        c += sum(p .== bt.tgt); tot += length(bt.tgt)
    end
    GC.gc(); dev !== cdev && CUDA.reclaim()
    return c / tot
end

"""
    exp_readout_ladder(; ...) -> file

Cumulative Tier-1 readout ladder × the two biggest knob disparities (FFN on/off
at spread; modes m1→m2 at long-range), on LCA + input_embed. Shows which readout
upgrade most improves accuracy and how it shifts each disparity.
"""
function exp_readout_ladder(; use_cuda::Bool = true, seeds = 1:2,
                            outdir::String = joinpath(_OUT, "readout_ladder"))
    dev = _dev(use_cuda); mkpath(outdir)
    file = joinpath(outdir, "ladder.csv")
    done = done_keys(file, (:rung, :probe, :arm, :seed))
    h = HARD
    rungs = [  # (name, readout_pool, loss_type, learn_codes, learn_beta)
        ("R0_meanpool_simloss", :mean,      :similarity, false, false),
        ("R1_softmaxCE",        :mean,      :softmax_ce, false, false),
        ("R2_learncodes",       :mean,      :softmax_ce, true,  false),
        ("R3_learnbeta",        :mean,      :softmax_ce, true,  true),
        ("R4_logsumexp",        :logsumexp, :softmax_ce, true,  true),
    ]
    probes = [  # (probe, arm, per-arm overrides)
        ("ffn",   "on",  (; ffn = :on,  n_blocks = 2, sig_max_frac = 1.0,  ffn_n_modes = 1)),
        ("ffn",   "off", (; ffn = :off, n_blocks = 2, sig_max_frac = 1.0,  ffn_n_modes = 1)),
        ("modes", "m1",  (; ffn = :on,  n_blocks = 3, sig_max_frac = 0.34, ffn_n_modes = 1, ffn_hippo_tau_max = 256f0)),
        ("modes", "m2",  (; ffn = :on,  n_blocks = 3, sig_max_frac = 0.34, ffn_n_modes = 2, ffn_hippo_tau_max = 256f0)),
    ]
    @info "readout ladder" device=string(dev) rungs=[r[1] for r in rungs] done=length(done)
    for (rname, pool, loss, lc, lb) in rungs, (probe, arm, kw) in probes, seed in seeds
        _key(rname, probe, arm, seed) in done && continue
        acc = ablation_trial(; D = 48, L = h.L, n_vals = h.n_vals, n_heads = 4, attn_kind = :lca,
                             n_anchors = 8, d_ff = 48, ffn_mode = :hippo, m_signal = h.m_signal,
                             n_distract = h.n_distract, noise = h.noise, B = 48,
                             n_train_batches = 16, n_eval_batches = 8, epochs = 40, lr = 1f-3,
                             rho = 0.9, seed = seed, dev = dev, input_embed = true,
                             readout_pool = pool, pool_frac = 0.25, loss_type = loss,
                             beta0 = 8f0, learn_codes = lc, learn_beta = lb, kw...)
        append_row(file, (; rung = rname, probe, arm, seed, acc = Float32(acc)))
        @info "ladder" rung=rname probe arm seed acc=round(acc, digits=3)
    end
    @info "ladder complete" file
    return file
end

# =====================================================================
# Reporting helper + orchestrators
# =====================================================================

"Print mean±std acc grouped by (kind, knobvals) for a rows vector with :acc."
function summarize(rows, knob::Symbol, kinds, knobvals, outdir, title)
    println("\n===== $title (mean±std acc) =====")
    print(rpad("kind", 8)); for k in knobvals; print(rpad(string(knob, "=", k), 14)); end; println()
    for kd in kinds
        print(rpad(string(kd), 8))
        for kv in knobvals
            rs = filter(r -> get(r, :attn_kind, kd) == kd && getproperty(r, knob) == kv, rows)
            if isempty(rs); print(rpad("-", 14)); continue; end
            a = getproperty.(rs, :acc)
            print(rpad(@sprintf("%.3f±%.3f", mean(a), length(a) > 1 ? std(a) : 0f0), 14))
        end
        println()
    end
    println(repeat("=", 40), "\n")
end

# Minimal CSV reader → Vector{Dict{String,String}} (values kept as strings).
function _read_csv(file)
    isfile(file) || return Dict{String,String}[]
    lines = readlines(file); length(lines) <= 1 && return Dict{String,String}[]
    hdr = String.(split(lines[1], ','))
    rows = Dict{String,String}[]
    for ln in lines[2:end]
        f = String.(split(ln, ','))
        length(f) == length(hdr) || continue
        push!(rows, Dict(hdr[i] => f[i] for i in eachindex(hdr)))
    end
    return rows
end
_fnum(x) = parse(Float32, x)

"Group `rows` by `by` (vector of col names), aggregate mean±std of `val` col."
function _agg(rows, by::Vector{String}, val::String)
    groups = Dict{Tuple,Vector{Float32}}()
    for r in rows
        all(haskey(r, b) for b in by) || continue
        k = Tuple(r[b] for b in by)
        push!(get!(groups, k, Float32[]), _fnum(r[val]))
    end
    return sort([(k, mean(v), length(v) > 1 ? std(v) : 0f0, length(v)) for (k, v) in groups])
end

"Print mean±std tables for every experiment CSV present under `outdir`."
function summarize_all(; outdir::String = _OUT)
    println("\n########## TEMPORAL SCALING — SUMMARY ##########  ($outdir)")
    d = _read_csv(joinpath(outdir, "depth.csv"))
    if !isempty(d)
        println("\n--- Exp B: depth as a scaling knob (acc mean±std[n]) ---")
        for (k, m, s, n) in _agg(d, ["attn_kind", "depth"], "acc")
            println(@sprintf("  %-4s depth=%s  acc=%.3f±%.3f [n=%d]", k[1], k[2], m, s, n))
        end
    end
    f = _read_csv(joinpath(outdir, "ffn.csv"))
    if !isempty(f)
        println("\n--- Exp C: FFN role (acc mean±std[n]) ---")
        for (k, m, s, n) in _agg(f, ["depth", "ffn", "d_ff"], "acc")
            println(@sprintf("  depth=%s ffn=%-3s d_ff=%-3s  acc=%.3f±%.3f [n=%d]", k[1], k[2], k[3], m, s, n))
        end
    end
    w = _read_csv(joinpath(outdir, "width.csv"))
    if !isempty(w)
        println("\n--- Exp D: depth vs width (acc mean±std, params) ---")
        for (k, m, s, n) in _agg(w, ["arm", "knob", "n_params"], "acc")
            println(@sprintf("  arm=%-5s knob=%-3s params=%-6s  acc=%.3f±%.3f [n=%d]", k[1], k[2], k[3], m, s, n))
        end
    end
    ig = _read_csv(joinpath(outdir, "integration.csv"))
    if !isempty(ig)
        acc = filter(r -> r["probe"] == "accumulate", ig)
        lr  = filter(r -> r["probe"] == "longrange", ig)
        if !isempty(acc)
            println("\n--- Exp E(i): accumulation — acc vs #evidence frames ---")
            for (k, m, s, n) in _agg(acc, ["m_signal"], "acc")
                println(@sprintf("  m_signal=%-2s  acc=%.3f±%.3f [n=%d]", k[1], m, s, n))
            end
        end
        if !isempty(lr)
            println("\n--- Exp E(ii): long-range — sig_max_frac × ffn_mode ---")
            for (k, m, s, n) in _agg(lr, ["sig_max_frac", "ffn_mode"], "acc")
                println(@sprintf("  frac=%-4s ffn=%-7s  acc=%.3f±%.3f [n=%d]", k[1], k[2], m, s, n))
            end
        end
    end
    mq = _read_csv(joinpath(outdir, "depth_mqar.csv"))
    if !isempty(mq)
        println("\n--- Contrast: MQAR routing, far-acc vs depth ---")
        for (k, m, s, n) in _agg(mq, ["depth"], "far_acc")
            println(@sprintf("  depth=%s  far=%.3f±%.3f [n=%d]", k[1], m, s, n))
        end
    end
    # --- knob studies ---
    cap = _read_csv(joinpath(outdir, "capacity.csv"))
    if !isempty(cap)
        println("\n--- Knob 4: capacity (width D × d_ff) — acc, params ---")
        for (k, m, s, n) in _agg(cap, ["D", "d_ff", "n_params"], "acc")
            println(@sprintf("  D=%-3s d_ff=%-3s params=%-6s  acc=%.3f±%.3f [n=%d]", k[1], k[2], k[3], m, s, n))
        end
    end
    an = _read_csv(joinpath(outdir, "anchors.csv"))
    if !isempty(an)
        println("\n--- Knob 5: LCA n_anchors — acc, params ---")
        for (k, m, s, n) in _agg(an, ["n_anchors", "n_params"], "acc")
            println(@sprintf("  n_anchors=%-3s params=%-6s  acc=%.3f±%.3f [n=%d]", k[1], k[2], m, s, n))
        end
    end
    ta = _read_csv(joinpath(outdir, "tau.csv"))
    if !isempty(ta)
        println("\n--- Knob 3: FFN hippo_tau_max × evidence location — acc ---")
        for (k, m, s, n) in _agg(ta, ["sig_max_frac", "tau_max"], "acc")
            println(@sprintf("  frac=%-4s tau_max=%-6s  acc=%.3f±%.3f [n=%d]", k[1], k[2], m, s, n))
        end
    end
    mo = _read_csv(joinpath(outdir, "modes.csv"))
    if !isempty(mo)
        println("\n--- Knob 1: FFN modes/channel × evidence location — acc, params ---")
        for (k, m, s, n) in _agg(mo, ["sig_max_frac", "n_modes", "n_params"], "acc")
            println(@sprintf("  frac=%-4s n_modes=%-2s params=%-6s  acc=%.3f±%.3f [n=%d]", k[1], k[2], k[3], m, s, n))
        end
    end
    println("\n################################################\n")
end

"Fast CPU smoke — validates task, model (FFN on/off), train loop, all drivers."
function smoke(; use_cuda::Bool = false)
    dev = _dev(use_cuda)
    @info "SMOKE (temporal_scaling)" device=string(dev)
    # tiny TIR trial, FFN on and off
    for ffn in (:on, :off)
        r = tir_trial(; D = 24, L = 16, n_vals = 6, n_heads = 3, n_blocks = 2,
                      attn_kind = :lsa, attn_mode = :default, ffn, d_ff = 24, ffn_mode = :hippo,
                      m_signal = 3, n_distract = 4, noise = 0.2f0, sig_max_frac = 1.0,
                      B = 16, n_train_batches = 4, n_eval_batches = 2, epochs = 2, lr = 1f-3,
                      seed = 1, dev)
        @info "smoke TIR" ffn acc=round(r.acc, digits=3) n_params=r.n_params
    end
    # knob 1 (modes) + knob 3 (tau) path
    rm = tir_trial(; D = 24, L = 16, n_vals = 6, n_heads = 3, n_blocks = 2,
                   attn_kind = :lsa, attn_mode = :default, ffn = :on, d_ff = 24, ffn_mode = :hippo,
                   ffn_n_modes = 4, ffn_hippo_tau_max = 256f0,
                   m_signal = 3, n_distract = 4, noise = 0.2f0, sig_max_frac = 0.34,
                   B = 16, n_train_batches = 4, n_eval_batches = 2, epochs = 2, lr = 1f-3,
                   seed = 1, dev)
    @info "smoke TIR modes=4" acc=round(rm.acc, digits=3) n_params=rm.n_params
    # knob 5 (LCA anchors) path
    ra = tir_trial(; D = 24, L = 16, n_vals = 6, n_heads = 3, n_blocks = 2,
                   attn_kind = :lca, n_anchors = 16, attn_mode = :default, ffn = :on,
                   d_ff = 24, ffn_mode = :hippo, m_signal = 3, n_distract = 4, noise = 0.2f0,
                   sig_max_frac = 1.0, B = 16, n_train_batches = 4, n_eval_batches = 2,
                   epochs = 2, lr = 1f-3, seed = 1, dev)
    @info "smoke TIR lca anchors=16" acc=round(ra.acc, digits=3) n_params=ra.n_params
    @info "smoke ok"
end

"""
    run_all(; use_cuda, hard)

Run the whole study sequentially on GPU (HARD is pre-tuned; run `calibrate_tir`
separately to re-tune). Every driver appends per-trial and resumes, so this is
safe to re-run after an interruption — completed trials are skipped.
"""
function run_all(; use_cuda::Bool = true, hard = HARD)
    exp_depth_tir(; use_cuda, hard)
    exp_ffn_tir(; use_cuda, hard)
    exp_width_tir(; use_cuda, hard)
    exp_integration_tir(; use_cuda, hard)
    exp_depth_mqar(; use_cuda)
    summarize_all()
    @info "temporal_scaling study done" outdir=_OUT
end

"""
    run_knobs(; use_cuda)

Run the four performance-knob studies (4 width/FFN, 5 anchors, 3 λ-range,
1 modes/channel) sequentially, then print the summary. Cheap→expensive order;
all drivers append-per-trial and resume.
"""
function run_knobs(; use_cuda::Bool = true, input_embed::Bool = false,
                   pool_frac::Float64 = 0.0, outdir::String = _OUT)
    exp_capacity_tir(; use_cuda, input_embed, pool_frac, outdir)   # knob 4
    exp_anchors_tir(; use_cuda, input_embed, pool_frac, outdir)    # knob 5
    exp_tau_tir(; use_cuda, input_embed, pool_frac, outdir)        # knob 3
    exp_modes_tir(; use_cuda, input_embed, pool_frac, outdir)      # knob 1
    summarize_all(; outdir)
    @info "knob study done" outdir input_embed pool_frac
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_all()
end
