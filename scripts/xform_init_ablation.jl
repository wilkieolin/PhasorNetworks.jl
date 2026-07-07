#!/usr/bin/env julia
#
# scripts/xform_init_ablation.jl
#   HiPPO placement ablation for PhasorTransformerBlock: read heads vs memory tape.
#
# Question
# --------
# The block has two families of PhasorDense sublayers, each with a per-channel
# decay spectrum (λ) set by `init_mode`:
#   • QKV attention projections   (PhasorLSA/LCA `init_mode`)   — "read heads"
#   • the residual-stream FFN      (PhasorTransformerBlock `ffn_init_mode`) — "tape"
#
# Hypothesis: the multi-timescale HiPPO basis (short recent detail through a long
# memory tape, now τ ∈ [0.5, 64] steps after the kernels.jl fix) belongs in the
# RESIDUAL STREAM (FFN), while the QKV projections should be single-timescale
# (:default) read heads that just process what is present now.
#
# We run the 2×2 init ablation:
#     A: QKV :hippo   / FFN :default   (current package default)
#     B: QKV :default / FFN :hippo     (proposed)
#     C: QKV :hippo   / FFN :hippo     (all-hippo control)
#     D: QKV :default / FFN :default   (all-uniform control)
# Only the λ INIT differs; λ stays trainable, param counts identical.
#
# λ only shapes dynamics in the 3D SSM path, so the whole experiment runs 3D
# Phase (D, L, B) through the causal-conv path.
#
# Task: mixed near+far delayed-cue recall (isolates the memory tape)
# -------------------------------------------------------------------
# A length-L sequence of D-dim phasors, mostly neutral filler, with:
#   • V_far  (a value from the vocab) at position 1
#   • cue_far  (a FIXED phasor, same every example) at position qfar → read V_far
#   • V_near (a value from the vocab) at position pnear
#   • cue_near (a FIXED phasor) at position qnear = L → read V_near
# Layout is causal and separated so the far readout is uncontaminated:
#   1:V_far … qfar:cue_far (gap≈L) | pnear:V_near … L:cue_near (gap=near_gap)
#
# The cues are CONSTANT and carry no content pointer, so attention cannot
# content-match to locate a value — the value can only reach its readout by
# being carried forward through the per-channel conv memory (λ). That makes λ
# the load-bearing mechanism: the far readout needs the long tape (τ≈64), the
# near readout a fast tap. Readout cleans up against the value vocabulary
# (similarity → class); we report FAR- and NEAR-accuracy separately.
#
# Usage
# -----
#   julia --project=. scripts/xform_init_ablation.jl                       # full run
#   julia --project=. -e 'include("scripts/xform_init_ablation.jl"); main(; smoke=true)'
#
using PhasorNetworks, Lux, Zygote, Optimisers, CUDA, LuxCUDA
using OneHotArrays: onehotbatch
using Random: Xoshiro, AbstractRNG
using Statistics: mean, std, median
using Printf: @sprintf
import PhasorNetworks: angle_to_complex, similarity_outer, HIPPO_TAU_MIN, HIPPO_TAU_MAX

const PLOTS_OK = try
    @eval using Plots
    true
catch err
    @warn "Plots unavailable; figures will be skipped" exception = err
    false
end

const cdev = cpu_device()

# 2×2 design: (QKV init_mode, FFN init_mode).
const CONFIGS = (
    A = (attn = :hippo,   ffn = :default, label = "A: QKV=hippo/FFN=default (current)"),
    B = (attn = :default, ffn = :hippo,   label = "B: QKV=default/FFN=hippo (proposed)"),
    C = (attn = :hippo,   ffn = :hippo,   label = "C: all-hippo"),
    D = (attn = :default, ffn = :default, label = "D: all-uniform"),
)

# ---------------------------------------------------------------------
# Data — mixed near+far associative recall
# ---------------------------------------------------------------------

"Wrap phase floats back into [-1,1]."
_remap(p) = mod.(p .+ 1f0, 2f0) .- 1f0

"Positions for the causal near+far layout (far in the first half, near at the end)."
function _layout(L, near_gap)
    pfar  = 1
    qfar  = L - near_gap - 1     # far cue, kept BEFORE the near value (no contamination)
    pnear = L - near_gap         # near value
    qnear = L                    # near cue
    return pfar, qfar, pnear, qnear
end

"""
    gen_cue_batch(rng, Vfloat, cue_far, cue_near; D, L, n_vocab, B, near_gap) -> NamedTuple

Build one delayed-cue batch. `Vfloat` is the fixed (D, n_vocab) value vocabulary
and `cue_far`/`cue_near` are the two fixed recall-cue phasors (all shared across
batches). Non-value/non-cue positions are neutral filler (phase 0). Returns
`(x::Array{Phase,3} (D,L,B), tgt_far, tgt_near, qfar, qnear)` (1-based targets).
"""
function gen_cue_batch(rng::AbstractRNG, Vfloat::Matrix{Float32},
                       cue_far::Vector{Float32}, cue_near::Vector{Float32};
                       D::Int, L::Int, n_vocab::Int, B::Int, near_gap::Int)
    @assert L > near_gap + 3 "L too short for near_gap"
    pfar, qfar, pnear, qnear = _layout(L, near_gap)

    X = zeros(Float32, D, L, B)              # neutral filler = phase 0 everywhere
    tgt_far  = Vector{Int}(undef, B)
    tgt_near = Vector{Int}(undef, B)

    for b in 1:B
        vf = rand(rng, 1:n_vocab); vn = rand(rng, 1:n_vocab)
        tgt_far[b] = vf; tgt_near[b] = vn
        X[:, pfar,  b] = @view Vfloat[:, vf]
        X[:, pnear, b] = @view Vfloat[:, vn]
        X[:, qfar,  b] = cue_far             # constant cue — no content pointer
        X[:, qnear, b] = cue_near
    end
    return (x = Phase.(X), tgt_far = tgt_far, tgt_near = tgt_near,
            qfar = qfar, qnear = qnear)
end

"Fixed task: value vocabulary + the two constant recall cues (shared train/eval)."
function setup_task(rng; D, n_vocab)
    Vfloat   = 2f0 .* rand(rng, Float32, D, n_vocab) .- 1f0
    cue_far  = 2f0 .* rand(rng, Float32, D) .- 1f0
    cue_near = 2f0 .* rand(rng, Float32, D) .- 1f0
    return Vfloat, cue_far, cue_near
end

"Generate `n_batches` delayed-cue batches for a FIXED task (Vfloat, cues)."
function gen_batches(rng, Vfloat, cue_far, cue_near; D, L, n_vocab, B, near_gap, n_batches)
    return [gen_cue_batch(rng, Vfloat, cue_far, cue_near; D, L, n_vocab, B, near_gap)
            for _ in 1:n_batches]
end

# ---------------------------------------------------------------------
# Data — MQAR along a noisy tape (routing + interference; stresses read heads)
# ---------------------------------------------------------------------
#
# Separate key/value tokens (MQAR-standard): a (key, value) pair is two ADJACENT
# tokens [key@p, value@p+1]. The query presents a bare key; the answer is the
# value that followed it. Two targets (far, near); the rest of the tape is filled
# with random single-token distractors at probability `density` (task-irrelevant
# noise). With full attention the *gap* is largely shortcut, so `density` (SNR) is
# the discriminating knob — it stresses the read heads' ability to extract the
# clean key/value from noise (and the key→value+1 induction).

"Positions for the MQAR layout: far pair at the front, near pair at the end, both queried."
function _mqar_layout(L, near_gap)
    pfar_k, pfar_v = 1, 2                 # far pair (key, value)
    pnear_k = L - near_gap                # near key
    pnear_v = L - near_gap + 1            # near value
    qnear   = L                           # near query key
    qfar    = pnear_k - 2                 # far query key (before the near pair)
    return pfar_k, pfar_v, qfar, pnear_k, pnear_v, qnear
end

"Fixed MQAR task: key alphabet (routing) + value alphabet (readout classes)."
function setup_mqar_task(rng; D, n_keys, n_vals)
    Kfloat = 2f0 .* rand(rng, Float32, D, n_keys) .- 1f0
    Vfloat = 2f0 .* rand(rng, Float32, D, n_vals) .- 1f0
    return Kfloat, Vfloat
end

"""
    gen_mqar_batch(rng, Kfloat, Vfloat; D, L, n_keys, n_vals, B, near_gap, density)

One MQAR batch. Two target pairs (far@1-2, near@L-near_gap..). Non-reserved
positions get a random distractor token with probability `density`, else neutral
filler. Returns `(x, tgt_far, tgt_near, qfar, qnear)` (1-based value targets).
"""
function gen_mqar_batch(rng::AbstractRNG, Kfloat::Matrix{Float32}, Vfloat::Matrix{Float32};
                        D::Int, L::Int, n_keys::Int, n_vals::Int, B::Int,
                        near_gap::Int, density::Float64)
    @assert n_keys ≥ 2 "need ≥2 keys for two distinct targets"
    pfar_k, pfar_v, qfar, pnear_k, pnear_v, qnear = _mqar_layout(L, near_gap)
    reserved = (pfar_k, pfar_v, qfar, pnear_k, pnear_v, qnear)

    X = zeros(Float32, D, L, B)
    tgt_far  = Vector{Int}(undef, B)
    tgt_near = Vector{Int}(undef, B)

    for b in 1:B
        kf = rand(rng, 1:n_keys)
        kn = rand(rng, 1:n_keys); while kn == kf; kn = rand(rng, 1:n_keys); end
        vf = rand(rng, 1:n_vals); vn = rand(rng, 1:n_vals)
        tgt_far[b] = vf; tgt_near[b] = vn

        X[:, pfar_k,  b] = @view Kfloat[:, kf]
        X[:, pfar_v,  b] = @view Vfloat[:, vf]
        X[:, pnear_k, b] = @view Kfloat[:, kn]
        X[:, pnear_v, b] = @view Vfloat[:, vn]
        X[:, qfar,    b] = @view Kfloat[:, kf]      # far query key
        X[:, qnear,   b] = @view Kfloat[:, kn]      # near query key

        for p in 1:L
            p in reserved && continue
            if rand(rng) < density                   # task-irrelevant noise token
                X[:, p, b] = 2f0 .* rand(rng, Float32, D) .- 1f0
            end                                       # else neutral filler (0)
        end
    end
    return (x = Phase.(X), tgt_far = tgt_far, tgt_near = tgt_near,
            qfar = qfar, qnear = qnear)
end

"Generate `n_batches` MQAR batches at a fixed distractor `density`."
function gen_mqar_batches(rng, Kfloat, Vfloat; D, L, n_keys, n_vals, B, near_gap, density, n_batches)
    return [gen_mqar_batch(rng, Kfloat, Vfloat; D, L, n_keys, n_vals, B, near_gap, density)
            for _ in 1:n_batches]
end

# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------

"Stack of `n_blocks` PhasorTransformerBlocks with the given QKV / FFN init modes."
function build_model(; D::Int, n_heads::Int, n_blocks::Int,
                     attn_mode::Symbol, ffn_mode::Symbol, attn_kind::Symbol = :lsa,
                     n_anchors::Int = 8, recenter::Bool = true)
    blocks = ntuple(n_blocks) do _
        attn = attn_kind === :lca ?
            PhasorLCA(D => D, n_heads, n_anchors; init_mode = attn_mode) :
            PhasorLSA(D => D, n_heads; init_mode = attn_mode)
        PhasorTransformerBlock(D, attn; ffn_init_mode = ffn_mode, gate = :rezero, recenter = recenter)
    end
    return Chain(blocks...)
end

# ---------------------------------------------------------------------
# Loss / metrics
# ---------------------------------------------------------------------

"Readout at the two query positions → (sim_far, sim_near) each (n_vocab, B)."
function _readout(x, model, ps, st, qfar, qnear, Vc)
    y, _ = model(x, ps, st)                       # (D, L, B) Phase
    yf = angle_to_complex(y[:, qfar,  :])         # (D, B) complex
    yn = angle_to_complex(y[:, qnear, :])
    sf = similarity_outer(yf, Vc)                 # (n_vocab, B)
    sn = similarity_outer(yn, Vc)
    return sf, sn
end

function recall_loss(x, model, ps, st, qfar, qnear, Vc, yoh_far, yoh_near)
    sf, sn = _readout(x, model, ps, st, qfar, qnear, Vc)
    return mean(evaluate_loss(sf, yoh_far,  :similarity)) +
           mean(evaluate_loss(sn, yoh_near, :similarity))
end

onehot_dev(y, n, dev) = Float32.(onehotbatch(y, 1:n)) |> dev

"Separate FAR / NEAR accuracy over a set of batches."
function eval_accuracy(model, ps, st, batches, Vc, n_vocab, dev)
    cf = 0; cn = 0; tot = 0
    for bt in batches
        sf, sn = _readout(bt.x |> dev, model, ps, st, bt.qfar, bt.qnear, Vc)
        pf = predict(cdev(sf), :similarity); pn = predict(cdev(sn), :similarity)
        cf += sum(pf .== bt.tgt_far); cn += sum(pn .== bt.tgt_near)
        tot += length(bt.tgt_far)
    end
    return cf / tot, cn / tot
end

# ---------------------------------------------------------------------
# Learned-λ spectrum probe (attn read heads vs FFN tape)
# ---------------------------------------------------------------------

"Collect log_neg_lambda leaves split by residual branch: attn (:attn_res) vs ffn (:ffn_res)."
function collect_lnl(ps)
    attn = Float32[]; ffn = Float32[]
    for (kp, v) in Optimisers.trainables(ps, path = true)
        last(kp.keys) === :log_neg_lambda || continue
        # keypath under Chain is (:layer_i, :attn_res|:ffn_res, …) — classify by
        # which residual branch it lives in, regardless of Chain nesting depth.
        dest = (:attn_res in kp.keys) ? attn : ffn
        append!(dest, Float32.(vec(Array(v))))
    end
    return attn, ffn
end

# ---------------------------------------------------------------------
# Gradient-health probe (near-origin complex_to_angle events)
# ---------------------------------------------------------------------

"Run one backward with the _cta_probe armed; return (min|z|, max|dz|, init_loss)."
function grad_health(model, ps, st, bt, Vc, yoh_far, yoh_near, dev)
    xd = bt.x |> dev
    PhasorNetworks._cta_probe[] = Tuple{Float32,Float32}[]
    l, back = Zygote.pullback(p -> recall_loss(xd, model, p, st, bt.qfar, bt.qnear,
                                               Vc, yoh_far, yoh_near), ps)
    back(one(l))
    events = PhasorNetworks._cta_probe[]
    PhasorNetworks._cta_probe[] = nothing
    minz = isempty(events) ? NaN32 : minimum(first.(events))
    maxd = isempty(events) ? NaN32 : maximum(last.(events))
    return (init_loss = Float32(l), min_absz = minz, max_absdz = maxd)
end

# ---------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------

function boost_alpha_lr!(opt_state, ps, lr_alpha)
    for (kp, _) in Optimisers.trainables(ps, path = true)
        if last(kp.keys) === :alpha
            node = opt_state
            for k in kp.keys
                node = k isa Integer ? node[k] : getproperty(node, k)
            end
            Optimisers.adjust!(node, Float32(lr_alpha))
        end
    end
end

function train!(model, ps, st, batches, Vc, n_vocab, epochs, lr, dev; alpha_lr_mult = 5f0)
    opt = Optimisers.setup(Optimisers.RMSProp(Float32(lr)), ps)
    alpha_lr_mult != 1 && boost_alpha_lr!(opt, ps, Float32(lr) * Float32(alpha_lr_mult))
    losses = Float32[]
    for _ in 1:epochs
        el = 0f0; nb = 0
        for bt in batches
            xd = bt.x |> dev
            yof = onehot_dev(bt.tgt_far,  n_vocab, dev)
            yon = onehot_dev(bt.tgt_near, n_vocab, dev)
            l, back = Zygote.pullback(p -> recall_loss(xd, model, p, st, bt.qfar, bt.qnear,
                                                       Vc, yof, yon), ps)
            g = back(one(l))[1]
            opt, ps = Optimisers.update(opt, ps, g)
            el += Float32(l); nb += 1
        end
        push!(losses, el / nb)
    end
    return ps, losses
end

# ---------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------

function main(; D::Int = 64, L::Int = 48, n_heads::Int = 4, n_blocks::Int = 2,
              n_vocab::Int = 16, near_gap::Int = 3, B::Int = 64,
              n_train_batches::Int = 150, n_eval_batches::Int = 40,
              epochs::Int = 25, lr::Real = 1f-3, seeds = 1:3,
              attn_kind::Symbol = :lsa, use_cuda::Bool = true,
              outdir::String = joinpath(@__DIR__, "..", "results", "xform_init"),
              smoke::Bool = false)

    if smoke
        D = 32; L = 16; n_blocks = 1; n_vocab = 8; near_gap = 2; B = 16
        n_train_batches = 6; n_eval_batches = 3; epochs = 2; seeds = 1:1
    end

    dev = (use_cuda && CUDA.functional()) ? gpu_device() : cpu_device()
    @info "device" dev L near_gap gap_far = (L - near_gap - 2) HIPPO_TAU_MAX
    mkpath(outdir)

    # One fixed task (vocab + cues) shared across ALL configs and seeds, and one
    # fixed eval set — so only the model init/λ-placement varies.
    Vfloat, cue_far, cue_near = setup_task(Xoshiro(777); D, n_vocab)
    Vc = angle_to_complex(Phase.(Vfloat)) |> dev
    eval_b = gen_batches(Xoshiro(9999), Vfloat, cue_far, cue_near;
                         D, L, n_vocab, B, near_gap, n_batches = n_eval_batches)

    rows = NamedTuple[]
    lnl_snaps = Dict{Symbol,Any}()          # config => (attn_after, ffn_after) from seed 1

    for (cfg_key, cfg) in pairs(CONFIGS)
        @info "=== config $cfg_key : $(cfg.label) ==="
        for seed in seeds
            train_b = gen_batches(Xoshiro(seed), Vfloat, cue_far, cue_near;
                                  D, L, n_vocab, B, near_gap, n_batches = n_train_batches)

            model = build_model(; D, n_heads, n_blocks, attn_mode = cfg.attn,
                                ffn_mode = cfg.ffn, attn_kind)
            ps, st = Lux.setup(Xoshiro(seed + 1), model)
            ps = ps |> dev; st = st |> dev

            attn0, ffn0 = collect_lnl(cdev(ps))
            yof0 = onehot_dev(train_b[1].tgt_far,  n_vocab, dev)
            yon0 = onehot_dev(train_b[1].tgt_near, n_vocab, dev)
            gh = grad_health(model, ps, st, train_b[1], Vc, yof0, yon0, dev)

            ps, losses = train!(model, ps, st, train_b, Vc, n_vocab, epochs, lr, dev)

            far_acc, near_acc = eval_accuracy(model, ps, st, eval_b, Vc, n_vocab, dev)
            attn1, ffn1 = collect_lnl(cdev(ps))

            @info(@sprintf("  seed %d: far=%.3f near=%.3f  loss %.3f→%.3f  min|z|=%.2e max|dz|=%.1e",
                           seed, far_acc, near_acc, losses[1], losses[end], gh.min_absz, gh.max_absdz))

            push!(rows, (config = String(cfg_key), label = cfg.label, seed = seed,
                         far_acc = far_acc, near_acc = near_acc,
                         init_loss = losses[1], final_loss = losses[end],
                         min_absz = gh.min_absz, max_absdz = gh.max_absdz,
                         attn_tau_med0 = _med_tau(attn0), ffn_tau_med0 = _med_tau(ffn0),
                         attn_tau_med1 = _med_tau(attn1), ffn_tau_med1 = _med_tau(ffn1)))
            if seed == first(seeds)
                lnl_snaps[cfg_key] = (attn0 = attn0, ffn0 = ffn0, attn1 = attn1, ffn1 = ffn1)
            end
        end
    end

    write_csv(joinpath(outdir, "results.csv"), rows)
    summarize(rows)
    PLOTS_OK && make_plots(rows, lnl_snaps, outdir)
    @info "done" outdir
    return rows, lnl_snaps
end

"Median time-constant τ=1/|λ| from a vector of log_neg_lambda."
_med_tau(lnl) = isempty(lnl) ? NaN32 : Float32(median(1f0 ./ exp.(lnl)))

function write_csv(file, rows)
    isempty(rows) && return
    ks = keys(rows[1])
    open(file, "w") do io
        println(io, join(String.(ks), ","))
        for r in rows
            println(io, join((string(getproperty(r, k)) for k in ks), ","))
        end
    end
    @info "wrote $file"
end

function summarize(rows)
    println("\n================ SUMMARY (mean ± std over seeds) ================")
    println("cfg    FAR-acc        NEAR-acc       loss   τ(FFN)med0→1  τ(QKV)med0→1  min|z|")
    for cfg in (:A, :B, :C, :D)
        rs = filter(r -> r.config == String(cfg), rows)
        isempty(rs) && continue
        m(f) = mean(getproperty.(rs, f)); s(f) = length(rs) > 1 ? std(getproperty.(rs, f)) : 0f0
        println(@sprintf("%-6s far=%.3f±%.3f  near=%.3f±%.3f  loss=%.3f  τ_ffn %.1f→%.1f  τ_qkv %.1f→%.1f  min|z|=%.1e",
                         String(cfg), m(:far_acc), s(:far_acc), m(:near_acc), s(:near_acc),
                         m(:final_loss), m(:ffn_tau_med0), m(:ffn_tau_med1),
                         m(:attn_tau_med0), m(:attn_tau_med1), m(:min_absz)))
    end
    println("=================================================================\n")
end

function make_plots(rows, lnl_snaps, outdir)
    cfgs = (:A, :B, :C, :D)
    m(cfg, f) = (rs = filter(r -> r.config == String(cfg), rows); mean(getproperty.(rs, f)))
    s(cfg, f) = (rs = filter(r -> r.config == String(cfg), rows); length(rs) > 1 ? std(getproperty.(rs, f)) : 0f0)

    # (1) FAR vs NEAR accuracy bars per config
    xs = collect(1:length(cfgs))
    far = [m(c, :far_acc) for c in cfgs]; farerr = [s(c, :far_acc) for c in cfgs]
    near = [m(c, :near_acc) for c in cfgs]; nearerr = [s(c, :near_acc) for c in cfgs]
    p1 = bar(xs .- 0.2, far; bar_width = 0.4, yerror = farerr, label = "far (gap≈L)",
             xticks = (xs, String.(collect(cfgs))), ylabel = "accuracy", ylims = (0, 1),
             title = "Recall accuracy by init placement", legend = :topright)
    bar!(p1, xs .+ 0.2, near; bar_width = 0.4, yerror = nearerr, label = "near (gap=3)")
    savefig(p1, joinpath(outdir, "accuracy.png"))

    # (2) learned λ time-constant spectra (seed 1): FFN vs QKV, before/after
    plts = []
    for c in cfgs
        haskey(lnl_snaps, c) || continue
        sn = lnl_snaps[c]
        τ(lnl) = isempty(lnl) ? Float32[] : (1f0 ./ exp.(lnl))
        pp = histogram(log10.(max.(τ(sn.ffn1), 1f-3)); bins = 20, alpha = 0.55,
                       label = "FFN", xlabel = "log10 τ (steps)", title = "cfg $c (trained)")
        histogram!(pp, log10.(max.(τ(sn.attn1), 1f-3)); bins = 20, alpha = 0.55, label = "QKV")
        push!(plts, pp)
    end
    if !isempty(plts)
        p2 = plot(plts...; layout = (2, 2), size = (900, 700))
        savefig(p2, joinpath(outdir, "lambda_spectra.png"))
    end
    @info "wrote plots" dir = outdir
end

# ---------------------------------------------------------------------
# MQAR density-sweep driver (routing / read-head stress)
# ---------------------------------------------------------------------

"""
    main_mqar(; densities, ...) -> rows

Sweep distractor `density` on the MQAR-along-a-noisy-tape task for all four
init configs. Headline: far/near accuracy vs density (expect curves to fan out
if a scheme is more noise-robust for routing). Writes results/xform_mqar/.
"""
function main_mqar(; D::Int = 64, L::Int = 48, n_heads::Int = 4, n_blocks::Int = 2,
                   n_keys::Int = 8, n_vals::Int = 8, near_gap::Int = 4, B::Int = 64,
                   densities = [0.0, 0.25, 0.5, 0.75, 1.0],
                   n_train_batches::Int = 50, n_eval_batches::Int = 15,
                   epochs::Int = 40, lr::Real = 1f-3, seeds = 1:1,
                   attn_kind::Symbol = :lsa, use_cuda::Bool = true,
                   outdir::String = joinpath(@__DIR__, "..", "results", "xform_mqar"),
                   smoke::Bool = false)

    if smoke
        D = 32; L = 20; n_blocks = 1; n_keys = 4; n_vals = 4; near_gap = 3; B = 16
        densities = [0.0, 1.0]; n_train_batches = 6; n_eval_batches = 3; epochs = 2; seeds = 1:1
    end

    dev = (use_cuda && CUDA.functional()) ? gpu_device() : cpu_device()
    @info "device (MQAR)" dev L near_gap densities
    mkpath(outdir)

    Kfloat, Vfloat = setup_mqar_task(Xoshiro(777); D, n_keys, n_vals)
    Vc = angle_to_complex(Phase.(Vfloat)) |> dev

    rows = NamedTuple[]
    for density in densities
        eval_b = gen_mqar_batches(Xoshiro(9999), Kfloat, Vfloat;
                                  D, L, n_keys, n_vals, B, near_gap, density, n_batches = n_eval_batches)
        for (cfg_key, cfg) in pairs(CONFIGS)
            for seed in seeds
                train_b = gen_mqar_batches(Xoshiro(seed), Kfloat, Vfloat;
                                           D, L, n_keys, n_vals, B, near_gap, density, n_batches = n_train_batches)
                model = build_model(; D, n_heads, n_blocks, attn_mode = cfg.attn,
                                    ffn_mode = cfg.ffn, attn_kind)
                ps, st = Lux.setup(Xoshiro(seed + 1), model)
                ps = ps |> dev; st = st |> dev

                ps, losses = train!(model, ps, st, train_b, Vc, n_vals, epochs, lr, dev)
                far_acc, near_acc = eval_accuracy(model, ps, st, eval_b, Vc, n_vals, dev)

                @info(@sprintf("  density %.2f  cfg %s seed %d: far=%.3f near=%.3f  loss %.3f→%.3f",
                               density, String(cfg_key), seed, far_acc, near_acc, losses[1], losses[end]))
                push!(rows, (density = density, config = String(cfg_key), label = cfg.label,
                             seed = seed, far_acc = far_acc, near_acc = near_acc,
                             init_loss = losses[1], final_loss = losses[end]))
            end
        end
    end

    write_csv(joinpath(outdir, "results.csv"), rows)
    summarize_mqar(rows, densities)
    PLOTS_OK && make_mqar_plots(rows, densities, outdir)
    @info "done (MQAR)" outdir
    return rows
end

function summarize_mqar(rows, densities)
    println("\n============ MQAR SUMMARY: far-acc (mean±std) vs density ============")
    print(rpad("cfg", 6)); for d in densities; print(rpad(@sprintf("d=%.2f", d), 14)); end; println()
    for cfg in (:A, :B, :C, :D)
        print(rpad(String(cfg), 6))
        for d in densities
            rs = filter(r -> r.config == String(cfg) && r.density == d, rows)
            if isempty(rs); print(rpad("-", 14)); continue; end
            m = mean(getproperty.(rs, :far_acc)); s = length(rs) > 1 ? std(getproperty.(rs, :far_acc)) : 0f0
            print(rpad(@sprintf("%.2f±%.2f", m, s), 14))
        end
        println()
    end
    println("====================================================================\n")
end

function make_mqar_plots(rows, densities, outdir)
    cfgs = (:A, :B, :C, :D)
    mean_at(cfg, d, f) = (rs = filter(r -> r.config == String(cfg) && r.density == d, rows);
                          isempty(rs) ? NaN : mean(getproperty.(rs, f)))
    p = plot(; xlabel = "distractor density", ylabel = "far-acc", ylims = (0, 1),
             title = "MQAR far-recall vs tape noise", legend = :bottomleft)
    for c in cfgs
        ys = [mean_at(c, d, :far_acc) for d in densities]
        plot!(p, densities, ys; marker = :circle, label = String(c))
    end
    savefig(p, joinpath(outdir, "far_vs_density.png"))
    pn = plot(; xlabel = "distractor density", ylabel = "near-acc", ylims = (0, 1),
              title = "MQAR near-recall vs tape noise", legend = :bottomleft)
    for c in cfgs
        ys = [mean_at(c, d, :near_acc) for d in densities]
        plot!(pn, densities, ys; marker = :circle, label = String(c))
    end
    savefig(pn, joinpath(outdir, "near_vs_density.png"))
    @info "wrote MQAR plots" dir = outdir
end

# ---------------------------------------------------------------------
# PhaseRecenter ablation (usefulness + gradient-blowup source)
# ---------------------------------------------------------------------

"""
    main_recenter(; ...) -> rows

Toggle `recenter ∈ {true,false}` on the MQAR clean task (d=0) for a chosen
config (default B, which ran closest to the origin in earlier probes). Reports
far/near accuracy AND grad-health (min|z|, max|dz| from the `_cta_probe`, which
captures the recenter's own `complex_to_angle` backward) at init and after
training — so we can see if PhaseRecenter helps and whether it is a blow-up
source.
"""
function main_recenter(; D::Int = 64, L::Int = 48, n_heads::Int = 4, n_blocks::Int = 2,
                       n_keys::Int = 8, n_vals::Int = 8, near_gap::Int = 4, B::Int = 64,
                       configs = (:B, :C), n_train_batches::Int = 50, n_eval_batches::Int = 15,
                       epochs::Int = 40, lr::Real = 1f-3, seeds = 1:5,
                       attn_kind::Symbol = :lsa, use_cuda::Bool = true,
                       outdir::String = joinpath(@__DIR__, "..", "results", "xform_recenter"),
                       smoke::Bool = false)

    if smoke
        D = 32; L = 20; n_blocks = 1; n_keys = 4; n_vals = 4; near_gap = 3; B = 16
        configs = (:B,); n_train_batches = 6; n_eval_batches = 3; epochs = 2; seeds = 1:1
    end

    dev = (use_cuda && CUDA.functional()) ? gpu_device() : cpu_device()
    @info "device (recenter)" dev configs
    mkpath(outdir)

    Kfloat, Vfloat = setup_mqar_task(Xoshiro(777); D, n_keys, n_vals)
    Vc = angle_to_complex(Phase.(Vfloat)) |> dev
    eval_b = gen_mqar_batches(Xoshiro(9999), Kfloat, Vfloat;
                              D, L, n_keys, n_vals, B, near_gap, density = 0.0, n_batches = n_eval_batches)

    rows = NamedTuple[]
    for cfg_key in configs
        cfg = getproperty(CONFIGS, cfg_key)
        for recenter in (true, false)
            for seed in seeds
                train_b = gen_mqar_batches(Xoshiro(seed), Kfloat, Vfloat;
                                           D, L, n_keys, n_vals, B, near_gap, density = 0.0, n_batches = n_train_batches)
                model = build_model(; D, n_heads, n_blocks, attn_mode = cfg.attn,
                                    ffn_mode = cfg.ffn, attn_kind, recenter)
                ps, st = Lux.setup(Xoshiro(seed + 1), model)
                ps = ps |> dev; st = st |> dev

                yof0 = onehot_dev(train_b[1].tgt_far, n_vals, dev)
                yon0 = onehot_dev(train_b[1].tgt_near, n_vals, dev)
                gh0 = grad_health(model, ps, st, train_b[1], Vc, yof0, yon0, dev)
                ps, losses = train!(model, ps, st, train_b, Vc, n_vals, epochs, lr, dev)
                gh1 = grad_health(model, ps, st, train_b[1], Vc, yof0, yon0, dev)
                far_acc, near_acc = eval_accuracy(model, ps, st, eval_b, Vc, n_vals, dev)

                @info(@sprintf("  cfg %s recenter=%-5s seed %d: far=%.3f near=%.3f  loss %.3f→%.3f  max|dz| init=%.1e trn=%.1e min|z|=%.1e",
                               String(cfg_key), string(recenter), seed, far_acc, near_acc,
                               losses[1], losses[end], gh0.max_absdz, gh1.max_absdz, min(gh0.min_absz, gh1.min_absz)))
                push!(rows, (config = String(cfg_key), recenter = recenter, seed = seed,
                             far_acc = far_acc, near_acc = near_acc,
                             init_loss = losses[1], final_loss = losses[end],
                             max_absdz_init = gh0.max_absdz, max_absdz_trained = gh1.max_absdz,
                             min_absz = min(gh0.min_absz, gh1.min_absz)))
            end
        end
    end

    write_csv(joinpath(outdir, "results.csv"), rows)
    summarize_recenter(rows, configs)
    @info "done (recenter)" outdir
    return rows
end

function summarize_recenter(rows, configs)
    println("\n===== RECENTER ABLATION (mean±std over seeds) =====")
    println("cfg  recenter  far-acc         near-acc        max|dz|(init) max|dz|(trn)  min|z|")
    for cfg in configs, rc in (true, false)
        rs = filter(r -> r.config == String(cfg) && r.recenter == rc, rows)
        isempty(rs) && continue
        m(f) = mean(getproperty.(rs, f)); s(f) = length(rs) > 1 ? std(getproperty.(rs, f)) : 0f0
        println(@sprintf("%-4s %-9s far=%.3f±%.3f near=%.3f±%.3f  %.1e      %.1e     %.1e",
                         String(cfg), string(rc), m(:far_acc), s(:far_acc), m(:near_acc), s(:near_acc),
                         m(:max_absdz_init), m(:max_absdz_trained), m(:min_absz)))
    end
    println("==================================================\n")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
