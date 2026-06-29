#!/usr/bin/env julia
#
# scripts/lsa_lca_residual_sweep.jl
#   Does the identity-at-init residual fix (commit 3f2d86f) — validated only on
#   plain PhasorDense MLP stacks in depth_sweep_fashionmnist.jl — also make the
#   framework's signature local-attention layers (PhasorLSA / PhasorLCA)
#   *stackable* into deep transformer towers?
#
# Setup
# -----
# Sequential FashionMNIST, one image ROW per timestep (C=28, L=28, B) — light
# memory. One base chain with a swappable attention kind and a swappable
# residual treatment, stacked `depth` deep:
#
#   _row_phase                              # (28,28,B) → (28,28,B) Phase
#   PhasorDense(28 => D)                    # input encoder (SSM dynamics)
#   ScanStack(PhasorTransformerBlock(D, attn; TREATMENT), depth)
#   SSMReadout(D => 10)                     # similarity readout over last 25%
#
# Attention kinds (the unique contributions + a reference):
#   :local_self   PhasorLSA   (across-head attention; pointwise in L)
#   :local_cross  PhasorLCA   (Hopfield anchor retrieval + V binding)
#   :ssm_self     SSMSelfAttention   (across-time attention; reference)
#
# Residual treatments (the independent variable — "before vs after" the fix):
#   :old              gate=:none,  branch_init_scale=1.0, recenter=false   (pre-fix)
#   :downscaled_ffn   gate=:none,  branch_init_scale=0.1, recenter=false   (FFN near-identity)
#   :rezero           gate=:rezero,branch_init_scale=0.1, recenter=false   (attn identity via α)
#   :rezero_recenter  gate=:rezero,branch_init_scale=0.1, recenter=true    (+ phase pre-norm)
#
# Metrics
# -------
#   Exp 1 (init, no training):
#     • extended gradient probe — per-block ∇ L2 norms for :weight AND the
#       attention-specific :scale / :anchors / :alpha; encoder/last ratio +
#       log-linear decay slope (the vanishing/exploding signature).
#     • forward stats at init — cumulative circular drift from the encoder
#       representation, and channel circular variance (rank/diversity collapse).
#   Exp 2 (trained): final loss, test accuracy vs depth, NaN/instability count.
#   Exp 3 (focused): learned α / attention-scale β trajectories (in the saved
#     params), and a discrete-vs-spiking accuracy gap at one depth.
#
# Reuses the validated harness in depth_sweep_fashionmnist.jl (ScanStack,
# load_subset, make_batches, _row_phase, write_csv, depth_loss, onehot_dev,
# test_accuracy, train_config!) via include().
#
# Usage
# -----
#   # smoke (CPU, fast — validates the harness):
#   julia --project=. -e 'include("scripts/lsa_lca_residual_sweep.jl"); smoke()'
#   # init-only gradient probe sweep (cheap, no training):
#   julia --project=. -e 'include("scripts/lsa_lca_residual_sweep.jl"); main_lsa_lca_sweep(probe_only=true)'
#   # full medium-scope sweep (GPU):
#   julia --project=. scripts/lsa_lca_residual_sweep.jl

using Statistics: mean, std

# Pull in the shared depth-sweep infrastructure (ScanStack, data loaders,
# write_csv, depth_loss, onehot_dev, test_accuracy, train_config!, cdev, …).
# The include is guarded against running depth_sweep's own main().
include(joinpath(@__DIR__, "depth_sweep_fashionmnist.jl"))

# ---------------------------------------------------------------------
# Model: attention kind × residual treatment, stacked `depth` deep
# ---------------------------------------------------------------------

const ATTN_KINDS = (:local_self, :local_cross, :ssm_self)

"Build a fresh d_model⇒d_model phase attention layer of the requested kind."
function make_attn(kind::Symbol, D::Int; n_heads::Int = 4, n_anchors::Int = 32,
                   init_mode::Symbol = :hippo, spk_args::SpikingArgs = SpikingArgs())
    if kind === :local_self
        return PhasorLSA(D => D, n_heads; init_mode = init_mode, spk_args = spk_args)
    elseif kind === :local_cross
        return PhasorLCA(D => D, n_heads, n_anchors; init_mode = init_mode, spk_args = spk_args)
    elseif kind === :ssm_self
        return SSMSelfAttention(D => D, normalize_to_unit_circle)
    else
        error("unknown attention kind :$kind (use one of $(ATTN_KINDS))")
    end
end

"Residual-treatment knobs as keyword tuples for PhasorTransformerBlock."
function treatment_kw(t::Symbol)
    if t === :old
        return (; gate = :none,   branch_init_scale = 1f0,  recenter = false, alpha0 = 0.1f0)
    elseif t === :downscaled_ffn
        return (; gate = :none,   branch_init_scale = 0.1f0, recenter = false, alpha0 = 0.1f0)
    elseif t === :rezero
        return (; gate = :rezero, branch_init_scale = 0.1f0, recenter = false, alpha0 = 0.1f0)
    elseif t === :rezero_recenter
        return (; gate = :rezero, branch_init_scale = 0.1f0, recenter = true,  alpha0 = 0.1f0)
    else
        error("unknown treatment :$t")
    end
end
const TREATMENTS = (:old, :downscaled_ffn, :rezero, :rezero_recenter)

"""
    build_attn_model(kind, treatment, D, depth; n_heads, n_anchors, scan=true, ...) -> Chain

Encoder → `depth` PhasorTransformerBlock(kind) under the given residual
treatment → SSMReadout. `scan=true` runs the stack through one compile-once
`ScanStack`.
"""
function build_attn_model(kind::Symbol, treatment::Symbol, D::Int, depth::Int;
                          n_heads::Int = 4, n_anchors::Int = 32,
                          n_classes::Int = 10, readout_frac::Float32 = 0.25f0,
                          init_mode::Symbol = :hippo,
                          spk_args::SpikingArgs = SpikingArgs(), scan::Bool = true,
                          checkpoint::Bool = false)
    tkw = treatment_kw(treatment)
    enc = PhasorDense(28 => D, normalize_to_unit_circle;
                      init_mode = init_mode, use_bias = false, spk_args = spk_args)
    readout = SSMReadout(D => n_classes; readout_frac = readout_frac)
    mk_block() = PhasorTransformerBlock(D, make_attn(kind, D; n_heads, n_anchors, init_mode, spk_args);
                                        tkw...)
    if depth == 0
        return Chain(_row_phase, enc, readout)
    elseif scan
        return Chain(_row_phase, enc, ScanStack(mk_block(), depth; checkpoint), readout)
    else
        mids = ntuple(_ -> mk_block(), depth)
        return Chain(_row_phase, enc, mids..., readout)
    end
end

# ---------------------------------------------------------------------
# Exp 1a: extended gradient probe (named leaves, not just :weight)
# ---------------------------------------------------------------------

const PROBE_KEYS = (:weight, :scale, :anchors, :alpha, :log_neg_lambda)

"Collect L2 norms of named gradient leaves into kind → [norms] (traversal order)."
function collect_named_grad_norms(g)
    acc = Dict{Symbol,Vector{Float32}}()
    _walkn!(acc, g)
    return acc
end
function _walkn!(acc, g::NamedTuple)
    for k in keys(g)
        v = getfield(g, k)
        if v isa AbstractArray{<:Number} && k in PROBE_KEYS
            push!(get!(acc, k, Float32[]), sqrt(sum(abs2, Array(v))))
        else
            _walkn!(acc, v)
        end
    end
end
_walkn!(acc, g::Tuple) = (for v in g; _walkn!(acc, v); end)
_walkn!(acc, g::AbstractVector{<:Number}) = nothing
_walkn!(acc, g::AbstractVector) = (for v in g; _walkn!(acc, v); end)
_walkn!(acc, g) = nothing

"Log-linear slope of a positive sequence vs index (≈ per-layer ∇ growth rate)."
function log_slope(v::AbstractVector{<:Real})
    n = length(v)
    n < 2 && return 0f0
    y = log.(Float32.(v) .+ 1f-12)
    x = Float32.(1:n)
    xm = mean(x); ym = mean(y)
    sxx = sum((x .- xm) .^ 2)
    sxx == 0 && return 0f0
    return Float32(sum((x .- xm) .* (y .- ym)) / sxx)
end

"""
    attn_grad_probe(model, ps, st, x, y, dev) -> NamedTuple

∇_ps loss at init on a fixed batch. Returns init loss plus, per probed leaf
kind, the encoder/last ratio and decay slope of the gradient-norm profile.
"""
function attn_grad_probe(model, ps, st, x, y, dev)
    xd = x |> dev
    yoh = onehot_dev(y, dev)
    init_loss, back = Zygote.pullback(p -> depth_loss(xd, yoh, model, p, st), ps)
    grads = back(one(init_loss))[1]
    named = collect_named_grad_norms(grads)
    w = get(named, :weight, Float32[])
    eps = 1f-12
    return (init_loss = Float32(init_loss),
            n_weight = length(w),
            grad_w_encoder = isempty(w) ? NaN32 : w[1],
            grad_w_last = isempty(w) ? NaN32 : w[end],
            grad_w_ratio = isempty(w) ? NaN32 : w[1] / (w[end] + eps),
            grad_w_slope = log_slope(w),
            grad_scale_mean = (haskey(named, :scale)  ? mean(named[:scale])  : NaN32),
            grad_anchor_mean = (haskey(named, :anchors) ? mean(named[:anchors]) : NaN32),
            grad_alpha_mean = (haskey(named, :alpha)  ? mean(named[:alpha])  : NaN32),
            named = named)
end

# ---------------------------------------------------------------------
# Exp 1b: init forward stats — drift and channel diversity
# ---------------------------------------------------------------------

"Mean circular variance across channels (dim 1): 1 - |mean_c exp(iπθ)|, ∈ [0,1]."
function channel_circvar(x::AbstractArray{<:Phase})
    z = angle_to_complex(x)
    r = abs.(sum(z, dims = 1)) ./ Float32(size(x, 1))   # resultant length per (l,b)
    return Float32(mean(1f0 .- r))
end

"""
    forward_stats(kind, treatment, D, depth; x, seed, dev, ...) -> (drift, circvar0, circvarD)

Build encoder + `depth` blocks at init, run on `x`. `drift` = mean circular
distance between the post-stack rep and the post-encoder rep (0 ⇔ identity
stack). `circvar0/circvarD` = channel circular variance just after the encoder
vs after the stack (collapse ⇒ circvarD → 0).
"""
function forward_stats(kind::Symbol, treatment::Symbol, D::Int, depth::Int;
                       x, seed::Int, dev, n_heads::Int = 4, n_anchors::Int = 32,
                       init_mode::Symbol = :hippo, spk_args::SpikingArgs = SpikingArgs())
    rng = Xoshiro(seed)
    enc_chain = Chain(_row_phase,
                      PhasorDense(28 => D, normalize_to_unit_circle;
                                  init_mode, use_bias = false, spk_args))
    ps_e, st_e = Lux.setup(rng, enc_chain)
    r0, _ = enc_chain(x |> dev, ps_e |> dev, st_e |> dev)
    circvar0 = channel_circvar(r0)

    tkw = treatment_kw(treatment)
    r = r0
    for _ in 1:depth
        blk = PhasorTransformerBlock(D, make_attn(kind, D; n_heads, n_anchors, init_mode, spk_args); tkw...)
        psb, stb = Lux.setup(rng, blk)
        r, _ = blk(r, psb |> dev, stb |> dev)
    end
    drift = circ_dist(r, r0)              # circ_dist is from depth_sweep_fashionmnist.jl
    return (; drift = Float32(drift), circvar0, circvarD = channel_circvar(r))
end

# ---------------------------------------------------------------------
# Exp 3: learned-parameter trajectories + spiking gap
# ---------------------------------------------------------------------

"Mean of every length-1 :scale (attention β) leaf in a param tree."
collect_scales(ps) = (acc = Float32[]; _walkscale!(acc, ps); acc)
function _walkscale!(o, g::NamedTuple)
    for k in keys(g)
        v = getfield(g, k)
        (k === :scale && v isa AbstractArray) ? push!(o, Float32(Array(v)[1])) : _walkscale!(o, v)
    end
end
_walkscale!(o, g::Tuple) = (for v in g; _walkscale!(o, v); end)
_walkscale!(o, g::AbstractVector{<:Number}) = nothing
_walkscale!(o, g::AbstractVector) = (for v in g; _walkscale!(o, v); end)
_walkscale!(o, g) = nothing

"""
    spiking_gap(model, ps, st, batches, spk_args, dev, D) -> (disc_acc, spk_acc)

Discrete accuracy vs spiking-dispatch accuracy: run the input encoder through
the ODE pathway (`return_type=:potential`) + `sample_phases_at_periods`, then
the transformer tail (ScanStack + SSMReadout) in discrete Phase dispatch.
Mirrors `local_attention_compare.jl::evaluate_spiking`. Layout assumed:
layer_1 = _row_phase (fn), layer_2 = encoder PhasorDense, layer_3.. = tail.
"""
function spiking_gap(model::Chain, ps, st, batches, spk_args::SpikingArgs, dev, D::Int;
                     init_mode::Symbol = :hippo)
    spiking_enc = PhasorDense(28 => D, normalize_to_unit_circle;
                              init_mode, use_bias = false, spk_args,
                              return_type = SolutionType(:potential))
    ps_enc, st_enc = ps.layer_2, st.layer_2
    tail_keys = collect(keys(ps))[3:end]

    disc_c = 0; spk_c = 0; total = 0
    for (x, y) in batches
        xd = x |> dev
        yoh = onehot_dev(y, dev)
        truth = vec(getindex.(argmax(Array(yoh), dims = 1), 1))

        # discrete reference
        sims_d, _ = model(xd, ps, st)
        disc_c += sum(vec(getindex.(argmax(Array(sims_d), dims = 1), 1)) .== truth)

        # spiking encoder → per-period phases → discrete tail
        xph = _row_phase(xd)
        L = size(xph, 2)
        sc = SpikingCall(ssm_phases_to_train(xph; spk_args = spk_args),
                         spk_args, (0f0, Float32(L) * spk_args.t_period))
        sol, _ = spiking_enc(sc, ps_enc, st_enc)
        z = sample_phases_at_periods(sol, L, spk_args;
                                     activation = normalize_to_unit_circle, unrotate = true)
        for k in tail_keys
            z, _ = model.layers[k](z, ps[k], st[k])
        end
        spk_c += sum(vec(getindex.(argmax(Array(z), dims = 1), 1)) .== truth)
        total += length(y)
    end
    return disc_c / total, spk_c / total
end

# ---------------------------------------------------------------------
# Resume support: skip (kind,treatment,depth,seed) tuples already in a CSV
# ---------------------------------------------------------------------

"Read a sweep_summary.csv and return the Set of completed (kind,treatment,depth,seed) keys."
function done_configs(file::String)
    done = Set{Tuple{String,String,Int,Int}}()
    isfile(file) || return done
    lines = readlines(file)
    length(lines) <= 1 && return done
    for ln in lines[2:end]
        f = split(ln, ',')
        length(f) >= 4 || continue
        try
            push!(done, (String(f[1]), String(f[2]), parse(Int, f[3]), parse(Int, f[4])))
        catch
        end
    end
    return done
end

"Append rows to a CSV (writing the header only when the file is new/empty)."
function append_csv(file::String, rows::Vector{<:NamedTuple})
    isempty(rows) && return
    isdir(dirname(file)) || mkpath(dirname(file))
    newfile = !isfile(file) || filesize(file) == 0
    open(file, "a") do io
        newfile && println(io, join(string.(keys(rows[1])), ","))
        for r in rows
            println(io, join((string(getfield(r, c)) for c in keys(r)), ","))
        end
    end
end

# ---------------------------------------------------------------------
# Sweep driver
# ---------------------------------------------------------------------

"""
    main_lsa_lca_sweep(; kinds, treatments, depths, seeds, epochs, lr, D,
                       n_heads, n_anchors, batchsize, n_train, n_test, use_cuda,
                       probe_only, spiking_depth, outdir)

Sweep attention kind × residual treatment × depth × seed on sequential
FashionMNIST. Per config: extended init gradient probe + forward stats, then
(unless probe_only) train and test. Writes `<outdir>/sweep_summary.csv` and
`<outdir>/grad_profile.csv`. When `spiking_depth > 0`, also records the
discrete-vs-spiking gap at that depth for each kind under the best treatment.
"""
function main_lsa_lca_sweep(; kinds = (:local_self, :local_cross, :ssm_self),
                            treatments = TREATMENTS,
                            depths = (1, 2, 4, 8, 16),
                            seeds = 1:3,
                            epochs::Int = 8,
                            lr::Real = 3f-4,
                            D::Int = 64,
                            n_heads::Int = 4,
                            n_anchors::Int = 32,
                            batchsize::Int = 48,
                            n_train::Int = 10_000,
                            n_test::Int = 2_000,
                            use_cuda::Bool = true,
                            checkpoint::Bool = true,
                            probe_only::Bool = false,
                            init_probe::Bool = true,
                            spiking_depth::Int = 8,
                            best_treatment::Symbol = :rezero,
                            resume::Bool = true,
                            max_configs::Int = typemax(Int),
                            outdir::String = "results/lsa_lca_residual")
    dev = (use_cuda && CUDA.functional()) ? gpu_device() : cdev
    isdir(outdir) || mkpath(outdir)
    spk_args = SpikingArgs()

    (Xtr, ytr), (Xte, yte) = load_subset(; n_train, n_test)
    train_batches = make_batches(Xtr, ytr, batchsize)
    test_batches  = make_batches(Xte, yte, batchsize)
    probe_x, probe_y = train_batches[1]

    @info "lsa/lca residual sweep" kinds treatments depths=collect(depths) seeds=collect(seeds) epochs lr D batchsize checkpoint probe_only spiking_depth device=string(dev)

    summary_file = joinpath(outdir, "sweep_summary.csv")
    profile_file = joinpath(outdir, "grad_profile.csv")
    # Resume: keep prior rows and skip configs already recorded; otherwise start clean.
    if !resume
        rm(summary_file, force = true); rm(profile_file, force = true)
    end
    done = resume ? done_configs(summary_file) : Set{Tuple{String,String,Int,Int}}()
    isempty(done) || @info "resuming — $(length(done)) configs already done"
    summary_rows = NamedTuple[]
    profile_rows = NamedTuple[]
    expected = length(kinds) * length(treatments) * length(depths) * length(seeds)
    n_new = 0

    for kind in kinds, t in treatments, depth in depths, seed in seeds
        (string(kind), string(t), depth, seed) in done && continue
        try
            rng = Xoshiro(seed)
            model = build_attn_model(kind, t, D, depth; n_heads, n_anchors, spk_args, checkpoint)
            ps, st = Lux.setup(rng, model); ps = ps |> dev; st = st |> dev
            n_params = Lux.parameterlength(model)

            # Init-time diagnostics (grad probe + forward stats). Skippable for
            # the trained sweep when Exp 1 already recorded them for every config
            # — saves both memory (no depth-deep block-building here) and time.
            local probe, fs
            if init_probe
                probe = attn_grad_probe(model, ps, st, probe_x, probe_y, dev)
                fs = forward_stats(kind, t, D, depth; x = probe_x, seed, dev, n_heads, n_anchors, spk_args)
            else
                probe = (; init_loss = NaN32, grad_w_ratio = NaN32, grad_w_slope = NaN32,
                         grad_scale_mean = NaN32, grad_anchor_mean = NaN32, grad_alpha_mean = NaN32,
                         named = Dict{Symbol,Vector{Float32}}())
                fs = (; drift = NaN32, circvar0 = NaN32, circvarD = NaN32)
            end

            final_loss = NaN32; test_acc = NaN32; nan_loss = false
            if !probe_only
                ps, final_loss = train_config!(model, ps, st, train_batches, epochs, lr, dev;
                                               alpha_lr_mult = (treatment_kw(t).gate === :rezero ? 5f0 : 1f0))
                nan_loss = !isfinite(final_loss)
                test_acc = test_accuracy(model, ps, st, test_batches, dev)
            end

            srow = (; kind, treatment = t, depth, seed, n_params,
                    init_loss = probe.init_loss, final_loss, test_acc, nan_loss,
                    grad_w_ratio = probe.grad_w_ratio, grad_w_slope = probe.grad_w_slope,
                    grad_scale_mean = probe.grad_scale_mean,
                    grad_anchor_mean = probe.grad_anchor_mean,
                    grad_alpha_mean = probe.grad_alpha_mean,
                    drift = fs.drift, circvar0 = fs.circvar0, circvarD = fs.circvarD)
            prows = NamedTuple[]
            for (kk, vv) in probe.named, (li, gn) in enumerate(vv)
                push!(prows, (; kind, treatment = t, depth, seed, leaf = kk, idx = li, grad_norm = gn))
            end
            push!(summary_rows, srow); append!(profile_rows, prows)
            append_csv(summary_file, [srow]); append_csv(profile_file, prows)
            n_new += 1

            @info "config" kind treatment=t depth seed test_acc=round(test_acc, digits=4) grad_w_ratio=round(probe.grad_w_ratio, digits=3) drift=round(fs.drift, digits=3) circvarD=round(fs.circvarD, digits=3)
        catch err
            @error "config failed; skipping" kind treatment=t depth seed exception=(err, catch_backtrace())
        end
        GC.gc(); dev === gpu_device() && CUDA.reclaim()
        # Process-chunking: exit cleanly after `max_configs` new configs so the
        # launcher can spawn a fresh process (full memory reset). Defeats the
        # cross-config memory growth that stalls a single long-lived process.
        if n_new >= max_configs
            @info "reached max_configs this run — exiting for memory reset" n_new
            break
        end
    end

    # Exp 3: spiking gap at one depth, best treatment, per kind.
    # Only after the full matrix is done (so process-chunked runs don't start it early).
    if !probe_only && spiking_depth > 0 && (length(done) + n_new) >= expected
        gap_file = joinpath(outdir, "spiking_gap.csv")
        gap_done = Set{String}()
        if isfile(gap_file)
            gl = readlines(gap_file)
            length(gl) > 1 && foreach(ln -> (f = split(ln, ','); !isempty(f) && push!(gap_done, String(f[1]))), gl[2:end])
        end
        for kind in kinds
            string(kind) in gap_done && continue
            try
                rng = Xoshiro(first(seeds))
                model = build_attn_model(kind, best_treatment, D, spiking_depth; n_heads, n_anchors, spk_args, checkpoint)
                ps, st = Lux.setup(rng, model); ps = ps |> dev; st = st |> dev
                ps, _ = train_config!(model, ps, st, train_batches, epochs, lr, dev;
                                      alpha_lr_mult = (treatment_kw(best_treatment).gate === :rezero ? 5f0 : 1f0))
                d_acc, s_acc = spiking_gap(model, ps, st, test_batches, spk_args, dev, D)
                grow = (; kind, treatment = best_treatment, depth = spiking_depth,
                        disc_acc = d_acc, spk_acc = s_acc, gap = d_acc - s_acc)
                append_csv(gap_file, [grow])
                @info "spiking gap" kind disc_acc=round(d_acc, digits=4) spk_acc=round(s_acc, digits=4)
            catch err
                @error "spiking gap failed; skipping" kind exception=(err, catch_backtrace())
            end
            GC.gc(); dev === gpu_device() && CUDA.reclaim()
        end
    end

    @info "sweep done" outdir
    return summary_rows
end

"Fast CPU smoke run to validate the harness end-to-end."
smoke(; use_cuda::Bool = false) = main_lsa_lca_sweep(;
    kinds = (:local_self, :local_cross), treatments = (:old, :rezero),
    depths = (1, 8), seeds = 1:1, epochs = 1, D = 24,
    batchsize = 64, n_train = 512, n_test = 256, use_cuda,
    spiking_depth = 8, best_treatment = :rezero,
    outdir = "results/lsa_lca_residual/smoke")

if abspath(PROGRAM_FILE) == @__FILE__
    main_lsa_lca_sweep()
end
