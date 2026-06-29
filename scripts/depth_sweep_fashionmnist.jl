#!/usr/bin/env julia
#
# scripts/depth_sweep_fashionmnist.jl
#   FashionMNIST depth / trainability sweep for phasor networks.
#
# Question
# --------
# How does trainability of a phasor MLP scale with depth, and do the two
# conditioning tricks help?
#   (1) layer bias  — shifts the per-channel origin in the complex plane away
#       from 0+0im (default_bias = 1+0im). Improves layer conditioning.
#   (2) residual binding — `ResidualBlock` computes `y = v_bind(x, f(x))`
#       (= remap_phase(x + f(x))), a phase-domain skip connection that lets
#       gradients bypass the feed-forward path in deep stacks.
#
# We sweep depth = number of D=>D middle blocks over
#       [1..10] ∪ {15,20,25,30,35,40,45,50}
# in the 2x2 design (use_bias ∈ {false,true}) × (use_residual ∈ {false,true}),
# over several seeds, measuring BOTH:
#   • an init-time gradient-pass-through probe (per-layer ∇weight L2 norm on a
#     fixed batch; the encoder/first-layer norm collapsing with depth is the
#     vanishing-gradient signature), and
#   • trained outcome (initial loss, final loss, test accuracy).
#
# Two interchangeable forward paths (same readout/loss, so metrics compare):
#   :static     — 2D demo-style:  Flatten → LayerNorm → Phase(tanh) →
#                 PhasorDense(784=>D) → [D=>D]*depth → Codebook(D=>10)
#   :sequential — 3D SSM, 28 pixels (one image row) per timestep:
#                 rowphase (C=28,L=28,B) → PhasorDense(28=>D) →
#                 [D=>D]*depth → last-step → Codebook(D=>10)
#
# Usage
# -----
#   julia --project=scripts scripts/depth_sweep_fashionmnist.jl              # full :static sweep
#   julia --project=scripts -e 'include("scripts/depth_sweep_fashionmnist.jl"); main_depth_sweep(path=:both)'
#   # quick smoke run:
#   julia --project=scripts -e 'include("scripts/depth_sweep_fashionmnist.jl"); main_depth_sweep(path=:both, depths=[1,5,10], seeds=1:1, epochs=1, n_train=1000, n_test=500, use_cuda=false)'
#
# First-time env setup:
#   julia --project=scripts -e 'using Pkg; Pkg.instantiate()'

using PhasorNetworks, Lux, Zygote, Optimisers, CUDA, LuxCUDA
using OneHotArrays: onehotbatch
using Random: Xoshiro, AbstractRNG
using Statistics: mean, std
import PhasorNetworks: default_bias

# Load Plots at include time (not inside make_plots) so the binding is in an
# older world than the call — Julia 1.12 forbids calling a just-@eval'd binding
# in the same world. Skip figures gracefully if Plots is unavailable.
const PLOTS_OK = try
    @eval using Plots
    true
catch err
    @warn "Plots unavailable; figures will be skipped" exception = err
    false
end

const DEFAULT_DEPTHS = vcat(1:10, 15:5:50)
const cdev = cpu_device()

"Read an optional field from an arm NamedTuple with a default."
armget(arm, k::Symbol, default) = hasproperty(arm, k) ? getproperty(arm, k) : default

# ---------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------

"""
    load_subset(; n_train, n_test) -> ((Xtr, ytr), (Xte, yte))

FashionMNIST as `28×28×N` Float32 in [0,1] with `Int` labels in 0..9.
"""
function load_subset(; n_train::Int = 10_000, n_test::Int = 2_000)
    train = fashion_mnist_data(:train)
    test  = fashion_mnist_data(:test)
    n_train = min(n_train, length(train.targets))
    n_test  = min(n_test,  length(test.targets))
    Xtr = Float32.(train.features[:, :, 1:n_train]); ytr = Int.(train.targets[1:n_train])
    Xte = Float32.(test.features[:, :, 1:n_test]);   yte = Int.(test.targets[1:n_test])
    return (Xtr, ytr), (Xte, yte)
end

"Split `(28,28,N)` images + labels into a vector of `(x_batch, y_batch)` (raw Float32, labels 0..9)."
function make_batches(X::Array{Float32,3}, y::Vector{Int}, B::Int)
    N = length(y)
    out = Vector{Tuple{Array{Float32,3}, Vector{Int}}}()
    for i in 1:B:N
        e = min(i + B - 1, N)
        push!(out, (X[:, :, i:e], y[i:e]))
    end
    return out
end

# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------

"`:flat` per-channel log(-lambda): single timescale α = 5/L so state survives L steps."
flat_lnl(D::Int; L::Int = 28) = fill(Float32(log(5f0 / Float32(L))), D)

"Encode `(28,28,B)` image batch as `(C=28, L=28, B)` Phase — one image row per timestep."
function _row_phase(X::AbstractArray{<:Real,3})
    Xp = permutedims(X, (2, 1, 3))          # (W=28 channels, H=28 timesteps, B)
    return Phase.(Float32.(Xp) ./ 2f0)      # v∈[0,1] → phase v/2 ∈ [0,0.5]
end

# --- identity-at-init knobs --------------------------------------------------
# A residual block should start ≈ identity; for v_bind that means the branch
# output phase ≈ 0 at init. These shrink the branch's init output.

"glorot weight init scaled by γ (γ<1 ⇒ smaller branch output at init)."
scaled_glorot(γ::Real) = (rng, dims...) -> Float32(γ) .* Lux.glorot_uniform(rng, dims...)

"complex bias init of magnitude m on the +real axis (large m ⇒ output phase→0)."
bias_mag(m::Real) = (rng, dims) -> ComplexF32(m) .* ones(ComplexF32, dims)

"""
    ReZeroResidualBlock(D, act; kwargs...)

Phasor analog of ReZero: `y = v_bind(x, α .* ff(x))` with a single learnable
scalar `α` initialized to 0. At init the branch contributes nothing (exact
identity AND `dy/dx = I`); `α` grows during training. Path-agnostic — sidesteps
the 3D bias frame-correction accounting that complicates the bias-magnitude knob.
"""
struct ReZeroResidualBlock{F} <: Lux.AbstractLuxLayer
    ff::F
    alpha0::Float32
end
ReZeroResidualBlock(D::Int, act = normalize_to_unit_circle; alpha0::Real = 0f0, kwargs...) =
    ReZeroResidualBlock(PhasorDense(D => D, act; kwargs...), Float32(alpha0))

Lux.initialparameters(rng::AbstractRNG, b::ReZeroResidualBlock) =
    (ff = Lux.initialparameters(rng, b.ff), alpha = Float32[b.alpha0])
Lux.initialstates(rng::AbstractRNG, b::ReZeroResidualBlock) =
    (ff = Lux.initialstates(rng, b.ff),)
Lux.parameterlength(b::ReZeroResidualBlock) = Lux.parameterlength(b.ff) + 1

function (b::ReZeroResidualBlock)(x, ps, st)
    ff_out, st_ff = b.ff(x, ps.ff, st.ff)
    y = v_bind(x, ps.alpha .* ff_out)        # α=0 ⇒ y = x
    return y, (ff = st_ff,)
end

# NOTE: `PhaseRecenter` (phase-domain pre-norm) now lives in `src/ssm.jl` and is
# exported by PhasorNetworks; the former script-local definition was removed to
# avoid clobbering the exported name.

"""
    make_block(D; use_bias, use_residual, lnl, residual_mode=:bind,
               init_scale=1, bias_magnitude=1, recenter=false)

Build one D=>D middle block. `residual_mode` (when `use_residual`):
`:bind` → `ResidualBlock` (v_bind skip), `:rezero` → `ReZeroResidualBlock`.
`init_scale` scales the branch weight init; `bias_magnitude` sets the (real)
bias magnitude; `recenter=true` appends a `PhaseRecenter` after the block.
"""
function make_block(D::Int; use_bias::Bool, use_residual::Bool, lnl,
                    residual_mode::Symbol = :bind, init_scale::Real = 1f0,
                    bias_magnitude::Real = 1f0, recenter::Bool = false,
                    alpha0::Real = 0f0)
    kw = (use_bias = use_bias, init_weight = scaled_glorot(init_scale))
    kw = use_bias ? merge(kw, (init_bias = bias_mag(bias_magnitude),)) : kw
    kw = lnl === nothing ? kw : merge(kw, (init_log_neg_lambda = lnl,))

    block = if !use_residual
        PhasorDense(D => D, normalize_to_unit_circle; kw...)
    elseif residual_mode === :rezero
        ReZeroResidualBlock(D, normalize_to_unit_circle; alpha0, kw...)
    elseif residual_mode === :bind
        ResidualBlock((D, D), normalize_to_unit_circle; kw...)
    else
        error("residual_mode must be :bind or :rezero, got :$residual_mode")
    end
    return recenter ? Chain(block, PhaseRecenter()) : block
end

"""
    ScanStack(block, depth; checkpoint=false) <: Lux.AbstractLuxLayer

Apply `depth` independent copies of `block` (same shape, distinct params) in
sequence via a runtime loop instead of a length-`depth` `Lux.Chain`.

Why: a `Chain` of N layers is a length-N tuple, so every distinct depth forces
a *new* `applychain` specialization — across a 1..50 sweep that is N separate
compiles. A loop over a homogeneous parameter container compiles **one** block
body and reuses it for every depth. Per-step FLOPs are unchanged (depth is
FLOP-bound); the win is compile-once + (optionally) checkpointed memory.

Params: `(blocks = [p_1, …, p_depth],)` — a Vector of the block's own param
NamedTuples (distinct random init per layer). State: the block's (shared,
param-free) state. `checkpoint=true` wraps each step in `Zygote.checkpointed`
to recompute activations in the backward pass (O(1) tape instead of O(depth)).
"""
struct ScanStack{B} <: Lux.AbstractLuxLayer
    block::B
    depth::Int
    checkpoint::Bool
end
ScanStack(block, depth::Int; checkpoint::Bool = false) = ScanStack(block, depth, checkpoint)

function Lux.initialparameters(rng::AbstractRNG, s::ScanStack)
    return (blocks = [Lux.initialparameters(rng, s.block) for _ in 1:s.depth],)
end
Lux.initialstates(rng::AbstractRNG, s::ScanStack) = (block = Lux.initialstates(rng, s.block),)
# Default parameterlength doesn't recurse the `blocks` Vector → undercounts.
Lux.parameterlength(s::ScanStack) = s.depth * Lux.parameterlength(s.block)

# One block application (top-level so Zygote.checkpointed can target it).
_apply_block(block, x, p, bst) = first(block(x, p, bst))

function (s::ScanStack)(x, ps, st)
    bst = st.block
    for i in 1:s.depth
        p = ps.blocks[i]
        x = s.checkpoint ? Zygote.checkpointed(_apply_block, s.block, x, p, bst) :
                           _apply_block(s.block, x, p, bst)
    end
    return x, st
end

"""
    build_depth_model(path, D, depth; use_bias, use_residual, scan=false, checkpoint=false) -> Lux.Chain

`depth` = number of D=>D middle blocks (total phasor layers = depth+1). When
`scan=true` the middle blocks are run by a single `ScanStack` (compile-once for
any depth) rather than splatted into the `Chain`; `checkpoint=true` enables
gradient checkpointing inside the stack.
"""
# Per-path pieces: pre-layers tuple (ends in encoder), mid-block λ init, post tuple.
function _model_parts(path::Symbol, D::Int; use_bias::Bool)
    enc_bias_kw = use_bias ? (use_bias = true, init_bias = default_bias) : (use_bias = false,)
    if path === :static
        pre = (FlattenLayer(), LayerNorm((28^2,)), x -> Phase.(tanh.(x)),
               PhasorDense(28^2 => D, normalize_to_unit_circle; enc_bias_kw...))
        return pre, nothing, (Codebook(D => 10),)
    elseif path === :sequential
        lnl = flat_lnl(D; L = 28)
        pre = (_row_phase,
               PhasorDense(28 => D, normalize_to_unit_circle; enc_bias_kw..., init_log_neg_lambda = lnl))
        return pre, lnl, (x -> x[:, end, :], Codebook(D => 10))
    else
        error("path must be :static or :sequential, got :$path")
    end
end

function build_depth_model(path::Symbol, D::Int, depth::Int;
                           use_bias::Bool, use_residual::Bool,
                           scan::Bool = false, checkpoint::Bool = false,
                           residual_mode::Symbol = :bind, init_scale::Real = 1f0,
                           bias_magnitude::Real = 1f0, recenter::Bool = false,
                           alpha0::Real = 0f0)
    @assert depth >= 0
    pre, lnl, post = _model_parts(path, D; use_bias)
    blkkw = (; residual_mode, init_scale, bias_magnitude, recenter, alpha0)

    if depth == 0
        return Chain(pre..., post...)
    elseif scan
        proto = make_block(D; use_bias, use_residual, lnl, blkkw...)
        return Chain(pre..., ScanStack(proto, depth; checkpoint), post...)
    else
        mids = ntuple(_ -> make_block(D; use_bias, use_residual, lnl, blkkw...), depth)
        return Chain(pre..., mids..., post...)
    end
end

# ---------------------------------------------------------------------
# Loss / metrics
# ---------------------------------------------------------------------

# Loss takes a pre-built, on-device one-hot target (no device logic inside the AD path).
function depth_loss(x, yoh, model, ps, st)
    yp, _ = model(x, ps, st)
    return mean(evaluate_loss(yp, yoh, :similarity))
end

onehot_dev(y, dev) = Float32.(onehotbatch(y, 0:9)) |> dev

"Test-set accuracy via similarity argmax (predict returns 1-based labels)."
function test_accuracy(model, ps, st, batches, dev)
    correct = 0; total = 0
    for (x, y) in batches
        yp, _ = model(x |> dev, ps, st)
        preds = predict(cdev(yp), :similarity)   # 1-based
        correct += sum(preds .== (y .+ 1))
        total   += length(y)
    end
    return correct / total
end

# ---------------------------------------------------------------------
# Gradient-pass-through probe (at init)
# ---------------------------------------------------------------------

"Recursively collect L2 norms of every `:weight` leaf in a gradient tree, in traversal order."
function collect_weight_grad_norms(g)
    norms = Float32[]
    _walk!(norms, g)
    return norms
end
function _walk!(norms, g::NamedTuple)
    for k in keys(g)
        v = getfield(g, k)
        if k === :weight && v !== nothing
            push!(norms, sqrt(sum(abs2, Array(v))))
        else
            _walk!(norms, v)
        end
    end
end
_walk!(norms, g::Tuple)  = (for v in g; _walk!(norms, v); end)
# Numeric vectors (incl. GPU CuArrays like log_neg_lambda/bias_*) are LEAVES —
# iterating them would scalar-index a GPU array. Must come before the generic
# AbstractVector method below.
_walk!(norms, g::AbstractVector{<:Number}) = nothing
# ScanStack params live in a Vector of block-NamedTuples (`blocks = [p_1, …]`);
# this (host) container vector is recursed element-wise.
_walk!(norms, g::AbstractVector) = (for v in g; _walk!(norms, v); end)
_walk!(norms, g)         = nothing   # numbers / nothing / numeric arrays — leaves we don't index by name

"Collect the learned ReZero gate `α` (scalar per block) from a param tree, in block order."
function collect_alphas(ps)
    out = Float32[]; _walkα!(out, ps); return out
end
function _walkα!(o, g::NamedTuple)
    for k in keys(g)
        v = getfield(g, k)
        k === :alpha ? push!(o, Float32(Array(v)[1])) : _walkα!(o, v)
    end
end
_walkα!(o, g::Tuple) = (for v in g; _walkα!(o, v); end)
_walkα!(o, g::AbstractVector{<:Number}) = nothing
_walkα!(o, g::AbstractVector) = (for v in g; _walkα!(o, v); end)
_walkα!(o, g) = nothing

"""
    grad_probe(model, ps, st, x, y, dev) -> NamedTuple

Compute ∇_ps loss at initialization on one fixed batch and summarize the
per-phasor-layer weight-gradient norm profile (g[1] = encoder, g[end] = last block).
"""
function grad_probe(model, ps, st, x, y, dev)
    xd = x |> dev
    yoh = onehot_dev(y, dev)
    init_loss, back = Zygote.pullback(p -> depth_loss(xd, yoh, model, p, st), ps)
    grads = back(one(init_loss))[1]
    profile = collect_weight_grad_norms(grads)
    eps = 1f-12
    return (init_loss = Float32(init_loss),
            profile = profile,
            grad_encoder = profile[1],
            grad_last = profile[end],
            grad_ratio = profile[1] / (profile[end] + eps),
            grad_min = minimum(profile),
            grad_max = maximum(profile))
end

# ---------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------

# Give the ReZero gate α a higher learning rate (RMSProp is gradient-scale
# invariant, so this must be a real per-leaf η change, not grad scaling).
# Mirrors the package's _adjust_ssm_lr!, but targets :alpha and supports the
# integer indices introduced by ScanStack's `blocks` Vector.
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

function train_config!(model, ps, st, train_batches, epochs, lr, dev; alpha_lr_mult::Real = 1f0)
    opt_state = Optimisers.setup(Optimisers.RMSProp(Float32(lr)), ps)
    alpha_lr_mult != 1 && boost_alpha_lr!(opt_state, ps, Float32(lr) * Float32(alpha_lr_mult))
    final_loss = NaN32
    for _ in 1:epochs
        for (x, y) in train_batches
            xd = x |> dev
            yoh = onehot_dev(y, dev)
            loss_val, back = Zygote.pullback(p -> depth_loss(xd, yoh, model, p, st), ps)
            grads = back(one(loss_val))[1]
            opt_state, ps = Optimisers.update(opt_state, ps, grads)
            final_loss = Float32(loss_val)
        end
    end
    return ps, final_loss
end

# ---------------------------------------------------------------------
# Residual-study diagnostics (init-only, no AD)
# ---------------------------------------------------------------------

"Mean circular distance (in π units, ∈[0,1]) between two phase arrays."
circ_dist(a, b) = Float32(mean(abs.(Float32.(v_bind(a, .- b)))))

"""
    forward_drift(path, D, depth; arm knobs…, dev, x, seed) -> (drift, bdisp)

Build encoder + `depth` middle blocks at init and run them stepwise on `x`.
`bdisp[k]` = per-block displacement (circular dist between consecutive reps) —
the residual branch's init output magnitude (≈0 ⇔ identity). `drift[k]` =
cumulative circular distance of the rep after k blocks from the post-encoder rep.
"""
function forward_drift(path::Symbol, D::Int, depth::Int; dev, x, seed::Int = 1,
                       use_bias::Bool = true, use_residual::Bool = true,
                       residual_mode::Symbol = :bind, init_scale::Real = 1f0,
                       bias_magnitude::Real = 1f0, recenter::Bool = false,
                       alpha0::Real = 0f0)
    rng = Xoshiro(seed)
    pre, lnl, _ = _model_parts(path, D; use_bias)
    pre_chain = Chain(pre...)
    ps_pre, st_pre = Lux.setup(rng, pre_chain)
    r0, _ = pre_chain(x |> dev, ps_pre |> dev, st_pre |> dev)

    drift = Float32[]; bdisp = Float32[]
    r = r0
    for _ in 1:depth
        blk = make_block(D; use_bias, use_residual, lnl, residual_mode,
                         init_scale, bias_magnitude, recenter, alpha0)
        psb, stb = Lux.setup(rng, blk)
        rprev = r
        r, _ = blk(r, psb |> dev, stb |> dev)
        push!(bdisp, circ_dist(r, rprev))
        push!(drift, circ_dist(r, r0))
    end
    return (; drift, bdisp)
end

# ---------------------------------------------------------------------
# Gold standard: real-valued ResNet-MLP (additive residual, zero-init branch)
# ---------------------------------------------------------------------

"Real ResNet-MLP: Dense encoder → depth × [x + Dense(relu)∘LayerNorm, 2nd Dense zero-init] → readout."
function build_resnet_mlp(depth::Int; D::Int = 64)
    resblock() = SkipConnection(
        Chain(LayerNorm((D,)), Dense(D => D, relu), Dense(D => D; init_weight = zeros32)), +)
    mids = ntuple(_ -> resblock(), depth)
    return Chain(FlattenLayer(), LayerNorm((28^2,)), Dense(28^2 => D),
                 mids..., Dense(D => 10), softmax)
end

ce_loss_real(p, yoh) = -mean(sum(yoh .* log.(p .+ 1f-8), dims = 1))

function run_resnet_baseline(; D, depths, seeds, epochs, lr, batchsize,
                             n_train, n_test, dev, outdir)
    (Xtr, ytr), (Xte, yte) = load_subset(; n_train, n_test)
    trb = make_batches(Xtr, ytr, batchsize)
    teb = make_batches(Xte, yte, batchsize)
    rows = NamedTuple[]
    file = joinpath(outdir, "resnet_baseline.csv")
    for depth in depths, seed in seeds
        try
            model = build_resnet_mlp(depth; D)
            ps, st = Lux.setup(Xoshiro(seed), model); ps = ps |> dev; st = st |> dev
            opt = Optimisers.setup(Optimisers.Adam(Float32(lr)), ps)
            init_loss = NaN32; final_loss = NaN32
            for ep in 1:epochs, (x, y) in trb
                xd = x |> dev; yoh = onehot_dev(y, dev)
                lv, back = Zygote.pullback(p -> ce_loss_real(first(model(xd, p, st)), yoh), ps)
                opt, ps = Optimisers.update(opt, ps, back(one(lv))[1])
                ep == 1 && isnan(init_loss) && (init_loss = Float32(lv)); final_loss = Float32(lv)
            end
            correct = 0; total = 0
            for (x, y) in teb
                p, _ = model(x |> dev, ps, st)
                correct += sum(vec(getindex.(argmax(Array(p), dims = 1), 1)) .== (y .+ 1)); total += length(y)
            end
            push!(rows, (; depth, seed, init_loss, final_loss, test_acc = correct / total))
            @info "resnet" depth seed final_loss=round(final_loss, digits=4) test_acc=round(correct/total, digits=4)
        catch err
            @error "resnet config failed; skipping" depth seed exception=(err, catch_backtrace())
        end
        write_csv(file, rows); GC.gc(); dev === gpu_device() && CUDA.reclaim()
    end
    return rows
end

# ---------------------------------------------------------------------
# Sweep driver
# ---------------------------------------------------------------------

function run_path(path::Symbol; D, depths, seeds, epochs, lr, batchsize,
                  n_train, n_test, dev, outdir, probe_only, scan, checkpoint)
    (Xtr, ytr), (Xte, yte) = load_subset(; n_train, n_test)
    train_batches = make_batches(Xtr, ytr, batchsize)
    test_batches  = make_batches(Xte, yte, batchsize)
    probe_x, probe_y = train_batches[1]   # fixed batch for the init gradient probe

    conditions = [(use_bias = b, use_residual = r) for r in (false, true) for b in (false, true)]

    summary_rows = NamedTuple[]
    profile_rows = NamedTuple[]

    summary_file = joinpath(outdir, "$(path)_summary.csv")
    profile_file = joinpath(outdir, "$(path)_gradprofile.csv")

    for depth in depths, cond in conditions, seed in seeds
        # Each config is isolated: a failure (e.g. OOM on a very deep chain)
        # is logged and skipped so the rest of the sweep still completes.
        try
            rng = Xoshiro(seed)
            model = build_depth_model(path, D, depth; cond.use_bias, cond.use_residual, scan, checkpoint)
            ps, st = Lux.setup(rng, model)
            ps = ps |> dev; st = st |> dev
            n_params = Lux.parameterlength(model)

            probe = grad_probe(model, ps, st, probe_x, probe_y, dev)

            if probe_only
                final_loss = NaN32; test_acc = NaN32
            else
                ps, final_loss = train_config!(model, ps, st, train_batches, epochs, lr, dev)
                test_acc = test_accuracy(model, ps, st, test_batches, dev)
            end

            push!(summary_rows, (; path, depth, cond.use_bias, cond.use_residual, seed,
                                 init_loss = probe.init_loss, final_loss, test_acc,
                                 grad_encoder = probe.grad_encoder, grad_last = probe.grad_last,
                                 grad_ratio = probe.grad_ratio, grad_min = probe.grad_min,
                                 grad_max = probe.grad_max, n_params))
            for (li, gn) in enumerate(probe.profile)
                push!(profile_rows, (; path, depth, cond.use_bias, cond.use_residual, seed,
                                     layer_index = li, grad_norm = gn))
            end

            @info "config" path depth bias=cond.use_bias res=cond.use_residual seed init_loss=round(probe.init_loss,digits=4) final_loss=round(final_loss,digits=4) test_acc=round(test_acc,digits=4) grad_ratio=round(probe.grad_ratio,digits=3)
        catch err
            @error "config failed; skipping" path depth bias=cond.use_bias res=cond.use_residual seed exception=(err, catch_backtrace())
        end

        # Persist after every config so a later stall never loses prior results.
        write_csv(summary_file, summary_rows)
        write_csv(profile_file, profile_rows)
        GC.gc()
        dev === gpu_device() && CUDA.reclaim()
    end

    return summary_rows
end

# ---------------------------------------------------------------------
# Output: CSV + plots
# ---------------------------------------------------------------------

function write_csv(file, rows::Vector{<:NamedTuple})
    isempty(rows) && return
    isdir(dirname(file)) || mkpath(dirname(file))
    cols = keys(rows[1])
    open(file, "w") do io
        println(io, join(string.(cols), ","))
        for r in rows
            println(io, join((string(getfield(r, c)) for c in cols), ","))
        end
    end
end

"Aggregate mean over seeds → (depths, mean) for a metric, filtered to a condition."
function _curve(rows, use_bias, use_residual, metric)
    sub = filter(r -> r.use_bias == use_bias && r.use_residual == use_residual, rows)
    ds = sort(unique(getfield.(sub, :depth)))
    ms = Float64[]; ss = Float64[]
    for d in ds
        vals = Float64[getfield(r, metric) for r in sub if r.depth == d && !isnan(getfield(r, metric))]
        push!(ms, isempty(vals) ? NaN : mean(vals))
        push!(ss, length(vals) > 1 ? std(vals) : 0.0)
    end
    return ds, ms, ss
end

function make_plots(path, rows, outdir)
    PLOTS_OK || (@info "Plots unavailable; skipping figures for $path"; return)
    conds = [(false,false,"plain"), (true,false,"bias"),
             (false,true,"resid"),  (true,true,"bias+resid")]
    specs = [(:test_acc, "test accuracy", :identity),
             (:final_loss, "final loss", :identity),
             (:grad_ratio, "encoder/last ∇ ratio", :log10)]
    for (metric, ylab, yscale) in specs
        plt = Plots.plot(; xlabel = "depth (D=>D blocks)", ylabel = ylab,
                         title = "$(path): $(ylab) vs depth", legend = :outertopright,
                         yscale = yscale)
        any_data = false
        for (b, r, lab) in conds
            ds, ms, ss = _curve(rows, b, r, metric)
            (isempty(ds) || all(isnan, ms)) && continue
            any_data = true
            if metric === :grad_ratio
                Plots.plot!(plt, ds, ms; label = lab, marker = :circle, lw = 2)
            else
                Plots.plot!(plt, ds, ms; ribbon = ss, label = lab, marker = :circle, lw = 2)
            end
        end
        any_data || continue
        f = joinpath(outdir, "$(path)_$(metric).png")
        Plots.savefig(plt, f)
        @info "saved figure" f
    end
end

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

"""
    main_depth_sweep(; path=:static, D=64, depths=DEFAULT_DEPTHS, seeds=1:3,
                     epochs=5, lr=1e-3, batchsize=128, n_train=10_000,
                     n_test=2_000, use_cuda=true, probe_only=false,
                     scan=false, checkpoint=false,
                     skip_plots=false, outdir="results/depth_sweep")

Run the depth × (bias × residual) × seed sweep. `path=:both` runs `:static`
then `:sequential`. Writes `<path>_summary.csv`, `<path>_gradprofile.csv`,
and PNG figures to `outdir`.

`scan=true` runs the middle blocks through a single `ScanStack` (compile-once
for any depth — recommended for the deep `:sequential` range); `checkpoint=true`
adds gradient checkpointing inside the stack for memory headroom.
"""
function main_depth_sweep(; path::Symbol = :static,
                          D::Int = 64,
                          depths = DEFAULT_DEPTHS,
                          seeds = 1:3,
                          epochs::Int = 5,
                          lr::Real = 1f-3,
                          batchsize::Int = 128,
                          n_train::Int = 10_000,
                          n_test::Int = 2_000,
                          use_cuda::Bool = true,
                          probe_only::Bool = false,
                          scan::Bool = false,
                          checkpoint::Bool = false,
                          skip_plots::Bool = false,
                          outdir::String = "results/depth_sweep")
    dev = (use_cuda && CUDA.functional()) ? gpu_device() : cdev
    isdir(outdir) || mkpath(outdir)
    paths = path === :both ? (:static, :sequential) : (path,)

    @info "depth sweep" path D depths=collect(depths) seeds=collect(seeds) epochs lr batchsize n_train n_test probe_only scan checkpoint device=string(dev)

    all_results = Dict{Symbol,Vector{NamedTuple}}()
    for p in paths
        @info "=== path $p ==="
        rows = run_path(p; D, depths, seeds, epochs, lr, batchsize,
                        n_train, n_test, dev, outdir, probe_only, scan, checkpoint)
        all_results[p] = rows
        skip_plots || make_plots(p, rows, outdir)
    end
    @info "done" outdir
    return all_results
end

# ---------------------------------------------------------------------
# Residual study: identity-at-init interventions + ResNet gold standard
# ---------------------------------------------------------------------

"Intervention arms (all use_bias=true, use_residual=true)."
const RESIDUAL_ARMS = [
    (name = "baseline",  residual_mode = :bind,   init_scale = 1f0,   bias_magnitude = 1f0,  recenter = false),
    (name = "bias4",     residual_mode = :bind,   init_scale = 1f0,   bias_magnitude = 4f0,  recenter = false),
    (name = "bias16",    residual_mode = :bind,   init_scale = 1f0,   bias_magnitude = 16f0, recenter = false),
    (name = "wscale0.1", residual_mode = :bind,   init_scale = 0.1f0, bias_magnitude = 1f0,  recenter = false),
    (name = "rezero",    residual_mode = :rezero, init_scale = 1f0,   bias_magnitude = 1f0,  recenter = false),
    (name = "recenter",  residual_mode = :bind,   init_scale = 1f0,   bias_magnitude = 1f0,  recenter = true),
]

"""
ReZero-focused arms. Tests the warmup hypothesis (α₀ = 0 vs warm-started) and
whether combining the learnable gate with a small weight init (γ=0.1) helps,
against the non-gated γ=0.1 reference (prior winner) and the baseline.
"""
const REZERO_ARMS = [
    (name = "baseline",     residual_mode = :bind,   init_scale = 1f0,   bias_magnitude = 1f0, recenter = false, alpha0 = 0f0),
    (name = "wscale0.1",    residual_mode = :bind,   init_scale = 0.1f0, bias_magnitude = 1f0, recenter = false, alpha0 = 0f0),
    (name = "rezero_a0",    residual_mode = :rezero, init_scale = 1f0,   bias_magnitude = 1f0, recenter = false, alpha0 = 0f0),
    (name = "rezero_warm",  residual_mode = :rezero, init_scale = 1f0,   bias_magnitude = 1f0, recenter = false, alpha0 = 0.1f0),
    (name = "rezero_g",     residual_mode = :rezero, init_scale = 0.1f0, bias_magnitude = 1f0, recenter = false, alpha0 = 0f0),
    (name = "rezero_warm_g",residual_mode = :rezero, init_scale = 0.1f0, bias_magnitude = 1f0, recenter = false, alpha0 = 0.1f0),
]

"""
    main_residual_study(; path=:sequential, D=64, depths=DEFAULT_DEPTHS, seeds=1:1,
                        epochs=3, lr=1e-3, batchsize=128, n_train=6000, n_test=2000,
                        arms=RESIDUAL_ARMS, run_resnet=true, use_cuda=true,
                        outdir="results/residual_study")

For each intervention arm (bias-magnitude / weight-scale / ReZero / recenter),
sweep depth and record trained test accuracy + the init-time forward-drift
diagnostics (per-block displacement and cumulative drift). Also runs the
real-valued ResNet-MLP gold standard at the same depths. All phasor models use
`scan=true` (compile-once). Writes `intervention_summary.csv`,
`resnet_baseline.csv`, and figures.
"""
function main_residual_study(; path::Symbol = :sequential,
                             D::Int = 64,
                             depths = DEFAULT_DEPTHS,
                             seeds = 1:1,
                             epochs::Int = 3,
                             lr::Real = 1f-3,
                             batchsize::Int = 128,
                             n_train::Int = 6000,
                             n_test::Int = 2000,
                             arms = RESIDUAL_ARMS,
                             run_resnet::Bool = true,
                             use_cuda::Bool = true,
                             skip_plots::Bool = false,
                             outdir::String = "results/residual_study")
    dev = (use_cuda && CUDA.functional()) ? gpu_device() : cdev
    isdir(outdir) || mkpath(outdir)
    (Xtr, ytr), (Xte, yte) = load_subset(; n_train, n_test)
    train_batches = make_batches(Xtr, ytr, batchsize)
    test_batches  = make_batches(Xte, yte, batchsize)
    probe_x, probe_y = train_batches[1]

    @info "residual study" path D depths=collect(depths) seeds=collect(seeds) epochs n_train arms=[a.name for a in arms] device=string(dev)

    rows = NamedTuple[]
    alpha_rows = NamedTuple[]
    file = joinpath(outdir, "intervention_summary.csv")
    afile = joinpath(outdir, "alpha_profile.csv")
    for arm in arms, depth in depths, seed in seeds
        try
            a0 = armget(arm, :alpha0, 0f0)
            mode = armget(arm, :residual_mode, :bind)
            sc   = armget(arm, :init_scale, 1f0)
            bm   = armget(arm, :bias_magnitude, 1f0)
            rc   = armget(arm, :recenter, false)
            model = build_depth_model(path, D, depth; use_bias = true, use_residual = true,
                                      scan = true, residual_mode = mode, init_scale = sc,
                                      bias_magnitude = bm, recenter = rc, alpha0 = a0)
            ps, st = Lux.setup(Xoshiro(seed), model); ps = ps |> dev; st = st |> dev
            probe = grad_probe(model, ps, st, probe_x, probe_y, dev)
            fd = forward_drift(path, D, depth; dev, x = probe_x, seed,
                               residual_mode = mode, init_scale = sc, bias_magnitude = bm,
                               recenter = rc, alpha0 = a0)
            almult = armget(arm, :alpha_lr_mult, 1f0)
            ps, final_loss = train_config!(model, ps, st, train_batches, epochs, lr, dev; alpha_lr_mult = almult)
            test_acc = test_accuracy(model, ps, st, test_batches, dev)
            alphas = collect_alphas(ps)   # learned gate per block ([] for non-rezero)
            push!(rows, (; arm = arm.name, path, depth, seed,
                         residual_mode = mode, init_scale = sc, bias_magnitude = bm,
                         recenter = rc, alpha0 = a0, alpha_lr_mult = almult,
                         init_loss = probe.init_loss, final_loss, test_acc,
                         grad_ratio = probe.grad_ratio,
                         bdisp_mean = Float32(mean(fd.bdisp)), drift_final = fd.drift[end],
                         alpha_mean = isempty(alphas) ? NaN32 : mean(alphas),
                         alpha_max  = isempty(alphas) ? NaN32 : maximum(alphas)))
            for (bi, av) in enumerate(alphas)
                push!(alpha_rows, (; arm = arm.name, depth, seed, block_index = bi, alpha = av))
            end
            @info "arm" arm=arm.name depth seed test_acc=round(test_acc, digits=4) bdisp=round(mean(fd.bdisp), digits=4) drift=round(fd.drift[end], digits=4) alpha_mean=(isempty(alphas) ? NaN32 : round(mean(alphas), digits=3))
        catch err
            @error "arm config failed; skipping" arm=arm.name depth seed exception=(err, catch_backtrace())
        end
        write_csv(file, rows); write_csv(afile, alpha_rows)
        GC.gc(); dev === gpu_device() && CUDA.reclaim()
    end

    resnet_rows = run_resnet ?
        run_resnet_baseline(; D, depths, seeds, epochs, lr, batchsize, n_train, n_test, dev, outdir) :
        NamedTuple[]

    skip_plots || make_residual_plots(rows, resnet_rows, outdir)
    @info "residual study done" outdir
    return (; arms = rows, resnet = resnet_rows)
end

"Plot test_acc vs depth per arm (with ResNet reference) and init drift vs depth."
function make_residual_plots(rows, resnet_rows, outdir)
    PLOTS_OK || (@info "Plots unavailable; skipping residual figures"; return)
    isempty(rows) && return
    arms = unique(getfield.(rows, :arm))
    # test accuracy vs depth, one curve per arm + ResNet reference
    plt = Plots.plot(; xlabel = "depth", ylabel = "test accuracy",
                     title = "residual interventions: test acc vs depth",
                     legend = :outertopright)
    for a in arms
        sub = sort(filter(r -> r.arm == a, rows), by = r -> r.depth)
        Plots.plot!(plt, getfield.(sub, :depth), getfield.(sub, :test_acc); label = a, marker = :circle, lw = 2)
    end
    if !isempty(resnet_rows)
        rs = sort(resnet_rows, by = r -> r.depth)
        Plots.plot!(plt, getfield.(rs, :depth), getfield.(rs, :test_acc);
                    label = "resnet (real)", ls = :dash, lw = 3, color = :black)
    end
    Plots.savefig(plt, joinpath(outdir, "intervention_test_acc.png"))

    # init drift (cumulative, at deepest) vs depth per arm — log y
    plt2 = Plots.plot(; xlabel = "depth", ylabel = "init cumulative drift (π units)",
                      title = "init forward drift vs depth", legend = :outertopright)
    for a in arms
        sub = sort(filter(r -> r.arm == a, rows), by = r -> r.depth)
        Plots.plot!(plt2, getfield.(sub, :depth), getfield.(sub, :drift_final); label = a, marker = :circle, lw = 2)
    end
    Plots.savefig(plt2, joinpath(outdir, "intervention_drift.png"))

    # per-block displacement at init vs depth per arm
    plt3 = Plots.plot(; xlabel = "depth", ylabel = "mean per-block displacement (π units)",
                      title = "init branch displacement vs depth", legend = :outertopright)
    for a in arms
        sub = sort(filter(r -> r.arm == a, rows), by = r -> r.depth)
        Plots.plot!(plt3, getfield.(sub, :depth), getfield.(sub, :bdisp_mean); label = a, marker = :circle, lw = 2)
    end
    Plots.savefig(plt3, joinpath(outdir, "intervention_bdisp.png"))

    # learned ReZero gate (mean α) vs depth — only arms that have it
    αarms = filter(a -> any(r -> r.arm == a && !isnan(r.alpha_mean), rows), arms)
    if !isempty(αarms)
        plt4 = Plots.plot(; xlabel = "depth", ylabel = "learned mean α (per block)",
                          title = "ReZero gate opening vs depth", legend = :outertopright)
        for a in αarms
            sub = sort(filter(r -> r.arm == a && !isnan(r.alpha_mean), rows), by = r -> r.depth)
            Plots.plot!(plt4, getfield.(sub, :depth), getfield.(sub, :alpha_mean); label = a, marker = :circle, lw = 2)
        end
        Plots.savefig(plt4, joinpath(outdir, "intervention_alpha.png"))
    end
    @info "saved residual figures" outdir
end

if abspath(PROGRAM_FILE) == @__FILE__
    main_depth_sweep()
end
