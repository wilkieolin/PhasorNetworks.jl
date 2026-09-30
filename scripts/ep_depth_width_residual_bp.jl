#!/usr/bin/env julia
#
# scripts/ep_depth_width_residual_bp.jl
# Phase 1: Backprop validation of ResidualBlock for deep phasor networks
#
# Tests whether ResidualBlock with v_bind skip + ReZero gate enables
# training at depth >= 3 using standard Zygote backprop.
#
# Usage:
#   julia --project=scripts scripts/ep_depth_width_residual_bp.jl
#   julia --project=scripts -e 'include("scripts/ep_depth_width_residual_bp.jl"); main_residual_bp_sweep(depths=[1,2,3,4,5], widths=[64,256], seeds=1:1, epochs=2)'

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
Pkg.activate(joinpath(repo_root, "scripts"))

using PhasorNetworks, Lux, Zygote, Optimisers, CUDA, LuxCUDA
using OneHotArrays: onehotbatch
using Random: Xoshiro, AbstractRNG
using Statistics: mean, std
using CSV, DataFrames
using Printf

# Load Plots at include time
const PLOTS_OK = try
    @eval using Plots
    true
catch err
    @warn "Plots unavailable; figures will be skipped" exception = err
    false
end

const DEFAULT_DEPTHS = [1, 2, 3, 4, 5]
const DEFAULT_WIDTHS = [64, 256, 1024]
const cdev = cpu_device()

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

"Split `(28,28,N)` images + labels into a vector of `(x_batch, y_batch)`."
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
# Model: ResidualBlock with ReZero gate
# ---------------------------------------------------------------------

"glorot weight init scaled by γ (γ<1 ⇒ smaller branch output at init)."
scaled_glorot(γ::Real) = (rng, dims...) -> Float32(γ) .* Lux.glorot_uniform(rng, dims...)

"complex bias init of magnitude m on the +real axis (large m ⇒ output phase→0)."
bias_mag(m::Real) = (rng, dims) -> ComplexF32(m) .* ones(ComplexF32, dims)

"""
    build_residual_model(D, depth; use_bias, gate, alpha0, branch_init_scale) -> Lux.Chain

Build encoder + `depth` ResidualBlock middle layers + Codebook readout.
`depth` = number of D=>D ResidualBlock layers (total phasor layers = depth + 1).
"""
function build_residual_model(D::Int, depth::Int;
                              use_bias::Bool = true,
                              gate::Symbol = :rezero,
                              alpha0::Real = 0.1f0,
                              branch_init_scale::Real = 0.1f0)
    enc_bias_kw = use_bias ? (use_bias = true, init_bias = bias_mag(1f0)) : (use_bias = false,)
    
    encoder = Chain(
        FlattenLayer(),
        LayerNorm((28^2,)),
        x -> Phase.(tanh.(x)),
        PhasorDense(28^2 => D, normalize_to_unit_circle; enc_bias_kw...)
    )
    
    # Middle blocks: ResidualBlock with ReZero
    if depth == 0
        middle = identity
    else
        blocks = []
        for _ in 1:depth
            rb = ResidualBlock((D, D), normalize_to_unit_circle;
                               gate = gate,
                               alpha0 = alpha0,
                               branch_init_scale = branch_init_scale,
                               use_bias = use_bias,
                               init_weight = scaled_glorot(branch_init_scale),
                               init_bias = bias_mag(1f0))
            push!(blocks, rb)
        end
        middle = Chain(blocks...)
    end
    
    readout = Codebook(D => 10)
    
    return Chain(encoder, middle, readout)
end

# ---------------------------------------------------------------------
# Loss / metrics
# ---------------------------------------------------------------------

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
        preds = predict(cdev(yp), :similarity)
        correct += sum(preds .== (y .+ 1))
        total   += length(y)
    end
    return correct / total
end

# ---------------------------------------------------------------------
# Gradient-pass-through probe (at init)
# ---------------------------------------------------------------------

"Recursively collect L2 norms of every `:weight` leaf in a gradient tree."
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
_walk!(norms, g::AbstractVector{<:Number}) = nothing
_walk!(norms, g::AbstractVector) = (for v in g; _walk!(norms, v); end)
_walk!(norms, g) = nothing

"Collect learned ReZero gate α (scalar per block) from param tree."
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
per-phasor-layer weight-gradient norm profile.
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
# Forward drift diagnostics (init-only, no AD)
# ---------------------------------------------------------------------

"Mean circular distance (in π units, ∈[0,1]) between two phase arrays."
circ_dist(a, b) = Float32(mean(abs.(Float32.(v_bind(a, .- b)))))

"""
    forward_drift(D, depth; dev, x, seed, gate, alpha0, branch_init_scale) -> (drift, bdisp)

Build encoder + `depth` middle blocks at init and run them stepwise on `x`.
`bdisp[k]` = per-block displacement (circular dist between consecutive reps).
`drift[k]` = cumulative circular distance from post-encoder rep.
"""
function forward_drift(D::Int, depth::Int; dev, x, seed::Int = 1,
                       gate::Symbol = :rezero, alpha0::Real = 0.1f0,
                       branch_init_scale::Real = 0.1f0)
    rng = Xoshiro(seed)
    
    # Encoder
    enc_bias_kw = (use_bias = true, init_bias = bias_mag(1f0))
    encoder = Chain(
        FlattenLayer(),
        LayerNorm((28^2,)),
        x -> Phase.(tanh.(x)),
        PhasorDense(28^2 => D, normalize_to_unit_circle; enc_bias_kw...)
    )
    ps_enc, st_enc = Lux.setup(rng, encoder)
    r0, _ = encoder(x |> dev, ps_enc |> dev, st_enc |> dev)
    
    drift = Float32[]; bdisp = Float32[]
    r = r0
    for _ in 1:depth
        rb = ResidualBlock((D, D), normalize_to_unit_circle;
                           gate = gate,
                           alpha0 = alpha0,
                           branch_init_scale = branch_init_scale,
                           use_bias = true,
                           init_weight = scaled_glorot(branch_init_scale),
                           init_bias = bias_mag(1f0))
        psb, stb = Lux.setup(rng, rb)
        rprev = r
        r, _ = rb(r, psb |> dev, stb |> dev)
        push!(bdisp, circ_dist(r, rprev))
        push!(drift, circ_dist(r, r0))
    end
    return (; drift, bdisp)
end

# ---------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------

"Boost ReZero gate α learning rate (mirrors package's _adjust_alpha_lr!)."
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
    opt_state = Optimisers.setup(Optimisers.Adam(Float32(lr)), ps)
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
# Sweep driver
# ---------------------------------------------------------------------

function run_sweep(; D, depths, seeds, epochs, lr, batchsize,
                   n_train, n_test, dev, outdir,
                   gate = :rezero, alpha0 = 0.1f0, branch_init_scale = 0.1f0)
    (Xtr, ytr), (Xte, yte) = load_subset(; n_train, n_test)
    train_batches = make_batches(Xtr, ytr, batchsize)
    test_batches  = make_batches(Xte, yte, batchsize)
    probe_x, probe_y = train_batches[1]

    summary_rows = NamedTuple[]
    profile_rows = NamedTuple[]
    drift_rows = NamedTuple[]
    alpha_rows = NamedTuple[]

    summary_file = joinpath(outdir, "residual_bp_summary.csv")
    profile_file = joinpath(outdir, "residual_bp_gradprofile.csv")
    drift_file = joinpath(outdir, "residual_bp_drift.csv")
    alpha_file = joinpath(outdir, "residual_bp_alphas.csv")

    for depth in depths, seed in seeds
        try
            rng = Xoshiro(seed)
            model = build_residual_model(D, depth; use_bias = true, gate = gate,
                                         alpha0 = alpha0, branch_init_scale = branch_init_scale)
            ps, st = Lux.setup(rng, model)
            ps = ps |> dev; st = st |> dev
            n_params = Lux.parameterlength(model)

            # Init gradient probe
            probe = grad_probe(model, ps, st, probe_x, probe_y, dev)

            # Forward drift at init
            fd = forward_drift(D, depth; dev, x = probe_x, seed,
                               gate = gate, alpha0 = alpha0, branch_init_scale = branch_init_scale)

            # Train
            almult = (gate === :rezero) ? 5f0 : 1f0  # higher LR for ReZero gate
            ps, final_loss = train_config!(model, ps, st, train_batches, epochs, lr, dev; alpha_lr_mult = almult)
            test_acc = test_accuracy(model, ps, st, test_batches, dev)

            # Collect learned alphas
            alphas = collect_alphas(ps)

            push!(summary_rows, (; D, depth, seed,
                                 init_loss = probe.init_loss, final_loss, test_acc,
                                 grad_encoder = probe.grad_encoder, grad_last = probe.grad_last,
                                 grad_ratio = probe.grad_ratio, grad_min = probe.grad_min,
                                 grad_max = probe.grad_max, n_params,
                                 bdisp_mean = Float32(mean(fd.bdisp)), drift_final = fd.drift[end],
                                 alpha_mean = isempty(alphas) ? NaN32 : mean(alphas),
                                 alpha_max  = isempty(alphas) ? NaN32 : maximum(alphas)))
            for (li, gn) in enumerate(probe.profile)
                push!(profile_rows, (; D, depth, seed, layer_index = li, grad_norm = gn))
            end
            for (bi, (bd, dr)) in enumerate(zip(fd.bdisp, fd.drift))
                push!(drift_rows, (; D, depth, seed, block_index = bi, bdisp = bd, drift = dr))
            end
            for (bi, av) in enumerate(alphas)
                push!(alpha_rows, (; D, depth, seed, block_index = bi, alpha = av))
            end

            @info "config" D depth seed init_loss=round(probe.init_loss,digits=4) final_loss=round(final_loss,digits=4) test_acc=round(test_acc,digits=4) grad_ratio=round(probe.grad_ratio,digits=3) bdisp=round(mean(fd.bdisp),digits=4) drift=round(fd.drift[end],digits=4) alpha_mean=(isempty(alphas) ? NaN32 : round(mean(alphas),digits=3))
        catch err
            @error "config failed; skipping" D depth seed exception=(err, catch_backtrace())
        end

        write_csv(summary_file, summary_rows)
        write_csv(profile_file, profile_rows)
        write_csv(drift_file, drift_rows)
        write_csv(alpha_file, alpha_rows)
        GC.gc()
        dev === gpu_device() && CUDA.reclaim()
    end

    return summary_rows
end

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

"Aggregate mean over seeds → (depths, mean) for a metric."
function _curve(rows, D, metric)
    sub = filter(r -> r.D == D, rows)
    ds = sort(unique(getfield.(sub, :depth)))
    ms = Float64[]; ss = Float64[]
    for d in ds
        vals = Float64[getfield(r, metric) for r in sub if r.depth == d && !isnan(getfield(r, metric))]
        push!(ms, isempty(vals) ? NaN : mean(vals))
        push!(ss, length(vals) > 1 ? std(vals) : 0.0)
    end
    return ds, ms, ss
end

function make_plots(rows, outdir)
    PLOTS_OK || (@info "Plots unavailable; skipping figures"; return)
    isempty(rows) && return

    specs = [
        (:test_acc, "test accuracy", :identity),
        (:final_loss, "final loss", :identity),
        (:grad_ratio, "encoder/last ∇ ratio (log10)", :log10),
        (:grad_encoder, "encoder ∇ norm", :log10),
        (:grad_last, "last layer ∇ norm", :log10),
        (:bdisp_mean, "mean per-block displacement (π units)", :identity),
        (:drift_final, "cumulative init drift (π units)", :identity),
        (:alpha_mean, "mean ReZero α", :identity),
    ]

    for (metric, ylab, yscale) in specs
        plt = Plots.plot(; xlabel = "depth (ResidualBlock layers)", ylabel = ylab,
                         title = "ResidualBlock backprop: $(ylab) vs depth", legend = :outertopright,
                         yscale = yscale)
        any_data = false
        for D in unique(getfield.(rows, :D))
            ds, ms, ss = _curve(rows, D, metric)
            (isempty(ds) || all(isnan, ms)) && continue
            any_data = true
            if metric === :grad_ratio
                Plots.plot!(plt, ds, ms; label = "D=$D", marker = :circle, lw = 2)
            else
                Plots.plot!(plt, ds, ms; ribbon = ss, label = "D=$D", marker = :circle, lw = 2)
            end
        end
        any_data || continue
        f = joinpath(outdir, "residual_bp_$(metric).png")
        Plots.savefig(plt, f)
        @info "saved figure" f
    end

    # Alpha profile per block index (for deepest depth) - read from CSV
    alpha_file = joinpath(outdir, "residual_bp_alphas.csv")
    if isfile(alpha_file)
        alpha_df = CSV.read(alpha_file, DataFrame)
        max_depth = maximum(getfield.(rows, :depth))
        alpha_df = filter(r -> r.depth == max_depth, alpha_df)
        if !isempty(alpha_df) && any(!isnan, alpha_df.alpha)
            plt_a = Plots.plot(; xlabel = "block index", ylabel = "mean α",
                               title = "ReZero gate α vs block index (depth=$max_depth)", legend = :outertopright)
            for D in unique(alpha_df.D)
                sub = filter(r -> r.D == D, alpha_df)
                bis = sort(unique(sub.block_index))
                avs = Float64[mean(filter(r -> r.D == D && r.block_index == bi, sub).alpha) for bi in bis]
                Plots.plot!(plt_a, bis, avs; label = "D=$D", marker = :circle, lw = 2)
            end
            Plots.savefig(plt_a, joinpath(outdir, "residual_bp_alpha_profile.png"))
            @info "saved alpha profile figure"
        end
    end
end

# ---------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------

"""
    main_residual_bp_sweep(; D=256, depths=DEFAULT_DEPTHS, widths=DEFAULT_WIDTHS, seeds=1:3,
                            epochs=5, lr=1e-3, batchsize=128, n_train=10_000, n_test=2_000,
                            use_cuda=true, gate=:rezero, alpha0=0.1f0, branch_init_scale=0.1f0,
                            outdir="results/ep_residual_bp_sweep")

Run the depth × width × seed sweep with ResidualBlock + ReZero using Zygote backprop.
"""
function main_residual_bp_sweep(; D::Int = 256,
                                depths = DEFAULT_DEPTHS,
                                widths = DEFAULT_WIDTHS,
                                seeds = 1:3,
                                epochs::Int = 5,
                                lr::Real = 1f-3,
                                batchsize::Int = 128,
                                n_train::Int = 10_000,
                                n_test::Int = 2_000,
                                use_cuda::Bool = true,
                                gate::Symbol = :rezero,
                                alpha0::Real = 0.1f0,
                                branch_init_scale::Real = 0.1f0,
                                outdir::String = "results/ep_residual_bp_sweep")
    dev = (use_cuda && CUDA.functional()) ? gpu_device() : cdev
    isdir(outdir) || mkpath(outdir)

    @info "ResidualBlock backprop sweep" D depths=collect(depths) widths=collect(widths) seeds=collect(seeds) epochs lr batchsize n_train n_test gate alpha0 branch_init_scale device=string(dev)

    all_rows = NamedTuple[]
    for W in widths
        @info "=== Width D=$W ==="
        rows = run_sweep(; D=W, depths=depths, seeds=seeds, epochs=epochs, lr=lr, batchsize=batchsize,
                         n_train=n_train, n_test=n_test, dev=dev, outdir=outdir,
                         gate=gate, alpha0=alpha0, branch_init_scale=branch_init_scale)
        append!(all_rows, rows)
    end

    make_plots(all_rows, outdir)
    @info "done" outdir
    return all_rows
end

if abspath(PROGRAM_FILE) == @__FILE__
    main_residual_bp_sweep()
end