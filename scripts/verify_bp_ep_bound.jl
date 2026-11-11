#!/usr/bin/env julia
# scripts/verify_bp_ep_bound.jl — measure the constants of Supplementary Proof B
# on a backprop-trained phasor chain.
#
# Trains a PhasorDense chain with backpropagation, settles the same weights as
# an EP network, and reports the three quantities the appendix turns on:
#
#   1. the relative state deviation  δ̂_ℓ = ||z*_ℓ - z^ff_ℓ|| / sqrt(d_ℓ);
#   2. the drive moduli and the resulting gain matrix G, whose spectral radius
#      must be below 1 for the perturbation bound to apply;
#   3. the margin distribution and the per-sample accuracy bound.
#
# Both passes use the HARD projection u(g) = g/|g| — `normalize_to_unit_circle`
# for the feedforward states, and the library's `phasor_settle` at
# `project = :hard` for the EP fixed point. An earlier version of this script
# substituted the soft projection g/sqrt(|g|^2 + ε^2) with ε = 0.1 and hand-
# rolled its own settle loop; that measured a model the package does not run,
# and its per-layer C_ℓ/D_ℓ recurrence came from a superseded form of the
# appendix (the bias term cancels identically, and the Lipschitz constant used
# there was wrong). Neither survives here.

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

using PhasorNetworks, Lux, LinearAlgebra, Statistics, Random, Optimisers, Zygote, OneHotArrays
using Random: Xoshiro
using LinearAlgebra: svdvals, norm, dot, eigvals
using CSV, DataFrames, Printf
using PhasorNetworks: cpu_device, select_device, NullCost

const SEED     = 42
const HID      = 256
const DOUT     = 64
const SCALE    = 0.4f0
const EPOCHS   = 5
const LR       = 0.001
const BATCH    = 128
const T_SETTLE = 3000
const DT       = 0.5f0
const NTEST    = 500

const OUT = joinpath(repo_root, "results", "bp_ep_bound_verification")
mkpath(OUT)

const CDEV = cpu_device()
const DEV  = select_device(:cuda)

"""
    encode_phase(imgs)

Per-image standardisation of raw pixels into `Phase` values.

# Arguments
- `imgs`: `(28, 28, N)` `Float32` array of images.

# Returns
`(784, N)` array of `Phase` in `[-1, 1]`.
"""
function encode_phase(imgs::AbstractArray{Float32,3})
    n = size(imgs, 3)
    flat = reshape(imgs, :, n)
    mu = mean(flat; dims = 1)
    sigma = std(flat; dims = 1) .+ 1.0f-6
    return Phase.(0.5f0 .* tanh.((flat .- mu) ./ sigma))
end

"""
    build_chain(rng, dims; scale = SCALE)

`PhasorDense` chain of arbitrary depth, with hard projection and complex bias.

# Arguments
- `dims`: layer widths from input to output, e.g. `[784, 256, 64]` for the
  two-layer chain of the main text.

# Returns
`(chain, ps, st)`, with every weight matrix scaled by `scale` at
initialisation.
"""
function build_chain(rng::Xoshiro, dims::AbstractVector{Int}; scale::Float32 = SCALE)
    layers = [PhasorDense(dims[i] => dims[i + 1], normalize_to_unit_circle,
                          use_bias = true) for i in 1:length(dims)-1]
    chain = Chain(layers...)
    ps, st = Lux.setup(rng, chain)
    names = keys(ps)
    ps = NamedTuple{names}(map(k -> merge(ps[k], (weight = scale .* ps[k].weight,)),
                               names))
    return chain, ps, st
end

"""
    layer_bias(ps_l)

Complex bias of one `PhasorDense` layer, or `nothing` when `use_bias = false`.
"""
function layer_bias(ps_l)
    return haskey(ps_l, :bias_real) ?
        ps_l.bias_real .+ 1.0f0im .* ps_l.bias_imag : nothing
end

"""
    feedforward_states(ps, layer_keys, z0)

Per-layer complex states of the feedforward (backpropagation) pass.

# Arguments
- `ps`: chain parameters on the CPU.
- `layer_keys`: ordered layer names.
- `z0`: `(d_0, B)` complex input on the unit torus.

# Returns
Vector of `(d_ℓ, B)` complex matrices, one per layer.

# Implementation
Mirrors the phase-mode forward of `PhasorDense` — drive `W*z + b`, then
`normalize_to_unit_circle` — rather than calling `Lux.apply`, so that the
hidden states are available and not just the output.
"""
function feedforward_states(ps, layer_keys, z0::AbstractMatrix{ComplexF32})
    states = Matrix{ComplexF32}[]
    z = z0
    for key in layer_keys
        b = layer_bias(ps[key])
        g = ps[key].weight * z
        g = b === nothing ? g : g .+ b
        z = normalize_to_unit_circle(g)
        push!(states, z)
    end
    return states
end

"""
    ep_drives(ps, layer_keys, z0, states)

Drives `g_ℓ = W_ℓ z_{ℓ-1} + Wᵀ_{ℓ+1} z_{ℓ+1} + b_ℓ` entering each layer.

# Arguments
- `states`: per-layer complex states to evaluate the drive at.
- `z0`: clamped input.

# Returns
Vector of `(d_ℓ, B)` complex drive matrices. Passing the feedforward states
with the feedback term dropped is not the same thing — use
[`feedforward_drives`](@ref) for that.
"""
function ep_drives(ps, layer_keys, z0::AbstractMatrix{ComplexF32}, states)
    n = length(layer_keys)
    drives = Matrix{ComplexF32}[]
    for l in 1:n
        key = layer_keys[l]
        z_in = l == 1 ? z0 : states[l - 1]
        b = layer_bias(ps[key])
        g = ps[key].weight * z_in
        g = b === nothing ? g : g .+ b
        if l < n
            g = g .+ transpose(ps[layer_keys[l + 1]].weight) * states[l + 1]
        end
        push!(drives, g)
    end
    return drives
end

"""
    feedforward_drives(ps, layer_keys, z0)

Drives of the feedforward pass, i.e. [`ep_drives`](@ref) without feedback.
"""
function feedforward_drives(ps, layer_keys, z0::AbstractMatrix{ComplexF32})
    drives = Matrix{ComplexF32}[]
    z = z0
    for key in layer_keys
        b = layer_bias(ps[key])
        g = ps[key].weight * z
        push!(drives, b === nothing ? g : g .+ b)
        z = normalize_to_unit_circle(drives[end])
    end
    return drives
end

"""
    gain_matrix(rho, mu)

Gain matrix `G` of Supplementary Proof B.

# Arguments
- `rho`: per-layer weight spectral norms `||W_ℓ||_2`.
- `mu`: per-layer drive-modulus bound. `Float32` drive statistics are
  accepted and promoted; the eigenvalue solve runs in `Float64`.

# Returns
`L × L` matrix with `G[ℓ, ℓ-1] = rho[ℓ]/mu[ℓ]` and
`G[ℓ, ℓ+1] = rho[ℓ+1]/mu[ℓ]`.

# Implementation
Both `rho` and `mu` scale linearly under a common rescaling of the weights,
so `G` is invariant under it — that invariance is the point of the
construction and is worth preserving in any edit here.
"""
function gain_matrix(rho::AbstractVector{<:Real}, mu::AbstractVector{<:Real})
    n = length(rho)
    @assert length(mu) == n "rho and mu must have one entry per layer"
    r = Float64.(rho)
    m = Float64.(mu)
    G = zeros(Float64, n, n)
    for l in 1:n
        if l > 1
            G[l, l - 1] = r[l] / m[l]
        end
        if l < n
            G[l, l + 1] = r[l + 1] / m[l]
        end
    end
    return G
end

spectral_radius(A::AbstractMatrix) = maximum(abs.(eigvals(A)))

"""
    codebook_margin(logits, y)

Feedforward margin: true-class logit minus the best competing logit.
"""
function codebook_margin(logits::AbstractMatrix, y::Integer)
    k = size(logits, 1)
    return logits[y] - maximum(@view logits[setdiff(1:k, y)])
end

# ============================================================
# Driver
# ============================================================

"""
    run_verification(dims; seed, epochs, scale, t_settle, dt, ntest, verbose)

Train one chain with backpropagation, settle it as an EP network, and return
the constants of Supplementary Proof B.

# Arguments
- `dims`: layer widths from input to output, e.g. `[784, 256, 64]`.
- `ntest`: number of test images the EP fixed point is computed for.

# Returns
A `NamedTuple` with the per-layer weight norms, drive statistics, gain-matrix
spectral radii, relative deviations, accuracies, margin summary and the
per-sample accuracy bound, plus the per-sample table as `:per_sample`.

# Implementation
Both passes use the hard projection: `normalize_to_unit_circle` for the
feedforward states and `phasor_settle(..., project = :hard)` for the EP fixed
point. `settle_resid` is the fixed-point residual and should be checked before
any other field is believed -- a deep chain may fail to settle within
`t_settle`, in which case the deviations describe a non-equilibrium state.
"""
function run_verification(dims::AbstractVector{Int};
                          seed::Int = SEED, epochs::Int = EPOCHS,
                          scale::Float32 = SCALE, t_settle::Int = T_SETTLE,
                          dt::Float32 = DT, ntest::Int = NTEST,
                          verbose::Bool = true)
    d_out = dims[end]
    rng = Xoshiro(seed)
    chain, ps, st = build_chain(rng, dims; scale = scale)
    codes = ComplexF32.(angle_to_complex(orthogonal_codes(rng, d_out, 10)))

    tr = fashion_mnist_data(:train)
    te = fashion_mnist_data(:test)
    ntr = length(tr.targets)
    x_train = encode_phase(Float32.(tr.features))
    y_train = Int.(tr.targets) .+ 1
    x_test = encode_phase(Float32.(te.features[:, :, 1:ntest]))
    y_test = Int.(te.targets[1:ntest]) .+ 1

    ps = ps |> DEV
    st = st |> DEV
    codes_dev = codes |> DEV
    opt_state = Optimisers.setup(Optimisers.Adam(Float32(LR)), ps)

    for epoch in 1:epochs
        losses = Float64[]
        perm = randperm(Xoshiro(seed + epoch), ntr)
        for i in 1:BATCH:ntr
            idx = perm[i:min(i + BATCH - 1, ntr)]
            xb = x_train[:, idx] |> DEV
            yb = y_train[idx]
            loss_fn = p -> begin
                z_out, _ = Lux.apply(chain, xb, p, st)
                z_c = ComplexF32.(angle_to_complex(z_out))
                logits = similarity_outer(z_c, codes_dev)
                y_onehot = onehotbatch(yb .- 1, 0:9) |> DEV
                log_probs = logits .- log.(sum(exp.(logits); dims = 1))
                return -mean(sum(y_onehot .* log_probs; dims = 1))
            end
            lossval, grads = Zygote.withgradient(loss_fn, ps)
            push!(losses, lossval)
            opt_state, ps = Optimisers.update(opt_state, ps, grads[1])
        end
        verbose && @printf("    epoch %d  loss %.4f\n", epoch, mean(losses))
        verbose && flush(stdout)
    end

    ps = ps |> CDEV
    st = st |> CDEV
    layer_keys = collect(keys(ps))
    n_layers = length(layer_keys)
    dims_out = [size(ps[k].weight, 1) for k in layer_keys]
    rho = [maximum(svdvals(Float64.(ps[k].weight))) for k in layer_keys]

    z0 = ComplexF32.(angle_to_complex(x_test))
    ff = feedforward_states(ps, layer_keys, z0)
    ep = phasor_settle(chain, ps, st, x_test, NullCost(), 0.0f0;
                       T = t_settle, dt = dt, K_mode = :zero,
                       project = :hard, carrier = nothing)
    resid = maximum(maximum(abs.(normalize_to_unit_circle(g) .- ep[l]))
                    for (l, g) in enumerate(ep_drives(ps, layer_keys, z0, ep)))

    rel_dev = [[norm(ep[l][:, i] - ff[l][:, i]) / sqrt(dims_out[l]) for i in 1:ntest]
               for l in 1:n_layers]
    phase_removed = mean(norm(ep[end][:, i] -
                              cis(angle(dot(ff[end][:, i], ep[end][:, i]))) * ff[end][:, i]) /
                         sqrt(dims_out[end]) for i in 1:ntest)

    g_ff = feedforward_drives(ps, layer_keys, z0)
    g_ep = ep_drives(ps, layer_keys, z0, ep)
    both = [vcat(vec(abs.(g_ff[l])), vec(abs.(g_ep[l]))) for l in 1:n_layers]
    mu_min = [minimum(b) for b in both]
    mu_med = [median(b) for b in both]

    logits_ff = real.(adjoint(codes) * ff[end]) ./ Float32(d_out)
    logits_ep = real.(adjoint(codes) * ep[end]) ./ Float32(d_out)
    pred_ff = [argmax(@view logits_ff[:, i]) for i in 1:ntest]
    pred_ep = [argmax(@view logits_ep[:, i]) for i in 1:ntest]
    acc_bp = mean(pred_ff .== y_test)
    acc_ep = mean(pred_ep .== y_test)
    margin = [codebook_margin(@view(logits_ff[:, i:i]), y_test[i]) for i in 1:ntest]
    shift = 2 .* rel_dev[end]
    at_risk = count(i -> margin[i] > 0 && margin[i] <= shift[i], 1:ntest)

    per_sample = DataFrame(idx = 1:ntest, y = y_test, pred_ff = pred_ff,
                           pred_ep = pred_ep, match = pred_ff .== pred_ep,
                           margin = margin, shift = shift)
    for l in 1:n_layers
        per_sample[!, Symbol("rel_dev_$l")] = rel_dev[l]
    end

    return (dims = dims, depth = n_layers, width = length(dims) > 2 ? dims[2] : dims[2],
            rho = rho, rho_ratio = n_layers > 1 ? rho[end] / rho[1] : 1.0,
            mu_min = mu_min, mu_median = mu_med,
            rho_G_min = spectral_radius(gain_matrix(rho, mu_min)),
            rho_G_median = spectral_radius(gain_matrix(rho, mu_med)),
            rel_dev_mean = [mean(d) for d in rel_dev],
            rel_dev_max = [maximum(d) for d in rel_dev],
            out_dev_mean = mean(rel_dev[end]), out_dev_max = maximum(rel_dev[end]),
            phase_removed = phase_removed,
            acc_bp = acc_bp, acc_ep = acc_ep, drop = acc_bp - acc_ep,
            agreement = mean(pred_ff .== pred_ep),
            margin_median = median(margin), shift_mean = mean(shift),
            bound = at_risk / ntest, at_risk = at_risk,
            settle_resid = resid, per_sample = per_sample)
end

"""
    report(r)

Print one `run_verification` result in the layout of Supplementary Proof B.
"""
function report(r)
    @printf("\n=== %s ===\n", join(r.dims, "->"))
    @printf("  settle residual %.2e   ||W_l|| = %s   ratio %.3f\n",
            r.settle_resid, string(round.(r.rho, digits = 3)), r.rho_ratio)
    @printf("  (1) rel deviation mean %s   max %s   [trivial max 2]\n",
            string(round.(r.rel_dev_mean, digits = 4)),
            string(round.(r.rel_dev_max, digits = 4)))
    @printf("      output layer, global phase removed: %.4f\n", r.phase_removed)
    @printf("  (2) mu min %s  median %s   rho(G) min %.1f  median %.3f  [need < 1]\n",
            string(round.(r.mu_min, digits = 4)), string(round.(r.mu_median, digits = 3)),
            r.rho_G_min, r.rho_G_median)
    @printf("  (3) BP %.4f  EP %.4f  drop %.4f  agreement %.4f\n",
            r.acc_bp, r.acc_ep, r.drop, r.agreement)
    @printf("      median margin %.4f  mean displacement %.4f  bound %.4f  [%d at risk]\n",
            r.margin_median, r.shift_mean, r.bound, r.at_risk)
    flush(stdout)
end

if abspath(PROGRAM_FILE) == @__FILE__
    println("=== BP->EP bound verification (hard projection) ===")
    println("seed=$SEED  init scale=$SCALE  T_settle=$T_SETTLE  dt=$DT  n_test=$NTEST")
    result = run_verification([784, HID, DOUT])
    report(result)
    CSV.write(joinpath(OUT, "per_sample_$(SEED).csv"), result.per_sample)
    summary = DataFrame(layer = 1:result.depth, rho = result.rho,
                        mu_min = result.mu_min, mu_median = result.mu_median,
                        rel_dev_mean = result.rel_dev_mean,
                        rel_dev_max = result.rel_dev_max)
    CSV.write(joinpath(OUT, "summary_$(SEED).csv"), summary)
    println("\nwrote $(OUT)/per_sample_$(SEED).csv and summary_$(SEED).csv")
end
