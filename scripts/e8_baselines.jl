#!/usr/bin/env julia
# E8: matched baselines for the method comparison.
#
# Two arms that train in seconds, run at many seeds so their contribution to the
# pooled error is negligible next to lock-in EP's:
#
#   relu       bog-standard MLP, 784->256->64->10, ReLU + softmax, cross-entropy.
#              Same widths as the phasor net; only the readout differs, a linear
#              10-way head instead of the codebook similarity.
#   phasor_bp_flip
#              identical to phasor_bp, except each input dimension is offset by a
#              fixed random half-turn (0 or pi) drawn per seed. The encoder is
#              monotone in intensity, so background pixels all standardise to a
#              similar phase and ~28.5% of input power sits in a single common
#              mode (|R| = 0.52 measured over 4000 images); the flip drops that to
#              0.14%. This arm tests whether the phasor net's deficit against the
#              ReLU MLP is partly an encoding artefact. Costs nothing: no extra
#              parameters, one broadcast.
#   phasor_bp  the PhasorDense chain 784->256->64 with the codebook readout,
#              trained by ordinary feedforward backprop. NOT BPTT -- PhasorDense's
#              forward (src/network.jl:396) is a static complex matmul, so there
#              is no time axis to unroll. The settle appears only at evaluation.
#
# Protocol is matched to e6_basin_training.jl (60k train, 10k test, batch 256,
# 8 epochs) so these sit alongside lock-in EP at its best operating point,
# point B: eps=0.03, omega_p=0.02, i.e. omega_p/kappa=0.25.
#
#   JULIA_CUDA_HARD_MEMORY_LIMIT=8GiB julia --project=scripts scripts/e8_baselines.jl
#
# Env: E8_SEEDS (default 20), E8_EPOCHS (8), E8_ARMS ("relu,phasor_bp").

using Pkg
function find_repo_root(start_dir::String = pwd())
    dir = start_dir
    while !(isfile(joinpath(dir, "Project.toml")) && isdir(joinpath(dir, ".git")))
        parent = dirname(dir); parent == dir && error("no repo root"); dir = parent
    end
    return dir
end
repo_root = find_repo_root(@__DIR__); cd(repo_root); Pkg.activate(joinpath(repo_root, "scripts"))

using PhasorNetworks, Lux, MLUtils, Statistics, Random, Zygote, Optimisers, CUDA
using LinearAlgebra, CSV, DataFrames, Printf
using Random: Xoshiro
import PhasorNetworks: normalize_to_unit_circle

_envi(k, d) = parse(Int, get(ENV, k, string(d)))

const NSEEDS   = _envi("E8_SEEDS", 20)
const EPOCHS   = _envi("E8_EPOCHS", 8)
const ARMS     = strip.(split(get(ENV, "E8_ARMS", "relu,phasor_bp"), ','))
# Phase values are half-turns (angle_to_complex(x) = exp(i*pi*x)), so a sign flip
# of the phasor is an offset of exactly 1. Wrap back into [-1, 1) to keep values
# canonical.
wrap_ht(x) = mod.(x .+ 1f0, 2f0) .- 1f0
const BATCH    = 256           # matches e6_basin_training.jl
const LR       = 0.001
const HID, DOUT, SCALE = 256, 64, 0.4f0
const NTRAIN, NTEST = 60000, 10000
const USE_CUDA = CUDA.functional()

const OUT = joinpath(repo_root, "results", "e8_baselines"); mkpath(OUT)
const GITREV = try
    rev = strip(read(`git -C $(repo_root) rev-parse --short HEAD`, String))
    d   = read(`git -C $(repo_root) diff HEAD -- src`, String)
    isempty(strip(d)) ? rev : rev * "-d" * string(hash(d), base = 16)[1:8]
catch; "unknown" end

cdev = cpu_device(); gdev = gpu_device(); dev = USE_CUDA ? gdev : cdev
to_dev(x) = USE_CUDA ? x |> gdev : x

# ---- data. Both arms see the same per-image standardisation; the phasor arm
# additionally wraps it onto the circle, exactly as e3/e6 do.
function standardise(imgs::AbstractArray{Float32,3})
    flat = reshape(imgs, :, size(imgs, 3))
    μ = mean(flat; dims = 1); σ = std(flat; dims = 1) .+ 1f-6
    return (flat .- μ) ./ σ
end
println("Loading FashionMNIST...")
tr = fashion_mnist_data(:train); te = fashion_mnist_data(:test)
ntr = min(NTRAIN, length(tr.targets)); nte = min(NTEST, length(te.targets))
Str = standardise(Float32.(tr.features[:, :, 1:ntr]))
Ste = standardise(Float32.(te.features[:, :, 1:nte]))
ytr = Int.(tr.targets[1:ntr]) .+ 1
yte = Int.(te.targets[1:nte]) .+ 1
Htr = to_dev(0.5f0 .* tanh.(Str))              # phasor input, in half-turns
Hte = to_dev(0.5f0 .* tanh.(Ste))
Ptr = Phase.(Htr)                              # phasor arm input
Pte = Phase.(Hte)
Rtr = to_dev(Str)                              # relu arm input
Rte = to_dev(Ste)
onehot(y) = (m = zeros(Float32, 10, length(y)); for (i, c) in enumerate(y); m[c, i] = 1f0; end; to_dev(m))
Ytr = onehot(ytr)

logsoftmax_ce(logits, oh) = -mean(sum(oh .* (logits .- log.(sum(exp.(logits); dims = 1))); dims = 1))

# ---- arm 1: standard ReLU MLP -------------------------------------------
build_relu(rng) = begin
    m = Chain(Dense(784 => HID, relu), Dense(HID => DOUT, relu), Dense(DOUT => 10))
    ps, st = Lux.setup(rng, m)
    m, (USE_CUDA ? ps |> gdev : ps), (USE_CUDA ? st |> gdev : st)
end
relu_logits(m, ps, st, x) = first(Lux.apply(m, x, ps, st))

# ---- arm 2: PhasorDense + codebook, feedforward backprop ----------------
build_phasor(rng) = begin
    m = Chain(PhasorDense(784 => HID, normalize_to_unit_circle, use_bias = true),
              PhasorDense(HID => DOUT, normalize_to_unit_circle, use_bias = true))
    ps, st = Lux.setup(rng, m)
    ps = (layer_1 = merge(ps.layer_1, (weight = SCALE .* ps.layer_1.weight,)),
          layer_2 = merge(ps.layer_2, (weight = SCALE .* ps.layer_2.weight,)))
    m, (USE_CUDA ? ps |> gdev : ps), (USE_CUDA ? st |> gdev : st)
end
phasor_logits(m, ps, st, x, codes) =
    similarity_outer(ComplexF32.(angle_to_complex(first(Lux.apply(m, x, ps, st)))), codes)

function accuracy(logits_fn, X, y; chunk = 1024)
    correct = 0
    for i in 1:chunk:length(y)
        e = min(i + chunk - 1, length(y))
        lg = logits_fn(X[:, i:e]) |> cdev
        correct += sum([argmax(view(lg, :, b))[1] for b in 1:(e - i + 1)] .== y[i:e])
    end
    return correct / length(y)
end

rows = DataFrame(gitrev = String[], arm = String[], seed = Int[], epoch = Int[],
                 test_acc = Float64[], loss = Float64[], elapsed_s = Float64[],
                 epochs = Int[], batch = Int[], lr = Float64[], hidden = Int[],
                 output = Int[], ntrain = Int[])
csv_file = joinpath(OUT, "e8_baselines_$(GITREV).csv")
isfile(csv_file) || CSV.write(csv_file, rows)

for arm in ARMS, s in 1:NSEEDS
    seed = 1000 + s
    rng = Xoshiro(seed)
    t0 = time()
    if arm == "relu"
        model, ps, st = build_relu(rng)
        X, Xt = Rtr, Rte
        lossfn = (p, x, oh) -> logsoftmax_ce(relu_logits(model, p, st, x), oh)
        accfn = p -> accuracy(x -> relu_logits(model, p, st, x), Xt, yte)
    else
        model, ps, st = build_phasor(rng)
        codes = to_dev(ComplexF32.(angle_to_complex(orthogonal_codes(Xoshiro(seed), DOUT, 10))))
        if arm == "phasor_bp_flip"
            # one fixed half-turn offset per input dimension, drawn per seed
            off = to_dev(Float32.(rand(Xoshiro(seed + 31337), Bool, 784)))
            X, Xt = Phase.(wrap_ht(Htr .+ off)), Phase.(wrap_ht(Hte .+ off))
        else
            X, Xt = Ptr, Pte
        end
        lossfn = (p, x, oh) -> logsoftmax_ce(phasor_logits(model, p, st, x, codes), oh)
        accfn = p -> accuracy(x -> phasor_logits(model, p, st, x, codes), Xt, yte)
    end
    opt = Optimisers.setup(Optimisers.Adam(LR), ps)
    n = length(ytr); idx = collect(1:n)
    for epoch in 1:EPOCHS
        shuffle!(Xoshiro(seed + 7919 * epoch), idx)
        tot = 0.0; nb = 0
        for i in 1:BATCH:n
            e = min(i + BATCH - 1, n); b = idx[i:e]
            l, gs = Zygote.withgradient(p -> lossfn(p, X[:, b], Ytr[:, b]), ps)
            opt, ps = Optimisers.update(opt, ps, gs[1])
            tot += l; nb += 1
        end
        acc = accfn(ps)
        row = DataFrame(gitrev = GITREV, arm = arm, seed = seed, epoch = epoch,
                        test_acc = acc, loss = tot / nb, elapsed_s = time() - t0,
                        epochs = EPOCHS, batch = BATCH, lr = LR, hidden = HID,
                        output = DOUT, ntrain = ntr)
        CSV.write(csv_file, row; append = true)
        @printf("  %-10s seed %d epoch %d/%d  loss %.4f  test %.4f  [%.0f s]\n",
                arm, seed, epoch, EPOCHS, tot / nb, acc, time() - t0)
    end
end
println("\nwrote $csv_file")
