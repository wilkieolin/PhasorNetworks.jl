#!/usr/bin/env julia
# E6: does the adiabatic basin predict whether training works?
#
# `results/ep_adiabatic/grid.csv` measures GRADIENT FIDELITY on a factorial
# (eps, omega_p, n_cycles) grid. It says nothing about training, because it
# never trains anything. This script closes that gap: it trains the same
# 784->256->64 phasor chain with LockinEP at a ladder of operating points that
# sit at KNOWN locations on that map, and logs the loss curve at each.
#
# The settle configuration below is copied from scripts/ep_adiabatic_sweep.jl
# (DT = 0.5, T_FREE = 200, WARMUP = 1, n_cycles = 4) and must stay matched to
# it. If it drifts, the stars plotted on the Figure-2 heatmap no longer mark
# the points these curves were measured at, and the figure becomes a lie.
#
# Operating points, with the median cos_l1 the grid reports for each
# (n_cycles = 4, pooled over 8 input draws):
#
#   A  eps=0.03  w_p=0.01   cos 0.997   deep inside the basin
#   B  eps=0.03  w_p=0.02   cos 0.989   the documented operating point
#   C  eps=0.03  w_p=0.05   cos 0.885   near the boundary, 2/8 draws fail
#   D  eps=0.03  w_p=0.10   cos 0.254   outside: demodulation has decorrelated
#   E  eps=0.003 w_p=0.01   cos 0.34    below the lower eps bound (see note)
#
# A-D walk the frequency boundary at fixed amplitude. E is the control for the
# OTHER wall of the window: the response eps*dz/dbeta has to clear the readout
# floor, so the usable eps range is two-sided, and the grid's bottom row is the
# only place that shows it.
#
# GPU: DGX Spark GB10, unified memory, ~110 GB. Run under
# JULIA_CUDA_HARD_MEMORY_LIMIT -- the CUDA pool grows into whatever headroom it
# is given and exceeding physical memory pages to disk and locks the machine.
#
#   JULIA_CUDA_HARD_MEMORY_LIMIT=8GiB julia --project=scripts scripts/e6_basin_training.jl
#
# Env overrides for a cheap smoke test:
#   E6_EPOCHS, E6_SEEDS, E6_BATCH, E6_NTRAIN, E6_NTEST, E6_POINTS (e.g. "D"),
#   E6_TAG (suffix on the output filename)

using Pkg
function find_repo_root(start_dir::String = pwd())
    dir = start_dir
    while !(isfile(joinpath(dir, "Project.toml")) && isdir(joinpath(dir, ".git")))
        parent = dirname(dir)
        parent == dir && error("Repository root not found from $(start_dir)")
        dir = parent
    end
    return dir
end
repo_root = find_repo_root(@__DIR__)
cd(repo_root); Pkg.activate(joinpath(repo_root, "scripts"))

using PhasorNetworks, Lux, MLUtils, Statistics, Random, Optimisers, CUDA
using LinearAlgebra, CSV, DataFrames, Printf, Dates
using Random: Xoshiro
import PhasorNetworks: normalize_to_unit_circle

_envi(k, d) = haskey(ENV, k) ? parse(Int, ENV[k]) : d
_envf(k, d) = haskey(ENV, k) ? parse(Float64, ENV[k]) : d

# ---- settle configuration: MUST match scripts/ep_adiabatic_sweep.jl ----
const DT       = 0.5f0
const T_FREE   = 200
const WARMUP   = 1      # T_warmup_cycles
const NCYCLES  = 4
const K_MODE   = :zero
const PROJECT  = :hard

# ---- architecture: matches E3 and the grid ----
const HID    = 256
const DOUT   = 64
const SCALE  = 0.4f0

# ---- training ----
const EPOCHS    = _envi("E6_EPOCHS", 5)
const BATCHSIZE = _envi("E6_BATCH", 256)
const LR        = _envf("E6_LR", 0.003)
const WD        = _envf("E6_WD", 0.0)   # E1: no decay + centered was the best result in the repo
const NTRAIN    = _envi("E6_NTRAIN", 60000)
const NTEST     = _envi("E6_NTEST", 2000)   # per-epoch eval subset; kept small, EP settle is the cost
# The first three are the original run and must stay first, so re-running with a
# larger E6_SEEDS reproduces those rows rather than shifting them. E6_SEED_LIST
# overrides entirely (comma-separated), which is how the point-B arm is sharded
# across machines for the method comparison.
const ALL_SEEDS = [42, 123, 456, 789, 999, 1337, 2718, 3141, 4242, 5150, 6180, 7071]
const SEEDS     = haskey(ENV, "E6_SEED_LIST") ?
    [parse(Int, t) for t in split(ENV["E6_SEED_LIST"], ',') if !isempty(strip(t))] :
    ALL_SEEDS[1:_envi("E6_SEEDS", 3)]

# ---- operating points ----
const ALL_POINTS = [
    (name = "A", eps = 0.03f0,  omega_p = 0.01f0, region = "inside"),
    (name = "B", eps = 0.03f0,  omega_p = 0.02f0, region = "operating"),
    (name = "C", eps = 0.03f0,  omega_p = 0.05f0, region = "boundary"),
    (name = "D", eps = 0.03f0,  omega_p = 0.10f0, region = "outside"),
    (name = "E", eps = 0.003f0, omega_p = 0.01f0, region = "below_eps_floor"),
]
const POINTS = haskey(ENV, "E6_POINTS") ?
    filter(p -> occursin(p.name, ENV["E6_POINTS"]), ALL_POINTS) : ALL_POINTS

const USE_CUDA = CUDA.functional()
const OUT = joinpath(repo_root, "results", "e6_basin_training")
mkpath(OUT)

const GITREV = try
    rev = strip(read(`git -C $(repo_root) rev-parse --short HEAD`, String))
    d   = read(`git -C $(repo_root) diff HEAD -- src`, String)
    isempty(strip(d)) ? rev : rev * "-d" * string(hash(d), base = 16)[1:8]
catch
    "unknown"
end

cdev = cpu_device(); gdev = gpu_device()
dev  = USE_CUDA ? gdev : cdev
to_device(x) = USE_CUDA ? x |> gdev : x

# ---- data ----
function encode_phase(imgs::AbstractArray{Float32,3})
    N = size(imgs, 3)
    flat = reshape(imgs, :, N)
    mu = mean(flat; dims = 1)
    sd = std(flat; dims = 1) .+ 1f-6
    # half-plane range is deliberate: phases wrap, so a full-range map is not injective
    return Phase.(0.5f0 .* tanh.((flat .- mu) ./ sd))
end

println("Loading FashionMNIST...")
tr = fashion_mnist_data(:train); te = fashion_mnist_data(:test)
ntr = min(NTRAIN, length(tr.targets)); nte = min(NTEST, length(te.targets))
Xtr = to_device(encode_phase(Float32.(tr.features[:, :, 1:ntr])))
Xte = to_device(encode_phase(Float32.(te.features[:, :, 1:nte])))
ytr = Int.(tr.targets[1:ntr]) .+ 1
yte = Int.(te.targets[1:nte]) .+ 1
train_loader = DataLoader((Xtr, ytr), batchsize = BATCHSIZE, shuffle = true)

function build_chain(rng)
    chain = Chain(PhasorDense(784 => HID, normalize_to_unit_circle, use_bias = true),
                  PhasorDense(HID => DOUT, normalize_to_unit_circle, use_bias = true))
    ps, st = Lux.setup(rng, chain)
    ps = (layer_1 = merge(ps.layer_1, (weight = SCALE .* ps.layer_1.weight,)),
          layer_2 = merge(ps.layer_2, (weight = SCALE .* ps.layer_2.weight,)))
    return chain, ps, st
end

function ep_accuracy(model, X, y, ps, st, codebook)
    correct = 0
    for i in 1:512:length(y)
        e = min(i + 511, length(y))
        logits = ep_predict(model, ps, st, X[:, i:e], codebook;
                            T = T_FREE, dt = DT, K_mode = K_MODE)
        pred = [argmax(view(logits, :, b))[1] for b in 1:(e - i + 1)]
        correct += sum(pred .== y[i:e])
    end
    return correct / length(y)
end

# T-vs-2T settle drift, measured AT THE ESTIMATOR'S OWN T_free.
# Measuring it at some oracle's longer T does not predict fidelity -- that was
# established in E1 (results/ep_trained_vs_rescaled/FINDINGS.md).
function settle_drift(model, ps, st, x, cost)
    s1 = phasor_settle(model, ps, st, x, cost, 0f0;
                       T = T_FREE, dt = DT, K_mode = K_MODE, project = PROJECT)
    s2 = phasor_settle(model, ps, st, x, cost, 0f0;
                       T = 2 * T_FREE, dt = DT, K_mode = K_MODE, project = PROJECT)
    num = sum(sum(abs2, a .- b) for (a, b) in zip(s1, s2))
    den = sum(sum(abs2, b) for b in s2) + 1f-30
    return sqrt(num / den)
end

csv_file = joinpath(OUT, "e6_basin_$(GITREV)$(get(ENV, "E6_TAG", "")).csv")
if !isfile(csv_file)
    CSV.write(csv_file, DataFrame(
        gitrev = String[], point = String[], region = String[],
        eps = Float32[], omega_p = Float32[], n_cycles = Int[],
        seed = Int[], epoch = Int[],
        loss = Float32[], test_acc = Float32[],
        settle_drift = Float32[], w1_norm = Float32[], w2_norm = Float32[],
        elapsed_s = Float64[], steps_per_grad = Int[],
        dt = Float32[], T_free = Int[], batchsize = Int[], lr = Float32[],
        weight_decay = Float32[], hidden = Int[], output = Int[], scale = Float32[]))
end

period_steps(w) = round(Int, 2pi / (w * DT))
steps_per_grad(w) = T_FREE + (WARMUP + NCYCLES) * period_steps(w)

println("=== E6: training across the adiabatic basin ===")
println("gitrev=$GITREV  cuda=$USE_CUDA  epochs=$EPOCHS  batch=$BATCHSIZE  ntrain=$ntr  ntest=$nte")
println("seeds=$SEEDS")
for p in POINTS
    @printf("  %s  eps=%-7g w_p=%-6g %-16s %6d settle-steps/gradient\n",
            p.name, p.eps, p.omega_p, p.region, steps_per_grad(p.omega_p))
end
println("output: $csv_file")

for p in POINTS, seed in SEEDS
    @printf("\n--- point %s (eps=%g, w_p=%g, %s)  seed %d ---\n",
            p.name, p.eps, p.omega_p, p.region, seed)
    rng = Xoshiro(seed)
    chain, ps, st = build_chain(rng)
    codes = to_device(ComplexF32.(angle_to_complex(orthogonal_codes(rng, DOUT, 10))))
    chain = to_device(chain); ps = to_device(ps); st = to_device(st)

    method = LockinEP(ε = p.eps, ω_p = p.omega_p, n_cycles = NCYCLES,
                      T_warmup_cycles = WARMUP, T_free = T_FREE, dt = DT,
                      K_mode = K_MODE, project = PROJECT)
    args = Args(lr = LR, epochs = EPOCHS, weight_decay = WD, rng = Xoshiro(seed + 6000))

    # a fixed probe batch for the drift diagnostic, so it is comparable epoch to epoch
    xprobe = Xte[:, 1:min(64, nte)]
    cprobe = CodebookCost(codes, yte[1:min(64, nte)])

    t0 = time()
    cb = function (epoch, ps_e, st_e, epoch_loss)
        acc  = ep_accuracy(chain, Xte, yte, ps_e, st_e, codes)
        drft = settle_drift(chain, ps_e, st_e, xprobe, cprobe)
        el   = time() - t0
        @printf("  epoch %2d  loss %.4f  test_acc %.4f  drift %.2e  (%.0f s)\n",
                epoch, epoch_loss, acc, drft, el)
        CSV.write(csv_file, DataFrame(
            gitrev = GITREV, point = p.name, region = p.region,
            eps = p.eps, omega_p = p.omega_p, n_cycles = NCYCLES,
            seed = seed, epoch = epoch,
            loss = Float32(epoch_loss), test_acc = Float32(acc),
            settle_drift = Float32(drft),
            w1_norm = Float32(norm(ps_e.layer_1.weight)),
            w2_norm = Float32(norm(ps_e.layer_2.weight)),
            elapsed_s = el, steps_per_grad = steps_per_grad(p.omega_p),
            dt = DT, T_free = T_FREE, batchsize = BATCHSIZE, lr = Float32(LR),
            weight_decay = Float32(WD), hidden = HID, output = DOUT, scale = SCALE),
            append = true)
        return nothing
    end

    # epoch 0: the untrained reference, so every curve has a common origin
    cb(0, ps, st, ep_loss(cprobe, phasor_settle(chain, ps, st, xprobe, cprobe, 0f0;
            T = T_FREE, dt = DT, K_mode = K_MODE, project = PROJECT)[end]))

    ep_train(chain, ps, st, train_loader, args;
             method = method, cost_fn = yb -> CodebookCost(codes, yb),
             optimiser = Optimisers.Adam, callback = cb)
end

println("\n=== E6 complete -> $csv_file ===")
