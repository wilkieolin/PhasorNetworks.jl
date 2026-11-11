#!/usr/bin/env julia
# E2b: LockinEP vs StaticEP fine-tuning, PAIRED.
#
# Why this replaces scripts/e2_lockin_verification.jl
# ---------------------------------------------------
# The Figure-4 recovery sweep (scripts/e2_full_sweep.jl, 377 rows) tunes every
# impaired network with StaticEP, because StaticEP is ~8x cheaper per gradient.
# That is only legitimate if StaticEP and LockinEP recover the same accuracy,
# so the sweep needs a verification arm. The existing one cannot serve:
#
#   1. It draws impairments with Xoshiro(hash((type, param, rep, "backprop",
#      "lockin"))) -- note the trailing "lockin" -- while the main sweep uses
#      Xoshiro(hash((type, param, rep, pretrain_name))). Different key, so
#      different damage, so no row in one file pairs with any row in the other.
#      Confirmed empirically: zero of its 13 rows match a main-sweep row on
#      (type, param, rep, acc_clean, acc_impaired).
#   2. Two of its four cells (lognormal 0.3, stuck_zero 0.3) are undamaged --
#      acc_impaired is within noise of acc_clean -- so they measure recovery
#      from nothing. Only ~5 of the 13 rows carry real damage.
#   3. Its LockinEP is configured n_cycles=2, T_free=100, which is not the
#      operating point the rest of the paper characterises.
#
# This script fixes all three: identical base network, identical impairment
# draws, both estimators run from the SAME impaired parameters so each row is a
# true paired observation, and LockinEP at the documented operating point.
#
# Cells are restricted to severities where the damage is real (acc_impaired
# well below acc_clean in the main sweep), because an agreement test on
# undamaged networks is uninformative.
#
#   JULIA_CUDA_HARD_MEMORY_LIMIT=16GiB julia --project=scripts scripts/e2_lockin_paired.jl
#
# Env: E2P_REPS, E2P_NTRAIN, E2P_NTEST, E2P_FTEPOCHS, E2P_CELLS, E2P_TAG

using Pkg
function find_repo_root(start_dir::String = pwd())
    dir = start_dir
    while !(isfile(joinpath(dir, "Project.toml")) && isdir(joinpath(dir, ".git")))
        parent = dirname(dir); parent == dir && error("no repo root"); dir = parent
    end
    return dir
end
repo_root = find_repo_root(@__DIR__); cd(repo_root); Pkg.activate(joinpath(repo_root, "scripts"))

using PhasorNetworks, Lux, MLUtils, OneHotArrays, Statistics, Random, Zygote, Optimisers, CUDA
using LinearAlgebra, CSV, DataFrames, Printf
using Random: Xoshiro
import PhasorNetworks: normalize_to_unit_circle

_envi(k, d) = haskey(ENV, k) ? parse(Int, ENV[k]) : d

# ---- identical to scripts/e2_full_sweep.jl; do not drift ----
const SEED         = 0x42
const HID          = 256
const DOUT         = 64
const SCALE        = 0.4f0
const BATCH        = 128
const BP_EPOCHS    = 5
const BP_LR        = 0.001
const WEIGHT_DECAY = 0.0001

# ---- fine-tuning ----
const N_TRAIN   = _envi("E2P_NTRAIN", 10000)   # recovery converges well inside 3 epochs
const N_TEST    = _envi("E2P_NTEST", 10000)
const FT_EPOCHS = _envi("E2P_FTEPOCHS", 3)
const REPS      = _envi("E2P_REPS", 6)

# ---- LockinEP at the documented operating point (matches ep_adiabatic_sweep.jl) ----
const LK_EPS = 0.03f0
const LK_WP  = 0.02f0
const LK_NC  = 4
const LK_WARM = 1
const LK_TFREE = 200
const LK_DT  = 0.5f0

# Only cells where the main sweep shows real damage (median acc_impaired, backprop arm)
const ALL_CELLS = [
    (:gaussian,   0.1f0),   # 0.575
    (:gaussian,   0.3f0),   # 0.143
    (:gaussian,   0.5f0),   # 0.117
    (:stuck_sat,  0.1f0),   # 0.612
    (:stuck_sat,  0.3f0),   # 0.198
    (:stuck_sat,  0.5f0),   # 0.131
    (:stuck_zero, 0.7f0),   # 0.480
    (:stuck_zero, 0.9f0),   # 0.100
    (:lognormal,  1.0f0),   # 0.798
    (:lognormal,  1.5f0),   # 0.467
]
const CELLS = haskey(ENV, "E2P_CELLS") ?
    filter(c -> occursin(String(c[1]), ENV["E2P_CELLS"]), ALL_CELLS) : ALL_CELLS

const USE_CUDA = CUDA.functional()
const OUT = joinpath(repo_root, "results", "ep_analog_finetune"); mkpath(OUT)
const GITREV = try
    rev = strip(read(`git -C $(repo_root) rev-parse --short HEAD`, String))
    d   = read(`git -C $(repo_root) diff HEAD -- src`, String)
    isempty(strip(d)) ? rev : rev * "-d" * string(hash(d), base = 16)[1:8]
catch; "unknown" end

cdev = cpu_device(); gdev = gpu_device()
to_device(x) = USE_CUDA ? x |> gdev : x

function build_chain(rng::Xoshiro; scale = SCALE)
    chain = Chain(PhasorDense(784 => HID, normalize_to_unit_circle, use_bias = true),
                  PhasorDense(HID => DOUT, normalize_to_unit_circle, use_bias = true))
    ps, st = Lux.setup(rng, chain)
    ps = (layer_1 = merge(ps.layer_1, (weight = scale .* ps.layer_1.weight,)),
          layer_2 = merge(ps.layer_2, (weight = scale .* ps.layer_2.weight,)))
    return chain, ps, st
end
make_codes(rng) = ComplexF32.(angle_to_complex(orthogonal_codes(rng, DOUT, 10)))

function encode_phase(imgs::AbstractArray{Float32,3})
    N = size(imgs, 3); flat = reshape(imgs, :, N)
    mu = mean(flat; dims = 1); sd = std(flat; dims = 1) .+ 1f-6
    return Phase.(0.5f0 .* tanh.((flat .- mu) ./ sd))
end

function bp_loss(x, y, model, ps, st, codebook)
    z_out, _ = Lux.apply(model, x, ps, st)
    logits = similarity_outer(ComplexF32.(angle_to_complex(z_out)), codebook)
    y_onehot = onehotbatch(y .- 1, 0:9) |> (USE_CUDA ? gdev : identity)
    return -mean(sum(y_onehot .* (logits .- log.(sum(exp.(logits); dims = 1))); dims = 1))
end

function train_bp(chain, ps, st, loader, epochs, lr, codebook)
    opt_state = Optimisers.setup(Optimisers.Adam(lr), ps)
    for epoch in 1:epochs
        ls = Float64[]
        for (x, y) in loader
            lv, gs = Zygote.withgradient(p -> bp_loss(x, y, chain, p, st, codebook), ps)
            push!(ls, lv); opt_state, ps = Optimisers.update(opt_state, ps, gs[1])
        end
        @printf("  BP epoch %d/%d loss %.4f\n", epoch, epochs, mean(ls))
    end
    return ps
end

function ep_accuracy(model, X, y, ps, st, codebook; batch = 512)
    correct = 0
    for i in 1:batch:length(y)
        e = min(i + batch - 1, length(y))
        logits = ep_predict(model, ps, st, X[:, i:e], codebook; T = 200, dt = 0.5f0, K_mode = :zero)
        correct += sum([argmax(view(logits, :, b))[1] for b in 1:(e - i + 1)] .== y[i:e])
    end
    return correct / length(y)
end

# --- the two estimators under test, run from identical impaired parameters ---
function tune_static(chain, ps, st, loader, codes; weight_mask = nothing)
    args = Args(lr = 0.001, epochs = FT_EPOCHS, weight_decay = WEIGHT_DECAY, rng = Xoshiro(SEED + 1))
    m = StaticEP(β = 0.005f0, T_free = 200, T_nudge = 100, dt = 0.5f0, centered = true)
    _, p, s = ep_train(chain, ps, st, loader, args; method = m,
                       cost_fn = yb -> CodebookCost(codes, yb),
                       optimiser = Optimisers.Adam, weight_mask = weight_mask)
    return p, s
end

function tune_lockin(chain, ps, st, loader, codes; weight_mask = nothing)
    args = Args(lr = 0.001, epochs = FT_EPOCHS, weight_decay = WEIGHT_DECAY, rng = Xoshiro(SEED + 2))
    m = LockinEP(ε = LK_EPS, ω_p = LK_WP, n_cycles = LK_NC, T_warmup_cycles = LK_WARM,
                 T_free = LK_TFREE, dt = LK_DT, K_mode = :zero, project = :hard)
    _, p, s = ep_train(chain, ps, st, loader, args; method = m,
                       cost_fn = yb -> CodebookCost(codes, yb),
                       optimiser = Optimisers.Adam, weight_mask = weight_mask)
    return p, s
end

# --- impairments: byte-identical to scripts/e2_full_sweep.jl:225-265 ---
function apply_lognormal(ps, sig, rng)
    W1 = ps.layer_1.weight .* (1f0 .+ exp.(sig .* randn(rng, Float32, size(ps.layer_1.weight))) .- 1f0)
    W2 = ps.layer_2.weight .* (1f0 .+ exp.(sig .* randn(rng, Float32, size(ps.layer_2.weight))) .- 1f0)
    return merge(ps, (layer_1 = merge(ps.layer_1, (weight = W1,)),
                      layer_2 = merge(ps.layer_2, (weight = W2,)))), nothing
end
function apply_gaussian(ps, sig, rng)
    W1 = ps.layer_1.weight .+ sig .* randn(rng, Float32, size(ps.layer_1.weight))
    W2 = ps.layer_2.weight .+ sig .* randn(rng, Float32, size(ps.layer_2.weight))
    return merge(ps, (layer_1 = merge(ps.layer_1, (weight = W1,)),
                      layer_2 = merge(ps.layer_2, (weight = W2,)))), nothing
end
function _mask_nt(ps, m1, m2)
    (layer_1 = (weight = m1, bias_real = ones(Float32, size(ps.layer_1.bias_real)),
                bias_imag = ones(Float32, size(ps.layer_1.bias_imag))),
     layer_2 = (weight = m2, bias_real = ones(Float32, size(ps.layer_2.bias_real)),
                bias_imag = ones(Float32, size(ps.layer_2.bias_imag))))
end
function apply_stuck_zero(ps, frac, rng)
    m1 = Float32.(rand(rng, Float32, size(ps.layer_1.weight)) .>= frac)
    m2 = Float32.(rand(rng, Float32, size(ps.layer_2.weight)) .>= frac)
    return merge(ps, (layer_1 = merge(ps.layer_1, (weight = ps.layer_1.weight .* m1,)),
                      layer_2 = merge(ps.layer_2, (weight = ps.layer_2.weight .* m2,)))), _mask_nt(ps, m1, m2)
end
function apply_stuck_sat(ps, frac, rng)
    mx1 = maximum(abs.(ps.layer_1.weight)); mx2 = maximum(abs.(ps.layer_2.weight))
    m1 = Float32.(rand(rng, Float32, size(ps.layer_1.weight)) .>= frac)
    m2 = Float32.(rand(rng, Float32, size(ps.layer_2.weight)) .>= frac)
    s1 = rand(rng, Float32[1, -1], size(ps.layer_1.weight))
    s2 = rand(rng, Float32[1, -1], size(ps.layer_2.weight))
    W1 = ps.layer_1.weight .* m1 .+ (1f0 .- m1) .* s1 .* mx1
    W2 = ps.layer_2.weight .* m2 .+ (1f0 .- m2) .* s2 .* mx2
    return merge(ps, (layer_1 = merge(ps.layer_1, (weight = W1,)),
                      layer_2 = merge(ps.layer_2, (weight = W2,)))), _mask_nt(ps, m1, m2)
end

# ---- data & base network ----
println("Loading FashionMNIST...")
tr = fashion_mnist_data(:train); te = fashion_mnist_data(:test)
ntr = min(N_TRAIN, length(tr.targets)); nte = min(N_TEST, length(te.targets))
Xtr = to_device(encode_phase(Float32.(tr.features[:, :, 1:ntr])))
Xte = to_device(encode_phase(Float32.(te.features[:, :, 1:nte])))
ytr = Int.(tr.targets[1:ntr]) .+ 1; yte = Int.(te.targets[1:nte]) .+ 1
loader = DataLoader((Xtr, ytr), batchsize = BATCH, shuffle = true)

rng = Xoshiro(SEED)
chain, ps_init, st = build_chain(rng)
codes = make_codes(rng)
chain = to_device(chain); ps_init = to_device(ps_init); st = to_device(st); codes = to_device(codes)

println("Pretraining base network with backprop ($(BP_EPOCHS) epochs)...")
ps_bp = train_bp(chain, ps_init, st, loader, BP_EPOCHS, BP_LR, codes)
acc_clean = ep_accuracy(chain, Xte, yte, ps_bp, st, codes)
@printf("Clean EP accuracy: %.4f\n", acc_clean)

csv_file = joinpath(OUT, "lockin_paired_$(GITREV)$(get(ENV, "E2P_TAG", "")).csv")
isfile(csv_file) || CSV.write(csv_file, DataFrame(
    gitrev = String[], pretrain = String[], impairment_type = String[], param = Float32[],
    rep = Int[], acc_clean = Float32[], acc_impaired = Float32[],
    acc_static = Float32[], acc_lockin = Float32[],
    lk_eps = Float32[], lk_omega_p = Float32[], lk_n_cycles = Int[],
    ft_epochs = Int[], n_train = Int[], hidden = Int[], output = Int[]))

println("\n=== paired LockinEP vs StaticEP: $(length(CELLS)) cells x $(REPS) reps ===")
for (imp_type, param) in CELLS, rep in 1:REPS
    # SAME key as scripts/e2_full_sweep.jl:337 -- this is what makes the rows pair
    rng_rep = Xoshiro(hash((imp_type, param, rep, "backprop")) % UInt64)
    ps_cpu = ps_bp |> cdev
    ps_imp, wmask = imp_type === :lognormal  ? apply_lognormal(ps_cpu, param, rng_rep)  :
                    imp_type === :gaussian   ? apply_gaussian(ps_cpu, param, rng_rep)   :
                    imp_type === :stuck_zero ? apply_stuck_zero(ps_cpu, param, rng_rep) :
                                               apply_stuck_sat(ps_cpu, param, rng_rep)
    ps_imp = to_device(ps_imp)
    wmask  = wmask === nothing ? nothing : to_device(wmask)

    acc_imp = ep_accuracy(chain, Xte, yte, ps_imp, st, codes)
    ps_s, st_s = tune_static(chain, ps_imp, st, loader, codes; weight_mask = wmask)
    acc_s = ep_accuracy(chain, Xte, yte, ps_s, st_s, codes)
    ps_l, st_l = tune_lockin(chain, ps_imp, st, loader, codes; weight_mask = wmask)
    acc_l = ep_accuracy(chain, Xte, yte, ps_l, st_l, codes)

    @printf("  %-11s %-5g rep %d | impaired %.4f -> static %.4f  lockin %.4f  (d = %+.4f)\n",
            imp_type, param, rep, acc_imp, acc_s, acc_l, acc_l - acc_s)
    CSV.write(csv_file, DataFrame(
        gitrev = GITREV, pretrain = "backprop", impairment_type = String(imp_type),
        param = param, rep = rep, acc_clean = Float32(acc_clean), acc_impaired = Float32(acc_imp),
        acc_static = Float32(acc_s), acc_lockin = Float32(acc_l),
        lk_eps = LK_EPS, lk_omega_p = LK_WP, lk_n_cycles = LK_NC,
        ft_epochs = FT_EPOCHS, n_train = ntr, hidden = HID, output = DOUT), append = true)
end
println("\n=== done -> $csv_file ===")
