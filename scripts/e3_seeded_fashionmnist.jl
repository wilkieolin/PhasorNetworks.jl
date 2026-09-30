#!/usr/bin/env julia
# E3: Seeded FashionMNIST — StaticEP vs LockinEP vs Backprop
# Runs 5 seeds × {StaticEP, LockinEP, Backprop} on 784→256→64 PhasorDense
# Outputs CSV for Figure 1a (or 4a depending on paper structure)

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

using PhasorNetworks, Lux, MLUtils, OneHotArrays, Statistics, Random, Zygote, Optimisers, CUDA
using Dates, LinearAlgebra, CSV, DataFrames, Printf
using Random: Xoshiro

const SEEDS = [789, 999]  # Remaining 2 seeds (42, 123, 456 already done)
const BATCHSIZE = 128
const EPOCHS = 5
const LR = 0.001
const USE_CUDA = CUDA.functional()
const HID = 256
const DOUT = 64
const SCALE = 0.4f0

const OUT = joinpath(repo_root, "results", "e3_seeded_fashionmnist")
mkpath(OUT)

cdev = cpu_device()
gdev = gpu_device()
dev = USE_CUDA ? gdev : cdev

# Provenance
const GITREV = try
    rev = strip(read(`git -C $(repo_root) rev-parse --short HEAD`, String))
    d   = read(`git -C $(repo_root) diff HEAD -- src`, String)
    isempty(strip(d)) ? rev : rev * "-d" * string(hash(d), base = 16)[1:8]
catch
    "unknown"
end

# Device transfer helper
to_device(x) = USE_CUDA ? x |> gdev : x

# ---- encode_phase ----
function encode_phase(imgs::AbstractArray{Float32,3})
    N = size(imgs, 3)
    flat = reshape(imgs, :, N)
    μ = mean(flat; dims=1)
    σ = std(flat; dims=1) .+ 1f-6
    return Phase.(0.5f0 .* tanh.((flat .- μ) ./ σ))
end

# Load FashionMNIST
println("Loading FashionMNIST...")
tr = fashion_mnist_data(:train)
te = fashion_mnist_data(:test)
ntr = min(60000, length(tr.targets))
nte = min(10000, length(te.targets))
Xtr = encode_phase(Float32.(tr.features[:, :, 1:ntr]))
Xte = encode_phase(Float32.(te.features[:, :, 1:nte]))
ytr = Int.(tr.targets[1:ntr]) .+ 1
yte = Int.(te.targets[1:nte]) .+ 1

# Move features to device (labels stay on CPU for CodebookCost)
Xtr = to_device(Xtr)
Xte = to_device(Xte)
# ytr, yte stay on CPU

# DataLoaders (now on device)
train_loader = DataLoader((Xtr, ytr), batchsize=BATCHSIZE, shuffle=true)
test_loader = DataLoader((Xte, yte), batchsize=BATCHSIZE, shuffle=false)

# Build EP chain
import PhasorNetworks: default_bias, normalize_to_unit_circle

function build_chain(rng)
    chain = Chain(
        PhasorDense(784 => HID, normalize_to_unit_circle, use_bias=true),
        PhasorDense(HID => DOUT, normalize_to_unit_circle, use_bias=true)
    )
    ps, st = Lux.setup(rng, chain)
    ps = (layer_1 = merge(ps.layer_1, (weight = SCALE .* ps.layer_1.weight,)),
          layer_2 = merge(ps.layer_2, (weight = SCALE .* ps.layer_2.weight,)))
    return chain, ps, st
end

function make_codes(rng)
    return ComplexF32.(angle_to_complex(orthogonal_codes(rng, DOUT, 10)))
end

# Backprop loss (feedforward) - one-hot created outside gradient
function bp_loss(x, y, model, ps, st, codebook, y_onehot)
    z_out, _ = Lux.apply(model, x, ps, st)
    z_out_c = ComplexF32.(angle_to_complex(z_out))
    logits = similarity_outer(z_out_c, codebook)
    log_probs = logits .- log.(sum(exp.(logits); dims=1))
    loss = -mean(sum(y_onehot .* log_probs; dims=1))
    return loss
end

# Create one-hot encoding on GPU (CPU loop then transfer)
function make_onehot(y, dev)
    y_0based = (y .- 1) .|> Int
    y_onehot_cpu = zeros(Float32, 10, length(y))
    for i in 1:length(y)
        y_onehot_cpu[y_0based[i] + 1, i] = 1f0
    end
    return y_onehot_cpu |> dev
end

# Backprop accuracy (feedforward)
function bp_accuracy(model, data_loader, ps, st, codebook)
    total_correct = 0
    total_samples = 0
    for (x, y) in data_loader
        z_out, _ = Lux.apply(model, x, ps, st)
        z_out_c = ComplexF32.(angle_to_complex(z_out))
        logits = similarity_outer(z_out_c, codebook)
        logits_cpu = logits |> cdev
        pred_indices = argmax(logits_cpu; dims=1)
        pred_labels = [idx[1] for idx in vec(pred_indices)]
        y_cpu = y |> cdev
        batch_correct = sum(pred_labels .== y_cpu)
        total_correct += batch_correct
        total_samples += length(y_cpu)
    end
    return total_correct / total_samples
end

# EP settle accuracy
function ep_accuracy(model, X, y, ps, st, codebook; T=200, dt=0.5f0, K_mode=:zero)
    correct = 0
    for i in 1:512:length(y)
        e = min(i + 512 - 1, length(y))
        x_batch = X[:, i:e]
        logits = ep_predict(model, ps, st, x_batch, codebook; T=T, dt=dt, K_mode=K_mode)
        pred = [argmax(view(logits, :, b))[1] for b in 1:(e - i + 1)]
        correct += sum(pred .== y[i:e])
    end
    return correct / length(y)
end

# CSV output
csv_file = joinpath(OUT, "e3_seeded_$(GITREV).csv")
isfile(csv_file) || CSV.write(csv_file, DataFrame(
    gitrev=String[],
    seed=Int[],
    method=String[],
    bp_acc=Float32[],
    ep_acc=Float32[],
    bp_epochs=Int[],
    lr=Float32[],
    hidden=Int[],
    output=Int[],
    scale=Float32[]
))

# Methods to test
methods = [
    ("StaticEP", :staticep),
    ("LockinEP", :lockinep),
    ("Backprop", :backprop)
]

println("=== E3 Seeded FashionMNIST ===")
println("Seeds: $SEEDS")
println("Methods: $(map(first, methods))")
println("Output: $csv_file")

for seed in SEEDS
    println("\n========== Seed $seed ==========")
    
    rng = Xoshiro(seed)
    chain, ps, st = build_chain(rng)
    cb_codes = make_codes(rng)
    
    # Move model to device
    chain = to_device(chain)
    ps = to_device(ps)
    st = to_device(st)
    cb_codes = to_device(cb_codes)
    
    for (method_name, method_sym) in methods
        println("\n--- $method_name (seed=$seed) ---")
        
        # Reset params for each method
        _, ps_reset, st_reset = build_chain(rng)
        chain_reset = to_device(chain)
        ps_reset = to_device(ps_reset)
        st_reset = to_device(st_reset)
        
        if method_sym == :backprop
            # Pure backprop training
            optimiser = Optimisers.Adam(LR)
            opt_state = Optimisers.setup(optimiser, ps_reset)
            
            for epoch in 1:EPOCHS
                epoch_losses = Float64[]
                for (x, y) in train_loader
                    y_onehot = make_onehot(y, dev)
                    lf = p -> bp_loss(x, y, chain_reset, p, st_reset, cb_codes, y_onehot)
                    lossval, gs = withgradient(lf, ps_reset)
                    push!(epoch_losses, lossval)
                    opt_state, ps_reset = Optimisers.update(opt_state, ps_reset, gs[1])
                end
                println("  Epoch $epoch: loss = $(mean(epoch_losses))")
            end
            
            bp_acc = bp_accuracy(chain_reset, test_loader, ps_reset, st_reset, cb_codes)
            ep_acc = ep_accuracy(chain_reset, Xte, yte, ps_reset, st_reset, cb_codes)
            
        elseif method_sym == :staticep
            # StaticEP training
            static_method = StaticEP(β=0.005f0, T_free=200, T_nudge=100, dt=0.5f0, centered=true)
            args = Args(lr=LR, epochs=EPOCHS, weight_decay=0.0001, rng=Xoshiro(seed + 1000))
            
            losses, ps_trained, st_trained = ep_train(chain_reset, ps_reset, st_reset, train_loader, args;
                method=static_method,
                cost_fn=yb -> CodebookCost(cb_codes, yb),
                optimiser=Optimisers.Adam)
            
            bp_acc = bp_accuracy(chain_reset, test_loader, ps_trained, st_trained, cb_codes)
            ep_acc = ep_accuracy(chain_reset, Xte, yte, ps_trained, st_trained, cb_codes)
            
        elseif method_sym == :lockinep
            # LockinEP training
            lockin_method = LockinEP(ε=0.05f0, ω_p=0.05f0, n_cycles=4, T_warmup_cycles=2, T_free=100, dt=0.1f0)
            args = Args(lr=LR, epochs=EPOCHS, weight_decay=0.0001, rng=Xoshiro(seed + 2000))
            
            losses, ps_trained, st_trained = ep_train(chain_reset, ps_reset, st_reset, train_loader, args;
                method=lockin_method,
                cost_fn=yb -> CodebookCost(cb_codes, yb),
                optimiser=Optimisers.Adam)
            
            bp_acc = bp_accuracy(chain_reset, test_loader, ps_trained, st_trained, cb_codes)
            ep_acc = ep_accuracy(chain_reset, Xte, yte, ps_trained, st_trained, cb_codes)
        end
        
        println("  BP (feedforward) accuracy: $bp_acc")
        println("  EP (settle) accuracy:      $ep_acc")
        println("  Drop:                      $(bp_acc - ep_acc)")
        
        # Write CSV row
        row = DataFrame(
            gitrev = GITREV,
            seed = seed,
            method = method_name,
            bp_acc = Float32(bp_acc),
            ep_acc = Float32(ep_acc),
            bp_epochs = EPOCHS,
            lr = Float32(LR),
            hidden = HID,
            output = DOUT,
            scale = Float32(SCALE)
        )
        CSV.write(csv_file, row, append=true)
    end
end

println("\n=== E3 Complete ===")
println("Results: $csv_file")