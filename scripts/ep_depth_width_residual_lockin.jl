#!/usr/bin/env julia
# LockinEP depth/width sweep for ResidualBlock
# Tests operating zone of LockinEP with ResidualBlock architecture

using Pkg
Pkg.activate(@__DIR__)

using PhasorNetworks
using Lux
using MLUtils
using OneHotArrays
using Random
using LinearAlgebra
using Statistics
using CSV
using DataFrames
using Dates
using Optimisers

# Configuration
const DEPTHS = [1, 2, 3, 4, 5]  # number of ResidualBlocks
const WIDTHS = [64, 256]
const BATCH_SIZE = 128
const EPOCHS = 5
const LR = 0.001f0
const SEED = 1234

# LockinEP hyperparameters (from A5 sweep)
const ε = 0.1f0
const ω_p = 0.02f0
const n_cycles = 8
const T_warmup_cycles = 2
const T_free = 200
const dt = 0.5f0

function build_residual_model(input_dim, hidden_dim, output_dim, n_residual; gate=:rezero, alpha0=0.1f0)
    layers = []
    push!(layers, PhasorDense(input_dim => hidden_dim, normalize_to_unit_circle; use_bias=true))
    
    for i in 1:n_residual
        push!(layers, ResidualBlock(
            (hidden_dim, hidden_dim),
            normalize_to_unit_circle;
            gate=gate,
            alpha0=alpha0,
            branch_init_scale=0.1f0,
            use_bias=true
        ))
    end
    
    push!(layers, PhasorDense(hidden_dim => output_dim, normalize_to_unit_circle; use_bias=true))
    
    return Lux.Chain(layers...)
end

function make_codebook(output_dim, n_classes, rng)
    codes = randn(rng, ComplexF32, output_dim, n_classes)
    return PhasorNetworks.normalize_to_unit_circle(codes)
end

function evaluate_accuracy(chain, ps, st, test_loader, codebook)
    correct = 0
    total = 0
    
    for (x, y) in test_loader
        x = Float32.(x)  # Input is already phase in [0,1]
        x = reshape(x, 784, size(x, 3))  # Flatten (28,28,B) -> (784,B)
        y_onehot = onehotbatch(y, 0:9)
        batch_cost = PhasorNetworks.CodebookCost(codebook, y_onehot)
        s_free = PhasorNetworks.phasor_settle(chain, ps, st, x, batch_cost, 0f0; T=200, dt=0.5f0)
        z_out = s_free[end]
        # Get predictions from similarities
        sims = PhasorNetworks.similarity(z_out, codebook)
        preds = argmax(sims, dims=1)
        preds = preds .+ 0  # 0-based classes
        c = sum(preds .== y)
        correct += c
        total += length(y)
    end
    return correct / total
end

function train_one_config(depth, width, train_loader, test_loader, codebook; seed=SEED)
    println("  DEBUG: train_one_config called with depth=$depth, width=$width, seed=$seed")
    rng = Xoshiro(seed)
    
    # Build model
    chain = build_residual_model(784, width, 10, depth)
    
    # Initialize
    ps, st = Lux.setup(rng, chain)
    st = Lux.initialstates(rng, chain)
    
    # LockinEP optimizer
    opt = PhasorNetworks.LockinEP(
        ε=ε, ω_p=ω_p, n_cycles=n_cycles, 
        T_warmup_cycles=T_warmup_cycles, T_free=T_free, dt=dt
    )
    opt_state = Optimisers.setup(Optimisers.Adam(LR), ps)
    
    losses = Float32[]
    accuracies = Float32[]
    
    for epoch in 1:EPOCHS
        epoch_loss = 0f0
        n_batches = 0
        
        for (x, y) in train_loader
            x = Float32.(x)  # Input is already phase in [0,1]
            x = reshape(x, 784, size(x, 3))  # Flatten (28,28,B) -> (784,B)
            
            # Create cost with one-hot targets for this batch
            y_onehot = onehotbatch(y, 0:9)
            batch_cost = PhasorNetworks.CodebookCost(codebook, y_onehot)
            
            # LockinEP gradient step
            grads, st = PhasorNetworks.ep_gradient(opt, chain, ps, st, x, batch_cost)
            
            # Update
            ps, opt_state = Optimisers.update(opt_state, ps, grads)
            
            # Compute loss on free phase
            y_onehot = onehotbatch(y, 0:9)
            batch_cost = PhasorNetworks.CodebookCost(codebook, y_onehot)
            s_free = PhasorNetworks.phasor_settle(chain, ps, st, x, batch_cost, 0f0; T=T_free, dt=dt)
            loss = PhasorNetworks.ep_loss(batch_cost, s_free[end])
            epoch_loss += loss
            n_batches += 1
        end
        
        push!(losses, epoch_loss / n_batches)
        
        # Evaluate accuracy
        acc = evaluate_accuracy(chain, ps, st, test_loader, codebook)
        push!(accuracies, acc)
        
        println("  Epoch $epoch: loss=$(losses[end]), acc=$(accuracies[end])")
    end
    
    return losses, accuracies, ps, st
end

function run_sweep()
    rng = Xoshiro(SEED)
    
    println("="^60)
    println("ResidualBlock LockinEP Depth/Width Sweep")
    println("="^60)
    println("Depths: $DEPTHS")
    println("Widths: $WIDTHS")
    println("Epochs: $EPOCHS")
    println("LR: $LR")
    println("LockinEP: ε=$ε, ω_p=$ω_p, n_cycles=$n_cycles")
    println()
    
    # Load data once
    println("Loading FashionMNIST...")
    train_data = PhasorNetworks.fashion_mnist_data(:train)
    test_data = PhasorNetworks.fashion_mnist_data(:test)
    train_loader = DataLoader(train_data, batchsize=BATCH_SIZE, shuffle=true, rng=rng)
    test_loader = DataLoader(test_data, batchsize=BATCH_SIZE)
    
    # Create fixed codebook for evaluation
    codebook = make_codebook(10, 10, rng)
    # CodebookCost needs one-hot targets; we'll create it per-batch in training
    # For evaluation, we'll use the codebook directly with similarity
    
    results = DataFrame(
        depth=Int[],
        width=Int[],
        seed=Int[],
        epoch=Int[],
        loss=Float32[],
        accuracy=Float32[]
    )
    
    for depth in DEPTHS
        for width in WIDTHS
            println("\nTesting depth=$depth, width=$width")
            
            for seed in [SEED, SEED+1, SEED+2]  # 3 seeds
                println("  Seed: $seed")
                try
                    losses, accs, ps, st = train_one_config(depth, width, train_loader, test_loader, codebook; seed=seed)
                    
                    for (epoch, (loss, acc)) in enumerate(zip(losses, accs))
                        push!(results, (depth, width, seed, epoch, loss, acc))
                    end
                catch e
                    println("  ERROR: $e")
                    push!(results, (depth, width, seed, -1, NaN, NaN))
                end
            end
        end
    end
    
    # Save results
    timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    csv_file = "results/ep_residual_lockin_sweep/residual_lockin_$(timestamp).csv"
    mkpath(dirname(csv_file))
    CSV.write(csv_file, results)
    println("\nResults saved to $csv_file")
    
    # Summary
    println("\n" * "="^60)
    println("SUMMARY")
    println("="^60)
    for depth in DEPTHS
        for width in WIDTHS
            subset = filter(r -> r.depth == depth && r.width == width, results)
            if nrow(subset) > 0
                final_accs = subset[subset.epoch .== EPOCHS, :accuracy]
                mean_acc = mean(skipmissing(final_accs))
                std_acc = std(skipmissing(final_accs))
                println("depth=$depth, width=$width: acc = $(round(mean_acc, digits=4)) ± $(round(std_acc, digits=4))")
            end
        end
    end
    
    return results
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_sweep()
end