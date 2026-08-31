using Pkg; Pkg.activate(".")
using PhasorNetworks
using Lux
using MLUtils
using OneHotArrays
using Random
using LinearAlgebra
using Statistics
using Optimisers

function main()
    rng = Xoshiro(1234)
    train_data = PhasorNetworks.fashion_mnist_data(:train)
    test_data = PhasorNetworks.fashion_mnist_data(:test)
    train_loader = DataLoader(train_data, batchsize=128, shuffle=true, rng=rng)
    test_loader = DataLoader(test_data, batchsize=128)

    # Build model - depth=1, width=64 - use a flat chain
    layers = [
        PhasorNetworks.PhasorDense(784 => 64, PhasorNetworks.normalize_to_unit_circle; use_bias=true),
        PhasorNetworks.ResidualBlock((64, 64), PhasorNetworks.normalize_to_unit_circle; gate=:rezero, alpha0=0.1f0, branch_init_scale=0.1f0, use_bias=true),
        PhasorNetworks.PhasorDense(64 => 10, PhasorNetworks.normalize_to_unit_circle; use_bias=true)
    ]
    chain = Lux.Chain(layers...)

    ps, st = Lux.setup(rng, chain)
    st = Lux.initialstates(rng, chain)
    lux_st = st  # Keep Lux state separate from settled states

    opt = PhasorNetworks.LockinEP(ε=0.1f0, ω_p=0.02f0, n_cycles=8, T_warmup_cycles=2, T_free=200, dt=0.5f0)
    opt_state = Optimisers.setup(Optimisers.Adam(0.001f0), ps)

    codebook = PhasorNetworks.normalize_to_unit_circle(randn(rng, ComplexF32, 10, 10))

    println("Starting training...")
    for epoch in 1:1
        epoch_loss = 0f0
        n_batches = 0
        
        for (x, y) in train_loader
            x = Float32.(x)
            x = reshape(x, 784, size(x, 3))
            
            y_onehot = onehotbatch(y, 0:9)
            batch_cost = PhasorNetworks.CodebookCost(codebook, y_onehot)
            
            # LockinEP gradient step
            grads, _ = PhasorNetworks.ep_gradient(opt, chain, ps, lux_st, x, batch_cost)
            
            # Update
            opt_state, ps = Optimisers.update(opt_state, ps, grads)
            
            # Compute loss on free phase
            y_onehot = onehotbatch(y, 0:9)
            batch_cost = PhasorNetworks.CodebookCost(codebook, y_onehot)
            s_free = PhasorNetworks.phasor_settle(chain, ps, lux_st, x, batch_cost, 0f0; T=200, dt=0.5f0)
            loss = PhasorNetworks.ep_loss(batch_cost, s_free[end])
            epoch_loss += loss
            n_batches += 1
            
            if n_batches % 20 == 0
                println("  Batch $n_batches, loss=$loss")
            end
            
            # Only run first 30 batches for quick test
            n_batches >= 30 && break
        end
        println("Epoch $epoch: loss=$(epoch_loss/n_batches)")
        
        # Evaluate (simplified - just check loss on test batch)
        test_x, test_y = first(test_loader)
        test_x = Float32.(test_x)
        test_x = reshape(test_x, 784, size(test_x, 3))
        test_y_onehot = onehotbatch(test_y, 0:9)
        test_cost = PhasorNetworks.CodebookCost(codebook, test_y_onehot)
        s_free = PhasorNetworks.phasor_settle(chain, ps, lux_st, test_x, test_cost, 0f0; T=200, dt=0.5f0)
        test_loss = PhasorNetworks.ep_loss(test_cost, s_free[end])
        println("  Test loss: $test_loss")
    end

    println("Done!")
end

main()