using Pkg; Pkg.activate(@__DIR__)
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

# Test a single config
rng = Xoshiro(1234)
train_data = PhasorNetworks.fashion_mnist_data(:train)
test_data = PhasorNetworks.fashion_mnist_data(:test)
train_loader = DataLoader(train_data, batchsize=128, shuffle=true, rng=rng)
test_loader = DataLoader(test_data, batchsize=128)

# Build model
chain = PhasorNetworks.PhasorDense(784 => 64, PhasorNetworks.normalize_to_unit_circle; use_bias=true)
for i in 1:1
    global chain = Lux.Chain(chain, PhasorNetworks.ResidualBlock((64, 64), PhasorNetworks.normalize_to_unit_circle; gate=:rezero, alpha0=0.1f0, branch_init_scale=0.1f0, use_bias=true))
end
global chain = Lux.Chain(chain, PhasorNetworks.PhasorDense(64 => 10, PhasorNetworks.normalize_to_unit_circle; use_bias=true))

ps, st = Lux.setup(rng, chain)
st = Lux.initialstates(rng, chain)

opt = PhasorNetworks.LockinEP(ε=0.1f0, ω_p=0.02f0, n_cycles=8, T_warmup_cycles=2, T_free=200, dt=0.5f0)
opt_state = Optimisers.setup(Optimisers.Adam(0.001f0), ps)

codebook = PhasorNetworks.normalize_to_unit_circle(randn(rng, ComplexF32, 10, 10))

for epoch in 1:2
    epoch_loss = 0f0
    n_batches = 0
    for (x, y) in train_loader
        x = PhasorNetworks.phase_to_potential(Float32.(x))
        y_onehot = onehotbatch(y, 0:9)
        batch_cost = PhasorNetworks.CodebookCost(codebook, y_onehot)
        grads, st = PhasorNetworks.ep_gradient(opt, chain, ps, st, x, y_onehot, batch_cost)
        ps, opt_state = Optimisers.update(opt_state, ps, grads)
        
        # Compute loss
        y_onehot = onehotbatch(y, 0:9)
        batch_cost = PhasorNetworks.CodebookCost(codebook, y_onehot)
        s_free = PhasorNetworks.phasor_settle(chain, ps, st, x, batch_cost, 0f0; T=200, dt=0.5f0)
        loss = PhasorNetworks.ep_loss(batch_cost, s_free[end])
        epoch_loss += loss
        n_batches += 1
    end
    println("Epoch $epoch: loss=$(epoch_loss/n_batches)")
    
    # Evaluate
    correct = 0
    total = 0
    for (x, y) in test_loader
        x = PhasorNetworks.phase_to_potential(Float32.(x))
        y_onehot = onehotbatch(y, 0:9)
        batch_cost = PhasorNetworks.CodebookCost(codebook, y_onehot)
        s_free = PhasorNetworks.phasor_settle(chain, ps, st, x, batch_cost, 0f0; T=200, dt=0.5f0)
        z_out = s_free[end]
        sims = PhasorNetworks.similarity(z_out, codebook)
        preds = argmax(sims, dims=1) .+ 0
        c = sum(preds .== y)
        println("  DEBUG: c=$c, total=$(length(y))")
        correct += c
        total += length(y)
    end
    println("  Final: correct=$correct, total=$total, acc=$(correct/total)")
end