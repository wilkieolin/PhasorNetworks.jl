function network_tests()
    #load the dataset and a single batch for testing
    args = Args()
    train_loader, test_loader = getdata(args)
    x, y = first(train_loader)

    model, ps, st = build_mlp(args)
    spk_model, _, _ = build_spiking_mlp(args, spk_args)
    ode_model, _, _ = build_ode_mlp(args, spk_args)

    pretrain_chk = correlation_test(model, spk_model, ps, st, x)
    train_chk, ps_train, st_train = train_test(model, args, ps, st, train_loader, test_loader)
    acc_chk = accuracy_test(model, ps_train, st_train, test_loader)
    posttrain_chk = correlation_test(model, spk_model, ps_train, st_train, x)
    spk_acc_chk = spiking_accuracy_test(spk_model, ps_train, st_train, [(x, y),])
    y_cor_chk, lval_chk, grad_chk = ode_correlation(model, ode_model, ps, st, x, y)

    all_pass = reduce(*, [pretrain_chk,
                        train_chk, 
                        acc_chk, 
                        posttrain_chk, 
                        spk_acc_chk,
                        lval_chk,
                        grad_chk])

    return all_pass
end

function bullseye_data(n_s::Int, rng::AbstractRNG)
    d = Normal(0.0, 0.08)
    #determine the class labels
    y = rand(rng, (0, 1), n_s)
    #determine the polar coordinates
    r = rand(rng, d, n_s) .+ (0.4 .* y)
    phi = (rand(rng, Float64, n_s) .- 1) .* (2 * pi)
    #convert to cartesian
    x_x = r .* cos.(phi)
    x_y = r .* sin.(phi)

    data = Float32.(cat(x_x, x_y, dims=2)' )
    labels = onehotbatch(y, 0:1)

    return data, labels
end

function getdata(args)
    #make synthetic bullseye data (no needing an external data repository)
    train_loader = [bullseye_data(args.batchsize, args.rng) for i in 1:100];
    test_loader = [bullseye_data(args.batchsize, args.rng) for i in 1:10];

    return train_loader, test_loader
end

function build_mlp(args)
    phasor_model = Chain(LayerNorm((2,)), 
                x -> tanh_fast.(x), 
                x -> x, 
                PhasorDense(2 => 128), 
                x -> x,
                PhasorDense(128 => 2))
    ps, st = Lux.setup(args.rng, phasor_model)
    return phasor_model, ps, st
end

function build_spiking_mlp(args, spk_args)
    phasor_model = Chain(LayerNorm((2,)), 
                x -> tanh_fast.(x), 
                MakeSpiking(spk_args, repeats), 
                PhasorDense(2 => 128), 
                x -> x,
                PhasorDense(128 => 2))
    ps, st = Lux.setup(args.rng, phasor_model)
    return phasor_model, ps, st
end

function build_ode_mlp(args, spk_args)
    ode_model = Chain(LayerNorm((2,)),
                x -> tanh_fast.(x),
                x -> phase_to_current(x, spk_args=spk_args, tspan=(0.0, 10.0)),
                PhasorDense(2 => 128, return_solution=true),
                x -> mean_phase(x, 1, spk_args=spk_args, offset=0.0),
                PhasorDense(128 => 2))
    ps, st = Lux.setup(args.rng, ode_model)
    return ode_model, ps, st
end

function ode_correlation(model, ode_model, ps, st, x, y)
    @info "Running ODE correlation test..."
    y_f, _ = model(x, ps, st)
    y_ode, _ = ode_model(x, ps, st)
    y_cor_chk = cor_realvals(vec(y_f), vec(y_ode)) > 0.90
    @test y_cor_chk

    psf = ComponentArray(ps)
    lval, grads = withgradient(p -> mean(quadrature_loss(model(x, p, st)[1], y)), psf)
    lval_ode, grads_ode = withgradient(p -> mean(quadrature_loss(ode_model(x, p, st)[1], y)), psf)
    lval_chk = abs(lval_ode - lval) < 0.02
    grad_chk = cor_realvals(vec(grads[1].layer_4.weight), vec(grads_ode[1].layer_4.weight)) > 0.95
    @test lval_chk
    @test grad_chk

    return y_cor_chk, lval_chk, grad_chk
end

function test_correlation(model, spk_model, ps, st, x)
    #make the regular (static) call
    y, _ = model(x, ps, st)
    #make the spiking (dynamic) call
    y_spk, _ = spk_model(x, ps, st)
    yp = train_to_phase(y_spk)
    #measure the correlation between results
    c = cycle_correlation(y, yp)
    return c
end

function correlation_test(model, spk_model, ps, st, x)
    @info "Running spiking correlation test..."
    c_naive = test_correlation(model, spk_model, ps, st, x)
    #test the final full cycle of the network - use 70% correlation as the baseline
    corr_chk = c_naive[end-1] > 0.70
    @test corr_chk
    return corr_chk
end

function loss(x, y, model, ps, st)
    y_pred, st = model(x, ps, st)
    return mean(quadrature_loss(y_pred, y)), st
end


function train(model, ps, st, train_loader, args; verbose::Bool = false)
     if CUDA.functional() && args.use_cuda
        @info "Training on CUDA GPU"
        CUDA.allowscalar(false)
        device = gpu_device()
    else
        @info "Training on CPU"
        device = cpu_device()
    end

    ## Optimizer
    opt_state = Optimisers.setup(Adam(args.η), ps)
    losses = []

    ## Training
    for epoch in 1:args.epochs
        for (x, y) in train_loader
            x = x |> device
            y = y |> device
            
            lf = p -> loss(x, y, model, p, st)[1]
            lossval, gs = withgradient(lf, ps)
            if verbose
                println(reduce(*, ["Epoch ", string(epoch), " loss: ", string(lossval)]))
            end
            append!(losses, lossval)
            opt_state, ps = Optimisers.update(opt_state, ps, gs[1]) ## update parameters
        end
    end

    return losses, ps, st
end

function train_test(model, args, ps, st, train_loader, test_loader)
    @info "Running training test..."

    losses, ps, st = train(model, ps, st, train_loader, args)

    #check the final loss against the usual ending value
    loss_check = losses[end] < 0.36
    @test loss_check

    return loss_check, ps, st
end

function accuracy_test(model, ps, st, test_loader)
    _, accuracy = loss_and_accuracy(test_loader, model, ps, st)
    #usually reaches ~80% after 6 epochs
    acc_check = accuracy > 0.75
    @test acc_check
    return acc_check
end

function spiking_accuracy_test(model, ps, st, test_batch)
    @info "Running spiking accuracy test..."
    acc = spiking_accuracy(test_batch, model, ps, st, repeats)
    #make sure accuracy is above the baseline (~70% for spiking)
    acc_check = acc[end-1] > 0.70
    @test acc_check
    return acc_check
end

 