function network_tests()
    #load the dataset and a single batch for testing
    args = Args()
    train_loader, test_loader = getdata(args)
    x, y = first(train_loader)

    model, ps, st = build_mlp(args)
    pretrain_chk = correlation_test(model, ps, st, x)
    train_chk, ps_train, st_train = train_test(model, args, ps, st, train_loader, test_loader)
    acc_chk = accuracy_test(model, ps_train, st_train, test_loader)
    posttrain_chk = correlation_test(model, ps_train, st_train, x)
    spk_acc_chk = spiking_accuracy_test(model, ps_train, st_train, [(x, y),])

    all_pass = reduce(*, [pretrain_chk, train_chk, acc_chk, posttrain_chk, spk_acc_chk])

    return all_pass
end

"""
Check if a value is within the bounds determined by epsilon
"""
function in_tolerance(x)
    return abs(x) < epsilon ? true : false
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

    return cat(x_x, x_y, dims=2)', onehotbatch(y, 0:1)
end

function getdata(args)
    #make synthetic bullseye data (no needing an external data repository)
    train_loader = [bullseye_data(args.batchsize, args.rng) for i in 1:100];
    test_loader = [bullseye_data(args.batchsize, args.rng) for i in 1:10];

    return train_loader, test_loader
end

function build_mlp(args)
    phasor_model = Chain(LayerNorm((2,)), PhasorDense(2 => 128), PhasorDense(128 => 2))
    ps, st = Lux.setup(args.rng, phasor_model)
    return phasor_model, ps, st
end

function test_correlation(model, ps, st, x)
    #make the regular (static) call
    y, _ = model(x, ps, st)
    #make the spiking (dynamic) call
    x_train = phase_to_train(x, spk_args, repeats = repeats)
    x_call = SpikingCall(x_train, spk_args, tspan)
    y_spk, _ = model(x_call, ps, st)
    yp = train_to_phase(y_spk)
    #measure the correlation between results
    c = cycle_correlation(y, yp)
    return c
end

function correlation_test(model, ps, st, x)
    @info "Running spiking correlation test..."
    c_naive = test_correlation(model, ps, st, x)
    #test the final full cycle of the network - use 70% correlation as the baseline
    corr_chk = c_naive[end-1] > 0.70
    @test corr_chk
    return corr_chk
end

function train_test(model, args, ps, st, train_loader, test_loader)
    @info "Running training test..."

    # if CUDA.functional() && args.use_cuda
    #     @info "Training on CUDA GPU"
    #     CUDA.allowscalar(false)
    #     device = gpu_device()
    # else
        @info "Training on CPU"
        device = cpu_device()
    # end

    ## Construct model
    # model = model |> device

    ## Optimizer
    opt_state = Optimisers.setup(Adam(args.η), ps)
    losses = []

    ## Training
    for epoch in 1:args.epochs
        for (x, y) in train_loader
            x, y = device(x), device(y) ## transfer data to device
            loss, gs = withgradient(p -> mean(quadrature_loss(model(x, p, st)[1], y)), ps) ## compute gradient of the loss
            append!(losses, loss)
            opt_state, ps = Optimisers.update(opt_state, ps, gs[1]) ## update parameters
        end

        ## Report on train and test
        train_loss, train_acc = loss_and_accuracy(train_loader, model, ps, st)
        test_loss, test_acc = loss_and_accuracy(test_loader, model, ps, st)
    end

    #check the final loss against the usual ending value
    loss_error = 0.03 - losses[end] 
    loss_check = in_tolerance(loss_error)
    @test loss_check

    return loss_check, ps, st
end

function accuracy_test(model, ps, st, test_loader)
    _, accuracy = loss_and_accuracy(test_loader, model, ps, st)
    #usually reaches ~83% after 6 epochs
    acc_check = accuracy > 0.80
    @test acc_check
    return acc_check
end

function spiking_accuracy_test(model, ps, st, test_batch)
    @info "Running spiking accuracy test..."
    acc = spiking_accuracy(test_batch, model, ps, st,
                            spk_args = spk_args, 
                            t_span = tspan, 
                            repeats = repeats)
    #make sure accuracy is above the baseline (~80%)
    acc_check = acc[end-1] > 0.80
    @test acc_check
    return acc_check
end

 