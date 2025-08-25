# This file is intended to be included and run from runtests.jl
# Ensure that runtests.jl has already loaded PhasorNetworks and other common dependencies.

function network_tests()
    @testset "Network Tests" begin
        @info "Running network tests..."

        #load the dataset and a single batch for testing
        args = Args()
        train_loader, test_loader = getdata(args)
        x, y = first(train_loader)

        model, ps, st = build_mlp(args)
        model_nb, ps_nb, st_nb = build_mlp_no_bias(args)
        spk_model, _, _ = build_spiking_mlp(args, spk_args)
        ode_model, _, _ = build_ode_mlp(args, spk_args)

        correlation_test(model, spk_model, ps, st, x)
        ps_train, st_train = train_test(model, args, ps, st, train_loader, test_loader)
        accuracy_test(model, ps_train, st_train, test_loader, args)
        correlation_test(model, spk_model, ps_train, st_train, x)
        spiking_accuracy_test(spk_model, ps_train, st_train, [(x, y),], args)
        ode_correlation(model_nb, ode_model, ps_nb, st_nb, x, y)
    end
end
 
function getdata(args)
    #make synthetic bullseye data (no needing an external data repository)
    train_loader = [bullseye_data(args.batchsize, args.rng) for i in 1:100];
    test_loader = [bullseye_data(args.batchsize, args.rng) for i in 1:10];

    return train_loader, test_loader
end

function build_mlp(args)
    phasor_model = Chain(x -> tanh_fast.(x),
                x -> x, 
                PhasorDense(2 => 128, complex_to_angle), 
                x -> x,
                PhasorDense(128 => 2, complex_to_angle))
    ps, st = Lux.setup(args.rng, phasor_model)
    return phasor_model, ps, st
end

function build_mlp_no_bias(args)
    phasor_model = Chain(x -> tanh_fast.(x),
                x -> x, 
                PhasorDense(2 => 128, complex_to_angle, use_bias=false,), 
                x -> x,
                PhasorDense(128 => 2, complex_to_angle, use_bias=false,))
    ps, st = Lux.setup(args.rng, phasor_model)
    return phasor_model, ps, st
end

function build_spiking_mlp(args, spk_args)
    phasor_model = Chain(x -> tanh_fast.(x), 
                MakeSpiking(spk_args, repeats), 
                PhasorDense(2 => 128, soft_angle), 
                x -> x,
                PhasorDense(128 => 2, soft_angle))
    ps, st = Lux.setup(args.rng, phasor_model)
    return phasor_model, ps, st
end

function build_ode_mlp(args, spk_args)
    ode_model = Chain(
                x -> tanh_fast.(x),
                x -> phase_to_current(x, spk_args=spk_args, tspan=(0.0f0, 10.0f0)),
                PhasorDense(2 => 128, complex_to_angle, return_solution=true, use_bias=false),
                x -> end_phase(x, spk_args=spk_args, offset=0.0f0),
                PhasorDense(128 => 2, complex_to_angle, use_bias=false))
    ps, st = Lux.setup(args.rng, ode_model)
    return ode_model, ps, st
end

function ode_correlation(model, ode_model, ps, st, x, y)
    @testset "ODE Correlation Test" begin
        @info "Running ODE correlation test..."
        y_f, _ = model(x, ps, st)
        y_ode, _ = ode_model(x, ps, st)
        @test cor_realvals(vec(y_f), vec(y_ode)) > 0.90

        psf = ComponentArray(ps)
        lval, grads = withgradient(p -> mean(quadrature_loss(model(x, p, st)[1], y)), psf)
        lval_ode, grads_ode = withgradient(p -> mean(quadrature_loss(ode_model(x, p, st)[1], y)), psf)
        @test abs(lval_ode - lval) < 0.02
        @test cor_realvals(vec(real.(grads[1].layer_3.layer.weight)), vec(real.(grads_ode[1].layer_3.layer.weight))) > 0.95
    end
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
    @testset "Spiking correlation test" begin
        c_naive = test_correlation(model, spk_model, ps, st, x)
        #test the final full cycle of the network - use 70% correlation as the baseline
        @test c_naive[end-1] > 0.70
    end
end

function loss(x, y, model, ps, st)
    y_pred, _ = model(x, ps, st)
    return mean(quadrature_loss(y_pred, y))
end

function train_test(model, args, ps, st, train_loader, test_loader)
    @testset "Training Test" begin
        losses, ps, st = train(model, ps, st, train_loader, loss, args)

        #check the final loss against the usual ending value
        @test losses[end] < 0.36
        return ps, st
    end
end

function accuracy_test(model, ps, st, test_loader, args)
    @testset "Accuracy Test" begin
        _, accuracy = loss_and_accuracy(test_loader, model, ps, st, args)
        #usually reaches ~80% after 6 epochs
        @test accuracy > 0.75
    end
end

function spiking_accuracy_test(model, ps, st, test_batch, args)
    @testset "Spiking Accuracy Test" begin
        @info "Running spiking accuracy test..."
        acc = spiking_accuracy(test_batch, model, ps, st, args)
        #make sure accuracy is above the baseline (~70% for spiking)
        @test acc[end-1] > 0.70
    end
end

 