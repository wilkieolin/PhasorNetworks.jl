# This file is intended to be included and run from runtests.jl
# Ensure that runtests.jl has already loaded PhasorNetworks and other common dependencies.

using Test
using PhasorNetworks

function network_tests()
    @testset "Network Tests" begin
        @info "Running network tests..."

        #load the dataset and a single batch for testing
        args = Args()
        train_loader, test_loader = getdata(args)
        x, y = first(train_loader)

        model, ps, st = build_mlp(args, true)
        model_nb, ps_nb, st_nb = build_mlp(args, false)

        spk_model, _, _ = build_spiking_mlp(args, spk_args, true)
        spk_model_nb, _, _ = build_spiking_mlp(args, spk_args, false)

        ode_model, _, _ = build_ode_mlp(args, spk_args, true)
        ode_model_nb, _, _ = build_ode_mlp(args, spk_args, false)

        #test the outputs of the normal/spiking models
        correlation_test(model, spk_model, ps, st, x, true)
        correlation_test(model_nb, spk_model_nb, ps_nb, st_nb, x, false)
        #go through training on the normal model
        ps_train, st_train = train_test(model, args, ps, st, train_loader, test_loader, true)
        #test for accuracy & re-check spiking correlation
        accuracy_test(model, ps_train, st_train, test_loader, args, true)
        correlation_test(model, spk_model, ps_train, st_train, x, true)
        #do the spiking domain accuracy test
        spiking_accuracy_test(spk_model, ps_train, st_train, [(x, y),], args, true)
        #check the gradients we get through integrating phases as currents & phases as static values match
        ode_correlation(model, ode_model, ps, st, x, y, true)
        ode_correlation(model_nb, ode_model_nb, ps_nb, st_nb, x, y, false)
    end
end
 
function getdata(args)
    #make synthetic bullseye data (no needing an external data repository)
    train_loader = [bullseye_data(args.batchsize, args.rng) for i in 1:100];
    test_loader = [bullseye_data(args.batchsize, args.rng) for i in 1:10];

    return train_loader, test_loader
end

function build_mlp(args, bias::Bool=true)
    phasor_model = Chain(x -> tanh_fast.(x),
                x -> x, 
                PhasorDense(2 => 128, complex_to_angle, use_bias=bias), 
                x -> x,
                PhasorDense(128 => 2, complex_to_angle, use_bias=bias))
    ps, st = Lux.setup(args.rng, phasor_model)
    return phasor_model, ps, st
end

function build_spiking_mlp(args, spk_args, bias::Bool=true)
    phasor_model = Chain(x -> tanh_fast.(x), 
                MakeSpiking(spk_args, repeats), 
                PhasorDense(2 => 128, soft_angle, use_bias=bias), 
                x -> x,
                PhasorDense(128 => 2, soft_angle, use_bias=bias))
    ps, st = Lux.setup(args.rng, phasor_model)
    return phasor_model, ps, st
end

function build_ode_mlp(args, spk_args, bias::Bool=true)
    ode_model = Chain(
                x -> tanh_fast.(x),
                x -> phase_to_current(x, spk_args=spk_args, tspan=(0.0f0, 10.0f0)),
                PhasorDense(2 => 128,
                        complex_to_angle,
                        return_type=SolutionType(:phase),
                        use_bias=bias),
                x -> x[end],
                PhasorDense(128 => 2,
                        complex_to_angle,
                        use_bias=bias))
    ps, st = Lux.setup(args.rng, ode_model)
    return ode_model, ps, st
end

function ode_correlation(model, ode_model, ps, st, x, y, bias::Bool=true)
    @testset "ODE Correlation Test (bias=$bias)" begin
        @info "Running ODE correlation test (bias=$bias)..."
        y_f, _ = model(x, ps, st)
        y_ode, _ = ode_model(x, ps, st)
        @test cor_realvals(vec(y_f), vec(y_ode)) > 0.90

        if !bias
            psf = ComponentArray(ps)
            lval, grads = withgradient(p -> mean(evaluate_loss(model(x, p, st)[1], y, :quadrature)), psf)
            lval_ode, grads_ode = withgradient(p -> mean(evaluate_loss(ode_model(x, p, st)[1], y, :quadrature)), psf)
            @test abs(lval_ode - lval) < 0.02
            @test cor_realvals(vec(real.(grads[1].layer_3.layer.weight)), vec(real.(grads_ode[1].layer_3.layer.weight))) > 0.95
        end
    end
end

function test_correlation(model, spk_model, ps, st, x)
    #make the regular (static) call
    y, _ = model(x, ps, st)
    #make the spiking (dynamic) call
    y_spk, _ = spk_model(x, ps, st)
    yp = train_to_phase(y_spk)
    #measure the correlation between results - cycle is now last dimension
    c = cycle_correlation(y, yp)
    return c
end

function correlation_test(model, spk_model, ps, st, x, bias::Bool=true)
    @testset "Spiking correlation test (bias=$bias)" begin
        c_naive = test_correlation(model, spk_model, ps, st, x)
        #test the final full cycle of the network - use 70% correlation as the baseline
        @test c_naive[end-1] > 0.70
    end
end

function loss(x, y, model, ps, st)
    y_pred, _ = model(x, ps, st)
    loss = mean(evaluate_loss(y_pred, y, :quadrature))
    return loss
end

function train_test(model, args, ps, st, train_loader, test_loader, bias::Bool=true)
    @testset "Training Test (bias=$bias)" begin
        losses, ps, st = train(model, ps, st, train_loader, loss, args)

        #check the final loss against the usual ending value
        @test losses[end] < 0.36
        return ps, st
    end
end

function accuracy_test(model, ps, st, test_loader, args, bias::Bool=true)
    @testset "Accuracy Test (bias=$bias)" begin
        _, accuracy = loss_and_accuracy(test_loader, model, ps, st, args, encoding=:quadrature)
        #usually reaches ~80% after 6 epochs
        @test accuracy > 0.75
    end
end

function spiking_accuracy_test(model, ps, st, test_batch, args, bias::Bool=true)
    @testset "Spiking Accuracy Test (bias=$bias)" begin
        @info "Running spiking accuracy test (bias=$bias)..."
        _, accuracy = spiking_loss_and_accuracy(test_batch, model, ps, st, args, encoding=:quadrature)
        #make sure accuracy is above the baseline (~70% for spiking)
        @test accuracy[end-1] > 0.70
    end
end

 