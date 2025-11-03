# This file is intended to be included and run from runtests.jl
# Ensure that runtests.jl has already loaded PhasorNetworks and other common dependencies.

phases = -1.0f0:0.01f0:1.0f0 |> collect
offsets = 0.0f0:0.25f0:1.0f0 |> collect

function domain_tests()
    @testset "Domain Conversions" begin
        @info "Running domain conversion tests..."
        phase_time_test()
        phase_train_test()
        potential_phase_test()
        potential_time_test()
        cmpx_real_tests()
    end
end

function phase_time_test()
    @testset "Phase <-> Time Conversion" begin
        tms = [phase_to_time(phases, spk_args=spk_args, offset=o) for o in offsets]
        phase_tms = [time_to_phase(tms[i], spk_args=spk_args, offset=o) for (i,o) in enumerate(offsets)]
        errors = [arc_error(phases .- p) for p in phase_tms]
        max_error = maximum(maximum.(errors))
        @test max_error < 5e-6
    end
end

function phase_train_test()
    @testset "Phase <-> Spike Train Conversion" begin
        trains = [phase_to_train(phases, spk_args=spk_args, offset=o) for o in offsets];
        rec_phases = [train_to_phase(t, spk_args=spk_args) for t in trains];
        errors = [mapslices(x -> x .- phases, t, dims=(1)) for t in rec_phases]
        errors = map(y -> maximum(arc_error(filter(x -> !isnan(x), y))), errors)
        max_error = maximum(maximum.(errors))
        @test max_error < 5e-6
    end
end

function potential_phase_test()
    @testset "Potential <-> Phase Conversion" begin
        tms = 0.0:0.1:6.0 |> collect
        us = [phase_to_potential(phases, tms, offset = o, spk_args=spk_args) for o in offsets]
        rec_phases = [potential_to_phase(us[i], tms, offset=offsets[i], spk_args=spk_args) for i in axes(offsets,1)]
        errors = map(y -> arc_error.(mapslices(x -> abs.(x .- phases), y, dims=1)), rec_phases)
        max_error = maximum(maximum.(errors))
        @test max_error < 5e-6
    end
end

function potential_time_test()
    @testset "Potential <-> Time (via Phase) Conversion" begin
        tms = 0.0:0.1:6.0 |> collect
        us = [phase_to_potential(phases, tms, offset = o, spk_args=spk_args) for o in offsets]
        ts = [potential_to_time(u, tms, spk_args=spk_args) for u in us]
        us_rec = [time_to_potential(t, tms, spk_args=spk_args) for t in ts]
        errors = [arc_error.(angle.(u[1]) .- angle.(u[2])) for u in zip(us, us_rec)]
        max_error = maximum(maximum.(errors))
        @test max_error < 5e-6
    end
end

function cmpx_real_tests()
    @testset "Complex <-> Real Vector Conversion" begin
        s = rand(ComplexF32, (10, 60))
        sr = cmpx_to_realvec(s)
        ss = realvec_to_cmpx(sr)
        @test all(ss .== s)
    end
end