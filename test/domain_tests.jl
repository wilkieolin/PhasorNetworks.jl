phases = -1.0:0.01:1.0 |> collect
offsets = 0.0:0.25:1.0 |> collect

function domain_tests()
    phase_time_test()
    phase_train_test()
end

function phase_time_test()
    tms = [phase_to_time(phases, spk_args=sa, offset=o) for o in offsets]
    phase_tms = [time_to_phase(tms[i], spk_args=sa, offset=o) for (i,o) in enumerate(offsets)]
    errors = [maximum(arc_error(phases .- p)) for p in phase_tms]
    max_error = maximum(errors)
    @test max_error < 1e-6
end

function phase_train_test()
    trains = [phase_to_train(phases, spk_args=sa, offset=o) for o in offsets];
    rec_phases = [train_to_phase(t, sa) for t in trains];
    errors = [mapslices(x -> x .- phases, t, dims=(2)) for t in rec_phases]
    max_error = maximum(errors)
    @test max_error < 1e-6
end

function potential_phase_test()
    tms = 0.0:0.1:6.0 |> collect
    potentials = [phase_to_potential(phases, tms, offset = o, spk_args=sa) for o in offsets]
    rec_phases = [potential_to_phase(us[i], tms, dim=2, offset=offsets[i], spk_args=sa) for i in axes(offsets,1)]
    errors = map(y -> arc_error.(mapslices(x -> abs.(x .- phases), y, dims=1)), rec_phases)
    max_error = maximum(maximum.(errors))
    @test max_error < 1e-6
end
