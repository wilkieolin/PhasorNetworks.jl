# This file is intended to be included and run from runtests.jl
# Ensure that runtests.jl has already loaded PhasorNetworks and other common dependencies.

"""
Run all the basic VSA tests and check they pass
"""
function vsa_tests()
    @testset "VSA Tests" begin
        @info "Running VSA tests..."

        test_orthogonal()
        test_outer()
        test_binding()
        test_bundling()
    end
end

"""
Check if a value is within the bounds determined by epsilon
"""
function in_tolerance(x)
    return abs(x) < epsilon ? true : false
end

"""
Produce all possible pairs of angles
"""
function make_angle_pairs()
    phases = collect([[x, y] for x in range(-1.0, 1.0, n_x), y in range(-1.0, 1.0, n_y)]) |> stack
    phases = reshape(phases, (1,2,:))
    return phases
end

"""
Remove NaN values
"""
function remove_nan(x)
    y = filter(x -> !isnan(x), x)
    return y
end

"""
Produce the error of angular values from zero
"""
function sin_error(x)
    return sin.(pi .* x) |> mean
end

"""
Basic test that random VSA symbols are orthogonal
"""
function test_orthogonal()
    @testset "Orthogonality Test" begin
        x = random_symbols((100, 1024))
        y = random_symbols((100, 1024))

        s = similarity(x, y)
        @test mean(s) < 0.1 ? true : false
    end
end

"""
Test the outer similarity function with normal and spiking arguments
"""
function test_outer()
    @testset "Outer Similarity Test" begin
        function check_phase(matrix)
            in_phase = diag(matrix)
            anti_phase = diag(matrix, convert(Int, round(n_x / 2)))

            v1 = reduce(*, map(x -> x > 1.0 - epsilon, in_phase))
            v2 = reduce(*, map(x -> x < -1.0 + epsilon, anti_phase))
            return v1, v2
        end

        #check the non-spiking implementation
        phase_x = reshape(range(-1.0, 1.0, n_x), (n_vsa, n_x, 1)) |> collect
        phase_y = reshape(range(-1.0, 1.0, n_y), (n_vsa, n_y, 1)) |> collect
        sims = similarity_outer(phase_x, phase_y, dims=2)[:,:,1]
        v1, v2 = check_phase(sims)
        @test v1 && v2

        #check the spiking implementation
        st_x = phase_to_train(phase_x, spk_args = spk_args, repeats = repeats)
        st_y = phase_to_train(phase_y, spk_args = spk_args, repeats = repeats)
        sims_2 = similarity_outer(st_x, st_y, tspan=tspan, spk_args = spk_args)
        #check at the last time step
        sims_spk = sims_2[end][:,:,1]
        v1s, v2s = check_phase(sims_spk)
        @test v1s && v2s

        #check the cross-implementation error
        avg_error = mean(sims .- sims_spk)
        @test avg_error < epsilon
    end
end


"""
Test the binding/unbinding functions in both versions & cross-check errors
"""
function test_binding()
    @testset "Binding Test" begin
        phases = make_angle_pairs()
        #check binding and unbinding functions
        b = v_bind(phases, dims=2)
        ub = v_unbind(phases[1:1,1:1,:], phases[1:1,2:2,:])

        #check binding via oscillators
        st_x = phase_to_train(phases[1:1,1:1,:], spk_args=spk_args, repeats = repeats)
        st_y = phase_to_train(phases[1:1,2:2,:], spk_args=spk_args, repeats = repeats)
        soln = v_bind(st_x, st_y, spk_args=spk_args, tspan=tspan, return_solution=true);
        decoded = solution_to_phase(soln, tbase, spk_args=spk_args);
        u_err = mean(decoded[1,:,:,:] .- b[1,:,:], dims=(1,2))[end]
        u_check = in_tolerance(u_err)
        @test u_check

        #check with spiking outputs
        b2 = v_bind(st_x, st_y, spk_args=spk_args, tspan=tspan, return_solution=false)
        b2d = train_to_phase(b2, spk_args=spk_args)
        enc_error = remove_nan(vec(b2d[5,:,:,:]) .- vec(b)) |> mean
        enc_check = in_tolerance(enc_error)
        @test enc_check 

        #check unbinding operation
        ub_soln = v_unbind(st_x, st_y, tspan=tspan, spk_args=spk_args, return_solution=true)
        decoded = solution_to_phase(ub_soln, tbase, spk_args=spk_args)
        err = mean(decoded[1,:,:,:] .- ub[1,:,:], dims=(1,2))[end]
        unbind_chk = in_tolerance(err)
        @test unbind_chk 

        #check unbinding with spiking outputs
        ub2 = v_unbind(st_x, st_y, spk_args=spk_args, tspan=tspan, return_solution=false)
        ub2d = train_to_phase(ub2, spk_args=spk_args)
        ub_enc_error = remove_nan(vec(ub2d[5,:,:,:]) .- vec(ub)) |> mean
        ub_enc_check = in_tolerance(ub_enc_error)
        @test ub_enc_check 
    end
end

function test_bundling()
    @testset "Bundling test" begin
        phases = make_angle_pairs()
        #check bundling function
        b = v_bundle(phases, dims=2);

        st = phase_to_train(phases, spk_args=spk_args, repeats=6)
        #check potential encodings
        b2_sol = v_bundle(st, dims=2, spk_args=spk_args, tspan=tspan, return_solution=true)
        b2_phase = solution_to_phase(b2_sol, tbase, spk_args=spk_args, offset=0.0)
        b2_phase_error = vec(b2_phase[1,1,:,end]) .- vec(b) |> mean
        b2_phase_check = in_tolerance(b2_phase_error)
        @test b2_phase_check 

        #check spiking encoding
        b2 = v_bundle(st, dims=2, spk_args=spk_args, tspan=tspan, return_solution=false)
        decoded = train_to_phase(b2, spk_args=spk_args)
        b2_error = remove_nan(vec(decoded[end-1,:,:,:]) .- vec(b)) |> sin_error
        b2_spike_check = in_tolerance(b2_error)
        @test b2_spike_check
    end
end