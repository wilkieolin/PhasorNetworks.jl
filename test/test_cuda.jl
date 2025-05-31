# This file is intended to be included and run from runtests.jl
# Ensure that runtests.jl has already loaded PhasorNetworks and other common dependencies.

using CUDA
using Adapt
using DifferentialEquations # For Heun solver

function cuda_core_tests()
    spk_args_cuda = SpikingArgs(solver = Heun(),
                solver_args = Dict(:adaptive => false, 
                                :dt => 0.01),
                                threshold = 0.001)
    
    function bundling_test(spk_args::SpikingArgs, device="cpu")
        repeats = 6
        tspan = (0.0, repeats*1.0)
        tbase = collect(0.0:0.01:tspan[2])
        n_x = 21
        n_y = 21
        phases = collect([[x, y] for x in range(-1.0, 1.0, n_x), y in range(-1.0, 1.0, n_y)]) |> stack
        phases = reshape(phases, (1,2,:))
        cdev = cpu_device()
        
        b = v_bundle(phases, dims=2)
        st = phase_to_train(phases, spk_args=spk_args, repeats=6)

        if device == "gpu"
            st = SpikeTrainGPU(st)
        end

        #check potential encodings
        b2_sol = v_bundle(st, dims=2, spk_args=spk_args, tspan=tspan, return_solution=true)
        b2_phase = solution_to_phase(b2_sol, tbase, spk_args=spk_args, offset=0.0)

        if device == "gpu"
            b2_phase = b2_phase |> cdev
        end

        b2_phase_error = vec(b2_phase[1,1,:,end]) .- vec(b)
        
        return b2_sol, b2_phase_error
    end

    if CUDA.functional()
        @testset "CUDA Core Functionality Tests" begin
            sol_cpu, err_cpu = bundling_test(spk_args_cuda, "cpu")
            sol_gpu, err_gpu = bundling_test(spk_args_cuda, "gpu")

            return err_cpu, err_gpu
            
            max_comparative_error = maximum(err_cpu .- err_gpu)
            @test max_comparative_error < 1e-3
        end
    else
        # This else block might not be strictly necessary if runtests.jl already skips calling cuda_core_tests,
        # but kept for robustness if test_cuda.jl were somehow run directly in a non-CUDA env.
        @info "CUDA not functional. Skipping GPU-specific operations within cuda_core_tests."
    end
end