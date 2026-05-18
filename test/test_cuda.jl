# This file is intended to be included and run from runtests.jl
# Ensure that runtests.jl has already loaded PhasorNetworks and other common dependencies.

using CUDA
using Adapt
using DifferentialEquations: Tsit5

function cuda_core_tests()
    spk_args_cuda = SpikingArgs(solver = Tsit5(),
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

    if !CUDA.functional()
        # This else block might not be strictly necessary if runtests.jl already skips calling cuda_core_tests,
        # but kept for robustness if test_cuda.jl were somehow run directly in a non-CUDA env.
        @info "CUDA not functional. Skipping GPU-specific operations within cuda_core_tests."
        return
    end

    @testset "CUDA Core Functionality Tests" begin
        sol_cpu, err_cpu = bundling_test(spk_args_cuda, "cpu")
        sol_gpu, err_gpu = bundling_test(spk_args_cuda, "gpu")

        # CPU and GPU bundle the same spike train through the same ODE; their
        # phase outputs should agree. But phases live on a circle of period 2
        # (units of π), so a raw subtraction `err_cpu .- err_gpu` is wrong at
        # the wrap point: a CPU phase of `+1` and a GPU phase of `−1` are the
        # *same* angle (π), differing only in branch choice. Use the existing
        # `arc_error` (sin(π·δ)) which is the canonical circular distance —
        # smooth, zero at δ = 0, and zero at δ = ±2 (full wrap). Without this,
        # the 22 degenerate grid points (where the static bundle is identically
        # zero — antipodal phases summing to 0 + 0i, e.g. x=0.5/y=−0.5) flip
        # sign on round-off and produce spurious ~2.0 errors.
        max_comparative_error = maximum(abs.(arc_error(Float32.(err_cpu) .- Float32.(err_gpu))))
        @test max_comparative_error < 1e-3
    end
end