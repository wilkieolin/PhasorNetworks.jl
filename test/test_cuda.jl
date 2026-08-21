# This file is intended to be included and run from runtests.jl
# Ensure that runtests.jl has already loaded PhasorNetworks and other common dependencies.

using CUDA
using Adapt
using KernelAbstractions
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

    # Phasor-EP device parity (defined in test_ep.jl, which runtests.jl
    # includes before reaching here).
    ep_gpu_parity_tests(gpu_device())

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

# Direct compile+correctness coverage for the raised-cosine KA kernels.
#
# Regression guard for the bug where `pi_f32` was a non-`const` module global:
# referencing it inside a `@kernel` made the type `Any`, so every downstream
# op (`*`, `/`, `cos`, `+`) became a dynamic dispatch and the kernel failed to
# compile with `InvalidIRError: unsupported dynamic function invocation`.
# These kernels back `spike_current(::SpikeTrainGPU)` / `bias_current` — a
# central GPU spiking path that the bundling test above does not exercise.
# We invoke each kernel directly (forcing GPU codegen) and check its output
# against the scalar host helper.
function gpu_kernel_compile_tests()
    if !(CUDA.functional() || ONEAPI_AVAILABLE)
        @info "No functional GPU; skipping gpu_kernel_compile_tests."
        return
    end

    @testset "GPU raised-cosine kernel compilation" begin
        dev = gpu_device()
        t = 0.5f0
        t_sigma = 0.1f0
        t_period = 1.0f0
        times_cpu = collect(range(0.0f0, 1.0f0; length = 32))
        times = times_cpu |> dev
        backend = KernelAbstractions.get_backend(times)
        n = length(times)

        # raised_cosine
        out = KernelAbstractions.zeros(backend, Float32, n)
        PhasorNetworks.raised_cosine_kernel_ka!(backend)(out, times, t, t_sigma; ndrange = n)
        KernelAbstractions.synchronize(backend)
        ref = PhasorNetworks.raised_cosine_kernel_gpu.(times_cpu, t, t_sigma)
        @test maximum(abs.(Array(out) .- ref)) < 1e-5

        # periodic_raised_cosine
        out_p = KernelAbstractions.zeros(backend, Float32, n)
        PhasorNetworks.periodic_raised_cosine_kernel_ka!(backend)(out_p, times, t, t_sigma, t_period; ndrange = n)
        KernelAbstractions.synchronize(backend)
        ref_p = PhasorNetworks.periodic_raised_cosine_kernel_gpu.(times_cpu, t, t_sigma, t_period)
        @test maximum(abs.(Array(out_p) .- ref_p)) < 1e-5

        # gaussian (no globals — sanity baseline)
        out_g = KernelAbstractions.zeros(backend, Float32, n)
        PhasorNetworks.gaussian_kernel_ka!(backend)(out_g, times, t, t_sigma; ndrange = n)
        KernelAbstractions.synchronize(backend)
        ref_g = PhasorNetworks.gaussian_kernel_gpu.(times_cpu, t, t_sigma)
        @test maximum(abs.(Array(out_g) .- ref_g)) < 1e-5
    end
end