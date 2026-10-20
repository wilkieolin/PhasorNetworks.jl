# This file is intended to be included and run from runtests.jl, which
# defines `GPU_DEV` (the selected device) and `GPU_AVAILABLE`. It is
# backend-agnostic: nothing here names CUDA or oneAPI.

using Adapt
using MLDataDevices: get_device, AbstractGPUDevice
using KernelAbstractions
using DifferentialEquations: Tsit5

function gpu_core_tests()
    spk_args_gpu = SpikingArgs(solver = Tsit5(),
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

    if !GPU_AVAILABLE
        @info "No functional GPU device. Skipping gpu_core_tests."
        return
    end

    @testset "SpikeTrainGPU device placement" begin
        inds = [CartesianIndex(1, 1), CartesianIndex(2, 1), CartesianIndex(1, 2)]
        tr = SpikeTrain(inds, Float32[0.1, 0.2, 0.3], (2, 2), 0.0f0)

        # Default device: the auto-selected GPU.
        tr_gpu = SpikeTrainGPU(tr)
        @test on_gpu(tr_gpu)
        @test get_device(tr_gpu.times) isa AbstractGPUDevice
        @test get_device(tr_gpu.indices) isa AbstractGPUDevice
        @test get_device(tr_gpu.linear_indices) isa AbstractGPUDevice
        @test Array(tr_gpu.linear_indices) == [1, 2, 3]

        # Explicit device, and round trip back to the host.
        tr_gpu2 = SpikeTrainGPU(tr; device=GPU_DEV)
        tr_back = SpikeTrain(tr_gpu2)
        @test tr_back.indices == tr.indices
        @test tr_back.times == tr.times
        @test tr_back.shape == tr.shape
        @test tr_back.offset == tr.offset

        # Building from GPU-resident times follows the data's device.
        tr_gpu3 = SpikeTrainGPU(inds, tr_gpu.times, (2, 2), 0.0f0)
        @test get_device(tr_gpu3.times) isa AbstractGPUDevice

        # delay_train keeps a GPU train on the GPU, and train_to_phase
        # returns its result on the train's device.
        tr_delayed = delay_train(tr_gpu, 0.5f0, 0.0f0)
        @test tr_delayed isa SpikeTrainGPU
        @test Array(tr_delayed.times) ≈ tr.times .+ 0.5f0
        ph = train_to_phase(tr_gpu, spk_args=SpikingArgs())
        @test get_device(ph) isa AbstractGPUDevice
    end

    # Phasor-EP device parity (defined in test_ep.jl, which runtests.jl
    # includes before reaching here).
    ep_gpu_parity_tests(GPU_DEV)

    @testset "GPU Core Functionality Tests" begin
        sol_cpu, err_cpu = bundling_test(spk_args_gpu, "cpu")
        sol_gpu, err_gpu = bundling_test(spk_args_gpu, "gpu")

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
    if !GPU_AVAILABLE
        @info "No functional GPU; skipping gpu_kernel_compile_tests."
        return
    end

    @testset "GPU raised-cosine kernel compilation" begin
        dev = GPU_DEV
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

    # Issue #59: CPU and GPU `spike_current` must agree when a train carries
    # coincident spikes on one input, and both must apply `spk_scale`.
    @testset "spike_current CPU/GPU parity with duplicate indices" begin
        for scale in (1.0f0, 2.5f0)
            spk_args = SpikingArgs(spk_scale = scale)
            inds = [CartesianIndex(1, 1), CartesianIndex(1, 1), CartesianIndex(2, 1),
                    CartesianIndex(3, 2), CartesianIndex(3, 2), CartesianIndex(3, 2)]
            times = Float32[1.0, 1.005, 1.0, 0.5, 0.503, 0.51]
            tr = SpikeTrain(inds, times, (3, 2), 0.0f0)
            tr_gpu = SpikeTrainGPU(tr)
            for t in (1.0f0, 0.5f0, 0.0f0)
                c_cpu = PhasorNetworks.spike_current(tr, t, spk_args)
                c_gpu = Array(PhasorNetworks.spike_current(tr_gpu, t, spk_args))
                @test size(c_gpu) == size(c_cpu)
                @test maximum(abs.(c_gpu .- c_cpu)) < 1e-5
            end
            # Duplicates really are summed, not overwritten.
            @test PhasorNetworks.spike_current(tr, 1.0f0, spk_args)[1, 1] ≈ scale * (1.0f0 +
                PhasorNetworks.raised_cosine_kernel(Float32[1.005], 1.0f0, spk_args.t_window)[1])
        end
    end

    # `solution_to_train` on a device array: its period offsets are built on
    # the host from `times` and must be moved to the device before the
    # broadcast (failed to compile on CUDA and oneAPI before).
    @testset "solution_to_train on device matches host" begin
        spk_args = SpikingArgs()
        u = randn(Xoshiro(5), ComplexF32, 3, 4, 5)
        times = collect(1.0f0:5.0f0)
        tr = solution_to_train(u, times; spk_args = spk_args, offset = 0.0f0)
        tr_gpu = solution_to_train(u |> GPU_DEV, times; spk_args = spk_args, offset = 0.0f0)
        @test on_gpu(tr_gpu)
        @test Array(tr_gpu.indices) == tr.indices
        @test maximum(abs.(Array(tr_gpu.times) .- tr.times)) < 1e-5
    end
end