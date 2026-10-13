using PhasorNetworks
using Test
using MLDataDevices: AbstractGPUDevice, CPUDevice

# Which vendor backend (if any) runtests.jl loaded. When this file is run
# on its own, nothing is loaded and the CPU expectations apply.
const _TEST_BACKEND = @isdefined(PHASOR_BACKEND) ? PHASOR_BACKEND : :cpu

@testset "Backend Abstraction" begin
    @testset "Args construction" begin
        # Default backend resolves at run time to whatever is loaded.
        a = PhasorNetworks.Args()
        @test a.backend == :auto

        # Explicit backend
        a_cpu = PhasorNetworks.Args(backend=:cpu)
        @test a_cpu.backend == :cpu

        # Backward compat: use_cuda kwarg
        a_old = PhasorNetworks.Args(use_cuda=false)
        @test a_old.backend == :cpu

        a_old2 = PhasorNetworks.Args(use_cuda=true)
        @test a_old2.backend == :cuda
    end

    @testset "Args use_cuda property compat" begin
        a = PhasorNetworks.Args(backend=:cuda)
        @test a.use_cuda == true

        a2 = PhasorNetworks.Args(backend=:cpu)
        @test a2.use_cuda == false

        # Write via use_cuda
        a3 = PhasorNetworks.Args()
        a3.use_cuda = false
        @test a3.backend == :cpu

        a3.use_cuda = true
        @test a3.backend == :cuda
    end

    @testset "select_device" begin
        @test select_device(:cpu) isa CPUDevice

        # :auto never throws: a GPU device when a backend is loaded and
        # functional, the CPU otherwise.
        dev_auto = select_device(:auto)
        if _TEST_BACKEND == :cpu
            @test dev_auto isa CPUDevice
        else
            @test dev_auto isa AbstractGPUDevice
        end

        # Vendor backends are provided by package extensions: without the
        # vendor package loaded, asking for one is an error that names the
        # package to load. With it loaded, a device comes back.
        for backend in (:cuda, :oneapi)
            if backend == _TEST_BACKEND
                @test select_device(backend) isa AbstractGPUDevice
            else
                @test_throws ErrorException select_device(backend)
            end
        end

        # Unknown backend should error
        @test_throws ErrorException select_device(:invalid)
    end

    @testset "on_gpu / to_device_of / reclaim_device!" begin
        # CPU arrays should return false
        @test on_gpu([1, 2, 3]) == false
        @test on_gpu(zeros(Float32, 3)) == false

        # Multiple CPU arrays
        @test on_gpu([1, 2], [3, 4]) == false

        # Host reference: identity.
        x = Float32[1, 2, 3]
        @test to_device_of(zeros(Float32, 2), x) == x
        @test to_device_of(zeros(Float32, 2), x) isa Array

        # Generic reclaim hook is a no-op on the CPU device.
        @test reclaim_device!(select_device(:cpu)) === nothing
    end
end
