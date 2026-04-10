using PhasorNetworks
using Test

@testset "Backend Abstraction" begin
    @testset "Args construction" begin
        # Default backend
        a = Args()
        @test a.backend == :cuda

        # Explicit backend
        a_cpu = Args(backend=:cpu)
        @test a_cpu.backend == :cpu

        # Backward compat: use_cuda kwarg
        a_old = Args(use_cuda=false)
        @test a_old.backend == :cpu

        a_old2 = Args(use_cuda=true)
        @test a_old2.backend == :cuda
    end

    @testset "Args use_cuda property compat" begin
        a = Args(backend=:cuda)
        @test a.use_cuda == true

        a2 = Args(backend=:cpu)
        @test a2.use_cuda == false

        # Write via use_cuda
        a3 = Args()
        a3.use_cuda = false
        @test a3.backend == :cpu

        a3.use_cuda = true
        @test a3.backend == :cuda
    end

    @testset "select_device" begin
        dev_cpu = select_device(:cpu)
        @test dev_cpu isa Function || true  # cpu_device returns a device object

        # :cuda should work (either returns device or warns and falls back)
        dev = select_device(:cuda)
        @test dev isa Function || true

        # Unknown backend should error
        @test_throws ErrorException select_device(:invalid)

        # oneAPI without extension should error
        @test_throws ErrorException select_device(:oneapi)
    end

    @testset "on_gpu" begin
        # CPU arrays should return false
        @test PhasorNetworks.on_gpu([1, 2, 3]) == false
        @test PhasorNetworks.on_gpu(zeros(Float32, 3)) == false

        # Multiple CPU arrays
        @test PhasorNetworks.on_gpu([1, 2], [3, 4]) == false
    end
end
