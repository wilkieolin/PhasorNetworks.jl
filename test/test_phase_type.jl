using Test
using PhasorNetworks
using PhasorNetworks: remap_phase
using Zygote

function phase_type_tests()
    @testset "Phase Type Tests" begin
        @testset "Construction and isbits" begin
            p = Phase(0.5f0)
            @test p.value == 0.5f0
            @test isbitstype(Phase)
            @test sizeof(Phase) == 4

            # Construction from different numeric types
            @test Phase(0.5).value == 0.5f0
            @test Phase(1).value == 1.0f0
            @test Phase(0).value == 0.0f0

            # NaN is allowed (sub-threshold sentinel)
            p_nan = Phase(NaN32)
            @test isnan(p_nan)
        end

        @testset "Conversion and promotion" begin
            p = Phase(0.5f0)

            # Conversion to Float32/Float64
            @test Float32(p) === 0.5f0
            @test Float64(p) === 0.5
            @test float(p) === 0.5f0

            # Promotion: Phase + Phase -> Float32
            @test Phase(0.3f0) + Phase(0.2f0) isa Float32
            @test Phase(0.3f0) + Phase(0.2f0) ≈ 0.5f0

            # Promotion: Phase + Float32 -> Float32
            @test Phase(0.5f0) + 0.1f0 isa Float32

            # Promotion: Phase + Float64 -> Float64
            @test Phase(0.5f0) + 0.1 isa Float64

            # Phase is a subtype of Real
            @test Phase <: Real
        end

        @testset "Array operations" begin
            arr = Phase.([0.1f0, 0.2f0, 0.3f0])
            @test eltype(arr) == Phase
            @test length(arr) == 3

            # Broadcasting with Float32 promotes to Float32
            result = arr .+ 0.1f0
            @test eltype(result) == Float32

            # Can create Matrix of Phase
            mat = Phase.(rand(Float32, 3, 4) .* 2.0f0 .- 1.0f0)
            @test size(mat) == (3, 4)
            @test eltype(mat) == Phase

            # Float32 conversion of arrays
            f32_arr = Float32.(arr)
            @test eltype(f32_arr) == Float32
            @test f32_arr ≈ [0.1f0, 0.2f0, 0.3f0]
        end

        @testset "Comparison and predicates" begin
            @test Phase(0.5f0) == Phase(0.5f0)
            @test Phase(0.3f0) < Phase(0.5f0)
            @test isfinite(Phase(0.5f0))
            @test !isnan(Phase(0.5f0))
            @test !isinf(Phase(0.5f0))
            @test zero(Phase) == Phase(0.0f0)
            @test one(Phase) == Phase(1.0f0)
        end

        @testset "AD transparency" begin
            # Gradient should pass through Phase wrapping as identity
            grad = Zygote.gradient(x -> sum(Float32.(Phase.(x))), [1.0f0, 2.0f0])
            @test grad[1] ≈ [1.0f0, 1.0f0]

            # Gradient through arithmetic
            grad2 = Zygote.gradient(x -> sum(Phase.(x) .* 2.0f0), [1.0f0, 2.0f0])
            @test grad2[1] ≈ [2.0f0, 2.0f0]
        end

        @testset "Producer functions return Phase" begin
            # complex_to_angle
            z = [1.0f0 + 0.0f0im, 0.0f0 + 1.0f0im]
            angles = complex_to_angle(z)
            @test eltype(angles) == Phase

            # random_symbols
            syms = random_symbols((5, 3))
            @test eltype(syms) == Phase

            # remap_phase
            remapped = remap_phase(Phase(1.5f0))
            @test remapped isa Phase

            remapped_arr = remap_phase(Phase.([1.5f0, -1.5f0]))
            @test eltype(remapped_arr) == Phase

            # angular_mean
            phases = Phase.(rand(Float32, 10, 3) .* 2.0f0 .- 1.0f0)
            am = angular_mean(phases, dims=1)
            @test eltype(am) == Phase
        end
    end
end
