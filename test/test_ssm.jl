using Test
using PhasorNetworks
using Lux
using Random: Xoshiro
using Zygote: withgradient
using Statistics: mean

function ssm_tests()
    @testset "SSM Tests" begin
        @info "Running SSM tests..."
        phasor_kernel_tests()
        causal_conv_tests()
        hippo_tests()
        phasor_ssm_layer_tests()
        ssm_readout_tests()
        psk_encode_tests()
        impulse_encode_tests()
        ssm_chain_tests()
        ssm_gradient_tests()
    end
end

function phasor_kernel_tests()
    @testset "phasor_kernel" begin
        C, L = 8, 16
        λ = fill(-0.1f0, C)
        ω = Float32.(collect(range(0.2f0, 2.5f0; length=C)))

        K = phasor_kernel(λ, ω, 1f0, L)

        # Shape and type
        @test size(K) == (C, L)
        @test eltype(K) <: Complex

        # All values finite
        @test all(isfinite, K)

        # Decay behavior: magnitudes should decrease for negative λ
        mags = abs.(K)
        for c in 1:C
            @test mags[c, 1] > mags[c, L]
        end

        # Zero frequency produces real kernel (imaginary part ≈ 0)
        K_real = phasor_kernel(fill(-0.1f0, 2), fill(0f0, 2), 1f0, 8)
        @test all(abs.(imag.(K_real)) .< 1f-5)
    end
end

function causal_conv_tests()
    @testset "causal_conv" begin
        C, L, B = 4, 10, 3
        K = randn(ComplexF32, C, L)
        H = randn(ComplexF32, C, L, B)

        Z = causal_conv(K, H)

        # Shape preservation
        @test size(Z) == (C, L, B)
        @test eltype(Z) <: Complex

        # All values finite
        @test all(isfinite, Z)

        # Causality: zeroing future inputs doesn't change past outputs
        H2 = copy(H)
        H2[:, 6:end, :] .= 0f0
        Z2 = causal_conv(K, H2)
        # First 5 time steps should be identical
        @test Z[:, 1:5, :] ≈ Z2[:, 1:5, :]
    end
end

function hippo_tests()
    @testset "hippo_legs_diagonal" begin
        N = 32
        λ, ω = hippo_legs_diagonal(N)

        # Returns vectors of length N
        @test length(λ) == N
        @test length(ω) == N

        # λ all negative
        @test all(λ .< 0f0)

        # ω all positive
        @test all(ω .> 0f0)

        # clip_decay works
        λc, ωc = hippo_legs_diagonal(N; clip_decay=2.0)
        @test all(abs.(λc) .<= 2.0f0 + 1f-6)

        # Types
        @test eltype(λ) == Float32
        @test eltype(ω) == Float32
    end
end

function phasor_ssm_layer_tests()
    @testset "PhasorSSM layer" begin
        rng = Xoshiro(42)
        in_dim, out_dim = 8, 16
        L, B = 10, 4

        layer = PhasorSSM(in_dim => out_dim, normalize_to_unit_circle)
        ps, st = Lux.setup(rng, layer)

        # Parameter shapes
        @test size(ps.weight) == (out_dim, in_dim)
        @test size(ps.log_neg_lambda) == (out_dim,)
        @test size(ps.omega) == (out_dim,)

        # parameterlength
        @test Lux.parameterlength(layer) == out_dim * in_dim + 2 * out_dim

        # Forward pass
        x = randn(ComplexF32, in_dim, L, B)
        y, st_new = layer(x, ps, st)

        @test size(y) == (out_dim, L, B)
        @test all(isfinite, y)

        # Unit circle activation: magnitudes ≈ 1
        @test all(abs.(abs.(y) .- 1f0) .< 1f-5)

        # Different inputs give different outputs
        x2 = randn(ComplexF32, in_dim, L, B)
        y2, _ = layer(x2, ps, st)
        @test y != y2

        # HiPPO init
        layer_hippo = PhasorSSM(in_dim => out_dim, identity; init=:hippo)
        ps_h, _ = Lux.setup(rng, layer_hippo)
        @test size(ps_h.weight) == (out_dim, in_dim)
        y_h, _ = layer_hippo(x, ps_h, st)
        @test size(y_h) == (out_dim, L, B)
        @test all(isfinite, y_h)
    end
end

function ssm_readout_tests()
    @testset "SSMReadout" begin
        rng = Xoshiro(42)
        C, L, B = 16, 20, 4

        layer = SSMReadout(0.25f0)
        ps, st = Lux.setup(rng, layer)

        # No trainable parameters
        @test ps == NamedTuple()

        z = randn(ComplexF32, C, L, B)
        out, st_new = layer(z, ps, st)

        # Shape reduction: (C, L, B) → (C, B)
        @test size(out) == (C, B)

        # Output is Phase type
        @test eltype(out) == Phase

        # Bounds [-1, 1]
        @test all(Float32.(out) .>= -1f0) && all(Float32.(out) .<= 1f0)
    end
end

function psk_encode_tests()
    @testset "psk_encode" begin
        H, W, B = 28, 28, 4
        images = rand(Float32, H, W, B)

        x = psk_encode(images)

        # Shape: (H, W, B) → (W, H, B)
        @test size(x) == (W, H, B)
        @test eltype(x) <: Complex

        # Unit magnitude
        mags = abs.(x)
        @test all(abs.(mags .- 1f0) .< 1f-5)

        # n_repeats doubles time dim
        x2 = psk_encode(images; n_repeats=2)
        @test size(x2) == (W, 2 * H, B)
    end
end

function impulse_encode_tests()
    @testset "impulse_encode" begin
        H, W, B = 8, 6, 2
        images = rand(Float32, H, W, B)
        substeps = 4

        x = impulse_encode(images; substeps)

        # Shape: (H, W, B) → (W, H*substeps, B)
        @test size(x) == (W, H * substeps, B)
        @test eltype(x) <: Complex

        # Non-negative real part (von-Mises pulses are non-negative)
        @test all(real.(x) .>= -1f-6)

        # Zero imaginary part
        @test all(abs.(imag.(x)) .< 1f-6)
    end
end

function ssm_chain_tests()
    @testset "SSM Chain (end-to-end)" begin
        rng = Xoshiro(42)
        C_in, D_hidden, n_classes = 8, 16, 5
        L, B = 10, 4

        model = Chain(
            PhasorSSM(C_in => D_hidden, normalize_to_unit_circle),
            PhasorSSM(D_hidden => D_hidden, identity),
            SSMReadout(0.25f0),
            Codebook(D_hidden => n_classes),
        )
        ps, st = Lux.setup(rng, model)

        x = randn(ComplexF32, C_in, L, B)
        y, st_new = model(x, ps, st)

        # Correct output shape: (n_classes, B)
        @test size(y) == (n_classes, B)
        @test all(isfinite, y)
    end
end

function ssm_gradient_tests()
    @testset "SSM Gradients" begin
        rng = Xoshiro(42)
        in_dim, out_dim = 4, 8
        L, B = 6, 2

        layer = PhasorSSM(in_dim => out_dim, normalize_to_unit_circle)
        ps, st = Lux.setup(rng, layer)

        x = randn(ComplexF32, in_dim, L, B)

        loss_fn = function(ps)
            y, _ = layer(x, ps, st)
            return sum(abs2, y)
        end

        val, grads = withgradient(loss_fn, ps)

        @test isfinite(val)
        @test grads[1] !== nothing

        # All gradient components finite
        @test all(isfinite, grads[1].weight)
        @test all(isfinite, grads[1].log_neg_lambda)
        @test all(isfinite, grads[1].omega)
    end
end

function ssm_gpu_tests()
    @testset "SSM GPU Tests" begin
        @info "Running SSM GPU tests..."
        rng = Xoshiro(42)
        device = gpu_device()

        @testset "PhasorSSM forward on GPU" begin
            in_dim, out_dim = 8, 16
            L, B = 10, 4

            layer = PhasorSSM(in_dim => out_dim, normalize_to_unit_circle)
            ps, st = Lux.setup(rng, layer)
            ps = ps |> device
            st = st |> device

            x = randn(ComplexF32, in_dim, L, B) |> device
            y, _ = layer(x, ps, st)

            @test size(y) == (out_dim, L, B)
            @test all(isfinite, Array(y))
        end

        @testset "psk_encode on GPU" begin
            images = CUDA.rand(Float32, 28, 28, 4)
            x = psk_encode(images)
            @test size(x) == (28, 28, 4)
            @test all(abs.(abs.(Array(x)) .- 1f0) .< 1f-5)
        end

        @testset "SSM Chain on GPU" begin
            C_in, D_hidden, n_classes = 8, 16, 5
            L, B = 10, 4

            model = Chain(
                PhasorSSM(C_in => D_hidden, normalize_to_unit_circle),
                PhasorSSM(D_hidden => D_hidden, identity),
                SSMReadout(0.25f0),
                Codebook(D_hidden => n_classes),
            )
            ps, st = Lux.setup(rng, model)
            ps = ps |> device
            st = st |> device

            x = randn(ComplexF32, C_in, L, B) |> device
            y, _ = model(x, ps, st)

            @test size(y) == (n_classes, B)
            @test all(isfinite, Array(y))
        end
    end
end
