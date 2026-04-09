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
        causal_conv_fft_tests()
        dirac_discretization_tests()
        hippo_tests()
        phasor_ssm_layer_tests()
        ssm_readout_tests()
        psk_encode_tests()
        impulse_encode_tests()
        ssm_chain_tests()
        ssm_gradient_tests()
        ssm_cross_attention_tests()
        ssm_self_attention_tests()
        ssm_attention_chain_tests()
        ssm_phases_to_train_tests()
        ssm_spiking_dispatch_tests()
        ssm_spiking_correlation_tests()
        ssm_spiking_chain_tests()
        phasor_stft_tests()
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

function causal_conv_fft_tests()
    @testset "causal_conv_fft" begin
        @testset "matches Toeplitz for short sequences" begin
            C, L, B = 4, 10, 3
            K = randn(ComplexF32, C, L)
            H = randn(ComplexF32, C, L, B)

            Z_toeplitz = PhasorNetworks._causal_conv_toeplitz(K, H)
            Z_fft = causal_conv_fft(K, H)

            @test size(Z_fft) == (C, L, B)
            @test eltype(Z_fft) <: Complex
            @test Z_fft ≈ Z_toeplitz atol=1f-4
        end

        @testset "matches Toeplitz for medium sequences" begin
            C, L, B = 8, 128, 4
            λ = fill(-0.1f0, C)
            ω = Float32.(collect(range(0.2f0, 2.5f0; length=C)))
            K = phasor_kernel(λ, ω, 1f0, L)
            H = randn(ComplexF32, C, L, B)

            Z_toeplitz = PhasorNetworks._causal_conv_toeplitz(K, H)
            Z_fft = causal_conv_fft(K, H)

            @test Z_fft ≈ Z_toeplitz atol=1f-4
        end

        @testset "causality" begin
            C, L, B = 4, 64, 2
            K = randn(ComplexF32, C, L)
            H = randn(ComplexF32, C, L, B)

            Z = causal_conv_fft(K, H)
            H2 = copy(H)
            H2[:, 6:end, :] .= 0f0
            Z2 = causal_conv_fft(K, H2)

            # First 5 time steps should be identical
            @test Z[:, 1:5, :] ≈ Z2[:, 1:5, :] atol=1f-5
        end

        @testset "long sequence (L=512)" begin
            C, L, B = 8, 512, 2
            λ = fill(-0.01f0, C)
            ω = Float32.(collect(range(0.1f0, 1.0f0; length=C)))
            K = phasor_kernel(λ, ω, 1f0, L)
            H = randn(ComplexF32, C, L, B)

            Z = causal_conv_fft(K, H)
            @test size(Z) == (C, L, B)
            @test all(isfinite, Z)
        end

        @testset "gradient flow" begin
            C, L, B = 4, 32, 2
            K = randn(ComplexF32, C, L)
            H = randn(ComplexF32, C, L, B)

            loss_fn = K -> sum(abs2, causal_conv_fft(K, H))
            val, grads = withgradient(loss_fn, K)
            @test isfinite(val)
            @test grads[1] !== nothing
            @test all(isfinite, grads[1])
        end

        @testset "auto-dispatch uses FFT for long sequences" begin
            C, L, B = 4, 128, 2
            K = randn(ComplexF32, C, L)
            H = randn(ComplexF32, C, L, B)

            # causal_conv should auto-dispatch to FFT for L > 64
            Z_auto = causal_conv(K, H)
            Z_fft = causal_conv_fft(K, H)
            @test Z_auto ≈ Z_fft atol=1f-6
        end
    end
end

function dirac_discretization_tests()
    @testset "Dirac Discretization" begin
        rng = Xoshiro(42)

        @testset "dirac_encode shape and finiteness" begin
            C_in, C_out, L, B = 4, 8, 16, 3
            phases = 2f0 .* rand(rng, Float32, C_in, L, B) .- 1f0
            λ = fill(-0.1f0, C_out)
            ω = Float32.(collect(range(0.2f0, 2.5f0; length=C_out)))

            enc = dirac_encode(phases, λ, ω, 1f0)
            @test size(enc) == (C_out, C_in, L, B)
            @test eltype(enc) <: Complex
            @test all(isfinite, enc)
        end

        @testset "causal_conv_dirac shape and finiteness" begin
            C_in, C_out, L, B = 4, 8, 16, 3
            phases = 2f0 .* rand(rng, Float32, C_in, L, B) .- 1f0
            W = randn(rng, Float32, C_out, C_in)
            λ = fill(-0.1f0, C_out)
            ω = Float32.(collect(range(0.2f0, 2.5f0; length=C_out)))

            Z = causal_conv_dirac(phases, W, λ, ω, 1f0)
            @test size(Z) == (C_out, L, B)
            @test eltype(Z) <: Complex
            @test all(isfinite, Z)
        end

        @testset "gradient flow through Dirac path" begin
            C_in, C_out, L, B = 4, 8, 10, 2
            phases = 2f0 .* rand(rng, Float32, C_in, L, B) .- 1f0
            W = randn(rng, Float32, C_out, C_in)
            λ = fill(-0.1f0, C_out)
            ω = Float32.(collect(range(0.2f0, 2.5f0; length=C_out)))

            loss_fn = W -> sum(abs2, causal_conv_dirac(phases, W, λ, ω, 1f0))
            val, grads = withgradient(loss_fn, W)
            @test isfinite(val)
            @test grads[1] !== nothing
            @test all(isfinite, grads[1])
        end

        @testset "PhasorDense Phase 3D uses Dirac for SSM init" begin
            C_in, C_out, L, B = 4, 8, 10, 3
            phases = Phase.(2f0 .* rand(rng, Float32, C_in, L, B) .- 1f0)

            # SSM-mode layer (init_mode != :default) should use Dirac
            layer_ssm = PhasorDense(C_in => C_out, normalize_to_unit_circle;
                                    init_mode=:uniform, use_bias=false)
            ps, st = Lux.setup(rng, layer_ssm)
            y_dirac, _ = layer_ssm(phases, ps, st)
            @test size(y_dirac) == (C_out, L, B)
            @test eltype(y_dirac) <: Phase
            @test all(isfinite, Float32.(y_dirac))

            # Default layer should use ZOH (backward compat)
            layer_default = PhasorDense(C_in => C_out, normalize_to_unit_circle;
                                        init_mode=:default, use_bias=false)
            ps_d, st_d = Lux.setup(rng, layer_default)
            y_zoh, _ = layer_default(phases, ps_d, st_d)
            @test size(y_zoh) == (C_out, L, B)
            @test eltype(y_zoh) <: Phase

            # Complex 3D input always uses ZOH regardless of init_mode
            x_cmpx = angle_to_complex(phases)
            y_cmpx, _ = layer_ssm(x_cmpx, ps, st)
            @test size(y_cmpx) == (C_out, L, B)
            @test eltype(y_cmpx) <: Complex
        end

        @testset "Dirac single-oscillator consistency" begin
            @info "Running Dirac consistency check..."
            C_in, C_out = 4, 8
            L, B = 8, 3

            layer = PhasorDense(C_in => C_out, normalize_to_unit_circle;
                                init_mode=:uniform, use_bias=false)
            ps, st = Lux.setup(rng, layer)

            phases = Phase.(2f0 .* rand(rng, Float32, C_in, L, B) .- 1f0)

            # Dirac discrete path (single-oscillator)
            y_dirac, _ = layer(phases, ps, st)

            # Manual computation: same formula, verify we get identical results
            λ = -exp.(ps.log_neg_lambda)
            ω = st.omega
            Z_manual = causal_conv_dirac(Float32.(phases), ps.weight, λ, ω, 1f0)
            y_manual = complex_to_angle(normalize_to_unit_circle(Z_manual))

            @test Float32.(y_dirac) ≈ Float32.(y_manual) atol=1f-5

            # Verify output is non-degenerate (spread of phase values)
            ph_range = maximum(Float32.(y_dirac)) - minimum(Float32.(y_dirac))
            @test ph_range > 0.5
        end
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

        # Parameter shapes (PhasorSSM now returns PhasorDense)
        @test size(ps.weight) == (out_dim, in_dim)
        @test size(ps.log_neg_lambda) == (out_dim,)
        @test size(st.omega) == (out_dim,)

        # parameterlength
        @test Lux.parameterlength(layer) == out_dim * in_dim + out_dim

        # Forward pass (3D complex dispatch)
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
        ps_h, st_h = Lux.setup(rng, layer_hippo)
        @test size(ps_h.weight) == (out_dim, in_dim)
        y_h, _ = layer_hippo(x, ps_h, st_h)
        @test size(y_h) == (out_dim, L, B)
        @test all(isfinite, y_h)
    end
end

function ssm_readout_tests()
    @testset "SSMReadout" begin
        rng = Xoshiro(42)
        C, L, B = 16, 20, 4
        n_classes = 5

        layer = SSMReadout(C => n_classes)
        ps, st = Lux.setup(rng, layer)

        # No trainable parameters
        @test ps == NamedTuple()

        # Codebook in state
        @test size(st.codes) == (C, n_classes)

        z = randn(ComplexF32, C, L, B)
        out, st_new = layer(z, ps, st)

        # Shape: (C, L, B) → (n_classes, B) similarity logits
        @test size(out) == (n_classes, B)

        # Output is Float32 similarities in [-1, 1]
        @test eltype(out) == Float32
        @test all(out .>= -1f0 .- 1f-6) && all(out .<= 1f0 .+ 1f-6)

        # All values finite
        @test all(isfinite, out)
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
            SSMReadout(D_hidden => n_classes),
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
                SSMReadout(D_hidden => n_classes),
            )
            ps, st = Lux.setup(rng, model)
            ps = ps |> device
            st = st |> device

            x = randn(ComplexF32, C_in, L, B) |> device
            y, _ = model(x, ps, st)

            @test size(y) == (n_classes, B)
            @test all(isfinite, Array(y))
        end

        @testset "SSMCrossAttention on GPU" begin
            C_in, d_model, n_keys = 8, 16, 6
            L, B = 10, 4

            layer = SSMCrossAttention(C_in => d_model, n_keys)
            ps, st = Lux.setup(rng, layer)
            ps = ps |> device
            st = st |> device

            x = randn(ComplexF32, C_in, L, B) |> device
            y, _ = layer(x, ps, st)

            @test size(y) == (d_model, n_keys, B)
            @test all(isfinite, Array(y))
        end

        @testset "SSMSelfAttention on GPU" begin
            C_in, d_model = 8, 16
            L, B = 10, 4

            layer = SSMSelfAttention(C_in => d_model)
            ps, st = Lux.setup(rng, layer)
            ps = ps |> device
            st = st |> device

            x = randn(ComplexF32, C_in, L, B) |> device
            y, _ = layer(x, ps, st)

            @test size(y) == (d_model, L, B)
            @test all(isfinite, Array(y))
        end
    end
end

function ssm_cross_attention_tests()
    @testset "SSMCrossAttention" begin
        rng = Xoshiro(42)
        C_in, d_model, n_keys = 8, 16, 6
        L, B = 10, 4

        layer = SSMCrossAttention(C_in => d_model, n_keys)
        ps, st = Lux.setup(rng, layer)

        @testset "parameter shapes" begin
            @test size(ps.weight_q) == (d_model, C_in)
            @test size(ps.weight_v) == (d_model, C_in)
            @test size(ps.keys) == (d_model, n_keys)
            @test length(ps.scale) == 1
        end

        @testset "output shape and finiteness" begin
            x = randn(rng, ComplexF32, C_in, L, B)
            y, st_new = layer(x, ps, st)

            @test size(y) == (d_model, n_keys, B)
            @test eltype(y) <: Complex
            @test all(isfinite, y)
        end

        @testset "unit circle normalization" begin
            x = randn(rng, ComplexF32, C_in, L, B)
            y, _ = layer(x, ps, st)
            mags = abs.(y)
            @test all(abs.(mags .- 1f0) .< 0.1f0)
        end

        @testset "different inputs produce different outputs" begin
            x1 = randn(rng, ComplexF32, C_in, L, B)
            x2 = randn(rng, ComplexF32, C_in, L, B)
            y1, _ = layer(x1, ps, st)
            y2, _ = layer(x2, ps, st)
            @test y1 != y2
        end

        @testset "gradient computation" begin
            x = randn(rng, ComplexF32, C_in, L, B)
            loss_fn = ps -> begin
                y, _ = layer(x, ps, st)
                sum(abs2.(y))
            end
            val, grads = withgradient(loss_fn, ps)
            g = grads[1]
            @test isfinite(val)
            @test all(isfinite, g.weight_q)
            @test all(isfinite, g.weight_v)
            @test all(isfinite, g.keys)
            @test all(isfinite, g.scale)
        end

        @testset "parameterlength" begin
            expected = d_model * C_in * 2 + d_model * n_keys + 1
            @test Lux.parameterlength(layer) == expected
        end
    end
end

function ssm_self_attention_tests()
    @testset "SSMSelfAttention" begin
        rng = Xoshiro(42)
        C_in, d_model = 8, 16
        L, B = 10, 4

        layer = SSMSelfAttention(C_in => d_model)
        ps, st = Lux.setup(rng, layer)

        @testset "parameter shapes" begin
            @test size(ps.weight_q) == (d_model, C_in)
            @test size(ps.weight_k) == (d_model, C_in)
            @test size(ps.weight_v) == (d_model, C_in)
            @test length(ps.scale) == 1
        end

        @testset "output shape preserves temporal dim" begin
            x = randn(rng, ComplexF32, C_in, L, B)
            y, _ = layer(x, ps, st)

            @test size(y) == (d_model, L, B)
            @test eltype(y) <: Complex
            @test all(isfinite, y)
        end

        @testset "unit circle normalization" begin
            x = randn(rng, ComplexF32, C_in, L, B)
            y, _ = layer(x, ps, st)
            mags = abs.(y)
            @test all(abs.(mags .- 1f0) .< 0.1f0)
        end

        @testset "different inputs produce different outputs" begin
            x1 = randn(rng, ComplexF32, C_in, L, B)
            x2 = randn(rng, ComplexF32, C_in, L, B)
            y1, _ = layer(x1, ps, st)
            y2, _ = layer(x2, ps, st)
            @test y1 != y2
        end

        @testset "gradient computation" begin
            x = randn(rng, ComplexF32, C_in, L, B)
            loss_fn = ps -> begin
                y, _ = layer(x, ps, st)
                sum(abs2.(y))
            end
            val, grads = withgradient(loss_fn, ps)
            g = grads[1]
            @test isfinite(val)
            @test all(isfinite, g.weight_q)
            @test all(isfinite, g.weight_k)
            @test all(isfinite, g.weight_v)
            @test all(isfinite, g.scale)
        end

        @testset "parameterlength" begin
            expected = d_model * C_in * 3 + 1
            @test Lux.parameterlength(layer) == expected
        end
    end
end

function ssm_attention_chain_tests()
    @testset "SSM Attention Chain" begin
        rng = Xoshiro(42)
        C_in, D_hidden, n_classes = 8, 16, 5
        L, B = 10, 4

        @testset "self-attention in chain" begin
            model = Chain(
                PhasorSSM(C_in => D_hidden, normalize_to_unit_circle),
                SSMSelfAttention(D_hidden => D_hidden, normalize_to_unit_circle),
                PhasorSSM(D_hidden => D_hidden, identity),
                SSMReadout(D_hidden => n_classes),
            )
            ps, st = Lux.setup(rng, model)
            x = randn(rng, ComplexF32, C_in, L, B)

            y, _ = model(x, ps, st)
            @test size(y) == (n_classes, B)
            @test all(isfinite, y)
        end

        @testset "cross-attention in chain" begin
            model = Chain(
                PhasorSSM(C_in => D_hidden, normalize_to_unit_circle),
                SSMCrossAttention(D_hidden => D_hidden, L, normalize_to_unit_circle),
                PhasorSSM(D_hidden => D_hidden, identity),
                SSMReadout(D_hidden => n_classes),
            )
            ps, st = Lux.setup(rng, model)
            x = randn(rng, ComplexF32, C_in, L, B)

            y, _ = model(x, ps, st)
            @test size(y) == (n_classes, B)
            @test all(isfinite, y)
        end

        @testset "gradient through chain with self-attention" begin
            model = Chain(
                PhasorSSM(C_in => D_hidden, normalize_to_unit_circle),
                SSMSelfAttention(D_hidden => D_hidden, normalize_to_unit_circle),
                SSMReadout(D_hidden => n_classes),
            )
            ps, st = Lux.setup(rng, model)
            x = randn(rng, ComplexF32, C_in, L, B)
            y_target = Float32.(rand(rng, n_classes, B))

            loss_fn = ps -> begin
                y, _ = model(x, ps, st)
                sum(abs2.(y .- y_target))
            end
            val, grads = withgradient(loss_fn, ps)
            @test isfinite(val)
            @test all(isfinite, grads[1].layer_1.weight)
            @test all(isfinite, grads[1].layer_2.weight_q)
        end
    end
end

# ---- SSM Spiking Tests ----

function ssm_phases_to_train_tests()
    @testset "ssm_phases_to_train" begin
        C, L, B = 4, 6, 2
        phases = Phase.(rand(Float32, C, L, B) .* 2f0 .- 1f0)

        train = ssm_phases_to_train(phases, spk_args=spk_args)

        @testset "shape and spike count" begin
            @test train.shape == (C, B)
            @test length(train.times) == C * L * B
            @test length(train.indices) == C * L * B
        end

        @testset "time range" begin
            t_period = spk_args.t_period
            @test all(train.times .>= 0f0)
            @test all(train.times .<= Float32(L) * t_period)
        end

        @testset "times are ordered within periods" begin
            t_period = spk_args.t_period
            # Each period l maps to [(l-1)*t_period, l*t_period)
            for l in 1:L
                t_lo = Float32(l - 1) * t_period
                t_hi = Float32(l) * t_period
                mask = (train.times .>= t_lo) .& (train.times .< t_hi .+ 1f-6)
                @test sum(mask) == C * B
            end
        end
    end
end

function ssm_spiking_dispatch_tests()
    @testset "SSM Spiking Dispatch" begin
        rng = Xoshiro(42)
        C_in, C_out = 4, 8
        L, B = 6, 2

        # Create input as complex, encode as spikes
        x_cmpx = randn(rng, ComplexF32, C_in, L, B)
        phases_in = complex_to_angle(normalize_to_unit_circle(x_cmpx))
        train = ssm_phases_to_train(phases_in, spk_args=spk_args)
        tspan_spk = (0.0f0, Float32(L) * spk_args.t_period)
        sc = SpikingCall(train, spk_args, tspan_spk)

        @testset "PhasorSSM SpikingCall dispatch" begin
            layer = PhasorSSM(C_in => C_out, normalize_to_unit_circle)
            ps, st = Lux.setup(rng, layer)

            y, st_new = layer(sc, ps, st)

            @test size(y) == (C_out, L, B)
            @test eltype(y) <: Phase
            @test all(isfinite, Float32.(y))
        end

        @testset "PhasorSSM CurrentCall dispatch" begin
            layer = PhasorSSM(C_in => C_out, normalize_to_unit_circle)
            ps, st = Lux.setup(rng, layer)

            cc = CurrentCall(sc)
            y, st_new = layer(cc, ps, st)

            @test size(y) == (C_out, L, B)
            @test eltype(y) <: Phase
            @test all(isfinite, Float32.(y))
        end

        @testset "PhasorSSM potential return type" begin
            layer = PhasorSSM(C_in => C_out, normalize_to_unit_circle,
                              return_type=SolutionType(:potential))
            ps, st = Lux.setup(rng, layer)

            sol, st_new = layer(sc, ps, st)
            # Returns ODE solution of single-stage system (C_out, B)
            @test !(sol isa AbstractArray)
            sampled = sol(tspan_spk[2])
            @test sampled isa AbstractArray
            @test size(sampled) == (C_out, B)
        end

        @testset "PhasorSSM spiking return type" begin
            layer = PhasorSSM(C_in => C_out, normalize_to_unit_circle,
                              return_type=SolutionType(:spiking))
            ps, st = Lux.setup(rng, layer)

            result, st_new = layer(sc, ps, st)
            @test result isa SpikingCall
            @test result.train.shape == (C_out, B)
        end

        @testset "MakeSpikingSSM layer" begin
            make_layer = MakeSpikingSSM(spk_args)
            ps_m, st_m = Lux.setup(rng, make_layer)

            result, _ = make_layer(x_cmpx, ps_m, st_m)
            @test result isa SpikingCall
            @test result.train.shape == (C_in, B)
            @test length(result.train.times) == C_in * L * B
        end

        @testset "SSMSelfAttention SpikingCall dispatch" begin
            layer = SSMSelfAttention(C_in => C_out)
            ps, st = Lux.setup(rng, layer)

            y, _ = layer(sc, ps, st)
            @test size(y) == (C_out, L, B)
            @test all(isfinite, y)
        end

        @testset "SSMCrossAttention SpikingCall dispatch" begin
            n_keys = 4
            layer = SSMCrossAttention(C_in => C_out, n_keys)
            ps, st = Lux.setup(rng, layer)

            y, _ = layer(sc, ps, st)
            @test size(y) == (C_out, n_keys, B)
            @test all(isfinite, y)
        end

        @testset "SSMReadout SpikingCall dispatch" begin
            n_classes = 5
            layer = SSMReadout(C_in => n_classes)
            ps, st = Lux.setup(rng, layer)

            y, _ = layer(sc, ps, st)
            @test size(y) == (n_classes, B)
            @test eltype(y) == Float32
            @test all(isfinite, y)
        end
    end
end

function ssm_spiking_correlation_tests()
    @testset "SSM Dirac vs Spiking ODE Correlation" begin
        rng = Xoshiro(42)
        C_in, C_out = 4, 8
        L, B = 8, 3

        # Phase-returning layer for both Dirac and spiking comparison
        layer = PhasorSSM(C_in => C_out, normalize_to_unit_circle)
        ps, st = Lux.setup(rng, layer)

        # Random phase input
        phases_in = Phase.(2f0 .* rand(rng, Float32, C_in, L, B) .- 1f0)

        # Dirac convolution (exact analytical)
        phases_dirac, _ = layer(phases_in, ps, st)

        # Spiking ODE (single-stage: dz/dt = k·z + W·I(t))
        train = ssm_phases_to_train(phases_in, spk_args=spk_args)
        tspan_spk = (0.0f0, Float32(L) * spk_args.t_period)
        sc = SpikingCall(train, spk_args, tspan_spk)
        phases_spiking, _ = layer(sc, ps, st)

        @testset "output shapes match" begin
            @test size(phases_dirac) == size(phases_spiking)
        end

        @testset "Dirac matches manual causal_conv_dirac" begin
            λ_c = -exp.(ps.log_neg_lambda)
            ω_c = st.omega
            Z_manual = causal_conv_dirac(Float32.(phases_in), ps.weight, λ_c, ω_c, 1f0)
            phases_manual = complex_to_angle(normalize_to_unit_circle(Z_manual))
            @test Float32.(phases_dirac) ≈ Float32.(phases_manual) atol=1f-5
        end

        @testset "spiking output is non-degenerate" begin
            ph_range = maximum(Float32.(phases_spiking)) - minimum(Float32.(phases_spiking))
            @test ph_range > 0.5
        end

        @testset "spiking ODE correlates with Dirac" begin
            c = cor_realvals(vec(Float32.(phases_dirac)),
                             vec(Float32.(phases_spiking)))
            # The single-stage ODE should correlate positively with the
            # exact Dirac result. The gap comes from the finite-width spike
            # kernel and ODE solver temporal discretization.
            @test c > 0.3
        end
    end
end

function ssm_spiking_chain_tests()
    @testset "SSM Spiking Chain (end-to-end)" begin
        rng = Xoshiro(42)
        C_in, D_hidden, n_classes = 4, 8, 5
        L, B = 6, 2

        @testset "MakeSpikingSSM → PhasorSSM → SSMReadout" begin
            model = Chain(
                MakeSpikingSSM(spk_args),
                PhasorSSM(C_in => D_hidden, normalize_to_unit_circle,
                          return_type=SolutionType(:spiking)),
                SSMReadout(D_hidden => n_classes),
            )
            ps, st = Lux.setup(rng, model)

            x = normalize_to_unit_circle(randn(rng, ComplexF32, C_in, L, B))
            y, _ = model(x, ps, st)

            @test size(y) == (n_classes, B)
            @test all(isfinite, y)
        end

        @testset "MakeSpikingSSM → PhasorSSM → SSMSelfAttention → SSMReadout" begin
            model = Chain(
                MakeSpikingSSM(spk_args),
                PhasorSSM(C_in => D_hidden, normalize_to_unit_circle,
                          return_type=SolutionType(:spiking)),
                SSMSelfAttention(D_hidden => D_hidden, normalize_to_unit_circle),
                SSMReadout(D_hidden => n_classes),
            )
            ps, st = Lux.setup(rng, model)

            x = normalize_to_unit_circle(randn(rng, ComplexF32, C_in, L, B))
            y, _ = model(x, ps, st)

            @test size(y) == (n_classes, B)
            @test all(isfinite, y)
        end
    end
end

function phasor_stft_tests()
    @testset "PhasorSTFT" begin
        rng = Xoshiro(42)
        in_dim, n_freqs = 8, 16
        L, B = 10, 4

        layer = PhasorSTFT(in_dim => n_freqs, normalize_to_unit_circle)
        ps, st = Lux.setup(rng, layer)

        # Parameter structure: weight, log_neg_lambda, omega all trainable
        @test size(ps.weight) == (n_freqs, in_dim)
        @test size(ps.log_neg_lambda) == (n_freqs,)
        @test size(ps.omega) == (n_freqs,)

        # State: omega_out is fixed
        @test haskey(st, :omega_out)
        @test st.omega_out ≈ Float32(2π)

        # Omega initialized as uniform spread
        @test ps.omega[1] ≈ 0.2f0
        @test ps.omega[end] ≈ 2.5f0

        # parameterlength
        @test Lux.parameterlength(layer) == n_freqs * in_dim + n_freqs + n_freqs

        # 3D complex forward pass
        x = randn(rng, ComplexF32, in_dim, L, B)
        y, st_new = layer(x, ps, st)

        @test size(y) == (n_freqs, L, B)
        @test all(isfinite, y)
        # Unit circle activation: magnitudes ≈ 1
        @test all(abs.(abs.(y) .- 1f0) .< 1f-5)

        # Different inputs give different outputs
        x2 = randn(rng, ComplexF32, in_dim, L, B)
        y2, _ = layer(x2, ps, st)
        @test y != y2

        # 3D Phase forward pass
        x_phase = Phase.(2f0 .* rand(rng, Float32, in_dim, L, B) .- 1f0)
        y_phase, _ = layer(x_phase, ps, st)

        @test size(y_phase) == (n_freqs, L, B)
        @test eltype(y_phase) == Phase
        @test all(isfinite, Float32.(y_phase))

        # Gradient finiteness
        loss_fn(ps) = begin
            y, _ = layer(x, ps, st)
            mean(abs2.(y))
        end
        val, grads = withgradient(loss_fn, ps)
        @test isfinite(val)
        @test all(isfinite, grads[1].weight)
        @test all(isfinite, grads[1].log_neg_lambda)
        @test all(isfinite, grads[1].omega)

        # Frequency shift identity: when omega_out == omega, shift is no-op
        st_match = (omega_out = ps.omega[1],)
        layer_id = PhasorSTFT(in_dim => 1, identity; omega_lo=ps.omega[1], omega_hi=ps.omega[1])
        ps_id, _ = Lux.setup(rng, layer_id)
        # Override omega to match omega_out exactly
        ps_id = merge(ps_id, (omega = Float32[st_match.omega_out],))
        x_small = randn(rng, ComplexF32, in_dim, L, B)
        y_shift, _ = layer_id(x_small, ps_id, st_match)
        # Build expected output without shift (standard causal conv)
        λ_id = -exp.(ps_id.log_neg_lambda)
        K_id = phasor_kernel(λ_id, ps_id.omega, 1f0, L)
        xr = reshape(x_small, in_dim, L * B)
        Hr = complex.(ps_id.weight * real.(xr), ps_id.weight * imag.(xr))
        H_id = reshape(Hr, 1, L, B)
        Z_expected = causal_conv(K_id, H_id)
        @test y_shift ≈ Z_expected atol=1f-5

        # With bias
        layer_bias = PhasorSTFT(in_dim => n_freqs, normalize_to_unit_circle; use_bias=true)
        ps_b, st_b = Lux.setup(rng, layer_bias)
        @test haskey(ps_b, :bias_real)
        @test haskey(ps_b, :bias_imag)
        y_b, _ = layer_bias(x, ps_b, st_b)
        @test size(y_b) == (n_freqs, L, B)
        @test all(isfinite, y_b)
    end
end
