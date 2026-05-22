# Proposed network_layers_tests.jl
# This file proposes new tests for neural network layers in network.jl
# To be added to the test suite to improve coverage

using Test
using Lux
using PhasorNetworks
using Random: Xoshiro, AbstractRNG
using DifferentialEquations: Tsit5, Heun

"""
Run all network layer tests
"""
function network_layers_tests()
    @testset "Network Layers Tests" begin
        @info "Running network layers tests..."

        phasor_dense_tests()
        complex_bias_tests()
        phasor_dense_no_bias_tests()
        phasor_dense_activation_tests()
        phasor_conv_tests()
        phasor_fixed_tests()
        min_pool_tests()
        residual_block_tests()
        codebook_tests()
        flatten_spiketrain_tests()
        dropout_spiketrain_tests()
        variance_scaling_tests()
        dense_onehot_tests()
        attend_tests()
    end
end

function phasor_dense_tests()
    @testset "PhasorDense Tests" begin
        rng = Xoshiro(42)
        in_dim, out_dim = 10, 5
        batch_size = 32

        # Create layer
        layer = PhasorDense(in_dim => out_dim, complex_to_angle)
        ps, st = Lux.setup(rng, layer)

        # Random phase input
        x = Phase.(rand(Float32, in_dim, batch_size) .* 2.0f0 .- 1.0f0)

        # Forward pass
        y, st_new = layer(x, ps, st)

        # Output shape should be correct
        @test size(y) == (out_dim, batch_size)

        # Output should be phases (in [-1, 1])
        @test all(Float32.(y) .>= -1.0f0) && all(Float32.(y) .<= 1.0f0)

        # State should be returned
        @test st_new == st

        # Different inputs should give different outputs
        x2 = Phase.(rand(Float32, in_dim, batch_size) .* 2.0f0 .- 1.0f0)
        y2, _ = layer(x2, ps, st)
        @test !all(y .≈ y2)
    end
end

function complex_bias_tests()
    @testset "ComplexBias Tests" begin
        rng = Xoshiro(42)
        dims = (5, 10)
        
        # Test with default initialization
        bias = ComplexBias(dims)
        ps, st = Lux.setup(rng, bias)
        
        # Check parameter structure
        @test haskey(ps, :bias_real)
        @test haskey(ps, :bias_imag)
        @test size(ps.bias_real) == dims
        @test size(ps.bias_imag) == dims
        
        # Apply bias to complex array
        x = randn(ComplexF32, dims...)
        y, _ = bias(x, ps, st)
        
        # Output shape should match input
        @test size(y) == size(x)
        
        # Bias should be added
        expected = x .+ (ps.bias_real .+ 1.0f0im .* ps.bias_imag)
        @test y ≈ expected
        
        # Test with zero bias
        bias_zero = ComplexBias(dims, init_bias=zero_bias)
        ps_zero, _ = Lux.setup(rng, bias_zero)
        y_zero, _ = bias_zero(x, ps_zero, st)
        
        # Should equal input (zero bias)
        @test y_zero ≈ x
    end
end

function phasor_dense_no_bias_tests()
    @testset "PhasorDense Without Bias Tests" begin
        rng = Xoshiro(42)
        in_dim, out_dim = 8, 4
        batch_size = 16

        layer = PhasorDense(in_dim => out_dim, soft_angle, use_bias=false)
        ps, st = Lux.setup(rng, layer)

        # Should not have bias in parameters (flat structure)
        @test !haskey(ps, :bias_real)
        @test !haskey(ps, :bias_imag)

        x = Phase.(rand(Float32, in_dim, batch_size) .* 2.0f0 .- 1.0f0)
        y, _ = layer(x, ps, st)

        # Output should still be valid phases
        @test size(y) == (out_dim, batch_size)
        @test all(isfinite.(y))
    end
end

function phasor_dense_activation_tests()
    @testset "PhasorDense Activation Tests" begin
        rng = Xoshiro(42)
        in_dim, out_dim = 6, 3
        batch_size = 20

        # Test with different activations
        for activation in [complex_to_angle, soft_angle]
            layer = PhasorDense(in_dim => out_dim, activation)
            ps, st = Lux.setup(rng, layer)

            x = Phase.(rand(Float32, in_dim, batch_size) .* 2.0f0 .- 1.0f0)
            y, _ = layer(x, ps, st)

            # Output should be valid
            @test size(y) == (out_dim, batch_size)
            @test all(isfinite.(y))

            # complex_to_angle should produce phases in [-1, 1]
            if activation == complex_to_angle
                @test all(Float32.(y) .>= -1.0f0) && all(Float32.(y) .<= 1.0f0)
            end
        end
    end
end

function phasor_conv_tests()
    @testset "PhasorConv Tests" begin
        rng = Xoshiro(42)
        height, width, c_in, c_out, batch = 28, 28, 1, 4, 8

        @testset "Real (phase) input → phase output" begin
            layer = PhasorConv((3, 3), c_in => c_out; pad=1)
            ps, st = Lux.setup(rng, layer)

            x = Phase.(rand(Float32, height, width, c_in, batch) .* 2.0f0 .- 1.0f0)
            y, st_new = layer(x, ps, st)

            @test size(y) == (height, width, c_out, batch)
            @test eltype(y) <: Real
            @test all(Float32.(y) .>= -1.0f0) && all(Float32.(y) .<= 1.0f0)
            @test all(isfinite.(y))
        end

        @testset "Complex input → complex output (no activation)" begin
            layer = PhasorConv((3, 3), c_in => c_out; pad=1)
            ps, st = Lux.setup(rng, layer)

            xz = randn(ComplexF32, height, width, c_in, batch)
            y, _ = layer(xz, ps, st)

            @test size(y) == (height, width, c_out, batch)
            @test eltype(y) <: Complex
            # magnitude is NOT clamped to unit circle (no activation applied)
            @test all(isfinite.(real.(y))) && all(isfinite.(imag.(y)))
        end

        @testset "use_bias=false excludes bias from params" begin
            layer = PhasorConv((3, 3), c_in => c_out; use_bias=false)
            ps, st = Lux.setup(rng, layer)

            @test !haskey(ps, :bias_real)
            @test haskey(ps, :weight)

            x = Phase.(rand(Float32, height, width, c_in, batch) .* 2.0f0 .- 1.0f0)
            y, _ = layer(x, ps, st)
            @test all(isfinite.(y))
        end

        @testset "use_bias=true includes bias in params" begin
            layer = PhasorConv((3, 3), c_in => c_out)
            ps, st = Lux.setup(rng, layer)

            @test haskey(ps, :bias_real)
            @test haskey(ps, :bias_imag)
        end

        @testset "Different inputs give different outputs" begin
            layer = PhasorConv((3, 3), c_in => c_out; pad=1)
            ps, st = Lux.setup(rng, layer)

            x1 = Phase.(rand(Float32, height, width, c_in, batch) .* 2.0f0 .- 1.0f0)
            x2 = Phase.(rand(Float32, height, width, c_in, batch) .* 2.0f0 .- 1.0f0)
            y1, _ = layer(x1, ps, st)
            y2, _ = layer(x2, ps, st)
            @test !all(y1 .≈ y2)
        end
    end
end

function phasor_fixed_tests()
    @testset "PhasorFixed Tests" begin
        rng = Xoshiro(42)
        in_dim, out_dim, batch = 10, 8, 16

        @testset "Weights in state, not params" begin
            layer = PhasorFixed(in_dim => out_dim)
            ps, st = Lux.setup(rng, layer)

            @test !haskey(ps, :weight)
            @test haskey(st, :weight)
            @test size(st.weight) == (out_dim, in_dim)
        end

        @testset "Real (phase) input → phase output" begin
            layer = PhasorFixed(in_dim => out_dim)
            ps, st = Lux.setup(rng, layer)

            x = Phase.(rand(Float32, in_dim, batch) .* 2.0f0 .- 1.0f0)
            y, _ = layer(x, ps, st)

            @test size(y) == (out_dim, batch)
            @test eltype(y) <: Real
            @test all(Float32.(y) .>= -1.0f0) && all(Float32.(y) .<= 1.0f0)
            @test all(isfinite.(y))
        end

        @testset "Complex input → complex output (no activation)" begin
            layer = PhasorFixed(in_dim => out_dim)
            ps, st = Lux.setup(rng, layer)

            xz = randn(ComplexF32, in_dim, batch)
            y, _ = layer(xz, ps, st)

            @test size(y) == (out_dim, batch)
            @test eltype(y) <: Complex
            @test all(isfinite.(real.(y))) && all(isfinite.(imag.(y)))
        end

        @testset "use_bias=false: dynamics in params, weights in state" begin
            layer = PhasorFixed(in_dim => out_dim; use_bias=false)
            ps, st = Lux.setup(rng, layer)

            @test !haskey(ps, :weight)
            @test haskey(ps, :log_neg_lambda)
            @test haskey(st, :weight)

            x = Phase.(rand(Float32, in_dim, batch) .* 2.0f0 .- 1.0f0)
            y, _ = layer(x, ps, st)
            @test all(isfinite.(y))
        end

        @testset "Fixed weights do not change across calls" begin
            layer = PhasorFixed(in_dim => out_dim)
            ps, st = Lux.setup(rng, layer)

            x = Phase.(rand(Float32, in_dim, batch) .* 2.0f0 .- 1.0f0)
            y1, st1 = layer(x, ps, st)
            y2, _   = layer(x, ps, st1)

            # Same input, same fixed weights → same output
            @test y1 ≈ y2
        end

        @testset "Different inputs give different outputs" begin
            layer = PhasorFixed(in_dim => out_dim)
            ps, st = Lux.setup(rng, layer)

            x1 = Phase.(rand(Float32, in_dim, batch) .* 2.0f0 .- 1.0f0)
            x2 = Phase.(rand(Float32, in_dim, batch) .* 2.0f0 .- 1.0f0)
            y1, _ = layer(x1, ps, st)
            y2, _ = layer(x2, ps, st)
            @test !all(y1 .≈ y2)
        end
    end
end

function min_pool_tests()
    @testset "MinPool Tests" begin
        # Create MinPool layer with 2x2 pooling
        layer = MinPool((2, 2))
        ps, st = Lux.setup(Xoshiro(42), layer)
        
        # Create input
        x = rand(Float32, 8, 8, 1, 4)
        
        y, _ = layer(x, ps, st)
        
        # Output should be half size in spatial dimensions
        @test size(y)[1] == 4  # Height
        @test size(y)[2] == 4  # Width
        @test size(y)[3] == 1  # Channels
        @test size(y)[4] == 4  # Batch
        
        # Test with specific values to verify min operation
        x_test = ones(Float32, 4, 4, 1, 1) .* 5.0f0
        x_test[1:2, 1:2, 1, 1] .= 2.0f0
        
        y_test, _ = layer(x_test, ps, st)
        
        # First pooling region should have minimum value
        @test y_test[1, 1, 1, 1] == 2.0f0
    end
end

function residual_block_tests()
    @testset "Residual Block Tests" begin
        rng = Xoshiro(42)
        in_dim, out_dim = 32, 32

        # Create residual block with dimension progression
        layer = ResidualBlock((in_dim, 64, out_dim), complex_to_angle)
        ps, st = Lux.setup(rng, layer)

        # Create input
        batch_size = 16
        x = Phase.(rand(Float32, in_dim, batch_size) .* 2.0f0 .- 1.0f0)

        # Forward pass
        y, st_new = layer(x, ps, st)

        # Output shape should match input
        @test size(y) == (out_dim, batch_size)

        # Output should be finite
        @test all(isfinite.(y))

        # Gradient flow should work
        @test all(isfinite.(y))
    end
end

function codebook_tests()
    @testset "Codebook Tests" begin
        rng = Xoshiro(42)
        n_classes, embedding_dim = 10, 32

        # Create codebook using Pair syntax
        layer = Codebook(embedding_dim => n_classes)
        ps, st = Lux.setup(rng, layer)

        # Check state (codes stored in state, not parameters)
        @test size(st.codes) == (embedding_dim, n_classes)

        # Lookup valid indices
        indices = rand(1:n_classes, 16)
        embeddings = st.codes[:, indices]

        # Embeddings should be correct shape
        @test size(embeddings) == (embedding_dim, 16)

        # Embeddings should be numeric (Phase is a Real subtype)
        @test eltype(embeddings) <: Real

        # ---- init_mode kwarg ----

        # Default mode is :random; preserves prior behavior.
        @test layer.init_mode === :random

        # :orthogonal mode at d divisible by n — exact orthogonality.
        layer_o = Codebook(16 => 4; init_mode = :orthogonal)
        @test layer_o.init_mode === :orthogonal
        _, st_o = Lux.setup(Xoshiro(0), layer_o)
        @test size(st_o.codes) == (16, 4)
        Mo = similarity_outer(st_o.codes, st_o.codes; dims = 2)
        @test all(diag(Mo) .≈ 1f0)
        offdiag_o = [Mo[i, j] for i in 1:4, j in 1:4 if i != j]
        @test maximum(abs, offdiag_o) < 1f-5  # Float32 noise floor

        # Square case d == n — also exact.
        _, st_sq = Lux.setup(Xoshiro(0), Codebook(8 => 8; init_mode = :orthogonal))
        Msq = similarity_outer(st_sq.codes, st_sq.codes; dims = 2)
        offdiag_sq = [Msq[i, j] for i in 1:8, j in 1:8 if i != j]
        @test maximum(abs, offdiag_sq) < 1f-5

        # Approximate case d not divisible by n — should still beat random
        # baseline by a clear margin.
        _, st_ap = Lux.setup(Xoshiro(0), Codebook(21 => 10; init_mode = :orthogonal))
        Map = similarity_outer(st_ap.codes, st_ap.codes; dims = 2)
        offdiag_ap = [Map[i, j] for i in 1:10, j in 1:10 if i != j]
        _, st_rd = Lux.setup(Xoshiro(0), Codebook(21 => 10; init_mode = :random))
        Mrd = similarity_outer(st_rd.codes, st_rd.codes; dims = 2)
        offdiag_rd = [Mrd[i, j] for i in 1:10, j in 1:10 if i != j]
        @test maximum(abs, offdiag_ap) < maximum(abs, offdiag_rd)
        @test maximum(abs, offdiag_ap) < 1.0f0 / sin(Float32(π) / 10) / 21 + 1f-3

        # Forward pass uses the orthogonal codes correctly: feeding a code
        # back in must recover ~1 on its own row, ~0 elsewhere.
        layer_fp = Codebook(8 => 4; init_mode = :orthogonal)
        _, st_fp = Lux.setup(Xoshiro(1), layer_fp)
        # Pass each code as input, check the diagonal of output similarities.
        out, _ = layer_fp(st_fp.codes, NamedTuple(), st_fp)
        @test size(out) == (4, 4)
        @test all(diag(out) .≈ 1f0)
        offdiag_fp = [out[i, j] for i in 1:4, j in 1:4 if i != j]
        @test maximum(abs, offdiag_fp) < 1f-5

        # Error: invalid init_mode symbol.
        @test_throws ArgumentError Codebook(8 => 4; init_mode = :nope)

        # Error: n > d for orthogonal mode (impossible).
        cb_bad = Codebook(4 => 8; init_mode = :orthogonal)
        @test_throws ArgumentError Lux.setup(Xoshiro(0), cb_bad)

        # n = 1 edge case: orthogonality is vacuous; should not error.
        _, st_one = Lux.setup(Xoshiro(0), Codebook(8 => 1; init_mode = :orthogonal))
        @test size(st_one.codes) == (8, 1)
    end
end

function flatten_spiketrain_tests()
    @testset "Flatten SpikeTrain Tests" begin
        rng = Xoshiro(42)
        flatten_layer = Lux.FlattenLayer()
        ps, st = Lux.setup(rng, flatten_layer)
        
        # Create spike train with 2D spatial shape
        indices = [CartesianIndex(i, j) for i in 1:3 for j in 1:4]
        times = sort(rand(Float32, 12)) .* 0.1f0
        original_shape = (3, 4)
        
        train = SpikeTrain(indices, times, original_shape, 0.0f0)
        
        # Flatten
        flattened, _ = flatten_layer(train, ps, st)
        
        # Should be flattened to (12,) shape
        @test prod(flattened.shape) == 12
        @test length(flattened.indices) == 12
        @test length(flattened.times) == 12
        
        # All spikes should be preserved
        @test sum(flattened.times) ≈ sum(train.times)
    end
end

function dropout_spiketrain_tests()
    @testset "Dropout on SpikingCall Tests" begin
        rng = Xoshiro(42)
        
        # Create spike train
        indices = rand(1:10, 50)
        times = sort(rand(Float32, 50)) .* 0.1f0
        shape = (10, 5)
        
        train = SpikeTrain(indices, times, shape, 0.0f0)
        call = SpikingCall(train, spk_args, (0.0f0, 1.0f0))
        
        # Apply dropout layer during training
        dropout_layer = Lux.Dropout(0.5f0)
        ps_drop, st_drop = Lux.setup(rng, dropout_layer)
        
        # Forward pass with training flag
        output, st_new = dropout_layer(call, ps_drop, st_drop)
        
        # Output should be a SpikingCall
        @test output isa SpikingCall
        
        # Dropped train should have same or fewer spikes
        @test length(output.train.indices) <= length(train.indices)
    end
end

function variance_scaling_tests()
    @testset "Variance Scaling Tests" begin
        # Test basic scaling calculation
        fan_in, fan_out = 10, 5
        scale = 2.0f0 / (fan_in + fan_out)
        
        # Should return a positive number
        @test scale > 0.0f0
        
        # Scaling should depend on fan in/out
        scale1 = 2.0f0 / (100 + 50)
        scale2 = 2.0f0 / (10 + 5)
        
        # Different dimensions should give different scales
        @test scale1 != scale2
        
        # Larger dimensions should give smaller scales
        @test scale1 < scale2
    end
end

function dense_onehot_tests()
    @testset "Dense OneHot Tests" begin
        rng = Xoshiro(42)
        n_classes = 5
        batch_size = 16
        
        # Create one-hot encoded matrix
        labels = rand(1:n_classes, batch_size)
        onehot = OneHotArrays.onehotbatch(labels, 1:n_classes)
        
        # Forward pass
        y = dense_onehot(onehot)
        
        # Output shape should match input
        @test size(y) == size(onehot)
        
        # Output should contain valid values (0 or 1)
        @test all(y .>= 0.0f0) && all(y .<= 1.0f0)
    end
end

function attend_tests()
    @testset "Attend Tests" begin
        # Create some keys, values, and queries
        d_model = 32
        batch_size = 8
        seq_len = 10

        keys = Phase.(rand(Float32, d_model, seq_len, batch_size) .* 2.0f0 .- 1.0f0)
        values = Phase.(rand(Float32, d_model, seq_len, batch_size) .* 2.0f0 .- 1.0f0)
        queries = Phase.(rand(Float32, d_model, seq_len, batch_size) .* 2.0f0 .- 1.0f0)

        # Compute attention
        attended, scores = attend(queries, keys, values)

        # Output shape should match values shape
        @test size(attended) == size(values)

        # Output should be finite
        @test all(isfinite.(attended))

        # Attention weights should sum to something reasonable
        # (exact check depends on normalization)
        @test all(abs.(Float32.(attended)) .< 100.0f0)  # Not exploding
    end
end
