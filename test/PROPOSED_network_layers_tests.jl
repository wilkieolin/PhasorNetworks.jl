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
        x = (rand(Float32, in_dim, batch_size) .* 2.0f0 .- 1.0f0)
        
        # Forward pass
        y, st_new = layer(x, ps, st)
        
        # Output shape should be correct
        @test size(y) == (out_dim, batch_size)
        
        # Output should be phases (in [-1, 1])
        @test all(y .>= -1.0f0) && all(y .<= 1.0f0)
        
        # State should be updated
        @test st_new != st
        
        # Different inputs should give different outputs
        x2 = (rand(Float32, in_dim, batch_size) .* 2.0f0 .- 1.0f0)
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
        
        # Should not have bias in parameters
        @test !haskey(ps, :bias)
        
        x = rand(Float32, in_dim, batch_size) .* 2.0f0 .- 1.0f0
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
            
            x = rand(Float32, in_dim, batch_size) .* 2.0f0 .- 1.0f0
            y, _ = layer(x, ps, st)
            
            # Output should be valid
            @test size(y) == (out_dim, batch_size)
            @test all(isfinite.(y))
            
            # complex_to_angle should produce phases in [-1, 1]
            if activation == complex_to_angle
                @test all(y .>= -1.0f0) && all(y .<= 1.0f0)
            end
        end
    end
end

function phasor_conv_tests()
    @testset "PhasorConv Tests" begin
        rng = Xoshiro(42)
        
        # Create convolutional layer
        layer = PhasorConv((3, 3), 1 => 4, complex_to_angle)
        ps, st = Lux.setup(rng, layer)
        
        # Create phase input
        height, width, channels, batch = 28, 28, 1, 8
        x = rand(Float32, height, width, channels, batch) .* 2.0f0 .- 1.0f0
        
        # Forward pass
        y, st_new = layer(x, ps, st)
        
        # Output shape should have correct channel dimension
        @test size(y)[end] == batch  # Batch size preserved
        @test ndims(y) == 4
        
        # Output values should be valid
        @test all(isfinite.(y))
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
        dims = 32
        
        # Create residual block
        layer = ResidualBlock(dims)
        ps, st = Lux.setup(rng, layer)
        
        # Create input
        batch_size = 16
        x = randn(ComplexF32, dims, batch_size)
        
        # Forward pass
        y, st_new = layer(x, ps, st)
        
        # Output shape should match input
        @test size(y) == size(x)
        
        # Output should be complex
        @test eltype(y) <: Complex
        
        # Gradient flow should work
        @test all(isfinite.(y))
    end
end

function codebook_tests()
    @testset "Codebook Tests" begin
        rng = Xoshiro(42)
        n_classes, embedding_dim = 10, 32
        
        # Create codebook
        layer = Codebook(n_classes, embedding_dim)
        ps, st = Lux.setup(rng, layer)
        
        # Check parameters
        @test size(ps.embeddings) == (embedding_dim, n_classes)
        
        # Lookup valid indices
        indices = rand(1:n_classes, 16)
        embeddings = ps.embeddings[:, indices]
        
        # Embeddings should be correct shape
        @test size(embeddings) == (embedding_dim, 16)
        
        # Embeddings should be complex (likely)
        @test eltype(embeddings) <: Complex
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
        
        # Should be flattened to (12, 1) or similar
        @test flattened.shape[1] == 12
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
        call = SpikingCall(train, default_spk_args(), (0.0f0, 1.0f0))
        
        # Apply dropout with 50% drop rate
        dropped_call, _, rng_new = Lux.Dropout(0.5f0)(rng, call, 0.5f0, true, 2.0f0, :)
        
        # Dropped train should have fewer spikes
        @test length(dropped_call.train.indices) <= length(train.indices)
        
        # Some spikes should be kept (with high probability)
        @test length(dropped_call.train.indices) > 0
        
        # RNG should be updated
        @test rng_new != rng
    end
end

function variance_scaling_tests()
    @testset "Variance Scaling Tests" begin
        # Test basic scaling
        scale = variance_scaling(10, 5)
        
        # Should return a positive number
        @test scale > 0.0f0
        
        # Scaling should depend on fan in/out
        scale1 = variance_scaling(100, 50)
        scale2 = variance_scaling(10, 5)
        
        # Different dimensions should give different scales
        @test scale1 != scale2
        
        # Larger dimensions should generally give smaller scales
        @test scale1 < scale2
    end
end

function dense_onehot_tests()
    @testset "Dense OneHot Tests" begin
        rng = Xoshiro(42)
        n_classes = 5
        batch_size = 16
        
        # Create layer
        layer = dense_onehot(n_classes)
        ps, st = Lux.setup(rng, layer)
        
        # Create input (phases)
        x = rand(Float32, n_classes, batch_size) .* 2.0f0 .- 1.0f0
        
        # Forward pass
        y, _ = layer(x, ps, st)
        
        # Output shape should match input
        @test size(y) == size(x)
        
        # Output should contain valid phases
        @test all(isfinite.(y))
    end
end

function attend_tests()
    @testset "Attend Tests" begin
        # Create some keys, values, and queries
        d_model = 32
        batch_size = 8
        seq_len = 10
        
        keys = randn(Float32, d_model, seq_len, batch_size)
        values = randn(Float32, d_model, seq_len, batch_size)
        queries = randn(Float32, d_model, seq_len, batch_size)
        
        # Compute attention
        attended = attend(queries, keys, values)
        
        # Output shape should match values shape
        @test size(attended) == size(values)
        
        # Output should be finite
        @test all(isfinite.(attended))
        
        # Attention weights should sum to something reasonable
        # (exact check depends on normalization)
        @test all(abs.(attended) .< 100.0f0)  # Not exploding
    end
end
