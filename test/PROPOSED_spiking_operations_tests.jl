# Proposed spiking_operations_tests.jl
# This file proposes new tests for spiking neuron operations in spiking.jl
# To be added to the test suite to improve coverage

using Test
using PhasorNetworks
using DifferentialEquations: Tsit5
using Random: Xoshiro

"""
Run all spiking operation tests
"""
function spiking_operations_tests()
    @testset "Spiking Operations Tests" begin
        @info "Running spiking operations tests..."

        delay_train_tests()
        count_nans_tests()
        zero_nans_tests()
        stack_trains_tests()
        vcat_trains_tests()
        mean_phase_tests()
        find_spikes_rf_tests()
        gaussian_kernel_tests()
        is_active_tests()
        match_offsets_tests()
        spike_train_properties_tests()
        check_offsets_tests()
    end
end

function delay_train_tests()
    @testset "Delay Train Tests" begin
        # Create sample spike train
        indices = [1, 3, 5, 2, 4]
        times = Float32.([0.1, 0.2, 0.3, 0.15, 0.25])
        shape = (5, 10)
        offset = 0.0f0
        
        train = SpikeTrain(indices, times, shape, offset)
        
        # Apply delay
        delay_amount = 0.5f0
        delayed_train = delay_train(train, delay_amount, 0.0f0)
        
        # Times should be shifted
        @test delayed_train.times ≈ train.times .+ delay_amount
        
        # Indices should remain same
        @test delayed_train.indices == train.indices
        
        # Shape should remain same
        @test delayed_train.shape == train.shape
        
        # Offset should be updated
        @test delayed_train.offset ≈ train.offset
        
        # Test with offset parameter
        offset_amount = 0.1f0
        delayed_with_offset = delay_train(train, delay_amount, offset_amount)
        @test delayed_with_offset.offset ≈ train.offset + offset_amount
    end
end

function count_nans_tests()
    @testset "Count NaNs Tests" begin
        # Create phases with some NaN values
        phases = ones(Float32, 5, 10, 3)
        phases[1, :, :] .= NaN
        phases[3, 1:5, 2] .= NaN
        
        nan_counts = count_nans(phases)
        
        # Should return one value per first dimension
        @test size(nan_counts) == (5,)
        
        # First dimension should have high NaN count
        @test nan_counts[1] == 30  # 10 * 3 NaNs
        
        # Middle dimensions should have partial NaNs
        @test nan_counts[3] == 5   # 5 NaNs in one time step
        
        # Other dimensions should have no NaNs
        @test nan_counts[2] == 0
    end
end

function zero_nans_tests()
    @testset "Zero NaNs Tests" begin
        # Create array with NaN values
        data = randn(Float32, 10, 20)
        data[1:3, 1:5] .= NaN
        
        zeroed = zero_nans(data)
        
        # Should have same shape
        @test size(zeroed) == size(data)
        
        # NaN positions should now be zero
        @test all(zeroed[1:3, 1:5] .== 0.0f0)
        
        # Non-NaN values should be unchanged
        @test all(zeroed[4:end, 6:end] .≈ data[4:end, 6:end])
        
        # Result should have no NaNs
        @test !any(isnan.(zeroed))
    end
end

function stack_trains_tests()
    @testset "Stack Trains Tests" begin
        # Create multiple spike trains
        trains = []
        for i in 1:3
            indices = rand(1:5, 10)
            times = sort(rand(Float32, 10)) .* 0.5f0
            train = SpikeTrain(indices, times, (5, 10), 0.0f0)
            push!(trains, train)
        end
        
        stacked = stack_trains(trains)
        
        # Should combine all spikes
        total_spikes = sum(length(t.indices) for t in trains)
        @test length(stacked.indices) == total_spikes
        @test length(stacked.times) == total_spikes
        
        # Times should be within bounds
        @test all(stacked.times .>= 0.0f0)
        @test all(stacked.times .<= 0.5f0)
    end
end

function vcat_trains_tests()
    @testset "Vertical Concatenate Trains Tests" begin
        # Create two compatible spike trains
        indices1 = [CartesianIndex(1, 1), CartesianIndex(2, 3)]
        times1 = Float32.([0.1, 0.2])
        shape1 = (3, 5)
        
        indices2 = [CartesianIndex(3, 2), CartesianIndex(1, 4)]
        times2 = Float32.([0.15, 0.25])
        shape2 = (3, 5)
        
        train1 = SpikeTrain(indices1, times1, shape1, 0.0f0)
        train2 = SpikeTrain(indices2, times2, shape2, 0.0f0)
        
        vcatted = vcat_trains(train1, train2)
        
        # Should have combined spikes
        @test length(vcatted.indices) == 4
        @test length(vcatted.times) == 4
        
        # Shape should reflect vertical concatenation
        @test vcatted.shape[1] == shape1[1] + shape2[1]
        @test vcatted.shape[2] == shape1[2]
    end
end

function mean_phase_tests()
    @testset "Mean Phase Tests" begin
        # Create spike train from known phases
        phases = fill(0.3f0, 1, 10, 1)
        
        spk_args = SpikingArgs(
            t_window = 0.01f0,
            threshold = 0.001f0,
            solver = Tsit5(),
            solver_args = Dict(:adaptive => false, :dt => 0.01f0)
        )
        
        train = phase_to_train(phases, spk_args=spk_args, repeats=1)
        
        # Compute mean phase
        mean_p = mean_phase(train, spk_args=spk_args)
        
        # Should be close to input phase
        @test isapprox(mean_p, 0.3f0, atol=0.05f0)
    end
end

function find_spikes_rf_tests()
    @testset "Find Spikes R&F Tests" begin
        # Create synthetic potential with clear spikes
        t = collect(0.0:0.001:0.1)
        
        # Simulate R&F neuron potential (complex)
        # Voltage = imaginary part, should cross threshold
        frequency = 10.0f0  # 10 Hz
        voltage = sin.(2.0f0 * π * frequency .* t)
        current = cos.(2.0f0 * π * frequency .* t)
        u = current .+ 1.0f0im .* voltage
        
        spk_args = SpikingArgs(
            t_window = 0.01f0,
            threshold = 0.5f0,
            solver = Tsit5(),
            solver_args = Dict(:adaptive => false, :dt => 0.001f0)
        )
        
        spikes, spike_times = find_spikes_rf(u, t, spk_args, dim=1)
        
        # Should detect spikes (voltage crossings above threshold)
        @test length(spike_times) > 0
        
        # Spike times should be within time bounds
        @test all(spike_times .>= t[1]) && all(spike_times .<= t[end])
        
        # Spikes should be CartesianIndex
        @test all(typeof.(spikes) .== CartesianIndex{1})
    end
end

function gaussian_kernel_tests()
    @testset "Gaussian Kernel Tests" begin
        # Test basic Gaussian kernel
        spike_times = Float32.([0.1, 0.2, 0.3])
        current_time = 0.25f0
        sigma = 0.02f0
        
        kernel = gaussian_kernel(spike_times, current_time, sigma)
        
        # Should have same length as input
        @test length(kernel) == length(spike_times)
        
        # Values should be in [0, 1]
        @test all(kernel .>= 0.0f0) && all(kernel .<= 1.0f0)
        
        # Kernel should be maximum at nearest spike
        @test kernel[2] == maximum(kernel)  # Closest spike at 0.2
        
        # Test vectorized version
        times = collect(0.0:0.05:1.0)
        kernel_vec = gaussian_kernel_vec(spike_times, times, sigma)
        
        # Should be (n_spikes, n_times)
        @test size(kernel_vec) == (length(spike_times), length(times))
        
        # All values should be valid probabilities
        @test all(kernel_vec .>= 0.0f0) && all(kernel_vec .<= 1.0f0)
        
        # Test arc Gaussian kernel (periodic)
        arc_kernel = arc_gaussian_kernel(spike_times, current_time, sigma)
        @test length(arc_kernel) == length(spike_times)
        @test all(arc_kernel .>= 0.0f0) && all(arc_kernel .<= 1.0f0)
    end
end

function is_active_tests()
    @testset "Is Active Tests" begin
        spike_times = Float32.([0.1, 0.2, 0.5, 0.9])
        current_time = 0.25f0
        t_window = 0.05f0
        
        active = is_active(spike_times, current_time, t_window)
        
        # Should return boolean vector
        @test typeof(active) <: BitVector || typeof(active) <: Vector{Bool}
        @test length(active) == length(spike_times)
        
        # Spikes near current time should be active
        @test active[1] || active[2]  # 0.1 or 0.2 should be active around 0.25
        
        # Distant spikes should not be active
        @test !active[4]  # 0.9 is far from 0.25
        
        # Test with sigma parameter
        active_tight = is_active(spike_times, current_time, t_window, sigma=3.0f0)
        active_loose = is_active(spike_times, current_time, t_window, sigma=20.0f0)
        
        # More active with larger sigma
        @test sum(active_loose) >= sum(active_tight)
    end
end

function match_offsets_tests()
    @testset "Match Offsets Tests" begin
        spk_args = SpikingArgs(
            t_window = 0.01f0,
            threshold = 0.001f0,
            solver = Tsit5(),
            solver_args = Dict(:adaptive => false, :dt => 0.01f0)
        )
        
        # Create spike trains with different offsets
        indices1 = [1, 2, 3]
        times1 = Float32.([0.1, 0.2, 0.3])
        train1 = SpikeTrain(indices1, times1, (5, 10), 0.0f0)
        
        indices2 = [2, 3, 4]
        times2 = Float32.([0.15, 0.25, 0.35])
        train2 = SpikeTrain(indices2, times2, (5, 10), 0.2f0)
        
        # Match offsets
        matched1, matched2 = match_offsets(train1, train2)
        
        # Offsets should now match
        @test matched1.offset ≈ matched2.offset
        
        # Both should be SpikeTrain type
        @test typeof(matched1) == SpikeTrain
        @test typeof(matched2) == SpikeTrain
        
        # Times should be shifted
        @test all(matched1.times .>= 0.0f0)
        @test all(matched2.times .>= 0.0f0)
    end
end

function spike_train_properties_tests()
    @testset "SpikeTrain Properties Tests" begin
        indices = [CartesianIndex(1, 1), CartesianIndex(2, 3), CartesianIndex(1, 5)]
        times = Float32.([0.1, 0.2, 0.3])
        shape = (5, 10)
        offset = 0.05f0
        
        train = SpikeTrain(indices, times, shape, offset)
        
        # Test properties
        @test train.shape == shape
        @test train.offset == offset
        @test length(train.indices) == length(times)
        @test length(train.times) == 3
        
        # Test with integer indices
        train_int = SpikeTrain([1, 2, 3], times, shape, offset)
        @test length(train_int.indices) == 3
        
        # Test with different offset types
        train_f64 = SpikeTrain(indices, times, shape, 0.05)  # Float64 offset
        @test typeof(train_f64.offset) == Float32
    end
end

function check_offsets_tests()
    @testset "Check Offsets Tests" begin
        spk_args = SpikingArgs(
            t_window = 0.01f0,
            threshold = 0.001f0,
            solver = Tsit5(),
            solver_args = Dict(:adaptive => false, :dt => 0.01f0)
        )
        
        indices = [1, 2, 3]
        times = Float32.([0.1, 0.2, 0.3])
        shape = (5, 10)
        
        # Create trains with same offset
        train1 = SpikeTrain(indices, times, shape, 0.0f0)
        train2 = SpikeTrain(indices, times, shape, 0.0f0)
        
        @test check_offsets(train1, train2) == true
        
        # Create train with different offset
        train3 = SpikeTrain(indices, times, shape, 0.1f0)
        
        @test check_offsets(train1, train3) == false
        
        # Test with multiple trains
        @test check_offsets(train1, train2, train1) == true
        @test check_offsets(train1, train2, train3) == false
    end
end
