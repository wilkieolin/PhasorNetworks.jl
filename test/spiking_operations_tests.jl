# spiking_operations_tests.jl
#
# Coverage for spike-train and spiking-kernel utilities in src/spiking.jl and
# src/domains.jl that were previously untested. Revived and corrected from the
# old PROPOSED_ draft: assertions now match the actual implementations, the
# removed `arc_gaussian_kernel` test is replaced by `periodic_gaussian_kernel`,
# and the broken `mean_phase` / redundant constructor tests were dropped.
#
# Note: several functions exercised here (find_spikes_rf, is_active,
# check_offsets, gaussian_kernel*, periodic_gaussian_kernel) are internal and
# not exported, so they are called via the `PhasorNetworks.` qualifier.

using Test
using PhasorNetworks
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
        find_spikes_rf_tests()
        gaussian_kernel_tests()
        is_active_tests()
        match_offsets_tests()
        check_offsets_tests()
    end
end

function delay_train_tests()
    @testset "Delay Train Tests" begin
        indices = [1, 3, 5, 2, 4]
        times = Float32.([0.1, 0.2, 0.3, 0.15, 0.25])
        shape = (5, 10)
        train = SpikeTrain(indices, times, shape, 0.0f0)

        delay_amount = 0.5f0
        delayed_train = delay_train(train, delay_amount, 0.0f0)

        # Times shifted by the delay
        @test delayed_train.times ≈ train.times .+ delay_amount
        # Indices and shape preserved
        @test delayed_train.indices == train.indices
        @test delayed_train.shape == train.shape
        # With offset arg 0, the train offset is unchanged
        @test delayed_train.offset ≈ train.offset

        # The third arg adds to the train's offset
        offset_amount = 0.1f0
        delayed_with_offset = delay_train(train, delay_amount, offset_amount)
        @test delayed_with_offset.offset ≈ train.offset + offset_amount
    end
end

function count_nans_tests()
    @testset "Count NaNs Tests" begin
        # (channels, time, batch) phases with NaNs in known positions
        phases = ones(Float32, 5, 10, 3)
        phases[1, :, :] .= NaN       # 10 * 3 = 30 NaNs
        phases[3, 1:5, 2] .= NaN     # 5 NaNs

        nan_counts = count_nans(phases)

        # One count per first dimension
        @test size(nan_counts) == (5,)
        @test nan_counts[1] == 30
        @test nan_counts[3] == 5
        @test nan_counts[2] == 0
    end
end

function zero_nans_tests()
    @testset "Zero NaNs Tests" begin
        data = randn(Xoshiro(0), Float32, 10, 20)
        original = copy(data)
        data[1:3, 1:5] .= NaN

        zeroed = zero_nans(data)

        @test size(zeroed) == size(data)
        # NaN positions are now zero
        @test all(zeroed[1:3, 1:5] .== 0.0f0)
        # Non-NaN values are unchanged from the original
        @test all(zeroed[4:end, 6:end] .≈ original[4:end, 6:end])
        # No NaNs remain
        @test !any(isnan.(zeroed))
    end
end

function stack_trains_tests()
    @testset "Stack Trains Tests" begin
        # stack_trains requires identical shape + offset and CartesianIndex
        # spatial indices (it prepends a new leading dimension per train).
        rng = Xoshiro(1)
        shape = (5, 10)
        trains = SpikeTrain[]
        for _ in 1:3
            idx = [CartesianIndex(rand(rng, 1:5), rand(rng, 1:10)) for _ in 1:10]
            times = sort(rand(rng, Float32, 10)) .* 0.5f0
            push!(trains, SpikeTrain(idx, times, shape, 0.0f0))
        end

        stacked = stack_trains(trains)

        total_spikes = sum(length(t.indices) for t in trains)
        @test length(stacked.indices) == total_spikes
        @test length(stacked.times) == total_spikes
        # A new leading dimension is added: (n_trains, shape...)
        @test stacked.shape == (length(trains), shape...)
        @test stacked.offset == 0.0f0
        @test all(stacked.times .>= 0.0f0) && all(stacked.times .<= 0.5f0)
    end
end

function vcat_trains_tests()
    @testset "Concatenate Trains Tests" begin
        # vcat_trains combines spike events on an identically-shaped grid;
        # the shape is preserved (it does NOT grow).
        shape = (3, 5)
        indices1 = [CartesianIndex(1, 1), CartesianIndex(2, 3)]
        times1 = Float32.([0.1, 0.2])
        indices2 = [CartesianIndex(3, 2), CartesianIndex(1, 4)]
        times2 = Float32.([0.15, 0.25])

        train1 = SpikeTrain(indices1, times1, shape, 0.0f0)
        train2 = SpikeTrain(indices2, times2, shape, 0.0f0)

        vcatted = vcat_trains(train1, train2)

        @test length(vcatted.indices) == 4
        @test length(vcatted.times) == 4
        # Shape is unchanged by concatenation
        @test vcatted.shape == shape
        @test vcatted.offset == 0.0f0
    end
end

function find_spikes_rf_tests()
    @testset "Find Spikes R&F Tests" begin
        # Two identical R&F neurons (rows) oscillating at 10 Hz over 0.2 s.
        # Voltage = imag, current = real; spikes occur at voltage maxima
        # (current zero-crossings, +→-) above threshold.
        t = collect(0.0f0:0.001f0:0.2f0)
        freq = 10.0f0
        voltage = sin.(2.0f0 * π * freq .* t)
        current = cos.(2.0f0 * π * freq .* t)
        row_v = vcat(voltage', voltage')      # (2, n_t)
        row_c = vcat(current', current')
        u = row_c .+ 1.0f0im .* row_v

        spk_args = SpikingArgs(threshold = 0.5f0)

        channels, spike_times = PhasorNetworks.find_spikes_rf(u, t, spk_args; dim=2)

        # Two periods of oscillation → spikes detected
        @test length(spike_times) > 0
        @test length(channels) == length(spike_times)
        # Spike times within the simulated window
        @test all(spike_times .>= t[1]) && all(spike_times .<= t[end])
        # Spatial locations are CartesianIndex (one spatial axis here)
        @test all(c -> c isa CartesianIndex, channels)
    end
end

function gaussian_kernel_tests()
    @testset "Gaussian Kernel Tests" begin
        spike_times = Float32.([0.1, 0.2, 0.3])
        sigma = 0.02f0
        current_time = 0.21f0    # strictly closest to the 0.2 spike

        kernel = PhasorNetworks.gaussian_kernel(spike_times, current_time, sigma)

        @test length(kernel) == length(spike_times)
        @test all(kernel .>= 0.0f0) && all(kernel .<= 1.0f0)
        # Peaks at the nearest spike
        @test argmax(kernel) == 2

        # Vectorized variant: (n_spikes, n_times)
        times = collect(0.0f0:0.05f0:1.0f0)
        kernel_vec = PhasorNetworks.gaussian_kernel_vec(spike_times, times, sigma)
        @test size(kernel_vec) == (length(spike_times), length(times))
        @test all(kernel_vec .>= 0.0f0) && all(kernel_vec .<= 1.0f0)

        # Periodic (ring-distance) variant replaces the old arc_gaussian_kernel.
        t_period = 1.0f0
        periodic = PhasorNetworks.periodic_gaussian_kernel(spike_times, current_time, sigma, t_period)
        @test length(periodic) == length(spike_times)
        @test all(periodic .>= 0.0f0) && all(periodic .<= 1.0f0)
        # On the ring, a spike exactly at current_time would give 1; the
        # nearest spike (0.2) still dominates.
        @test argmax(periodic) == 2
    end
end

function is_active_tests()
    @testset "Is Active Tests" begin
        spike_times = Float32.([0.1, 0.2, 0.5, 0.9])
        current_time = 0.25f0
        t_window = 0.05f0

        active = PhasorNetworks.is_active(spike_times, current_time, t_window)

        @test length(active) == length(spike_times)
        @test eltype(active) == Bool
        # Default sigma=9 → window 0.25 ± 0.45 = [-0.2, 0.7]
        @test active[1] && active[2] && active[3]
        @test !active[4]   # 0.9 lies outside the window

        # Wider sigma admits at least as many spikes as a tighter one
        active_tight = PhasorNetworks.is_active(spike_times, current_time, t_window; sigma=3.0f0)
        active_loose = PhasorNetworks.is_active(spike_times, current_time, t_window; sigma=20.0f0)
        @test sum(active_loose) >= sum(active_tight)
    end
end

function match_offsets_tests()
    @testset "Match Offsets Tests" begin
        indices1 = [1, 2, 3]
        times1 = Float32.([0.1, 0.2, 0.3])
        train1 = SpikeTrain(indices1, times1, (5, 10), 0.0f0)

        indices2 = [2, 3, 4]
        times2 = Float32.([0.15, 0.25, 0.35])
        train2 = SpikeTrain(indices2, times2, (5, 10), 0.2f0)

        matched1, matched2 = match_offsets(train1, train2)

        # Offsets now agree
        @test matched1.offset ≈ matched2.offset
        @test matched1 isa SpikeTrain
        @test matched2 isa SpikeTrain
        @test all(matched1.times .>= 0.0f0)
        @test all(matched2.times .>= 0.0f0)
    end
end

function check_offsets_tests()
    @testset "Check Offsets Tests" begin
        indices = [1, 2, 3]
        times = Float32.([0.1, 0.2, 0.3])
        shape = (5, 10)

        train1 = SpikeTrain(indices, times, shape, 0.0f0)
        train2 = SpikeTrain(indices, times, shape, 0.0f0)
        train3 = SpikeTrain(indices, times, shape, 0.1f0)

        # Two-arg form
        @test PhasorNetworks.check_offsets(train1, train2) == true
        @test PhasorNetworks.check_offsets(train1, train3) == false

        # Variadic form
        @test PhasorNetworks.check_offsets(train1, train2, train1) == true
        @test PhasorNetworks.check_offsets(train1, train2, train3) == false
    end
end
