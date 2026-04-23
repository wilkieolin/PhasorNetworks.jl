"""
    on_gpu(args...) -> Bool

Check if any of the provided arguments are CUDA arrays.

Used to determine if computation should proceed on GPU or CPU path.
Returns true if at least one argument is a CuArray.
"""
function on_gpu(args...)
    locs = [typeof(x) <: CuArray for x in args]
    return reduce(+, locs) > 0
end

#Kernels 

function threads_blks(l::Int, threads::Int = N_THREADS)
    blocks = cld(l, threads)
    return threads, blocks
end

"""
    gaussian_kernel_gpu(x::Float32, t::Float32, t_sigma::Float32) -> Float32

GPU-optimized version of the Gaussian kernel used in spike current calculations.
All inputs are Float32 for CUDA compatibility.

# Arguments
- `x::Float32`: Time of the spike
- `t::Float32`: Current time
- `t_sigma::Float32`: Width of the Gaussian kernel

Returns the spike current value at time t for a spike at time x.
"""
function gaussian_kernel_gpu(x::Float32, t::Float32, t_sigma::Float32)
    i = exp(-1.0f0 * ((t - x) / (2.0f0 * t_sigma))^2.0f0)
    return i
end

"""
    gaussian_kernel_gpu!(output, times, t, t_sigma)

CUDA kernel that computes Gaussian kernel values for all spike times in parallel.
Avoids dynamic dispatch overhead from broadcasting on GPU arrays.
"""
function gaussian_kernel_gpu!(output, times, t::Float32, t_sigma::Float32)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= length(times)
        x = times[i]
        output[i] = exp(-1.0f0 * ((t - x) / (2.0f0 * t_sigma))^2.0f0)
    end
    return nothing
end

"""
    raised_cosine_kernel_gpu(x::Float32, t::Float32, t_sigma::Float32) -> Float32

GPU-optimized version of the raised cosine kernel used in spike current calculations.
Provides smoother gradients and better numerical stability than Gaussian kernels.

# Arguments
- `x::Float32`: Time of the spike
- `t::Float32`: Current time
- `t_sigma::Float32`: Width parameter (support is ±2*t_sigma)

Returns the spike current value at time t for a spike at time x.
"""
function raised_cosine_kernel_gpu(x::Float32, t::Float32, t_sigma::Float32)
    dt = t - x
    half_width = 2.0f0 * t_sigma
    if abs(dt) <= half_width
        return 0.5f0 * (1.0f0 + CUDA.cos(3.1415927f0 * dt / half_width))
    else
        return 0.0f0
    end
end

"""
    raised_cosine_kernel_gpu!(output, times, t, t_sigma)

CUDA kernel that computes raised cosine kernel values for all spike times in parallel.
Provides better gradient stability for adjoint-based differentiation.
"""
function raised_cosine_kernel_gpu!(output, times, t::Float32, t_sigma::Float32)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= length(times)
        x = times[i]
        dt = t - x
        half_width = 2.0f0 * t_sigma
        if abs(dt) <= half_width
            output[i] = 0.5f0 * (1.0f0 + CUDA.cos(3.1415927f0 * dt / half_width))
        else
            output[i] = 0.0f0
        end
    end
    return nothing
end

"""
    periodic_raised_cosine_kernel_gpu(x::Float32, t::Float32, t_sigma::Float32, t_period::Float32) -> Float32

GPU-optimized periodic raised cosine kernel for oscillatory systems.
Computes the shortest distance on a periodic time domain before applying the raised cosine.
"""
function periodic_raised_cosine_kernel_gpu(x::Float32, t::Float32, t_sigma::Float32, t_period::Float32)
    # Compute shortest distance on ring of circumference t_period
    dt = mod(t - x + t_period/2.0f0, t_period) - t_period/2.0f0
    half_width = 2.0f0 * t_sigma
    if abs(dt) <= half_width
        return 0.5f0 * (1.0f0 + CUDA.cos(3.1415927f0 * dt / half_width))
    else
        return 0.0f0
    end
end

"""
    periodic_raised_cosine_kernel_gpu!(output, times, t, t_sigma, t_period)

CUDA kernel for periodic raised cosine kernel values.
"""
function periodic_raised_cosine_kernel_gpu!(output, times, t::Float32, t_sigma::Float32, t_period::Float32)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= length(times)
        x = times[i]
        # Compute shortest distance on ring of circumference t_period
        dt = mod(t - x + t_period/2.0f0, t_period) - t_period/2.0f0
        half_width = 2.0f0 * t_sigma
        if abs(dt) <= half_width
            output[i] = 0.5f0 * (1.0f0 + CUDA.cos(3.1415927f0 * dt / half_width))
        else
            output[i] = 0.0f0
        end
    end
    return nothing
end

function scatter_add_kernel!(output, values, indices)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= length(indices)
        index = indices[i]
        value = values[i]
        CUDA.@atomic output[index] += value
    end
    return nothing
end

"""
    parallel_scatter_add(indices::CuArray{Int}, values::CuArray{T}, output_size::Int) where T -> CuArray{T}

Perform parallel scatter-add operation on GPU using atomic operations.
Used for efficiently accumulating spike currents in the network.

# Arguments
- `indices::CuArray{Int}`: Target indices for the values
- `values::CuArray{T}`: Values to be scattered and added
- `output_size::Int`: Size of the output array

# Implementation
Uses CUDA atomic operations to safely accumulate values in parallel.
Each thread processes one (index, value) pair and atomically adds
to the corresponding location in the output array.

Returns a CuArray of size `output_size` containing the accumulated values.
"""
function parallel_scatter_add(indices::CuArray{Int}, values::CuArray{T}, output_size::Int) where T
    @assert length(indices) == length(values) "Length of indices and values must match"
    
    output = CUDA.zeros(T, output_size)
    threads = 256
    blocks = cld(length(indices), threads)
    
    @cuda threads=threads blocks=blocks scatter_add_kernel!(output, values, indices)
    
    return output
end

"""
    parallel_scatter_add!(output::CuArray, indices::CuArray{Int}, values::CuArray)

In-place version of parallel_scatter_add that reuses a pre-allocated output buffer.
The output array must be pre-zeroed before calling this function.
"""
function parallel_scatter_add!(output::CuArray, indices::CuArray{Int}, values::CuArray)
    @assert length(indices) == length(values) "Length of indices and values must match"
    
    threads = 256
    blocks = cld(length(indices), threads)
    
    @cuda threads=threads blocks=blocks scatter_add_kernel!(output, values, indices)
    
    return nothing
end

function interference_kernel(A, B, output, X, M, N, D)
    m = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    n = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    x = blockIdx().z

    if x <= X && m <= M && n <= N
        acc = 0.0f0
        @inbounds for d in 1:D
            a = A[d, m, x]
            b = B[d, n, x]
            
            sum_real = real(a) + real(b)
            sum_imag = imag(a) + imag(b)
            interference = sqrt(sum_real^2 + sum_imag^2)
            
            magnitude = clamp(interference, 0.0f0, 2.0f0)
            half_angle = acos(0.5f0 * magnitude)
            sim = cos(2.0f0 * half_angle)
            acc += sim
        end
        output[m, n, x] = acc / D
    end
    return
end

# Domains

"""
    potential_to_phase(potential::CuArray, ts::AbstractVector; spk_args::SpikingArgs, offset::Real=0.0f0, threshold::Bool=false) -> CuArray

GPU implementation of potential to phase conversion for neural states.
Extends the CPU version in domains.jl with CUDA-optimized array operations.

# Arguments
- `potential::CuArray`: Complex-valued neural potentials on GPU
- `ts::AbstractVector`: Time points corresponding to potential values
- `spk_args::SpikingArgs`: Spiking neuron parameters
- `offset::Real=0.0f0`: Time offset for phase calculation
- `threshold::Bool=false`: Whether to apply threshold checks

# Implementation
1. Computes reference zero-phase potentials
2. Calculates phase differences using complex angles
3. Normalizes to [-1, 1] range
4. Marks sub-threshold neurons with NaN

See also: [`potential_to_phase`](@ref) for CPU version
"""
function potential_to_phase(potential::CuArray, ts::AbstractVector; spk_args::SpikingArgs, offset::Real=0.0f0, threshold::Bool=false)
    @assert size(potential)[end] == length(ts) "Time dimensions must match"
    dims = collect(1:ndims(potential))

    #find the angle of a neuron representing 0 phase at the current moment in time
    current_zeros = cu(phase_to_potential.(0.0f0, ts, offset=offset, spk_args=spk_args))

    #get the arc subtended in the complex plane between that reference and our neuron potentials
    potential = permutedims(potential, reverse(dims))
    arc = angle.(current_zeros) .- angle.(potential)
    
    #normalize by pi and shift to -1, 1
    phase = Phase.(mod.((arc ./ pi_f32 .+ 1.0f0), 2.0f0) .- 1.0f0)

    #replace silent neurons with NaN (only if threshold=true)
    if threshold
        silent = abs.(potential) .< spk_args.threshold
        phase[silent] .= Phase(NaN)
    end
    phase = permutedims(phase, reverse(dims))

    return phase
end

function phase_to_train(phases::CuArray; spk_args::SpikingArgs, repeats::Int = 1, offset::Real = 0.0f0)
    shape = phases |> size
    indices = collect(CartesianIndices(shape)) |> vec
    times = phase_to_time(phases, spk_args=spk_args, offset=offset) |> vec

    if repeats > 1
        n_t = times |> length
        offsets = cu(repeat(collect(0:repeats-1) .* spk_args.t_period, inner=n_t))
        times = repeat(times, repeats) .+ offsets
        indices = repeat(indices, repeats)
    end

    train = SpikeTrainGPU(indices, times, shape, offset)
    return train
end

#Spiking

function parallel_current(stg::SpikeTrainGPU, t::Float32, spk_args::SpikingArgs)
    n = length(stg.times)
    currents = CUDA.zeros(Float32, n)

    threads = N_THREADS
    blocks = cld(n, threads)

    @cuda threads=threads blocks=blocks raised_cosine_kernel_gpu!(currents, stg.times, t, Float32(spk_args.t_window))

    output = parallel_scatter_add(stg.linear_indices, currents, stg.linear_shape)
    return output
end

function spike_current(train::SpikeTrainGPU, t::Float32, spk_args::SpikingArgs)
    scale = spk_args.spk_scale
    current = parallel_current(train, t, spk_args)
    current = reshape(current, train.shape)
    
    return current
end

"""
    spike_current!(output::CuArray, train::SpikeTrainGPU, t::Float32, spk_args::SpikingArgs,
                   currents_buffer::CuArray, scatter_buffer::CuArray)

In-place version of spike_current that reuses pre-allocated buffers to avoid
GPU memory allocations during ODE integration. This is critical for performance
when spike_current is called many times per simulation.

# Arguments
- `output::CuArray`: Pre-allocated output array with shape matching train.shape
- `train::SpikeTrainGPU`: Input spike train
- `t::Float32`: Current time
- `spk_args::SpikingArgs`: Spiking neuron parameters
- `currents_buffer::CuArray`: Pre-allocated buffer for kernel values (size = n_spikes)
- `scatter_buffer::CuArray`: Pre-allocated buffer for scatter-add (size = linear_shape)
"""
function spike_current!(output::CuArray{Float32}, train::SpikeTrainGPU, t::Float32, spk_args::SpikingArgs,
                        currents_buffer::CuArray{Float32}, scatter_buffer::CuArray{Float32})
    scale = spk_args.spk_scale
    t_window = Float32(spk_args.t_window)

    # Compute kernel values in-place using CUDA kernel
    threads = N_THREADS
    blocks = cld(length(train.times), threads)
    @cuda threads=threads blocks=blocks raised_cosine_kernel_gpu!(currents_buffer, train.times, t, t_window)
    currents_buffer .*= scale

    # Zero the scatter buffer and accumulate
    scatter_buffer .= 0.0f0
    parallel_scatter_add!(scatter_buffer, train.linear_indices, currents_buffer)

    # Copy reshaped result to output
    output .= reshape(scatter_buffer, train.shape)

    return nothing
end

function bias_current(bias::CuArray{<:Complex}, t::Real, t_offset::Real, spk_args::SpikingArgs)
    phase = complex_to_angle(bias)
    mag = abs.(bias)
    return bias_current(phase, mag, t, t_offset, spk_args)
end

function bias_current(phase::CuArray{<:Real}, mag::CuArray{<:Real}, t::Real, t_offset::Real, spk_args::SpikingArgs)
    #what times to the bias values correlate to?
    times = phase_to_time(phase, spk_args=spk_args, offset=t_offset)
    #determine the time within the cycle
    t_mod = Float32(mod(t, spk_args.t_period))
    #compute kernel values using CUDA kernel (periodic version for bias)
    n = length(times)
    kernel_vals = CUDA.zeros(Float32, n)
    threads = N_THREADS
    blocks = cld(n, threads)
    @cuda threads=threads blocks=blocks periodic_raised_cosine_kernel_gpu!(kernel_vals, times, t_mod, Float32(spk_args.t_window), Float32(spk_args.t_period))
    #scale by bias magnitude and reshape to match phase shape
    bias = reshape(mag .* kernel_vals, size(phase))

    return bias
end

function f32_tspan(tspan::Tuple{<:Real, <:Real})
    tspan = (Float32(tspan[1]), Float32(tspan[2]))
    return tspan
end

function oscillator_bank(u0::CuArray, dzdt::Function; tspan::Tuple{<:Float32, <:Float32}, spk_args::SpikingArgs)
    #solve the ODE
    prob = ODEProblem(dzdt, u0, tspan)
    sol = solve(prob, spk_args.solver; spk_args.solver_args...)

    return sol
end

"""
    oscillator_bank(x::SpikeTrainGPU; tspan=(0.0f0, 10.0f0), spk_args::SpikingArgs) -> ODESolution

GPU implementation of oscillator bank simulation for spiking neural networks.
Simulates the dynamics of a bank of oscillators driven by spike inputs.

# Arguments
- `x::SpikeTrainGPU`: Input spike train on GPU
- `tspan::Tuple{<:Real, <:Real}`: Time span for simulation
- `spk_args::SpikingArgs`: Parameters for neuron dynamics

# Implementation
Sets up and solves the ODE system:
dz/dt = k*z + spike_current(x, t), where k = leakage + i*(2pi/t_period)

See also: [`oscillator_bank`](@ref) in spiking.jl for CPU version
"""
function oscillator_bank(x::SpikeTrainGPU; tspan::Tuple{<:Real, <:Real} = (0.0f0, 10.0f0), spk_args::SpikingArgs)
    tspan = tspan |> f32_tspan 

    #set up compartments for each sample
    u0 = CUDA.zeros(ComplexF32, x.shape)
    #resonate in time with the input spikes
    dzdt(u, p, t) = resonant_update(u, spk_args.leakage, spk_args.t_period) .+ spike_current(x, t, spk_args)

    sol = oscillator_bank(u0, dzdt, tspan=tspan, spk_args=spk_args)

    return sol
end

"""
    similarity_outer(A::CuArray{ComplexF32,3}, B::CuArray{ComplexF32,3}; dims=2) -> CuArray{Float32,3}

GPU pairwise interference-similarity between two batches of complex vectors.
Permutes inputs to canonical `(features, n_vectors, batch)` layout and delegates
to [`_similarity_outer_canonical_complex`](@ref) — the shared kernel that
carries a closed-form `rrule` so the backward pass avoids the
`(features, M, N, batch)` tape Zygote would otherwise hold.

# Arguments
- `A::CuArray{ComplexF32,3}`: First set of vectors
- `B::CuArray{ComplexF32,3}`: Second set of vectors
- `dims::Int=2`: Dimension along which to slice for pairwise comparison
  (the other two dims are inferred as features and batch)

# Returns
Output shape `(n_vectors_A, n_vectors_B, batch)`; for `dims=2` inputs of shape
`(features, n_vectors, batch)` this is the natural attention-score layout.

See also: [`similarity_outer`](@ref) for CPU dispatches in `vsa.jl`,
and [`_similarity_outer_canonical_complex`](@ref) for the shared kernel.
"""
function similarity_outer(A::CuArray{ComplexF32,3}, B::CuArray{ComplexF32,3}; dims::Int=2)
    # Permute to canonical (D, M, X) / (D, N, X) layout, then delegate to the
    # shared kernel in vsa.jl which carries a closed-form rrule for the
    # backward pass (avoiding the (D,M,N,X) intermediates Zygote would
    # otherwise pin in the tape).

    @assert size(A, dims) > 0 "dims=$dims out of range for array with $(ndims(A)) dimensions"

    if dims == 2
        feature_dim, batch_dim = 1, 3
    elseif dims == 1
        feature_dim, batch_dim = 2, 3
    elseif dims == 3
        feature_dim, batch_dim = 1, 2
    else
        error("dims must be 1, 2, or 3 for 3D arrays")
    end

    sz_A = size(A); sz_B = size(B)
    @assert sz_B[batch_dim]   == sz_A[batch_dim]   "Batch size mismatch"
    @assert sz_B[feature_dim] == sz_A[feature_dim] "Feature dimension mismatch"

    perm_to_canonical = (feature_dim, dims, batch_dim)
    A_canonical = permutedims(A, perm_to_canonical)
    B_canonical = permutedims(B, perm_to_canonical)

    return _similarity_outer_canonical_complex(A_canonical, B_canonical)
end

function similarity_outer(A::CuArray{ComplexF32,2}, B::CuArray{ComplexF32,2}; dims::Int=2)
    # Reuse the 3D path with a singleton batch axis. Final transpose matches
    # the CPU 2D real convention (vsa.jl:408), which returns (N, M).
    dims in (1, 2) || error("dims must be 1 or 2 for 2D arrays")
    A3 = reshape(A, size(A, 1), size(A, 2), 1)
    B3 = reshape(B, size(B, 1), size(B, 2), 1)
    out3 = similarity_outer(A3, B3; dims=dims)
    return permutedims(dropdims(out3, dims=3), (2, 1))
end

# Support dispatch when inputs are real-valued CuArrays by converting to
# ComplexF32 on the device and delegating to the complex-valued GPU kernels.
function similarity_outer(A::CuArray{<:Real,3}, B::CuArray{<:Real,3}; dims::Int=2)
    # Convert phase angles to complex representation on GPU and delegate to
    # the complex-valued, vectorized implementation.
    A_c = angle_to_complex(A)
    B_c = angle_to_complex(B)
    return similarity_outer(A_c, B_c; dims=dims)
end

function similarity_outer(A::CuArray{<:Real,2}, B::CuArray{<:Real,2}; dims::Int=2)
    A_c = angle_to_complex(A)
    B_c = angle_to_complex(B)
    return similarity_outer(A_c, B_c; dims=dims)
end