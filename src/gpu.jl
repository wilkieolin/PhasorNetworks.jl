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
    phase = mod.((arc ./ pi_f32 .+ 1.0f0), 2.0f0) .- 1.0f0

    #replace silent neurons with NaN
    silent = abs.(potential) .< spk_args.threshold
    phase[silent] .= Float32(NaN)
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
    
    @cuda threads=threads blocks=blocks gaussian_kernel_gpu!(currents, stg.times, t, Float32(spk_args.t_window))
    
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
    
    # Compute kernel values in-place
    currents_buffer .= gaussian_kernel_gpu.(train.times, t, t_window) .* scale
    
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
    t = Float32(mod(t, spk_args.t_period))
    #add the active currents, scaled by the gaussian kernel & bias magnitude
    bias = mag .* gaussian_kernel_gpu.(times, t, Float32(spk_args.t_window))

    return bias
end

function f32_tspan(tspan::Tuple{<:Real, <:Real})
    tspan = (Float32(tspan[1]), Float32(tspan[2]))
    return tspan
end

function oscillator_bank(u0::CuArray, dzdt::Function; tspan::Tuple{<:Float32, <:Float32}, spk_args::SpikingArgs)
    #solve the memory compartment
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
dz/dt = update_fn(z) + spike_current(x, t)
where z is the complex potential of each neuron.

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

function oscillator_bank(x::SpikeTrainGPU, w::AbstractMatrix, b::AbstractVecOrMat; kwargs...)
    return oscillator_bank(x, cu(w), cu(b); kwargs...)
end

function oscillator_bank(x::SpikeTrainGPU{2}, w::CuArray, b::CuArray; tspan::Tuple{<:Real, <:Real}, spk_args::SpikingArgs)
    tspan = tspan |> f32_tspan
    #get the number of batches & output neurons
    output_shape = (size(w, 1), x.shape[2])
    u0 = CUDA.zeros(ComplexF32, output_shape)

    #solve the ODE over the given time span
    dzdt(u, p, t) = resonant_update(u, spk_args.leakage, spk_args.t_period) + w * spike_current(x, t, spk_args) .+ bias_current(b, t, x.offset, spk_args)
    sol = oscillator_bank(u0, dzdt, tspan=tspan, spk_args=spk_args)

    #return full solution
    return sol
end

"""
    oscillator_bank(x::SpikeTrainGPU{3}, w::CuArray, b::CuArray; tspan, spk_args) -> ODESolution

GPU-optimized neural network layer simulation for 3D spike trains (batched 2D data).
Implements the weighted connections and bias terms for network layers.

# Arguments
- `x::SpikeTrainGPU{3}`: Input spike train with shape (features, spatial_dim, batch)
- `w::CuArray`: Weight matrix
- `b::CuArray`: Bias terms
- `tspan`: Time span for simulation
- `spk_args::SpikingArgs`: Neuron parameters

# Implementation
Solves the neural ODE:
dz/dt = update_fn(z) + W * spike_current(x, t) + bias_current(b, t)

Returns an ODESolution object containing the network dynamics.

See also: Other `oscillator_bank` methods for different dimensionalities
"""
function oscillator_bank(x::SpikeTrainGPU{3}, w::CuArray, b::CuArray; tspan::Tuple{<:Real, <:Real}, spk_args::SpikingArgs)
    tspan = tspan |> f32_tspan
    #set up functions to define the neuron's differential equations
    update_fn = spk_args.update_fn
    #get the number of batches & output neurons
    output_shape = (size(w, 1), x.shape[2], x.shape[3])
    u0 = CUDA.zeros(ComplexF32, output_shape)

    #solve the ODE over the given time span
    dzdt(u, p, t) = update_fn(u) + batched_mul(x, spike_current(x, t, spk_args)) .+ bias_current(b, t, x.offset, spk_args)
    sol = oscillator_bank(u0, dzdt, tspan=tspan, spk_args=spk_args)

    #return full solution
    return sol
end

function oscillator_bank(x::SpikeTrainGPU{4}, w::CuArray, b::CuArray; tspan::Tuple{<:Real, <:Real}, spk_args::SpikingArgs)
    tspan = tspan |> f32_tspan
    #set up functions to define the neuron's differential equations
    update_fn = spk_args.update_fn
    #get the number of batches, channels, & output neurons
    output_shape = (size(w, 1), size(w,2), x.shape[2], x.shape[3])
    u0 = CUDA.zeros(ComplexF32, output_shape)

    #solve the ODE over the given time span
    dzdt(u, p, t) = update_fn(u) + batched_mul(x, spike_current(x, t, spk_args)) .+ bias_current(b, t, x.offset, spk_args)
    sol = oscillator_bank(u0, dzdt, tspan=tspan, spk_args=spk_args)

    #return full solution
    return sol
end

"""
    similarity_outer(A::CuArray{ComplexF32,3}, B::CuArray{ComplexF32,3}; dims=2) -> CuArray{Float32,3}

GPU-optimized pairwise similarity computation between sets of complex-valued vectors.
Provides a vectorized implementation that is efficient on GPUs and compatible with automatic differentiation.

# Arguments
- `A::CuArray{ComplexF32,3}`: First set of vectors
- `B::CuArray{ComplexF32,3}`: Second set of vectors
- `dims::Int=2`: Dimension along which to slice vectors for pairwise comparison

For `dims=2` with shape `(features, n_vectors, batch)`:
- Returns shape `(n_vectors_A, n_vectors_B, batch)` matching CPU real-valued implementation

Note: The 2D version returns `(n_vectors_B, n_vectors_A)` to match CPU's transposed output.

# Implementation Details
- Uses broadcasting for vectorized operations
- Avoids custom CUDA kernels for AD compatibility (e.g., with Zygote)
- Optimizes similarity calculation using trigonometric identities

See also: [`similarity_outer`](@ref) in vsa.jl for CPU version
"""
function similarity_outer(A::CuArray{ComplexF32,3}, B::CuArray{ComplexF32,3}; dims::Int=2)
    # Vectorized implementation that avoids custom CUDA kernels and uses
    # broadcasting + reductions. This form is much simpler for AD backends
    # (Zygote) to trace through on GPU arrays.
    #
    # To match CPU behavior: we slice along `dims`, compute pairwise interference
    # similarity, and return (n_slices_A, n_slices_B, batch).
    
    ndA = ndims(A)
    @assert size(A, dims) > 0 "dims=$dims out of range for array with $(ndA) dimensions"
    
    # Determine which dimensions are: slice_dim (dims), feature_dim, batch_dim
    # For 3D arrays, we have 3 dims. `dims` is the slice dimension.
    # The remaining two are feature and batch. We assume:
    # - The dimension before `dims` (or dim 1 if dims=1) is features
    # - The dimension after `dims` (or last dim if dims=last) is batch
    # Following CPU convention for dims=2: (features, vectors, batch)
    
    if dims == 2
        # Standard case: (features, vectors, batch)
        feature_dim = 1
        batch_dim = 3
    elseif dims == 1
        # (vectors, features, batch) - slice along first dim
        feature_dim = 2
        batch_dim = 3
    elseif dims == 3
        # (features, batch, vectors) - slice along last dim
        feature_dim = 1
        batch_dim = 2
    else
        error("dims must be 1, 2, or 3 for 3D arrays")
    end
    
    # Get sizes
    sz_A = size(A)
    sz_B = size(B)
    M = sz_A[dims]  # number of slices in A
    N = sz_B[dims]  # number of slices in B
    D = sz_A[feature_dim]  # feature dimension
    X = sz_A[batch_dim]  # batch dimension
    
    @assert sz_B[batch_dim] == X "Batch size mismatch"
    @assert sz_B[feature_dim] == D "Feature dimension mismatch"
    
    # Permute arrays to canonical (D, M, X) and (D, N, X) layout for computation
    perm_to_canonical = (feature_dim, dims, batch_dim)
    
    A_canonical = permutedims(A, perm_to_canonical)  # -> (D, M, X)
    B_canonical = permutedims(B, perm_to_canonical)  # -> (D, N, X)
    
    # Separate into real and imaginary parts
    realA = real.(A_canonical)  # (D, M, X)
    imagA = imag.(A_canonical)
    realB = real.(B_canonical)  # (D, N, X)
    imagB = imag.(B_canonical)

    # Reshape for broadcasting to (D, M, N, X)
    realA4 = reshape(realA, D, M, 1, X)
    realB4 = reshape(realB, D, 1, N, X)
    imagA4 = reshape(imagA, D, M, 1, X)
    imagB4 = reshape(imagB, D, 1, N, X)

    # Compute pairwise sums across vectors for each feature d and batch x
    sum_real = realA4 .+ realB4
    sum_imag = imagA4 .+ imagB4

    # Instead of sqrt -> acos -> cos, simplify using trig identity:
    # sim = cos(2 * acos(0.5 * mag)) == 0.5 * mag^2 - 1
    # where mag = clamp(|u+v|, 0, 2). So mag^2 = clamp(|u+v|^2, 0, 4).
    sq = sum_real .^ 2 .+ sum_imag .^ 2
    sq_clamped = clamp.(sq, 0.0f0, 4.0f0)
    sim_per_d = 0.5f0 .* sq_clamped .- 1.0f0  # (D, M, N, X)

    # Average over the D (feature) dimension
    sim_avg = mean(sim_per_d, dims=1)  # (1, M, N, X)
    sim_avg = dropdims(sim_avg, dims=1) # (M, N, X)

    # CPU returns (M, N, X) = (n_slices_A, n_slices_B, batch)
    # Our sim_avg is already (M, N, X), so no permutation needed
    return sim_avg
end

function similarity_outer(A::CuArray{ComplexF32,2}, B::CuArray{ComplexF32,2}; dims::Int=2)
    # Treat 2D inputs as 3D with a singleton batch dimension and delegate
    # to the vectorized 3D implementation.
    # 
    # For dims=2 (default): input is (features, vectors)
    # We add singleton batch dim at the end -> (features, vectors, 1)
    # Output from 3D version is (M, N, 1), we squeeze to (M, N)
    
    if dims == 2
        # (features, vectors) -> (features, vectors, 1)
        A3 = reshape(A, size(A, 1), size(A, 2), 1)
        B3 = reshape(B, size(B, 1), size(B, 2), 1)
        out3 = similarity_outer(A3, B3; dims=2)
    elseif dims == 1
        # (vectors, features) -> (vectors, features, 1) 
        # then slice along dim 1
        A3 = reshape(A, size(A, 1), size(A, 2), 1)
        B3 = reshape(B, size(B, 1), size(B, 2), 1)
        out3 = similarity_outer(A3, B3; dims=1)
    else
        error("dims must be 1 or 2 for 2D arrays")
    end
    
    # out3 has shape (M, N, 1), squeeze the batch dimension
    out2 = dropdims(out3, dims=3)
    # CPU 2D version returns (N, M) due to permutedims(..., (2,1)), so transpose to match
    return permutedims(out2, (2, 1))
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