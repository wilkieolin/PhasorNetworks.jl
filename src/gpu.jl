# ================================================================
# GPU Kernels (KernelAbstractions — backend-agnostic)
# ================================================================

@kernel function gaussian_kernel_ka!(output, times, t::Float32, t_sigma::Float32)
    i = @index(Global)
    x = times[i]
    output[i] = exp(-1.0f0 * ((t - x) / (2.0f0 * t_sigma))^2.0f0)
end

@kernel function raised_cosine_kernel_ka!(output, times, t::Float32, t_sigma::Float32)
    i = @index(Global)
    x = times[i]
    dt = t - x
    half_width = 2.0f0 * t_sigma
    if abs(dt) <= half_width
        output[i] = 0.5f0 * (1.0f0 + cos(3.1415927f0 * dt / half_width))
    else
        output[i] = 0.0f0
    end
end

@kernel function periodic_raised_cosine_kernel_ka!(output, times, t::Float32, t_sigma::Float32, t_period::Float32)
    i = @index(Global)
    x = times[i]
    dt = mod(t - x + t_period/2.0f0, t_period) - t_period/2.0f0
    half_width = 2.0f0 * t_sigma
    if abs(dt) <= half_width
        output[i] = 0.5f0 * (1.0f0 + cos(3.1415927f0 * dt / half_width))
    else
        output[i] = 0.0f0
    end
end

@kernel function scatter_add_kernel_ka!(output, values, indices)
    i = @index(Global)
    index = indices[i]
    value = values[i]
    Atomix.@atomic output[index] += value
end

# Scalar helper functions (plain Julia math, no GPU kernel dependency)

function gaussian_kernel_gpu(x::Float32, t::Float32, t_sigma::Float32)
    return exp(-1.0f0 * ((t - x) / (2.0f0 * t_sigma))^2.0f0)
end

function raised_cosine_kernel_gpu(x::Float32, t::Float32, t_sigma::Float32)
    dt = t - x
    half_width = 2.0f0 * t_sigma
    if abs(dt) <= half_width
        return 0.5f0 * (1.0f0 + cos(3.1415927f0 * dt / half_width))
    else
        return 0.0f0
    end
end

function periodic_raised_cosine_kernel_gpu(x::Float32, t::Float32, t_sigma::Float32, t_period::Float32)
    dt = mod(t - x + t_period/2.0f0, t_period) - t_period/2.0f0
    half_width = 2.0f0 * t_sigma
    if abs(dt) <= half_width
        return 0.5f0 * (1.0f0 + cos(3.1415927f0 * dt / half_width))
    else
        return 0.0f0
    end
end

# ================================================================
# Scatter-Add Operations
# ================================================================

"""
    parallel_scatter_add(indices, values, output_size) -> AbstractGPUArray

Perform parallel scatter-add operation on GPU using atomic operations.
Backend-agnostic via KernelAbstractions.
"""
function parallel_scatter_add(indices::AbstractGPUArray{Int}, values::AbstractGPUArray{T}, output_size::Int) where T
    @assert length(indices) == length(values) "Length of indices and values must match"

    backend = get_backend(values)
    output = KernelAbstractions.zeros(backend, T, output_size)
    scatter_add_kernel_ka!(backend)(output, values, indices; ndrange=length(indices))

    return output
end

"""
    parallel_scatter_add!(output, indices, values)

In-place version of parallel_scatter_add that reuses a pre-allocated output buffer.
"""
function parallel_scatter_add!(output::AbstractGPUArray, indices::AbstractGPUArray{Int}, values::AbstractGPUArray)
    @assert length(indices) == length(values) "Length of indices and values must match"

    backend = get_backend(output)
    scatter_add_kernel_ka!(backend)(output, values, indices; ndrange=length(indices))

    return nothing
end

# ================================================================
# Domain Conversions (GPU)
# ================================================================

"""
    potential_to_phase(potential::AbstractGPUArray, ts::AbstractVector; ...) -> AbstractGPUArray

GPU implementation of potential to phase conversion for neural states.
"""
function potential_to_phase(potential::AbstractGPUArray, ts::AbstractVector; spk_args::SpikingArgs, offset::Real=0.0f0, threshold::Bool=false)
    @assert size(potential)[end] == length(ts) "Time dimensions must match"
    dims = collect(1:ndims(potential))

    current_zeros = gdev(phase_to_potential.(0.0f0, ts, offset=offset, spk_args=spk_args))

    potential = permutedims(potential, reverse(dims))
    arc = angle.(current_zeros) .- angle.(potential)

    phase = Phase.(mod.((arc ./ pi_f32 .+ 1.0f0), 2.0f0) .- 1.0f0)

    if threshold
        silent = abs.(potential) .< spk_args.threshold
        phase[silent] .= Phase(NaN)
    end
    phase = permutedims(phase, reverse(dims))

    return phase
end

function phase_to_train(phases::AbstractGPUArray; spk_args::SpikingArgs, repeats::Int = 1, offset::Real = 0.0f0)
    shape = phases |> size
    indices = collect(CartesianIndices(shape)) |> vec
    times = phase_to_time(phases, spk_args=spk_args, offset=offset) |> vec

    if repeats > 1
        n_t = times |> length
        offsets = gdev(repeat(collect(0:repeats-1) .* spk_args.t_period, inner=n_t))
        times = repeat(times, repeats) .+ offsets
        indices = repeat(indices, repeats)
    end

    train = SpikeTrainGPU(indices, times, shape, offset)
    return train
end

# ================================================================
# Spiking (GPU)
# ================================================================

function parallel_current(stg::SpikeTrainGPU, t::Float32, spk_args::SpikingArgs)
    n = length(stg.times)
    backend = get_backend(stg.times)
    currents = KernelAbstractions.zeros(backend, Float32, n)

    raised_cosine_kernel_ka!(backend)(currents, stg.times, t, Float32(spk_args.t_window); ndrange=n)

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
    spike_current!(output, train::SpikeTrainGPU, t, spk_args, currents_buffer, scatter_buffer)

In-place version of spike_current that reuses pre-allocated buffers.
"""
function spike_current!(output::AbstractGPUArray{Float32}, train::SpikeTrainGPU, t::Float32, spk_args::SpikingArgs,
                        currents_buffer::AbstractGPUArray{Float32}, scatter_buffer::AbstractGPUArray{Float32})
    scale = spk_args.spk_scale
    t_window = Float32(spk_args.t_window)

    backend = get_backend(currents_buffer)
    raised_cosine_kernel_ka!(backend)(currents_buffer, train.times, t, t_window; ndrange=length(train.times))
    currents_buffer .*= scale

    scatter_buffer .= 0.0f0
    parallel_scatter_add!(scatter_buffer, train.linear_indices, currents_buffer)

    output .= reshape(scatter_buffer, train.shape)

    return nothing
end

function bias_current(bias::AbstractGPUArray{<:Complex}, t::Real, t_offset::Real, spk_args::SpikingArgs)
    phase = complex_to_angle(bias)
    mag = abs.(bias)
    return bias_current(phase, mag, t, t_offset, spk_args)
end

function bias_current(phase::AbstractGPUArray{<:Real}, mag::AbstractGPUArray{<:Real}, t::Real, t_offset::Real, spk_args::SpikingArgs)
    times = phase_to_time(phase, spk_args=spk_args, offset=t_offset)
    t_mod = Float32(mod(t, spk_args.t_period))
    n = length(times)
    backend = get_backend(times)
    kernel_vals = KernelAbstractions.zeros(backend, Float32, n)
    periodic_raised_cosine_kernel_ka!(backend)(kernel_vals, times, t_mod, Float32(spk_args.t_window), Float32(spk_args.t_period); ndrange=n)
    bias = reshape(mag .* kernel_vals, size(phase))

    return bias
end

function f32_tspan(tspan::Tuple{<:Real, <:Real})
    return (Float32(tspan[1]), Float32(tspan[2]))
end

function oscillator_bank(u0::AbstractGPUArray, dzdt::Function; tspan::Tuple{<:Float32, <:Float32}, spk_args::SpikingArgs)
    prob = ODEProblem(dzdt, u0, tspan)
    sol = solve(prob, spk_args.solver; spk_args.solver_args...)
    return sol
end

"""
    oscillator_bank(x::SpikeTrainGPU; tspan, spk_args) -> ODESolution

GPU oscillator bank simulation. Solves dz/dt = k*z + spike_current(x, t).
"""
function oscillator_bank(x::SpikeTrainGPU; tspan::Tuple{<:Real, <:Real} = (0.0f0, 10.0f0), spk_args::SpikingArgs)
    tspan = tspan |> f32_tspan

    backend = get_backend(x.times)
    u0 = KernelAbstractions.zeros(backend, ComplexF32, x.shape...)
    dzdt(u, p, t) = resonant_update(u, spk_args.leakage, spk_args.t_period) .+ spike_current(x, t, spk_args)

    sol = oscillator_bank(u0, dzdt, tspan=tspan, spk_args=spk_args)

    return sol
end

"""
    similarity_outer(A::AbstractGPUArray{ComplexF32,3}, B::AbstractGPUArray{ComplexF32,3}; dims=2)

GPU-optimized pairwise similarity computation between sets of complex-valued vectors.
Backend-agnostic via AbstractGPUArray. Compatible with automatic differentiation.

# Arguments
- `A::AbstractGPUArray{ComplexF32,3}`: First set of vectors
- `B::AbstractGPUArray{ComplexF32,3}`: Second set of vectors
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
function similarity_outer(A::AbstractGPUArray{ComplexF32,3}, B::AbstractGPUArray{ComplexF32,3}; dims::Int=2)
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

function similarity_outer(A::AbstractGPUArray{ComplexF32,2}, B::AbstractGPUArray{ComplexF32,2}; dims::Int=2)
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

# Support dispatch when inputs are real-valued GPU arrays by converting to
# ComplexF32 on the device and delegating to the complex-valued implementation.
function similarity_outer(A::AbstractGPUArray{<:Real,3}, B::AbstractGPUArray{<:Real,3}; dims::Int=2)
    # Convert phase angles to complex representation on GPU and delegate to
    # the complex-valued, vectorized implementation.
    A_c = angle_to_complex(A)
    B_c = angle_to_complex(B)
    return similarity_outer(A_c, B_c; dims=dims)
end

function similarity_outer(A::AbstractGPUArray{<:Real,2}, B::AbstractGPUArray{<:Real,2}; dims::Int=2)
    A_c = angle_to_complex(A)
    B_c = angle_to_complex(B)
    return similarity_outer(A_c, B_c; dims=dims)
end