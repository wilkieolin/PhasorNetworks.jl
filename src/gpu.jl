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

"""
    ssm_phases_to_train(phases::CuArray{<:Phase, 3}; spk_args) -> SpikeTrainGPU

GPU dispatch. Mirrors the CPU `ssm_phases_to_train` in `src/ssm.jl:719`:
each `(c, l, b)` element produces one spike at time
`(l-1)·t_period + phase_to_time(phases[c, l, b])`, all spikes share
shape `(C, B)`.

Implementation note: the CPU body iterates with explicit `for l in 1:L`
loops and scalar `all_times[idx] = ...` writes, which are forbidden on
CuArrays. The GPU rewrite fuses everything into:
  1) `phase_to_time` broadcast over the (C, L, B) tensor,
  2) an `l`-indexed offset added via broadcast against a `(1, L, 1)` GPU
     constant,
  3) `permutedims (1, 3, 2)` + `vec` to flatten so the per-spike order
     matches the CPU version (outer loop `l`, inner column-major over
     `(c, b)`).
The CartesianIndex pattern stays on CPU (cheap, `C·B·L` integers) — the
existing `phase_to_train(::CuArray, ...)` follows the same convention.

# TODO(ka-migration)
Migrate to `AbstractGPUArray` dispatch + `KernelAbstractions.zeros(get_backend(phases), …)`
for `cu(reshape(...))` to make this work on OneAPI / AMD / Metal.
"""
function ssm_phases_to_train(phases::CuArray{<:Phase, 3}; spk_args::SpikingArgs)
    C, L, B = size(phases)
    shape   = (C, B)
    period  = spk_args.t_period

    # Per-element time within one period (CuArray, (C, L, B)).
    times_within = phase_to_time(phases, spk_args=spk_args, offset=0.0f0)

    # `l`-dependent offset, broadcast across (C, B).
    l_offsets = cu(reshape(Float32.((0:L-1) .* period), 1, L, 1))
    times_3d  = times_within .+ l_offsets

    # Column-major flatten matching the CPU iteration order: l outermost,
    # then (c, b) inner.
    times_perm = permutedims(times_3d, (1, 3, 2))   # (C, B, L)
    all_times  = vec(times_perm)                     # CuArray, length C·B·L

    # Spike indices share the same (c, b) pattern across all `l`.
    base_indices = vec(CartesianIndices((C, B)))     # CPU, length C·B
    all_indices  = repeat(base_indices, L)            # CPU, length C·B·L

    return SpikeTrainGPU(all_indices, all_times, shape, 0.0f0)
end

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

GPU pairwise interference-similarity between two batches of complex vectors.
Backend-agnostic via `AbstractGPUArray` (CUDA, OneAPI, and any other backend
that implements the GPUArraysCore interface). Permutes inputs to canonical
`(features, n_vectors, batch)` layout and delegates to
[`_similarity_outer_canonical_complex`](@ref) — the shared kernel that
carries a closed-form `rrule` so the backward pass avoids the
`(features, M, N, batch)` tape Zygote would otherwise hold.

# Arguments
- `A::AbstractGPUArray{ComplexF32,3}`: First set of vectors
- `B::AbstractGPUArray{ComplexF32,3}`: Second set of vectors
- `dims::Int=2`: Dimension along which to slice for pairwise comparison
  (the other two dims are inferred as features and batch)

# Returns
Output shape `(n_vectors_A, n_vectors_B, batch)`; for `dims=2` inputs of shape
`(features, n_vectors, batch)` this is the natural attention-score layout.

See also: [`similarity_outer`](@ref) for CPU dispatches in `vsa.jl`,
and [`_similarity_outer_canonical_complex`](@ref) for the shared kernel.
"""
function similarity_outer(A::AbstractGPUArray{ComplexF32,3}, B::AbstractGPUArray{ComplexF32,3}; dims::Int=2)
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

function similarity_outer(A::AbstractGPUArray{ComplexF32,2}, B::AbstractGPUArray{ComplexF32,2}; dims::Int=2)
    # Reuse the 3D path with a singleton batch axis. Final transpose matches
    # the CPU 2D real convention (vsa.jl:408), which returns (N, M).
    dims in (1, 2) || error("dims must be 1 or 2 for 2D arrays")
    A3 = reshape(A, size(A, 1), size(A, 2), 1)
    B3 = reshape(B, size(B, 1), size(B, 2), 1)
    out3 = similarity_outer(A3, B3; dims=dims)
    return permutedims(dropdims(out3, dims=3), (2, 1))
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