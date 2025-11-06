include("spiking.jl")

"""
    v_bind(x::AbstractArray; dims) -> AbstractArray

Bind vectors in a Vector Symbolic Architecture (VSA) by summing phases along specified dimensions.
This operation preserves the structure while combining information from multiple vectors.

# Arguments
- `x::AbstractArray`: Input array of phase values
- `dims`: Dimensions along which to perform binding

Returns the bound phase values remapped to [-1, 1].
"""
function v_bind(x::AbstractArray; dims)
    bz = sum(x, dims = dims)
    y = remap_phase(bz)
    return y
end

function v_bind(x::AbstractArray, y::AbstractArray)
    y = remap_phase(x .+ y)
    return y
end

function v_bind(x::Tuple{Vararg{AbstractArray}}; dims=1)
    x = cat((x...), dims=dims)
    return v_bind(x, dims=dims)
end

function v_bind(x::SpikingCall, y::SpikingCall; return_solution::Bool = false, unbind::Bool=false, automatch::Bool=true)
    output = v_bind(x.train, y.train; 
                tspan=x.t_span, 
                spk_args=x.spk_args,
                unbind=unbind,
                automatch=automatch,
                return_solution=return_solution)
    
    if return_solution
        return output
    end

    next_call = SpikingCall(output, x.spk_args, x.t_span)
    return next_call
end

function v_bind(x::SpikingTypes, y::SpikingTypes; tspan::Tuple{<:Real, <:Real} = (0.0f0, 10.0f0), spk_args::SpikingArgs, return_solution::Bool = false, unbind::Bool=false, automatch::Bool=true)
    if !automatch
        if check_offsets(x::SpikingTypes, y::SpikingTypes) @warn "Offsets between spike trains do not match - may not produce desired phases" end
    else
        x, y = match_offsets(x, y)
    end

    #set up functions to define the neuron's differential equations
    k = neuron_constant(spk_args)

    #get the number of batches & output neurons
    output_shape = x.shape

    #find the complex state induced by the spikes
    sol_x = oscillator_bank(x, tspan=tspan, spk_args=spk_args)
    sol_y = oscillator_bank(y, tspan=tspan, spk_args=spk_args)
    
    #create a reference oscillator to generate complex values for each moment in time
    u_ref = t -> phase_to_potential(0.0f0, t, offset = x.offset, spk_args = spk_args)

    #find the first chord
    chord_x = t -> sol_x(t)
    #find the second chord
    if unbind
        chord_y = t -> sol_x(t) .* conj.((sol_y(t) .- u_ref(t))) .* u_ref(t)
    else
        chord_y = t -> sol_x(t) .* (sol_y(t) .- u_ref(t)) .* conj(u_ref(t))
    end

    sol_output = t -> chord_x(t) .+ chord_y(t)
    
    if return_solution
        return sol_output
    end

    train = solution_to_train(sol_output, tspan, spk_args=spk_args, offset=x.offset)
    return train
end

"""
    v_bundle(x::AbstractArray; dims::Int) -> AbstractArray

Bundle vectors in VSA representation by converting phases to complex numbers,
summing along specified dimensions, and converting back to phase angles.
This operation is used to create superpositions of multiple vectors.

# Arguments
- `x::AbstractArray`: Input array of phase values
- `dims::Int`: Dimension along which to perform bundling

Returns bundled phases representing the superposition of input vectors.
"""
function v_bundle(x::AbstractArray; dims::Int)
    xz = angle_to_complex(x)
    bz = sum(xz, dims = dims)
    y = complex_to_angle(bz)
    return y
end

function v_bundle(x::Tuple{Vararg{AbstractArray}}; dims=1)
    x = cat((x...), dims=dims)
    return v_bundle(x, dims=dims)
end

function v_bundle(x::SpikingCall; dims::Int)
    train = v_bundle(x.train, dims=dims, tspan=x.t_span, spk_args=x.spk_args)
    next_call = SpikingCall(train, x.spk_args, x.t_span)
    return next_call
end

function v_bundle(x::SpikingTypes; dims::Int, tspan::Tuple{<:Real, <:Real} = (0.0f0, 10.0f0), spk_args::SpikingArgs, return_solution::Bool=false)
    #let compartments resonate in sync with inputs
    sol = oscillator_bank(x, tspan=tspan, spk_args=spk_args)
    tbase = sol.t
    #combine the potentials (interfere) along the bundling axis
    f_sol = x -> sum(normalize_potential.(sol(x)), dims=dims)

    if return_solution
        return f_sol
    end
    
    out_train = solution_to_train(f_sol, tspan, spk_args=spk_args, offset=x.offset)
    return out_train
end

"""
    v_bundle_project(x::AbstractArray, w::AbstractMatrix, b::AbstractVecOrMat) -> AbstractArray

Project bundled vectors through a linear transformation followed by a soft angle conversion.
Used in neural network layers to transform VSA representations while maintaining phase-based encoding.

# Arguments
- `x::AbstractArray`: Input array of phase values
- `w::AbstractMatrix`: Weight matrix for linear transformation
- `b::AbstractVecOrMat`: Bias term

Returns transformed phases using a soft angle conversion for stable gradients.
"""
function v_bundle_project(x::AbstractArray, w::AbstractMatrix, b::AbstractVecOrMat)
    xz = batched_mul(w, angle_to_complex(x)) .+ b
    #y = complex_to_angle(xz)
    y = soft_angle(xz, 0.01f0, 0.1f0)
    return y
end

function v_bundle_project(x::SpikingCall, w::AbstractMatrix, b::AbstractVecOrMat; return_solution::Bool=false)
    sol = oscillator_bank(x.train, w, b, tspan=x.t_span, spk_args=x.spk_args)
    if return_solution
        return sol
    end

    train = solution_to_train(sol, x.t_span, spk_args=x.spk_args, offset=x.train.offset)
    next_call = SpikingCall(train, x.spk_args, x.t_span)
    return next_call
end

function v_bundle_project(x::CurrentCall, w::AbstractMatrix, b::AbstractVecOrMat; return_solution::Bool=false)
    sol = oscillator_bank(x.current, w, b, tspan=x.t_span, spk_args=x.spk_args)
    if return_solution
        return sol
    end
    
    train = solution_to_train(sol, x.tspan, spk_args=x.spk_args, offset=x.offset)
    next_call = SpikingCall(train, x.spk_args, x.t_span)
    return next_call
end

function v_bundle_project(x::CurrentCall, params; return_solution::Bool=false)
    sol = oscillator_bank(x.current, params, tspan=x.t_span, spk_args=x.spk_args)
    if return_solution
        return sol
    end
    
    train = solution_to_train(sol, x.tspan, spk_args=x.spk_args, offset=x.offset)
    next_call = SpikingCall(train, x.spk_args, x.t_span)
    return next_call
end

"""
    chance_level(nd::Int, samples::Int) -> Float32

Calculate the expected standard deviation of similarities between random VSA symbols.
This function helps in determining the statistical significance of similarity measurements.

# Arguments
- `nd::Int`: Number of dimensions for the VSA symbols
- `samples::Int`: Number of random samples to generate

Returns the standard deviation of similarities between random symbols, which represents
the expected variation in similarity scores due to chance.
"""
function chance_level(nd::Int, samples::Int)
    symbol_0 = random_symbols((1, nd))
    symbols = random_symbols((samples, nd))
    sim = similarity_outer(symbol_0, symbols, dims=1) |> vec
    dev = std(sim)

    return dev
end

"""
    random_symbols(size::Tuple{Vararg{Int}}) -> Array{Float32}
    random_symbols(rng::AbstractRNG, size::Tuple{Vararg{Int}}) -> Array{Float32}

Generate random VSA symbols with values uniformly distributed in [-1, 1].
These symbols serve as base vectors for VSA operations.

# Arguments
- `size::Tuple{Vararg{Int}}`: Dimensions of the output array
- `rng::AbstractRNG`: Optional random number generator for reproducibility

Returns an array of random phases suitable for VSA operations.
"""
function random_symbols(size::Tuple{Vararg{Int}})
    y = 2.0f0 .* rand(Float32, size) .- 1.0f0
    return y
end

function random_symbols(rng::AbstractRNG, size::Tuple{Vararg{Int}})
    y = 2.0f0 .* rand(rng, Float32, size) .- 1.0f0
    return y
end

"""
    remap_phase(x::Real) -> Float32
    remap_phase(x::AbstractArray) -> AbstractArray

Remap phase values to the interval [-1, 1] using modular arithmetic.
This function maintains the cyclic nature of phases while keeping them in a consistent range.

# Arguments
- `x`: Phase value(s) to remap

The operation is performed within `ignore_derivatives` to avoid tracking through the modulo operation
in automatic differentiation.

Returns phase values normalized to [-1, 1].
"""
function remap_phase(x::Real)
    ignore_derivatives() do
        x = x + 1.0f0
        x = mod(x, 2.0f0)
        x = x - 1.0f0
    end
    return x
end

function remap_phase(x::AbstractArray)
    ignore_derivatives() do
        x = x .+ 1.0f0
        x = mod.(x, 2.0f0)
        x = x .- 1.0f0
    end
    return x
end

"""
    similarity(x::AbstractArray, y::AbstractArray; dim::Int = 1) -> AbstractArray

Compute the similarity between two arrays of phase values using cosine distance.
The similarity is calculated by taking the cosine of the phase difference and averaging.

# Arguments
- `x::AbstractArray`: First array of phase values
- `y::AbstractArray`: Second array of phase values
- `dim::Int`: Dimension along which to compute similarity (default: 1, use -1 for last dimension)

Returns similarity scores in [-1, 1], where:
- 1 indicates identical phases
- 0 indicates orthogonal phases
- -1 indicates opposite phases
"""
function similarity(x::AbstractArray, y::AbstractArray; dim::Int = 1)
    if dim == -1
        dim = ndims(x)
    end

    dx = cos.(pi_f32 .* (x .- y))
    s = mean(dx, dims = dim)
    s = dropdims(s, dims = dim)
    return s
end

function similarity(x::SpikingTypes, y::SpikingTypes; spk_args::SpikingArgs, tspan::Tuple{<:Real, <:Real}, automatch::Bool=true)
    if !automatch
        if check_offsets(x::SpikingTypes, y::SpikingTypes) @warn "Offsets between spike trains do not match - may not produce desired phases" end
    else
        x, y = match_offsets(x, y)
    end

    sol_x = oscillator_bank(x, tspan = tspan, spk_args = spk_args)
    sol_y = oscillator_bank(y, tspan = tspan, spk_args = spk_args)

    u_x = normalize_potential.(Array(sol_x))
    u_y = normalize_potential.(Array(sol_y))

    interference = abs.(u_x .+ u_y)
    avg_sim = interference_similarity(interference, dim=1)
    return avg_sim
end

"""
    interference_similarity(interference::AbstractArray; dim::Int=-1) -> AbstractArray

Calculate similarity from interference patterns between complex-valued VSA representations.
This function converts interference magnitudes to similarity scores using geometric relationships.

# Arguments
- `interference::AbstractArray`: Array of interference magnitudes (typically |u₁ + u₂|)
- `dim::Int`: Dimension along which to average (default: -1 for last dimension)

# Details
1. Clamps interference magnitudes to [0, 2]
2. Converts to half-angles using arccos
3. Computes similarity using cosine of double angle
4. Averages along specified dimension

Returns similarity scores in [-1, 1] range, averaged over the specified dimension.
"""
function interference_similarity(interference::AbstractArray; dim::Int=-1)
    if dim == -1
        dim = ndims(interference)
    end

    magnitude = clamp.(interference, 0.0f0, 2.0f0)
    half_angle = acos.(0.5f0 .* magnitude)
    sim = cos.(2.0f0 .* half_angle)
    avg_sim = mean(sim, dims=dim)
    avg_sim = dropdims(avg_sim, dims=dim)
    
    return avg_sim
end

function similarity_outer(x::SpikingCall, y::SpikingCall; automatch::Bool=true)
    @assert x.spk_args == y.spk_args "Spiking arguments must be identical to calculate similarity"
    new_span = match_tspans(x.t_span, y.t_span)
    return similarity_outer(x.train, y.train, tspan=new_span, spk_args=x.spk_args, automatch=automatch)
end

function similarity_outer(x::SpikingTypes, y::SpikingTypes; tspan::Tuple{<:Real, <:Real} = (0.0f0, 10.0f0), spk_args::SpikingArgs, automatch::Bool=true)
    # Allow arbitrary dimensionality for spike-train batches. We will slice along
    # the last two dimensions (batch x features) by default in downstream
    # similarity_outer for arrays, so no strict shape assertion is required here.
    if !automatch
        if check_offsets(x::SpikingTypes, y::SpikingTypes) @warn "Offsets between spike trains do not match - may not produce desired phases" end
    else
        x, y = match_offsets(x, y)
    end

    sol_x = oscillator_bank(x, tspan = tspan, spk_args = spk_args)
    sol_y = oscillator_bank(y, tspan = tspan, spk_args = spk_args)

    u_x = normalize_potential.(sol_x.u)
    u_y = normalize_potential.(sol_y.u)
    
    #add up along the slices
    sim = similarity_outer.(u_x, u_y)
    return sim
end

function similarity_outer(x::CurrentCall, y::CurrentCall)
    @assert x.spk_args == y.spk_args "Spiking arguments must be identical to calculate similarity"
    new_span = match_tspans(x.t_span, y.t_span)

    sol_x = oscillator_bank(x)
    sol_y = oscillator_bank(y)

    u_x = normalize_potential.(sol_x.u)
    u_y = normalize_potential.(sol_y.u)
    
    #add up along the slices
    return similarity_outer.(u_x, u_y)
end

function similarity_self(x::AbstractArray; dims)
    return similarity_outer(x, x, dims=dims)
end

"""
    similarity_outer(x::AbstractArray, y::AbstractArray; dims=2) -> AbstractArray

Compute pairwise similarities between slices of two arrays, supporting both real-valued phases
and complex-valued representations.

# Arguments
- `x::AbstractArray`: First array of values
- `y::AbstractArray`: Second array of values
- `dims::Int`: Dimension along which to slice the arrays (default: 2)

# Methods
- For real-valued 3D arrays: Returns similarities with shape (N₁, N₂, B) where N₁,N₂ are slice dimensions and B is batch
- For real-valued 2D arrays: Returns similarities with shape (N₁, N₂) where N₁,N₂ are slice dimensions
- For complex-valued arrays: Uses interference-based similarity with shape (N₁, N₂, B)

Returns a similarity matrix reshaped to maintain batch dimension as the last dimension.
"""
function similarity_outer(x::AbstractArray{<:Real,3}, y::AbstractArray{<:Real,3}; dims=2)
    s = [similarity(xs, ys) for xs in eachslice(x, dims=dims), ys in eachslice(y, dims=dims)]
    #stack and reshape to batch-last
    s = permutedims(stack(s), (2,3,1))
    return s
end

function similarity_outer(x::AbstractArray{<:Real,2}, y::AbstractArray{<:Real,2}; dims=2)
    s = [similarity(xs, ys) for xs in eachslice(x, dims=dims), ys in eachslice(y, dims=dims)]
    #stack and reshape to batch-last
    s = permutedims(stack(s), (2,1))
    return s
end

function similarity_outer(x::AbstractArray{<:Complex}, y::AbstractArray{<:Complex}; dims=2)
    s = [interference_similarity(abs.(xs .+ ys), dim=dims) for xs in eachslice(x, dims=dims), ys in eachslice(y, dims=dims)]
    #stack and reshape to batch-last
    s = permutedims(stack(s), (2,3,1))
    return s
end

#Note - additional definitions for similarity_outer included in gpu.jl

"""
    v_unbind(x::AbstractArray, y::AbstractArray) -> AbstractArray
    v_unbind(x::SpikingTypes, y::SpikingTypes; kwargs...) -> SpikingTypes

Unbind two VSA vectors by subtracting their phases (inverse of binding).
This operation is used to recover bound components.

# Arguments
- For arrays:
  - `x::AbstractArray`: First array of phase values
  - `y::AbstractArray`: Second array of phase values
- For spiking types:
  - `x::SpikingTypes`: First spike train
  - `y::SpikingTypes`: Second spike train
  - `kwargs...`: Additional arguments passed to `v_bind`

The array method performs phase subtraction with remapping to [-1, 1].
The spiking method uses `v_bind` with `unbind=true` for consistent handling.

Returns unbound phases or spike train.
"""
function v_unbind(x::AbstractArray, y::AbstractArray)
    y = remap_phase(x .- y)
    return y
end

function v_unbind(x::SpikingTypes, y::SpikingTypes; kwargs...)
    return v_bind(x, y, unbind=true; kwargs...)
end