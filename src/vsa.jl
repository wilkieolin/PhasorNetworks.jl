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
    y = complex_to_angle(xz)
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
    y = Phase.(2.0f0 .* rand(Float32, size) .- 1.0f0)
    return y
end

function random_symbols(rng::AbstractRNG, size::Tuple{Vararg{Int}})
    y = Phase.(2.0f0 .* rand(rng, Float32, size) .- 1.0f0)
    return y
end

"""
    orthogonal_codes([rng,] d::Int, n::Int) -> Array{Phase, 2}

Construct `n` mutually (near-)orthogonal `d`-dimensional phasor codes for use
as a `Codebook` (or other VSA classifier) initial state. Output shape is
`(d, n)` — one column per code.

# Construction
The construction is exactly the binding-based one used in phasor VSA: a
single random base symbol `b` is bundled with `n` integer multiples of a
DFT shift symbol `s`,

    code_i = remap_phase( b + (i − 1) · s ),    s[k] = 2 · ((k−1) mod n) / n

Pairwise differences are `(i − j) · s mod 2`, whose per-dimension cosines
sum to zero by DFT orthogonality:

    sim(code_i, code_j) = (1/d) · Σ_k cos(2π · (i − j) · ((k−1) mod n) / n)

This is **exactly zero** when `n` divides `d` (each n-period block
contributes 0). When it does not, the sum has a bounded residual
`≤ 1 / sin(π/n)` from the partial trailing block, giving similarity
`O(n / d)` — still strictly better than the `O(1/√d)` standard deviation
of random codes for small `d`. Diagonal entries (self-similarity) are
exactly 1.

# Errors
Throws when `n > d`: at most `d` mutually orthogonal vectors fit in
`d`-dimensional space.

# Notes
- The base symbol `b` injects per-dimension randomness without changing
  the orthogonality property — useful for breaking symmetry in downstream
  layers.
- For `n = 1` the orthogonality condition is vacuous; returns a single
  random symbol.

See also: [`Codebook`](@ref), [`random_symbols`](@ref), [`v_bind`](@ref).
"""
function orthogonal_codes(rng::AbstractRNG, d::Int, n::Int)
    n > d && throw(ArgumentError(
        "orthogonal_codes requires d ≥ n (cannot fit $n orthogonal vectors in $d dims)"))
    if n == 1
        return random_symbols(rng, (d, 1))
    end
    # Random base offset — does not change pairwise similarity.
    base = 2.0f0 .* rand(rng, Float32, d) .- 1.0f0
    # DFT shift pattern: tile (k mod n)/n across d dimensions.
    shift = Float32[2 * ((k - 1) % n) / n for k in 1:d]
    codes = Matrix{Float32}(undef, d, n)
    for i in 1:n
        @views codes[:, i] .= mod.(Float32(i - 1) .* shift .+ base .+ 1.0f0, 2.0f0) .- 1.0f0
    end
    return Phase.(codes)
end

orthogonal_codes(d::Int, n::Int) = orthogonal_codes(GLOBAL_RNG, d, n)

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
    return Phase(x)
end

function remap_phase(x::AbstractArray)
    ignore_derivatives() do
        x = x .+ 1.0f0
        x = mod.(x, 2.0f0)
        x = x .- 1.0f0
    end
    return Phase.(x)
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

"""
    similarity_outer(x::AbstractArray{<:Complex,3}, y::AbstractArray{<:Complex,3}; dims=2)

Pairwise interference-similarity for 3-D complex arrays on CPU. Permutes to
canonical `(features, n_vectors, batch)` layout and delegates to
[`_similarity_outer_canonical_complex`](@ref) — the same kernel used by the
GPU dispatch — so CPU and GPU paths agree on output shape `(M, N, X)` and
share the closed-form rrule.

(Previous comprehension-based implementation averaged over the batch
dimension instead of features, returning shape `(M, N, D)`. That was
inconsistent with the GPU and CPU real-valued dispatches and is replaced
here.)
"""
function similarity_outer(x::AbstractArray{<:Complex,3}, y::AbstractArray{<:Complex,3}; dims=2)
    if dims == 2
        feature_dim, batch_dim = 1, 3
    elseif dims == 1
        feature_dim, batch_dim = 2, 3
    elseif dims == 3
        feature_dim, batch_dim = 1, 2
    else
        error("dims must be 1, 2, or 3 for 3D arrays")
    end

    sz_x = size(x); sz_y = size(y)
    @assert sz_y[batch_dim]   == sz_x[batch_dim]   "Batch size mismatch"
    @assert sz_y[feature_dim] == sz_x[feature_dim] "Feature dimension mismatch"

    perm = (feature_dim, dims, batch_dim)
    A = permutedims(ComplexF32.(x), perm)
    B = permutedims(ComplexF32.(y), perm)
    return _similarity_outer_canonical_complex(A, B)
end

"""
    similarity_outer(x::AbstractArray{<:Complex,2}, y::AbstractArray{<:Complex,2}; dims=2)

2-D complex variant: wraps as 3-D with a singleton batch axis, delegates to
the 3-D path, then squeezes and transposes to match the CPU real-valued 2-D
convention `(N, M)` (see [`similarity_outer(::AbstractArray{<:Real,2}, ...)`](@ref)).
"""
function similarity_outer(x::AbstractArray{<:Complex,2}, y::AbstractArray{<:Complex,2}; dims=2)
    dims in (1, 2) || error("dims must be 1 or 2 for 2D arrays")
    x3 = reshape(x, size(x, 1), size(x, 2), 1)
    y3 = reshape(y, size(y, 1), size(y, 2), 1)
    out3 = similarity_outer(x3, y3; dims=dims)
    return permutedims(dropdims(out3, dims=3), (2, 1))
end

"""
    _similarity_outer_canonical_complex(A, B) -> AbstractArray{Float32,3}

Vectorized pairwise interference similarity for complex inputs in canonical
layout: `A :: (D, M, X)` and `B :: (D, N, X)`, returning `(M, N, X)` with

    sim[m,n,x] = (1/D) * Σ_d (½ |A[d,m,x] + B[d,n,x]|² − 1)

Used as the shared compute kernel for both the GPU 3-D `similarity_outer`
dispatch (after permutation) and the rrule that gives it a memory-efficient
backward pass.

The clamp `max(|A+B|², 4)` from the original GPU formulation is dropped
because every production caller passes unit-modulus inputs (output of
`angle_to_complex`), for which `|A+B|² ≤ 4` holds exactly. Removing it
preserves the bilinear structure required for a closed-form backward.

# Memory layout

The defining sum is decomposed before any `(D, M, N, X)` tensor is
materialized:

    ½|A_d+B_d|² − 1
       = ½(|A_d|² + |B_d|²) + (Ar_d Br_d + Ai_d Bi_d) − 1

Summing over `d`:

    sim[m,n,x] = (1/D) [ ½·a2[m,x] + ½·b2[n,x] + cross[m,n,x] ] − 1

with `a2 = Σ_d |A|²` shape `(M, X)`, `b2 = Σ_d |B|²` shape `(N, X)`, and
`cross = Σ_d (Ar Br + Ai Bi)` shape `(M, N, X)` computed as two batched
real GEMMs via `NNlib.batched_mul`. Forward peak transient is
`O(D·max(M,N)·X + M·N·X)` instead of `O(D·M·N·X)`. At
`D=64, M=N=784, X=32` this drops the broadcast-time peak from ~19 GiB
to ~270 MiB (≈ 70× reduction), which matters for end-to-end attention on
long sequences. The closed-form rrule below remains correct since it is
derived from the math, not the implementation.
"""
function _similarity_outer_canonical_complex(A::AbstractArray{ComplexF32,3},
                                             B::AbstractArray{ComplexF32,3})
    D, M, X = size(A)
    N = size(B, 2)
    @assert size(B, 1) == D "feature dim mismatch: size(A,1)=$D vs size(B,1)=$(size(B,1))"
    @assert size(B, 3) == X "batch dim mismatch: size(A,3)=$X vs size(B,3)=$(size(B,3))"
    invD = inv(Float32(D))

    Ar = real.(A); Ai = imag.(A)        # (D, M, X) Float32 each
    Br = real.(B); Bi = imag.(B)        # (D, N, X) Float32 each

    a2 = reshape(sum(Ar .^ 2 .+ Ai .^ 2; dims=1), M, 1, X)   # (M, 1, X)
    b2 = reshape(sum(Br .^ 2 .+ Bi .^ 2; dims=1), 1, N, X)   # (1, N, X)

    Ar_T = permutedims(Ar, (2, 1, 3))   # (M, D, X)
    Ai_T = permutedims(Ai, (2, 1, 3))   # (M, D, X)
    cross = batched_mul(Ar_T, Br) .+ batched_mul(Ai_T, Bi)   # (M, N, X)

    return invD .* (0.5f0 .* (a2 .+ b2) .+ cross) .- 1.0f0
end

"""
    rrule(_similarity_outer_canonical_complex, A, B)

Closed-form pullback that avoids materializing the `(D, M, N, X)`
intermediates the broadcast chain would otherwise pin in the tape.

Given output cotangent `ḡ :: (M, N, X)`:

    dA[d,m,x] = (1/D) [ A[d,m,x] · Σ_n ḡ[m,n,x]  +  Σ_n B[d,n,x] · ḡ[m,n,x] ]
    dB[d,n,x] = (1/D) [ B[d,n,x] · Σ_m ḡ[m,n,x]  +  Σ_m A[d,m,x] · ḡ[m,n,x] ]

Each contraction is one batched complex GEMM. Saved tape is just `A` and
`B` (the function inputs). Memory ratio versus the broadcast-traced
backward is ≈ `(M+N) / (k · M · N)`; for `M=N=L` and ~5 saved
intermediates this is `4/(5L)` (≈ 1/160 at `L = 128`).
"""
function ChainRulesCore.rrule(::typeof(_similarity_outer_canonical_complex),
                              A::AbstractArray{ComplexF32,3},
                              B::AbstractArray{ComplexF32,3})
    out = _similarity_outer_canonical_complex(A, B)
    D, M, X = size(A)
    N = size(B, 2)
    invD = inv(Float32(D))

    function _similarity_outer_canonical_complex_pullback(ḡ_)
        ḡ = unthunk(ḡ_)
        # Lift ḡ to ComplexF32 once so batched_mul can contract against the
        # complex inputs; the (M,N,X) transient is freed at the end of bwd.
        ḡc = ComplexF32.(ḡ)

        g_row = reshape(sum(ḡ, dims=2), 1, M, X)   # Σ_n ḡ
        g_col = reshape(sum(ḡ, dims=1), 1, N, X)   # Σ_m ḡ

        # Σ_n B[d,n,x] · ḡ[m,n,x]  → (D, M, X)
        AB_term = batched_mul(B, permutedims(ḡc, (2, 1, 3)))
        # Σ_m A[d,m,x] · ḡ[m,n,x]  → (D, N, X)
        BA_term = batched_mul(A, ḡc)

        dA = invD .* (A .* g_row .+ AB_term)
        dB = invD .* (B .* g_col .+ BA_term)
        return (NoTangent(), dA, dB)
    end
    return out, _similarity_outer_canonical_complex_pullback
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