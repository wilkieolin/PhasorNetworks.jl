using ComponentArrays, SciMLSensitivity, DifferentialEquations, Lux
using Random: AbstractRNG
using Lux: glorot_uniform, truncated_normal
using LinearAlgebra: diagind

include("vsa.jl")

LuxParams = Union{NamedTuple, ComponentArray}

struct MakeSpiking <: Lux.AbstractExplicitLayer
    spk_args::SpikingArgs
    repeats::Int
    tspan::Tuple{<:Real, <:Real}
    offset::Real
end

function MakeSpiking(spk_args::SpikingArgs, repeats::Int)
    return MakeSpiking(spk_args, repeats, (0.0, 1.0 * repeats), 0.0)
end

function (a::MakeSpiking)(x::AbstractArray, params::LuxParams, state::NamedTuple)
    train = phase_to_train(x, spk_args = a.spk_args, repeats = a.repeats, offset = a.offset)
    call = SpikingCall(train, a.spk_args, a.tspan)
    return call, state
end

function (a::MakeSpiking)(x::ODESolution, params::LuxParams, state::NamedTuple)
    train = solution_to_train(x, a.tspan, spk_args = a.spk_args, offset = 0.0)
    call = SpikingCall(train, a.spk_args, a.tspan)
    return call, state
end


###
### Phasor Dense definitions
###

struct PhasorDense <: Lux.AbstractExplicitLayer
    shape::Tuple{<:Int, <:Int}
    in_dims::Int
    out_dims::Int
    init_weight
    init_bias

    PhasorDense(shape, init_weight, init_bias) = new(shape, shape[1], shape[2], init_weight, init_bias)
end

## Constructors

#instantiate a layer from a passed weight and bias
function PhasorDense(W::AbstractMatrix, b::AbstractVecOrMat)
    return PhasorDense(size(W), () -> copy(W), () -> copy(b))
end

function PhasorDense(W::AbstractMatrix)
    b = ones(ComplexF32, axes(W,1))
    return PhasorDense(W, b)
end

#setup the layer with a shape and initializers
function PhasorDense((in, out)::Pair{<:Integer, <:Integer};
                init = variance_scaling)

    return PhasorDense((in, out), variance_scaling, () -> ones(ComplexF32, out))
end

function Lux.initialparameters(rng::AbstractRNG, layer::PhasorDense)
    params = (weight = layer.init_weight(rng, layer.out_dims, layer.in_dims), bias = layer.init_bias())
end

# Calls

function (a::PhasorDense)(x::AbstractVecOrMat, params::LuxParams, state::NamedTuple)
    y = v_bundle_project(x, params.weight, params.bias)
    return y, state
end

function (a::PhasorDense)(x::SpikingCall, params::LuxParams, state::NamedTuple; return_solution::Bool=false)
    y = v_bundle_project(x, params.weight, params.bias, return_solution=return_solution)
    return y, state
end

function (a::PhasorDense)(x::CurrentCall, params::LuxParams, state::NamedTuple; return_solution::Bool=false)
    y = v_bundle_project(x.current, params.weight, params.bias, x.t_span, x.spk_args, return_solution=return_solution)
    return y, state
end

function Base.show(io::IO, l::PhasorDense)
    print(io, "PhasorDense(", l.shape)
    print(io, ")")
end

###
### Same as PhasorDense, but made with F32 parameters so ComponentArrays doesn't get confused and lead to the gradients getting mixed up
###
struct PhasorDenseF32 <: Lux.AbstractExplicitLayer
    shape::Tuple{<:Int, <:Int}
    in_dims::Int
    out_dims::Int
    init_weight
    init_bias_real
    init_bias_imag
    return_solution::Bool

    PhasorDenseF32(shape, 
                    init_weight, 
                    init_bias_real, 
                    init_bias_imag,
                    return_solution) = 
                        new(shape,
                        shape[1],
                        shape[2],
                        init_weight,
                        init_bias_real,
                        init_bias_imag,
                        return_solution
                        )
end

## Constructors

function PhasorDenseF32(W::AbstractMatrix, b_real::AbstractVecOrMat, b_imag::AbstractVecOrMat; return_solution=false)
    return PhasorDenseF32(size(W), size(W,2), size(W,1), () -> copy(W), () -> copy(b_real), () -> copy(b_imag), return_solution)
end

function PhasorDenseF32(W::AbstractMatrix; return_solution::Bool)
    b_real = ones(Float32, axes(W,1))
    b_imag = zeros(Float32, axes(W,1))
    return PhasorDenseF32(W, b_real, b_imag, return_solution=return_solution)
end

function PhasorDenseF32((in, out)::Pair{<:Integer, <:Integer};
                init = variance_scaling,
                return_solution::Bool = false)

    return PhasorDenseF32((in, out), variance_scaling, () -> ones(Float32, out), () -> zeros(Float32, out), return_solution)
end

function Lux.initialparameters(rng::AbstractRNG, layer::PhasorDenseF32)
    params = (weight = layer.init_weight(rng, layer.out_dims, layer.in_dims), bias_real = layer.init_bias_real(), bias_imag = layer.init_bias_imag())
end

# Calls

function (a::PhasorDenseF32)(x::AbstractVecOrMat, params::LuxParams, state::NamedTuple)
    y = v_bundle_project(x, params.weight, params.bias_real .+ 1im .* params.bias_imag)
    return y, state
end

function (a::PhasorDenseF32)(x::SpikingCall, params::LuxParams, state::NamedTuple)
    y = v_bundle_project(x, params.weight, params.bias_real .+ 1im .* params.bias_imag, return_solution=a.return_solution)
    return y, state
end

function (a::PhasorDenseF32)(x::CurrentCall, params::LuxParams, state::NamedTuple)
    y = v_bundle_project(x.current, params, tspan = x.t_span, spk_args = x.spk_args, return_solution = a.return_solution)    
    return y, state
end

function Base.show(io::IO, l::PhasorDenseF32)
    print(io, "PhasorDenseF32(", l.shape)
    print(io, ")")
end

###
### Layer which resonates with incoming input currents - mainly with one input and weakly with others
###
struct PhasorResonant <: Lux.AbstractExplicitLayer
    shape::Int
    init_weight
    init_leakage
    init_t_period
    return_solution::Bool
end

function PhasorResonant(n::Int, spk_args::SpikingArgs, return_solution::Bool = true)
    init_w = rng -> square_variance(rng, n)
    init_leakage = () -> [spk_args.leakage,]
    init_t_period = () -> [spk_args.t_period,]
    return PhasorResonant(n, init_w, init_leakage, init_t_period, return_solution)
end

function Lux.initialparameters(rng::AbstractRNG, layer::PhasorResonant)
    params = (weight = layer.init_weight(rng), leakage = layer.init_leakage(), t_period = layer.init_t_period())
end

# Calls

function (a::PhasorResonant)(x::CurrentCall, params::LuxParams, state::NamedTuple)
    y = v_bundle_project(x.current, params, tspan = x.t_span, return_solution = a.return_solution)
    return y, state
end

function (a::PhasorResonant)(x::SpikingCall, params::LuxParams)
    y = v_bundle_project(x, params, return_solution = a.return_solution)
    return y, state
end

"""
Phasor QKV Attention
"""

function attend(q::Array{<:Real, 3}, k::Array{<:Real, 3}, v::Array{<:Real, 3})
    #compute qk scores
    #produces (1 b qt kt)
    scores = similarity_outer(q, k, dims=2)
    #do complex-domain matrix multiply of values by scores (v kt b)
    v = angle_to_complex(v)
    #multiply each value by the scores across batch
    #(v kt b) * (1 b qt kt) ... (v kt) * (kt qt) over b
    output = stack([v[:,:,i] * scores[1,i,:,:]' for i in axes(v, 3)])
    output = complex_to_angle(output)
    return output
end

function attend(q::SpikeTrain, k::SpikeTrain, v::SpikeTrain; tspan::Tuple{<:Real, <:Real}=(0.0, 10.0), spk_args::SpikingArgs=default_spk_args(), return_solution::Bool = false)
    #compute the similarity between the spike trains
    #produces [q k][1 1 time]
    scores = similarity_outer(q, k, dims=2)
    #convert the values to potentials
    values = phase_memory(v, tspan=tspan, spk_args=spk_args)
    #multiply by the scores found at each time step
    output_u = stack([values[:,:,b,t] * scores[1,1,t,:,:]' for b in axes(v_u, 3), t in axes(v_u,4)])
    if return_solution 
        return output_u 
    end
    output = find_spikes_rf(output_u, values.t, spk_args)
    return output
end

struct PhasorAttention{M<:AbstractArray, B} <: Lux.AbstractExplicitLayer
    shape::Tuple{<:Int, <:Int}
    in_dims::Int
    out_dims::Int
    init_weight::Function
    init_bias::Function
end

function (a::PhasorAttention)(query::AbstractArray, keyvalue::AbstractArray)
    q = a.query_network(query)
    k = a.key_network(keyvalue)
    v = a.value_network(keyvalue)

    result = attend(q, k, v)

    output = a.output_network(result)

    return output
end


"""
Phasor Self-Attention Module
"""
struct PhasorSA{M<:AbstractArray, B} <: Lux.AbstractExplicitLayer
    shape::Tuple{<:Int, <:Int}
    n_heads::Int
    in_dims::Int
    out_dims::Int
    init_weight::Function
    init_bias::Function

    function PhasorDense(W::M, b::B) where {M<:AbstractArray, B<:AbstractVector}
      new{M,typeof(b)}(size(W), size(W,2), size(W,1), () -> copy(W), () -> copy(b))
    end
end

"""
Other utilities
"""

function variance_scaling(rng::AbstractRNG, shape::Integer...; mode::String = "avg", scale::Real = 0.66)
    fan_in = shape[end]
    fan_out = shape[1]

    if mode == "fan_in"
        scale /= max(1.0, fan_in)
    elseif mode == "fan_out"
        scale /= max(1.0, fan_out)
    else
        scale /= max(1.0, (fan_in + fan_out) / 2.0)
    end

    stddev = sqrt(scale) / 0.87962566103423978
    return truncated_normal(rng, shape..., mean = 0.0, std = stddev)
end

function square_variance(rng::AbstractRNG, shape::Integer; kwargs...)
    weights = variance_scaling(rng, shape, shape; kwargs...)
    weights[diagind(weights)] .= 1.0
    return weights
end