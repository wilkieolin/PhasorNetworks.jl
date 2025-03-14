using ComponentArrays, SciMLSensitivity, DifferentialEquations, Lux
using Random: AbstractRNG
using Lux: glorot_uniform, truncated_normal
using LinearAlgebra: diagind, I
import LuxLib: dropout

include("vsa.jl")

LuxParams = Union{NamedTuple, ComponentArray, SubArray}

struct MakeSpiking <: Lux.AbstractLuxLayer
    spk_args::SpikingArgs
    repeats::Int
    tspan::Tuple{<:Real, <:Real}
    offset::Real
end

function MakeSpiking(spk_args::SpikingArgs, repeats::Int)
    return MakeSpiking(spk_args, repeats, (0.0, spk_args.t_period * repeats), 0.0)
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

function dropout(rng::AbstractRNG, x::SpikingCall, p::T, training, invp::T, dims) where {T}
    train = x.train
    n_s = length(train.indices)

    keep_inds = rand(rng, Float32, (n_s)) .>= p
    new_inds = train.indices[keep_inds]
    new_tms = train.times[keep_inds]
    new_train = SpikeTrain(new_inds, new_tms, train.shape, train.offset)
    new_call = SpikingCall(new_train, x.spk_args, x.t_span)

    return new_call, (), rng
end

###
### Phasor Dense definitions
###
struct PhasorDense <: Lux.AbstractLuxLayer
    shape::Tuple{<:Int, <:Int}
    in_dims::Int
    out_dims::Int
    init_weight
    init_bias_real
    init_bias_imag
    return_solution::Bool

    PhasorDense(shape, 
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

function PhasorDense(W::AbstractMatrix, b_real::AbstractVecOrMat, b_imag::AbstractVecOrMat; return_solution=false)
    return PhasorDense(size(W), size(W,2), size(W,1), () -> copy(W), () -> copy(b_real), () -> copy(b_imag), return_solution)
end

function PhasorDense(W::AbstractMatrix; return_solution::Bool, phase_bias::Bool=true)
    if phase_bias
        b_real = ones(Float32, axes(W,1))
    else
        b_real = zeros(Float32, axes(W,1))
    end
    b_imag = zeros(Float32, axes(W,1))
    return PhasorDense(W, b_real, b_imag, return_solution=return_solution)
end

function PhasorDense((in, out)::Pair{<:Integer, <:Integer};
                init = variance_scaling,
                return_solution::Bool = false,
                phase_bias::Bool = true,)

    if phase_bias
        layer = PhasorDense((in, out), variance_scaling, () -> ones(Float32, out), () -> zeros(Float32, out), return_solution)
    else
        layer = PhasorDense((in, out), variance_scaling, () -> zeros(Float32, out), () -> zeros(Float32, out), return_solution)
    end

    return layer
end

function Lux.initialparameters(rng::AbstractRNG, layer::PhasorDense)
    params = (weight = layer.init_weight(rng, layer.out_dims, layer.in_dims), bias_real = layer.init_bias_real(), bias_imag = layer.init_bias_imag())
end

# Calls

function (a::PhasorDense)(x::AbstractVecOrMat, params::LuxParams, state::NamedTuple)
    y = v_bundle_project(x, params.weight, params.bias_real .+ 1im .* params.bias_imag)
    return y, state
end

function (a::PhasorDense)(x::SpikingCall, params::LuxParams, state::NamedTuple)
    y = v_bundle_project(x, params.weight, params.bias_real .+ 1im .* params.bias_imag, return_solution=a.return_solution)
    return y, state
end

function (a::PhasorDense)(x::CurrentCall, params::LuxParams, state::NamedTuple)
    y = v_bundle_project(x, params, return_solution = a.return_solution)
    return y, state
end

function Base.show(io::IO, l::PhasorDense)
    print(io, "PhasorDense(", l.shape)
    print(io, ")")
end

###
### Layer which resonates with incoming input currents - mainly with one input and weakly with others
###
struct PhasorResonant <: Lux.AbstractLuxLayer
    shape::Int
    init_weight
    return_solution::Bool
    static::Bool
end

function PhasorResonant(n::Int, spk_args::SpikingArgs, return_solution::Bool = true, static::Bool = true)
    if static
        init_w = () -> Matrix(ones(Float32, 1) .* I(n))
    else
        init_w = rng -> square_variance(rng, n)
    end
        
    return PhasorResonant(n, init_w, return_solution, static)
end

function Lux.initialparameters(rng::AbstractRNG, layer::PhasorResonant)
    if layer.static
        params = NamedTuple()
    else
        params = (weight = layer.init_weight(rng))
    end
end

# Calls

function (a::PhasorResonant)(x::CurrentCall, params::LuxParams, state::NamedTuple)
    if a.static
        y = v_bundle_project(x.current, a.init_weight(), zeros(ComplexF32, (a.shape)), spk_args=x.spk_args, tspan = x.t_span, return_solution = a.return_solution)
    else    
        y = v_bundle_project(x.current, params, spk_args=x.spk_args, tspan = x.t_span, return_solution = a.return_solution)
    end

    return y, state
end

function (a::PhasorResonant)(x::SpikingCall, params::LuxParams)
    if a.static
        y = v_bundle_project(x, a.init_weight(), zeros(ComplexF32, (a.shape)), return_solution = a.return_solution)
    else
        y = v_bundle_project(x, params, spk_args = x.spk_args, return_solution = a.return_solution)
    end

    return y, state
end

"""
Phasor QKV Attention
"""

function attend(q::AbstractArray{<:Real, 3}, k::AbstractArray{<:Real, 3}, v::AbstractArray{<:Real, 3})
    #compute qk scores
    #produces (1 b qt kt)
    scores = similarity_outer(q, k, dims=2)
    #do complex-domain matrix multiply of values by scores (b kt v)
    v = angle_to_complex(v)
    #multiply each value by the scores across batch
    #(b kt v) * (1 b qt kt) ... (v kt) * (kt qt) over b
    output = stack([v[i,:,:]' * scores[i,1,:,:] for i in axes(v, 1)])
    output = complex_to_angle(output)
    return output
end

function attend(q::SpikeTrain, k::SpikeTrain, v::SpikeTrain; spk_args::SpikingArgs, tspan::Tuple{<:Real, <:Real}=(0.0, 10.0), return_solution::Bool = false)
    #compute the similarity between the spike trains
    #produces [q k][1 1 time]
    scores = similarity_outer(q, k, dims=2)
    #convert the values to potentials
    values = oscillator_bank(v, tspan=tspan, spk_args=spk_args)
    #multiply by the scores found at each time step
    output_u = stack([values[:,:,b,t] * scores[1,1,t,:,:]' for b in axes(v_u, 3), t in axes(v_u,4)])
    if return_solution 
        return output_u 
    end
    output = find_spikes_rf(output_u, values.t, spk_args)
    return output
end

struct PhasorAttention{M<:AbstractArray, B} <: Lux.AbstractLuxLayer
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
struct PhasorSA{M<:AbstractArray, B} <: Lux.AbstractLuxLayer
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

struct TrackOutput{L<:Lux.AbstractLuxLayer} <: Lux.AbstractLuxLayer
    layer::L
end

# Forward parameter initialization to inner layer
Lux.initialparameters(rng::AbstractRNG, t::TrackOutput) = 
    (layer=Lux.initialparameters(rng, t.layer),)

# Forward state initialization and add output tracking
function Lux.initialstates(rng::AbstractRNG, t::TrackOutput)
    st_layer = Lux.initialstates(rng, t.layer)
    return merge(st_layer, (outputs=(),))
end

function (t::TrackOutput)(x, ps, st)
    y, st_layer = Lux.apply(t.layer, x, ps.layer, st)
    new_st = merge(st_layer, (outputs=(st.outputs..., y),))
    return y, new_st
end

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