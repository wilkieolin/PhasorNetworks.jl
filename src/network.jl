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

function (a::PhasorDense)(x::AbstractArray, params::LuxParams, state::NamedTuple)
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
Residual blocks
"""

struct ResidualBlock <: LuxCore.AbstractLuxContainerLayer{(:ff,)}
    ff
end

function ResidualBlock(dimensions::Tuple{Vararg{Int}};)
    @assert length(dimensions) >= 2 "Must have at least 1 layer"
    #construct a Phasor MLP based on the given dimensions
    pairs = [dimensions[i] => dimensions[i+1] for i in 1:length(dimensions) - 1]
    layers = [PhasorDense(pair) for pair in pairs]
    ff = Chain(layers...)

    return ResidualBlock(ff)
end

function (rb::ResidualBlock)(x, ps, st)
    # MLP path
    ff_out, st_ff = rb.ff(x, ps.ff, st.ff)
    x = v_bind(x, ff_out)
    
    return x, st_ff
end

"""
Phasor QKV Attention
"""

function attend(q::AbstractArray{<:Real, 3}, k::AbstractArray{<:Real, 3}, v::AbstractArray{<:Real, 3}; scale::AbstractArray=[1.0f0,])
    #compute qk scores
    #produces (qt kt b)
    d_k = size(k,2)
    scores = exp.(scale .* similarity_outer(q, k, dims=2)) ./ d_k
    #do complex-domain matrix multiply of values by scores (kt v b)
    v = angle_to_complex(v)
    #multiply each value by the scores across batch
    #(v kt b) * (kt qt b) ... (v kt) * (kt qt) over b
    output = batched_mul(v, scores)
    output = complex_to_angle(output)
    return output, scores
end

function score_scale(potential::CuArray{<:Complex,3}, scores::CuArray{<:Real,3}; d_k::Int, scale::AbstractArray=[1.0f0,])
    @assert size(potential, 3) == size(scores,3) "Batch dimensions of inputs must match"

    scores = permutedims(scores, (2,1,3))
    d_k = size(scores,1)
    scores = exp.(scale .* scores) ./ d_k
    scaled = batched_mul(potential, scores)
    return scaled
end

function attend(q::SpikingTypes, k::SpikingTypes, v::SpikingTypes; spk_args::SpikingArgs, tspan::Tuple{<:Real, <:Real}=(0.0, 10.0), return_solution::Bool = false, scale::AbstractArray=[1.0f0,])
    #compute the similarity between the spike trains
    #produces [time][b qt kt]
    scores = similarity_outer(q, k, spk_args=spk_args, tspan=tspan)
    #convert the values to potentials
    d_k = size(k)[2]
    values = oscillator_bank(v, tspan=tspan, spk_args=spk_args)
    #multiply by the scores found at each time step
    output_u = score_scale.(values.u, scores, scale=scale, d_k=d_k)
    if return_solution 
        return output_u 
    end

    output = solution_to_train(output_u, values.t, tspan, spk_args=spk_args, offset=v.offset)
    return output, scores
end

struct PhasorAttention <: Lux.AbstractLuxLayer
    init_scale::Real
end

function PhasorAttention()
    return PhasorAttention(1.0f0)
end

function Lux.initialparameters(rng::AbstractRNG, attention::PhasorAttention)
    params = (scale = [attention.init_scale,],)
end

function (a::PhasorAttention)(q::AbstractArray, k::AbstractArray, v::AbstractArray, ps::LuxParams, st::NamedTuple)
    result, scores = attend(q, k, v, scale=ps.scale)

    return result, (scores=scores,)
end

identity_layer = Chain(x -> x,)

struct SingleHeadAttention <: LuxCore.AbstractLuxContainerLayer{(:q_proj, :k_proj, :v_proj, :attention, :out_proj)}
    q_proj
    k_proj
    v_proj
    attention
    out_proj
end

function SingleHeadAttention(d_input::Int, d_model::Int; init=variance_scaling, kwargs...)
    default_model = () -> Chain(ResidualBlock((d_input, d_model)))

    q_proj = get(kwargs, :q_proj, default_model())
    k_proj = get(kwargs, :k_proj, default_model())
    v_proj = get(kwargs, :v_proj, default_model())
    scale = get(kwargs, :scale, 1.0f0)
    attention = get(kwargs, :attention, PhasorAttention(scale))
    out_proj = get(kwargs, :out_proj, PhasorDense(d_model => d_input; init))
    

    SingleHeadAttention(
        q_proj,  # Query
        k_proj,  # Key
        v_proj,  # Value
        attention, # Attention mechanism
        out_proj,   # Output
    )
end

function (m::SingleHeadAttention)(q, kv, ps, st)
    q, _ = m.q_proj(q, ps.q_proj, st.q_proj)
    k, _ = m.k_proj(kv, ps.k_proj, st.k_proj)
    v, _ = m.v_proj(kv, ps.v_proj, st.v_proj)
    
    # Single-head attention (nheads=1)
    attn_out, scores = m.attention(q, k, v, ps.attention, st.attention)
    output = m.out_proj(attn_out, ps.out_proj, st.out_proj)[1]
    
    return output, (scores = scores,)
end

struct SingleHeadCABlock <: LuxCore.AbstractLuxContainerLayer{(:attn, :q_norm, :kv_norm, :ff_norm, :ff)}
    attn::SingleHeadAttention
    q_norm
    kv_norm
    ff_norm
    ff
end

function SingleHeadCABlock(d_input::Int, d_model::Int, n_q::Int, n_kv::Int; dropout::Real=0.1, kwargs...)
    SingleHeadCABlock(
        SingleHeadAttention(d_input, d_model; kwargs...),
        LayerNorm((d_model, n_q)),
        LayerNorm((d_model, n_kv)),
        LayerNorm((d_model, n_q)),
        Chain(PhasorDense(d_input => d_model),
            Dropout(dropout),
            PhasorDense(d_model => d_input)),
    )
end

function (tb::SingleHeadCABlock)(q, kv, mask, ps, st)
    # Attention path
    norm_q = tb.q_norm(q, ps.q_norm, st.q_norm)[1]
    norm_kv = tb.kv_norm(kv, ps.kv_norm, st.kv_norm)[1]
    attn_out, st_attn = tb.attn(q, kv, ps.attn, st.attn)
    x = v_bind(q, attn_out)
    
    # Feed-forward path
    norm_x = tb.ff_norm(x, ps.ff_norm, st.ff_norm)[1]
    ff_out, st_ff = tb.ff(x, ps.ff, st.ff)
    x = v_bind(x, ff_out)
    
    return x, merge(st_attn, st_ff)
end

"""
Training loops/primitives
"""
@kwdef mutable struct Args
    η::Float64 = 3e-4       ## learning rate
    batchsize::Int = 256    ## batch size
    epochs::Int = 10        ## number of epochs
    use_cuda::Bool = true   ## use gpu (if cuda available)
    rng::Xoshiro = Xoshiro(42) ## global rng
end

function train(model, ps, st, train_loader, loss, args; verbose::Bool = false)
    if CUDA.functional() && args.use_cuda
       @info "Training on CUDA GPU"
       #CUDA.allowscalar(false)
       device = gpu_device()
   else
       @info "Training on CPU"
       device = cpu_device()
   end

   ## Optimizer
   opt_state = Optimisers.setup(Adam(args.η), ps)
   losses = []

   ## Training
   for epoch in 1:args.epochs
       for (x, y) in train_loader
           x = x |> device
           y = y |> device
           
           lf = p -> loss(x, y, model, p, st)
           lossval, gs = withgradient(lf, ps)
           if verbose
               println(reduce(*, ["Epoch ", string(epoch), " loss: ", string(lossval)]))
           end
           append!(losses, lossval)
           opt_state, ps = Optimisers.update(opt_state, ps, gs[1]) ## update parameters
       end
   end

   return losses, ps, st
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