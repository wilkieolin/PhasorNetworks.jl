"""
MakeSpiking - a layer to include in Chains to convert phase tensors
into SpikeTrains
"""
struct MakeSpiking <: Lux.AbstractLuxLayer
    spk_args::SpikingArgs
    repeats::Int
    tspan::Tuple{<:Real, <:Real}
    offset::Real
end

function MakeSpiking(spk_args::SpikingArgs, repeats::Int)
    return MakeSpiking(spk_args, repeats, (0.0f0, spk_args.t_period * repeats), 0.0f0)
end

function (a::MakeSpiking)(x::AbstractArray, params::LuxParams, state::NamedTuple)
    train = phase_to_train(x, spk_args = a.spk_args, repeats = a.repeats, offset = a.offset)
    call = SpikingCall(train, a.spk_args, a.tspan)
    return call, state
end

function (a::MakeSpiking)(x::ODESolution, params::LuxParams, state::NamedTuple)
    train = solution_to_train(x, a.tspan, spk_args = a.spk_args, offset = 0.0f0)
    call = SpikingCall(train, a.spk_args, a.tspan)
    return call, state
end

# Extend Lux.Flatten for SpikeTrain and SpikingCall types, preserving the last dimension (batch)
function (f::Lux.FlattenLayer)(x::SpikeTrain, params::LuxParams, state::NamedTuple)
    original_shape = x.shape
    N_dims = length(original_shape)

    if N_dims == 0
        # Cannot meaningfully flatten a 0-dimensional SpikeTrain for batch processing
        return x, state
    end

    local new_shape::Tuple
    local feature_shape_tuple::Tuple

    if N_dims == 1 # Input is (Batch,), new shape will be (1, Batch)
        batch_size = original_shape[1]
        new_shape = (1, batch_size)
        feature_shape_tuple = () # No feature dimensions to flatten
    else # Input is (F1, ..., Fk, Batch), new shape (F1*...*Fk, Batch)
        batch_size = original_shape[end]
        feature_shape_tuple = original_shape[1:end-1]
        num_features_flat = prod(feature_shape_tuple)
        new_shape = (num_features_flat, batch_size)
    end

    # Convert original indices to new CartesianIndices for the new_shape
    new_indices = map(x.indices) do original_idx_val
        # Ensure original_idx_val is CartesianIndex for original_shape
        ci_orig::CartesianIndex = original_idx_val isa CartesianIndex ? 
                                  original_idx_val : 
                                  CartesianIndices(original_shape)[original_idx_val]
        
        # Batch index is the component corresponding to the last dimension of original_shape
        batch_val = ci_orig[N_dims]

        # Extract feature coordinates as a tuple of integers
        # For N_dims=1, (N_dims - 1) is 0, so ntuple returns ()
        feature_coords_as_tuple = ntuple(d -> ci_orig[d], N_dims - 1)
        
        # Calculate linear index for the (potentially) flattened features
        # If feature_shape_tuple is empty (e.g. original was (Batch,)), linear_feature_idx is 1.
        linear_feature_idx = isempty(feature_shape_tuple) ? 
                             1 : 
                             LinearIndices(feature_shape_tuple)[feature_coords_as_tuple...]
        
        CartesianIndex(linear_feature_idx, batch_val)
    end

    flattened_train = SpikeTrain(new_indices, x.times, new_shape, x.offset)
    return flattened_train, state
end

function (f::Lux.FlattenLayer)(x::SpikeTrainGPU, params::LuxParams, state::NamedTuple)
    original_shape = x.shape
    N_dims = length(original_shape)

    if N_dims == 0
        return x, state
    end

    local new_shape::Tuple
    local feature_shape_tuple::Tuple

    if N_dims == 1 # Input is (Batch,), new shape will be (1, Batch)
        batch_size = original_shape[1]
        new_shape = (1, batch_size)
        feature_shape_tuple = ()
    else # Input is (F1, ..., Fk, Batch), new shape (F1*...*Fk, Batch)
        batch_size = original_shape[end]
        feature_shape_tuple = original_shape[1:end-1]
        num_features_flat = prod(feature_shape_tuple)
        new_shape = (num_features_flat, batch_size)
    end

    # Convert original linear_indices (for N-D shape) to new CartesianIndices (for 2D shape) on CPU
    linear_indices_cpu = Array(x.linear_indices)
    cart_indices_obj_orig = CartesianIndices(original_shape)
    original_cart_indices_cpu = map(li -> cart_indices_obj_orig[li], linear_indices_cpu)

    new_cart_indices_cpu = map(original_cart_indices_cpu) do ci_orig
        batch_val = ci_orig[N_dims]
        feature_coords_as_tuple = ntuple(d -> ci_orig[d], N_dims - 1)
        linear_feature_idx = isempty(feature_shape_tuple) ? 1 : LinearIndices(feature_shape_tuple)[feature_coords_as_tuple...]
        CartesianIndex(linear_feature_idx, batch_val)
    end
    
    # SpikeTrainGPU constructor takes AbstractArray for indices (here, CPU Array of CartesianIndex),
    # and will convert it to CuArray and compute new linear_indices for the new_shape.
    # x.times is already a CuArray.
    flattened_train = SpikeTrainGPU(new_cart_indices_cpu, x.times, new_shape, x.offset)
    return flattened_train, state
end

function (f::Lux.FlattenLayer)(call::SpikingCall, params::LuxParams, state::NamedTuple)
    flattened_train, _ = f(call.train, params, state) # Dispatch to SpikeTrain or SpikeTrainGPU method
    new_call = SpikingCall(flattened_train, call.spk_args, call.t_span)
    return new_call, state
end

"""
Extension of dropout to SpikeTrains
"""
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


"""
    ComplexBias <: LuxCore.AbstractLuxLayer

Layer that adds learnable complex-valued biases to phase networks.

# Fields
- `dims`: Dimensions of the bias terms
- `init_bias`: Function to initialize bias values (default: ones)

# Initialization Options
- `default_bias`: Initialize with ones in complex plane
- `zero_bias`: Initialize with zeros
- Custom initialization function with signature (rng, dims) -> ComplexF32 array

Used as a component in [`PhasorDense`](@ref) and [`PhasorConv`](@ref) layers
to provide phase shifts in the complex plane.
"""
struct ComplexBias <: LuxCore.AbstractLuxLayer
    dims
    init_bias
end

function default_bias(rng::AbstractRNG, dims::Tuple{Vararg{Int}})
    return ones(ComplexF32, dims)
end

function zero_bias(rng::AbstractRNG, dims::Tuple{Vararg{Int}})
    return zeros(ComplexF32, dims)
end

function ComplexBias(dims::Tuple{Vararg{Int}}; init_bias = default_bias)
    if init_bias === nothing
        init_bias = (rng, dims) -> zeros(ComplexF32, dims)
    end

    return ComplexBias(dims, init_bias)
end

function Base.show(io::IO, b::ComplexBias)
    print(io, "ComplexBias($(b.dims))")
end

function (b::ComplexBias)(x::AbstractArray{<:Complex}, params::LuxParams, state::NamedTuple)
    bias_val = params.bias_real .+ 1.0f0im .* params.bias_imag
    return x .+ bias_val, state 
end

function (b::ComplexBias)(params::LuxParams, state::NamedTuple; offset::Real = 0.0f0, spk_args::SpikingArgs)
    bias_val = params.bias_real .+ 1.0f0im .* params.bias_imag
    return t -> bias_current(bias_val, t, offset, spk_args)
end

function Lux.initialparameters(rng::AbstractRNG, bias::ComplexBias)
    bias = bias.init_bias(rng, bias.dims)
    bias_real = real.(bias)
    bias_imag = imag.(bias)
    params = (bias_real = bias_real, bias_imag = bias_imag)
    return params
end

function Lux.initialstates(rng::AbstractRNG, bias::ComplexBias)
    # ComplexBias is stateless by itself, but Lux convention is to return an empty NamedTuple
    # if no specific state is needed.
    return NamedTuple()
end

"""
    PhasorDense <: LuxCore.AbstractLuxContainerLayer{(:layer, :bias)}

A dense (fully-connected) layer that operates on phase/complex-valued inputs.
Combines a standard dense layer with complex bias and phase-based activation.

# Fields
- `layer`: Standard dense layer for linear transformation
- `bias`: Complex-valued bias for phase shift
- `activation`: Function to convert complex values to phases
- `use_bias::Bool`: Whether to apply complex bias
- `return_type::SolutionType`: Output format for spiking inputs (:phase, :potential, :current, or :spiking)

# Layer Operation
1. Converts input phases to complex numbers
2. Applies linear transformation separately to real/imaginary parts
3. Optionally applies complex bias
4. Applies activation function to map back to phases

Supports both direct phase inputs and spiking inputs through ODEs.
"""
struct PhasorDense <: LuxCore.AbstractLuxContainerLayer{(:layer, :bias)}
    layer # the conventional layer used to transform inputs
    bias # the bias in the complex domain used to shift away from the origin
    activation # the activation function which converts complex values to real phases
    use_bias::Bool # apply the layer with the bias if true
    init_leakage::Function # initializer for the values to scale the leakge in spk_args
    init_period::Function # initializer for the values to scale the t_period in spk_args
    trainable_leakage::Bool # can the ODE optimizer access leakage in params
    trainable_period::Bool # can the ODE optimizer access period in params
    return_type::SolutionType # return the full ODE solution from a spiking input
end

function PhasorDense(shape::Pair{<:Integer,<:Integer}, 
                    activation = identity;
                    return_type::SolutionType = SolutionType(:spiking),
                    init_bias = default_bias,
                    use_bias::Bool = true,
                    init_leakage = ones32,
                    init_period = ones32,
                    trainable_leakage::Bool = false,
                    trainable_period::Bool = false,
                    kwargs...)
    layer = Dense(shape, identity; use_bias=false, kwargs...)
    bias = ComplexBias((shape[2],); init_bias = init_bias)
    return PhasorDense(layer, 
                        bias, 
                        activation,
                        use_bias,
                        init_leakage,
                        init_period,
                        trainable_leakage,
                        trainable_period,
                        return_type)
end

function Lux.initialparameters(rng::AbstractRNG, l::PhasorDense)
    ps_layer = Lux.initialparameters(rng, l.layer)
    ps_bias = Lux.initialparameters(rng, l.bias)
    parameters = (layer = ps_layer, bias = ps_bias,)

    n_out = l.layer.out_dims
    if l.trainable_leakage
        ps_leakage = l.init_leakage(rng, n_out,)
        parameters = merge(parameters, (leakage = ps_leakage,))
    end

    if l.trainable_period
        ps_period = l.init_leakage(rng, n_out,)
        parameters = merge(parameters, (period = ps_period,))
    end
    return parameters
end

function Lux.initialstates(rng::AbstractRNG, l::PhasorDense)
    st_layer = Lux.initialstates(rng, l.layer)
    st_bias = Lux.initialstates(rng, l.bias)
    state = (layer = st_layer, bias = st_bias,)

    n_out = l.layer.out_dims
    if !l.trainable_leakage
        st_leakage = l.init_leakage(rng, n_out,)
        state = merge(state, (leakage = st_leakage,))
    end

    if !l.trainable_period
        st_period = l.init_leakage(rng, n_out,)
        state = merge(state, (period = st_period,))
    end
    return state
end

# Calls
function (a::PhasorDense)(x::AbstractArray, params::LuxParams, state::NamedTuple)
    xz = angle_to_complex(x)
    #stateless calls to dense
    y_real, _ = a.layer(real.(xz), params.layer, state.layer)
    y_imag, _ = a.layer(imag.(xz), params.layer, state.layer)
    y = y_real .+ 1.0f0im .* y_imag

    if a.use_bias
        y_biased, st_updated_bias = a.bias(y, params.bias, state.bias)
        y_activated = a.activation(y_biased)
    else
        #passthrough
        st_updated_bias = state.bias
        y_activated = a.activation(y)
    end

    # New state for PhasorDense layer
    st_new = (dense = state.layer, bias = st_updated_bias)
    return y_activated, st_new
end

function (a::PhasorDense)(x::SpikingCall, params::LuxParams, state::NamedTuple)
    current_call = CurrentCall(x)
    return a(current_call, params, state)
end

function (a::PhasorDense)(x::CurrentCall, params::LuxParams, state::NamedTuple)
    #pass the params and dense kernel to the solver
    sol = oscillator_bank(x.current, a, params, state, tspan=x.t_span, spk_args=x.spk_args, use_bias=a.use_bias)
    if a.return_type.type == :phase
        u = unrotate_solution(sol.u, sol.t, spk_args=x.spk_args, offset=x.current.offset)
        y = a.activation(stack(u))
        return y, state
    elseif a.return_type.type == :potential
        return sol, state
    elseif a.return_type.type == :current
        i_fn = t -> potential_to_current(sol(t), spk_args=x.spk_args)
        next_call = CurrentCall(LocalCurrent(i_fn, x.current.shape, x.current.offset + spiking_offset(x.spk_args)),
                                x.spk_args,
                                x.t_span)
        return next_call, state
    else #spiking
        train = solution_to_train(sol, x.t_span, spk_args=x.spk_args, offset=x.current.offset)
        next_call = SpikingCall(train, x.spk_args, x.t_span)
        return next_call, state
    end
end

###
### Convolutional Phasor Layer
###

"""
    PhasorConv <: LuxCore.AbstractLuxContainerLayer{(:layer, :bias)}

Convolutional layer for phase-valued inputs and spiking neural networks.
Implements complex-valued convolution with phase-based activation.

# Fields
- `layer`: Standard convolutional layer for spatial operations
- `bias`: Complex-valued bias terms
- `activation`: Phase activation function
- `use_bias::Bool`: Whether to apply complex bias
- `return_type::SolutionType`: Output format for spiking inputs (:phase, :potential, :current, or :spiking)

# Implementation
- Separates input into real/imaginary components
- Applies convolution separately to each component
- Recombines into complex values
- Optionally applies complex bias and activation

See also: [`PhasorDense`](@ref) for fully-connected equivalent
"""
struct PhasorConv <: LuxCore.AbstractLuxContainerLayer{(:layer, :bias)}
    layer
    bias
    activation
    use_bias::Bool
    return_type::SolutionType
end

function PhasorConv(k::Tuple{Vararg{Integer}}, chs::Pair{<:Integer,<:Integer}, activation = identity; return_type::SolutionType = SolutionType(:spiking), init_bias = default_bias, use_bias::Bool = true, kwargs...)
    #construct the convolutional layer
    layer = Conv(k, chs, identity; use_bias=false, kwargs...)
    bias = ComplexBias(([1 for _ in 1:length(k)]...,chs[2],), init_bias = init_bias)
    return PhasorConv(layer, bias, activation, use_bias, return_type)
end

function Lux.initialstates(rng::AbstractRNG, l::PhasorConv)
    st_layer = Lux.initialstates(rng, l.layer)
    st_bias = Lux.initialstates(rng, l.bias)
    return (layer = st_layer, bias = st_bias)
end

function (pc::PhasorConv)(x::AbstractArray, ps::LuxParams, st::NamedTuple)
    x = angle_to_complex(x)
    x_real = real.(x)
    x_imag = imag.(x)

    y_real_conv, _ = pc.layer(x_real, ps.layer, st.layer)
    y_imag_conv, _ = pc.layer(x_imag, ps.layer, st.layer)
    y = y_real_conv .+ 1.0f0im .* y_imag_conv

    # Apply bias
    if pc.use_bias
        y_biased, st_updated_bias = pc.bias(y, ps.bias, st.bias)
        y_activated = pc.activation(y_biased)
    else
        #passthrough
        st_updated_bias = st.bias
        y_activated = a.activation(y)
    end

    st_new = (layer = st.layer, bias = st_updated_bias)
    return y_activated, st_new
end

function (a::PhasorConv)(x::SpikingCall, params::LuxParams, state::NamedTuple)
    current_call = CurrentCall(x)
    return a(current_call, params, state)
end

function (a::PhasorConv)(x::CurrentCall, params::LuxParams, state::NamedTuple)
    #pass the params and dense kernel to the solver
    sol = oscillator_bank(x.current, a, params, state, tspan=x.t_span, spk_args=x.spk_args, use_bias=a.use_bias)
    if a.return_type.type == :phase
        u = unrotate_solution(sol.u, sol.t, spk_args=x.spk_args, offset=x.current.offset)
        y = a.activation.(u)
        return y, state
    elseif a.return_type.type == :potential
        return sol, state
    elseif a.return_type.type == :current
        i_fn = t -> potential_to_current(sol(t), spk_args=x.spk_args)
        next_call = CurrentCall(LocalCurrent(i_fn, x.current.shape, x.current.offset + spiking_offset(x.spk_args)),
                                x.spk_args,
                                x.t_span)
        return next_call, state
    else #spiking
        train = solution_to_train(sol, x.t_span, spk_args=x.spk_args, offset=x.current.offset)
        next_call = SpikingCall(train, x.spk_args, x.t_span)
        return next_call, state
    end
end

###
### Codebook layer - converts a vector to a value of similarities
###

"""
    Codebook <: LuxCore.AbstractLuxLayer

Layer that accesses a fixed set of phase codes and computes similarities with inputs.
Used for discrete embedding or classification tasks in phase-based networks.

# Fields
- `dims::Pair{<:Int, <:Int}`: Input dimension => Number of codes

# State
- `codes`: Random phase symbols initialized as the codebook
- Codes are fixed after initialization (non-trainable)

# Forward Pass
1. For phase inputs: Computes similarity with all codes
2. For spiking inputs: Converts codes to currents and computes temporal similarity

# Use Cases
- Discrete symbol encoding in Vector Symbolic Architectures
- Classification by similarity to learned phase patterns
- Phase-based memory or lookup mechanisms

See also: [`similarity_outer`](@ref) for similarity computation
"""
struct Codebook <: LuxCore.AbstractLuxLayer
    dims
    Codebook(x::Pair{<:Int, <:Int}) = new(x)
end

function Base.show(io::IO, cb::Codebook)
    print(io, "Codebook($(cb.dims))")
end

function Lux.initialparameters(rng::AbstractRNG, cb::Codebook)
    return NamedTuple()
end

function Lux.initialstates(rng::AbstractRNG, cb::Codebook)
    state = (codes = random_symbols(rng, (cb.dims[1], cb.dims[2])),)
    return state
end

function (cb::Codebook)(x::AbstractArray{<:Real}, params::LuxParams, state::NamedTuple)
    return similarity_outer(x, state.codes), state
end

function (cb::Codebook)(x::SpikingCall, params::LuxParams, state::NamedTuple)
    current_call = CurrentCall(x)
    return cb(current_call, params, state)
end

function (cb::Codebook)(x::CurrentCall, params::LuxParams, state::NamedTuple)
    code_currents = phase_to_current(state.codes, spk_args=x.spk_args, offset=x.current.offset, tspan=x.t_span)
    similarities = similarity_outer(x, code_currents)
    
    return similarities, state
end


"""
    PhasorFixed <: LuxCore.AbstractLuxContainerLayer{(:bias,)}

A dense layer with non-trainable weights stored in state.
The weights are fixed after initialization and not updated during training.

# Fields
- `shape::Pair{<:Integer,<:Integer}`: Input => Output dimensions
- `bias`: Complex-valued bias for phase shift (optional)
- `activation`: Function to convert complex values to phases
- `use_bias::Bool`: Whether to apply complex bias
- `return_type::SolutionType`: Output format for spiking inputs (:phase, :potential, :current, or :spiking)
- `init_weight`: Function to initialize fixed weights

# Constructor
```julia
PhasorFixed(shape::Pair{<:Integer,<:Integer}, activation=identity;
    init_weight=nothing,  # Custom weight initialization, e.g. identity_init
    init_bias=default_bias,
    use_bias=false,
    return_type=:phase)  # :phase, :potential, :current, or :spiking
```

# Weight Initialization
- `init_weight=nothing`: Uses Glorot uniform initialization
- `init_weight=identity_init`: Creates identity matrix (requires square shape)
- Custom function: `(rng, in_dim, out_dim) -> weight_matrix`

# Forward Pass
1. Converts input phases to complex numbers
2. Applies fixed linear transformation separately to real/imaginary parts
3. Optionally applies complex bias
4. Applies activation function to map back to phases

# Use Cases
- Fixed pattern detection
- Identity or permutation mapping
- Non-trainable projection layers
- Frozen layers in transfer learning

See also: [`PhasorDense`](@ref) for trainable version
"""

struct PhasorFixed <: LuxCore.AbstractLuxContainerLayer{(:layer, :bias,)}
    layer # the conventional layer used to transform inputs (fixed weights)
    bias # the bias in the complex domain used to shift away from the origin
    activation # the activation function which converts complex values to real phases
    use_bias::Bool # apply the layer with the bias if true
    return_type::SolutionType # the type of solution to return from a spiking input
    init_weight # function to initialize fixed weights, or nothing for default
end

function PhasorFixed(shape::Pair{<:Integer,<:Integer}, activation = identity; 
                     return_type::SolutionType = SolutionType(:spiking), 
                     init_bias = default_bias, 
                     use_bias::Bool = false,
                     init_weight = nothing,
                     kwargs...)
    # Create a fixed Dense layer with use_bias=false and weights initialized later
    layer = Dense(shape, identity; use_bias=false, kwargs...)
    bias = ComplexBias((shape[2],); init_bias = init_bias)
    return PhasorFixed(layer, bias, activation, use_bias, return_type, init_weight)
end

function Lux.initialparameters(rng::AbstractRNG, l::PhasorFixed)
    # No trainable parameters - weights are stored in state
    return NamedTuple()
end

function Lux.initialstates(rng::AbstractRNG, l::PhasorFixed)
    # Initialize weights in state (non-trainable)
    in_dim = l.layer.in_dims
    out_dim = l.layer.out_dims
    if l.init_weight === nothing
        # Default: Glorot uniform initialization
        weight = glorot_uniform(rng, out_dim, in_dim)
    else
        # Custom initialization
        weight = l.init_weight(rng, in_dim, out_dim)
    end
    
    # Initialize bias parameters (stored in state, non-trainable)
    ps_bias = Lux.initialparameters(rng, l.bias)
    st_bias = Lux.initialstates(rng, l.bias)
    
    # Store weight in a nested structure matching what oscillator_bank expects
    return (weight = weight, layer = (weight = weight,), bias_params = ps_bias, bias = st_bias)
end

function (a::PhasorFixed)(x::AbstractArray, params::LuxParams, state::NamedTuple)
    xz = angle_to_complex(x)
    
    # Apply fixed weights from state
    y_real = state.weight * real.(xz)
    y_imag = state.weight * imag.(xz)
    y = y_real .+ 1.0f0im .* y_imag

    if a.use_bias
        # Use bias params from state (non-trainable)
        y_biased, st_updated_bias = a.bias(y, state.bias_params, state.bias)
        y_activated = a.activation(y_biased)
        st_new = (weight = state.weight, bias_params = state.bias_params, bias = st_updated_bias)
    else
        y_activated = a.activation(y)
        st_new = state
    end

    return y_activated, st_new
end

function (a::PhasorFixed)(x::SpikingCall, params::LuxParams, state::NamedTuple)
    current_call = CurrentCall(x)
    return a(current_call, params, state)
end

function (a::PhasorFixed)(x::CurrentCall, params::LuxParams, state::NamedTuple)
    # Create a pseudo-params structure for oscillator_bank that includes weights from state
    # The layer field must contain weight for the Dense layer call in oscillator_bank
    fixed_params = (layer = (weight = state.weight,), bias = state.bias_params)
    # Create a state structure that oscillator_bank expects (with layer field)
    fixed_state = (layer = NamedTuple(), bias = state.bias)
    
    # Pass the fixed params to the solver
    sol = oscillator_bank(x.current, a, fixed_params, fixed_state, tspan=x.t_span, spk_args=x.spk_args, use_bias=a.use_bias)
    if a.return_type.type == :phase
        u = unrotate_solution(sol.u, sol.t, spk_args=x.spk_args, offset=x.current.offset)
        y = a.activation.(u)
        return y, state
    elseif a.return_type.type == :potential
        return sol, state
    elseif a.return_type.type == :current
        i_fn = t -> potential_to_current(sol(t), spk_args=x.spk_args)
        next_call = CurrentCall(LocalCurrent(i_fn, x.current.shape, x.current.offset + spiking_offset(x.spk_args)),
                                x.spk_args,
                                x.t_span)
        return next_call, state
    else #spiking
        train = solution_to_train(sol, x.t_span, spk_args=x.spk_args, offset=x.current.offset)
        next_call = SpikingCall(train, x.spk_args, x.t_span)
        return next_call, state
    end
end

###
### Random Projection Layer
###

struct RandomProjection <: Lux.AbstractLuxLayer
    dim::Int # The dimension being projected, typically the feature dimension
end

function Lux.initialparameters(rng::AbstractRNG, layer::RandomProjection)
    return NamedTuple() # No trainable parameters
end

function Lux.initialstates(rng::AbstractRNG, layer::RandomProjection)
    # Create a random projection matrix W of size (dim, dim).
    # This matrix will project a vector of length `dim` to another vector of length `dim`.
    # Stored in state as it's non-trainable.
    projection_weights = randn(rng, Float32, layer.dim, layer.dim)
    return (weights = projection_weights, rng = Lux.replicate(rng))
end

# Call
function (rp::RandomProjection)(x::AbstractArray, params::LuxParams, state::NamedTuple)
    # x is expected to have its first dimension match rp.dim
    # e.g., x can be (dim, batch_size) or (dim, H, W, batch_size)
    
    current_size = size(x)
    if current_size[1] != rp.dim
        error("Input first dimension $(current_size[1]) must match layer dimension $(rp.dim)")
    end

    local y::AbstractArray
    if ndims(x) == 1 # Input is a vector (dim,)
        y = state.weights * x
    else # Input is a batched tensor (dim, other_dims...)
        x_reshaped = reshape(x, rp.dim, :)
        y_reshaped = state.weights * x_reshaped
        y = reshape(y_reshaped, current_size)
    end
    
    return y, state # State is not modified in the forward pass for this layer
end

struct RandomPhaseProjection <: LuxCore.AbstractLuxLayer
    dims
end

function RandomPhaseProjection(dims::Tuple{Vararg{Int}})
    return RandomPhaseProjection(dims)
end

function Lux.initialparameters(rng::AbstractRNG, layer::RandomPhaseProjection)
    return NamedTuple() # No trainable parameters
end

function Lux.initialstates(rng::AbstractRNG, layer::RandomPhaseProjection)
    # Create a random projection matrix W of size (dim, dim).
    # This matrix will project a vector of length `dim` to another vector of length `dim`.
    # Stored in state as it's non-trainable.
    projection_weights = rand(rng, (-1.0f0, 1.0f0), layer.dims)
    return (weights = projection_weights, rng = Lux.replicate(rng))
end

function (p::RandomPhaseProjection)(x::AbstractArray, params::LuxParams, state::NamedTuple)
    y = batched_mul(x, state.weights)
    return y, state
end

"""
Residual blocks
"""

"""
    ResidualBlock <: LuxCore.AbstractLuxContainerLayer{(:ff,)}

Residual block for phase-based neural networks, implementing skip connections
through phase binding.

# Fields
- `ff`: Feed-forward chain of phase-based layers

# Implementation Details
1. Processes input through feed-forward path
2. Binds (combines) original input with processed output
3. Maintains phase-based representation throughout

Used to build deep phase networks while mitigating phase degradation,
similar to residual connections in standard neural networks but using
phase binding for combination.

See also: [`v_bind`](@ref) for the phase binding operation
"""
struct ResidualBlock <: LuxCore.AbstractLuxContainerLayer{(:ff,)}
    ff
end

function ResidualBlock(dimensions::Tuple{Vararg{Int}}, activation::Function; kwargs...)
    @assert length(dimensions) >= 2 "Must have at least 1 layer"
    #construct a Phasor MLP based on the given dimensions
    pairs = [dimensions[i] => dimensions[i+1] for i in 1:length(dimensions) - 1]
    layers = [PhasorDense(pair, activation, kwargs...) for pair in pairs]
    ff = Chain(layers...)

    return ResidualBlock(ff)
end

function (rb::ResidualBlock)(x, ps, st)
    # MLP path
    ff_out, st_ff = rb.ff(x, ps.ff, st.ff)
    y = v_bind(x, ff_out)
    
    return y, st_ff
end

"""
Phasor QKV Attention
"""
function score_scale(scores::AbstractArray{<:Real,3}; scale::AbstractVector{<:Real})
    d_k = size(scores,1)
    return exp.(scale .* scores) ./ d_k
end

function score_scale(potential::AbstractArray{<:Complex,3}, scores::AbstractArray{<:Real,3}; scale::AbstractVector)
    @assert size(potential, 3) == size(scores,3) "Batch dimensions of inputs must match"

    scores = permutedims(scores, (2,1,3))
    d_k = size(scores,1)
    scores = exp.(scale .* scores) ./ d_k
    scaled = batched_mul(potential, scores)
    return scaled
end

function attend(q::AbstractArray{<:Real, 3}, k::AbstractArray{<:Real, 3}, v::AbstractArray{<:Real, 3}; scale::AbstractArray=[1.0f0,])
    #compute qk scores
    #produces (qt kt b)
    scores = score_scale(similarity_outer(q, k, dims=2), scale=scale)
    #do complex-domain matrix multiply of values by scores (kt v b)
    v = angle_to_complex(v)
    #multiply each value by the scores across batch
    #(v kt b) * (kt qt b) ... (v kt) * (kt qt) over b
    output = batched_mul(v, scores)
    output = complex_to_angle(output)
    return output, scores
end



"""
    attend(q::SpikingTypes, k::SpikingTypes, v::SpikingTypes; spk_args, tspan, return_solution=false, scale=[1.0f0]) -> Tuple

Compute attention between spiking neural representations using phase similarity.
Core attention mechanism for spiking transformer architectures.

# Arguments
- `q, k, v`: Query, key, and value spike trains
- `spk_args::SpikingArgs`: Spiking neuron parameters
- `tspan`: Time span for simulation
- `return_solution::Bool`: Whether to return raw potentials
- `scale::AbstractArray`: Attention scaling factor

# Implementation
1. Computes temporal similarity between query and key spikes
2. Converts value spikes to oscillator potentials
3. Scales and combines values based on similarities
4. Optionally converts back to spike train

Returns:
- Spike train or potentials representing attended values
- Attention scores over time

See also: [`attend`](@ref) for phase-based version
"""
function attend(q::SpikingTypes, k::SpikingTypes, v::SpikingTypes; spk_args::SpikingArgs, tspan::Tuple{<:Real, <:Real}=(0.0f0, 10.0f0), return_solution::Bool = false, scale::AbstractArray=[1.0f0,])
    #compute the similarity between the spike trains
    #produces [time][b qt kt]
    scores = similarity_outer(q, k, spk_args=spk_args, tspan=tspan)
    #convert the values to potentials
    d_k = size(k)[2]
    values = oscillator_bank(v, tspan=tspan, spk_args=spk_args)
    #multiply by the scores found at eachb time step
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

"""
    SingleHeadAttention <: LuxCore.AbstractLuxContainerLayer{(:q_proj, :k_proj, :v_proj, :attention, :out_proj)}

Single-head attention mechanism for phase-based transformers.
Implements attention using phase similarity for key-query interactions.

# Fields
- `q_proj`: Query projection layer
- `k_proj`: Key projection layer
- `v_proj`: Value projection layer
- `attention`: Attention scoring mechanism
- `out_proj`: Output projection layer

# Implementation Details
1. Projects input to query/key/value representations
2. Computes attention scores using phase similarity
3. Combines values weighted by attention scores
4. Projects combined values to output space

Can operate on both direct phase inputs and spiking representations.
See also: [`attend`](@ref) for the core attention computation
"""
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

function SingleHeadCABlock(d_input::Int, d_model::Int, n_q::Int, n_kv::Int; dropout::Real=0.1f0, kwargs...)
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
    train(model, ps, st, train_loader, loss, args; optimiser=Optimisers.Adam, verbose=false, sample_gradients=0)

Train a phase-based neural network using gradient descent.

# Arguments
- `model`: Network model (any Lux.jl compatible architecture)
- `ps`: Model parameters
- `st`: Model state
- `train_loader`: Data loader providing (x, y) batches
- `loss`: Loss function(x, y, model, params, state)
- `args::Args`: Training configuration
- `optimiser`: Optimization algorithm (default: Adam)
- `verbose::Bool`: Whether to print loss values
- `sample_gradients::Int`: Frequency of gradient sampling (0 to disable)

# Returns
- `losses`: Array of loss values during training
- `ps`: Updated parameters
- `st`: Updated state
- `gradients`: Sampled gradients if enabled

Automatically handles CPU/GPU device placement based on args.use_cuda.
"""
function train(model, ps, st, train_loader, loss, args; optimiser = Optimisers.Adam, verbose::Bool = false, sample_gradients::Int = 0)
    if CUDA.functional() && args.use_cuda
       @info "Training on CUDA GPU"
       #CUDA.allowscalar(false)
       device = gpu_device()
   else
       @info "Training on CPU"
       device = cpu_device()
   end

   ## Optimizer
   opt_state = Optimisers.setup(optimiser(args.lr), ps)
   losses = []
   gradients = []
   step_count = 0

   ## Training
   for epoch in 1:args.epochs
       for (x, y) in train_loader
           step_count += 1
           x = x |> device
           y = y |> device
           
           lf = p -> loss(x, y, model, p, st)
           lossval, gs = withgradient(lf, ps)
           if verbose
               println(reduce(*, ["Epoch ", string(epoch), " loss: ", string(lossval)]))
           end
           append!(losses, lossval)
           
           # Save gradients if sampling is enabled and we're at a sampling step
           if sample_gradients > 0 && (step_count % sample_gradients == 0)
               push!(gradients, deepcopy(gs[1]))
           end
           
           opt_state, ps = Optimisers.update(opt_state, ps, gs[1]) ## update parameters
       end
   end
   
   if sample_gradients > 0
    return losses, ps, st, gradients
   else
    return losses, ps, st
   end
end


"""
Other utilities
"""

struct MinPool <: LuxCore.AbstractLuxWrapperLayer{:pool}
    pool
end

function MinPool()
    return MinPool(MaxPool())
end

function (mp::MinPool)(x, ps, st)
    y = -1.0f0 .* mp(-1.0f0 .* x, ps, st)
    return y
end

"""
    (m::Union{MaxPool, MinPool})(x::SpikingCall, ps::LuxParams, st::NamedTuple)

Extend MaxPool to handle SpikeTrain inputs by selecting the spike with maximum
decoded phase value over the pooling dimensions.

# Arguments
- `x::SpikingCall`: Input spiking call
- `ps::LuxParams`: Layer parameters
- `st::NamedTuple`: Layer state

# Operation
1. Converts each spike to its corresponding phase value using `train_to_phase`
2. Finds the maximum phase value over the pooling dimensions
3. Returns a new SpikeTrain with the selected spike(s), preserving temporal offset

# Returns
- `output_train::SpikeTrain`: Spike train containing only maximum phase spikes
- `st::NamedTuple`: Unchanged state
"""
function (pool::Union{MaxPool, MinPool})(call::SpikingCall, params::LuxParams, state::NamedTuple)
    if on_gpu(call.train)
        #move to CPU if necessary
        gpu = true
        train = SpikeTrain(call.train)
    else
        gpu = false
        train = call.train
    end

    #calculate the phase values represented by each spike in each resonant period
    phases = train_to_phase(train, spk_args=call.spk_args)
    max_phases = map(x -> pool(x, params, state)[1],
                    eachslice(phases, dims=ndims(phases)))
    #convert those extremum phase values back to spikes
    trains = map(x -> phase_to_train(x, 
                                    spk_args=call.spk_args,
                                    repeats=1,
                                    offset=call.train.offset),
                                    max_phases)
    #look up what the offset for that spiking cycle is & adjust the spikes to match
    n_slices = size(phases, ndims(phases))
    cycles = generate_cycles(call.t_span, call.spk_args, call.train.offset)[1:n_slices]
    trains = map(x -> delay_train(x[1], call.spk_args.t_period * x[2], 0.0f0),
                        zip(trains, cycles))
    #concatenate each train together for a single pooled output
    new_train = vcat_trains(trains...)

    if gpu
        new_train = SpikeTrainGPU(new_train)
    end
    new_call = SpikingCall(new_train, call.spk_args, call.t_span)
    return new_call, state
end

"""
    TrackOutput{L<:Lux.AbstractLuxLayer} <: Lux.AbstractLuxLayer

Wrapper layer that records intermediate outputs during forward passes.
Useful for analyzing internal representations in phase networks.

# Fields
- `layer::L`: The layer whose outputs to track

# State
Maintains a tuple of all intermediate outputs in the state.outputs field.
Each forward pass appends its output to this tuple.

# Usage
```julia
tracked_layer = TrackOutput(PhasorDense(64 => 32))
y, st = tracked_layer(x, ps, st)
intermediate_outputs = st.outputs  # Access all recorded outputs
```

Useful for visualization, analysis, and debugging of phase networks.
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

"""
    variance_scaling(rng::AbstractRNG, shape::Integer...; mode="avg", scale=0.66f0)

Initialize network weights using variance scaling initialization.
Adapts the scale based on input/output dimensions to maintain stable variances.

# Arguments
- `rng::AbstractRNG`: Random number generator
- `shape::Integer...`: Dimensions of weight matrix/tensor
- `mode::String`: Scaling mode ("fan_in", "fan_out", or "avg")
- `scale::Real`: Base scaling factor (default: 0.66f0)

# Modes
- "fan_in": Scale based on input dimension
- "fan_out": Scale based on output dimension
- "avg": Scale based on average of input/output dimensions

Returns weights initialized from truncated normal distribution with
computed standard deviation.
"""
function variance_scaling(rng::AbstractRNG, shape::Integer...; mode::String = "avg", scale::Real = 0.66f0)
    fan_in = shape[end]
    fan_out = shape[1]

    if mode == "fan_in"
        scale /= max(1.0f0, fan_in)
    elseif mode == "fan_out"
        scale /= max(1.0f0, fan_out)
    else
        scale /= max(1.0f0, (fan_in + fan_out) / 2.0f0)
    end

    stddev = sqrt(scale) / 0.87962566103423978f0
    return truncated_normal(rng, shape..., mean = 0.0f0, std = stddev)
end

function square_variance(rng::AbstractRNG, shape::Integer; kwargs...)
    weights = variance_scaling(rng, shape, shape; kwargs...)
    weights[diagind(weights)] .= 1.0f0
    return weights
end