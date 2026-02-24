# Complex-native implementation of PhasorDense and related layers
# This file contains the refactored versions that work natively with complex values on the unit circle

# This file is designed to be included, so it uses the parent scope's imports

"""
    PhasorDenseComplex <: LuxCore.AbstractLuxContainerLayer{(:layer, :bias)}

A dense (fully-connected) layer for complex-valued neural networks with unit circle normalization.
This is a refactored version of PhasorDense that works natively with complex inputs/outputs,
applying configurable normalization to project values onto or toward the unit circle.

# Fields
- `layer`: Underlying `Dense` layer for linear transformation (no internal bias)
- `bias`: [`ComplexBias`](@ref) layer for phase shifts in the complex plane
- `normalization`: Function that projects complex values (z → f(z) with |f(z)| ≈ 1)
- `output_type::Symbol`: Output format - `:complex` (keep complex) or `:phase` (convert to real phases)
- `use_bias::Bool`: Whether to apply the complex bias term
- `init_leakage::Function`: Initializer for per-neuron leakage scaling factors
- `init_period::Function`: Initializer for per-neuron period scaling factors
- `trainable_leakage::Bool`: If `true`, leakage factors are trainable parameters
- `trainable_period::Bool`: If `true`, period factors are trainable parameters
- `return_type::SolutionType`: Output format for spiking inputs

# Constructor
```julia
PhasorDenseComplex(shape::Pair{<:Integer,<:Integer};
    normalization=normalize_to_unit_circle,  # Normalization function
    output_type=:phase,                       # :complex or :phase
    return_type=SolutionType(:spiking),      # For spiking inputs
    init_bias=default_bias,                   # Bias initialization function
    use_bias=true,                            # Apply complex bias
    init_leakage=ones32,                      # Leakage factor initializer
    init_period=ones32,                       # Period factor initializer
    trainable_leakage=false,                  # Make leakage trainable
    trainable_period=false,                   # Make period trainable
    kwargs...)                                # Passed to Dense layer
```

# Normalization Functions
- `normalize_to_unit_circle`: Hard normalization, projects exactly to unit circle (z → z/|z|)
- `soft_normalize_to_unit_circle`: Soft normalization, gradually pushes toward unit circle
- Custom function with signature: `f(z::AbstractArray{<:Complex}) -> AbstractArray{<:Complex}`

# Output Types
- `:complex`: Returns complex values (after normalization)
- `:phase`: Converts to real-valued phases via `complex_to_angle`

# Return Types (for spiking inputs)
- `:phase`: Extract phases from ODE solution, return array
- `:potential`: Return raw ODE solution object
- `:current`: Convert solution to current, return `CurrentCall` for next layer
- `:spiking`: Convert solution to spike train, return `SpikingCall` (default)

# Forward Pass Architecture

The forward pass follows this sequence:
1. **Input**: Complex values on unit circle (or real phases that are converted)
2. **Linear transformation**: Apply weights separately to real and imaginary parts
3. **Bias**: Optionally add complex bias (shifts values in complex plane)
4. **Normalization**: Apply normalization function (the "activation", projects toward unit circle)
5. **Output conversion**: Optionally convert to phases if output_type=:phase

This architecture allows the normalization function to control the dynamics of the layer,
with different normalization strategies providing different behaviors.

# Examples
```julia
# Basic layer with hard normalization, phase output
layer = PhasorDenseComplex(64 => 32)

# Soft normalization for smoother gradients
layer = PhasorDenseComplex(64 => 32;
    normalization=soft_normalize_to_unit_circle)

# Keep complex outputs for chaining
layer = PhasorDenseComplex(64 => 32; output_type=:complex)

# Custom normalization with parameters
custom_norm = z -> soft_normalize_to_unit_circle(z; r_lo=0.05f0, r_hi=0.15f0)
layer = PhasorDenseComplex(64 => 32; normalization=custom_norm)

# Chain complex layers efficiently
model = Chain(
    PhasorDenseComplex(64 => 32; output_type=:complex),  # Complex → Complex
    PhasorDenseComplex(32 => 16; output_type=:complex),  # Complex → Complex
    PhasorDenseComplex(16 => 10; output_type=:phase)     # Complex → Phase (final output)
)

# Initialize and apply
rng = Random.default_rng()
ps, st = Lux.setup(rng, layer)

# Can accept complex inputs
x_complex = angle_to_complex(randn(Float32, 64, 10))
y, st = layer(x_complex, ps, st)

# Or phase inputs (automatically converted)
x_phase = randn(Float32, 64, 10)
y, st = layer(x_phase, ps, st)
```

See also: [`PhasorDense`](@ref), [`normalize_to_unit_circle`](@ref),
[`soft_normalize_to_unit_circle`](@ref), [`ComplexBias`](@ref)
"""
struct PhasorDenseComplex <: LuxCore.AbstractLuxContainerLayer{(:layer, :bias)}
    layer                        # Dense layer for linear transformation
    bias                         # ComplexBias layer
    normalization::Function      # Function: Complex → Complex (projects toward unit circle)
    use_bias::Bool
    init_leakage::Function
    init_period::Function
    trainable_leakage::Bool
    trainable_period::Bool
    return_type::SolutionType
end

function PhasorDenseComplex(shape::Pair{<:Integer,<:Integer};
                            normalization::Function = normalize_to_unit_circle,
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

    return PhasorDenseComplex(layer,
                              bias,
                              normalization,
                              use_bias,
                              init_leakage,
                              init_period,
                              trainable_leakage,
                              trainable_period,
                              return_type)
end

# Parameter and state initialization (same as PhasorDense)
function Lux.initialparameters(rng::AbstractRNG, l::PhasorDenseComplex)
    ps_layer = Lux.initialparameters(rng, l.layer)
    parameters = (layer = ps_layer,)

    if l.use_bias
        ps_bias = Lux.initialparameters(rng, l.bias)
        parameters = merge(parameters, (bias = ps_bias,))
    end

    n_out = l.layer.out_dims
    if l.trainable_leakage
        ps_leakage = l.init_leakage(rng, n_out,)
        parameters = merge(parameters, (leakage = ps_leakage,))
    end

    if l.trainable_period
        ps_period = l.init_period(rng, n_out,)
        parameters = merge(parameters, (period = ps_period,))
    end
    return parameters
end

function Lux.initialstates(rng::AbstractRNG, l::PhasorDenseComplex)
    st_layer = Lux.initialstates(rng, l.layer)
    st_bias = Lux.initialstates(rng, l.bias)
    state = (layer = st_layer, bias = st_bias,)

    n_out = l.layer.out_dims
    if !l.trainable_leakage
        st_leakage = l.init_leakage(rng, n_out,)
        state = merge(state, (leakage = st_leakage,))
    end

    if !l.trainable_period
        st_period = l.init_period(rng, n_out,)
        state = merge(state, (period = st_period,))
    end
    return state
end

###
### Forward Pass Implementations
###

"""
    (a::PhasorDenseComplex)(x::AbstractArray{<:Complex}, params::LuxParams, state::NamedTuple)

Core complex-valued forward pass. This is the primary implementation that all other input types route to.

# Process
1. Apply linear transformation to real and imaginary parts separately
2. Add complex bias (if enabled)
3. Apply normalization function (projects toward/onto unit circle)
4. Convert to phases if output_type=:phase, otherwise keep complex
"""
function (a::PhasorDenseComplex)(x::AbstractArray{<:Complex}, params::LuxParams, state::NamedTuple)
    # Linear transformation on real and imaginary components
    y_real, _ = a.layer(real.(x), params.layer, state.layer)
    y_imag, _ = a.layer(imag.(x), params.layer, state.layer)
    y = y_real .+ 1.0f0im .* y_imag

    # Apply complex bias
    if a.use_bias
        y_biased, st_updated_bias = a.bias(y, params.bias, state.bias)
    else
        st_updated_bias = state.bias
        y_biased = y
    end

    # Apply normalization (this is the "activation" that shapes the complex values)
    y_normalized = a.normalization(y_biased)

    st_new = (layer = state.layer, bias = st_updated_bias)
    return y_normalized, st_new
end

"""
    (a::PhasorDenseComplex)(x::AbstractArray{<:Real}, params::LuxParams, state::NamedTuple)

Phase-based interface. Converts real-valued phases to complex and dispatches to complex method.
"""
function (a::PhasorDenseComplex)(x::AbstractArray{<:Real}, params::LuxParams, state::NamedTuple)
    # Convert phases to complex representation on unit circle
    xz = angle_to_complex(x)

    # Dispatch to complex-valued core implementation
    y_normalized, st_new = a(xz, params, state)
    y_phase = complex_to_angle(y_normalized)
    return y_phase, st_new
end

"""
    (a::PhasorDenseComplex)(x::SpikingCall, params::LuxParams, state::NamedTuple)

Spiking input interface. Converts to CurrentCall and processes through ODE integration.
"""
function (a::PhasorDenseComplex)(x::SpikingCall, params::LuxParams, state::NamedTuple)
    current_call = CurrentCall(x)
    return a(current_call, params, state)
end

"""
    (a::PhasorDenseComplex)(x::CurrentCall, params::LuxParams, state::NamedTuple)

Current-based spiking input. Integrates ODEs and returns based on return_type.
"""
function (a::PhasorDenseComplex)(x::CurrentCall, params::LuxParams, state::NamedTuple)
    # Pass to oscillator bank for ODE integration
    sol = oscillator_bank(x.current, a, params, state, tspan=x.t_span, spk_args=x.spk_args, use_bias=a.use_bias)

    if a.return_type.type == :phase
        u = unrotate_solution(sol.u, sol.t, spk_args=x.spk_args, offset=x.current.offset)
        # Apply normalization to the complex potentials before converting to phase
        u_normalized = map(u_t -> a.normalization(u_t), u)
        y = complex_to_angle.(u_normalized)
        return y, state
    elseif a.return_type.type == :potential
        return sol, state
    elseif a.return_type.type == :current
        i_fn = t -> potential_to_current(sol(t), spk_args=x.spk_args)
        next_call = CurrentCall(LocalCurrent(i_fn, x.current.shape, x.current.offset + spiking_offset(x.spk_args)),
                                x.spk_args,
                                x.t_span)
        return next_call, state
    else  # :spiking
        train = solution_to_train(sol, x.t_span, spk_args=x.spk_args, offset=x.current.offset)
        next_call = SpikingCall(train, x.spk_args, x.t_span)
        return next_call, state
    end
end

###
### Convenience Constructors for Common Normalization Types
###

"""
    PhasorDenseHard(shape; kwargs...)

Convenience constructor for PhasorDenseComplex with hard normalization (exact unit circle projection).
"""
function PhasorDenseHard(shape::Pair{<:Integer,<:Integer}; kwargs...)
    return PhasorDenseComplex(shape;
        normalization=normalize_to_unit_circle,
        kwargs...)
end

"""
    PhasorDenseSoft(shape; r_lo=0.1f0, r_hi=0.2f0, kwargs...)

Convenience constructor for PhasorDenseComplex with soft normalization (gradual transition to unit circle).

# Arguments
- `r_lo::Real`: Lower magnitude threshold for soft normalization (default: 0.1)
- `r_hi::Real`: Upper magnitude threshold for soft normalization (default: 0.2)
"""
function PhasorDenseSoft(shape::Pair{<:Integer,<:Integer};
                         r_lo::Real = 0.1f0,
                         r_hi::Real = 0.2f0,
                         kwargs...)
    # Create closure that captures r_lo and r_hi
    soft_norm = z -> soft_normalize_to_unit_circle(z; r_lo=r_lo, r_hi=r_hi)

    return PhasorDenseComplex(shape;
        normalization=soft_norm,
        kwargs...)
end
