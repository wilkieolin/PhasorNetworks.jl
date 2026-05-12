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
    PhasorDense <: Lux.AbstractLuxLayer

A dense (fully-connected) layer for phase-valued and spiking neural networks
with per-channel oscillator dynamics. Operates on inputs that have already
been moved into the phase domain — for continuous complex-valued sequences
use [`PhasorResonant`](@ref) (or [`ResonantSTFT`](@ref) when ω should be
trainable) as the encoder upstream.

Dispatches on input type AND dimensionality:
- 2D Phase `(C, B)` → angle_to_complex → 2D complex → activation → complex_to_angle
- 3D Phase `(C, L, B)` → Dirac discretization with causal convolution → complex_to_angle
- SpikingCall → convert to CurrentCall, delegate
- CurrentCall → single-stage ODE `dz/dt = k·z + W·I(t)`

A 2D Complex helper dispatch is kept as an internal building block for the
2D Phase path (it does the matmul + bias step on already-complex data); it
is not part of the documented forward-pass surface.

# Fields
- `in_dims::Int`: Input feature dimension
- `out_dims::Int`: Output feature dimension
- `activation::Function`: Normalization function (Complex → Complex)
- `use_bias::Bool`: Whether to apply complex bias
- `init_weight::Function`: Weight initializer `(rng, out, in) -> Matrix`
- `init_bias::Function`: Bias initializer `(rng, dims) -> ComplexF32 array`
- `init_mode::Symbol`: Initialization mode for dynamics (`:default` or `:hippo`)
- `return_type::SolutionType`: Output format for spiking inputs
- `init_log_neg_lambda::Union{Float32, Nothing}`: Optional uniform override
  for the initial value of `log_neg_lambda`. When `nothing` (default), each
  `init_mode` uses its own preset (see *Init Modes* below). Useful at long
  sequence lengths `L` where the per-mode default produces
  `λ·L < log(eps(Float32))` and the kernel underflows.

# Parameters (always present)
- `weight` — `(out, in)` Float32
- `log_neg_lambda` — `(out,)` Float32, per-channel decay (always trainable)
- `bias_real`, `bias_imag` — `(out,)` Float32 (when `use_bias=true`)

# State
- `omega` — `(out,)` Float32, fixed at `2π` for every channel.

# Per-channel ω rule
Every channel in this layer carries the same carrier frequency `ω = 2π`
(period = 1, matching the spiking convention `spk_args.t_period = 1`).
Per-channel ω diversity would desynchronize the phase carrier and break
HD-VSA invariance with downstream phase-locked layers. Cross-channel
diversity comes from `λ` and the weight matrix, not from `ω`. Use
[`ResonantSTFT`](@ref) when channel-dependent ω is genuinely the goal
(frequency decomposition); it re-encodes its outputs at a uniform
downstream `omega_out` so the rest of the network can resume phase-locked
operation.

# Init Modes
- `:default` — `log_neg_lambda = fill(log(0.2), out)`, single timescale.
- `:hippo`   — `log_neg_lambda` from log-spaced HiPPO-LegS λ spectrum
  (per-channel multi-timescale memory). ω stays at `2π`.

When `init_log_neg_lambda` is set, it replaces the per-channel value
uniformly across `out` for both modes (overrides the HiPPO spread in
`:hippo`).

See also: [`PhasorConv`](@ref), [`PhasorFixed`](@ref), [`ComplexBias`](@ref)
"""
struct PhasorDense <: Lux.AbstractLuxLayer
    in_dims::Int
    out_dims::Int
    activation::Function
    use_bias::Bool
    init_weight::Function
    init_bias::Function
    init_mode::Symbol
    return_type::SolutionType
    init_log_neg_lambda::Union{Vector{Float32}, Nothing}  # nothing → use init_mode default; otherwise per-channel override
end

function PhasorDense(shape::Pair{<:Integer,<:Integer},
                    activation = normalize_to_unit_circle;
                    return_type::SolutionType = SolutionType(:spiking),
                    init_bias = default_bias,
                    use_bias::Bool = true,
                    init_weight = nothing,
                    init = nothing,
                    init_mode::Symbol = :default,
                    init_log_neg_lambda::Union{Real, AbstractVector{<:Real}, Nothing} = nothing,
                    kwargs...)
    # Handle backward compatibility: 'init' kwarg maps to init_weight
    if init_weight === nothing && init !== nothing
        init_weight = init
    elseif init_weight === nothing
        init_weight = glorot_uniform
    end
    init_mode in (:default, :hippo) ||
        throw(ArgumentError("init_mode must be :default or :hippo (got :$init_mode). " *
                            "The :uniform mode was removed because it spread ω across " *
                            "channels, which breaks the per-channel ω rule. For multi-" *
                            "timescale dynamics use :hippo."))
    out = shape[2]
    lnl_vec = if init_log_neg_lambda === nothing
        nothing
    elseif init_log_neg_lambda isa Real
        fill(Float32(init_log_neg_lambda), out)
    else
        @assert length(init_log_neg_lambda) == out "init_log_neg_lambda must have length $(out), got $(length(init_log_neg_lambda))"
        Float32.(collect(init_log_neg_lambda))
    end
    return PhasorDense(shape[1], shape[2],
                        activation,
                        use_bias,
                        init_weight,
                        init_bias,
                        init_mode,
                        return_type,
                        lnl_vec)
end

# Per-mode (log_neg_lambda, omega) initializer. ω is uniformly 2π across
# channels in every mode (the per-channel ω rule); modes only differ in
# how λ is laid out. When `init_log_neg_lambda` is supplied (as a vector
# from the constructor — even for a scalar override the constructor lifts
# it to a per-channel vector), it overrides whichever mode-default would
# apply.
function _init_dynamics(l::PhasorDense)
    omega = fill(Float32(2π), l.out_dims)
    log_neg_lambda = if l.init_log_neg_lambda !== nothing
        copy(l.init_log_neg_lambda)
    elseif l.init_mode == :hippo
        λ_init, _ = hippo_legs_diagonal(l.out_dims)
        log.(-λ_init)
    else  # :default
        fill(Float32(log(0.2)), l.out_dims)
    end
    return log_neg_lambda, omega
end

# Helper to get omega from state. (Kept as a function so that adding a
# back-channel for the rare scalar-trainable-ω experiment later wouldn't
# require touching every call site.)
_get_omega(::PhasorDense, _params, state) = state.omega

function Lux.initialparameters(rng::AbstractRNG, l::PhasorDense)
    W = l.init_weight(rng, l.out_dims, l.in_dims)
    log_neg_lambda, omega = _init_dynamics(l)

    parameters = (weight = W, log_neg_lambda = log_neg_lambda)

    if l.use_bias
        bias = l.init_bias(rng, (l.out_dims,))
        parameters = merge(parameters, (bias_real = Float32.(real.(bias)),
                                         bias_imag = Float32.(imag.(bias))))
    end

    return parameters
end

function Lux.initialstates(rng::AbstractRNG, l::PhasorDense)
    _, omega = _init_dynamics(l)
    return (omega = omega,)
end

# ---- 2D Complex dispatch: W * x + bias, no activation ----

function (a::PhasorDense)(x::AbstractArray{<:Complex}, params::LuxParams, state::NamedTuple)
    y_real = params.weight * real.(x)
    y_imag = params.weight * imag.(x)
    y = y_real .+ 1.0f0im .* y_imag

    if a.use_bias
        bias_val = params.bias_real .+ 1.0f0im .* params.bias_imag
        y = y .+ bias_val
    end

    return y, state
end

# ---- 2D Phase dispatch ----

function (a::PhasorDense)(x::AbstractArray{<:Phase}, params::LuxParams, state::NamedTuple)
    xz = angle_to_complex(x)
    y, st_new = a(xz, params, state)
    y_normalized = a.activation(y)
    y_phase = complex_to_angle(y_normalized)
    return y_phase, st_new
end

# ---- 3D Phase dispatch ----

function (a::PhasorDense)(x::AbstractArray{<:Phase, 3}, params::LuxParams, state::NamedTuple)
    return _forward_3d_dirac(a, x, params, state)
end

# ---- 3D Phase Dirac path ----

function _forward_3d_dirac(a::PhasorDense, x::AbstractArray{<:Phase, 3},
                           params::LuxParams, state::NamedTuple)
    λ = -exp.(params.log_neg_lambda)
    ω = _get_omega(a, params, state)
    L = size(x, 2)

    Z = causal_conv_dirac(x, params.weight, λ, ω, 1f0)

    if a.use_bias
        # Per-period bias drive: bias enters as a constant kick at every
        # period and is accumulated by the SSM kernel — the discrete
        # equivalent of `bias_current` in the CurrentCall path. Without
        # this scaling the bias is ~150× weaker than the accumulated
        # signal at slow-decay encoder settings (α = 5/L), so silent
        # neurons drift to the origin instead of defaulting to the
        # bias-direction "wave" the static-MLP convention assumes.
        bias_val = params.bias_real .+ 1.0f0im .* params.bias_imag       # (C,)
        G = bias_kernel_accumulation(λ, ω, 1f0, L)                       # (C, L)
        Z = Z .+ reshape(bias_val, :, 1, 1) .* reshape(G, size(G, 1), size(G, 2), 1)
    end

    # Skip normalization when the activation is normalize_to_unit_circle:
    # the function divides through a positive real (whether |z| or
    # √(|z|² + ε)), so complex_to_angle ∘ normalize_to_unit_circle
    # reduces to complex_to_angle on nonzero inputs, and both return 0
    # angle at z = 0.
    if a.activation === normalize_to_unit_circle
        return complex_to_angle(Z), state
    else
        Y = a.activation(Z)
        return complex_to_angle(Y), state
    end
end

# ---- SpikingCall dispatch ----

function (a::PhasorDense)(x::SpikingCall, params::LuxParams, state::NamedTuple)
    current_call = CurrentCall(x)
    return a(current_call, params, state)
end

# ---- CurrentCall dispatch: continuous ODE mode ----
#
# Solves dz_c/dt = k_c · z_c + Σ_j W[c,j] · I_j(t)
# where k_c = λ_c + iω_c is the per-channel complex eigenvalue.
# This is the continuous-time equivalent of the discrete causal convolution
# used in the 3D complex/Phase dispatch paths.

function (a::PhasorDense)(x::CurrentCall, params::LuxParams, state::NamedTuple)
    spk_args = x.spk_args
    tspan = x.t_span

    sample_I = x.current.current_fn(Float32(tspan[1]))
    out_shape = ndims(sample_I) >= 2 ? (a.out_dims, size(sample_I)[2:end]...) : (a.out_dims,)
    u0 = similar(sample_I, ComplexF32, out_shape)
    ignore_derivatives() do
        u0 .= zero(ComplexF32)
    end

    use_bias = a.use_bias && haskey(params, :bias_real)

    function dzdt(u, p, t)
        λ = -exp.(p.log_neg_lambda)
        # ω lives in state across all phase-locked layers (per-channel ω rule).
        ω_val = state.omega
        k = ComplexF32.(λ .+ im .* ω_val)
        I_transformed = p.weight * x.current.current_fn(t)
        result = k .* u .+ I_transformed
        if use_bias
            bias_val = p.bias_real .+ 1im .* p.bias_imag
            result = result .+ bias_current(bias_val, t, x.current.offset, spk_args)
        end
        return result
    end

    prob = ODEProblem(dzdt, u0, tspan, params)
    sol = solve(prob, spk_args.solver, p=params; spk_args.solver_args...)

    if a.return_type.type == :potential
        return sol, state
    end

    if a.return_type.type == :phase
        u = unrotate_solution(sol.u, sol.t, spk_args=spk_args, offset=x.current.offset)
        y = a.activation.(u)
        phase = complex_to_angle.(y)
        return phase, state
    elseif a.return_type.type == :current
        i_fn = t -> potential_to_current(sol(t), spk_args=spk_args)
        next_call = CurrentCall(LocalCurrent(i_fn, x.current.shape, x.current.offset + spiking_offset(spk_args)),
                                spk_args,
                                tspan)
        return next_call, state
    else # :spiking
        train = solution_to_train(sol, tspan, spk_args=spk_args, offset=x.current.offset)
        next_call = SpikingCall(train, spk_args, tspan)
        return next_call, state
    end
end

###
### PhasorResonant — Complex → Phase encoder (fixed ω, ZOH SSM)
###

"""
    PhasorResonant(in_dims => out_dims, activation = normalize_to_unit_circle;
                   omega = nothing, omega_lo = 0.2, omega_hi = 2.5,
                   use_bias = false, init_weight = nothing,
                   init_bias = default_bias,
                   init_log_neg_lambda = log(0.1)) <: Lux.AbstractLuxLayer

Encoder layer that consumes a continuous complex-valued sequence and produces
a phase-valued sequence suitable for downstream phase-only layers
(`PhasorDense`, `Codebook`, attention, …).

Mathematically a per-channel resonate-and-fire SSM with **fixed** angular
frequencies `ω` and trainable decay `λ = -exp(log_neg_lambda)`. Each output
channel `c` integrates the weight-mixed input via the discrete-time
zero-order-hold recurrence

    z_c[n+1] = exp(k_c) · z_c[n] + B_c · I_c[n],    k_c = λ_c + i · ω_c

precomputed once as the impulse-response kernel
`K_c[n] = exp(k_c·n) · (exp(k_c) − 1)/k_c` and applied via [`causal_conv`](@ref).
The output is then projected to the unit circle and converted to phase via
`complex_to_angle`.

This layer is the **single-frequency, multi-timescale** sibling of
[`ResonantSTFT`](@ref). All `out_dims` resonators share one carrier
frequency `ω` (so their output phases remain commensurable for downstream
phase-locked operations), but each can have its own decay rate `λ` —
giving a multi-timescale memory bank without breaking phase invariance.
Use `ResonantSTFT` when channel-dependent ω is genuinely the goal (e.g.
trainable frequency decomposition); use `PhasorResonant` everywhere else.

# Arguments
- `in_dims => out_dims`: input feature count → number of resonators.
- `activation`: applied to the complex membrane potential before
  `complex_to_angle`. Default `normalize_to_unit_circle` (which is then
  short-circuited because angle is magnitude-independent).
- `omega::Real`: shared carrier frequency for all channels (default `2π`,
  matching the `t_period = 1` spiking convention). Constant by design —
  per-channel ω would desynchronize the phase carrier and break HD-VSA
  invariance with downstream layers.
- `use_bias`: optional complex bias added after the convolution.
- `init_weight`: weight init `(rng, out, in) -> Matrix`. Default
  `glorot_uniform`.
- `init_bias`: bias init `(rng, dims) -> ComplexF32 array`. Default
  `default_bias`.
- `init_log_neg_lambda`: initial value(s) for `log_neg_lambda`. Pass a
  `Real` for a uniform decay across all channels, or an
  `AbstractVector` of length `out_dims` for a multi-timescale spread
  (e.g. `log.(-hippo_legs_diagonal(out_dims)[1])`).

# Parameters
- `weight` — `(out_dims, in_dims)` Float32.
- `log_neg_lambda` — `(out_dims,)` Float32 (trainable per-channel decay).
- `bias_real`, `bias_imag` — `(out_dims,)` Float32 (when `use_bias=true`).

# State
- `omega` — Float32 scalar, fixed at construction.

# Forward
- 3D Complex `(C_in, L, B)` → 3D Phase `(out_dims, L, B)`.

See also: [`ResonantSTFT`](@ref) (per-channel trainable ω for frequency
decomposition), [`PhasorDense`](@ref) (downstream phase-only layer),
[`phasor_kernel`](@ref), [`causal_conv`](@ref).
"""
struct PhasorResonant <: Lux.AbstractLuxLayer
    in_dims::Int
    out_dims::Int
    activation::Function
    use_bias::Bool
    init_weight::Function
    init_bias::Function
    omega::Float32                          # shared carrier frequency
    init_log_neg_lambda::Vector{Float32}    # per-channel; allows multi-timescale spread
end

function PhasorResonant(shape::Pair{<:Integer,<:Integer},
                        activation = normalize_to_unit_circle;
                        use_bias::Bool = false,
                        init_weight = nothing,
                        init_bias = default_bias,
                        omega::Real = 2.0f0 * Float32(π),
                        init_log_neg_lambda = log(0.1))
    if init_weight === nothing
        init_weight = glorot_uniform
    end
    out = shape[2]
    lnl_vec = if init_log_neg_lambda isa Real
        fill(Float32(init_log_neg_lambda), out)
    else
        @assert length(init_log_neg_lambda) == out "init_log_neg_lambda must have length $(out), got $(length(init_log_neg_lambda))"
        Float32.(collect(init_log_neg_lambda))
    end
    return PhasorResonant(shape[1], out, activation, use_bias,
                          init_weight, init_bias, Float32(omega), lnl_vec)
end

function Lux.initialparameters(rng::AbstractRNG, l::PhasorResonant)
    W = l.init_weight(rng, l.out_dims, l.in_dims)
    parameters = (weight = W, log_neg_lambda = copy(l.init_log_neg_lambda))
    if l.use_bias
        bias = l.init_bias(rng, (l.out_dims,))
        parameters = merge(parameters, (bias_real = Float32.(real.(bias)),
                                         bias_imag = Float32.(imag.(bias))))
    end
    return parameters
end

Lux.initialstates(::AbstractRNG, l::PhasorResonant) = (omega = l.omega,)

function Lux.parameterlength(l::PhasorResonant)
    n = l.out_dims * l.in_dims + l.out_dims  # weight + log_neg_lambda
    if l.use_bias
        n += 2 * l.out_dims
    end
    return n
end

function (a::PhasorResonant)(x::AbstractArray{<:Complex, 3}, ps::LuxParams, st::NamedTuple)
    C_in, L, B = size(x)

    λ = -exp.(ps.log_neg_lambda)
    # Shared carrier ω for every channel — phase-locked by construction.
    # Built on the same device as λ via `zero(λ) .+ scalar` so GPU
    # parameters get a CuArray ω without explicit device tracking.
    ω = zero(λ) .+ st.omega

    K = phasor_kernel(λ, ω, 1f0, L)

    xr = reshape(x, C_in, L * B)
    Hr = complex.(ps.weight * real.(xr), ps.weight * imag.(xr))
    H  = reshape(Hr, a.out_dims, L, B)

    Z = causal_conv(K, H)

    if a.use_bias
        bias_val = ps.bias_real .+ 1.0f0im .* ps.bias_imag
        G = bias_kernel_accumulation(λ, ω, 1f0, L)                       # (C_out, L)
        Z = Z .+ reshape(bias_val, :, 1, 1) .* reshape(G, size(G, 1), size(G, 2), 1)
    end

    # complex_to_angle is magnitude-independent, so when the activation is
    # the angle-preserving normalize_to_unit_circle we can skip it.
    if a.activation === normalize_to_unit_circle
        return complex_to_angle(Z), st
    else
        Y = a.activation(Z)
        return complex_to_angle(Y), st
    end
end

###
### ResonantSTFT — Trainable Frequency Decomposition
###

"""
    ResonantSTFT <: Lux.AbstractLuxLayer

Multi-compartment neuron layer that performs trainable frequency decomposition
(STFT) on complex phasor input sequences, then re-encodes the result at a
uniform carrier frequency for downstream synchronized layers.

Each frequency channel acts as a two-compartment neuron:
- **Signal compartment**: driven by weighted input, evolves at trainable `(λ, ω)`
- **Reference compartment**: free-running at the same `(λ, ω)`, analytically known

The invariant phase — the difference between signal and reference — represents
the input's spectral content at frequency `ω`. This is re-encoded at the
downstream carrier `omega_out` via frequency-shift modulation:

    z_out[n] = z_sig[n] * exp(i * (ω_out - ω_f) * n * Δt)

Dispatches on input dimensionality (3D only — STFT is inherently temporal):
- 3D complex `(C, L, B)` → weight + causal_conv + freq_shift + activation
- 3D Phase `(C, L, B)` → causal_conv_dirac + freq_shift + complex_to_angle

# Fields
- `in_dims::Int`: Input feature dimension
- `n_freqs::Int`: Number of frequency analysis channels (output dimension)
- `activation::Function`: Normalization function (Complex → Complex)
- `use_bias::Bool`: Whether to apply complex bias
- `init_weight::Function`: Weight initializer `(rng, out, in) -> Matrix`
- `init_bias::Function`: Bias initializer `(rng, dims) -> ComplexF32 array`
- `omega_lo::Float32`: Lower bound for omega initialization
- `omega_hi::Float32`: Upper bound for omega initialization
- `omega_out::Float32`: Downstream carrier frequency (fixed in state)
- `init_log_neg_lambda::Float32`: Initial value used to fill `log_neg_lambda`
  for every channel (default `log(0.1)` ⇒ `λ = -0.1`). Override at long
  sequence lengths `L` where `λ · L < log(eps(Float32)) ≈ -88` would cause
  `phasor_kernel` to underflow.

# State
- `omega_out` — Float32, downstream carrier frequency (fixed)

See also: [`PhasorDense`](@ref)
"""
struct ResonantSTFT <: Lux.AbstractLuxLayer
    in_dims::Int
    n_freqs::Int
    activation::Function
    use_bias::Bool
    init_weight::Function
    init_bias::Function
    omega_lo::Float32
    omega_hi::Float32
    omega_out::Float32
    init_log_neg_lambda::Float32
end

function ResonantSTFT(shape::Pair{<:Integer,<:Integer},
                    activation = normalize_to_unit_circle;
                    use_bias::Bool = false,
                    init_weight = nothing,
                    init_bias = default_bias,
                    omega_lo::Real = 0.2f0,
                    omega_hi::Real = 2.5f0,
                    omega_out::Real = Float32(2π),
                    init_log_neg_lambda::Real = log(0.1))
    if init_weight === nothing
        init_weight = glorot_uniform
    end
    return ResonantSTFT(shape[1], shape[2],
                      activation,
                      use_bias,
                      init_weight,
                      init_bias,
                      Float32(omega_lo),
                      Float32(omega_hi),
                      Float32(omega_out),
                      Float32(init_log_neg_lambda))
end

# Frequency-shift modulation: re-encode from per-channel omega to uniform omega_out
function _freq_shift(Z::AbstractArray{<:Complex, 3},
                     omega::AbstractVector,
                     omega_out::Real,
                     Δt::Real)
    n_freqs, L, B = size(Z)
    Δω = Float32(omega_out) .- omega                        # (n_freqs,)
    ns_cpu = Float32.(0:L-1)
    ns = reshape(typeof(omega)(ns_cpu), 1, L)               # (1, L), GPU-safe
    shift = exp.(1.0f0im .* Δω .* Float32(Δt) .* ns)       # (n_freqs, L)
    return Z .* reshape(shift, n_freqs, L, 1)               # (n_freqs, L, B)
end

function Lux.initialparameters(rng::AbstractRNG, l::ResonantSTFT)
    W = l.init_weight(rng, l.n_freqs, l.in_dims)
    log_neg_lambda = fill(l.init_log_neg_lambda, l.n_freqs)
    omega = Float32.(collect(range(l.omega_lo, l.omega_hi; length=l.n_freqs)))

    parameters = (weight = W, log_neg_lambda = log_neg_lambda, omega = omega)

    if l.use_bias
        bias = l.init_bias(rng, (l.n_freqs,))
        parameters = merge(parameters, (bias_real = Float32.(real.(bias)),
                                         bias_imag = Float32.(imag.(bias))))
    end

    return parameters
end

function Lux.initialstates(rng::AbstractRNG, l::ResonantSTFT)
    return (omega_out = l.omega_out,)
end

function Lux.parameterlength(l::ResonantSTFT)
    n = l.n_freqs * l.in_dims + l.n_freqs + l.n_freqs  # weight + log_neg_lambda + omega
    if l.use_bias
        n += 2 * l.n_freqs
    end
    return n
end

# ---- 3D Complex dispatch ----

function (a::ResonantSTFT)(x::AbstractArray{<:Complex, 3}, params::LuxParams, state::NamedTuple)
    C_in, L, B = size(x)

    λ = -exp.(params.log_neg_lambda)
    ω = params.omega

    # Build impulse-response kernel for each frequency channel
    K = phasor_kernel(λ, ω, 1f0, L)

    # Weight mixing: project input channels to frequency channels
    xr = reshape(x, C_in, L * B)
    Hr = complex.(params.weight * real.(xr), params.weight * imag.(xr))
    H = reshape(Hr, a.n_freqs, L, B)

    # Temporal integration via causal convolution
    Z_sig = causal_conv(K, H)

    # Per-period bias drive accumulated through the SSM kernel at the
    # signal-compartment frequency `ω`, then carried through `_freq_shift`
    # like the rest of the signal (matches the ODE semantics; bias is a
    # constant current injected into the same dynamics, not a post-hoc
    # offset).
    if a.use_bias
        bias_val = params.bias_real .+ 1.0f0im .* params.bias_imag
        G = bias_kernel_accumulation(λ, ω, 1f0, L)                       # (n_freqs, L)
        Z_sig = Z_sig .+ reshape(bias_val, :, 1, 1) .* reshape(G, size(G, 1), size(G, 2), 1)
    end

    # Frequency shift: re-encode at downstream carrier omega_out
    Z = _freq_shift(Z_sig, ω, state.omega_out, 1f0)

    Y = a.activation(Z)
    return Y, state
end

# ---- 3D Phase dispatch ----

function (a::ResonantSTFT)(x::AbstractArray{<:Phase, 3}, params::LuxParams, state::NamedTuple)
    λ = -exp.(params.log_neg_lambda)
    ω = params.omega
    L = size(x, 2)

    # Dirac discretization: causal convolution on phase inputs
    Z_sig = causal_conv_dirac(x, params.weight, λ, ω, 1f0)

    if a.use_bias
        bias_val = params.bias_real .+ 1.0f0im .* params.bias_imag
        G = bias_kernel_accumulation(λ, ω, 1f0, L)
        Z_sig = Z_sig .+ reshape(bias_val, :, 1, 1) .* reshape(G, size(G, 1), size(G, 2), 1)
    end

    # Frequency shift: re-encode at downstream carrier omega_out
    Z = _freq_shift(Z_sig, ω, state.omega_out, 1f0)

    if a.activation === normalize_to_unit_circle
        return complex_to_angle(Z), state
    else
        Y = a.activation(Z)
        return complex_to_angle(Y), state
    end
end

###
### Convolutional Phasor Layer
###

"""
    PhasorConv <: Lux.AbstractLuxLayer

A convolutional layer for phase-valued and spiking neural networks.
Flat structure with per-channel oscillator dynamics, matching PhasorDense.

# Fields
- `_conv`: Internal Conv layer for convolution mechanics
- `activation::Function`: Normalization function (Complex → Complex)
- `use_bias::Bool`: Whether to apply complex bias
- `init_bias::Function`: Bias initializer
- `init_mode::Symbol`: Dynamics initialization (`:default` or `:hippo`)
- `return_type::SolutionType`: Output format for spiking inputs
- `init_log_neg_lambda::Union{Float32, Nothing}`: Optional uniform override
  for `log_neg_lambda`; semantics match [`PhasorDense`](@ref).

# Parameters
- `weight` — Conv weight tensor
- `log_neg_lambda` — `(out_chs,)` per-channel decay
- `bias_real`, `bias_imag` — bias (when `use_bias=true`)

# State
- `omega` — `(out_chs,)` Float32, fixed at `2π` (per-channel ω rule —
  see [`PhasorDense`](@ref) for the full rationale).

!!! note "Architectural direction"
    PhasorConv still accepts complex-valued inputs and runs the ZOH SSM
    internally — mirroring the pre-refactor PhasorDense layout. The
    intended direction (already applied to `PhasorDense`) is to split
    that responsibility off into a dedicated complex→phase encoder layer
    (cf. [`PhasorResonant`](@ref)) and have `PhasorConv` operate purely
    in the phase domain. When you next touch this layer, consider doing
    the same split: keep the Phase paths here, move the complex-input
    SSM kernel into a `PhasorResonantConv` (or similar) sibling.
"""
struct PhasorConv <: Lux.AbstractLuxLayer
    _conv  # Internal Conv for forward pass mechanics
    activation::Function
    use_bias::Bool
    init_bias::Function
    init_mode::Symbol
    return_type::SolutionType
    init_log_neg_lambda::Union{Float32, Nothing}
end

function PhasorConv(k::Tuple{Vararg{Integer}}, chs::Pair{<:Integer,<:Integer}, activation = normalize_to_unit_circle;
                    return_type::SolutionType = SolutionType(:spiking),
                    init_bias = default_bias,
                    use_bias::Bool = true,
                    init_mode::Symbol = :default,
                    init_log_neg_lambda::Union{Real, Nothing} = nothing,
                    kwargs...)
    conv = Conv(k, chs, identity; use_bias=false, kwargs...)
    init_mode in (:default, :hippo) ||
        throw(ArgumentError("init_mode must be :default or :hippo (got :$init_mode). " *
                            "The :uniform mode was removed because it spread ω across " *
                            "channels, which breaks the per-channel ω rule."))
    return PhasorConv(conv,
                      activation,
                      use_bias,
                      init_bias,
                      init_mode,
                      return_type,
                      init_log_neg_lambda === nothing ? nothing : Float32(init_log_neg_lambda))
end

# Helper to get output channels
_out_chs(l::PhasorConv) = l._conv.out_chs

# Per-mode (log_neg_lambda, omega) initializer. Same per-channel ω rule
# as PhasorDense: ω = 2π uniformly; modes differ only in λ layout.
function _init_conv_dynamics(l::PhasorConv)
    n = _out_chs(l)
    omega = fill(Float32(2π), n)
    log_neg_lambda = if l.init_mode == :hippo
        λ_init, _ = hippo_legs_diagonal(n)
        l.init_log_neg_lambda === nothing ?
            log.(-λ_init) : fill(l.init_log_neg_lambda, n)
    else  # :default
        ll = l.init_log_neg_lambda === nothing ? Float32(log(0.2)) : l.init_log_neg_lambda
        fill(ll, n)
    end
    return log_neg_lambda, omega
end

function Lux.initialparameters(rng::AbstractRNG, l::PhasorConv)
    conv_ps = Lux.initialparameters(rng, l._conv)
    log_neg_lambda, _ = _init_conv_dynamics(l)

    parameters = (weight = conv_ps.weight, log_neg_lambda = log_neg_lambda)

    if l.use_bias
        n = _out_chs(l)
        bias_dims = ([1 for _ in 1:length(size(conv_ps.weight))-2]..., n)
        bias = l.init_bias(rng, bias_dims)
        parameters = merge(parameters, (bias_real = Float32.(real.(bias)),
                                         bias_imag = Float32.(imag.(bias))))
    end

    return parameters
end

function Lux.initialstates(rng::AbstractRNG, l::PhasorConv)
    conv_st = Lux.initialstates(rng, l._conv)
    _, omega = _init_conv_dynamics(l)
    return (_conv = conv_st, omega = omega)
end

_get_omega(::PhasorConv, _params, state) = state.omega

# ---- Complex dispatch ----

function (pc::PhasorConv)(x::AbstractArray{<:Complex}, ps::LuxParams, st::NamedTuple)
    conv_ps = (weight = ps.weight,)
    conv_st = haskey(st, :_conv) ? st._conv : NamedTuple()
    y_real, _ = pc._conv(real.(x), conv_ps, conv_st)
    y_imag, _ = pc._conv(imag.(x), conv_ps, conv_st)
    y = y_real .+ 1.0f0im .* y_imag

    if pc.use_bias
        bias_val = ps.bias_real .+ 1.0f0im .* ps.bias_imag
        y = y .+ bias_val
    end

    return y, st
end

# ---- Phase dispatch ----

function (pc::PhasorConv)(x::AbstractArray{<:Phase}, ps::LuxParams, st::NamedTuple)
    xz = angle_to_complex(x)
    y, st_new = pc(xz, ps, st)
    y_normalized = pc.activation(y)
    y_phase = complex_to_angle(y_normalized)
    return y_phase, st_new
end

# ---- Spiking dispatch ----

function (a::PhasorConv)(x::SpikingCall, params::LuxParams, state::NamedTuple)
    current_call = CurrentCall(x)
    return a(current_call, params, state)
end

function (a::PhasorConv)(x::CurrentCall, params::LuxParams, state::NamedTuple)
    spk_args = x.spk_args
    tspan = x.t_span

    # Compute output shape by running conv on a sample
    conv_ps = (weight = params.weight,)
    conv_st = haskey(state, :_conv) ? state._conv : NamedTuple()
    sample_out, _ = a._conv(x.current.current_fn(Float32(tspan[1])), conv_ps, conv_st)
    u0 = similar(sample_out, ComplexF32)
    ignore_derivatives() do
        u0 .= zero(ComplexF32)
    end

    ω_val = _get_omega(a, params, state)
    use_bias = a.use_bias && haskey(params, :bias_real)

    function dzdt(u, p, t)
        λ = -exp.(p.log_neg_lambda)
        k = ComplexF32.(λ .+ im .* ω_val)
        # Reshape k for conv broadcasting: (1...1, out_chs, 1)
        k_shaped = reshape(k, [1 for _ in 1:ndims(u)-2]..., length(k), 1)
        conv_p = (weight = p.weight,)
        I_transformed, _ = a._conv(x.current.current_fn(t), conv_p, conv_st)
        result = k_shaped .* u .+ I_transformed
        if use_bias
            bias_val = p.bias_real .+ 1im .* p.bias_imag
            # bias_val is shaped for conv broadcasting
            result = result .+ bias_val
        end
        return result
    end

    prob = ODEProblem(dzdt, u0, tspan, params)
    sol = solve(prob, spk_args.solver, p=params; spk_args.solver_args...)

    if a.return_type.type == :phase
        u = unrotate_solution(sol.u, sol.t, spk_args=spk_args, offset=x.current.offset)
        y = a.activation.(u)
        phase = complex_to_angle.(y)
        return phase, state
    elseif a.return_type.type == :potential
        return sol, state
    elseif a.return_type.type == :current
        i_fn = t -> potential_to_current(sol(t), spk_args=spk_args)
        next_call = CurrentCall(LocalCurrent(i_fn, x.current.shape, x.current.offset + spiking_offset(spk_args)),
                                spk_args,
                                tspan)
        return next_call, state
    else #spiking
        train = solution_to_train(sol, tspan, spk_args=spk_args, offset=x.current.offset)
        next_call = SpikingCall(train, spk_args, tspan)
        return next_call, state
    end
end

###
### Codebook layer - converts a vector to a value of similarities
###

"""
    Codebook(d => n; init_mode = :random) <: LuxCore.AbstractLuxLayer

Layer that holds `n` fixed `d`-dimensional phase codes and returns
similarities with its input.

# Arguments
- `d => n`: input feature dimension `d` paired with number of codes `n`.
- `init_mode::Symbol`: how the codebook is initialized.
    - `:random` (default) — i.i.d. uniform phases via [`random_symbols`](@ref).
      Pairwise similarities have standard deviation `O(1/√d)`; reliable
      separation only when `d` is large relative to `n`.
    - `:orthogonal` — DFT-shifted codes via [`orthogonal_codes`](@ref).
      Pairwise similarities are exactly zero when `d` is divisible by `n`,
      and bounded by `O(n/d)` otherwise. Requires `n ≤ d`. Useful when
      `d` is small enough that random initialization may not give enough
      separation between classes.

# Fields
- `dims::Pair{Int,Int}`: `d => n`.
- `init_mode::Symbol`: `:random` or `:orthogonal`.

# State
- `codes::Array{Phase, 2}` of shape `(d, n)` — fixed after initialization
  (non-trainable).

# Forward Pass
- Phase inputs: returns `similarity_outer(input, codes)`.
- Spiking inputs: converts codes to currents and returns temporal
  similarity.

See also: [`similarity_outer`](@ref), [`orthogonal_codes`](@ref),
[`random_symbols`](@ref).
"""
struct Codebook <: LuxCore.AbstractLuxLayer
    dims
    init_mode::Symbol
end

function Codebook(x::Pair{<:Int, <:Int}; init_mode::Symbol = :random)
    init_mode in (:random, :orthogonal) || throw(ArgumentError(
        "Codebook init_mode must be :random or :orthogonal, got :$init_mode"))
    return Codebook(x, init_mode)
end

function Base.show(io::IO, cb::Codebook)
    print(io, "Codebook($(cb.dims); init_mode=:$(cb.init_mode))")
end

function Lux.initialparameters(rng::AbstractRNG, cb::Codebook)
    return NamedTuple()
end

function Lux.initialstates(rng::AbstractRNG, cb::Codebook)
    d, n = cb.dims
    codes = cb.init_mode === :orthogonal ?
        orthogonal_codes(rng, d, n) :
        random_symbols(rng, (d, n))
    return (codes = codes,)
end

function (cb::Codebook)(x::AbstractArray{<:Phase}, params::LuxParams, state::NamedTuple)
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
    PhasorFixed <: Lux.AbstractLuxLayer

A dense layer with non-trainable (fixed) weights for phase-valued and spiking networks.
Weights and bias are stored in state, dynamics parameters in params.

# Fields
- `in_dims::Int`, `out_dims::Int`: Layer dimensions
- `activation::Function`: Normalization function
- `use_bias::Bool`: Whether to apply complex bias
- `init_weight`: Weight initializer or `nothing` for glorot_uniform
- `init_bias::Function`: Bias initializer
- `init_mode::Symbol`: Dynamics init (`:default` or `:hippo`)
- `return_type::SolutionType`: Output format for spiking inputs

# Parameters
- `log_neg_lambda` — per-channel decay (always trainable)

# State
- `weight` — fixed weight matrix
- `bias_real`, `bias_imag` — bias (when `use_bias=true`)
- `omega` — `(out,)` Float32, fixed at `2π` (per-channel ω rule).
"""
struct PhasorFixed <: Lux.AbstractLuxLayer
    in_dims::Int
    out_dims::Int
    activation::Function
    use_bias::Bool
    init_weight
    init_bias::Function
    init_mode::Symbol
    return_type::SolutionType
end

function PhasorFixed(shape::Pair{<:Integer,<:Integer}, activation = normalize_to_unit_circle;
                     return_type::SolutionType = SolutionType(:spiking),
                     init_bias = default_bias,
                     use_bias::Bool = false,
                     init_weight = nothing,
                     init_mode::Symbol = :default,
                     kwargs...)
    init_mode in (:default, :hippo) ||
        throw(ArgumentError("init_mode must be :default or :hippo (got :$init_mode). " *
                            "The :uniform mode was removed because it spread ω across " *
                            "channels, which breaks the per-channel ω rule."))
    return PhasorFixed(shape[1], shape[2],
                       activation,
                       use_bias,
                       init_weight,
                       init_bias,
                       init_mode,
                       return_type)
end

function _init_dynamics(l::PhasorFixed)
    omega = fill(Float32(2π), l.out_dims)
    log_neg_lambda = if l.init_mode == :hippo
        λ_init, _ = hippo_legs_diagonal(l.out_dims)
        log.(-λ_init)
    else  # :default
        fill(Float32(log(0.2)), l.out_dims)
    end
    return log_neg_lambda, omega
end

function Lux.initialparameters(rng::AbstractRNG, l::PhasorFixed)
    log_neg_lambda, _ = _init_dynamics(l)
    return (log_neg_lambda = log_neg_lambda,)
end

function Lux.initialstates(rng::AbstractRNG, l::PhasorFixed)
    # Weight in state (non-trainable)
    if l.init_weight !== nothing
        W = l.init_weight(rng, l.in_dims, l.out_dims)
    else
        W = glorot_uniform(rng, l.out_dims, l.in_dims)
    end
    state = (weight = W,)

    # Bias in state (non-trainable)
    if l.use_bias
        bias = l.init_bias(rng, (l.out_dims,))
        state = merge(state, (bias_real = Float32.(real.(bias)),
                               bias_imag = Float32.(imag.(bias))))
    end

    _, omega = _init_dynamics(l)
    state = merge(state, (omega = omega,))
    return state
end

_get_omega(::PhasorFixed, _params, state) = state.omega

# ---- Complex dispatch (weights from state) ----

function (a::PhasorFixed)(x::AbstractArray{<:Complex}, params::LuxParams, state::NamedTuple)
    y_real = state.weight * real.(x)
    y_imag = state.weight * imag.(x)
    y = y_real .+ 1.0f0im .* y_imag

    if a.use_bias
        bias_val = state.bias_real .+ 1.0f0im .* state.bias_imag
        y = y .+ bias_val
    end

    return y, state
end

# ---- Phase dispatch ----

function (a::PhasorFixed)(x::AbstractArray{<:Phase}, params::LuxParams, state::NamedTuple)
    xz = angle_to_complex(x)
    y, st_new = a(xz, params, state)
    y_normalized = a.activation(y)
    y_phase = complex_to_angle(y_normalized)
    return y_phase, st_new
end

# ---- Spiking dispatch ----

function (a::PhasorFixed)(x::SpikingCall, params::LuxParams, state::NamedTuple)
    current_call = CurrentCall(x)
    return a(current_call, params, state)
end

function (a::PhasorFixed)(x::CurrentCall, params::LuxParams, state::NamedTuple)
    spk_args = x.spk_args
    tspan = x.t_span

    # Compute output shape
    sample_I = x.current.current_fn(Float32(tspan[1]))
    out_shape = ndims(sample_I) >= 2 ? (a.out_dims, size(sample_I)[2:end]...) : (a.out_dims,)
    u0 = similar(sample_I, ComplexF32, out_shape)
    ignore_derivatives() do
        u0 .= zero(ComplexF32)
    end

    # Weight from state, dynamics from params
    W = state.weight
    use_bias = a.use_bias && haskey(state, :bias_real)

    function dzdt(u, p, t)
        λ = -exp.(p.log_neg_lambda)
        ω_val = _get_omega(a, p, state)
        k = ComplexF32.(λ .+ im .* ω_val)
        I_transformed = W * x.current.current_fn(t)
        result = k .* u .+ I_transformed
        if use_bias
            bias_val = state.bias_real .+ 1im .* state.bias_imag
            result = result .+ bias_current(bias_val, t, x.current.offset, spk_args)
        end
        return result
    end

    prob = ODEProblem(dzdt, u0, tspan, params)
    sol = solve(prob, spk_args.solver, p=params; spk_args.solver_args...)

    if a.return_type.type == :phase
        u = unrotate_solution(sol.u, sol.t, spk_args=spk_args, offset=x.current.offset)
        y = a.activation.(u)
        phase = complex_to_angle.(y)
        return phase, state
    elseif a.return_type.type == :potential
        return sol, state
    elseif a.return_type.type == :current
        i_fn = t -> potential_to_current(sol(t), spk_args=spk_args)
        next_call = CurrentCall(LocalCurrent(i_fn, x.current.shape, x.current.offset + spiking_offset(spk_args)),
                                spk_args,
                                tspan)
        return next_call, state
    else #spiking
        train = solution_to_train(sol, tspan, spk_args=spk_args, offset=x.current.offset)
        next_call = SpikingCall(train, spk_args, tspan)
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
    layers = [PhasorDense(pair, activation; kwargs...) for pair in pairs]
    ff = Chain(layers...)

    return ResidualBlock(ff)
end

function (rb::ResidualBlock)(x, ps, st)
    # MLP path
    ff_out, st_ff = rb.ff(x, ps.ff, st.ff)
    y = v_bind(x, ff_out)
    
    return y, (ff=st_ff,)
end

"""
Phasor QKV Attention
"""


function score_scale(scores::AbstractArray{<:Real,3}; scale::AbstractVector{<:Real})
    #this function takes interference scores on [-1,1] and exponentially scales them
    d_k = size(scores,1)
    return exp.(scale .* scores) ./ d_k
end

function score_scale(potential::AbstractArray{<:Complex,3}, scores::AbstractArray{<:Real,3}; scale::AbstractVector)
    #this function takes an array of oscillator potentials on the complex domain
    # and multiplies them by using the real-valued interference scores transformed by the scaled exponential
    @assert size(potential, 3) == size(scores,3) "Batch dimensions of inputs must match"

    scores = permutedims(scores, (2,1,3))
    d_k = size(scores,1)
    scores = exp.(scale .* scores) ./ d_k
    scaled = batched_mul(potential, scores)
    return scaled
end

function attend(q::AbstractArray{<:Phase, 3}, k::AbstractArray{<:Phase, 3}, v::AbstractArray{<:Phase, 3}; scale::AbstractArray=[1.0f0,])
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
function attend(q::SpikingTypes, k::SpikingTypes, v::SpikingTypes; spk_args::SpikingArgs, tspan::Tuple{<:Real, <:Real}=(0.0f0, 10.0f0), return_solution::Bool = false, scale::AbstractArray=[3.0f0,])
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
    return PhasorAttention(3.0f0)
end

function Lux.initialparameters(rng::AbstractRNG, attention::PhasorAttention)
    params = (scale = [attention.init_scale,],)
end

function Lux.initialstates(rng::AbstractRNG, attention::PhasorAttention)
    return NamedTuple()
end

function (a::PhasorAttention)(q::AbstractArray{<:Phase,3}, k::AbstractArray{<:Phase,3}, v::AbstractArray{<:Phase,3}, ps::LuxParams, st::NamedTuple)
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
    attn_mask::AbstractArray
    q_norm
    kv_norm
    ff_norm
    ff
end

function SingleHeadCABlock(d_input::Int, d_model::Int, n_q::Int, n_kv::Int; attn_mask=ones(Float32, n_q, n_kv), dropout::Real=0.1f0, kwargs...)
    SingleHeadCABlock(
        SingleHeadAttention(d_input, d_model; kwargs...),
        attn_mask,
        LayerNorm((d_model, n_q)),
        LayerNorm((d_model, n_kv)),
        LayerNorm((d_model, n_q)),
        Chain(PhasorDense(d_input => d_model),
            Dropout(dropout),
            PhasorDense(d_model => d_input)),
    )
end

function (tb::SingleHeadCABlock)(q, kv, ps, st)
    # Attention path
    #norm_q = tb.q_norm(q, ps.q_norm, st.q_norm)[1]
    #norm_kv = tb.kv_norm(kv, ps.kv_norm, st.kv_norm)[1]
    attn_out, st_attn = tb.attn(q, kv, ps.attn, st.attn)
    attn_masked = attn_out .* tb.attn_mask
    x = v_bind(q, attn_masked)
    
    # Feed-forward path
    #norm_x = tb.ff_norm(x, ps.ff_norm, st.ff_norm)[1]
    ff_out, st_ff = tb.ff(x, ps.ff, st.ff)
    x = v_bind(x, ff_out)
    
    return x, merge(st_attn, st_ff)
end

"""
    _adjust_ssm_lr!(opt_state, ps, lr_ssm)

Walk the optimizer state tree and set the learning rate for SSM dynamics parameters
(`log_neg_lambda`, `omega`) to `lr_ssm`, leaving all other parameters unchanged.
"""
function _adjust_ssm_lr!(opt_state, ps, lr_ssm)
    ssm_keys = (:log_neg_lambda, :omega)
    for (kp, _) in Optimisers.trainables(ps, path=true)
        if last(kp.keys) in ssm_keys
            node = opt_state
            for k in kp.keys
                node = getproperty(node, k)
            end
            Optimisers.adjust!(node, lr_ssm)
        end
    end
end

"""
    _apply_weight_decay(gs, ps, wd)

Add L2 weight decay to gradients for `:weight` parameters only.
Returns a modified gradient tree. SSM dynamics parameters (log_neg_lambda, omega)
and bias parameters are not penalized.
"""
function _apply_weight_decay(gs, ps, wd)
    decay_keys = (:weight,)
    for (kp, param) in Optimisers.trainables(ps, path=true)
        if last(kp.keys) in decay_keys
            # Navigate to the corresponding gradient leaf
            g_node = gs
            for k in kp.keys[1:end-1]
                g_node = getproperty(g_node, k)
            end
            old_g = getproperty(g_node, last(kp.keys))
            # gs is a NamedTuple tree — we can't mutate it, so we accumulate
            # the decay into the gradient via broadcasting (creates new array)
            new_g = old_g .+ eltype(old_g)(wd) .* param
            # Rebuild the leaf. Since NamedTuples are immutable, we use
            # Optimisers.trainables to identify locations but apply decay
            # in-place on the gradient array (which IS mutable).
            old_g .= new_g
        end
    end
    return gs
end

"""
    train(model, ps, st, train_loader, loss, args; optimiser=Optimisers.Adam, verbose=false, sample_gradients=0, early_stop=false, early_stop_window=5)

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
- `early_stop::Bool`: Stop training if the mean loss over the last `early_stop_window`
  batches exceeds the mean over the preceding `early_stop_window` batches (default: false)
- `early_stop_window::Int`: Number of batches in each comparison window (default: 5).
  Requires at least `2 * early_stop_window` completed batches before checking.

# Returns
- `losses`: Array of loss values during training
- `ps`: Updated parameters
- `st`: Updated state
- `gradients`: Sampled gradients (only when `sample_gradients > 0`)

Automatically handles CPU/GPU device placement based on args.use_cuda.

# Training features
- **Differential LR**: Set `args.lr_ssm` to use a lower learning rate for SSM dynamics
  parameters (`log_neg_lambda`, `omega`) vs connection weights.
- **Weight decay**: Set `args.weight_decay > 0` to apply L2 regularization to weight
  matrices only (not SSM dynamics parameters).
- **Cosine schedule**: Set `args.cosine_schedule = true` to anneal learning rates from
  initial values down to `args.lr_min` over training.
- **GC interval**: Set `args.gc_interval > 0` to reduce garbage collection frequency
  (useful for SSM convolution workloads that don't accumulate ODE solution objects).
"""
function train(model, ps, st, train_loader, loss, args;
               optimiser = Optimisers.Adam,
               verbose::Bool = false,
               sample_gradients::Int = 0,
               early_stop::Bool = false,
               early_stop_window::Int = 5)
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

   # Differential learning rates for SSM dynamics parameters
   use_ssm_lr = args.lr_ssm > 0 && args.lr_ssm != args.lr
   if use_ssm_lr
       _adjust_ssm_lr!(opt_state, ps, args.lr_ssm)
   end

   # Precompute total steps for cosine schedule
   use_cosine = args.cosine_schedule
   total_steps = args.epochs * length(train_loader)

   # GC interval
   gc_every = args.gc_interval > 0 ? args.gc_interval : 1

   losses = []
   gradients = []
   step_count = 0
   stopped_early = false

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

           # Apply weight decay to gradient (weights only, not SSM params)
           if args.weight_decay > 0
               _apply_weight_decay(gs[1], ps, args.weight_decay)
           end

           opt_state, ps = Optimisers.update(opt_state, ps, gs[1]) ## update parameters

           # Cosine LR schedule: anneal from initial LR to lr_min
           if use_cosine
               progress = step_count / total_steps
               cos_mult = 0.5 * (1.0 + cos(Float64(pi) * progress))
               lr_t = args.lr_min + (args.lr - args.lr_min) * cos_mult
               Optimisers.adjust!(opt_state, lr_t)
               if use_ssm_lr
                   lr_ssm_t = args.lr_min + (args.lr_ssm - args.lr_min) * cos_mult
                   _adjust_ssm_lr!(opt_state, ps, lr_ssm_t)
               end
           end

           # Early stopping: compare mean of last window vs preceding window
           if early_stop && length(losses) >= 2 * early_stop_window
               w = early_stop_window
               current_avg = sum(losses[end-w+1:end]) / w
               prev_avg    = sum(losses[end-2w+1:end-w]) / w
               if current_avg > prev_avg
                   verbose && @info "Early stopping at step $step_count: mean loss increased from $(round(prev_avg, digits=6)) to $(round(current_avg, digits=6))"
                   stopped_early = true
                   break
               end
           end

           # Prompt Julia's GC to reclaim Zygote tape and ODE solution objects from
           # this step before the next forward/backward solve begins.  Without this,
           # large sol.u arrays (one per saveat point) accumulate on the host and GPU.
           if step_count % gc_every == 0
               GC.gc(false)
               CUDA.functional() && CUDA.reclaim()
           end
       end
       stopped_early && break
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
    # Default constructor creates a MaxPool with default parameters
    return MinPool(MaxPool())
end

# Additional constructors to allow specifying the pooling size directly.
# Accept a tuple of dimensions (e.g., (2, 2)) and construct the underlying
# MaxPool layer accordingly.
function MinPool(pool_size::Tuple{Vararg{Int}})
    return MinPool(MaxPool(pool_size))
end

# Allow constructing from an existing MaxPool instance.
# This is the inner constructor which will be called automatically
# function MinPool(pool::MaxPool)
#     # The default struct constructor will be used
# end

function (mp::MinPool)(x, ps, st)
    # Apply the underlying MaxPool (stored in `mp.pool`) to the negated input,
    # then negate the result to achieve min‑pooling behavior.
    y, st_new = mp.pool(-1.0f0 .* x, ps, st)
    return -1.0f0 .* y, st_new
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