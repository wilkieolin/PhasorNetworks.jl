# ================================================================
# AttractorPhasorSSM — selective recurrent layer with attractor coupling
# ================================================================
#
# Extends the time-invariant linear SSM
#
#     dz/dt = (λ + iω)·z + W·I(t)
#
# with a Hopfield-style attractor pull toward learned phasor codes:
#
#     dz/dt = (λ + iω)·z + W·I(t) + α·(pull(z, codes) − z)
#     pull(z, C) = C · softmax(β · sim(z, C))
#
# The codes are stored phasor patterns; α controls pull strength
# (sigmoid of a trainable scalar so it's in (0, 1)); β controls
# retrieval sharpness (exp of a trainable scalar so it's positive).
# With α = 0 the layer reduces to today's linear SSM exactly.
#
# Two dispatch paths are provided, mirroring PhasorDense:
#   • Phase 3D — explicit per-step recurrence using the same Dirac
#     discretization causal_conv_dirac uses (kernels.jl), interleaved
#     with the attractor pull. Operator-split, first-order accurate
#     in the period T.
#   • CurrentCall — extends the dzdt closure with the pull term and
#     hands it off to DifferentialEquations.jl, same as PhasorDense.
#   • SpikingCall — trampoline through CurrentCall.
#
# See also: PhasorDense, Codebook, similarity_outer.

"""
    AttractorPhasorSSM(in_dims => state_dims, n_codes; kwargs...)

A phasor SSM layer with a Hopfield-style content-addressable memory
of `n_codes` learnable phasor patterns. Each forward step blends a
linear SSM update with a softmax-weighted retrieval from the code
bank, producing **selective, content-addressable persistence** that
the bare linear SSM cannot express.

# Keyword arguments

- `use_bias::Bool = true` — learnable bias on the input projection.
- `init_weight::Function = glorot_uniform` — `(rng, out, in) -> Matrix`.
- `init_bias::Function = default_bias` — `(rng, out) -> Vector`.
- `init_codes::Symbol = :random` — `:random` (uniform Phase) or
  `:orthogonal` (DFT-shifted, near-orthogonal). Requires `state_dims ≥ n_codes`
  for `:orthogonal`.
- `init_log_neg_lambda::Union{Real,AbstractVector,Nothing} = nothing` —
  per-channel `log(-λ)` override (vector or scalar). When `nothing`,
  defaults to `log(0.2)` per channel (mild decay).
- `init_log_alpha::Real = log(0.4)` — initial value of `log_alpha`
  (sigmoided to `α ≈ 0.4` pull strength). Set very negative to start
  near linear-only.
- `init_log_beta::Real = log(8.0)` — initial value of `log_beta`
  (exponentiated to `β ≈ 8.0` softmax sharpness).
- `period::Real = 1.0` — discrete sample interval `T`.
- `trainable_codes::Bool = true` — codes in `params` (true) or `state`
  (false).

# Parameter / state layout

| name | shape | location |
|------|-------|----------|
| `weight` | `(state_dims, in_dims)` Float32 | params |
| `bias_real`, `bias_imag` | `(state_dims,)` Float32 | params (if `use_bias`) |
| `log_neg_lambda` | `(state_dims,)` Float32 | params |
| `log_alpha` | scalar Float32 | params |
| `log_beta` | scalar Float32 | params |
| `codes` | `(state_dims, n_codes)` Phase | params if `trainable_codes`, else state |
| `omega` | `(state_dims,)` Float32 | state (per-channel ω rule, frozen at 2π) |
"""
struct AttractorPhasorSSM <: Lux.AbstractLuxLayer
    in_dims::Int
    state_dims::Int
    n_codes::Int
    use_bias::Bool
    init_weight::Function
    init_bias::Function
    init_codes::Symbol
    init_log_neg_lambda::Union{Vector{Float32}, Nothing}
    init_log_alpha::Float32
    init_log_beta::Float32
    period::Float32
    trainable_codes::Bool
end

function AttractorPhasorSSM(shape::Pair{<:Integer,<:Integer}, n_codes::Integer;
                             use_bias::Bool = true,
                             init_weight::Function = glorot_uniform,
                             init_bias::Function = default_bias,
                             init_codes::Symbol = :random,
                             init_log_neg_lambda::Union{Real,AbstractVector{<:Real},Nothing} = nothing,
                             init_log_alpha::Real = log(0.4),
                             init_log_beta::Real = log(8.0),
                             period::Real = 1.0,
                             trainable_codes::Bool = true)
    init_codes in (:random, :orthogonal) ||
        throw(ArgumentError("init_codes must be :random or :orthogonal, got :$init_codes"))
    state_dims = shape[2]
    lnl_vec = if init_log_neg_lambda === nothing
        nothing
    elseif init_log_neg_lambda isa Real
        fill(Float32(init_log_neg_lambda), state_dims)
    else
        @assert length(init_log_neg_lambda) == state_dims "init_log_neg_lambda must have length $(state_dims)"
        Float32.(collect(init_log_neg_lambda))
    end
    return AttractorPhasorSSM(shape[1], state_dims, Int(n_codes),
                               use_bias, init_weight, init_bias,
                               init_codes, lnl_vec,
                               Float32(init_log_alpha), Float32(init_log_beta),
                               Float32(period), trainable_codes)
end

function Base.show(io::IO, l::AttractorPhasorSSM)
    print(io, "AttractorPhasorSSM($(l.in_dims) => $(l.state_dims), $(l.n_codes); ")
    print(io, "init_codes=:$(l.init_codes), trainable_codes=$(l.trainable_codes), period=$(l.period))")
end

# ---- Lux interface -----------------------------------------------------

_init_codes(rng::AbstractRNG, l::AttractorPhasorSSM) =
    l.init_codes === :orthogonal ?
        orthogonal_codes(rng, l.state_dims, l.n_codes) :
        random_symbols(rng, (l.state_dims, l.n_codes))

function Lux.initialparameters(rng::AbstractRNG, l::AttractorPhasorSSM)
    W = l.init_weight(rng, l.state_dims, l.in_dims)
    lnl = l.init_log_neg_lambda === nothing ?
        fill(Float32(log(0.2)), l.state_dims) :
        copy(l.init_log_neg_lambda)
    base = (weight = W,
            log_neg_lambda = lnl,
            log_alpha = Float32[l.init_log_alpha],
            log_beta  = Float32[l.init_log_beta])
    if l.use_bias
        b_complex = l.init_bias(rng, (l.state_dims,))
        base = merge(base, (bias_real = real.(b_complex),
                            bias_imag = imag.(b_complex)))
    end
    if l.trainable_codes
        base = merge(base, (codes = _init_codes(rng, l),))
    end
    return base
end

function Lux.initialstates(rng::AbstractRNG, l::AttractorPhasorSSM)
    omega = fill(Float32(2π), l.state_dims)
    if l.trainable_codes
        return (omega = omega,)
    else
        return (omega = omega, codes = _init_codes(rng, l))
    end
end

# ---- Internal helpers --------------------------------------------------

# Sigmoid for bounding α ∈ (0, 1) without depending on NNlib.sigmoid_fast
# (keep AD-clean, broadcastable, stable for moderate magnitudes).
_sigmoid(x) = inv(one(x) + exp(-x))

# Resolve trainable vs. fixed codes, returning an (out, n) Phase matrix.
_get_codes(l::AttractorPhasorSSM, ps, st) =
    l.trainable_codes ? ps.codes : st.codes

# Resolve current bias (complex), or zero if absent.
function _get_bias(l::AttractorPhasorSSM, ps)
    if l.use_bias && haskey(ps, :bias_real)
        return ps.bias_real .+ 1im .* ps.bias_imag
    else
        return nothing
    end
end

"""
    attractor_pull(z::AbstractMatrix{<:Complex},
                   codes_cplx::AbstractMatrix{<:Complex}, β) -> Matrix

Hopfield-style soft retrieval. Given current state `z :: (D, B)` and
the code bank `codes_cplx :: (D, K)`, returns `(D, B)` — the
softmax-weighted convex combination of codes, where the softmax
operates across the K codes axis with sharpness `β`. Used inside the
layer's per-step update to compute the "target" the state is nudged
toward.
"""
function attractor_pull(z::AbstractMatrix{<:Complex},
                        codes_cplx::AbstractMatrix{<:Complex},
                        β)
    sims = similarity_outer(z, codes_cplx)        # (K, B) Real (CPU 2D Complex dispatch)
    weights = softmax(β .* sims; dims = 1)        # (K, B)
    return codes_cplx * weights                   # (D, B)
end

# ---- Discrete dispatch: Phase 3D --------------------------------------
#
# Per-step recurrence using the Dirac discretization (matches
# causal_conv_dirac). Reuses `_exp_kdt(k, dt)` for the per-period spike
# kick — it's exactly the same broadcast as kernels.jl line 283, but
# evaluated one timestep at a time so we can interleave the attractor
# pull. The full-period decay `exp(k·T)` carries the previous state
# forward.
#
# Built with non-mutating accumulation (`Base.foldl` over 1:L,
# accumulating tuple of (z, list_of_outputs)) for Zygote AD compatibility
# — no `push!` to a Vector.

function (l::AttractorPhasorSSM)(x::AbstractArray{<:Phase, 3},
                                  ps::LuxParams, st::NamedTuple)
    in_dims = l.in_dims
    D       = l.state_dims
    L       = size(x, 2)
    B       = size(x, 3)
    @assert size(x, 1) == in_dims "input channel mismatch: $(size(x,1)) vs $(in_dims)"

    λ      = -exp.(ps.log_neg_lambda)                          # (D,) Real
    ω_val  = st.omega                                           # (D,) Real
    k      = ComplexF32.(λ .+ 1im .* ω_val)                     # (D,) Cplx
    A_step = exp.(k .* l.period)                                # (D,) full-period decay
    α      = _sigmoid(ps.log_alpha[1])                          # scalar in (0,1)
    β      = exp(ps.log_beta[1])                                # scalar > 0
    W_c    = ComplexF32.(ps.weight)                             # (D, in_dims)
    bias_c = _get_bias(l, ps)                                   # nothing or (D,) Cplx

    codes_phase = _get_codes(l, ps, st)                         # (D, K) Phase
    codes_cplx  = angle_to_complex(codes_phase)                 # (D, K) Cplx

    # Initial state: zero on the same device as input weights.
    z0 = ignore_derivatives() do
        z = similar(W_c, ComplexF32, D, B)
        z .= zero(ComplexF32)
        return z
    end

    # Per-period Dirac kick for output channel D, given input phases (in_dims, B).
    function _step_kick(phases_t::AbstractMatrix)
        # phases_t :: (in_dims, B) Phase
        dt_t = l.period .* (0.5f0 .- Float32.(phases_t) ./ 2f0)  # (in_dims, B) Float32
        # _exp_kdt expects rank-3 inputs: k as (D,1,1), dt as (1,in,B).
        enc  = _exp_kdt(reshape(k, D, 1, 1),
                        reshape(dt_t, 1, in_dims, B))            # (D, in_dims, B)
        h_t  = dropdims(sum(reshape(W_c, D, in_dims, 1) .* enc; dims = 2);
                        dims = 2)                                # (D, B)
        if bias_c !== nothing
            h_t = h_t .+ bias_c                                  # broadcast (D,) over (D,B)
        end
        return h_t
    end

    # One recurrent step: linear SSM half + attractor pull half.
    function _step(z, phases_t)
        h_t   = _step_kick(phases_t)
        z_lin = A_step .* z .+ h_t                               # (D, B) linear half
        target = attractor_pull(z_lin, codes_cplx, β)            # (D, B) pull target
        z_new  = (1 - α) .* z_lin .+ α .* target
        return normalize_to_unit_circle(z_new)
    end

    # Non-mutating fold: accumulator is (z_current, list_of_outputs).
    # `map` over `1:L` returns the list of per-step outputs, but since
    # each step depends on the previous, we use a recursive fold.
    function _scan(z, t)
        if t > L
            return (z, ())
        end
        z_next = _step(z, x[:, t, :])
        rest_z, rest_outs = _scan(z_next, t + 1)
        return (rest_z, (complex_to_angle(z_next), rest_outs...))
    end

    _, outs_tuple = _scan(z0, 1)
    Y = stack(collect(outs_tuple); dims = 2)                     # (D, L, B) Phase
    return Y, st
end

# ---- Continuous dispatch: CurrentCall ---------------------------------
#
# Extends PhasorDense's dzdt closure with the attractor pull term.
# Solves the augmented ODE
#     dz/dt = k·z + W·I(t) + bias(t) + α·(pull(z, codes) − z)
# via the same DifferentialEquations setup PhasorDense uses
# (network.jl:455–520).

function (l::AttractorPhasorSSM)(x::CurrentCall,
                                  ps::LuxParams, st::NamedTuple)
    spk_args = x.spk_args
    tspan    = x.t_span
    L        = round(Int, (tspan[2] - tspan[1]) / spk_args.t_period)
    use_period_sampling = L > 1

    sample_I = x.current.current_fn(Float32(tspan[1]))
    out_shape = ndims(sample_I) >= 2 ? (l.state_dims, size(sample_I)[2:end]...) : (l.state_dims,)
    u0 = ignore_derivatives() do
        u = similar(sample_I, ComplexF32, out_shape)
        u .= zero(ComplexF32)
        return u
    end

    use_bias_local = l.use_bias && haskey(ps, :bias_real)
    codes_phase = _get_codes(l, ps, st)
    codes_cplx  = angle_to_complex(codes_phase)

    ω_val = st.omega

    function dzdt(u, p, t)
        λ = -exp.(p.log_neg_lambda)
        k = ComplexF32.(λ .+ 1im .* ω_val)
        I_transformed = p.weight * x.current.current_fn(t)
        result = k .* u .+ I_transformed
        if use_bias_local
            bias_val = p.bias_real .+ 1im .* p.bias_imag
            result = result .+ bias_current(bias_val, t, x.current.offset, spk_args)
        end
        # Attractor pull (continuous form: α·(pull(z) − z))
        α_eff = _sigmoid(p.log_alpha[1])
        β_eff = exp(p.log_beta[1])
        if ndims(u) >= 2
            target = attractor_pull(u, codes_cplx, β_eff)        # (D, B)
        else
            # 1D state: lift to (D, 1) for the pull, then drop back.
            target_2d = attractor_pull(reshape(u, :, 1), codes_cplx, β_eff)
            target = vec(target_2d)
        end
        return result .+ α_eff .* (target .- u)
    end

    prob = ODEProblem(dzdt, u0, tspan, ps)
    sol  = solve(prob, spk_args.solver, p = ps; spk_args.solver_args...)

    if use_period_sampling
        T = spk_args.t_period
        sample_times = Float32.([j * T for j in 1:L])
        samples = [sol(t) for t in sample_times]
        if ndims(samples[1]) >= 2
            Z = cat([reshape(s, size(s, 1), 1, size(s, 2)) for s in samples]...; dims = 2)
        else
            Z = reduce(hcat, [reshape(s, :, 1) for s in samples])
        end
        return complex_to_angle(Z), st
    else
        u_last = unrotate_solution(sol.u, sol.t, spk_args = spk_args, offset = x.current.offset)
        phase  = complex_to_angle.(u_last)
        return phase, st
    end
end

# ---- Spiking dispatch -------------------------------------------------

function (l::AttractorPhasorSSM)(x::SpikingCall,
                                  ps::LuxParams, st::NamedTuple)
    return l(CurrentCall(x), ps, st)
end
