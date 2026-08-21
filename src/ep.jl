# ================================================================
# Phasor Equilibrium Propagation (vanilla EP, phase-domain)
# ================================================================
#
# Implements vanilla equilibrium propagation on phase-based phasor
# networks. The state of each neuron is a unit-modulus complex value
# z = exp(i·π·θ); the "saturation" of vanilla EP's tanh is replaced
# by the topological constraint that states live on the unit circle
# (no holomorphic activation, so Liouville does not apply — see
# docs/phasor_ep_design.md).
#
# Current scope:
#   * PhasorDense layers only. A new layer type needs methods for
#     ep_drive / ep_feedback / ep_self_force / ep_hebbian (§2);
#     ep_energy_contribution is diagnostic and optional. Note that
#     phasor_settle also requires an `out_dims` field, so parameterless
#     layers (e.g. Codebook) cannot sit inside an EP chain — pull their
#     codes out and pass them to CodebookCost instead.
#   * Gradient estimators: StaticEP (one-sided or centered finite
#     difference) and LockinEP (temporal real-probe demodulation).
#   * Costs: SimilarityCost and CodebookCost, both batched.
#   * Self-energy: K_mode = :zero (default) or :stored. The :stored path
#     needs dt <= 0.1 or an omega override to settle at all.
#   * use_bias is supported and FD-verified, and is effectively required
#     in practice — real W plus an entrywise unit projection is
#     axis-preserving, so without a complex bias whole input classes stay
#     locked to one axis (see demos/lockin_demo.ipynb §7.1).
#
# Batching: states are `(out_dims,)` for a single sample or
# `(out_dims, B)` for a minibatch, on one code path. Every operation in
# _phasor_step is column-separable, so a batched settle is exactly B
# independent settles. The 1/B normalization lives on the Hebbian
# (§ep_hebbian), NOT on nudge_force — each sample must see the full
# per-sample nudge amplitude or the linear response shrinks by B.
#
# Energy function being descended:
#   Phi = sum_l Re<W_l z_{l-1}, z_l>  -  beta * C(z_L, y)
# with cost
#   C(z_L, y) = 1 - (1/d) * Re<y, z_L>.

# ================================================================
# 1. Cost types
# ================================================================

abstract type AbstractEPCost end

# ---- batch-shape helpers ----------------------------------------
# EP states are either a plain `(d,)` vector (single sample, the
# original Phase-1 interface) or a `(d, B)` matrix (minibatch). Every
# operation in `_phasor_step` is column-separable, so a batched settle
# is exactly B independent settles; these helpers keep the two shapes
# on one code path.
_feature_dim(z::AbstractVector) = length(z)
_feature_dim(z::AbstractMatrix) = size(z, 1)

_batch_size(z::AbstractVector) = 1
_batch_size(z::AbstractMatrix) = size(z, 2)

# Sum over the batch dimension, returning a `(d,)` vector either way.
_sum_batch(z::AbstractVector) = z
_sum_batch(z::AbstractMatrix) = vec(sum(z; dims = 2))

"""
    SimilarityCost(y::AbstractVector{<:Complex})

Cosine-distance cost against a fixed unit-complex target `y`:

    C(z, y) = 1 - (1/d) * Re<y, z>

The nudge force `(β/d) * y` uses the **real-parameter convention**
(no factor of 1/2); this is what matches finite differences on real
weights — see `docs/phasor_ep_design.md`, section "Subtle convention
point".
"""
struct SimilarityCost{T<:AbstractVecOrMat{<:Complex}} <: AbstractEPCost
    y::T
end

"""
    ep_loss(cost::AbstractEPCost, z_o) -> Float32

Scalar training loss at the free-phase equilibrium output `z_o` under
`cost`. Plugs into [`ep_train`](@ref) for per-epoch loss reporting.

Defined per-cost:

- `ep_loss(c::SimilarityCost, z_o) = 1 - Re⟨c.y, z_o⟩ / length(z_o)` —
  cosine-distance to the unit-modulus target.
- `ep_loss(c::CodebookCost, z_o)` — softmax cross-entropy of the
  per-class similarities against `c.y_onehot`; see [`CodebookCost`](@ref).
"""
function ep_loss end

# Real-parameter convention: factor of 1/d, NOT 1/(2d).
# Do not "fix" — see docs/phasor_ep_design.md.
#
# Batch convention (see `ep_hebbian`): the nudge is applied at FULL
# per-sample amplitude β — it is NOT divided by the batch size. Each
# column of `z_o` is an independent settle, and shrinking the nudge by
# B would shrink the linear response by B and destroy the
# finite-difference SNR the EP estimate depends on. The 1/B lives on
# the Hebbian instead. `ep_loss` does average over the batch, since
# that is a reporting quantity rather than a force.
nudge_force(c::SimilarityCost, z_o, β) = (Float32(β) / _feature_dim(z_o)) .* c.y

# Single-sample path keeps the original `dot` reduction verbatim. Both
# this loss and `fd_gradient_phasor` are cancellation-sensitive in
# Float32 (FD at ε=1e-5 against an O(1) loss), so changing the
# summation algorithm here visibly moves the FD oracle. Do not merge
# these two methods.
ep_loss(c::SimilarityCost, z_o::AbstractVector) =
    one(Float32) - real(dot(c.y, z_o)) / length(z_o)

function ep_loss(c::SimilarityCost, z_o::AbstractMatrix)
    d = size(z_o, 1)
    s = real.(sum(conj.(c.y) .* z_o; dims = 1)) ./ Float32(d)
    return one(Float32) - Float32(mean(s))
end

"""
    CodebookCost(codes::AbstractMatrix{<:Complex}, y_onehot::AbstractVector)
    CodebookCost(codes, y_class::Integer)

Softmax-cross-entropy cost over similarity to a complex codebook
of class codewords (one column per class):

    s_c = (1/d) · Re⟨code_c, z_o⟩
    C   = -Σ_c y_c · log softmax(s)_c

The codes are assumed to be unit-modulus complex (e.g. produced by
`angle_to_complex(codes_phase)` from `Codebook`'s state). The class
target is either a one-hot vector of length `n_classes`, or a class
index that's converted to one-hot internally.

Real-parameter convention (matches FD on real W):

    nudge_force(c, z_o, β) = -(β/d) · codes · (softmax(s) − y)

Sign flip relative to the SimilarityCost case is correct: we're
pulling z_o toward the target codeword, which means pulling AWAY
from the wrongly-active codewords (`err > 0`).
"""
struct CodebookCost{C<:AbstractMatrix{<:Complex}, T<:AbstractVecOrMat{<:Real}} <: AbstractEPCost
    codes::C
    y_onehot::T
end

function CodebookCost(codes::AbstractMatrix{<:Complex}, y_class::Integer)
    n_classes = size(codes, 2)
    y_oh = zeros(Float32, n_classes)
    @assert 1 <= y_class <= n_classes "y_class out of range: got $y_class, expected 1..$n_classes"
    y_oh[y_class] = one(Float32)
    return CodebookCost(codes, _match_device(codes, y_oh))
end

# Move a host-built array onto whatever device `ref` lives on. The one-hot
# target is assembled with scalar indexing (illegal on a GPU array), so it
# is always built on the host and then copied across.
_match_device(ref::AbstractArray, A::AbstractArray) = A
function _match_device(ref::AbstractGPUArray, A::AbstractArray)
    D = similar(ref, eltype(A), size(A))
    copyto!(D, A)
    return D
end

# Batched: a vector of 1-based class indices becomes a (n_classes, B)
# one-hot matrix, one column per sample.
function CodebookCost(codes::AbstractMatrix{<:Complex},
                      y_classes::AbstractVector{<:Integer})
    n_classes = size(codes, 2)
    @assert all(1 .<= y_classes .<= n_classes) "y_classes out of range: expected 1..$n_classes"
    y_oh = zeros(Float32, n_classes, length(y_classes))
    for (b, cls) in enumerate(y_classes)
        y_oh[cls, b] = one(Float32)
    end
    return CodebookCost(codes, _match_device(codes, y_oh))
end

function _codebook_logits(c::CodebookCost, z_o)
    d = _feature_dim(z_o)
    return real.(adjoint(c.codes) * z_o) ./ Float32(d)
end

# Column-wise softmax. `dims=1` matters: a global reduction would mix
# samples across the batch.
function _softmax(s::AbstractArray{<:Real})
    e = exp.(s .- maximum(s; dims = 1))
    return e ./ sum(e; dims = 1)
end

function ep_loss(c::CodebookCost, z_o)
    s = _codebook_logits(c, z_o)
    s_shift = s .- maximum(s; dims = 1)
    log_probs = s_shift .- log.(sum(exp.(s_shift); dims = 1))
    return -sum(c.y_onehot .* log_probs) / Float32(_batch_size(z_o))
end

# Full per-sample nudge amplitude — no 1/B here; see the note on
# `nudge_force(::SimilarityCost, ...)`.
function nudge_force(c::CodebookCost, z_o, β)
    s   = _codebook_logits(c, z_o)
    err = _softmax(s) .- c.y_onehot
    d   = _feature_dim(z_o)
    return -(Float32(β) / d) .* (c.codes * err)
end

# Cost placeholder for inference-only settles. `phasor_settle` requires
# an AbstractEPCost, but at β = 0 the cost is never consulted for the
# settle direction — this makes that explicit rather than passing a
# throwaway target.
struct NullCost <: AbstractEPCost end
nudge_force(::NullCost, z_o, β) = zero(z_o)
ep_loss(::NullCost, z_o)        = zero(Float32)

"""
    codebook_logits(codes::AbstractMatrix{<:Complex}, z_o) -> AbstractArray

Per-class similarity logits `s_c = (1/d)·Re⟨code_c, z_o⟩` for a settled
output state `z_o` (a `(d,)` vector or a `(d, B)` batch). Returns
`(n_classes,)` or `(n_classes, B)`.

This is the same kernel `CodebookCost` nudges against, exposed so that
a settled network can be scored without constructing a cost. Feeds
`predict(·, :similarity)` and `evaluate_accuracy` directly — note those
return **1-based** class indices.
"""
codebook_logits(codes::AbstractMatrix{<:Complex}, z_o) =
    real.(adjoint(codes) * z_o) ./ Float32(_feature_dim(z_o))

# ================================================================
# 2. Per-layer EP interface
# ================================================================
#
# Five free functions dispatching on layer type. Layer authors add
# methods for their layer; the chain-level machinery (phasor_settle,
# ep_gradient) calls them generically. Phase 1 implements PhasorDense
# only.

"""
    ep_drive(layer, ps, st, z_in) -> z_drive

Forward drive into this layer given the previous layer's complex
state `z_in`. For PhasorDense this is `weight * z_in` (no
activation, no projection — those are applied in `phasor_settle`).
"""
function ep_drive end

"""
    ep_feedback(layer, ps, st, z_out) -> z_back

Backward feedback from this layer's state `z_out`, contributing to
the gradient of the previous layer's state. For PhasorDense with
real weights this is `transpose(weight) * z_out`.
"""
function ep_feedback end

"""
    ep_self_force(layer, ps, st, z_self) -> z_force

Self-energy gradient (½·K·z) for this layer's own state. Returns
zero in Phase 1 (K = 0). When per-channel dynamics are added in
Phase 2, this returns `½·(λ + iω)·z` from the layer's stored params.
"""
function ep_self_force end

"""
    ep_hebbian(layer, ps, st, z_in, z_self) -> NamedTuple

Per-parameter Hebbian outer products at the equilibrium states
`(z_in, z_self)`. Returns a NamedTuple matching the layer's
trainable-parameter structure (with zero entries for parameters
that EP does not update — e.g. log_neg_lambda in Phase 1).
"""
function ep_hebbian end

"""
    ep_energy_contribution(layer, ps, st, z_in, z_self) -> Float32

Real-valued scalar: this layer's contribution to the total energy
Φ. Diagnostic — used for sanity-checking settling, not for gradient
extraction.
"""
function ep_energy_contribution end

# ---- PhasorDense implementations ----

"""
    ep_drive(layer::PhasorDense, ps, st, z_in)

Reuse the layer's existing 2D Complex functor (network.jl line 370):
it computes `weight * z_in (+ bias)` without applying the
activation, which is exactly the raw drive the energy gradient
wants. This insulates EP from future internal changes to
PhasorDense's pre-activation step.
"""
function ep_drive(layer::PhasorDense, ps, st, z_in)
    return first(layer(z_in, ps, st))
end

function ep_feedback(layer::PhasorDense, ps, st, z_out)
    @assert eltype(ps.weight) <: Real "EP feedback assumes real-valued PhasorDense weights"
    return transpose(ps.weight) * z_out
end

# ep_self_force: half-K times z_self. K_mode = :zero ignores the
# stored per-channel dynamics (matches the prototype's K=0
# settling); K_mode = :stored pulls λ = -exp(log_neg_lambda) and ω
# from the layer's spk_args (via _get_omega). With ω ≈ 0 (e.g. by
# passing omega_override = zeros(...) or constructing the layer
# with SpikingArgs(t_period = Inf)) the equilibrium remains close
# to the unit circle; with the layer's default ω = 2π and dt = 0.5
# the per-step rotation is too large for damped fixed-point
# iteration to settle, so :stored mode typically requires a smaller
# dt or an ω override.
function ep_self_force(layer::PhasorDense, ps, st, z_self;
                       K_mode::Symbol = :zero,
                       omega_override::Union{Nothing, AbstractVector} = nothing)
    K_mode == :zero && return zero(z_self)
    λ = -exp.(ps.log_neg_lambda)
    ω = omega_override === nothing ? _get_omega(layer) : omega_override
    return Float32(0.5) .* ComplexF32.(λ .+ im .* ω) .* z_self
end

"""
    ep_hebbian(::PhasorDense, ps, st, z_in, z_self)

Per-parameter Hebbian outer products at the equilibrium states
`(z_in, z_self)`:

* `weight = real(z_self * z_in')` — uses `adjoint` (not
  `transpose`) because the states are complex and the energy
  derivative requires conjugation. (hep.jl uses transpose because
  it does the holomorphic Wirtinger derivative deliberately —
  different convention; do not copy.)
* `bias_real = real(z_self)`, `bias_imag = imag(z_self)` — derived
  from `Φ_bias = Re⟨bias, z_self⟩` with bias = bias_real + i·bias_imag.

Returns a NamedTuple matching `ps`'s trainable-parameter structure,
with zero gradients on `log_neg_lambda` (and `omega` if trainable)
since EP does not update per-channel dynamics in this Phase 2.
"""
function ep_hebbian(::PhasorDense, ps, st, z_in, z_self)
    # Batch normalization: `z_self * adjoint(z_in)` is a matmul, so for
    # `(out,B) * (B,in)` it already SUMS the per-sample outer products.
    # Dividing by B here turns that sum into the mean-over-batch
    # gradient. B = 1 for the single-sample path, so this is a no-op
    # for the original interface.
    invB = one(Float32) / Float32(_batch_size(z_self))
    g = (weight = real.(z_self * adjoint(z_in)) .* invB,)
    if haskey(ps, :bias_real)
        zb = _sum_batch(z_self)   # (out,) either way
        g = merge(g, (bias_real = Float32.(real.(zb)) .* invB,
                      bias_imag = Float32.(imag.(zb)) .* invB))
    end
    # Match ps shape exactly so Optimisers.update doesn't warn /
    # silently skip. Dynamics params are not EP-updated.
    if haskey(ps, :log_neg_lambda)
        g = merge(g, (log_neg_lambda = zero(ps.log_neg_lambda),))
    end
    return g
end

function ep_energy_contribution(layer::PhasorDense, ps, st, z_in, z_self;
                                K_mode::Symbol = :zero,
                                omega_override::Union{Nothing, AbstractVector} = nothing)
    e = Float32(real(dot(z_self, ps.weight * z_in)))
    if haskey(ps, :bias_real)
        bias = ps.bias_real .+ 1f0im .* ps.bias_imag
        e += Float32(real(dot(z_self, bias)))
    end
    if K_mode == :stored
        λ = -exp.(ps.log_neg_lambda)
        ω = omega_override === nothing ? _get_omega(layer) : omega_override
        K = ComplexF32.(λ .+ im .* ω)
        # Energy contribution: ½·Re<z, K·z>
        e += Float32(0.5) * Float32(real(dot(z_self, K .* z_self)))
    end
    return e
end

# ================================================================
# 3. Chain settling
# ================================================================

"""
    phasor_settle(chain, ps, st, x, cost, β; T=100, dt=0.5, init=nothing)

Damped projected fixed-point iteration on a `Lux.Chain` of
EP-compatible layers. Returns one complex-state vector per layer.

* Initializes states at zero by default. The hard branch of
  `normalize_to_unit_circle(·; ε=0)` returns `1+0im` for
  sub-threshold inputs, so the first iteration moves cleanly out of
  the origin instead of stalling.
* `init` (optional) — a Vector of complex states to warm-start from
  (used by the nudged phase, which starts from the free
  equilibrium).
* Update rule (per layer per step):

      grad_l = ep_drive(layer_l, …, z_{l-1})
             + ep_self_force(layer_l, …, z_l)
             + ep_feedback(layer_{l+1}, …, z_{l+1})         # if l < n
             + nudge_force(cost, z_l, β)                    # if l == n and β ≠ 0
      z_l   ← (1-dt) · z_l + dt · normalize_to_unit_circle(grad_l; ε=0)
"""
function phasor_settle(chain::Lux.Chain, ps, st, x, cost::AbstractEPCost, β::Real;
                       T::Int = 100, dt::Real = 0.5f0,
                       init::Union{Nothing,Vector} = nothing,
                       K_mode::Symbol = :zero,
                       omega_override::Union{Nothing, Vector} = nothing)
    layer_keys = collect(keys(ps))
    dt_f = Float32(dt)
    β_f  = Float32(β)

    z0 = _phase_input_to_complex(x)

    states = init === nothing ?
        _init_states(chain, layer_keys, z0) :
        [ComplexF32.(s) for s in init]

    # Hoist the input drive: layer 1's `ep_drive` is `W₁·z₀ (+ bias)`,
    # and both `ps` and `z₀` are fixed for the whole settle — yet the
    # original loop recomputed it every step. At MLP width this is the
    # single largest term in the step (2.6x on the per-step linear
    # algebra at 784→256, B=128).
    drive0 = _input_drive(chain, ps, st, layer_keys, z0)

    for _ in 1:T
        states = _phasor_step(chain, ps, st, layer_keys, z0, cost,
                              β_f, dt_f, states; K_mode=K_mode,
                              omega_override=omega_override, drive0=drive0)
    end
    return states
end

# Zero-initialized per-layer states, matching the input's batch shape.
# A `(d,)` input gives `(out,)` states; a `(d,B)` input gives `(out,B)`.
# These must agree — a 1-D init against a 2-D drive would silently
# broadcast in the damping term rather than erroring.
function _init_states(chain::Lux.Chain, layer_keys, z0::AbstractVector)
    return [gpu_zeros(z0, ComplexF32, chain.layers[k].out_dims) for k in layer_keys]
end

function _init_states(chain::Lux.Chain, layer_keys, z0::AbstractMatrix)
    B = size(z0, 2)
    return [gpu_zeros(z0, ComplexF32, chain.layers[k].out_dims, B) for k in layer_keys]
end

# The constant first-layer drive, computed once per settle.
function _input_drive(chain::Lux.Chain, ps, st, layer_keys, z0)
    k = layer_keys[1]
    return ep_drive(chain.layers[k], ps[k], st[k], z0)
end

# Single damped projected update step across all layers, with the
# given time-varying nudge β. Factored out so phasor_settle and the
# lock-in gradient extraction share the per-step logic. `K_mode`
# selects the self-energy treatment (see `ep_self_force`).
# `omega_override` is a Vector of per-layer overrides (each either
# `nothing` to use the layer's spk_args ω, or an AbstractVector to
# replace it) — or `nothing` to use defaults for every layer.
function _phasor_step(chain::Lux.Chain, ps, st, layer_keys, z0,
                      cost::AbstractEPCost, β::Float32, dt::Float32, states;
                      K_mode::Symbol = :zero,
                      omega_override::Union{Nothing, Vector} = nothing,
                      drive0 = nothing)
    n = length(layer_keys)
    # `map` (rather than a preallocated `Vector{Vector{ComplexF32}}`)
    # lets the element type be inferred, so the same code path yields
    # `Vector` states for a single sample and `Matrix` states for a
    # batch.
    return map(1:n) do l
        key  = layer_keys[l]
        ps_l = ps[key]; st_l = st[key]
        z_self = states[l]
        ω_l    = omega_override === nothing ? nothing : omega_override[l]

        grad_l = if l == 1
            drive0 === nothing ?
                ep_drive(chain.layers[key], ps_l, st_l, z0) : drive0
        else
            ep_drive(chain.layers[key], ps_l, st_l, states[l-1])
        end

        # Skip the self-force entirely under K_mode=:zero rather than
        # allocating a zero array and broadcasting it in every step.
        if K_mode != :zero
            grad_l = grad_l .+ ep_self_force(chain.layers[key], ps_l, st_l, z_self;
                                              K_mode=K_mode, omega_override=ω_l)
        end

        if l < n
            key_n = layer_keys[l+1]
            grad_l = grad_l .+ ep_feedback(chain.layers[key_n],
                                            ps[key_n], st[key_n], states[l+1])
        end
        if l == n && β != 0f0
            grad_l = grad_l .+ nudge_force(cost, z_self, β)
        end

        # Hard projection (ε = 0) — matches prototype, avoids
        # sub-threshold magnitude bias from the safe-mode default.
        (1 - dt) .* z_self .+ dt .* normalize_to_unit_circle(grad_l; ε = 0)
    end
end

# Convert any phase-typed input (Phase array, raw real array
# interpreted as phases in [-1,1], or already-complex array) into a
# complex unit-modulus vector for the first layer's drive.
_phase_input_to_complex(x::AbstractArray{<:Complex}) = ComplexF32.(x)
_phase_input_to_complex(x::AbstractArray{<:Phase})   = ComplexF32.(angle_to_complex(x))
_phase_input_to_complex(x::AbstractArray{<:Real})    = ComplexF32.(angle_to_complex(Phase.(x)))

# ================================================================
# 4. Finite-difference ground truth
# ================================================================

"""
    fd_gradient_phasor(chain, ps, st, x, cost::AbstractEPCost; ε=1e-5, T=200, dt=0.5, K_mode=:zero)
    fd_gradient_phasor(chain, ps, st, x, y; kwargs...)

Coordinate-by-coordinate forward finite-difference gradient of
`L(ps) = ep_loss(cost, z_o*_free)` with respect to each EP-trained
parameter (`weight`, plus `bias_real` and `bias_imag` when present)
of every PhasorDense layer in the chain.

The convenience form `fd_gradient_phasor(..., x, y::AbstractVector{<:Complex}; kwargs...)`
wraps `y` in a `SimilarityCost`.

Returns a NamedTuple matching `ps`'s structure; entries for
parameters EP does not update (e.g., `log_neg_lambda`, `omega`)
are zero.

This is the **ground-truth oracle** for the EP gradient. O(n_params)
expensive — for a chain with n_p trainable params (weight + bias),
runs n_p + 1 free-phase settles. Use for tests and small-network
analysis only.
"""
function fd_gradient_phasor(chain::Lux.Chain, ps, st, x,
                            cost::AbstractEPCost;
                            ε::Real = 1e-5, T::Int = 200, dt::Real = 0.5f0,
                            K_mode::Symbol = :zero,
                            omega_override::Union{Nothing, Vector} = nothing)
    ε_f  = Float32(ε)

    function loss_at(ps_perturbed)
        s = phasor_settle(chain, ps_perturbed, st, x, cost, 0f0;
                          T=T, dt=dt, K_mode=K_mode,
                          omega_override=omega_override)
        return ep_loss(cost, s[end])
    end

    base = loss_at(ps)

    # Walk every layer; FD each EP-trained parameter (weight, plus
    # bias if present), build a matched-shape gradient NamedTuple.
    pairs = Pair{Symbol,Any}[]
    for key in keys(ps)
        layer_ps = ps[key]
        filled = NamedTuple()
        for pname in (:weight, :bias_real, :bias_imag)
            haskey(layer_ps, pname) || continue
            P = layer_ps[pname]
            gP = zeros(Float32, size(P))
            for i in eachindex(P)
                Pp = copy(P)
                Pp[i] += ε_f
                ps_perturbed = _replace_param(ps, key, pname, Pp)
                gP[i] = (loss_at(ps_perturbed) - base) / ε_f
            end
            filled = merge(filled, NamedTuple{(pname,)}((gP,)))
        end
        push!(pairs, key => _zero_other_params(layer_ps, filled))
    end
    return NamedTuple(pairs)
end

# Backwards-compatible: y as a complex vector → SimilarityCost.
function fd_gradient_phasor(chain::Lux.Chain, ps, st, x,
                            y::AbstractVector{<:Complex}; kwargs...)
    return fd_gradient_phasor(chain, ps, st, x,
                              SimilarityCost(ComplexF32.(y)); kwargs...)
end

# Build a per-layer gradient NamedTuple matching the parameter
# structure: keep entries from `g_filled`, zero out any other arrays
# in `ps_layer` that aren't in `g_filled`.
function _zero_other_params(ps_layer::NamedTuple, g_filled::NamedTuple)
    out = Pair{Symbol,Any}[]
    for k in keys(ps_layer)
        if haskey(g_filled, k)
            push!(out, k => g_filled[k])
        else
            v = ps_layer[k]
            if v isa AbstractArray
                push!(out, k => zeros(eltype(v), size(v)))
            end
        end
    end
    return NamedTuple(out)
end

# Replace ps[key][pname] with V, returning a new NamedTuple.
function _replace_param(ps::NamedTuple, key::Symbol, pname::Symbol, V::AbstractArray)
    inner = ps[key]
    new_inner = merge(inner, NamedTuple{(pname,)}((V,)))
    return merge(ps, NamedTuple{(key,)}((new_inner,)))
end

# ================================================================
# 5. StaticEP + ep_gradient
# ================================================================

abstract type AbstractEPMethod end

"""
    StaticEP(; β=0.1, T_free=100, T_nudge=50, dt=0.5, K_mode=:zero, centered=false)

Vanilla EP gradient extraction with a single static real β. The
nudged phase warm-starts from the free equilibrium for tighter
linear-response sampling.

`centered = false` (default) is the original one-sided estimator
`-(h_β - h_0)/β`, which carries an O(β) bias. `centered = true` also
settles at `-β` and uses the symmetric difference `-(h_₊ - h_₋)/(2β)`,
cancelling the O(β) term at the cost of one extra nudged settle. Use
the centered form when `StaticEP` is serving as a gradient *oracle*
(e.g. calibrating `LockinEP` at a width where `fd_gradient_phasor` is
unaffordable), where its own bias would otherwise confound the
comparison.

!!! warning "Gradient fidelity degrades at large ‖W‖"
    EP assumes the nudged settle is a smooth deformation of the *same*
    fixed point as the free settle. On a 784→256→64 chain that holds at
    the usual init scale (‖W₁‖ ≈ 8: cosine 0.995 against a small-β
    reference, flat in β) but fails once ‖W₁‖ grows past roughly 15 —
    and ‖W₁‖ does grow, unbounded, because `normalize_to_unit_circle`
    makes the states scale-invariant so nothing in the loss penalizes it.

    The signature is diagnostic: the one-sided relative error scales as
    1/β (measured 28.7, 96.9, 292, 976 at β = 0.1, 0.03, 0.01, 0.003 for
    ‖W₁‖ = 31), meaning `h_nudge - h_free` retains a β-INDEPENDENT term.
    That is a basin hop — the two settles converge to different fixed
    points — not a linearization error, so shrinking β does not help and
    actively makes the estimate worse. The free settle is still perfectly
    stationary throughout (residual ≲1e-6), so this is invisible to a
    convergence check.

    `centered = true` recovers a substantial part of it (cosine 0.07 →
    0.38 at ‖W₁‖ = 31, 0.25 → 0.68 at ‖W₁‖ = 126) and is recommended for
    any long training run. Caveat: these numbers come from randomly
    initialized matrices rescaled to the given norm; trained weights of
    the same norm appear better behaved, since training at ‖W₁‖ ≈ 27
    still makes progress.

`K_mode = :zero` (default) ignores the layer's stored
`log_neg_lambda` / `omega` for self-energy — matches the
prototype's K = 0 phase-consensus settling. `K_mode = :stored` adds
the `½·(λ + iω)·z` self-force; in this mode the equilibrium reflects
the layer's per-channel SSM dynamics, but settling is sensitive to
the per-step rotation `dt·ω`, so a smaller `dt` and / or chain with
zeroed `ω` is typically required (see `docs/phasor_ep_design.md`).
"""
Base.@kwdef struct StaticEP <: AbstractEPMethod
    β::Float32      = 0.1f0
    T_free::Int     = 100
    T_nudge::Int    = 50
    dt::Float32     = 0.5f0
    K_mode::Symbol  = :zero
    centered::Bool  = false
end

"""
    ep_gradient(method, chain, ps, st, x, cost::AbstractEPCost) -> (grads, states_free)
    ep_gradient(method, chain, ps, st, x, y::AbstractVector{<:Complex}) -> (grads, states_free)

Compute the EP gradient for all trainable parameters of `chain`.
Returns a NamedTuple `grads` matching the structure of `ps` (one
entry per layer, with `weight` and `bias_real` / `bias_imag`
populated as appropriate, and other params zeroed) and the
free-phase equilibrium states.

The convenience form taking a complex vector `y` wraps it in a
`SimilarityCost` for backward compatibility.
"""
function ep_gradient(m::StaticEP, chain::Lux.Chain, ps, st, x,
                     cost::AbstractEPCost;
                     omega_override::Union{Nothing, Vector} = nothing)
    s_free  = phasor_settle(chain, ps, st, x, cost, 0f0;
                            T=m.T_free,  dt=m.dt, K_mode=m.K_mode,
                            omega_override=omega_override)
    s_pos   = phasor_settle(chain, ps, st, x, cost, m.β;
                            T=m.T_nudge, dt=m.dt, init=s_free, K_mode=m.K_mode,
                            omega_override=omega_override)

    h_free = chain_hebbians(chain, ps, st, x, s_free)
    h_pos  = chain_hebbians(chain, ps, st, x, s_pos)

    if m.centered
        # Symmetric (centered) estimator: settle at -β as well and use
        # -(h₊ - h₋)/(2β). The one-sided difference carries an O(β)
        # bias; the symmetric one cancels it, leaving O(β²). Costs one
        # extra nudged settle.
        s_neg = phasor_settle(chain, ps, st, x, cost, -m.β;
                              T=m.T_nudge, dt=m.dt, init=s_free, K_mode=m.K_mode,
                              omega_override=omega_override)
        h_neg = chain_hebbians(chain, ps, st, x, s_neg)
        grads = _ep_diff_gradient(ps, h_neg, h_pos, 2f0 * m.β)
    else
        # EP estimate: -(hebb_nudge - hebb_free) / β. Sign flip because
        # Φ contains -β·C and we want dL/dW.
        grads = _ep_diff_gradient(ps, h_free, h_pos, m.β)
    end
    return grads, s_free
end

# Back-compat: y as a complex vector → SimilarityCost.
function ep_gradient(m::AbstractEPMethod, chain::Lux.Chain, ps, st, x,
                     y::AbstractVector{<:Complex}; kwargs...)
    return ep_gradient(m, chain, ps, st, x, SimilarityCost(ComplexF32.(y)); kwargs...)
end

"""
    chain_hebbians(chain::Lux.Chain, ps, st, x, states::Vector) -> NamedTuple

Per-layer Hebbian outer products at the equilibrium states `states`. The
input `x` becomes the first layer's complex drive `z_0`; subsequent
layers consume the upstream equilibrium state. For each layer that has a
weight (`PhasorDense` and friends), returns `ep_hebbian(layer, ps, st,
z_in, z_self)` — see [`ep_hebbian`](@ref). Layers without a weight get
a zero-shaped gradient slot to keep the NamedTuple shape matching `ps`.

Used internally by [`ep_gradient`](@ref) (both `StaticEP` and `LockinEP`
paths) to compute the per-equilibrium Hebbian snapshots that the EP
gradient theorem differences. Useful directly when implementing custom
gradient estimators or batched EP variants.
"""
function chain_hebbians(chain::Lux.Chain, ps, st, x, states::Vector)
    layer_keys = collect(keys(ps))
    n = length(layer_keys)
    z0 = _phase_input_to_complex(x)
    pairs = Pair{Symbol,Any}[]
    for l in 1:n
        key = layer_keys[l]
        z_in   = (l == 1) ? z0 : states[l-1]
        z_self = states[l]
        h_l = haskey(ps[key], :weight) ?
            ep_hebbian(chain.layers[key], ps[key], st[key], z_in, z_self) :
            _zero_grad(ps[key])
        push!(pairs, key => h_l)
    end
    return NamedTuple(pairs)
end

# Build the gradient NamedTuple by differencing per-layer Hebbians
# and dividing by β. Handles weight + bias (when present) and
# zeros-out non-EP-trained params (log_neg_lambda, omega).
function _ep_diff_gradient(ps, h_free, h_nudge, β)
    inv_β = -1f0 / Float32(β)
    pairs = Pair{Symbol,Any}[]
    for key in keys(ps)
        if haskey(ps[key], :weight)
            entry = (weight = inv_β .* (h_nudge[key].weight .- h_free[key].weight),)
            if haskey(ps[key], :bias_real)
                entry = merge(entry, (
                    bias_real = inv_β .* (h_nudge[key].bias_real .- h_free[key].bias_real),
                    bias_imag = inv_β .* (h_nudge[key].bias_imag .- h_free[key].bias_imag),
                ))
            end
            entry = _pad_dynamics_zeros(entry, ps[key])
            push!(pairs, key => entry)
        else
            push!(pairs, key => _zero_grad(ps[key]))
        end
    end
    return NamedTuple(pairs)
end

# Add zero gradients for log_neg_lambda so the returned NamedTuple
# matches `ps` shape exactly (avoids Optimisers warnings). ω was
# removed from PhasorDense parameters/state (it's now derived from
# spk_args), so no :omega slot needs padding.
function _pad_dynamics_zeros(entry::NamedTuple, layer_ps::NamedTuple)
    if haskey(layer_ps, :log_neg_lambda)
        entry = merge(entry, (log_neg_lambda = zero(layer_ps.log_neg_lambda),))
    end
    return entry
end

# ================================================================
# 6. LockinEP — temporal Cauchy / lock-in detection
# ================================================================
#
# Drive the nudge as a real cosine probe β(t) = ε·cos(ω_p t), then
# extract the linear-response coefficient by demodulating the
# Hebbian outer products at +ω_p (DC-subtracted, integer cycles,
# warm-up phase). Equivalent to hEP's spatial contour but laid out
# in time — see docs/phasor_ep_design.md for the derivation.
#
# Why a real probe (not complex e^{iω_p t}): with non-holomorphic
# unit_project, z*(β,β̄) depends on both β and β̄. A complex probe
# extracts only the Wirtinger ∂/∂β; a real cosine probe excites both
# sidebands so the +ω_p Fourier coefficient picks up the full
# d/dβ_real = ∂/∂β + ∂/∂β̄, matching FD on real weights.

"""
    LockinEP(; ε=0.05, ω_p=0.05, n_cycles=8, T_warmup_cycles=2,
             T_free=200, dt=0.1)

Lock-in / temporal-Cauchy EP gradient extraction. The nudge is
swept as `β(t) = ε·cos(ω_p t)` and the gradient is recovered by
demodulating the per-layer Hebbian outer products at the probe
frequency.

# Knobs

* `ε` — probe amplitude. Smaller → more linear, but eventually
  hits the FD-precision noise floor.
* `ω_p` — probe angular frequency (rad / time-unit, where one
  step is `dt` time-units). Must be slow enough that the network
  tracks the probe adiabatically — i.e. `ω_p ≪ relaxation_rate ≈
  1/T_settle`.
* `n_cycles` — integer number of probe periods over which to
  integrate the lock-in. More → better demodulator selectivity at
  proportional compute cost.
* `T_warmup_cycles` — discarded probe periods at the start to let
  the equilibrium catch up to the modulation. Two is usually
  enough.
* `T_free` — free-phase settle steps (β = 0) to reach the
  base equilibrium.
* `dt` — per-step time increment. The product `ω_p · dt` is the
  per-step phase increment of the probe, so `dt` and `ω_p` are
  coupled — fine `dt` lets you use higher `ω_p` without aliasing.

# Defaults

The defaults `(ε=0.05, ω_p=0.05, dt=0.1, n_cycles=8)` give roughly
the deep-adiabatic regime visible in
`demos/phasor_ep_demo.ipynb` Section 6 (matches FD to a few
percent on a 2-layer chain).
"""
Base.@kwdef struct LockinEP <: AbstractEPMethod
    ε::Float32                 = 0.05f0
    ω_p::Float32               = 0.05f0
    n_cycles::Int              = 8
    T_warmup_cycles::Int       = 2
    T_free::Int                = 200
    dt::Float32                = 0.1f0
    K_mode::Symbol             = :zero
end

function ep_gradient(m::LockinEP, chain::Lux.Chain, ps, st, x,
                     cost::AbstractEPCost;
                     omega_override::Union{Nothing, Vector} = nothing)
    # 1. Free settle to the β=0 equilibrium and snapshot the DC hebbians.
    s_free = phasor_settle(chain, ps, st, x, cost, 0f0;
                           T=m.T_free, dt=m.dt, K_mode=m.K_mode,
                           omega_override=omega_override)
    h_dc   = chain_hebbians(chain, ps, st, x, s_free)

    # 2. Lock-in setup.
    layer_keys = collect(keys(ps))
    z0 = _phase_input_to_complex(x)
    period_steps = round(Int, 2π / (m.ω_p * m.dt))
    T_warmup = m.T_warmup_cycles * period_steps
    T_lockin = m.n_cycles        * period_steps

    states = [copy(s) for s in s_free]
    drive0 = _input_drive(chain, ps, st, layer_keys, z0)

    # 3. Warm-up — drive the probe but don't accumulate (transients die).
    for t in 1:T_warmup
        β_t = m.ε * cos(m.ω_p * t * m.dt)
        states = _phasor_step(chain, ps, st, layer_keys, z0, cost,
                              β_t, m.dt, states; K_mode=m.K_mode,
                              omega_override=omega_override, drive0=drive0)
    end

    # 4. Accumulators.
    #
    # `Ẑ[l] = Σ_t z_l(t)·e^{-iω_p t}` — the demodulated state of every
    # layer, `(out_l, B)`. This gives the bias gradient for every layer
    # directly, and for layer 1 it also gives the WEIGHT gradient
    # without ever forming a per-step outer product: layer 1's `z_in`
    # is `z₀`, which is constant in t, so it factors out of the sum
    #
    #     Σ_t z₁(t)·z₀' ·e^{-iω_p t} = (Σ_t z₁(t)e^{-iω_p t})·z₀'
    #
    # turning ~10⁴ full `(out×in)` complex outer products into ~10⁴
    # cheap `(out×B)` accumulations plus one matmul at the end (19x at
    # 784→256, B=128).
    #
    # `HW[l]` for l > 1 still needs per-step accumulation because
    # `z_in = states[l-1]` varies in t — but via the 5-arg `mul!`, which
    # fuses scale-and-add into one BLAS call with no temporaries.
    #
    # `c = Σ_t e^{-iω_p t}` carries the DC subtraction out of the loop
    # too: `Σ_t (h(t) - h_dc)·e^{-iω_p t} = Σ_t h(t)e^{-iω_p t} - c·h_dc`.
    # It is ≈0 over integer cycles but is kept exact.
    Zhat = [gpu_zeros(states[l], ComplexF32, size(states[l])...) for l in 1:length(layer_keys)]
    HW   = Vector{Any}(nothing, length(layer_keys))
    for (l, key) in enumerate(layer_keys)
        (l > 1 && haskey(ps[key], :weight)) || continue
        HW[l] = gpu_zeros(ps[key].weight, ComplexF32, size(ps[key].weight)...)
    end
    c = zero(ComplexF32)

    # 5. Integration: settle + demodulated accumulation.
    for t in 1:T_lockin
        β_t   = m.ε * cos(m.ω_p * t * m.dt)
        states = _phasor_step(chain, ps, st, layer_keys, z0, cost,
                              β_t, m.dt, states; K_mode=m.K_mode,
                              omega_override=omega_override, drive0=drive0)
        demod = ComplexF32(exp(-im * m.ω_p * t * m.dt))
        c += demod
        for l in 1:length(layer_keys)
            Zhat[l] .+= states[l] .* demod
            HW[l] === nothing && continue
            # H += demod · z_l · z_{l-1}'  (adjoint, not transpose —
            # the energy derivative requires conjugation; see ep_hebbian)
            mul!(HW[l], states[l], adjoint(states[l-1]), demod, one(ComplexF32))
        end
    end

    # 6. Convert to gradient: dL/d(real-param) = -2·Re(H) / (T_lockin · ε).
    #    The factor of 2 comes from the real cosine probe — see the
    #    design doc, section "Implementation sketch". For bias the
    #    real/imag parts of H_b give the bias_real / bias_imag grads
    #    respectively (since H_b's "complex" packaging is z_self and
    #    Re(z_self), Im(z_self) are independent params).
    H_W, H_b = _lockin_accumulators(ps, layer_keys, Zhat, HW, z0, h_dc, c)
    grads = _ep_lockin_gradient(ps, H_W, H_b, T_lockin, m.ε)
    return grads, s_free
end

# Close out the lock-in accumulators into DC-subtracted, batch-normalized
# complex Hebbians keyed by layer. `h_dc` entries are already divided by
# B (see `ep_hebbian`), so the raw accumulators get the same treatment
# before the DC term is subtracted.
function _lockin_accumulators(ps, layer_keys, Zhat, HW, z0, h_dc, c)
    H_W = Dict{Symbol, Any}()
    H_b = Dict{Symbol, Any}()
    for (l, key) in enumerate(layer_keys)
        haskey(ps[key], :weight) || continue
        invB = one(Float32) / Float32(_batch_size(Zhat[l]))
        raw  = l == 1 ? Zhat[1] * adjoint(z0) : HW[l]
        H_W[key] = raw .* invB .- c .* ComplexF32.(h_dc[key].weight)
        if haskey(ps[key], :bias_real)
            dc_b = ComplexF32.(h_dc[key].bias_real .+ 1f0im .* h_dc[key].bias_imag)
            H_b[key] = _sum_batch(Zhat[l]) .* invB .- c .* dc_b
        end
    end
    return H_W, H_b
end

function _ep_lockin_gradient(ps, H_W::AbstractDict, H_b::AbstractDict,
                              T_lockin::Int, ε)
    norm_factor = Float32(T_lockin) * Float32(ε)
    pairs = Pair{Symbol,Any}[]
    for key in keys(ps)
        if haskey(ps[key], :weight)
            entry = (weight = -2f0 .* real.(H_W[key]) ./ norm_factor,)
            if haskey(H_b, key)
                # Re(H_b) → bias_real grad; Im(H_b) → bias_imag grad.
                entry = merge(entry, (
                    bias_real = -2f0 .* real.(H_b[key]) ./ norm_factor,
                    bias_imag = -2f0 .* imag.(H_b[key]) ./ norm_factor,
                ))
            end
            entry = _pad_dynamics_zeros(entry, ps[key])
            push!(pairs, key => entry)
        else
            push!(pairs, key => _zero_grad(ps[key]))
        end
    end
    return NamedTuple(pairs)
end

# ================================================================
# 7. Training loop
# ================================================================

"""
    ep_train(model, ps, st, train_loader, args;
             method=StaticEP(), cost_fn=default_cost_fn, verbose=false)

Train a `Lux.Chain` of EP-compatible layers via equilibrium
propagation. Returns `(losses, ps, st)` — same shape as `train` and
`hep_train` so it's a drop-in for existing users.

`train_loader` is any iterable of `(x, y)` batches where `x` is a
phase-typed (or real-valued, interpreted as phase) array and `y` is
whatever your `cost_fn` consumes (a complex vector for the default
`SimilarityCost`, an integer class index for `CodebookCost`, etc.).

`cost_fn(y) -> AbstractEPCost`. The default constructs
`SimilarityCost(ComplexF32.(y))`, preserving the Phase-1 interface.
For codebook-style classification, pass
`cost_fn = y -> CodebookCost(codes_complex, y)`.

`args` is the global `Args` struct (see `test/runtests.jl`); `lr`,
`epochs`, and `weight_decay` are read.

`optimiser` is a constructor (not an instance), matching [`train`](@ref).
The default `Optimisers.Descent` is kept for backward compatibility, but
it is a poor choice at scale — on FashionMNIST at 217K parameters it
plateaus near 0.50 accuracy at any learning rate while `Optimisers.Adam`
reaches 0.80 (see `demos/ep_fashionmnist.jl`, `EP_MODE=sweep`).

`callback(epoch, ps, st, epoch_loss)` runs after each epoch — use it for
test-set evaluation, checkpointing, or tracking the settle residual.
"""
function ep_train(model::Lux.Chain, ps, st, train_loader, args;
                  method::AbstractEPMethod = StaticEP(),
                  cost_fn::Function = _default_cost_fn,
                  optimiser = Optimisers.Descent,
                  callback = nothing,
                  omega_override::Union{Nothing, Vector} = nothing,
                  verbose::Bool = false)
    opt_state = Optimisers.setup(optimiser(Float32(args.lr)), ps)
    losses = Float32[]
    for epoch in 1:args.epochs
        epoch_start = length(losses) + 1
        for (x, y) in train_loader
            cost = cost_fn(y)
            grads, s_free = ep_gradient(method, model, ps, st, x, cost;
                                        omega_override=omega_override)
            # Weight decay is read here (ep_train previously ignored it).
            # It measurably helps EP training on FashionMNIST at 1e-4, but
            # NOT by bounding ‖W‖ — measured, 1e-4 leaves the weight-norm
            # trajectory almost unchanged while clearly improving accuracy,
            # and a larger 1e-3 bounds ‖W‖ much more while performing
            # worse. Treat it as ordinary regularization; the EP-specific
            # large-‖W‖ failure is described under `StaticEP`.
            if args.weight_decay > 0
                grads = _apply_weight_decay(grads, ps, args.weight_decay)
            end
            opt_state, ps = Optimisers.update(opt_state, ps, grads)
            push!(losses, ep_loss(cost, s_free[end]))
            if verbose
                println("epoch=$epoch loss=$(losses[end])")
            end
        end
        if callback !== nothing
            epoch_loss = mean(@view losses[epoch_start:end])
            callback(epoch, ps, st, epoch_loss)
        end
    end
    return losses, ps, st
end

"""
    ep_predict(chain, ps, st, x, codes; T=100, dt=0.5, K_mode=:zero) -> logits

Settle `chain` to its free (β = 0) equilibrium on input `x` and score the
output state against a complex codebook, returning `(n_classes,)` or
`(n_classes, B)` similarity logits.

This is the inference counterpart to training with `CodebookCost`. It
exists because `loss_and_accuracy` (`src/metrics.jl`) assumes a
feedforward `model(x, ps, st)` call, which an EP-settled network cannot
provide. The returned matrix feeds `predict(·, :similarity)` and
`evaluate_accuracy` unchanged — both of which return **1-based** class
indices, while `fashion_mnist_data` targets are 0-based.
"""
function ep_predict(chain::Lux.Chain, ps, st, x, codes::AbstractMatrix{<:Complex};
                    T::Int = 100, dt::Real = 0.5f0, K_mode::Symbol = :zero,
                    omega_override::Union{Nothing, Vector} = nothing)
    s = phasor_settle(chain, ps, st, x, NullCost(), 0f0;
                      T=T, dt=dt, K_mode=K_mode, omega_override=omega_override)
    return codebook_logits(codes, s[end])
end

_default_cost_fn(y) = SimilarityCost(ComplexF32.(y))
