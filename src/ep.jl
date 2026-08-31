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
#   * Self-energy: K_mode = :zero (default) or :stored. :stored adds the
#     dissipative half-lambda*z only; it settles fine at the default
#     dt = 0.5 and the layer's default omega = 2*pi. (It used to need
#     dt <= 0.1 or an omega override, because the force wrongly carried
#     an extra half-i-omega*z — see ep_self_force.)
#   * Carrier: phasor_settle(carrier=omega) runs the lab frame for a
#     rotating (resonate-and-fire) substrate. Exactly equivalent to the
#     default co-rotating frame for one shared omega, so for the analog
#     settle it is a verification tool rather than a different model.
#     This stops at a quantized readout (readout_delta > 0), which is not
#     U(1)-equivariant — see phasor_settle's docstring.
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

# ep_self_force: the DISSIPATIVE half of the per-channel dynamics,
# ½·λ·z. K_mode = :zero skips it entirely (the K = 0 phase-consensus
# settle); K_mode = :stored pulls λ = -exp(log_neg_lambda).
#
# ω is deliberately NOT here. The self-energy term of Φ is
# ½·Re⟨z, K·z⟩ with K = λ + iω, and
#
#     Re⟨z, (λ + iω)z⟩ = Re((λ + iω)|z|²) = λ|z|²
#
# — the rotation contributes NOTHING to the energy. ω is symplectic,
# not dissipative: it generates the U(1) flow rather than descending Φ.
# An earlier version added ½(λ + iω)z into the pre-projection drive,
# which made the force inconsistent with `ep_energy_contribution` (where
# ω cancels automatically) and mixed the rotation into the nonlinear
# unit projection. That is why `:stored` used to need dt ≪ 0.5 or an ω
# override to settle at all.
#
# The carrier is instead applied as an EXACT multiplicative rotation
# after the projection — see the `carrier` kwarg of `phasor_settle`.
# `omega_override` is accepted and ignored, kept so existing call sites
# and the `ep_energy_contribution` signature stay source-compatible.
function ep_self_force(layer::PhasorDense, ps, st, z_self;
                       K_mode::Symbol = :zero,
                       omega_override::Union{Nothing, AbstractVector} = nothing)
    K_mode == :zero && return zero(z_self)
    λ = -exp.(ps.log_neg_lambda)
    return Float32(0.5) .* λ .* z_self
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
        # Energy contribution: ½·Re⟨z, K·z⟩. Note this is ALREADY blind to
        # ω — `dot(z, K.*z) = Σ K_i|z_i|²`, so `real(...)` keeps only λ.
        # `ep_self_force` returning ½λz is therefore exactly this term's
        # gradient; the two agree by construction. (They did not before: the
        # force carried an extra ½iωz with no counterpart here. See
        # `ep_self_force` and docs/phasor_ep_design.md.) ω is retained in the
        # expression only so the arithmetic mirrors the SSM eigenvalue.
        e += Float32(0.5) * Float32(real(dot(z_self, K .* z_self)))
    end
    return e
end

# ---- PhasorBind implementations ----

"""
    ep_drive(layer::PhasorBind, ps, st, z_in)

Forward drive for binding layer: `k ⊙ z_in + bias`.
"""
function ep_drive(layer::PhasorBind, ps, st, z_in)
    return first(layer(z_in, ps, st))
end

function ep_feedback(layer::PhasorBind, ps, st, z_out)
    k = ps.key
    return conj.(k) .* z_out
end

function ep_self_force(layer::PhasorBind, ps, st, z_self;
                       K_mode::Symbol = :zero,
                       omega_override::Union{Nothing, AbstractVector} = nothing)
    # Binding has no self-dynamics (K = 0)
    return zero(z_self)
end

function ep_hebbian(::PhasorBind, ps, st, z_in, z_self)
    invB = one(Float32) / Float32(_batch_size(z_self))
    # Gradient w.r.t. key: real(z_self ⊙ conj(z_in)) = real(z_self * z_in')
    g = (key = real.(z_self .* adjoint(z_in)) .* invB,)
    if haskey(ps, :bias_real)
        zb = _sum_batch(z_self)
        g = merge(g, (bias_real = Float32.(real.(zb)) .* invB,
                      bias_imag = Float32.(imag.(zb)) .* invB))
    end
    return g
end

function ep_energy_contribution(layer::PhasorBind, ps, st, z_in, z_self;
                                K_mode::Symbol = :zero,
                                omega_override::Union{Nothing, AbstractVector} = nothing)
    k = ps.key
    e = Float32(real(dot(z_self, k .* z_in)))
    if haskey(ps, :bias_real)
        bias = ps.bias_real .+ 1f0im .* ps.bias_imag
        e += Float32(real(dot(z_self, bias)))
    end
    return e
end

# ================================================================
# ResidualBlock implementations (Phase 2: skip connections)
# ================================================================

"""
    ep_drive(rb::ResidualBlock, ps, st, z_in)

Forward drive for ResidualBlock: returns the pre-normalization target
z_target = z_in ⊙ (z_branch .^ α).

The branch state z_branch is computed from z_in using the internal chain.
"""
function ep_drive(rb::ResidualBlock, ps, st, z_in)
    # Compute branch drive using internal chain's ep_drive
    d_branch = ep_drive(rb.ff, ps.ff, st.ff, z_in)
    # Project to unit circle to get branch state
    z_branch = _project_damp(ComplexF32(1), d_branch, 1f0, 1f-10)
    # Apply ReZero gate: z_branch^α
    α = haskey(ps, :alpha) ? ps.alpha[1] : 1f0
    z_branch_α = z_branch .^ α
    # v_bind = complex multiplication = phase addition
    return z_in .* z_branch_α
end

"""
    ep_feedback(rb::ResidualBlock, ps, st, z_out)

Backward feedback through ResidualBlock's v_bind (complex multiplication).
If z_out is a tuple (z_out, z_branch), extracts branch state directly.
Otherwise recomputes branch from z_out as approximation.
"""
function ep_feedback(rb::ResidualBlock, ps, st, z_out)
    # Handle tuple state from settle: (z_out, z_branch)
    if z_out isa Tuple
        z_out_val, z_branch = z_out
    else
        # Recompute branch state from z_out as approximation
        branch_ps = ps.ff.layer_1; branch_st = st.ff.layer_1
        branch_layer = rb.ff.layers[1]  # assuming single PhasorDense branch
        d_branch = ep_drive(branch_layer, branch_ps, branch_st, z_out)
        z_branch = _project_damp.(ComplexF32(1), d_branch, 1f0, 1f-10)
        z_out_val = z_out
    end
    α = haskey(ps, :alpha) ? ps.alpha[1] : 1f0
    z_branch_α = z_branch .^ α
    # Adjoint through z_in .* z_branch_α: ∂/∂z_in = conj(z_branch_α)
    fb_zin = conj.(z_branch_α) .* z_out_val
    # Plus feedback through branch weights
    branch_ps = ps.ff.layer_1; branch_st = st.ff.layer_1
    branch_layer = rb.ff.layers[1]
    fb_branch = ep_feedback(branch_layer, branch_ps, branch_st, z_branch_α .* z_out_val)
    return fb_zin + fb_branch
end

"""
    ep_self_force(rb::ResidualBlock, ps, st, z_self; K_mode, omega_override)

Self-energy for ResidualBlock output state: zero (no self-dynamics).
Branch self-dynamics are handled by the branch chain's own ep_self_force.
"""
function ep_self_force(rb::ResidualBlock, ps, st, z_self;
                       K_mode::Symbol = :zero,
                       omega_override::Union{Nothing, AbstractVector} = nothing)
    return zero(z_self)
end

"""
    ep_hebbian(::ResidualBlock, ps, st, z_in, z_self)

Per-parameter Hebbian for ResidualBlock:
- Branch weights: reuse PhasorDense hebbian with (z_in, z_branch)
- Alpha gradient: hand derivative d/dα (z^α) = z^α log(z)
  For |z|=1: log(z) = i·angle(z), so d/dα (z^α) = i·angle(z)·z^α
"""
function ep_hebbian(rb::ResidualBlock, ps, st, z_in, z_self)
    invB = one(Float32) / Float32(_batch_size(z_self))
    
    # Compute branch state using internal chain's ep_drive
    branch_ps = ps.ff.layer_1; branch_st = st.ff.layer_1
    branch_layer = rb.ff.layers[1]  # assuming single PhasorDense branch
    d_branch = ep_drive(branch_layer, branch_ps, branch_st, z_in)
    z_branch = _project_damp.(ComplexF32(1), d_branch, 1f0, 1f-10)
    α = haskey(ps, :alpha) ? ps.alpha[1] : 1f0
    z_branch_α = z_branch .^ α
    
    # Branch weight hebbian: reuse PhasorDense logic with (z_in, z_branch)
    g_branch = ep_hebbian(branch_layer, branch_ps, branch_st, z_in, z_branch)
    
    # Alpha gradient: dE/dα = imag( (z_self ⊙ conj(z_branch_α)) ⊙ phase(z_branch) ) / B
    # phase(z_branch) ∈ [-0.5, 0.5] where 1.0 = 2π rad = full circle
    phase_zb = angle.(z_branch) ./ (2f0 * pi_f32)
    alpha_grad = sum(imag.(z_self .* conj.(z_branch_α) .* phase_zb)) .* invB
    
    # Wrap branch hebbian under ff.layer_1 to match ResidualBlock param structure
    g = (ff = (layer_1 = g_branch,), alpha = [alpha_grad])
    return g
end

function ep_energy_contribution(rb::ResidualBlock, ps, st, z_in, z_self;
                                K_mode::Symbol = :zero,
                                omega_override::Union{Nothing, AbstractVector} = nothing)
    # Energy: Re⟨z_self, z_in ⊙ z_branch^α⟩
    branch_ps = ps.ff.layer_1; branch_st = st.ff.layer_1
    branch_layer = rb.ff.layers[1]
    d_branch = ep_drive(branch_layer, branch_ps, branch_st, z_in)
    z_branch = _project_damp.(ComplexF32(1), d_branch, 1f0, 1f-10)
    α = haskey(ps, :alpha) ? ps.alpha[1] : 1f0
    z_branch_α = z_branch .^ α
    z_target = z_in .* z_branch_α
    return Float32(real(dot(z_self, z_target)))
end

# ================================================================
# Weight cache for hoisted drive/feedback
# ================================================================

struct _WeightCache{T}
    drive::T        # per-layer ComplexF32 weight, or nothing
    feedback::T     # per-layer ComplexF32 transpose(weight), or nothing
end

# ================================================================
# 3. Chain settling
# ================================================================

"""
    phasor_settle(chain, ps, st, x, cost, β; T=100, dt=0.5, init=nothing,
                  carrier=nothing, t0=0)

Damped projected fixed-point iteration on a `Lux.Chain` of
EP-compatible layers. Returns one complex-state vector per layer.

# Frames

By default the settle runs in the **co-rotating frame**: states are
relative phases and there is no carrier. This is the frame EP is defined
in, and it is the one to use.

`carrier = ω` instead runs the **lab frame**, where every state spins at
the shared carrier ω, as a resonate-and-fire neuron physically does. Pass
`t0` to continue an existing trajectory (e.g. a nudged settle warm-started
from a free equilibrium must resume at `t0 = T_free·dt`, not 0).

The two frames are **exactly equivalent**, not approximately. Under
`z_l = w_l·e^{iωt}` applied to every layer including the input,

    ż_l = (λ + iω)z_l + W_l·z_{l-1}   ⟹   ẇ_l = λw_l + W_l·w_{l-1}

because Φ = Σ_l Re⟨W_l z_{l-1}, z_l⟩ is U(1)-invariant and the hard
projection is U(1)-equivariant (`û(e^{iθ}g) = e^{iθ}û(g)`), so the
*discrete* step commutes with the rotation too. The only terms that break
the symmetry are the bias and the cost target, and in this package both
physically co-rotate — `bias_current` (`src/spiking.jl`) injects a
`periodic_raised_cosine_kernel` pulse once per period at a fixed phase, so
its fundamental is `b·e^{iωt}`, and the codebook is made of neurons at the
same ω. The lab-frame path therefore puts both on the carrier and the
equivalence is exact.

Consequence: **for the analog settle**, the lab frame is a validation
tool, not a cheaper or a truer model — see `scripts/ep_rotating_gates.jl`,
which asserts the equivalence.

The equivalence stops at the readout. `_quantize_phase` rounds onto a
fixed phase grid, and rounding commutes with a rotation only when the
rotation is an exact multiple of the grid spacing δ. It is therefore the
one non-U(1)-equivariant operation in the pipeline, and the carrier
reduction above does not reach it. With `readout_δ > 0` the two frames
give materially different estimators: measured over 8 draws at
δ = 0.005 turns and no jitter, median cos against a centered `StaticEP`
reference is 0.44/0.82 co-rotating but 0.998/0.998 in the lab at an
incommensurate ω — the carrier sweeps the state across tens of bins per
step and dithers the quantizer for free, which is precisely the dead zone
the co-rotating model suffers. A commensurate carrier (ω = 2π at dt = 0.5
is 0.25 turns = exactly 50 bins of δ = 0.005) reproduces the co-rotating
result, as the commensurability argument requires. So with quantization
the frame is a modelling choice about where the readout clock lives, not
a free change of variables. Gate F in `scripts/ep_rotating_gates.jl` pins
the equivariance boundary; `scripts/ep_readout_frame_check.jl` reproduces
the measurement.

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
                       omega_override::Union{Nothing, Vector} = nothing,
                       carrier::Union{Nothing, Real, Vector{<:Real}} = nothing,
                       t0::Real = 0,
                       project::Symbol = :hard,
                       soft_ε::Real = 0.1f0,
                       cache::Union{Nothing, _WeightCache} = nothing)
    layer_keys = collect(keys(ps))
    n_layers = length(layer_keys)
    dt_f = Float32(dt)
    β_f  = Float32(β)
    
    # Handle carrier: single value (shared) or vector (per-layer detuning)
    if carrier === nothing
        carriers = nothing
    elseif carrier isa AbstractVector
        @assert length(carrier) == n_layers "carrier vector must match number of layers ($n_layers)"
        carriers = Float32.(carrier)
    else
        carriers = fill(Float32(carrier), n_layers)
    end
    t_f = Float32(t0)

    z0 = _phase_input_to_complex(x)

    states = init === nothing ?
        _init_states(chain, layer_keys, z0) :
        [s isa Tuple ? s : ComplexF32.(s) for s in init]

    # Hoist the input drive: layer 1's `ep_drive` is `W₁·z₀ (+ bias)`,
    # and both `ps` and `z₀` are fixed for the whole settle — yet the
    # original loop recomputed it every step. At MLP width this is the
    # single largest term in the step (2.6x on the per-step linear
    # algebra at 784→256, B=128).
    cache = cache === nothing ? _weight_cache(chain, ps, layer_keys) : cache
    drive0 = _input_drive(chain, ps, st, layer_keys, z0; cache=cache)

    for _ in 1:T
        states = _phasor_step(chain, ps, st, layer_keys, z0, cost,
                              β_f, dt_f, states; K_mode=K_mode,
                              omega_override=omega_override, drive0=drive0,
                              cache=cache, carriers=carriers, t_now=t_f,
                              project=project, soft_ε=Float32(soft_ε))
        t_f += dt_f
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
function _input_drive(chain::Lux.Chain, ps, st, layer_keys, z0; cache = nothing)
    k = layer_keys[1]
    return _cached_drive(cache, 1, chain, k, ps[k], st[k], z0)
end

# Single damped projected update step across all layers, with the
# given time-varying nudge β. Factored out so phasor_settle and the
# lock-in gradient extraction share the per-step logic. `K_mode`
# selects the self-energy treatment (see `ep_self_force`).
# `omega_override` is VESTIGIAL in this path: ω no longer enters the
# self-force at all (it is symplectic — see `ep_self_force`), so the value
# threaded here is accepted and ignored. It is still live in
# `ep_energy_contribution`, and the kwarg is retained so existing call
# sites keep working. To actually rotate, use `carriers` below.
# `carriers` can be `nothing` (co-rotating frame), a single Float32 (shared
# carrier for all layers), or a Vector{Float32} (per-layer carrier for detuning).
function _phasor_step(chain::Lux.Chain, ps, st, layer_keys, z0,
                      cost::AbstractEPCost, β::Float32, dt::Float32, states;
                      K_mode::Symbol = :zero,
                      omega_override::Union{Nothing, Vector} = nothing,
                      drive0 = nothing, cache = nothing,
                      carriers::Union{Nothing, Float32, Vector{Float32}} = nothing,
                      t_now::Float32 = 0f0,
                      project::Symbol = :hard,
                      soft_ε::Float32 = 0.1f0)
    n = length(layer_keys)
    th = 1.0f-10
    
    # Lab-frame carrier(s). `carriers` can be nothing (co-rotating), 
    # a single carrier (shared), or per-layer vector (detuning).
    # Precompute c_t and rot per layer.
    c_t_vec = if carriers === nothing
        fill(nothing, n)
    elseif carriers isa AbstractVector
        @assert length(carriers) == n
        [ComplexF32(cis(mod(Float64(carriers[l]) * Float64(t_now), 2π))) for l in 1:n]
    else
        fill(ComplexF32(cis(mod(Float64(carriers) * Float64(t_now), 2π))), n)
    end
    
    rot_vec = if carriers === nothing
        fill(nothing, n)
    elseif carriers isa AbstractVector
        [(r = ComplexF32(cis(mod(Float64(carriers[l]) * Float64(dt), 2π))); r / abs(r)) for l in 1:n]
    else
        fill((r = ComplexF32(cis(mod(Float64(carriers) * Float64(dt), 2π))); r / abs(r)), n)
    end
    
    # `map` (rather than a preallocated `Vector{Vector{ComplexF32}}`)
    # lets the element type be inferred, so the same code path yields
    # `Vector` states for a single sample and `Matrix` states for a
    # batch.
    return map(1:n) do l
        key  = layer_keys[l]
        ps_l = ps[key]; st_l = st[key]
        layer = chain.layers[key]
        z_self = states[l]
        
        # Extract z_out from tuple if ResidualBlock
        z_out = z_self isa Tuple ? z_self[1] : z_self
        
        ω_l    = omega_override === nothing ? nothing : omega_override[l]
        c_t    = c_t_vec[l]
        rot    = rot_vec[l]

        if layer isa ResidualBlock
            # ResidualBlock-specific settle logic
            # Get z_in (input to this block)
            z_in = if l == 1
                z0
            else
                z_prev = states[l-1]
                z_prev isa Tuple ? z_prev[1] : z_prev
            end
            
            # Compute branch state using internal chain
            branch_ps = ps_l.ff.layer_1; branch_st = st_l.ff.layer_1
            branch_layer = layer.ff.layers[1]
            d_branch = ep_drive(branch_layer, branch_ps, branch_st, z_in)
            z_branch = _project_damp.(ComplexF32(1), d_branch, 1f0, 1f-10)
            
            # Apply ReZero gate: z_branch^α
            α = haskey(ps_l, :alpha) ? ps_l.alpha[1] : 1f0
            z_branch_α = z_branch .^ α
            
            # Block target: z_target = z_in ⊙ z_branch_α
            z_target = z_in .* z_branch_α
            grad_l = z_target
            
            # Self-force (zero for ResidualBlock output, branch handled internally)
            if K_mode != :zero
                grad_l = grad_l .+ ep_self_force(layer, ps_l, st_l, z_out;
                                                  K_mode=K_mode, omega_override=ω_l)
            end
            
            # Feedback from next layer
            if l < n
                key_n = layer_keys[l+1]
                next_layer = chain.layers[key_n]
                next_z_self = states[l+1]
                next_z_out = next_z_self isa Tuple ? next_z_self[1] : next_z_self
                if next_layer isa ResidualBlock
                    # Next layer is also ResidualBlock - use its ep_feedback with tuple
                    grad_l = grad_l .+ ep_feedback(next_layer, ps[key_n], st[key_n], next_z_self)
                else
                    grad_l = grad_l .+ _cached_feedback(cache, l+1, chain, key_n,
                                                        ps[key_n], st[key_n], next_z_out)
                end
            end
            
            # Nudge at output layer
            if l == n && β != 0f0
                grad_l = grad_l .+ (c_t === nothing ?
                    nudge_force(cost, z_out, β) :
                    c_t .* nudge_force(cost, z_out .* conj(c_t), β))
            end
            
            # Project
            if project === :soft
                z_out_new = rot === nothing ?
                    _project_damp_soft.(z_out, grad_l, dt, soft_ε) :
                    _project_damp_soft_rot.(z_out, grad_l, dt, soft_ε, rot)
            else
                z_out_new = rot === nothing ? _project_damp.(z_out, grad_l, dt, th) :
                                              _project_damp_rot.(z_out, grad_l, dt, th, rot)
            end
            
            # Return tuple (z_out, z_branch) for this ResidualBlock
            return (z_out_new, z_branch)
        else
            # Standard layer logic (PhasorDense, PhasorBind, etc.)
            # Get z_in for drive computation
            z_in = if l == 1
                z0
            else
                z_prev = states[l-1]
                z_prev isa Tuple ? z_prev[1] : z_prev
            end
            
            grad_l = if l == 1
                d = drive0 === nothing ?
                    _cached_drive(cache, l, chain, key, ps_l, st_l, z0) : drive0
                c_t === nothing ? d : c_t .* d
            else
                _cached_drive(cache, l, chain, key, ps_l, st_l, z_in;
                              carrier_phase=c_t)
            end

            if K_mode != :zero
                grad_l = grad_l .+ ep_self_force(layer, ps_l, st_l, z_out;
                                                  K_mode=K_mode, omega_override=ω_l)
            end

            if l < n
                key_n = layer_keys[l+1]
                next_z_self = states[l+1]
                next_z_out = next_z_self isa Tuple ? next_z_self[1] : next_z_self
                if chain.layers[key_n] isa ResidualBlock
                    grad_l = grad_l .+ ep_feedback(chain.layers[key_n], ps[key_n], st[key_n], next_z_self)
                else
                    grad_l = grad_l .+ _cached_feedback(cache, l+1, chain, key_n,
                                                        ps[key_n], st[key_n], next_z_out)
                end
            end
            if l == n && β != 0f0
                grad_l = grad_l .+ (c_t === nothing ?
                    nudge_force(cost, z_out, β) :
                    c_t .* nudge_force(cost, z_out .* conj(c_t), β))
            end

            if project === :soft
                rot === nothing ?
                    _project_damp_soft.(z_out, grad_l, dt, soft_ε) :
                    _project_damp_soft_rot.(z_out, grad_l, dt, soft_ε, rot)
            else
                rot === nothing ? _project_damp.(z_out, grad_l, dt, th) :
                                  _project_damp_rot.(z_out, grad_l, dt, th, rot)
            end
        end
    end
end

# Fused per-element settle update: hard unit projection of the drive
# followed by the damped step toward it. Written as a scalar kernel so the
# whole thing is ONE broadcast pass — the equivalent expression
# `(1-dt).*z .+ dt.*normalize_to_unit_circle(g; ε=0)` materializes four
# temporaries and recomputes |g| twice. Same operations in the same order,
# so results are bit-identical. GPU-safe (`abs`/`ifelse` on ComplexF32).
@inline function _project_damp(z::ComplexF32, g::ComplexF32,
                               dt::Float32, th::Float32)
    r = abs(g)
    u = ifelse(r > th, g / max(r, th), ComplexF32(1, 0))
    return (1 - dt) * z + dt * u
end

# Lab-frame variant: the same projected damp, followed by ONE exact
# carrier step. Written multiplicatively (`rot = cis(ω·dt)`), never as an
# additive `iω·z` in the drive — that is what makes the lab frame exactly
# conjugate to the co-rotating frame. Substituting z_n = w_n·cis(ω·n·dt):
#
#     cis(ω(n+1)dt)·w_{n+1} = cis(ω·dt)·[(1-dt)·cis(ω·n·dt)·w_n + dt·û(g)]
#
# and since g is U(1)-covariant, g = cis(ω·n·dt)·g_w and
# û(g) = cis(ω·n·dt)·û(g_w), so the carrier divides straight out:
#
#     w_{n+1} = (1-dt)·w_n + dt·û(g_w)
#
# which is `_project_damp` verbatim. Exact at any dt — no Nyquist limit,
# because the rotation is never discretized. Kept as a separate kernel so
# the co-rotating path pays nothing for it.
@inline function _project_damp_rot(z::ComplexF32, g::ComplexF32,
                                   dt::Float32, th::Float32, rot::ComplexF32)
    return rot * _project_damp(z, g, dt, th)
end

# Soft-projection variant. `_project_damp`'s hard branch,
# `ifelse(r > th, g/r, 1+0im)`, is DISCONTINUOUS: as |g| → 0 the output
# jumps to 1+0im rather than approaching it. Near-zero drives are not
# exotic — `test/test_ep.jl` downscales its weights by 0.4 precisely
# because "the default glorot is wide enough that some initial drives can
# have small magnitude during settling".
#
# That discontinuity is a candidate explanation for the bimodal
# lock-in failures in results/ep_adiabatic/ANTICORRELATED_GRADIENT_NOTE.md,
# which concluded the fix was "operational, not structural" (use a smaller
# ε). Bimodality across draws is the signature of a threshold being
# crossed, not of an amplitude being too large — so this offers the
# structural alternative as something measurable rather than argued.
#
# Softening is `u = g / sqrt(|g|² + ε²)`: exactly `g/|g|` for |g| ≫ ε,
# smoothly → 0 as |g| → 0, and continuous everywhere.
#
# NOTE — `soft_normalize_to_unit_circle` (src/activations.jl) is NOT reused
# here, deliberately. It interpolates the PHASE from 0 toward angle(g),
#
#     u = exp(i · blend(|g|) · angle(g))
#
# which is not U(1)-equivariant: `blend·angle(e^{iφ}g) ≠ φ + blend·angle(g)`.
# Phase 0 is a fixed point of that compression, i.e. it installs a preferred
# direction in the complex plane. For a feedforward activation that is
# harmless; here it is fatal twice over — it breaks the carrier cancellation
# that makes the rotating case tractable at all (`phasor_settle`'s
# docstring), and it breaks the U(1) invariance of the Hebbian that lets a
# spiking substrate read the gradient off relative spike times.
#
# Measured, on the toy chain: the phase-interpolating form settles perfectly
# (residual 0) but degrades EP-vs-FD from 0.023 to 0.198 (K=:zero) and 0.011
# to 0.318 (K=:stored). The form below is equivariant by construction —
# `u(e^{iφ}g) = e^{iφ}u(g)` — so it keeps both properties.
#
# Unlike the hard branch, this does not force |z| = 1: a weakly driven
# neuron settles to a small amplitude rather than snapping to 1+0im in an
# arbitrary direction. The Hebbian then weights by amplitude, which is the
# physically sensible reading of a neuron that barely fired.
@inline function _project_damp_soft(z::ComplexF32, g::ComplexF32, dt::Float32,
                                    εs::Float32)
    u = g / sqrt(abs2(g) + εs * εs)
    return (1 - dt) * z + dt * u
end

@inline function _project_damp_soft_rot(z::ComplexF32, g::ComplexF32, dt::Float32,
                                        εs::Float32, rot::ComplexF32)
    return rot * _project_damp_soft(z, g, dt, εs)
end

# Per-settle weight cache. `ps.weight` is real, but the states are complex:
# the PhasorDense functor therefore splits into `W*real(x)` and `W*imag(x)`,
# two real gemms plus five array temporaries. Promoting the weight to
# ComplexF32 ONCE per settle lets BLAS run a single cgemm instead —
# measured 2.3x on the layer drive and 1.4x on the feedback at 784→256→64,
# B=128, despite doing nominally 2x the arithmetic. The win is memory
# traffic, not flops.
#
# Do NOT "simplify" this by calling `mul!` with a real `transpose(W)`
# against a complex operand: that combination misses the BLAS path
# entirely and falls back to a generic kernel ~19x SLOWER than the
# allocating `transpose(W) * z`.

function _weight_cache(chain::Lux.Chain, ps, layer_keys)
    drive = Any[]; feedback = Any[]
    for k in layer_keys
        lyr = chain.layers[k]
        if lyr isa PhasorDense && haskey(ps[k], :weight) && eltype(ps[k].weight) <: Real
            W = ps[k].weight
            push!(drive,    ComplexF32.(W))
            push!(feedback, ComplexF32.(transpose(W)))
        else
            push!(drive, nothing); push!(feedback, nothing)
        end
    end
    return _WeightCache(drive, feedback)
end

# Cached drive/feedback with a fallback to the generic per-layer interface
# for any layer the cache could not handle.
# `carrier_phase` (lab frame only) multiplies the BIAS and nothing else:
# `z_in` already carries the carrier, but the stored bias is a
# co-rotating-frame quantity. `nothing` leaves the expression bit-identical
# to the original so the co-rotating hot path is unaffected.
@inline function _cached_drive(cache, l, chain, key, ps_l, st_l, z_in;
                               carrier_phase::Union{Nothing, ComplexF32} = nothing)
    Wc = cache === nothing ? nothing : cache.drive[l]
    Wc === nothing && return ep_drive(chain.layers[key], ps_l, st_l, z_in)
    y = Wc * z_in
    if haskey(ps_l, :bias_real)
        b = ps_l.bias_real .+ 1f0im .* ps_l.bias_imag
        y = y .+ (carrier_phase === nothing ? b : carrier_phase .* b)
    end
    return y
end

@inline function _cached_feedback(cache, l, chain, key, ps_l, st_l, z_out)
    Wt = cache === nothing ? nothing : cache.feedback[l]
    Wt === nothing && return ep_feedback(chain.layers[key], ps_l, st_l, z_out)
    return Wt * z_out
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
                            omega_override::Union{Nothing, Vector} = nothing,
                            project::Symbol = :hard)
    ε_f  = Float32(ε)

    function loss_at(ps_perturbed)
        s = phasor_settle(chain, ps_perturbed, st, x, cost, 0f0;
                          T=T, dt=dt, K_mode=K_mode,
                          omega_override=omega_override, project=project)
        return ep_loss(cost, s[end])
    end

    base = loss_at(ps)

    # Walk every layer; FD each EP-trained parameter (weight, plus
    # bias if present), build a matched-shape gradient NamedTuple.
    pairs = Pair{Symbol,Any}[]
    for key in keys(ps)
        layer_ps = ps[key]
        if haskey(layer_ps, :ff) && haskey(layer_ps, :alpha)
            # ResidualBlock: recursively FD the branch chain
            ff_grad = _fd_nested_params(ps, key, layer_ps.ff, loss_at, ε_f)
            # FD alpha
            P = layer_ps.alpha
            gP = zeros(Float32, size(P))
            for i in eachindex(P)
                Pp = copy(P)
                Pp[i] += ε_f
                ps_perturbed = _replace_param(ps, key, :alpha, Pp)
                gP[i] = (loss_at(ps_perturbed) - base) / ε_f
            end
            filled = (ff = ff_grad, alpha = gP)
            push!(pairs, key => filled)
        else
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
    end
    return NamedTuple(pairs)
end

# FD for nested param structures (ResidualBlock branch chains)
function _fd_nested_params(ps, layer_key, ps_struct, loss_at, ε_f)
    pairs = Pair{Symbol,Any}[]
    for k in keys(ps_struct)
        ps_k = ps_struct[k]
        if haskey(ps_k, :weight)
            filled = NamedTuple()
            for pname in (:weight, :bias_real, :bias_imag)
                haskey(ps_k, pname) || continue
                P = ps_k[pname]
                gP = zeros(Float32, size(P))
                for i in eachindex(P)
                    Pp = copy(P)
                    Pp[i] += ε_f
                    # Perturb the nested param
                    ps_perturbed = _replace_nested_param(ps, layer_key, k, pname, Pp)
                    gP[i] = (loss_at(ps_perturbed) - loss_at(ps)) / ε_f
                end
                filled = merge(filled, NamedTuple{(pname,)}((gP,)))
            end
            push!(pairs, k => _zero_other_params(ps_k, filled))
        else
            push!(pairs, k => _fd_nested_params(ps, layer_key, ps_k, loss_at, ε_f))
        end
    end
    return NamedTuple(pairs)
end

# Replace a nested param: ps[layer_key].ff[k][pname] = V
function _replace_nested_param(ps, layer_key, inner_key, pname, V)
    inner = ps[layer_key]
    ff = inner.ff
    ff_k = ff[inner_key]
    new_ff_k = merge(ff_k, NamedTuple{(pname,)}((V,)))
    new_ff = merge(ff, NamedTuple{(inner_key,)}((new_ff_k,)))
    new_inner = merge(inner, NamedTuple{( :ff,)}((new_ff,)))
    return merge(ps, NamedTuple{(layer_key,)}((new_inner,)))
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

`K_mode = :zero` (default) skips the self-force entirely — the K = 0
phase-consensus settle. `K_mode = :stored` adds `½·λ·z`, so the
equilibrium reflects the layer's per-channel decay. It settles at the
default `dt = 0.5` and the layer's default `ω = 2π`; ω does not enter
(it is symplectic — see [`ep_self_force`](@ref)), so `omega_override`
has no effect on either mode. To actually rotate, pass `carrier` to
[`phasor_settle`](@ref).
"""
Base.@kwdef struct StaticEP <: AbstractEPMethod
    β::Float32      = 0.1f0
    T_free::Int     = 100
    T_nudge::Int    = 50
    dt::Float32     = 0.5f0
    K_mode::Symbol  = :zero
    centered::Bool  = false
    # :hard (default) is the discontinuous unit projection; :soft blends
    # phase in with a sigmoid in |g|. See `_project_damp_soft`.
    project::Symbol = :hard
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
                            omega_override=omega_override, project=m.project)
    s_pos   = phasor_settle(chain, ps, st, x, cost, m.β;
                            T=m.T_nudge, dt=m.dt, init=s_free, K_mode=m.K_mode,
                            omega_override=omega_override, project=m.project)

    h_free = chain_hebbians(chain, ps, st, x, s_free)
    h_pos  = chain_hebbians(chain, ps, st, x, s_pos)

    if m.centered
        # Symmetric (centered) estimator: settle at -β as well and use
        # -(h₊ - h₋)/(2β). The one-sided difference carries an O(β)
        # bias; the symmetric one cancels it, leaving O(β²). Costs one
        # extra nudged settle.
        s_neg = phasor_settle(chain, ps, st, x, cost, -m.β;
                              T=m.T_nudge, dt=m.dt, init=s_free, K_mode=m.K_mode,
                              omega_override=omega_override, project=m.project)
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
        layer = chain.layers[key]
        ps_l = ps[key]; st_l = st[key]
        
        # Extract z_in from previous state (handle ResidualBlock tuple)
        z_in = if l == 1
            z0
        else
            z_prev = states[l-1]
            z_prev isa Tuple ? z_prev[1] : z_prev
        end
        
        # Handle ResidualBlock: states[l] is (z_out, z_branch)
        z_self = states[l]
        if layer isa ResidualBlock
            z_out, z_branch = z_self
            # For ResidualBlock, ep_hebbian takes (z_in, z_self) where z_self is z_out
            # It internally computes branch hebbian using z_branch
            h_l = ep_hebbian(layer, ps_l, st_l, z_in, z_out)
        else
            z_out = z_self isa Tuple ? z_self[1] : z_self
            h_l = haskey(ps_l, :weight) ?
                ep_hebbian(layer, ps_l, st_l, z_in, z_out) :
                _zero_grad(ps_l)
        end
        push!(pairs, key => h_l)
    end
    return NamedTuple(pairs)
end

# Build the gradient NamedTuple by differencing per-layer Hebbians
# and dividing by β. Handles weight + bias (when present), alpha (for
# ResidualBlock), and zeros-out non-EP-trained params (log_neg_lambda).
function _ep_diff_gradient(ps, h_free, h_nudge, β)
    inv_β = -1f0 / Float32(β)
    pairs = Pair{Symbol,Any}[]
    for key in keys(ps)
        layer_ps = ps[key]
        if haskey(layer_ps, :weight)
            # PhasorDense, PhasorBind, etc.
            entry = (weight = inv_β .* (h_nudge[key].weight .- h_free[key].weight),)
            if haskey(layer_ps, :bias_real)
                entry = merge(entry, (
                    bias_real = inv_β .* (h_nudge[key].bias_real .- h_free[key].bias_real),
                    bias_imag = inv_β .* (h_nudge[key].bias_imag .- h_free[key].bias_imag),
                ))
            end
            # Alpha gradient for ResidualBlock (if somehow present)
            if haskey(layer_ps, :alpha)
                entry = merge(entry, (
                    alpha = inv_β .* (h_nudge[key].alpha .- h_free[key].alpha),
                ))
            end
            entry = _pad_dynamics_zeros(entry, layer_ps)
            push!(pairs, key => entry)
        elseif haskey(layer_ps, :ff) && haskey(layer_ps, :alpha)
            # ResidualBlock: params are (ff = (layer_1 = ...), alpha = ...)
            # Hebbians are (ff = (layer_1 = ...), alpha = ...)
            ff_free = h_free[key].ff
            ff_nudge = h_nudge[key].ff
            alpha_free = h_free[key].alpha
            alpha_nudge = h_nudge[key].alpha
            
            # Diff the branch chain params (recursively handle nested structure)
            ff_grad = _diff_nested_params(layer_ps.ff, ff_free, ff_nudge, inv_β)
            alpha_grad = inv_β .* (alpha_nudge .- alpha_free)
            
            entry = (ff = ff_grad, alpha = alpha_grad)
            push!(pairs, key => entry)
        else
            push!(pairs, key => _zero_grad(layer_ps))
        end
    end
    return NamedTuple(pairs)
end

# Recursively difference nested parameter structures (for ResidualBlock branch chains)
function _diff_nested_params(ps_struct, h_free, h_nudge, inv_β)
    pairs = Pair{Symbol,Any}[]
    for k in keys(ps_struct)
        ps_k = ps_struct[k]
        if haskey(ps_k, :weight)
            entry = (weight = inv_β .* (h_nudge[k].weight .- h_free[k].weight),)
            if haskey(ps_k, :bias_real)
                entry = merge(entry, (
                    bias_real = inv_β .* (h_nudge[k].bias_real .- h_free[k].bias_real),
                    bias_imag = inv_β .* (h_nudge[k].bias_imag .- h_free[k].bias_imag),
                ))
            end
            entry = _pad_dynamics_zeros(entry, ps_k)
            push!(pairs, k => entry)
        else
            push!(pairs, k => _diff_nested_params(ps_k, h_free[k], h_nudge[k], inv_β))
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
# ---- spike-timing readout floor ---------------------------------
#
# On a real spiking substrate a neuron's phase is not read off a complex
# number — it is inferred from WHEN the neuron spiked, and that time is
# resolvable only to about the spike-kernel width. `SpikingArgs` defaults
# to `t_window = 0.01` against `t_period = 1.0`, and `bias_current` smears
# each pulse over `±2·t_window`, so the readout quantum is a few percent
# of a full turn.
#
# That matters because it puts a LOWER bound on the probe amplitude ε: the
# lock-in has to resolve a response of order ε·χ above this floor. The
# usual advice for the estimator's other failure mode — the hard
# projection basin-hopping at large ε — is to shrink ε, which drives
# straight into this floor from the other side. The usable zone for a
# spiking implementation is therefore TWO-SIDED, and a sweep on exact
# complex states can only ever see the upper edge of it.
#
# Note this is not a foregone conclusion: the lock-in integrates over many
# probe cycles, and the state sweeps across bin boundaries as it goes, so
# the probe dithers the quantizer and time-averaging recovers some
# sub-quantum resolution. How much is exactly what the sweep measures.
#
# The grid here is fixed in whatever frame the states are in, which for
# the default settle is the CO-ROTATING frame. That is the pessimistic
# case: the envelope is slowly varying, so the only thing sweeping the
# state across bins is the probe, and the probe sweeps by less than a bin
# — hence the dead zone. A physical spike-time clock may instead be fixed
# in the LAB frame, where the carrier sweeps the state across tens of bins
# per step and dithers the quantizer for free. That is not a free change
# of variables (rounding is not U(1)-equivariant) and it measurably
# matters: 0.44 -> 0.998 median cos at delta = 0.005 turns. Which frame is
# right is a hardware question about what the readout clock is locked to;
# see phasor_settle's docstring and docs/ep_rotating_followups.md.
#
# Modelled here as readout-only: the dynamics stay analog (a membrane
# potential is continuous) and only the value entering the Hebbian is
# quantized. Quantizing inter-layer communication as well is a strictly
# stronger constraint and is not attempted here.
@inline function _quantize_phase(z::ComplexF32, inv_δ::Float32)
    turns = angle(z) * (1f0 / (2f0 * pi_f32))
    q     = round(turns * inv_δ) / inv_δ
    return abs(z) * ComplexF32(cis(2f0 * pi_f32 * q))
end

# Dithered readout: jitter the phase (in turns) BEFORE quantizing.
#
# This matters more than it looks. A deterministic quantizer is the
# *no-dither* limit: if the probe response is smaller than one bin, the
# observed value never changes, the demodulated sum is identically zero,
# and the estimated gradient is zero — not noisy, ZERO. Measured on the
# 784→256→64 grid, a 0.005-turn quantum takes the best median cos from
# 0.999 to 0.07 with 100% failure at every (ε, ω_p, n_cycles).
#
# But real spike timing is noisy, and noise dithers a quantizer: it
# converts a dead zone into a biased coin whose mean tracks the sub-bin
# value, which time-averaging over the lock-in window can then recover.
# So the honest question is not "does quantization hurt" (it does,
# catastrophically) but "does the jitter that accompanies it in any real
# device buy the resolution back". `readout_jitter` is what measures that.
@inline function _dither_quantize(z::ComplexF32, inv_δ::Float32, n::Float32)
    turns = angle(z) * (1f0 / (2f0 * pi_f32)) + n
    q     = round(turns * inv_δ) / inv_δ
    return abs(z) * ComplexF32(cis(2f0 * pi_f32 * q))
end

# `noise` must be pre-drawn by the caller: `ep_gradient` owns one RNG for
# the whole estimate so the result is reproducible from the method's seed
# and so no RNG is allocated per step.
# `δ` and `jitter` are both in TURNS (so t_window/t_period, not radians).
# δ = jitter = 0 returns the argument untouched and allocation-free.
function _readout(z, δ::Float32, jitter::Float32, noise)
    δ <= 0f0 && jitter <= 0f0 && return z          # exact
    if δ <= 0f0                                     # jitter only, no grid
        return abs.(z) .* ComplexF32.(cis.(angle.(z) .+ 2f0 .* pi_f32 .* noise))
    end
    # Quantized. `noise === nothing` is the undithered case and must route
    # to the plain quantizer — the dither kernel takes a Float32 per element.
    noise === nothing && return _quantize_phase.(z, 1f0 / δ)
    return _dither_quantize.(z, 1f0 / δ, noise)
end

Base.@kwdef struct LockinEP <: AbstractEPMethod
    ε::Float32                 = 0.05f0
    ω_p::Float32               = 0.05f0
    n_cycles::Int              = 8
    T_warmup_cycles::Int       = 2
    T_free::Int                = 200
    dt::Float32                = 0.1f0
    K_mode::Symbol             = :zero
    # Spike-timing phase quantum in turns (t_window / t_period). 0 = exact
    # complex readout, i.e. the original estimator. See `_readout`.
    readout_δ::Float32         = 0f0
    # :hard (default) or :soft — see `_project_damp_soft`.
    project::Symbol            = :hard
    # Spike-time jitter, std in TURNS, applied before quantization. See
    # `_dither_quantize` — this is what tests whether device noise buys
    # back the resolution that `readout_δ` destroys.
    readout_jitter::Float32    = 0f0
    readout_seed::Int          = 1234
    # Lab-frame carrier ω for resonate-and-fire substrates. When set, the
    # settle runs in the lab frame (states rotate at ω) and the readout
    # frame determines whether quantization happens in the lab or
    # co-rotating frame.
    # Can be: nothing (co-rotating), single Float32 (shared carrier),
    # or Vector{Float32} (per-layer carrier for detuning experiments).
    carrier::Union{Nothing, Float32, Vector{Float32}} = nothing
    # Readout frame: :co_rotating (quantize then demodulate, original
    # behavior) or :lab (demodulate carrier off, then quantize).
    readout_frame::Symbol      = :co_rotating
    # Subsampling factor for the lock-in integration loop. 1 = every step,
    # period_steps = one sample per probe period. Affects T_lockin and
    # demodulator phase increment.
    sample_every::Int          = 1
end

function ep_gradient(m::LockinEP, chain::Lux.Chain, ps, st, x,
                     cost::AbstractEPCost;
                     omega_override::Union{Nothing, Vector} = nothing)
    # 1. Free settle to the β=0 equilibrium and snapshot the DC hebbians.
    #    If carrier is set, run in lab frame with t0=0.
    # Support per-layer carriers for detuning experiments.
    if m.carrier === nothing
        carriers_f = nothing
    elseif m.carrier isa AbstractVector
        carriers_f = Float32.(m.carrier)
    else
        carriers_f = fill(Float32(m.carrier), length(keys(ps)))
    end
    s_free = phasor_settle(chain, ps, st, x, cost, 0f0;
                           T=m.T_free, dt=m.dt, K_mode=m.K_mode,
                           omega_override=omega_override, project=m.project,
                           carrier=carriers_f, t0=0f0)
    # Time at end of free settle (start of warmup).
    t_free_end = Float32(m.T_free * m.dt)

    # The DC Hebbian is subtracted from the demodulated accumulators, so it
    # must be read through the SAME quantizer — otherwise the mismatch
    # between an exact DC term and a quantized AC term would masquerade as
    # a response.
    # One RNG per gradient estimate: reproducible from `readout_seed`, and
    # no allocation inside the integration loop.
    ro_rng = Xoshiro(m.readout_seed)
    _noise(a) = m.readout_jitter <= 0f0 ? nothing :
                m.readout_jitter .* randn(ro_rng, Float32, size(a))

    # For per-layer carriers, use the first layer's carrier for readout frame
    carrier_f = carriers_f === nothing ? nothing : carriers_f[1]
    
    # Readout function depends on frame.
    # In co-rotating frame: demodulate carrier (if present), then quantize.
    #   This matches the original LockinEP behavior when carrier=nothing.
    #   When carrier is set, this corresponds to a lab-frame settle with
    #   co-rotating frame readout (carrier demodulated before quantization).
    # In lab frame: quantize the lab-frame state (which rotates with carrier),
    # then demodulate the carrier by multiplying by conj(carrier_phase).
    # This matches the physical scenario where the readout clock is in the lab frame.
    _ro = if m.readout_frame === :co_rotating
        (a, t) -> begin
            if carrier_f === nothing
                return _readout(a, m.readout_δ, m.readout_jitter, _noise(a))
            end
            # Demodulate carrier first, then quantize
            ph = ComplexF32(cis(mod(Float64(carrier_f) * Float64(t), 2π)))
            return _readout(a .* conj(ph), m.readout_δ, m.readout_jitter, _noise(a))
        end
    elseif m.readout_frame === :lab
        (a, t) -> begin
            if carrier_f === nothing
                return _readout(a, m.readout_δ, m.readout_jitter, _noise(a))
            end
            # Quantize lab-frame state, then demodulate carrier
            ph = ComplexF32(cis(mod(Float64(carrier_f) * Float64(t), 2π)))
            return _readout(a, m.readout_δ, m.readout_jitter, _noise(a)) .* conj(ph)
        end
    else
        error("Unknown readout_frame: $(m.readout_frame). Expected :co_rotating or :lab")
    end

    z0c    = _phase_input_to_complex(x)

    # For lab frame, the input must be put on the carrier before readout.
    # Carrier phase at t_free_end (using first layer's carrier).
    ph_free_end = carrier_f === nothing ? ComplexF32(1) :
                  ComplexF32(cis(mod(Float64(carrier_f) * Float64(t_free_end), 2π)))
    z0_lab_free = z0c .* ph_free_end

    z0_ro  = _ro(z0_lab_free, t_free_end)
    h_dc   = chain_hebbians(chain, ps, st, z0_ro, [_ro(z, t_free_end) for z in s_free])

    # 2. Lock-in setup.
    layer_keys = collect(keys(ps))
    z0 = _phase_input_to_complex(x)
    period_steps = round(Int, 2π / (m.ω_p * m.dt))
    T_warmup = m.T_warmup_cycles * period_steps
    T_lockin = m.n_cycles        * period_steps

    # Subsampling: only accumulate every `sample_every` steps.
    # This affects the effective T_lockin and the demodulator phase increment.
    se = max(1, m.sample_every)
    T_lockin_eff = div(T_lockin, se)
    if T_lockin_eff == 0
        T_lockin_eff = 1
    end

    # Copy states, handling tuples for ResidualBlock
    _copy_state(s) = s isa Tuple ? (copy(s[1]), copy(s[2])) : copy(s)
    states = [_copy_state(s) for s in s_free]
    cache  = _weight_cache(chain, ps, layer_keys)
    drive0 = _input_drive(chain, ps, st, layer_keys, z0; cache=cache)

    # 3. Warm-up — drive the probe but don't accumulate (transients die).
    #    Track absolute time: warmup starts at t_free_end.
    for t in 1:T_warmup
        β_t = m.ε * cos(m.ω_p * t * m.dt)
        t_now = t_free_end + Float32((t - 1) * m.dt)
        states = _phasor_step(chain, ps, st, layer_keys, z0, cost,
                              β_t, m.dt, states; K_mode=m.K_mode,
                              omega_override=omega_override, drive0=drive0,
                              cache=cache, project=m.project,
                              carriers=carriers_f, t_now=t_now)
    end

    # 4. Accumulators.
    #
    # `Ẑ[l] = Σ_t z_l(t)·e^{-iω_p t}` — the demodulated state of every
    # layer, `(out_l, B)`. This gives the bias gradient for every layer
    # directly.
    #
    # For layer 1's weight gradient: in the co-rotating frame, the input `z₀`
    # is constant, so `Σ_t z₁(t)·z₀' ·e^{-iω_p t} = (Σ_t z₁(t)e^{-iω_p t})·z₀'`
    # factors out (optimization). In the lab frame, the input rotates with
    # the carrier, so we must accumulate the outer product per step.
    #
    # `HW[l]` for l ≥ 1 accumulates `Σ_t demod · z_l · z_{l-1}'` (adjoint).
    # For l=1, `z_{l-1}` is the input `z₀` (put on carrier for lab frame).
    #
    # `c = Σ_t e^{-iω_p t}` carries the DC subtraction out of the loop
    # too: `Σ_t (h(t) - h_dc)·e^{-iω_p t} = Σ_t h(t)e^{-iω_p t} - c·h_dc`.
    # It is ≈0 over integer cycles but is kept exact.
    _state_template(s) = s isa Tuple ? s[1] : s
    _state_size(s) = s isa Tuple ? size(s[1]) : size(s)
    _branch_template(s) = s isa Tuple ? s[2] : nothing
    _branch_size(s) = s isa Tuple ? size(s[2]) : nothing
    Zhat = [gpu_zeros(_state_template(states[l]), ComplexF32, _state_size(states[l])...) for l in 1:length(layer_keys)]
    Zhat_branch = [(_branch_template(states[l]) === nothing) ? nothing : gpu_zeros(_branch_template(states[l]), ComplexF32, _branch_size(states[l])...) for l in 1:length(layer_keys)]
    HW   = Vector{Any}(nothing, length(layer_keys))
    for (l, key) in enumerate(layer_keys)
        layer_ps = ps[key]
        if haskey(layer_ps, :weight)
            HW[l] = gpu_zeros(ps[key].weight, ComplexF32, size(ps[key].weight)...)
        elseif haskey(layer_ps, :ff) && haskey(layer_ps, :alpha)
            # ResidualBlock: allocate HW for the branch layer
            branch_ps = layer_ps.ff.layer_1
            HW[l] = gpu_zeros(branch_ps.weight, ComplexF32, size(branch_ps.weight)...)
        end
    end
    c = zero(ComplexF32)

    # Time at end of warmup (start of lock-in integration).
    t_warm_end = t_free_end + Float32(T_warmup * m.dt)

    # 5. Integration: settle + demodulated accumulation.
    #    Subsample by `se`: only accumulate on steps where (t-1) % se == 0.
    for t in 1:T_lockin
        β_t   = m.ε * cos(m.ω_p * t * m.dt)
        t_now = t_warm_end + Float32((t - 1) * m.dt)
        states = _phasor_step(chain, ps, st, layer_keys, z0, cost,
                              β_t, m.dt, states; K_mode=m.K_mode,
                              omega_override=omega_override, drive0=drive0,
                              cache=cache, project=m.project,
                              carriers=carriers_f, t_now=t_now)

        # Subsample the lock-in accumulation.
        if (t - 1) % se == 0
            # Effective lock-in step index (0-based).
            t_eff = div(t - 1, se)
            # Demodulator phase: tracks the probe phase which is ω_p * t * dt
            # where t is the lock-in step (1-based). For subsampled steps,
            # the effective probe phase is ω_p * (t_eff * se + 1) * dt.
            # The demodulator uses the same phase as the probe at that step.
            # Using 0-based: phase = ω_p * (t_eff * se + 1) * dt
            # But the original uses t * dt for step t (1-based), so for
            # effective step t_eff (0-based) corresponding to original step t = t_eff*se + 1:
            demod_phase = m.ω_p * Float32(t_eff * se + 1) * m.dt
            demod = ComplexF32(exp(-im * demod_phase))
            c += demod

            # Readout time: absolute time for carrier demodulation in lab frame.
            # The state returned by _phasor_step at step t is at time t_now + dt.
            t_sample = t_warm_end + Float32(t_eff * se + 1) * m.dt
            _extract_obs(z) = z isa Tuple ? z[1] : z
            if m.readout_δ <= 0f0 && m.readout_jitter <= 0f0
                obs = [_extract_obs(z) for z in states]
            else
                obs = [_ro(_extract_obs(z), t_sample) for z in states]
            end

            # Lab-frame input at this sample time (for layer 1's Hebbian).
            ph_sample = carrier_f === nothing ? ComplexF32(1) :
                        ComplexF32(cis(mod(Float64(carrier_f) * Float64(t_sample), 2π)))
            z0_lab_sample = z0c .* ph_sample
            z0_ro_sample = _ro(z0_lab_sample, t_sample)

            for l in 1:length(layer_keys)
                Zhat[l] .+= obs[l] .* demod
                # Track branch state for ResidualBlock alpha gradient
                if Zhat_branch[l] !== nothing
                    z_branch = states[l] isa Tuple ? states[l][2] : nothing
                    if z_branch !== nothing
                        Zhat_branch[l] .+= z_branch .* demod
                    end
                end
                HW[l] === nothing && continue
                # H += demod · z_l · z_{l-1}'  (adjoint, not transpose —
                # the energy derivative requires conjugation; see ep_hebbian)
                z_in = (l == 1) ? z0_ro_sample : obs[l-1]
                mul!(HW[l], obs[l], adjoint(z_in), demod, one(ComplexF32))
            end
        end
    end

    # 6. Convert to gradient: dL/d(real-param) = -2·Re(H) / (T_lockin_eff · ε).
    #    The factor of 2 comes from the real cosine probe — see the
    #    design doc, section "Implementation sketch". For bias the
    #    real/imag parts of H_b give the bias_real / bias_imag grads
    #    respectively (since H_b's "complex" packaging is z_self and
    #    Re(z_self), Im(z_self) are independent params).
    H_W, H_b = _lockin_accumulators(ps, layer_keys, Zhat, Zhat_branch, HW, z0_ro, h_dc, c)
    grads = _ep_lockin_gradient(ps, H_W, H_b, T_lockin_eff, m.ε)
    return grads, s_free
end

# Close out the lock-in accumulators into DC-subtracted, batch-normalized
# complex Hebbians keyed by layer. `h_dc` entries are already divided by
# B (see `ep_hebbian`), so the raw accumulators get the same treatment
# before the DC term is subtracted.
function _lockin_accumulators(ps, layer_keys, Zhat, Zhat_branch, HW, z0, h_dc, c)
    H_W = Dict{Symbol, Any}()
    H_b = Dict{Symbol, Any}()
    for (l, key) in enumerate(layer_keys)
        layer_ps = ps[key]
        invB = one(Float32) / Float32(_batch_size(Zhat[l]))
        if haskey(layer_ps, :weight)
            # PhasorDense, PhasorBind, etc.
            raw = HW[l]
            H_W[key] = raw .* invB .- c .* ComplexF32.(h_dc[key].weight)
            if haskey(layer_ps, :bias_real)
                dc_b = ComplexF32.(h_dc[key].bias_real .+ 1f0im .* h_dc[key].bias_imag)
                H_b[key] = _sum_batch(Zhat[l]) .* invB .- c .* dc_b
            end
        elseif haskey(layer_ps, :ff) && haskey(layer_ps, :alpha)
            # ResidualBlock: params are (ff = (layer_1 = ...), alpha = ...)
            # Hebbians are (ff = (layer_1 = ...), alpha = ...)
            # The HW[l] for ResidualBlock is the branch layer's HW
            # We need to recursively extract branch Hebbians
            ff_H_W, ff_H_b = _lockin_nested_accumulators(layer_ps.ff, h_dc[key].ff, HW[l], Zhat_branch[l], invB, c)
            H_W[key] = (ff = ff_H_W,)
            # Alpha gradient from demodulated branch state
            # dE/dα = imag(z_out ⊙ conj(z_branch^α) ⊙ phase(z_branch)) / B
            # Lock-in extracts: demod · z_branch component at ω_p
            # The alpha gradient is -2 * Re(Zhat_branch_alpha) / (T_lockin * ε)
            if Zhat_branch[l] !== nothing
                # Alpha hebbian is the demodulated branch state
                # For alpha: H_b_alpha = Σ_t demod · z_branch / B  (demodulated at ω_p)
                # The final alpha gradient uses: -2 * real(H_b_alpha) / (T_lockin * ε)
                # But we need z_branch^α * phase(z_branch) - approximate from demodulated z_branch
                # For small α near 1, z_branch^α ≈ z_branch, phase(z_branch) is angle(z_branch)/(2π)
                # This is an approximation; full alpha gradient needs more careful treatment
                # For now, sum over channels to make it scalar (matching alpha param size)
                branch_hebbian = _sum_batch(Zhat_branch[l]) .* invB .- c .* h_dc[key].alpha
                # Sum over channels to get scalar
                alpha_hebbian = sum(branch_hebbian)
                H_b[key] = (ff = ff_H_b, alpha = [alpha_hebbian])
            else
                H_b[key] = (ff = ff_H_b, alpha = zero.(h_dc[key].alpha))
            end
        end
    end
    return H_W, H_b
end

# Recursively compute lock-in accumulators for nested parameter structures (for ResidualBlock branch chains)
# Returns (weight_hebbians, bias_hebbians) where bias_hebbians are complex vectors (real=bias_real, imag=bias_imag)
function _lockin_nested_accumulators(ps_struct, h_dc_struct, HW_raw, Zhat_branch_layer, invB, c)
    weight_pairs = Pair{Symbol, Any}[]
    bias_pairs = Pair{Symbol, Any}[]
    # For ResidualBlock with single PhasorDense branch, HW_raw is the branch's HW
    for k in keys(ps_struct)
        ps_k = ps_struct[k]
        if haskey(ps_k, :weight)
            raw = HW_raw
            # Weight hebbian
            push!(weight_pairs, k => (raw .* invB .- c .* ComplexF32.(h_dc_struct[k].weight)))
            # Bias hebbian: from demodulated branch state (Zhat_branch_layer)
            if Zhat_branch_layer !== nothing
                bias_hebbian = _sum_batch(Zhat_branch_layer) .* invB .- c .* ComplexF32.(h_dc_struct[k].bias_real .+ 1f0im .* h_dc_struct[k].bias_imag)
                push!(bias_pairs, k => bias_hebbian)
            else
                push!(bias_pairs, k => zero.(h_dc_struct[k].bias_real) .+ 1f0im .* zero.(h_dc_struct[k].bias_imag))
            end
        end
    end
    return NamedTuple(weight_pairs), NamedTuple(bias_pairs)
end

function _ep_lockin_gradient(ps, H_W::AbstractDict, H_b::AbstractDict,
                              T_lockin::Int, ε)
    norm_factor = Float32(T_lockin) * Float32(ε)
    pairs = Pair{Symbol,Any}[]
    for key in keys(ps)
        layer_ps = ps[key]
        if haskey(layer_ps, :weight)
            # PhasorDense, PhasorBind, etc.
            entry = (weight = -2f0 .* real.(H_W[key]) ./ norm_factor,)
            if haskey(H_b, key)
                # Re(H_b) → bias_real grad; Im(H_b) → bias_imag grad.
                entry = merge(entry, (
                    bias_real = -2f0 .* real.(H_b[key]) ./ norm_factor,
                    bias_imag = -2f0 .* imag.(H_b[key]) ./ norm_factor,
                ))
            end
            # Alpha gradient for ResidualBlock (if somehow present)
            if haskey(layer_ps, :alpha)
                entry = merge(entry, (
                    alpha = -2f0 .* real.(H_b[key].alpha) ./ norm_factor,
                ))
            end
            entry = _pad_dynamics_zeros(entry, layer_ps)
            push!(pairs, key => entry)
        elseif haskey(layer_ps, :ff) && haskey(layer_ps, :alpha)
            # ResidualBlock: params are (ff = (layer_1 = ...), alpha = ...)
            # Hebbians are (ff = (layer_1 = ...), alpha = ...)
            ff_H_W = H_W[key].ff
            ff_H_b = H_b[key].ff
            alpha_H_b = H_b[key].alpha
            
            # Diff the branch chain params (recursively handle nested structure)
            ff_grad = _lockin_nested_gradient(layer_ps.ff, ff_H_W, ff_H_b, norm_factor)
            alpha_grad = -2f0 .* real.(alpha_H_b) ./ norm_factor
            
            entry = (ff = ff_grad, alpha = alpha_grad)
            push!(pairs, key => entry)
        else
            push!(pairs, key => _zero_grad(layer_ps))
        end
    end
    return NamedTuple(pairs)
end

# Recursively compute lock-in gradient for nested parameter structures (for ResidualBlock branch chains)
function _lockin_nested_gradient(ps_struct, H_W, H_b, norm_factor)
    pairs = Pair{Symbol,Any}[]
    for k in keys(ps_struct)
        ps_k = ps_struct[k]
        if haskey(ps_k, :weight)
            entry = (weight = -2f0 .* real.(H_W[k]) ./ norm_factor,)
            if haskey(H_b, k)
                entry = merge(entry, (
                    bias_real = -2f0 .* real.(H_b[k]) ./ norm_factor,
                    bias_imag = -2f0 .* imag.(H_b[k]) ./ norm_factor,
                ))
            end
            entry = _pad_dynamics_zeros(entry, ps_k)
            push!(pairs, k => entry)
        else
            push!(pairs, k => _zero_grad(ps_k))
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

`weight_mask` (optional) — a NamedTuple matching `ps` structure with
values in [0,1] to scale gradients per parameter. Used to freeze
stuck synapses (mask=0.0) during fine-tuning with impaired weights.
"""
function ep_train(model::Lux.Chain, ps, st, train_loader, args;
                  method::AbstractEPMethod = StaticEP(),
                  cost_fn::Function = _default_cost_fn,
                  optimiser = Optimisers.Descent,
                  callback = nothing,
                  omega_override::Union{Nothing, Vector} = nothing,
                  verbose::Bool = false,
                  weight_mask::Union{Nothing, NamedTuple} = nothing)
    opt_state = Optimisers.setup(optimiser(Float32(args.lr)), ps)
    losses = Float32[]
    for epoch in 1:args.epochs
        epoch_start = length(losses) + 1
        for (x, y) in train_loader
            cost = cost_fn(y)
            grads, s_free = ep_gradient(method, model, ps, st, x, cost;
                                        omega_override=omega_override)
            if args.weight_decay > 0
                grads = _apply_weight_decay(grads, ps, args.weight_decay)
            end
            if weight_mask !== nothing
                grads = _apply_weight_mask(grads, weight_mask)
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

# Apply weight mask to gradients (zero out stuck synapses)
# mask has same structure as ps, with 0.0 for stuck, 1.0 for free
function _apply_weight_mask(grads, mask)
    pairs = Pair{Symbol,Any}[]
    for key in keys(grads)
        g = grads[key]
        m = mask[key]
        if haskey(g, :weight)
            entry = (weight = g.weight .* m.weight,)
            if haskey(g, :bias_real)
                entry = merge(entry, (
                    bias_real = g.bias_real .* m.bias_real,
                    bias_imag = g.bias_imag .* m.bias_imag,
                ))
            end
            push!(pairs, key => merge(entry, _pad_dynamics_zeros(entry, g)))
        else
            push!(pairs, key => g)
        end
    end
    return NamedTuple(pairs)
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
