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
# Phase 1 scope:
#   * PhasorDense layers only
#   * StaticEP method only (lock-in / contour follow later)
#   * SimilarityCost only (codebook-cross-entropy follows later)
#   * K = 0 self-energy (ignore log_neg_lambda / omega)
#   * use_bias = false required
#
# Energy function being descended:
#   Phi = sum_l Re<W_l z_{l-1}, z_l>  -  beta * C(z_L, y)
# with cost
#   C(z_L, y) = 1 - (1/d) * Re<y, z_L>.

# ================================================================
# 1. Cost types
# ================================================================

abstract type AbstractEPCost end

"""
    SimilarityCost(y::AbstractVector{<:Complex})

Cosine-distance cost against a fixed unit-complex target `y`:

    C(z, y) = 1 - (1/d) * Re<y, z>

The nudge force `(β/d) * y` uses the **real-parameter convention**
(no factor of 1/2); this is what matches finite differences on real
weights — see `docs/phasor_ep_design.md`, section "Subtle convention
point".
"""
struct SimilarityCost{T<:AbstractVector{<:Complex}} <: AbstractEPCost
    y::T
end

# Real-parameter convention: factor of 1/d, NOT 1/(2d).
# Do not "fix" — see docs/phasor_ep_design.md.
nudge_force(c::SimilarityCost, z_o, β) = (β / length(z_o)) .* c.y
ep_loss(c::SimilarityCost, z_o)        = one(Float32) - real(dot(c.y, z_o)) / length(z_o)

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
struct CodebookCost{C<:AbstractMatrix{<:Complex}, T<:AbstractVector{<:Real}} <: AbstractEPCost
    codes::C
    y_onehot::T
end

function CodebookCost(codes::AbstractMatrix{<:Complex}, y_class::Integer)
    n_classes = size(codes, 2)
    y_oh = zeros(Float32, n_classes)
    @assert 1 <= y_class <= n_classes "y_class out of range: got $y_class, expected 1..$n_classes"
    y_oh[y_class] = one(Float32)
    return CodebookCost(codes, y_oh)
end

function _codebook_logits(c::CodebookCost, z_o)
    d = length(z_o)
    return real.(adjoint(c.codes) * z_o) ./ Float32(d)
end

function _softmax(s::AbstractVector{<:Real})
    s_shift = s .- maximum(s)
    e = exp.(s_shift)
    return e ./ sum(e)
end

function ep_loss(c::CodebookCost, z_o)
    s = _codebook_logits(c, z_o)
    s_shift = s .- maximum(s)
    log_probs = s_shift .- log(sum(exp.(s_shift)))
    return -sum(c.y_onehot .* log_probs)
end

function nudge_force(c::CodebookCost, z_o, β)
    s   = _codebook_logits(c, z_o)
    err = _softmax(s) .- c.y_onehot
    d   = length(z_o)
    return -(Float32(β) / d) .* (c.codes * err)
end

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
# from the layer's params/state. With ω ≈ 0 (e.g. by zeroing
# state.omega) the equilibrium remains close to the unit circle;
# with the layer's default ω = 2π and dt = 0.5 the per-step
# rotation is too large for damped fixed-point iteration to settle,
# so :stored mode typically requires a smaller dt or a chain whose
# omega has been overridden.
function ep_self_force(layer::PhasorDense, ps, st, z_self;
                       K_mode::Symbol = :zero)
    K_mode == :zero && return zero(z_self)
    λ = -exp.(ps.log_neg_lambda)
    ω = layer.trainable_omega ? ps.omega : st.omega
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
    g = (weight = real.(z_self * adjoint(z_in)),)
    if haskey(ps, :bias_real)
        g = merge(g, (bias_real = Float32.(real.(z_self)),
                      bias_imag = Float32.(imag.(z_self))))
    end
    # Match ps shape exactly so Optimisers.update doesn't warn /
    # silently skip. Dynamics params are not EP-updated.
    if haskey(ps, :log_neg_lambda)
        g = merge(g, (log_neg_lambda = zeros(Float32, size(ps.log_neg_lambda)),))
    end
    if haskey(ps, :omega)
        g = merge(g, (omega = zeros(Float32, size(ps.omega)),))
    end
    return g
end

function ep_energy_contribution(::PhasorDense, ps, st, z_in, z_self;
                                K_mode::Symbol = :zero)
    e = Float32(real(dot(z_self, ps.weight * z_in)))
    if haskey(ps, :bias_real)
        bias = ps.bias_real .+ 1f0im .* ps.bias_imag
        e += Float32(real(dot(z_self, bias)))
    end
    if K_mode == :stored
        λ = -exp.(ps.log_neg_lambda)
        ω = layer_omega(z_self, ps, st)   # see helper below
        K = ComplexF32.(λ .+ im .* ω)
        # Energy contribution: ½·Re<z, K·z>
        e += Float32(0.5) * Float32(real(dot(z_self, K .* z_self)))
    end
    return e
end

# Tiny helper used only by ep_energy_contribution; the runtime
# dispatch in ep_self_force does the trainable_omega check itself.
function layer_omega(z_self, ps, st)
    haskey(ps, :omega) && return ps.omega
    return st.omega
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
                       K_mode::Symbol = :zero)
    layer_keys = collect(keys(ps))
    dt_f = Float32(dt)
    β_f  = Float32(β)

    states = init === nothing ?
        [zeros(ComplexF32, chain.layers[k].out_dims) for k in layer_keys] :
        [ComplexF32.(s) for s in init]

    z0 = _phase_input_to_complex(x)

    for _ in 1:T
        states = _phasor_step(chain, ps, st, layer_keys, z0, cost,
                              β_f, dt_f, states; K_mode=K_mode)
    end
    return states
end

# Single damped projected update step across all layers, with the
# given time-varying nudge β. Factored out so phasor_settle and the
# lock-in gradient extraction share the per-step logic. `K_mode`
# selects the self-energy treatment (see `ep_self_force`).
function _phasor_step(chain::Lux.Chain, ps, st, layer_keys, z0,
                      cost::AbstractEPCost, β::Float32, dt::Float32, states;
                      K_mode::Symbol = :zero)
    n = length(layer_keys)
    new_states = Vector{Vector{ComplexF32}}(undef, n)
    for l in 1:n
        key  = layer_keys[l]
        ps_l = ps[key]; st_l = st[key]
        z_in   = (l == 1) ? z0 : states[l-1]
        z_self = states[l]

        grad_l = ep_drive(chain.layers[key], ps_l, st_l, z_in)
        grad_l = grad_l .+ ep_self_force(chain.layers[key], ps_l, st_l, z_self;
                                          K_mode=K_mode)

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
        new_states[l] = (1 - dt) .* z_self .+
                        dt .* normalize_to_unit_circle(grad_l; ε = 0)
    end
    return new_states
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
                            K_mode::Symbol = :zero)
    ε_f  = Float32(ε)

    function loss_at(ps_perturbed)
        s = phasor_settle(chain, ps_perturbed, st, x, cost, 0f0;
                          T=T, dt=dt, K_mode=K_mode)
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
    StaticEP(; β=0.1, T_free=100, T_nudge=50, dt=0.5, K_mode=:zero)

Vanilla EP gradient extraction with a single static real β. The
nudged phase warm-starts from the free equilibrium for tighter
linear-response sampling.

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
                     cost::AbstractEPCost)
    s_free  = phasor_settle(chain, ps, st, x, cost, 0f0;
                            T=m.T_free,  dt=m.dt, K_mode=m.K_mode)
    s_nudge = phasor_settle(chain, ps, st, x, cost, m.β;
                            T=m.T_nudge, dt=m.dt, init=s_free, K_mode=m.K_mode)

    h_free  = chain_hebbians(chain, ps, st, x, s_free)
    h_nudge = chain_hebbians(chain, ps, st, x, s_nudge)

    # EP estimate: -(hebb_nudge - hebb_free) / β. Sign flip because
    # Φ contains -β·C and we want dL/dW.
    grads = _ep_diff_gradient(ps, h_free, h_nudge, m.β)
    return grads, s_free
end

# Back-compat: y as a complex vector → SimilarityCost.
function ep_gradient(m::AbstractEPMethod, chain::Lux.Chain, ps, st, x,
                     y::AbstractVector{<:Complex})
    return ep_gradient(m, chain, ps, st, x, SimilarityCost(ComplexF32.(y)))
end

# Walk the chain, computing per-layer Hebbian outer products at the
# given equilibrium states.
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

# Add zero gradients for log_neg_lambda / omega so the returned
# NamedTuple matches `ps` shape exactly (avoids Optimisers warnings).
function _pad_dynamics_zeros(entry::NamedTuple, layer_ps::NamedTuple)
    if haskey(layer_ps, :log_neg_lambda)
        entry = merge(entry, (log_neg_lambda = zeros(Float32, size(layer_ps.log_neg_lambda)),))
    end
    if haskey(layer_ps, :omega)
        entry = merge(entry, (omega = zeros(Float32, size(layer_ps.omega)),))
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
                     cost::AbstractEPCost)
    # 1. Free settle to the β=0 equilibrium and snapshot the DC hebbians.
    s_free = phasor_settle(chain, ps, st, x, cost, 0f0;
                           T=m.T_free, dt=m.dt, K_mode=m.K_mode)
    h_dc   = chain_hebbians(chain, ps, st, x, s_free)

    # 2. Lock-in setup.
    layer_keys = collect(keys(ps))
    z0 = _phase_input_to_complex(x)
    period_steps = round(Int, 2π / (m.ω_p * m.dt))
    T_warmup = m.T_warmup_cycles * period_steps
    T_lockin = m.n_cycles        * period_steps

    states = [copy(s) for s in s_free]

    # 3. Warm-up — drive the probe but don't accumulate (transients die).
    for t in 1:T_warmup
        β_t = m.ε * cos(m.ω_p * t * m.dt)
        states = _phasor_step(chain, ps, st, layer_keys, z0, cost,
                              β_t, m.dt, states; K_mode=m.K_mode)
    end

    # 4. Per-layer complex Hebbian accumulator. Weight + bias_real
    #    + bias_imag entries when present.
    H_W = Dict{Symbol, Matrix{ComplexF32}}()
    H_b = Dict{Symbol, Vector{ComplexF32}}()
    for key in layer_keys
        haskey(ps[key], :weight) || continue
        H_W[key] = zeros(ComplexF32, size(ps[key].weight))
        if haskey(ps[key], :bias_real)
            H_b[key] = zeros(ComplexF32, size(ps[key].bias_real))
        end
    end

    # 5. Integration: settle + DC-subtracted, demodulated accumulation.
    for t in 1:T_lockin
        β_t   = m.ε * cos(m.ω_p * t * m.dt)
        states = _phasor_step(chain, ps, st, layer_keys, z0, cost,
                              β_t, m.dt, states; K_mode=m.K_mode)
        demod = exp(-im * m.ω_p * t * m.dt)
        for (l, key) in enumerate(layer_keys)
            haskey(ps[key], :weight) || continue
            z_in   = (l == 1) ? z0 : states[l-1]
            z_self = states[l]
            # Weight outer product
            h_W = z_self * adjoint(z_in)
            H_W[key] .+= (h_W .- ComplexF32.(h_dc[key].weight)) .* demod
            # Bias hebbians (per-channel z_self real / imag) packaged
            # as complex so the demodulation is consistent.
            if haskey(H_b, key)
                h_b_complex = z_self
                H_b[key] .+= (h_b_complex .- ComplexF32.(h_dc[key].bias_real .+ 1f0im .* h_dc[key].bias_imag)) .* demod
            end
        end
    end

    # 6. Convert to gradient: dL/d(real-param) = -2·Re(H) / (T_lockin · ε).
    #    The factor of 2 comes from the real cosine probe — see the
    #    design doc, section "Implementation sketch". For bias the
    #    real/imag parts of H_b give the bias_real / bias_imag grads
    #    respectively (since H_b's "complex" packaging is z_self and
    #    Re(z_self), Im(z_self) are independent params).
    grads = _ep_lockin_gradient(ps, H_W, H_b, T_lockin, m.ε)
    return grads, s_free
end

function _ep_lockin_gradient(ps,
                              H_W::Dict{Symbol,Matrix{ComplexF32}},
                              H_b::Dict{Symbol,Vector{ComplexF32}},
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

`args` is the global `Args` struct (see `test/runtests.jl`); only
`lr` and `epochs` are read.
"""
function ep_train(model::Lux.Chain, ps, st, train_loader, args;
                  method::AbstractEPMethod = StaticEP(),
                  cost_fn::Function = _default_cost_fn,
                  verbose::Bool = false)
    opt_state = Optimisers.setup(Optimisers.Descent(Float32(args.lr)), ps)
    losses = Float32[]
    for epoch in 1:args.epochs
        for (x, y) in train_loader
            cost = cost_fn(y)
            grads, s_free = ep_gradient(method, model, ps, st, x, cost)
            opt_state, ps = Optimisers.update(opt_state, ps, grads)
            push!(losses, ep_loss(cost, s_free[end]))
            if verbose
                println("epoch=$epoch loss=$(losses[end])")
            end
        end
    end
    return losses, ps, st
end

_default_cost_fn(y) = SimilarityCost(ComplexF32.(y))
