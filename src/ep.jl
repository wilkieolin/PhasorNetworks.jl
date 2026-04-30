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

# Phase 1: K = 0 (no self-energy term). Returns a zero array of
# matching shape so the settling code can unconditionally `.+= ` it.
ep_self_force(::PhasorDense, ps, st, z_self) = zero(z_self)

"""
    ep_hebbian(::PhasorDense, ps, st, z_in, z_self)

Hebbian outer product `Re(z_self * z_in')`. Uses `adjoint` (not
`transpose`) because the states are complex and the energy
derivative requires conjugation. (hep.jl uses transpose because it
is doing the holomorphic Wirtinger derivative deliberately —
different convention; do not copy.)

Returns a NamedTuple matching `ps`'s trainable-parameter structure,
with zero gradients on `log_neg_lambda` (and `omega` if trainable)
since Phase 1 EP does not update per-channel dynamics.
"""
function ep_hebbian(::PhasorDense, ps, st, z_in, z_self)
    g = (weight = real.(z_self * adjoint(z_in)),)
    # Match ps shape exactly so Optimisers.update doesn't warn /
    # silently skip. Zero out the dynamics params (Phase 1: K = 0).
    if haskey(ps, :log_neg_lambda)
        g = merge(g, (log_neg_lambda = zeros(Float32, size(ps.log_neg_lambda)),))
    end
    if haskey(ps, :omega)
        g = merge(g, (omega = zeros(Float32, size(ps.omega)),))
    end
    return g
end

function ep_energy_contribution(::PhasorDense, ps, st, z_in, z_self)
    return Float32(real(dot(z_self, ps.weight * z_in)))
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
                       init::Union{Nothing,Vector} = nothing)
    layer_keys = collect(keys(ps))
    dt_f = Float32(dt)
    β_f  = Float32(β)

    states = init === nothing ?
        [zeros(ComplexF32, chain.layers[k].out_dims) for k in layer_keys] :
        [ComplexF32.(s) for s in init]

    z0 = _phase_input_to_complex(x)

    for _ in 1:T
        states = _phasor_step(chain, ps, st, layer_keys, z0, cost, β_f, dt_f, states)
    end
    return states
end

# Single damped projected update step across all layers, with the
# given time-varying nudge β. Factored out so phasor_settle and the
# lock-in gradient extraction share the per-step logic.
function _phasor_step(chain::Lux.Chain, ps, st, layer_keys, z0,
                      cost::AbstractEPCost, β::Float32, dt::Float32, states)
    n = length(layer_keys)
    new_states = Vector{Vector{ComplexF32}}(undef, n)
    for l in 1:n
        key  = layer_keys[l]
        ps_l = ps[key]; st_l = st[key]
        z_in   = (l == 1) ? z0 : states[l-1]
        z_self = states[l]

        grad_l = ep_drive(chain.layers[key], ps_l, st_l, z_in)
        grad_l = grad_l .+ ep_self_force(chain.layers[key], ps_l, st_l, z_self)

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
    fd_gradient_phasor(chain, ps, st, x, y; ε=1e-5, T=200, dt=0.5)

Coordinate-by-coordinate forward finite-difference gradient of
`L(ps) = ep_loss(SimilarityCost(y), z_o*_free)` with respect to
each `weight` parameter of every PhasorDense layer in the chain.

Returns a NamedTuple matching `ps`'s structure; entries for
parameters other than `weight` are zero (we don't FD them in Phase
1, since they aren't EP-updated either).

This is the **ground-truth oracle** for the EP gradient. O(n_params)
expensive — for a chain with n_w trainable weight entries, runs
n_w + 1 free-phase settles. Use for tests and small-network
analysis only.
"""
function fd_gradient_phasor(chain::Lux.Chain, ps, st, x, y;
                            ε::Real = 1e-5, T::Int = 200, dt::Real = 0.5f0)
    cost = SimilarityCost(ComplexF32.(y))
    ε_f  = Float32(ε)

    function loss_at(ps_perturbed)
        s = phasor_settle(chain, ps_perturbed, st, x, cost, 0f0; T=T, dt=dt)
        return ep_loss(cost, s[end])
    end

    base = loss_at(ps)

    # Walk every layer; for layers with `weight`, FD each entry and
    # build a matched-shape gradient NamedTuple from scratch.
    pairs = Pair{Symbol,Any}[]
    for key in keys(ps)
        if haskey(ps[key], :weight)
            W = ps[key].weight
            gW = zeros(Float32, size(W))
            for i in eachindex(W)
                Wp = copy(W)
                Wp[i] += ε_f
                ps_perturbed = _replace_weight(ps, key, Wp)
                gW[i] = (loss_at(ps_perturbed) - base) / ε_f
            end
            push!(pairs, key => _zero_other_params(ps[key], (weight = gW,)))
        else
            push!(pairs, key => _zero_other_params(ps[key], NamedTuple()))
        end
    end
    return NamedTuple(pairs)
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

# Replace ps[key].weight with W, returning a new NamedTuple.
function _replace_weight(ps::NamedTuple, key::Symbol, W::AbstractArray)
    inner = ps[key]
    new_inner = merge(inner, (weight = W,))
    return merge(ps, NamedTuple{(key,)}((new_inner,)))
end

# ================================================================
# 5. StaticEP + ep_gradient
# ================================================================

abstract type AbstractEPMethod end

"""
    StaticEP(; β=0.1, T_free=100, T_nudge=50, dt=0.5)

Vanilla EP gradient extraction with a single static real β. The
nudged phase warm-starts from the free equilibrium for tighter
linear-response sampling.
"""
Base.@kwdef struct StaticEP <: AbstractEPMethod
    β::Float32     = 0.1f0
    T_free::Int    = 100
    T_nudge::Int   = 50
    dt::Float32    = 0.5f0
end

"""
    ep_gradient(method, chain, ps, st, x, y) -> (grads, states_free)

Compute the EP gradient for all trainable parameters of `chain`.
Returns a NamedTuple `grads` matching the structure of `ps` (one
entry per layer, with `weight` populated and other params zeroed in
Phase 1) and the free-phase equilibrium states.
"""
function ep_gradient(m::StaticEP, chain::Lux.Chain, ps, st, x, y)
    cost   = SimilarityCost(ComplexF32.(y))
    s_free  = phasor_settle(chain, ps, st, x, cost, 0f0;
                            T=m.T_free,  dt=m.dt)
    s_nudge = phasor_settle(chain, ps, st, x, cost, m.β;
                            T=m.T_nudge, dt=m.dt, init=s_free)

    h_free  = chain_hebbians(chain, ps, st, x, s_free)
    h_nudge = chain_hebbians(chain, ps, st, x, s_nudge)

    # EP estimate: -(hebb_nudge - hebb_free) / β. Sign flip because
    # Φ contains -β·C and we want dL/dW.
    grads = _ep_diff_gradient(ps, h_free, h_nudge, m.β)
    return grads, s_free
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
# and dividing by β. Layers without `weight` get `_zero_grad` of
# their params.
function _ep_diff_gradient(ps, h_free, h_nudge, β)
    pairs = Pair{Symbol,Any}[]
    for key in keys(ps)
        if haskey(ps[key], :weight)
            hf = h_free[key].weight
            hn = h_nudge[key].weight
            gW = -(hn .- hf) ./ Float32(β)
            entry = (weight = gW,)
            if haskey(ps[key], :log_neg_lambda)
                entry = merge(entry, (log_neg_lambda = zeros(Float32, size(ps[key].log_neg_lambda)),))
            end
            if haskey(ps[key], :omega)
                entry = merge(entry, (omega = zeros(Float32, size(ps[key].omega)),))
            end
            push!(pairs, key => entry)
        else
            push!(pairs, key => _zero_grad(ps[key]))
        end
    end
    return NamedTuple(pairs)
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
end

function ep_gradient(m::LockinEP, chain::Lux.Chain, ps, st, x, y)
    cost = SimilarityCost(ComplexF32.(y))

    # 1. Free settle to the β=0 equilibrium and snapshot the DC hebbians.
    s_free = phasor_settle(chain, ps, st, x, cost, 0f0;
                           T=m.T_free, dt=m.dt)
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
                              β_t, m.dt, states)
    end

    # 4. Per-layer complex Hebbian accumulator.
    H_acc = Dict{Symbol, Matrix{ComplexF32}}()
    for key in layer_keys
        if haskey(ps[key], :weight)
            H_acc[key] = zeros(ComplexF32, size(ps[key].weight))
        end
    end

    # 5. Integration: settle + DC-subtracted, demodulated accumulation.
    for t in 1:T_lockin
        β_t   = m.ε * cos(m.ω_p * t * m.dt)
        states = _phasor_step(chain, ps, st, layer_keys, z0, cost,
                              β_t, m.dt, states)
        demod = exp(-im * m.ω_p * t * m.dt)
        for (l, key) in enumerate(layer_keys)
            haskey(ps[key], :weight) || continue
            z_in   = (l == 1) ? z0 : states[l-1]
            z_self = states[l]
            h_complex = z_self * adjoint(z_in)
            H_acc[key] .+= (h_complex .- ComplexF32.(h_dc[key].weight)) .* demod
        end
    end

    # 6. Convert to gradient: dL/dW = -2·Re(H) / (T_lockin · ε).
    #    The factor of 2 comes from the real cosine probe — see the
    #    design doc, section "Implementation sketch".
    grads = _ep_lockin_gradient(ps, H_acc, T_lockin, m.ε)
    return grads, s_free
end

function _ep_lockin_gradient(ps, H_acc::Dict{Symbol,Matrix{ComplexF32}},
                              T_lockin::Int, ε)
    norm_factor = Float32(T_lockin) * Float32(ε)
    pairs = Pair{Symbol,Any}[]
    for key in keys(ps)
        if haskey(ps[key], :weight)
            gW = -2f0 .* real.(H_acc[key]) ./ norm_factor
            entry = (weight = gW,)
            if haskey(ps[key], :log_neg_lambda)
                entry = merge(entry, (log_neg_lambda = zeros(Float32, size(ps[key].log_neg_lambda)),))
            end
            if haskey(ps[key], :omega)
                entry = merge(entry, (omega = zeros(Float32, size(ps[key].omega)),))
            end
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
    ep_train(model, ps, st, train_loader, args; method=StaticEP())

Train a `Lux.Chain` of EP-compatible layers via equilibrium
propagation. Returns `(losses, ps, st)` — same shape as `train` and
`hep_train` so it's a drop-in for existing users.

`train_loader` is any iterable of `(x, y)` batches where `x` is a
phase-typed (or real-valued, interpreted as phase) array and `y` is
a complex unit-modulus target matching the chain's last layer's
output dimension.

`args` is the global `Args` struct (see `test/runtests.jl`); only
`lr` and `epochs` are read.
"""
function ep_train(model::Lux.Chain, ps, st, train_loader, args;
                  method::AbstractEPMethod = StaticEP(),
                  verbose::Bool = false)
    opt_state = Optimisers.setup(Optimisers.Descent(Float32(args.lr)), ps)
    losses = Float32[]
    for epoch in 1:args.epochs
        for (x, y) in train_loader
            grads, s_free = ep_gradient(method, model, ps, st, x, y)
            opt_state, ps = Optimisers.update(opt_state, ps, grads)

            cost = SimilarityCost(ComplexF32.(y))
            push!(losses, ep_loss(cost, s_free[end]))

            if verbose
                println("epoch=$epoch loss=$(losses[end])")
            end
        end
    end
    return losses, ps, st
end
