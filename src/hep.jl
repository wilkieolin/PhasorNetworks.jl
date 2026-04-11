# ================================================================
# Holomorphic Equilibrium Propagation (hEP)
# ================================================================
#
# Implements holomorphic equilibrium propagation (Laborieux & Zenke,
# NeurIPS 2022) for PhasorNetworks via coupled phasor recurrences
# with demodulated inter-layer coupling.
#
# Each neuron's state z_l(t) oscillates at frequency omega_l. The
# information is in the relative phase, obtained by demodulating
# against a reference oscillation:
#
#     phi_l[n] = z_l[n] .* conj(ref_l[n])
#     ref_l[n] = exp(i * omega_l * n * dt)
#
# The phasor kernel handles carrier dynamics (oscillation + decay),
# while inter-layer coupling operates on the demodulated states.
# The demodulation reference is a fixed function of time, so
# conj(ref) is a constant w.r.t. holomorphic derivatives in z.
#
# See docs/phasor_hep_derivation.tex for the formal derivation.

# ================================================================
# 1. Holomorphic Activation Functions
# ================================================================

"""
    holotanh(z; a=1.0f0)

Complex-valued hyperbolic tangent — holomorphic (complex-differentiable)
everywhere except at isolated poles on the imaginary axis.

For real inputs, reduces to standard tanh.
"""
function holotanh(z; a::Float32 = 1.0f0)
    return tanh.(a .* z)
end

function holotanh(a::Float32)
    return z -> holotanh(z; a=a)
end

"""
    holotanh_deriv(z; a=1.0f0)

Derivative of holotanh: σ'(z) = a * (1 - tanh(a*z)^2).
"""
function holotanh_deriv(z; a::Float32 = 1.0f0)
    t = tanh.(a .* z)
    return a .* (1 .- t .* t)
end

# ================================================================
# 2. Holomorphic Cost Functions
# ================================================================

"""
    hep_cost_xent(z_output, y)

Complex cross-entropy cost, holomorphic in z_output.
"""
function hep_cost_xent(z_output, y)
    z_shift = z_output .- maximum(real.(z_output), dims=1)
    log_probs = z_shift .- log.(sum(exp.(z_shift), dims=1))
    batch = size(z_output, 2)
    return -sum(y .* log_probs) / batch
end

"""
    hep_cost_xent_grad(z_output, y)

Gradient of complex cross-entropy: softmax(z) - y, normalized by batch.
"""
function hep_cost_xent_grad(z_output, y)
    z_shift = z_output .- maximum(real.(z_output), dims=1)
    probs = exp.(z_shift) ./ sum(exp.(z_shift), dims=1)
    batch = size(z_output, 2)
    return (probs .- y) ./ batch
end

# ================================================================
# 3. Holomorphic Readout Layer
# ================================================================

"""
    HolomorphicReadout <: LuxCore.AbstractLuxLayer

Interference-based readout layer for holomorphic EP.

Computes class logits via complex dot product of network output
with conjugated codebook prototypes:

    logit_c = (1/d) * sum(z .* conj(code_c))

This is holomorphic in z (the conjugated codes are fixed constants).
When z is on the unit circle, the real part of each logit equals
the standard codebook cosine similarity: mean(cos(π(θ_z - θ_code))).

Physically: measures interference between the output oscillator state
and a reference oscillator at the class prototype phase.
"""
struct HolomorphicReadout <: LuxCore.AbstractLuxLayer
    in_dims::Int
    n_classes::Int
end

function HolomorphicReadout(dims::Pair{<:Int, <:Int})
    return HolomorphicReadout(dims.first, dims.second)
end

function Base.show(io::IO, hr::HolomorphicReadout)
    print(io, "HolomorphicReadout($(hr.in_dims) => $(hr.n_classes))")
end

function Lux.initialparameters(::AbstractRNG, ::HolomorphicReadout)
    return NamedTuple()
end

function Lux.initialstates(rng::AbstractRNG, hr::HolomorphicReadout)
    codes = random_symbols(rng, (hr.in_dims, hr.n_classes))
    codes_conj = conj.(angle_to_complex(Float32.(codes)))
    return (codes_conj = codes_conj,)
end

function (hr::HolomorphicReadout)(z::AbstractArray{<:Complex}, params::LuxParams, state::NamedTuple)
    logits = transpose(state.codes_conj) * z ./ Float32(hr.in_dims)
    return logits, state
end

function (hr::HolomorphicReadout)(z::AbstractArray{<:Phase}, params::LuxParams, state::NamedTuple)
    zc = angle_to_complex(Float32.(z))
    logits, st = hr(zc, params, state)
    return real.(logits), st
end

function (hr::HolomorphicReadout)(z::AbstractArray{<:Real}, params::LuxParams, state::NamedTuple)
    zc = angle_to_complex(z)
    logits, st = hr(zc, params, state)
    return real.(logits), st
end

"""
    hep_interference_cost(logits, y)

Cross-entropy cost on interference logits. Holomorphic in logits.
"""
function hep_interference_cost(logits, y)
    z_shift = logits .- maximum(real.(logits), dims=1)
    log_probs = z_shift .- log.(sum(exp.(z_shift), dims=1))
    batch = size(logits, 2)
    return -sum(y .* log_probs) / batch
end

"""
    hep_interference_cost_grad(logits, y)

Gradient of interference cost w.r.t. logits: softmax(z) - y.
"""
function hep_interference_cost_grad(logits, y)
    z_shift = logits .- maximum(real.(logits), dims=1)
    probs = exp.(z_shift) ./ sum(exp.(z_shift), dims=1)
    batch = size(logits, 2)
    return (probs .- y) ./ batch
end

# ================================================================
# 4. Phasor Kernel Matrices
# ================================================================

"""
    _phasor_AB(log_neg_lambda, omega, dt)

Compute phasor kernel matrices A = exp(k*dt), B = (A-1)/k
where k = -exp(log_neg_lambda) + i*omega.
"""
function _phasor_AB(log_neg_lambda, omega, dt::Float32)
    lambda = -exp.(log_neg_lambda)
    k = ComplexF32.(lambda .+ im .* omega)
    A = exp.(k .* dt)
    B = (A .- 1) ./ k
    return A, B
end

# ================================================================
# 5. Demodulated Coupled Phasor Recurrence
# ================================================================

"""
    _demodulate(z, omega, n, dt)

Demodulate state z by removing the carrier oscillation at step n:
    phi = z .* conj(exp(i * omega * n * dt))

The demodulated state phi encodes the relative phase — the
information-carrying variable — while the carrier exp(i*omega*t)
is factored out. Since conj(ref) depends only on omega and n
(not on z), this operation is holomorphic in z.
"""
function _demodulate(z, omega, n::Int, dt::Float32)
    ref_conj = exp.(ComplexF32.(-im .* omega .* (n * dt)))
    return z .* ref_conj
end

"""
    hep_equilibrium(weights, biases, k_arrays, x, beta, y;
                    T, dt, readout_conj, init)

Find equilibrium via the demodulated coupled phasor recurrence.

At each step n, for each layer l:
1. Demodulate states: phi_l = z_l .* conj(ref_l)
2. Compute inter-layer coupling on demodulated states:
   - Feedforward:  W_l * phi_{l-1}
   - Feedback:     W_{l+1}^T * phi_{l+1}
   - Teaching:     -β * ∂C/∂z_L  (output layer only)
3. Advance oscillator: z_l[n+1] = A_l .* z_l[n] + B_l .* I_l[n]

The oscillator kernel (A, B) handles carrier dynamics (rotation +
decay). Inter-layer coupling operates on relative phases (phi),
avoiding the DC gain attenuation that occurs when coupling raw
oscillating states.

# Arguments
- `weights`: Tuple of weight matrices
- `biases`: Tuple of bias vectors (or nothing)
- `k_arrays`: Tuple of (log_neg_lambda, omega) pairs per layer
- `x`: Input data (encoded as complex, e.g. exp(i*pi*theta))
- `beta`: Nudge parameter (real or complex)
- `y`: Target labels (one-hot)
- `T::Int=100`: Recurrence steps
- `dt::Float32=0.1f0`: Time step (should be < period for meaningful oscillation)
- `readout_conj`: Conjugated codebook for interference readout (or nothing)
- `init`: Optional initial states tuple (raw z, not demodulated)
"""
function hep_equilibrium(weights, biases, k_arrays, x, beta, y;
                         T::Int = 100,
                         dt::Float32 = 0.1f0,
                         readout_conj = nothing,
                         init = nothing)
    n_layers = length(weights)
    ET = _state_eltype(beta, x)

    # Extract omega per layer and compute A, B
    omegas = [ComplexF32.(ka[2]) for ka in k_arrays]
    AB = [_phasor_AB(ka[1], ka[2], dt) for ka in k_arrays]

    # Initialize states
    if init !== nothing
        states = [ET.(copy(s)) for s in init]
    else
        states = _forward_init_demod(weights, biases, x, ET)
    end

    # Coupled recurrence with demodulation
    for n in 1:T
        new_states = Vector{Any}(undef, n_layers)

        # Demodulate all states at current step
        phis = [_demodulate(states[l], omegas[l], n-1, dt) for l in 1:n_layers]
        # Demodulate input (input has no carrier, so phi_0 = x)
        phi_inputs = (x, phis[1:end-1]...)

        for l in 1:n_layers
            A_l, B_l = AB[l]

            # --- Inter-layer coupling on demodulated states ---

            # Feedforward: W_l * phi_{l-1}
            I_l = weights[l] * phi_inputs[l]
            if biases[l] !== nothing
                I_l = I_l .+ biases[l]
            end

            # Feedback: W_{l+1}^T * phi_{l+1}
            if l < n_layers
                I_l = I_l .+ transpose(weights[l+1]) * phis[l+1]
            end

            # Teaching signal on output layer
            if l == n_layers && beta != 0
                if readout_conj !== nothing
                    d = size(readout_conj, 1)
                    logits = transpose(readout_conj) * phis[l] ./ Float32(d)
                    dC_dlogits = hep_interference_cost_grad(logits, y)
                    dC_dphi = readout_conj * dC_dlogits ./ Float32(d)
                else
                    dC_dphi = hep_cost_xent_grad(phis[l], y)
                end
                I_l = I_l .- beta .* dC_dphi
            end

            # Phasor recurrence: z[n+1] = A .* z[n] + B .* I
            new_states[l] = A_l .* states[l] .+ B_l .* I_l
        end

        states = new_states
    end

    return Tuple(states)
end

"""
Determine element type for states.
"""
function _state_eltype(beta, x)
    if beta isa Complex || eltype(x) <: Complex
        return ComplexF32
    else
        return Float32
    end
end

"""
Initialize states from a single feedforward pass (no activation —
the oscillator dynamics provide the nonlinearity via demodulation).
"""
function _forward_init_demod(weights, biases, x, ET)
    n_layers = length(weights)
    states = Vector{Any}(undef, n_layers)
    h = ET.(x)
    for l in 1:n_layers
        h = weights[l] * h
        if biases[l] !== nothing
            h = h .+ biases[l]
        end
        states[l] = h
    end
    return states
end

# ================================================================
# 6. Energy Function
# ================================================================

"""
    hep_energy(states, weights, biases, omegas, x, y, beta, n, dt)

Compute the interference-based energy at step n:

    Φ = Σ_l <phi_l, W_l * phi_{l-1}> - β * C(phi_L, y)

where phi_l = demodulate(z_l) is the relative-phase state.
Uses the complex bilinear form (no conjugation on states).
"""
function hep_energy(states, weights, biases, omegas, x, y, beta;
                    n::Int = 0, dt::Float32 = 0.1f0,
                    readout_conj = nothing)
    phi = zero(ComplexF32)

    # Demodulate states
    phis = [_demodulate(states[l], omegas[l], n, dt) for l in 1:length(weights)]
    phi_inputs = (x, phis[1:end-1]...)

    for l in 1:length(weights)
        pre = weights[l] * phi_inputs[l]
        if biases[l] !== nothing
            pre = pre .+ biases[l]
        end
        phi = phi + sum(pre .* phis[l])
    end

    if readout_conj !== nothing
        d = size(readout_conj, 1)
        logits = transpose(readout_conj) * phis[end] ./ Float32(d)
        cost = hep_interference_cost(logits, y)
    else
        cost = hep_cost_xent(phis[end], y)
    end
    phi = phi - beta * cost

    return phi
end

# ================================================================
# 7. Energy Gradient w.r.t. Parameters
# ================================================================

"""
    _energy_param_gradients(states, weights, biases, omegas, x, n, dt)

Compute ∂Φ/∂θ at equilibrium using demodulated states.

    ∂Φ/∂W_l = phi_l * phi_{l-1}^T

(Hebbian outer product on demodulated states.)
"""
function _energy_param_gradients(states, weights, biases, omegas, x;
                                 n::Int = 0, dt::Float32 = 0.1f0)
    n_layers = length(weights)

    phis = [_demodulate(states[l], omegas[l], n, dt) for l in 1:n_layers]
    phi_inputs = (x, phis[1:end-1]...)

    weight_grads = []
    bias_grads = []

    for l in 1:n_layers
        # ∂Φ/∂W_l = phi_l * phi_{l-1}^T
        w_grad = phis[l] * transpose(phi_inputs[l])
        push!(weight_grads, w_grad)

        if biases[l] !== nothing
            b_grad = mean(phis[l], dims=ndims(phis[l]))
            if ndims(phis[l]) > 1
                b_grad = dropdims(b_grad, dims=ndims(phis[l]))
            end
            push!(bias_grads, b_grad)
        else
            push!(bias_grads, nothing)
        end
    end

    return weight_grads, bias_grads
end

# ================================================================
# 8. Contour Integration
# ================================================================

"""
    hep_gradient(weights, biases, k_arrays, x, y; N, r, T_free, T_nudge, dt, ...)

Compute parameter gradients via holomorphic EP contour integration.
"""
function hep_gradient(weights, biases, k_arrays, x, y;
                      N::Int = 4,
                      r::Float32 = 0.5f0,
                      T_free::Int = 100,
                      T_nudge::Int = 30,
                      dt::Float32 = 0.1f0,
                      readout_conj = nothing)
    n_layers = length(weights)
    omegas = [ComplexF32.(ka[2]) for ka in k_arrays]

    # 1. Free phase
    states_free = hep_equilibrium(weights, biases, k_arrays, x, 0.0f0, y;
                                  T=T_free, dt=dt, readout_conj=readout_conj)

    # 2. Contour integration
    w_grads_accum = [zeros(ComplexF32, size(w)) for w in weights]
    b_grads_accum = [b === nothing ? nothing : zeros(ComplexF32, size(b)) for b in biases]

    for n_contour in 0:N-1
        angle_n = 2.0f0 * Float32(pi) * n_contour / N
        beta_n = r * exp(1.0f0im * angle_n)

        states_n = hep_equilibrium(weights, biases, k_arrays, x, beta_n, y;
                                   T=T_nudge, dt=dt, init=states_free,
                                   readout_conj=readout_conj)

        # Parameter gradients at this contour point (using final step index)
        wg, bg = _energy_param_gradients(states_n, weights, biases, omegas, x;
                                         n=T_free+T_nudge, dt=dt)

        phase_weight = exp(-1.0f0im * angle_n)
        for l in 1:n_layers
            w_grads_accum[l] .+= ComplexF32.(wg[l]) .* phase_weight
            if b_grads_accum[l] !== nothing && bg[l] !== nothing
                b_grads_accum[l] .+= ComplexF32.(bg[l]) .* phase_weight
            end
        end
    end

    w_grads = [real.(wg) ./ N for wg in w_grads_accum]
    b_grads = [bg === nothing ? nothing : real.(bg) ./ N for bg in b_grads_accum]

    return w_grads, b_grads
end

# ================================================================
# 9. Lux-Compatible Wrappers
# ================================================================

"""
    extract_hep_params(ps::NamedTuple, st::NamedTuple)

Extract weights, biases, dynamics parameters, and readout codes.
"""
function extract_hep_params(ps::NamedTuple, st::NamedTuple)
    weights = []
    biases = []
    k_arrays = []
    layer_keys = Symbol[]
    readout_conj = nothing

    for key in keys(ps)
        p = ps[key]
        s = st[key]
        if haskey(p, :weight)
            push!(weights, p.weight)
            if haskey(p, :bias_real) && haskey(p, :bias_imag)
                push!(biases, p.bias_real .+ 1.0f0im .* p.bias_imag)
            else
                push!(biases, nothing)
            end
            lnl = p.log_neg_lambda
            omega = haskey(p, :omega) ? p.omega : s.omega
            push!(k_arrays, (lnl, omega))
            push!(layer_keys, key)
        elseif haskey(s, :codes_conj)
            readout_conj = s.codes_conj
        end
    end

    return Tuple(weights), Tuple(biases), Tuple(k_arrays), layer_keys, readout_conj
end

function extract_hep_params(ps::NamedTuple)
    weights = []
    biases = []
    k_arrays = []
    layer_keys = Symbol[]

    for key in keys(ps)
        p = ps[key]
        if haskey(p, :weight)
            push!(weights, p.weight)
            if haskey(p, :bias_real) && haskey(p, :bias_imag)
                push!(biases, p.bias_real .+ 1.0f0im .* p.bias_imag)
            else
                push!(biases, nothing)
            end
            lnl = p.log_neg_lambda
            omega = haskey(p, :omega) ? p.omega : fill(Float32(2pi), size(lnl))
            push!(k_arrays, (lnl, omega))
            push!(layer_keys, key)
        end
    end

    return Tuple(weights), Tuple(biases), Tuple(k_arrays), layer_keys, nothing
end

"""
    pack_hep_gradients(w_grads, b_grads, layer_keys, ps)

Pack hEP gradients into Lux-compatible NamedTuple.
"""
function pack_hep_gradients(w_grads, b_grads, layer_keys, ps)
    grad_dict = Dict{Symbol, Any}()

    layer_idx = 1
    for key in keys(ps)
        p = ps[key]
        if key in layer_keys
            g = Dict{Symbol, Any}()
            g[:weight] = Float32.(real.(w_grads[layer_idx]))

            if haskey(p, :log_neg_lambda)
                g[:log_neg_lambda] = zeros(Float32, size(p.log_neg_lambda))
            end
            if haskey(p, :omega)
                g[:omega] = zeros(Float32, size(p.omega))
            end

            if b_grads[layer_idx] !== nothing
                b_complex = b_grads[layer_idx]
                g[:bias_real] = Float32.(real.(b_complex))
                g[:bias_imag] = Float32.(imag.(b_complex))
            end

            grad_dict[key] = NamedTuple{Tuple(Symbol.(keys(g)))}(values(g))
            layer_idx += 1
        else
            grad_dict[key] = _zero_grad(p)
        end
    end

    return NamedTuple{Tuple(keys(ps))}([grad_dict[k] for k in keys(ps)])
end

function _zero_grad(p)
    if p isa NamedTuple
        if length(p) == 0
            return NamedTuple()
        end
        vals = [v isa AbstractArray ? zeros(eltype(v), size(v)) : nothing for v in values(p)]
        valid = [(k, v) for (k, v) in zip(keys(p), vals) if v !== nothing]
        if isempty(valid)
            return NamedTuple()
        end
        ks = Tuple(first.(valid))
        vs = last.(valid)
        return NamedTuple{ks}(Tuple(vs))
    else
        return p isa AbstractArray ? zeros(eltype(p), size(p)) : nothing
    end
end

# ================================================================
# 10. High-Level Training Interface
# ================================================================

"""
    hep_train(model, ps, st, train_loader, args; kwargs...)

Train a phasor network using holomorphic equilibrium propagation.
"""
function hep_train(model, ps, st, train_loader, args;
                   N::Int = 4,
                   r::Float32 = 0.5f0,
                   T_free::Int = 100,
                   T_nudge::Int = 30,
                   dt::Float32 = 0.1f0,
                   verbose::Bool = false)
    opt_state = Optimisers.setup(Optimisers.Adam(args.lr), ps)
    losses = Float32[]

    for epoch in 1:args.epochs
        for (x, y) in train_loader
            weights, biases, k_arrays, layer_keys, readout_conj = extract_hep_params(ps, st)

            w_grads, b_grads = hep_gradient(weights, biases, k_arrays, x, y;
                                            N=N, r=r, dt=dt,
                                            T_free=T_free, T_nudge=T_nudge,
                                            readout_conj=readout_conj !== nothing ? ComplexF32.(readout_conj) : nothing)

            grad_nt = pack_hep_gradients(w_grads, b_grads, layer_keys, ps)
            opt_state, ps = Optimisers.update(opt_state, ps, grad_nt)

            # Track loss via model forward pass
            y_pred, _ = model(x, ps, st)
            lossval = if y_pred isa AbstractArray{<:Complex}
                real(hep_interference_cost(y_pred, y))
            else
                mean(evaluate_loss(y_pred, y, :quadrature))
            end
            push!(losses, Float32(lossval))

            if verbose
                println("Loss: ", lossval)
            end
        end
    end

    return losses, ps, st
end
