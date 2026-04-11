# ================================================================
# Holomorphic Equilibrium Propagation (hEP)
# ================================================================
#
# Implements holomorphic equilibrium propagation for PhasorNetworks
# via coupled phasor recurrences with a consistent energy function.
#
# The energy function explicitly includes the oscillator self-energy:
#
#   Phi = sum_l [(1/2)<z_l, K_l*z_l> + <sigma(W_l*z_{l-1}+b_l), z_l>]
#         - beta * C(z_L, y)
#
# The gradient dPhi/dz_l = K_l*z_l + sigma(W_l*z_{l-1}+b_l) + feedback
# is exactly the phasor ODE: dz/dt = K*z + drive. This ensures
# the EP gradient theorem holds: the Hebbian parameter gradient at
# equilibrium equals the loss gradient.
#
# Demodulation (removing the carrier oscillation) is applied only
# at the readout layer for the cost function. Inter-layer coupling
# operates on raw oscillating states.
#
# See docs/phasor_hep_derivation.tex for the formal derivation.

# ================================================================
# 1. Holomorphic Activation Functions
# ================================================================

"""
    holotanh(z; a=1.0f0)

Complex-valued hyperbolic tangent — holomorphic everywhere except
at isolated poles on the imaginary axis at z = i*pi*(n+1/2)/a.
"""
function holotanh(z; a::Float32 = 1.0f0)
    return tanh.(a .* z)
end

function holotanh(a::Float32)
    return z -> holotanh(z; a=a)
end

"""
    holotanh_deriv(z; a=1.0f0)

Derivative of holotanh: sigma'(z) = a * (1 - tanh(a*z)^2).
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

Gradient of complex cross-entropy: (softmax(z) - y) / batch.
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

Interference-based readout: logit_c = (1/d) * sum(z .* conj(code_c)).
Holomorphic in z (conjugated codes are fixed constants).
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

Cross-entropy on complex interference logits.
"""
function hep_interference_cost(logits, y)
    z_shift = logits .- maximum(real.(logits), dims=1)
    log_probs = z_shift .- log.(sum(exp.(z_shift), dims=1))
    batch = size(logits, 2)
    return -sum(y .* log_probs) / batch
end

"""
    hep_interference_cost_grad(logits, y)

Gradient of interference cost: (softmax(z) - y) / batch.
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

Compute A = exp(k*dt), B = (A-1)/k where k = -exp(log_neg_lambda) + i*omega.
Returns (A, B) as per-channel vectors.
"""
function _phasor_AB(log_neg_lambda, omega, dt::Float32)
    lambda = -exp.(log_neg_lambda)
    k = ComplexF32.(lambda .+ im .* omega)
    A = exp.(k .* dt)
    B = (A .- 1) ./ k
    return A, B
end

"""
    _demodulate(z, omega, n, dt)

Remove carrier oscillation: phi = z .* exp(-i*omega*n*dt).
Holomorphic in z (the reference is a fixed constant).
"""
function _demodulate(z, omega, n::Int, dt::Float32)
    ref_conj = exp.(ComplexF32.(-im .* omega .* (n * dt)))
    return z .* ref_conj
end

# ================================================================
# 5. Coupled Phasor Recurrence with Consistent Energy
# ================================================================

"""
    hep_equilibrium(weights, biases, k_arrays, x, beta, y; ...)

Find equilibrium via the coupled phasor recurrence derived from
the energy function:

    Phi = sum_l [(1/2)<z_l, K_l*z_l> + <sigma(W_l*z_{l-1}+b_l), z_l>]
          - beta * C(z_L, y)

The gradient dPhi/dz_l gives the phasor ODE:

    dz_l/dt = K_l*z_l + sigma(W_l*z_{l-1}+b_l) + feedback - beta*dC/dz_l

Discretized as: z_l[n+1] = A_l .* z_l[n] + B_l .* I_l[n]

Inter-layer coupling operates on RAW states (not demodulated).
Demodulation is applied only at the readout for the cost function.

Weight matrices should be initialized with scale ~ |k| to prevent
compound signal attenuation through layers.
"""
function hep_equilibrium(weights, biases, k_arrays, x, beta, y;
                         T::Int = 100,
                         dt::Float32 = 0.1f0,
                         activation = holotanh,
                         readout_conj = nothing,
                         init = nothing)
    n_layers = length(weights)

    # Compute A, B per layer
    AB = [_phasor_AB(ka[1], ka[2], dt) for ka in k_arrays]
    omegas = [ComplexF32.(ka[2]) for ka in k_arrays]

    # Initialize states
    if init !== nothing
        states = [ComplexF32.(copy(s)) for s in init]
    else
        states = _forward_init(weights, biases, x, activation)
    end

    # Input is fixed (z_0 = x)
    x_c = ComplexF32.(x)

    # Coupled recurrence: dz/dt = Kz + sigma(W*z_prev+b) + feedback
    for n in 1:T
        new_states = Vector{Any}(undef, n_layers)
        inputs = (x_c, states[1:end-1]...)

        for l in 1:n_layers
            A_l, B_l = AB[l]

            # Feedforward: sigma(W_l * z_{l-1} + b_l)
            pre_l = weights[l] * inputs[l]
            if biases[l] !== nothing
                pre_l = pre_l .+ biases[l]
            end
            I_l = activation(pre_l)

            # Feedback: W_{l+1}^T * [sigma'(W_{l+1}*z_l + b_{l+1}) .* z_{l+1}]
            if l < n_layers
                pre_above = weights[l+1] * states[l]
                if biases[l+1] !== nothing
                    pre_above = pre_above .+ biases[l+1]
                end
                I_l = I_l .+ transpose(weights[l+1]) * (holotanh_deriv(pre_above) .* states[l+1])
            end

            # Teaching signal (output layer only)
            # Cost is computed on DEMODULATED states at the readout
            if l == n_layers && beta != 0
                if readout_conj !== nothing
                    # Demodulate output for cost computation
                    phi_L = _demodulate(states[l], omegas[l], n-1, dt)
                    d = size(readout_conj, 1)
                    logits = transpose(readout_conj) * phi_L ./ Float32(d)
                    dC_dlogits = hep_interference_cost_grad(logits, y)
                    # Chain rule: dC/dz_L = dC/dphi_L * dphi_L/dz_L
                    # dphi_L/dz_L = conj(ref_L) (diagonal, fixed)
                    ref_conj = exp.(ComplexF32.(-im .* omegas[l] .* ((n-1) * dt)))
                    dC_dz = ref_conj .* (readout_conj * dC_dlogits ./ Float32(d))
                else
                    dC_dz = hep_cost_xent_grad(states[l], y)
                end
                I_l = I_l .- beta .* dC_dz
            end

            # Phasor recurrence step
            new_states[l] = A_l .* states[l] .+ B_l .* I_l
        end

        states = new_states
    end

    return Tuple(states)
end

"""
Initialize states from a single forward pass with activation.
"""
function _forward_init(weights, biases, x, activation)
    n_layers = length(weights)
    states = Vector{Any}(undef, n_layers)
    h = ComplexF32.(x)
    for l in 1:n_layers
        pre = weights[l] * h
        if biases[l] !== nothing
            pre = pre .+ biases[l]
        end
        h = activation(pre)
        states[l] = h
    end
    return states
end

function _state_eltype(beta, x)
    if beta isa Complex || eltype(x) <: Complex
        return ComplexF32
    else
        return Float32
    end
end

# ================================================================
# 6. Energy Function
# ================================================================

"""
    hep_energy(states, weights, biases, k_arrays, x, y, beta; ...)

Compute the phasor EP energy:

    Phi = sum_l [(1/2)<z_l, K_l*z_l> + <sigma(W_l*z_{l-1}+b_l), z_l>]
          - beta * C(z_L, y)

The self-energy (1/2)<z, Kz> is the oscillator's intrinsic energy.
The coupling <sigma(W*z_prev), z> measures inter-layer coherence.
Uses the complex bilinear form (no conjugation).
"""
function hep_energy(states, weights, biases, k_arrays, x, y, beta;
                    dt::Float32 = 0.1f0, n::Int = 0,
                    activation = holotanh,
                    readout_conj = nothing)
    phi = zero(ComplexF32)
    inputs = (ComplexF32.(x), states[1:end-1]...)

    for l in 1:length(weights)
        # Self-energy: (1/2)<z_l, K_l*z_l>
        lambda = -exp.(k_arrays[l][1])
        omega = k_arrays[l][2]
        K_l = ComplexF32.(lambda .+ im .* omega)
        phi = phi + sum(K_l .* states[l] .* states[l]) / 2

        # Coupling: <sigma(W_l*z_{l-1}+b_l), z_l>
        pre = weights[l] * inputs[l]
        if biases[l] !== nothing
            pre = pre .+ biases[l]
        end
        phi = phi + sum(activation(pre) .* states[l])
    end

    # Cost (on demodulated output for readout)
    omegas_L = ComplexF32.(k_arrays[end][2])
    if readout_conj !== nothing
        phi_L = _demodulate(states[end], omegas_L, n, dt)
        d = size(readout_conj, 1)
        logits = transpose(readout_conj) * phi_L ./ Float32(d)
        cost = hep_interference_cost(logits, y)
    else
        cost = hep_cost_xent(states[end], y)
    end
    phi = phi - beta * cost

    return phi
end

# ================================================================
# 7. Energy Gradient w.r.t. Parameters
# ================================================================

"""
    _energy_param_gradients(states, weights, biases, x; activation)

Compute dPhi/dW_l = [sigma'(W_l*z_{l-1}+b_l) .* z_l] * z_{l-1}^T

This is the derivative of <sigma(W_l*z_{l-1}+b_l), z_l> w.r.t. W_l.
Uses transpose (not adjoint) to preserve holomorphicity.
"""
function _energy_param_gradients(states, weights, biases, x;
                                 activation = holotanh)
    n_layers = length(weights)
    inputs = (ComplexF32.(x), states[1:end-1]...)

    weight_grads = []
    bias_grads = []

    for l in 1:n_layers
        pre = weights[l] * inputs[l]
        if biases[l] !== nothing
            pre = pre .+ biases[l]
        end

        # dPhi/dW_l = [sigma'(pre_l) .* z_l] * z_{l-1}^T
        sigma_prime = holotanh_deriv(pre)
        modulated = sigma_prime .* states[l]
        w_grad = modulated * transpose(inputs[l])
        push!(weight_grads, w_grad)

        if biases[l] !== nothing
            b_grad = mean(modulated, dims=ndims(modulated))
            if ndims(modulated) > 1
                b_grad = dropdims(b_grad, dims=ndims(modulated))
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
    hep_gradient(weights, biases, k_arrays, x, y; ...)

Compute parameter gradients via holomorphic EP contour integration.
"""
function hep_gradient(weights, biases, k_arrays, x, y;
                      N::Int = 4,
                      r::Float32 = 0.5f0,
                      T_free::Int = 100,
                      T_nudge::Int = 30,
                      dt::Float32 = 0.1f0,
                      activation = holotanh,
                      readout_conj = nothing)
    n_layers = length(weights)

    # Free phase
    states_free = hep_equilibrium(weights, biases, k_arrays, x, 0.0f0, y;
                                  T=T_free, dt=dt, activation=activation,
                                  readout_conj=readout_conj)

    # Contour integration
    w_grads_accum = [zeros(ComplexF32, size(w)) for w in weights]
    b_grads_accum = [b === nothing ? nothing : zeros(ComplexF32, size(b)) for b in biases]

    for n_c in 0:N-1
        angle_n = 2.0f0 * Float32(pi) * n_c / N
        beta_n = r * exp(1.0f0im * angle_n)

        states_n = hep_equilibrium(weights, biases, k_arrays, x, beta_n, y;
                                   T=T_nudge, dt=dt, init=states_free,
                                   activation=activation, readout_conj=readout_conj)

        wg, bg = _energy_param_gradients(states_n, weights, biases, x;
                                         activation=activation)

        phase_weight = exp(-1.0f0im * angle_n)
        for l in 1:n_layers
            w_grads_accum[l] .+= ComplexF32.(wg[l]) .* phase_weight
            if b_grads_accum[l] !== nothing && bg[l] !== nothing
                b_grads_accum[l] .+= ComplexF32.(bg[l]) .* phase_weight
            end
        end
    end

    # Negate: the EP energy gradient dPhi/dW at equilibrium gives the
    # NEGATIVE of the loss gradient (because Phi = ... - beta*C, so
    # the cost enters with a minus sign). Negate to get dL/dW for descent.
    w_grads = [-real.(wg) ./ N for wg in w_grads_accum]
    b_grads = [bg === nothing ? nothing : -real.(bg) ./ N for bg in b_grads_accum]

    return w_grads, b_grads
end

# ================================================================
# 9. Lux-Compatible Wrappers
# ================================================================

"""
    extract_hep_params(ps, st)

Extract weights, biases, dynamics, and readout from Lux params/state.
"""
function extract_hep_params(ps::NamedTuple, st::NamedTuple)
    weights = []; biases = []; k_arrays = []; layer_keys = Symbol[]
    readout_conj = nothing

    for key in keys(ps)
        p = ps[key]; s = st[key]
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
    weights = []; biases = []; k_arrays = []; layer_keys = Symbol[]
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
                g[:bias_real] = Float32.(real.(b_grads[layer_idx]))
                g[:bias_imag] = Float32.(imag.(b_grads[layer_idx]))
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
        length(p) == 0 && return NamedTuple()
        vals = [v isa AbstractArray ? zeros(eltype(v), size(v)) : nothing for v in values(p)]
        valid = [(k, v) for (k, v) in zip(keys(p), vals) if v !== nothing]
        isempty(valid) && return NamedTuple()
        return NamedTuple{Tuple(first.(valid))}(Tuple(last.(valid)))
    else
        return p isa AbstractArray ? zeros(eltype(p), size(p)) : nothing
    end
end

# ================================================================
# 10. Training Interface
# ================================================================

"""
    hep_train(model, ps, st, train_loader, args; ...)

Train using holomorphic EP. Loss is evaluated from the hEP
free-phase equilibrium (aligned with training dynamics).
"""
function hep_train(model, ps, st, train_loader, args;
                   N::Int = 4,
                   r::Float32 = 0.5f0,
                   T_free::Int = 100,
                   T_nudge::Int = 30,
                   dt::Float32 = 0.1f0,
                   activation = holotanh,
                   verbose::Bool = false)
    opt_state = Optimisers.setup(Optimisers.Descent(args.lr), ps)
    losses = Float32[]

    for epoch in 1:args.epochs
        for (x, y) in train_loader
            weights, biases, k_arrays, layer_keys, readout_conj = extract_hep_params(ps, st)
            rc = readout_conj !== nothing ? ComplexF32.(readout_conj) : nothing

            w_grads, b_grads = hep_gradient(weights, biases, k_arrays, x, y;
                                            N=N, r=r, dt=dt,
                                            T_free=T_free, T_nudge=T_nudge,
                                            activation=activation, readout_conj=rc)

            grad_nt = pack_hep_gradients(w_grads, b_grads, layer_keys, ps)
            opt_state, ps = Optimisers.update(opt_state, ps, grad_nt)

            # Aligned loss from hEP equilibrium
            w2, b2, k2, _, rc2 = extract_hep_params(ps, st)
            rc_e = rc2 !== nothing ? ComplexF32.(rc2) : nothing
            omegas = [ComplexF32.(ka[2]) for ka in k2]
            st_eval = hep_equilibrium(w2, b2, k2, ComplexF32.(x), 0.0f0, y;
                                      T=T_free, dt=dt, activation=activation,
                                      readout_conj=rc_e)

            if rc_e !== nothing
                phi_L = _demodulate(st_eval[end], omegas[end], T_free, dt)
                d = size(rc_e, 1)
                logits = transpose(rc_e) * phi_L ./ Float32(d)
                lossval = real(hep_interference_cost(logits, y))
            else
                lossval = real(hep_cost_xent(st_eval[end], y))
            end
            push!(losses, Float32(lossval))

            if verbose
                println("Loss: ", lossval)
            end
        end
    end

    return losses, ps, st
end
