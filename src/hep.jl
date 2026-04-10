# ================================================================
# Holomorphic Equilibrium Propagation (hEP)
# ================================================================
#
# Implements holomorphic equilibrium propagation (Laborieux & Zenke,
# NeurIPS 2022) for PhasorNetworks via coupled phasor recurrences.
#
# The EP settling dynamics are realized by damped oscillator
# recurrences:  z_l[n+1] = A_l * z_l[n] + B_l * I_l[n]
# where A = exp(k*dt), B = (A-1)/k are the phasor kernel matrices,
# and I_l is the EP energy gradient (feedforward + feedback + nudge).
#
# The holomorphic extension (complex beta) enables exact gradient
# computation via contour integration / first Fourier coefficient.
#
# See docs/phasor_hep_derivation.tex for the full derivation.

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
Needed for the feedback term and parameter gradients in the
EP energy gradient (Eq. 7 in the derivation).
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

Treats z_output as complex logits and applies log-softmax:
    C = -sum(y .* log_softmax(z)) / batch_size

Since exp and log are holomorphic, this cost preserves the
holomorphic structure needed for exact contour gradients.
"""
function hep_cost_xent(z_output, y)
    # log-softmax: z_c - log(sum(exp(z)))
    # Shift for numerical stability (subtract max per sample)
    z_shift = z_output .- maximum(real.(z_output), dims=1)
    log_probs = z_shift .- log.(sum(exp.(z_shift), dims=1))
    batch = size(z_output, 2)
    return -sum(y .* log_probs) / batch
end

"""
    hep_cost_xent_grad(z_output, y)

Gradient of complex cross-entropy w.r.t. z_output (holomorphic derivative).

    ∂C/∂z = (softmax(z) - y) / batch_size
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

The physical interpretation: each logit measures the interference
between the output oscillator state and a reference oscillator at
the class prototype phase. Constructive interference (phase match)
yields large magnitude; destructive interference yields small.

# Fields
- `in_dims::Int`: Feature dimension (must match network output)
- `n_classes::Int`: Number of classification targets

# State
- `codes_conj`: Conjugated codebook entries, (in_dims, n_classes) ComplexF32
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

"""
    (hr::HolomorphicReadout)(z, params, state) -> (logits, state)

Compute interference logits. z can be complex or Phase.

Output shape: (n_classes, batch_size), real-valued for inference
or complex-valued during hEP dynamics.
"""
function (hr::HolomorphicReadout)(z::AbstractArray{<:Complex}, params::LuxParams, state::NamedTuple)
    # z: (in_dims, batch) complex
    # codes_conj: (in_dims, n_classes) complex
    # logit_c = (1/d) * sum_j z_j * codes_conj_j,c
    # = (1/d) * codes_conj^T * z  →  (n_classes, batch)
    logits = transpose(state.codes_conj) * z ./ Float32(hr.in_dims)
    return logits, state
end

function (hr::HolomorphicReadout)(z::AbstractArray{<:Phase}, params::LuxParams, state::NamedTuple)
    zc = angle_to_complex(Float32.(z))
    logits, st = hr(zc, params, state)
    # For Phase input (inference), return real similarities
    return real.(logits), st
end

function (hr::HolomorphicReadout)(z::AbstractArray{<:Real}, params::LuxParams, state::NamedTuple)
    # Real input: treat as phase values, convert to complex
    zc = angle_to_complex(z)
    logits, st = hr(zc, params, state)
    return real.(logits), st
end

"""
    hep_interference_cost(logits, y)

Cross-entropy cost on interference logits. Holomorphic in logits.

Takes the real part of complex logits before softmax, since
exp(a+ib) has magnitude exp(a) — the softmax is naturally driven
by the real component (which encodes phase alignment strength).
"""
function hep_interference_cost(logits, y)
    # Use real part for softmax (holomorphic: Re(z) = (z + conj(z))/2,
    # but conj(z) is constant w.r.t. holomorphic derivatives since
    # d/dz conj(z) = 0 in Wirtinger calculus)
    # Actually, Re() is not holomorphic. Use the full complex logits.
    z_shift = logits .- maximum(real.(logits), dims=1)
    log_probs = z_shift .- log.(sum(exp.(z_shift), dims=1))
    batch = size(logits, 2)
    return -sum(y .* log_probs) / batch
end

"""
    hep_interference_cost_grad(logits, y)

Gradient of interference cost w.r.t. logits (holomorphic).
Same as softmax cross-entropy gradient: softmax(z) - y.
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

Compute the phasor kernel matrices A and B from dynamics parameters.

    k = -exp(log_neg_lambda) + i*omega
    A = exp(k * dt)
    B = (A - 1) / k

Returns (A, B) as vectors (diagonal matrices stored as vectors).
"""
function _phasor_AB(log_neg_lambda, omega, dt::Float32)
    lambda = -exp.(log_neg_lambda)
    k = ComplexF32.(lambda .+ im .* omega)
    A = exp.(k .* dt)
    B = (A .- 1) ./ k
    return A, B
end

# ================================================================
# 4. Coupled Phasor Recurrence Equilibrium Solver
# ================================================================

"""
    hep_equilibrium(weights, biases, k_arrays, x, beta, y;
                    T, dt, activation, readout_conj, init)

Find equilibrium via the coupled phasor recurrence:

    z_l[n+1] = A_l .* z_l[n] + B_l .* I_l[n]

where I_l is the EP energy gradient for layer l:
- Feedforward:  σ(W_l * z_{l-1} + b_l)
- Feedback:     W_{l+1}^T * [σ'(pre_{l+1}) .* z_{l+1}]
- Teaching:     -β * ∂C/∂z_L  (output layer only)

The phasor eigenvalues k = λ + iω provide:
- Decay (λ < 0): drives system toward equilibrium
- Oscillation (ω): preserves phasor phase structure

# Arguments
- `weights`: Tuple of weight matrices (W_1, ..., W_L)
- `biases`: Tuple of bias vectors (or nothing entries)
- `k_arrays`: Tuple of (log_neg_lambda, omega) pairs per layer
- `x`: Input data (fixed, not updated)
- `beta`: Nudge parameter (real or complex)
- `y`: Target labels (one-hot)
- `T::Int=100`: Number of recurrence steps
- `dt::Float32=1.0f0`: Discretization time step
- `activation`: Holomorphic activation (default: holotanh)
- `readout_conj`: Conjugated codebook matrix (in_dims, n_classes) for
  interference-based readout. If nothing, cost is applied directly
  to output states via hep_cost_xent.
- `init`: Optional initial states tuple
"""
function hep_equilibrium(weights, biases, k_arrays, x, beta, y;
                         T::Int = 100,
                         dt::Float32 = 1.0f0,
                         activation = holotanh,
                         readout_conj = nothing,
                         init = nothing)
    n_layers = length(weights)
    ET = _state_eltype(beta, x)

    # Compute phasor kernel matrices A, B per layer
    AB = [_phasor_AB(ka[1], ka[2], dt) for ka in k_arrays]

    # Initialize states
    if init !== nothing
        states = [ET.(copy(s)) for s in init]
    else
        states = _forward_init(weights, biases, x, activation, ET)
    end

    # Coupled recurrence
    for t in 1:T
        new_states = Vector{Any}(undef, n_layers)
        inputs = (x, states[1:end-1]...)

        for l in 1:n_layers
            A_l, B_l = AB[l]

            # --- Compute I_l: EP energy gradient for layer l ---

            # Feedforward: σ(W_l * z_{l-1} + b_l)
            pre_l = weights[l] * inputs[l]
            if biases[l] !== nothing
                pre_l = pre_l .+ biases[l]
            end
            I_l = activation(pre_l)

            # Feedback from above: W_{l+1}^T * [σ'(pre_{l+1}) .* z_{l+1}]
            if l < n_layers
                pre_above = weights[l+1] * states[l]
                if biases[l+1] !== nothing
                    pre_above = pre_above .+ biases[l+1]
                end
                sigma_prime = holotanh_deriv(pre_above)
                I_l = I_l .+ transpose(weights[l+1]) * (sigma_prime .* states[l+1])
            end

            # Teaching signal: -β * ∂C/∂z_L (output layer only)
            if l == n_layers && beta != 0
                if readout_conj !== nothing
                    # Interference readout: logits = codes_conj^T * z / d
                    d = size(readout_conj, 1)
                    logits = transpose(readout_conj) * states[l] ./ Float32(d)
                    dC_dlogits = hep_interference_cost_grad(logits, y)
                    # Chain rule: ∂C/∂z = codes_conj * ∂C/∂logits / d
                    dC_dz = readout_conj * dC_dlogits ./ Float32(d)
                else
                    dC_dz = hep_cost_xent_grad(states[l], y)
                end
                I_l = I_l .- beta .* dC_dz
            end

            # Phasor recurrence step: z[n+1] = A .* z[n] + B .* I
            new_states[l] = A_l .* states[l] .+ B_l .* I_l
        end

        states = new_states
    end

    return Tuple(states)
end

"""
Determine element type for states. Complex beta requires complex states.
"""
function _state_eltype(beta, x)
    if beta isa Complex || eltype(x) <: Complex
        return ComplexF32
    else
        return Float32
    end
end

"""
Initialize states from a single forward pass.
"""
function _forward_init(weights, biases, x, activation, ET)
    n_layers = length(weights)
    states = Vector{Any}(undef, n_layers)
    h = ET.(x)
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

# ================================================================
# 5. Energy Function (for verification / monitoring)
# ================================================================

"""
    hep_energy(states, weights, biases, x, y, beta; activation)

Compute the Hopfield energy Φ (Eq. 2 in derivation):

    Φ = Σ_l <σ(W_l z_{l-1} + b_l), z_l> - β C(z_L, y)

Uses the complex bilinear form (no conjugation).
"""
function hep_energy(states, weights, biases, x, y, beta;
                    activation = holotanh)
    phi = zero(ComplexF32)
    inputs = (x, states[1:end-1]...)

    for l in 1:length(weights)
        pre = weights[l] * inputs[l]
        if biases[l] !== nothing
            pre = pre .+ biases[l]
        end
        # Bilinear form: sum(σ(pre) .* z), no conjugation
        phi = phi + sum(activation(pre) .* states[l])
    end

    cost = hep_cost_xent(states[end], y)
    phi = phi - beta * cost

    return phi
end

# ================================================================
# 6. Energy Gradient w.r.t. Parameters
# ================================================================

"""
    _energy_param_gradients(states, weights, biases, x; activation)

Compute ∂Φ/∂θ at the given equilibrium states (Eq. 15 in derivation):

    ∂Φ/∂W_l = [σ'(pre_l) .* z_l] * z_{l-1}^T

Note: uses transpose (not adjoint) for z_{l-1}^T to preserve
holomorphicity.
"""
function _energy_param_gradients(states, weights, biases, x;
                                 activation = holotanh)
    n_layers = length(weights)
    inputs = (x, states[1:end-1]...)

    weight_grads = []
    bias_grads = []
    k_grads = []  # (log_neg_lambda_grad, omega_grad) per layer

    for l in 1:n_layers
        pre = weights[l] * inputs[l]
        if biases[l] !== nothing
            pre = pre .+ biases[l]
        end

        # ∂Φ/∂W_l = [σ'(pre_l) .* z_l] * z_{l-1}^T
        sigma_prime = holotanh_deriv(pre)
        modulated = sigma_prime .* states[l]
        # transpose (not adjoint ') to avoid conjugation
        w_grad = modulated * transpose(inputs[l])
        push!(weight_grads, w_grad)

        # ∂Φ/∂b_l = mean over batch of [σ'(pre_l) .* z_l]
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
# 7. Contour Integration for Gradient Computation
# ================================================================

"""
    hep_gradient(weights, biases, k_arrays, x, y;
                 N, r, T_free, T_nudge, dt, kwargs...)

Compute parameter gradients via holomorphic EP contour integration:

1. Free phase: settle with β=0
2. For n = 0..N-1: settle with β_n = r exp(2πin/N), compute ∂Φ/∂θ
3. Gradient = (1/N) Σ_n ∂Φ/∂θ|_{z*(β_n)} exp(-2πin/N)

# Arguments
- `weights`: Tuple of weight matrices
- `biases`: Tuple of bias vectors (or nothing)
- `k_arrays`: Tuple of (log_neg_lambda, omega) per layer
- `x`: Input batch
- `y`: Target labels (one-hot)
- `N::Int=4`: Contour points
- `r::Float32=0.5f0`: Contour radius
- `T_free::Int=100`: Free phase settling steps
- `T_nudge::Int=30`: Nudged phase settling steps

# Returns
(weight_gradients, bias_gradients) tuples.
"""
function hep_gradient(weights, biases, k_arrays, x, y;
                      N::Int = 4,
                      r::Float32 = 0.5f0,
                      T_free::Int = 100,
                      T_nudge::Int = 30,
                      dt::Float32 = 1.0f0,
                      activation = holotanh,
                      readout_conj = nothing)
    n_layers = length(weights)

    # 1. Free phase
    states_free = hep_equilibrium(weights, biases, k_arrays, x, 0.0f0, y;
                                  T=T_free, dt=dt, activation=activation,
                                  readout_conj=readout_conj)

    # 2. Contour integration
    w_grads_accum = [zeros(ComplexF32, size(w)) for w in weights]
    b_grads_accum = [b === nothing ? nothing : zeros(ComplexF32, size(b)) for b in biases]

    for n in 0:N-1
        angle_n = 2.0f0 * Float32(pi) * n / N
        beta_n = r * exp(1.0f0im * angle_n)

        # Nudged equilibrium from free-phase init
        states_n = hep_equilibrium(weights, biases, k_arrays, x, beta_n, y;
                                   T=T_nudge, dt=dt, init=states_free,
                                   activation=activation, readout_conj=readout_conj)

        # Energy gradient w.r.t. parameters at this contour point
        wg, bg = _energy_param_gradients(states_n, weights, biases, x;
                                         activation=activation)

        # Accumulate with inverse contour phase weighting
        phase_weight = exp(-1.0f0im * angle_n)
        for l in 1:n_layers
            w_grads_accum[l] .+= ComplexF32.(wg[l]) .* phase_weight
            if b_grads_accum[l] !== nothing && bg[l] !== nothing
                b_grads_accum[l] .+= ComplexF32.(bg[l]) .* phase_weight
            end
        end
    end

    # 3. First Fourier coefficient (real part)
    w_grads = [real.(wg) ./ N for wg in w_grads_accum]
    b_grads = [bg === nothing ? nothing : real.(bg) ./ N for bg in b_grads_accum]

    return w_grads, b_grads
end

# ================================================================
# 8. Lux-Compatible Wrappers
# ================================================================

"""
    extract_hep_params(ps::NamedTuple, st::NamedTuple)

Extract weights, biases, dynamics parameters, and readout codes from
a Lux Chain's parameter/state NamedTuples.

Returns (weights, biases, k_arrays, layer_keys, readout_conj) where:
- weights: Tuple of weight matrices (PhasorDense layers only)
- biases: Tuple of bias vectors (or nothing)
- k_arrays: Tuple of (log_neg_lambda, omega) pairs
- layer_keys: Symbol keys mapping back to Chain structure
- readout_conj: Conjugated codebook (from HolomorphicReadout state), or nothing
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

# Version without state — uses default dynamics, no readout
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

Pack hEP gradients back into a NamedTuple matching the Lux parameter
structure for use with Optimisers.update.

Dynamics parameters (log_neg_lambda, omega) receive zero gradients
in this implementation. Future work: compute dynamics gradients via
the contour integration (they're holomorphic in k).
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
# 9. High-Level Training Interface
# ================================================================

"""
    hep_train(model, ps, st, train_loader, args; kwargs...)

Train a phasor network using holomorphic equilibrium propagation.

# Arguments
- `model`: Lux Chain model
- `ps`, `st`: Model parameters and states
- `train_loader`: Iterable of (x, y) batches
- `args`: Must have .lr, .epochs fields
- `N::Int=4`: Contour points
- `r::Float32=0.5f0`: Contour radius
- `T_free::Int=100`: Free phase settling steps
- `T_nudge::Int=30`: Nudged phase settling steps
- `dt::Float32=1.0f0`: Discretization step

# Returns
(losses, ps, st)
"""
function hep_train(model, ps, st, train_loader, args;
                   N::Int = 4,
                   r::Float32 = 0.5f0,
                   T_free::Int = 100,
                   T_nudge::Int = 30,
                   dt::Float32 = 1.0f0,
                   activation = holotanh,
                   verbose::Bool = false)
    opt_state = Optimisers.setup(Optimisers.Adam(args.lr), ps)
    losses = Float32[]

    for epoch in 1:args.epochs
        for (x, y) in train_loader
            weights, biases, k_arrays, layer_keys, readout_conj = extract_hep_params(ps, st)

            w_grads, b_grads = hep_gradient(weights, biases, k_arrays, x, y;
                                            N=N, r=r, dt=dt,
                                            T_free=T_free, T_nudge=T_nudge,
                                            activation=activation,
                                            readout_conj=readout_conj !== nothing ? ComplexF32.(readout_conj) : nothing)

            grad_nt = pack_hep_gradients(w_grads, b_grads, layer_keys, ps)
            opt_state, ps = Optimisers.update(opt_state, ps, grad_nt)

            # Track loss via the model's own forward pass
            y_pred, _ = model(x, ps, st)
            lossval = if y_pred isa AbstractArray{<:Complex}
                # HolomorphicReadout produces complex logits
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
