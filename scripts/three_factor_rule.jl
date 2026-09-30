# The three-factor form of the lock-in update, extracted verbatim from
# scripts/e4_three_factor_scale.jl (lines 50-166) so that the E4 sweep and the
# component-pair export in scripts/e4_export_update_pairs.jl cannot drift apart.
# ep_three_factor_stdp.jl carries its own older copy; that one drives the toy
# 4->8->2 chain and is not used by the manuscript figures.
#
# Callers must already have imported the PhasorNetworks internals this uses:
#   phasor_settle, chain_hebbians, _phase_input_to_complex, _weight_cache,
#   _input_drive, _phasor_step, _pad_dynamics_zeros, _zero_grad, gpu_zeros

Base.@kwdef struct ThreeFactorLockin
    ε::Float32          = 0.05f0
    ω_p::Float32        = 0.05f0
    n_cycles::Int       = 4
    T_warmup_cycles::Int = 2
    T_free::Int         = 100
    dt::Float32         = 0.1f0
    K_mode::Symbol      = :zero
    project::Symbol     = :hard
end

function three_factor_gradient(m::ThreeFactorLockin, chain::Lux.Chain, ps, st, x,
                               cost::AbstractEPCost;
                               omega_override::Union{Nothing, Vector} = nothing)
    # 1. Free settle
    s_free = phasor_settle(chain, ps, st, x, cost, 0f0;
                           T=m.T_free, dt=m.dt, K_mode=m.K_mode,
                           omega_override=omega_override, project=m.project)
    t_free_end = Float32(m.T_free * m.dt)

    # 2. DC Hebbian (for DC subtraction)
    h_dc = chain_hebbians(chain, ps, st, x, s_free)

    # 3. Warmup
    layer_keys = collect(keys(ps))
    z0 = _phase_input_to_complex(x)
    period_steps = round(Int, 2π / (m.ω_p * m.dt))
    T_warmup = m.T_warmup_cycles * period_steps
    T_lockin = m.n_cycles * period_steps

    states = [copy(s) for s in s_free]
    cache = _weight_cache(chain, ps, layer_keys)
    drive0 = _input_drive(chain, ps, st, layer_keys, z0; cache=cache)

    # Warmup
    for t in 1:T_warmup
        β_t = m.ε * cos(m.ω_p * t * m.dt)
        t_now = t_free_end + Float32((t - 1) * m.dt)
        states = _phasor_step(chain, ps, st, layer_keys, z0, cost,
                              β_t, m.dt, states; K_mode=m.K_mode,
                              omega_override=omega_override, drive0=drive0,
                              cache=cache, carriers=nothing, t_now=t_now,
                              project=m.project, soft_ε=0.1f0)
    end

    # 4. Three-factor accumulation
    t_warm_end = t_free_end + Float32(T_warmup * m.dt)

    # Eligibility accumulators
    E_W = Dict{Symbol, Any}()
    E_b = Dict{Symbol, Any}()
    for (l, key) in enumerate(layer_keys)
        haskey(ps[key], :weight) || continue
        E_W[key] = gpu_zeros(ps[key].weight, ComplexF32, size(ps[key].weight)...)
        if haskey(ps[key], :bias_real)
            E_b[key] = gpu_zeros(states[l], ComplexF32, size(states[l])...)
        end
    end

    # Integration with explicit three-factor rule
    for t in 1:T_lockin
        β_t = m.ε * cos(m.ω_p * t * m.dt)
        t_now = t_warm_end + Float32((t - 1) * m.dt)
        states = _phasor_step(chain, ps, st, layer_keys, z0, cost,
                              β_t, m.dt, states; K_mode=m.K_mode,
                              omega_override=omega_override, drive0=drive0,
                              cache=cache, carriers=nothing, t_now=t_now,
                              project=m.project, soft_ε=0.1f0)

        demod = ComplexF32(exp(-im * m.ω_p * Float32(t) * m.dt))

        for l in 1:length(layer_keys)
            key = layer_keys[l]
            haskey(ps[key], :weight) || continue

            z_l = states[l]
            z_in = (l == 1) ? z0 : states[l-1]

            # Eligibility trace: Hebbian outer product (adjoint for complex)
            elig_W = z_l * adjoint(z_in)
            elig_b = z_l

            # DC-subtracted eligibility
            elig_W_dc = elig_W .- h_dc[key].weight
            elig_b_dc = elig_b .- ComplexF32.(h_dc[key].bias_real .+ 1f0im .* h_dc[key].bias_imag)

            # Three-factor: accumulate eligibility × demodulation
            E_W[key] .+= elig_W_dc .* demod
            if haskey(E_b, key)
                E_b[key] .+= elig_b_dc .* demod
            end
        end
    end

    # 5. Convert to real-parameter gradients
    norm_factor = Float32(T_lockin) * m.ε
    grads = NamedTuple()

    for key in layer_keys
        if haskey(ps[key], :weight)
            entry = (weight = -2f0 .* real.(E_W[key]) ./ norm_factor,)
            if haskey(ps[key], :bias_real)
                entry = merge(entry, (
                    bias_real = -2f0 .* real.(E_b[key]) ./ norm_factor,
                    bias_imag = -2f0 .* imag.(E_b[key]) ./ norm_factor,
                ))
            end
            entry = _pad_dynamics_zeros(entry, ps[key])
            grads = merge(grads, NamedTuple{(key,)}((entry,)))
        else
            grads = merge(grads, NamedTuple{(key,)}((_zero_grad(ps[key]),)))
        end
    end

    return grads, s_free
end
