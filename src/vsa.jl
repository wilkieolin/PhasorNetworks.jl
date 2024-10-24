using Statistics: mean
using Random: AbstractRNG
using DifferentialEquations

include("spiking.jl")

function v_bind(x::AbstractArray; dims)
    bz = sum(x, dims = dims)
    y = remap_phase(bz)
    return y
end

function v_bind(x::AbstractArray, y::AbstractArray)
    y = remap_phase(x + y)
    return y
end

function v_bind(x::SpikingCall, y::SpikingCall; kwargs...)
    train = v_bind(x.train, y.train; tspan=x.t_span, spk_args=x.spk_args, kwargs...)
    next_call = SpikingCall(train, x.spk_args, x.t_span)
    return next_call
end

function v_bind(x::SpikeTrain, y::SpikeTrain; tspan::Tuple{<:Real, <:Real} = (0.0, 10.0), spk_args::SpikingArgs, return_solution::Bool = false, unbind::Bool=false, automatch::Bool=true)
    if !automatch
        if check_offsets(x::SpikeTrain, y::SpikeTrain) @warn "Offsets between spike trains do not match - may not produce desired phases" end
    else
        x, y = match_offsets(x, y)
    end

    #set up functions to define the neuron's differential equations
    k = neuron_constant(spk_args)

    #get the number of batches & output neurons
    output_shape = x.shape

    #find the complex state induced by the spikes
    sol_x = phase_memory(x, tspan=tspan, spk_args=spk_args)
    sol_y = phase_memory(y, tspan=tspan, spk_args=spk_args)
    
    #create a reference oscillator to generate complex values for each moment in time
    u_ref = t -> phase_to_potential(0.0, t, offset = x.offset, spk_args = spk_args)

    #find the first chord
    chord_x = t -> sol_x(t)
    #find the second chord
    if unbind
        chord_y = t -> sol_x(t) .* conj.((sol_y(t) .- u_ref(t))) .* u_ref(t)
    else
        chord_y = t -> sol_x(t) .* (sol_y(t) .- u_ref(t)) .* conj(u_ref(t))
    end

    sol_output = t -> chord_x(t) .+ chord_y(t)
    
    if return_solution
        return sol_output
    end

    train = solution_to_train(sol_output, tspan, spk_args=spk_args, offset=x.offset)
    return train
end

function v_bundle(x::AbstractArray; dims::Int)
    xz = angle_to_complex(x)
    bz = sum(xz, dims = dims)
    y = complex_to_angle(bz)
    return y
end

function v_bundle(x::SpikingCall; dims::Int)
    train = v_bundle(x.train, dims=dims, tspan=x.t_span, spk_args=x.spk_args)
    next_call = SpikingCall(train, x.spk_args, x.t_span)
    return next_call
end

function v_bundle(x::SpikeTrain; dims::Int, tspan::Tuple{<:Real, <:Real} = (0.0, 10.0), spk_args::SpikingArgs, return_solution::Bool=false)
    #let compartments resonate in sync with inputs
    sol = phase_memory(x, tspan=tspan, spk_args=spk_args)
    tbase = sol.t
    #combine the potentials (interfere) along the bundling axis
    f_sol = x -> sum(normalize_potential.(sol(x)), dims=dims)

    if return_solution
        return f_sol
    end
    
    out_train = solution_to_train(f_sol, tspan, spk_args=spk_args, offset=x.offset)
    return out_train
end

function v_bundle_project(x::AbstractMatrix, w::AbstractMatrix, b::AbstractVecOrMat)
    xz = w * angle_to_complex(x) .+ b
    y = complex_to_angle(xz)
    return y
end

function v_bundle_project(x::SpikingCall, w::AbstractMatrix, b::AbstractVecOrMat; return_solution::Bool=false)
    train = v_bundle_project(x.train, w, b, tspan=x.t_span, spk_args=x.spk_args, return_solution = return_solution)
    if return_solution
        return train
    end
    next_call = SpikingCall(train, x.spk_args, x.t_span)
    return next_call
end

function v_bundle_project(x::SpikeTrain, w::AbstractMatrix, b::AbstractVecOrMat; tspan::Tuple{<:Real, <:Real}, spk_args::SpikingArgs, return_solution::Bool=false)
    #set up functions to define the neuron's differential equations
    update_fn = spk_args.update_fn
    #get the number of batches & output neurons
    output_shape = (size(w, 1), x.shape[2])
    u0 = zeros(ComplexF32, output_shape)
    dzdt(u, p, t) = update_fn(u) + w * spike_current(x, t, spk_args) .+ bias_current(b, t, x.offset, spk_args)
    #solve the ODE over the given time span
    prob = ODEProblem(dzdt, u0, tspan)
    sol = solve(prob, spk_args.solver; spk_args.solver_args...)
    #return full solution
    if return_solution return sol end

    #convert the full solution (potentials) to spikes
    train = solution_to_train(sol, tspan, spk_args = spk_args, offset = x.offset)
    return train
end

function v_bundle_project(x::LocalCurrent, w::AbstractArray{<:Real,2}, b::AbstractArray{<:Complex,1}; tspan::Tuple{<:Real, <:Real}, spk_args::SpikingArgs, return_solution::Bool=false)
    #set up functions to define the neuron's differential equations
    update_fn = spk_args.update_fn
    output_shape = (size(w, 1), x.shape[2])
    #make the initial potential the bias value
    u0 = zeros(ComplexF32, output_shape)
    #shift the solver span by the function's time offset
    tspan = tspan .+ x.offset
    dzdt(u, p, t) = update_fn(u) + w * x.current_fn(t) .+ bias_current(b, t, x.offset, spk_args)
    #solve the ODE over the given time span
    prob = ODEProblem(dzdt, u0, tspan)
    sol = solve(prob, spk_args.solver; spk_args.solver_args...)
    #return full solution
    if return_solution return sol end
    
    #convert the full solution (potentials) to spikes
    train = solution_to_train(sol, tspan, spk_args = spk_args, offset = x.offset)
    next_call = SpikingCall(train, spk_args, tspan)
    return next_call
end

function v_bundle_project(x::LocalCurrent, params; tspan::Tuple{<:Real, <:Real}, spk_args::SpikingArgs, return_solution::Bool=false)
    #set up functions to define the neuron's differential equations
    output_shape = (size(params.weight, 1), x.shape[2])
    #make the initial potential the bias value
    u0 = zeros(ComplexF32, output_shape)
    #shift the solver span by the function's time offset
    tspan = tspan .+ x.offset
    
    #override spk args with params leakage and frequency if provided
    function calc_k(p)
        if haskey(p, :leakage) && haskey(p, :t_period)
            angular_frequency = 2 * pi / p.t_period[1]
            k = (p.leakage[1] + 1im * angular_frequency)
        else
            k = neuron_constant(spk_args)
        end
        return k
    end
    
    function dzdt(u, p, t)
        k = calc_k(p)
        du = k .* u + p.weight * x.current_fn(t) .+ bias_current(p.bias_real .+ 1im .* p.bias_imag, t, x.offset, spk_args)
        return du
    end

    function dzdt_nobias(u, p, t)
        k = calc_k(p)
        du = k .* u + p.weight * x.current_fn(t)
        return du
    end
    
    #enable bias if used
    if haskey(params, :bias_real) && haskey(params, :bias_imag)
        prob = ODEProblem(dzdt, u0, tspan, params)
    else
        prob = ODEProblem(dzdt_nobias, u0, tspan, params)
    end
    
    sol = solve(prob, spk_args.solver; spk_args.solver_args...)
    #return full solution
    if return_solution return sol end
    
    #convert the full solution (potentials) to spikes
    train = solution_to_train(sol, tspan, spk_args = spk_args, offset = x.offset)
    next_call = SpikingCall(train, spk_args, tspan)
    return next_call
end

function chance_level(nd::Int, samples::Int)
    symbol_0 = random_symbols((1, nd))
    symbols = random_symbols((samples, nd))
    sim = similarity_outer(symbol_0, symbols, dims=1) |> vec
    dev = std(sim)

    return dev
end

function random_symbols(size::Tuple{Vararg{Int}})
    y = 2 .* rand(Float32, size) .- 1.0
    return y
end

function random_symbols(size::Tuple{Vararg{Int}}, rng::AbstractRNG)
    y = 2 .* rand(rng, Float32, size) .- 1.0
    return y
end

function remap_phase(x::AbstractArray)
    x = x .+ 1
    x = mod.(x, 2.0)
    x = x .- 1
    return x
end

function similarity(x::AbstractArray, y::AbstractArray; dim::Int = -1)
    if dim == -1
        dim = ndims(x)
    end

    dx = cos.(pi .* (x .- y))
    s = mean(dx, dims = dim)
    return s
end

function similarity(x::SpikeTrain, y::SpikeTrain; spk_args::SpikingArgs, tspan::Tuple{<:Real, <:Real}, automatch::Bool=true)
    if !automatch
        if check_offsets(x::SpikeTrain, y::SpikeTrain) @warn "Offsets between spike trains do not match - may not produce desired phases" end
    else
        x, y = match_offsets(x, y)
    end

    sol_x = phase_memory(x, tspan = tspan, spk_args = spk_args)
    sol_y = phase_memory(y, tspan = tspan, spk_args = spk_args)

    u_x = normalize_potential.(Array(sol_x))
    u_y = normalize_potential.(Array(sol_y))

    interference = abs.(u_x .+ u_y)
    avg_sim = interference_similarity(interference, dim=1)
    return avg_sim
end

function interference_similarity(interference::AbstractArray; dim::Int=-1)
    if dim == -1
        dim = ndims(interference)
    end

    magnitude = clamp.(interference, 0.0, 2.0)
    half_angle = acos.(0.5 .* magnitude)
    sim = cos.(2.0 .* half_angle)
    avg_sim = mean(sim, dims=dim)
    
    return avg_sim
end

function similarity_outer(x::SpikingCall, y::SpikingCall; dims, reduce_dim::Int=-1, automatch::Bool=true)
    @assert x.spk_args == y.spk_args "Spiking arguments must be identical to calculate similarity"
    new_span = match_tspans(x.t_span, y.t_span)
    return similarity_outer(x.train, y.train, dims=dims, reduce_dim=reduce_dim, tspan=new_span, spk_args=x.spk_args, automatch=automatch)
end

function similarity_outer(x::SpikeTrain, y::SpikeTrain; dims, reduce_dim::Int=-1, tspan::Tuple{<:Real, <:Real} = (0.0, 10.0), spk_args::SpikingArgs, automatch::Bool=true)
    if !automatch
        if check_offsets(x::SpikeTrain, y::SpikeTrain) @warn "Offsets between spike trains do not match - may not produce desired phases" end
    else
        x, y = match_offsets(x, y)
    end

    sol_x = phase_memory(x, tspan = tspan, spk_args = spk_args)
    sol_y = phase_memory(y, tspan = tspan, spk_args = spk_args)
    if reduce_dim == -1
        reduce_dim = ndims(sol_x)
    end

    u_x = normalize_potential.(Array(sol_x))
    u_y = normalize_potential.(Array(sol_y))
    
    #add up along the slices
    interference = [abs.(u_xs .+ u_ys) for u_xs in eachslice(u_x, dims=dims), u_ys in eachslice(u_y, dims=dims)]
    avg_sim = interference_similarity.(interference, dim=reduce_dim-1)
    return avg_sim
end

function similarity_self(x::AbstractArray; dims)
    return similarity_outer(x, x, dims=dims)
end

"""
Slicing each array along 'dims', find the similarity between each corresponding slice and
reduce along 'reduce_dim'
"""
function similarity_outer(x::AbstractArray, y::AbstractArray; dims, reduce_dim::Int=-1)
    s = stack([similarity(xs, ys, dim=reduce_dim) for xs in eachslice(x, dims=dims), ys in eachslice(y, dims=dims)])
    return s
end

function v_unbind(x::AbstractArray, y::AbstractArray)
    y = remap_phase(x .- y)
    return y
end

function v_unbind(x::SpikeTrain, y::SpikeTrain; kwargs...)
    return v_bind(x, y, unbind=true; kwargs...)
end