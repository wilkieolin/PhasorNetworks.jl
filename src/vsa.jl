using Statistics: mean
using Random: AbstractRNG
using DifferentialEquations

include("spiking.jl")

function angle_to_complex(x::AbstractArray)
    k = convert(ComplexF32, pi * (0.0 + 1.0im))
    return exp.(k .* x)
end

function bind(x::AbstractArray; dims)
    bz = sum(x, dims = dims)
    y = remap_phase(bz)
    return y
end

function bind(x::AbstractArray, y::AbstractArray)
    y = remap_phase(x + y)
    return y
end

function bind(x::SpikingCall, y::SpikingCall; kwargs...)
    train = bind(x.train, y.train; tspan=x.t_span, spk_args=x.spk_args, kwargs...)
    next_call = SpikingCall(train, x.spk_args, x.t_span)
    return next_call
end

function bind(x::SpikeTrain, y::SpikeTrain; tspan::Tuple{<:Real, <:Real} = (0.0, 10.0), spk_args::SpikingArgs = default_spk_args(), return_solution::Bool = false, unbind::Bool=false, automatch::Bool=true)
    if !automatch
        if check_offsets(x::SpikeTrain, y::SpikeTrain) @warn "Offsets between spike trains do not match - may not produce desired phases" end
    else
        x, y = match_offsets(x, y)
    end

    #set up functions to define the neuron'sa differential equations
    k = neuron_constant(spk_args)
    tbase = tspan[1]:spk_args.dt:tspan[2]

    #get the number of batches & output neurons
    output_shape = x.shape

    #find the complex state induced by the spikes
    sol_x = phase_memory(x, tspan=tspan, spk_args=spk_args)
    sol_y = phase_memory(y, tspan=tspan, spk_args=spk_args)

    to_array = x -> normalize_potential.(Array(x))
    u_x = to_array(sol_x)
    u_y = to_array(sol_y)

    n_t = length(sol_x.t)
    ref_shape = (ones(Int, length(output_shape))..., n_t)
    #create a reference oscillator to generate complex values for each moment in time
    u_ref = phase_to_potential(0.0, sol_x.t, x.offset, spk_args)
    u_ref = reshape(u_ref, ref_shape)

    #return u_x, u_y, u_ref
    
    #find the first chord
    chord_x = u_x
    #find the second chord
    if unbind
        chord_y = u_x .* conj.((u_y .- u_ref)) .* u_ref
    else
        chord_y = u_x .* (u_y .- u_ref) .* conj(u_ref)
    end

    u_output = chord_x .+ chord_y
    
    if return_solution
        return u_output
    end
    
    indices, times = find_spikes_rf(u_output, tbase, spk_args, dim=ndims(u_output))
    #construct the spike train and call for the next layer
    train = SpikeTrain(indices, times, output_shape, x.offset + spiking_offset(spk_args))
    return train

end

function bundle(x::AbstractArray; dims::Int)
    xz = angle_to_complex(x)
    bz = sum(xz, dims = dims)
    y = complex_to_angle(bz)
    return y
end

function bundle(x::SpikingCall; dims::Int)
    train = bundle(x.train, dims=dims, tspan=x.t_span, spk_args=x.spk_args)
    next_call = SpikingCall(train, x.spk_args, x.t_span)
    return next_call
end

function bundle(x::SpikeTrain; dims::Int, tspan::Tuple{<:Real, <:Real} = (0.0, 10.0), spk_args::SpikingArgs=default_spk_args(), return_solution::Bool=false)
    #let compartments resonate in sync with inputs
    sol = phase_memory(x, tspan=tspan, spk_args=spk_args)
    #extract their potentials
    u = Array(sol)
    nu = normalize_potential.(u)
    #combine the potentials (interfere) along the bundling axis
    bundled = sum(nu, dims=dims)
    if return_solution
        return bundled
    end
    
    #detect spiking outputs
    out_shape = size(bundled)[1:end-1]
    out_inds, out_tms = find_spikes_rf(bundled, sol.t, spk_args)
    out_offset = x.offset + spiking_offset(spk_args)
    out_train = SpikeTrain(out_inds, out_tms, out_shape, out_offset)
    return out_train
end

function bundle_project(x::AbstractMatrix, w::AbstractMatrix, b::AbstractVecOrMat)
    xz = w * angle_to_complex(x) .+ b
    y = complex_to_angle(xz)
    return y
end

#TODO - make dimensions constant with static call
function bundle_project(x::SpikeTrain, w::AbstractMatrix, b::AbstractVecOrMat, tspan::Tuple{<:Real, <:Real}, spk_args::SpikingArgs; return_solution::Bool=false)
    #set up functions to define the neuron's differential equations
    k = neuron_constant(spk_args)
    #get the number of batches & output neurons
    output_shape = (size(w, 1), x.shape[2])
    u0 = zeros(ComplexF32, output_shape)
    dzdt(u, p, t) = k .* u + w * spike_current(x, t, spk_args) .+ bias_current(b, t, x.offset, spk_args)
    #solve the ODE over the given time span
    prob = ODEProblem(dzdt, u0, tspan)
    sol = solve(prob, Heun(), adaptive=false, dt=spk_args.dt)
    #option for early exit (mostly for debug)
    if return_solution return sol end

    #convert the full solution (potentials) to spikes
    indices, times = find_spikes_rf(sol, spk_args)
    #construct the spike train and call for the next layer
    train = SpikeTrain(indices, times, output_shape, x.offset + spiking_offset(spk_args))
    next_call = SpikingCall(train, spk_args, tspan)
    return next_call
end

function bundle_project(x::LocalCurrent, w::AbstractMatrix, b::AbstractVecOrMat, tspan::Tuple{<:Real, <:Real}, spk_args::SpikingArgs; return_solution::Bool=false)
    #set up functions to define the neuron's differential equations
    angular_frequency = 2 * pi / spk_args.t_period
    k = (spk_args.leakage + 1im * angular_frequency)
    output_shape = (size(w, 1), x.shape[2])
    u0 = zeros(ComplexF32, output_shape)
    dzdt(u, p, t) = k .* u + w * x.current_fn(t) .+ bias_current(b, t, x.offset, spk_args)
    #solve the ODE over the given time span
    prob = ODEProblem(dzdt, u0, tspan)
    sol = solve(prob, Heun(), adaptive=false, dt=spk_args.dt)
    #option for early exit (mostly for debug)
    if return_solution return sol end
    
    #convert the full solution (potentials) to spikes
    indices, times = find_spikes_rf(sol, spk_args)
    #construct the spike train and call for the next layer
    train = SpikeTrain(indices, times, output_shape, x.offset + spiking_offset(spk_args))
    next_call = SpikingCall(train, spk_args, tspan)
    return next_call
end

function complex_to_angle(x::AbstractArray)
    return angle.(x) ./ pi
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

function similarity_outer(x::SpikeTrain, y::SpikeTrain; dims, reduce_dim::Int=-1, tspan::Tuple{<:Real, <:Real} = (0.0, 10.0), spk_args::SpikingArgs = default_spk_args(), automatch::Bool=true)
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

function unbind(x::AbstractArray, y::AbstractArray)
    y = remap_phase(x .- y)
    return y
end

function unbind(x::SpikeTrain, y::SpikeTrain; kwargs...)
    return bind(x, y, unbind=true; kwargs...)
end