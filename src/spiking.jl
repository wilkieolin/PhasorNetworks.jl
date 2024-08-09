using DifferentialEquations: ODESolution

include("domains.jl")

function angular_mean(phases::AbstractArray; dims)
    u = exp.(pi * 1im .* phases)
    u_mean = mean(u, dims=dims)
    phase = angle.(u_mean) ./ pi
    return phase
end

function bias_current(bias::AbstractArray{<:Complex}, t::Real, t_offset::Real, spk_args::SpikingArgs; sigma::Real=9.0)
    phase = complex_to_angle(bias)
    mag = abs.(bias)
    return bias_current(phase, mag, t, t_offset, spk_args, sigma=sigma)
end

function bias_current(phase::AbstractArray{<:Real}, mag::AbstractArray{<:Real}, t::Real, t_offset::Real, spk_args::SpikingArgs; sigma::Real=9.0)
    #what times to the bias values correlate to?
    times = phase_to_time(phase, spk_args=spk_args, offset=t_offset)
    #determine the time within the cycle
    t = mod(t, spk_args.t_period)

    #add the active currents, scaled by the gaussian kernel & bias magnitude
    current_kernel = x -> gaussian_kernel(x, t, spk_args.t_window)
    bias = mag .* current_kernel(times)

    return bias
end

function check_offsets(x::SpikeTrain, y::SpikeTrain)
    if x.offset != y.offset
        return false
    else
        return true
    end
end

function check_offsets(x::SpikeTrain...)
    offset = x[1].offset
    for st in x
        if st.offset != offset
            return false
        end
    end
    return true
end

function count_nans(phases::Array{<:Real,3})
    return mapslices(x->sum(isnan.(x)), phases, dims=(2,3)) |> vec
end

function delay_train(train::SpikeTrain, t::Real, offset::Real)
    times = train.times .+ t
    new_train = SpikeTrain(train.indices, times, train.shape, train.offset + offset)
    return new_train
end

function find_spikes_rf(sol::ODESolution, spk_args::SpikingArgs; dim::Int=-1)
    @assert typeof(sol.u) <: Vector{<:Array{<:Complex}} "This method is for R&F neurons with complex potential"    
    t = sol.t
    u = solution_to_potential(sol, t)

    return find_spikes_rf(u, t, spk_args, dim=dim)
end

function find_spikes_rf(u::AbstractArray, t::AbstractVector, spk_args::SpikingArgs; dim::Int=-1)
    #choose the last dimension as default
    if dim == -1
        dim = ndims(u)
    end

    #if potential is from an R&F neuron, it is complex and voltage is the imaginary part
    voltage = imag.(u)
    current = real.(u)

    #find the local voltage maxima through the first derivative (current)
    op = x -> x .< 0
    #find maxima along the temporal dimension
    maxima = findall(op(diff(sign.(current), dims=dim)))
    peak_voltages = voltage[maxima]
    #check voltages at these peaks are above the threshold
    above_threshold = peak_voltages .> spk_args.threshold
    spikes = maxima[above_threshold]

    #retrieve the indices of the spiking neurons
    ax = 1:ndims(u) |> collect
    spatial_ax = setdiff(ax, dim)
    spatial_idx = [getindex.(spikes, i) for i in spatial_ax]
    channels = CartesianIndex.(spatial_idx...) 
    #retrieve the times they spiked at
    times = t[getindex.(spikes, dim)]
    
    return channels, times
end

function gaussian_kernel(x::AbstractVecOrMat, t::Real, t_sigma::Real)
    i = exp.(-1 .* ((t .- x) / (2 .* t_sigma)).^2)
    return i
end

function generate_cycles(tspan::Tuple{<:Real, <:Real}, spk_args::SpikingArgs, offset::Real)
    #determine what the cycle offset should be
    offset = mod(offset, spk_args.t_period)
    #determine when each cycles begins and ends
    r = tspan[1]:spk_args.t_period:tspan[2]
    r = collect(r) .+ offset
    return r[2:end]
end

function is_active(times::AbstractArray, t::Real, t_window::Real; sigma::Real=9.0)
    active = (times .> (t - sigma * t_window)) .* (times .< (t + sigma * t_window))
    return active
end

"""
Delay spike trains as necessary to make the represented phases between them match
"""
function match_offsets(x::SpikeTrain, y::SpikeTrain)
    xo = x.offset
    yo = y.offset

    if xo == yo
        return x, y
    elseif xo > yo
        dy = xo - yo
        yp = delay_train(y, dy, dy)
        return x, yp
    else
        dx = yo - xo
        xp = delay_train(x, dx, dx)
        return xp, y
    end
end

"""
Delay the spike trains in a vector as necessary to make their offsets match
"""
function match_offsets(x::Vector{<:SpikeTrain})
    offsets = getfield.(x, :offset)
    final = maximum(offsets)
    dt = final .- offsets
    new_trains = [delay_train(st, dt[i], dt[i]) for (i, st) in enumerate(x)]
    return new_trains
end

function match_tspans(spans::Tuple{<:Real, <:Real}...)
    start = minimum([s[1] for s in spans])
    stop = maximum([s[2] for s in spans])
    return (start, stop)
end

function mean_phase(solution::ODESolution, i_warmup::Int; spk_args::SpikingArgs, offset::Real=0.0, kwargs...)
    inds = solution.t .> (i_warmup * spk_args.t_period)

    u = Array(solution)[:,:,inds]
    t = solution.t[inds]
    phase = potential_to_phase(u, t, offset=offset, spk_args=spk_args, kwargs...)
    phase = angular_mean(phase, dims=(3))[:,:,1]

    return phase
end

function normalize_potential(u::Complex)
    a = abs(u)
    if a == 0.0
        return u
    else
        return u / a
    end
end

function phase_to_current(phases::AbstractArray; spk_args::SpikingArgs, offset::Real = 0.0, tspan::Tuple{<:Real, <:Real})
    shape = size(phases)
    
    function inner(t::Real)
        output = zero(phases)

        ignore_derivatives() do
            times = phase_to_time(phases, spk_args = spk_args, offset = offset)

            #add currents into the active synapses
            current_kernel = x -> gaussian_kernel(x, mod(t, spk_args.t_period), spk_args.t_window)
            impulses = current_kernel(times)
            output .+= impulses
        end

        return output
    end

    current = LocalCurrent(inner, shape, offset)
    call = CurrentCall(current, spk_args, tspan)

    return call
end

function spike_current(train::SpikeTrain, t::Real, spk_args::SpikingArgs; sigma::Real = 9.0)
    current = zeros(Float32, train.shape)

    ignore_derivatives() do
        #find which channels are active 
        times = train.times
        active = is_active(times, t, spk_args.t_window, sigma=sigma)
        active_inds = train.indices[active]

        #add currents into the active synapses
        current_kernel = x -> gaussian_kernel(x, t, spk_args.t_window)
        impulses = current_kernel(train.times[active])
        
        current[active_inds] .+= impulses
    end

    return current
end

function spiking_offset(spk_args::SpikingArgs)
    return spk_args.t_period / 4.0
end

function stack_trains(trains::Array{<:SpikeTrain,1})
    check_offsets(trains...)
    n_t = length(trains)
    shape = trains[1].shape
    offset = trains[1].offset
    for t in trains
        @assert shape == t.shape "Spike trains must have identical shape to be stacked"
        @assert offset == t.offset "Spike trains must have identical offsets"
    end

    new_shape = (n_t, shape...)
    all_indices = []

    for (i, train) in enumerate(trains)
        old_indices = train.indices
        #add the new dimension for each index
        new_indices = [CartesianIndex((i, Tuple(idx)...)) for idx in old_indices]
        append!(all_indices, new_indices)
    end

    all_indices = vcat(all_indices...)
    all_times = reduce(vcat, collect(t.times for t in trains))

    new_train = SpikeTrain(all_indices, all_times, new_shape, offset)
    return new_train
end

function phase_memory(x::SpikeTrain; tspan::Tuple{<:Real, <:Real} = (0.0, 10.0), spk_args::SpikingArgs)
    #set up functions to define the neuron's differential equations
    k = neuron_constant(spk_args)

    #set up compartments for each sample
    u0 = zeros(ComplexF32, x.shape)
    #resonate in time with the input spikes
    dzdt(u, p, t) = k .* u .+ spike_current(x, t, spk_args)
    #solve the memory compartment
    prob = ODEProblem(dzdt, u0, tspan)
    sol = solve(prob, spk_args.solver; spk_args.solver_args...)

    return sol
end

function phase_memory(x::LocalCurrent; tspan::Tuple{<:Real, <:Real} = (0.0, 10.0), spk_args::SpikingArgs)
    #set up functions to define the neuron's differential equations
    k = neuron_constant(spk_args)

    #set up compartments for each sample
    u0 = zeros(ComplexF32, x.shape)
    #resonate in time with the input spikes
    dzdt(u, p, t) = k .* u .+ x.current_fn(t)
    #solve the memory compartment
    prob = ODEProblem(dzdt, u0, tspan)
    sol = solve(prob, spk_args.solver; spk_args.solver_args...)

    return sol
end

function phase_memory(x::CurrentCall)
    return phase_memory(x.current, tspan=x.t_span, spk_args=x.spk_args)
end

function vcat_trains(trains::Array{<:SpikeTrain,1})
    check_offsets(trains...)
    n_t = length(trains)
    shape = trains[1].shape
    offset = trains[1].offset
    for t in trains
        @assert shape == t.shape "Spike trains must have identical shape to be stacked"
        @assert offset == t.offset "Spike trains must have identical offsets"
    end

    new_shape = (n_t, shape[2:end]...)
    all_indices = []

    for (i, train) in enumerate(trains)
        old_indices = train.indices
        #add the new dimension for each index
        new_indices = [CartesianIndex((i, Tuple(idx)[2:end]...)) for idx in old_indices]
        append!(all_indices, new_indices)
    end

    all_indices = vcat(all_indices...)
    all_times = reduce(vcat, collect(t.times for t in trains))

    new_train = SpikeTrain(all_indices, all_times, new_shape, offset)
    return new_train
end

function zero_nans(phases::AbstractArray)
    nans = isnan.(phases)
    phases[nans] .= 0.0
    return phases
end