using DifferentialEquations: ODESolution

struct SpikeTrain
    indices::Array{<:Union{Int, CartesianIndex},1}
    times::Array{<:Real,1}
    shape::Tuple
    offset::Real
end

struct LocalCurrent
    current_fn::Function
    shape::Tuple
    offset::Real
end

function Base.show(io::IO, train::SpikeTrain)
    print(io, "Spike Train: ", train.shape, " with ", length(train.times), " spikes.")
end

struct SpikingArgs
    leakage::Real
    t_period::Real
    t_window::Real
    threshold::Real
    dt::Real
end


function SpikingArgs(; leakage::Real = -0.2, 
                    t_period::Real = 1.0,
                    t_window::Real = 0.01,
                    threshold::Real = 0.02,
                    dt::Real = 0.01)
    return SpikingArgs(leakage, t_period, t_window, threshold, dt)
end

function default_spk_args()
    return SpikingArgs()
end

function Base.show(io::IO, spk_args::SpikingArgs)
    print(io, "Neuron parameters: Period ", spk_args.t_period, " (s)\n")
    print(io, "Current kernel duration: ", spk_args.t_window, " (s)\n")
    print(io, "Threshold: ", spk_args.threshold, " (V)\n")
end

struct SpikingCall
    train::SpikeTrain
    spk_args::SpikingArgs
    t_span::Tuple{<:Real, <:Real}
end

struct CurrentCall
    current::LocalCurrent
    spk_args::SpikingArgs
    t_span::Tuple{<:Real, <:Real}
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

function spike_current(train::SpikeTrain, t::Real, spk_args::SpikingArgs)
    #get constants
    t_window = spk_args.t_window
    dt = spk_args.dt

    #determine which synapses will have incoming currents
    #snap spike times to the grid points
    times = train.times .- mod.(train.times, dt)
    active = (times .> (t - 2 * t_window)) .* (times .< (t))
    # relative_time = abs.(times .- t)
    # active = relative_time .<= t_window
    active_inds = train.indices[active]

    #add currents into the active synapses
    current = zeros(Float32, train.shape)
    current[active_inds] .+= (1.0)

    return current
end

function bias_current(bias::AbstractArray, t::Real, t_offset::Real, spk_args::SpikingArgs)
    #get constants
    t_bias = spk_args.t_period / 2.0
    t_window = spk_args.t_window
    #determine the time within the cycle
    t_relative = mod((t - t_offset), spk_args.t_period)
    #determine if the bias is active
    active = (t_relative > (t_bias - t_window)) && (t_relative < (t_bias + t_window))

    if active
        return bias
    else
        return zero(bias)
    end
end

function find_spikes_rf(sol::ODESolution, spk_args::SpikingArgs; dim::Int=-1)
    @assert typeof(sol.u) <: Vector{<:Array{<:Complex}} "This method is for R&F neurons with complex potential"    
    t = sol.t
    u = Array(sol)

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

function neuron_constant(spk_args::SpikingArgs)
    angular_frequency = period_to_angfreq(spk_args.t_period)
    k = (spk_args.leakage + 1im * angular_frequency)
    return k
end

function normalize_potential(u::Complex)
    a = abs(u)
    if a == 0.0
        return u
    else
        return u / a
    end
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


function period_to_angfreq(t_period::Real)
    angular_frequency = 2 * pi / t_period
    return angular_frequency
end

function phase_memory(x::SpikeTrain; tspan::Tuple{<:Real, <:Real} = (0.0, 10.0), spk_args::SpikingArgs = default_spk_args())
    #set up functions to define the neuron's differential equations
    k = neuron_constant(spk_args)

    #set up compartments for each sample
    u0 = zeros(ComplexF32, x.shape)
    #resonate in time with the input spikes
    dzdt(u, p, t) = k .* u .+ spike_current(x, t, spk_args)
    #solve the memory compartment
    prob = ODEProblem(dzdt, u0, tspan)
    sol = solve(prob, Heun(), adaptive=false, dt=spk_args.dt)

    return sol
end

"""
Convert a static phase to the complex potential of an R&F neuron
"""
function phase_to_potential(phase::Real, ts::AbstractVector, offset::Real, spk_args::SpikingArgs)
    return [phase_to_potential(phase, t, offset, spk_args) for t in ts]
end

function phase_to_potential(phase::AbstractArray, ts::AbstractVector, offset::Real, spk_args::SpikingArgs)
    return [phase_to_potential(p, t, offset, spk_args) for p in phase, t in ts]
end

function phase_to_potential(phase::Real, t::Real, offset::Real, spk_args::SpikingArgs)
    period = spk_args.t_period
    k = 1im * imag(neuron_constant(spk_args))
    potential = exp.(k .* ((t .+ offset) .- (phase - 1)/2))
    return potential
end

"""
Convert the potential of a neuron at an arbitrary point in time to its phase relative to a reference
"""
function potential_to_phase(potential::AbstractArray, t::Real, offset::Real, spk_args::SpikingArgs)
    #find the angle of a neuron representing 0 phase at the current moment in time
    current_zero = phase_to_potential(0.0, t, offset, spk_args)
    #get the arc subtended in the complex plane between that reference and our neuron potentials
    arc = angle(current_zero) .- angle.(potential) 
    #normalize by pi and shift to -1, 1
    phase = mod.((arc ./ pi .+ 1.0), 2.0) .- 1.0
end

function potential_to_phase(potential::AbstractArray, t::AbstractVector; dim::Int, spk_args::SpikingArgs, offset::Real=0.0)
    @assert size(potential, dim) == length(t) "Time dimensions must match"
    phases = [potential_to_phase(uslice, t[i], offset, spk_args) for (i, uslice) in enumerate(eachslice(potential, dims=dim))]
    phases = stack(phases)
    
    return phases
end
"""
Converts a matrix of phases into a spike train via phase encoding

phase_to_train(phases::AbstractMatrix, spk_args::SpikingArgs, repeats::Int = 1, offset::Real = 0.0)
"""
function phase_to_train(phases::AbstractArray, spk_args::SpikingArgs; repeats::Int = 1, offset::Real = 0.0)
    t_phase0 = spk_args.t_period / 2.0
    shape = phases |> size
    indices = collect(CartesianIndices(shape)) |> vec
    times = phases .* t_phase0 .+ t_phase0 |> vec

    if repeats > 1
        n_t = times |> length
        offsets = repeat(0:repeats-1, inner=n_t)
        times = repeat(times, repeats) .+ offsets
        indices = repeat(indices, repeats)
    end

    train = SpikeTrain(indices, times, shape, offset)
    return train
end

function time_to_phase(times::AbstractArray, period::Real, offset::Real)
    times = mod.((times .- offset), period) ./ period
    times = (times .- 0.5) .* 2.0
    return times
end

function train_to_phase(train::SpikeTrain, spk_args::SpikingArgs)
    if length(train.times) == 0
        return missing
    end

    #decode each spike's phase within a cycle
    relative_phase = time_to_phase(train.times, spk_args.t_period, train.offset)
    relative_time = train.times .- train.offset
    #what is the number of cycles in this train?
    n_cycles = maximum(relative_time) ÷ spk_args.t_period + 1
    #what is the cycle in which each spike occurs?
    cycle = floor.(Int, relative_time .÷ spk_args.t_period .+ 1)
    phases = [NaN .* zeros(train.shape...) for i in 1:n_cycles]

    for i in eachindex(relative_phase)
        phases[cycle[i]][train.indices[i]] = relative_phase[i]
    end

    #stack the arrays to cycle, batch, neuron
    phases = mapreduce(x->reshape(x, 1, train.shape...), vcat, phases)
    return phases

end

function train_to_phase(call::SpikingCall)
    return train_to_phase(call.train, call.spk_args)
end

function zero_nans(phases::AbstractArray)
    nans = isnan.(phases)
    phases[nans] .= 0.0
    return phases
end