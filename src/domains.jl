struct SpikeTrain
    indices::Array{<:Union{Int, CartesianIndex},1}
    times::Array{<:Real,1}
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
    solver
    solver_args::Dict
end

function SpikingArgs(; leakage::Real = -0.2, 
                    t_period::Real = 1.0,
                    t_window::Real = 0.01,
                    threshold::Real = 0.001,
                    solver = Heun(),
                    solver_args = Dict(:dt => 0.01,
                                    :adaptive => false,))
    return SpikingArgs(leakage, t_period, t_window, threshold, solver, solver_args)
end

function Base.show(io::IO, spk_args::SpikingArgs)
    print(io, "Neuron parameters: Period ", spk_args.t_period, " (s)\n")
    print(io, "Current kernel duration: ", spk_args.t_window, " (s)\n")
    print(io, "Threshold: ", spk_args.threshold, " (V)\n")
end

struct LocalCurrent
    current_fn::Function
    shape::Tuple
    offset::Real
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

function arc_error(phase::Real)
    return sin(pi * phase)
end

function arc_error(phases::AbstractArray)
    return arc_error.(phases)
end

function angle_to_complex(x::AbstractArray)
    k = convert(ComplexF32, pi * (0.0 + 1.0im))
    return exp.(k .* x)
end

function complex_to_angle(x::AbstractArray)
    return angle.(x) ./ pi
end

###
### PHASE - SPIKE
###

"""
Converts a matrix of phases into a spike train via phase encoding

phase_to_train(phases::AbstractMatrix, spk_args::SpikingArgs, repeats::Int = 1, offset::Real = 0.0)
"""
function phase_to_time(phases::AbstractArray; offset::Real = 0.0, spk_args::SpikingArgs)
    return phase_to_time(phases, spk_args.t_period, offset)
end

function phase_to_time(phases::AbstractArray, period::Real, offset::Real = 0.0)
    #convert a potential to the time at which the voltage is maximum - 90* behind phase
    phases = (phases ./ 2.0) .+ 0.5
    times = phases .* period
    #add any additional offset
    times .+= offset
    #make all times positive
    times = mod.(times, period)
   
    return times
end

function time_to_phase(times::AbstractArray; spk_args::SpikingArgs, offset::Real)
    return time_to_phase(times, spk_args.t_period, offset)
end

function time_to_phase(times::AbstractArray, period::Real, offset::Real)
    times = mod.((times .- offset), period) ./ period
    phase = (times .- 0.5) .* 2.0
    return phase
end

function phase_to_train(phases::AbstractArray; spk_args::SpikingArgs, repeats::Int = 1, offset::Real = 0.0)
    shape = phases |> size
    indices = collect(CartesianIndices(shape)) |> vec
    times = phase_to_time(phases, spk_args=spk_args, offset=offset) |> vec

    if repeats > 1
        n_t = times |> length
        offsets = repeat(0:repeats-1, inner=n_t)
        times = repeat(times, repeats) .+ offsets
        indices = repeat(indices, repeats)
    end

    train = SpikeTrain(indices, times, shape, offset)
    return train
end

function train_to_phase(train::SpikeTrain; spk_args::SpikingArgs)
    return train_to_phase(train, spk_args)
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

###
### PHASE - POTENTIAL
###

"""
Convert a static phase to the complex potential of an R&F neuron
"""
function phase_to_potential(phase::Real, ts::AbstractVector; offset::Real=0.0, spk_args::SpikingArgs)
    return [phase_to_potential(phase, t, offset=offset, spk_args=spk_args) for t in ts]
end

function phase_to_potential(phase::AbstractArray, ts::AbstractVector; offset::Real=0.0, spk_args::SpikingArgs)
    return [phase_to_potential(p, t, offset=offset, spk_args=spk_args) for p in phase, t in ts]
end

function phase_to_potential(phase::Real, t::Real; offset::Real=0.0, spk_args::SpikingArgs)
    period = spk_args.t_period
    k = 1im * imag(neuron_constant(spk_args))
    potential = exp.(k .* ((t .- offset) .- (phase - 1)/2))
    return potential
end

"""
Convert the potential of a neuron at an arbitrary point in time to its phase relative to a reference
"""
function potential_to_phase(potential::AbstractArray, t::Real; offset::Real=0.0, spk_args::SpikingArgs)
    #find the angle of a neuron representing 0 phase at the current moment in time
    current_zero = phase_to_potential(0.0, t, offset=offset, spk_args=spk_args)
    #get the arc subtended in the complex plane between that reference and our neuron potentials
    arc = angle(current_zero) .- angle.(potential) 
    #normalize by pi and shift to -1, 1
    phase = mod.((arc ./ pi .+ 1.0), 2.0) .- 1.0
end

function potential_to_phase(potential::AbstractArray, t::AbstractVector; dim::Int, spk_args::SpikingArgs, offset::Real=0.0)
    @assert size(potential, dim) == length(t) "Time dimensions must match"
    phases = [potential_to_phase(uslice, t[i], offset=offset, spk_args=spk_args) for (i, uslice) in enumerate(eachslice(potential, dims=dim))]
    phases = stack(phases)
    
    return phases
end

function solution_to_potential(func_sol::Union{ODESolution, Function}, t::Array)
    u = func_sol.(t)
    d = ndims(u[1])
    #stack the vector of solutions along a new final axis
    u = stack(u, dims = d + 1)
    return u
end

function solution_to_phase(sol::Union{ODESolution, Function}, t::Array; offset::Real=0.0, spk_args::SpikingArgs)
    #call the solution at the provided times
    u = solution_to_potential(sol, t)
    dim = ndims(u)
    #calculate the phase represented by that potential
    p = potential_to_phase(u, t, dim=dim, offset=offset, spk_args=spk_args)
    return p
end

###
### POTENTIAL - TIME
###

function period_to_angfreq(t_period::Real)
    angular_frequency = 2 * pi / t_period
    return angular_frequency
end

function neuron_constant(spk_args::SpikingArgs)
    angular_frequency = period_to_angfreq(spk_args.t_period)
    k = (spk_args.leakage + 1im * angular_frequency)
    return k
end

function potential_to_time(u::AbstractArray, t::Real; spk_args::SpikingArgs)
    spiking_angle = pi / 2

    #find out given this potential, how much time until the neuron spikes (ideally)
    angles = mod.(-1 .* angle.(u), 2*pi) #flip angles and move onto the positive domain
    arc_to_spike = spiking_angle .+ angles
    time_to_spike = arc_to_spike ./ period_to_angfreq(spk_args.t_period)
    spikes = t .+ time_to_spike
    
    #make all times positive
    spikes[findall(x -> x < 0.0, spikes)] .+= spk_args.t_period
    return spikes
end

function potential_to_time(u::AbstractArray, ts::AbstractVector; spk_args::SpikingArgs, dim::Int=-1)
    if dim == -1
        dim = ndims(u)
    end
    @assert size(u, dim) == length(ts) "Time dimension of array must match list of times"

    u_slices = eachslice(u, dims=dim)
    spikes = [potential_to_time(x[1], x[2], spk_args=spk_args) for x in zip(u_slices, ts)]
    spikes = stack(spikes, dims=dim)
    return spikes
end

function time_to_potential(spikes::AbstractArray, t::Real; spk_args::SpikingArgs)
    spiking_angle = pi / 2

    #find out given this time, what is the (normalized) potential at a given moment?
    time_from_spike = spikes .- t
    arc_from_spike = time_from_spike .* period_to_angfreq(spk_args.t_period)
    angles = -1 .* (arc_from_spike .- spiking_angle)
    potentials = angle_to_complex(angles ./ pi)

    return potentials
end

function time_to_potential(spikes::AbstractArray, ts::AbstractVector; spk_args::SpikingArgs, dim::Int=-1)
    if dim == -1
        dim = ndims(spikes)
    end
    @assert size(spikes, dim) == length(ts) "Time dimension of array must match list of times"

    t_slices = eachslice(spikes, dims=dim)
    potential = [time_to_potential(x[1], x[2], spk_args=spk_args) for x in zip(t_slices, ts)]
    potential = stack(potential, dims=dim)
    return potential
end

function solution_to_train(sol::Union{ODESolution,Function}, tspan::Tuple{<:Real, <:Real}; spk_args::SpikingArgs, offset::Real)
    #determine the ending time of each cycle
    cycles = generate_cycles(tspan, spk_args, offset)

    #sample the potential at the end of each cycle
    u = solution_to_potential(sol, cycles)
    spiking = abs.(u) .> spk_args.threshold
    
    #convert the phase represented by that potential to a spike time
    tms = potential_to_time(u, cycles, spk_args = spk_args)

    #return only the times where the neuron is spiking
    cut_index = i -> CartesianIndex(Tuple(i)[1:end-1])
    inds = findall(spiking)
    tms = tms[inds]
    inds = cut_index.(inds)
    train = SpikeTrain(inds, tms, size(u)[1:end-1], offset + spiking_offset(spk_args))

    return train
end