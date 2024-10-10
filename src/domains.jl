using ChainRulesCore: ignore_derivatives
using Random: GLOBAL_RNG

struct SpikeTrain
    indices::Array{<:Union{Int, CartesianIndex},1}
    times::Array{<:Real,1}
    shape::Tuple
    offset::Real
end

function Base.show(io::IO, train::SpikeTrain)
    print(io, "Spike Train: ", train.shape, " with ", length(train.times), " spikes.")
end

function Base.size(x::SpikeTrain)
    return x.shape
end

function remap(indices::Vector{<:CartesianIndex})
    #find the dimensions of unique elements for the new array
    n_dims = length(indices[1])
    elements = [map(x -> getindex(x, i), indices) for i in 1:n_dims]
    unique_elements = unique.(elements)
    new_ranges = [1:length(e) for e in unique_elements]
    #for each dimension and element, construct the map of old to new elements
    match = (x, i) -> x => new_ranges[i][findfirst(unique_elements[i] .== x)[1]]
    mapping = [Dict(match.(elements[i], i)) for i in 1:n_dims]

    #map each old index to the new index
    new_indices = map(idx -> CartesianIndex([mapping[i][idx[i]] for i in 1:n_dims]...), indices)
    new_shape = Tuple([r[end] for r in new_ranges])
    return new_indices, new_shape
end

function Base.getindex(x::SpikeTrain, inds...)
    #find the relevant entries
    idxs = CartesianIndices(size(x))
    selected = idxs[inds...]
    matches = [idx in selected for idx in x.indices]
    #downselect
    sel_indices = x.indices[matches]
    sel_times = x.times[matches]
    #map the old indicies to new values
    new_inds, new_shape = remap(sel_indices)
    new_train = SpikeTrain(new_inds, sel_times, new_shape, x.offset)

    return new_train
end

function increment_indices(indices::Vector{CartesianIndex{N}}, dim::Int, value::Int) where N
    # Check if the dimension is valid
    if dim < 1 || dim > N
        error("Invalid dimension. Must be between 1 and $N.")
    end

    # Increment the indices along the specified dimension
    return [CartesianIndex(ntuple(i -> i == dim ? idx[i] + value : idx[i], Val(N))) for idx in indices]
end

function Base.cat(x::SpikeTrain...; dim)
    n_trains = length(x)

    inds = x[1].indices |> deepcopy
    tms = x[1].times |> deepcopy
    offset = x[1].offset
    n_dims = length(x[1].shape)

    if n_trains > 1
        i = 2

        for i in 2:n_trains
            #calculate how far the indices need to be shifted
            selected = getindex.(inds, dim)
            idx_offset = maximum(selected)
            
            #offset the indices
            new_inds = increment_indices(x[i].indices, dim, idx_offset)
            #append them to the new train
            append!(inds, new_inds)
            append!(tms, x[i].times)
            @assert x[i].offset == offset "Spike trains must have idential offset to concatentate"
        end
    end

    shape = Tuple([maximum(getindex.(inds, i)) for i in 1:n_dims])

    #create and return the new train
    new_train = SpikeTrain(inds, tms, shape, offset)
end

function Base.size(x::SpikeTrain, d::Int)
    return x.shape[d]
end

struct SpikingArgs
    leakage::Real
    t_period::Real
    t_window::Real
    threshold::Real
    solver
    solver_args::Dict
    update_fn::Function
end

function SpikingArgs(; leakage::Real = -0.2, 
                    t_period::Real = 1.0,
                    t_window::Real = 0.01,
                    threshold::Real = 0.001,
                    solver = Heun(),
                    solver_args = Dict(:dt => 0.01,
                                    :adaptive => false,
                                    :sensealg => InterpolatingAdjoint(; autojacvec=ZygoteVJP(allow_nothing=false)),
                                    :save_start => true))
                    
    return SpikingArgs(leakage,
            t_period,
            t_window,
            threshold,
            solver,
            solver_args,
            u -> neuron_constant(leakage, t_period) .* u,)
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

function Base.getindex(x::SpikingCall, inds...)
    new_train = getindex(x.train, inds...)
    new_call = SpikingCall(new_train, x.spk_args, x.t_span)
    return new_call
end

struct LocalCurrent
    current_fn::Function
    shape::Tuple
    offset::Real
end

struct CurrentCall
    current::LocalCurrent
    spk_args::SpikingArgs
    t_span::Tuple{<:Real, <:Real}
end

function angle_to_complex(x::AbstractArray)
    k = convert(ComplexF32, pi * (0.0 + 1.0im))
    return exp.(k .* x)
end

function complex_to_angle(x::AbstractArray)
    return angle.(x) ./ pi
end

function cmpx_to_realvec(u::Array{<:Complex})
    nd = ndims(u)
    reals = real.(u)
    imags = imag.(u)
    mat = stack((reals, imags), dims=1)
    return mat
end

function realvec_to_cmpx(u::Array{<:Real})
    @assert size(u)[1] == 2 "Must have first dimension contain real and imaginary values"
    slices = eachslice(u, dims=1)
    mat = slices[1] .+ 1im .* slices[2]
    return mat
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
    times = phases .* period .+ offset
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
        offsets = repeat(collect(0:repeats-1) .* spk_args.t_period, inner=n_t)
        times = repeat(times, repeats) .+ offsets
        indices = repeat(indices, repeats)
    end

    train = SpikeTrain(indices, times, shape, offset)
    return train
end

function train_to_phase(train::SpikeTrain; spk_args::SpikingArgs)
    return train_to_phase(train, spk_args)
end

function train_to_phase(train::SpikeTrain, spk_args::SpikingArgs; offset::Real = 0.0)
    if length(train.times) == 0
        return missing
    end

    @assert reduce(*, train.times .>= 0.0) "Spike train times must be positive"

    #decode each spike's phase within a cycle
    relative_phase = time_to_phase(train.times, spk_args.t_period, train.offset)
    relative_time = train.times .- (train.offset + offset)
    #what is the cycle in which each spike occurs?
    cycle = floor.(Int, relative_time .÷ spk_args.t_period)
    #re-number cycles to be positive
    cycle = cycle .+ (1 - minimum(cycle))
    #what is the number of cycles in this train?
    n_cycles = maximum(cycle)
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
function potential_to_phase(potential::AbstractArray, t::Real; offset::Real=0.0, spk_args::SpikingArgs, threshold::Bool=false)
    current_zero = ones(ComplexF32, (1))

    ignore_derivatives() do
        #find the angle of a neuron representing 0 phase at the current moment in time
        current_zero = phase_to_potential(0.0, t, offset=offset, spk_args=spk_args)
    end
    #get the arc subtended in the complex plane between that reference and our neuron potentials
    arc = angle(current_zero) .- angle.(potential) 

    #normalize by pi and shift to -1, 1
    phase = mod.((arc ./ pi .+ 1.0), 2.0) .- 1.0

    #replace silent neurons with random values
    ignore_derivatives() do
        if threshold
            silent = findall(abs.(potential) .<= spk_args.threshold)
            for i in silent
                phase[i] = NaN
            end
        end
    end

    return phase
end

function potential_to_phase(potential::AbstractArray, ts::AbstractVector; spk_args::SpikingArgs, offset::Real=0.0, threshold::Bool=false)
    @assert size(potential)[end] == length(ts) "Time dimensions must match"
    current_zeros = ones(ComplexF32, (length(ts)))
    dims = collect(1:ndims(potential))

    ignore_derivatives() do
        #find the angle of a neuron representing 0 phase at the current moment in time
        current_zeros = phase_to_potential.(0.0, ts, offset=offset, spk_args=spk_args)
    end
    #get the arc subtended in the complex plane between that reference and our neuron potentials
    potential = permutedims(potential, reverse(dims))
    arc = angle.(current_zeros) .- angle.(potential) 
    

    #normalize by pi and shift to -1, 1
    phase = mod.((arc ./ pi .+ 1.0), 2.0) .- 1.0

    #replace silent neurons with random values
    ignore_derivatives() do
        if threshold
            silent = findall(abs.(potential) .<= spk_args.threshold)
            for i in silent
                phase[i] = NaN
            end
        end
    end

    phase = permutedims(phase, reverse(dims))
    return phase
end

function solution_to_potential(func_sol::Union{ODESolution, Function}, t::Array)
    u = func_sol.(t)
    d = ndims(u[1])
    #stack the vector of solutions along a new final axis
    u = stack(u, dims = d + 1)
    return u
end

function solution_to_potential(ode_sol::ODESolution)
    return Array(ode_sol)
end

function solution_to_phase(sol::ODESolution; final_t::Bool=true, offset::Real=0.0, spk_args::SpikingArgs, kwargs...)
    #convert the ODE solution's saved points to an array
    u = solution_to_potential(sol)
    if final_t
        u = u[:,:,end]
        p = potential_to_phase(u, sol.t[end], offset=offset, spk_args=spk_args; kwargs...)
    else
        dim = ndims(u)
        #calculate the phase represented by that potential
        p = potential_to_phase(u, sol.t, offset=offset, spk_args=spk_args; kwargs...)
    end

    return p
end

function solution_to_phase(sol::Union{ODESolution, Function}, t::Array; offset::Real=0.0, spk_args::SpikingArgs, kwargs...)
    #call the solution at the provided times
    u = solution_to_potential(sol, t)
    #calculate the phase represented by that potential
    p = potential_to_phase(u, t, offset=offset, spk_args=spk_args; kwargs...)
    return p
end

###
### POTENTIAL - TIME
###

function period_to_angfreq(t_period::Real)
    angular_frequency = 2 * pi / t_period
    return angular_frequency
end

function angfreq_to_period(angfreq::Real)
    #auto-inverting transform
    return period_to_angfreq(angfreq)
end

function neuron_constant(leakage::Real, t_period::Real)
    angular_frequency = period_to_angfreq(t_period)
    k = (leakage + 1im * angular_frequency)
    return k
end

function neuron_constant(spk_args::SpikingArgs)
    k = neuron_constant(spk_args.leakage, spk_args.t_period)
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