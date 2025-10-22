include("imports.jl")

pi_f32 = convert(Float32, pi)

@kwdef mutable struct Args #lr is intentionally Float64 for Optimisers compatibility with some AD backends if not careful
    lr::Float64 = 0.0003       ## learning rate
    batchsize::Int = 128    ## batch size
    epochs::Int = 10        ## number of epochs
    use_cuda::Bool = true   ## use gpu (if cuda available)
    rng::Xoshiro = Xoshiro(42) ## global rng
end

struct SpikeTrain{N}
    indices::Array{<:Union{Int, CartesianIndex},1}
    times::Array{Float32,1}
    shape::Tuple
    offset::Float32

    function SpikeTrain(indices::AbstractArray,
        times::AbstractArray,
        shape::Tuple,
        offset::Real)
        times_f32 = eltype(times) == Float32 ? times : Float32.(times)
        return new{length(shape)}(indices,
                            times_f32,
                            shape,
                            Float32(offset))
    end
end


struct SpikeTrainGPU{N}
    indices::CuArray
    linear_indices::CuArray
    times::CuArray{<:Real}
    shape::Tuple
    linear_shape::Int
    offset::Float32

    function SpikeTrainGPU(indices::AbstractArray,
                            times::AbstractArray,
                            shape::Tuple,
                            offset::Real)
        times_f32 = eltype(times) == Float32 ? times : Float32.(times)
        cu_times_f32 = cu(times_f32)
        return new{length(shape)}(cu(indices),
                CuArray(LinearIndices(shape)[indices]),
                cu_times_f32,
                shape,
                reduce(*, shape),
                Float32(offset))
    end
end

function SpikeTrainGPU(st::SpikeTrain)
    return SpikeTrainGPU(st.indices,
                        st.times,
                        st.shape,
                        st.offset)
end

function SpikeTrain(stg::SpikeTrainGPU)
    st = SpikeTrain(Array(stg.indices),
                    Array(stg.times),
                    stg.shape,
                    stg.offset)
    return st
end

function Base.convert(::Type{SpikeTrain}, stg::SpikeTrainGPU)
    return SpikeTrain(stg)
end

function Base.convert(::Type{SpikeTrainGPU}, st::SpikeTrain)
    return SpikeTrainGPU(st)
end

SpikingTypes = Union{SpikeTrain, SpikeTrainGPU}
LuxParams = Union{NamedTuple, ComponentArray, ComponentVector, SubArray}

function Base.show(io::IO, train::SpikeTrain)
    print(io, "Spike Train: ", train.shape, " with ", length(train.times), " spikes.")
end

function Base.size(x::Union{SpikeTrain, SpikeTrainGPU})
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

function get_time(st::SpikeTrainGPU, times::Tuple{<:Real, <:Real})
    return get_time(SpikeTrain(st), times)
end

function get_time(st::SpikeTrain, times::Tuple{<:Real, <:Real})
    valid = findall((st.times .> times[1]) .* (st.times .< times[2]))
    st = SpikeTrain(st.indices[valid], st.times[valid], st.shape, st.offset)
    return st
end

struct SpikingArgs
    leakage::Float32
    t_period::Float32
    t_window::Float32
    spk_scale::Float32
    threshold::Float32
    spike_kernel::Union{Symbol, Function}
    solver # Solver type can be kept generic
    solver_args::Dict
    update_fn::Function
end

function SpikingArgs(; leakage::Real = -0.2f0, 
                    t_period::Real = 1.0f0,
                    t_window::Real = 0.01f0,
                    spk_scale::Real = 1.0f0,
                    threshold::Real = 0.001f0,
                    spike_kernel = :gaussian,
                    solver = Heun(),
                    solver_args = Dict(:dt => 0.01f0,
                                    :adaptive => false,
                                    :sensealg => InterpolatingAdjoint(; autojacvec=ZygoteVJP(allow_nothing=false)),
                                    :save_start => true))
                    
    return SpikingArgs(Float32(leakage),
            Float32(t_period),
            Float32(t_window),
            Float32(spk_scale),
            Float32(threshold),
            spike_kernel,
            solver,
            solver_args,
            u -> neuron_constant(Float32(leakage), Float32(t_period)) .* u,)
end

function SpikingArgs_NN(; leakage::Real = -0.2f0, # Changed default to Float32 literal
    t_period::Real = 1.0f0,
    t_window::Real = 0.01f0, # Ensure all Real args are consistently handled
    spk_scale::Real = 1.0f0,
    threshold::Real = 0.001f0,
    spike_kernel = :gaussian,
    solver = Heun(),
    solver_args = Dict(:dt => 0.01f0,
                    :adaptive => false,
                    :sensealg => InterpolatingAdjoint(; autojacvec=ZygoteVJP(allow_nothing=false)),
                    :save_start => true),
    update_fn::Function)

    return SpikingArgs(Float32(leakage),
            Float32(t_period),
            Float32(t_window),
            Float32(spk_scale),
            Float32(threshold),
            spike_kernel,
            solver,
            solver_args,
            update_fn)
end

function Base.show(io::IO, spk_args::SpikingArgs)
    print(io, "Neuron parameters: Period ", spk_args.t_period, " (s)\n")
    print(io, "Current kernel duration: ", spk_args.t_window, " (s)\n")
    print(io, "Threshold: ", spk_args.threshold, " (V)\n")
end


struct SpikingCall
    train::SpikingTypes
    spk_args::SpikingArgs
    t_span::Tuple{Float32, Float32}
end

function Base.size(x::SpikingCall)
    return x.train.shape
end

function Base.getindex(x::SpikingCall, inds...)
    new_train = getindex(x.train, inds...)
    new_call = SpikingCall(new_train, x.spk_args, x.t_span)
    return new_call
end

struct LocalCurrent
    current_fn::Function
    shape::Tuple
    offset::Float32
end

function LocalCurrent(current_fn::Function, shape::Tuple, offset::Real)
    return LocalCurrent(current_fn, shape, Float32(offset))
end

function LocalCurrent(current_fn::Function, shape::Tuple) # Default offset
    return LocalCurrent(current_fn, shape, 0.0f0)
end

function LocalCurrent(st::SpikingTypes, spk_args::SpikingArgs)
    return LocalCurrent(t -> spike_current(st, t, spk_args),
                                        st.shape,
                                        st.offset)
end

function Base.size(x::LocalCurrent)
    return x.shape
end

struct CurrentCall
    current::LocalCurrent
    spk_args::SpikingArgs
    t_span::Tuple{<:Real, <:Real}
end

function CurrentCall(sc::SpikingCall)
    return CurrentCall(LocalCurrent(sc.train, sc.spk_args), 
                        sc.spk_args,
                        sc.t_span)
end

function Base.size(x::CurrentCall)
    return x.current.shape
end

PhaseInput = Union{SpikeTrain, SpikingCall, LocalCurrent, CurrentCall, AbstractArray, ODESolution}

function angle_to_complex(x::AbstractArray)
    k = pi_f32 * (0.0f0 + 1.0f0im)
    return exp.(k .* x)
end

function complex_to_angle(x::AbstractArray)
    return angle.(x) ./ pi_f32
end

function complex_to_angle(x_real::Real, x_imag::Real)
    return atan(x_imag, x_real) / pi_f32
end

function soft_angle(x::AbstractArray{<:Complex}, r_lo::Real = 0.1f0, r_hi::Real = 0.2f0)
    s = similar(real.(x))

    ignore_derivatives() do
        r = abs.(x)
        m = (r .- r_lo) ./ (r_hi - r_lo)
        s .= sigmoid_fast(3.0f0 .* m .- (r_hi - r_lo))
    end

    return s .* angle.(x) / pi_f32
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
    mat = slices[1] .+ 1.0f0im .* slices[2]
    return mat
end

###
### PHASE - SPIKE
###

"""
Converts a matrix of phases into a spike train via phase encoding

phase_to_train(phases::AbstractMatrix, spk_args::SpikingArgs, repeats::Int = 1, offset::Real = 0.0)
"""
function phase_to_time(phases::AbstractArray; offset::Real = 0.0f0, spk_args::SpikingArgs)
    return phase_to_time(phases, spk_args.t_period, Float32(offset))
end

function phase_to_time(phases::AbstractArray, period::Real, offset::Real = 0.0f0)
    phases = eltype(phases) == Float32 ? phases : Float32.(phases)
    period = Float32(period)
    offset = Float32(offset)
    #convert a potential to the time at which the voltage is maximum - 90* behind phase
    phases = (phases ./ 2.0f0) .+ 0.5f0
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
    phase = (times .- 0.5f0) .* 2.0f0
    return phase
end

function phase_to_train(phases::AbstractArray; spk_args::SpikingArgs, repeats::Int = 1, offset::Real = 0.0f0)
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

function train_to_phase(call::SpikingCall)
    return train_to_phase(call.train, spk_args=call.spk_args)
end

function train_to_phase(train::SpikeTrainGPU; spk_args::SpikingArgs)
    train = SpikeTrain(train)
    return train_to_phase(train, spk_args=spk_args, offset=train.offset)
end

function train_to_phase(train::SpikeTrain; spk_args::SpikingArgs, offset::Real = 0.0f0)
    if length(train.times) == 0
        return missing
    end

    @assert reduce(*, train.times .>= 0.0f0) "Spike train times must be positive"

    #decode each spike's phase within a cycle
    relative_phase = time_to_phase(train.times, spk_args.t_period, train.offset)
    relative_time = train.times .- (train.offset + offset)
    #what is the cycle in which each spike occurs?
    cycle = floor.(Int, relative_time .÷ spk_args.t_period)
    #re-number cycles to be positive
    cycle = cycle .+ (1 - minimum(cycle))
    #what is the number of cycles in this train?
    n_cycles = maximum(cycle)
    phases = [fill(Float32(NaN), train.shape...) for i in 1:n_cycles]

    for i in eachindex(relative_phase)
        phases[cycle[i]][train.indices[i]] = relative_phase[i]
    end

    #stack the arrays to cycle, batch, neuron
    phases = mapreduce(x->reshape(x, 1, train.shape...), vcat, phases)
    return phases
end

function phase_to_current(phases::AbstractArray; spk_args::SpikingArgs, offset::Real = 0.0f0, tspan::Tuple{<:Real, <:Real}, rng::Union{AbstractRNG, Nothing} = nothing, zeta::Real=Float32(0.0))
    shape = size(phases)
    
    function inner(t::Real)
        output = similar(phases)

        ignore_derivatives() do
            p = time_to_phase([t,], spk_args = spk_args, offset = offset)[1]
            current_kernel = x -> arc_gaussian_kernel(x, p, spk_args.t_window * period_to_angfreq(spk_args.t_period))
            impulses = current_kernel(phases)

            if zeta > 0.0f0
                noise = zeta .* randn(rng, Float32, size(impulses))
                impulses .+= noise
            end
            
            output .= impulses
        end

        return output
    end

    current = LocalCurrent(inner, shape, offset)
    call = CurrentCall(current, spk_args, tspan)

    return call
end

###
### PHASE - POTENTIAL
###

"""
Convert a static phase to the complex potential of an R&F neuron
"""
function phase_to_potential(phase::Real, ts::AbstractVector; offset::Real=0.0f0, spk_args::SpikingArgs)
    return [phase_to_potential(phase, t, offset=offset, spk_args=spk_args) for t in ts]
end

function phase_to_potential(phase::AbstractArray, ts::AbstractVector; offset::Real=0.0f0, spk_args::SpikingArgs)
    return [phase_to_potential(p, t, offset=offset, spk_args=spk_args) for p in phase, t in ts]
end

function phase_to_potential(phase::Real, t::Real; offset::Real=0.0f0, spk_args::SpikingArgs)
    period = Float32(spk_args.t_period)
    k = ComplexF32(1.0f0im * imag(neuron_constant(spk_args)))
    potential = ComplexF32(exp.(k .* ((t .- offset) .- (phase - 1.0f0)/2.0f0 * period)))
    return potential
end

"""
Convert the potential of a neuron at an arbitrary point in time to its phase relative to a reference
"""
function potential_to_phase(potential::AbstractArray, t::Real; offset::Real=0.f0, spk_args::SpikingArgs, threshold::Bool=false)
    current_zero = similar(potential, ComplexF32, (1))

    ignore_derivatives() do
        #find the angle of a neuron representing 0 phase at the current moment in time
        current_zero = phase_to_potential(0.0f0, t, offset=offset, spk_args=spk_args)
    end
    #get the arc subtended in the complex plane between that reference and our neuron potentials
    arc = angle(current_zero) .- angle.(potential) 

    #normalize by pi and shift to -1, 1
    phase = mod.((arc ./ pi_f32 .+ 1.0f0), 2.0f0) .- 1.0f0

    #replace silent neurons with NaN values
    ignore_derivatives() do
        if threshold
            silent = findall(abs.(potential) .<= spk_args.threshold)
            for i in silent
                phase[i] = Float32(NaN)
            end
        end
    end

    return phase
end

"""
    potential_to_phase(ut::Tuple{<:AbstractVector{<:AbstractArray}, <:AbstractVector}; spk_args::SpikingArgs, kwargs...)

Decodes the phase from a tuple of potentials and times, as produced by an `ODESolution`.
This is a convenience function for handling the output of ODE solvers like `(sol.u, sol.t)`.
"""
function potential_to_phase(ut::Tuple{<:AbstractVector{<:AbstractArray}, <:AbstractVector}; spk_args::SpikingArgs, kwargs...)
    u_vec = ut[1]
    ts = ut[2]

    # Stack the vector of arrays into a single multi-dimensional array, adding a time dimension.
    potential = stack(u_vec, dims=ndims(u_vec[1]) + 1)

    return potential_to_phase(potential, ts; spk_args=spk_args, kwargs...)
end

function potential_to_phase(potential::AbstractArray, ts::AbstractVector; spk_args::SpikingArgs, offset::Real=0.0f0, threshold::Bool=false)
    @assert size(potential)[end] == length(ts) "Time dimensions must match"
    current_zeros = similar(potential, ComplexF32, (length(ts)))
    dims = collect(1:ndims(potential))

    ignore_derivatives() do
        #find the angle of a neuron representing 0 phase at the current moment in time
        current_zeros = phase_to_potential.(0.0f0, ts, offset=offset, spk_args=spk_args)
    end
    #get the arc subtended in the complex plane between that reference and our neuron potentials
    potential = permutedims(potential, reverse(dims))
    arc = angle.(current_zeros) .- angle.(potential) 
    
    #normalize by pi and shift to -1, 1
    phase = mod.((arc ./ pi_f32 .+ 1.0f0), 2.0f0) .- 1.0f0

    #replace silent neurons with random values
    ignore_derivatives() do
        if threshold
            silent = findall(abs.(potential) .<= spk_args.threshold)
            for i in silent
                phase[i] = Float32(NaN)
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

function solution_to_phase(sol::ODESolution; final_t::Bool=false, offset::Real=0.0f0, spk_args::SpikingArgs, kwargs...)
    #convert the ODE solution's saved points to an array
    u = solution_to_potential(sol)
    if final_t
        u = u[:,:,end]
        p = potential_to_phase(u, sol.t[end], offset=offset, spk_args=spk_args; kwargs...)
    else
        #calculate the phase represented by that potential
        p = potential_to_phase(u, sol.t, offset=offset, spk_args=spk_args; kwargs...)
    end

    return p
end

function solution_to_phase(sol::Union{ODESolution, Function}, t::Array; offset::Real=0.0f0, spk_args::SpikingArgs, kwargs...)
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
    angular_frequency = 2.0f0 * pi_f32 / t_period
    return angular_frequency
end

function angfreq_to_period(angfreq::Real)
    #auto-inverting transform
    return period_to_angfreq(angfreq)
end

function neuron_constant(leakage::Real, t_period::Real)
    angular_frequency = period_to_angfreq(t_period)
    k = ComplexF32(leakage + 1.0f0im * angular_frequency)
    return k
end

function neuron_constant(spk_args::SpikingArgs)
    k = neuron_constant(spk_args.leakage, spk_args.t_period)
    return k
end

function potential_to_time(u::AbstractArray, t::Real; spk_args::SpikingArgs)
    spiking_angle = pi_f32 / 2.0f0

    #find out given this potential, how much time until the neuron spikes (ideally)
    angles = mod.(-1.0f0 .* angle.(u), 2.0f0*pi_f32) #flip angles and move onto the positive domain
    arc_to_spike = spiking_angle .+ angles
    time_to_spike = arc_to_spike ./ period_to_angfreq(spk_args.t_period)
    spikes = t .+ time_to_spike
    
    #make all times positive
    spikes[findall(x -> x < 0.0f0, spikes)] .+= spk_args.t_period
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
    spiking_angle = pi_f32 / 2.0f0

    #find out given this time, what is the (normalized) potential at a given moment?
    time_from_spike = spikes .- t
    arc_from_spike = time_from_spike .* period_to_angfreq(spk_args.t_period)
    angles = -1.0f0 .* (arc_from_spike .- spiking_angle)
    potentials = angle_to_complex(angles ./ pi_f32)

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
    train = solution_to_train(u, cycles, spk_args=spk_args, offset=offset)
    return train
end

"""
This implementation takes a full solution (represented by a vector of arrays) and finds the spikes from it.
"""
function solution_to_train(u::AbstractVector{<:AbstractArray}, t::AbstractVector{<:Real}, tspan::Tuple{<:Real, <:Real}; spk_args::SpikingArgs, offset::Real)
    #determine the ending time of each cycle
    cycles = generate_cycles(tspan, spk_args, offset)
    inds = [argmin(abs.(t .- t_c)) for t_c in cycles]

    #sample the potential at the end of each cycle
    u = u[inds] |> stack
    ts = t[inds]
    train = solution_to_train(u, ts, spk_args=spk_args, offset=offset)
    return train
end

"""
This implementation takes a single matrix at pre-selected, representative times and converts each temporal slice
to spikes.
"""
function solution_to_train(u::AbstractArray{<:Complex}, times::AbstractVector{<:Real}; spk_args::SpikingArgs, offset::Real)
    #determine the ending time of each cycle
    spiking = abs.(u) .> spk_args.threshold
    
    #convert the phase represented by that potential to a spike time
    tms = potential_to_time(u, times, spk_args = spk_args)
    
    if on_gpu(tms)
        gpu = true
        spiking = spiking |> cdev
        tms = tms |> cdev
    else
        gpu = false
    end

    #return only the times where the neuron is spiking
    cut_index = i -> CartesianIndex(Tuple(i)[1:end-1])
    inds = findall(spiking)
    tms = tms[inds]
    inds = cut_index.(inds)
    train = SpikeTrain(inds, tms, size(u)[1:end-1], offset + spiking_offset(spk_args))

    if gpu
        train = SpikeTrainGPU(train)
    end

    return train
end