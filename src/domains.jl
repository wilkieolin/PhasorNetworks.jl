"""
    angle_to_complex(x::AbstractArray)

Convert an array of angles (in units of π radians) to complex numbers on the unit circle.
Each angle θ is mapped to exp(iπθ), resulting in complex numbers with unit magnitude.

# Arguments
- `x::AbstractArray`: Array of angles in units of π radians

# Returns
- Complex array where each element is exp(iπθ) for the corresponding angle θ
"""
function angle_to_complex(x::AbstractArray)
    k = pi_f32 * (0.0f0 + 1.0f0im)
    return exp.(k .* x)
end

"""
    complex_to_angle(x::AbstractArray)

Convert an array of complex numbers to their angles in units of π radians.

# Arguments
- `x::AbstractArray`: Array of complex numbers

# Returns
- Array of angles in units of π radians, in range [-1, 1]
"""
function complex_to_angle(x::AbstractArray)
    return angle.(x) ./ pi_f32
end

"""
    complex_to_angle(x_real::Real, x_imag::Real)

Convert real and imaginary components to an angle in units of π radians.

# Arguments
- `x_real::Real`: Real component of complex number
- `x_imag::Real`: Imaginary component of complex number

# Returns
- Angle in units of π radians, in range [-1, 1]
"""
function complex_to_angle(x_real::Real, x_imag::Real)
    return atan(x_imag, x_real) / pi_f32
end

"""
    soft_angle(x::AbstractArray{<:Complex}, r_lo::Real = 0.1f0, r_hi::Real = 0.2f0)

Calculate angles of complex numbers with a soft threshold based on magnitude.
The output is scaled by a sigmoid function of the magnitude, which smoothly 
transitions between 0 and 1 in the range [r_lo, r_hi].

# Arguments
- `x::AbstractArray{<:Complex}`: Array of complex numbers
- `r_lo::Real = 0.1f0`: Lower threshold for magnitude scaling
- `r_hi::Real = 0.2f0`: Upper threshold for magnitude scaling

# Returns
- Array of angles in units of π radians, scaled by magnitude-dependent sigmoid
"""
function soft_angle(x::AbstractArray{<:Complex}, r_lo::Real = 0.1f0, r_hi::Real = 0.2f0)
    s = similar(real.(x))

    ignore_derivatives() do
        r = abs.(x)
        m = (r .- r_lo) ./ (r_hi - r_lo)
        s .= sigmoid_fast(3.0f0 .* m .- (r_hi - r_lo))
    end

    return s .* angle.(x) / pi_f32
end


"""
    cmpx_to_realvec(u::Array{<:Complex})

Convert an array of complex numbers to a real-valued array by stacking real and imaginary parts.
The output array has an additional first dimension of size 2, containing real parts in index 1
and imaginary parts in index 2.

# Arguments
- `u::Array{<:Complex}`: Input array of complex numbers

# Returns
- Array of real numbers with shape (2, size(u)...)
"""
function cmpx_to_realvec(u::Array{<:Complex})
    nd = ndims(u)
    reals = real.(u)
    imags = imag.(u)
    mat = stack((reals, imags), dims=1)
    return mat
end

"""
    realvec_to_cmpx(u::Array{<:Real})

Convert a real-valued array with a leading dimension of size 2 back to complex numbers.
The first slice along dimension 1 becomes the real part, and the second slice becomes
the imaginary part.

# Arguments
- `u::Array{<:Real}`: Input array with shape (2, ...)

# Returns
- Complex array with shape matching input dimensions excluding the first
  
# Throws
- AssertionError if first dimension is not of size 2
"""
function realvec_to_cmpx(u::Array{<:Real})
    @assert size(u)[1] == 2 "Must have first dimension contain real and imaginary values"
    slices = eachslice(u, dims=1)
    mat = slices[1] .+ 1.0f0im .* slices[2]
    return mat
end

"""
    gaussian_kernel(x::AbstractArray, t::Real, t_sigma::Real) -> Array{Float32}
    gaussian_kernel_vec(x::AbstractVector, ts::Vector, t_sigma::Real) -> Array{Float32}
    arc_gaussian_kernel(x::AbstractVecOrMat, t::Real, t_sigma::Real) -> Array{Float32}

Family of kernel functions for computing spike-induced currents.

# Arguments
- `x`: Spike times or phase values
- `t`: Current time (or vector of times for _vec variant)
- `t_sigma`: Width parameter of the kernel

# Variants
- `gaussian_kernel`: Standard Gaussian kernel for spike times
- `gaussian_kernel_vec`: Vectorized version for multiple evaluation times
- `periodic_gaussian_kernel`: Periodic version using modulo distances

See also: [`gaussian_kernel_gpu`](@ref) for GPU implementation
"""
function gaussian_kernel(x::AbstractArray, t::Real, t_sigma::Real)
    i = exp.(-1.0f0 .* ((t .- x) / (2.0f0 .* t_sigma)).^2.0f0)
    return i
end

function gaussian_kernel_vec(x::AbstractVector, ts::Vector, t_sigma::Real)
    i = exp.(-1.0f0 .* ((ts' .- x) / (2.0f0 .* t_sigma)).^2.0f0)
    return i
end

function periodic_gaussian_kernel(x::AbstractArray, t::Real, t_sigma::Real, t_period::Real)
    # Compute the shortest distance on the ring of circumference t_period
    dt = mod.(t .- x .+ t_period/2.0f0, t_period) .- t_period/2.0f0
    i = exp.(-1.0f0 .* (dt ./ (2.0f0 .* t_sigma)).^2.0f0)
    return i
end

# Converts membrane potential to current by multiplying a sigmoidal function over
# absolute magnitude with a gaussian function over the complex angle
function potential_to_current(potential::AbstractArray{<:Complex}; spk_args::SpikingArgs)
    steepness = spk_args.steepness
    threshold = spk_args.threshold
    phase_window = spk_args.t_window / spk_args.t_period
    abs_scale = sigmoid_fast((abs.(potential) .- threshold) .* steepness)
    phase_scale = exp.(-1.0f0 .* (complex_to_angle(potential) .^ 2.0f0 / (2.0f0 * phase_window ^ 2.0f0)))
    current = abs_scale .* phase_scale
    return current
end 

###
### PHASE - SPIKE
###


"""
    phase_to_time(phases::AbstractArray; offset::Real = 0.0f0, spk_args::SpikingArgs)

Convert phases to spike times using spiking neuron parameters. Phases are interpreted 
as relative positions within a neuron's oscillation period.

# Arguments
- `phases::AbstractArray`: Array of phases in range [-1, 1]
- `offset::Real = 0.0f0`: Time offset to add to all spike times
- `spk_args::SpikingArgs`: Spiking neuron parameters including oscillation period

# Returns
- Array of spike times in absolute time units
"""
function phase_to_time(phases::AbstractArray; offset::Real = 0.0f0, spk_args::SpikingArgs)
    return phase_to_time(phases, spk_args.t_period, Float32(offset))
end

"""
    phase_to_time(phases::AbstractArray, period::Real, offset::Real = 0.0f0)

Convert phases to spike times using a specified oscillation period. This is the core
implementation that handles the actual conversion math.

# Arguments
- `phases::AbstractArray`: Array of phases in range [-1, 1]
- `period::Real`: Oscillation period length
- `offset::Real = 0.0f0`: Time offset to add to all spike times

# Returns
- Array of spike times in absolute time units, normalized to be within [0, period)

# Details 
The conversion maps phase φ to time t as:
t = (φ/2 + 0.5) * period + offset
followed by modulo operation to ensure positive times within one period.
"""
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

"""
    time_to_phase(times::AbstractArray; spk_args::SpikingArgs, offset::Real)

Convert spike times to phases using spiking neuron parameters.
This is a convenience wrapper that uses the neuron's period from spk_args.

# Arguments
- `times::AbstractArray`: Array of spike times
- `spk_args::SpikingArgs`: Spiking neuron parameters containing period information
- `offset::Real`: Time offset to subtract from all spike times

# Returns
- Array of phases in range [-1, 1]
"""
function time_to_phase(times::AbstractArray; spk_args::SpikingArgs, offset::Real)
    return time_to_phase(times, spk_args.t_period, offset)
end

"""
    time_to_phase(times::AbstractArray, period::Real, offset::Real)

Convert spike times to phases using a specified oscillation period.
This is the inverse operation of phase_to_time.

# Arguments
- `times::AbstractArray`: Array of spike times
- `period::Real`: Oscillation period length
- `offset::Real`: Time offset to subtract from all spike times

# Returns
- Array of phases in range [-1, 1]

# Details
The conversion maps time t to phase φ as:
φ = 2(((t - offset) mod period)/period - 0.5)
"""
function time_to_phase(times::AbstractArray, period::Real, offset::Real)
    times = mod.((times .- offset), period) ./ period
    phase = (times .- 0.5f0) .* 2.0f0
    return phase
end

"""
    phase_to_train(phases::AbstractArray; spk_args::SpikingArgs, repeats::Int = 1, offset::Real = 0.0f0)

Convert an array of phases to a SpikeTrain object, optionally repeating the spike pattern
multiple times.

# Arguments
- `phases::AbstractArray`: Array of phases in range [-1, 1]
- `spk_args::SpikingArgs`: Spiking neuron parameters
- `repeats::Int = 1`: Number of times to repeat the spike pattern
- `offset::Real = 0.0f0`: Time offset for the spike train

# Returns
- `SpikeTrain`: Object containing spike times and their corresponding indices

# Details
For each non-NaN phase value, a spike is generated at the corresponding time.
If repeats > 1, the spike pattern is repeated with appropriate time offsets.
The spatial structure of the input array is preserved in the SpikeTrain's shape.
"""
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

"""
    train_to_phase(call::SpikingCall)

Convert a SpikingCall's spike train to phases using its own spiking parameters.

# Arguments
- `call::SpikingCall`: Contains a spike train and associated spiking parameters

# Returns
- Array of phases in range [-1, 1]
"""
function train_to_phase(call::SpikingCall)
    return train_to_phase(call.train, spk_args=call.spk_args)
end

"""
    train_to_phase(train::SpikeTrainGPU; spk_args::SpikingArgs)

Convert a GPU-based spike train to phases. The output remains on the GPU.

# Arguments
- `train::SpikeTrainGPU`: GPU-based spike train
- `spk_args::SpikingArgs`: Spiking neuron parameters

# Returns
- GPU array of phases in range [-1, 1]
"""
function train_to_phase(train::SpikeTrainGPU; spk_args::SpikingArgs)
    train = SpikeTrain(train)
    #preserve device on output
    phases = train_to_phase(train, spk_args=spk_args, offset=train.offset) |> gdev
    return phases
end

"""
    train_to_phase(train::SpikeTrain; spk_args::SpikingArgs, offset::Real = 0.0f0)

Convert a spike train to a sequence of phase snapshots, one for each oscillation cycle.
For each cycle, creates a phase array matching the original spatial dimensions, with 
NaN values for neurons that did not spike in that cycle.

# Arguments
- `train::SpikeTrain`: The spike train to convert
- `spk_args::SpikingArgs`: Spiking neuron parameters
- `offset::Real = 0.0f0`: Additional time offset for phase calculation

# Returns
- Array of phases with shape (original_shape..., n_cycles), where n_cycles is determined
  by the temporal span of the spike train. Each slice along the last dimension represents
  the phases in one oscillation cycle.
- Returns `missing` if the spike train is empty

# Throws
- AssertionError if any spike times are negative

# Details
1. Converts each spike time to a phase within its cycle
2. Determines which cycle each spike belongs to
3. Creates a phase array for each cycle, filling with NaN by default
4. Places each spike's phase in the appropriate cycle and spatial location
5. Stacks all cycles into a single array along a new final dimension
"""
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

    #stack the arrays to batch, neuron, cycle
    phases = mapreduce(x->reshape(x, train.shape..., 1), (a,b)->cat(a, b, dims=ndims(a)), phases)
    return phases
end

"""
    phase_to_current(phases::AbstractArray; spk_args::SpikingArgs, offset::Real = 0.0f0, 
                    tspan::Tuple{<:Real, <:Real}, rng::Union{AbstractRNG, Nothing} = nothing, 
                    zeta::Real=Float32(0.0))

Convert a set of phases to a time-varying current function using a Gaussian kernel.
The current at each time point is computed based on the phase difference between the
input phases and the current time, with optional noise.

# Arguments
- `phases::AbstractArray`: Array of phases in range [-1, 1]
- `spk_args::SpikingArgs`: Spiking neuron parameters
- `offset::Real = 0.0f0`: Time offset for phase calculations
- `tspan::Tuple{<:Real, <:Real}`: Time span over which the current will be defined
- `rng::Union{AbstractRNG, Nothing} = nothing`: Random number generator for noise
- `zeta::Real=Float32(0.0)`: Noise amplitude (0 for no noise)

# Returns
- `CurrentCall`: Object containing a LocalCurrent function that computes the current
  at any given time point, along with the spiking parameters and time span

# Details
The current is computed using a Gaussian kernel centered at each phase, with width
determined by spk_args.t_window. Optional Gaussian noise can be added with amplitude zeta.
The returned function preserves the input array's shape and can be evaluated at any time
within the specified time span.
"""
function phase_to_current(phases::AbstractArray; spk_args::SpikingArgs, offset::Real = 0.0f0, tspan::Tuple{<:Real, <:Real}, rng::Union{AbstractRNG, Nothing} = nothing, zeta::Real=Float32(0.0))
    shape = size(phases)
    
    function inner(t::Real)
        # Ensure t is Float32 to avoid mixed-precision issues with Float32 weights
        times = phase_to_time(phases, spk_args.t_period, offset)
        impulses = periodic_gaussian_kernel(times, Float32(t), spk_args.t_window, spk_args.t_period)

        ignore_derivatives() do
            if zeta > 0.0f0
                noise = zeta .* randn(rng, Float32, size(impulses))
                impulses .+= noise
            end
        end

        return impulses
    end

    current = LocalCurrent(inner, shape, offset)
    call = CurrentCall(current, spk_args, tspan)

    return call
end

function (a::CurrentCall)(time::Real)
    return a.current.current_fn(time)
end

###
### PHASE - POTENTIAL
###

"""
    phase_to_potential(phase::Real, ts::AbstractVector; offset::Real=0.0f0, spk_args::SpikingArgs)

Convert a single phase value to a sequence of complex potentials at specified time points
for a Resonate-and-Fire (R&F) neuron.

# Arguments
- `phase::Real`: Phase value in range [-1, 1]
- `ts::AbstractVector`: Vector of time points at which to compute the potential
- `offset::Real=0.0f0`: Time offset for phase calculations
- `spk_args::SpikingArgs`: Spiking neuron parameters

# Returns
- Vector of complex potentials, one for each time point in ts

# Details
The R&F neuron potential follows a circular trajectory in the complex plane, with
its phase determined by the input phase and time point. The trajectory's frequency
and damping are specified in spk_args.
"""
function phase_to_potential(phase::Real, ts::AbstractVector; offset::Real=0.0f0, spk_args::SpikingArgs)
    return [phase_to_potential(phase, t, offset=offset, spk_args=spk_args) for t in ts]
end

"""
    phase_to_potential(phase::AbstractArray, ts::AbstractVector; offset::Real=0.0f0, spk_args::SpikingArgs)

Convert an array of phases to a matrix of complex potentials, computing the potential
for each phase at each specified time point.

# Arguments
- `phase::AbstractArray`: Array of phase values in range [-1, 1]
- `ts::AbstractVector`: Vector of time points
- `offset::Real=0.0f0`: Time offset for phase calculations
- `spk_args::SpikingArgs`: Spiking neuron parameters

# Returns
- Matrix of complex potentials with size (length(phase), length(ts))

# Details
Creates a matrix where each row corresponds to a phase value and each column corresponds
to a time point, containing the complex potential of an R&F neuron with that phase at
that time.
"""
function phase_to_potential(phase::AbstractArray, ts::AbstractVector; offset::Real=0.0f0, spk_args::SpikingArgs)
    return [phase_to_potential(p, t, offset=offset, spk_args=spk_args) for p in phase, t in ts]
end

"""
    phase_to_potential(phase::Real, t::Real; offset::Real=0.0f0, spk_args::SpikingArgs)

Convert a single phase value to a complex potential at a specific time point.
This is the core implementation of the phase-to-potential conversion.

# Arguments
- `phase::Real`: Phase value in range [-1, 1]
- `t::Real`: Time point at which to compute the potential
- `offset::Real=0.0f0`: Time offset for phase calculations
- `spk_args::SpikingArgs`: Spiking neuron parameters

# Returns
- Complex number representing the neuron's potential at time t

# Details
The potential is computed as:
z(t) = exp(ik * (t - offset - (phase - 1)/2 * period))
where k is the neuron's complex frequency constant (incorporating both oscillation
frequency and leakage/damping).
"""
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

function unrotate_solution(potentials::AbstractVector{<:AbstractArray}, ts::AbstractVector; offset::Real=0.0f0, spk_args::SpikingArgs)
    current_zeros = similar(potentials[1], ComplexF32, (length(ts)))

    ignore_derivatives() do
        #find the angle of a neuron representing 0 phase at the current moment in time
        current_zeros = phase_to_potential(0.0f0, ts, offset=offset, spk_args=spk_args)
    end

    potentials = current_zeros .* conj.(potentials)
    return potentials
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

"""
    solution_to_potential(func_sol::Union{ODESolution, Function}, t::Array)

Convert a solution function or ODESolution to an array of complex potentials at specified times.

# Arguments
- `func_sol::Union{ODESolution, Function}`: ODE solution object or interpolating function
- `t::Array`: Array of time points at which to evaluate the solution

# Returns
- Array of complex potentials with time as the last dimension

# Details
Takes a solution (either as a function or ODESolution object) and evaluates it at
specified time points, stacking the results along a new final dimension. This is
useful for converting continuous solution functions into discretely sampled arrays
of potentials.
"""
function solution_to_potential(func_sol::Union{ODESolution, Function}, t::Array)
    u = func_sol.(t)
    d = ndims(u[1])
    #stack the vector of solutions along a new final axis
    u = stack(u, dims = d + 1)
    return u
end

"""
    solution_to_potential(ode_sol::ODESolution)

Convert an ODESolution directly to an array of complex potentials.

# Arguments
- `ode_sol::ODESolution`: ODE solution object

# Returns
- Array of complex potentials sampled at the solution's saved time points

# Details
Simple conversion of an ODESolution to an array format, using the solution's
internally saved time points. Useful when you want to work with the exact
time points at which the ODE solver saved its results.
"""
function solution_to_potential(ode_sol::ODESolution)
    return Array(ode_sol)
end

"""
    solution_to_phase(sol::ODESolution; final_t::Bool=false, offset::Real=0.0f0, 
                     spk_args::SpikingArgs, kwargs...)

Convert an ODESolution to phases, either at all time points or just the final time.

# Arguments
- `sol::ODESolution`: ODE solution object
- `final_t::Bool=false`: If true, only compute phase at final time point
- `offset::Real=0.0f0`: Time offset for phase calculations
- `spk_args::SpikingArgs`: Spiking neuron parameters
- `kwargs...`: Additional arguments passed to potential_to_phase

# Returns
- If final_t is true: Array of phases at final time point
- If final_t is false: Array of phases at all saved time points

# Details
Converts the ODE solution to potentials and then to phases. Can either process
the entire time series or just the final state, which is useful for different
analysis scenarios.
"""
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

"""
    solution_to_phase(sol::Union{ODESolution, Function}, t::Array; offset::Real=0.0f0, 
                     spk_args::SpikingArgs, kwargs...)

Convert a solution (ODE or function) to phases at specified time points.

# Arguments
- `sol::Union{ODESolution, Function}`: ODE solution object or interpolating function
- `t::Array`: Array of time points at which to compute phases
- `offset::Real=0.0f0`: Time offset for phase calculations
- `spk_args::SpikingArgs`: Spiking neuron parameters
- `kwargs...`: Additional arguments passed to potential_to_phase

# Returns
- Array of phases at the specified time points

# Details
Evaluates the solution at given time points, converts to potentials, and then
computes the corresponding phases. This allows for flexible sampling of the
solution's phase representation at arbitrary time points.
"""
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

"""
    period_to_angfreq(t_period::Real)

Convert a time period to angular frequency.

# Arguments
- `t_period::Real`: Time period (τ)

# Returns
- Angular frequency (ω = 2π/τ) in radians per unit time

# Details
Implements the standard relationship between period and angular frequency:
ω = 2π/τ where τ is the period and ω is the angular frequency.
"""
function period_to_angfreq(t_period::Real)
    angular_frequency = 2.0f0 * pi_f32 / t_period
    return angular_frequency
end

function period_to_angfreq(t_period::AbstractArray)
    angular_frequency = 2.0f0 .* pi_f32 ./ t_period
    return angular_frequency
end

"""
    angfreq_to_period(angfreq::Real)

Convert an angular frequency to time period.

# Arguments
- `angfreq::Real`: Angular frequency (ω) in radians per unit time

# Returns
- Time period (τ = 2π/ω)

# Details
This function is auto-inverting due to the reciprocal relationship between
period and angular frequency. The implementation uses period_to_angfreq
since τ = 2π/ω = 2π/(2π/τ₀) = τ₀.
"""
function angfreq_to_period(angfreq::Real)
    #auto-inverting transform
    return period_to_angfreq(angfreq)
end

"""
    neuron_constant(leakage::Real, t_period::Real)

Calculate the complex frequency constant for a Resonate-and-Fire neuron.

# Arguments
- `leakage::Real`: Leakage/damping rate of the neuron
- `t_period::Real`: Oscillation period

# Returns
- Complex frequency constant k = λ + iω, where:
  - λ is the leakage rate
  - ω is the angular frequency (2π/period)

# Details
This complex constant determines both the frequency of oscillation and the rate
of decay in the neuron's dynamics. The real part (leakage) controls damping,
while the imaginary part sets the oscillation frequency.
"""
function neuron_constant(leakage::Real, t_period::Real)
    angular_frequency = period_to_angfreq(t_period)
    k = ComplexF32(leakage + 1.0f0im * angular_frequency)
    return k
end

function neuron_constant(leakage::AbstractArray, t_period::AbstractArray)
    angular_frequency = period_to_angfreq(t_period)
    k = ComplexF32.(leakage .+ 1.0f0im .* angular_frequency)
    return k
end

"""
    neuron_constant(spk_args::SpikingArgs)

Convenience function to calculate the neuron's complex frequency constant from SpikingArgs.

# Arguments
- `spk_args::SpikingArgs`: Spiking neuron parameters containing leakage and period

# Returns
- Complex frequency constant k = λ + iω using parameters from spk_args

# Details
This is a wrapper around neuron_constant(leakage, t_period) that extracts the
parameters from a SpikingArgs struct.
"""
function neuron_constant(spk_args::SpikingArgs)
    k = neuron_constant(spk_args.leakage, spk_args.t_period)
    return k
end

"""
    potential_to_time(u::AbstractArray, t::Real; spk_args::SpikingArgs)

Calculate expected spike times for an array of neuron potentials at a given time.

# Arguments
- `u::AbstractArray`: Array of complex potentials
- `t::Real`: Current time point
- `spk_args::SpikingArgs`: Spiking neuron parameters

# Returns
- Array of predicted spike times

# Details
For each complex potential:
1. Calculates angle in complex plane
2. Determines angular distance to π/2 (spiking threshold)
3. Converts this angle to time using neuron frequency
4. Adds to current time to get absolute spike time
5. Ensures all times are positive by adding period if needed

The spiking angle π/2 represents the phase at which a neuron generates a spike
in the Resonate-and-Fire model.
"""
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

"""
    potential_to_time(u::AbstractArray, ts::AbstractVector; spk_args::SpikingArgs, dim::Int=-1)

Calculate spike times for an array of neuron potentials over multiple time points.

# Arguments
- `u::AbstractArray`: Array of complex potentials
- `ts::AbstractVector`: Vector of time points
- `spk_args::SpikingArgs`: Spiking neuron parameters
- `dim::Int=-1`: Dimension along which time varies (defaults to last dimension)

# Returns
- Array of predicted spike times with same shape as input

# Throws
- AssertionError if size along time dimension doesn't match length of ts

# Details
Processes each time slice of the potential array separately, computing spike times
for each potential at the corresponding time point. The results are stacked back
together along the specified dimension.
"""
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

"""
    time_to_potential(spikes::AbstractArray, t::Real; spk_args::SpikingArgs)

Calculate complex potentials at a given time for neurons that spiked at specified times.

# Arguments
- `spikes::AbstractArray`: Array of spike times
- `t::Real`: Time at which to compute the potentials
- `spk_args::SpikingArgs`: Spiking neuron parameters

# Returns
- Array of complex potentials

# Details
For each spike time:
1. Computes time elapsed since/until spike
2. Converts to angular displacement using neuron frequency
3. Adjusts relative to spiking angle (π/2)
4. Converts to complex potential on unit circle

The resulting potentials represent the state each neuron would have at time t,
given their spike times, assuming ideal oscillatory behavior.
"""
function time_to_potential(spikes::AbstractArray, t::Real; spk_args::SpikingArgs)
    spiking_angle = pi_f32 / 2.0f0

    #find out given this time, what is the (normalized) potential at a given moment?
    time_from_spike = spikes .- t
    arc_from_spike = time_from_spike .* period_to_angfreq(spk_args.t_period)
    angles = -1.0f0 .* (arc_from_spike .- spiking_angle)
    potentials = angle_to_complex(angles ./ pi_f32)

    return potentials
end

"""
    time_to_potential(spikes::AbstractArray, ts::AbstractVector; spk_args::SpikingArgs, dim::Int=-1)

Calculate complex potentials over multiple time points for neurons with specified spike times.

# Arguments
- `spikes::AbstractArray`: Array of spike times
- `ts::AbstractVector`: Vector of time points at which to compute potentials
- `spk_args::SpikingArgs`: Spiking neuron parameters
- `dim::Int=-1`: Dimension along which time varies (defaults to last dimension)

# Returns
- Array of complex potentials with same shape as input

# Throws
- AssertionError if size along time dimension doesn't match length of ts

# Details
Processes each time slice of the spike times array separately, computing potentials
at each corresponding time point. The results are stacked back together along the
specified dimension, maintaining the original array structure.
"""
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

"""
    solution_to_train(sol::Union{ODESolution,Function}, tspan::Tuple{<:Real, <:Real}; 
                     spk_args::SpikingArgs, offset::Real)

Convert a continuous ODE solution or interpolating function to a discrete spike train
by sampling at cycle boundaries.

# Arguments
- `sol::Union{ODESolution,Function}`: Either an ODESolution object or a function that can
  be evaluated at arbitrary time points to get the system state
- `tspan::Tuple{<:Real, <:Real}`: Time span (t_start, t_end) over which to generate spikes
- `spk_args::SpikingArgs`: Spiking neuron parameters
- `offset::Real`: Time offset for spike timing calculations

# Returns
- `SpikeTrain`: Object containing the detected spikes and their timing information

# Details
1. Determines cycle boundary times within the specified time span
2. Samples the solution at these cycle boundaries
3. Converts the sampled potentials to spike times using threshold detection

This function provides a way to discretize a continuous dynamical solution into
a sequence of spikes, which is useful for analyzing the system's behavior in
terms of discrete events.
"""
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