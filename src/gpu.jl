include("domains.jl")

using CUDA
const N_THREADS = 256

#Kernels 

function threads_blks(l::Int, threads::Int = N_THREADS)
    blocks = cld(l, threads)
    return threads, blocks
end

function gaussian_kernel_gpu(x::Float32, t::Float32, t_sigma::Float32)
    i = exp(-1.0f0 * ((t - x) / (2.0f0 * t_sigma))^2.0f0)
    return i
end

function scatter_add_kernel!(output, values, indices)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= length(indices)
        index = indices[i]
        value = values[i]
        CUDA.@atomic output[index] += value
    end
    return nothing
end

function parallel_scatter_add(indices::CuArray{Int}, values::CuArray{T}, output_size::Int) where T
    @assert length(indices) == length(values) "Length of indices and values must match"
    
    output = CUDA.zeros(T, output_size)
    threads, blocks = threads_blks(length(indices))
    
    @cuda threads=threads blocks=blocks scatter_add_kernel!(output, values, indices)
    
    return output
end

# Domains

function potential_to_phase(potential::CuArray, ts::AbstractVector; spk_args::SpikingArgs, offset::Real=0.0, threshold::Bool=false)
    @assert size(potential)[end] == length(ts) "Time dimensions must match"
    dims = collect(1:ndims(potential))

    #find the angle of a neuron representing 0 phase at the current moment in time
    current_zeros = cu(phase_to_potential.(0.0f0, ts, offset=offset, spk_args=spk_args))

    #get the arc subtended in the complex plane between that reference and our neuron potentials
    potential = permutedims(potential, reverse(dims))
    arc = angle.(current_zeros) .- angle.(potential)
    
    #normalize by pi and shift to -1, 1
    phase = mod.((arc ./ pi .+ 1.0), 2.0) .- 1.0

    #replace silent neurons with NaN
    silent = abs.(potential) .< spk_args.threshold
    phase[silent] .= NaN
    phase = permutedims(phase, reverse(dims))
    
    return phase
end

#Spiking

function parallel_current(stg::SpikeTrainGPU, t::Float32, spk_args::SpikingArgs)
    currents = gaussian_kernel_gpu.(stg.times, t, Float32(spk_args.t_window))
    output = parallel_scatter_add(stg.linear_indices, currents, stg.linear_shape)

    return output
end

function spike_current(train::SpikeTrainGPU, t::Float32, spk_args::SpikingArgs)
    scale = spk_args.spk_scale
    current = parallel_current(train, t, spk_args)
    current = reshape(current, train.shape)
    
    return current
end

function phase_memory(x::SpikeTrainGPU; tspan::Tuple{<:Real, <:Real} = (0.0, 10.0), spk_args::SpikingArgs)
    update_fn = spk_args.update_fn

    #set up compartments for each sample
    u0 = CUDA.zeros(ComplexF32, x.shape)
    
    #resonate in time with the input spikes
    function dzdt!(du, u, p, t)
        du .= spk_args.update_fn(u) .+ spike_current(x, t, spk_args)
        return nothing
    end
    
    prob = ODEProblem(dzdt!, u0, tspan)
    return prob
    #sol = solve(prob, spk_args.solver; spk_args.solver_args...)

    return sol
end