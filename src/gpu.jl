include("domains.jl")

using CUDA
const N_THREADS = 256

function threads_blks(l::Int, threads::Int = N_THREADS)
    blocks = cld(l, threads)
    return threads, blocks
end

function gaussian_kernel_gpu(x::Real, t::Real, t_sigma::Real)
    i = exp(-1 * ((t - x) / (2 * t_sigma))^2)
    return i
end

function gaussian_kernel_gpu(x::Float32, t::Float32, t_sigma::Float32)
    i = exp(-1.0f0 * ((t - x) / (2.0f0 * t_sigma))^2.0f0)
    return i
end

function parallel_current(stg::SpikeTrainGPU, t::Real, spk_args::SpikingArgs)
    currents = gaussian_kernel_gpu.(stg.times, t, spk_args.t_window)
    currents = parallel_scatter_add(stg.linear_indices, currents, stg.linear_shape)
    return currents
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
    output = CUDA.zeros(T, output_size)
    threads = 256
    blocks = cld(length(indices), threads)
    
    @cuda threads=threads blocks=blocks scatter_add_kernel!(output, values, indices)
    
    return output
end