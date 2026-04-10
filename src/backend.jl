"""
    on_gpu(args...) -> Bool

Check if any of the provided arguments are GPU arrays (any backend).
Uses `AbstractGPUArray` from GPUArraysCore for backend-agnostic detection.
"""
function on_gpu(args...)
    return any(x -> x isa AbstractGPUArray, args)
end

"""
    select_device(backend::Symbol)

Return a Lux device function for the given backend symbol.

# Supported backends
- `:cuda` — CUDA GPU (requires CUDA.jl, default)
- `:cpu` — CPU fallback
- `:oneapi` — Intel GPU (requires oneAPI.jl extension)

# Returns
A device function compatible with Lux's `gpu_device()` / `cpu_device()`.
"""
function select_device(backend::Symbol)
    if backend == :cuda
        if CUDA.functional()
            return gpu_device()
        else
            @warn "CUDA requested but not functional, falling back to CPU"
            return cpu_device()
        end
    elseif backend == :cpu
        return cpu_device()
    elseif backend == :oneapi
        error("oneAPI backend requires `using oneAPI` — load the package to activate the extension")
    else
        error("Unknown backend: $backend. Supported: :cuda, :cpu, :oneapi")
    end
end

"""
    gpu_zeros(ref, T, dims...)

Allocate a zero array of type `T` and shape `dims` on the same backend as `ref`.
"""
function gpu_zeros(ref::AbstractGPUArray, T::Type, dims...)
    return KernelAbstractions.zeros(get_backend(ref), T, dims...)
end

function gpu_zeros(ref::AbstractArray, T::Type, dims...)
    return zeros(T, dims...)
end
