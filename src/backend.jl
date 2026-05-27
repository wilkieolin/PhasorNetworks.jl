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
- `:oneapi` — Intel GPU (provided by `PhasorNetworksOneAPIExt`; requires
  `using oneAPI`)

# Returns
A device function compatible with Lux's `gpu_device()` / `cpu_device()`.

# Implementation
Dispatches through `Val(backend)` so package extensions can register
new backends by adding methods on `Val{:<name>}` rather than
overwriting the `::Symbol` entry point (which Julia's strict
precompile checks forbid). The package ships the `:cuda` and `:cpu`
methods; `PhasorNetworksOneAPIExt` adds `:oneapi`.
"""
select_device(backend::Symbol) = select_device(Val(backend))

function select_device(::Val{:cuda})
    if CUDA.functional()
        return gpu_device()
    else
        @warn "CUDA requested but not functional, falling back to CPU"
        return cpu_device()
    end
end

select_device(::Val{:cpu}) = cpu_device()

# Fallback for unknown / unloaded backends. `:oneapi` lands here unless
# `PhasorNetworksOneAPIExt` has been activated by `using oneAPI`.
function select_device(::Val{B}) where {B}
    if B === :oneapi
        error("oneAPI backend requires `using oneAPI` — load the package to activate the extension")
    end
    error("Unknown backend: :$B. Supported: :cuda, :cpu, :oneapi")
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
