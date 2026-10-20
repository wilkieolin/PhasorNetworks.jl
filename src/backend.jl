"""
    on_gpu(args...) -> Bool

Check if any of the provided arguments are GPU arrays (any backend).
Uses `AbstractGPUArray` from GPUArraysCore for backend-agnostic detection.
"""
function on_gpu(args...)
    return any(x -> x isa AbstractGPUArray || x isa SpikeTrainGPU, args)
end

"""
    to_device_of(ref, x)

Move `x` onto the device that `ref` lives on (`MLDataDevices.get_device`).
For a host `ref` this is the identity. Use it to lift host-built constants
(offsets, noise, index tables) onto GPU data without naming a backend.
"""
to_device_of(ref, x) = get_device(ref)(x)

"""
    reclaim_device!(device)

Release cached device memory back to the driver, if the backend supports
it. The generic method is a no-op; `PhasorNetworksCUDAExt` adds
`CUDA.reclaim()` for `CUDADevice`.
"""
reclaim_device!(::Any) = nothing

# Device for a new SpikeTrainGPU: follow the data when either array is
# already on a GPU, otherwise the auto-selected GPU device.
function _spike_train_device(indices, times)
    dev = times isa AbstractGPUArray ? get_device(times) :
          indices isa AbstractGPUArray ? get_device(indices) :
          gpu_device()
    dev isa AbstractGPUDevice ||
        error("SpikeTrainGPU requires a functional GPU device; none is loaded. " *
              "Load a backend (`using LuxCUDA` / `using oneAPI`) or pass `device=`.")
    return dev
end

"""
    select_device(backend::Symbol)

Return a Lux device function for the given backend symbol.

# Supported backends
- `:auto` — first loaded and functional GPU backend (via
  `MLDataDevices.gpu_device()`), else CPU. Default for `Args`.
- `:cpu` — CPU
- `:cuda` — NVIDIA GPU (provided by `PhasorNetworksCUDAExt`; requires
  `using CUDA`, plus cuDNN for conv layers — `using LuxCUDA` loads both)
- `:oneapi` — Intel GPU (provided by `PhasorNetworksOneAPIExt`; requires
  `using oneAPI`)

# Returns
A device function compatible with Lux's `gpu_device()` / `cpu_device()`.

# Implementation
Dispatches through `Val(backend)` so package extensions can register
new backends by adding methods on `Val{:<name>}` rather than
overwriting the `::Symbol` entry point (which Julia's strict
precompile checks forbid). The package ships the `:auto` and `:cpu`
methods; `PhasorNetworksCUDAExt` adds `:cuda` and
`PhasorNetworksOneAPIExt` adds `:oneapi`.
"""
select_device(backend::Symbol) = select_device(Val(backend))

select_device(::Val{:auto}) = gpu_device()

select_device(::Val{:cpu}) = cpu_device()

# Trigger packages for the vendor backends, for the error message below.
const _BACKEND_TRIGGERS = (cuda = "CUDA (plus cuDNN for conv layers: `using LuxCUDA`)",
                           oneapi = "oneAPI")

# Fallback for unknown / unloaded backends. `:cuda` / `:oneapi` land here
# unless the matching extension has been activated by loading its package.
function select_device(::Val{B}) where {B}
    if haskey(_BACKEND_TRIGGERS, B)
        error("Backend :$B requested but its extension is not active — " *
              "`using $(_BACKEND_TRIGGERS[B])` first")
    end
    error("Unknown backend: :$B. Supported: :auto, :cpu, :cuda, :oneapi")
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

"""
    _bmul_matched(A, B)

`batched_mul(A, B)` with the real operand promoted to the other's complex
type, so the product is a same-type gemm. NNlib sends mixed element types
(ComplexF32 × Float32) to `batched_mul_generic!`, a per-batch `mul!` over
views that runs GPUArrays' generic Julia matmul kernel instead of the
vendor BLAS; on oneAPI that kernel read out of bounds (PhasorLCA at L 500,
B ≥ 4, Aurora 2026-10-20). Same-type inputs pass through unchanged.
"""
function _bmul_matched(A::AbstractArray, B::AbstractArray)
    T = promote_type(eltype(A), eltype(B))
    return batched_mul(_as_eltype(T, A), _as_eltype(T, B))
end
_as_eltype(::Type{T}, X::AbstractArray{T}) where {T} = X
_as_eltype(::Type{T}, X::AbstractArray) where {T} = T.(X)
