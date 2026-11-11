module PhasorNetworksCUDAExt

# NVIDIA backend for PhasorNetworks. Activated by `using CUDA` in the
# calling environment (CUDA is a weak dependency of the package). Mirrors
# ext/PhasorNetworksOneAPIExt.jl: the only backend-specific code the
# package needs is the device selector and a memory-reclaim hook. All GPU
# kernels in src/gpu.jl are KernelAbstractions kernels dispatched on
# `AbstractGPUArray`, so they need nothing from this file.

using PhasorNetworks
using CUDA
using MLDataDevices: gpu_device, cpu_device, CUDADevice

function PhasorNetworks.select_device(::Val{:cuda})
    if !CUDA.functional()
        @warn "CUDA requested but not functional, falling back to CPU"
        return cpu_device()
    end
    dev = gpu_device()
    # MLDataDevices only auto-selects CUDA when cuDNN is loaded as well
    # (NNlib's conv kernels need it). Fall back to a bare CUDADevice so
    # non-conv models still run, but say why conv layers would fail.
    if !(dev isa CUDADevice)
        @warn "CUDA is functional but `gpu_device()` did not select it; " *
              "load cuDNN too (`using LuxCUDA`) for conv layers. Using a bare CUDADevice." maxlog=1
        dev = CUDADevice()
    end
    return dev
end

PhasorNetworks.reclaim_device!(::CUDADevice) = CUDA.reclaim()

end
