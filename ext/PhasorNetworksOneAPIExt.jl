module PhasorNetworksOneAPIExt

using PhasorNetworks
using oneAPI

# Override select_device to support :oneapi backend
function PhasorNetworks.select_device(backend::Symbol)
    if backend == :oneapi
        return oneAPI.oneAPIDevice()
    elseif backend == :cuda
        if PhasorNetworks.CUDA.functional()
            return PhasorNetworks.gpu_device()
        else
            @warn "CUDA requested but not functional, falling back to CPU"
            return PhasorNetworks.cpu_device()
        end
    elseif backend == :cpu
        return PhasorNetworks.cpu_device()
    else
        error("Unknown backend: $backend. Supported: :cuda, :cpu, :oneapi")
    end
end

end
