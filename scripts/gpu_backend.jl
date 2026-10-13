# scripts/gpu_backend.jl
#
# Platform-resolved GPU backend loading for scripts and the test launcher.
#
#     include(joinpath(@__DIR__, "gpu_backend.jl"))
#     backend, dev = load_gpu_backend()            # :auto → detect platform
#     backend, dev = load_gpu_backend(backend=:cpu)
#
# `load_gpu_backend` loads the vendor package for the selected backend
# (`LuxCUDA` for :cuda — CUDA plus cuDNN; `oneAPI` for :oneapi), which
# activates the matching PhasorNetworks extension, and returns the
# backend symbol and the MLDataDevices device to move data with.
#
# The vendor package must be resolvable from the active environment, so
# run GPU scripts from the per-backend environments:
#
#     julia --project=envs/cuda   scripts/foo.jl
#     julia --project=envs/oneapi scripts/foo.jl
#
# Neither this file nor `PhasorNetworks` names a vendor package, so the
# same script runs on the DGX (CUDA) and on Aurora (oneAPI) unchanged.
#
# Selection order: explicit `backend=` argument → `ENV["PHASOR_BACKEND"]`
# → platform probe (`detect_gpu_backend`).
#
# This file deliberately has no top-level `using`: test/gpu.jl includes
# it before any environment is activated, for `detect_gpu_backend` only.

"""
    detect_gpu_backend() -> Symbol

Guess the GPU backend for this machine from the host environment:
`:cuda` if an NVIDIA device node or `nvidia-smi` is present, `:oneapi`
if Intel Level Zero / SYCL tooling or an Aurora hostname is present,
otherwise `:cpu`. Override with `PHASOR_BACKEND` or an explicit argument.
"""
function detect_gpu_backend()
    if ispath("/dev/nvidia0") || Sys.which("nvidia-smi") !== nothing
        return :cuda
    end
    if haskey(ENV, "ZE_AFFINITY_MASK") || haskey(ENV, "ZE_FLAT_DEVICE_HIERARCHY") ||
       Sys.which("sycl-ls") !== nothing || occursin("aurora", lowercase(gethostname()))
        return :oneapi
    end
    return :cpu
end

"""
    resolve_gpu_backend(backend=:auto) -> Symbol

Normalise a requested backend: `:auto` consults `PHASOR_BACKEND` and then
`detect_gpu_backend()`. Returns one of `:cpu`, `:cuda`, `:oneapi`.
"""
function resolve_gpu_backend(backend::Symbol = :auto)
    if backend == :auto
        backend = Symbol(lowercase(get(ENV, "PHASOR_BACKEND", "auto")))
        backend == :auto && (backend = detect_gpu_backend())
    end
    backend in (:cpu, :cuda, :oneapi) ||
        error("Unknown GPU backend :$backend; expected :auto, :cpu, :cuda or :oneapi")
    return backend
end

"""
    load_gpu_backend(; backend=:auto, cuda_memory_limit="16GiB") -> (backend, device)

Load the vendor package for `backend` into `Main` and return
`(backend, device)`. For `:cuda`, `JULIA_CUDA_HARD_MEMORY_LIMIT` is set to
`cuda_memory_limit` unless already set (guards the DGX Spark's unified
memory against paging lock-ups). Requires `using PhasorNetworks` in `Main`.
"""
function load_gpu_backend(; backend::Symbol = :auto, cuda_memory_limit::AbstractString = "16GiB")
    backend = resolve_gpu_backend(backend)
    if backend == :cuda
        haskey(ENV, "JULIA_CUDA_HARD_MEMORY_LIMIT") ||
            (ENV["JULIA_CUDA_HARD_MEMORY_LIMIT"] = cuda_memory_limit)
        Core.eval(Main, :(using LuxCUDA))
    elseif backend == :oneapi
        Core.eval(Main, :(using oneAPI))
    end
    ENV["PHASOR_BACKEND"] = String(backend)
    # invokelatest: the extension methods were added after this function
    # was compiled, so look them up in the newest world.
    device = Base.invokelatest(Main.PhasorNetworks.select_device, backend)
    return backend, device
end
