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
    site_oneapi_library_dirs() -> Vector{String}

`LD_LIBRARY_PATH` entries that belong to an Aurora site oneAPI module
(`/opt/aurora/<release>/oneapi/...`). They shadow the SYCL adapters
(`libur_adapter_level_zero`, `libumf`) bundled with oneAPI.jl's support
library — `LD_LIBRARY_PATH` wins over that library's RUNPATH — and a
version mismatch aborts the process (SIGABRT, "level_zero backend failed
with error: 45 (UR_RESULT_ERROR_INVALID_ARGUMENT)") inside
`sycl::platform::get_platforms` on the first oneMKL call. Empty on any
other machine.
"""
function site_oneapi_library_dirs()
    entries = split(get(ENV, "LD_LIBRARY_PATH", ""), ':'; keepempty=false)
    return String[e for e in entries if occursin(r"^/opt/aurora/[^/]+/oneapi/", e)]
end

"""
    check_oneapi_environment()

Fail fast when `site_oneapi_library_dirs()` is non-empty: the dynamic loader
reads `LD_LIBRARY_PATH` at process start, so this cannot be repaired from
inside Julia — the shell that launches Julia has to drop those entries
(`scripts/aurora_test.sh` does). Set `PHASOR_ALLOW_SITE_ONEAPI=1` to run
anyway.
"""
function check_oneapi_environment()
    dirs = site_oneapi_library_dirs()
    isempty(dirs) && return nothing
    get(ENV, "PHASOR_ALLOW_SITE_ONEAPI", "0") == "1" && return nothing
    error("""
        LD_LIBRARY_PATH contains Aurora's site oneAPI module directories:
            $(join(dirs, "\n    "))
        They shadow the SYCL runtime bundled with oneAPI.jl and abort the first
        oneMKL call (matmul on a oneArray). Launch Julia with those entries
        removed, e.g.

            export LD_LIBRARY_PATH="\$(echo "\${LD_LIBRARY_PATH:-}" | tr ':' '\\n' | grep -v '^/opt/aurora/[^/]*/oneapi/' | paste -sd:)"

        (scripts/aurora_test.sh does this), or set PHASOR_ALLOW_SITE_ONEAPI=1
        to proceed regardless.""")
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
        check_oneapi_environment()
        Core.eval(Main, :(using oneAPI))
    end
    ENV["PHASOR_BACKEND"] = String(backend)
    # invokelatest: the extension methods were added after this function
    # was compiled, so look them up in the newest world.
    device = Base.invokelatest(Main.PhasorNetworks.select_device, backend)
    return backend, device
end
