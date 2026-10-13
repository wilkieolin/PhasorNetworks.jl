#!/usr/bin/env julia
#
# GPU test launcher: resolve the backend for this machine, activate the
# matching environment under envs/, and run the full test suite there.
#
#     julia test/gpu.jl            # auto-detect (PHASOR_BACKEND overrides)
#     julia test/gpu.jl cuda
#     julia test/gpu.jl oneapi
#     julia test/gpu.jl cpu        # same as Pkg.test()
#
# `Pkg.test()` cannot be used for the GPU suites: its sandbox hard-sets
# the load path to the package's own deps plus the test target, and no
# GPU vendor package is a dependency of the package. The vendor packages
# are resolved only inside envs/cuda and envs/oneapi, so a machine never
# has to resolve both stacks.

using Pkg

const REPO = normpath(joinpath(@__DIR__, ".."))
include(joinpath(REPO, "scripts", "gpu_backend.jl"))   # detect_gpu_backend, resolve_gpu_backend

backend = resolve_gpu_backend(isempty(ARGS) ? :auto : Symbol(lowercase(ARGS[1])))

if backend == :cpu
    @info "Running the CPU suite via Pkg.test()"
    Pkg.activate(REPO)
    Pkg.test()
else
    env = joinpath(REPO, "envs", String(backend))
    isdir(env) || error("No environment for backend :$backend at $env")
    isfile(joinpath(env, "Manifest.toml")) ||
        @warn "$env has no Manifest.toml; Pkg.instantiate() will resolve one (needs network access)."

    if backend == :cuda
        # Must be set before CUDA initialises. Guards the DGX Spark's
        # unified memory: exceeding ~110 GB pages to disk and locks the box.
        haskey(ENV, "JULIA_CUDA_HARD_MEMORY_LIMIT") ||
            (ENV["JULIA_CUDA_HARD_MEMORY_LIMIT"] = "16GiB")
    end

    @info "Running the test suite on backend :$backend" env
    Pkg.activate(env)
    Pkg.instantiate()
    ENV["PHASOR_BACKEND"] = String(backend)
    include(joinpath(REPO, "test", "runtests.jl"))
end
