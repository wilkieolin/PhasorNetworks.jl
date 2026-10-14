using Lux, OneHotArrays, Statistics, PhasorNetworks, Test
using DifferentialEquations, SciMLSensitivity, ChainRulesCore
using MLDataDevices: AbstractGPUDevice

# Backend selection.
#
# `Pkg.test()` runs the CPU suite: no GPU vendor package is a test
# dependency, so nothing below loads one. GPU suites are launched with
#
#     julia test/gpu.jl [cuda|oneapi|auto]      # auto = detect platform
#
# which activates the matching environment under envs/ (the only place a
# vendor package is resolved), sets PHASOR_BACKEND, and includes this
# file. Loading the vendor package here activates the package extension
# (PhasorNetworksCUDAExt / PhasorNetworksOneAPIExt) that registers the
# `select_device(:cuda)` / `select_device(:oneapi)` methods.
const PHASOR_BACKEND = Symbol(lowercase(get(ENV, "PHASOR_BACKEND", "cpu")))
PHASOR_BACKEND in (:cpu, :cuda, :oneapi) ||
    error("PHASOR_BACKEND must be one of cpu, cuda, oneapi; got $(PHASOR_BACKEND)")
PHASOR_BACKEND == :cuda   && @eval using LuxCUDA   # CUDA + cuDNN
PHASOR_BACKEND == :oneapi && @eval using oneAPI
# On the CPU suite `select_device(:auto)` (exercised by test_backend.jl)
# would print MLDataDevices' "no functional GPU" advice; that is expected.
PHASOR_BACKEND == :cpu && get!(ENV, "MLDATADEVICES_SILENCE_WARN_NO_GPU", "1")
const GPU_DEV = PHASOR_BACKEND == :cpu ? cpu_device() : select_device(PHASOR_BACKEND)
const GPU_AVAILABLE = GPU_DEV isa AbstractGPUDevice
if PHASOR_BACKEND != :cpu && !GPU_AVAILABLE
    @warn "PHASOR_BACKEND=$(PHASOR_BACKEND) requested but no functional device was found; GPU tests will be skipped."
end
using Random: Xoshiro, AbstractRNG
using Base: @kwdef
using Zygote: withgradient
using Optimisers, ComponentArrays
using Statistics: mean
using LinearAlgebra: diag
using Distributions: Normal
using DifferentialEquations: Tsit5
#global args for all tests
# Phase-grid resolution (vsa_tests). Was 101 — coarser is fine for the
# check_phase logic and the comprehension-built outer matrix is O(n²).
# Smaller (e.g. 33) breaks the bundling test's mean-error threshold:
# the test averages a per-element systematic bias (~0.01 magnitude)
# over n² samples, so the empirical mean has noise ~bias/√(n²); 51
# clears the 0.025 threshold with margin.
n_x = 51
n_y = 51
n_vsa = 1
# Spike-train cycle count — controls tspan = (0, repeats); each repeat
# is one ODE integration window. 10 was empirically required for the
# spiking model to reach the cycle-correlation / spiking-accuracy
# thresholds (~0.7) in network_tests; lowering it broke those without
# meaningfully widening the convergence assertions. The cost is in
# per-call ODE integration; we already bring that down by shrinking
# batch sizes in network_tests.
repeats = 10
epsilon = 0.025
#solver_args = Dict(:adaptive => false, :dt => 0.01)
solver_args = Dict(:adaptive => false, 
                    :dt => 0.005,
                    :sensealg => BacksolveAdjoint(; autojacvec=ZygoteVJP()),
                    :save_start => true)

spk_args = SpikingArgs(t_window = 0.01,
                    threshold = 0.001,
                    solver=Tsit5(),
                    solver_args = solver_args)
tspan = (0.0, repeats*1.0)
tbase = collect(tspan[1]:spk_args.solver_args[:dt]:tspan[2])

@kwdef mutable struct Args
    lr::Float64 = 3e-4       ## learning rate
    lr_ssm::Float64 = 0.0    ## SSM dynamics learning rate (0 = use lr)
    lr_alpha::Float64 = 0.0  ## ReZero gate (alpha) learning rate (0 = use lr)
    weight_decay::Float64 = 0.0 ## weight decay (weights only, not SSM params)
    cosine_schedule::Bool = false ## cosine LR annealing
    lr_min::Float64 = 1e-6   ## minimum LR for cosine schedule
    gc_interval::Int = 0     ## GC every N batches (0 = every batch)
    batchsize::Int = 256    ## batch size
    epochs::Int = 10        ## number of epochs
    backend::Symbol = :cpu   ## GPU backend (:cuda, :cpu, :oneapi)
    rng::Xoshiro = Xoshiro(42) ## global rng
end

include("data.jl")
include("domain_tests.jl")
include("vsa_tests.jl")
include("network_tests.jl")
include("metrics_tests.jl")
include("network_layers_tests.jl")
include("test_phase_type.jl")
include("test_ssm.jl")
include("test_local_attention.jl")
include("test_transformer_block.jl")
include("test_multimode.jl")
include("test_attractor_ssm.jl")
include("test_wave.jl")
include("test_excitable.jl")
include("test_hep.jl")
include("test_ep.jl")
include("spiking_operations_tests.jl")

@testset "PhasorNetworks.jl" begin
    # Backend abstraction tests — self-contained @testset, just include it.
    include("test_backend.jl")

    # Backend-agnostic GPU test gate. `GPU_DEV` is the device selected in
    # the preamble from PHASOR_BACKEND (CUDA on NVIDIA, oneAPI on Intel).
    # The test bodies route everything through `GPU_DEV` and
    # `AbstractGPUArray` dispatch, so the same suite runs on either.
    # Runs before the CPU suite so a device-level failure (driver abort,
    # missing vendor-library hook) surfaces in minutes, not after the
    # ten-minute CPU run — relevant on Aurora's one-hour debug queue.
    if GPU_AVAILABLE
        @info "GPU backend :$(PHASOR_BACKEND) → $(typeof(GPU_DEV)). Running GPU tests..."
        @testset "GPU Specific Tests" begin
            try
                include("test_gpu.jl")
                gpu_core_tests()
                gpu_kernel_compile_tests()
                ssm_gpu_tests()
                local_attention_gpu_tests()
            catch e
                @error "Error during GPU tests:" exception=(e, catch_backtrace())
                @test false # Explicitly fail GPU test section on error
            end
        end
    else
        @info "No GPU backend selected (PHASOR_BACKEND=$(PHASOR_BACKEND)). Skipping GPU tests; use `julia test/gpu.jl` to run them."
    end

    domain_tests()
    vsa_tests()
    network_tests()
    metrics_tests()
    network_layers_tests()
    phase_type_tests()
    ssm_tests()
    local_attention_tests()
    transformer_block_tests()
    multimode_tests()
    attractor_ssm_tests()
    wave_tests()
    excitable_tests()
    hep_tests()
    ep_tests()
    spiking_operations_tests()
end
