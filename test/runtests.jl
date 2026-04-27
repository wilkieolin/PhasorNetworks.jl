using Lux, OneHotArrays, Statistics, PhasorNetworks, Test
using DifferentialEquations, SciMLSensitivity, CUDA, LuxCUDA, ChainRulesCore
using Random: Xoshiro, AbstractRNG
using Base: @kwdef
using Zygote: withgradient
using Optimisers, ComponentArrays
using Statistics: mean
using LinearAlgebra: diag
using Distributions: Normal
using DifferentialEquations: Heun, Tsit5
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
    weight_decay::Float64 = 0.0 ## weight decay (weights only, not SSM params)
    cosine_schedule::Bool = false ## cosine LR annealing
    lr_min::Float64 = 1e-6   ## minimum LR for cosine schedule
    gc_interval::Int = 0     ## GC every N batches (0 = every batch)
    batchsize::Int = 256    ## batch size
    epochs::Int = 10        ## number of epochs
    use_cuda::Bool = false   ## use gpu (if cuda available)
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
include("test_hep.jl")
#include("PROPOSED_spiking_operations_tests.jl")

@testset "PhasorNetworks.jl" begin
    domain_tests()
    vsa_tests()
    network_tests()
    metrics_tests()
    network_layers_tests()
    phase_type_tests()
    ssm_tests()
    hep_tests()
    #spiking_operations_tests()

    if CUDA.functional()
        @info "CUDA device detected and functional. Running CUDA tests..."
        @testset "CUDA Specific Tests" begin
            try
                include("test_cuda.jl")
                cuda_core_tests() # Call the main test function from test_cuda.jl
                ssm_gpu_tests()
            catch e
                @error "Error during CUDA tests:" exception=(e, catch_backtrace())
                @test false # Explicitly fail CUDA test section on error
            end
        end
    else
        @info "No functional CUDA device detected or CUDA.jl is not functional. Skipping CUDA tests."
        # Optionally, explicitly mark as skipped if your test framework supports it.
    end
end
