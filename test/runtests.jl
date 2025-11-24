using Lux, MLUtils, MLDatasets, OneHotArrays, Statistics, PhasorNetworks, Test
using DifferentialEquations, SciMLSensitivity, CUDA, LuxCUDA
using Random: Xoshiro, AbstractRNG
using Base: @kwdef
using Zygote: withgradient
using Optimisers, ComponentArrays
using Statistics: mean
using LinearAlgebra: diag
using Distributions: Normal
using DifferentialEquations: Heun, Tsit5
#global args for all tests
n_x = 101
n_y = 101
n_vsa = 1
epsilon = 0.10
repeats = 10
epsilon = 0.025
#solver_args = Dict(:adaptive => false, :dt => 0.01)
solver_args = Dict(:adaptive => false, 
                    :dt => 0.01,
                    :sensealg => InterpolatingAdjoint(; autojacvec=ZygoteVJP()),
                    :save_start => true)

spk_args = SpikingArgs(t_window = 0.01, 
                    threshold = 0.001,
                    solver=Tsit5(), 
                    solver_args = solver_args)
tspan = (0.0, repeats*1.0)
tbase = collect(tspan[1]:spk_args.solver_args[:dt]:tspan[2])

@kwdef mutable struct Args
    lr::Float64 = 3e-4       ## learning rate
    batchsize::Int = 256    ## batch size
    epochs::Int = 10        ## number of epochs
    use_cuda::Bool = false   ## use gpu (if cuda available)
    rng::Xoshiro = Xoshiro(42) ## global rng
end

include("data.jl")
include("domain_tests.jl")
include("vsa_tests.jl")
include("network_tests.jl")
#include("PROPOSED_metrics_tests.jl")
#include("PROPOSED_network_layers_tests.jl")
#include("PROPOSED_spiking_operations_tests.jl")

@testset "PhasorNetworks.jl" begin
    domain_tests()
    vsa_tests()
    network_tests()
    #new tests
    #metrics_tests()
    #network_layers_tests()
    #spiking_operations_tests()

    if CUDA.functional()
        @info "CUDA device detected and functional. Running CUDA tests..."
        @testset "CUDA Specific Tests" begin
            try
                include("test_cuda.jl")
                cuda_core_tests() # Call the main test function from test_cuda.jl
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
