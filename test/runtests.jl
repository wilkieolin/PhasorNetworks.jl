using Lux, MLUtils, MLDatasets, OneHotArrays, Statistics, PhasorNetworks, Test
using Random: Xoshiro
using Base: @kwdef
using Zygote: withgradient
using Optimisers, ComponentArrays
using Statistics: mean
using LinearAlgebra: diag
using PhasorNetworks: bind

#global args for all tests
n_x = 101
n_y = 101
n_vsa = 1
epsilon = 0.10
repeats = 6
epsilon = 0.02
spk_args = default_spk_args()
tspan = (0.0, repeats*1.0)
tbase = collect(tspan[1]:spk_args.dt:tspan[2])

@kwdef mutable struct Args
    η::Float64 = 3e-4       ## learning rate
    batchsize::Int = 256    ## batch size
    epochs::Int = 4        ## number of epochs
    use_cuda::Bool = false   ## use gpu (if cuda available)
    rng::Xoshiro = Xoshiro(42) ## global rng
end

include("vsa_tests.jl")
include("network_tests.jl")

@testset "PhasorNetworks.jl" begin
    vsa_tests()
    network_tests()
end
