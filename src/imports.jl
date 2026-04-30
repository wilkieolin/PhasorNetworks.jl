using CUDA, LuxCUDA
using ComponentArrays, Lux
using SciMLSensitivity, DifferentialEquations, Optimisers

using Lux: glorot_uniform, truncated_normal
using LinearAlgebra: diagind, I, dot
using DifferentialEquations: ODESolution
using ChainRulesCore: ignore_derivatives, NoTangent, unthunk
import ChainRulesCore
import Random
using Random: GLOBAL_RNG, AbstractRNG
using Interpolations: linear_interpolation
using Statistics: cor, mean
using LinearAlgebra: diag
using OneHotArrays: OneHotMatrix
using NNlib: batched_mul
using FFTW  # registers CPU FFT methods with AbstractFFTs
using AbstractFFTs: fft, ifft
using Base: @kwdef
using Zygote: withgradient
using Random: Xoshiro
using WeightInitializers: ones32

import LuxLib: dropout