using CUDA, LuxCUDA
using ComponentArrays, SciMLSensitivity, DifferentialEquations, Lux

using Lux: glorot_uniform, truncated_normal
using LinearAlgebra: diagind, I
using DifferentialEquations: ODESolution
using ChainRulesCore: ignore_derivatives
using Random: GLOBAL_RNG, AbstractRNG
using Interpolations: linear_interpolation
using Statistics: cor, mean
using LinearAlgebra: diag
using OneHotArrays: OneHotMatrix
using NNLib: batched_mul

import LuxLib: dropout