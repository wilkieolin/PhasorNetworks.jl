# GPU vendor packages (CUDA, oneAPI) are weak dependencies: they are never
# imported here. Backend-specific code lives in ext/PhasorNetworks*Ext.jl and
# is activated by `using CUDA` / `using oneAPI` in the calling environment.
import MLDataDevices
using MLDataDevices: get_device, gpu_device, cpu_device,
                     AbstractDevice, AbstractGPUDevice, CPUDevice
using KernelAbstractions
using GPUArraysCore: AbstractGPUArray
using Atomix
using ComponentArrays, Lux
using SciMLSensitivity, DifferentialEquations, Optimisers

using Lux: glorot_uniform, truncated_normal
using LinearAlgebra: diagind, I, dot, mul!, adjoint
using DifferentialEquations: ODESolution
using ChainRulesCore: ignore_derivatives, NoTangent, unthunk
import ChainRulesCore
import Random
using Random: GLOBAL_RNG, AbstractRNG
using Interpolations: linear_interpolation
using Statistics: cor, mean, std, quantile
using LinearAlgebra: diag
using OneHotArrays: OneHotMatrix
using NNlib: batched_mul, softmax, sigmoid
using FFTW  # registers CPU FFT methods with AbstractFFTs
using AbstractFFTs: fft, ifft
using Base: @kwdef
using Zygote: withgradient, Buffer, checkpointed
using Random: Xoshiro

import LuxLib: dropout