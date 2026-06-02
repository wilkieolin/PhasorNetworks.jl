module PhasorNetworksOneAPIExt

using PhasorNetworks
using oneAPI
using oneAPI.oneL0: ZePtr   # oneAPI exports `oneL0` (submodule) and `oneMKL` (submodule)
                            # and `oneArray`, but NOT `ZePtr` at the top level.
                            # Must import explicitly via the oneL0 submodule.
using NNlib

# Add a Val{:oneapi} method to PhasorNetworks.select_device. The package
# defines select_device(::Symbol) → select_device(Val(backend)) plus
# methods for :cuda, :cpu, and a fallback. Extending via Val keeps each
# method's signature unique, so Julia's strict precompile (Aurora) does
# not flag this as method overwriting.
function PhasorNetworks.select_device(::Val{:oneapi})
    return oneAPI.oneAPIDevice()
end

# ------------------------------------------------------------------
# NNlib.batched_mul backend for oneArray
# ------------------------------------------------------------------
#
# NNlib's batched matmul pipeline routes any DenseArray{BlasFloat}-typed
# storage through `_batched_try_gemm!`, which finally dispatches on
# `_batched_gemm!(::Type{<:storage}, ...)`. NNlib ships methods for
# Array (generic BLAS) and CuArray (via NNlibCUDAExt → CUBLAS), but
# nothing for oneArray, so `batched_mul` on oneAPI tensors hits a
# MethodError. `_causal_conv_toeplitz` in kernels.jl is the primary
# consumer in this package.
#
# oneMKL exposes `gemm_strided_batched!` with the exact same calling
# convention as CUBLAS (transA, transB, α, A, B, β, C) for Float16/32/64
# and ComplexF32/64, so the wrapper is a direct hand-off.
NNlib._batched_gemm!(::Type{<:oneArray}, transA::Char, transB::Char,
                     α::Number, A, B, β::Number, C) =
    oneMKL.gemm_strided_batched!(transA, transB, α, A, B, β, C)

# NNlib's BatchedAdjoint/BatchedTranspose wrappers are passed straight
# through to the BLAS C call; the underlying ccall needs to extract a
# raw device pointer from them. Mirror NNlibCUDAExt's CuPtr shim using
# ZePtr (imported above from oneAPI.oneL0 — not re-exported by the
# top-level oneAPI module despite the internal `using .oneL0`).
Base.unsafe_convert(::Type{ZePtr{T}}, A::NNlib.BatchedAdjOrTrans{T}) where {T} =
    Base.unsafe_convert(ZePtr{T}, parent(A))

end
