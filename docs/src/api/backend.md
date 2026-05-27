# Backend Abstraction API

Backend-agnostic helpers used by the rest of the package to dispatch
across GPU vendors. Built on
[`GPUArraysCore.AbstractGPUArray`](https://github.com/JuliaGPU/GPUArrays.jl)
and [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl).

- [`select_device`](@ref) picks a Lux `gpu_device` / `cpu_device`
  by symbol (`:cuda`, `:cpu`, `:oneapi`). Dispatches through `Val`
  so package extensions can register additional backends (e.g.
  `PhasorNetworksOneAPIExt` adds `Val{:oneapi}`).
- `on_gpu(args...)` checks whether any argument is an
  `AbstractGPUArray`, regardless of vendor.
- `gpu_zeros(ref, T, dims...)` allocates a zero array on the same
  backend as `ref`, using `KernelAbstractions.zeros(get_backend(ref), …)`
  for GPU inputs and a plain `Array` for CPU inputs.

See `src/gpu.jl` for the cross-backend kernels that consume these
helpers, and the `Args.backend` field on the [`Args`](@ref) type for
the training-loop entry point.

```@autodocs
Modules = [PhasorNetworks]
Pages = ["src/backend.jl"]
```
