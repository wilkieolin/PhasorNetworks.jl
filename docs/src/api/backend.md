# Backend Abstraction API

Backend-agnostic helpers used by the rest of the package to dispatch
across GPU vendors. Built on
[`GPUArraysCore.AbstractGPUArray`](https://github.com/JuliaGPU/GPUArrays.jl),
[KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl)
and [MLDataDevices.jl](https://github.com/LuxDL/Lux.jl/tree/main/lib/MLDataDevices).

## Backends resolve per platform

The package itself depends on no GPU vendor package. CUDA.jl and oneAPI.jl
are *weak* dependencies: loading one in the calling environment activates
the matching extension (`PhasorNetworksCUDAExt`, `PhasorNetworksOneAPIExt`),
which registers that backend with [`select_device`](@ref).

| Backend   | Load                        | Environment     |
|-----------|-----------------------------|-----------------|
| `:cuda`   | `using LuxCUDA` (CUDA+cuDNN)| `envs/cuda`     |
| `:oneapi` | `using oneAPI`              | `envs/oneapi`   |
| `:cpu`    | nothing                     | any             |
| `:auto`   | whichever of the above is loaded and functional, else CPU | |

`Args.backend` defaults to `:auto`, so the same code picks CUDA on an
NVIDIA machine and oneAPI on Aurora, depending only on which vendor
package the environment loaded. The per-backend environments under
`envs/` are the only places a vendor package is resolved, so a machine
never has to resolve both stacks.

- `julia test/gpu.jl [cuda|oneapi|auto]` runs the full test suite on a GPU
  backend (auto-detects the platform). `Pkg.test()` runs the CPU suite.
  The GPU section runs first, so device-level failures surface early.
- `scripts/gpu_backend.jl` provides `load_gpu_backend()` for scripts.

### Aurora (oneAPI) note

Aurora's site oneAPI module puts its own SYCL adapters on
`LD_LIBRARY_PATH`, where they shadow the ones bundled with oneAPI.jl and
abort the process on the first oneMKL call (a matmul on a `oneArray`).
`scripts/aurora_test.sh` strips those entries before launching Julia;
`test/gpu.jl oneapi` and `load_gpu_backend()` refuse to start while they
are present (`PHASOR_ALLOW_SITE_ONEAPI=1` overrides). KernelAbstractions
kernels are unaffected either way — only the oneMKL paths are.

## Helpers

- [`select_device`](@ref) picks an MLDataDevices device by symbol
  (`:auto`, `:cpu`, `:cuda`, `:oneapi`). Dispatches through `Val` so
  package extensions register additional backends.
- `on_gpu(args...)` checks whether any argument is an `AbstractGPUArray`
  or a `SpikeTrainGPU`, regardless of vendor.
- `to_device_of(ref, x)` moves a host array `x` onto the device `ref`
  lives on ("device follows the data"); identity for host `ref`.
- `gpu_zeros(ref, T, dims...)` allocates a zero array on the same
  backend as `ref`, using `KernelAbstractions.zeros(get_backend(ref), …)`
  for GPU inputs and a plain `Array` for CPU inputs.
- `reclaim_device!(device)` releases cached device memory where the
  backend supports it (`CUDA.reclaim()`); a no-op elsewhere.

Inside the package, never name a vendor type or module: dispatch on
`AbstractGPUArray`, allocate with `gpu_zeros` / `to_device_of`, and move
data with `MLDataDevices.get_device(x)`.

See `src/gpu.jl` for the cross-backend kernels that consume these
helpers, and the `Args.backend` field on the [`Args`](@ref) type for
the training-loop entry point.

```@autodocs
Modules = [PhasorNetworks]
Pages = ["src/backend.jl"]
```
