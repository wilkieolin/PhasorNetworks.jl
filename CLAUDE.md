# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PhasorNetworks.jl is a Julia package for phasor neural networks combining oscillatory computing, neuromorphic engineering, and Vector Symbolic Architectures (VSA). Information is represented as phases (angles in [-1, 1] in units of pi) rather than scalar activations. Networks can execute via atemporal floating-point computation or via coupled oscillator ODEs producing spike trains.

Built on Lux.jl (functional neural network framework), with Zygote.jl for AD, DifferentialEquations.jl for ODE solving, and CUDA.jl for GPU acceleration with hand-written kernels.

## Common Commands

```bash
# Run full test suite
julia --project=. -e 'using Pkg; Pkg.test()'

# Run from REPL
julia --project=.
using Pkg; Pkg.test()

# Run a specific test file interactively
julia --project=.
include("test/runtests.jl")  # loads globals, then call e.g. network_tests()

# Build docs locally
julia --project=docs docs/make.jl

# Training script
julia scripts/train_fashionmnist.jl --lr 0.001 --epochs 5 --optimizer rmsprop --batchsize 128 --use_cuda true
```

There is no linter or formatter configured. Julia 1.11+ is required.

## Architecture

### Three Execution Modes via Multiple Dispatch

Every layer supports three calling conventions through Julia's multiple dispatch:

1. **Phase mode** (`AbstractArray` input) — direct floating-point phase computation, fast inference
2. **Spiking mode** (`SpikingCall` input) — event-driven ODE integration via `oscillator_bank`
3. **Current mode** (`CurrentCall` input) — continuous current input, intermediate representation

### Lux Layer Contract

All layers extend `Lux.AbstractLuxLayer` and follow `(layer)(x, params, state) -> (output, state)`. Must implement `Lux.initialparameters(rng, layer)` and `Lux.initialstates(rng, layer)`. Trainable values go in parameters, fixed values in state.

### Data Flow

```
Input → Phase Encoding (angle_to_complex / phase_to_train)
      → Network Layers (PhasorDense / PhasorConv / PhasorAttention)
      → Codebook (similarity-based classification)
      → Loss & Metrics
```

### Source File Responsibilities

| File | Role |
|------|------|
| `types.jl` | `SpikeTrain`, `SpikeTrainGPU`, `SpikingArgs`, `SpikingCall`, `CurrentCall`, `Phase` |
| `domains.jl` | Phase↔complex↔potential↔spike conversions, kernels, normalization, `bias_to_complex_offset` |
| `vsa.jl` | `v_bind`, `v_unbind`, `v_bundle`, `similarity`, `codebook_loss` |
| `network.jl` | `PhasorDense`, `PhasorConv`, `ComplexBias`, `Codebook`, `PhasorAttention`, `train()` |
| `spiking.jl` | `oscillator_bank`, `spike_current`, `neuron_constant`, spike detection |
| `gpu.jl` | CUDA kernels mirroring CPU paths for spike processing, scatter-add, similarity |
| `metrics.jl` | `evaluate_accuracy`, `evaluate_loss`, confusion matrices, ROC curves |
| `network_complex.jl` | Complex-native `PhasorDenseComplex` variant (in development on `cmpx_layers`) |

### Key Type Aliases

- `Phase <: Real` — scalar type wrapping `Float32`, representing a phase angle in [-1, 1] (units of pi). `isbits`, 4 bytes. Arithmetic with other `Real` types promotes to `Float32`. Network layers dispatch on `AbstractArray{<:Phase}` for type safety.
- `LuxParams = Union{NamedTuple, AbstractArray}` — used for layer parameter type annotations
- `SolutionType` enum: `:phase`, `:potential`, `:current`, `:spiking`

## Critical Conventions

### Float32 Everywhere

Use `Float32` for all neural data. Write `1.0f0` not `1.0`. GPU kernels require this. Avoid Float64 except where external libraries demand it (e.g., Optimisers.jl learning rates).

### Phase Range [-1, 1]

Phases are represented as `Phase` values in [-1, 1] (units of pi). Producer functions (`complex_to_angle`, `soft_angle`, `remap_phase`, `random_symbols`, `time_to_phase`, `potential_to_phase`, `angular_mean`) return `Phase` arrays. Network layers (`PhasorDense`, `PhasorConv`, `PhasorFixed`, `Codebook`, `attend`, `PhasorAttention`) dispatch on `AbstractArray{<:Phase}` for their phase-mode forward pass. Use `Phase.()` to wrap raw `Float32` data at network input boundaries. Arithmetic on `Phase` promotes to `Float32` — use `remap_phase(x)` after arithmetic that may exceed bounds. Use circular distance metrics (`arc_error`) not naive subtraction for phase comparisons.

### Bias Application: Post-ODE Only

**Never inject bias during ODE solving** — it causes adjoint gradient instability (NaN/Inf). Always:
1. Solve ODE with `use_bias=false`
2. Apply bias afterward with `bias_to_complex_offset(bias, tspan; spk_args)`

### Zygote AD Compatibility

- Never mutate arrays in differentiable code paths — use `map` and broadcasting
- Use `ChainRulesCore.ignore_derivatives` for bookkeeping logic
- Use `ifelse` not `if/else` for conditional logic in hot paths
- Avoid dictionaries, try/catch, and foreign calls in differentiable paths
- ComponentArrays need special handling for bias parameters

### ODE Gradient Strategy

Uses `BacksolveAdjoint` with `ZygoteVJP` for sensitivity analysis through DifferentialEquations.jl. Default solver is `Tsit5()` with `dt=0.005, adaptive=false`.

### GPU/CPU Parity

Functions in `gpu.jl` must mirror CPU paths. Check both `Array` and `CuArray` code paths. Avoid scalar indexing on GPU arrays.

### Exports

All public API functions must be listed in the `export` block in `src/PhasorNetworks.jl`.

## Testing

Test globals defined in `test/runtests.jl`: `n_x=101, n_y=101, epsilon=0.025, repeats=10`. Default `SpikingArgs`: `leakage=-0.2, t_period=1.0, t_window=0.01`. CUDA tests run conditionally when `CUDA.functional()` is true. Use appropriate tolerances for floating-point phase comparisons.

## Code Style

- 4-space indentation
- `snake_case` for functions, `PascalCase` for types
- Docstrings with `# Arguments`, `# Returns`, `# Implementation` sections
