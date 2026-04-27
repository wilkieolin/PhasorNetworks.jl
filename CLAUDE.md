# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PhasorNetworks.jl is a Julia package for phasor neural networks combining oscillatory computing, neuromorphic engineering, and Vector Symbolic Architectures (VSA). Information is represented as phases (angles in [-1, 1] in units of pi) rather than scalar activations.

The architecture is built on a unified State Space Model (SSM): each layer of neurons is defined by a single equation `dz/dt = k*z + W*I(t)` where `k = lambda + i*omega` is a per-channel complex eigenvalue. This equation can be evaluated in multiple equivalent modes: discrete (phase matrices, causal convolution, FFT), or continuous ODE (spiking/current input). The discrete kernel `K[n] = A^n * B` (where `A = exp(k*dt)`, `B = (A-1)/k`) links the modes.

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

### Unified SSM Model

Every layer implements a single defining equation with per-channel oscillator dynamics:

    dz_c/dt = k_c * z_c + W * I(t),  where k_c = lambda_c + i*omega_c

This equation is evaluated via multiple dispatch on the input type. Two
families of layers share the same defining equation but differ in what
input domain they accept:

**Encoder layers** (Complex 3D in → Phase 3D out, fixed/trainable ω):
- `PhasorResonant` — fixed per-channel ω, ZOH SSM via `phasor_kernel +
  causal_conv`, then `complex_to_angle`. Use this to bring a sampled
  complex sequence into the phase domain.
- `ResonantSTFT` — same shape as `PhasorResonant` but with **trainable**
  ω; learned frequency decomposition.

**Phase-domain layers** (Phase 3D / 2D in → Phase out):
- `PhasorDense`, `PhasorConv`, `PhasorFixed`, `Codebook`, attention.
  These dispatch on `<:Phase` arrays and use `causal_conv_dirac`
  (Dirac/spike-time encoding) for 3D Phase, or matmul for 2D Phase.
- They also accept `SpikingCall` (→ CurrentCall → ODE) and `CurrentCall`
  (single-stage ODE `dz/dt = k*z + W*I(t)` solved via
  DifferentialEquations.jl).

Common dispatch summary across these layers:

1. **2D Phase/Complex** (`AbstractArray{<:Phase}` or `AbstractArray{<:Complex}`) — direct linear transform `W*x + bias`, fast single-step inference
2. **3D Phase** (`AbstractArray{<:Phase, 3}`) — Dirac discretization + causal convolution via `causal_conv_dirac`, returns Phase
3. **3D Complex** (`AbstractArray{<:Complex, 3}`) — only on `PhasorResonant` / `ResonantSTFT` (the encoder layers); `PhasorDense` no longer has this dispatch
4. **SpikingCall** — converts to CurrentCall, then ODE integration
5. **CurrentCall** — continuous ODE: `dz/dt = k*z + W*I(t)`, solved via DifferentialEquations.jl

The discrete kernel `K[n] = A^n * B` (where `A = exp(k*dt)`, `B = (A-1)/k`) is mathematically equivalent to the continuous ODE, linking all modes.

`PhasorConv` currently still has the legacy "complex 3D in" path baked
in; its docstring carries a note about migrating it to follow the
encoder/phase-layer split when next touched.

### Lux Layer Contract

All layers extend `Lux.AbstractLuxLayer` and follow `(layer)(x, params, state) -> (output, state)`. Must implement `Lux.initialparameters(rng, layer)` and `Lux.initialstates(rng, layer)`. Trainable values go in parameters, fixed values in state.

### Per-Channel Dynamics Parameters

Each layer has per-channel trainable dynamics:
- `log_neg_lambda` — `(out,)` Float32, per-channel decay (always trainable, parameterized as `lambda = -exp(log_neg_lambda)`)
- `omega` — `(out,)` Float32, per-channel angular frequency (trainable or fixed in state)
- `weight` — `(out, in)` Float32, connection weights

Init modes: `:default` (uniform dynamics), `:uniform` (spread omega), `:hippo` (HiPPO-LegS multi-timescale)

### Data Flow

```
Input → Encoding (Phase, complex, or spike train)
      → Network Layers (PhasorDense / PhasorConv / ResonantSTFT / PhasorAttention)
      │   Discrete path: weight mixing → causal_conv(phasor_kernel, input)
      │   ODE path:      dz/dt = k*z + W*I(t) via oscillator_bank
      → Readout (Codebook / SSMReadout — similarity-based classification)
      → Loss & Metrics
```

### Source File Responsibilities

| File | Role |
|------|------|
| `types.jl` | `SpikeTrain`, `SpikeTrainGPU`, `SpikingArgs`, `SpikingCall`, `CurrentCall`, `Phase`, `Args` |
| `domains.jl` | Phase↔complex↔potential↔spike conversions, spike kernels, normalization, `bias_to_complex_offset` |
| `kernels.jl` | Discrete phasor kernels (`phasor_kernel`), causal convolution (Toeplitz/FFT), Dirac encoding, HiPPO init |
| `network.jl` | `PhasorResonant`, `ResonantSTFT` (complex→phase encoders), `PhasorDense`, `PhasorConv`, `PhasorFixed`, `ComplexBias`, `Codebook`, `PhasorAttention`, `train()` |
| `ssm.jl` | `SSMReadout`, `SSMCrossAttention`, `SSMSelfAttention`, encoding helpers, spiking dispatch, deprecated `PhasorSSM` compat |
| `spiking.jl` | `oscillator_bank`, `spike_current`, `neuron_constant`, spike detection |
| `vsa.jl` | `v_bind`, `v_unbind`, `v_bundle`, `similarity`, `codebook_loss` |
| `gpu.jl` | CUDA kernels mirroring CPU paths for spike processing, scatter-add, similarity |
| `metrics.jl` | `evaluate_accuracy`, `evaluate_loss`, confusion matrices, ROC curves |

### Key Type Aliases

- `Phase <: Real` — scalar type wrapping `Float32`, representing a phase angle in [-1, 1] (units of pi). `isbits`, 4 bytes. Arithmetic with other `Real` types promotes to `Float32`. Network layers dispatch on `AbstractArray{<:Phase}` for type safety.
- `LuxParams = Union{NamedTuple, AbstractArray}` — used for layer parameter type annotations
- `SolutionType` enum: `:phase`, `:potential`, `:current`, `:spiking`

## Critical Conventions

### Float32 Everywhere

Use `Float32` for all neural data. Write `1.0f0` not `1.0`. GPU kernels require this. Avoid Float64 except where external libraries demand it (e.g., Optimisers.jl learning rates).

### Phase Range [-1, 1]

Phases are represented as `Phase` values in [-1, 1] (units of pi). Producer functions (`complex_to_angle`, `soft_angle`, `remap_phase`, `random_symbols`, `time_to_phase`, `potential_to_phase`, `angular_mean`) return `Phase` arrays. Network layers (`PhasorDense`, `PhasorConv`, `PhasorFixed`, `Codebook`, `attend`, `PhasorAttention`) dispatch on `AbstractArray{<:Phase}` for their phase-mode forward pass. Use `Phase.()` to wrap raw `Float32` data at network input boundaries. Arithmetic on `Phase` promotes to `Float32` — use `remap_phase(x)` after arithmetic that may exceed bounds. Use circular distance metrics (`arc_error`) not naive subtraction for phase comparisons.

### Bias Application

In **discrete mode** (3D complex/Phase), bias is applied directly as a complex offset after causal convolution.

In **ODE mode** (CurrentCall), bias can be injected as periodic current via `bias_current`, or applied post-hoc via `bias_to_complex_offset(bias, tspan; spk_args)`. For training stability, prefer post-ODE application when gradients are unstable.

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
