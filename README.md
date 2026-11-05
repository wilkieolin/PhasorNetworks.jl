# PhasorNetworks

[![Dev docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://wilkieolin.github.io/PhasorNetworks.jl/dev/)
[![Build Status](https://github.com/wilkieolin/PhasorNetworks.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/wilkieolin/PhasorNetworks.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/wilkieolin/PhasorNetworks.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/wilkieolin/PhasorNetworks.jl)

This package provides methods for manipulating phase-based vector-symbolic systems and creating deep networks utilizing these operations. It supports executing these systems via atemporal floating-point operations or via dynamical systems interfering and exchanging impulses through time.

Information is represented as phases (angles in [-1, 1], in units of π) rather than scalar activations. Each layer implements one state-space equation, `dz/dt = k*z + W*I(t)` with `k = λ + iω`, which can be evaluated either in discrete form (matrix products and causal convolution) or as a continuous ODE driven by spikes or currents. Layers are built on [Lux.jl](https://github.com/LuxDL/Lux.jl) and differentiate with Zygote.

## Installation

You need [Julia](https://julialang.org/downloads/) 1.12, the version CI runs on. [juliaup](https://github.com/JuliaLang/juliaup) is the easiest way to install it.

The package is not in the General registry, so install it from its URL rather than by name. To use it as a library in your own project:

```julia
using Pkg
Pkg.add(url="https://github.com/wilkieolin/PhasorNetworks.jl")
Pkg.add("Lux")   # the quick start below builds Lux models directly
```

To work on the package itself, clone it and use the pinned environment:

```bash
git clone https://github.com/wilkieolin/PhasorNetworks.jl
cd PhasorNetworks.jl
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

The first install downloads and precompiles a large dependency tree (DifferentialEquations, Lux, Zygote), so expect it to take several minutes.

## Quick start

```julia
using PhasorNetworks, Lux, Random

rng = Xoshiro(42)

# Symbols are arrays of phases in [-1, 1] (units of π), with shape (dims, batch).
a = random_symbols(rng, (1024, 4))
b = random_symbols(rng, (1024, 4))

# VSA: binding produces a dissimilar symbol, and unbinding recovers the original.
c = v_bind(a, b)
similarity(c, a)                # ≈ 0 for each of the 4 samples
similarity(v_unbind(c, b), a)   # ≈ 1

# A Lux model on phases: a dense layer followed by a similarity readout.
model = Chain(PhasorDense(1024 => 128), Codebook(128 => 10))
ps, st = Lux.setup(rng, model)
scores, st = model(a, ps, st)
size(scores)                    # (10, 4): one score per class and sample
```

For an end-to-end training run on Fashion-MNIST (the dataset downloads on first use):

```bash
julia --project=. scripts/train_fashionmnist.jl --backend cpu --epochs 5
```

## GPU setup

The package has no GPU vendor dependency. CUDA and oneAPI are weak dependencies, and loading one in your environment activates the matching backend. In a clone, use the pinned environment for your hardware:

| Hardware | Environment | Set up with |
|----------|-------------|-------------|
| NVIDIA | `envs/cuda` | `make env-cuda` |
| Intel (e.g. Aurora) | `envs/oneapi` | `make env-oneapi` |

```bash
julia --project=envs/cuda scripts/train_fashionmnist.jl --backend auto
```

In your own project, add the vendor package and select the device:

```julia
using PhasorNetworks, LuxCUDA   # or: using PhasorNetworks, oneAPI
dev = select_device(:cuda)      # or :oneapi; :auto picks a loaded, functional backend
```

A [Dockerfile](Dockerfile) builds the CUDA environment; run it with `docker run --gpus all <image>` (this needs the NVIDIA Container Toolkit on the host).

## Running the tests

```bash
make test                 # CPU suite, same as Pkg.test()
make test-gpu             # GPU suite on the backend detected for this machine
make test-gpu BACKEND=cuda
```

On Aurora, launch through [`scripts/aurora_test.sh`](scripts/aurora_test.sh), which clears the library paths that otherwise abort oneAPI.

## Documentation

- **[API reference](https://wilkieolin.github.io/PhasorNetworks.jl/dev/)**, with pages for layers, spiking, VSA operations, GPU backends and the training methods.
- **Design notes and derivations**, kept in [`docs/`](docs):

| Topic | Start with |
|-------|------------|
| State-space model derivation | [`phasor_ssm_derivation.tex`](docs/phasor_ssm_derivation.tex) |
| Equilibrium propagation | [`phasor_ep_design.md`](docs/phasor_ep_design.md), [`ep_program_status.md`](docs/ep_program_status.md) |
| Holomorphic equilibrium propagation | [`hep_development_summary.md`](docs/hep_development_summary.md), [`phasor_hep_derivation.tex`](docs/phasor_hep_derivation.tex) |
| Local attention | [`local_attention_companion.md`](docs/local_attention_companion.md) |
| Fractional-power encoding and resonators | [`fpe_resonator.md`](docs/fpe_resonator.md) |
| Wave sheets | [`wave_dispersion_derivation.md`](docs/wave_dispersion_derivation.md), [`rf_wave_network_implementation.md`](docs/rf_wave_network_implementation.md) |
| Velocity sensing hardware | [`wavesheet_hardware_ladder.md`](docs/wavesheet_hardware_ladder.md) |

To build the API reference locally:

```bash
julia --project=docs -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
julia --project=docs docs/make.jl     # output in docs/build/
```

## Repository layout

| Path | Contents |
|------|----------|
| `src/` | The package |
| `ext/` | CUDA and oneAPI backend extensions |
| `test/` | Test suite; `test/gpu.jl` launches the GPU runs |
| `envs/` | Pinned per-backend GPU environments |
| `scripts/` | Experiments and figures, with their own environment: `julia --project=scripts scripts/make_all_figures.jl` |
| `demos/` | Standalone demonstration scripts; those that plot run with `--project=scripts`, the rest with `--project=.` |
| `docs/` | API reference source, design notes and derivations |
| `results/` | Recorded experiment outputs |

Earlier Jupyter tutorials and demo notebooks are no longer maintained or kept in the repository. The last commit containing them is [`903cf06`](https://github.com/wilkieolin/PhasorNetworks.jl/tree/903cf06).
