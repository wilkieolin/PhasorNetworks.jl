# Repository Overview

## Project Description

**PhasorNetworks.jl** is a Julia package for implementing and simulating phasor neural networks that combine concepts from oscillatory computing, neuromorphic engineering, and Vector Symbolic Architectures (VSA).

- **Main purpose:** Provide methods for manipulating phase-based vector-symbolic systems and creating deep networks utilizing these operations. It supports both atemporal floating-point computation and dynamical systems that interfere and exchange impulses through time (spiking neural networks via ODE solvers).
- **Key goals:** Enable researchers to build, train, and evaluate phasor neural networks that operate on phase representations rather than traditional scalar activations, with GPU acceleration for large-scale experiments.

### Key Technologies

- **Language:** Julia 1.11+
- **Deep Learning Framework:** [Lux.jl](https://github.com/LuxDL/Lux.jl) (functional, stateless neural network library)
- **Automatic Differentiation:** Zygote.jl with ChainRulesCore.jl for custom rules
- **ODE Solvers:** DifferentialEquations.jl with SciMLSensitivity.jl for differentiable ODE solving
- **GPU Acceleration:** CUDA.jl / LuxCUDA.jl with hand-written CUDA kernels
- **Optimisation:** Optimisers.jl (Adam, RMSProp, etc.)
- **Data Handling:** MLDatasets.jl, MLUtils.jl, OneHotArrays.jl
- **Documentation:** Documenter.jl (deployed to GitHub Pages)
- **CI:** GitHub Actions (test on Julia 1.11, codecov coverage)

## Architecture Overview

### Unified SSM Model

Phasor networks represent information as **phases** (angles in [-1, 1] in units of pi) rather than scalar activations. Each layer of neurons is defined by a single equation:

    dz_c/dt = k_c * z_c + W * I(t),  where k_c = lambda_c + i*omega_c

This equation discretizes to a linear recurrence `z[n+1] = A*z[n] + B*I[n]` which unrolls into a causal convolution `z = K * I` with kernel `K[n] = A^n * B`. This links three equivalent execution modes:

1. **Discrete mode** — phase matrices or complex sequences processed via causal convolution (Toeplitz for short, FFT for long sequences). Fastest for training.
2. **Continuous ODE mode** — spike train or current inputs integrated via `oscillator_bank` using DifferentialEquations.jl. Enables neuromorphic deployment.
3. **Dirac mode** — phase inputs encoded as instantaneous spikes, processed via `causal_conv_dirac` for efficient phase-to-phase computation.

Frequencies are locked across layers (or explicitly shifted via `PhasorSTFT`) to allow communication with invariant phase differences.

The package also provides:
- **VSA operations** — binding (element-wise phase addition) and bundling (circular mean) for compositional representations
- **GPU acceleration** — custom CUDA kernels for spike-current computation, scatter-add, and interference calculations

### Data Flow

```
Input Data
    |
    v
Encoding (Phase, complex, or spike train)
    |
    v
Network Layers (PhasorDense / PhasorConv / PhasorSTFT / PhasorAttention)
    |  Discrete path: weight mixing -> causal_conv(phasor_kernel, input)
    |  ODE path:      dz/dt = k*z + W*I(t) via oscillator_bank
    |  Dirac path:    causal_conv_dirac(phases, W, lambda, omega, T)
    v
Readout (Codebook / SSMReadout — similarity-based classification)
    |
    v
Loss & Metrics (similarity_loss, codebook_loss, evaluate_accuracy)
```

### Main Components

| Component | File | Responsibility |
|-----------|------|----------------|
| **Types** | `src/types.jl` | Core data structures: `SpikeTrain`, `SpikeTrainGPU`, `SpikingArgs`, `SpikingCall`, `CurrentCall`, `Phase`, `Args` |
| **Domains** | `src/domains.jl` | Conversions between phase, complex, time, potential, spike train, and real-vector representations |
| **Kernels** | `src/kernels.jl` | Discrete phasor kernels (`phasor_kernel`), causal convolution (Toeplitz/FFT), Dirac encoding (`dirac_encode`, `causal_conv_dirac`), HiPPO-LegS initialization |
| **Network** | `src/network.jl` | Lux layer definitions: `PhasorDense`, `PhasorConv`, `PhasorSTFT`, `PhasorFixed`, `ComplexBias`, `Codebook`, `ResidualBlock`, `PhasorAttention`, `SingleHeadAttention`, `MinPool`, `MakeSpiking`, `train()` |
| **SSM** | `src/ssm.jl` | `SSMReadout`, `SSMCrossAttention`, `SSMSelfAttention`, encoding helpers (`psk_encode`, `impulse_encode`), spiking dispatch for SSM layers, deprecated `PhasorSSM` constructor |
| **VSA** | `src/vsa.jl` | Vector Symbolic Architecture ops: `v_bind`, `v_unbind`, `v_bundle`, `v_bundle_project`, `similarity`, `random_symbols` |
| **Spiking** | `src/spiking.jl` | ODE-mode utilities: `oscillator_bank`, `spike_current`, `bias_current`, `neuron_constant`, spike detection |
| **GPU** | `src/gpu.jl` | CUDA kernels and GPU-optimised paths for spike processing, scatter-add, interference, `oscillator_bank`, `similarity_outer` |
| **Metrics** | `src/metrics.jl` | Evaluation: `arc_error`, `angular_mean`, `evaluate_loss`, `evaluate_accuracy`, `confusion_matrix`, ROC curves |
| **Constants** | `src/constants.jl` | Global constants: `N_THREADS` (CUDA), `pi_f32`, device handles |
| **Imports** | `src/imports.jl` | Centralised `using`/`import` statements |

### Layer Dispatch via Multiple Dispatch

All custom layers extend `Lux.AbstractLuxLayer` and follow `(layer)(x, params, state) -> (output, state)`. Input type determines execution mode:

- **2D Complex** `(C, B)` — linear transform `W*x + bias`, no temporal kernel
- **3D Complex** `(C, L, B)` — weight mixing + causal convolution with per-channel kernel + activation
- **2D Phase** — `angle_to_complex` -> 2D complex -> activation -> `complex_to_angle`
- **3D Phase** — Dirac discretization with `causal_conv_dirac` -> `complex_to_angle`
- **SpikingCall** — converts to CurrentCall, delegates
- **CurrentCall** — continuous ODE `dz/dt = k*z + W*I(t)`, sampled at period boundaries

Per-channel trainable parameters: `log_neg_lambda` (decay), `omega` (angular frequency), `weight` (connections). Init modes: `:default`, `:uniform`, `:hippo` (HiPPO-LegS multi-timescale memory).

## Directory Structure

```
PhasorNetworks.jl/
├── src/                        # Package source code
│   ├── PhasorNetworks.jl       # Module definition and exports
│   ├── imports.jl              # Centralised dependency imports
│   ├── constants.jl            # Global constants (N_THREADS, pi_f32, devices)
│   ├── types.jl                # Core type definitions
│   ├── domains.jl              # Domain conversion functions
│   ├── kernels.jl              # Discrete phasor kernels, causal convolution, HiPPO
│   ├── network.jl              # Neural network layers and training
│   ├── ssm.jl                  # SSM readout, attention, encoding, spiking helpers
│   ├── vsa.jl                  # Vector Symbolic Architecture operations
│   ├── spiking.jl              # ODE-mode utilities (oscillator_bank, spike_current)
│   ├── gpu.jl                  # CUDA kernels and GPU-optimised operations
│   └── metrics.jl              # Evaluation metrics and analysis
├── test/                       # Test suite
│   ├── runtests.jl             # Test entry point
│   ├── data.jl                 # Test data helpers
│   ├── domain_tests.jl         # Domain conversion tests
│   ├── vsa_tests.jl            # VSA operation tests
│   ├── network_tests.jl        # Network forward/backward pass tests
│   ├── network_layers_tests.jl # Individual layer tests
│   ├── test_ssm.jl             # SSM kernel, causal conv, readout, attention tests
│   ├── test_phase_type.jl      # Phase scalar type tests
│   ├── metrics_tests.jl        # Metrics tests
│   ├── test_cuda.jl            # CUDA-specific tests
│   └── PROPOSED_spiking_operations_tests.jl  # Proposed spiking tests (not yet active)
├── scripts/                    # Training scripts
│   ├── train_fashionmnist.jl   # FashionMNIST training script (CLI)
│   ├── train_fashionmnist_conv.jl  # Convolutional variant
│   └── dispatch_polaris.sh     # HPC job submission (PBS/Polaris)
├── demos/                      # Jupyter notebooks demonstrating features
│   ├── binding.ipynb           # VSA binding demos
│   ├── bundling.ipynb          # VSA bundling demos
│   ├── network fashionmnist.ipynb  # End-to-end training demo
│   ├── layer.ipynb / layer conv.ipynb  # Layer-level demos
│   ├── oscillator bank.ipynb   # Spiking oscillator demos
│   └── ...                     # Many more demo notebooks
├── tutorial/                   # Step-by-step tutorial notebooks
│   ├── 00 julia crash course.ipynb
│   ├── 01 representation.ipynb
│   ├── 02 oscillators.ipynb
│   ├── 03 similarity.ipynb
│   └── 04 neural network.ipynb
├── docs/                       # Documenter.jl documentation
│   ├── make.jl                 # Doc build script
│   └── src/                    # Doc source (index.md, api/*.md)
├── runs/                       # Saved training run results (.jld2)
├── Project.toml                # Julia package manifest (deps + compat)
├── Manifest.toml               # Locked dependency versions
└── .github/workflows/          # CI configuration
    ├── CI.yml                  # Tests + docs deployment
    ├── CompatHelper.yml        # Dependency compat auto-updates
    └── TagBot.yml              # Auto-tagging releases
```

### Key Entry Points

- **Package entry:** `src/PhasorNetworks.jl` — defines the module, exports, and includes
- **Test entry:** `test/runtests.jl` — runs the test suite
- **Training scripts:** `scripts/train_fashionmnist.jl` — CLI-driven training with ArgParse
- **Doc build:** `docs/make.jl` — generates API documentation

## Development Workflow

### Environment Setup

```bash
# Clone the repository
git clone https://github.com/wilkieolin/PhasorNetworks.jl.git
cd PhasorNetworks.jl

# Start Julia and activate the project
julia --project=.

# Install dependencies
using Pkg
Pkg.instantiate()
```

### Building and Running

```julia
# Load the package in development mode
using PhasorNetworks

# Run a training script from the command line
julia scripts/train_fashionmnist.jl --lr 0.001 --epochs 5 --optimizer rmsprop --batchsize 128 --use_cuda true
```

### Testing

```bash
# Run the full test suite
julia --project=. -e 'using Pkg; Pkg.test()'

# Or from the Julia REPL
using Pkg
Pkg.test()
```

The test suite covers: domain conversions, VSA operations (binding, bundling, similarity, orthogonality), network forward/backward passes, individual layer behaviour (PhasorDense, PhasorConv, ComplexBias, Codebook, MinPool, ResidualBlock, etc.), metrics, and optionally CUDA-specific tests.

**Note:** Many test sets in `runtests.jl` are currently commented out; only `network_tests()` is active by default. Uncomment others as needed: `domain_tests()`, `vsa_tests()`, `metrics_tests()`, `network_layers_tests()`.

### Documentation

```bash
# Build docs locally
julia --project=docs docs/make.jl
# Output is in docs/build/
```

Documentation is auto-deployed to GitHub Pages via CI on pushes to `main`.

### Code Style and Conventions

- **No formal linter/formatter** is configured. Follow existing code style: 4-space indentation, descriptive function/type names in snake_case (functions) and PascalCase (types).
- **Docstrings:** Functions should have docstrings with `# Arguments`, `# Returns`, and `# Implementation` sections (see `src/domains.jl` and `src/gpu.jl` for good examples).
- **Type annotations:** Use `Float32` throughout for GPU compatibility. The `LuxParams = Union{NamedTuple, AbstractArray}` type alias is used for layer parameters.
- **Multiple dispatch:** Each operation typically has methods for `AbstractArray` (phase mode), `SpikingCall` (spike mode), and `CurrentCall` (current mode), plus CPU and GPU specialisations.
- **Exports:** All public API functions must be listed in the `export` block in `src/PhasorNetworks.jl`.

### HPC Deployment

The `scripts/dispatch_polaris.sh` script provides a PBS job template for running on the Polaris supercomputer at ALCF, loading Julia 1.11 and dispatching GPU training.
