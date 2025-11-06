```@meta
CurrentModule = PhasorNetworks
```

# PhasorNetworks.jl

PhasorNetworks.jl is a Julia package for implementing and simulating phasor neural networks, combining concepts from oscillatory computing and neuromorphic engineering.

## Features

- **Phasor Neural Networks**: Implementation of neural networks using phase-based computation
- **Spiking Neural Networks**: Tools for working with spiking neural networks and spike trains
- **Vector Symbolic Architecture**: Operations for VSA-based computations
- **GPU Acceleration**: CUDA support for large-scale simulations
- **Domain Transformations**: Utilities for converting between different neural representations
- **Performance Metrics**: Tools for analyzing and evaluating network performance

## Installation

```julia
using Pkg
Pkg.add("PhasorNetworks")
```

## Quick Start

```julia
using PhasorNetworks

# Create a basic phasor network
spk_args = SpikingArgs_NN(
    t_period = 1.0f0,
    t_window = 0.01f0,
    threshold = 0.001f0
)

# Example code will go here...
```

## Package Structure

The package functionality is organized into several core components:

- [`Types`](api/types.md): Core types and type system
- [`Networks`](api/network.md): Neural network implementation
- [`Spiking`](api/spiking.md): Spiking neural network functionality
- [`Domains`](api/domains.md): Domain transformation utilities
- [`VSA`](api/vsa.md): Vector Symbolic Architecture operations
- [`GPU`](api/gpu.md): GPU acceleration support
- [`Metrics`](api/metrics.md): Performance metrics and analysis

See the respective API documentation pages for detailed information about each component.

```@index
Pages = ["api/types.md", "api/network.md", "api/spiking.md", "api/domains.md", 
         "api/vsa.md", "api/gpu.md", "api/metrics.md"]
```
