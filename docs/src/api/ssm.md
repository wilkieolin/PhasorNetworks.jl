# SSM (State Space Models)

Discrete state space model support for phasor networks: causal convolution kernels,
readout layers, attention, and encoding helpers.

SSM functionality is integrated directly into the main layer types (`PhasorDense`,
`PhasorConv`) via `init_mode` and per-channel dynamics parameters. The layers in this
section provide readout, attention, and encoding utilities for SSM workflows.

## Kernels and Convolution

```@docs
phasor_kernel
causal_conv
causal_conv_fft
causal_conv_dirac
dirac_encode
hippo_legs_diagonal
```

## Layers

```@docs
SSMReadout
SSMCrossAttention
SSMSelfAttention
```

!!! note "PhasorSSM is deprecated"
    `PhasorSSM` is a backward-compatible constructor that returns a `PhasorDense` with
    SSM-appropriate defaults. New code should use `PhasorDense` directly with
    `init_mode=:uniform` or `init_mode=:hippo`.

```@docs
PhasorSSM
```

## Encoding

```@docs
psk_encode
impulse_encode
```
