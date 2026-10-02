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
PhasorLSA
PhasorLCA
```

!!! note "PhasorSSM is deprecated"
    `PhasorSSM` has been removed. New code should use `PhasorDense` directly with
    `init_mode=:default` or `init_mode=:hippo`.

## Spiking encoding

```@docs
MakeSpikingSSM
ssm_phases_to_train
ssm_train_to_phases
```

## Residual blocks and stacks

Residual combination on the torus (phase addition of the branch onto the skip),
the transformer block built from it, and `ScanStack` for weight-tied or stacked
blocks. All three run in discrete, Dirac and spiking modes. In spiking mode,
`spike_phase_bind` adds phases by shifting spike times, and
`spike_phase_recenter` rotates a spike train by a phase offset.

```@docs
PhasorResidual
PhaseRecenter
PhasorTransformerBlock
ScanStack
spike_phase_bind
spike_phase_recenter
```

## Output extraction

```@docs
sample_phases_at_periods
reconstruct_from_current
```

## Encoding

```@docs
psk_encode
impulse_encode
```
