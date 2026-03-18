# SSM (State Space Models)

Discrete state space model layers for phasor networks: causal convolution kernels,
SSM layers, attention, encoding, and readout.

## Kernels and Convolution

```@docs
phasor_kernel
causal_conv
hippo_legs_diagonal
```

## Layers

```@docs
PhasorSSM
SSMReadout
SSMCrossAttention
SSMSelfAttention
```

## Encoding

```@docs
psk_encode
impulse_encode
```
