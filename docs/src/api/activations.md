# Activations API

Complex-domain activation functions used by phasor layers (`PhasorDense`,
`PhasorConv`, `PhasorResonant`, `ResonantSTFT`, `PhasorFixed`, and the
`SSM*Attention` family). These take a complex-valued post-`W·x + bias`
output and return a complex-valued result for the next stage.

```@autodocs
Modules = [PhasorNetworks]
Pages = ["activations.jl"]
```
