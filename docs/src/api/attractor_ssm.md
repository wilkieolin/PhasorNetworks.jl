# Attractor SSM API

Selective recurrent SSM layer with a Hopfield-style attractor pull toward
learned phasor codes. Extends the linear time-invariant SSM
`dz/dt = (λ + iω)·z + W·I(t)` with the additional term
`α·(pull(z, codes) − z)`, where `pull(z, C) = C · softmax(β · sim(z, C))`.

```@autodocs
Modules = [PhasorNetworks]
Pages = ["src/attractor_ssm.jl"]
```
