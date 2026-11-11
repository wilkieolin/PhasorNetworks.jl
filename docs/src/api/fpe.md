# Fractional Power Encoding API

Fractional power encoding (FPE) and residue codes: [`FPEEncoder`](@ref) writes
real values into phases, `v_bind` adds encoded values without carries, and
[`Resonator`](@ref) factors a bound address back into its codebook entries.
See `docs/fpe_resonator.md` for measured capacity and the status of the spiking
path.

```@autodocs
Modules = [PhasorNetworks]
Pages = ["src/fpe.jl"]
```
