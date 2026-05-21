# Holomorphic Equilibrium Propagation API

Holomorphic EP (hEP) recovers the loss gradient by sampling a *spatial*
Cauchy contour `β = r·e^{2πin/N}` in the complex nudge plane and reading
the first Fourier coefficient of the network response. Requires a
holomorphic activation ([`holotanh`](@ref)) so the equilibrium
`z*(β)` depends only on `β` (no `β̄`); for non-holomorphic recurrences
(e.g. unit-magnitude projection), see the temporal lock-in approach in
[`LockinEP`](@ref).

[`hep_equilibrium`](@ref) settles the coupled phasor recurrence under a
given nudge; [`hep_gradient`](@ref) does the contour integration;
[`hep_train`](@ref) is the per-example training loop;
[`HolomorphicReadout`](@ref) is the demodulating readout layer aligned
with the hEP training dynamics.

See `demos/hep_demo.ipynb`, `docs/phasor_hep_derivation.tex` (formal
derivation), and `docs/hep_development_summary.md` (development
history and known issues) for background.

!!! warning "Gradient/loss alignment"
    The current hEP implementation's Hebbian gradient is uncorrelated
    with the true loss gradient on multi-example objectives — see
    `docs/hep_development_summary.md`. `hep_train`'s reported losses
    decrease for single-pattern memorization, but the parameter updates
    are not gradient descent on a global loss. Treat this as a
    research-grade interface.

```@autodocs
Modules = [PhasorNetworks]
Pages = ["src/hep.jl"]
```
