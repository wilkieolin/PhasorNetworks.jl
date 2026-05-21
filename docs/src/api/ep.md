# Equilibrium Propagation API

Phasor equilibrium propagation: train a [`PhasorDense`](@ref) chain by
running it to equilibrium and reading the loss gradient from the
difference between a free and a nudged settle (`StaticEP`) or by
demodulating the response to a cosine probe in the nudge amplitude
(`LockinEP`).

Cost functions ([`SimilarityCost`](@ref), [`CodebookCost`](@ref))
define the teaching signal; [`ep_gradient`](@ref) computes the
gradient; [`ep_train`](@ref) is the per-example training loop;
[`fd_gradient_phasor`](@ref) is the finite-difference oracle used for
verification.

See `demos/phasor_ep_demo.ipynb` (package-API walkthrough),
`demos/lockin_demo.ipynb` (lock-in derivation), and
`docs/phasor_ep_design.md` (full design doc, including the temporal-
Cauchy lock-in derivation) for background.

!!! note "Known limitation"
    `ep_train` does per-example SGD: parameters are updated after every
    `(x, y)` pair. On multi-pattern classification tasks (e.g. XOR with
    `normalize_to_unit_circle` activation) per-example updates from one
    example partially un-do the previous one's, capping accuracy short of
    what backprop would achieve. See `demos/lockin_demo.ipynb` §7 for a
    concrete demonstration.

```@autodocs
Modules = [PhasorNetworks]
Pages = ["src/ep.jl"]
```
