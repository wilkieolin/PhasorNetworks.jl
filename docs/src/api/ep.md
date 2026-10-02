# Equilibrium Propagation API

Phasor equilibrium propagation: train a [`PhasorDense`](@ref) chain by
running it to equilibrium and reading the loss gradient from the
difference between a free and a nudged settle (`StaticEP`) or by
demodulating the response to a cosine probe in the nudge amplitude
(`LockinEP`).

Cost functions ([`SimilarityCost`](@ref), [`CodebookCost`](@ref))
define the teaching signal; [`ep_gradient`](@ref) computes the
gradient; [`ep_train`](@ref) is the training loop;
[`ep_predict`](@ref) scores a settled network against a codebook; and
[`fd_gradient_phasor`](@ref) is the finite-difference oracle used for
verification.

See `demos/phasor_ep_demo.ipynb` (package-API walkthrough),
`demos/lockin_demo.ipynb` (lock-in derivation),
`demos/ep_fashionmnist.jl` (FashionMNIST MLP), and
`docs/phasor_ep_design.md` (full design doc, including the temporal-
Cauchy lock-in derivation) for background.

## Minibatching

EP states are `(out_dims,)` for a single sample or `(out_dims, B)` for a
minibatch — one code path, selected by the shape of the input you pass to
[`phasor_settle`](@ref). Every operation in the settle is
column-separable, so a batched settle is exactly `B` independent settles
and the batched gradient equals the mean of the per-sample gradients (to
Float32 roundoff; verified in `test/test_ep.jl`).

The `1/B` normalization lives on the **Hebbian**, not on the nudge. Each
sample must receive the full per-sample nudge amplitude `β` — dividing the
nudge by `B` would shrink the linear response by `B` and destroy the
finite-difference SNR the EP estimate depends on.

To batch, pass a `(d, B)` phase matrix and a batched cost
(`CodebookCost(codes, labels::AbstractVector{<:Integer})`, or a
`SimilarityCost` whose target is either shared or `(d, B)`).

!!! note "Estimator bias"
    `StaticEP` defaults to the one-sided difference `-(h_β - h_0)/β`,
    which carries an `O(β)` bias. Pass `centered = true` for the symmetric
    form `-(h_₊ - h_₋)/(2β)`, which cancels that term at the cost of one
    extra nudged settle. This matters most when `StaticEP` is being used
    as a gradient *oracle* — e.g. calibrating `LockinEP` at a width where
    `fd_gradient_phasor` (`O(n_params)` settles) is unaffordable.

!!! note "Finite-difference step size"
    `fd_gradient_phasor` defaults to `ε = 1e-3`, the bottom of the Float32
    cancellation/truncation U-curve for `O(1)` losses. The old default of
    `1e-5` left about 1% absolute Float32 noise in every component, so the
    *oracle*, not EP, dominated the measured error, and the value depended
    on CPU and BLAS (0.067–0.132 relative error on the toy bias test, vs
    0.002 at `1e-3`). Measured on a 49→12→10 `CodebookCost` proxy, EP-vs-FD
    relative error runs 0.24 at `ε=1e-5`, 0.027 at `1e-4`, **0.004 at
    `1e-3`**, 0.036 at `1e-2`. Re-tune `ε` if the loss scale is far from 1.

```@autodocs
Modules = [PhasorNetworks]
Pages = ["src/ep.jl"]
```
