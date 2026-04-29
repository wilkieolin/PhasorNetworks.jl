# Phasor EP — porting vanilla EP to phase-based networks

This note sketches how vanilla equilibrium propagation (EP) can be adapted
to phasor networks, sidestepping the holomorphic/meromorphic activation
constraint that limits hEP. It is intended as design background for
`demos/phasor_ep_demo.ipynb`, which implements the simplest version.

## Why phasor EP

Holomorphic EP (hEP) requires the activation function to be
holomorphic — at minimum meromorphic — so that the equilibrium
`z*(β)` is a holomorphic function of `β` and the contour integral
`(1/2πi) ∮ f(β)/β² dβ` extracts an exact derivative. Liouville's
theorem says any non-constant entire function that is bounded
*everywhere* must be constant, so we cannot have a globally-bounded
("normalizing") holomorphic activation. The standard escape route is
`tanh(z)` — meromorphic, with poles at `z = iπ(n+½)/a` — which gives a
hard ceiling on the hEP contour radius `r`.

Phase-based networks bypass this entirely. The state of each neuron
is a phase `θ ∈ [-1, 1]` (units of π), equivalently a unit complex
`z = e^{iπθ}`. The "saturation" we'd need a complex `tanh` for is
replaced by the **topological constraint** that the state lives on the
unit circle — a non-trivial nonlinear manifold without any saturating
function. No holomorphicity worries, no Liouville problem.

## State space

A layer of `N` phasor neurons is a vector `z ∈ T^N` (the N-torus). In
complex coordinates, each entry has unit modulus; in phase coordinates,
each entry is a real number in `[-1, 1]` interpreted modulo 2 (units
of π).

The continuous-time SSM that defines each layer is

    dz_l / dt = K_l · z_l + W_l · I_l(t),    K_l = diag(λ_l + i·ω_l)

with per-channel decay `λ_l < 0` and per-channel angular frequency
`ω_l`. We want this dynamics to fall out of `−∂Φ/∂z̄` for some
energy `Φ`, so that EP's free/nudged-phase machinery applies.

## Candidate energy

Three terms per layer, mirroring vanilla EP's structure:

    Φ({z_l}; θ, β) =
        Σ_l ½ Re⟨z_l, K_l z_l⟩            ← self-energy (SSM)
      + Σ_l Re⟨bundle_W_l(z_{l-1}), z_l⟩  ← cross-layer coupling
      − β · C(z_L, y)                     ← nudge (only at output)

The inner product is the standard Hermitian one `⟨a, b⟩ = Σ_i a_i · b̄_i`.

### Self-energy term

`K_l = diag(λ_l + iω_l)` is the per-channel SSM eigenvalue matrix.
Its Wirtinger gradient is

    ∂(½ Re⟨z, Kz⟩) / ∂z̄ = ½ K z

which reproduces the `K · z` term of the ODE. Without this term the
EP gradient theorem cannot give the SSM dynamics; with it, the energy
is "consistent" with the underlying dynamics in the same sense that
hEP's energy is.

### Cross-layer coupling

The "drive" into neuron `i` of layer `l` is the weighted complex
superposition of the previous layer's states:

    bundle_W_l(z_{l-1})_i = Σ_j W_l[i, j] · z_{l-1, j}

This is the **VSA bundle primitive** with weights `W_l[i, :]`. The
energy contribution `Re⟨bundle_W(z_{l-1}), z_l⟩` is, channel-by-channel,
the **complex cosine similarity** between the drive into a neuron and
that neuron's state.

The Wirtinger gradients are:

    ∂Φ_coupling / ∂z̄_l     = W_l · z_{l-1}                    (forward drive)
    ∂Φ_coupling / ∂z̄_{l-1} = W_l^* · z_l                       (backward feedback)

The forward gradient is the `W_l · I` term of the SSM; the backward
gradient is the EP feedback that requires symmetric (here:
conjugate-transpose) weights.

### Cost / nudge

For a target phase pattern `y` (also unit-modulus complex), the
natural cost is one-minus-similarity:

    C(z_L, y) = 1 − sim(z_L, y) = 1 − (1/d) · Re⟨z_L, y⟩

This is **cosine similarity to a target**, the same metric used for
similarity-based classification at inference time. Properties:

1. `C ∈ [0, 2]`, minimized at `C = 0` when `z_L = y`.
2. `∂C / ∂z̄_L = −y / (2d)` — a *constant* pull toward `y` in the
   complex plane, scaled by `β` in the nudged phase. No saturation,
   no slope factor — just a uniform force.
3. The loss surface and the prediction surface are aligned, since
   the same similarity is used for both.

For codebook-based classification (matching the existing `Codebook`
layer in this repo), generalize to softmax-cross-entropy over a
similarity vector:

    C(z_L, y_class) = −log softmax(similarity_outer(z_L, codebook))[y_class]

## Why no Liouville problem

The Liouville constraint forced hEP's activations to be meromorphic,
which capped the contour radius `r`. Phasor EP never applies a
complex-domain activation. The "nonlinearity" comes from two
structural sources:

1. **Unit-circle constraint** on `z`. Dynamics constrained to
   `|z| = 1` give a non-trivial nonlinear manifold without needing a
   saturating function.
2. **Per-channel oscillation** via `K_l`. The `iω` part of `k_l`
   rotates each channel at its own frequency — this is the analog of
   "different neurons" but it's geometric (rotation in ℂ), not
   "saturating" in the squashing sense.

Because of (1), the natural variational principle is gradient flow on
the **torus**, not on `ℂ^N`. In phase coordinates `z_l = e^{iπθ_l}`:

    dθ_{l,i} / dt = −∂Φ / ∂θ_{l,i} = 2π · Im(z̄_{l,i} · ∂Φ/∂z_{l,i})

This is automatic — no special functions, no poles, no
holomorphicity worries. The "force" on each phase is the imaginary
part of the Hermitian inner product of the drive with the state,
which is exactly the **arc-distance** between the drive's angle and
the state's angle. The codebase already computes this kind of
quantity (`arc_error` and friends).

## Algorithmic recipe

Same shape as vanilla EP:

1. **Free phase**: settle the network with `β = 0` by integrating
   the phase ODE (or running ZOH steps on the phase representation).
   Each layer's phase is pulled toward `angle(bundle_W(z_{l-1}))`
   from below and `angle(W_{l+1}^* z_{l+1})` from above (the
   feedback term — symmetric weights still required, just like
   vanilla EP).
2. **Nudged phase**: re-settle with `β > 0` added; the output layer
   gets an additional rotation toward `y`'s phases.
3. **Hebbian gradient**: `Σ Re(z_l · z̄_{l-1}^T)` at each
   equilibrium, differenced and divided by `β`. Same structure as
   vanilla EP, just with complex outer products evaluated
   channel-wise.
4. **Training rule**: `W_l ← W_l + η·(hebb_nudge − hebb_free)/β`.
   The update is itself complex-valued (or, equivalently, you can
   keep `W` complex and update both real and imaginary parts).

## Implementation notes

The simplest first cut is **discrete time** — one ZOH step per "EP
iteration" — so you can reuse the existing `phasor_kernel +
causal_conv` infrastructure verbatim. The energy function above is
what those kernels are *implicitly* descending in the free phase;
making it explicit is what lets you add the nudged phase and extract
the EP gradient.

A minimal `phasor_ep_settle!` is a port of `ep_settle!` from the
vanilla-EP demo notebook with two substitutions:

* `tanh.(grad)` → `normalize_to_unit_circle.(grad)` (project to
  unit circle instead of squashing).
* MSE cost → `1 − similarity(z_L, y)` (cosine distance to target).

For the very first prototype, set `K = 0` (no decay, no
oscillation). Then the only forces are coupling and nudge, and the
equilibrium is purely a "phase-consensus" fixed point. Add `λ` and
then `ω` once that prototype is working.

## Subtlety: equilibrium isn't strictly on the unit circle

The self-energy term `½ Re⟨z, Kz⟩` acts like a quadratic confinement
toward `z = 0`. On the unit-circle constraint surface that's a
degenerate force (zero perpendicular to the surface), but it
interacts with the oscillation `iω · z`. In practice this means the
EP equilibrium for phasor networks is a balance of magnitude AND
phase, not pure phase — the magnitude relaxes to whatever balances
the input drive against the decay `λ`, and the phase rotates at `ω`
until the cross-layer coupling locks it in. That's exactly the
behavior the discrete-time ZOH kernel already produces; making the
energy explicit just lets us also compute the EP nudge.

## Open questions

* **Trainable `K`?** In the existing PhasorDense, `λ` is always
  trainable (via `log_neg_lambda`) and `ω` is optionally trainable.
  EP's gradient extraction extends naturally to these per-channel
  dynamics parameters, but we'd need to derive the Hebbian update for
  `K` separately (it's the "self" gradient of the self-energy term).
* **Symmetric weights.** Vanilla EP requires `W` symmetric; phasor
  EP requires `W` Hermitian (i.e., `W = W^H`). With real `W`, this
  is just `W = W^T` — the existing constraint. With complex `W`, it
  becomes a stronger condition.
* **Continuous-time vs. discrete-time.** Discrete-time is the easy
  starting point and reuses the existing kernels. Continuous-time
  needs an ODE solver in the EP loop, which is more code but more
  consistent with how the rest of PhasorNetworks treats SSMs.
* **Codebook readout.** The cross-entropy nudge through
  `similarity_outer` is straightforward in the energy formulation,
  but needs a Wirtinger derivative through the softmax. Should match
  what `hep_cost_xent_grad` already does in `src/hep.jl`.

## Temporal Cauchy / lock-in detection for phasor EP

### Why spatial hEP doesn't directly transfer

The phasor recurrence uses `unit_project(z) = z / |z|`, which is
non-holomorphic in `z` because `|z| = √(z · z̄)` involves `z̄`.
Propagated through to the equilibrium `z*(β)`, this means

    z*(β) = z*(β, β̄)

is real-differentiable in β but not holomorphic. If you tried hEP's
contour formula

    f'(0) = (1/2πi) ∮ f(β)/β² dβ ≈ (1/N) Σ f(β_n)/β_n,    β_n = r·e^{2πin/N}

the integrand picks up a `conj(β)/β² = e^{-2iφ}/r²·e^{-2iφ}` part
that aliases at every contour radius. Increasing `N` doesn't help —
the contamination isn't at higher Taylor powers, it's the conjugate
sideband.

### The temporal version: lock-in detection

Instead of sampling β on a *spatial* contour, drive it as a
*temporal* trajectory. Two flavors:

* **Real probe** `β(t) = ε·cos(ω_p t)`: real-axis perturbation,
  the natural analog of vanilla EP's static β.
* **Complex probe** `β(t) = ε·e^{iω_p t}`: orbital perturbation
  in the complex β-plane, the analog of hEP's spatial contour.

Lock-in detect the response by multiplying any observable `f(t)` by
`e^{-iω_p t}` and time-averaging over an integer number of probe
cycles:

    H_+ = (1/T) ∫_0^T f(t) · e^{-iω_p t} dt

For small `ε`, the equilibrium response is

    z*(t) = z*(0) + ε·e^{iω_p t}·∂z*/∂β + ε·e^{-iω_p t}·∂z*/∂β̄ + O(ε²)

Substituting and integrating gives different things for the two
probe choices:

| probe | what `2·Re(H_+/ε)` extracts | matches FD on real W? |
|---|---|---|
| real (`cos`) | `∂z*/∂β + ∂z*/∂β̄ = d/dβ_real` | **yes** |
| complex (`e^{iω_p t}`) | `2·Re(∂z*/∂β)` (Wirtinger only) | only if system is holomorphic in β |

**This is the key correction over the hEP picture.** When the
underlying map is holomorphic in β (the hEP regime, with
meromorphic activations like tanh), `∂/∂β̄ = 0` and the two probes
agree — the complex probe gives a clean derivative, which is what
hEP exploits. When the map is *not* holomorphic in β (the phasor-EP
regime with `unit_project`), the complex probe extracts only the
holomorphic Wirtinger half, which is **not** the gradient on
real-valued weights. The real probe is what reproduces FD ground
truth.

### Why this still beats spatial hEP

For phasor networks specifically, the lock-in formulation has three
conceptual advantages:

1. **No Liouville constraint on the activation.** Lock-in only
   requires Wirtinger-differentiability of the recurrence, not
   holomorphicity. `unit_project` is fine. Any non-holomorphic
   activation is fine. This is the activation freedom that hEP had
   to give up.
2. **Native dynamical fit.** Phasor SSMs are already oscillating
   systems. Adding a probe oscillation is in their native language —
   the lock-in mixer can be implemented using the existing
   `oscillator_bank` infrastructure. There's no "probe machinery"
   that's foreign to the network.
3. **Frequency-multiplexed gradients.** Because each parameter
   direction's gradient lives at a chosen `ω_p`, you can use
   *several* probe frequencies simultaneously (one per parameter
   direction or per output channel) and lock-in detect each
   independently in a single settle. Spatial hEP forces one
   contour-evaluation per parameter direction; temporal lock-in
   supports gradient parallelism via FDM.

### Accuracy requirements

Three knobs, three error sources — the temporal analog of hEP's
`(N, r)` trade-off.

| knob | controls | too small | too large |
|---|---|---|---|
| `ε` (probe amplitude) | linearity | numerical noise floor | nonlinear O(ε²) bias |
| `ω_p` (probe frequency) | separation from natural ω_l, adiabatic following | resonance with channel oscillations | non-adiabatic (system can't track) |
| `T_int` (integration cycles) | demodulator selectivity | aliasing from imperfectly-cancelled harmonics | computational cost |

Stability constraints specific to lock-in:

* **Frequency separation**: `|ω_p − ω_l| > |λ|` for every channel `l`
  (probe must lie outside every channel's natural bandwidth).
* **Adiabatic following**: `ω_p ≪ relaxation_rate ≈ |λ| + ‖W‖`
  (probe period much longer than the network's slowest fixed-point
  convergence time).
* **Integer cycles**: integrate for `T_int = 2π/ω_p · k` with
  integer `k ≥ 2`, otherwise even-harmonic terms leak through.
* **Probe-phase warm-up**: discard the first ~2 cycles of probe
  steady-state so the equilibrium has caught up to the modulation.
* **DC subtraction**: explicitly subtract the free-equilibrium
  Hebbian before demodulating; otherwise DC leakage from a
  non-integer cycle count or a transient dominates.

The cliff analog of hEP's "r ≈ 0.3" breakdown is here a
**probe-amplitude cliff** set by the equilibrium's basin-of-
attraction radius. Beyond it, the probe pushes the network through
a bifurcation and lock-in is meaningless.

### Implementation sketch

```julia
function phasor_ep_lockin(net, x, y; ε=0.05, ω_p=0.05, n_cycles=8,
                          T_free=400, T_warmup_cycles=2, dt=0.1)
    # Free settle to equilibrium
    z_h = zeros(ComplexF64, ...); z_o = zeros(ComplexF64, ...)
    phasor_settle!(net, x, z_h, z_o, y, 0.0; T=T_free, dt=dt)
    H_dc = z_h * adjoint(x)   # DC hebb to subtract

    period_steps = round(Int, 2π / (ω_p * dt))
    # Warm-up — drive probe but don't accumulate
    for t in 1:(T_warmup_cycles * period_steps)
        β_t = ε * cos(ω_p * t * dt)
        # ... settle step with β_t ...
    end
    # Lock-in over n_cycles complete probe periods
    H_acc = zeros(ComplexF64, size(net.W1))
    for t in 1:(n_cycles * period_steps)
        β_t = ε * cos(ω_p * t * dt)
        # ... settle step with β_t ...
        H_t = (z_h * adjoint(x)) - H_dc        # DC-subtracted hebb
        H_acc += H_t * exp(-im * ω_p * t * dt) # demodulate
    end
    return -2 .* real.(H_acc) ./ (n_cycles * period_steps * ε)
end
```

The factor of 2 comes from `Re(cos(ω_p t)·e^{-iω_p t}) = 1/2` over
an integer cycle: the lock-in coefficient at `+ω_p` of a real
probe of amplitude `ε` is `ε/2 · d/dβ_real(observable)`.
