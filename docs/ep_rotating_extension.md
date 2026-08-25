# Rotating (resonate-and-fire) EP: what changes, and what doesn't

> **Formal derivation:** `docs/phasor_lockin_derivation.tex` §Rotating Substrate &nbsp;|&nbsp;
> **Gates:** `scripts/ep_rotating_gates.jl` &nbsp;|&nbsp;
> **CI:** `ep_carrier_tests()` in `test/test_ep.jl`
>
> **Measurements:** `results/ep_readout_floor/FINDINGS.md` &nbsp;|&nbsp;
> **Open questions:** `docs/ep_rotating_followups.md` &nbsp;|&nbsp;
> **Background:** `docs/ep_adiabatic_sweep_design.md`

This file is the narrative account: what changed, what was measured, and what
the earlier prototype got wrong. The theorems and proofs live in the `.tex`.

## The question

The adiabatic sweep characterized `LockinEP` on a *static* settle: `K_mode=:zero`
drops the per-channel dynamics, so neither λ nor ω ever entered `phasor_settle`.
The goal is now a learning rule that runs **while a symmetrically-connected
spiking network is running** — where each neuron's instantaneous phase rotates at
a shared carrier ω, and the information lives in the relative phases, which
don't.

## The answer: ω cancels exactly

Φ = Σ_l Re⟨W_l z_{l−1}, z_l⟩ is invariant under the global rotation
z_l → e^{iθ}z_l applied to **every** layer including the input z₀. Substituting
z_l = w_l·e^{iωt}:

    ż_l = (λ + iω)z_l + W_l·z_{l−1}   ⟹   ẇ_l = λw_l + W_l·w_{l−1}

The carrier cancels against the frame derivative identically. `_project_damp` is
U(1)-equivariant (`û(e^{iθ}g) = e^{iθ}û(g)`), so the *discrete* step commutes
with the rotation too — this is exact at any `dt`, not an adiabatic
approximation.

The only terms that break the symmetry are the bias and the cost target, and in
this package **both physically co-rotate**:

- `bias_current` (`src/spiking.jl:12`) injects a `periodic_raised_cosine_kernel`
  pulse **once per period at a phase-determined time**, wrapped on the ring of
  circumference `t_period`. Its fundamental is `b·e^{iωt}` — the same thing a
  presynaptic spike delivers. It is not a DC offset.
- The codebook is made of neurons rotating at the same ω.

**Consequence: for a chain with one shared carrier, the rotating problem *is* the
static problem.** The existing zone map (`results/ep_adiabatic/`) already applies.
A sweep in the lab frame buys nothing, and the lab frame is strictly worse
numerically (see Gate C).

This is a strong validation of the per-channel ω rule (`CLAUDE.md`): the rule
exists so phase-locked communication works, and exact carrier cancellation is
what that buys.

## Why the Hebbian's `adjoint` is load-bearing

In the co-rotating frame `real(z_self·z_in') = cos(π(θ_self − θ_in))` — a
**relative phase**. It is U(1)-invariant, and it is measurable locally from
pre/post spike-time differences. That is the whole reason this program can run on
a spiking substrate at all.

Substituting `transpose` gives `real(z_self·z_in) = cos(π(θ_self + θ_in))`, which
spins at 2ω under a global rotation and is not a spike-timing-measurable quantity.
Gate D asserts the adjoint form is invariant **and** that the transpose form
visibly breaks, so it cannot pass vacuously.

## What actually changed in the code

### λ and ω were conflated in `ep_self_force`

It returned `½(λ + iω)z`, added into the pre-projection drive. But the
self-energy term is `½Re⟨z, Kz⟩` with K = λ + iω, and

    Re⟨z, (λ + iω)z⟩ = Re((λ + iω)|z|²) = λ|z|²

so **the rotation contributes nothing to the energy**. ω is symplectic: it
generates the U(1) flow rather than descending Φ. The force was therefore
inconsistent with `ep_energy_contribution`, where ω cancels automatically, and
the rotation was being mixed into the nonlinear unit projection.

That is why `:stored` was documented as needing `dt ≪ 0.5` or an ω override to
settle at all. `ep_self_force` now returns `½λz`, and `:stored` runs at the
layer's default ω = 2π and dt = 0.5 (Gate A).

**Where it came from — and where it didn't.** The formal derivation had this
right the whole time. `docs/phasor_lockin_derivation.tex` App. A states
`∂Φ/∂z̄ = ½Λz + …` with `Λ = diag(λ)`, explicitly "because ½Re⟨z,Kz⟩ =
½Σλ_j|z_j|² depends only on Re K", and correctly identifies `½iωz` as a
non-variational rotational term that is not the gradient of any real energy.

What went wrong is that the *informal* companion documents disagreed with it and
the code followed those. `docs/phasor_ep_design.md` asserted
`∂(½Re⟨z,Kz⟩)/∂z̄ = ½Kz`, "which reproduces the `K·z` term of the ODE", and
`demos/lockin_demo.ipynb` repeated it (harmlessly there, since it sets K = 0
throughout). Both are now corrected in place.

The useful lesson is not "a doc was wrong" but that the rigorous statement and
the implementable one had drifted apart with nothing checking them against each
other. Gate A now does exactly that: it FD-verifies the implemented force
against the settle's own fixed point, so the two cannot silently diverge again.

The stronger statement is worth keeping in view: `Re⟨z, iωz⟩ = 0` for any z, so
**no** real energy can have a rotation as its gradient. A rotating network is a
gradient system *plus* a U(1) generator, and EP's fixed point becomes a
*relative* equilibrium — a fixed point of the co-rotating dynamics, carried
around at `e^{iωt}`. That is precisely why the co-rotating frame is the right
place to do the work.

### The carrier is now an exact multiplicative rotation

`phasor_settle(carrier=ω, t0=…)` runs the lab frame:

    z ← cis(ω·dt) · [(1−dt)·z + dt·û(g)]

never as an additive `iω·z`. Gate C includes ω = 2π at dt = 0.5, where
ω·dt = π — **exactly Nyquist**. A discretized rotation is unusable there; the
multiplicative one has no dt limit because the rotation is never discretized.

The bias and cost target are put on the carrier at the current time. The nudge is
evaluated on the *demodulated* state and the resulting force rotated back, so
every cost type works unchanged — including `CodebookCost`, whose softmax is
nonlinear and could not simply be rotated.

## Gate results

| gate | asserts | measured |
|---|---|---|
| **A** | corrected `½λz` force still matches true FD, both `K_mode` | EP↔FD rel-err **0.001–0.008** |
| **B** | `carrier=0` reproduces the co-rotating settle | **exactly 0** (bit-identical) |
| **C** | lab frame ≡ co-rotating frame after demodulation | rel-err **1e-7 – 1.3e-4**, cos ≥ **0.9999998** |
| **D** | Hebbians U(1)-invariant (and transpose breaks) | **9.3e-8** invariant / **1.84** broken |
| **E** | lab-frame *gradient* ≡ co-rotating gradient | rel-err **4.5e-5 – 8.2e-5**, cos = **1.00000000** |

Two notes on reading these:

- **Gate A's tolerance is set by the oracle, not by EP.** A forward FD at ε on an
  O(1) Float32 loss carries roundoff ~eps/ε in the gradient. Evaluated at
  ε = 1e-5 and 1e-4, FD disagrees *with itself* by 0.004–0.096 — larger than
  EP's disagreement with it in every case. The gate reports that floor per
  parameter rather than trusting a single FD call.
- **Gate C's residual is accumulated Float32 rounding, not a frame error.** It
  tracks T (the step count) rather than ω, dt, or ω·dt, and the direction stays
  exact. The lab frame applies `rot` once per step and accrues ~T·eps of drift
  that the co-rotating frame never incurs — one more reason to do the real work
  in the co-rotating frame.

## What is genuinely new: a two-sided ε window

On a spiking substrate, phase is not read off a complex number — it is inferred
from *when* a neuron spiked, resolvable only to about the spike-kernel width.
`SpikingArgs` defaults to `t_window = 0.01` against `t_period = 1.0`, and each
pulse is smeared over ±2·t_window, so the readout quantum is a few percent of a
turn.

That puts a **lower** bound on the probe amplitude ε: the lock-in must resolve a
response of order ε·χ above the floor. The standing advice for the estimator's
*other* failure mode — hard-projection basin hopping at large ε
(`results/ep_adiabatic/ANTICORRELATED_GRADIENT_NOTE.md`) — is to shrink ε, which
drives straight into this floor from the other side.

So the usable zone for a spiking implementation is **two-sided**, and a sweep on
exact complex states can only ever see its upper edge. `LockinEP(readout_δ=…)`
models it as readout-only quantization (dynamics stay analog; only what the
synapse observes is snapped to the spike-time grid).

This is not a foregone conclusion. The lock-in integrates over many probe cycles
and the state sweeps across bin boundaries as it goes, so the probe **dithers**
the quantizer and time-averaging recovers some sub-quantum resolution. How much
is exactly what the `readout` axis measures.

## The soft projection has to be the equivariant one

`_project_damp`'s hard branch, `ifelse(r > th, g/r, 1+0im)`, is discontinuous:
as |g| → 0 the output *jumps* to 1+0im rather than approaching it. Near-zero
drives are routine — `test/test_ep.jl` downscales its weights by 0.4 precisely
because "the default glorot is wide enough that some initial drives can have
small magnitude during settling". That discontinuity is the structural
candidate for the bimodal lock-in failures in
`results/ep_adiabatic/ANTICORRELATED_GRADIENT_NOTE.md`, which concluded the fix
was operational ("use a smaller ε"). Bimodality across draws is the signature of
a threshold being crossed, not of an amplitude being too large.

The obvious softening is the package's existing
`soft_normalize_to_unit_circle`, which blends phase from 0 toward `angle(g)`.
**It is the wrong tool here.** `blend·angle(e^{iφ}g) ≠ φ + blend·angle(g)`, so
it is not U(1)-equivariant: phase 0 is a fixed point of the compression, which
installs a preferred direction in the complex plane. That breaks both things
this document depends on — the carrier cancellation, and the U(1) invariance of
the Hebbian that lets a spiking substrate read the gradient off relative spike
times. Fine for a feedforward activation; fatal here.

Measured on the toy chain, FD oracle using the same projection:

| projection | K=:zero | K=:stored |
|---|---|---|
| hard | 0.023 | 0.011 |
| soft, phase-interpolating | **0.198** | **0.318** |
| soft, `g/sqrt(abs2(g)+ε²)` | 0.032 | 0.016 |

So `_project_damp_soft` uses `u = g/sqrt(abs2(g)+ε²)`, equivariant by
construction. It does not force |z| = 1 — a weakly driven neuron settles to a
small amplitude (measured |z| ≈ 0.99) instead of snapping to 1+0im in an
arbitrary direction, and the Hebbian then weights by amplitude, which is the
sensible reading of a neuron that barely fired.

Whether this removes the bimodal failures is what the `project` sweep axis
measures. Its median fidelity is slightly *worse* than hard; the claim is about
the tail, so the number to watch is `fail%`, not the median.

## Cost, in the unit that matters

Settle steps are the wrong unit for a rule meant to run on live hardware; what a
physical network spends is **carrier cycles**. With `t_period = 1`:

- R_relax ≈ 0.08–0.13 ⟹ the network relaxes in **8–12 carrier cycles**
- ω_p = 0.02 ⟹ one probe period is **~314 carrier cycles**
- n_cycles = 4 plus 1 warmup cycle, plus the 200-step free settle ⟹
  **~1670 carrier cycles per gradient** ((1+4)·314 + 200·0.5)

At a 1 kHz carrier that is ~1.7 s per gradient. The sweep now reports this
alongside step counts.

## Post-mortem: the prototype

An earlier attempt shipped a 120-cell sweep (`results/ep_adiabatic_rot/`) in
which every cell failed, and the failure was reported as physics. It was not.

1. **`rotate_frame=true` was a no-op.** The branch set `w_self_local = z_self`
   and never unrotated anything, then ran the identical drive / feedback / nudge
   / projection. Bit-identical to `rotate_frame=false`.
2. **ω was never in the settle.** Everything ran `K_mode=:zero`, which returns
   before λ or ω are read — so there was no rotation to remove, which is why a
   no-op could look self-consistent.
3. **The demodulator searched an empty band.** It accumulated at ω_p ± 2π while
   the only AC content was at ω_p; and at dt = 0.5 the reference
   `exp(-i·2π·t·0.5)` alternates ±1, fully aliased.
4. **The grid never read its own sweep variables.** `stage_grid` under `ROTATE`
   did not call `ep_gradient`; `(ε, ω_p, n_cycles)` were written to CSV and
   otherwise unused. The data showed it: 100 rows, `steps=200` constant, and 8
   unique cosines for 8 replicates — the result was a function of `rep` alone.
5. **The reference was not a gradient.** `unified_reference` returned a single
   free-phase Hebbian — no nudge, no differencing, no 1/β — with `invB` applied
   twice to the bias and a `transpose`/`adjoint` inversion via
   `unrotate_solution` (which returns `conj(w)`, not `w`).
6. **Stage 0 never ran.** `stage_variance` under `ROTATE` referenced an
   undefined `cost_ref`. The design doc's own gate was skipped.

The prototype also broke **8 existing EP tests** — the entire `LockinEP` group
plus a batched case — which stood as a direct, already-available refutation of
the dual-sideband change. The suite now passes 127/127.

The deeper problem was one of allocation: ω is the gap that is provably free,
while λ, the hard projection, and spike-time readout are the gaps that actually
separate `phasor_settle` from a running spiking network. The effort went to the
free one.

## Harness changes

- Every CSV row carries `gitrev`. The schema guard catches a *changed* set of
  columns; it cannot catch the same columns filled by different code, which is
  what put two mutually inconsistent code versions in one results directory.
  `read_rows` now warns when a file spans commits.
- The grid records a **settle-stationarity residual** per point. Above a locking
  threshold there is no fixed point, only drift, and a cosine against a drifting
  snapshot reads as noise — indistinguishable from "the estimator is broken"
  unless this is recorded next to it.

## Deliberately not done

**Detuning.** `ω_l = ω̄ + Δω_l` (device mismatch) leaves a residual `i·Δω_l·w_l`
after the frame transform, which is genuinely non-gradient. The new dimensionless
group is `Δω/R_relax` and the predicted boundary is an Adler/Kuramoto
injection-locking threshold rather than an adiabaticity limit. This is the only
direction with new *theory*; the readout floor above is the one with new
*engineering*. Flagged rather than assumed.
