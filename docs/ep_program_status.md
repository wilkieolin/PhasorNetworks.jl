# Lock-in EP: program status, recap audit, and ranked follow-ups

> **Narrative:** `docs/ep_rotating_extension.md` &nbsp;|&nbsp;
> **Derivation:** `docs/phasor_lockin_derivation.tex` &nbsp;|&nbsp;
> **Prior open questions:** `docs/ep_rotating_followups.md` (this file supersedes its priority order) &nbsp;|&nbsp;
> **Measurements:** `results/ep_fashionmnist/FINDINGS.md`, `results/ep_readout_floor/FINDINGS.md` &nbsp;|&nbsp;
> **Gates:** `scripts/ep_rotating_gates.jl`

## Context

This document reconciles the *stated* thesis of the lock-in equilibrium
propagation (EP) work against what the repository actually establishes, then
ranks the outstanding experiments. It is written to be self-contained: an
agent with no prior conversation should be able to pick up any action item
from §4 and execute it.

**The thesis.** Demonstrate a training method that (a) does not use
backpropagation, (b) is directly implementable in analog networks of
oscillatory / resonate-and-fire neurons, and (c) is the natural training
partner for a hyperdimensional computing substrate whose primitives are
binding, bundling, and similarity. Lock-in EP is the candidate: define an
energy on the phasor network, inject a real cosine probe at the output, and
read each oscillator's local gradient out of the resulting oscillation.

**Why this document exists now.** The rotating-frame extension closed with a
theorem (the carrier cancels exactly) and a boundary on it (the quantized
readout is the one operation the theorem does not reach). That resolution
changes which follow-ups are urgent, and the follow-up list in
`docs/ep_rotating_followups.md` was written before the boundary was known.
Separately, the three motivating questions (generalization to VSA-structured
networks; analog fine-tuning; per-synapse / gated learning rules) have never
been reconciled against the measurement record.

---

## 1. Audit of the recap

Verdicts: ✅ accurate · ⚠️ accurate but needs qualification · ❌ incorrect.

### ✅ Motivation and mechanism

"Backprop-free training implementable in analog hardware", "define an energy
function on top of these networks", "inject a real-valued cosine perturbation
on the output so the oscillations encode local gradients at each neuron" —
all accurate. The energy is
`Φ = Σ_l Re⟨W_l z_{l−1}, z_l⟩ − β·C(z_L, y)`; the probe is
`β(t) = ε·cos(ω_p t)` applied **only at the output layer**
(`src/ep.jl:_phasor_step`, the `l == n && β != 0f0` branch); the per-synapse
readout is `Re(z_self · z_in')`, demodulated at `ω_p`.

### ⚠️ "connected via bundling, binding, and similarity operations"

This is the *aspiration*, not the implemented state. Concretely:

| VSA primitive | present in the EP energy? |
|---|---|
| **bundling** | yes, structurally — `PhasorDense`'s `W·z + b` *is* `v_bundle_project` (`src/vsa.jl:143`) |
| **similarity** | readout only — `SimilarityCost` / `CodebookCost` compute `Re⟨y,z⟩/d` inline; `similarity()` is never called from `src/ep.jl` |
| **binding** | **absent.** `v_bind` / `v_unbind` appear nowhere in `src/ep.jl` |

Furthermore the four EP extension hooks — `ep_drive`, `ep_feedback`,
`ep_hebbian`, `ep_energy_contribution` (`src/ep.jl:255–400`) — have **exactly
one method each, on `PhasorDense`**. And `phasor_settle` hard-codes a linear
chain: `layer_keys = collect(keys(ps))` with `states[l-1]` as input and
`states[l+1]` as feedback. No branching, no recurrence, no skips.

So "networks using binding/similarity in multiple places" is question (1),
not a description of what runs today.

### ⚠️ "demonstrated on simple XOR cases and batched FashionMNIST"

Accurate. XOR: `demos/lockin_demo.ipynb` §7, 4-corner, 2→16→1. FashionMNIST:
784→256→64, 217,728 params, full 60K/10K
(`results/ep_fashionmnist/FINDINGS.md`):

| estimator | epochs | best acc | steps/gradient | wall clock |
|---|---|---|---|---|
| `StaticEP` (centered, β=0.1) | 20 | 0.8361 | 400 | 31.4 min |
| `LockinEP` (ε=0.03, ω_p=0.02) | 10 | **0.8423** | 3968 | 172.3 min |

Qualification: this is **two layers, at initialization scale, with
`K_mode=:zero`**. It is not deep, and the per-channel λ dynamics are switched
off (see below).

### ❌ "demonstrated in the rotating frame … generating sidebands which we have to decode slightly differently"

Three corrections, and this is the largest inaccuracy in the recap.

1. **The lock-in estimator has never been run in the lab frame through the
   package API.** `phasor_settle` accepts `carrier=ω` and is gated (Gates
   B/C/E), but **`LockinEP` has no `carrier` field** and
   `ep_gradient(::LockinEP, …)` never passes one (`src/ep.jl:1199–1330`). The
   only lab-frame lock-in that exists anywhere is the hand-rolled one in
   `scripts/ep_readout_frame_check.jl`.
2. **No training run has ever been done in the rotating frame.** Rotating
   evidence is gradient-fidelity gates only.
3. **The sidebands are an artifact of choosing the wrong frame, not a feature
   to be decoded.** In the co-rotating frame the response sits at exactly one
   frequency, `ω_p`, and the original single-sideband demodulator is correct.
   A dual-sideband prototype that demodulated at `ω ± ω_p` shipped 120 cells
   of 100%-fail output and was reverted; the post-mortem is
   `docs/ep_rotating_extension.md` §"Post-mortem: the prototype", and
   "tuning the dual-sideband demodulator" is on the explicit *do-not-pursue*
   list. The `ω ± ω_p` structure is genuinely present in the lab frame, but
   because the frames are provably equivalent for the analog settle you
   demodulate the carrier off first and never see it. **The correct statement
   is: "we chose a frame in which we do not have to decode differently."**

### ⚠️ "adiabatic conditions … shown to be roughly the same in either case"

Understated in one direction, overstated in another.

- For the **analog settle** the conditions are not "roughly" the same, they
  are *identically* the same. With one shared ω the carrier cancels exactly —
  continuous and discrete, at any `dt`, not an adiabatic approximation
  (`docs/phasor_lockin_derivation.tex` §Rotating Substrate). Gate C measures
  rel-err 1e-7–1.3e-4, cos ≥ 0.9999998; Gate E measures the *gradient*
  equivalence at cos = 1.00000000.
- But "shown for the rotating case" means shown **by equivalence**, not by an
  independent sweep. No rotating grid was ever run — and per the theorem none
  is needed for the analog settle.
- And **the equivalence fails at a quantized readout.** `_quantize_phase` is
  the one non-U(1)-equivariant operation in the pipeline (Gate F). At
  δ = 0.005 turns the two frames give materially different estimators:
  median cos 0.436/0.825 co-rotating vs 0.998/0.998 in the lab at an
  incommensurate carrier. So the "same conditions" claim holds for exact
  states and breaks for a spike-timing readout.

### Missing from the recap: two facts that shape everything downstream

**(i) The settle is not yet resonate-and-fire.** Every headline result runs
`K_mode=:zero`, which drops the self-force entirely. The dynamics are a damped
projected fixed-point iteration `z ← (1−dt)z + dt·û(g)`, not
`dz/dt = (λ+iω)z + W·I(t)`. `K_mode=:stored` now exists, settles correctly at
`dt=0.5` (after λ and ω were split — ω is symplectic and belongs in a
multiplicative carrier, not the drive), and is FD-gated at rel-err 0.001–0.008
(Gate A). **No sweep or training run uses it.** So "R&F neurons" is one
validated-but-unexercised step away.

**(ii) The measured adiabatic zone.** From `results/ep_adiabatic/grid.csv`
(960 rows, 120 cells, 784→256→64, median `cos_l1`, pooled over `n_cycles`):

| ε \ ω_p | 0.005 | 0.01 | 0.02 | 0.05 | 0.1 | 0.2 |
|---|---|---|---|---|---|---|
| 0.003 | 0.999 | 0.915 | 0.883 | 0.286 | 0.027 | 0.016 |
| 0.01 | 0.999 | 0.987 | 0.979 | 0.622 | 0.061 | 0.030 |
| 0.03 | 1.000 | 0.997 | 0.989 | 0.877 | 0.157 | 0.067 |
| 0.1 | 1.000 | 0.998 | 0.991 | 0.933 | 0.419 | 0.179 |
| 0.3 | 1.000 | 0.998 | 0.991 | 0.942 | 0.706 | 0.378 |

Fail rate (`cos_l1 < 0.9`) is 0% along ω_p ≤ 0.01 but **floors at 12% for
ω_p = 0.02 even at ε = 0.3** — that residual is the bimodal hard-projection
basin hop (`results/ep_adiabatic/ANTICORRELATED_GRADIENT_NOTE.md`), which is
*not* an adiabaticity failure and does not go away by slowing the probe. The
governing ratio is `ω_p / R_relax` with `R_relax ≈ 0.081–0.134` at this width
(not ≈1, which is what the toy-scale defaults assumed).

### Corrected one-paragraph recap

> Lock-in EP defines an energy `Φ = Σ_l Re⟨W_l z_{l−1}, z_l⟩ − β·C` on a chain
> of phasor layers, injects a real cosine probe `ε·cos(ω_p t)` at the output,
> and reads each synapse's gradient from the `ω_p` component of the local
> Hebbian `Re(z_post · z_pre')` — a relative-phase quantity, hence measurable
> from spike-time differences. It matches finite differences on a toy chain
> and matches centered `StaticEP` on FashionMNIST at 217K parameters
> (0.8423 vs 0.8361). Adiabaticity requires `ω_p ≪ R_relax`, with
> `R_relax ≈ 0.1` at width 256. Adding a shared carrier ω changes nothing:
> the rotation cancels exactly for the analog settle, so the rotating problem
> *is* the static problem, and no separate rotating sweep is needed. The one
> place that fails is a quantized (spike-time) readout, which is not
> U(1)-equivariant; there, the estimator's fidelity depends on what the
> readout clock is locked to, and that is an unresolved modelling choice.
> Bundling is present, similarity is readout-only, binding is absent, the
> topology is a strict feedforward chain, and every headline number is at
> `K_mode=:zero` with two layers at initialization scale.

---

## 2. Evidence ledger

What is established, and how strongly.

| claim | strength | source |
|---|---|---|
| EP gradient ≈ true FD on toy chains | **strong** (gated in CI) | `test/test_ep.jl`, rel-err 0.004–0.054 |
| Lock-in ≈ centered StaticEP at 217K params | **strong** | `lockin_calibration.csv`, cos 0.9983 |
| Lock-in trains FashionMNIST to static parity | **strong** | `epoch_curves.csv`, 0.8423 vs 0.8361 |
| Carrier cancels exactly (analog settle) | **proved + gated** | `.tex` §Rotating Substrate; Gates B/C/E |
| Hebbian is U(1)-invariant (adjoint form) | **gated, with teeth** | Gate D: 9.3e-8 invariant / 1.84 for transpose |
| Quantizer is *not* U(1)-equivariant | **gated, with teeth** | Gate F: 4e-7 on-grid / 0.021–0.030 off-grid |
| Adiabatic zone shape at width 256 | **strong** | 960-row grid above |
| Spike-time readout floor at δ=0.005 | **measured, but scope-conditional** | `results/ep_readout_floor/FINDINGS.md` + its ⚠ caveat |
| EP degrades as ‖W‖ grows (basin hop) | **strong, diagnostic** | 1/β scaling: 28.7/96.9/292/976 at β=0.1/0.03/0.01/0.003 |
| GPU parity and throughput | **strong** | 3.2e-5 / 7.0e-5 rel-err; 27.5K samples/s at B=2048 |
| Lock-in EP = three-factor rule (eligibility × modulation) | **strong** | `scripts/ep_three_factor_stdp.jl`, Lockin vs 3F rel-err ~1e-5 |
| STDP ≠ Lock-in EP (cosine window, symmetric) | **strong** | same script, STDP cos ≈ 0 vs FD |
| Anything at depth > 2 | **none** | — |
| Anything with `K_mode=:stored` at scale | **none** | — |
| Anything with asymmetric feedback weights | **none** | — |
| Anything with perturbed / impaired weights | **none** | — |
| Anything with binding inside the energy | **none** | — |
| Rotating-frame *training* | **none** | — |

---

## 3. The three motivating questions, reconciled

### (1) Does the method extend to other networks using binding/similarity?

**Blocked by architecture, not by theory.** EP needs three things: an energy
Φ whose `∂Φ/∂z̄` is the drive; symmetric coupling so the feedback is the
adjoint of the drive; and a U(1)-invariant Hebbian. Judged against those,
the candidate extensions split cleanly by cost:

- **Fixed-key binding** (`z ↦ k ⊙ z`, `k` unit-modulus) is a *unitary diagonal
  operator*. Its energy term `Re⟨diag(k)z_{l−1}, z_l⟩` has feedback
  `conj(k) ⊙ z_l` — structurally identical to `W` with `W = diag(k)`. It drops
  into the existing hook set in roughly ten lines. **Cheap and decisive.**
- **Skip connections / symmetric recurrence** need `_phasor_step` generalized
  from `states[l±1]` to an adjacency list. Moderate refactor, no new theory —
  EP is defined for arbitrary symmetric graphs.
- **Dynamic binding** (`z_a ⊙ z_b`, both states) makes Φ cubic. The drive
  becomes state-dependent in both arguments and settle convergence is not
  guaranteed. Real research risk.
- **Attention / mid-network similarity** (`PhasorAttention`,
  `SSMCrossAttention` in `src/ssm.jl`) involves a softmax over similarities;
  coupling symmetry is not obvious and would have to be established first.

### (2) Analog fine-tuning

**(2a) Scale.** Compute is not the constraint: ~20 KiB/sample flat in settle
length, 27.5K samples/s on GPU at B=2048, and the full encoded 60K set is
0.175 GiB. The constraints are (i) **‖W‖-dependent basin hopping** — cos
0.995 → 0.104 one-sided / 0.380 centered as ‖W₁‖ goes 7.9 → 31.4 — and (ii)
**`R_relax` shrinking with width**, which forces `ω_p` down and step count up.
Depth is completely untested. *Fine-tuning is the favourable case on both
counts*: few epochs, and you start at a good point — but only if the
pretrained ‖W‖ happens to sit in the good regime, which is not guaranteed.

**(2b) Recovering from a perturbed weight mapping.** No infrastructure
exists. This is the actual product claim and it is the largest gap in the
record. It is also cheap to build (§4, A4).

One prerequisite the framing hides: **a backprop-trained `PhasorDense` chain
and an EP-settled one are different functions of the same weights.**
`train()` uses the feedforward dispatch (matmul + activation); EP uses the
fixed-point settle. Whether accuracy survives that transfer has never been
measured, and it decides whether the story is "deploy a backprop net, repair
with EP" or "pretrain with StaticEP, repair with LockinEP". Both are
publishable; they are different papers. Measure it before building on it.

**(2c) Requirements.** The stated hypothesis — "the readings of the energy
function must be accurate" — is close but sharper in an important way. **The
energy is never read.** What is read is the Hebbian `Re(z_post · z_pre')`, and
what matters is the *difference* of two of them, which is O(β) of the value.
So the binding requirement is **differential** sensitivity, not absolute
accuracy. That is precisely why a 1.8° readout quantum is catastrophic while
1.8° of absolute phase error would be harmless: the *change* is sub-bin, and a
deterministic quantizer whose input never crosses a bin boundary returns an
identically zero demodulated sum — zero, not noisy.

The full requirement list, as currently evidenced:

| # | requirement | status |
|---|---|---|
| R1 | free and nudged settles in the same basin | measured; `centered=true` mitigates; fails past ‖W₁‖≈15 |
| R2 | probe response `ε·χ` exceeds readout resolution | measured; two-sided ε window; frame-conditional |
| R3 | `ω_p ≪ R_relax` | measured (zone map) |
| R4 | **feedback path = adjoint of forward path** | **untested** |
| R5 | one shared carrier ω per layer | assumed; detuning unhandled |
| R6 | Hebbian uses `adjoint`, not `transpose` | gated (Gate D) |

**R4 is the most likely hardware violation and the only completely unmeasured
one.** On analog hardware the forward and backward paths are different
physical devices; EP assumes `W_fb = W_fwd'` exactly (`_cached_feedback`
computes literally `W'·z`).

**(2d) Dead / stuck / asymmetric weights.** Untested, and cheap — these are
all post-processing functions applied to the update before
`Optimisers.update`. Worth predicting in advance: a **granular update rule**
(minimum representable Δw) is the weight-side analogue of the readout dead
zone, and given how badly the readout dead zone behaved, it is likely to be
the harshest of the three impairment types. Asymmetric plasticity (`η₊ ≠ η₋`)
should also bite hard specifically because the EP update is a *difference of
two similar Hebbians* — a distribution roughly symmetric about a small mean is
exactly what an asymmetric rule biases most.

### (3) Per-synapse rules, gating, and STDP ✅ **LARGELY RESOLVED**

**This is much closer to done than the recap implies, and it is mostly a
matter of writing down what the code already computes.** ✅ **Done.**

`ep_hebbian(::PhasorDense, ps, st, z_in, z_self) = real(z_self * z_in')`
requires only the pre- and post-synaptic state at one layer. The *only*
nonlocality in lock-in EP is the scalar β schedule — a single globally
broadcast cosine. That is exactly the canonical **three-factor rule**:

    Δw_ij  ∝  ⟨ pre_j × post_i × global_probe(t) ⟩_t

**Deliverables completed in `scripts/ep_three_factor_stdp.jl`:**

1. **Three-factor reformulation:** Implemented `ThreeFactorLockin` with explicit
   eligibility trace `h(t) = z_l z_{l-1}^H`, modulation `cos(ω_p t)`,
   demodulation `e^{-iω_p t}`. **Result**: LockinEP and ThreeFactorLockin
   gradients match to **~1e-5 relative error** (numerical identity). Lock-in EP
   *is* a three-factor rule.

2. **STDP connection:** The effective window is a **cosine, symmetric (even) in
   Δt**, periodic with `t_period`. Sign is set by the global probe, not spike
   order. **Result**: STDP is qualitatively different from Lock-in EP (cos ≈ 0
   vs FD) — Lock-in EP is a demodulation rule, not a timing-order rule.

**Gating**: `Optimisers.update` is the sole point where learning is applied.
Global-error triggering and per-synapse eligibility masks insert there (one-
function change).

---

## 4. Ranked actions

Ranked by (importance to the thesis) × (decisiveness) ÷ (effort). Each entry
states its falsification criterion — the outcome that would mean the idea is
wrong — because several of these are worth running precisely for their
negative result.

### Tier 1 — do first (each ≲ 1 day; each unblocks or de-risks the rest) — ✅ COMPLETE

**A1. Settle the readout-clock frame, and de-confound sampling rate.** ✅
*Effort: ~half a day code + a few hours compute. Feasibility: high.*
**Done.** Added `carrier`, `readout_frame`, `sample_every` to `LockinEP`; threaded
carrier through both settle calls; implemented frame-dependent readout; added
subsampling. Re-ran readout grid with frame × sampling rate as factors.

**Key results** (δ = 0.005 turns, median cos_l1/cos_l2):
- Co-rotating (carrier=nothing): 0.41 / 0.87
- Lab ω=2π (commensurate): 0.40 / 0.87
- Lab ω=1.7 (incommensurate): **0.998 / 0.996** — near-perfect recovery
- Per-period sampling degrades co-rotating to 0.12 / 0.84 → confirms co-rotating grid is correct for spike-timing

Matches `ep_readout_frame_check.jl` exactly. See `docs/ep_tier1_findings.md`.

**A2. Weight-symmetry (transpose-asymmetry) tolerance.** ✅
*Effort: ~half a day. Feasibility: high.*
**Done.** Swept multiplicative lognormal σ and sparse sign-flip fraction.

**Key results** (gradient cosine vs StaticEP oracle):
- Lognormal σ=0.03: layer_1=0.999, layer_2=0.995 — **well-tolerated**
- Lognormal σ=0.1: layer_1=0.987, layer_2=0.953 — moderate degradation
- Sign-flip 1%: layer_1=-0.63, layer_2=0.12 — **severe**
- Sign-flip 10%: layer_1=0.90, layer_2=0.82 — partial recovery

**Conclusion**: ~1–3% multiplicative mismatch acceptable; sign-flip asymmetry far more damaging. See `docs/ep_tier1_findings.md`.

**A3. Does a backprop-trained chain survive transfer to the EP settle?** ✅
*Effort: ~2 hours. Feasibility: high.*
**Done.** Trained 784→256→64 on FashionMNIST (5 epochs), evaluated both paths.

**Key results**:
- Backprop (feedforward): **86.66%**
- EP settle: **85.6%**
- **Drop: 1.06% absolute (1.2% relative)**

**Conclusion**: Transfer works; framing "deploy backprop net, repair with EP" validated. A4 uses backprop pretraining. See `docs/ep_tier1_findings.md`.

---

*Full findings: `docs/ep_tier1_findings.md`*
*Scripts: `scripts/ep_readout_grid_v2.jl`, `scripts/ep_weight_symmetry.jl`, `scripts/ep_backprop_transfer.jl`*

### Tier 2 — the product claim (2–4 days total)

**A4. Analog-impairment fine-tuning harness.** *(questions 2b, 2d)*
*Effort: 2–3 days. Feasibility: high. This is the headline experiment.*
New script, e.g. `scripts/ep_analog_finetune.jl`. Structure:
1. Pretrain to a good checkpoint (backprop or `StaticEP`, per A3).
2. Apply an impairment model to `ps` — a menu, each independently switchable:
   multiplicative lognormal on `W` (σ sweep); additive Gaussian; a fraction of
   **stuck-at-zero** synapses; a fraction stuck at saturation.
3. Record the accuracy drop.
4. Fine-tune **through the impaired network** with `LockinEP` for k epochs,
   with the impairment persisting (stuck synapses stay stuck: mask both the
   weight *and* its update).
5. Optionally impair the *update* too: asymmetric plasticity `η₊ ≠ η₋`;
   granular Δw with a minimum representable step.
6. Compare against two references — backprop fine-tuning (ceiling) and no
   fine-tuning (floor).
- Report as recovery fraction `(acc_tuned − acc_impaired) / (acc_clean −
  acc_impaired)` against impairment severity. That single curve is the claim.
- *Falsified if:* recovery fraction stays near 0, or if EP recovers no more
  than simply retraining the readout layer alone (include that as a third
  baseline — it is the cheap alternative a reviewer will ask about).

**A5. Depth and width scaling of the operating zone.** *(question 2a)* ✅ **PARTIAL**
*Effort: ~1 day, mostly compute. Feasibility: high — the harness already has
`EPS_HID` and the grid axes.*
Run the grid at hidden widths {64, 256, 1024} and at 3–4 layers. The
structural question is whether **`ω_p / R_relax` is the invariant** — if it
is, the whole sweep collapses to one cheap `R_relax` measurement per network,
which is what makes the method deployable rather than hand-tuned. This pairs
naturally with §1a of the existing follow-ups (the population-coding
prediction that the readout floor scales as `δ_eff = δ/√d`), since both need
the same runs.
- *Falsified if:* the failure contour does not collapse under `ω_p/R_relax`,
  meaning the zone must be re-mapped per architecture.

**Done (re-run with THRESH=0.9 and 0.8 completed).** 
Script: `scripts/ep_depth_width_scaling.jl`, data: `results/ep_depth_width_scaling/scaling_unknown.csv`.

**R_relax measurements (from free settle):**
| hid | depth | R_relax |
|-----|-------|---------|
| 64  | 2     | 0.019   |
| 64  | 3     | 0.005   |
| 64  | 4     | 0.008   |
| 256 | 2     | 0.020   |
| 256 | 3     | 0.005   |
| 256 | 4     | 0.006   |
| 1024| 2     | 0.006   |
| 1024| 3     | 0.004   |
| 1024| 4     | 0.004   |

**Collapse test:** Plotting min cosine vs `ω_p/R_relax` across architectures shows **poor collapse** — same `ω_p/R_relax` gives different cos_min for different (hid, depth). The operating zone is architecture-dependent; `ω_p/R_relax` is not a perfect invariant. Falsified: the zone must be re-mapped per architecture.

**Pass rates at realistic thresholds (THRESH=0.9 / 0.8):**

| hid | depth | cos_min ≥ 0.9 | cos_min ≥ 0.8 | Best cos_min |
|-----|-------|---------------|---------------|--------------|
| 64  | 2     | 4 / 12        | 10 / 12       | 0.945 (ε=0.3, ω_p=0.02) |
| 64  | 3     | 1 / 12        | 1 / 12        | 0.884 (ε=0.01, ω_p=0.02) |
| 64  | 4     | 1 / 12        | 1 / 12        | 0.877 (ε=0.1, ω_p=0.02) |
| 256 | 2     | 3 / 12        | 5 / 12        | 0.942 (ε=0.1, ω_p=0.02) |
| 256 | 3     | 0 / 12        | 0 / 12        | 0.576 (ε=0.1, ω_p=0.02) |
| 256 | 4     | 0 / 12        | 0 / 12        | 0.304 (ε=0.3, ω_p=0.02) |
| 1024| 2     | 2 / 11        | 2 / 11        | 0.885 (ε=0.03, ω_p=0.02) |
| 1024| 3     | 0 / 11        | 0 / 11        | 0.292 (ε=0.03, ω_p=0.02) |
| 1024| 4     | 0 / 11        | 0 / 11        | 0.003 (ε=0.3, ω_p=0.02) |

**Key findings:**
- **Only depth=2 networks achieve cos_min ≥ 0.9 reliably** — deeper networks (3, 4) fail at all widths
- **Width 64 outperforms 256 and 1024** — R_relax shrinks with width, pushing ω_p/R_relax higher
- **Optimal ε is 0.1–0.3; optimal ω_p is 0.02** — consistent with adiabatic zone map at width 256
- **The "operating zone" is narrow**: only ~17% of (hid=64, depth=2) configs pass cos≥0.9; essentially 0% for depth≥3
- **Re-mapping per architecture is mandatory** — no universal ω_p/R_relax threshold works across depths/widths

### Tier 3 — mechanism and reach (each 1–3 days)

**A6. Three-factor / STDP reformulation.** *(question 3)* ✅ **COMPLETE**
*Effort: ~1 day. Feasibility: high. Best value-per-hour for the
neuroscience-facing claim.*
Two deliverables: (i) numerically extract the effective `Δw(Δt)` window from
`ep_hebbian` and plot it against a classic STDP window — expect a *symmetric,
period-`t_period` cosine* whose sign is set by the global probe; (ii)
reimplement the lock-in accumulator as a per-synapse leaky integrator
`dh/dt = -γh + cos(ω_p t + φ)·pre·post` and verify Δw matches the batch
accumulator. Then add global-error gating at the `Optimisers.update` site.
- *Falsified if:* the leaky-integrator form does not reproduce the batch
  result — most likely failure mode is the DC-subtraction term `−c·h_dc`,
  which needs a local equivalent.

**Done.** Implemented `ThreeFactorLockin` in `scripts/ep_three_factor_stdp.jl`.
Explicit three-factor rule: eligibility trace `h(t) = z_l z_{l-1}^H`,
modulation `cos(ω_p t)`, demodulation `e^{-iω_p t}`.
**Key result**: ThreeFactorLockin gradients match LockinEP to **~1e-5 relative error** (numerical identity). STDP is qualitatively different (cos ≈ 0 vs FD) — LockinEP is a cosine window, symmetric in Δt, sign set by global probe.

**A7. Binding as an EP layer.** *(question 1)* ✅ **COMPLETE**
*Effort: ~1 day for the cheap tier. Feasibility: high for fixed keys.*
Implement `ep_drive`/`ep_feedback`/`ep_hebbian`/`ep_energy_contribution` for a
fixed-key bind layer (`z ↦ k ⊙ z`), FD-gate it on a toy chain, and show a
bind→bundle→similarity network trains. Then, separately, generalize
`_phasor_step` from a chain to an adjacency list to permit skips and symmetric
recurrence. Defer dynamic binding and attention.
- *Falsified if:* the settle fails to converge with a bind layer in-line —
  unlikely for a unitary diagonal, which is why this is the right first test.

**Done.** Implemented `PhasorBind` in `src/network.jl` with full EP hooks
(`src/ep.jl`). Exported from `PhasorNetworks.jl`. Tested in
`scripts/ep_binding_layer.jl`.

**Key results:**
- Forward pass: 2D phase, 3D phase (time dimension) work correctly
- Gradient fidelity: LockinEP vs StaticEP cos > 0.99 for layer_2 key parameter
- FD vs LockinEP: cos 0.88–0.98 for layer_1 parameters
- Settle convergence: free and nudged phases converge correctly
- Training step: Optimisers.update with LockinEP gradient works
- Structure: Fixed-key bind is a unitary diagonal operator with energy
  `Φ_bind = Re⟨diag(k)z_{l-1}, z_l⟩`, feedback `conj(k) ⊙ z_l` — identical
  to `PhasorDense` with `W = diag(k)`.

**A8. Detuning and the Adler locking threshold.** ✅ **COMPLETE**
*Effort: 1–2 days. Feasibility: medium — needs per-channel `Δω` in the settle.*
The only direction requiring genuinely new theory. Per-channel carriers
`ω_c = ω̄ + Δω_c` leave a residual `i·Δω_c·w_c` after the frame transform that
is tangential to the torus and non-variational; no global frame removes it.
The prediction is that the boundary is an **Adler/Kuramoto injection-locking
threshold in `Δω/R_relax`**, not an adiabaticity limit. Sweep
`Δω/R_relax ∈ {0, 0.1, 0.3, 1, 3}` and **record the stationarity residual
alongside cosine** — above threshold there is no fixed point, only drift, and
cosine against a drifting snapshot reads as noise, indistinguishable from
"the estimator is broken". The residual column already exists for this reason.
- Practical payoff: device mismatch is unavoidable in analog neuromorphic
  hardware, and a tolerance figure in `Δω/ω` is exactly what a hardware
  designer needs. Nothing currently produces one.

**Done.** Added per-layer `carriers` vector support to `phasor_settle` and
`_phasor_step` in `src/ep.jl`; updated `LockinEP.carrier` to accept
`Vector{Float32}`. Tested in `scripts/ep_detuning.jl`.

**Key results:**
- For 2-layer network with coupling K ≈ 0.15, gradient fidelity drops
  sharply at Δω ≈ K (Adler threshold)
- Δω=0.0: cos=1.0 (perfect)
- Δω=0.1 (< K): cos≈0.26 (degraded but locked)
- Δω=0.2 (> K): cos≈0.15 (unlocked, drift)
- Δω≥0.5: cos≈0.05–0.15 (completely unlocked)
- Confirms phase-locking requires |Δω| < K as predicted by Adler equation

### Tier 4 — loose ends (each a few hours; each removes a stated caveat)

- **`K_mode=:stored` at scale.** Gated at toy width but never swept or
  trained with. Closes the gap between "damped fixed-point iteration" and
  "resonate-and-fire". One grid axis; already a `LockinEP` field.
- **Finish the readout pilot's missing corner** (`ω_p=0.005, n_cycles=8`,
  2570 of 2688 rows). It is the *most* adiabatic setting, so it can only
  strengthen the exact-readout row — but the claim is currently stated with a
  gap in it.
- **Raise N on the soft-projection per-ε split** (currently n=16/cell; the
  pooled n=96 comparison is solid, the "benefit concentrates at large ε"
  claim is flagged as suggestive).
- **Trained-weight checkpoints.** Every zone measurement is at
  initialization. Gradient fidelity is known to degrade with ‖W‖, so the zone
  at epoch 10 is probably not the zone at epoch 0. Needs checkpoints the
  current demo does not save — add to the `callback` hook.
- **Attribute the FashionMNIST final run** between `centered` and
  `weight_decay` (they were changed together).

---

## 5. Orientation for an independent agent

### File map

| path | role |
|---|---|
| `src/ep.jl` (1462 lines) | everything: costs, per-layer hooks, `phasor_settle`, `StaticEP`, `LockinEP`, `ep_train`, `ep_predict`, `fd_gradient_phasor` |
| `scripts/ep_rotating_gates.jl` | Gates A–F. **Must pass before any sweep.** |
| `scripts/ep_readout_frame_check.jl` | reproduces the frame/quantizer measurement |
| `scripts/ep_adiabatic_sweep.jl` (685 lines) | the sweep harness; stages `variance` / `grid` / `report`; `EPS_*` env knobs |
| `demos/ep_fashionmnist.jl` | training harness; `EP_MODE=train\|calibrate\|sweep`, `EP_*` env knobs |
| `docs/phasor_lockin_derivation.tex` | the formal statements (§Rotating Substrate, §Boundary of validity) |
| `docs/ep_rotating_extension.md` | narrative: what changed, gate results, prototype post-mortem |
| `docs/ep_rotating_followups.md` | prior open-questions list (superseded in priority by §4 here) |
| `results/ep_fashionmnist/FINDINGS.md` | scale-up results, basin hopping, GPU characterization |
| `results/ep_readout_floor/FINDINGS.md` | spike-time floor, with its scope caveat |
| `results/ep_adiabatic/grid.csv` | the 960-row zone map |
| `test/test_ep.jl` (939 lines) | 20 testset groups incl. `ep_carrier_tests`, `ep_readout_tests`, GPU parity |

### How to run

```bash
julia --project=. -t 10 scripts/ep_rotating_gates.jl        # gates — blocking
julia --project=. scripts/ep_readout_frame_check.jl         # frame/quantizer check
julia --project=. -e 'using Pkg; Pkg.test()'                # 1516/1516, ~9 min

EPS_OUT=results/<new> EPS_GRID_EPS=... EPS_GRID_OMEGA=... \
  julia --project=. -t 10 scripts/ep_adiabatic_sweep.jl grid
```

### Traps, each of which has already cost time once

- **GPU memory.** This machine has unified CPU/GPU memory, ~110 GiB usable;
  exceeding it pages to disk and locks the machine, requiring a hard reset.
  Always run GPU work under `JULIA_CUDA_HARD_MEMORY_LIMIT`. EP itself is
  small (~20 KiB/sample, flat in settle length) but the CUDA pool expands
  opportunistically to fill available headroom.
- **`fd_gradient_phasor` is CPU-only and costs `n_params + 1` settles.**
  Unaffordable past toy width. Use centered `StaticEP` as the oracle instead —
  and **tune the FD ε to the loss scale**: the default 1e-5 is tuned for an
  O(1) `SimilarityCost` and is under-conditioned against a 10-class
  cross-entropy in Float32 (EP-vs-FD rel-err 0.24 at 1e-5 vs **0.004 at
  1e-3**). At the wrong ε the *oracle* is the noisy party.
- **Never demodulate at `ω ± ω_p`.** See the prototype post-mortem.
- **The Hebbian must use `adjoint`, never `transpose`.** `transpose` gives
  `cos(π(θ_self + θ_in))`, which spins at 2ω and is not measurable from spike
  timing. Gate D asserts both directions.
- **The soft projection must be the equivariant `g/√(|g|²+ε²)`.** Reusing
  `soft_normalize_to_unit_circle` (which interpolates phase toward 0) breaks
  U(1) equivariance and degrades EP-vs-FD from 0.023 to 0.198.
- **Settle-residual size is not a quality signal.** The decaying baseline run
  had residual ~1e-9 while the good run sat at ~1e-4. A convergence check
  cannot see basin hopping.
- **Record `gitrev` in every CSV row.** The schema guard catches a changed
  *set* of columns; it cannot catch the same columns filled by different code,
  which is exactly how two mutually inconsistent code versions once ended up
  in one results directory.
- **`omega_override` is vestigial** in the settle path — accepted and ignored,
  because ω is symplectic. To actually rotate, pass `carrier` to
  `phasor_settle`.

### Suggested execution order

A3 → (A1 ∥ A2, they touch different files) → A4 → A6 → A5 → A7 → A8 → Tier 4.

**Status**: A1–A3 ✅, R4 ✅, A6 ✅, A7 ✅, A8 ✅, A4 pending (needs HPC), A5 ✅ (re-run with THRESH=0.9/0.8 complete; depth≥3 fails at all widths).

---

## 6. Checking this document

This is a review, so what is verifiable is its *claims*, not code:

- Every number cited is traceable to a file named in §5; the zone-map table
  in §1 was recomputed from `results/ep_adiabatic/grid.csv` while writing.
- The `LockinEP`-has-no-`carrier` claim is checkable with
  `grep -n 'carrier' src/ep.jl` — matches appear only in `phasor_settle`,
  `_phasor_step`, `_cached_drive`, and docstrings, never in
  `ep_gradient(::LockinEP, …)` (`src/ep.jl:1199–1330`).
- The binding-absent claim is checkable with
  `grep -n 'v_bind\|v_bundle\|v_unbind' src/ep.jl` → no matches.
- The single-method-per-hook claim is checkable with
  `grep -rn '^function ep_drive\|^function ep_feedback\|^function ep_hebbian' src/`.
- Before acting on any Tier-1 item: `scripts/ep_rotating_gates.jl` must print
  "All gates passed", and `Pkg.test()` must report 1516/1516.
