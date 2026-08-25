# Open questions after the rotating-substrate result

> **Derivation:** `docs/phasor_lockin_derivation.tex` §Rotating Substrate &nbsp;|&nbsp;
> **Narrative:** `docs/ep_rotating_extension.md` &nbsp;|&nbsp;
> **Measurements:** `results/ep_readout_floor/FINDINGS.md` &nbsp;|&nbsp;
> **Gates:** `scripts/ep_rotating_gates.jl`

## Where things stand

Two results bound what follows.

1. **The carrier is free.** For a single shared ω the rotation cancels
   exactly — continuous and discrete, at any `dt` — so the rotating problem
   *is* the static problem and the existing zone map already applies. Nothing
   further to measure here.
2. **The spike-time readout is not free.** A deterministic phase quantum of
   0.005 turns (1.8°) takes best-achievable fidelity from 0.999 to 0.07, with
   100% failure at every (ε, ω_p, n_cycles). Dithering roughly doubles that and
   no more; a smoother projection helps the tail by ~8 points of fail rate. The
   window for a spiking implementation is **closed** at the package's default
   `t_window/t_period = 0.01`.

So the open work is not about ω. It is about the readout, and about the one
place where a rotating substrate genuinely does change the theory.

---

## 1. Beat the readout floor, or establish that it cannot be beaten

The binding question. Everything in Result 2 follows from needing the probe
response `ε·χ` to exceed one phase bin, and from a deterministic quantizer
having a dead zone: if the state never crosses a bin boundary, the demodulated
sum is *identically* zero, and no amount of averaging recovers a signal that
never moved.

Three routes, in rough order of expected value.

### 1a. Population / multi-spike coding *(most promising)*

A single neuron's phase is resolvable to ~`t_window`. A *population* of N
neurons carrying the same phase is not obviously limited the same way: if their
timing errors are independent, the population mean has resolution ~`t_window/√N`.
The lock-in already sums over channels when it forms the Hebbian, so some of
this may be available for free.

**Test:** the readout floor should then scale as `δ_eff = δ/√d` with layer
width `d`. Re-run the `readout` axis at 64 / 256 / 1024 hidden units and check
whether the failure contour moves as `√d`. Cheap — the harness already sweeps
readout, and `EPS_HID` is already a knob.

**What would falsify it:** the contour not moving with width, which would mean
the errors are common-mode (they are correlated through the shared drive) and
the floor is per-layer rather than per-neuron.

### 1b. An estimator that does not resolve sub-degree phase

Everything about the floor is downstream of *what* is being measured. Lock-in
EP needs a small AC phase shift; a rule reading coincidence counts, spike-count
rates, or spike-order statistics would not inherit this constraint at all. This
is a larger design question than a sweep, but it is the one that follows most
directly from the negative result — the estimator, not the substrate, is what
makes the floor binding.

### 1c. Extend the linear range so a larger ε is admissible

Ruled out as stated: the floor needs ε ≫ 0.3, and 0.3 is already where the hard
projection basin-hops. The soft projection moved fail% from 31% to 19% at
ε = 0.3, which is real but leaves the estimator far from usable. Worth
revisiting only if 1a or 1b lowers the floor enough to make the two ranges
overlap.

### Also worth settling cheaply

- **Is quantizing the readout the right model at all?** The current model
  quantizes what the synapse observes while leaving the dynamics analog. The
  stronger, more faithful model quantizes inter-layer *communication* too —
  the downstream layer sees spikes, not a complex number. That is a strictly
  harder constraint and it is not yet known whether the settle even converges
  under it.
- **Sampling rate vs. resolution.** A real neuron reports once per carrier
  cycle; the current model re-reads every settle step (2 per cycle at
  `dt = 0.5`). Close enough that it probably does not matter, but it is one
  line to check and would remove an assumption.

---

## 2. Detuning: the only place the theory actually changes

With per-channel carriers `ω_c = ω̄ + Δω_c`, the frame transform at `ω̄` leaves
a residual `i·Δω_c·w_c` that is tangential to the torus and non-variational —
the obstruction identified in the derivation. No global frame removes it.

The prediction is that the system still has a relative equilibrium as long as
the inter-layer coupling phase-locks the detuned channels, making the boundary
an **Adler/Kuramoto locking threshold** in `Δω / R_relax`, not an adiabaticity
limit. That is a qualitatively different kind of boundary from anything the
existing sweep maps, and it is the one genuinely new piece of theory a rotating
substrate demands.

**Why it matters practically:** device mismatch is not optional in analog
neuromorphic hardware. A tolerance figure in `Δω/ω` is exactly what a hardware
designer needs, and nothing currently produces one.

**How to run it:** add a per-channel `Δω` to the settle (the carrier is
currently a scalar) and sweep `Δω/R_relax ∈ {0, 0.1, 0.3, 1, 3}`. Record the
**stationarity residual** alongside cosine — above threshold there is no fixed
point, only drift, and a cosine against a drifting snapshot reads as noise,
which is indistinguishable from "the estimator is broken" without it. The
residual column already exists in the harness for exactly this reason.

**Expected shape:** fidelity flat below threshold, collapsing above it, with
the threshold scaling as coupling strength — i.e. as ‖W‖, which is itself known
to drift during training.

---

## 3. Loose ends in what was measured

Small, cheap, and each removes a stated caveat.

- **Finish the readout pilot's expensive corner.** `ω_p = 0.005, n_cycles = 8`
  is missing (2570 of 2688 rows). It is the *most* adiabatic setting, so it
  would if anything strengthen the exact-readout row and leave the quantized
  rows unchanged — but the claim is currently stated with a gap in it.
- **Raise N on the soft-projection per-ε split.** The pooled hard-vs-soft
  comparison is n = 96 and solid; the claim that the benefit concentrates at
  large ε rests on n = 16 cells and is flagged as suggestive. This is the one
  number most worth more samples.
- **Sweep width.** The whole reason the original defaults failed is that the
  zone moves with width. Whether `ω_p/R_relax` is the right invariant — which
  would replace the sweep with a single cheap `R_relax` measurement — remains
  the highest-leverage structural question about the estimator itself, and is
  unchanged by anything here. Pairs naturally with 1a, which needs the same runs.
- **Trained weights.** Everything is at initialization. Gradient fidelity is
  known to degrade as ‖W‖ grows, so the zone at epoch 10 is probably not the
  zone at epoch 0. Needs checkpoints the current demo does not save.

---

## 4. Deliberately not pursued

- **A lab-frame sweep.** The frames are provably equivalent and the lab frame is
  strictly worse numerically (it accumulates `O(N·u)` carrier rounding the
  co-rotating frame never incurs). Its only role is verification, which
  `scripts/ep_rotating_gates.jl` already performs.
- **Tuning the dual-sideband demodulator.** In the co-rotating frame there is
  one sideband, at ω_p. The `ω ± ω_p` structure is an artifact of choosing the
  wrong frame, and chasing it was what produced 120 cells of uninterpretable
  output.
- **A GP surrogate over the grid.** Still only justified as a post-hoc contour
  smoother, and the current contours are not close enough to a boundary for
  smoothing to be the limiting factor.
