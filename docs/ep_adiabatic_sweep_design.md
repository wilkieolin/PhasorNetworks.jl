# Mapping the LockinEP adiabatic operating zone — sweep design

> **Harness:** `scripts/ep_adiabatic_sweep.jl` &nbsp;|&nbsp;
> **Output:** `results/ep_adiabatic/` &nbsp;|&nbsp;
> **Background:** `results/ep_fashionmnist/FINDINGS.md`

## Why

`LockinEP`'s defaults (`ε=0.05, ω_p=0.05, n_cycles=8, dt=0.1`) were tuned on a
4→8→2 chain. At 784→256→64 they are **not adiabatic**: the fitted relaxation
rate is `R_relax ≈ 0.08–0.13 /time-unit`, not the ≈1 a naive reading of the
damped iteration suggests, so `ω_p=0.05` sits at a ratio of ~0.4 rather than
~0.05. Recalibrating by hand found `ε=0.03, ω_p=0.02, dt=0.5` at 3968
steps/gradient, but that came from a 12-point ad-hoc grid on one weight draw.

Now that the settle is 1.7–2.0× faster (CPU) the sweep is cheap enough to do
properly. The goal is a **map of the usable zone and its boundary**, portable
to other widths, not one lucky operating point.

## What is actually being optimized

Cost is analytic — there is nothing to learn about it:

    steps(p) = T_free + (T_warmup_cycles + n_cycles) · round(2π / (ω_p · dt))

So the problem is constrained, not free:

    minimize   steps(p)
    subject to cos(g_lockin, g_ref) ≥ threshold

The constraint surface is what we do not know. The deliverable is the
**contour** of that constraint plus the efficient frontier along it.

### Fidelity reference

`fd_gradient_phasor` needs `n_params + 1` settles — hopeless at 217K
parameters. The reference is instead **centered `StaticEP` at small β**
(`β=0.005`, `T_free=400`, `T_nudge=200`, `centered=true`), whose O(β) bias
cancels to O(β²).

This is only legitimate if the reference is itself stable, which Stage 0 tests
directly by comparing two independently-parameterized references
(`ref_vs_ref`). Early measurement: **cos = 0.99986, std 2.1e-5** — the oracle
is far tighter than anything we are trying to resolve, so it is not the
limiting noise source.

### Metric

Worst-layer cosine, not mean. An average across layers hides a layer whose
gradient has gone orthogonal, which is exactly the failure the zone boundary
should catch. Relative error is recorded too but is not the criterion: it
diverges as 1/β under basin-hopping (see FINDINGS §3) and so conflates
direction error with magnitude error.

## (a) Dispatch: CPU threads, not GPU

Measured on the production shape (784→256→64):

| strategy | evals/s |
|---|---|
| CPU serial, BLAS=10, B=32 | 6.7 |
| **CPU 10-way parallel, BLAS=1, B=32** | **14.6** |
| GPU serial, B=16–512 | 3.4–5.0 |
| GPU 4-way concurrent, B=512 | 4.6 |
| CPU serial, B=512 | 0.52 |

Two facts decide it:

1. **GPU eval rate is flat from B=16 to B=512.** The settle is ~730 sequential
   small kernels, so it is launch-latency-bound; a 32× larger batch is free.
   Concurrent streams add only 1.36× at K=4 and regress at K=8 — the launches
   serialize.
2. **Estimator variance does not fall with batch size.** Measured across 5
   independent draws, std(cos) is 1.9e-2 at B=16 and 5.0e-2 at B=1024 —
   flat-to-worse. Both the lock-in gradient and the reference are batch means,
   so a larger batch tightens each vector but evidently not the angle between
   them at these widths.

Together those kill the GPU's only advantage for this task: its free large
batch buys nothing, because a large batch is not what reduces the noise.
Replication over draws is. And at the small batch replication implies, CPU is
2.9× faster per evaluation.

So: `Threads.@spawn` across configurations with `BLAS.set_num_threads(1)`.
The parallelism belongs *across* configurations, not inside one gemm.

**Caveat.** CPU scaling is only 2.16× on 10 cores — memory-bandwidth bound, not
core bound. Further throughput needs separate *processes* with independent BLAS
instances, or more memory channels; more Julia threads will not help.

**When to revisit.** If evaluations ever need a production batch (sweeping at
B≥512, or much wider networks), the crossover flips hard: at B=512 GPU is 9.4×
faster. The harness keeps `B` as a knob for exactly that reason.

## (b) Search: staged grid, not Bayesian optimization

BO is the wrong tool here, for three independent reasons:

1. **The deliverable is a boundary, not an argmax.** BO's whole value is
   concentrating samples near the optimum and deliberately under-sampling
   elsewhere — precisely the wrong allocation when the answer is a contour.
   The technique that matches this goal is *level-set estimation* (e.g. the
   straddle acquisition), not BO.
2. **Cost is already known in closed form.** Half of what a cost-aware BO
   would spend evaluations learning is a one-line formula here.
3. **Evaluations cost ~70 ms.** At 14.6 evals/s the full 5×6×3 grid with 8
   replicates is 720 evaluations — minutes. BO earns its complexity when
   evaluations cost minutes to hours; here it would take longer to implement
   than to brute-force.

### Why `dt` is not swept

`ω_p` and `dt` are **degenerate in cost** — only the product enters
`period_steps = 2π/(ω_p·dt)` — but not in physics: `ω_p` alone sets
adiabaticity through `ω_p/R_relax`, while `dt` sets the integration accuracy of
the settle. Sweeping both spends budget on a degenerate direction. `dt` is
pinned at 0.5, the largest value the settle is stable at and the same step the
static settle already uses, and `ω_p` is searched. (This degeneracy is what
made the original recalibration free: moving `dt` 0.1 → 0.5 bought a 5× slower
probe at identical step count.)

### Grid

`ε ∈ {0.003, 0.01, 0.03, 0.1, 0.3}` × `ω_p ∈ {0.005, 0.01, 0.02, 0.05, 0.1,
0.2}` × `n_cycles ∈ {2, 4, 8}`, log-spaced, 8 replicates = 720 evaluations.

Three implementation points that matter more than the search algorithm:

- **The reference is shared.** It depends only on (weights, input, labels), not
  on any lock-in knob, so it is computed once per replicate and reused across
  all 90 configurations. That is 8 reference settles instead of 720 — the
  single biggest saving in the sweep.
- **Cheapest-first ordering.** Cost varies ~59× across the grid (389 to 22817
  steps). Jobs are sorted by analytic cost so a killed run still leaves a
  usable map of the cheap region rather than a random half.
- **Feasibility is gated on the lower confidence bound** (`mean - 2·stderr ≥
  threshold`), so a configuration cannot enter the zone on one lucky draw.

### Stage 0 first

Because the estimator is noisy and batch size does not fix it, the grid would
otherwise risk mapping contours of its own measurement noise. Stage 0 isolates
three candidate sources before any of that budget is spent:

| probe | question |
|---|---|
| `n_cycles` scan | does demodulator selectivity tighten the spread? |
| `ref_vs_ref` | is the oracle itself the limiting noise? |
| `vary_input` / `vary_weights` | is the zone input-dependent or weight-dependent? |

The third matters for generalization: if `vary_weights` spread dominates, the
zone is a property of a particular weight draw and the grid must average over
weight draws too, not just inputs — and any single recommended operating point
should be treated as provisional.

## Running it

```bash
julia --project=. -t 10 scripts/ep_adiabatic_sweep.jl variance   # Stage 0
julia --project=. -t 10 scripts/ep_adiabatic_sweep.jl grid       # Stage 1
julia --project=. -t 10 scripts/ep_adiabatic_sweep.jl report     # Stage 2
```

Both data stages append per-point to CSV and skip work already on disk, so they
are killable and resumable. Knobs: `EPS_HID`, `EPS_DOUT`, `EPS_B`, `EPS_REPS`,
`EPS_DT`, `EPS_SCALE`, `EPS_TFREE`, `EPS_WARMUP`, `EPS_THRESH`, `EPS_OUT`.

## Deliberately out of scope

- **A GP surrogate.** Worth adding *after* the grid, purely as a post-hoc
  smoother to draw a clean contour from noisy points — not as a search driver.
  Only justified if the grid turns out too coarse near the boundary.
- **Sweeping width.** The zone almost certainly moves with width (that is the
  whole reason the toy defaults failed). Re-running this grid at 128/512/1024
  hidden units would test whether `ω_p/R_relax` is the right invariant — which,
  if it holds, replaces the sweep with a single cheap `R_relax` measurement.
  That is the real prize, but it needs this baseline first.
- **Trained weights.** Everything here is at initialization. FINDINGS §3 shows
  gradient fidelity collapses as ‖W‖ grows, so the zone at epoch 10 is probably
  not the zone at epoch 0. Mapping that needs checkpoints the current demo does
  not save.
