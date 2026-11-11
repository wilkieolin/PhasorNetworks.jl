# The ‖W‖ decorrelation is a property of random matrices, not of weight norm

> **Harness:** `scripts/ep_trained_vs_rescaled.jl` (main sweep),
> `scripts/e1_settle_convergence.jl` (T ladder × batch),
> `scripts/e1_free_vs_nudge.jl` (T_free × T_nudge grid) &nbsp;|&nbsp;
> **Analysis:** `scripts/e1_summarize.py` &nbsp;|&nbsp;
> **Answers:** the ‖W‖ caveat and open question 4 in
> `results/ep_fashionmnist/FINDINGS.md` &nbsp;|&nbsp; gitrev `ddbaa3d`

## Why this was run

`results/ep_fashionmnist/FINDINGS.md` §3 reports that the EP gradient
decorrelates past ‖W₁‖ ≈ 15 — one-sided cos falls 0.995 → 0.104 between
‖W₁‖ = 7.9 and 31.4, with relative error scaling as 1/β (28.7, 96.9, 292, 976
at β = 0.1, 0.03, 0.01, 0.003). That 1/β scaling is the signature of a
β-independent term surviving in `h_nudge - h_free`, i.e. the free and nudged
settles converging to different fixed points.

But the headline run trains at ‖W₁‖ ≈ 20–29 throughout and reaches 0.83. Both
statements cannot be load-bearing. §3's own caveat is that the probe used
**randomly initialized matrices rescaled to the stated norm**, and that trained
weights of the same norm may behave better. That was a hypothesis. This is the
measurement.

## Method, and one methodological point that matters

784 → 256 → 64, StaticEP (centered, β=0.1, T_free=200, T_nudge=100, dt=0.5),
Adam 3e-3, batch 128, full 60K/10K, seed 7, 20 epochs. Two trajectories,
`wd=0` and `wd=1e-4`, snapshotted every 2 epochs. Three arms per snapshot:

| arm | construction |
|---|---|
| `trained` | the snapshot as trained |
| `rescaled` | the run's own random init, rescaled so ‖W₁‖ and ‖W₂‖ match the snapshot |
| `trained_rescaled` | the **epoch-2** snapshot rescaled to the current snapshot's norms |

The third arm is what separates structure from scale. `rescaled` differs from
`trained` in both, so on its own it cannot attribute the effect.

**The oracle cannot be centered StaticEP.** The published probe scores EP
against centered StaticEP at small β. That is fine for calibrating lock-in, but
it is circular here: basin hopping is a property of the settle, so it
contaminates the reference too, and a reference sharing the failure mode cannot
detect it. Full FD needs n_params+1 settles (217K) and is unaffordable.

So the oracle is **directional** central finite differences: K = 256 fixed
pseudo-random unit directions in `layer_1.weight`, `<∇L, d_k>` by central
difference. Two reasons over coordinate FD:

- *Conditioning.* One entry of a 784×256 matrix moves a Float32 cross-entropy
  near 1.7 by ~1e-6 against ~1e-7 resolution — the oracle becomes the noisy
  party, exactly as `ep_fashionmnist/FINDINGS.md` warns for
  `fd_gradient_phasor`'s default ε. A dense unit direction moves every entry at
  once: measured |ΔL| is 2.1e-5 / 6.4e-5 / 3.0e-4 at ε = 0.01 / 0.03 / 0.1.
  (An earlier coordinate-FD version of this script agreed with itself to only
  cos 0.94 across step sizes where the settle residual was 1e-9.)
- *It measures the right thing.* By Johnson–Lindenstrauss the cosine between
  the two projection vectors estimates the cosine between the **full**
  gradients, not the cosine on a coordinate slice.

Every probe point runs a three-step bracket {0.01, **0.03**, 0.1} and reports
agreement of the reference with each neighbour. Trust is decided by the
**small**-step end only: convergence as the step shrinks certifies the linear
regime, while the large step is *expected* to disagree wherever the loss has
curvature. (Gating on the large step, as a first version did, discards exactly
the points of interest.)

## Result 0: the harness reproduces the published run

| | final acc | ‖W₁‖ |
|---|---|---|
| published `final` (`ep_fashionmnist/epoch_curves.csv`) | 0.8268 | 25.3 |
| this script, `wd=1e-4` | **0.8296** | 25.7 |

0.3 points and 0.4 in norm. Everything below is measured on the same object the
headline run trains.

## Result 1: at matched norm, trained weights show no decorrelation at all

‖W₁‖ ≈ 32, the published probe's 31.4, one-sided StaticEP against the FD oracle:

| β | 0.3 | 0.1 | 0.03 | 0.01 | 0.003 |
|---|---|---|---|---|---|
| `trained` cos | 0.9999 | **1.0000** | 1.0000 | 0.9998 | 0.9977 |
| `trained` rel-err | 0.012 | 0.008 | 0.010 | 0.022 | 0.070 |
| `rescaled` cos | −0.031 | **−0.059** | −0.069 | −0.072 | −0.073 |
| `rescaled` rel-err | 12.0 | 35.8 | 119.2 | 357.3 | **1190.8** |

The control's relative error triples for every 3.33× drop in β — slope −1 to
two figures, matching the published 28.7 → 976. **The published finding
reproduces exactly when the construction matches, and vanishes when it does
not.** Both cells are FD-TRUSTED (small-step agreement 0.9997 and 0.9910), so
neither column is an oracle artefact.

## Result 2: scale is not the variable — structure is

`trained_rescaled` — epoch-2 weights scaled up along the whole trajectory:

| ‖W₁‖ | 41 | 50 | 56 | 64 | 69 | 75 | 81 | 85 | **89** |
|---|---|---|---|---|---|---|---|---|---|
| cos (1-sided, β=0.1) | .9999 | .9999 | .9999 | .9999 | .9998 | .9998 | .9997 | .9997 | **.9996** |

It also *keeps its accuracy* under the rescale — 0.8324 → 0.8362 across that
row, against 0.13 (chance) for the random control at every norm. Rescaling a
trained matrix preserves both the function and the gradient estimate.

Trained structure at 2.8× the norm where the published probe reports cos = 0.104
gives 0.9996. Weight norm is not the variable. What breaks EP is the *randomness*
of the matrix, and rescaling a random matrix does not make it less random.

Independent corroboration from the FD bracket itself: **all 165 untrusted rows
are in the `rescaled` arm; none are in either trained arm.** At three quarters
of the random-matrix probe points the loss is not locally linear at any step in
the bracket. The construction is not merely hard for EP — it is a place where
the directional derivative barely exists.

## Result 3: `centered` removes the β-independent term outright

On the `rescaled` control at ‖W₁‖ = 32, where one-sided blows up as 1/β:

| β | 0.3 | 0.1 | 0.03 | 0.01 | 0.003 |
|---|---|---|---|---|---|
| centered cos | 0.9293 | 0.9290 | 0.9290 | 0.9290 | 0.9287 |
| centered rel-err | 0.514 | 0.515 | 0.515 | 0.515 | 0.515 |

Flat to three figures across two decades of β. The published table gives
centered cos = 0.380 at this norm and reads as a partial recovery; measured
against an independent oracle it is 0.929, and the *flatness* is the real
result — centered does not attenuate the β-independent term, it cancels it.

## Result 4: the trained arm is far more robust, but not immune — and it is the *same* mechanism

The `trained` arm is not uniformly clean. At the package's `T_free = 200`, 8 of
the 20 snapshots fall below cos 0.99 one-sided at β = 0.1:

| run | ‖W₁‖ | 1-sided | centered |
|---|---|---|---|
| nodecay ep 6 | 49.9 | 0.837 | 0.998 |
| nodecay ep 8 | 55.9 | 0.072 | **−0.059** |
| nodecay ep 10 | 63.8 | −0.624 | 1.000 |
| nodecay ep 14 | 75.1 | 0.607 | **−0.882** |
| nodecay ep 16 | 81.1 | 0.467 | 1.000 |
| nodecay ep 18 | 85.2 | 0.603 | 1.000 |
| wd1e-4 ep 8 | 24.8 | 0.020 | 0.961 |
| wd1e-4 ep 10 | 24.3 | 0.970 | 1.000 |

The other 12 sit at 0.9965–1.0000. Note the failures are **not monotone in
‖W₁‖** — ep 12 (69.3) and ep 20 (89.4) are perfect while ep 10 and ep 14 either
side of them are not.

**An earlier draft of this note called these a second, distinct failure mode**,
on the grounds that they were un-converged settles rather than basin hops and
that `centered` did not rescue them. The β sweep refutes that. Relative error at
β = 0.3 / 0.1 / 0.03 / 0.01 / 0.003, where a 1/β law is a ×3.33 ratio per step:

| | 0.3 | 0.1 | 0.03 | 0.01 | 0.003 |
|---|---|---|---|---|---|
| nodecay ep 8, 1-sided | 127 | 428 | 1466 | 4428 | 14795 |
| nodecay ep 10, 1-sided | 1.90 | 5.68 | 18.90 | 56.70 | 188.97 |
| nodecay ep 14, 1-sided | 1.97 | 7.27 | 31.49 | 101.89 | 348.61 |
| nodecay ep 16, 1-sided | 0.45 | 1.34 | 4.45 | 13.33 | 44.43 |
| wd1e-4 ep 8, 1-sided | 0.68 | 1.48 | 4.24 | 12.11 | 39.65 |
| *(rescaled ep 2, for reference)* | *12.0* | *35.8* | *119.2* | *357.3* | *1190.8* |

Every row is slope −1 to two figures. These are basin hops, the same
β-independent term as the rescaled control, differing only in magnitude. And
`centered` flattens them exactly as Result 3 describes — nodecay ep 10 goes
0.011 / 0.012 / 0.012 / 0.018 / 0.050 (flat, cos 1.000), wd1e-4 ep 8 goes
0.309 / 0.310 / 0.310 / 0.310 / 0.307 (flat, cos 0.961).

So there is **one mechanism, not two.** What differs between the arms is its
*rate*: at matched norm the rescaled control hops at essentially every probe
point, the trained arm at 8 of 20, and `centered` clears 6 of those 8. The two
it does not clear (ep 8, ep 14) also go flat in β — 21.8 and 4.50, unchanging
across two decades — so `centered` removes the β-dependence there too and simply
lands on the wrong answer. That residual is not characterised.

### `T_free` is a lever on the same mechanism

Gridding `T_free` × `T_nudge` at three failing points, oracle fixed at T = 3200
(`free_vs_nudge.csv`, one-sided β = 0.1):

| **nodecay ep 8** | T_nudge=100 | 200 | 400 | 800 |
|---|---|---|---|---|
| T_free=200 | 0.319 | 0.239 | 0.239 | 0.239 |
| 400 | 0.991 | 0.991 | 0.991 | 0.991 |
| 800 | 1.000 | 1.000 | 1.000 | 1.000 |
| 1600 | 1.000 | 1.000 | 1.000 | 1.000 |

| **nodecay ep 14** | 100 | 200 | 400 | 800 |
|---|---|---|---|---|
| T_free=200 | −0.505 | 0.028 | 0.069 | 0.069 |
| 400 | 0.587 | 0.588 | 0.589 | 0.589 |
| 800 | 0.999 | 1.000 | 1.000 | 1.000 |
| 1600 | 0.999 | 1.000 | 1.000 | 1.000 |

| **wd1e-4 ep 4 batch 3** | 100 | 200 | 400 | 800 |
|---|---|---|---|---|
| T_free=200 | 0.731 | 0.678 | 0.595 | 0.513 |
| 400 | 0.919 | 0.902 | 0.858 | 0.788 |
| 800 | **0.299** | **0.013** | **0.046** | **0.047** |
| 1600 | 0.996 | 0.999 | 1.000 | 1.000 |

For the two nodecay cells the reading is clean — rows move, columns do not, so
the free settle is the binding constraint and `T_nudge` is irrelevant. **The
third cell does not cooperate, and it is the reason the clean reading cannot be
stated generally.** There `T_nudge` does matter (0.731 → 0.513 across the top
row) and `T_free` is *non-monotone*: 0.92 at 400, collapsing to 0.01–0.30 at
800, recovering to 0.999 at 1600. Which basin the free/nudged pair lands in is
not a monotone function of settle length.

The defensible statement is the weak one: **`T_free = 200` is not always
sufficient at width 256, `T_free = 1600` was sufficient at every point probed,
and intermediate values are not safe to interpolate.**

## Result 5: the drift diagnostic works — but only at the estimator's own `T_free`

`‖z(2T) − z(T)‖/√N`, on `settle_convergence.csv`, where drift and the estimator
share the same T (n = 96):

| drift | n | 1-sided min cos | median | centered min cos |
|---|---|---|---|---|
| < 1e-8 | 85 | 0.9986 | 0.9999 | 0.9986 |
| 1e-8 … 1e-5 | 5 | 0.9998 | 0.9999 | 0.9996 |
| 1e-5 … 1e-3 | 3 | 0.9688 | 0.9929 | 0.9925 |
| > 1e-3 | **3** | **−0.2198** | −0.0494 | 0.0263 |

Clean separation: no cell above drift 1e-3 has cos ≥ 0.99, and only one cell
below it falls under 0.99 (0.9688, at drift 6.8e-5 — a mild degradation, not an
inversion). Also note the **step-to-step residual misses this**: in the worst
cell the residual is 9.3e-5 against a drift of 1.3e-1, three orders apart.

**The caveat that matters.** In the main sweep, drift is computed at the
oracle's `FD_T = 400` while the estimator runs at `T_free = 200`. There the
diagnostic fails outright: **7 of 34 probe points have drift < 1e-5 and cos <
0.99**, including nodecay ep 10 at drift 8.8e-8 with cos −0.624. That is not a
counterexample to the table above — it is the mismatch. A converged settle at
T = 400 says nothing about a settle stopped at 200. So the check is only
meaningful run at the same `T_free` the estimator uses, which is a
one-line-to-get-wrong detail worth stating explicitly.

## What this changes in `results/ep_fashionmnist/FINDINGS.md`

1. **§3's caveat was right, and understated.** Trained weights at matched norm
   do not merely "behave better" — they show no decorrelation whatsoever. The
   ‖W₁‖ ≈ 15 threshold should not be stated as a property of the estimator.
2. **§3's "a convergence check cannot see this failure" is too strong.** It is
   true of the step-to-step residual, which misses everything here. The T-vs-2T
   drift, run at the estimator's own `T_free`, separates cleanly (Result 5). The
   failure is still a basin hop in both arms — that part of §3 is right — but it
   is not invisible to convergence checking, only to the wrong one.
3. **Open question 4 is answered.** "Why does wd=1e-4 help without bounding
   ‖W‖?" — with `centered` on, it does not help. Final accuracy 0.8636 at
   ‖W₁‖ = 89.4 with **no** decay, against 0.8296 at ‖W₁‖ = 25.7 with it: decay
   costs 3.4 points while bounding the norm to a third. The published
   `baseline_nodecay` collapse (0.726 at ‖W₁‖ = 108.9) was the one-sided
   estimator failing, not the norm growing, and `centered` removes it.
4. **Open question 3 is answered** ("does basin-hopping onset differ for trained
   vs randomly-scaled weights of equal norm?"). Yes, completely.

## Limitations

- **One seed, one architecture.** The trained-vs-rescaled contrast (1.000 vs
  −0.059) is far too large to be seed noise. The "8 of 20 trained snapshots"
  figure is one trajectory pair at one probe batch and is **not** a failure
  rate; do not quote it as a probability.
- The `rescaled` arm has only 55 trusted rows of 220. Its summary statistics are
  over the minority of its own probe points where the oracle is valid — which
  biases them *optimistic*, since the excluded points are the roughest.
- Probe batch is 16 samples. Batch 3 of `wd1e-4` epoch 4 fails at T_free = 400
  where batches 1 and 2 are perfect, so a parameter×batch interaction exists on
  top of the settle-length effect and is not characterised here.
- `trained_rescaled` uses the epoch-2 snapshot as its structure donor. Whether a
  *later* snapshot rescaled *down* behaves the same is untested.
- All CPU. No CUDA path is touched, so nothing here needs a GPU memory cap.
