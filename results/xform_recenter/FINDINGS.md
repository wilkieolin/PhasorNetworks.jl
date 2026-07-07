# PhaseRecenter ablation: useful, or a gradient-blow-up source?

**Question.** `PhasorTransformerBlock` puts a `PhaseRecenter` (circular-mean
pre-norm) at the head of each residual branch when `recenter=true`. It computes

    mθ = complex_to_angle(sum(angle_to_complex(x), dims=1))   # mean angle over channels

which is **ill-conditioned when the channel phasors cancel** (`|sum| → 0`) — the
same origin singularity as `complex_to_angle`. Is it worth the risk?

## Setup

MQAR clean task (d=0), config B (uniform QKV + hippo FFN — the config that ran
*closest* to the origin in earlier probes, so the best stress case), full budget
(40 epochs, 50 batches), 5 seeds, `recenter ∈ {true, false}`. Grad-health from
the `_cta_probe` (which captures the recenter's own `complex_to_angle` backward)
at init and after training.

## Results (5 seeds)

| recenter | far-acc | near-acc | max\|dz\| (trained) | min\|z\| |
|---|---|---|---|---|
| **true**  | 0.904 ± 0.205 | 0.997 ± 0.006 | 6.3e-2 | 9.6e-5 |
| **false** | **1.000 ± 0.000** | **1.000 ± 0.000** | **1.9e-2** | 2.8e-4 |

Per-seed with recenter=true: far = {1.00, 0.54, 1.00, 1.00, 0.98} — seed 2 stuck;
seed 5 drove min|z|=1.4e-7 with a trained `max|dz|`=2.2e-1 spike. With
recenter=false: far = {1.00, 1.00, 1.00, 1.00, 1.00}.

## Verdict — remove it (default `recenter=false`)

`PhaseRecenter` is **both useless and a hazard** on this task:

1. **Hurts trainability.** recenter=false solved all 5 seeds (far=1.000±0.000);
   recenter=true left 2/5 imperfect. Removing it makes training *more* reliable.
2. **Amplifies near-origin gradients.** Trained `max|dz|` is 3.3× larger with
   recenter (6.3e-2 vs 1.9e-2) and it pushes closer to the origin (min|z|
   9.6e-5 vs 2.8e-4; worst single event 1.4e-7 → dz spike 2.2e-1). Confirms
   `complex_to_angle(sum(z))` blows up under channel cancellation.

**Change:** `PhasorTransformerBlock` default `recenter=true → false`. All 1045
tests pass.

## Caveat

This is a **shallow** stack (2 blocks). Pre-norm's usual justification is
stabilizing *deep* stacks against activation drift, which 2 blocks don't
exercise. So `recenter=false` is the right default for typical/shallow use, but
`recenter=true` may still help very deep phasor transformers — re-enable and
verify there. If a deep use-case wants pre-norm without the blow-up, the fix is
a `|mean|`-aware recenter: the `complex_to_angle` here acts on a *sum of C
phasors* (|sum| ~ √C for random phase), so its gate threshold should scale with
C rather than the unit-phasor default (1e-3), and it should pass through
unchanged (skip recentering) when |mean| is below that.

## Reproduce

```
julia --project=. -e 'include("scripts/xform_init_ablation.jl"); main_recenter()'
```
Artifacts: `results/xform_recenter/results.csv`.
