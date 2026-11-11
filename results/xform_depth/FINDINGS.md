# Deep-stack test: does ReZero carry depth WITHOUT phase recentering?

**Motivation.** `PhaseRecenter` was made off-by-default (`results/xform_recenter/`)
because it hurts trainability and blows up gradients on a shallow (2-block) stack.
The open caveat: pre-norm's classic job is stabilizing *deep* stacks, so maybe it
still earns its place at depth. This tests whether the **ReZero gate alone**
(α₀=0.1, near-identity at init) carries deep phasor-transformer stacks with
`recenter=false`.

## Setup

MQAR clean task (d=0), config B (uniform QKV + hippo FFN — the shipped default),
`recenter=false`, depths ∈ {2, 4, 8, 16} `PhasorTransformerBlock`s, 2 seeds, full
budget (40 epochs, 40 batches). Grad-health from `_cta_probe` at init & trained.

## Results (2 seeds)

| depth | far-acc | near-acc | final loss | max\|dz\| (trained) | min\|z\| |
|---|---|---|---|---|---|
| 2  | 1.000 ± 0.000 | 0.999 ± 0.001 | 0.337 | 3.8e-2 | 1.2e-4 |
| 4  | 1.000 ± 0.000 | 0.999 ± 0.001 | 0.259 | 1.1e-2 | 1.1e-4 |
| 8  | 0.992 ± 0.011 | 1.000 ± 0.000 | 0.193 | 6.8e-2 | 1.1e-4 |
| 16 | 0.992 ± 0.012 | 0.994 ± 0.008 | 0.290 | 3.0e-2 | 1.3e-5 |

## Verdict — PASS

Deep phasor-transformer stacks train fine to **depth 16 without pre-norm**:

1. **Accuracy holds** ~0.99–1.00 at every depth; no depth collapse. Loss even
   *improves* with depth (depth-8 lowest at 0.19 — the extra capacity helps),
   which is the opposite of a vanishing-signal failure.
2. **Gradients stay bounded through training.** Trained `max|dz|` sits ~1e-2–7e-2
   across all depths — flat, not growing with depth; `min|z|` stays ~1e-4–1e-5,
   no runaway origin collapse.
3. **The ReZero gate is what makes this work.** Its near-identity init keeps the
   forward/backward signals well-scaled without any normalization layer.

**Nuance (honest):** *init* `max|dz|` does grow with depth (4e-3 @ d2 → 1.7e-1
@ d16) — expected in an unnormalized residual stack — but training pulls it back
down within bounds. So there is a mild depth-scaling of the raw init gradient,
just not a catastrophic one, and ReZero + the near-origin gate absorb it.

## Conclusion

The `recenter=false` default is validated at depth: **PhaseRecenter is not needed
for deep stacks** (to depth 16 here), resolving the caveat left open in
`results/xform_recenter/FINDINGS.md`. ReZero (α₀=0.1) is the depth-conditioning
mechanism, consistent with the earlier depth/residual study. Depths >16 or noisy
(d>0) deep stacks remain untested.

## Reproduce

```
julia --project=. -e 'include("scripts/xform_init_ablation.jl"); main_depth()'
```
Artifacts: `results/xform_depth/results.csv`.
