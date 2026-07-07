# MQAR-along-a-noisy-tape: which init scheme, and does hippo-in-QKV hurt?

**Companion to** `results/xform_init/FINDINGS.md`. That experiment (delayed-cue
recall) showed the memory tape is *necessary* but couldn't discriminate *where*
HiPPO belongs, because it never stressed content routing. This one does.

## Task

Multi-query associative recall (MQAR) with **separate key/value tokens**: a pair
is `[key@p, value@p+1]`; a query presents a bare key; the answer is the value
that followed it. Two targets — **far** (front of the sequence, gap≈40) and
**near** (gap≈4). Non-target positions are filled with **random single-token
distractors** at probability `density` (task-irrelevant noise along the tape).

With full attention the *gap* is largely content-addressable, so `density` (SNR)
is the difficulty knob, and it stresses the **read heads** (extracting the clean
key/value from noise + the key→value+1 induction) rather than the FFN tape.

Same 2×2 init ablation (A: QKV=hippo/FFN=default; B: QKV=default/FFN=hippo;
C: all-hippo; D: all-uniform), same harness.

## Results — far-recall vs density (chance = 0.125)

**Preliminary (1 seed, 40 epochs):**

| config | QKV / FFN | d=0.0 | d=0.5 | d=1.0 |
|---|---|---|---|---|
| **B** | uniform / **hippo** | **1.00** | 0.23 | 0.19 |
| D | uniform / uniform | 0.86 | 0.12 | 0.12 |
| C | hippo / hippo | 0.61 | 0.25 | 0.19 |
| A | hippo / uniform | 0.32 | 0.13 | 0.12 |

near-recall is ~1.00 for all at d=0. **⚠ The 1-seed numbers above did NOT
replicate — see below.**

**3-seed confirmation (finer density [0,0.1,0.2,0.3,0.4], 30 epochs, far-acc
mean±std):**

| cfg | QKV/FFN | d=0.0 | d=0.1 | d=0.2 | d=0.3 | d=0.4 |
|---|---|---|---|---|---|---|
| A | hippo/uniform  | 0.70±0.25 | 0.27±0.25 | 0.13 | 0.13 | 0.11 |
| B | uniform/hippo  | 0.44±0.08 | 0.37±0.02 | 0.29 | 0.44±0.32 | 0.22 |
| C | hippo/hippo    | 0.69±0.27 | 0.35±0.05 | 0.29 | 0.26 | 0.25 |
| D | uniform/uniform| 0.45±0.13 | 0.14 | 0.13 | 0.12 | 0.13 |

## Interpretation — the QKV claim did NOT survive replication

The 1-seed run (B=1.00, A=0.32 at d=0) was a **single-seed outlier**. With 3
seeds the clean point is A=0.70, B=0.44, C=0.69, D=0.45 — huge variance, and if
anything *hippo*-QKV (A, C) sits *higher*. Per-seed d=0: A={0.46,0.68,0.95},
C={0.48,0.61,1.00} each had a seed converge to ~1.0, while B and D never did.

**Confound:** the 3-seed run used a reduced training budget (30 epochs / 40
batches vs the 1-seed's 40 / 50) to bound runtime, which undertrained the clean
d=0 point (nobody reached 1.0) — muddying exactly the discriminating regime. A
clean d=0 × 5-seed × full-budget run (`results/xform_mqar_d0/`) settles this.

**What holds robustly across seeds:**

1. **FFN=hippo helps far-recall under distractor noise.** B, C (hippo-FFN) beat
   A, D (uniform-FFN) at every d≥0.1 (e.g. d=0.2: 0.29/0.29 vs 0.13/0.13). ✔ the
   "tape" half.
2. **FFN=hippo hurts near-recall under noise** (trade-off): at d=0.4 near-acc
   D≈0.94 vs B,C≈0.79. The tape's long integration absorbs noise, hurting the
   *recent* readout. Holds across all seeds.
3. **QKV mode is inconclusive.** B vs C (both hippo-FFN) is a wash; A ≥ D. **No
   support for "uniform QKV."** The originally-hypothesized "hippo-in-QKV is
   harmful" is *not* borne out.

## Clean confirmation (d=0, 5 seeds, FULL budget) — `results/xform_mqar_d0/`

The 3-seed muddiness was the training-budget confound. Re-running clean d=0 at
the full 40-epoch / 50-batch budget × 5 seeds restores the effect. far-acc 2×2:

|              | FFN=uniform      | FFN=hippo        |
|--------------|------------------|------------------|
| **QKV=hippo**   | A: 0.67 ± 0.32 | C: 0.83 ± 0.22 |
| **QKV=uniform** | D: 0.81 ± 0.26 | **B: 0.90 ± 0.21** |

Both main effects point the hypothesized way in *every* within-pair comparison:

- uniform-QKV ≥ hippo-QKV: **B>C** (0.90>0.83) and **D>A** (0.81>0.67).
- hippo-FFN ≥ uniform-FFN: **B>D** (0.90>0.81) and **C>A** (0.83>0.67).
- **B (uniform QKV + hippo FFN) is the best cell; A (old default) the worst.**

Reliability (seeds solved, far≥0.95): **B 4/5, C 3/5, A 2/5, D 1/5** — B
converges most reliably. Effects are modest (0.07–0.16) with high seed variance
(far-recall is near-bimodal: a seed either converges to ~1.0 or sticks ~0.3–0.6),
but the **direction is consistent across all four comparisons**.

## Decision

Defaults flipped to **config B** and **kept** — supported by the clean run:
`PhasorLSA`/`PhasorLCA` `init_mode` `:hippo→:default` (sharp uniform read heads);
`PhasorTransformerBlock` `ffn_init_mode` `:default→:hippo` (multi-timescale tape
in the residual stream). All 1045 tests pass.

Confidence: **FFN→`:hippo`** is well-supported (clean 2×2 + noise-robustness +
delayed-cue tape-necessity). **QKV→`:default`** is supported but *modestly*
(consistent direction, overlapping error bars) — worth revisiting if a future
task shows hippo-QKV helping. The near-recall-under-noise trade-off (uniform-FFN
more robust to recent noise) remains the main reason a workload might prefer
uniform FFN.

## Reproduce

```
julia --project=. -e 'include("scripts/xform_init_ablation.jl"); main_mqar()'
# quick: main_mqar(; smoke=true, use_cuda=false)
```
Artifacts: `results/xform_mqar/{results.csv, far_vs_density.png, near_vs_density.png}`.
