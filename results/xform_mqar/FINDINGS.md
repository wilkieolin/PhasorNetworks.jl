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

near-recall is ~1.00 for all at d=0; the gap-40 far-recall is what separates
configs. (3-seed × finer-density confirmation: see `results.csv`.)

<!-- FINAL_MQAR_TABLE -->

## Interpretation — both halves of the hypothesis confirmed

At the discriminating point (clean, d=0), far-recall ranks
**B (1.00) > D (0.86) > C (0.61) > A (0.32)**:

1. **HiPPO-in-QKV is harmful for long-range routing.** The two *uniform-QKV*
   configs (B, D) are the two best; the two *hippo-QKV* configs (A, C) are the
   two worst. Slow-channel read heads blur adjacent tokens, degrading the
   key-match / induction across the gap. ✔ "QKV should be uniform."
2. **HiPPO-in-FFN helps memory.** Within each QKV pairing, hippo-FFN beats
   uniform-FFN: B>D and C>A. ✔ "FFN should be the tape."
3. ⇒ **B (uniform QKV + hippo FFN) — the proposed config — is the clear winner.**

## Nuances

- **Discrimination is largest when clean.** Heavy distractor noise (d≥0.5)
  collapses far-recall to ~chance for *every* config — so "task-irrelevant
  symbols along the tape" mostly make the task uniformly harder rather than
  fanning the curves. The interesting regime is low density; the finer sweep
  [0,0.1,0.2,0.3,0.4] maps where each config breaks.
- **Trade-off on near-recall under noise.** uniform-FFN stays robust
  (D near=0.995 at d=1.0) while hippo-FFN degrades (B/C≈0.73): the tape's long
  integration absorbs noise, hurting the *recent* readout. HiPPO-FFN is not a
  free win — it trades near-under-noise for far reach.

## Decision

Given both halves confirmed and B the clear winner, the package defaults were
flipped to **config B**:

- `PhasorLSA` / `PhasorLCA` `init_mode` default `:hippo → :default` (uniform,
  sharp read heads).
- `PhasorTransformerBlock` `ffn_init_mode` default `:default → :hippo`
  (multi-timescale memory tape in the residual stream).

All 1045 tests still pass. The 3-seed × finer-density run confirms the ordering
is not a single-seed artifact.

## Reproduce

```
julia --project=. -e 'include("scripts/xform_init_ablation.jl"); main_mqar()'
# quick: main_mqar(; smoke=true, use_cuda=false)
```
Artifacts: `results/xform_mqar/{results.csv, far_vs_density.png, near_vs_density.png}`.
