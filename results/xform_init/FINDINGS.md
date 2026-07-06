# HiPPO placement in `PhasorTransformerBlock`: read heads vs memory tape

**Question.** The block has two families of `PhasorDense` sublayers, each with a
per-channel decay spectrum `λ` set by `init_mode`:

- **QKV attention projections** (`PhasorLSA`/`PhasorLCA` `init_mode`) — the
  intra-block "read heads."
- **the residual-stream FFN** (`PhasorTransformerBlock(...; ffn_init_mode)`) —
  the "memory tape" carried between transformations.

Hypothesis (going in): the multi-timescale HiPPO basis belongs in the
**residual stream (FFN)**, while the QKV projections should be single-timescale
(`:default`) read heads — and putting HiPPO in the QKV projections may be
wasteful or even harmful.

## Prerequisite fix: `:hippo` was not actually a long tape

`hippo_legs_diagonal` previously log-spaced `|λ|` from `0.5 … N-0.5`, i.e.
time-constants τ=1/|λ| from **2 steps down to ~0.016 steps** — a spread of
*short* timescales, not a memory tape. Its extreme fast end (λ≈−63.5) also drove
projection outputs to the origin (the `complex_to_angle` gradient blow-up seen in
the projection-bias branch).

We redefined `:hippo` to log-space τ over **[`HIPPO_TAU_MIN`=0.5,
`HIPPO_TAU_MAX`=64] steps** (`src/kernels.jl`), so the slowest channel is a
genuine long-memory integrator (λ≈−0.016, ~half of a unit phasor preserved
across 43 steps) and the fast end is gentled. `λ` only shapes dynamics in the 3D
SSM / ODE path, so the whole study runs 3D Phase `(D,L,B)`.

## Task: delayed-cue recall (isolates the tape from attention)

A length-L=48 sequence of D=64 phasors, mostly neutral filler, with two values
from an 8-symbol vocabulary and two **constant** recall cues:

```
1:V_far … 44:cue_far (gap≈43)  |  45:V_near … 48:cue_near (gap=3)
```

Because the cues are **constant** (no content pointer), attention cannot
content-match to locate a value — the value can only reach its readout by being
carried forward through the per-channel conv memory (`λ`). The far readout needs
the long tape (τ≈64), the near readout a fast tap. Readout cleans up against the
value vocabulary (similarity → 8-way class); we report FAR- and NEAR-accuracy.

## 2×2 ablation (init only; `λ` stays trainable; identical param counts)

| config | QKV λ | FFN λ |
|---|---|---|
| A | `:hippo`   | `:default` (current package default) |
| B | `:default` | `:hippo`   (proposed) |
| C | `:hippo`   | `:hippo`   |
| D | `:default` | `:default` |

## Results (3 seeds, 40 epochs, chance = 0.125)

| config | far-acc (mean±std) | near-acc | final loss |
|---|---|---|---|
| A (QKV=hippo / FFN=default) | **1.000 ± 0.000** | 1.000 | 0.060 |
| B (QKV=default / FFN=hippo) | **1.000 ± 0.000** | 1.000 | 0.155 |
| C (all-hippo)               | **1.000 ± 0.000** | 1.000 | 0.112 |
| **D (all-uniform)**         | **0.834 ± 0.144** | 1.000 | 0.219 |

Per-seed far-acc for D: {0.747, 1.000, 0.754} — high variance, on the edge; the
other three configs are 1.000 on every seed. Trained τ stays ≈4–5 steps in D
(the trainable λ never grows a real tape from a uniform start), so D's occasional
success is a lucky init/attention artifact rather than a learned long memory.

## Interpretation

1. **The long-`:hippo` fix works.** Far recall across a 43-step gap is now
   achievable. Under the old short-hippo (τ_max=2) A⁴³≈4e-10 — impossible
   everywhere. The learned-τ columns confirm the new long spectrum is active.

2. **The memory tape is necessary.** All-uniform (**D**, τ=5 ⇒ A⁴³≈1e-4) is the
   only config that fails far recall (0.834 ± 0.144, dipping to 0.75) — it
   cannot reliably carry information 43 steps, and its trainable λ never grows a
   real tape (trained τ stays ≈4). A/B/C are 1.000 on every seed. Near recall
   (gap 3) is trivial for every config.

3. **Placement is flexible on this task — and HiPPO-in-QKV is *not* harmful.**
   HiPPO in *either* the QKV projections (A) or the FFN (B) suffices; both
   saturate at far=1.00. Config A reaches far=1.00 by routing its `v_proj`
   slow-channel memory through attention. This **partially revises the
   hypothesis**: on a pure memory task, loading the read heads with the
   multi-timescale basis does not hurt — it provides the tape just as well.

## Caveat / why placement didn't discriminate

With a **constant cue**, the recall target is identifiable only by its fixed
*lag* (the conv's positional weighting), so the tape merely has to carry one
impulse — which any path with a slow channel does equally well. To test whether
HiPPO-in-QKV is genuinely *wasteful/harmful*, a task is needed where the QKV
projections must **simultaneously** act as sharp content-based read heads (real
associative routing) *and* long memory is required — only then does spending QKV
capacity on slow channels trade off against its read job. That is a natural
follow-up experiment.

## Recommendation

- **Keep the long-`:hippo` fix** (`src/kernels.jl`) — it is a strict improvement:
  it makes `:hippo` a real short-AND-long basis and gentles the origin-collapse
  the fast end used to cause.
- **Do not flip the package default yet.** The data support "a long tape must
  exist somewhere," not "the FFN is the uniquely correct home for it." The
  current default (QKV=hippo, FFN=default, config A) performs as well as the
  proposed swap (B) on this task, so there is no evidence to justify changing it
  until the discriminating (routing + memory) task is run.

## Reproduce

```
julia --project=. -e 'include("scripts/xform_init_ablation.jl"); main()'
# quick: main(; smoke=true, use_cuda=false)
```
Artifacts: `results/xform_init/{results.csv, accuracy.png, lambda_spectra.png}`.
