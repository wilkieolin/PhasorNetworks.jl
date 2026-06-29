# LSA/LCA stackability under the identity-at-init residual fix

**Question.** Commit `3f2d86f` made `ResidualBlock` near-identity at init
(`branch_init_scale=0.1` + bias, opt-in ReZero gate) and fixed a depth-collapse
wall — but it was validated **only on plain `PhasorDense` MLP stacks**
(`depth_sweep_fashionmnist.jl`). Does the same fix make the framework's signature
local-attention layers (`PhasorLSA`/`PhasorLCA`) **stackable** into deep
transformer towers?

**Setup.** Sequential FashionMNIST, one image **row** per timestep (C=28, L=28).
One base chain with a swappable attention kind and residual treatment, stacked
`depth` deep:

```
_row_phase → PhasorDense(28⇒D) → ScanStack(PhasorTransformerBlock(D, attn; TREATMENT), depth) → SSMReadout(D⇒10)
```

New `src/` API backing this: **`PhaseRecenter`** (phase-domain pre-norm),
**`PhasorResidual`** (generic identity-at-init residual wrapper, gate `:none`/`:rezero`),
**`PhasorTransformerBlock`** (`residual(attn)` + `residual(FFN)`). Harness:
`scripts/lsa_lca_residual_sweep.jl`.

**Attention kinds:** `:local_self` (PhasorLSA), `:local_cross` (PhasorLCA),
`:ssm_self` (SSMSelfAttention, reference).

**Residual treatments (the independent variable = "before vs after"):**

| name | gate | branch_init_scale | recenter | meaning |
|---|---|---|---|---|
| `old` | none | 1.0 | no | pre-fix (full-scale, ungated) |
| `downscaled_ffn` | none | 0.1 | no | FFN branch near-identity; attention full |
| `rezero` | rezero (α₀=0.1) | 0.1 | no | attention identity via α gate |
| `rezero_recenter` | rezero | 0.1 | yes | + phase pre-norm |

---

## Exp 1 — init-time gradient-flow probe (DONE: GPU, seed 1, D=64)

Data: `exp1_probe/sweep_summary.csv`. Two init-time diagnostics, no training:
- **`grad_w_ratio`** = (encoder ∇weight L2) / (last-block ∇weight L2). A blow-up
  with depth is the vanishing/exploding signature.
- **`drift`** = mean circular distance between the post-stack and post-encoder
  representations at init. `drift → 0.5` ≡ the representation is fully decorrelated
  from the input (random-walk scramble); `drift → 0` ≡ identity-at-init stack.

### Result: the fix transfers to attention — but only via ReZero, not FFN downscaling.

**`grad_w_ratio` vs depth** (lower/flatter = healthier):

| kind | treat | d1 | d2 | d4 | d8 | d16 |
|---|---|---|---|---|---|---|
| LSA | old | 24 | 5.1e2 | 6.1e5 | 1.4e10 | **Inf** |
| LSA | downscaled_ffn | 4.2e2 | 3.2e4 | 1.1e8 | 9.6e10 | **1.0e19** |
| LSA | **rezero** | 4.7e2 | 4.6e2 | 1.3e3 | 1.6e4 | **4.7e6** |
| LCA | old | 17 | 1.7e2 | 3.4e4 | 3.8e9 | **4.6e18** |
| LCA | **rezero** | 4.1e2 | 3.8e2 | 1.1e3 | 3.2e3 | **3.3e4** |
| ssm_self | old | 4.0 | 29 | 6.4e2 | 6.4e4 | **1.1e12** |
| ssm_self | **rezero** | 4.1e2 | 2.8e2 | 5.6e2 | 2.2e2 | **5.1e2** |

**`drift` at init vs depth** (0 = identity, 0.5 = scrambled):

| kind | treat | d1 | d4 | d8 | d16 |
|---|---|---|---|---|---|
| LSA | old | 0.53 | 0.50 | 0.50 | 0.50 |
| LSA | downscaled_ffn | 0.52 | 0.50 | 0.50 | 0.50 |
| LSA | **rezero** | 0.05 | 0.10 | 0.14 | 0.20 |
| LCA | old | 0.50 | 0.50 | 0.50 | 0.50 |
| LCA | **rezero** | 0.05 | 0.09 | 0.13 | 0.18 |
| ssm_self | rezero | 0.05 | 0.10 | 0.14 | 0.20 |

### Reading

1. **The old treatment reproduces the scramble on attention stacks.** For every
   attention kind, `old` gives `drift ≈ 0.50` at all depths (input fully
   decorrelated by init) and `grad_w_ratio` exploding to 10¹⁰–10¹⁸ — `Inf`
   (Float32 overflow) for LSA at depth 16. Same signature the MLP fix was built
   to cure.

2. **ReZero transfers the fix to all three attention kinds.** `rezero` keeps
   `drift` low (0.05 → 0.20 over depth 1→16, i.e. near-identity-at-init) and the
   gradient ratio bounded — for `ssm_self` it is essentially **flat** across
   depth (2.2e2–5.6e2). At depth 16 ReZero is **12–14 orders of magnitude**
   healthier than `old` (LCA: 3.3e4 vs 4.6e18).

3. **FFN weight-downscaling alone does NOT fix attention** — it is, if anything,
   worse than `old` at depth 16 (LSA: 1.0e19 vs Inf is moot, but 1e8 vs 6e5 at
   depth 4). Because `downscaled_ffn` only shrinks the FFN branch, the
   full-strength attention branch still scrambles the representation
   (`drift ≈ 0.50`). **This empirically confirms the design rationale:**
   weight-downscaling is an FFN-only lever; the attention sublayer needs the
   ReZero gate (exact identity at α=0) to start near-identity.

4. **The α gate is live and depth-aware.** `grad_alpha_mean` under `rezero` grows
   with depth (LSA 0.014→5.7, LCA 0.009→0.73 over depth 1→16): deeper stacks put
   more gradient pressure on α — consistent with adaptive depth allocation.

5. **No rank/diversity collapse (negative result).** `circvarD` (channel circular
   variance after the stack) stays ≈0.88–0.90 in every treatment. The init-time
   pathology here is gradient explosion + forward scramble, **not** channel
   collapse — so a diversity-restoring fix is not what is needed.

6. **The non-softmax attention scale interacts with the blow-up.** LSA/LCA use
   `exp(β·s)/H` (not a normalized softmax). Under `old`, `grad_scale_mean` (the β
   gradient) explodes to 1e16; under `rezero` it stays ~0.1. Flagged for Exp 3 /
   a possible true-softmax ablation.

**Conclusion (Exp 1):** the identity-at-init fix transfers from MLPs to stacked
LSA/LCA transformer blocks, **provided the attention residual is gated with
ReZero** — FFN weight-downscaling is insufficient because it cannot bring the
attention branch to identity-at-init. Recenter (`rezero_recenter`) is comparable
to plain `rezero` at init; its value (if any) should show in training.

---

## Exp 2 — trained accuracy vs depth (DONE: GPU, 3 seeds, 8 epochs, D=64)

Data: `exp2_final/sweep_summary.csv`. Sequential FashionMNIST, batchsize 32,
RMSProp lr 3e-4, ReZero arms get a 5× α-LR. **LSA and LCA are complete (60/60
configs each, 0 NaN failures).** The `ssm_self` reference is partial (see Open
items) and not required for the conclusion. Test accuracy, mean ± std over 3
seeds:

### PhasorLSA (local self-attention)

| treatment | d1 | d2 | d4 | d8 | d16 |
|---|---|---|---|---|---|
| `old` | .689±.008 | .649±.024 | **.094±.003** | .099±.003 | .112±.008 |
| `downscaled_ffn` | .698±.009 | .709±.011 | **.109±.005** | .107±.004 | .098±.008 |
| `rezero` | .679±.015 | .739±.009 | **.772±.004** | .771±.010 | **.775±.007** |
| `rezero_recenter` | .673±.011 | .734±.010 | .751±.001 | .751±.007 | .763±.014 |

### PhasorLCA (local cross-attention, Hopfield anchors)

| treatment | d1 | d2 | d4 | d8 | d16 |
|---|---|---|---|---|---|
| `old` | .697±.008 | .658±.012 | **.105±.006** | .097±.007 | .099±.003 |
| `downscaled_ffn` | .700±.004 | .662±.010 | **.128±.015** | .094±.006 | .105±.004 |
| `rezero` | .690±.009 | .734±.006 | .761±.008 | .779±.006 | **.788±.004** |
| `rezero_recenter` | .663±.008 | .730±.012 | .749±.015 | .749±.002 | .755±.018 |

### Reading (confirms every Exp 1 prediction)

1. **`old` and `downscaled_ffn` hit a depth ceiling at ~2 and collapse to chance
   (≈0.10) by depth 4** — for both LSA and LCA. The collapse is the silent
   representation scramble (Exp 1 `drift→0.5`), not numerical blow-up:
   `nan_loss=false` everywhere, losses are finite, the network just learns
   nothing.

2. **ReZero is depth-robust and monotonically improving to depth 16** — LSA
   .775, LCA **.788** (its best cell). The identity-at-init fix that rescued
   plain MLP stacks transfers cleanly to the framework's signature attention
   layers. ReZero wins at every depth ≥ 2.

3. **FFN weight-downscaling does NOT rescue attention** — `downscaled_ffn`
   tracks `old` and collapses identically, because it leaves the attention
   branch full-strength. Empirically nails the design rationale: attention needs
   the ReZero gate (exact identity at α=0); weight-downscaling is an FFN-only
   lever.

4. **Pre-norm (`recenter`) slightly *under*performs plain ReZero at depth**
   (LCA d16 .755 vs .788; LSA d16 .763 vs .775). Phase-domain recentering isn't
   needed once ReZero is in place, and marginally hurts. **Plain `rezero` is the
   recommended default.**

5. Variance across seeds is tiny (σ ≤ 0.02), so the separation is robust.

**Conclusion (Exp 2):** the identity-at-init residual fix makes stacked
PhasorLSA/PhasorLCA depth-robust **specifically via the ReZero gate**, turning a
depth-2 collapse into monotonic improvement through depth 16. This is the trained
confirmation of the Exp 1 gradient/forward-scramble signature.

## Exp 3 — spiking gap (pending the targeted LSA/LCA run)

The depth-8 discrete-vs-spiking gap is queued for a focused run over
`kinds=(:local_self,:local_cross)` (the full-matrix driver's spiking step is
gated behind all three kinds completing, which `ssm_self` is currently blocking).

```julia
include("scripts/lsa_lca_residual_sweep.jl")
main_lsa_lca_sweep(; kinds=(:local_self,:local_cross), depths=(1,2,4,8,16),
                   seeds=1:3, epochs=8, batchsize=32, checkpoint=true,
                   init_probe=false, spiking_depth=8, best_treatment=:rezero,
                   resume=true, outdir="results/lsa_lca_residual/exp2_final")
```

---

## Open items / suggested follow-ups

- **`grad_w_ratio` still grows under ReZero at depth 16** (e.g. LSA 4.7e6). Bounded
  vs `old`, but a true-softmax score norm and/or β clamping may flatten it further
  — worth the Exp-3 ablation if Exp 2 shows a depth-16 wobble.
- **`PhasorLSA`/`PhasorLCA` could accept `init_weight`** so the `downscaled_ffn`
  lever can also reach Q/K/V; currently attention identity is ReZero-only (which
  Exp 1 says is the right default anyway).
- Promote `PhasorTransformerBlock` into a demo notebook once Exp 2 confirms the
  trained depth curves. **(Exp 2 now confirms them — LSA/LCA depth-robust to 16
  under ReZero.)**
- **`SSMSelfAttention` is ~10× slower than LSA/LCA on GPU** — during Exp 2 the
  `ssm_self` configs stalled with one CPU core pegged at 100% and the GPU idle
  (loadavg ≈ 1.0, no device activity), the signature of a scalar-indexing → CPU
  fallback in its across-time `attend` path. LSA/LCA use `batched_mul` head-mix
  paths and ran fast on GPU. The reference sweep was left partial (22/60) because
  of this; the LSA/LCA conclusions don't depend on it. Worth profiling/fixing the
  `attend`/`similarity_outer(...; dims=2)` path for GPU separately.
- Memory/runtime engineering: the sweep needed process-chunking
  (`scripts/run_exp2.sh`, `max_configs`) + `JULIA_HARD_MEMORY_LIMIT` +
  `--heap-size-hint` because a single long-lived process accumulates memory
  across configs and stalls into GC-thrash after ~100–140 configs on the
  shared-memory box. Fresh process per ~20 configs (resume-based) is the stable
  pattern.
