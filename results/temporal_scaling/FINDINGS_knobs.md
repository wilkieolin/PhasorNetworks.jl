# SSM performance knobs: width/FFN (4), LCA anchors (5), λ-range (3), modes/channel (1)

> **⚠ Regime caveat (2026-07 reconciliation).** These results were measured with
> the *single-position headless* TIR readout and **no input embedding**. The
> phasor_torch/audio reconciliation (`phasor_torch/results/LINCHPIN_FINDINGS.md`)
> showed that regime **over-states the FFN** (the audio pipeline's input embedding
> + pooling readout make the FFN redundant). Re-verified under the corrected
> regime (`input_embed=true, pool_frac=0.25`) in
> `results/temporal_scaling/fixed_regime/`. Read the conclusions below alongside
> that re-run.


**Follow-up to** `FINDINGS.md` (depth is a weak lever; FFN load-bearing; width >
depth). Having reframed the architecture as an **LRU/S5-style SSM-MLP**, we test
four knobs the deep-SSM literature says should matter, all on the **FFN** (the
load-bearing sublayer) and the headless **TIR** task. Harness:
`scripts/temporal_scaling_sweep.jl` (`run_knobs`), branch `ssm-knobs`. Shrunk
scale (D=48 base, L=32, 2 seeds, 40 epochs). The discriminating probe for the
*temporal* knobs (3, 1) is the **long-range** setting `sig_max_frac=0.34`
(evidence planted early, query at L) vs the spread setting `1.0`.

New library support (branch `ssm-knobs`, all tests green — 1134/1134):
- `MultiModePhasorDense` — SSM **state expansion**: M timescale-modes per output
  channel (distinct log-spaced τ, shared carrier ω), block-diagonal complex
  mix-down. `n_modes=1` reduces exactly to `PhasorDense`.
- `hippo_tau_max`/`hippo_tau_min` threaded through `PhasorDense` /
  `PhasorTransformerBlock` (the λ-range knob).
- `PhasorTransformerBlock(...; ffn_n_modes, ffn_hippo_tau_max/min)`.

---

## Knob 4 — width `D` × FFN expansion `d_ff` (acc, params)

| D | d_ff | params | acc |
|---|---|---|---|
| 48 | 48 | 24k | 0.445 |
| 48 | 96 | 33k | 0.486 |
| 48 | 192 | 52k | 0.520 |
| 64 | 64 | 42k | 0.529 |
| 64 | 128 | 59k | 0.587 |
| 96 | 96 | 94k | **0.622** |
| 96 | 192 | 131k | **0.641** |
| 96 | 384 | 206k | 0.648 |
| 128 | 128 | 166k | 0.625 |
| 128 | 512 | 365k | 0.637 |

**Width is the strongest *general* capacity knob** (0.44 → 0.65), **plateauing
around D≈96**; `d_ff` is a secondary lever that **saturates at ~2×**. Parameter
efficiency favours moderate width: **D=96/d_ff=96 (0.622, 94k) ≈ D=128/d_ff=128
(0.625, 166k)** for ~half the params. Recommendation: **D≈96, d_ff≈1–2×**.

## Knob 5 — LCA associative-memory size `n_anchors` (acc, params)

| n_anchors | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|
| acc | 0.414 | 0.418 | 0.431 | 0.448 | **0.486** |
| params | 19.6k | 20.0k | 20.7k | 22.3k | 25.4k |

**Cheap and monotonic, but modest** (+7 pts from 4→64 anchors for ~+6k params).
The Hopfield anchor bank is a low-cost knob worth scaling for content-heavy
workloads, but it's not a primary lever on this integration task.

## Knob 3 — FFN λ-timescale range `hippo_tau_max` × evidence location (acc)

| tau_max | 16 | 64 | 256 | 1024 |
|---|---|---|---|---|
| **long-range** (frac 0.34) | 0.370 | 0.583 | 0.725 | **0.768** |
| spread (frac 1.0) | 0.344 | **0.474** | 0.430 | 0.418 |

**A long-range–specific knob with a memory/resolution tradeoff.** When evidence
sits far from the query, widening the timescale range helps **monotonically and
hugely** (+40 pts). When evidence is nearby/spread, very long timescales
**over-smooth and hurt** — the spread curve peaks at τ_max≈64 then declines.
Tune `tau_max` to the temporal scale of the discriminative evidence.

## Knob 1 — SSM state expansion: FFN modes/channel `ffn_n_modes` (acc, params)

*(base `ffn_hippo_tau_max=256`; `n_modes=1` here == knob-3 τ=256 row — consistent)*

| n_modes | params | long-range (0.34) | spread (1.0) |
|---|---|---|---|
| 1 | 36k | 0.725 | 0.430 |
| 2 | 51k | **0.938** | 0.488 |
| 4 | 81k | 0.845 ± .061 | 0.383 ± .169 |
| 8 | 139k | 0.819 ± .103 | 0.188 |

**The biggest long-range win, and it's small-and-cheap: `n_modes=2` jumps
long-range accuracy to 0.938** (from 0.725 at M=1, +21 pts, +15k params) — and it
**beats even the widest single-timescale tape** (τ_max=1024, M=1 → 0.768). Two
timescales per channel carry the far signal far better than one wide-range
channel. But **M > 2 overfits/destabilizes** at fixed budget (M=8 collapses to
0.188 on the spread task, high variance). A **small mode-bank (M=2)** is the
recommendation — exactly the S5 "state expansion beats width for temporal reach"
result, in miniature.

---

## Synthesis — the knob ranking for the audio scale-up

1. **SSM state expansion (`ffn_n_modes=2`) — highest leverage for long-range.**
   A 2-mode FFN bank is the single most effective change for carrying evidence
   across time (0.73 → 0.94), cheaper and better than any single-timescale
   alternative. Keep it small (M=2); larger banks destabilize.
2. **Width `D` — the strongest general knob**, up to a plateau (~D=96 here); pair
   with a modest FFN expansion (`d_ff` ≈ 1–2×). Best parameter efficiency at
   moderate width, not maximal.
3. **λ-range `tau_max` — match it to the evidence timescale.** Long inputs →
   large `tau_max`; short/dense evidence → moderate (over-long memory hurts).
   Combine with modes (modes + τ=256 already gave the 0.94 point).
4. **LCA `n_anchors` — cheap monotonic top-up** (LCA workloads only), modest size.

**Depth remains the knob NOT to turn** (prior `FINDINGS.md`). Concretely for the
audio model: a **shallow (2–3 block), moderately wide (D≈96) stack, FFN with
`n_modes=2` and `tau_max` matched to the utterance/evidence timescale** is the
configuration these results point to.

## Deferred knobs (noted, not yet run)
2 selectivity/Mamba (input-dependent λ), 6 Perceiver-style cross-time latent
bottleneck (adds the missing cross-position routing), 7 learnable temporal
readout/pooling, 8 attention head count, 9 multi-ω / `ResonantSTFT` front end.
Given knob-1's result, **(2) input-dependent selection over a small mode-bank**
is the natural next experiment.

## Caveats
Shrunk scale + 2 seeds (same `Phase`-GPU cost caveat as `FINDINGS.md`); M≥4 and
long-τ points carry real seed variance (reported), but the sweet-spot findings
(M=2, D≈96, τ matched to range) are well separated from noise. All drivers
append-per-trial and resume, so more seeds / larger D extend cheaply.

## Corrected regime (input_embed + pooling readout) — re-run

The numbers above used the **single-position headless** readout. The audio
pipeline has an input embedding + a **pooling** readout, which the linchpin
(`phasor_torch/results/LINCHPIN_FINDINGS.md`) showed is what makes the FFN
redundant. Re-ran all four knobs with `input_embed=true, pool_frac=0.25`
(`fixed_regime/*.csv`, ρ=0.9). The pooling readout does much of the temporal
integration itself, so it **compresses the dynamic range of every
temporal-integration knob:**

| knob | original (single-pos) | corrected (embed+pool) | verdict |
|---|---|---|---|
| **Width `D`** | 0.44→0.62 (peak D96) | 0.43→0.58 (peak D96) | ✓ **holds** — strongest, plateau ~96 |
| **FFN `d_ff`** | helps, saturates ~2× | helps, saturates | ✓ holds (modest) |
| **Modes `n_modes=2` (long-range)** | 0.73→**0.94** (Δ+0.21) | 0.74→**0.80** (Δ+0.06) | ✓ direction holds; effect **~3× smaller** |
| **λ-range (long-range)** | 0.37→0.77 (Δ+0.40) | 0.63→0.74 (Δ+0.11) | ✓ holds, smaller (pool lifts low-τ end) |
| **λ-range (short-range)** | peaks @τ=64 | peaks @τ=64 | ✓ over-long τ still hurts |
| **LCA `n_anchors`** | 0.41→0.49 monotonic | 0.39–0.43 **flat/noisy** | ✗ **does NOT hold** — was a readout artifact |

**Revised takeaways (audio-representative regime):** width remains the robust
lever (plateau ~D=96); `n_modes=2` is still the long-range peak (M>2 still worse)
but the gain is modest once a pooling readout exists; λ-range still helps long
range with the same short-range over-smoothing tradeoff; **anchors was a readout
artifact** (flat under pooling). General lesson: a benchmark's *readout* can
silently inflate temporal-integration components — evaluate knobs under a readout
that matches deployment.

## Tier-1 readout ablation — what shifts the needle (readout/loss > body knobs)

The reconciliation showed the *readout* controls the knob conclusions. So we
ablated a **Tier-1 readout upgrade** (cumulative) on LCA + input_embed, measuring
the two biggest disparities (FFN on/off at spread; modes m1→m2 at long-range),
2 seeds. Harness: `exp_readout_ladder` (`readout_ladder/ladder.csv`).

| rung (cumulative) | FFN on | FFN off | Δ FFN | modes m1 | modes m2 | Δ modes |
|---|---|---|---|---|---|---|
| R0 mean-pool + sim-loss (current) | 0.408 | 0.385 | +0.022 | 0.749 | 0.826 | +0.077 |
| R1 + softmax-CE (contrastive) | 0.466 | 0.453 | +0.013 | 0.785 | 0.866 | +0.081 |
| R2 + learnable codes | 0.466 | 0.454 | +0.012 | 0.801 | 0.867 | +0.066 |
| R3 + learnable β | 0.467 | 0.448 | +0.020 | 0.796 | 0.850 | +0.055 |
| **R4 + logsumexp-over-time** | **0.503** | **0.543** | **−0.040** | 0.823 | 0.828 | **+0.005** |

**Findings:**
1. **Contrastive softmax-CE = biggest general accuracy lever** (+~0.06 over the
   non-contrastive `similarity_loss`, which only pulls toward the true prototype).
2. **LogSumExp-over-time pooling = biggest structural lever** — lifts accuracy
   further (attn-only 0.448→**0.543**) and **flips the FFN delta negative**
   (attn-only now *beats* the FFN model) and **collapses the modes advantage**
   (Δ→+0.005). A max-over-time readout does the temporal aggregation itself, so
   the FFN *and* the mode-bank become redundant-to-harmful. This is the strongest
   reproduction of audio's "no-FFN wins," and explains the mechanism (KWS = "is
   the keyword present *somewhere*" → max-pool is the right inductive bias).
3. **Learnable codes / β = negligible** — fixed random codes + fixed temperature
   were fine; the *loss* and *pooling* are the levers, not prototype geometry.
4. Total Tier-1 accuracy gain: **+0.10 (FFN model) to +0.16 (attn-only)** — a
   large lever living entirely in the readout/loss.

**Audio recommendation:** swap `similarity_loss → softmax-CE` and `SSMReadout
mean-pool → logsumexp-over-time`; expect a bump past 79.3%, largest for the
no-FFN config.

## Reproduce
```julia
include("scripts/temporal_scaling_sweep.jl")
exp_readout_ladder(; use_cuda=true)                         # Tier-1 readout ablation
run_knobs(; use_cuda=true)                                   # original single-pos regime
run_knobs(; use_cuda=true, input_embed=true, pool_frac=0.25, # corrected audio-representative regime
          outdir="results/temporal_scaling/fixed_regime")
```
Artifacts: `results/temporal_scaling/{capacity,anchors,tau,modes}.csv` (original)
and `results/temporal_scaling/fixed_regime/*.csv` (corrected).
