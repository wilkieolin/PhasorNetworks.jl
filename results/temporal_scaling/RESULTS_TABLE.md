# Temporal-scaling & SSM-knob experiments — results summary

> **⚠ Read with the reconciliation.** These tables use the single-position headless
> TIR readout, which the phasor_torch/audio reconciliation showed **over-states
> temporal-integration knobs** (FFN, modes, λ, anchors). Under the audio-representative
> pooling readout (`fixed_regime/`), width holds, modes/λ shrink ~3–4×, and anchors
> vanishes. See `FINDINGS_knobs.md` §Corrected regime, `findings_report.html`
> §Reconciliation, and `phasor_torch/results/LINCHPIN_FINDINGS.md`.

All on the headless **TIR** task (temporal integration recall, chance = 0.0625),
shrunk scale (D=48 base / L=32 / 2 seeds / 40 epochs) unless noted. Full detail:
[`FINDINGS.md`](FINDINGS.md) (scaling & FFN), [`FINDINGS_knobs.md`](FINDINGS_knobs.md)
(SSM knobs, branch `ssm-knobs`). CSVs: `{depth,ffn,width,integration,depth_mqar,capacity,anchors,tau,modes}.csv`.

## Top line — every knob and its verdict

| Knob | Range tested | Effect | Verdict |
|---|---|---|---|
| **Depth** (# blocks) | 1→4 | 0.42 → 0.47 → 0.43 (peak@3, regresses@4) | ✗ weak, unreliable |
| **FFN present?** | off vs on | 0.17 (≈chance) → 0.49 | ✓✓ load-bearing |
| **Width `D`** | 48→128 | 0.44 → 0.65 (plateau ~96) | ✓✓ strongest general knob |
| **FFN `d_ff`** | 1×→4× | +0.04–0.07, saturates ~2× | ✓ secondary |
| **Width vs depth** @matched params | ~48k | width 0.561 vs depth 0.428 | ✓ width wins (+13pts) |
| **λ-range `tau_max`** | 16→1024 | long-range +0.40; short-range peaks@64 | ✓ match to evidence timescale |
| **Modes/channel** (SSM state exp.) | 1→8 | long-range 0.73 → 0.94 @M=2 | ✓✓ biggest long-range win |
| **LCA `n_anchors`** | 4→64 | 0.41 → 0.49 | ✓ cheap, monotonic, modest |
| **Integration real?** (m evidence frames) | 1→8 | 0.08 → 1.00 | ✓ task validated |
| **MQAR routing** (contrast) | depth 1→4 | all ≈ chance | — inconclusive at this scale |

## Study 1 — scaling & FFN (`FINDINGS.md`)

**Depth (mean of 2 seeds):**

| depth | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| LSA | 0.421 | 0.445 | **0.474** | 0.428 |
| LCA | 0.428 | 0.418 | **0.465** | 0.408 |

**FFN role:**

| FFN | off | d_ff=D/2 | d_ff=D | d_ff=2D |
|---|---|---|---|---|
| depth 2 | 0.176 | 0.370 | 0.445 | **0.486** |
| depth 4 | 0.164 | 0.401 | 0.428 | **0.474** |

**Depth vs width @ matched params:**

| ~params | deepen (D=48) | widen (depth 2) |
|---|---|---|
| 24k | 0.445 | 0.445 |
| 36k | 0.474 | **0.509** |
| 48k | 0.428 | **0.561** |

**Integration (accumulation over m evidence frames):**

| m frames | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| acc | 0.083 | 0.246 | 0.690 | **0.997** |

**Long-range integration (evidence location × FFN λ-init):**

| evidence | FFN=hippo | FFN=default |
|---|---|---|
| spread (frac 1.0) | 0.474 | 0.331 |
| far (frac 0.34) | **0.583** | 0.167 (≈chance) |

**MQAR routing contrast (far-acc, shrunk, 1 seed):** depth 1/2/4 = 0.094 / 0.112 / 0.073 (all ≈ chance — inconclusive at this scale).

## Study 2 — SSM knobs (`FINDINGS_knobs.md`, branch `ssm-knobs`)

**Width × FFN expansion (capacity):**

| D \ d_ff | 1× | 2× | 4× |
|---|---|---|---|
| 48 | 0.445 | 0.486 | 0.520 |
| 64 | 0.529 | 0.587 | 0.585 |
| 96 | 0.622 | **0.641** | 0.648 |
| 128 | 0.625 | 0.620 | 0.637 |

**λ-range `tau_max`:**

| tau_max | 16 | 64 | 256 | 1024 |
|---|---|---|---|---|
| long-range (0.34) | 0.370 | 0.583 | 0.725 | **0.768** |
| spread (1.0) | 0.344 | **0.474** | 0.430 | 0.418 |

**Modes/channel (SSM state expansion; base τ_max=256):**

| n_modes | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| long-range (0.34) | 0.725 | **0.938** | 0.845 | 0.819 |
| spread (1.0) | 0.430 | 0.488 | 0.383 | 0.188 |
| params | 36k | 51k | 81k | 139k |

**LCA anchors:**

| n_anchors | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|
| acc | 0.414 | 0.418 | 0.431 | 0.448 | **0.486** |

## Bottom line for the audio scale-up

**Shallow (2–3 blocks), D≈96, FFN `n_modes=2`, `tau_max` matched to the evidence
timescale.** Width + a small (M=2) SSM state-expansion bank are the high-leverage
levers; depth is not. Natural next experiment: knob 2 (Mamba-style input-dependent
selection over the mode-bank).

## Caveats

Shrunk scale + 2 seeds (the `Phase`-typed GPU path is allocation-heavy, so trials
are expensive). M≥4 / long-τ points carry real seed variance; the sweet-spot
findings (M=2, D≈96, τ matched, width>depth) are well separated from noise. All
drivers append-per-trial and resume — more seeds / larger D extend cheaply.
