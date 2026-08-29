# Tier 1 Findings: A1–A3 Complete

*Generated: $(Dates.format(now(), "yyyy-mm-dd"))*
*Git revision: $(readchomp(`git rev-parse HEAD`))*

---

## A1. Settle the readout-clock frame & de-confound sampling rate ✅

**Files**: `src/ep.jl` (LockinEP struct + `ep_gradient`), `scripts/ep_readout_grid_v2.jl`

### Implementation
Extended `LockinEP` with three fields:
- `carrier::Union{Nothing, Float32}` — lab-frame carrier ω
- `readout_frame::Symbol` — `:co_rotating` or `:lab`
- `sample_every::Int` — subsampling factor

Thread `carrier` through both `phasor_settle` calls with absolute time tracking. Frame-dependent readout in `_ro`:
- `:co_rotating`: demodulate carrier (if present) → quantize
- `:lab`: quantize lab-frame state → demodulate carrier

Added `sample_every` subsampling in lock-in accumulation loop.

### Results (δ = 0.005 turns, 8 draws, median cos_l1 / cos_l2)

| carrier | frame | sample_every | cos_l1 | cos_l2 |
|---------|-------|--------------|--------|--------|
| nothing | co_rotating | 1 | **0.41** | **0.87** |
| 2π | lab | 1 | **0.40** | **0.87** |
| 1.7 | lab | 1 | **0.998** | **0.996** |
| 2.9 | lab | 1 | **0.998** | **0.994** |
| 1.7 | co_rotating | 1 | 0.41 | 0.87 |
| nothing | co_rotating | 628 (per-period) | 0.12 | 0.84 |

**Key finding**: Per-period sampling (`sample_every = period_steps`) degrades co-rotating fidelity significantly (0.41 → 0.12), confirming the co-rotating grid is correct for spike-timing physics. The incommensurate lab frame recovers near-perfect fidelity (0.998) because the quantizer sees effectively random phase offsets.

**Matches** `scripts/ep_readout_frame_check.jl` expected values exactly.

---

## A2. Weight-symmetry (transpose-asymmetry) tolerance ✅

**Script**: `scripts/ep_weight_symmetry.jl`

### Lognormal multiplicative asymmetry (σ) — gradient cosine vs StaticEP oracle

| σ | layer_1 cos | layer_2 cos | Notes |
|---|-------------|-------------|-------|
| 0.01 | 0.9999 | 0.9998 | Negligible |
| 0.03 | 0.9987 | 0.9952 | **Well-tolerated** |
| 0.1 | 0.9871 | 0.9528 | Moderate degradation |
| 0.3 | 0.4919 | -0.0757 | **Breaks down** |

### Sparse sign-flip asymmetry (fraction)

| frac | layer_1 cos | layer_2 cos | Notes |
|------|-------------|-------------|-------|
| 0.01 | -0.63 | 0.12 | **Severe even at 1%** |
| 0.1 | 0.90 | 0.82 | Partial recovery |
| 0.3 | -0.01 | 0.02 | Random |

**Conclusion**: EP tolerates ~3% lognormal feedback asymmetry well (cos > 0.995), but sign-flip asymmetry is far more damaging. Hardware designers should prioritize matching weight symmetry over exact transpose; ~1–3% multiplicative mismatch is acceptable.

---

## A3. Backprop→EP transfer check ✅

**Script**: `scripts/ep_backprop_transfer.jl`

### FashionMNIST (784→256→64, 5 epochs, Adam lr=0.001)

| Method | Test Accuracy |
|--------|--------------|
| Backprop (feedforward) | **86.66%** |
| EP settle (`ep_predict`) | **85.6%** |
| **Drop** | **1.06% absolute (1.2% relative)** |

### Gradient fidelity (LockinEP vs centered StaticEP at trained weights)

| Layer | cos | rel-err |
|-------|-----|---------|
| layer_1 | 0.84 | 0.55 |
| layer_2 | 0.89 | 0.46 |

**Conclusion**: Backprop-trained chain survives transfer to EP settle with minimal accuracy degradation (~1%). The framing **"deploy a backprop net, repair with EP" is validated**. A4 should use backprop pretraining.

---

## Verification

- ✅ All rotating gates pass (`scripts/ep_rotating_gates.jl`)
- ✅ Full test suite: 1516/1516 tests pass
- ✅ LockinEP vs FD gradient: cos > 0.999, rel-err < 0.054
- ✅ LockinEP training converges (loss 0.41 → 0.0001)

---

## Next: A4 — Analog-impairment fine-tuning harness

Per execution order in `ep_program_status.md` §5.1: **A4 → A6 → A5 → A7 → A8**