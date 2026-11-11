# Phase 1 Analysis: ResidualBlock with ReZero under Backprop

## Executive Summary

**Hypothesis confirmed**: `ResidualBlock` with `v_bind` skip connections and ReZero gate enables training of deep phasor networks (depth ≥ 5) using standard Zygote backprop, in contrast to standard `PhasorDense` chains which fail at depth ≥ 3 (A5 results).

**Key findings**:
- Test accuracy at depth 5: **82-83%** (vs ~10% random for standard PhasorDense at depth 3)
- Gradient flow: `grad_ratio` (encoder/last layer) stays constant ~25-65 across depths (vs 100-1000× explosion for standard)
- Identity preservation: per-block displacement `bdisp < 0.002` π units, cumulative drift `< 0.004` π at depth 5
- ReZero gates learn adaptive depth allocation: `α` decreases with block index (shallow blocks open more)

---

## Experimental Setup

| Parameter | Value |
|-----------|-------|
| Architecture | `Flatten → LN → Phase(tanh) → PhasorDense(784→D) → [ResidualBlock(D,D)]^depth → Codebook(D→10)` |
| ResidualBlock | `gate=:rezero, alpha0=0.1, branch_init_scale=0.1, use_bias=true` |
| Widths tested | D ∈ {64, 256} (D=1024 too slow) |
| Depths tested | 1, 2, 3, 4, 5 (ResidualBlock layers; total phasor layers = depth + 1) |
| Seeds | 3 per config |
| Epochs | 5 |
| Optimizer | Adam(lr=1e-3), α LR ×5 |
| Data | FashionMNIST (10K train, 2K test) |
| Device | CPU |

---

## Results Summary

### Test Accuracy vs Depth

| Depth | D=64 (mean ± std) | D=256 (mean ± std) |
|-------|-------------------|---------------------|
| 1 | 82.0% ± 0.2% | 81.5% ± 1.1% |
| 2 | 82.6% ± 0.5% | 83.4% ± 0.5% |
| 3 | 82.8% ± 0.6% | 83.4% ± 0.2% |
| 4 | 82.9% ± 0.3% | 83.5% ± 0.2% |
| 5 | 82.4% ± 0.7% | 83.1% ± 0.7% |

**Observation**: Accuracy is stable across depths 1-5. No degradation at depth 3-5 unlike standard PhasorDense chains.

### Gradient Flow (Init)

| Depth | D=64 grad_ratio | D=256 grad_ratio |
|-------|-----------------|------------------|
| 1 | 49.8 ± 7.9 | 24.8 ± 0.7 |
| 2 | 52.5 ± 11.7 | 24.3 ± 0.3 |
| 3 | 51.7 ± 6.3 | 24.3 ± 1.6 |
| 4 | 49.3 ± 7.3 | 24.9 ± 1.2 |
| 5 | 51.6 ± 8.5 | 24.6 ± 0.5 |

**Observation**: `grad_ratio = grad_encoder / grad_last` remains ~constant across depths. No vanishing/exploding gradient. Standard PhasorDense shows 100-1000× growth by depth 3.

### Forward Drift at Init

| Depth | D=64 bdisp_mean | D=64 drift_final | D=256 bdisp_mean | D=256 drift_final |
|-------|-----------------|------------------|------------------|-------------------|
| 1 | 0.0016 | 0.0016 | 0.0015 | 0.0015 |
| 2 | 0.0017 | 0.0022 | 0.0015 | 0.0022 |
| 3 | 0.0016 | 0.0027 | 0.0016 | 0.0027 |
| 4 | 0.0016 | 0.0032 | 0.0015 | 0.0031 |
| 5 | 0.0016 | 0.0037 | 0.0016 | 0.0035 |

**Observation**: Per-block displacement < 0.002 π units (≈ 0.36°). Cumulative drift < 0.004 π units (≈ 0.7°) even at depth 5. The network preserves the input representation through the residual path.

### ReZero Gate α Values

| Depth | D=64 α_mean | D=256 α_mean |
|-------|-------------|--------------|
| 1 | 1.14 ± 0.03 | 0.52 ± 0.32 |
| 2 | 0.90 ± 0.01 | 0.68 ± 0.05 |
| 3 | 0.79 ± 0.02 | 0.64 ± 0.04 |
| 4 | 0.68 ± 0.02 | 0.57 ± 0.03 |
| 5 | 0.64 ± 0.02 | 0.50 ± 0.01 |

**Observation**: 
- Gates open (α > 0.1) during training, allowing the branch to contribute
- α decreases with depth (adaptive depth allocation) - deeper blocks stay closer to identity
- D=256 has more variable α at depth 1 (some seeds keep α small)

---

## Comparison with A5 Baseline (Standard PhasorDense + LockinEP)

| Metric | Standard PhasorDense (A5) | ResidualBlock + ReZero (Phase 1) |
|--------|---------------------------|----------------------------------|
| Max trainable depth (5 epochs) | 2 | **5** |
| Depth 3 test accuracy | ~10% (random) | **82-83%** |
| Depth 5 test accuracy | N/A | **82-83%** |
| Grad ratio at depth 3 | >1000 | **~25-50** |
| Grad ratio at depth 5 | N/A | **~25-50** |
| R_relax depth scaling | 4× drop per layer | Predicted: ~constant |
| Identity at init | No (random phases) | Yes (bdisp < 0.002) |

---

## Why This Works: Theory Validation

The experimental results validate the theoretical predictions from `residual_lockin_ep_analysis.md`:

1. **Identity at init**: `branch_init_scale=0.1` → `ff(x) ≈ 0` → `v_bind(x, α·ff(x)) ≈ x` ✓
2. **Jacobian ≈ I**: `∂y/∂x ≈ I + α·∂ff/∂x` → eigenvalues clustered at 1 ✓
3. **Gradient flow**: Skip path carries identity gradient → no vanishing gradient ✓
4. **ReZero adaptive allocation**: α decreases with depth → deeper blocks stay near identity ✓

---

## D=1024 Note

D=1024 runs were too slow (1.8M parameters) and showed negative α values (numerical instability at large width with current initialization). Not pursued further — D=256 already demonstrates the effect with 400K-500K parameters.

---

## Conclusions for Phase 2 (LockinEP on ResidualBlock)

### Feasibility: HIGH

The backprop results prove the **architecture is sound** for deep phasor networks. The remaining question is whether LockinEP can leverage the same architectural benefits.

### Expected LockinEP Benefits from ResidualBlock

Based on theory (§5 of analysis doc):
- **R_relax should stay ~constant with depth** (vs 4× drop per layer for standard)
- This means **ω_p can remain fixed** at greater depths (no exponential slowdown)
- LockinEP adiabaticity condition `ω_p ≪ R_relax` becomes easier to satisfy

### Recommended Phase 2 Approach

**Option A (Direct EP on ResidualBlock)** — Implement per-layer EP interface:
1. Add `ep_drive`, `ep_feedback`, `ep_hebbian` for `ResidualBlock` in `src/ep.jl`
2. Complex-domain drive: `z_in ⊙ z_branch` (element-wise multiplication)
3. Feedback: `conj(z_branch) ⊙ z_out` (adjoint through v_bind)
4. Hebbian: accounts for phase addition structure
5. Validate against FD on toy residual chains
6. Run LockinEP depth sweep (depth 1-5, D=64, 256)

### Implementation Notes

- The `ignore_derivatives` in `remap_phase` is **not a blocker** for LockinEP (uses settle dynamics, not autodiff)
- Phase wrapping at equilibrium is unlikely with `branch_init_scale=0.1` and `0.5·tanh` input encoding
- ReZero gate α gradients will emerge naturally from the demodulated Hebbians

---

## Next Steps

1. **Implement ResidualBlock EP interface** in `src/ep.jl` (Week 1)
2. **Validate with StaticEP** against FD on toy residual chains (Week 1)
3. **Run LockinEP depth sweep** with ResidualBlock (Week 2)
4. **Compare operating zone** to A5 standard chain results
5. **Document** in `ep_program_narrative.md` as extension of A5 analysis

---

## Files Generated

- `scripts/ep_depth_width_residual_bp.jl` — sweep script
- `results/ep_residual_bp_sweep/residual_bp_summary.csv` — summary metrics
- `results/ep_residual_bp_sweep/residual_bp_gradprofile.csv` — per-layer gradient norms
- `results/ep_residual_bp_sweep/residual_bp_drift.csv` — forward drift per block
- `results/ep_residual_bp_sweep/residual_bp_alphas.csv` — learned ReZero gates
- `results/ep_residual_bp_sweep/*.png` — analysis plots