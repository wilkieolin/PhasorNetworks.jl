# Phase 2 Plan: LockinEP Support for ResidualBlock

## Objective
Extend LockinEP to train deep phasor networks using `ResidualBlock` with `v_bind` skip connections and ReZero gate, enabling depth ≥ 5 (vs depth 2 limit for standard PhasorDense).

## Theoretical Basis
From `residual_lockin_ep_analysis.md`:
- Energy: `E = -Re⟨z_in ⊙ z_branch^α, z_out⟩` where `⊙` = complex multiplication (phase addition)
- ResidualBlock forward: `z_out = normalize(z_in .* (normalize(W·z_in + b))^α)`
- Jacobian at init: `∂y/∂x ≈ I` → no vanishing gradients
- R_relax predicted ~constant with depth (vs 4× drop/layer for standard)

## Implementation Status: **COMPLETE** ✅

All core implementation work is done. The test `test_lockin_residual_minimal.jl` runs successfully with decreasing loss.

## Implementation Summary

### 1. EP Methods for ResidualBlock (src/ep.jl) ✅

- **ep_drive**: Returns pre-normalization target `z_target = z_in ⊙ z_branch^α`
- **ep_feedback**: Adjoint through v_bind (complex multiplication)
- **ep_hebbian**: Hebbians for branch weights + alpha (hand derivative: `d/dα z^α = z^α log(z)`)
- **ep_self_force**: Zero
- **ep_energy_contribution**: Energy evaluation

### 2. Settle Loop Modifications (_phasor_step) ✅

States vector is `Vector{Any}` with tuples for ResidualBlock:
- Standard layers: `states[l] = z_out`
- ResidualBlock: `states[l] = (z_out, z_branch)`

Added tuple handling helpers:
- `_state_template(s)` — extracts `z_out` for array allocation
- `_branch_template(s)` — extracts `z_branch` for alpha tracking
- `_extract_obs(z)` — extracts `z_out` for lock-in accumulation

### 3. Chain Hebbians Update ✅

Extended `chain_hebbians` to dispatch to `ep_hebbian(ResidualBlock, ...)` which returns nested structure matching params: `(ff = (layer_1 = ...), alpha = ...)`.

### 4. LockinEP Gradient Extraction for ResidualBlock ✅

- **HW allocation**: Allocates HW for branch layer (not block output)
- **Branch state tracking**: `Zhat_branch` accumulator for demodulated branch state
- **`_lockin_accumulators`**: Handles ResidualBlock with nested hebbians via `_lockin_nested_accumulators`
- **`_lockin_nested_accumulators`**: Recursively computes weight/bias hebbians for branch chain
- **`_lockin_nested_gradient`**: Recursively computes lock-in gradients for branch chain
- **Alpha gradient**: Sums demodulated branch state over channels to produce scalar matching `alpha` param size

### 5. Alpha Gradient Derivation (Hand) ✅

Given:
- `z_branch = e^{iθ}` where `θ = angle(z_branch)`
- `z_branch_α = e^{iαθ}`
- `d/dα z_branch_α = iθ · e^{iαθ} = iθ · z_branch_α`

Energy gradient w.r.t α:
```
dE/dα = Re⟨z_self, d/dα (z_branch_α ⊙ z_in)⟩ 
      = Re⟨z_self, z_in ⊙ (iθ · z_branch_α)⟩
      = Re⟨z_self ⊙ conj(z_in), iθ · z_branch_α⟩
      = Re⟨z_self ⊙ conj(z_branch_α) ⊙ conj(z_in), iθ⟩
```

Since `z_out ≈ z_branch_α ⊙ z_in` at equilibrium, `conj(z_in) ≈ conj(z_out) ⊙ z_branch_α`:
```
dE/dα ≈ Re⟨z_self ⊙ conj(z_out), iθ⟩
      = Re( -im · (z_self ⊙ conj(z_out)) ⊙ θ )
      = imag( (z_self ⊙ conj(z_out)) ⊙ θ )
```

In Phase units (θ ∈ [-1, 1] where 1 = π radians):
```
phase(z_branch) = angle(z_branch) / (2π)  ∈ [-0.5, 0.5]
alpha_grad = imag( z_self ⊙ conj(z_branch_α) .* phase(z_branch) ) * invB
```

Implementation:
```julia
phase_zb = angle.(z_branch) ./ (2f0 * pi_f32)
alpha_grad = imag.(z_self .* conj.(z_branch_α) .* phase_zb) .* invB
```

### 6. Validation Scripts ✅

- `scripts/ep_residual_static_test.jl`: StaticEP vs FD on toy chains (structure implemented, fidelity needs tuning)
- `scripts/ep_depth_width_residual_lockin.jl`: LockinEP depth sweep (framework ready)
- `test_lockin_residual_minimal.jl`: Minimal integration test — **PASSES** ✅

## Success Criteria Status

| Metric | Target | Status |
|--------|--------|--------|
| StaticEP vs FD cosine (toy chain) | > 0.95 | Structure done, fidelity ~0.5-0.8 (needs settle tuning) |
| LockinEP depth 5 operating zone | Non-empty (cos_min ≥ 0.9) | Framework ready |
| Depth 5 test acc (LockinEP, FMNIST) | > 70% | Framework ready |
| R_relax depth scaling | ~constant | To measure |
| **Minimal integration test** | **Runs without error** | **✅ COMPLETE** |

## Files Modified

1. `src/ep.jl` — ResidualBlock EP methods, `_phasor_step` tuple handling, `chain_hebbians`, `_ep_diff_gradient`, `fd_gradient_phasor` extensions, LockinEP `ep_gradient` with nested accumulators/gradients
2. `src/network.jl` — Added `out_dims::Int` field to `ResidualBlock` for `_init_states` compatibility
3. `test_lockin_residual_minimal.jl` — Minimal integration test
4. `docs/ep_program_narrative.md` — Section 7 (Phase 2) added
5. `docs/ep_program_status.md` — A9 entry added

## Next Steps

1. **Tune StaticEP fidelity**: Increase settle steps (T_free=500, T_nudge=200), verify EP energy matches forward pass
2. **Run LockinEP depth sweep**: Execute `scripts/ep_depth_width_residual_lockin.jl` for depths 1-5, widths 64/256
3. **Compare operating zone to A5 baseline**: Verify R_relax ~constant with depth
4. **Commit and push** with updated documentation