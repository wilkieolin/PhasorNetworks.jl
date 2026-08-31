# Phase 2 Plan: LockinEP Support for ResidualBlock

## Objective
Extend LockinEP to train deep phasor networks using `ResidualBlock` with `v_bind` skip connections and ReZero gate, enabling depth ≥ 5 (vs depth 2 limit for standard PhasorDense).

## Theoretical Basis
From `residual_lockin_ep_analysis.md`:
- Energy: `E = -Re⟨z_in ⊙ z_branch^α, z_out⟩` where `⊙` = complex multiplication (phase addition)
- ResidualBlock forward: `z_out = normalize(z_in .* (normalize(W·z_in + b))^α)`
- Jacobian at init: `∂y/∂x ≈ I` → no vanishing gradients
- R_relax predicted ~constant with depth (vs 4× drop/layer for standard)

## Implementation Design

### 1. EP Methods for ResidualBlock (src/ep.jl)

**ep_drive**: Returns pre-normalization target `z_target = z_in ⊙ z_branch^α`
```julia
function ep_drive(rb::ResidualBlock, ps, st, z_in)
    d_branch = ep_drive(rb.ff, ps.ff, st.ff, z_in)
    z_branch = _project_damp(ComplexF32(1), d_branch, 1f0, 1f-10)  # normalize
    α = haskey(ps, :alpha) ? ps.alpha[1] : 1f0
    z_branch_α = z_branch .^ α
    return z_in .* z_branch_α
end
```

**ep_feedback**: Adjoint through v_bind (complex multiplication)
- Uses invertibility: `z_in = z_out ./ z_branch_α` since `|z_branch_α| = 1`
- Returns `fb_zin + fb_branch` where:
  - `fb_zin = conj(z_branch_α) .* z_out`
  - `fb_branch = ep_feedback(rb.ff, ps.ff, st.ff, z_branch_α .* z_out)`

**ep_hebbian**: Hebbians for branch weights + alpha
- Branch weights: reuses `ep_hebbian(rb.ff, ...)` with `(z_in, z_branch)`
- Alpha gradient (hand derivative):
  ```
  d/dα (z_branch .^ α) = z_branch .^ α .* log(z_branch)
  For |z|=1: log(z) = i·angle(z) = i·phase(z)
  alpha_grad = real(z_self ⊙ conj(z_branch_α) .* log(z_branch)) * invB
            = real(z_self ⊙ conj(z_branch_α) .* (im * phase(z_branch))) * invB
            = -imag(z_self ⊙ conj(z_branch_α) .* phase(z_branch)) * invB
  ```
  Since `phase(z_branch) = angle(z_branch) / (2π)` in our Phase units.

**ep_self_force**: Zero (no self-dynamics for block output)

### 2. Settle Loop Modifications (_phasor_step)

States vector becomes `Vector{Any}`:
- Standard layers: `states[l] = z_out` (ComplexF32 vector/matrix)
- ResidualBlock: `states[l] = (z_out, z_branch)` (tuple)

In settle loop:
```julia
if layer isa ResidualBlock
    z_in = (l == 1) ? z0 : (states[l-1] isa Tuple ? states[l-1][1] : states[l-1])
    # Compute branch state
    d_branch = ep_drive(rb.ff, ps.ff, st.ff, z_in)
    z_branch = _project_damp(z_branch, d_branch, dt, th)
    α = haskey(ps_rb, :alpha) ? ps_rb.alpha[1] : 1f0
    z_branch_α = z_branch .^ α
    z_target = z_in .* z_branch_α
    # Add feedback, self-force, nudge
    # Project
    z_out = project(z_self, grad_l, dt, th)
    return (z_out, z_branch)
else
    # existing logic
end
```

### 3. Chain Hebbians Update

```julia
if layer isa ResidualBlock
    z_in = (l == 1) ? z0 : (states[l-1] isa Tuple ? states[l-1][1] : states[l-1])
    z_self = states[l][1]
    z_branch = states[l][2]
    h_branch = ep_hebbian(rb.ff, ps.ff, st.ff, z_in, z_branch)
    h_alpha = ...  # computed in ep_hebbian
    push!(pairs, key => merge(h_branch, (alpha = h_alpha,)))
end
```

### 4. Alpha Gradient Derivation (Hand)

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

Implementation uses `Phase` type:
```julia
phase_zb = angle.(z_branch) ./ (2f0 * pi_f32)
alpha_grad = imag.(z_self .* conj.(z_branch_α) .* phase_zb) .* invB
```

### 5. Validation Scripts

- `scripts/ep_residual_static_test.jl`: StaticEP vs FD on toy chains
- `scripts/ep_depth_width_residual_lockin.jl`: LockinEP depth sweep (1-5, D=64,256)

## Success Criteria

| Metric | Target |
|--------|--------|
| StaticEP vs FD cosine (toy chain) | > 0.95 |
| LockinEP depth 5 operating zone | Non-empty (cos_min ≥ 0.9) |
| Depth 5 test acc (LockinEP, FMNIST) | > 70% |
| R_relax depth scaling | ~constant |

## Files to Modify

1. `src/ep.jl` - Add ResidualBlock EP methods, modify `_phasor_step`, `chain_hebbians`
2. `scripts/ep_residual_static_test.jl` - New validation script
3. `scripts/ep_depth_width_residual_lockin.jl` - New sweep script
4. `docs/ep_program_narrative.md` - Update with Phase 2 plan
5. `docs/ep_program_status.md` - Mark A5 extension in progress