# Theoretical Foundations: ResidualBlock with v_bind Skip Connections under LockinEP

## Executive Summary

This document analyzes why `ResidualBlock` (with `v_bind` skip connections) should enable deep phasor networks under LockinEP. The key findings are:

1. **Energy function compatibility**: The residual phase network has a well-defined energy landscape where the fixed point condition `ff(z*) ≈ 0` (mod 2π) naturally drives the branch toward identity.

2. **`remap_phase` with `ignore_derivatives` is NOT a blocker for LockinEP**: LockinEP extracts gradients from the *settle dynamics*, not from autodiff through `remap_phase`. The modulo operation affects the forward pass phase values but the EP gradient flows through the differential equations governing the settle.

3. **Linearized dynamics are favorable**: At initialization with `branch_init_scale=0.1`, the Jacobian `∂y/∂x ≈ I`, preventing vanishing/exploding gradients through the skip path.

4. **ReZero gate provides adaptive depth allocation**: The learnable `α` scales the effective nudge at each block, naturally shrinking with depth.

5. **Predicted `R_relax` scaling**: Residual connections provide direct coupling paths, improving relaxation rates from ~4× drop per layer (standard) to near-constant (residual).

---

## 1. Energy Function for Residual Phase Network

### 1.1 Standard PhasorDense Energy

For a standard chain of `PhasorDense` layers, the energy being descended in EP is:

```
Φ_standard = Σ_l Re⟨W_l z_{l-1}, z_l⟩ - β·C(z_L, y)
```

where `z_l = normalize_to_unit_circle(W_l z_{l-1} + b_l)` and `C` is the cost (e.g., `SimilarityCost` or `CodebookCost`).

### 1.2 Residual Phase Network Energy

For a `ResidualBlock` with `v_bind` skip connection:

```
y = v_bind(x, ff(x)) = remap_phase(x + ff(x))
```

The forward pass performs **phase addition** (binding in VSA terminology). In the complex domain, this corresponds to:

```
z_out = z_in ⊙ z_branch = exp(iπ·x) · exp(iπ·ff(x)) = exp(iπ·(x + ff(x)))
```

where `⊙` denotes element-wise complex multiplication (phase addition).

The energy function for a chain of `L` residual blocks becomes:

```
Φ_residual = Σ_{l=1}^L Re⟨z_{l-1} ⊙ z_{branch,l}, z_l⟩ - β·C(z_L, y)
           = Σ_{l=1}^L Re⟨z_{l-1} ⊙ normalize_to_unit_circle(W_l z_{l-1} + b_l), z_l⟩ - β·C(z_L, y)
```

### 1.3 Fixed Point Condition

At equilibrium, the settle dynamics satisfy:

```
z_l^* = normalize_to_unit_circle( z_{l-1}^* ⊙ z_{branch,l}^* + feedback_l )
```

For the free phase (β = 0), and neglecting feedback for a single block analysis:

```
z^* = normalize_to_unit_circle( z^* ⊙ z_branch^* )
```

In the phase domain, this is:

```
θ^* = remap_phase( θ^* + θ_branch^* )  ⇒  θ_branch^* ≡ 0 (mod 2)
```

**Interpretation**: The residual branch must drive the phase toward **identity** (zero phase shift). This is exactly what the initialization scheme achieves: with `branch_init_scale=0.1`, the branch output `ff(x) ≈ 0` at initialization, so the block starts at the fixed point.

---

## 2. Linearized Dynamics Around Identity

### 2.1 Jacobian Analysis

At initialization, `ff(x) ≈ 0` (due to `branch_init_scale=0.1`), so:

```
y = remap_phase(x + ff(x)) ≈ remap_phase(x) = x
```

The Jacobian of the residual block output with respect to input:

```
∂y/∂x = ∂/∂x remap_phase(x + ff(x))
      = ∂remap_phase/∂z |_{z=x+ff(x)} · (I + ∂ff/∂x)
```

Since `remap_phase` is locally the identity away from the wrap boundaries (±1):

```
∂remap_phase/∂z ≈ I  (for |x + ff(x)| < 1)
```

Thus:
```
∂y/∂x ≈ I + ∂ff/∂x
```

With `branch_init_scale=0.1`, `||∂ff/∂x|| ≈ 0.1`, so:

```
∂y/∂x ≈ I + O(0.1)
```

**Key result**: The skip path carries the identity gradient. No vanishing/exploding gradients — the Jacobian eigenvalues are clustered around 1.

### 2.2 ReZero Gate Effect

With `gate=:rezero`, the block computes:

```
y = v_bind(x, α · ff(x)) = remap_phase(x + α · ff(x))
```

Jacobian:
```
∂y/∂x ≈ I + α · ∂ff/∂x
```

At initialization `α = alpha0 = 0.1`, the branch contribution is further scaled. During training, `α` adapts per-block (typically shrinking with depth), providing **adaptive depth allocation**.

---

## 3. LockinEP Probe Propagation Through v_bind

### 3.1 LockinEP Mechanism Recap

LockinEP applies a temporal cosine probe:
```
β(t) = ε · cos(ω_p · t)
```

The nudge force enters at the output layer:
```
nudge_force(cost, z_L, β(t)) = (β(t)/d) · y_target   (for SimilarityCost)
```

This force propagates **backward through the settle dynamics** via the feedback terms in `_phasor_step`:
```
grad_l = ep_drive(layer_l, z_{l-1}) + ep_self_force(layer_l, z_l) 
       + ep_feedback(layer_{l+1}, z_{l+1}) + [nudge if l=L]
```

### 3.2 v_bind in Forward Pass vs. EP Gradient Flow

**Crucial distinction**: The `ignore_derivatives()` in `remap_phase` only affects **automatic differentiation (Zygote)**. LockinEP does **not** use Zygote to differentiate through the forward pass. Instead, it:

1. Runs the settle dynamics (forward Euler on the energy gradient)
2. Records Hebbian outer products `z_l ⊗ z_{l-1}^*` at equilibrium
3. Demodulates the Hebbians at the probe frequency `ω_p`
4. Computes gradients from the demodulated accumulators

The EP gradient theorem states:
```
dL/dW ≈ -2 · Re(H_ω_p) / (T_lockin · ε)
```
where `H_ω_p` is the Fourier coefficient of the Hebbian at `ω_p`.

**The `remap_phase` modulo operation affects the *forward* phase values that enter the settle dynamics, but the gradient extraction happens via the *dynamics themselves*, not through autodiff of `remap_phase`.**

### 3.3 Adjoint of v_bind for EP

In the energy framework, the `v_bind` operation corresponds to the energy term:
```
E_bind = -Re⟨z_in ⊙ z_branch, z_out⟩
```

The gradient with respect to `z_in` (feedback) is:
```
∂E_bind/∂z_in^* = -z_branch ⊙ z_out
```

In the phase domain, this is phase addition of the branch output and the downstream state — which is exactly what the settle dynamics compute via the feedback term `ep_feedback`.

**Conclusion**: The EP gradient flow through `v_bind` is naturally handled by the existing `ep_feedback` mechanism. No special adjoint for `remap_phase` is needed because EP doesn't differentiate through it — it uses the energy gradient which is well-defined in the complex domain.

---

## 4. ReZero Gate in EP Context

### 4.1 Effective Nudge Scaling

With ReZero gate, the block output is:
```
y = remap_phase(x + α · ff(x))
```

During the nudged phase of LockinEP, the probe `β(t)` enters at the output. The effective nudge seen by block `l` is scaled by the product of gates downstream:

```
β_eff(l) = β(t) · Π_{k=l+1}^L α_k
```

Since `α_k` typically shrinks with depth (adaptive depth allocation), **deeper blocks receive a smaller effective nudge**. This is actually beneficial:

- Shallow blocks (near output) need stronger nudges to overcome the accumulated transformations
- Deep blocks (near input) operate closer to identity, so smaller nudges suffice
- The gate `α` naturally learns this allocation

### 4.2 Gradient Expression with ReZero

The Hebbian for block `l` with ReZero is:
```
H_l = z_l · (α_l · z_branch,l)^*  (for weight)
```

The LockinEP demodulation extracts:
```
H_l,ω_p = Σ_t H_l(t) · e^{-iω_p t}
```

The gradient for the branch weights:
```
dL/dW_branch,l ∝ -2 · Re(H_l,ω_p) / (T_lockin · ε)
```

The gate parameter `α_l` has its own gradient:
```
dL/dα_l ∝ -2 · Re( Σ_t z_l(t) · conj(z_branch,l(t)) · e^{-iω_p t} ) / (T_lockin · ε)
```

---

## 5. R_relax Improvement with Residual Connections

### 5.1 Relaxation Rate Definition

The relaxation rate `R_relax` governs the exponential convergence to equilibrium:
```
||z(t) - z*|| ∝ exp(-R_relax · t)
```

It is determined by the spectral radius of the linearized settle dynamics Jacobian.

### 5.2 Standard Chain (No Skip)

For a standard chain of `PhasorDense` layers with damped iteration:
```
z_l ← (1-dt)·z_l + dt·normalize(W_l z_{l-1} + b_l + feedback)
```

The coupling Jacobian has eigenvalues approaching 1 as depth increases. Empirically (from `ep_fashionmnist.jl`):
- Depth 2: `R_relax ≈ 0.134` / time-unit
- Depth 3: `R_relax` drops ~4× (to ~0.03)
- Each additional layer multiplies the slowest mode by the layer's contraction factor

### 5.3 Residual Chain (With v_bind Skip)

For a residual block:
```
z_l ← (1-dt)·z_l + dt·normalize(z_{l-1} ⊙ z_branch,l + feedback)
```

Linearized around identity (`z_branch ≈ 0`):
```
δz_l ← (1-dt)·δz_l + dt·(δz_{l-1} + ∂ff/∂x · δz_{l-1} + feedback')
```

The identity term `δz_{l-1}` provides a **direct coupling path** with eigenvalue 1 (before damping). The effective Jacobian for the residual chain has the form:

```
J_residual = (1-dt)I + dt·(I + J_branch) ≈ I - dt·I + dt·J_branch
```

Compared to standard:
```
J_standard = (1-dt)I + dt·J_dense
```

The residual connection adds `dt·I` to the diagonal, shifting eigenvalues away from zero. The slowest mode relaxation rate becomes:

```
R_relax,residual ≈ R_relax,standard + O(dt)
```

**Prediction**: With residual connections, `R_relax` becomes **approximately constant with depth** (or degrades much more slowly), enabling LockinEP to use larger `ω_p` (faster probes) at greater depths.

### 5.4 LockinEP Adiabaticity Condition

LockinEP requires:
```
ω_p ≪ R_relax  (adiabatic tracking)
```

With standard chains: `R_relax` drops 4× per layer → must decrease `ω_p` exponentially with depth.

With residual chains: `R_relax` ≈ constant → `ω_p` can remain fixed, or depth scaling is polynomial not exponential.

---

## 6. Compatibility of `remap_phase` with LockinEP

### 6.1 The `ignore_derivatives()` Issue

`remap_phase` (vsa.jl:285-316) wraps the modulo operation in `ignore_derivatives()`:

```julia
function remap_phase(x::AbstractArray)
    ignore_derivatives() do
        x = x .+ 1.0f0
        x = mod.(x, 2.0f0)
        x = x .- 1.0f0
    end
    return Phase.(x)
end
```

**This blocks Zygote gradients** through the modulo operation. However:

### 6.2 Why It Doesn't Block LockinEP

| Aspect | Zygote (Backprop) | LockinEP |
|--------|-------------------|----------|
| Gradient source | Autodiff through forward pass | Settle dynamics (physics) |
| `remap_phase` role | Differentiated through | Part of forward dynamics |
| Gradient flow | `∂L/∂x = ∂L/∂y · ∂y/∂x` | `dL/dW ∝ ⟨z_out ⊗ z_in⟩_ω_p` |
| Modulo effect | Zero gradient at wrap | Phase value enters dynamics |

The modulo operation simply **wraps the phase value** that enters the settle dynamics. The energy gradient with respect to that phase value is still well-defined (it's the conjugate phase difference). LockinEP measures how the *equilibrium phase* responds to the nudge, not how the *forward computation* would change under infinitesimal perturbation.

### 6.3 Potential Issue: Phase Wrapping at Equilibrium

If the equilibrium phase sits near the wrap boundary (±1), small nudges could cause discontinuous jumps in the wrapped phase, creating a **non-smooth energy landscape**.

**Mitigation**: 
1. The activation `normalize_to_unit_circle` and `soft_normalize_to_unit_circle` keep phases away from the origin (where phase is undefined), but don't prevent wrap-boundary proximity.
2. In practice, with `branch_init_scale=0.1`, the residual branch output is small, so the equilibrium stays near the input phase, away from wrap boundaries unless the input itself is near ±1.
3. The `Phase` type uses `[-1, 1]` in units of π, so the wrap is at ±π. The `0.5·tanh` encoding in `ep_fashionmnist.jl` keeps inputs in `[-0.5, 0.5]`, far from boundaries.

**Recommendation**: Monitor equilibrium phase distributions during training. If phases cluster near ±1, consider:
- Using `soft_remap_phase` with a smooth modulo (e.g., `atan(sin, cos)` based)
- Adding a small phase bias to keep representations centered

---

## 7. Recommendations for Phase 2 Implementation

### 7.1 Option A: Direct LockinEP on ResidualBlock (Recommended)

**Approach**: Implement `ep_drive`, `ep_feedback`, `ep_hebbian` for `ResidualBlock` and run LockinEP directly.

**Required methods**:
```julia
# Forward drive: x + branch_output (in complex domain: z_in ⊙ z_branch)
function ep_drive(rb::ResidualBlock, ps, st, z_in)
    # z_branch = ep_drive(rb.ff, ps.ff, st.ff, z_in)
    # return z_in .* z_branch  (phase addition = complex multiplication)
end

# Feedback: adjoint through v_bind
function ep_feedback(rb::ResidualBlock, ps, st, z_out)
    # z_branch = ep_drive(rb.ff, ps.ff, st.ff, z_in)  # need z_in!
    # return conj(z_branch) .* z_out
end

# Hebbian: real outer product through v_bind
function ep_hebbian(rb::ResidualBlock, ps, st, z_in, z_self)
    # Weight gradient involves z_self and z_in through v_bind
    # For ReZero: include α scaling
end
```

**Complexity**: Need to handle the phase addition properly in the complex domain. The `v_bind` in complex domain is `z_in ⊙ z_branch = z_in .* z_branch` (element-wise multiplication).

**Advantage**: Most direct path; uses existing LockinEP infrastructure.

### 7.2 Option B: Smooth Modulo for Autodiff Compatibility

**Approach**: Replace `remap_phase` with a smooth version for the EP path, or add a `smooth_remap` variant.

```julia
# Smooth remap using atan2 (differentiable everywhere)
function smooth_remap_phase(x::AbstractArray)
    # x in radians (not π units)
    return Phase.(atan.(sin.(π .* x), cos.(π .* x)) ./ π)
end
```

This avoids `ignore_derivatives` and enables Zygote gradients through the residual connection. But **LockinEP doesn't need this** (see §6.2).

**Advantage**: Enables hybrid training (EP + backprop) and gradient checking.

### 7.3 Option C: Equilibrium Propagation on the Complex Domain Directly

**Approach**: Run EP entirely in the complex domain (unit circle), avoiding `remap_phase` entirely. The residual connection becomes complex multiplication `z_out = z_in .* z_branch`, which is holomorphic and has a clean adjoint.

**Advantage**: Cleanest mathematical formulation; matches the physical spiking substrate.

**Challenge**: The `Codebook` readout and `similarity` cost are defined in the phase domain. Need consistent complex-domain equivalents.

---

## 8. Implementation Priority

### 8.1 Immediate (Week 1-2)
1. **Add `ep_drive`, `ep_feedback`, `ep_hebbian` for `ResidualBlock`** in `ep.jl`
2. **Test with StaticEP first** (simpler, validated against FD)
3. **Verify gradient matches FD** on toy chains with residual blocks

### 8.2 Short-term (Week 3-4)
1. **Run LockinEP on residual chains** at increasing depths
2. **Measure `R_relax` vs depth** for residual vs standard
3. **Tune `ω_p` and `ε`** for the improved relaxation rates

### 8.3 Medium-term (Month 2)
1. **Add smooth remap variant** for autodiff compatibility (Option B)
2. **Implement complex-domain EP** (Option C) for spiking substrate alignment
3. **Run depth sweep** with `ResidualBlock` + LockinEP on FashionMNIST

---

## 9. Mathematical Appendix

### 9.1 Energy Gradient for Residual Block

For a single residual block with complex states:
```
z_out = z_in ⊙ z_branch = z_in .* normalize(W z_in + b)
```

Energy contribution:
```
E = -Re⟨z_in ⊙ z_branch, z_out⟩
```

Gradients:
```
∂E/∂W^* = -z_branch ⊙ z_out ⊗ z_in^* = -(z_branch .* z_out) · z_in^†
∂E/∂b^* = -z_branch ⊙ z_out
∂E/∂z_in^* = -W^† (z_branch ⊙ z_out) - z_branch ⊙ z_out  (feedback)
```

The feedback term has two parts: through the branch weights and through the direct skip (identity).

### 9.2 LockinEP Demodulation with Residual

The demodulated Hebbian for layer `l`:
```
H_l = (1/T) Σ_t z_l(t) ⊗ z_{l-1}(t)^* · e^{-iω_p t}
```

For residual block `l`:
```
z_l(t) = z_{l-1}(t) ⊙ z_branch,l(t)
```

So:
```
H_l = (1/T) Σ_t [z_{l-1}(t) ⊙ z_branch,l(t)] ⊗ z_{l-1}(t)^* · e^{-iω_p t}
    = (1/T) Σ_t z_branch,l(t) ⊗ (z_{l-1}(t) ⊙ z_{l-1}(t)^*) · e^{-iω_p t}  (element-wise)
```

Since `z_{l-1} ⊙ z_{l-1}^* = |z_{l-1}|² = 1` (unit modulus):
```
H_l ≈ (1/T) Σ_t z_branch,l(t) · e^{-iω_p t}
```

**Beautiful result**: The residual structure makes the Hebbian directly measure the branch output's response to the probe, uncontaminated by the skip connection's magnitude.

---

## 10. Conclusion

**ResidualBlock with v_bind is fundamentally compatible with LockinEP**. The theoretical analysis shows:

1. ✅ **Well-defined energy landscape** with identity fixed points
2. ✅ **Favorable linearized dynamics** (Jacobian ≈ I at init)
3. ✅ **No blocker from `ignore_derivatives`** — LockinEP uses settle dynamics, not autodiff
4. ✅ **Improved relaxation rates** enabling deeper networks
5. ✅ **ReZero gate provides adaptive nudge scaling**

**Recommended path**: Implement the per-layer EP interface for `ResidualBlock` (Option A) and validate against FD on toy chains, then scale to FashionMNIST depth sweeps with LockinEP.

