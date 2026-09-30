# Review: `manuscript_liep.tex` vs. Primer & Implementation

**Date**: 2025-09-01  
**Documents compared**:
- `manuscript_liep.tex` (hand-written manuscript, 147 lines)
- `phasor_lockin_ep_primer.tex` (pedagogical primer, 446 lines)
- `src/ep.jl` (implementation, 2077 lines)

---

## Executive Summary

| Aspect | Alignment |
|--------|-----------|
| Conceptual narrative | ✅ Strong — Manuscript follows Primer structure |
| Mathematical notation | ✅ Consistent (Wirtinger, energy, cost/loss) |
| **Discrete settle equation** | ❌ **Manuscript Eq 12 ≠ Implementation** |
| **Decay term origin** | ⚠️ Manuscript says "from energy"; actually design choice |
| **Rotating frame / carrier** | ❌ Missing from Manuscript |
| **Soft projection / near-zero drives** | ❌ Missing from both |
| **Sign convention (cost in energy)** | ⚠️ Manuscript `+βC` vs Primer/Impl `-βC` |
| **Gradient sign** | ⚠️ Manuscript `+` vs Primer/Impl `-` |
| **Full Wirtinger derivation** | ✅ Primer Appendix A has it; Manuscript states result only |
| **Lock-in demodulation math** | ✅ Both have; Primer shows `2/ε` origin more explicitly |
| **Operating regime / limitations** | ⚠️ Both have placeholders |

**Overall**: Manuscript is a correct *conceptual* condensation of the Primer, but **sweeps 4 critical implementation details under the rug** and has **2 sign-convention errors** that must be fixed before submission.

---

## Section-by-Section Comparison

| Manuscript § | Primer § | Topic | Status |
|--------------|----------|-------|--------|
| 1.1 | 1 | Vanilla EP intro | ✅ Aligned |
| 1.2.1 | 2.1 + Table 1 | Phase energy functions | ✅ Manuscript more compact |
| 1.2.2 | 2.2–2.4 | Energy descent, flow, projection | ⚠️ **Conflates 3 steps; Eq 12 wrong** |
| 1.2.3 | 3.1–3.4 | Non-holomorphic map, AC probe | ✅ Manuscript skips Fig 3 (sidebands) |
| 1.2.4 | 4.1–4.2 | Lock-in gradient | ✅ Primer adds §4.1 bridge |
| — | 5 (placeholder) | Operating regime | ❌ Both missing |
| — | 6 (placeholder) | Scope/limitations | ❌ Both missing |
| — | Appendix A | Full Wirtinger derivation | ✅ Primer only |
| — | Appendix A.2 | Decay origin | ✅ Primer only |
| — | Appendix A.3 | Projected Euler proof | ✅ Primer only |

---

## Critical Discrepancies (10 Items)

### 1. Discrete Settle Equation ≠ Implementation
| Source | Equation |
|--------|----------|
| **Manuscript Eq 12** / **Primer Eq 28** | `z ← unit_project(z + dt·(-z + W z_{ℓ-1} + W^T z_{ℓ+1}))` |
| **Implementation** (`_project_damp`) | `z_new = (1-dt)·z + dt·unit_project(-z + W z_{ℓ-1} + W^T z_{ℓ+1})` |

**Why it matters**: These are different integrators. The implementation form is a **convex combination** of old state and projected drive; the textbook form projects the full Euler step. They are *equivalent to O(dt)* but differ at finite dt.

**Required**: Show both in Primer; add equivalence remark. Manuscript should note the implementation form.

---

### 2. Decay Term Origin
| Source | Claim |
|--------|-------|
| **Manuscript Eq 10** | "linear decay term `-γ z_ℓ` ... with `γ > 0` to ensure Lyapunov-stable" (implies from energy) |
| **Primer §2.3** | "not derived from Φ; design choice ... γ=1" |
| **Implementation** | `-z` term comes from `(1-dt)z` damping in `_project_damp`; `K_mode=:stored` adds `½λz` (dissipative only); ω (symplectic) applied as exact rotation AFTER projection |

**Required**: Manuscript must correct: decay is **not from energy**; it's a stabilizing design choice (or from oscillator physics, not Φ).

---

### 3. Rotating Frame / Exact Carrier (Missing from Manuscript)
| Source | Content |
|--------|---------|
| **Manuscript** | No mention |
| **Primer** | Appendix mentions "rotating frame equivalence (Appendix C)" — not written |
| **Implementation** | `carriers` argument in `phasor_settle`; exact multiplicative rotation `rot = cis(ω·dt)` applied **after** projection (code lines 908–926). Exact at any dt — no Nyquist limit because rotation is never discretized. Co-rotating frame: `w_{n+1} = (1-dt)w_n + dt·û(g_w)` identical to non-rotating. |

**Required**: Add subsection in Primer §2.6; note in Manuscript.

---

### 4. Soft Projection / Near-Zero Drive Handling (Missing from Both)
| Source | Content |
|--------|---------|
| **Manuscript/Primer** | Only hard projector `g/|g|` |
| **Implementation** | `_project_damp_soft` with `ε` regularization: `u = g / sqrt(|g|² + ε²)`. Continuous at `g=0` (→ 0, not jump to `1+0im`). Hard projector is discontinuous at `|g|→0`; causes bimodal gradient failures (see `ANTICORRELATED_GRADIENT_NOTE.md`). Soft form is U(1)-equivariant: `u(e^{iφ}g) = e^{iφ}u(g)`. Phase-interpolating form (from `soft_normalize_to_unit_circle`) breaks carrier cancellation and U(1) invariance — degrades EP-vs-FD from 0.023→0.198. |

**Required**: Add Primer §2.7 with implementation form and rationale.

---

### 5. Cost Sign in Energy
| Source | Energy |
|--------|--------|
| **Manuscript Eq 22** | `F = E + β C(s,y)` |
| **Primer Eq 2 / 111** | `Φ = ∑ Re⟨W z_{ℓ-1}, z_ℓ⟩ - β C(z_L, y)` |
| **Implementation** (`ep_energy_contribution`) | Uses `-β * cost` |

**Required**: Manuscript **must change to minus sign** (Primer/Impl convention). This flips the nudged gradient direction.

---

### 6. Gradient Sign in EP Theorem
| Source | Formula |
|--------|---------|
| **Manuscript Eq 27** | `∂L/∂θ = lim (1/β)[∂E/∂θ|_β - ∂E/∂θ|_0]` |
| **Primer Eq 35** | `∂L/∂W = -2/ε ⟨H cos⟩` |
| **Implementation** (`ep_gradient`) | Returns **negative** of `∂L/∂θ` (docstring: "returned gradient is the negative of ∂L/∂θ") |

**Required**: Explicit sign convention statement in both. Primer convention (minus) matches implementation.

---

### 7. Wirtinger Derivative Factors (½ in Primer, Implicit in Manuscript)
| Source | Derivation |
|--------|------------|
| **Manuscript Eqs 5–6** | States `∂Φ/∂z̄_ℓ = W z_{ℓ-1} + W^T z_{ℓ+1}` directly |
| **Primer Appendix A.1** | Shows `∂Φ_ℓ/∂z̄_ℓ = ½ W z_{ℓ-1}` from each coupling; sums two → full result |
| **Implementation** | `ep_hebbian` returns `real(z_self * z_in')` = `Re(z_ℓ z_{ℓ-1}^H)` — matches Primer |

**Why it matters**: The ½ factors come from `Re⟨u,v⟩ = ½(u^H v + v^H u)`. Primer is more explicit; Manuscript is correct but opaque.

---

### 8. Sideband "Interference" Language
| Source | Text |
|--------|------|
| **Manuscript Eq 18 vicinity** | "sidebands now constructively interfere to directly produce the sum a+b" |
| **Primer Fig 3 caption** | "both sidebands carry the coherent sum a+b" |
| **Math** | `β=β̄` → both terms in linearized map are identical → response IS `ε(a+b)cos(ω_p t)` — not interference, just equality |

**Required**: Manuscript should say "both Wirtinger components contribute coherently" not "interfere".

---

### 9. Batch Dimension (Missing from Both)
| Source | Content |
|--------|---------|
| **Manuscript/Primer** | Single-sample notation `z ∈ ℂ^d` |
| **Implementation** | States can be `(d,)` (vector) or `(d,B)` (matrix). `_init_states`, `_phasor_step` handle both via `map` and broadcasting. Drive caching works for batches. |

**Required**: Add remark in Primer about batched settling.

---

### 10. Lock-in Demodulation Factor `2/ε` Derivation
| Source | Detail |
|--------|--------|
| **Manuscript Eq 22** | States result `≈ (ε/2) ∂Φ/∂W` |
| **Primer Eq 32** | Same, with integral |
| **Missing** | Explicit: `H(t) ≈ H_0 + (ε/2)(a+b)cos(ω_p t)`, multiply by `cos(ω_p t)`, `∫ cos² = T/2` → `(2/T)∫ = (ε/2)(a+b)`. Then EP theorem: `∂L/∂W = -(a+b)` → `∂L/∂W = -(2/ε) ⟨H cos⟩` |

**Required**: Expand Primer §4.1 to show full derivation.

---

## Manuscript Corrections List (Specific Edits)

| Location | Current | Corrected |
|----------|---------|-----------|
| Eq 22 | `F = E + β C` | `Φ = E - β C` |
| Eq 10 vicinity | "decay term ... to ensure Lyapunov-stable" | "decay term `-γ z` (γ=1) is a stabilizing design choice; not from Φ" |
| Eq 12 | `z ← unit_project(z + dt·g)` | Show both forms; note implementation uses `(1-dt)z + dt·unit_project(g)`; equivalent at O(dt) |
| Eq 27 | `∂L/∂θ = lim (1/β)[∂E/∂θ|_β - ∂E/∂θ|_0]` | Add sign convention: implementation returns `-∂L/∂θ`; with `Φ = E - βC`, the formula holds as written |
| §1.2.3 | "constructively interfere" | "contribute coherently" |
| New subsection | — | Add rotating frame / carrier (§1.2.5) |
| New subsection | — | Add soft projection note (§1.2.6) |

---

## Missing Content Inventory

### In Primer, Not in Manuscript
- Table 1 (three substitutions)
- Figure 1 (state space: line vs circle)
- Figure 2 (free settle spiral)
- Figure 3 (sideband comparison — complex vs real probe)
- Figure 4 (timeline: probe → Hebbian → demod → gradient)
- §2.3 explicit "decay not from energy"
- §2.4 projected Euler justification
- §4.1 bridge from state response to weight gradient
- Appendix A (full Wirtinger)
- Appendix A.2 (decay origin)
- Appendix A.3 (projected Euler proof)
- Bibliography

### In Manuscript, Not in Primer
- Citations (Scellier & Bengio 2017, Laborieux et al. 2022, etc.)
- Explicit "EP as recipe" framing
- Compact energy gradient statement (Eqs 5–6)

---

## Next Steps

1. **Create bibliography** (search + verify references)
2. **Write companion proof** `liep_companion_proof.tex` mirroring Manuscript §1.2 structure
3. **Revise primer** with all fixes above + §5–6 content
4. **Produce corrected manuscript** (optional patched version)

---

*End of review.*