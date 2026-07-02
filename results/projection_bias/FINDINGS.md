# complex_to_angle gradient explosion — diagnosis and fix

## Symptom
Deep phasor models (e.g. attention K/V projections) intermittently NaN at high lr.
Source: a phasor sum cancels to |z| ≈ 1e-9 right before `complex_to_angle`, whose
backward `dz = ȳ · i·z / (π · |z|²)` then explodes (max|dz| ~1e9), occasionally
tipping a step to NaN once Adam can no longer absorb it.

## What we ruled out: bias
Biasing the projections (origin shift) was investigated as the fix and **rejected**:

1. **The three-view equivalence is phase-only.** SSM ≡ ODE by derivation (magnitude
   *and* phase). Both converge to the 2D static solution only **in phase** — leakage
   has no 2D analogue, so high-leakage neurons converge slowly/imprecisely, and the
   pre-activation *magnitudes* differ from 2D by a λ-dependent SSM gain (measured:
   ~5× for `:default`, ~0.24× for `:hippo`). This gain is invisible in normal use
   because the output is angle-only.
2. **Bias is a magnitude operation**, so it reads that non-equivalent magnitude:
   `angle(signal + b)` depends on `|signal|/|b|`, which differs per view → a fixed
   bias does not behave consistently across 2D/3D/ODE.
3. **The singularity is a magnitude-zero event; the equivalence is magnitude-blind.**
   So no bias-based floor can both cure the singularity and preserve the equivalence
   — they are intrinsically at odds. (Also: the old 3D/ODE bias encoding was a
   λ-attenuated phantom-spike current that vanished on fast/HiPPO channels anyway.)

## Fix: near-origin gradient gate
A phasor collapsed to the origin carries **no useful phase** (its angle is
undefined/noise), so its gradient contributes nothing meaningful downstream. We
therefore gate it: in the `complex_to_angle` rrule, zero the cotangent for
`|z| < threshold`, with `threshold` raised `1e-10 → 1e-3`.

- Caps `max|dz|` at `≈ |ȳ|/(π·threshold)` and removes the NaN.
- **Forward pass unchanged** → parity-safe; all existing 2D/3D/ODE parity tests
  and the phase-equivalence are untouched.
- `|z| ≳ 1e-3` gates only genuinely-collapsed phasors; normal O(1) signals are
  unaffected. `threshold` is tunable per call.

Site: `src/domains.jl`, `rrule(::typeof(complex_to_angle), …)`. A guarded
`_cta_probe` diagnostic (records `(min|z|, max|dz|)` per backward, zero cost when
off) is kept for validation/debugging.

## Status
Bias-based changes reverted (attention projections back to `use_bias=false`).
The gate is the shipped fix.
