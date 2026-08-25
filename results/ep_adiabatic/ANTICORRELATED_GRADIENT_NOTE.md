# Anticorrelated gradient findings — LockinEP failure mode

## Observation (paired draw r=3, seed input=303)

- Lock-in gradient vs centered `StaticEP` reference (`ref1` vs `ref2`): `cos = -0.1197`, `relerr = 1.328` (layer 1); `cos = -0.065`, `relerr = 1.613` (layer 2).
- Reference self-consistency stays high (`cos = 0.974`), confirming the equilibrium is well-conditioned — the failure is in the demodulator, not the settle.

## Mathematical divergence from LockinEP derivation (`src/ep.jl` §6)

Derivation assumes:
```
z(β) = z_free + O(β)  (smooth deformation of the same fixed point)
```
and extracts the linear-response coefficient by demodulating `z(t)` at `ω_p`.

Numerical conditions violating this at `ε=0.03, ω_p=0.02, dt=0.5, n_cycles=2`:

1. **Hard projection (`_project_damp`, `ep.jl:517`)** — `normalize_to_unit_circle(g; ε=0)` forces `|z|=1` exactly. A large real probe `β(t) = ε·cos(ω_p t)` can push the gradient `g` to the opposite side of the unit circle, flipping `z` by π. The AC response `δz` then points opposite to the linearized gradient direction.

2. **Anti-correlation mechanism** — the demodulated accumulator `Zhat = Σ_t z(t)·e^(-iω_p t dt)` picks up the flipped-phase component. In `_ep_lockin_gradient` (`ep.jl:1028`):
```
H_W = raw · invB - c · ComplexF32.(h_dc.weight)
```
When `δz` is anti-parallel to `h_dc`, `Re(H_W)` acquires the opposite sign of the true gradient, giving `cos < 0` and `relerr > 1` (the estimated gradient is larger than the true gradient but opposite).

3. **Single-cycle basin hop** — this is the same mechanism as `StaticEP`'s large-`||W||` basin-hop (`FINDINGS.md §3`, `ep.jl:699-721`), but occurring within one probe period rather than across training epochs. The free settle (`β=0`) stays stationary; the nudged trajectory (`β(t)`) hops to a different fixed point under the projection.

4. **Insufficient integration** — `n_cycles=2` gives only `T_lockin = 1256` steps (period = 628 steps at `dt=0.5`). Nonlinear harmonic distortion from projection is not averaged out over so few cycles; more cycles (`n_cycles=8`) improve selectivity but do not eliminate the catastrophic failure mode (measured `fail%` remains >0 at best settings, just lower).

## Comparison — normal vs failure

| case | `ref1 vs ref2` | `lockin vs ref1` L1 | `lockin vs ref1` L2 | interpretation |
|---|---|---|---|---|
| r=23 (best) | cos 0.9995 | cos 0.995, relerr 0.11 | cos 0.995, relerr 0.10 | linear response holds |
| r=3 (fail) | cos 0.974 | cos **-0.120**, relerr **1.33** | cos **-0.065**, relerr **1.61** | phase-flip / basin hop |

## Conclusion

The anticorrelated gradient is not random noise. It is a deterministic divergence caused by the interaction of:
- large probe amplitude (`ε = 0.03`),
- hard unit projection (`ε=0` in projection step),
- and specific input/weight configurations that drive the state trajectory to the opposite side of the unit circle.

The fix is operational (smaller `ε`, slower `ω_p`, or centered reference for calibration), not structural — the derivation is correct when the linear-response assumption holds.
