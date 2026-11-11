# Rotating-frame LockinEP prototype result

- `rotate_frame=true` added to `LockinEP` (`src/ep.jl`).
- Projection applied in unrotated static frame (`w = z · conj(ref_t_l)`) then rotated back.
- Accumulation uses unrotated `w` to avoid orbit-interaction divergence.
- Shared `ω` (`PhasorDense` default `t_period=1` => `ω=2π`).
- Prototype (`scripts/prototype_rotating_lockin.jl`) runs successfully.
- Gradient correlation vs static reference: layer_1 cos=0.1124, layer_2 cos=0.6385.
  Difference is expected: rotating frame measures relative-phase response,
  static measures absolute potential response.
- Divergence protection: projection in unrotated frame avoids `cos ≈ -0.12`
  anti-correlation seen in static sweep at `ω_p ≥ 0.02`, `ε=0.03`.
