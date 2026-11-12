# The shared-ω ideal and real devices

> **Status:** the conventions here are fixed, and the device API lands in stages. In the library now: the card
> format (`load_device_card`) and the dimensionless device spec built from it (`device_spec`, `DeviceSpec`). Still to
> come: per-unit draws, class-R and class-L unit dynamics, per-unit ω on the discrete path, then EP. Until then the
> layers implement only the ideal described in §1.

## 1. The ideal: one shared carrier

The derivations behind the package all assume that every unit rotates at one carrier frequency `ω₀ = 2π/T`, with
`T = spk_args.t_period`. These include the SSM/Dirac equivalence, the `−conj(·)` frame correction in the discrete
path, `unrotate_solution`, and the rotating-frame theorem in `docs/phasor_lockin_derivation.tex` §Rotating Frame.
Information is relative phase, and downstream layers compare phases by interference (similarity, binding, bundling).
That comparison is exact only when all channels share the carrier.

The layers implement this ideal:
- In `PhasorDense`, `PhasorConv`, `PhasorFixed`, `PhasorResonant` and `MultiModePhasorDense`, `ω` is not stored. It is
  derived from `spk_args.t_period` at each call and is identical for every channel. Cross-channel diversity comes from
  `λ` and the weights.
- `ResonantSTFT` is the deliberate exception. It is a frequency-decomposition front end with trainable per-channel ω,
  and it re-encodes its outputs at one `omega_out` (`_freq_shift`).

The ideal is the reference, not a claim about hardware. Every run on a device model has an **ideal twin**: the same
network and weights on the shared-ω layers. Results are reported against that twin.

## 2. What real devices do

Oscillator hardware never matches the ideal. Two classes of unit matter.

**Class R (driven resonator)** is the library's neuron, `ż = (λ + iω)z + W·I(t)` with `λ < 0`. It decays to rest when
undriven. The fitted printed-MoS₂ device is class R. Suppose a unit's frequency is detuned, `ω_c = ω₀ + Δω`, and it is
driven at `ω₀`. Its steady-state response to the fundamental is multiplied by

    1 / (1 − iΔω/|λ|)

This is a static phase offset of `atan(Δω/|λ|)` and a gain of `1/√(1 + (Δω/λ)²)`. The error grows with `Δω/|λ|`, the
detuning per unit of decay rate. Units with long memory (small `|λ|`) are therefore the most sensitive, which is why
sequence models need much tighter frequency matching than feedforward ones.

**Class L (self-sustained limit cycle)** covers CMOS ring oscillators, VO₂ above threshold and spin-torque
oscillators. Such a unit runs without input and its amplitude is pinned by the device. Under a drive of effective
strength `K`, its phase relative to the drive obeys Adler's equation `φ̇ = Δω − K sin φ`.
- It locks, with a static lag `asin(Δω/K)`, when `|Δω| ≤ K`.
- Otherwise it slips cycles at the rate `√(Δω² − K²)`.

The relevant ratio is `σ_ω/K`.

**In EP** (`phasor_settle`), per-unit detuning leaves a residual `iΔω ⊙ w` in the co-rotating frame. That residual is
tangential to the torus and has no energy function. The rotating-frame cancellation then no longer holds, and a
relative equilibrium exists only if the couplings phase-lock the detuned units (`docs/phasor_lockin_derivation.tex`,
"Boundary of validity: non-uniform ω").

Other device properties covered by the same model:
- spread in decay `λ`
- white and 1/f frequency noise
- slow drift
- amplitude-dependent frequency (non-isochrony)
- orbit-speed distortion and the phase response to input
- emission rules
- wire delays and jitter
- weight precision

## 3. Working with variability

Frequency spread is expected, so the library treats ways to measure, correct or tolerate it as first-class features.
Outcomes from the ARC experiments:

| Method | What it does | Where it has worked |
|---|---|---|
| Frequency trim (per-unit, quantised) | removes most of `Δω` once, at calibration | required for any model with memory; 3–8 bits on every analog platform studied |
| Phase or delay trim | removes the static lag only | feedforward networks; insufficient where `Δω` accumulates through memory |
| Frequency-locked loop | tracks `Δω` online from the drive | only with an accurate model of amplitude-dependent frequency |
| Local demodulation reference | decodes against a reference unit in the same cycle instead of a global clock | removes reader-clock errors in lock-in EP |
| Injection / m-fold locking | pulls phases towards a reference alphabet | diagnostic only; it does not replace a frequency trim |
| Device-aware training | trains against frozen device draws | complements trims; does not remove systematic estimator bias in EP |

## 4. Conventions for device models

The device API will follow these conventions throughout.

| | Convention |
|---|---|
| C1 Units | The nominal period `T₀ = spk_args.t_period` and `ω₀ = 2π/T₀`. Cards store physical values; the simulation uses dimensionless ones: `λT`, `Q` (checked by `λT = −π/Q`), `μT`, `σ_ω/ω₀`, `σ_t/T`, `D·T`, `β`, `ν`, `K/ω₀`. |
| C2 Non-isochrony | `ω(r) = ω_c·F(ρ)`, with `ρ = r/r_ref` and `β = dF/d ln ρ` at `ρ = 1`. Shapes: `:loglinear` `1 + β·clamp(ln ρ, ±ln 3)`; `:linear` `1 + β(ρ − 1)`; `:quadratic` (Stuart–Landau, `β = −2νr₀²/ω₀`); `:table`. In EP the amplitude variable is the drive `|g|`, since states have unit modulus. |
| C3 Hill threshold | `θ` is in the library's units, `θ_D = θ_raw / (2·t_window·spk_scale)`, as used by `hill_charge` through `_spike_charge_fn`. |
| C4 Speed distortion | `S(θ)` is normalised so that `⟨1/S⟩_θ = 1`, which keeps the period unchanged. |
| C5 Orbit angle | `θ` is computed from `u/max(|u|, ε)`, never from `angle(u)`, which has no usable gradient at rest. |
| C6 Seeding | Device draws use a fixed integer mix of `(seed, layer id, stream)` into `Xoshiro`, never `Base.hash`. Static draws come from a device seed; dynamic noise comes from a separate noise seed. |
| C7 Decode clock | Spikes travel between layers as physical times. A clock matters only where something samples or reads, such as cycle-end emission, `:phase` outputs and the final readout. The default is the nominal clock, which is today's behaviour; alternatives are a reader clock with errors, the unit's own clock, and a local reference unit. |
