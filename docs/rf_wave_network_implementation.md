# Implementing self-regulating sparse waves in PhasorNetworks.jl

**Companion to `docs/rf_wave_network_plan.html`.** The plan (`rf_wave_network_plan.html`)
is a *biophysical* specification: complex resonate-and-fire (R&F) units on a 2-D
sheet, coupled by a delayed difference-of-Gaussians kernel, with spike-triggered
adaptation, short-term depression, and a hard threshold+reset, tuned to
criticality. This document translates that spec into the algebra this codebase
already speaks — the **phase-SSM view** — so that the resulting layer is (a)
demonstrably self-regulating, (b) GPU-native, (c) trainable end-to-end, and (d)
a genuine *wave-based* transformation rather than a stack of discrete layers.

Status: draft alongside the first prototype (`src/wave.jl`, `demos/wave_dispersion.jl`).

---

## 1. The core reframe

Our defining equation is already a lattice of coupled oscillators:

```
dz_c/dt = k_c · z_c + W · I(t),     k_c = λ_c + i·ω
```

Today `W` is a **feed-forward inter-layer** weight. Make it a **recurrent
within-sheet** coupling and the SSM *becomes a wave medium* — reusing
`phasor_kernel`, `causal_conv`, and the ODE path unchanged. Every piece of the
biophysical spec maps onto algebra we already have:

| Plan (biophysical) | Phase-SSM algebra | Where it lives |
|---|---|---|
| R&F unit `(−λ+iω)z` | `k = -exp(log_neg_lambda) + i·ω` | already the SSM eigenvalue |
| Distance kernel `W(r)` (DoG) | recurrent `W_rec·z`; translation-invariant ⇒ a **spatial convolution** | new: `_build_coupling` in `wave.jl` |
| Conduction delay `τ_ij` | **complex phase factor `e^{-iωτ}`** on the shared carrier (§2.1) | folded into the complex kernel |
| Spike-triggered adaptation `a_i` | slow real aux-state, `|z|`-driven (soft) | optional `use_adaptation` |
| Short-term depression `d_j` | slow multiplicative gain (soft) | future work |
| Threshold + reset | phase-only saturation (`normalize_to_unit_circle`) for training; hard threshold at demo time | `saturating` flag |
| Balanced `∝ g/√K`, `∫W≈0` | zero-mean DoG + gain `g` | kernel construction |
| Branching ratio σ≈1 | **spectral radius of the per-mode step multiplier ≈ 1** (§2.2) | `dispersion()` |

---

## 2. Three insights that make it fit the four constraints

### 2.1 Conduction delays become complex weights — no DDE solver

The architecture is phase-locked to one carrier `ω` (the per-channel-ω rule in
`CLAUDE.md`). On that carrier a delay `τ` is *exactly* a phase rotation
`e^{-iωτ}`. So the plan's `s_j(t − τ_ij)` collapses to `e^{-iωτ_ij}` multiplying a
complex coupling weight. Delays stay inside the **linear complex SSM** — no
`DelayDiffEq`, no history buffer, fully differentiable, and identical in the
discrete and ODE modes.

This is only available *because* of the shared-ω discipline, so the wave sheet is
consistent with — not an exception to — the per-channel-ω rule. Spatial diversity
lives in `λ` and the coupling kernel, exactly as `CLAUDE.md` specifies.

> **Caveat.** `e^{-iωτ}` is the exact delay only *on* the carrier `ω`. Broadband
> transients see an approximate delay. Fine for phase-locked operation; note it.

### 2.2 Translation-invariant coupling diagonalizes under spatial FFT

If the DoG coupling is homogeneous over the sheet (periodic boundary), a 2-D
spatial FFT diagonalizes `W_rec`. The linear wave SSM then *decouples into one
independent single-channel SSM per spatial wavevector* `q`, with per-step
multiplier

```
M(q) = A + g · Ŵ(q),     A = exp(k·T),   Ŵ = fft2(W_rec)
```

and continuous effective eigenvalue `k_eff(q) = log M(q) / T`. This one relation
gives us everything at once:

- **(b) GPU** — the per-step coupling is a circular convolution done as
  `ifft2(Ŵ .* fft2(z))`: `O(HW log HW)`, cuFFT-native, Zygote-differentiable via
  the AbstractFFTs rrules the codebase already relies on (`causal_conv_fft`).
- **Dispersion for free** — `arg M(q)` gives per-step phase advance ⇒ wave speed
  vs. wavelength; validates [Kerr–Ashwin–Wedgwood 2025] "resonance sets speed."
- **Criticality without a blind sweep** — the plan's "branching ratio ≈ 1" is
  `max_q |M(q)| ≈ 1` (spectral radius). The "keep one scalar gain `g`" knob
  becomes *solvable*: set `g` so the most unstable spatial mode is marginally
  stable. Training the DoG kernel = shaping which spatial-frequency waves sit at
  marginal stability. `dispersion(layer, ps, st)` returns `M`, the spectral
  radius, and `k_eff` directly.

### 2.3 A phase-only wave is self-limiting by construction

A *linear* SSM cannot be self-limiting (it decays or explodes). The plan spends
three mechanisms (adaptation, depression, inhibition) on bounding amplitude. But
phasor networks carry information in *phase*, not magnitude — and we already have
`normalize_to_unit_circle` / `soft_normalize_to_unit_circle` as activations.
Projecting the state back onto the unit circle each step gives amplitude
self-limitation *for free*, differentiably, with no discrete reset. This is what
`demos/wave_dispersion.jl` demonstrates: at a supercritical gain the *linear*
sheet's amplitude grows geometrically (`|z|ₘₐₓ ~ spectral_radiusᵗ`), while the
saturating sheet holds `|z| ≡ 1` yet keeps all its phase content.

> **Adaptation is a separate, harder story.** The plan's spike-triggered
> adaptation and the A-CANN "destabilize a standing bump into a traveling wave"
> mechanism are *amplitude* phenomena: a bump is a localized packet of high
> activity, which phase-only dynamics (`|z| ≡ 1` everywhere) cannot represent.
> `PhasorWaveSheet` exposes an optional `use_adaptation` slow-feedback state
> (differentiable, gradient-tested), but a genuinely *traveling* bump also needs
> symmetry-breaking (anisotropic coupling / directional delay) — which the plan
> itself flags as the research frontier. That belongs to the amplitude-preserving
> Tier-2 work, not this first milestone.

---

## 3. Two tiers, linked by `K[n] = Aⁿ·B`

Mirroring the existing SSM/ODE duality:

**Tier 1 — `PhasorWaveSheet` (discrete; trainable; GPU).** Forks the
`AttractorPhasorSSM` per-step `Buffer` loop (already proven Zygote-clean), but
swaps the attractor pull for spatial-conv coupling + delay phase factors:

```
a[t]  = ρ_a · a[t-1] + δ_a · |z[t-1]|                 # optional slow adaptation
z[t]  = A · z[t-1] + g · ifft2(Ŵ ⊙ fft2(z[t-1]))      # per-channel decay + DoG coupling+delays
        + drive[t] − i · a[t]                          # external drive; adaptation into voltage
z[t]  = normalize_to_unit_circle(z[t])                 # (if saturating) phase-only self-limit
```

**Tier 2 — continuous ODE (*shipped* as the `CurrentCall`/`SpikingCall`
dispatch on `PhasorWaveSheet`).** Faithful to the codebase's "one defining
equation, multiple modes via dispatch" architecture (as `PhasorSSM` was unified
into `PhasorDense`), the ODE mode is *the same struct*, not a separate `WaveODE`
type. It extends the `oscillator_bank` `dzdt` closure with the recurrent
FFT-coupling term,

```
dz/dt = k·z + g·ifft2(Ŵ ⊙ fft2(z)) + I(t)
```

solved by the existing `Tsit5 + BacksolveAdjoint/ZygoteVJP`. The coupling kernel
`Ŵ` is rebuilt from `p` inside `dzdt` so gradients flow to the coupling
parameters through the ODE adjoint; outputs are sampled at each period via
`saveat` (interpolation is disabled under the adjoint), making the path
differentiable end-to-end (verified: finite, nonzero grads for `log_g`,
`log_speed`, `B_inh`, … with `ComponentArray` params, as all ODE-mode layers
require). `wave_simulate(...; mode=:ode)` gives the pure-forward autonomous
integrator and `dispersion(...; mode=:continuous)` the exact operator eigenvalue
`k_eff(q)=k+g·Ŵ(q)`.

Because both tiers integrate the same equation, they agree: seeded from one
pulse at a subcritical gain, the discrete recurrence and the ODE keep a
**field similarity of 0.997–0.9997** across the run (`demos/wave_dispersion.jl`
§4) — the `K[n]=Aⁿ·B` SSM/ODE duality in action. They differ only by the
operator-splitting term `O(g·T)`, which is why the two `dispersion` modes
(`:discrete` `A+g·Ŵ` vs `:continuous` `exp((k+g·Ŵ)T)`) diverge as coupling
strengthens.

### What is *not* the linear SSM (honesty)

Genuine power-law avalanche statistics / branching-ratio criticality are
properties of the **nonlinear threshold+reset** system. Tier 1's trainable path
is near-critical *linear*; the avalanche diagnostics belong to the nonlinear
demo-mode (hard threshold at inference). Keep "trainable near-marginal" and
"biophysically critical" claims in separate tiers.

---

## 4. `PhasorWaveSheet` — API

```julia
layer = PhasorWaveSheet(H, W;
    saturating       = true,     # phase-only self-limiting each step (false ⇒ linear, exact dispersion)
    use_adaptation   = false,    # slow negative feedback (bump → traveling wave)
    init_log_g       = log(1.0), # recurrent gain — the criticality knob
    init_A_exc, init_log_sigma_exc, init_B_inh, init_log_sigma_inh,  # DoG shape
    init_log_speed   = log(8.0), # conduction speed c (pixels/period); τ(r)=r/c
    spk_args         = SpikingArgs())   # supplies shared ω, T
```

- **Lux forward** `(x::Phase{3}, ps, st)` with `x :: (H*W, L, B)` — Tier-1
  discrete mode; each timestep injects a spatial drive; returns the sheet state
  `(H*W, L, B)` Phase. This is the primary trainable interface (drop into a Lux
  chain; feed its output to `SSMReadout`/`Codebook`).
- **`CurrentCall` / `SpikingCall` dispatch** — Tier-2 continuous ODE mode of the
  same struct; returns `(H*W, L, B)` Phase sampled at each period. Differentiable
  through `BacksolveAdjoint` (use `ComponentArray` params).
- **`wave_simulate(layer, ps, st; z0, L, drive=nothing, mode=:discrete)`** — the
  pure-forward interface for demos/analysis: set initial complex state
  `z0 :: (H,W[,B])`, evolve `L` steps, return the full complex trajectory
  `(H,W,L[,B])`. `mode=:ode` integrates the continuous ODE (autonomous).
- **`dispersion(layer, ps, st; mode=:discrete)`** — `(; M, spectral_radius,
  k_eff, W_hat, growth_rate)` per spatial mode (§2.2). `mode=:continuous` returns
  the exact ODE operator eigenvalue.

Coupling defaults to a parametric difference-of-Gaussians (`coupling = :dog`, a
handful of interpretable trainable scalars), so ablations are one-liners:
`B_inh = 0` removes inhibition; `δ_a = 0` removes adaptation; scaling `g` detunes
criticality. `coupling = :stencil` swaps in a free learnable kernel (§5-bis).

---

## 5. Demos (in dependency order)

1. **Dispersion, wave-speed & self-limiting (no training)** — *shipped*,
   `demos/wave_dispersion.jl`. Pure forward simulation, lowest risk, visual
   payoff. Demonstrates: (i) closed-form dispersion + a computed critical gain
   (`spectral_radius = 1`); (ii) a seeded pulse propagating at a measurable
   speed; and the plan's decisive ablations — (iii) remove inhibition → the
   wavelength-selecting ring collapses to a uniform front (fastest mode → DC);
   (iv) detune `g` → extinguish / sustain / saturate; (v) phase-only saturation
   self-limits amplitude where the linear medium diverges; and (vi) **Tier-1 ↔
   Tier-2 equivalence** — the discrete recurrence and the continuous ODE track
   each other at field similarity 0.997–0.9997. *(The plan's "remove adaptation →
   standing bump" ablation is deferred — see §2.3.)*
2. **Trainable wave classifier on FashionMNIST** — *shipped*,
   `demos/wave_fashionmnist.jl`. Inject the image as a spatial drive on a
   28×28 sheet, propagate `L` steps, collapse to the last-step phase field, and
   classify with a `PhasorDense` head + `Codebook` similarity readout. Trains
   end-to-end through the discrete phase-SSM path (Zygote AD through the
   Buffer/FFT recurrence). Config is env-overridable (`WAVE_N_TRAIN`,
   `WAVE_EPOCHS`, `WAVE_STENCIL_R`, …) for scaling. Four-way comparison on a
   scaled CPU run (20k/5k subset, 8 epochs, L=5):

   | model | test acc | params |
   |---|---|---|
   | **wave — DoG coupling** | **0.834** | 50,375 |
   | wave — learnable stencil (R=3) | 0.824 | 50,468 |
   | dense baseline (same head, no wave) | 0.816 | 50,368 |
   | `PhasorConv` stack (repo-canonical) | 0.654 | 3,468 |

   Both wave variants top the matched dense baseline: **+1.7 points for +7
   trainable parameters** (DoG coupling), and the wave models train more stably
   (the dense baseline wobbles across epochs while the wave models hold ~0.83).
   The `PhasorConv` stack is a much leaner, different inductive bias (its
   16×16→8×8 kernels compress to 36 features). Demonstrates (c) trainability and
   (d) wave-based computation on a real task. *(Not leaderboard-tuned; the
   headline is the controlled wave-vs-dense delta at fixed head.)* On the
   learnable stencil (bookmark 2), see §5-bis; on **where the learning actually
   happens** (spoiler: mostly the head), see §5-ter.
3. **Associative memory via settling waves** *(bookmarked)* — bridges to
   `AttractorPhasorSSM` and the EP/hEP equilibrium machinery (`ep.jl`, `hep.jl`):
   the sheet's fixed point *is* an energy minimum.
4. **Sparse critical regime** — tune `g` to marginal stability, add hard
   threshold, measure participation fraction / avalanche sizes (nonlinear
   demo-mode; plan §05 diagnostics).

### 5-bis. Learnable coupling stencil — *shipped* (was bookmark 2)

`PhasorWaveSheet(...; coupling = :stencil, stencil_radius = R)` replaces the
9-scalar parametric DoG with a free, translation-invariant complex kernel of
radius `R` (`(2R+1)²` complex entries), built by a differentiable linear
scatter onto the sheet (`place · stencil_vec`, no mutation) then `fft`ed —
keeping the GPU/dispersion machinery intact. Seeded from the DoG so it starts
in the same regime, then trains freely; gradients reach the full stencil.

**Finding (honest):** on isotropic FashionMNIST the learnable stencil is *on
par with, not better than*, the structured DoG:

| coupling | test acc | coupling params | run |
|---|---|---|---|
| DoG | 0.834 | 7 | 20k/8ep |
| stencil R=3 | 0.824 | 100 | 20k/8ep |
| DoG | 0.812 | 7 | 8k/5ep |
| stencil R=6 | 0.810 | 338 | 8k/5ep |

The R=3 stencil slightly *trails* the DoG because its reach (r ≤ 4.2) truncates
the DoG's longer inhibitory tail (σ_I = 3 → reach ~6–9 px); giving it adequate
reach (R=6) closes the gap to a tie. So the extra capacity neither helps nor
hurts here — the DoG's structured bias already fits this isotropic task, and the
50k-param head dominates either way. The stencil's value is **flexibility for
couplings the DoG can't express** — anisotropic / patchy / feature-selective
connectivity (Davis 2024) — which this task doesn't exercise. That's the regime
to test it in next.

### 5-ter. Where does the learning happen? — mostly the head

The classifier's head (`PhasorDense` 784→64 → `Codebook`) has ~50k params; the
wave coupling has 7 (DoG). So how much does the wave sheet actually contribute?
`demos/wave_attribution.jl` decomposes it with three identical-head/data/seed
conditions (8k/2k, 6 epochs):

| condition | test acc | Δ |
|---|---|---|
| head only (dense, no wave) | 0.802 | — |
| head + wave coupling **frozen** at init | 0.793 | −0.009 vs head |
| head + wave coupling **trained** | 0.809 | +0.016 vs frozen |

Two honest conclusions:

1. **The head does ~99% of the work.** Head-alone is 0.802 of the full 0.809
   (and 0.816 of 0.834 in the 20k/8ep run) — a consistent ~99% across scales.
   The wave sheet is a small additive contribution, not the primary learner.
   (Unsurprising: 7 coupling params vs 50k head params.)
2. **The wave sheet's benefit comes from *training* the coupling, not from a
   fixed structural prior.** The frozen (init) coupling mildly *hurts* (−0.009);
   only when the ~7 coupling params are trained does it help (+0.016 over frozen,
   +0.007 over head-alone). So those few parameters do genuine work — they're
   just a small effect on top of a head that already does almost everything.

*Caveat:* these are ~1-point deltas at a single seed on a subset — within
run-to-run noise. The "head dominates" headline is solid; the finer
"frozen hurts / trained helps" split is directional and wants multiple seeds to
confirm. The takeaway for honest framing: **the wave sheet is a near-free,
learnable spatial-mixing adjunct to a conventional trainable head — not a
stand-alone learner.** Making the sheet carry more would mean shrinking the head
and giving the coupling real reach/capacity (cf. §5-bis), or a trainable-code
readout so the sheet's phase field is what gets classified.

### Still bookmarked

- **(3) Associative memory via settling waves** — demo 3 above; the settling
  fixed point as an energy minimum, tied into `AttractorPhasorSSM` + EP/hEP.

---

## 6. Answering the four questions

- **(a) self-regulatory sparse dynamics** — self-limiting from phase-only
  normalization (free) or smooth adaptation (trainable); sparsity from a `|z|`
  threshold (soft at train, hard at demo → participation-fraction metric);
  self-sustaining from tuning `g` to spectral radius ≈ 1.
- **(b) GPU-compatible** — FFT-diagonalized coupling on existing primitives
  (`causal_conv_fft`, cuFFT, KernelAbstractions). Avoid dense `HW×HW` coupling;
  conv+FFT is the scalable route.
- **(c) phase-SSM / trainable** — reuses the Dirac/`causal_conv` discrete path
  and the `BacksolveAdjoint` ODE path; the recurrent loop reuses the
  `AttractorPhasorSSM` `Buffer` template. Shared-ω makes delays-as-phase exact.
- **(d) wave-based vs. discrete layers** — one recurrent sheet unrolled `L` steps
  replaces a stack of layers (SSM depth-equivalence); VSA `v_bind`/`v_bundle` are
  interference ops a wave medium performs natively. Paradigm: *inject as spatial
  current → let the wave compute by propagating/interfering → read out by
  similarity*.

---

## 7. Risks / open questions

- **Criticality ≠ linear SSM** — train near-marginal (Tier 1); reserve
  avalanche/power-law claims for the nonlinear demo-mode.
- **Delay-as-phase exact only on the carrier ω** — approximate for broadband.
- **Traveling *bumps* (amplitude packets) need symmetry-breaking** — an
  isotropic kernel + symmetric seed spreads rather than translates; a genuine
  moving bump needs anisotropic coupling / directional delay (plan §06). The
  shipped demo therefore demonstrates *traveling waves* (phase fronts) and
  *self-limiting*, not moving bumps.
- **Patchy connectivity** (Davis 2024 feature-selective motifs) breaks FFT
  diagonalization — keeps differentiability but loses the closed-form
  dispersion/criticality readout; budget sparse-conv cost.
- **oneAPI/Aurora** — per project memory, validate any new GPU kernels on Aurora,
  not locally.

---

## References

- Izhikevich (2001), *Resonate-and-fire neurons*, Neural Networks — the substrate unit.
- [Kerr, Ashwin & Wedgwood (2025), arXiv:2511.05232](https://arxiv.org/abs/2511.05232) — resonance sets wave speed and locks ISIs; partial-participation waves need added stochasticity/heterogeneity.
- [Adaptive CANN (2024), arXiv:2410.06517](https://arxiv.org/abs/2410.06517) — adaptation destabilizes a bump so activity travels.
- Muller et al. (2018), *Cortical travelling waves*, Nat. Rev. Neurosci. — delays generate waves in recurrent networks.
- Davis et al. (2024), Cell Reports — patchy horizontal connectivity sculpts feature-selective wave motifs.
- van Vreeswijk & Sompolinsky (1996); Brunel (2000) — balanced sparse irregular state, `∝1/√K` scaling.
- Beggs & Plenz (2003); Poil et al. (2012) — branching ≈ 1, power-law avalanches from E/I balance.
