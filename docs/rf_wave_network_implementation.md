# Implementing self-regulating sparse waves in PhasorNetworks.jl

**Companion to `docs/rf_wave_network_plan.html`.** The plan (`rf_wave_network_plan.html`)
is a *biophysical* specification: complex resonate-and-fire (R&F) units on a 2-D
sheet, coupled by a delayed difference-of-Gaussians kernel, with spike-triggered
adaptation, short-term depression, and a hard threshold+reset, tuned to
criticality. This document translates that spec into the algebra this codebase
already speaks — the **phase-SSM view** — so that the resulting layer is (a)
demonstrably self-regulating, (b) GPU-native, (c) trainable end-to-end, and (d)
a genuine *wave-based* transformation rather than a stack of discrete layers.

Status: draft alongside the prototype (`src/wave.jl`, `demos/wave_dispersion.jl`).
Related: [`wave_dispersion_derivation.md`](wave_dispersion_derivation.md) (phonon/Kuramoto
dispersion theory), [`wavesheet_experts_design.md`](wavesheet_experts_design.md)
(sparse read→gate→re-bind "expert" modules that transform waves in transit), the
step-by-step build-up notebook `demos/wave_sheet_explained.ipynb`, and its
verification `demos/wave_dispersion_derivation.ipynb`.

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
| Spike-triggered adaptation `a_i` | slow real aux-state, `\|z\|`-driven (soft) | optional `use_adaptation` |
| Short-term depression `d_j` | slow multiplicative gain (soft) | future work |
| Threshold + reset | **emission threshold `θ`** on the transmitted spike, `z/√(\|z\|²+θ²)`; derived default, optionally homeostatic (§4-ter) | `init_log_theta`, `homeostasis` |
| Intrinsic excitability homeostasis | per-site slow `θ` adaptation; the refractoriness that makes the packet travel | `homeostasis = :local` |
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

> **Full derivation:** the phonon-style dispersion relation is worked out from
> first principles (Bloch ansatz, group velocity, the delay→dispersion /
> DoG→gain-band split, 2-D, *and* the nonlinear `:spike` Kuramoto phase-mode
> band) in **[`docs/wave_dispersion_derivation.md`](wave_dispersion_derivation.md)**,
> with numerical verification in **`demos/wave_dispersion_derivation.ipynb`**.

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
>
> *Update (matched-speed work, §4-bis):* the symmetry-breaking requirement applies
> to **directional** transport. Radially expanding waves with a moving amplitude
> envelope come out of the isotropic kernel alone once the conduction delay is
> matched — no adaptation and no anisotropy. Adaptation remains the route to
> refractoriness (no back-propagation), which the delay mechanism does not supply,
> and the Muller et al. review does *not* rank it as the primary wave generator.
>
> *Update (threshold work, §4-ter):* refractoriness now has a second, better
> implementation — `homeostasis = :local`, a slow per-site **firing threshold**.
> It beats `use_adaptation` on two counts: it modulates *emission* rather than
> adding `−ia` to the state, so it cannot spuriously rotate phase, and it is the
> same mechanism as the activity regulation the sheet needs anyway. Measured, it
> is what turns the standing localized blob that `:global` produces into a
> traveling one (net drift ≈3×). `use_adaptation` is untouched but is now the
> second-choice route.

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
    init_log_speed   = nothing,  # conduction speed c (sites/period); τ(r)=r/c.
                                 # DERIVED by default: c = 2σ_I/T (§4-bis)
    # --- emission threshold (transmit = :spike only), §4-ter ---
    init_log_theta   = nothing,  # firing threshold θ; emit = z/√(|z|²+θ²).
                                 # DERIVED by default: θ = frac · g·max|Ŵ|
    init_theta_frac  = 1.4,      # multiplier on that reference
    homeostasis      = :none,    # :none | :global | :local  — regulate θ to a target rate
    init_log_eta_g   = log(0.15), init_log_eta_l = log(0.02),
    init_logit_target = log(0.02/0.98),   # target firing rate, logit-parameterized
    init_log_theta_beta = log(0.05),
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
  k_eff, W_hat, growth_rate)` per spatial mode (§2.2; full derivation in
  [`wave_dispersion_derivation.md`](wave_dispersion_derivation.md)).
  `mode=:continuous` returns the exact ODE operator eigenvalue. On a `:spike`
  sheet it substitutes the subthreshold effective gain `g/θ` (§4-ter).
- **`radial_band` / `wave_transport` / `matched_conduction_speed`** — the
  standing-vs-traveling surface (§4-bis).
- **`emission_threshold` / `wave_homeostat_trace`** — the firing-threshold
  surface (§4-ter). The trace is the only way to see `θ`, which is rollout-local.

Coupling defaults to a parametric difference-of-Gaussians (`coupling = :dog`, a
handful of interpretable trainable scalars), so ablations are one-liners:
`B_inh = 0` removes inhibition; `δ_a = 0` removes adaptation; scaling `g` detunes
criticality. `coupling = :stencil` swaps in a free learnable kernel (§5-bis).

---

## 4-bis. Conduction speed: the standing-vs-traveling knob

A critical sheet is not automatically a *propagating* one. Because the kernel is
reflection-symmetric, `±q` are amplified equally, so an impulse always launches
counter-propagating pairs; whether they separate into a traveling ring or overlap
into a standing pattern is decided by the group velocity at the selected mode.
Weak conduction delay forces `v_g(q*) → 0` identically — see
[`wave_dispersion_derivation.md`](wave_dispersion_derivation.md) §3-bis for the
derivation. The escape is to match the delay phase across the surround to half a
cycle:

```
φ = ω σ_I / c ≈ π    ⟺    c = 2 σ_I / T      # matched_conduction_speed(σ_I, T)
```

`init_log_speed` therefore **derives** this by default rather than taking a fixed
value — a constant default silently drifts back into the standing regime whenever
`σ_I` or `t_period` changes. At the default shape this gives `c = 6`, versus the
`40` (and this document's earlier `8`) previously in use:

| | `c` | `φ` | `g_crit` | transport ratio | verdict |
|---|---|---|---|---|---|
| previous | 40 | 0.15π | 0.026 | 0.028 | standing |
| matched | 6 | 1.00π | 0.069 | 0.995 | **traveling** |

Check any sheet with `wave_transport(l, ps, st).verdict` (evaluate at criticality —
well above it the DC mode wins and everything reports `:standing`). The traveling
window is narrow, `c ∈ [3.2, 6.8]`.

Two caveats that matter for design, both detailed in §3-ter of the derivation:
`c` is a **phase parameter on an instantaneous coupling**, not a transport delay,
so `v/c` is not comparable to the cortical wave-speed/conduction-speed ratio; and
this analysis reaches `:spike` only in its *subthreshold* regime (§4-ter). There
is still no relay front in the axonal sense — the coupling is one instantaneous
FFT convolution per step whatever `c` is.

---

## 4-ter. Emission threshold: the knob that makes `:spike` analysable

`:spike` transmits `z/√(|z|²+θ²)`. Until recently `θ` was the hardcoded `ε` guard
of `normalize_to_unit_circle` (`√1e-8 = 1e-4`) — five orders of magnitude below
the scale the coupling operates at, so every site above `1e-4` emitted a *full*
spike and an impulse became an ignition cascade rather than a wave. `θ` is now an
exposed, trainable parameter. Full derivation and measurements in §3-quater of
[`wave_dispersion_derivation.md`](wave_dispersion_derivation.md); the design
consequences:

**It interpolates between the two transmission modes.** For `|z| ≫ θ` the emit is
`z/|z|` (hard spike); for `|z| ≪ θ` it is `z/θ` — *linear*, i.e. the `:potential`
medium at gain `g/θ`. So a spike sheet now has an exact subthreshold dispersion,
and `dispersion` / `radial_band` / `wave_transport` become valid for it. This is
what `dispersion` reports.

**The default is derived.** `θ_ref = g·max_q|Ŵ(q)|` — the drive a site receives
when its whole neighbourhood emits unit spikes in phase, i.e. *fire only on
near-maximal local coherence*. Measured `θ* ∝ g` **exactly**; the layer default is
`1.4 × θ_ref`, because `θ_ref` itself sits just below the flood boundary.

**A constant `θ` is not sufficient.** The usable band is only ≈1.3× wide and moves
with `g`, so a fixed value falls out of regime as soon as `g` trains — the same
failure mode as the old fixed `c = 40`.

| `homeostasis` | what it does | measured |
|---|---|---|
| `:none` | fixed `θ` | correct only at the `g` it was derived for |
| `:global` | scalar `θ_g` → target firing rate | tracks `g` over 100× (`θ_g/g` = 11.7 / 11.9 / 11.3); gives a **standing** localized blob |
| `:local` | `:global` + slow zero-sum per-site `θ_l` | refractoriness; net drift ≈3× the standing case at `η_l ∈ [0.01, 0.02]` |

Local-only is deliberately not offered — on its own the per-site term can only hit
its target by silencing whoever just fired, giving rate-regulated flicker
(step-to-step overlap of the active set measured at exactly 0).

Three implementation points that are load-bearing rather than incidental:
- The fire indicator is a **straight-through estimator** (hard `|z|>θ` forward,
  sigmoid backward). Any smooth pointwise indicator lets the homeostat hit its
  target with a *uniform subthreshold* sheet — measured: target met exactly, zero
  sites firing.
- `mode = :deq` **throws** with homeostasis on: a sweep-dependent `θ` breaks the
  fixed point's exact-reproduction guarantee.
- A real threshold means a **charging transient**. Unit drive alone saturates at
  `1/(1−|A|) ≈ 7.2`, below the default `θ ≈ 10.6`, so firing needs ~50 steps of
  recurrent build-up. Short rollouts see no spikes and `η_l` gets no gradient.

> **Validated at the default `λ = 0.15` only**, across `g ∈ [0.3, 3]`,
> `N ∈ [48, 96]`, couplings `:dog`/`:aniso`/`:stencil`. At `λ ≥ 0.5` the homeostat
> still regulates the rate to target but with a synchronously bursting sheet.
> Hitting the rate is necessary, not sufficient — check `std|z|` via
> `wave_homeostat_trace`.

---

## 5. Demos (in dependency order)

1. **Dispersion, wave-speed & self-limiting (no training)** — *shipped*,
   `demos/wave_dispersion.jl`. Pure forward simulation, lowest risk, visual
   payoff. Demonstrates: (i) closed-form dispersion + a computed critical gain
   (`spectral_radius = 1`); (ii) the gain band `Γ(|q|)` and the propagation band
   `v_r(|q|)` together, with the transport verdict (§4-bis) and a wavepacket
   measurement overlaid as an independent check; and the plan's decisive
   ablations — (iii) remove inhibition; (iv) detune `g` → extinguish / sustain /
   saturate; (v) phase-only saturation self-limits amplitude where the linear
   medium diverges; and (vi) Tier-1 ↔ Tier-2 comparison. *(The plan's "remove
   adaptation → standing bump" ablation is deferred — see §2.3.)*

   Two results here changed when the sheet moved to the matched conduction speed,
   and the earlier wording overstated both:

   - **The inhibition ablation is regime-dependent.** At weak delay the Mexican hat
     is the only wavelength selector, so `B_inh = 0` collapses the peak to DC (a
     uniform front) — that is what this document previously claimed flatly. At the
     matched speed the delay phase varies by ~π across the kernel and selects a
     wavelength on its own, so the ring *survives* the ablation (peak `|q|` moves
     0.524 → 1.242 rather than → 0). The demo now computes both regimes.
   - **Tier-1 ↔ Tier-2 agreement is not 0.997–0.9997 at this operating point.** The
     `O(g·T)` splitting gap scales with `g_crit`, which the matched regime raises
     2.7×; field similarity at `t=60` falls to 0.56. It tracks `g_crit`, not `c`.
     See §4 of the derivation doc.
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
- **Delay-as-phase exact only on the carrier ω** — approximate for broadband. And
  note it is a *phase*, not a lag: the coupling is instantaneous, so `c` shapes the
  band but does not gate how fast activity reaches a site (§4-bis, and §3-ter of
  the derivation doc). A true relay mechanism would need real delayed coupling.
- **Radial transport does *not* need symmetry-breaking — directional transport
  does.** This item previously read "an isotropic kernel + symmetric seed spreads
  rather than translates", and concluded the shipped demo showed traveling phase
  fronts. Both need correction. An isotropic kernel *can* produce a genuine
  radially-expanding wave with a moving amplitude envelope, provided the
  conduction delay is matched (§4-bis); measured on 192², the radial-profile peak
  marches at 0.71 sites/period. What the demo actually showed at the old `c = 40`
  was a **standing** ring pattern whose crests are pinned for hundreds of steps —
  only the *phase* advanced (at `v_p ≈ 0.08`), which is why a phase-coloured
  animation read as a traveling wave. Symmetry-breaking (`:aniso`, `:shift`) is
  what a *net-drift* wave needs, not an expanding one.
  - Caveat on `:aniso` at the matched speed: the DoG becomes a backward wave
    (`v_g` opposite to `q`), so `β_h > 0` biases growth toward `+q_h` while the
    packet travels toward `−h` — the drift sign inverts relative to the weak-delay
    convention (measured: `+0.67` at `c=40`, `−1.01` at `c=6`).
- **The threshold work is calibrated at one damping.** §4-ter's derived default
  and homeostat defaults were measured at `λ = 0.15`. At `λ ≥ 0.5` the homeostat
  regulates the firing rate to target but the sheet bursts synchronously rather
  than forming structure, and the `1.4` multiplier lands on the extinct side
  (`θ*/θ_ref` falls to 0.71 by `λ = 1.5`). Anyone moving `λ` must re-measure. This
  interacts with the open question of whether `λ` should move at all — the
  matched-speed work found `v/c` improves with damping up to `λ ≈ 1.2`, which is
  squarely in the un-validated region.
- **Homeostat stability has sharp edges.** Measured failures, all recoverable but
  none obvious: target `0.10` at `η_g = 0.15` collapses the sheet to uniform;
  `η_l ≥ 0.05` decoheres the packet; and adding a restoring prior toward `θ_ref`
  (which looks like an obvious robustness win, and is one at `λ = 0.15`)
  permanently kills the sheet at other `λ`.
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
