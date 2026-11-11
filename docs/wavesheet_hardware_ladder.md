# The wavesheet on superconducting hardware — a stepped ladder

Target platform: SNSPD sensing elements in a sub-4 K environment, with the
resonate-and-fire neuron and the coupling lattice built from other
superconducting elements. Target temporal resolution ≈ 10 ps, in the specific
sense that a particle transiting the sheet excites successive sites **10 ps
apart**, and we want the velocity and trajectory back.

Companion to `docs/wave_dispersion_derivation.md` (why the sheet is a poor wave
medium) and `src/velocity_bank.jl` (the matched-filter route that works).
Rung 0 is implemented in `demos/wave_velocity_bank_rung0.jl`; every ⓒ number
below is printed by that script.

Conventions: `T` is the carrier period, `ω = 2π/T`, `Δt` the site-to-site
arrival step, `κ = 2π·Δt/T` the deposited phase ramp in rad/site, `v = 2π/κ` the
speed in sites/period. Lattice spacing 1.

**Provenance convention.** Numbers are marked ⓒ (computed here) or ⓛ (device
measurements from the literature, with an arXiv ID). Do not mix them when
budgeting.

**All ⓒ figures below are 2D measurements at the worst-case heading** — half an
angle-bin off the nearest channel. An earlier 1D study informed the design and
several of its conclusions did not survive the move to 2D; where that happened
it is called out explicitly, because the 1D numbers are the optimistic ones.

---

## 0. Summary

10 ps per site is **not** the difficult part. At a 10 GHz carrier it maps to
`κ = 0.628 rad/site`, i.e. `v = 10 sites/period`, recovered to **0.17%** in the
clean case ⓒ. The difficulties are elsewhere.

Literature search moved four numbers:

| finding | consequence |
|---|---|
| nTron timing jitter < 60 ps ⓛ | **nTron cannot be in the timing path.** Direct reactive injection only. |
| Sheet inductance 3–30 pH/sq, not 50–1000 ⓛ | Delay lines 3–10× longer than first estimated; 33–146 ps/mm. |
| SNSPD jitter splits by material: 15–18 ps (NbN high-I_c) to 191 ps (WSi) ⓛ | Material choice is first-order. |
| Meander needs ≥ 390 µm pitch at R=1, ~1 mm at R=3 ⓒ | New floor on lattice pitch. |

And rung 0 moved three more, two of them against us:

| finding | consequence |
|---|---|
| **Detector jitter tolerance is far tighter in 2D than 1D** ⓒ | The binding constraint. See §2.4. |
| **R = 1 is not the stencil optimum in 2D** ⓒ | R=2 clean, R=3–4 under jitter; 2–6× more line. |
| **The readout interpolator must be 2D Cartesian** ⓒ | Worth 38× over argmax; separable polar is biased. |

**Detector jitter was the live risk, and the κ-aligned stencil resolves it.**
With an isotropic stencil at R=1 the bank holds only to 3 ps, degrading to 6.1%
at 15 ps and 16.3% at 25 ps ⓒ — the preceding 1D study predicted 25 ps at
0.63%, optimistic by more than an order of magnitude, and an oracle mask over
the true track reproduces the same numbers so it is not a readout artefact.

Elongating each channel's coupling along its **own** `κ_c` recovers it: at
R=4, aspect 0.5 the bank holds 0.78% at 15 ps and 1.81% at 25 ps ⓒ — a 9×
improvement at 25 ps — while costing about half of the isotropic stencil it beats
(25 taps vs 48, 7.3 mm vs 16.6 mm of line per site per channel). The
diagnosis was right: a 1D track in a 2D lattice couples mostly to off-track
neighbours, and pointing the stencil along the track fixes it. The detector
specification moves from "nothing available works" to "MoSi at 26 ps works,
high-I_c NbN has margin." The price is a lattice pitch of **~2.66 mm** — that
figure carries the full 192-channel layout cost, see §3.4. Curvature, which was
expected to be the other cost, turned out not to be (L1).

---

## 1. The mapping — time becomes a wavevector

Because `ωT = 2π` exactly, the carrier is stroboscopically invisible and a
particle crossing site `x` at continuous time `t(x)` leaves that neuron at
sub-cycle phase `θ = 2π·frac(t(x)/T)`. For straight motion the deposited field
is a plane wave whose wavevector *is* the velocity:

$$\kappa \;=\; \frac{2\pi\,\Delta t}{T}\ \text{rad/site},\qquad v \;=\; \frac{2\pi}{\kappa} \;=\; \frac{T}{\Delta t}\ \text{sites/period}.$$

### 1.1 Choosing the carrier

| `T` (ps) | `f` (GHz) | κ at Δt = 10 ps | v (sites/period) | unambiguous Δt |
|---|---|---|---|---|
| 40 | 25.0 | 1.571 = π/2 | 4.0 | 0–20 ps |
| **100** | **10.0** | **0.628 = π/5** | **10.0** | **0–50 ps** |
| 200 | 5.0 | 0.314 | 20.0 | 0–100 ps |

### 1.2 The match condition is frequency-independent

The coupling imparts phase `ωτ_c` per site; the signal carries `ωΔt` per site.
Matching gives

$$\boxed{\ \tau_c \;=\; \Delta t\ }$$

with `ω` cancelling. **A channel is a delay line whose delay equals the transit
time** — Jeffress' coincidence principle — so one delay layout serves any
carrier frequency. The carrier sets **resolution** (`δ(Δt) = δκ·T/2π`, higher
`f` finer) and **ambiguity** (`Δt < T/2`, higher `f` narrower), not matching.

A **two-carrier vernier** resolves ambiguity beyond `T/2` by CRT: two phase
estimates that must agree mod 2π, reusing one delay layout.

### 1.3 Measured accuracy

96×96 sheet, 192 channels (24 speeds × 8 headings), `T = 100 ps`, `R = 1`,
`margin = 0.9`, worst-case mid-bin heading ⓒ:

| condition | Δt = 10 ps recovered | error |
|---|---|---|
| clean | 9.983 ps | **0.17%** |
| 3 ps jitter | 10.018 ps | 0.30% |
| 15 ps jitter | 10.603 ps | 6.14% |
| 25 ps jitter | 11.608 ps | 16.3% |
| 50% dropout | 9.999 ps | 0.05% |
| 95% dropout | 9.964 ps | 1.17% |
| 10% delay scatter | 9.871 ps | 1.30% |
| 20% delay scatter | 9.723 ps | 2.86% |

Heading with this isotropic stencil is recovered to ≤ 0.002 rad across a full
0.785 rad angle bin when clean, but degrades to ~0.30 rad at 25 ps of jitter. The
recommended κ-aligned stencil inverts that trade — 0.084 rad worst-case clean,
~0.10 rad at 25 ps — and heading, not speed, is the quantity that ends up
limiting the design. See L1.

### 1.4 The readout must be a 2D Cartesian fit

Bare argmax is quantised to the channel grid. Interpolating is worth **38×
on average** (4.3–104× across the speed range) ⓒ — the single largest accuracy
win available, and free in hardware.

But *how* you interpolate matters. Refining `|κ|` and heading separably on the
polar channel grid carries a geometric bias: the channel best matching a target
at angular offset `Δθ` sits at radius `κ_true·cos Δθ`, not `κ_true`. A heading
falling mid-bin therefore foreshortens the speed estimate. Measured worst case
across one angle bin ⓒ:

| readout | worst-case speed error |
|---|---|
| argmax | 2.26% |
| separable polar log-parabola | 1.96% |
| **2D quadratic in Cartesian κ** | **0.18%** |

Separable refinement recovers almost nothing, because the bias is introduced by
the grid rather than the fit. This was a real bug found at rung 0, not a
theoretical nicety.

### 1.5 What physically sweeps at 10 ps/site

The genuine open question, and it moves the required pitch by four orders of
magnitude ⓒ:

| source | pitch for Δt = 10 ps |
|---|---|
| optical wavefront, grazing incidence | ≥ **3 mm** |
| 10 keV H⁺ (1.4 × 10⁶ m/s) | 14 µm |
| 10 keV, 100 amu ion (1.4 × 10⁵ m/s) | 1.4 µm |
| 20 keV, 10 kDa macromolecule | 0.2 µm |

The fix is a **deliberate per-site delay wedge**: insert `τ_n = n·τ₀` of
kinetic-inductance line between pixel *n* and the lattice, translating any
apparent speed into the 10 ps/site band at 129 µm per 10 ps of skew ⓒ. It makes
sensor geometry and network operating point independent design choices.

---

## 2. What the neuron must do

| spec | requirement | note |
|---|---|---|
| carrier | 10 GHz, `ωT = 2π` by construction | design choice |
| bare Q | ≈ 21 (`λT = 0.15`, ringdown 6.7 periods) | current default |
| coupled ringdown | 72 periods at margin 0.9 ⓒ | the margin buys 10× |
| **per-site timing precision** | **≤ 25 ps with a κ-aligned stencil** ⓒ | 3 ps only if isotropic — §2.4–2.5 |
| dropout tolerance | 95% of sites dead ⓒ | very benign |
| coupling stencil | **κ-aligned, R = 4, aspect 0.5** ⓒ | 25 taps/site, 7.3 mm/site/chan |
| coupling gain | `g = 0.9·g_crit`, `g_crit = (1−A)/ΣG` | closed form, no search |
| delay accuracy | 20% rms random → 2.9% ⓒ | systematic is a pure rescale |
| drive time resolution | ≥ 8 substeps per period ⓒ | see §2.3 |

### 2.1 Q ≈ 21 is easy; that is the point

Superconducting resonators routinely reach 10³–10⁶ and would need *spoiling*
down to 21. With `margin = 0.9` the coupled array holds 72 periods against the
bare resonator's 6.7, so even a 128-site track is comfortable.

### 2.2 `margin` should be 0.9, not 0.99

As `margin → 1` the matched mode integrates losslessly while mismatched modes
saturate at `1/(1−A) = 7.18`, so contrast is set by track length, not margin ⓒ:

| margin | L=10 | L=20 | L=45 | L=100 |
|---|---|---|---|---|
| 0.5 | 1.32× | 1.61× | 1.92× | 2.00× |
| 0.9 | 1.68× | 2.57× | 4.69× | 7.54× |
| 0.99 | 1.78× | 2.89× | 6.09× | 13.01× |

0.99 buys under 30% more contrast at `L = 45` but demands coupler/loss matching
at the 1% rather than the 10% level. **Default changed to 0.9.**

### 2.3 A correction: `substeps = 16` was not too coarse

An earlier estimate held that 16 substeps (6.25 ps at `T = 100 ps`) was
uncomfortably close to the quantity being measured. Measured, it is not ⓒ:

| substeps | resolves | error |
|---|---|---|
| 4 | 25.0 ps | 5.70% |
| **8** | **12.5 ps** | **0.15%** |
| 16 | 6.25 ps | 0.16% |
| 128 | 0.78 ps | 0.17% |

The floor is 8, not 64. The default was raised to 64 anyway — it is cheap and
removes the question — but the original 16 was adequate and that claim was
wrong.

### 2.4 The binding constraint: 2D jitter dilution

Measured degradation with frozen per-site detector jitter, as a function of
stencil radius ⓒ (rms Δt error, target 10 ps):

Truncation is by Euclidean distance, so `R = 1` is the **4-tap von Neumann**
neighbourhood — the diagonals sit at √2 — and `R = 1.5` the 8-tap Moore one.

| R | taps/site | line/site/chan | 0 ps | 3 ps | 15 ps | 25 ps | 35 ps |
|---|---|---|---|---|---|---|---|
| 1 | 4 | 0.52 mm | 0.17% | 0.26% | 5.59% | 16.02% | 47.25% |
| 2 | 12 | 2.28 mm | **0.02%** | 0.16% | 4.14% | 10.25% | 22.25% |
| 3 | 28 | 7.58 mm | 0.35% | 0.28% | 2.44% | 6.63% | 12.76% |
| 4 | 48 | 16.6 mm | 0.48% | 0.50% | **2.19%** | **6.31%** | **11.84%** |

Two conclusions, both corrections to earlier belief:

**R = 1 is not the optimum in 2D.** In 1D, R = 1 was best and larger stencils
measured worse. In 2D the clean optimum is R = 2 and the *operating* optimum
under jitter is R = 3–4. "Larger is worse" was a 1D-only result.

**The mechanism is geometric,** and it was confirmed by fixing it. A 1D track
embedded in a 2D lattice couples mostly to *off-track* neighbours carrying no
signal, so coherent gain is diluted in a way it never is in 1D. A wider
isotropic stencil buys some back, but only some, at 3–32× the delay line.

## 2.5 The κ-aligned anisotropic stencil

Point the stencil along the track instead of widening it. Each channel gets an
ellipse with semi-axis `R` along its own `κ_c` and `aspect·R` across, with the
Gaussian elongated to match, evaluated in that channel's rotated frame. Every
tap then lands on the track rather than beside it. `aspect = 1` recovers the
isotropic disc exactly.

Measured, 5 seeds, Δt = 10 ps, worst-case heading ⓒ:

| stencil | taps | mm/site/chan | 0 ps | 15 ps | 25 ps | 35 ps | 50 ps |
|---|---|---|---|---|---|---|---|
| iso R=1 | 4 | 0.52 | 0.20% | 5.16% | 16.13% | 45.40% | 153.8% |
| iso R=2 | 12 | 2.28 | **0.05%** | 3.74% | 9.14% | 19.26% | 99.1% |
| iso R=4 | 48 | 16.63 | 0.49% | 2.02% | 5.75% | 10.16% | 56.6% |
| **aniso 4 / 0.50** | **25** | **7.26** | 1.06% | **0.78%** | 1.81% | 3.42% | 21.6% |
| aniso 5 / 0.40 | 29 | 9.51 | 0.57% | 0.81% | **1.28%** | 3.71% | 18.4% |
| aniso 6 / 0.25 | 30 | 11.23 | 0.56% | 2.10% | 1.77% | **3.04%** | **15.8%** |

Regenerated at rung 1 with consistent settings (96×96, 5 seeds, default σ, half-bin
heading). `aniso 4/0.50` is recommended on cost: the three anisotropic rows are
within a seed-spread of each other from 15–35 ps, and it is the cheapest and has
the best clean heading. `5/0.40` and `6/0.25` pull ahead only at 50 ps, where
everything is already failing.

**`aniso 5/0.40` dominates `iso R=4` outright** — fewer taps, 57% of the line,
and 5.4× better at 25 ps. Against the cheap `iso R=1` it is 14× better at 25 ps
for 18× the line.

Across the speed range at 25 ps jitter ⓒ, the anisotropic stencil is also far
flatter — isotropic R=1 collapses on *fast* targets, where the ramp is shallow:

| stencil | 8 ps | 10 ps | 15 ps | 25 ps | 40 ps |
|---|---|---|---|---|---|
| iso R=1 | 31.3% | 15.9% | 5.9% | 2.1% | 2.4% |
| aniso 5 / 0.40 | 1.5% | 1.3% | 1.4% | 2.1% | 5.3% |
| aniso 6 / 0.25 | 1.9% | 1.0% | 1.3% | 1.0% | 3.2% |

**Curvature was expected to be the cost. It is not** — see L1, where the
elongated stencil holds 0.13–3.58% from straight down to a 10-site turn radius at
up to 25 ps of jitter ⓒ. An earlier draft reported a curvature penalty here; those
figures were measured before two test bugs were found and are withdrawn.

A caveat worth carrying: with a thin ellipse, *which* lattice points fall inside
depends on the channel's heading, so the effective stencil differs slightly
between channels, and clean accuracy is non-monotonic in aspect. The inclusion
test needs a tolerance (`≤ 1 + 10⁻⁴`) for exactly this reason — without it,
boundary taps drop out per-channel, the disc keeps 3.7 of its 4 taps at R=1, and
clean accuracy degrades by 13×. The real cost of the anisotropic stencil turned
out to be **heading resolution**, not curvature (L1).

---

## 3. Match to available hardware

### 3.1 SNSPDs — now the tight component

| device | jitter (FWHM) | source |
|---|---|---|
| high-I_c NbN | **15 ps intrinsic / 18 ps system** | arXiv:1308.0763 ⓛ |
| amorphous MoSi | 26 ps system | arXiv:1710.06740 ⓛ |
| SNAP | 62 ps, 4.9 ns reset | arXiv:1601.01719 ⓛ |
| multi-element array + SFQ | 50 ps | arXiv:1207.3902 ⓛ |
| WSi at 2.5 K | 191 ps | arXiv:1406.1810 ⓛ |

With an isotropic stencil the picture was bleak: at R = 1 only ~3 ps clears 3%
error, i.e. nothing available. The κ-aligned stencil (§2.5) changes the answer.
At R = 4 / aspect 0.5, 15 ps gives 0.78% and 25 ps gives 1.81%, so **high-I_c
NbN has comfortable margin and MoSi at 26 ps works.** 50 ps arrays sit at 21.6%
— usable for coarse velocity, not for the 10 ps target. WSi at 191 ps still
fails outright.

Dead time limits *event rate*, not intra-event resolution — the detector fires
once per track. The 95% dropout tolerance means sparse photon statistics are a
non-issue.

### 3.2 Kinetic-inductance delay lines — the enabling component

For a microstrip with kinetic inductance dominant, the width cancels:

$$v \;=\; \sqrt{\frac{d}{L_s\,\varepsilon_0\varepsilon_r}},\qquad
Z_0 \;=\; \frac{1}{w}\sqrt{\frac{L_s\,d}{\varepsilon_0\varepsilon_r}}.$$

Measured sheet inductances: NbN foundry **3 pH/sq**, Mo₂N **8 pH/sq**
(arXiv:2302.06830 ⓛ); NbN HKIL **8.5 pH/sq** (arXiv:2305.07607 ⓛ); NbTiN
**8.5 pH/sq** (arXiv:2310.11410 ⓛ); MgB₂ 40 nm "tens of pH/sq"
(arXiv:2305.15190 ⓛ).

| L_s (pH/sq) | dielectric | v/c | ps/mm | 10 ps | 40 ps | Z₀ (w=1 µm) |
|---|---|---|---|---|---|---|
| 3 | 100 nm | 0.102 | 33 | 307 µm | 1.23 mm | 92 Ω |
| 8.5 | 100 nm | 0.061 | 55 | 182 µm | 729 µm | 155 Ω |
| **8.5** | **50 nm** | **0.043** | **78** | **129 µm** | **516 µm** | **110 Ω** |
| 30 | 50 nm | 0.023 | 146 | 69 µm | 274 µm | 206 Ω |

Realistic: **33–146 ps/mm** ⓒ, versus free space 3.3 and silicon photonics ~14.

**Loss is not a problem.** At the reactive-to-dissipative ratio of 788 for
NbTiN microstrip (arXiv:2103.00656 ⓛ), a 32-hop path of 40 ps hops is 12.8
cycles: `exp(−π·12.8/788) = 0.95` ⓒ, negligible against the intentional
`g < g_crit ≈ 0.14` attenuation.

**Tunability is worth exploiting.** The same work reports inductance per unit
length changing up to 20% under DC bias, making channel delays electrically
trimmable — calibration becomes training, and `κ` is already trainable.

### 3.3 nTron — keep it out of the timing path

NbN nanocryotrons reach 48 dB gain at < 20 aJ/op with **timing jitter < 60 ps**
(arXiv:2409.17366 ⓛ) — more than the entire budget on its own, and now that the
budget has tightened to ~15 ps, decisively so.

The SNSPD's current step already carries a 10 GHz Fourier component whose phase
is `2πf·t_arrival`, so **direct reactive injection into the resonator needs no
active element in the timing path.** Reserve the nTron for fanout and readout
thresholding; prior art exists for exactly that (arXiv:2304.11700 ⓛ).

### 3.4 Layout — pitch floor

Line cost is **distance-weighted**: a tap at displacement `d` needs `|d|·Δt` of
delay, so counting taps understates every stencil beyond R = 1. The bank tested
has **192 channels** — 24 speeds × 8 headings, since 2D velocity needs both — and
each is a physically separate lattice. At 78 ps/mm with serpentine at 2 µm
line+space (available length = `pitch²/2 µm`) ⓒ:

| stencil | line/site (192 chan) | minimum pitch |
|---|---|---|
| iso R=1 | 255 mm | 0.71 mm |
| iso R=2 | 1116 mm | 1.49 mm |
| iso R=4 | 8140 mm | 4.03 mm |
| **aniso 4 / 0.50** | **3549 mm** | **2.66 mm** |
| aniso 5 / 0.40 | 4655 mm | 3.05 mm |
| aniso 6 / 0.25 | 5497 mm | 3.32 mm |

An earlier draft of this table costed 24 channels rather than 192 and so
understated every pitch by √8 ≈ 2.8×. Channel count is the dominant term in the
layout budget and is the first thing to attack if pitch becomes binding —
halving the heading count halves the line.

**The pitch fork still resolves for the optical scenario, but only just.** A
wavefront sweep needs ≥ 3 mm pitch to reach 10 ps/site (§1.5) and the
recommended κ-aligned stencil needs 2.66 mm — a margin close enough to be
uncomfortable rather than reassuring. Any of: fewer headings, a smaller Δt span,
a higher-inductance film, or multi-layer routing would restore margin. The µm-pitch ion scenario is incompatible; its delay
network must move to a separate tier.

### 3.5 A tension worth naming

Large pitch helps the delay network but **hurts the detector**: a large-area
SNSPD pixel has high self-inductance, giving slower reset and worse geometric
jitter. The 50–62 ps array and SNAP figures plausibly reflect this. With the κ-aligned
stencil the jitter budget is ~25 ps rather than ~3 ps, which makes this tension
manageable — but you still cannot grow the pixel to fill a 2.7 mm cell and expect
15 ps.

Mitigations in order: the κ-aligned stencil, which buys an order of magnitude of
jitter tolerance and is now the default recommendation; a small high-I_c NbN
active area at low fill factor with optical concentration; a SNAP to recover
speed at some jitter cost.

### 3.6 Readout multiplexing is the architectural payoff

A 128×128 SNSPD array conventionally needs 16 384 channels. Here the array
drives a passive lattice and you read **192 analog channels** — an 85× reduction,
with computation replacing the multiplexer. Precedent: CB-KID images 15×15 mm with **four**
readout channels (arXiv:2010.07491 ⓛ).

---

## 4. Alternatives considered

| approach | value | assessment |
|---|---|---|
| **κ-aligned anisotropic stencil** | 14× jitter tolerance at lower cost than iso R=4 | **done, §2.5** — the result that unblocks the programme |
| **Delay wedge at the pixel** | decouples physical velocity from electrical Δt | **strong** — 129 µm per 10 ps of skew |
| **2D Cartesian readout fit** | 38× over argmax, removes polar bias | **done** — free, already implemented |
| **Two-carrier vernier** | ambiguity beyond Δt = T/2 | strong; native to the phasor algebra |
| DC-tunable delay (20% range) | electrical trim; calibration becomes training | promising; range insufficient alone |
| Parametric up-conversion (KI TWPA) | higher f ⟹ finer δΔt | plausible; KI is intrinsically Kerr |
| Differential delay-line readout | proven imaging baseline | build as comparator, not target |
| Photonic delay network | ~14 ps/mm | dominated by KI lines 3–10× |
| MKID / TES | energy resolution | µs response — too slow |

---

## 5. The ladder

| rung | scope | deliverable | go/no-go |
|---|---|---|---|
| **L0** | Physical-units simulation | ps-unit spec sheet; stencil and readout characterised | **6/7 gates — done, see below** |
| **L1** | Device realism: curvature, yield, envelope | **3/3 gates — done, see below** | ≥ 80% yield at 10% delay scatter |
| **L2** | 2 pixels + 1 delay line | chain jitter measurement | total jitter < 25 ps |
| **L3** | 16-site line, 1 channel | Jeffress tuning curve | width matches prediction |
| **L4** | 1D bank, C = 8–16 | interpolated Δt readout | < 5% over 8–40 ps |
| **L5** | Add recurrence, margin 0.9 | contrast vs feedforward | measurable gain |
| **L6** | 32×32 2D | speed + direction, curved tracks | matches simulation |

### L0 — implemented, 6 of 7 gates passed

Run `julia --project=. demos/wave_velocity_bank_rung0.jl`. CPU only, a few
minutes, ~30 MB peak.

| gate | result |
|---|---|
| Δt = 10 ps recovered < 1% clean, worst-case heading | PASS — 0.17% |
| cheapest stencil R = 1 clears 1% clean | PASS — 0.17% (best tested 0.02% at R=2) |
| interpolation beats argmax by > 3× | PASS — 38.5× mean |
| FFT and real-space rollouts agree | PASS — 3.6 × 10⁻⁷ |
| tolerates ≥ 70% dropout | PASS — 95% |
| isotropic R = 1 tolerates ≥ 15 ps jitter | **FAIL — clears 3% only to 3 ps** |
| κ-aligned stencil tolerates ≥ 25 ps jitter | PASS — aniso 5/0.40 clears 3% to 25 ps |

The isotropic jitter gate is retained deliberately as the documented baseline
failure that motivated §2.5; it is not expected to pass.

Code changes made:

1. **`decode_velocity`: argmax → 2D quadratic in Cartesian κ**, with separable
   and argmax fallbacks. §1.4.
2. **`stencil_aspect`** — κ-aligned anisotropic coupling, per-channel envelope
   and per-channel `g_crit` so every channel sits at the same margin. §2.5.
3. **`margin` 0.99 → 0.9.** §2.2.
4. **`stencil_radius`** added, so stencil truncation is testable in 2D.
5. **`substeps` 16 → 64**, though 8 turns out to suffice. §2.3.
6. **`src/velocity_bank_hw.jl`** — physical units, frozen per-site jitter and
   dropout, per-link delay scatter with a real-space rollout, delay-line layout
   helper. The previous noise sweep used additive field noise, which is the
   wrong model for single-photon detection: a detector fires *once*, with *one*
   timing error, frozen for that event rather than resampled each substep.

The real-space and Fourier rollouts agree to 3.6 × 10⁻⁷ ⓒ, which cross-validates
both paths.

### L1 — implemented, 3 of 3 gates passed

Run `julia --project=. demos/wave_velocity_bank_rung1.jl`.

| gate | result |
|---|---|
| curvature × jitter, κ-aligned, to radius 10 sites | PASS — worst 3.58% |
| yield ≥ 80% at 10% delay scatter | PASS — 100% |
| yield ≥ 80% on a fully realistic device | PASS — 92% at 10/10/5% + 25 ps |

**Curvature does not interact with jitter.** This was the named risk and it did
not materialise: the κ-aligned stencil holds 0.13–3.58% across curvatures from
straight down to a 10-site radius, at up to 25 ps of jitter ⓒ. The reason is
structural — the readout is per-site and the stencil is only 4–5 sites long, so
it sees a locally straight chord. **Design rule: stencil half-length ≤ radius/2.**

**A false cliff, and a real bug.** An earlier version of this study showed
catastrophic failure at curvature 0.06 (269% error) — but for *every* stencil
including isotropic, and at *zero* jitter, which no stencil-shape or noise
explanation can produce. The arc was lapping: 150 sites of track around a
105-site circle, re-depositing at a different carrier phase. Arcs must be capped
below one circumference. Separately, `moving_drive` ignored `angle` whenever
`curvature ≠ 0`, so every curved test had silently run at heading 0 — a channel
bin *centre*, the most favourable case — while straight-track tests used the
worst case. Both are fixed.

**Yield is comfortable, and delay scatter is the only defect that matters** ⓒ:

| delay scatter | Q spread | gain error | jitter | median err | yield |
|---|---|---|---|---|---|
| 10% | — | — | — | 2.12% | **100%** |
| 20% | — | — | — | 5.26% | 42% |
| — | 30% | — | — | 0.85% | 100% |
| — | — | 10% | — | 1.11% | 100% |
| 10% | 10% | 5% | 25 ps | 1.59% | **92%** |

Resonator-Q non-uniformity to 30% and coupling-gain error to 10% are essentially
free. Delay scatter is the binding defect and the cliff is between 10% and 20%.

**Envelope width: investigated, largely a dead end.** `σ = 6` is flat across an
`R = 4` ellipse, so taps enter at full weight rather than tapering, and tying σ
to R ought to help. It does for `aniso 4/0.50` in the clean column (1.06% →
0.47% at σ = 3) — but across all six configs and five jitter levels the σ = 0.75R
and σ = 6 columns differ by less than the seed spread, and under jitter the
default is often marginally better ⓒ. At `R = 1` σ has *no* effect at all: one
tap distance makes the envelope a scalar that `g_crit` normalises away. An
earlier pass generalised that single clean cell into a "σ ≈ 0.75·R" design rule.
**It does not hold** — leave σ at the default.

**Heading is now the limiting quantity, not speed.** The κ-aligned stencil holds
0.084 rad (4.8°) worst-case clean and ~0.10 rad at 25 ps ⓒ. Widening the
envelope does not help. Isotropic reaches 0.001 rad clean but degrades to
0.30 rad at 25 ps, so it is not a fix and a hybrid bank would buy nothing in the
regime that matters.

**Worst case is not the same place for speed and heading.** Speed is worst when
the track falls half a bin off the nearest channel; heading is *best* there,
because the response is then symmetric between two channels. Heading is worst at
the quarter-bin offsets, where the fit is asymmetric — 0.084 rad against 0.011
at half-bin and 0.000 at bin centre ⓒ. Any sweep that tests only one offset will
flatter one of the two quantities.

**Not established.** Why heading is capped this way. The natural explanation is
Fourier reciprocity: a stencil elongated along the track is narrow in q-space
radially and wide tangentially, hence sharp in speed and blunt in heading. An
attempt to confirm this by measuring channel response half-widths **failed** —
the responses are broader than the channel grid, so the probe saturated against
the grid bounds and returned identical numbers for every configuration. The
explanation is reasoning, not measurement.

### The two open questions

**Heading accuracy, if 5° is not good enough.** This is the one performance
number that has stopped improving. Everything tried so far (envelope width,
isotropic channels, more headings) either fails to help or trades away the
jitter tolerance that made the design work. If the application needs better
than ~0.1 rad, it needs a different idea rather than a tuning pass — and the
first step is to actually establish the mechanism, since the attempt to measure
it failed.

**Wafer-scale delay-line uniformity** is now the binding fabrication number
rather than a loose one: yield is 100% at 10% scatter and 42% at 20%, so the
cliff sits inside the range of plausible process spread. No source consulted
states the achieved figure. This is a measurement on real wafers, not a
literature question, and it should gate any mask commitment.

Both questions that gated earlier passes — 2D jitter dilution (§2.5) and
curvature × jitter (L1) — are now **answered**.

**Wafer-scale sheet-inductance uniformity** remains unmeasured in the sources
consulted. 20% rms delay scatter costs 2.9% ⓒ, which is loose, and systematic
error is a pure rescale absorbed by one calibration constant — so this is
probably fine, but it is unverified. It is a measurement on rung 1, not a
literature question.

---

## Appendix — provenance

**Computed (ⓒ), `demos/wave_velocity_bank_rung0.jl`:** the κ ↔ Δt mapping and
carrier table; `τ_c = Δt` frequency-independence; Q and ringdown; transient
contrast vs margin; readout comparison; stencil radius vs jitter; substep floor;
jitter, dropout and delay-scatter tolerances; delay-line velocity, impedance and
loss; distance-weighted meander budget and pitch floor; pitch-vs-apparent-velocity
table; the κ-aligned stencil comparison (§2.5).

**Computed (ⓒ), `demos/wave_velocity_bank_rung1.jl`:** curvature × jitter;
the lap-limit guard; envelope-width sweep (negative result); heading error vs
position within the angle bin; fabrication yield over delay scatter, resonator-Q
spread and coupling-gain error.

**Withdrawn.** Curvature penalties reported in an earlier draft of §2.5 were
measured before two test bugs were found — `moving_drive` ignored `angle` under
curvature, and arcs were allowed to lap their own path — and are not reproducible.
A "σ ≈ 0.75·R" envelope rule was generalised from a single measurement and does
not hold. A claim that `substeps = 16` was too coarse was wrong; the floor is 8.
An attempt to measure channel selectivity in speed vs heading **failed** (the
probe saturated against the channel-grid bounds), so the Fourier-reciprocity
explanation for the heading limit is reasoning, not measurement.

**Literature (ⓛ):**

- arXiv:1308.0763 — SNSPD jitter, 15 ps intrinsic / 18 ps system, high-I_c NbN
- arXiv:1710.06740 — amorphous MoSi SNSPD, 26 ps system jitter, 80% SDE
- arXiv:1601.01719 — SNAP, 62 ps jitter, 4.9 ns reset
- arXiv:1207.3902 — multi-element SSPD array with SFQ, 50 ps
- arXiv:1406.1810 — WSi at 2.5 K, 191 ps
- arXiv:2409.17366 — nanocryotron, < 60 ps jitter, 48 dB gain, < 20 aJ/op
- arXiv:2304.11700 — nTron ripple counter for megapixel SNSPD arrays
- arXiv:2302.06830 — planarized NbN/Mo₂N process, 3 and 8 pH/sq
- arXiv:2305.07607 — SFQ process with 8.5 pH/sq NbN HKIL
- arXiv:2310.11410 — NbTiN films for KI-TWPAs, 8.5 pH/sq
- arXiv:2305.15190 — wafer-scale MgB₂, tens of pH/sq at 40 nm
- arXiv:2103.00656 — NbTiN microstrip, reactive/dissipative 788, 20% DC tuning
- arXiv:2010.07491 — CB-KID, 15×15 mm imaged with four readout channels
