# Designing wave media by purpose: the gain profile

**Status: simulation, and everything below was measured on 2026-08-13 unless
marked otherwise.** Every number in this document came out of a run; the ones
that are predictions rather than measurements say so. Companion to
`wave_dispersion_derivation.md` (the linear theory) and
`rf_wave_network_implementation.md` (the Tier-1 realization). Source:
`src/wave.jl`, `src/excitable.jl`, `demos/wave_mach_cone.jl`.

This is a design document, not a derivation. It exists because two media in this
codebase — `PhasorWaveSheet` and `ExcitableWaveSheet` — share a lattice, a
kernel, a carrier convention, and an FFT diagonalization, and yet cannot be
made to do each other's job. Working out *why* produced one concept that
generalizes, and it is the subject of §1.

---

## 1. The gain profile, and why dispersion cannot see it

Define the **pointwise gain**

    G(r) ≡ |emit(z)| / |z|        evaluated at r = |z|

This is a function of the *local amplitude*. It has no spatial frequency in it.

The dispersion relation's gain term, `g_eff = g/θ`, is **one point on this
curve — the tangent at the origin**, scaled by `g`. It arises from the
subthreshold limit `emit ≈ z/θ`, so `G(0) = 1/θ`. That is the whole of the
relationship between the two:

- **dispersion** — a function of spatial frequency `q`, at infinitesimal amplitude
- **gain profile** — a function of amplitude, at no particular `q`

Measured `G(r)/G(0)` at the shipped defaults (base `θ = 10.581`; excitable
`θ = 19.651, α = 6.0, β = 0.15`):

| r/θ | `PhasorWaveSheet` | `ExcitableWaveSheet` |
|---|---|---|
| **0.00** | **1.0000** | **1.0000** ← the only point dispersion sees |
| 0.30 | 0.9578 | 1.0037 |
| 0.80 | 0.7809 | 1.7449 |
| 1.00 | 0.7071 | 2.8070 |
| 1.36 | 0.5924 | **3.82 ← peak** |
| 2.00 | 0.4472 | 3.1034 |
| 20.0 | 0.0499 | 0.3469 |

The base curve is **monotonically decreasing** — maximal at rest — as a matter
of algebra: `G(r) = 1/√(r²+θ²)`. The excitable curve has a bump peaking at
**3.82× its rest value at r = 1.36θ**. That bump is FitzHugh–Nagumo's cubic
activator nullcline, written as a gain rather than as a nullcline.

**Dispersion is structurally blind to the bump.** `α` does not appear in
`M(q) = A + (g/θ)·Ŵ(q)` at all:

    base       g/θ = 0.0945   ρ_sub = 1.0797
    excitable  g/θ = 0.0509   ρ_sub = 0.9528     differs ONLY via θ, never via α

Set both to the same `theta_frac` and the two media have **identical dispersion
relations** while one is excitable and the other provably is not. A
linearization reports the slope at one point of a curve whose *shape* is the
entire question.

### 1.1 The monotonicity argument

Put both requirements on the same curve:

- **quiescence** — loop gain < 1 at `r = 0`
- **propagation** — loop gain > 1 at `r ≈ θ`

Both at once **requires `G` to be non-monotonic**. Since the base sheet's `G` is
monotonically decreasing by construction, the two conditions there are mutually
exclusive, and *no parameter can fix a monotonicity*. `g` scales the curve's
height uniformly; `θ` slides its knee along the amplitude axis; neither puts a
bump in it.

This also restates two older results:

- "`g` cannot stabilise the sheet in `:spike`" — `g` cannot change the curve's
  shape, only its height.
- The dimensionless control parameter is `frac = θ/(g·max|Ŵ|)`, not `g` — `frac`
  positions the medium's operating amplitude relative to the knee. Verified:
  firing outcomes identical across `g ∈ {0.3, 1, 3}` with `frac` held fixed.

---

## 2. Why `PhasorWaveSheet` is not an excitable medium

Measured at N=128, default λ and g, seeded with a disc of amplitude 20θ and
radius R ∈ {0,1,2,4,8,16,32} — that is **1 to 3209 sites, up to 20% of the
sheet**. Firing decays monotonically to **exactly 0 by t ≈ 31–50 in every
case**. `homeostasis ∈ {:none, :global, :local}` makes no difference: all three
give 0.0 sites firing over t = 20–60 from a point kick. There is no critical
nucleus, because there is nothing to nucleate.

### 2.1 The closed form, and why it leaves room that coherence then eats

A fully-firing neighbourhood delivers at most `g·Σ|W(d)|`; igniting the next
site needs that to exceed `θ = frac·g·max_q|Ŵ(q)|`, so propagation requires

    frac  <  Σ|W| / max|Ŵ|  ≡  frac_prop

Measured `frac_prop = 1.667` (`:dog`), `1.678` (`:aniso`). `g` cancels, which is
why the whole problem is scale-free in gain. But that bound assumes every
neighbour fires *and* their contributions arrive phase-aligned. They do not —
the DoG's inhibitory lobe subtracts, and the `e^{−iω r/c}` delay phases rotate
each neighbour's contribution by a distance-dependent amount:

| quantity | value |
|---|---|
| `θ` | 10.581 |
| `g·Σ\|W\|` (coherent upper bound) | 12.598 |
| ratio to θ | **1.191** — geometry allows propagation |
| actual max drive (3209 sites firing at \|emit\|=0.998) | **7.738** |
| as a fraction of the bound | **0.614** — the coherence loss |
| as a fraction of θ | **0.731** — short by 27% |

Geometry leaves 19% of headroom; coherence eats it and 27% more. The achievable
bound therefore lands **below** the flood boundary: propagation needs
`frac < 1.024`, quiescence needs `frac ≳ 1.25`. **The windows do not overlap,
and the gap is not marginal.**

Nor is it a kernel-design problem. For a purely excitatory, delay-free kernel
`Σ|W| = Ŵ(0) = max|Ŵ|`, so `frac_prop → 1.0` — strictly worse. Both boundaries
are set by the same ratio and move together.

### 2.2 The consequence for the "traveling firing ring"

The ring measured earlier at 512² is therefore most likely a **level set of the
growing linear mode** (`ρ_sub > 1`) marking where `|z|` crosses `θ`, not an
autonomous excitable front. Falsifiable test: a true excitable front's speed is
independent of seed amplitude; a level-set front's is not. **This has not been
run** — treat §2.2 as a hypothesis, not a retraction of the measurement.

A level-set front and a pushed front differ in ways that matter for any readout:

| | level set of a growing linear mode | pushed (excitable) front |
|---|---|---|
| energy source | the linear instability, everywhere at once | the front itself, locally |
| speed set by | linear dispersion (`v_g` at the marginal mode) | nonlinearity + recovery time |
| speed depends on seed amplitude | **yes** | no |
| survives `ρ_sub < 1` | **no** | yes — that is the point |
| behind the front | keeps growing → floods | returns to rest |
| colliding fronts | superpose | annihilate |
| constant profile over distance | no | yes |

Only the right-hand column gives a speed you can calibrate once and use as a
ruler, which is what any front-based velocimetry needs.

---

## 3. The two media, contrasted by purpose

Both specifications are statements about **what must not change**:

- **`PhasorWaveSheet`** — *the phase of the signal survives transport, while the
  substrate is free to transform it.*
- **`ExcitableWaveSheet`** — *the speed of the front is a constant of the medium,
  whatever lit it.*

These are visibly contradictory before any code is written: one says the signal
must dominate the medium, the other that the medium must erase the signal.

| | transport & compute | measure |
|---|---|---|
| what is preserved | the signal's phase | the medium's speed |
| what is discarded | amplitude, on the wire | everything about the stimulus |
| nonlinearity | **limiter** — `G` falls as a site activates | **amplifier** — `G` bumps above 1 near θ |
| dispersion | a feature (it *is* the computation) | fatal — one speed or no angle |
| memory | lives in `z`, must persist | must be actively erased |
| rest state | irrelevant; it is driven | mandatory and silent |
| object you reason with | `M(q) = A + gŴ(q)` | the front; **there is no `M`** |
| stability from | boundedness of emit | subcriticality + refractory reset |
| closed forms available | dispersion, criticality, parallel scan, DEQ | none of them |

The transport sheet needs `|Σ W·s| ≤ Σ|W|` for BIBO stability under arbitrary
input, and needs to stay FFT-diagonal so the kernel trains with closed-form
dispersion and criticality. The measurement sheet needs the front to *forget*
its stimulus — so the medium must inject its own amplitude, precisely the
regeneration the first design forbids — and needs to reset, so a second particle
can be measured at all.

**The entire difference is the sign of the emit gain's slope.** Same lattice,
same kernel, same carrier.

---

## 4. `ExcitableWaveSheet`: the two mechanisms

Full parameter documentation and the calibration table live in the header of
`src/excitable.jl`; this section records the design reasoning and the two
failures, which are more transferable than the final numbers.

### 4.1 Regenerative current — and why *threshold-localised* is the load-bearing word

    emit = z/√(|z|²+θe²) · (1 + α·σ((|z| − θe)/(β·θe)))

The point is not "more gain." It is that the boost is ≈1 below θ and `(1+α)`
above it, so it **raises the suprathreshold drive without touching `ρ_sub`**.
That decouples the propagation boundary from the flood boundary — exactly the
coupling that made §2 unfixable. Verified: quiescence held at every α tested,
at every threshold.

The general move: *when two requirements are coupled through a shared
parameter, do not hunt for a better value — add a term whose support is confined
to one regime.*

Note the boost is a **real positive scalar** multiplying the complex direction,
so `arg(emit) = arg(z)` exactly. **Regeneration is phase-preserving.** What
discards phase in this medium is the refractory reset and the threshold, not the
boost. See §7.

Alone, (a) gives a front that propagates but never recovers: interior fill
0.75 → 0.92, a filling disc rather than a pulse.

### 4.2 Refractory state — and two designs that both failed

    θe = θ·(1 + κ_r·u),      u ← min(1, ρ_u·u + rise·fire)

**Failure 1 — symmetric leaky integrator** (`ρ_u·u + (1−ρ_u)·fire`). Charges on
the *same* constant it discharges on, so at ρ_u = 0.98 a site needs ~35 steps of
continuous firing before `u` can gate it. Measured interior fill
**1.00 → 0.77 → 0.63 → 0.17 → 0.00**: the pulse travels as a filled disc and
only hollows out around t ≈ 75. That is slow adaptation, not refractoriness.

**Failure 2 — instantaneous** (`u ← max(ρ_u·u, fire)`). Worse. `θe` gates the
site's own *emission* as well as its excitability, so instant refractoriness
cuts the spike off before it can drive anyone. **Every** configuration in the
calibration grid went to a *negative* front speed — the pulse collapses inward.
In a real neuron the spike is emitted and *then* inactivation sets in.

**The fix: decouple the ON rate from the OFF rate.** They do different jobs —
the on-rate sets the pulse width, the off-rate sets the refractory wavelength.
This resolved a trade-off that had looked fundamental an hour earlier, going
from 6% amplitude invariance with a thick front to **1% with a clean annulus**.

### 4.3 Validation

`α = 0, κ_r = 0` reproduces `PhasorWaveSheet` to **1.5e-9** relative. That
assertion is in the test suite and is what makes this a fork rather than a
divergent copy: it pins the new terms to be strictly additive, so any drift in
the base dynamics surfaces immediately.

All four excitability criteria hold at N ∈ {96, 128, 160, 192, 256}:

| criterion | measured |
|---|---|
| quiescent rest state | 0 sites firing over 150 steps from a noise seed |
| critical nucleus | 1-site seed extinguishes, 5-site propagates |
| amplitude-invariant speed | **0.9–2.0% spread over a 167× range** (3θ → 500θ) |
| fronts annihilate | returns to complete rest at every size |

Suite: 1364/1364 (was 1328; +36 new in `test/test_excitable.jl`).

**Two traps worth carrying forward.** (i) The refractory wavelength
`≈ speed/(1−ρ_u)` **must exceed the sheet's half-width**: at ρ_u = 0.98 (λ ≈ 61)
a 192² torus does not annihilate — fragments seed re-entrant spirals and it
settles into turbulence; 0.99 (λ ≈ 122) fixes it. Small sheets hide this
completely — it passed at 160 and failed at 192. (ii) `front_speed` over a long
window silently averages in post-collision turbulence and returned 0.006 where
the truth was 1.22.

---

## 5. Mach cone: the measurement the medium was built for

`demos/wave_mach_cone.jl` → `demos/wave_out/mach_cone.gif`,
`mach_cone_geometry.png`. A particle crosses a 640² sheet at v = 3.0
sites/period; the medium's front speed is u = 1.265, so Mach 2.37 and
α = asin(u/v) = 24.9°. `u` is measured from a radial seed with no particle, so
the prediction is not fitted to the data it is checked against.

    predicted tan α = 0.465   (α = 24.9°)
    measured  tan α = 0.454   (α = 24.4°)          −2.8%
    implied u from the cone alone = 1.240   vs 1.265 measured directly

The ratio table is better evidence than the fit, because the `d_max` cutoff is
visible in it:

| d behind particle | measured half-width | d·tan α | ratio |
|---|---|---|---|
| 160 | 46 | 74.4 | 0.618 |
| 240 | 87 | 111.6 | 0.779 |
| 320 | 148 | 148.8 | **0.994** |
| 360 | 167 | 167.5 | **0.997** |
| 400 | 179 | 186.1 | 0.962 ← past d_max |
| 440 | 182 | 204.7 | 0.889 ← past d_max |

### 5.1 Three ways to measure the wrong thing here

**(i) The track must be wider than the critical nucleus.** At `moving_drive`'s
default `width = 0.8` the particle lays a line one site across — below the
nucleus — so the track fires but launches *no* lateral front: off-axis extent 1
site at mid-track, versus 66 at `width = 2.0`. What you see instead is the
circle from the *entry* point, where the drive dwells long enough to nucleate,
expanding and wrapping the torus. It looks like a wave. It is not the cone.

**(ii) The cone is in the instantaneous wavefront, not the ever-fired set.** The
ever-fired region is the *union* of the emitted discs, dominated by the largest;
a plane fit to the first-spike-time field over that union returns **v = 5.9
against a true 3.0**. Refractoriness is what rescues the picture — overlapping
fronts annihilate, so only the outer envelope survives. Hence an animation
rather than a still.

**(iii) The angle is approached asymptotically from below, and that is physics.**
Ratios 0.62 → 0.78 → 0.99 → 1.00. The near cone really is narrower than
`asin(u/v)`, because recently-emitted circles are small and strongly curved and
an excitable front obeys `v = v_plane − D·κ`: they have been expanding at less
than the plane-wave speed the prediction uses. A through-origin fit over the
whole range is biased low — two earlier attempts reported −13% and −21% for
exactly this reason. The cone is also only fully developed within

    d_max = v·L·(1 − sin²α)

because the envelope at distance `d` is set by the circle emitted
`x = d/(1−sin²α)` behind (maximise `√(x²sin²α − (x−d)²)` over x). Past `d_max`
the track has not existed long enough to supply it.

**One artifact left in deliberately:** the entry point emits a full circle whose
backward half wraps the torus and races round to meet the particle at
`t* ≈ N/(v+u)`. `L` is capped below `t*`. Push `MACH_L` past it and the tail of
the animation is that front annihilating against the cone — real behaviour, and
a decent annihilation demo, but not a cone.

---

## 6. Lessons for future variants

1. **Write the invariant first, then diff it against the parent's for
   contradiction.** "Self-limiting" and "excitable" are incompatible on paper.
   Both the `frac_prop` bound and the monotonicity argument of §1.1 were
   available in closed form before any simulation; they were found afterwards.

2. **The nonlinearity is the design; the kernel is decoration.** Every
   functional difference between these two media came from the emit function and
   one auxiliary variable. And kernel design cannot substitute: the "obvious"
   fix of removing inhibition and delay makes `frac_prop` *worse* (§2.1).

3. **Two boundaries that move together cannot be separated by tuning.** You need
   a term with support on one side only. §4.1.

4. **An auxiliary variable that both responds to and gates the primary state
   needs independent on/off rates.** §4.2 — it failed in both directions before
   this was obvious.

5. **Price the analysis machinery you are forfeiting, up front.** The transport
   sheet has `M(q)`, closed-form dispersion and criticality, a parallel scan and
   a DEQ solver. `excitable_simulate` has no `:scan` or `:deq` mode, because
   there is no fixed linear operator. That is the cost of regeneration and
   belongs in the spec, not in the discovery.

6. **Test the purpose, not the code.** The transport suite tests
   dispersion-vs-theory, scan/rollout equivalence, gradient flow — *fidelity*.
   The excitable suite tests quiescence, nucleus, amplitude invariance,
   annihilation — *reliability*. Neither would catch the other's failures. The
   cautionary data point: the full suite once passed unchanged while θ moved
   five orders of magnitude, because spike-mode coverage was shape-and-
   finiteness only.

7. **A summary statistic will happily be satisfied by the wrong mechanism.**
   Four instances in one session:
   - "ever fired" scored 5.27% and was labelled excitable; it was a
     `ρ_sub = 1.08` blowup at t ≈ 90, unrelated to the seed.
   - the homeostat hit its firing-rate target with a *uniform subthreshold*
     sheet — a mean cannot distinguish "5% at 1" from "100% at 0.05" (fixed with
     a hard forward pass / straight-through estimator).
   - `front_speed` returned 0.006 where the truth was 1.22.
   - the Mach fit read −13% then −21%, sampling a cone the track was too short
     to have produced.

   Each fix was an *additional discriminating measurement* — intermediate-time
   firing, a hard indicator, a wrap-bounded window, the `d_max` bound — never a
   loosened threshold. When a number disagrees with the model, the first
   hypothesis should be that you are measuring something else.

8. **Keep the degenerate reduction to the parent as a test.** §4.3.

### 6.1 Checklist

1. State the invariant — what must not change.
2. Diff it against the parent's invariant for contradiction.
3. Sketch `G(r)`. Is it monotone? That decides the class of medium.
4. Identify which boundaries are coupled; find the localizing term.
5. List the closed forms being given up.
6. Write purpose-shaped tests.
7. Keep the reduction to the parent green.

---

## 7. Open, and one thing the axis points at

**Unblocked but untested.** Ignition-front readouts — Mach cone, first-spike
trilateration, P/S two-speed ranging — needed an autonomous front with a
calibrated speed, and there now is one. §5 demonstrates the *medium* behaves
correctly; no velocimetry has been demonstrated. The recommendation to use
`PhasorVelocityBank` for velocimetry is unchanged, since none of its structural
advantages (dense per-site field, exact superposition for multi-track, no
criticality requirement) are affected.

**Untested hypothesis (§2.2).** Whether the 512² "traveling firing ring" is a
level set rather than a front. Sweep seed amplitude and watch whether 1.844
moves.

**The interior of the axis is unexplored.** These two media are endpoints of a
gain-profile axis. Because the regenerative boost is phase-preserving (§4.1),
the combination *regenerative boost, no refractory variable, dispersion
retained* is coherent: a **phase-preserving repeater** that restores amplitude
over long transport without overwriting the carried phase — which is exactly the
attenuation limit on the original wavesheet's range. Untested, and the obvious
risk is that `α > 0` with `ρ_sub` near 1 simply makes the medium unstable rather
than regenerative. But it is the design the axis points at, and it is cheap to
check with the layer that now exists.
