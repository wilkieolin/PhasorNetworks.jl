# The spike-timing readout floor closes the lock-in EP window (carrier-locked readout)

> **Harness:** `scripts/ep_adiabatic_sweep.jl` &nbsp;|&nbsp;
> **Theory:** `docs/ep_rotating_extension.md`,
> `docs/phasor_lockin_derivation.tex` §Rotating Substrate &nbsp;|&nbsp;
> **Gates:** `scripts/ep_rotating_gates.jl` &nbsp;|&nbsp;
> **What to do next:** `docs/ep_rotating_followups.md`

> ### ⚠ Scope caveat (added after these runs)
>
> Every number below was measured with the readout grid fixed in the
> **co-rotating** frame. That is the pessimistic case, and it is not the only
> defensible model. `_quantize_phase` is the one non-U(1)-equivariant operation
> in the pipeline, so the carrier-cancellation theorem does *not* extend to it:
> rounding commutes with a rotation only on exact multiples of the grid. With
> the same grid fixed in the **lab** frame and an incommensurate carrier, the
> carrier sweeps the state across tens of bins per step and dithers the
> quantizer for free — median cos recovers from 0.44 to 0.998 at δ = 0.005
> turns, no injected noise (8 draws, toy width). Reproduce with
> `julia --project=. scripts/ep_readout_frame_check.jl`; discussion in
> `docs/ep_rotating_extension.md`, "The exception: a quantized readout".
>
> So the floor below is real **for a readout clock phase-locked to the carrier**
> — one spike time per period, which is automatically commensurate — and largely
> not real for a clock that free-runs against it. Which one applies is a hardware
> question that this package does not currently commit to; `LockinEP` has no
> `carrier` field, so the co-rotating choice was implicit rather than argued.
> Resolving it is `docs/ep_rotating_followups.md` §0, and it now gates the
> interpretation of everything here. Gate F in `scripts/ep_rotating_gates.jl`
> pins the equivariance boundary so this cannot drift again.

## Why this was run

The goal is EP that runs *while a symmetrically-connected spiking network is
running*. The carrier turned out to be free for the analog settle — for one shared ω it
cancels identically, so the rotating *settle* is the static settle
(`docs/ep_rotating_extension.md`). It is not free at the readout, which is the
caveat above. That leaves the gap that is not free: a
spiking substrate does not read a complex number off a neuron, it infers phase
from **when the neuron spiked**, and that time is resolvable only to about the
spike-kernel width. `SpikingArgs` defaults to `t_window = 0.01` against
`t_period = 1.0`, with each pulse smeared over `±2·t_window`.

The prediction was that this puts a **lower** bound on the probe amplitude ε —
the lock-in must resolve a response of order ε·χ above the floor — making the
usable window two-sided, where the exact-state sweep could only ever see its
upper edge.

## Result 1: the window is two-sided, and with exact readout that is mild

From the readout pilot (`grid_readout_pilot.csv`, 2570 rows, 784→256→64),
best median cos over (ω_p, n_cycles) at each ε:

| δ (turns) | ε=0.0003 | 0.001 | 0.003 | 0.01 | 0.03 | 0.1 | 0.3 |
|---|---|---|---|---|---|---|---|
| **0 (exact)** | 0.9095 | 0.9904 | 0.9986 | 0.9993 | 0.9994 | 0.9994 | 0.9994 |
| 0.005 | -0.004 | 0.003 | 0.023 | 0.069 | 0.150 | 0.254 | 0.441 |
| 0.01 | 0.002 | -0.005 | 0.016 | 0.041 | 0.119 | 0.178 | 0.305 |
| 0.02 | -0.001 | 0.008 | 0.013 | 0.048 | 0.113 | 0.129 | 0.249 |

The exact-readout row does turn over: 0.9994 → 0.9095 as ε drops to 3e-4. So a
lower boundary exists even without spike timing — that one is the Float32
finite-difference floor of the estimator itself. It sits far below the upper
boundary, so the exact-state zone is comfortably wide. That is the regime the
original sweep mapped.

## Result 2: with a quantized readout the window does not narrow, it closes

Every quantized row is catastrophic. **0.005 turns — 1.8° — takes the best
achievable median from 0.999 to 0.07.** There is no ε that recovers it: cos
rises monotonically all the way to ε = 0.3 and still only reaches 0.44, and
ε = 0.3 is far outside the linear regime where the hard projection basin-hops
(`results/ep_adiabatic/ANTICORRELATED_GRADIENT_NOTE.md`). Failure is 100% at
every one of the 644 settings per quantum.

The mechanism is specific, and worse than "added noise". A **deterministic**
quantizer has a dead zone: if the probe response is smaller than one bin, the
observed value never changes across the whole lock-in window, the demodulated
sum is identically zero, and the estimated gradient is zero — not noisy, zero.
Time-averaging cannot recover a signal that never moved.

## Result 3: dithering does not rescue it

Real spike timing is noisy, and noise dithers a quantizer — it converts a dead
zone into a biased coin whose mean tracks the sub-bin value, which averaging can
in principle recover. This is the obvious escape route, so it was measured
directly (`readout_jitter`, jitter expressed as a multiple of the quantum).

Median cos, pooled over projection / ε / ω_p / n_cycles:

| δ (turns) | jitter=0 | 0.25δ | 1δ | 4δ |
|---|---|---|---|---|
| 0 (exact) | 0.905 | -0.006† | -0.004† | -0.002† |
| 0.005 | 0.124 | **0.281** | 0.113 | 0.023 |
| 0.01 | 0.104 | 0.156 | 0.057 | 0.009 |
| 0.02 | 0.063 | 0.091 | 0.020 | 0.003 |

(n = 192 per cell; 3072 evaluations total.)

† With no quantum there is nothing to scale against, so jitter is an *absolute*
std in turns — 0.25 there is 90° of phase noise. Those cells say nothing about
dithering; they are a fixture artifact and are shown only for completeness.

Dithering is real but small: a genuine stochastic-resonance optimum near 0.25×
the quantum roughly **doubles** fidelity at δ = 0.005 (0.124 → 0.281) and does
so at every quantum (0.104 → 0.156, 0.063 → 0.091), then noise swamps the
signal. Doubling 0.12 does not reach 0.999. **The floor stands.**

## Result 4: soft projection helps the tail, slightly

The hard projection `ifelse(r > th, g/r, 1+0im)` is discontinuous at |g| → 0,
which is the structural candidate for the bimodal lock-in failures that
`ANTICORRELATED_GRADIENT_NOTE.md` attributed to an over-large ε. At exact
readout and no jitter (n = 74 / 69):

| projection | median | p10 | fail% | min |
|---|---|---|---|---|
| hard | 0.8761 | 0.309 | 53% | 0.0736 |
| soft | 0.9104 | 0.378 | 45% | 0.0834 |

(n = 96 each.) Directionally consistent with the structural reading, and every
statistic moves the right way: fail% down 8 points, median, p10 and min all up.

Split by ε, the effect is concentrated where the structural account predicts it
should be — at *large* ε, where the drive most often swings across the
discontinuity (n = 16 per cell, so read the trend, not the individual numbers):

| ε | hard median / fail% / min | soft median / fail% / min |
|---|---|---|
| 0.003 | 0.546 / 81% / 0.074 | 0.541 / 75% / 0.083 |
| 0.03 | 0.923 / 47% / 0.500 | 0.934 / 41% / 0.613 |
| 0.3 | 0.962 / 31% / 0.658 | 0.943 / **19%** / 0.794 |

At ε = 0.3 soft cuts failures from 31% to 19% and lifts the worst draw from
0.66 to 0.79, while *lowering* the median slightly — precisely the trade of
median for tail that a discontinuity fix should produce. At ε = 0.003 the two
are indistinguishable, which also fits: that regime is dominated by the
Float32 FD floor, not by the projection.

So the bimodal failures are at least partly structural, not merely "ε too
large". Worth keeping as an option; still not a fix, since it leaves fail% at
19% in its best cell.

(Note the soft projection must be the *equivariant* one, `g/sqrt(|g|²+ε²)`.
Reusing `soft_normalize_to_unit_circle`, which interpolates phase toward 0,
degrades EP-vs-FD from 0.023 to 0.198 because it is not U(1)-equivariant — see
`docs/ep_rotating_extension.md`.)

## What this means

The carrier is free; the readout is not. For lock-in EP on a spiking substrate,
the binding constraint is **spike-time resolution**, and at the package's
default `t_window/t_period = 0.01` it is not close — three orders of magnitude
of fidelity, not a factor to be tuned away. Neither dithering nor a smoother
projection changes that conclusion.

Three routes remain, in rough order of plausibility:

1. **Finer effective timing resolution.** The floor scales with the quantum, so
   the question is how far below 0.005 turns a device can go, or whether
   multi-spike / population coding buys sub-quantum phase resolution the way
   averaging alone does not.
2. **An estimator that does not resolve a small AC phase response.** Everything
   here follows from needing ε·χ to exceed the bin. A rule reading coincidence
   counts or spike-count rates rather than sub-degree phase shifts would not
   inherit this floor.
3. **A larger probe.** Ruled out as stated — ε ≥ 0.3 is where the projection
   basin-hops — unless the soft projection extends the linear range far enough
   to matter, which Result 4 suggests it will not on its own.

## Reproducing

```bash
julia --project=. -t 10 scripts/ep_rotating_gates.jl        # must pass first
EPS_OUT=results/ep_readout_floor \
EPS_GRID_EPS=0.003,0.03,0.3 EPS_GRID_OMEGA=0.02,0.05 \
EPS_JITTER=0.0,0.25,1.0,4.0 EPS_KMODE=zero EPS_PROJECT=hard,soft \
  julia --project=. -t 10 scripts/ep_adiabatic_sweep.jl grid
```

**Status of the numbers above.** Results 3–4 are from the complete dither study
(3072 evaluations, 8 replicates per configuration). Results 1–2 are from the
readout pilot, which is complete except for its most expensive corner
(`ω_p=0.005, n_cycles=8`); 2570 of 2688 rows. That corner is the *most*
adiabatic setting, so if anything it would strengthen Result 1's exact-readout
row and leave the quantized rows — which fail at every ω_p — unchanged.

The one number that would most repay more sampling is Result 4's per-ε split,
at n = 16 per cell. The pooled comparison (n = 96) is solid; the claim that the
soft projection's benefit *concentrates at large ε* rests on the finer split
and should be treated as suggestive.
