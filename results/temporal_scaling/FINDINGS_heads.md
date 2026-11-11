# Head mechanism: does slicing the D-symbol into Dh=D/H heads limit VSA fidelity?

**Hypothesis (raised from the LCA>LSA observation on audio keyword spotting).**
LSA/LCA project the input to `D` then **reshape** into `H` heads of width `Dh=D/H`;
attention/similarity then run on `Dh`-vectors. If the representation is a
*holographically distributed* HD-VSA symbol (information spread uniformly, VSA
similarity/bind defined on the full vector), slicing into `Dh` sub-vectors should
**degrade** the comparisons (similarity SNR falls as `√H`). The proposed fix:
give each head the **full `D`** (project `D→D·H`, one full-`D` symbol per head),
so VSA ops act on whole symbols.

Two tests on the headless **TIR** task (D=48 base, L=32, 2 seeds), the
**long-range** setting (`sig_max_frac=0.34`) as the discriminator. Harness:
`scripts/temporal_scaling_sweep.jl` (`exp_heads_tir`, `exp_fulldhead_tir`);
data `heads.csv`, `fulldhead.csv`.

---

## Test 1 — head-count sweep at FIXED params (`heads.csv`)

Because the Q/K/V projection is always `D→D`, varying `H` changes **only the
reshape** — parameter count is identical across the whole sweep (LSA 35,865;
LCA 29,961). A clean isolation of the slicing question. *Prediction under the
hypothesis: acc falls as `H` grows / `Dh` shrinks.*

**Long-range (frac 0.34), acc mean of 2 seeds:**

| | H=1 (Dh=48) | H=2 (24) | H=4 (12) | H=8 (6) | H=16 (Dh=3) |
|---|---|---|---|---|---|
| **LSA** | 0.556 | **0.676** | 0.583 | 0.671 | 0.616 |
| **LCA** | 0.552 | 0.566 | 0.565 | 0.596 | **0.660** |

**Spread (frac 1.0):**

| | H=1 | H=2 | H=4 | H=8 | H=16 |
|---|---|---|---|---|---|
| LSA | 0.460 | 0.467 | 0.474 | 0.445 | 0.440 |
| LCA | 0.444 | 0.460 | 0.465 | 0.449 | 0.451 |

**Result: the opposite of the prediction.** The full-D single head (H=1, Dh=48)
is the **worst** long-range config for both layers. Shrinking `Dh` does not hurt;
for **LCA it monotonically helps** (0.55→0.66 as Dh goes 48→3), and even `Dh=3`
(a 3-dimensional phasor comparison) beats full-D. Spread is essentially flat.
So the `√H` slicing cost is real in isolation but **dominated by the benefit of
more routing units** — more heads = finer attention at zero param cost.

## Test 2 — full-D heads vs sliced, direct AND matched-params (`fulldhead.csv`)

`full_d_heads=true` gives each head a full-`D` projection (`D→D·H`), combined
across heads by **VSA bundling** (param-free superposition — the combine faithful
to the "distributed symbol" framing; a learned `W_O` would contradict it).

| variant | D | Dh | params | long-range (0.34) | spread (1.0) |
|---|---|---|---|---|---|
| sliced (baseline) | 48 | 12 | 35.9k | 0.583 | 0.474 |
| **full_d** (proposal) | 48 | 48 | 99.4k | **0.557** | 0.480 |
| sliced_matched (widen) | 80 | 20 | 98.2k | **0.773** | 0.596 |

**Result: full-D heads don't help, and it's not close.**
1. **Direct (same D):** full-D is no better — slightly *worse* long-range
   (0.557 vs 0.583) — despite **2.8× the parameters**.
2. **Matched params:** spending the same ~98k on **width** (sliced, D=80) beats
   full-D by **+22 pts** long-range (0.773 vs 0.557) and +12 pts spread.
3. **Tell-tale:** full-D's *training* loss (~0.37) equals the tiny sliced
   baseline's — the extra 63k params don't even fit the train set better
   (sliced_matched reaches ~0.13). The full-D projection+bundle is a genuinely
   **wasteful parameterization**, not merely a generalization gap.

---

## Conclusion

Both tests converge: **`Dh`-slicing is not a bottleneck.** The learned
projections pack each head's needed information into its slice regardless of `Dh`
(fine down to `Dh=3`), and the parameters a full-D scheme consumes are far better
spent on **residual width**. The holographic-fragility concern does not manifest
for these learned representations on this task.

**Actionable takeaways:**
- **Head count is a free, positive knob** (fixed params) — turn it *up*,
  especially for **LCA long-range** (monotonic to Dh=3). This is deferred knob 8,
  and the data says the opposite of what the slicing concern predicted.
- **Do not** pursue full-D heads for a fidelity gain; if extra capacity is wanted,
  **widen `D`** (the strongest lever from `FINDINGS_knobs.md`).
- The **LCA>LSA** advantage is therefore *not* about head dimensionality (both
  slice identically). It more likely comes from LCA's learned anchor memory +
  binding + independent heads — a separate, promising direction.

## Caveats
2 seeds; LSA long-range is non-monotonic/noisy (H=2,H=8 best), but clearly not
decreasing. The full-D combine tested is **bundling**; a `W_O`-projection combine
is untested (but would add more params and contradict the VSA premise, and Test 1
already falsifies the slicing premise independently).

## Reproduce
```julia
include("scripts/temporal_scaling_sweep.jl")
exp_heads_tir(; use_cuda=true)       # Test 1 — heads.csv
exp_fulldhead_tir(; use_cuda=true)   # Test 2 — fulldhead.csv
```
