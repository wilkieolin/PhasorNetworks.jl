# Scaling stacked LSA/LCA blocks: which knob buys accuracy?

**Goal.** Decide whether *stacking* PhasorLSA/PhasorLCA blocks is a reliable
scaling knob, quantify the role of the FFN sublayer, and produce results that
transfer to scaling up an **audio-classification** network. Companion to the
depth-robustness study in `results/lsa_lca_residual/` (which showed stacks *don't
collapse* to depth 16 under ReZero) — this study asks the harder question: does
adding blocks actually *improve* accuracy, and if not, what does?

Harness: `scripts/temporal_scaling_sweep.jl`. Shrunk scale (D=48, L=32, B=48,
2 seeds, 40 epochs) chosen for local tractability — see *Cost & caveats*.

---

## Architectural finding that reframes the whole question

Reading the forward passes (`src/ssm.jl:480` LSA, `:651` LCA; `src/vsa.jl:657`
`similarity_outer_heads`): **PhasorLSA and PhasorLCA are pointwise in the
sequence axis `L`.** Their attention scores are computed per `(l,b)` slice
(head×head for LSA, anchor×key for LCA) and mix values only *within* a position.
**No layer routes information across sequence positions.** The only cross-timestep
transport is the causal **λ-conv SSM memory** inside the q/k/v/FFN `PhasorDense`
projections.

Consequences:
1. Cross-position "multi-hop" tasks (depth = #position-to-position hops) are
   architecturally unsupported — abandoned before spending compute on them.
2. The audio-relevant capability (gather evidence spread across time, integrate
   it, decide) is carried by the **λ memory + the per-position FFN transform**,
   *not* by attention depth. This predicts depth will be a weak lever and the
   FFN/width will be the real ones — borne out below.

## Task: Temporal Integration Recall (TIR)

A length-`L` sequence of `D`-dim phasors. One target value `v*` (∈ `n_vals`
classes) is planted, **noisily**, at `m_signal` timesteps spread across the
sequence; `n_distract` other timesteps carry *one-off* random values (also
noisy); a **constant cue** sits at the read position `L` and never contains `v*`.
Readout = similarity of the post-stack representation at `L` to the value
codebook. Because `v*` is the only value that recurs coherently while distractors
appear once, **no single frame separates signal from noise — the readout must
integrate across timesteps.** This is a *headless* readout (no learnable head),
so accuracy measures the *stack's* capability, not a readout head.

"Hard" regime (headroom for scaling to show): `n_vals=16` (chance 0.0625),
`m_signal=3`, `n_distract=16`, `noise=0.35`, `L=32`.

---

## Exp B — depth as a scaling knob (acc, mean of 2 seeds)

| depth | params | LSA | LCA |
|---|---|---|---|
| 1 | 12.0k | 0.421 | 0.428 |
| 2 | 23.9k | 0.445 | 0.418 |
| 3 | 35.9k | **0.474** | **0.465** |
| 4 | 47.8k | 0.428 | 0.408 |

**Depth is a weak, non-monotonic lever.** Both attention kinds peak around
**depth 3 (+≈5 pts over depth 1)** then *regress at depth 4*. Training loss keeps
dropping with depth (more capacity fits the train set: LSA final-loss 0.41→0.36)
but eval accuracy does not follow, and depth 4 is unstable across seeds
(LSA d4: 0.479 / 0.378). Stacking is **not** a reliable accuracy knob here.

## Exp C — FFN role (acc, mean of 2 seeds)

| FFN | d_ff | depth 2 | depth 4 |
|---|---|---|---|
| **off** (attn-only) | – | 0.176 | 0.164 |
| on | D/2 (24) | 0.370 | 0.401 |
| on | D (48) | 0.445 | 0.428 |
| on | 2D (96) | **0.486** | **0.474** |

**The FFN is the load-bearing component.** Removing it (a bare
`PhasorResidual(attn)` block) collapses accuracy to ≈0.17 — barely above chance
(0.0625) — at *both* depths. Pointwise head-mixing attention alone cannot do the
task. Adding the FFN and **widening it improves accuracy monotonically**
(0.37 → 0.45 → 0.49 at depth 2). The FFN — the per-position nonlinear transform
carrying the multi-timescale λ tape — is where the work happens.

## Exp D — depth vs width at matched parameter count (acc)

| ~params | deepen (D=48, more blocks) | widen (depth 2, larger D) |
|---|---|---|
| ~24k | depth 2: 0.445 | D=48: 0.445 |
| ~36k | depth 3: 0.474 | D=60: **0.509** |
| ~48k | depth 4: 0.428 | D=68: **0.561** |

**Width beats depth decisively at equal parameter budget.** At ~48k params,
widening (D=68, depth 2 → 0.561) outperforms deepening (depth 4, D=48 → 0.428)
by **+13 points**, and width scales *monotonically* while depth regresses. For a
fixed parameter budget, spend it on width, not depth.

## Exp E(i) — integration is real (acc vs #evidence frames, depth 3)

| m_signal | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| acc | 0.083 | 0.246 | 0.690 | **0.997** |

With one noisy frame the network is at chance (0.083 ≈ 0.0625); accuracy climbs
steeply with more coherent frames to **≈1.0 at m=8**. Direct proof that the task
*requires and rewards cross-timestep accumulation*, and that the λ-SSM memory
performs it — the exact primitive audio classification needs.

## Exp E(ii) — long-range integration needs the multi-timescale tape (acc)

| evidence location | FFN=hippo | FFN=default |
|---|---|---|
| spread (`sig_max_frac=1.0`) | 0.474 | 0.331 |
| early / far from query (`0.34`) | **0.583** | **0.167** |

When evidence must be carried a long distance (planted early, query at `L`), the
single-timescale (`default`) FFN **collapses to near chance (0.167)** while the
long-`:hippo` λ tape holds **0.58**. The multi-timescale memory is *essential*
for long-range temporal integration — directly relevant to long audio inputs.

## Contrast — depth on routing MQAR (shrunk, inconclusive)

| depth | 1 | 2 | 4 |
|---|---|---|---|
| far-acc | 0.094 | 0.112 | 0.073 |

At the shrunk scale (D=48, L=32, `n_keys=n_vals=16`, density 0.3, 1 seed) the
routing MQAR task stayed **at chance (0.0625) for every depth** — the model never
learned the content-addressed recall in this budget, so depth cannot be assessed
here. This is consistent with "depth is not the lever" but is *inconclusive* on
its own; full-scale MQAR (where the task is solvable) is already characterized in
`results/xform_depth/` (depth improves loss but accuracy saturates by depth 2).
The TIR conclusions above do not depend on this contrast.

---

## Recommendations for the audio scale-up

1. **Scale by widening, not deepening.** Depth ≥ ~3 gives no reliable gain and
   destabilizes; width (`D`) and **FFN width (`d_ff`)** are the monotonic,
   parameter-efficient levers. Prefer a shallow (2–3 block), *wide* stack.
2. **Keep — and widen — the FFN.** It is load-bearing; an attention-only tower is
   near-broken. Treat `d_ff` (expansion ratio) as a primary capacity knob.
3. **Use the multi-timescale `:hippo` FFN λ init** — it is essential once
   evidence must travel across many timesteps (long utterances); single-timescale
   memory collapses on long-range integration.
4. **Attention (LSA vs LCA) is a wash here** and, being pointwise in `L`, is not
   the temporal-integration workhorse — the λ memory + FFN are. Don't rely on
   stacking attention to add temporal reach.

## Cost & caveats

- **Tractability drove the scale.** The `Phase`-typed GPU path hits a
  scalar-fallback (~4×10⁸ allocs/step), so each trial is expensive and cost
  scales with depth×steps. Hence D=48, L=32, B=48, 2 seeds. Absolute accuracies
  are modest by design (headroom regime); the *comparisons* (depth vs width, FFN
  on/off, hippo vs default) are the result, and they are large and consistent.
- **2 seeds**: depth's +5pt bump and its depth-4 regression have seed-level noise;
  the FFN-off collapse, width>depth gap (+13pts), and hippo long-range effect are
  far larger than seed variance.
- Equal-epoch budget may under-serve deeper stacks — but deeper stacks reached
  *lower train loss* without higher accuracy, so the ceiling is generalization,
  not optimization.
- All drivers append per-trial and resume, so the study is re-runnable/extendable
  (more seeds, larger D) without redoing completed trials.

## Reproduce

```julia
include("scripts/temporal_scaling_sweep.jl")
smoke()                       # CPU sanity
calibrate_tir(; use_cuda=true)   # difficulty calibration (Exp A)
run_all(; use_cuda=true)      # Exp B–E + MQAR contrast + summarize_all()
summarize_all()               # print mean±std tables from the CSVs
```
Artifacts: `results/temporal_scaling/{depth,ffn,width,integration,depth_mqar}.csv`.
