# Phasor EP on FashionMNIST — first scale-up past toy problems

**Question.** Vanilla phasor EP (`src/ep.jl`) had only ever been run on a single
fixed pattern and the 4-corner XOR (`demos/lockin_demo.ipynb` §7, a 2→16→1 chain).
Does it work on a real dataset — and specifically, does **`LockinEP`** (one settle
plus one driven trajectory, demodulated at the probe frequency) match **`StaticEP`**
(two equilibria, finite difference) at MLP scale? Lock-in is the hardware-relevant
estimator: a synchronous demodulator is an analog primitive, and it needs no second
equilibrium computation.

**Setup.** 784 → 256 → 64, two `PhasorDense` layers, `normalize_to_unit_circle`,
complex bias on both, `K_mode = :zero`, 217,728 parameters. Readout is
`CodebookCost` over 64×10 `orthogonal_codes`. Full 60K train / 10K test, batch 128,
Adam 3e-3, weight decay 1e-4, `centered = true`. Harness: `demos/ep_fashionmnist.jl`.
CPU only (10 BLAS threads).

Pixels → phase via `0.5·tanh(standardize(x))` → `[-0.5, 0.5]`. The half-plane range
is deliberate: phases wrap, so `Phase(1) ≡ Phase(-1) ≡ -1+0i` and a full-range map
is not injective. The repo's usual static path `Phase.(tanh.(LayerNorm(x)))`
saturates *toward* that collapse point.

## Headline result

| estimator | epochs | best acc | final acc | steps/gradient | wall clock |
|---|---|---|---|---|---|
| `StaticEP` (centered, β=0.1) | 20 | **0.8361** | 0.8268 | 400 | 31.4 min |
| `LockinEP` (ε=0.03, ω_p=0.02) | 10 | **0.8423** | 0.8301 | 3968 | 172.3 min |

**Lock-in matches static on a real 10-class problem at 217K parameters**, slightly
ahead on peak accuracy in half the epochs, at 5.5× the wall clock. Both are stable
across the whole run. Chance is 0.10; accuracy at init is 0.134.

Note the loss floor: `CodebookCost` logits are `(1/d)·Re⟨code_c, z⟩ ∈ [-1,1]`, so
softmax CE cannot go much below ~0.8 even with perfect separation. Loss around 1.67
is not a plateau in the usual sense — read accuracy.

Raw data: `epoch_curves.csv`, `run_final.txt`, `run_baseline_nodecay.txt`.

## Three things that had to be fixed, and one that didn't matter

### 1. The optimizer, not EP, was the first bottleneck

`ep_train` hardcoded `Optimisers.Descent`. That is what the XOR demo used with four
training patterns; at 217K parameters on 60K images it does not work at any learning
rate (`hyperparam_sweeps.csv`, 10K subset, 5 epochs):

| optimiser | lr | final acc |
|---|---|---|
| Descent | 0.05 / 0.02 / 0.005 | 0.476 / 0.462 / 0.493 |
| Adam | 0.01 / **0.003** / 0.001 | 0.763 / **0.798** / 0.772 |

`ep_train` now takes an `optimiser` kwarg (a constructor, matching `train`).

### 2. Lock-in adiabaticity had to be recalibrated at width — the toy defaults are wrong

The package default `ω_p = 0.05` is **not adiabatic** at this width. The relevant
quantity is the relaxation rate, fitted from the settle residual decay:

    R_relax ≈ 0.081–0.134 /time-unit   at 784→256→64

not ≈1. The damped iteration `z ← (1-dt)z + dt·unit(grad)` would relax at rate 1
only if the drive were independent of `z`; the inter-layer feedback puts the
coupling Jacobian's slowest mode near 1, and that mode sets the rate. So `ω_p=0.05`
is a ratio of ~0.4, not ~0.05.

The fix costs nothing. Lock-in cost is `period_steps = 2π/(ω_p·dt)`, so **ω_p and dt
trade off directly**: moving the lock-in settle from `dt=0.1` to `dt=0.5` buys a 5×
slower probe *in time units* at identical step count. `dt=0.5` is the same step the
static settle already uses, and the probe increment `ω_p·dt = 0.01` rad/step is far
from aliasing.

Calibration against centered `StaticEP` at small β (`lockin_calibration.csv`; FD is
unaffordable at 217K params — it needs `n_params + 1` settles):

| ε | ω_p | steps/grad | cos(L1) | rel-err(L1) |
|---|---|---|---|---|
| 0.03 | 0.005 | 15278 | 0.9999 | 0.015 |
| 0.03 | 0.010 | 7742 | 0.9987 | 0.052 |
| **0.03** | **0.020** | **3968** | **0.9983** | **0.062** |
| 0.03 | 0.050 | 1706 | 0.9811 | 0.197 |

ε=0.03/ω_p=0.02 gives width-8 gradient quality (the toy chain's reference is
rel-err 0.054) at half the original step count.

**Larger ε is better here** — the opposite of the width-8 guidance in
`lockin_demo.ipynb` §4 ("wider chains need a smaller ε"). At this width the
demodulator noise floor dominates over the O(ε²) nonlinearity: ε = 0.01/0.03/0.10 at
ω_p=0.02 give rel-err 0.078/0.062/0.059.

### 3. EP's gradient decorrelates as ‖W‖ grows — basin hopping, not linearization error

Without weight decay, accuracy peaks at epoch 3 (0.791) and decays to 0.726 by epoch
20 while `‖W₁‖` grows 26.7 → 108.9, roughly linearly and unbounded. Nothing in the
loss penalizes weight scale, because `normalize_to_unit_circle` makes the states
scale-invariant.

**The settle is not the problem — for *this* failure.** The stationarity residual
stays ≲1e-6 (often exactly 0) throughout the decay. A convergence check cannot see
the basin hop described below.

> **Amended.** The mechanism claim is right — it is a basin hop — but "a
> convergence check cannot see this failure" is too strong, and applies only to
> the *step-to-step residual* reported here.
> `results/ep_trained_vs_rescaled/FINDINGS.md` Result 5 measures the **T-vs-2T
> drift** `‖z(2T) − z(T)‖/√N` instead, and it separates cleanly: over 96 probe
> points where drift and the estimator share a settle length, no cell above
> drift 1e-3 reaches cos 0.99 and only one below it falls under (0.969). In the
> worst cell the residual reads 9.3e-5 while the drift reads 1.3e-1 — three
> orders apart, which is why one sees the failure and the other does not.
>
> Two details that are easy to get wrong. The drift must be evaluated at the
> **estimator's own `T_free`**: measured instead at a longer oracle settle, it
> misses 7 of 34 failing points, one of them at drift 8.8e-8 with cos −0.624.
> And `T_free = 200` — this run's setting — is itself sometimes the cause: it
> fails at 8 of 20 trained snapshots there, and raising it to 1600 fixed every
> point probed.

Direct probe (`gradient_fidelity_vs_weightnorm.csv`), EP against a small-β centered
reference on **randomly initialized matrices rescaled to the stated norm** — read
the amended caveat below before generalizing these to trained weights:

| ‖W₁‖ | β | cos(L1), one-sided | cos(L1), centered |
|---|---|---|---|
| 7.9 | any (0.003–0.3) | 0.995 | 0.995 |
| 31.4 | 0.1 | 0.104 | 0.380 |
| 125.8 | 0.1 | 0.254 | 0.684 |

At the init scale everything agrees and is flat in β. Past ‖W₁‖ ≈ 15 the one-sided
estimate decorrelates entirely, and the **relative error scales as 1/β** — 28.7,
96.9, 292, 976 at β = 0.1, 0.03, 0.01, 0.003 for ‖W₁‖ = 31.4.

That 1/β scaling is the diagnostic: it means `h_nudge - h_free` retains a
**β-independent** term. The free and nudged settles are converging to *different
fixed points*. This is a basin hop, not a failure to linearize — which is why
shrinking β does not help and actively makes the estimate worse. EP's premise is
that the nudged equilibrium is a smooth deformation of the free one; at large ‖W‖
that premise fails.

`centered = true` (settle at ±β, use `-(h₊-h₋)/(2β)`) recovers a substantial part of
it and is now recommended for any long run. Added as a `StaticEP` field.

**Caveat — now measured, and stronger than stated.** These numbers come from
randomly initialized matrices rescaled to the stated norm. That caveat was a
hypothesis when written; `results/ep_trained_vs_rescaled/FINDINGS.md` measured it
against an independent directional-FD oracle, on a run reproducing this one to 0.3
accuracy points and 0.4 in norm. At the matched norm ‖W₁‖ ≈ 32:

| β | 0.3 | 0.1 | 0.03 | 0.01 | 0.003 |
|---|---|---|---|---|---|
| **trained** cos / rel-err | 0.9999 / 0.012 | **1.0000 / 0.008** | 1.0000 / 0.010 | 0.9998 / 0.022 | 0.9977 / 0.070 |
| **rescaled** cos / rel-err | −0.031 / 12.0 | **−0.059 / 35.8** | −0.069 / 119.2 | −0.072 / 357.3 | −0.073 / 1190.8 |

The rescaled arm reproduces this table's 1/β blow-up to two figures (28.7 → 976
here, 35.8 → 1190.8 there). The trained arm shows **no decorrelation at all** and is
flat in β. Trained structure rescaled to ‖W₁‖ = 89.4 — 2.8× this threshold — still
gives cos 0.9996, so *scale is not the variable; randomness of the matrix is.*

So **‖W₁‖ ≈ 15 is not a threshold of the estimator** and should not be quoted as
one. It is a property of the random-matrix construction used to probe it, and that
construction is pathological in its own right: all 165 FD-untrusted rows in that
sweep are in the rescaled arm and none in either trained arm — at most of its probe
points the loss is not locally linear at any step size.

Two things this does *not* retract: `centered` is still the right mitigation
(measured there at cos 0.929 and **flat** in β — it cancels the β-independent term
rather than attenuating it, which is a cleaner claim than the 0.380 above), and
training at ‖W₁‖ ≈ 27 is fine.

### 4. Weight decay helps, but not for the reason we assumed

wd = 1e-4 clearly helps (10K/8-epoch probe: peak 0.771 → 0.793, final 0.721 → 0.775).
The obvious explanation — that it bounds ‖W‖ and so keeps EP in the good regime — is
**not supported**:

- wd=1e-4 leaves the weight-norm trajectory nearly unchanged (13.9→30.4 vs 13.1→27.1)
  while clearly improving accuracy.
- wd=1e-3 bounds ‖W‖ much harder (→15.7) and performs *worse* (0.699).

So the benefit at 1e-4 is not from bounding ‖W‖, and "more bounding" is not better.
Mechanism unresolved; treat it as ordinary regularization. The large-‖W‖ mitigation
is `centered`, not decay.

**Attribution caveat on the headline run.** The final configuration changed
`centered` and `weight_decay` together relative to the baseline, so the table below
confirms the combination works but does not apportion credit between them. The β-sweep
above is the evidence that `centered` specifically addresses the basin-hopping.

| | baseline (Adam only) | + centered + wd=1e-4 |
|---|---|---|
| peak | 0.7908 (epoch 3) | 0.8361 (epoch 8) |
| final | 0.7257 | 0.8268 |
| ‖W₁‖ | 26.7 → 108.9 | 20.4 → 25.3 (plateaus) |
| loss | 1.84, rising | 1.67, flat |

Also worth noting: the good run's settle residual sits around 1e-4, the decaying
baseline's around 1e-9. **Residual size on its own is not a quality signal.**

## Implementation work this required

`src/ep.jl` was strictly single-sample (`Vector{Vector{ComplexF32}}` states) and did
per-example SGD — a documented accuracy cap in `docs/src/api/ep.md`.

- **Batching.** States are `(out,)` or `(out, B)` on one code path (`map`-based type
  inference in `_phasor_step`, shape-matched init). Every operation in the settle is
  column-separable, so a batched settle is exactly B independent settles. The `1/B`
  goes on the **Hebbian, not the nudge** — each sample must see the full per-sample
  nudge amplitude or the linear response shrinks by B and the FD SNR collapses.
  Verified: batched gradient == mean of per-sample gradients to ≤2.8e-4.
- **Hoisting the input drive.** Layer 1's `ep_drive` is `W₁·z₀`, constant for the
  whole settle, but was recomputed every step. 2.6× on the per-step linear algebra at
  B=128 (4.8× at B=1).
- **Lock-in factorization.** Layer 1's `z_in` is `z₀`, constant in `t`, so it factors
  out of the demodulation sum:
  `Σₜ z₁(t)·z₀'·e^{-iω_p t} = (Σₜ z₁(t)e^{-iω_p t})·z₀' - c·H_dc`, with
  `c = Σₜ e^{-iω_p t}` a closed-form scalar. Turns ~10⁴ full `(256×784)` complex
  outer products into ~10⁴ `(256×B)` accumulations plus one matmul. **19× at B=128**;
  verified rel-err 3.4e-7 against the original loop. Layer 2 cannot be factorized
  (its `z_in` varies) and uses 5-arg `mul!` instead.
- Also: `centered` on `StaticEP`, `weight_decay` honored in `ep_train`, `callback`
  hook, `ep_predict`/`codebook_logits` (needed because `loss_and_accuracy` assumes a
  feedforward `model(x, ps, st)` call an EP-settled network cannot provide).

Tests 69 → 112 in `test/test_ep.jl`; full suite 1464/1464.

New in the test suite: batched-vs-looped equivalence for all three estimators, batched
cost identities, and a 49→12→10 FD proxy with a 10-class codebook (cos = 1.0,
rel-err 0.004).

**Aside — `fd_gradient_phasor`'s default ε is under-conditioned for this cost.**
ε=1e-5 is tuned for the toy chains' O(1) `SimilarityCost`. Against a 10-class
cross-entropy in Float32 the difference falls below machine resolution and the
*oracle* becomes the noisy party: EP-vs-FD relative error runs 0.24 at ε=1e-5, 0.027
at 1e-4, **0.004 at 1e-3**, 0.036 at 1e-2. Tune ε to the loss scale before drawing
conclusions about gradient quality.

## GPU port and characterization

`src/ep.jl` was host-allocating and would not run on a GPU at all. Making it
device-agnostic was a contained change — the math was already fine; only the
allocations were wrong:

- `_init_states` and the lock-in accumulators (`Zhat`, `HW`) now go through
  `gpu_zeros(ref, T, dims...)` (`src/backend.jl`) instead of bare `zeros`.
- `ep_hebbian` / `_pad_dynamics_zeros` use `zero(ps.log_neg_lambda)` rather than
  `zeros(Float32, size(...))`, preserving the array type.
- `CodebookCost` moves its one-hot target onto the codebook's device. The one-hot
  is assembled with scalar indexing, which is illegal on a GPU array, so it is
  built on the host and copied (`_match_device`).
- The lock-in accumulator `Dict`s and `_ep_lockin_gradient`'s signature were
  concretely typed to host `Matrix`/`Vector`; loosened.

`fd_gradient_phasor` stays CPU-only by design — it perturbs parameters with scalar
`Pp[i] += ε` indexing. It is a test oracle, not a training path.

Parity is verified in CI (`ep_gpu_parity_tests`, called from `test_cuda.jl`):
worst CPU-vs-GPU relative error **3.2e-5 (StaticEP), 7.0e-5 (LockinEP)**.

### Was the "GPU probably won't help" prediction right?

Partly. The prediction was that the settle is a long chain of *small sequential*
matmuls (after hoisting the input drive, each step is only 256×64 and 64×256), so
GPU would be launch-latency-bound and might even lose. Measured throughput in
samples/s, `StaticEP` at 400 steps/gradient (`gpu_throughput.csv`):

| B | CPU | GPU | speedup |
|---|---|---|---|
| 32 | 344 | 337 | 0.98× |
| 128 | 605 | 898 | 1.5× |
| 512 | 537 | 1663 | 3.1× |
| 2048 | 426 | 8546 | 20× |
| 8192 | 460 | 9559 | 21× |
| 32768 | — | 8342 | — |

Right about the small-batch regime: at B=32 it is a dead heat, confirming
launch-bound behaviour, and the crossover lands at B≈512–2048, close to the
predicted ~1024. Wrong that GPU would be *slower* — it ties at worst.

The more useful finding is about **CPU**, not GPU: CPU throughput peaks at B=128
(605 samples/s) and then *declines* with larger batches. So the batch size the
training runs used was already at the CPU optimum, and there is no CPU-side win
available from batching harder. All the headroom is on the device: **~16× at each
platform's best batch** (605 → 9559), ~21× at matched B=8192.

GPU timing is mildly non-monotonic (B=512 measured slower than B=2048) — treat
individual points as ±30%; the trend is what matters.

**These are pre-fusion numbers.** The step-fusion work below supersedes them; see
that table for current throughput. The CPU/GPU crossover point is unchanged.

Caveat: `LockinEP` numbers in the CSV use a reduced 289-step configuration so the
sweep was affordable, not the 3968-step production setting. Per-gradient times are
therefore not comparable across the two methods; compare each method to itself
across devices.

### Step fusion (the "buffer reuse" follow-up)

The settle allocated **2.8 MiB per step** at B=128 — 1.1 GiB of garbage per
400-step settle. Profiling the step (784→256→64, B=128) put the cost in three
places: the layer drive (92 µs), the feedback matmul (80 µs), and
normalize-plus-damping (73 µs) out of 335 µs total.

A full buffer-reuse rewrite — states split into real/imag `Float32` pairs so every
matmul is a `mul!` real gemm, zero allocation — was prototyped and measured at only
**1.44×**. That is not worth the churn: it would change the state representation
that `chain_hebbians`, `ep_hebbian` and the lock-in accumulators all consume.

Most of the available win turned out not to be buffer reuse at all:

1. **Promote the weights to complex once per settle** (`_weight_cache`). `ps.weight`
   is real and the states are complex, so the `PhasorDense` functor splits into
   `W*real(x)` and `W*imag(x)` — two real gemms plus five array temporaries.
   A single `cgemm` on a pre-promoted weight is 2.3× faster on the drive and 1.4×
   on the feedback, despite nominally doing 2× the arithmetic. The win is memory
   traffic, not flops.
2. **Fuse projection and damping into one broadcast** (`_project_damp`). The
   expression `(1-dt).*z .+ dt.*normalize_to_unit_circle(g; ε=0)` materializes four
   temporaries and computes `|g|` twice; the scalar kernel does one pass.

Both are non-mutating, so the code stays functional and GPU-clean. Results are
**bit-identical** — every EP-vs-FD figure in the test suite is unchanged to four
decimals.

Throughput before → after (samples/s, `StaticEP`, 400 steps/gradient):

| B | CPU before | CPU after | GPU before | GPU after |
|---|---|---|---|---|
| 32 | 344 | 528 (1.5×) | 337 | 706 (2.1×) |
| 128 | 605 | **1055** (1.7×) | 898 | 1985 (2.2×) |
| 512 | 537 | 977 (1.8×) | 1663 | 4086 (2.5×) |
| 2048 | 426 | 851 (2.0×) | 8546 | **27555** (3.2×) |
| 8192 | 460 | 776 (1.7×) | 9559 | 25028 (2.6×) |
| 32768 | — | — | 8342 | 25243 (3.0×) |

The end-to-end gain (1.7–2.0× CPU, 2.1–3.2× GPU) exceeds the 1.55× measured on a
single isolated step, because the fusion also removes GC pressure across the whole
gradient and, on GPU, collapses several kernel launches per layer into one.

Memory also improved, 27 → ~20 KiB/sample, and now scales cleanly (B=8192 measures
0.149 GiB rather than the previously noisy 2.9 GiB).

**Trap worth recording.** Do not "finish the job" by calling `mul!` with a real
`transpose(W)` against a complex operand. That combination misses BLAS entirely and
hits a generic fallback measured at **1561 µs vs 80 µs** for the allocating
`transpose(W) * z` — a 19× pessimization. Promote the transpose to complex too.

### Memory

Comfortably within bounds, and the interesting part is that it is **flat in settle
length** (`gpu_memory.csv`):

| B | T | peak | per-sample |
|---|---|---|---|
| 2048 | 25 | 0.053 GiB | 27.1 KiB |
| 2048 | 100 | 0.053 GiB | 27.1 KiB |
| 2048 | 400 | 0.052 GiB | 26.9 KiB |

The settle allocates fresh state arrays every step and never reuses buffers, so a
400-step settle at large batch looked alarming at first — naive readings suggested
8.6 GiB at B=2048. That was **pool reservation, not live data**: `CUDA.used_memory()`
sampled after the fact reflects what the allocator is holding, not the working set.
Sampling the true high-water mark concurrently shows ~27 KiB/sample independent of
T, which matches the analytic working set (z₀ at 6.3 KiB/sample plus a handful of
320-channel state temporaries).

The pool will opportunistically expand to fill whatever headroom it is given, so
set a hard limit. It respects one:

| B | cap | peak | outcome |
|---|---|---|---|
| 8192 | 2 GiB | 1.97 GiB | completes |
| 8192 | 4 GiB | 1.23 GiB | completes |
| 32768 | 6 GiB | 1.80 GiB | completes |

B=32768 — 16× the training batch — fits in under 2 GiB when capped. Against this
machine's ~110 GiB usable, there is no risk from EP at any batch size worth using.
The full encoded 60K training set resident on device is 0.175 GiB.

**Recommendation.** Keep training on CPU at B=128 unless you want large-batch runs;
the GPU path is now available and correct, and becomes worth using from B≈512 up,
where it is 3–21×. Always run it under `JULIA_CUDA_HARD_MEMORY_LIMIT`.

## Open

- Attribute the final run's gain between `centered` and `weight_decay`.
  **Partly answered** — see the next item; with `centered` on, decay is a net loss.
- ~~Why does wd=1e-4 help without bounding ‖W‖?~~ **Answered: with `centered` on it
  does not help.** `results/ep_trained_vs_rescaled/FINDINGS.md` trains both arms to
  20 epochs with `centered = true`: **0.8636 at ‖W₁‖ = 89.4 with no decay**, against
  0.8296 at ‖W₁‖ = 25.7 with wd=1e-4. Decay costs 3.4 points while bounding the norm
  to under a third. The `baseline_nodecay` collapse recorded above (0.726 at
  ‖W₁‖ = 108.9) was the **one-sided estimator failing as ‖W‖ grew, not the norm
  itself being harmful** — that run had `centered = false`, so it confounded the two.
  Once `centered` removes the β-independent term, unbounded ‖W‖ is not a problem and
  0.8636 is the best result in the repo on this architecture.
- ~~Does the basin-hopping onset differ for trained vs randomly-scaled weights of
  equal norm?~~ **Answered: yes, completely** — see the amended caveat in §3. There
  is no onset on trained weights out to ‖W₁‖ = 89.
- **New, from that work:** `T_free = 200` is not always sufficient at width 256.
  At β=0.1 one-sided it puts 8 of 20 trained snapshots below cos 0.99, four of
  them sign-inverted, non-monotonically in ‖W₁‖ — and `centered` clears 6 of the
  8. Raising `T_free` to 1600 fixed every point probed, but intermediate values
  are not safe to interpolate (one cell reads 0.92 at 400, 0.01 at 800, 0.999 at
  1600). Worth a per-epoch T-vs-2T drift check in `ep_train`, and worth asking
  whether the ±3-point epoch-to-epoch swings in `epoch_curves.csv` are this.
- Full buffer reuse in `_phasor_step` was measured at 1.44× on top of a functional
  implementation and rejected as not worth the state-representation change; the
  cheaper fusion above captured 1.7–3.2× instead. Revisit only if the remaining
  per-step state allocation shows up in a profile.
