# Three-View Mismatch Analysis: Static MLP vs Dirac Discrete vs Spiking ODE

> **Revision note (2026-05-14):** an earlier version of this document
> claimed the static MLP and the spiking-ODE views were inherently
> inconsistent. That was wrong. The static MLP and the
> ODE-via-`unrotate_solution` pair are mathematically equivalent and
> match in practice (verified below), which is exactly what
> `demos/network fashionmnist.ipynb` shows. The real mismatch is in the
> **3D Phase Dirac dispatch** of `PhasorDense` (and the analogous block
> in `ResonantSTFT`), which (a) operates in the rotating frame instead
> of the static phase frame, and (b) injects bias in a way that does
> not respect the spike-time encoding the rest of the SSM uses.
>
> **Status (2026-05-15):** issues (a), (b), (c), and §4.3 are all resolved.
> - (a) rotating-vs-static frame discrepancy: fixed in `src/network.jl` —
>   `_forward_3d_dirac` and `ResonantSTFT`'s 3D Phase dispatch now
>   apply `−conj(.)` derotation before `complex_to_angle`, so their
>   Phase output lives in the **static phase frame** matching the 2D
>   Phase MLP and the ODE-via-`unrotate_solution` pair. **Breaking
>   change**: 3D-Dirac-trained networks (`demos/long_range_demo.jl`,
>   `demos/run_ablation.jl`) need retraining because their codebook
>   prototypes were calibrated against the previous rotating-frame
>   output. See §4.1.
> - (b) bias sub-frame mismatch: fixed in `src/network.jl` —
>   `_forward_3d_dirac` and `ResonantSTFT`'s 3D Phase dispatch
>   apply the spike-time encoding `b_eff = b·exp(k_c·dt(b))` before
>   the kernel accumulator (Dirac semantics). See §3.
> - (c) `PhasorDense::CurrentCall` shape mismatch: resolved by adding
>   `sample_phases_at_periods` in `src/ssm.jl` and reconfiguring the
>   failing tests to use it on a `:potential` ODESolution. The
>   layer's `:phase` return type intentionally still returns the
>   dense per-save-point trajectory. See §4.2.
> - §4.3 ZOH bias paths: fixed in `src/network.jl` — `PhasorResonant`
>   and `ResonantSTFT`'s 3D Complex dispatches now multiply the bias
>   by the ZOH input gain `B = (exp(k·T)−1)/k` before the per-period
>   accumulator, matching the continuous-current semantic chosen for
>   these continuous-signal encoders. See §4.3.
>
> All four sub-issues are now closed. A new `zoh_bias_continuous_current_tests`
> regression test in `test/test_ssm.jl` pins the new ZOH bias semantics.

This note walks through what happens to a phase value as it flows
through each of the three computational views described in
`docs/ssm_rf_connections.md`, identifies where the Dirac discrete view
silently switches frames, and proposes a minimal fix that lets the
discrete and ODE outputs be compared on common ground.

The reproducers under `test/scratch/` produce the numbers quoted here:

- `three_view_compare.jl` — broad sweep across modes & L.
- `library_unrotate_check.jl` — confirms static ≡ ODE+library_unrotate.
- `dirac_frame_diag.jl` — isolates the 3D Dirac frame & bias issue.
- `bias_reconcile.jl` — shows the ODE bias semantics in detail.

## TL;DR (current state, post-§3 + §4.1 + §4.2 fixes)

| Pair                                                      | No bias                | With bias               |
| --------------------------------------------------------- | ---------------------- | ----------------------- |
| Static MLP  vs  ODE through `unrotate_solution`           | ≈ 0.02 (solver noise)  | ≈ 0.01 – 0.03           |
| **Live 3D Dirac dispatch  vs  Static MLP**                | **≈ 0.02 ✓**           | **≈ 0.01 – 0.03 ✓**     |
| **Live 3D Dirac dispatch  vs  ODE-via-`sample_phases_at_periods(unrotate=true)`, L=8, per period** | **ρ ≈ 0.97–1.00** | **ρ ≈ 0.90–1.00** (a few periods drop on near-zero variance) |

(Errors are mean modular arc distance in units of π. ω = 2π, T = 1,
default decay λ = −0.2, random `(C_in, C_out, B) = (4, 8, 3)` weights
and inputs. Numbers reproduce in `test/scratch/post_fix_check.jl`.)

For historical context, the **pre-fix** measurements showed:

| Pre-fix pair                                              | No bias                | With bias               |
| --------------------------------------------------------- | ---------------------- | ----------------------- |
| 3D Dirac (no unrotation, raw `b·G` bias)  vs  Static MLP  | ≈ 0.42 (frame off by π) | ≈ 0.49                  |
| 3D Dirac (no unrotation)  vs  `1 − θ_static`              | ≈ 0.02 ✓ (modular reflection) | ≈ 0.49 (bias breaks it) |
| 3D Dirac (with library unrotation, raw bias) vs Static MLP | ≈ 0.02 ✓             | ≈ 0.49 (bias still wrong) |

So:

- Static and ODE always agreed — `unrotate_solution` translates the
  rotating-frame ODE state back to the static phase frame.
- **The 3D Phase Dirac dispatch used to live in the rotating frame
  at integer sample times** (`θ_dirac ≡ 1 − θ_static (mod 2)` for
  ω = 2π, T = 1). §4.1 added the `−conj(.)` derotation inside the
  dispatch so it now matches the static frame natively.
- **Bias used to break even the simple frame-shift**, because the
  discrete bias entered as a raw complex offset while the signal
  carried the spike-time → period-end carrier rotation. §3 rewrote
  the bias as `b_eff = b·exp(k·dt(b))` so it lives in the same
  sub-frame as the signal; §4.1 then puts both into the static
  frame in one step.

## 1. The frame model, made explicit

A phase θ ∈ [−1, 1] (units of π) admits two distinct representations
that this codebase uses interchangeably:

1. **Static phase frame.** θ → `exp(+iπθ)` on the unit circle. The
   2D Phase dispatch of `PhasorDense` (and `Codebook`, and the
   `Codebook` similarity computation) lives here.
2. **Spike-time / rotating frame.** θ → spike at time
   `t_s = (θ/2 + 0.5)·T` within an oscillation period. A receiving
   neuron at eigenvalue `k = λ + iω` integrates the impulse for
   `dt = T·(0.5 − θ/2)` until the period boundary; the resulting
   complex potential is `exp(k·dt)`. This is what
   `causal_conv_dirac` and the spiking ODE produce internally.

For ω = 2π and T = 1, `exp(iω·dt) = exp(iπ − iπθ) = −exp(−iπθ)` — i.e.,
the spike-time encoding is the **conjugate of** the static-phase encoding,
multiplied by −1. Schematically:

    encoding(θ)        =  −exp(−iπθ) · exp(λ·dt(θ))           (Dirac)
    static_complex(θ)  =   exp(+iπθ)                          (2D Phase MLP)

The two are related by `encoding(θ) = −exp(λ·dt) · conj(static_complex(θ))`.

`unrotate_solution(z, t)` — the function the ODE dispatch applies to
its solution before extracting phase — is, at integer t = nT and
ω = 2π, exactly `−conj(z)`. That operation cancels out the
`−·conj(·)` introduced by the spike-time encoding, putting the
result back into the static phase frame.

This is why the static MLP and the ODE pipeline match: both
ultimately read out **static-frame** phases. The Dirac discrete
pipeline reads out **rotating-frame** phases, because it skips the
unrotation step.

## 2. Verification: static ≡ ODE-with-`unrotate_solution`

`test/scratch/library_unrotate_check.jl` builds an L=1 network,
encodes the same input phases through the 2D static MLP and through
`MakeSpiking → CurrentCall → ODE → unrotate_solution`, then compares
the resulting static-frame phases:

```
=== Static MLP  vs  ODE-with-LIBRARY-unrotate, L=1 ===
  no bias            : mean arc-err = 0.0219   max = 0.1779
  real-pos bias      : mean arc-err = 0.0132   max = 0.0782
  imag-pos bias      : mean arc-err = 0.0339   max = 0.4835
  mixed bias         : mean arc-err = 0.0172   max = 0.1546
```

Mean arc-err of ~0.02 is the ODE solver's discretization noise (Tsit5
at dt = 0.005 with t_window = 0.01). They agree to within numerical
tolerance — bias and all. This matches what
`demos/network fashionmnist.ipynb` shows when training the static
network and evaluating it through the spiking pipeline.

The relative magnitudes also work out: the spiking ODE dilutes BOTH
signal and bias by the same kernel-integration factor
`2·t_window·spk_scale ≈ 0.02`, so the bias-to-signal ratio is
preserved across the static → spiking transfer. This is why my
earlier "discrete bias is 50× larger" finding was misleading — that
ratio doesn't matter, only the within-mode ratio does, and the
within-mode ratios match.

## 3. The actual mismatch — the 3D Phase Dirac dispatch

`test/scratch/dirac_frame_diag.jl` runs the 3D Phase Dirac path
through the same `PhasorDense` layer at L = 1 (so the SSM kernel
collapses to identity) and applies the library unrotation to its
output before comparing to the static MLP:

```
=== 3D Dirac WITH library unrotation  vs  static MLP, L=1 ===
  no bias            : Dirac-unrotated vs static = 0.0195
  real-pos bias      : Dirac-unrotated vs static = 0.4940
  imag-pos bias      : Dirac-unrotated vs static = 0.0354
  mixed bias         : Dirac-unrotated vs static = 0.3696
```

- **No bias**: the unrotated 3D Dirac output matches the static MLP to
  within solver-noise tolerance (0.020). This confirms the frame
  diagnosis: the 3D Dirac signal path is correct *up to the missing
  unrotation step*.
- **Real-positive or mixed bias**: arc-err blows up to 0.5 — i.e.,
  half a full phase rotation. The bias and the signal are no longer
  in the same sub-frame after unrotation.

### Why bias breaks the otherwise-clean Dirac↔static correspondence

`_forward_3d_dirac` (`src/network.jl:402-434`) does:

```julia
Z = causal_conv_dirac(x, params.weight, λ, ω, 1f0)              # signal
if a.use_bias
    bias_val = params.bias_real .+ 1f0im .* params.bias_imag
    G = bias_kernel_accumulation(λ, ω, 1f0, L)
    Z = Z .+ reshape(bias_val, :, 1, 1) .* reshape(G, ..., 1)   # bias
end
return complex_to_angle(Z), state                                # frame: rotating
```

The signal entry `causal_conv_dirac` carries the encoding
`exp(k·dt(θ_in))`, which includes the spike-time → period-end carrier
rotation `exp(iω·dt(θ_in))`. The bias entry just multiplies the raw
complex offset `b` by the accumulator `G[c, n]` — it has **no**
spike-time rotation factor. When you then apply library unrotation
(`−conj(·)` at integer T) to the whole thing:

- Signal: `−conj(−exp(−iπθ_in)·exp(λ·dt))` = `exp(+iπθ_in)·exp(λ·dt)` →
  static-frame phase ≈ θ_in. Correct.
- Bias: `−conj(b·G)` = `−conj(b)·conj(G)` →
  static-frame phase ≈ −arg(b). For a real-positive b this comes out
  to phase 1 (negative real), not phase 0. **Wrong by π.**

In the ODE, the bias **does** pick up `exp(iω·dt(b))` because
`bias_current` injects it as a periodic spike at
`t_s = phase_to_time(arg(b))` — that's the same spike-time encoding
the input signal uses. So the ODE bias and signal are in the same
sub-frame, and library unrotation cancels them both correctly.

### A one-line bias fix that closes the gap

Replace the discrete bias accumulator's `b` by the spike-time-rotated
`b_eff`:

```julia
# inside _forward_3d_dirac, replacing the existing bias block
bphase = angle.(bias_val) ./ Float32(π)
dt_b   = 1f0 .* (0.5f0 .- bphase ./ 2f0)
k      = ComplexF32.(λ .+ 1im .* ω)
b_eff  = abs.(bias_val) .* exp.(k .* dt_b)         # spike-time encoding of b
G      = bias_kernel_accumulation(λ, ω, 1f0, L)
Z      = Z .+ reshape(b_eff, :, 1, 1) .* reshape(G, ..., 1)
```

`test/scratch/dirac_frame_diag.jl` runs this fix end-to-end:

```
=== Proposed fix: spike-time-rotate b first ===
  no bias            : FIXED Dirac vs static = 0.0195
  real-pos bias      : FIXED Dirac vs static = 0.0124
  imag-pos bias      : FIXED Dirac vs static = 0.0301
  mixed bias         : FIXED Dirac vs static = 0.0169

=== FIXED Dirac vs library-unrotated ODE per-period at L=8 ===
  no bias             per-period ρ = 0.974  0.991  0.983  0.999  1.000  1.000  1.000  1.000
  no bias             mean arc-err = 0.0289
  mixed bias          per-period ρ = 0.899  0.987  0.983  0.998  0.999  0.998  0.998  1.000
  mixed bias          mean arc-err = 0.0222
```

Both at L = 1 and across periods of an L = 8 sequence, the fixed
discrete path now agrees with the library-unrotated ODE to within
solver-noise tolerance, with and without bias.

This change does **not** affect anything for layers that don't use
bias — the early-out path is identical to today's code.

> **Note (post-§4.1):** with §4.1 also applied (the layer derotates
> internally before `complex_to_angle`), users no longer need to
> apply `unrotate_solution` themselves to compare the 3D Dirac
> output to the static MLP — the layer's Phase output is already in
> the static phase frame.

## 4. Resolved issues

§3 (bias sub-frame, Dirac), §4.1 (rotating-vs-static frame),
§4.2 (`PhasorDense::CurrentCall` shape), and §4.3 (ZOH bias paths
in `PhasorResonant` / `ResonantSTFT` Complex 3D) are all fixed.
This section retains the diagnosis, applied fix, and verification
notes for each.

### 4.1 The rotating-vs-static frame discrepancy (issue "a") — RESOLVED

**Resolution status (2026-05-15):** resolved by inserting
`Z = -conj.(Z)` before `complex_to_angle` in
`PhasorDense._forward_3d_dirac` and in `ResonantSTFT`'s 3D Phase
dispatch. The Phase output of these dispatches now lives in the
**static phase frame**, matching the 2D Phase MLP and the
ODE-via-`unrotate_solution` pair.

**Breaking change.** Networks trained end-to-end in the 3D Phase
Dirac dispatch (notably `demos/long_range_demo.jl` and
`demos/run_ablation.jl`) had codebook prototypes calibrated against
the prior rotating-frame output. After §4.1 the layer emits
static-frame phases, so those trained codebooks no longer match the
layer's output and accuracy will drop until retrained. The user has
explicitly accepted this — those demos are prototypes — and the
test surface caught no other consumer of the rotating-frame output.

**What was wrong before.** `_forward_3d_dirac` (and the analogous
Phase 3D dispatch in `ResonantSTFT`) returned `complex_to_angle(Z)`
directly, without applying the `unrotate_solution`-style `-conj(·)`
that the ODE pipeline applies via `unrotate_solution`. So the Phase
output of the 3D Dirac path was in the **rotating frame at the
sample boundary**. For ω = 2π and T = 1, the relationship between
the two frames was

    θ_dirac_rot  ≡  1 − θ_static  (mod 2)

i.e., a modular reflection about ½. The §3 bias fix brought the
bias sub-frame into alignment with the signal sub-frame, but both
still lived in the rotating frame, so they were *jointly* off by
the same modular reflection from the static MLP / ODE convention.

**The applied fix.**

```julia
# in _forward_3d_dirac, after the bias accumulator block:
Z = -conj.(Z)               # ω·T = 2π·integer specialization
# then:
if a.activation === normalize_to_unit_circle
    return complex_to_angle(Z), state
else
    Y = a.activation(Z)
    return complex_to_angle(Y), state
end
```

`-conj(.)` is exactly what `unrotate_solution` would compute: at
integer `t = n·T` with shared `ω = 2π`,
`phase_to_potential(0, n·T) · conj(z) = exp(iπ) · conj(z) = -conj(z)`.
For full generality (if `T` or `ω` ever stop matching the canonical
configuration), the right call is the more general
`unrotate_solution`-style multiplication by
`phase_to_potential(0, n·T)` per sample n. For PhasorNetworks'
default single-shared-ω = 2π and T = 1, `-conj(·)` is exact.

`-conj(.)` commutes with the magnitude-only activations in the
codebase (`identity`, `normalize_to_unit_circle`,
`soft_normalize_to_unit_circle`), so applying it before vs after
the activation gives the same final phase. We apply it before,
mirroring `unrotate_solution`'s convention of derotating the raw
potential.

**What this enables.**

- 2D Phase MLP, 3D Phase Dirac, and the spiking ODE all produce
  comparable static-frame phases — a single trained `(W, b, λ)`
  has the same downstream meaning across all three modes.
- `sample_phases_at_periods(sol, L, spk_args; unrotate=true)`
  produces phases that match the live 3D Phase Dirac output to
  within solver tolerance (per `test/scratch/post_fix_check.jl`).
- The `ssm_spiking_correlation_tests` "spiking ODE correlates with
  Dirac" assertion now compares like-with-like in the static frame
  (was rotating-frame on both sides, with the static-vs-rotating
  shift quietly absorbed because both Dirac and ODE happened to
  agree in their joint rotating frame); empirically still
  ρ ≈ 0.97–1.00.

**Verification.** `test/scratch/post_fix_check.jl` reports
mean arc-err 0.012–0.030 between the live 3D Phase Dirac output
and the 2D Phase static MLP across no-bias / real-pos / imag-pos
/ mixed bias configurations, **without** applying any manual
unrotation step (the layer does it). Full SSM test suite stays at
the post-§4.2 baseline of 557 passed / 2 failed / 0 errored
(the 2 failures are pre-existing — see §4.2).

### 4.2 The `PhasorDense::CurrentCall` shape mismatch (issue "c") — RESOLVED

**Resolution status (2026-05-15):** resolved by adding
[`sample_phases_at_periods`](@ref) in `src/ssm.jl` and rewriting the
two failing test blocks to use it. The layer's `:phase` return type
is **deliberately unchanged** — see "Why the layer wasn't changed"
below.

**What it was.** `PhasorDense::(::CurrentCall, …)` for
`return_type = SolutionType(:phase)` returns a `Vector{Matrix{Phase}}`
of length ≈ `L·T/dt + 1` (one matrix per ODE save point), instead of
an `(C_out, L, B)` Phase tensor. Specifically:

```julia
# src/network.jl ≈ 484
u  = unrotate_solution(sol.u, sol.t, …)   # Vector{Matrix{Complex}}
y  = a.activation.(u)                      # Vector{Matrix{Complex}}
phase = complex_to_angle.(y)               # Vector{Matrix{Phase}}
return phase, state
```

`a.activation.(u)` and `complex_to_angle.(u)` broadcast over the
outer Vector, so the per-save-point structure is preserved.
`ssm_spiking_correlation_tests` and the
"PhasorDense SSM SpikingCall/CurrentCall dispatch" assertions in
`ssm_spiking_dispatch_tests` expected an `(C_out, L, B)` tensor and
errored / failed at the shape comparison.

**Why the layer wasn't changed.** The dense per-save-point trajectory
is *useful*: it carries sub-period transients and kernel-window
interactions that period-boundary sampling at the layer would
discard. Different downstream consumers may want different sampling
cadences, so constraining the ODE output to boundary edges at the
layer level is wrong. `:potential` already exposes the raw
`ODESolution`, which any consumer can interpolate with `sol(t)`.

The right place to do period-boundary sampling is in a library
helper, not in the layer.

**What changed instead.**

1. Added `sample_phases_at_periods(sol, L, spk_args; activation,
   unrotate, offset)` in `src/ssm.jl`, sibling of
   `ssm_extract_phases` and `reconstruct_from_current`. Behaviour:
   - Builds `sample_ts = Float32[n·T + offset for n in 1:L]`.
   - `samples = [sol(t) for t in sample_ts]`.
   - If `unrotate=true`, applies `unrotate_solution` to put samples
     in the static phase frame (matches the static MLP / library
     ODE convention). Default `false` keeps them in the rotating
     frame, matching the 3D Dirac dispatch's native frame.
   - Stacks into `(C_out, L, B)` (or `(C_out, L)`).
   - Applies `activation` then `complex_to_angle`.
   - Returns a Phase tensor.
2. Exported from `src/PhasorNetworks.jl`.
3. The two failing test blocks (`ssm_spiking_dispatch_tests`'s
   "PhasorDense SSM SpikingCall dispatch" and "PhasorDense SSM
   CurrentCall dispatch", and `ssm_spiking_correlation_tests`) now
   request `return_type = :potential`, capture the `ODESolution`,
   and call `sample_phases_at_periods` to obtain the
   `(C_out, L, B)` per-period phases. The assertions on shape /
   eltype / finiteness / correlation are otherwise unchanged.
4. The `PhasorDense` docstring's `return_type` field gained a
   one-line note explaining that `:phase` is intentionally the
   dense trajectory and pointing at the helper.

**Verification.** Before the bias fix (§3) and this change, the SSM
test block reported 548 passed / 7 failed / 4 errored. After both
changes: 557 passed / 2 failed / 0 errored. The two remaining
failures (`PhasorDense SSM potential return type` line 849, and
`CUDA Specific Tests`) are pre-existing and unrelated:

- `@test !(sol isa AbstractArray)` (line 849) is a long-standing
  test bug: SciML's `ODESolution` *does* subtype `AbstractArray`,
  so the assertion was always wrong; it's mis-checking
  "is this the raw ODE solution object" via `isa AbstractArray`
  instead of e.g. `isa SciMLBase.AbstractTimeseriesSolution` or
  testing for callability. Out of scope here.
- `CUDA Specific Tests` errors at load time with
  `ArgumentError: Package Adapt not found in current path` —
  environment / dependency issue, not a test logic bug.

**Note for `PhasorConv`.** Its `(::CurrentCall, …)` dispatch has the
same `Vector{Matrix{Phase}}` pattern at `src/network.jl ≈ 1046`. No
test currently asserts the per-period shape against it, so it's
unchanged here. The new helper would work for it without modification
if a future test wants `(C_out, …, L, B)` Phase output.

### 4.3 The ZOH bias paths in `PhasorResonant` / `ResonantSTFT` — RESOLVED

**Resolution status (2026-05-15):** resolved by multiplying the bias
by the ZOH input gain `B = (exp(k·T) − 1) / k` before the per-period
kernel accumulator in both `PhasorResonant`'s 3D Complex dispatch
and `ResonantSTFT`'s 3D Complex dispatch. Semantic chosen by the
user: **bias as a constant complex current** — these layers are
encoders of continuous complex signals, and the bias is naturally a
held-constant additive current applied to the input current at every
period.

**What was wrong before.** Both 3D Complex dispatches accumulated
bias as `b · G[c, m]`, missing the same `B` factor that
`phasor_kernel` (`src/kernels.jl:70-78`) already applies to the
signal. Under ZOH the recurrence is

    z[n+1]  =  A · z[n]  +  B · H[n]

so a constant additive bias `b` (i.e., `H[n] += b` per step) gets
multiplied by `B` before accumulation. The correct closed-form
contribution at sample `m` is

    Z_bias[c, m]  =  b · B_c · G[c, m]
                  =  b · (A_c^(m+1) − 1) / k_c

(matches the continuous-time ODE bias integral
`b · (exp(k·t) − 1)/k` evaluated at `t = (m+1)·T`).

The pre-fix `b · G` accumulator was off by `~|k|/|A−1| ≈ 50×` in
magnitude and `~arg(B) ≈ 0.01·π` in phase at default decay
(`λ = −0.2, ω = 2π, T = 1`). `use_bias = false` is the default for
these layers, no test or live demo exercised `use_bias = true`
beyond shape/finiteness, so this was latent — but the bias
gradients these layers received were wrong by the missing factor
whenever a user enabled bias.

**The applied fix.**

```julia
# in both 3D Complex dispatches' use_bias branch:
bias_val = ps.bias_real .+ 1.0f0im .* ps.bias_imag                   # (C,)
k_c   = ComplexF32.(λ .+ 1im .* ω)                                   # (C,)
B_gain = (exp.(k_c .* 1f0) .- 1f0) ./ k_c                            # (C,) ZOH input gain
b_eff = B_gain .* bias_val                                           # (C,)
G = bias_kernel_accumulation(λ, ω, 1f0, L)                           # (C, L)
Z = Z .+ reshape(b_eff, :, 1, 1) .* reshape(G, size(G, 1), size(G, 2), 1)
```

`B_gain` is computed inline (the same expression appears in
`phasor_kernel`'s signal kernel construction, but extracting it as a
shared helper isn't worth the API surface for two call sites). The
formula has a removable singularity at `k = 0`, but `λ < 0` is
guaranteed by initialization (both layers compute
`λ = -exp(log_neg_lambda)`), so `|k| > 0` and no guard is needed.

**Contrast with the §3 spike-time bias semantics.** The `_forward_3d_dirac`
path (`PhasorDense` and `ResonantSTFT`'s 3D Phase dispatch) treats
bias as a phantom spike with timing `t_s = phase_to_time(arg(b))`,
accumulating `|b| · exp(k·dt(b)) · G[c, m]`. That's correct for the
spike-encoded input domain those dispatches consume. The 3D Complex
dispatches consume continuous signals, so they use the ZOH-current
semantics here. The two semantic choices are independent — neither
fix is wrong; they apply to different input domains.

**Verification.** New `zoh_bias_continuous_current_tests` in
`test/test_ssm.jl` bias-only-probes both layers (zero weights,
nonzero `b = 0.3 + 0.4i`) and asserts that the layer's complex
output equals the analytical `b · (A^(m+1) − 1)/k` reference within
solver tolerance (`maximum(abs.(y .- ref)) < 1f-5` for ResonantSTFT,
`max arc-err < 1f-4` for PhasorResonant's Phase output). The full
SSM test suite goes from 557 passed / 2 failed / 0 errored
(post-§4.1) to **561 passed / 2 failed / 0 errored** (post-§4.3),
gaining the 4 new asserts without regressions.

## 5. What this analysis withdraws from the previous version

- The claim that the static MLP and the spiking ODE are inherently
  inconsistent. They aren't, because `unrotate_solution` reconciles
  the two frames at the ODE output. The user's
  `network fashionmnist.ipynb` correctly observes this.
- The "discrete bias is ~50× too large" framing in absolute
  magnitude. Both signal and bias are diluted by the same
  `2·t_window·spk_scale` factor in the ODE pipeline, so the
  bias-to-signal ratio is preserved across static → spiking, and
  that ratio is what determines network behavior.

What this analysis still claims:

- The 3D Phase Dirac dispatch silently uses the rotating frame
  rather than the static phase frame.
- Within the 3D Dirac dispatch, the bias accumulator
  (`bias_kernel_accumulation` + raw `b`) and the signal path
  (`causal_conv_dirac`'s spike-time encoding) live in different
  sub-frames, so even after applying the library unrotation the bias
  ends up rotated by π relative to the signal.
- `PhasorDense::(::CurrentCall, :phase)` returns the wrong shape for
  comparison with the discrete output.

## 6. Reproducer index

| File                                          | Purpose                                                       |
| --------------------------------------------- | ------------------------------------------------------------- |
| `test/scratch/three_view_compare.jl`         | Sweep across three modes, ±bias, several L. (Note: uses a non-library unrotation; see §1.) |
| `test/scratch/library_unrotate_check.jl`     | Confirms static MLP ≡ ODE through `unrotate_solution`.        |
| `test/scratch/dirac_frame_diag.jl`           | Isolates the 3D Dirac frame issue and validates the bias fix in isolation. |
| `test/scratch/post_fix_check.jl`             | End-to-end test against the **modified** `_forward_3d_dirac`: live layer + library unrotation vs static MLP, vs ODE per period at L=8, ResonantSTFT smoke test. |
| `test/scratch/bias_reconcile.jl`             | Detailed bias semantics; ODE vs discrete in absolute terms.   |
| `test/scratch/static_vs_dirac_diag.jl`       | Algebraic relation between Dirac encoding and static encoding. |
