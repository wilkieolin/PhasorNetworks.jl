# Sparse experts on the wavesheet: read → gate → re-bind

**Design note.** The wavesheet (`src/wave.jl`, [`rf_wave_network_implementation.md`](rf_wave_network_implementation.md))
gives us a substrate that *propagates* information across a 2-D toroid as waves,
with a closed-form dispersion relation ([`wave_dispersion_derivation.md`](wave_dispersion_derivation.md))
that tells us how a wave's speed, growth, selected wavelength, and read-out phase
evolve as it travels. This note asks the next question: can we make the sheet
*actively, sparsely transform* information as it moves — not with a stack of
discrete downstream layers, but with **local "expert" modules that read a patch of
the sheet, gate themselves on or off, and re-bind their output into the passing
wave**?

The short answer from a literature sweep (24 primary/secondary sources, claims
adversarially verified 3-voter): **yes, the module is definable, buildable from
existing well-validated tooling, and lives almost entirely inside the phasor/FHRR/VSA
+ oscillator substrate we already use.** Every sub-mechanism has a primary-source
precedent. The load-bearing honesty is twofold: (1) *no* source demonstrates the
*integrated* module end-to-end, so feasibility is by composition, not by direct
evidence; and (2) the strongest neuroscience justification (Numenta / Thousand
Brains) **fails verification** and must not be leaned on.

Status: design note, no code yet. Companion to the wave implementation and
dispersion docs above. **Training method is the discrete phase-SSM (parallel
scan / BPTT), not the continuous-ODE adjoint** — §5 derives what the SSM must be
extended to cover for the expert module. The concrete first experiment is in §6.

---

## 1. The reframe: an expert is a bind that rides the carrier

The wavesheet already carries information as **relative phase on a shared carrier
`ω`** (the per-channel-ω rule in `CLAUDE.md`). Two facts make a "modulate the
passing wave" module natural rather than bolted-on:

- **Binding is a phase operation.** In the Fourier Holographic Reduced
  Representation (FHRR) — unit-magnitude complex/phasor vectors, exactly our
  domain — *binding is element-wise complex multiplication*, i.e. **phase
  addition**. To "attach a transformation `T` to a wave `z`" is `z ↦ z ⊙ φ_T`,
  a per-site phase rotation. This is already the kind of thing a wavesheet site
  does; an expert just applies a *learned, gated* rotation keyed to a patch.

- **The carrier is a shared clock.** Because every site rotates at the same `ω`,
  an expert reading patch `P` at time `t` and writing back at `t+Δ` can express
  its transport delay as the same `e^{-iωΔ}` phase factor the sheet already uses
  for conduction delay (§2.1 of the implementation doc). Read, transform, and
  write-back all stay inside the linear complex algebra — no separate control
  path, no DDE.

So an **expert module** is the triple:

```
read(P)  →  gate(g)  →  z_P ↦ z_P ⊙ (g · φ_expert)      (re-bind into the wave)
```

where `read(P)` samples the complex field on a patch, `gate(g) ∈ [0,1]` (soft) or
`{0,1}` (hard, top-K) decides whether this expert fires, and `φ_expert` is the
learned bind code the expert stamps onto the wave. Multiple experts tile the
toroid; only a sparse few fire per wave-crossing — the "columns" of a
Mountcastle sheet, or the active experts of a sparse MoE.

---

## 2. The gate: sparsely-gated Mixture-of-Experts, de-risked

The routing/gating stage maps cleanly onto modern MoE, and the two classic
trainability failures already have published fixes:

- **Fine-grained expert segmentation** (DeepSeekMoE). Splitting `N` experts into
  `mN` smaller ones and activating `mK` increases combinatorial routing
  flexibility (`C(mN,mK) ≫ C(N,K)`) and pushes each expert toward
  *non-overlapping, focused* knowledge — precisely the goal that selectively-fired
  modules on a *shared substrate* acquire distinct functions. For us, a
  fine-grained expert = a small phase-transform keyed to a small patch.

- **Auxiliary-loss-free load balancing** (DeepSeek, adopted in DeepSeek-V3). A
  load-balancing *auxiliary loss* "introduces non-negligible interference
  gradients and impairs model performance." The fix is a **per-expert bias added
  to the routing scores *before* top-K selection**, nudged by each expert's
  recent load. It balances utilization *without* polluting the gate's gradient.
  This is our recommended routing recipe.

- **Differentiable hard gate.** Gumbel-Softmax + straight-through gives a hard
  one-hot forward / relaxed-gradient backward, so "which expert fires" is
  trainable through the discrete choice.

> **Adaptation needed.** DeepSeek's "load" is a token count. Our load is
> **phase/patch occupancy on a 2-D toroid** — how often a spatial expert's patch
> carries a wave worth transforming. The per-expert-bias mechanism should port,
> but the load signal must be redefined spatially (open question, §7).

---

## 3. The bind/unbind: VSA + resonator networks, in our own substrate

This is the strongest part of the case, because it is native to the framework:

- **Binding = the equivariant transform.** VSA binding acts as *the equivariant
  operation for geometric transformations*; a scene is a sum of bound products.
  "Attach a transform" is a bind, "detach" is an unbind — both phase arithmetic
  in FHRR.

- **Resonator networks factor what got bound.** When a wave carries *several*
  superposed bound transforms, a resonator network recovers the individual
  factors by "searching in superposition" — interleaving VSA multiplication with
  codebook pattern-completion — reported "dramatically more effective than all
  alternative approaches" (ALS / gradient descent). This is our tractable *decode*
  step, and it reuses the `Codebook` / similarity machinery already in the repo.

- **It runs in spiking phasor hardware.** Complex-valued resonator networks have
  been implemented as a **multi-compartment spiking phasor neuron model on
  low-power neuromorphic hardware** — matching the wavesheet's Sakaguchi–Kuramoto
  `transmit=:spike` mode directly. The bind/factor ops are not just abstractly
  compatible; they have been physically realized in our exact regime.

- **FHRR + deep nets is a direct precedent** — and is authored by this repo's own
  git user (see References). It is the closest existing scaffold for exactly this
  framework.

> **Hard capacity ceiling (real limit, not a refutation).** Resonators have a
> finite *operational capacity*: beyond a search-space threshold, factorization
> fails. This bounds **how many bound transforms a single wave can carry and
> still be decoded**. It sets a budget on expert-stamps-per-wave and is Open
> Question §7.2.

---

## 4. Why the wave substrate can host this at all

Two ML results establish that waves *carry and transform* information and that
such networks *train*:

- **Wave-RNN** (ICLR 2024): a simple RNN whose hidden state exhibits traveling
  waves "invertibly storing a short-term memory of sequential stimuli," matching
  gated RNNs with fewer parameters.
- **Neural Wave Machine** (ICML 2023): a "locally coupled oscillatory recurrent
  neural network" whose traveling waves "encode observed transformations." Both
  train via **standard backprop**.

And two results show attention/gating layered on a *local lattice* without
abandoning local dynamics — the exact structural pattern the expert needs:

- **AKOrN** (Artificial Kuramoto Oscillatory Neurons, ICLR 2025 oral):
  generalized Kuramoto updates "bind neurons together through their
  synchronization dynamics" and **compose with convolutional or attentive
  connectivity** — phase synchronization *as* a gating/binding primitive that
  interfaces with attention rather than replacing it.
- **ViTCA** (NeurIPS 2022): a "spatially-localized yet globally-organized
  self-attention scheme" layered on local cellular-automata update rules —
  read-patch attention coexisting with a local update law.

The toroidal re-entry of a wave (it loops around and is re-processed) has a
1-D deep-learning analog in **recurrent-depth / looped transformers**, which
iterate one recurrent block to arbitrary latent depth through a fixed weight set.

*(Bridging lens, medium confidence: self-attention itself admits a VSA reading —
Q/K as role spaces, V as fillers, attention weights as soft unbinding, residual
as bundling. Verified but rests on a single non-peer-reviewed preprint; treat as
interpretive, not established.)*

---

## 5. Training method: the discrete SSM, **not** the ODE adjoint

The natural first instinct is to train the expert-augmented sheet the way the
sheet's Tier-2 continuous mode trains — `CurrentCall` → `oscillator_bank` →
`BacksolveAdjoint`+`ZygoteVJP`. **We should not.** The discrete phase-SSM we
already derived (`K[n] = Aⁿ·B`, `M(q) = A + g·Ŵ(q)`) is the more stable and more
parallelizable training substrate, and it should be the reference path for *both*
the sheet and the expert module. The ODE mode stays, but as a *verification and
continuous-spike-input* path, not the trainer. This section says what the SSM
must be *extended* to cover before it can train the module.

### 5.1 Why discrete-SSM beats ODE-adjoint here

- **Stability is a spectral-radius knob, not a solver gamble.** The discrete step
  multiplier is `M(q) = A + g·Ŵ(q)`, and `dispersion()` already returns
  `spectral_radius = max_q |M(q)|`. Training near `|M(q)| ≲ 1` gives a
  numerically bounded forward *and* backward by construction. `BacksolveAdjoint`
  instead reconstructs the trajectory in reverse time and accrues reconstruction
  error (and stiffness sensitivity) that we would have to fight — a fragility the
  wavesheet's own Tier-2 already exposed (needs `ComponentArray`, `saveat` not
  `sol(t)`, `ignore_derivatives` around solver bookkeeping).
- **Parallelism.** The linear sheet is LTI and diagonal per spatial mode, so its
  whole time evolution is either a causal convolution with the phasor kernel
  `K(q)[n] = M(q)ⁿ·B` (FFT in time *and* space) or an **associative parallel
  scan** — `O(log L)` depth instead of the ODE's inherently sequential
  time-stepping. Gradients flow through the same parallel structure.
- **It reuses code we already have.** `phasor_kernel`, `causal_conv[_fft]`,
  `normalize_to_unit_circle`, and the `M(q)` machinery in `dispersion()` are all
  present. The gap is the *scan/selective* generalization below, not new physics.

### 5.2 What the SSM already covers — the linear sheet

In `transmit=:potential` the per-step update `z[n+1] = A·z[n] + g·(W⊛z[n]) + B·I[n]`
is exactly `z[n+1] = M·z[n] + B·I[n]` with `M` diagonalized by the spatial FFT
into scalar `M(q)`. Per mode this is a textbook diagonal linear SSM. Two
non-recurrent forms train it in parallel:

1. **Time-convolution:** `ẑ_q[n] = Σ_m M(q)ᵐ B I_q[n−m]` — the existing
   `causal_conv` idea, now applied *per spatial mode* (FFT the drive in space,
   convolve in time with the geometric kernel, inverse-FFT). Fully parallel.
2. **Associative scan:** carry `(M(q), B·I_q[n])`, combine with
   `(M₂,b₂)∘(M₁,b₁) = (M₂M₁, M₂b₁+b₂)`. `O(log L)` depth, standard S5-style.

Today `_wave_rollout` is a sequential `Zygote.Buffer` loop. **Extension task #1:
add a parallel-scan / FFT-in-time forward for the linear segment.** This alone
makes the *sheet* train faster and more stably with zero architectural change.

### 5.3 The bind is a diagonal SSM insert — stay in the algebra

The expert re-bind `z_P ↦ z_P ⊙ (g·φ_expert)` is a **per-site diagonal
multiplication**: at the fired sites it left-multiplies the state by a diagonal
operator `D_expert = diag(g·φ_expert)`. That is the *same kind of object* as the
SSM transition `M` — an expert step is just an input-/state-dependent diagonal
transition spliced into the recurrence. So the module never leaves the SSM
algebra; it makes the transition **selective** (input-dependent), exactly the
generalization Mamba makes to diagonal SSMs. The question is only *what the
selection depends on*, and that determines whether parallelism survives.

**Does the per-cycle z-step reproduce the *sub-cycle* effect of a bind?** Yes,
exactly, for the diagonal part — and this is why the insert is legitimate at
per-cycle resolution. A bind is a phasor multiply, which by the delay↔phase
identity (`e^{-iωτ}`) *is* a sub-cycle temporal shift; the sub-cycle detail lives
in `arg(z)`. Within a cycle, free propagation is the scalar `e^{kt}`, so a
diagonal bind `d` inserted at any sub-cycle time `t_b` gives
`e^{k(T−t_b)}·d·e^{k·t_b}·z = d·e^{kT}·z = d·A·z` — the `t_b` cancels because
diagonal `D_expert` and diagonal `A = exp(k·T)` commute. So the sub-cycle
placement of a diagonal bind is a gauge the per-cycle step absorbs, and
`z[n+1] = D·A·z[n] + …` is the exact forward step (no sub-cycle potential). The
**one caveat**: the coupling `g·(W ⊛ z)` is *off-diagonal*, so `D` does not commute
with it — a bind that fires mid-cycle *and* must couple to neighbours within the
same cycle does depend on `t_b`. The discrete sheet already operator-splits
(`M(q)=A+g·Ŵ(q)`, coupling once per cycle from the start-of-cycle state), so a
boundary-ordered bind adds no error beyond the split; if genuine mid-cycle
bind→couple is ever needed, use Strang splitting (half-couple → bind →
half-couple), still per-cycle in the z-domain — not sub-cycle ODE integration.
Consistent with §5.5: only threshold+reset truly needs sub-cycle resolution.

### 5.4 The crux: input-conditioned vs state-conditioned experts

A selective SSM stays parallel-scan-trainable **iff its per-step coefficients can
be computed before the scan.** Mamba achieves this because its selection reads the
*external input* `x[n]` (known ahead of time), so all `A[n], B[n]` are
precomputed, then one parallel scan runs. Our expert has two regimes:

- **Input-conditioned expert (parallel, "Mamba-like").** The gate/bind is a
  function of the *drive* or a slowly-varying context — not the instantaneous
  propagating wave. Then every `D_expert[n]` is precomputable, so it can fold into
  the parallel scan. **One basis subtlety, load-bearing for the implementation:**
  a *spatially-local* patch bind `D = diag(g·φ)` is diagonal in the **site** basis,
  while the transport `M(q) = A + g·Ŵ(q)` is diagonal in the **Fourier** basis —
  they do *not* co-diagonalise, so a bind spliced into the *state transition*
  (`z[n+1] = D[n]·M·z[n] + …`) is **not** a scalar per-mode scan even when
  input-conditioned (the composed transition is dense; it needs BPTT or the chunk
  of regime 2). The scan stays scalar in exactly two cases: (i) the bind is
  spatially **global** (`D = c·I`, co-diagonal with `M`), or (ii) — the practical
  one — the selectivity modulates the **input injection** rather than the state:
  `z[n+1] = M·z[n] + D[n]·drive[n]`. Then the transition is just `M`, the modified
  drive `D[n]·drive[n]` is precomputed, and the *existing* `_wave_rollout_scan`
  runs unchanged. This is the **shipped** input-conditioned path (§5.8 #3): the
  expert reads the input patch, gates, and stamps its bind onto the injected
  wavefront, which the linear sheet then propagates. It is a genuine restriction
  (the expert modulates what is injected, not the wave already travelling) but it
  keeps the clean one-scan story and is the right *first* target.

- **State-conditioned expert (the ambitious version).** The gate reads the
  *passing wave itself* (`read(P)` of the live state) — the literal
  "modify a wave as it passes by." Now `D_expert[n] = f(z[n])` couples
  coefficient→state→coefficient and a single global scan no longer applies. Three
  ways to keep most of the parallelism and still avoid the ODE adjoint:

  1. **Event-based operator splitting (natural fit for *sparse* experts).**
     Propagate the linear sheet in parallel (scan/FFT) between firings; only at
     the sparse firing events do we read → gate → apply `D_expert` sequentially.
     If experts fire rarely — which is the entire point of *sparse* MoE — there
     are few sequential events separating long parallel linear segments. Cost ≈
     (#events) sequential steps, not `L`.
  2. **Chunked scan (Mamba-2 / chunked-linear-attention style).** Freeze the gate
     over a short chunk (compute it from the chunk's entry state), run a parallel
     linear scan within the chunk, update the gate sequentially *between* chunks.
     Trades exactness for `O(L/chunk)` sequential depth; the dispersion
     linearization in [`wave_dispersion_derivation.md`](wave_dispersion_derivation.md)
     tells us over what horizon "freeze the gate" is a good approximation.
  3. **Equilibrium / DEQ settling.** If an expert settles to a fixed point,
     train it with implicit-function-theorem gradients — which ties directly to
     the `AttractorPhasorSSM` + EP/hEP machinery already in the repo, again
     side-stepping the ODE adjoint.

### 5.5 The spike nonlinearity is a *per-cycle* operation — no sub-cycle potentials needed

A tempting worry is that `transmit=:spike`'s `normalize_to_unit_circle` forces us
back toward fine-grained (ODE-like) time resolution. It does not, and the reason
is the same trick that lets `PhasorDense` train discretely: **we track phase per
neuron per cycle, not the potential per sub-cycle timestep.** Two facts in the
code make this concrete:

- In `_build_coupling`, `T = spk_args.t_period` and `A_step = exp(k·T)`, so **one
  discrete step advances one whole carrier cycle.** `_wave_rollout` iterates `L`
  cycles, not `L` sub-cycle solver steps.
- `PhasorDense` is identical in spirit: `phasor_kernel` uses `Δt = 1` (period
  units), and `causal_conv_dirac` folds the *sub-cycle* spike timing into a
  one-shot phase→`dt = T·(0.5 − θ/2)` encoding rather than by stepping through the
  cycle.

So the trainable path never integrates a potential at an arbitrary sub-cycle time.
The state is the per-cycle phasor; a neuron's sub-cycle spike time *is* its phase.
This is a *stronger* statement than §5.1: the discrete SSM is not merely more
stable/parallel than the ODE — it runs at the semantically correct rate (one
sample per carrier cycle = one spike opportunity per neuron), and the ODE's
sub-cycle `dt` is a verifier artifact, not a training requirement.

What remains is a *linearity* residue, not a *resolution* one. `:spike` emits a
unit-magnitude spike each cycle (`src = z/|z|`), so
`z[n+1] = A·z[n] + g·(W ⊛ z[n]/|z[n]|) + drive` still carries one nonlinearity
per cycle-step and is therefore not a single pure-linear scan. But being
*per-cycle* (not sub-cycle) makes it cheap to handle, in order of fidelity:

- **BPTT over the per-cycle recurrence.** Reverse-mode AD over `L` cycle-steps —
  no ODE solver, no reverse-time reconstruction, batch-parallel. This is what
  Tier-1 already does and should be the default; the step count is `L`, not
  `L × steps-per-cycle`.
- **Phase-only ⇒ Kuramoto map.** Tracking phase alone (`|z|≡1`) turns the
  recurrence into the pure Sakaguchi–Kuramoto per-cycle map
  `θ[n+1] = angle(A·e^{iθ} + g·(W ⊛ e^{iθ}) + drive)` — the exact regime the
  dispersion linearization (dispersion doc §3) describes.
- **Chunk to recover a linear scan.** Normalize every `K` cycles instead of every
  cycle: each `K`-cycle segment is then a pure linear scan / FFT-in-time, with the
  dispersion linearization quantifying how large `K` can be before magnitude
  growth distorts phase. Only `:potential` mode is a pure linear scan with `K=L`.

**The only thing that genuinely needs the exact sub-cycle potential** is the
nonlinear **threshold+reset** demo mode — a crossing occurs at an arbitrary
sub-cycle time and reset is a discontinuity per-cycle sampling cannot resolve —
which is exactly why it is already scoped to ODE-only (§5.7). Conduction delays do
*not* qualify: `e^{-iωτ}` is a carrier phase factor, fully representable in the
per-cycle phase.

See §5.6 for the constructive follow-on: `:spike` mode *is* a phase-domain SSM,
and can be made to share `PhasorDense`'s exact encoding.

### 5.6 A phase-domain SSM for `:spike` mode — unify with `PhasorDense`

`PhasorDense`'s phase-SSM is **feed-forward**: it Dirac-encodes the *input* once
(`dirac_encode`: `θ → exp(k·dt)`, `dt = T·(0.5 − θ/2)`) and runs one linear
`causal_conv` with `phasor_kernel` (`Aⁿ·B`). The wavesheet `:spike` mode is the
same object closed into a **recurrent** loop — each neuron re-emits a spike from
its own state each cycle. Writing the step explicitly:

```
θ[n]  →  emit s[n]                         (currently e^{iθ}; PhasorDense: exp(k·dt))
      →  couple  c[n] = ifft(Ŵ · fft(s[n]))     (Ŵ carries the delay e^{-iωτ})
      →  integrate  z[n+1] = A·z[n] + g·c[n] + drive[n]
      →  extract  θ[n+1] = angle(z[n+1])
```

The two nonlinearities — spike **emission** (phase→Dirac) and phase **extraction**
(`angle`) — are per-cycle *boundary* ops; the integration between them is linear.
This is exactly `PhasorDense`'s structure, looped. Three usable forms follow:

- **(a) Exact per-cycle recurrence (share the encoding).** The current transmit
  emits the **unit phasor** `e^{iθ}` (magnitude 1, no sub-cycle leak);
  `dirac_encode` emits `exp(k·dt)` (magnitude `exp(λ·dt)`, phase `ω·dt = π(1−θ)`).
  They **agree in phase and differ only in sub-cycle leak bookkeeping.** Swapping
  the emission to `dirac_encode` makes the sheet and `PhasorDense` share one
  formalism and one code path — a real design choice (it reintroduces a per-cycle
  leak magnitude), not a free identity. Trained by BPTT over `L` cycles (§5.5).

- **(b) Fixed-point / DEQ formulation — the *parallel* spike trainer.** *Given* the
  emitted spike train `s[0..L−1]`, the state trajectory is a pure linear
  convolution `z[n] = Σ_m Aᵐ·B·(g·c[n−m] + drive[n−m])` — a `causal_conv` with
  `phasor_kernel`, fully parallel. The only nonlinearity is recomputing
  `s[n] = emit(angle(z[n]))`. So `:spike` mode is the fixed point

  ```
  s* = emit( angle( causal_conv(phasor_kernel, couple(s*) + drive) ) )
  ```

  Solve by iterating parallel linear sweeps with a cheap pointwise `emit` between
  them; differentiate via the implicit-function theorem. This is §5.4-regime-3
  (DEQ / EP settling) made concrete for spikes, and it reuses the
  `AttractorPhasorSSM` + EP/hEP machinery already in the repo. Heavy compute is the
  parallel kernel; the nonlinearity is a per-cycle pointwise map.

- **(c) Exact linear phase-SSM near coherence (already derived).** The dispersion
  doc's **phase-mode band** `ν(q')` *is* the linearization of the Sakaguchi–
  Kuramoto per-cycle map `θ[n+1] = angle(A·e^{iθ} + g·(W ⊛ e^{iθ}) + drive)`. Near
  a coherent state the perturbation obeys a linear diagonal-per-mode SSM
  `z′[n+1] ≈ M_phase(q)·z′[n]` — exact, scannable, and doubling as the stability
  oracle. So in the small-perturbation regime `:spike` already *has* a linear
  phase-SSM.

Net: `:spike` mode is a phase-domain SSM whose transport is linear and whose
emission/extraction are cheap per-cycle boundary nonlinearities — trainable by
BPTT (a), parallel DEQ sweeps (b), or exactly linearly near coherence (c), none
needing the ODE adjoint.

### 5.7 What ODE mode is still for

Keep Tier-2, but scoped: (i) genuine continuous spike-*current* input
(`SpikingCall`/`CurrentCall`), (ii) the nonlinear threshold+reset demo mode that
the discrete trainable path deliberately omits, and (iii) an
independent oracle for the discrete↔continuous equivalence check (already at
0.997–0.9997 field similarity). It is a *verifier*, not the *trainer*.

### 5.8 Concrete extension checklist

To make the discrete SSM cover the sheet **and** the expert module:

1. Parallel-scan / FFT-in-time forward+backward for the linear sheet segment
   (generalize `_wave_rollout`; reuse `phasor_kernel`/`causal_conv`). — **shipped:**
   `_wave_rollout_scan` in `src/wave.jl`, exposed as `wave_simulate(...; mode=:scan)`.
   Diagonalizes per mode (`M(q)=A+g·Ŵ(q)`), computes the homogeneous power term
   plus a causal `causal_conv` of the drive with kernel `K_q[m]=M_q^{m-1}` (built
   as `exp((m-1)·log M)` to stay AD-differentiable — `cumprod`'s `dims` adjoint is
   unsupported). Matches the sequential Buffer loop to ~6e-8 (autonomous) / ~1e-7
   (driven), gradients flow; `test_wave_scan_equivalence` (8 checks) in
   `test/test_wave.jl`. Errors on the nonlinear regimes.
2. Express the expert as a selective diagonal transition `D_expert = diag(g·φ)`
   spliced into the recurrence (§5.3). — **shipped (input-conditioned form):**
   `_apply_wave_experts` in `src/wave.jl` stamps each expert's unit bind phasor
   `φ_e` onto the drive inside its patch mask (`drive ⊙ (1 + Σ_e mask_e·g_e·(φ_e−1))`).
   Because a *local* bind is site-diagonal while transport is Fourier-diagonal
   (§5.4), the bind rides the **input injection**, not the state transition —
   which is what keeps it a single scan.
3. Ship the **input-conditioned** selective path first — one associative scan,
   fully parallel (§5.4, regime 1). — **shipped:** router → bind → the item-1
   scan. The precomputed modulated drive feeds `_wave_rollout_scan` unchanged;
   gradients reach the bind phasors and router logits. `test_wave_experts`
   (15 checks) in `test/test_wave.jl`.
4. Add the **state-conditioned** path via event-based operator splitting for
   sparse experts (§5.4, regime 2.1), with chunked scan as the dense-ish
   fallback. — **shipped:** `_wave_rollout_chunked` in `src/wave.jl` runs the
   linear sheet as a parallel scan within each chunk and applies an `interact(z,c)`
   event (state-conditioned gate+bind) at chunk boundaries; `_state_read` does the
   masked patch read of the live wave. Identity event ⇒ chunked == full scan
   exactly for any chunk size (linear transport composes across boundaries), so
   sequential cost is only the number of events. Gradients flow through the
   state→gate→bind feedback. `test_wave_state_experts` (15 checks) in
   `test/test_wave.jl`.
5. Route discrete-gate gradients with Gumbel-Softmax / straight-through, which
   compose with both BPTT and the precompute-then-scan structure. — **shipped:**
   `moe_gate` (straight-through top-1, DeepSeek loss-free per-expert bias steers
   selection only) + `update_moe_bias` (out-of-graph load balancing), in
   `src/wave.jl`, exported.
6. Unify `:spike` emission with `PhasorDense`'s `dirac_encode` so both share one
   phase-SSM formalism (§5.6a), and add the DEQ fixed-point sweep as the parallel
   spike trainer (§5.6b). — **shipped:** `_wave_rollout_deq` in `src/wave.jl`,
   exposed as `wave_simulate(...; mode=:deq, n_sweeps, emit)`. Solves the spike
   fixed point `s* = emit(linear_response(s*))` by `n_sweeps` parallel sweeps
   (scalar-`A` `causal_conv` + batched spatial coupling + pointwise `emit`); the
   causal map makes `n_sweeps == L` reproduce the sequential spike rollout exactly
   and fewer sweeps settle toward it. `emit=:unit` (`z/|z|`) or `:dirac`
   (`exp(k·dt)` via `_emit_dirac`, `PhasorDense`-consistent — §5.6a). Gradients
   flow through the sweeps. `test_wave_spike_deq` (8 checks) in `test/test_wave.jl`.
7. Keep ODE as verifier only (§5.7). — held (unchanged).

**§5 implementation status:** items #1–#6 are shipped and tested (131 wave tests
pass); #7 is a standing policy, not code. The pieces now exist to assemble the §6
prototype — a `PhasorWaveSheet` + expert Lux layer trained on the discrete SSM —
without further primitives.

**The remaining unproven link** is narrower than before: not "gate through an ODE
adjoint," but *does a straight-through gate gradient stay informative through a
long parallel/BPTT wave rollout, or collapse to one always-on expert?* That —
not solver fragility — is what §6 probes.

---

## 6. First prototype — probe the riskiest unknown first

> **Status — built & probed.** The prototype is the `WaveExpertSheet` Lux layer
> in `src/wave.jl` (composes a `PhasorWaveSheet` substrate + expert bank + top-1
> router; `route_stats` gives the go/no-go readout; `test_wave_expert_layer`
> covers it). The go/no-go experiment (`demos/wave_experts_gonogo.jl`, synthetic
> "which patch is coherent" routing task) **passes: no gate collapse.** On the
> discrete-SSM (`:potential`) substrate the router reaches 100% accuracy with
> per-expert load staying spread (load-entropy ≈ 1.2 of max ln 4 = 1.386; all four
> experts used), so the straight-through gate gradient *does* stay informative
> through the wave rollout — Open Q §7.1 resolves in favour of the architecture.
> The steps below are the original plan; ✓ marks what shipped.

The cheapest informative experiment isolates the make-or-break question: **does a
discrete gate gradient stay informative through the discrete-SSM wave rollout, or
collapse to one always-on expert?**

1. **Substrate.** ✓ `WaveExpertSheet` composes a `PhasorWaveSheet`; the substrate
   `transmit` is selectable (`:potential` → linear scan, `:spike` → DEQ). The
   go/no-go ran on `:potential`; `:spike` is supported via `mode=:deq`.
2. **Experts, patches.** ✓ A bank of `n_experts` tiled patches (`_tile_masks`);
   the top-1 router (`moe_gate`, **DeepSeek loss-free per-expert bias**, online via
   `update_moe_bias`) picks which fire; each fired expert stamps a learned
   unit-magnitude bind phasor `φ_e` inside its patch — the selective diagonal
   insert `D_expert` of §5.3.
3. **Decode check.** ✓ `demos/wave_experts_decode.jl` (+ `test_wave_expert_decode`).
   Findings: the stamped bind **is recoverable** — a *differential* decode (unbind
   the sheet output against a bind-free reference, `r·conj(r₀)`, then match to the
   codebook) recovers every expert's code, 100% up to E=64 with a coherent carrier
   and ~92–99% even at a fully incoherent carrier. A *blind* decode (no reference)
   is **coherence-limited** — 100% at σ=0 → 0% at σ=1, the same square-law channel
   as the readout study. Two limits surfaced: row-band tiling needs `E ≤ H`
   (now enforced), and the per-expert bind is a **1-D scalar phase** → codes decode
   independently on disjoint patches but cannot be superposed/factored at one site
   (Open Q §7.2).
4. **Train — discrete SSM, not ODE (per §5).** ✓ Both regimes implemented:
   **input-conditioned** (gate reads the drive → one selective scan, §5.4 r.1) and
   **state-conditioned** (event-split chunks, §5.4 r.2). Straight-through gate; no
   `BacksolveAdjoint`. The go/no-go trained the input-conditioned `:potential`
   path.
5. **Metric that decides go/no-go.** ✓ `route_stats` returns per-expert load +
   routing entropy. Result: **utilization stays balanced** (load-entropy ≈ 1.2 of
   ln 4) and accuracy hits 100% → *go.* Next: scale the expert bank and add the
   decode check (step 3).
6. **Prerequisite (§5.8 #1).** ✓ `_wave_rollout_scan` matches the Buffer loop to
   ~6e-8 (see §5.8 #1).

> **Real-task update.** `demos/wave_experts_fashionmnist.jl` wires `WaveExpertSheet`
> into FashionMNIST (7 experts on a 28×28 sheet, similarity readout). Quick run
> (4k/1.5k, 4 ep): wave+experts **0.770** vs plain-sheet **0.759** (+1.1% for +56
> params), and the **gate does not collapse** — entropy holds at 1.60/1.95 (82% of
> max), all 7 experts used. Consistent with the earlier attribution finding: the
> wave/expert layer is a near-free adjunct on top of a head-dominated pipeline.

Keep the memory budget in view (DGX Spark, ~110 GB unified): a 16×16 toroid ×
rollout length × batch is small, but a *bank* of experts each with its own
resonator can grow fast — size the expert count against the capacity ceiling of
§3, not against the number of patches.

---

## 7. Open questions

1. **Gate-through-substrate trainability.** — **resolved (first pass):** the §6
   go/no-go shows discrete top-1 routing stays trainable and balanced through the
   discrete-SSM rollout (100% acc, no collapse, load-entropy ≈ 1.2 of ln 4). Open
   remainder: does this hold at scale (many fine-grained experts), on the `:spike`
   DEQ substrate, and on a real task rather than the synthetic routing probe?
2. **Resonator capacity vs. wave payload.** How many simultaneously bound
   transforms can one traveling wave carry through the toroid before factorization
   degrades, and how does that scale with lattice / hypervector dimension?
   — *partly probed:* the decode check (§6 step 3) shows spatially-separated
   (disjoint-patch) scalar codes decode independently with no superposition ceiling
   (100% to E=64), but the prototype's 1-D scalar bind has **no** superposition
   capacity — you cannot bundle multiple codes at one site and factor them. The
   open work is high-D per-site bind codes + a resonator network to get genuine
   superposition capacity, and to measure its ceiling vs. hypervector dimension.
3. **Spatial load signal.** What is the right load-balancing signal for
   *spatially-localized* experts — can the recent-load per-expert bias be adapted
   when "load" is phase/patch occupancy on a toroid rather than a token count?
4. **Biological grounding (weakened).** Since the strong Numenta columnar-expert
   claims were refuted (§8), what actually supports treating a wavesheet patch as
   an independent "expert"? Mountcastle columnar *repetition* and traveling-wave
   phenomenology survive; column-as-complete-model does not.

---

## 8. Honesty: what did **not** survive verification

Two "cortical column as a complete computational expert" claims were **killed** in
adversarial verification, and the biological case must be adjusted accordingly:

- ❌ *"Grid-cell reference frames exist in every cortical column"* (**refuted
  3–0**). The source (Hawkins et al. 2018) is explicitly a *"Hypothesis and
  Theory"* article; the real sentence is "We **propose** that grid cells exist
  throughout the neocortex" — a proposal quoted as established fact.
- ❌ *"Each column is a complete sensorimotor modeling unit"* (**refuted 2–1**).
  The quote is genuine but overreaches: it rests on unconfirmed theoretical
  constructs, not experimental confirmation.

**Consequence.** Justify the repeated-expert unit from **Mountcastle columnar
repetition + cortical traveling-wave phenomenology** (which survived, chiefly via
the Wave-RNN / Neural Wave Machine engineering line), **not** from
Hawkins/Thousand-Brains reference-frame specifics. And remember the umbrella
caveat: every verified claim validates a *component* in isolation — the integrated
"expert-modulates-passing-wave" module is inferred by composition, and §6 exists
to test that inference.

---

## References

Deep-learning / VSA (all primary-source, claims verified 3–0 unless noted):

- [DeepSeekMoE, arXiv:2401.06066](https://arxiv.org/abs/2401.06066) — fine-grained expert segmentation; specialization as design goal.
- [Auxiliary-Loss-Free Load Balancing, arXiv:2408.15664](https://arxiv.org/abs/2408.15664) — per-expert bias before top-K; adopted in DeepSeek-V3.
- [Sparsely-Gated MoE, arXiv:1701.06538](https://arxiv.org/abs/1701.06538) — trainable gate → sparse expert combination (conditional computation).
- [Gumbel-Softmax, arXiv:1611.01144](https://arxiv.org/abs/1611.01144) — differentiable relaxation + straight-through for discrete gates.
- [Resonator Networks I, arXiv:2007.03748](https://arxiv.org/abs/2007.03748) and [analysis, arXiv:1906.11684](https://arxiv.org/abs/1906.11684) — factoring Hadamard-bound products by search in superposition.
- [Spiking phasor resonator on neuromorphic HW, arXiv:2208.12880](https://arxiv.org/abs/2208.12880) (Nat. Mach. Intell. 2024) — binding as equivariant transform; complex resonator as multi-compartment spiking phasor neurons.
- [FHRR + deep networks, arXiv:2207.08953](https://arxiv.org/abs/2207.08953) — VSA/FHRR merged with residual/attention nets (this repo's git author).
- [Wave-RNN, arXiv:2309.08045](https://arxiv.org/abs/2309.08045) (ICLR 2024) — traveling-wave hidden state as invertible short-term memory.
- [Neural Wave Machine, PMLR v202](https://proceedings.mlr.press/v202/keller23a.html) (ICML 2023) — locally-coupled oscillatory RNN whose waves encode transformations.
- [AKOrN, arXiv:2410.13821](https://arxiv.org/abs/2410.13821) (ICLR 2025 oral) — Kuramoto oscillator neurons compose with attentive connectivity.
- [ViTCA, arXiv:2211.01233](https://arxiv.org/abs/2211.01233) (NeurIPS 2022) — localized self-attention on local cellular-automata updates.
- [Recurrent-depth transformer, arXiv:2502.05171](https://arxiv.org/abs/2502.05171) — looped latent recurrence (1-D analog of toroidal re-entry).
- [Neural ODE adjoint, arXiv:1806.07366](https://arxiv.org/abs/1806.07366) and [SciMLSensitivity docs](https://docs.sciml.ai/SciMLSensitivity/stable/manual/differential_equation_sensitivities/) — black-box backprop through ODE solvers; BacksolveAdjoint+ZygoteVJP.
- [Self-attention as VSA, arXiv:2512.14709](https://arxiv.org/abs/2512.14709) — *interpretive*, single-author preprint (medium confidence).

Neuroscience (adjusted per §8):

- Mountcastle — columnar organization / repeated-unit hypothesis (surviving grounding: repetition, not column-as-complete-model).
- Muller et al. (2018), *Cortical travelling waves*, Nat. Rev. Neurosci. — delays generate waves in recurrent circuits.
- Hawkins et al. (2018/2019), Front. Neural Circuits — Thousand Brains / grid-cell reference frames. **Hypothesis-and-Theory; the "every column" and "complete model" claims were refuted in verification — do not cite as established.**
