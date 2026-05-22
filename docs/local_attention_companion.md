# Local Self-/Cross-Attention: Implementation & Test Plan

Companion to `docs/local_attention_derivation.tex`. The tex document is
the formal specification (definitions, propositions, equivalence proofs,
bibliography). This file is informal: a tie-back to the existing source
tree, a sketch of the planned Julia layers, a literature digest, and an
experimental test plan. Implementation is the next deliverable; nothing
in this document needs to be merged into source yet.

## 1. Why this fits the project

`PhasorDense` and `PhasorConv` are three-mode layers: they evaluate the
same per-channel equation `dz_c/dt = k_c z_c + W·I(t)` as a discrete
Dirac recurrence, a continuous ODE, or a parallel causal convolution,
and the three modes agree on the sample lattice. The current attention
layers in `src/network.jl` and `src/ssm.jl` (PhasorAttention,
SSMSelfAttention, SSMCrossAttention) do not have this property because
they compute the similarity score *across time* — producing an
`(L, L, B)` matrix that fuses every time slice into every other. There
is no per-slice update for that operation; spiking and streaming
deployments are blocked.

Local Self-Attention (LSA) and Local Cross-Attention (LCA) replace the
across-time score with a head-axis score. The result is pointwise in
time and inherits the three-view structure of the surrounding SSM. See
`docs/local_attention_derivation.tex` for the equations and the
equivalence proofs (Propositions 1 and 3).

## 2. Connection to existing code

### 2.1 What we reuse

| What | Where | Why |
|---|---|---|
| Fourier-HRR similarity | `src/vsa.jl:334` (`similarity`) | Identical to the per-slice head score; just applied along a different axis. |
| Pairwise similarity over slices | `src/vsa.jl:458` (`similarity_outer`, real 3D); `src/vsa.jl:486` (complex 3D, canonical form) | Score `(H, H, L, B)` or `(A, H, L, B)` is just `similarity_outer` slicing on a different axis (heads, not time). |
| Scaled exponential mix and complex-domain weighted sum | `src/network.jl:1606` (`score_scale`), `src/network.jl:1624` (`attend` Phase 3D) | Equations (lsa-scale, lsa-mix) reuse this verbatim once the score tensor's axes have been permuted to put the reduced head index in the contracted position. |
| Phase-domain query/key/value projections | `src/network.jl:288` (`PhasorDense` struct), `:410` (`_forward_3d_dirac`), `:495` (`CurrentCall` dispatch) | `Π_Q, Π_K, Π_V` in the derivation are PhasorDense layers, identical to those used today by SSMCrossAttention/SSMSelfAttention. |
| Trainable phase anchor bank | `src/ssm.jl:252` (`keys` in SSMCrossAttention's `initialparameters`) | LCA's `Q*` is exactly this pattern, generalized to shape `(D, A)` and reshaped to `(D_h, H, A)` per slice. |
| Spiking pathway via current reconstruction | `src/ssm.jl:605` (`reconstruct_from_current`), `:641-:674` (call sites in SSMSelfAttention / SSMCrossAttention spiking dispatch) | Same trampoline used for the new layers' SpikingCall path. |
| `causal_conv_dirac` and `phasor_kernel` | `src/kernels.jl:70` (`phasor_kernel`), `:139` (`causal_conv`) | Surrounding SSM convolution; LSA/LCA simply replace the input slice with `f_β(x[l])` before encoding. |
| Lux layer contract (`initialparameters` / `initialstates` / `parameterlength`) | `src/ssm.jl:251-:269` (SSMCrossAttention example) | Pattern for the new layers' parameter trees. |

### 2.2 What we add

- **A head-axis similarity primitive.** Two dispatches of
  `similarity_outer_heads(q, k)` (in `src/vsa.jl`), selected by rank:
  - LSA: `q, k :: (D_h, H, L, B)` Phase → `(H, H, L, B)` score.
  - LCA: `q :: (D_h, H, A)` (time-invariant anchors) and
    `k :: (D_h, H, L, B)` → `(A, H, L, B)` score.

  Both delegate to `_similarity_outer_canonical_complex` (the existing
  closed-form rrule-equipped kernel), so the primitive is
  end-to-end differentiable without manual rules.
- **`PhasorLSA` layer.** Lux layer with parameters `(q_proj, k_proj,
  v_proj, scale)` and the dispatch quartet below. Mirrors
  `SSMSelfAttention` (`src/ssm.jl:327`) but with the head-axis score
  and the four execution modes.
- **`PhasorLCA` layer.** Lux layer with parameters `(k_proj, v_proj,
  anchors, scale)`, where `anchors` is `(D, A)` Phase. The forward is
  a Hopfield-style content-addressable lookup:
  1. Score each anchor against the per-head `K` slice;
  2. Bundle the anchors in the complex domain weighted by
     `softmax(β · score)` to get a retrieved memory phasor `R`;
  3. Bind `R` onto `V` via element-wise complex multiplication
     (= phase addition) to produce the output.

  The binding step (3) is the key non-degeneracy fix relative to the
  literal "sum-product with V projected from input" form, which would
  collapse to a uniform amplitude rescaling of V (output phase = V's
  phase). See `docs/local_attention_derivation.tex` Remark
  *Why bind, instead of bundle V directly?* for the formal explanation.
- **Tests in `test/test_local_attention.jl`** (new file, wired into
  `test/runtests.jl`) covering each dispatch mode, gradient sanity,
  spiking-vs-discrete correlation, and a non-degeneracy regression
  guard on LCA.

## 3. Dispatch surface to mirror

Both `PhasorLSA` and `PhasorLCA` implement the same dispatch quartet
that `PhasorDense` already exposes (`src/network.jl:385-549`), plus a
Complex-3D back-compat trampoline:

- `(::PhasorLSA)(x::AbstractArray{<:Phase,2}, ps, st)` — single-slice
  (no time axis). Reshapes to `(D, 1, B)` internally and invokes the
  3D path with `L=1`, dropping the time dim from the result.
- `(::PhasorLSA)(x::AbstractArray{<:Phase,3}, ps, st)` — Dirac per-slice
  evaluation in parallel. Reshapes channels to `(D_h, H, L, B)`,
  computes the score with `similarity_outer_heads`, applies the
  `exp(β · score) / H` scaling, performs the value mix in the complex
  domain via `batched_mul`, reshapes back to `(D, L, B)`, returns
  Phase.
- `(::PhasorLSA)(x::AbstractArray{<:Complex,3}, ps, st)` —
  back-compat trampoline through `complex_to_angle` → Phase 3D path
  → `angle_to_complex`; returns Complex 3D.
- `(::PhasorLSA)(x::SpikingCall, ps, st)` — trampoline to `CurrentCall`
  via the same pattern as `src/network.jl:483-486`.
- `(::PhasorLSA)(x::CurrentCall, ps, st)` — `reconstruct_from_current`
  (`src/ssm.jl:605`) recovers a Complex 3D tensor, which is then
  routed through the Complex-3D back-compat dispatch. Output type is
  Complex 3D on the unit circle. (The per-ODE-step "in-loop" variant
  is a follow-up; the current baseline is the more numerically stable
  reconstruct-then-Phase-3D path, matching `SSMSelfAttention` /
  `SSMCrossAttention`.)

`PhasorLCA` has the same five signatures. The `(D_h, H, A)` anchor
bank is time-invariant, so the per-slice score reuses the same anchor
tensor across `L`; per-head bundling collapses `A` into a `(D_h, H, L, B)`
retrieved memory `R` which is then bound (element-wise complex
multiplication) onto `V_complex`.

## 4. Literature digest

The tex bibliography lists the formal references. The informal notes:

**Local / windowed transformer attention.** Longformer-style
sliding-window attention reduces the time-axis softmax to a window of
width `w`, achieving `O(L·w)` cost. The attention is still
across-time; locality is a budgetary restriction, not a structural
elimination. LSA is stronger: it removes the time axis from the score
entirely (cost `O(L · H²)`, independent of any window). The hard
locality is the property we need for spiking deployment.

**Modern Hopfield networks.** Ramsauer et al. (ICLR 2021) showed that
the continuous Hopfield retrieval rule
`ξ_new = X · softmax(β · X^T ξ)` is identical to a transformer
attention update with stored patterns `X` as keys/values. LCA's
*retrieval step* is exactly this rule restricted to the unit torus
with the Fourier-HRR inner product (Proposition 3 of the tex); the
*binding step* then multiplies the retrieved memory phasor onto the
input value `V` per channel, producing a content-addressable rotation
of `V`. Setting `V ≡ 0` recovers the pure Hopfield-retrieval limit.
The exponential capacity bound `A_max ~ exp(c · D_h)` from the
Hopfield theorem gives a useful design heuristic for the anchor
count `A`, though a precise unit-torus version is future work.

**State-space attention.** Mamba (selective SSM), RetNet (chunked
retention), S5 (diagonal SSM + linear attention) all blend SSM and
attention by making the SSM transition input-dependent. Our
construction is orthogonal: the transition stays fixed, the attention
is applied pointwise in time. We do not pursue input-dependent
transitions here; the multi-timescale memory from HiPPO already
supplies the cross-time mixing that selective-state methods chase.

**HD-VSA attention precedents.** Frady et al. (Neural Computation,
2018) and follow-up VSA-attention work use HRR similarity as the
score and complex-domain superposition as the mix. We inherit those
choices verbatim — the novelty is the head-axis restriction plus the
explicit three-view derivation against the Phasor SSM.

## 5. Experimental test plan

The plan below is tiered: sanity checks come first and are cheap;
the comparative training run is the largest item and should only run
after the sanity tier passes.

### 5.1 Sanity (must pass before any training)

- **Shape and dtype.** For each mode (2D Phase, 3D Phase, Complex 3D,
  SpikingCall, CurrentCall), confirm output type and shape. Documented
  promotion rules: Phase → Phase, Complex 3D → Complex 3D (unit
  modulus), SpikingCall and CurrentCall → Complex 3D (unit modulus,
  reconstructed from the ODE solution — matches
  `SSMSelfAttention` / `SSMCrossAttention`). Mirror `test/test_ssm.jl`
  style.
- **Three-view equivalence.** Fix a random `(D, L, B)` Phase input,
  random parameters, and a single `SpikingArgs`. Run the layer in
  discrete, parallel, and continuous modes and confirm
  `maximum(arc_error(y_disc, y_par)) < tol` and
  `maximum(arc_error(y_disc, sample(y_ode, l*T))) < tol_ode` for
  appropriate `tol`. The discrete vs. parallel tolerance should be
  near-machine-epsilon for Float32; the ODE tolerance depends on the
  Tsit5 step size.
- **H = 1 identity.** With `H = 1` and `Π_Q = Π_K = Π_V = I` (no
  projection), `f_β(x) = x` up to amplitude normalization. Use this as
  a degenerate sanity check.
- **Gradient sanity.** `Zygote.gradient` on a scalar loss
  `mean(arc_error(LSA(x; ps), target))` returns finite gradients for
  every parameter; no NaN, no `nothing` in the trainable subtree.
  Same for LCA on the anchors.

### 5.2 HD-VSA invariance

- **Bind/unbind symmetry under LCA.** With anchors `Q*` set to random
  symbols and the input set to `bind(Q*[:, a], r)` for a random
  rotation `r`, the LCA output should peak the score at row `a` and
  the value mix should recover `r` modulo the V projection. Confirms
  the modern-Hopfield equivalence numerically.
- **Capacity sweep for LCA.** Vary `A ∈ {16, 64, 256, 1024}` for
  fixed `D_h = 64`. For each, generate `A` random Phase anchors,
  perturb each by noise of varying SNR, query LCA, and measure recall
  accuracy. Compare the capacity curve to the Hopfield exponential
  bound. Expect a knee well below `exp(c · D_h)` due to the unit-torus
  restriction but qualitatively similar shape.

### 5.3 Comparative training (the big one)

- **Drop-in swap in FashionMNIST.** Take an existing chain in
  `scripts/train_fashionmnist.jl` that uses `SSMSelfAttention` /
  `SSMCrossAttention` and replace those layers with `PhasorLSA` /
  `PhasorLCA`. Match parameter counts (head count tuned). Train for
  the same epoch count, same optimizer, same learning rate.
- **Metrics.** Training loss curve, test accuracy, per-iteration wall
  time (CPU + GPU), peak memory. Memory budget for this DGX Spark:
  ~110 GB unified; estimate before submitting.
- **Discrete-vs-spiking parity.** Train in discrete mode, evaluate on
  the test set in spiking mode without retraining. Accuracy gap
  should be in the same ballpark as the parity already achieved by
  `PhasorDense` / `PhasorConv` (see `test/test_ssm.jl` discrete-vs-
  spiking sections). Confirms three-view equivalence under learned
  weights, not just at initialization.

### 5.4 Stretch goals

- **Streaming inference benchmark.** Run LSA in discrete (per-slice)
  mode with `L = 1` blocks and stream a long sequence; confirm
  constant per-step latency and bounded memory.
- **Composition with input-dependent SSM.** Try composing LSA with a
  Mamba-style selective transition (`A_c(x)`) and see whether the
  combination outperforms either alone. Out of scope for the
  derivation; useful future direction.

## 6. Open questions

- **Head sharing of λ.** The derivation says heads share `ω` (per
  the per-channel-ω rule from CLAUDE.md) but may have distinct `λ`.
  Whether per-head `λ` initialization should be HiPPO-style log-spaced
  *within* each head or shared *across* heads is an empirical
  question for §5.3.
- **Trainable head-mix `M` in LCA.** v1 hard-codes the per-head bundle
  of equation `lca-retrieve` (`M_{a,h,h'} = δ_{hh'}/A`). The tex's
  *Optional trainable head-mix* remark formalises the generalisation
  `R^{(M)}` that mixes the retrieval across heads with a trainable
  `(A, H, H)` tensor; whether the `A · H²` extra parameters buy
  accuracy is a §5.3 ablation candidate.
- **Scaling β.** The `scale` parameter in the existing
  `PhasorAttention` defaults to `3f0`. Whether LSA/LCA need different
  defaults is empirical; the modern-Hopfield interpretation suggests
  scaling `β` with `√D_h` like standard attention's `1/√d_k`.

## 7. Verification of this document

- LaTeX compilation: `pdflatex docs/local_attention_derivation.tex`
  twice (second pass for refs); inspect the PDF for missing references
  and broken equation labels.
- Source-link audit: every `src/...:N` reference in §2 must resolve to
  a real line. Re-grep before merging.
- Stakeholder sign-off on this plan before any new layer code lands in
  `src/`.
