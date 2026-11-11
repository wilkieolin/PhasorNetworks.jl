# Port note → phasor_torch: stackable LSA/LCA transformer blocks

**Audience:** the agent implementing this in `phasor_torch` (PyTorch).
**Goal:** port the phase-domain transformer stack so that `PhasorLSA`/`PhasorLCA`
become **depth-robust** (trainable through ≥16 stacked blocks) instead of
collapsing past depth ~2.

The single load-bearing result from the Julia experiment
(`results/lsa_lca_residual/FINDINGS.md`):

> Stacked phase attention is trainable at depth **only when each attention
> sublayer is wrapped in a residual with a ReZero gate** (a learnable scalar α
> init ≈ 0, giving *exact identity at init*). Down-scaling the projection/FFN
> weights does **not** rescue attention. Pre-norm recentering is optional and
> slightly hurts.

Reproduce that curve and you've ported the feature.

---

## 0. Phase-domain substrate (assumed to already exist in phasor_torch)

Phases θ ∈ [−1, 1] in units of π. Core maps (match your existing conventions):

- `to_complex(θ) = exp(iπθ)` → `torch.polar(torch.ones_like(θ), π*θ)`
- `to_phase(z)   = angle(z)/π` → `torch.angle(z) / π`
- `normalize(z)  = z / |z|`
- `similarity(a, b)` over the feature axis = `mean_c cos(π(a_c − b_c))`
  = `Re(mean_c exp(iπa_c)·conj(exp(iπb_c)))`, range [−1, 1].

**`v_bind(x, y)` = phase addition** = `wrap(x + y)` to [−1, 1]. Two non-negotiable
properties:
1. **Identity element is `y = 0`** (so a zero branch output ⇒ pass-through).
2. **Straight-through gradient 1 to both operands** — the wrap (mod) must be
   detached. In PyTorch:
   ```python
   def v_bind(x, y):
       s = x + y
       # wrap to [-1,1] with the correction detached (gradient flows as identity)
       return s - (2.0 * torch.floor((s + 1.0) / 2.0)).detach()
   ```
   (Equivalently: keep phases unwrapped through the residual chain and only wrap
   at the readout — `exp(iπθ)` is periodic so it's invariant either way. The
   detached-wrap above matches the Julia `remap_phase`/`ignore_derivatives`.)

---

## 1. Components to implement (the actual deliverable)

### 1a. `PhaseRecenter` (parameter-free pre-norm)
Subtract the per-token circular mean across the **channel** axis:
```python
m = to_phase(to_complex(x).sum(dim=CHANNEL, keepdim=True))   # circular mean angle
return v_bind(x, -m)
```
Phase analog of LayerNorm centering. (Optional — see §3.)

### 1b. `PhasorResidual(sublayer, gate="none"|"rezero", alpha0=0.1)`
The crux. Generic identity-at-init residual around any **shape-preserving** phase
layer (`sublayer: (…,D) → (…,D)`):
```python
branch = sublayer(x)
g = self.alpha if gate == "rezero" else 1.0      # alpha = nn.Parameter(full((1,), alpha0))
return v_bind(x, g * branch)
```
- `gate="rezero"` with `alpha0 → 0` ⇒ **block ≡ identity at init**, regardless of
  what `sublayer` computes. This is the mechanism that makes attention stack.
- `gate="none"` ⇒ identity-at-init only if `sublayer` itself emits ≈0 (true for a
  down-scaled FFN, **false** for attention — hence ReZero is required for attn).

### 1c. `PhasorTransformerBlock(d_model, attn, d_ff=d_model, gate="rezero", alpha0=0.1, branch_init_scale=0.1, recenter=True)`
Pre-norm `residual(attn) + residual(FFN)`:
```python
ffn = Sequential(PhasorDense(d_model, d_ff), PhasorDense(d_ff, d_model))   # init weights * branch_init_scale
attn_branch = Sequential(PhaseRecenter(), attn) if recenter else attn
ffn_branch  = Sequential(PhaseRecenter(), ffn) if recenter else ffn
self.attn_res = PhasorResidual(attn_branch, gate, alpha0)
self.ffn_res  = PhasorResidual(ffn_branch,  gate, alpha0)

def forward(x):
    return self.ffn_res(self.attn_res(x))
```
`attn` is a prebuilt `d_model→d_model` phase attention layer (LSA/LCA/etc.),
constructed by the caller.

---

## 2. The attention layers (port if not already present)

Both are **pointwise in the sequence axis L** — attention is computed *across
heads*, not across time — so they're cheap and compose with the residual stack.
Layout below is feature-first `(D, L, B)` as in Julia; **use batch-first
`(B, L, D)` in PyTorch** and adjust the einsums (given below in batch-first).

Reshape `D → (H heads, Dh = D/H)`.

### 2a. `PhasorLSA(d_model, n_heads, init_scale=3.0)` — local self-attention
```
Q,K,V = three phase projections of x         # (B,L,D) -> (B,L,H,Dh)
Qc,Kc,Vc = to_complex(Q),to_complex(K),to_complex(V)
scores[b,l,h,h'] = (1/Dh) * Re( einsum('blhd,blkd->blhk', Qc, conj(Kc)) )   # (B,L,H,H), in [-1,1]
weights = exp(beta * scores) / H              # beta = learnable scalar (Parameter, init 3.0). NOTE: NOT softmax.
Yc[b,l,h,:] = einsum('blhk,blkd->blhd', weights, Vc)    # complex weighted sum over the other-head axis
Y = to_phase(Yc).reshape(B,L,D)               # then activation (normalize/identity)
```

### 2b. `PhasorLCA(d_model, n_heads, n_anchors, init_scale=3.0)` — local cross-attn (Hopfield)
```
K,V = two phase projections of x              # (B,L,H,Dh)
anchors = learnable phase bank (D, A) -> (H, A, Dh); Ac = to_complex(anchors)
scores[b,l,h,a] = (1/Dh) * Re( einsum('blhd,had->blha', to_complex(K), conj(Ac)) )   # (B,L,H,A)
weights = exp(beta * scores) / A
Bundle[b,l,h,:] = einsum('blha,had->blhd', weights, Ac)     # complex anchor bundle
Yc = to_complex(V) * Bundle                   # elementwise complex mult == phase binding (VSA bind)
Y = to_phase(Yc).reshape(B,L,D)               # then activation
```

### 2c. The projections
The Julia Q/K/V projections are `PhasorDense` with per-channel SSM dynamics
(decay λ, `init_mode=:hippo`) on the way in. **The residual/ReZero result is
independent of the projection internals** — port whatever your `PhasorDense`
already is (at minimum a complex linear on `exp(iπθ)` + `normalize` + `to_phase`).
If you have the SSM-dynamics PhasorDense, use it; it only affects absolute
accuracy, not the depth-robustness mechanism.

---

## 3. Design decisions that MUST carry over (these are the findings)

1. **ReZero is the identity-at-init mechanism for attention — not weight
   downscaling.** `branch_init_scale` is an FFN-only lever; the attention branch
   reaches identity only via `alpha→0`. (Empirically, FFN-downscaling collapses
   identically to the un-fixed baseline.)
2. **Give α a higher learning rate (~5×) than the rest of the network** — it
   warms up from ≈0. (Mirror your optimizer’s per-parameter-group LR.)
3. **Default = plain `rezero` (no recenter).** Pre-norm recentering was neutral-
   to-slightly-worse at depth. Keep `recenter` as an option, default it off if you
   want the best result, on if you want to match the Julia default.
4. **Per-channel ω rule:** every channel shares one carrier frequency ω (=2π);
   reshaping `D→(H,Dh)` must NOT introduce per-head/per-channel ω. Phase-locking
   across channels is what makes the similarity/bind operations meaningful.
5. The attention scale uses `exp(β·s)/H` (or `/A`), **not** a normalized softmax;
   β is a single learnable scalar. (It's unbounded — a known wart; a true-softmax
   ablation is future work, not required for parity.)

---

## 4. Parity tests to port (from `test/test_transformer_block.jl`)

- **Exact identity at init:** with `gate="rezero", alpha0=0, recenter=False`,
  `block(x) == x` (atol 1e-5) for both LSA and LCA wrapped.
- `gate="none"` has no α parameter; `gate="rezero"` adds exactly one.
- Shape/type preserved for `(B,L,D)` and `(B,D)`; outputs are valid phases ∈[−1,1].
- Gradients finite through a depth-4 stack.
- `PhaseRecenter`: per-token circular mean over channels ≈ 0 after applying.

---

## 5. Validation target (reproduce this curve)

Sequential FashionMNIST, **one image row per timestep** (C_in=28, L=28),
encoder `PhasorDense(28→D)` → `depth ×` block → `SSMReadout(D→10)`.
D=64, H=4, A=32, RMSProp lr 3e-4, α-LR ×5, 8 epochs, batch 32, 3 seeds.

Test accuracy vs depth (Julia, σ≤0.02):

| treatment | d1 | d2 | d4 | d8 | d16 |
|---|---|---|---|---|---|
| `old` (gate=none, scale=1) | ~.69 | ~.65 | **~.10** | ~.10 | ~.10 |
| `downscaled_ffn` (gate=none, scale=.1) | ~.70 | ~.69 | **~.11** | ~.10 | ~.10 |
| `rezero` (gate=rezero, scale=.1) | ~.68 | ~.73 | ~.77 | ~.77 | **LSA .775 / LCA .788** |

**Pass criterion:** `old`/`downscaled_ffn` collapse to chance (~0.10) by depth 4,
while `rezero` stays ≳0.75 and is non-decreasing through depth 16. Absolute
numbers will differ with your PhasorDense/optimizer details; the **depth-collapse
vs depth-robust split between gate=none and gate=rezero is the thing to
reproduce.**

---

## 6. Suggested build order
1. `v_bind` straight-through wrap (+ unit test: gradient is 1 to both args).
2. `PhaseRecenter`, `PhasorResidual` (+ identity-at-init test).
3. `PhasorTransformerBlock` (+ shape/grad tests).
4. Confirm/port `PhasorLSA`, `PhasorLCA` forward (einsums in §2).
5. Reproduce the §5 depth curve on sequential FashionMNIST.

Reference implementation: PhasorNetworks.jl `src/ssm.jl`
(`PhaseRecenter`/`PhasorResidual`/`PhasorTransformerBlock`/`PhasorLSA`/`PhasorLCA`),
tests in `test/test_transformer_block.jl` + `test/test_local_attention.jl`,
results in `results/lsa_lca_residual/FINDINGS.md`.
