# `coupling = :rect` — a separable directed conveyor

**Status: specification, not yet implemented.** Target: `src/wave.jl`,
`test/test_wave.jl`. Companion to `wavesheet_media_design.md` (which medium for
which purpose) and `wavesheet_report.html` §10 (directed transport).

---

## 1. What it is, in one line

A learnable **1-D complex kernel along the transport axis only**, with zero
extent across the code axis:

    Ŵ_rect(q) = Σ_{k} w_k · e^{-i k q_h}        w ∈ ℂ^{2R+1},  k = −R…R,  w_0 ≡ 0

`Ŵ` is a function of `q_h` alone. That is the whole design.

## 2. Why separability is the point

Measured, not assumed. If the coupling acts only along `h` and the code varies
only along `w`, then for **any** row kernel `K`

    K ⊛_h (env ⊙ p) = (K ⊛_h env) ⊙ p

so the payload passes through exactly — for any tap values, integer or
fractional shift, dispersive or not. This came out of
`demos/wave_transport_fidelity.jl`: it is why the `s = 0.7` arm added to test the
conveyor turned out to test nothing, and it is the only reason any medium in that
benchmark transported a symbol at all (`:shift` ≥ 99.6 sites at fidelity 1.000;
every isotropic medium 0–11 sites).

`:rect` makes that factorization **structural** instead of accidental.
`coupling = :stencil` can *express* a rect kernel — just set the off-row taps to
zero — but they are free parameters with nothing holding them there, so training
destroys the property. `:rect` has no off-row parameters to drift.

**Design rule this encodes:** the medium transports, it does not compute. All
mixing across the code axis happens in discrete stations (expert bands,
LSA/LCA/FFN blocks), never in the substrate. Mixing in the substrate is exactly
what shreds a broadband VSA payload.

## 3. Parameterization

Reuses the existing `stencil_radius` field — no struct growth for the radius.

| name | shape | location | notes |
|---|---|---|---|
| `rect_re`, `rect_im` | `(2R+1,)` Float32 | params | taps `w_k`, `k = −R…R`, index `k+R+1` |
| `qh` | `(H,W)` Float32 | state | already built for `:shift`; reused verbatim |
| `ones_w` | `(1,W)` Float32 | state | broadcast helper (see §4) |

New constructor keywords:

- `rect_axis::Symbol = :h` — transport axis. `:h` uses `st.qh`, `:w` uses
  `st.qw`. `:h` matches every existing convention (`dispersion_diagnostics`
  defaults to `axis = :h`, `:aniso` uses `beta_h`, the fidelity benchmark puts
  the code on columns).
- `rect_onesided::Bool = true` — keep only `k ∈ 1…R` as parameters; the
  backward taps are structurally absent, not merely initialised to zero.
  **This is a hard guarantee, and it is the one the "retain codes moving in one
  direction" requirement actually needs:** with one-sided support a site is only
  ever driven by sites behind it, so information cannot flow upstream *whatever
  the tap values learn to be*. There is always a downstream, so stations can be
  ordered. Set `false` for a general 1-D dispersive medium with no directional
  guarantee.
- `init_rect_mode::Symbol = :shift` — see §5.

`w_0` (self) is structurally zero, matching `:dog` and `:stencil`: self-dynamics
live in `A_step`.

## 4. Construction (the `_build_coupling` branch)

Build in Fourier on the 1-D axis, then broadcast. Do **not** route through the
`place`-matrix scatter that `:stencil` uses.

```julia
elseif l.coupling === :rect
    R  = l.stencil_radius
    ks = l.rect_onesided ? (1:R) : filter(!=(0), -R:R)
    w  = ComplexF32.(ps.rect_re) .+ 1im .* ComplexF32.(ps.rect_im)   # (nk,)
    q1 = l.rect_axis === :h ? st.qh[:, 1] : st.qw[1, :]              # (H,) or (W,)
    Wq = sum(w[i] .* exp.((-1im * Float32(k)) .* q1) for (i, k) in enumerate(ks))
    W_hat = l.rect_axis === :h ? reshape(Wq, :, 1) .* st.ones_w :
                                 transpose(Wq) .* st.ones_h
    return A_step, g, W_hat
```

Three reasons for the Fourier route over the scatter:

1. **Exactness.** `Ŵ` is a function of `q_h` by construction, to machine
   precision. Going via a 2-D kernel and `fft` leaves the separability at the
   mercy of FFT roundoff and makes the §6.2 invariant a tolerance rather than an
   identity.
2. **Cost.** `O(H·R)` instead of `:stencil`'s `O(HW·(2R+1)²)` scatter plus a 2-D
   FFT of the kernel, every forward pass.
3. **Integer offsets only.** The taps sit at integer `k`, so the phase ramp is
   exactly `2π`-periodic on the lattice. This is not cosmetic — see §7.

The `(H,W)` materialisation is required because `_wave_step` does
`reshape(W_hat, H, W, 1)`. It is redundant (the array is constant along `w`) and
could be removed by teaching `_wave_step` to accept an `(H,1)` operand; noted as
an optimisation, not part of this spec.

## 5. Initialisation, and the reduction contract

`init_rect_mode`:

- **`:shift` (default)** — `w_{+1} = 1`, all others 0. This reproduces
  `coupling = :shift, init_shift_h = 1, init_shift_w = 0` **exactly**.
- `:dog` — the 1-D slice of the delayed difference-of-Gaussians along the
  transport axis, truncated to radius `R`, mirroring `_dog_stencil_init`. A
  dispersive, symmetric start; requires `rect_onesided = false` to be meaningful.

**The `:shift` reduction is a test, not a convenience.** It is the same contract
that makes `ExcitableWaveSheet` a fork rather than a divergent copy (`α = 0,
κ_r = 0` reproduces the base sheet to 1.5e-9, asserted in the suite). Pin it and
any drift in the shift path surfaces immediately in the rect path.

## 6. Tests (`test_wave_rect_coupling`)

1. **Reduction.** `:rect`/`:shift`-init `W_hat` equals `:shift`/`s=1` `W_hat` to
   `< 1e-6`.
2. **Separability invariant — the load-bearing one.**
   `maximum(abs, W_hat .- W_hat[:, 1]) < 1e-6`, **and again after 20 Adam steps
   on a reconstruction loss.** `:stencil` seeded identically must fail the
   post-training check. Without the post-training half this test does not
   distinguish `:rect` from a well-initialised `:stencil`, which is the entire
   claim.
3. **Code preservation.** Inject `env(row) ⊙ payload(col)` with a random FHRR
   payload, roll `L = 60`, decode differentially: fidelity stays `> 0.99` for
   every tap setting in a small random sweep. Same code on `:stencil` degrades.
4. **Directedness** (`rect_onesided = true`). Seed one row; after `t` steps no
   energy appears upstream of the seed beyond roundoff. This is the property that
   licenses ordered stations, so assert it directly rather than inferring it from
   a positive `v_g`.
5. **Dispersion.** `v_g` constant across `q_w`; with the shift init `v_g ≈ 1`,
   `gvd ≈ 0`, `gain_curv ≈ 0`.
6. **Gradients** reach `rect_re`/`rect_im`, both nonzero.
7. **Constructor** rejects `2R+1 > H` (transport-axis extent only — *not*
   `min(H,W)` as `:stencil` does; the kernel has no `w` extent to overflow), and
   rejects `rect_axis ∉ (:h,:w)`.

## 7. What this fixes, beyond the intended purpose

- **The Brillouin-zone diagnostic bug.** `dispersion_diagnostics` uses
  `circshift` finite differences, which assume the sampled band is `2π`-periodic
  on the grid. A non-integer `:shift` violates that: the ramp jumps by `2πs`
  across the zone edge and the diagnostics return `gvd* = 88.4`, `v_g* = 3.83`
  for a medium whose true values are `0` and `0.7`, and `transport_forecast` then
  predicts `t½ ≈ 1.6` for a provably lossless translation. `:rect` taps are at
  integer offsets by construction, so the artifact cannot arise.
- **The 1-D diagnostics stop being an approximation.** `dispersion_diagnostics`
  reads the transverse-DC slice. For a separable kernel that slice *is* the whole
  band, so the diagnostics become exact rather than a projection — and the two
  standing traps go away with it: "evaluate at criticality or DC wins" (no 2-D
  gain competition to lose to) and the off-axis critical mode that `radial_band`
  exists to catch.
- **A trainable dispersion budget.** `:shift` offers rigid-or-nothing and
  `:aniso` couples drift to instability (`v = gβ`, and flattening forces `g→0`,
  the §10.4 ceiling). `:rect` interpolates: the taps *are* the transport profile,
  so the network chooses how much dispersion to trade for what.

## 8. What it does not fix, and what to guard

- **`A ≈ 0` versus resonance is untouched.** Shift dominance still needs a strong
  leak. Measured: at `λ = 0.15` (86% self-memory retained) transport is still
  lossless but 7.5× slower, because `ρ ≈ |A| + g` forces `g` down. Under
  `:spike`, `A ≈ 0` keeps `|z| ≈ g ≈ 1` against `θ = 1.39` — permanently
  subthreshold, so the conveyor never actually spikes. `:rect` inherits all of
  this unchanged. It is now the binding constraint on anything downstream.
- **`emission_threshold` needs a `:rect` branch** in `_init_emission_threshold`
  (which has no `ps` and must mirror `_build_coupling` from the `init_*` fields).
  Build the 1-D `Ŵ` from the init taps and return `g · max|Ŵ|`. With the shift
  init `|Ŵ| ≡ 1`, so the existing *"not a criticality reference for `:shift`"*
  warning extends verbatim to `:rect` at default init.
- **`zero_gvd_speed` must reject `:rect`** — it needs `log_speed`, which `:rect`
  does not have. The existing guard is `l.coupling in (:dog, :aniso)`, so this is
  already correct; do not widen it.
- **`wave_transport` returns `phi = NaN`** for `:rect` (the `:dog`/`:aniso`
  check at `src/wave.jl:894`). Correct and already generic.
- **`radial_band` is meaningless here** — it ring-averages, which an anisotropic
  separable kernel has no business being subjected to. Same status as `:aniso`
  and `:shift` today: document, do not error.
- **Stations break translation invariance.** Anything spatially localised along
  the transport axis pays the price the experts overlay already pays
  (`wavesheet_experts_design.md` §8). `:rect` keeps the *substrate*
  FFT-diagonal; it does not buy the overlay a free pass.

## 9. Scope

Purely additive. New coupling symbol, new parameters only when selected, no
change to any existing default or code path. Acceptance: `wave_tests()` stays at
307/307 plus the new testset, `excitable_tests()` stays 36/36.

Touch list: `coupling in (...)` validation and the two new keywords in the
constructor; `Base.show`; `_dog_stencil_init` sibling for the 1-D seed;
`initialparameters`; `initialstates` (`qh`/`ones_w`); `_build_coupling`;
`_init_emission_threshold`; the docstring parameter/state tables; exports
unchanged.
