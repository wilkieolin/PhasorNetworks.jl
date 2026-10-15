# FPE encoder and resonator (A3-min)

Branch `fpe-resonator`, 2026-10-14. Implements the residue (grid) code natively:
a phasor's phase is a residue mod 2π, fractional power encoding (FPE) with K
root-of-unity bases is a K-modulus residue code, `v_bind` is carry-free
addition, and a resonator network factors a bound address back into its
residues. Source: `src/fpe.jl`; tests `test/test_fpe_resonator.jl`; full-size
numbers from `scripts/fpe_resonator_sweep.jl` (CSVs in
`results/fpe_resonator/`). The predictions tested are §3(e) of the ARC note
`grid_code_addressing.md` (Frady et al. 2021; Frady, Kent, Olshausen & Sommer
2020; Kymn et al. 2024; Fiete / Sreenivasan).

## API

```julia
enc = FPEEncoder(in_dims => D; mode = :residue, moduli = [m₁, …, m_K],
                 exponent = :random, layout = :bound, random_offset = false)
enc = FPEEncoder(1 => D; mode = :uniform, bandwidth = B)
enc = FPEEncoder(2 => D; mode = :hex, moduli = [m], layout = :blocks, exponent = :unit)
ps, st = Lux.setup(rng, enc)             # no parameters; Ω, offsets, blocks in state
θ, _  = enc(x, ps, st)                   # x: (in_dims, B) real → θ: (D, B) Phase

cbs = fpe_codebooks(enc, st)             # one (D, m_k) codebook per (modulus, axis)
res = Resonator(enc, st; iterations = 50, projection = :abs)
rps, rst = Lux.setup(rng, res)
out, rst = res(θ, rps, rst)              # out.indices (F, B), .converged, .iterations, .similarity
x̂ = crt_decode(out.indices[:, b] .- 1, enc.moduli)
fpe_table(enc, st)                       # modulus ↔ ω ↔ block table
```

`FPEEncoder` maps `x ↦ θ_j = Σ_a Ω[j,a] x_a + ψ_j (mod 2π)`, output in the
library's `Phase` convention, so `v_bind`, `Codebook`, `PhasorDense` and
`MakeSpiking` accept it unchanged. Modes:

| mode | Ω | kernel | use |
|---|---|---|---|
| `:uniform` | `ω ~ U(−B, B)` rad per unit | `sinc(BΔx/π)`: similar nearby, orthogonal beyond ≈ π/B | local attention over position (Frady 2021) |
| `:residue` | `2π r_j / m_k`, `r_j = 1` (`:unit`) or `r_j ~ U{0..m_k−1}` (`:random`) | periodic in `m_k`; `:random` gives a delta kernel (quasi-orthogonal residues) | addresses, time tags |
| `:hex` | `:residue` on three 2-D axes `k₁+k₂+k₃ = 0` | hexagonal | 2-D positions |

Layouts: `:bound` (every component carries every factor; the output is the
bind of the per-factor codes, one dense address vector; decode with the
resonator) and `:blocks` (one block of `D/(K·A)` components per factor;
decode block by block with a `Codebook`). Inputs may be `(in_dims, B)` or
`(in_dims, L, B)`. Internally each axis projection is reduced mod `m_k` before
the multiply, so large integers stay exact in Float32 (homomorphism error
7×10⁻⁵ at x ~ 3×10⁴ at D = 1024).

`Resonator` (Frady et al. 2020): for each factor `f`, unbind the other factors'
estimates from the input, score against codebook `f`, project, and return to
the torus; sequential (Gauss–Seidel) sweeps by default, stop when no estimate
moves by more than `tol`. Projections:

- `:abs` (default) — `x̂_f ← ∠ C_f |C_fᴴ ê_f|`: codebook bundle weighted by the
  *magnitude* of each complex similarity, then per-component normalisation.
  This is the loop of `demos/03_resonator.ipynb`, lifted into the library.
- `:linear` — `x̂_f ← ∠ C_f (C_fᴴ ê_f)`, the update of Frady 2020 / Kymn 2024 as
  written.
- `:codeword` — snap to the best codeword (coordinate ascent).

The magnitude weighting matters: with random starts the other factors'
estimates carry an arbitrary global phase during the search, and `|⟨c, ê⟩|`
is invariant to it. Measured at D = 1024, σ_φ = 0.3 rad (table below):
`:abs` factors ∏m = 3.5×10⁵ at 100 %, `:linear` fails from 8×10⁴ (80 %), and
`:codeword` never exceeds a few percent. One subtlety found on the way: the
`:random` exponents must be drawn from all `m` roots of unity, not from the
units of ℤ_m — for even `m` every unit is odd, codeword `l + m/2` is exactly
`−c_l`, and the magnitude weights cancel the pair (0 % accuracy on
13·15·16·17·19 before the fix, 62 % after at D = 1024, 87 % at D = 2048).

## Residue mapping table

`fpe_table(enc, st)` prints one row per factor. For `FPEEncoder(1 => 1024;
moduli = [7, 11, 13], layout = :blocks)`:

| factor | modulus m | ω = 2π/m (rad / unit x) | units of π | block (components) |
|---|---|---|---|---|
| 1 | 7  | 0.8976 | 2/7  | 1:341 |
| 2 | 11 | 0.5712 | 2/11 | 342:683 |
| 3 | 13 | 0.4833 | 2/13 | 684:1024 |

With `layout = :bound` every row has block `1:1024`; with `exponent = :random`
component `j` of modulus `k` rotates at `r_jk · 2π/m_k`, `r_jk ∈ {0..m_k−1}`.
Unique range = ∏m = 1001 (coprime moduli; a warning is issued otherwise). ZM1's
modulus is the natural first entry.

## Measured results (`scripts/fpe_resonator_sweep.jl`, CPU, ≈ 6 min)

### (i) Range law — holds; K_eff/K = 0.79–0.97

Blocks layout, per-modulus codebook decode + CRT, addresses drawn from `[0, X)`
with `X` on a geometric grid (20 points/decade); range = largest `X` with
accuracy ≥ 99 %. All moduli ≤ 13 (`1/δφ = 13`).

| K | moduli | ∏m | measured range (D = 1024, σ = 0.3 / 0.6 / 1.0 rad) | K_eff = log R / log 13 | K_eff / K |
|---|---|---|---|---|---|
| 1 | 13 | 13 | 13 | 1.00 | 1.00 |
| 2 | 11·13 | 143 | 143 | 1.93 | 0.97 |
| 3 | 9·11·13 | 1 287 | 1 287 | 2.79 | 0.93 |
| 4 | 7·9·11·13 | 9 009 | 9 009 | 3.55 | 0.89 |
| 5 | 5·7·9·11·13 | 45 045 | 45 045 | 4.18 | 0.84 |
| 6 | 4·5·7·9·11·13 | 180 180 | 180 180 | 4.72 | 0.79 |

Range grows exponentially in K (ratio ≥ 4 per added modulus) and equals ∏m at
every noise level at D = 1024; at D = 256 the K = 6 code (32 components per
block) drops to 98 % at σ = 1.0 rad and the K = 5 code to 40 146. The exponent
K_eff ≈ 0.9K at K = 4 is exactly Fiete's N_eff; it is the arithmetic of
distinct coprime moduli below 1/δφ, not a decoding loss.

### (ii) Resonator vs codebook — holds; ratio 1 646× at D = 1024

Bound layout, σ = 0.3 rad per component, 200 random addresses, ≤ 200 sweeps.
"direct" = each modulus decoded against its codebook with no unbinding (what a
`Codebook` layer alone can do with a bound vector): 0–1 % everywhere, chance.
"flat" = a flat codebook holding as many stored vectors as the resonator
(Σm_k): it decodes its Σm_k addresses at 98–100 % and cannot represent more.
Ratio = addresses the resonator decodes / Σm_k.

| K | moduli | ∏m | Σm | D = 512 | D = 1024 | D = 2048 | sweeps (1024) | ratio (1024) |
|---|---|---|---|---|---|---|---|---|
| 3 | 23·29·31 | 20 677 | 83 | 1.000 | 1.000 | 1.000 | 7 | 249 |
| 3 | 41·43·47 | 82 861 | 131 | 1.000 | 1.000 | 1.000 | 11 | 633 |
| 3 | 53·59·61 | 190 747 | 173 | 1.000 | 1.000 | 1.000 | 17 | 1 103 |
| 3 | 67·71·73 | 347 261 | 211 | 0.890 | 1.000 | 1.000 | 27 | 1 646 |
| 3 | 79·83·89 | 583 573 | 251 | 0.560 | 0.995 | 1.000 | 42 | 2 313 |
| 3 | 97·101·103 | 1 009 091 | 301 | 0.155 | 0.875 | 1.000 | 94 | 2 933 |
| 3 | 127·131·137 | 2 279 269 | 395 | 0.000 | 0.290 | 0.935 | 169 | 1 673 |
| 2 | 499·503 | 250 997 | 1 002 | 1.000 | 1.000 | 1.000 | 13 | 251 |
| 4 | 13·17·19·23 | 96 577 | 72 | 1.000 | 1.000 | 1.000 | 19 | 1 341 |
| 4 | 23·29·31·37 | 765 049 | 120 | 0.270 | 0.845 | 0.990 | 98 | 5 387 |

Operational capacity (≥ 99 %) at D = 1024, K = 3: between 3.5×10⁵ and 5.8×10⁵
addresses with σ = 0.3 rad (Kymn Fig. 2a reports > 10⁵; noiseless 0.935 at
10⁶); at σ = 0.6 rad it is ≈ 3.5×10⁵, and it roughly doubles per doubling of D.
Projection comparison at D = 1024, σ = 0.3: `:linear` 100 % up to 2×10⁴,
80 % at 8×10⁴, 47 % for 13·17·19·23; `:codeword` ≤ 8 %.

### (iii) Hex vs square — 3m² states (not 3m²−3m+1), isotropy holds

Continuous positions in one period cell, unit-exponent blocks (one per axis),
each axis phase decoded to the nearest of m levels, D = 1536, 20 000 positions.

| m | code | states / period | 3m²−3m+1 | states / unit area | noise flip σ=0.1 / 0.2 | LS MSE | anisotropy (max/min directional MSE) |
|---|---|---|---|---|---|---|---|
| 5 | square (2 axes) | 25 | — | 1.000 | 0.004 / 0.009 | 0.166 | 1.57 |
| 5 | hex (3 axes) | 75 | 61 | 2.598 | 0.008 / 0.019 | 0.111 | 1.06 |
| 9 | square | 81 | — | 1.000 | 0.009 / 0.017 | 0.167 | 1.52 |
| 9 | hex | 243 | 217 | 2.598 | 0.015 / 0.030 | 0.110 | 1.03 |
| 13 | square | 169 | — | 1.000 | 0.012 / 0.025 | 0.168 | 1.51 |
| 13 | hex | 507 | 469 | 2.598 | 0.025 / 0.044 | 0.112 | 1.06 |

The three-axis code has exactly 3m² distinct decoded states per period
(the three families of decision lines are in general position because the
half-integer boundaries cannot concur when k₁+k₂+k₃ = 0), i.e. log₂3 = 1.58
bits more per modulus for 1.5× the channels, versus the note's 3m²−3m+1 (the
finite-window count, which the periodic code exceeds). Per unit area the hex
code holds 3√3/2 = 2.6× the states, its quantisation MSE is 0.67× the
square's, and the directional error is isotropic (1.03–1.10 vs 1.51–1.57,
4-fold). Its per-position noise flip rate is about twice the square code's
because its cells are 2.6× smaller.

### (iv) Error-correction margin — falsified in channel form, holds with ≥ 16 phasors per channel

2¹⁶ addresses, von Mises noise σ per phasor, 400 trials, nearest-*valid*-codeword
decoding in the wrapped-phase metric vs per-channel rounding + CRT, against a
16-ary phase per hex digit (4 phases = 2¹⁶). `R` = independently noisy phasors
per channel (circularly averaged before decoding); R = 1 is Fiete's channel
form, R ≈ D/K the vector form. Entries: largest σ (rad, 0.05 steps) with error
≤ 1 %.

| moduli (∏m) | R | residue, nearest codeword | residue, per-channel | 16-ary, per phase | 16-ary, address |
|---|---|---|---|---|---|
| 13·15·16·17·19 (1.0×10⁶) | 1 | 0.05 | 0.05 | 0.05 | 0.05 |
|  | 4 | 0.15 | 0.10 | 0.15 | 0.10 |
|  | 16 | **0.35** | 0.20 | 0.30 | 0.25 |
|  | 64 | 0.50 | 0.45 | 0.50 | 0.45 |
| 29·31·32·33 (9.5×10⁵) | 1 | 0.05 | < 0.05 | 0.05 | 0.05 |
|  | 16 | 0.20 | 0.10 | 0.25 | 0.25 |

Channel form (R = 1): at σ = 0.2 rad the residue code's address error is 45 %
(16-ary: 33 % per phase, 80 % per address); at 0.3 rad it is 81 %. Adjacent
addresses differ by one level in *every* residue, so the minimum distance of
the 2¹⁶-subset is √K·2π/m ≈ 0.9 rad against a noise norm of √K·σ ≈ 0.7 rad;
sparsity in ∏m buys no margin at this rate. The prediction "near-zero error up
to 0.3 rad where a 16-ary phase fails at 0.2" is falsified for one oscillator
per modulus. With 16 phasors per channel the residue code with cleanup
survives 0.35 rad at ≤ 1 % error where the 16-ary address code reaches 0.25
rad, and cleanup over the valid set beats per-channel rounding at every R
(0.35 vs 0.20 rad at R = 16). Vector form, bound into one D-vector and
factored by the resonator (same 5 moduli, 2¹⁶ addresses): 62 % at D = 1024
and 87 % at D = 2048 at σ = 0.3 rad, falling to 46 % / 77 % at 0.6 rad —
this 10⁶-state K = 5 code sits at the resonator's capacity edge for these D;
a 4-modulus code with ∏m ≈ 10⁵ (13·17·19·23) decodes 100 % at σ = 0.6 rad.

### (v) Round trip — exact

At D = 1024 (max arc error): `v_unbind(v_bind(a,b),b)` vs `a` 3.8×10⁻⁷;
`bind(enc(x), enc(y))` vs `enc(x+y)` 7.0×10⁻⁵ (x, y ≤ 2¹⁵);
phase → `MakeSpiking` → `train_to_phase` 0; `spike_phase_bind` of two encoded
spike trains vs `v_bind` of the phases 0.

Test runtime (`fpe_resonator_tests()`, D ≤ 256): 7.5–9 s; the sweep 6 min.

## Spiking path

Implemented:
- Encoder: `Chain(FPEEncoder(…), MakeSpiking(spk_args, repeats))` emits one
  spike per component per carrier cycle at time `(θ/2 + ½)T`; this is the
  library's encoder convention (real x → phase is the one off-torus operation
  and it sits at the input boundary). Test (v) checks the round trip and that
  `spike_phase_bind` (phase addition as a spike delay) reproduces the bound
  address exactly, so binding and unbinding of addresses are available in the
  spike domain today.
- `Resonator(x::SpikingCall)`: reads the last carrier cycle's spike times as
  phases (`train_to_phase`, a readout) and runs the phase-domain loop; the
  test checks it returns the same factors as the phase path.

Not implemented — what a cycle-by-cycle spiking resonator needs:
1. One resonator sweep per carrier cycle: estimates `x̂_f` as spike trains,
   unbinding by `spike_phase_bind(…; gain = −1)` (exists), similarity by
   resonant integration of the unbound train against each codeword (exists as
   `similarity_outer` on `CurrentCall`, ODE-based and slow; an event-driven
   per-cycle version analogous to `_spike_time_table` would be cheap).
2. The projection as a circuit: per factor, `m_f` codeword units each
   integrating `|⟨c, ê⟩|`, then re-emitting a spike per component at the phase
   of `Σ_l w_l c_l` (a weighted resonator bank, the same unit as
   `spike_phase_bundle` with `m_f` inputs). The magnitude weights are a
   per-unit amplitude readout, which the ground rule allows at a readout but
   which here sits inside the loop — the `:linear` projection (complex
   weights, no magnitude) is the torus-native alternative and costs ≈ 4× in
   capacity (table (ii)).
3. Convergence detection from spike-time stability across cycles (the `tol`
   test on consecutive cycles' tables).
Estimated size: ~150 lines in `src/ssm.jl` style; the two measured gaps to
decide first are the cost of `:linear` vs `:abs` in capacity and whether the
per-cycle integration noise (σ_φ ≈ T/20 from F9.4) keeps the loop inside the
σ = 0.3–0.6 rad band where table (ii) was measured.

## Files

`src/fpe.jl`, `test/test_fpe_resonator.jl` (wired into `test/runtests.jl`),
`scripts/fpe_resonator_sweep.jl`, `results/fpe_resonator/*.csv`, this file,
and a `fpe.jl` row in `CLAUDE.md`. Demo credit: the resonator loop first
appeared in `demos/03_resonator.ipynb` (ported from phasor_julia).
