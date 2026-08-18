# Dispersion of the PhasorWaveSheet

A lattice-dynamics ("phonon") derivation of how waves propagate in
`PhasorWaveSheet`, for both coupling modes:

- **`:potential`** (full-state coupling) — a *linear* lattice with an exact
  dispersion relation `κ(q) = k + g·Ŵ(q)`.
- **`:spike`** (threshold-gated coupling, the default) — a *nonlinear*
  Sakaguchi–Kuramoto lattice; "dispersion" exists only as a **linearization**,
  either around a coherent reference (§3, a phase-mode Goldstone band) or — below
  the emission threshold `θ` — as the exact linear medium at gain `g/θ`
  (§3-quater).

Companion to `docs/rf_wave_network_implementation.md` and the notebook
`demos/wave_dispersion_derivation.ipynb` (numerical verification). Conventions:
lattice spacing 1, ring of `N` sites, `q ∈ (-π,π]`, `k = λ + iω`, per-period
propagator `A = e^{kT}`.

---

## 1. The sheet as a lattice

A site `n` holds a complex state `z_n(t)`. The continuous dynamics are a lattice
with an on-site oscillator plus range-dependent coupling:

$$\frac{dz_n}{dt} = k\,z_n \;+\; g\sum_m W_m\,c_{n-m},\qquad k=\lambda+i\omega,$$

where `c_j` is what a neuron *transmits* and `W_m = D(|m|)\,e^{-i\omega|m|/c}` is
the coupling: the difference-of-Gaussians magnitude `D(r)=A_E e^{-r^2/2\sigma_E^2}-A_I e^{-r^2/2\sigma_I^2}`
times the conduction-delay phase `e^{-i\omega|m|/c}`, with `W_0=0`. The two
transmission modes are

$$\text{`:potential`}:\ c_j = z_j, \qquad\qquad
  \text{`:spike`}:\ c_j = \frac{z_j}{\sqrt{|z_j|^2+\theta^2}}\;\xrightarrow[\ |z_j|\gg\theta\ ]{}\;\frac{z_j}{|z_j|}.$$

`\theta` is the **emission threshold** (§3-quater). Sections 2–3 below take the
`\theta\to 0` limit, i.e. `c_j = z_j/|z_j|`; §3-quater covers what finite `\theta`
changes, which is most of the observable behaviour.

Solid-state dictionary: `iω` is an on-site (optical-mode) gap, `λ` an on-site
loss (non-Hermitian), `g W_m` complex hopping/force-constants, `q` the Brillouin
zone, and the coupling's spatial Fourier transform is the "structure factor."

---

## 2. Potential mode — the exact (phonon) dispersion

With `c_j = z_j` the coupling is linear, so plane waves `z_n = e^{iqn}\zeta(t)`
are exact eigenmodes. The convolution diagonalizes:

$$\sum_m W_m e^{iq(n-m)} = e^{iqn}\,\hat W(q),\qquad \hat W(q)=\sum_m W_m e^{-iqm},$$

giving a decoupled mode equation and the **dispersion relation**

$$\boxed{\;\frac{d\zeta_q}{dt} = \kappa(q)\,\zeta_q,\qquad \kappa(q)=k+g\,\hat W(q)\;}$$

Split into growth and frequency:

$$\kappa(q)=\underbrace{\lambda+g\,\mathrm{Re}\,\hat W(q)}_{\Gamma(q)\ \text{(growth/decay)}}\;+\;i\underbrace{\big[\omega+g\,\mathrm{Im}\,\hat W(q)\big]}_{\Omega(q)\ \text{(band)}},\qquad z_n(t)\propto e^{\Gamma(q)t}\,e^{i(qn+\Omega(q)t)}.$$

Because the delay depends on **distance** `|m|`, `W_m` is complex but even, so
`\hat W(q)=2\sum_{m\ge1}D(m)e^{-i\omega m/c}\cos(qm)` and

$$\Gamma(q)=\lambda+2g\!\sum_{m\ge1}\!D(m)\cos\!\tfrac{\omega m}{c}\cos(qm),\qquad
\Omega(q)=\omega-2g\!\sum_{m\ge1}\!D(m)\sin\!\tfrac{\omega m}{c}\cos(qm).$$

This separates the two "modifications" cleanly:

- **Delays create the dispersion.** No delay (`c\to\infty`) ⇒ `\sin(\omega m/c)\to0` ⇒ `\Omega(q)=\omega` flat (dispersionless standing modes, zero group velocity). Finite `c` bends the band ⇒ traveling waves.
- **The DoG shapes the gain band.** `\Gamma(q)` is the DoG's band-pass profile; its peak `q^\*` is the fastest-growing wavelength (a ring in 2-D).

**Velocities.** With our `e^{+i\Omega t}` rotation convention (so constant-phase
surfaces obey `qn+\Omega t=\text{const}`):

$$v_p(q)=-\frac{\Omega(q)}{q},\qquad v_g(q)=-\frac{d\Omega}{dq}.$$

(Flip `\Omega\to-\Omega` for the textbook `e^{-i\Omega t}` form and `v_g=+d\Omega/dq`.)

**Stability / criticality.** Mode `q` grows iff `\Gamma(q)>0`; the sheet is
critical when `\max_q \Gamma(q)=0`. In discrete time the per-step multiplier is
`M(q)=A+g\hat W(q)` (`dispersion(...; mode=:discrete)`), differing from the exact
per-period factor `e^{\kappa(q)T}` (`mode=:continuous`) only at `O(gT)`. Spectral
radius `\max_q|M(q)|` is critical at 1.

**2-D.** Identical with `\mathbf q=(q_x,q_y)` and `\hat W(\mathbf q)=\mathrm{fft2}(W)`;
`\kappa(\mathbf q)=k+g\hat W(\mathbf q)`, `\mathbf v_g=-\nabla_{\mathbf q}\Omega`.
This `\hat W(\mathbf q)` is exactly what `dispersion()` returns as `W_hat`.

> **The 2-D velocity trap.** `\mathbf v_g` is a *vector*. For the isotropic `:dog`
> kernel the band depends only on `|\mathbf q|`, so `\mathbf v_g = v_r(|q|)\,\hat q`
> — it points radially, and **averaging the vector over a ring of constant `|q|`
> gives exactly zero** (verified to 4·10⁻⁸ at every `|q|`). Every outward mode is
> cancelled by its opposite-travelling partner. That zero is a property of the
> concentric geometry, not of the wave, and reading it as "the wave is not moving"
> is wrong. The quantity that is not zero, and that *is* the rate the wavefront
> expands, is the radial projection taken **before** averaging:
>
> $$v_r(|q|)=\big\langle \mathbf v_g\cdot\hat q\big\rangle_{\text{annulus}}=-\frac{d\Omega}{d|q|}.$$
>
> `radial_band` returns this, alongside the vector average as `v_vec` so the null
> is visible rather than silent. Compute the derivative branch-safely as
> `-\mathrm{Im}(\nabla M/M)`; no phase unwrapping is then needed even if `\arg M`
> approaches `\pm\pi`.
>
> Two scalars that look like wave speeds but are not: the **RMS radius** slope is a
> second moment of the whole sheet (window-dependent, ≈⅓ of any level-set speed),
> and the **level-set front speed** depends on the threshold you pick — at
> criticality it reads 0.65 sites/period at 0.1% of peak and 0.09 at 30%. When the
> medium is marginal there is no amplification to lock a rigid front, so the packet
> spreads dispersively and every level set moves at its own speed. The band
> `v_r(|q|)` is the well-defined answer; a single scalar is not.

**Verified** (`demos/wave_dispersion_derivation.ipynb`): a pure Fourier mode
scales by exactly `M(q)=A+g\hat W(q)` (`|Δ|≈6·10⁻⁸`), and a wave-packet centroid
travels at `-d(\arg M)/dq` (magnitude match to ~3 sig figs).

---

## 3. Spike mode — the Kuramoto linearization

With `c_j=z_j/|z_j|` the coupling is nonlinear, so plane waves are **not**
eigenmodes: there is no single global `κ(q)`. Write `z_n=\rho_n e^{i\theta_n}`;
the spike is `s_n=e^{i\theta_n}` (amplitude-blind). Separating real/imaginary:

$$\dot\rho_n=\lambda\rho_n+g\!\sum_m|W_m|\cos(\theta_{n-m}-\theta_n+\psi_m),\qquad
\dot\theta_n=\omega+\frac{g}{\rho_n}\!\sum_m|W_m|\sin(\theta_{n-m}-\theta_n+\psi_m),$$

with `\psi_m=\arg W_m=-\omega|m|/c`. The phase equation is a **Sakaguchi–Kuramoto
lattice**: oscillators at frequency `ω`, coupled by the DoG `|W_m|` through
`\sin` of phase differences, with the conduction delay reappearing as the
**phase-frustration** `\psi_m`. Amplitude does not propagate — it is slaved to
the phase pattern (`\rho` relaxes at rate `|\lambda|`).

### Nonlinear normal mode (the reference)

A uniform-amplitude wave `z_n^0=r\,e^{i(qn+\Omega t)}` is an *exact* solution when

$$r=\frac{g\,\mathrm{Re}\,\hat W(q)}{|\lambda|},\qquad \Omega=\omega+\frac{g}{r}\,\mathrm{Im}\,\hat W(q).$$

The amplitude self-selects so the leaky loss `\lambda r` balances the
**fixed-magnitude** spike drive `g\,\mathrm{Re}\hat W(q)`. Equivalently, in a
uniform-amplitude state the coupling collapses to the linear form with an
**effective gain `g/r`**: `\kappa_{\text{eff}}(q)=k+(g/r)\hat W(q)`. Since
`\mathrm{Re}\,\kappa_{\text{eff}}=0` at this `r`, the amplitude **self-tunes the
dominant mode to marginal stability** (automatic gain control), on top of the
hard BIBO bound `|z|\le g\sum_m|W_m|/(1-|A|)` from unit-magnitude spikes.

### Linearization → the phase-mode band

Perturb `\theta_n=qn+\Omega t+\eta_n`, `\rho_n=r+\delta_n`. To first order,
Fourier mode `q'` obeys a `2\times2` band (reference `q=0`, where the entries are
real):

$$\nu\begin{pmatrix}\hat\eta\\ \hat\delta\end{pmatrix}=
\begin{pmatrix}\tfrac{g}{r}\big(C(q')-\mathrm{Re}\hat W(0)\big) & -\tfrac{g}{r^2}\mathrm{Im}\hat W(0)\\[2pt]
-g\big(S(q')-\mathrm{Im}\hat W(0)\big) & \lambda\end{pmatrix}
\begin{pmatrix}\hat\eta\\ \hat\delta\end{pmatrix},$$

with `C(q')=\sum_m \mathrm{Re}(W_m)e^{-iq'm}` and `S(q')=\sum_m \mathrm{Im}(W_m)e^{-iq'm}`
(both real & even). The **phase (Goldstone) branch** is the eigenvalue with
`\nu\to0` as `q'\to0`; the other branch is the fast amplitude relaxation `\approx\lambda`.

**Clean closed form (no delay, `q=0`).** Then `\mathrm{Im}\hat W(0)=0`, the
matrix is diagonal, and the phase branch is exact:

$$\boxed{\;\nu(q')=\frac{g}{r}\big(\hat D(q')-\hat D(0)\big)=|\lambda|\Big(\frac{\hat D(q')}{\hat D(0)}-1\Big)\;}$$

a phase-diffusion band: `\nu(0)=0` (global phase shift is free), and its shape
follows the DoG's Fourier transform. If `\hat D` is band-pass (Mexican hat) it
peaks at `q^\*\ne0`, so `\nu(q^\*)>0`: the synchronized state is **unstable to
pattern formation** at wavelength `q^\*` (a Turing-type instability) — the system
spontaneously forms the traveling wave.

**Verified** (`demos/wave_dispersion_derivation.ipynb`): direct simulation of the
nonlinear spike lattice matches `\nu(q')=|\lambda|(\hat D(q')/\hat D(0)-1)` to
`~10^{-4}` across the band (no-delay, `q=0`); the delayed case follows the `2\times2`
eigenvalue in trend and magnitude (exact isolation needs projecting onto the
dominant eigenvector).

### Why it is only a linearization

- **Reference-dependent:** the band depends on the operating amplitude `r` (via
  the effective gain `g/r`) and reference `q`; there is a *family* of bands, not
  one exact `κ(q)`.
- **Small-perturbation only:** valid near a coherent state, not for large or
  incoherent (turbulent) regimes.
- **Different object:** it describes *phase* fluctuations of a Kuramoto lattice
  (a Goldstone band), not the amplitude-carrying phonon band of §2.

**2-D** is identical with `\mathbf q'` and `\hat D(\mathbf q')=\mathrm{fft2}(D)`.

---

## 3-bis. Standing rings vs traveling waves

A dispersion relation says which modes grow. It does **not** by itself say whether
an impulse produces a wave that travels. That is decided by one number, and for a
long time the sheet was on the wrong side of it.

### The criterion

The kernel is reflection-symmetric, `W_m = W_{-m}`, so `\hat W(q)` is even, so
`\Gamma(q)` is even: **modes at `+q` and `-q` are amplified identically.** A
localized impulse therefore always excites counter-propagating pairs at the
selected wavenumber `q^\*`. Their group velocities are equal and opposite
(`\Omega` even ⇒ `v_g` odd). So:

| | outcome |
|---|---|
| `v_g(q^\*) \ne 0` | the two packets **separate** → expanding ring of traveling wave |
| `v_g(q^\*) = 0`   | they never separate, overlap forever and interfere → **standing rings** |

In 2-D the ring `|q| = q^\*` contains modes pointing in every direction, so when
`v_r(q^\*) \ne 0` their packets fan out isotropically and the observable is a
concentric ring expanding at `|v_r(q^\*)|`.

### Weak delay forces the degenerate case

This is not a parameter accident. Expanding §2's sums to first order in
`\omega m/c` (weak delay: `\cos(\omega m/c)\to 1`, `\sin(\omega m/c)\to \omega m/c`):

$$\Gamma(q)\approx\lambda+g\,\hat D(q),\qquad
  \Omega(q)\approx\omega-g\tfrac{\omega}{c}\,\widehat{mD}(q).$$

Both are cosine transforms of nearly the same band-pass radial profile (`D` and
`mD` share their dominant length scale), so `\arg\max\Gamma` lands on an extremum
of `\Omega`, and `v_g = -d\Omega/dq` vanishes exactly where the gain peaks.
**Standing rings are the generic weak-delay outcome for any Mexican-hat kernel.**

### The matched conduction speed

Escaping requires enough delay to decouple the two transforms — the `\cos` and
`\sin` weightings must become genuinely different functions of `m`. That happens
once the conduction phase across the kernel's own width reaches half a cycle:

$$\boxed{\;\phi=\frac{\omega\,\sigma_I}{c}\approx\pi
  \qquad\Longleftrightarrow\qquad c\approx\frac{2\sigma_I}{T}\;}$$

Verified over `\sigma_I \in [1.5, 9]` (a 6× range), the `c` maximizing
`|v_g(q^\*)|` sits at `\phi/\pi = 1.17, 0.99, 1.15, 1.10, 1.09, 1.08`. The selected
wavelength tracks `\lambda^\* \approx 3\sigma_I`. **`\sigma_I` and `c` are not
independent** — `\sigma_I` sets the wavelength, and `c` is then forced.

At the default shape (`\sigma_I = 3`, `T = 1`), on a 48×48 sheet:

| | `c` | `\phi` | `g_\text{crit}` | `v_r(q^\*)` | transport ratio | verdict |
|---|---|---|---|---|---|---|
| previous default | 40 | 0.15π | 0.026 | +0.010 | 0.028 | standing |
| matched | 6 | 1.00π | 0.069 | −0.704 | 0.995 | **traveling** |

Confirmed by simulation on a 192² sheet. At `c=40` the ring crests are **pinned** —
`r = 6, 10, 16, \dots` stay put for hundreds of steps while new crests merely appear
at the growing edge, and the amplitude maximum never leaves the origin. Only the
boundary of the patterned disc creeps outward, and it *decelerates* (0.47
sites/period over `t=10\!-\!40`, 0.09 over `t=200\!-\!300`) as the transporting
low-`q` shoulder damps away. That is a standing pattern filling an expanding disc,
not a ripple. At `c=6` the radial-profile peak itself marches
`0.5 \to 25.5 \to 53.5 \to 96.5` at 0.71 sites/period, matching
`|v_g(q^\*)| = 0.704`.

> Worth knowing when reading an animation: at `c=40` the *phase* still advances at
> `v_p \approx 0.08` sites/period through the stationary crests. A phase-coloured
> render (`angle(z) .* mag`, as in `traveling_wave.gif`) therefore looks like a
> traveling wave even when the envelope is frozen. The envelope tracks `v_g`; only
> the phase tracks `v_p`. This is how the standing regime went unnoticed.

The traveling window is **narrow**: transport ratio ≥ 0.5 only for
`c \in [3.2, 6.8]`, falling off a cliff above `c \approx 7`. This is a resonance,
not a plateau — relevant to hardware tolerance.

### Damping sets the speed

`\phi=\pi` maximizes the *absolute* speed at the library's damping. The **ratio** of
wave speed to `c` is set by `\lambda`, not by kernel geometry. Fixing the DoG shape
and re-optimizing `c` at each `\lambda` (measured on 160², not just the band):

| `\lambda` | `c_\text{opt}` | band speed | measured `v` | `v/c` | `\tau = T/\lambda` at 25 ms |
|---|---|---|---|---|---|
| 0.15 (default) | 5.54 | 0.85 | 1.06 | 0.19 | 167 ms |
| 0.50 | 4.07 | 1.91 | 1.61 | 0.40 | 50 ms |
| 1.20 | 3.23 | 1.85 | 2.19 | 0.68 | 21 ms |
| ≥ 2.0 | — | *no selected traveling mode exists* | | | |

`\lambda \approx 1.2` puts the implied membrane time constant at ~21 ms, inside the
cortical 10–30 ms range, where the default 0.15 implies 167 ms. The `\phi=\pi` rule
drifts with damping (`\phi \approx 1.08\pi` at `\lambda=0.15`, `1.86\pi` at 1.2).

### Diagnostics

`wave_transport(l, ps, st)` returns the verdict as
`\text{transport ratio} = |v_r(q^\*)| / \max_q|v_r(q)|` — how much of the band's
available transport the *selected* mode actually carries — plus `delay_phase`
(`\phi`) and the critical mode located over the **full 2-D Brillouin zone**.
That last point matters: `dispersion_diagnostics` reads a transverse-DC axis slice,
and at the matched speed the critical mode is **off-axis** (diagonal), so the slice
misreports it. Evaluate at criticality: well above it the DC mode wins outright and
*any* sheet reports `:standing`.

---

## 3-ter. What `c` is, and what it is not

`c` enters **only** as the phase factor `e^{-i\omega r/c}` on a coupling that is
otherwise **instantaneous** — one FFT convolution per step. There is no time lag:
every site inside the kernel's numerical support is driven on the very next step,
whatever `c` is. This is deliberate (it keeps the sheet linear, FFT-diagonal and
differentiable — no delay-ODE solver), but two consequences follow.

**`v/c` is not the cortical ratio.** Cortex has wave speed ≈ conduction speed
because activity *relays* down axons; here `c` only reshapes the band and the wave
speed is an emergent group velocity. The two ratios measure different things.

**`:spike` had no relay front — at the legacy threshold.** Spike emission is
amplitude-blind, so any site above the emission threshold `\theta` emits a *full*
unit spike, which drives its neighbours at full strength. With the old hardcoded
`\theta = \sqrt{\varepsilon} = 10^{-4}` this meant essentially no threshold at all,
and measurements on 160² showed an accelerating cascade rather than a front
(`r = 3, 5, 17, 113` over four steps at `g=0.1`). A fine gain scan found no
constant-speed window — `g \le 0.025` dies, `g \ge 0.030` ignites the whole sheet in
~4 steps — and the speed was **flat in `c`** (27.4–27.6 sites/period from `c=6` to
`c=100`), i.e. effectively instantaneous. In that regime spike mode is a bistable
switch, not a wave medium.

> **Superseded in part by §3-quater.** `\theta` is now an exposed parameter with a
> derived default `\approx 1.4\,g\max_q|\hat W(q)|`, five orders of magnitude above
> the legacy value. The cascade above is what the sheet does at
> `\theta = 10^{-4}`, not what it must do. The conclusion that `c` is a phase and
> not a transport delay is unaffected — the coupling is still one instantaneous FFT
> convolution per step, so there is still no relay in the axonal sense. What changes
> is that the sheet now has a genuine quiescent state and a subthreshold linear
> regime, so §3-bis's transport analysis becomes applicable to `:spike` below
> threshold rather than to `:potential` only.

A genuine relay mechanism would still need real time-delayed coupling (a ring buffer
of past states), which the FFT-diagonal design deliberately avoids.

---

## 3-quater. The emission threshold

`:spike` transmission is

$$s_n = \frac{z_n}{\sqrt{|z_n|^2+\theta^2}},$$

the `\varepsilon` of `normalize_to_unit_circle` with `\varepsilon=\theta^2`. Until
recently `\theta` was a hardcoded numerical guard at `10^{-4}`. It is not a guard —
it is the sheet's **firing threshold**, and it is the single parameter that decides
whether `:spike` is a wave medium or a switch.

### `\theta` interpolates between the two transmission modes

$$|z|\gg\theta:\ s\approx z/|z| \quad\text{(hard spike)},\qquad
  |z|\ll\theta:\ s\approx z/\theta \quad\text{(\emph{linear})}.$$

The second limit is the important one: below threshold the spike sheet **is** the
`:potential` medium of §2 at effective gain `g_{\text{eff}} = g/\theta`. Verified by
sweeping `g/(\theta g_{\text{crit}})` through 1, which reproduces the criticality
knife edge (peak `|z|` at `t=80`: `0.5\to 5.8\!\times\!10^{-6}`,
`0.99\to 3.5\!\times\!10^{-3}`, `1.01\to 5.0\!\times\!10^{-3}`, `2.0\to 23.5`).

This replaces §3's `g/r` Kuramoto linearization — whose scale involved an unknown
mean-field amplitude `r` — with a computable one, and it is what makes
`dispersion`, `radial_band` and `wave_transport` meaningful on a spike sheet.
`dispersion` now applies `g_{\text{eff}}` automatically for `transmit = :spike`.

### Where the threshold has to sit

Bisecting the flood/extinct boundary `\theta^\*` on a seeded impulse (`N=64`,
`:dog`, matched `c`):

| `\lambda` | `\theta^\*/g` at `g=0.3` | `g=1` | `g=3` | `\theta^\*/(g\max\|\hat W\|)` |
|---|---|---|---|---|
| 0.15 | 8.750 | 8.797 | 8.815 | 1.164 |
| 0.50 | 6.178 | 6.179 | 6.179 | 0.818 |
| 1.50 | 5.337 | 5.338 | 5.338 | 0.706 |

`\theta^\* \propto g` **exactly**. So the derived reference is

> $$\boxed{\;\theta_{\text{ref}} = g\,\max_q|\hat W(q)|\;}$$

— the drive a site receives when its entire neighbourhood emits unit spikes in
phase, i.e. *fire only on near-maximal local coherence*. It is stable in grid size
(`\theta^\*/\theta_{\text{ref}} = 1.10` at `N=16`, then 1.13–1.17 for `N\ge 24`;
`\max|\hat W|` has converged to 4 s.f. by `N=32`) and holds for the other band-pass
couplings (`:aniso` 1.28, `:stencil` 1.34). It is **not** a criticality reference
for `:shift`, which is a unit-gain phase ramp (`|\hat W|\equiv 1`) with no
amplifying band.

The reference sits just *below* the boundary, so the layer default is
`1.4\,\theta_{\text{ref}}`. Using `\theta_{\text{ref}}` unmultiplied lands in the
saturating basin: the sheet floods and homogenizes (`\mathrm{std}|z| \to 10^{-4}`),
after which no threshold can recover structure.

### Why a constant `\theta` is not enough

The band of `\theta` giving partial (1–99%) activity is only **1.26–1.33× wide**,
and it *moves* as `\theta^\*\propto g`. A fixed `\theta` is wrong the same way the
old fixed `c=40` default was wrong: correct at one operating point, silently out of
regime once `g` trains. That, not biological realism, is the argument for
homeostasis.

### Two timescales, and why one is not enough

`homeostasis = :global` adapts a scalar `\theta_g` toward a target firing fraction.
It tracks `g` exactly (`N=48`, target 2%):

| `g` | 0.1 | 1.0 | 10.0 |
|---|---|---|---|
| final `\theta_g` | 1.174 | 11.895 | 112.5 |
| `\theta_g/g` | 11.74 | 11.90 | 11.25 |
| firing rate | 1.82% | 2.27% | 1.94% |
| `\mathrm{std}\|z\|/\mathrm{mean}\|z\|` | 0.49 | 0.53 | 0.49 |

A **per-site** homeostat *alone* is structurally broken: it regulates the mean rate
perfectly but the step-to-step overlap of the active set is exactly **0.000** at
every rate `\eta\in[0.02,1.0]` and every target `\in[0.005,0.4]`. It can only hit
its target by silencing whoever just fired, which is anti-persistence by
construction — rate-regulated flicker, not a wave.

`homeostasis = :local` therefore keeps the global term and adds a *slow, zero-sum*
per-site term (geometric mean pinned to 1, so it carries only the spatial pattern of
excitability and leaves the level entirely to `\theta_g`). That is refractoriness,
and it is what makes the packet travel:

| `\eta_{\text{local}}` | 0 | 0.005 | 0.01 | 0.02 | 0.03 | 0.05 |
|---|---|---|---|---|---|---|
| net centroid drift / 300 steps | 101 | 67 | **360** | **306** | 160 | 62 |
| step-to-step overlap | 0.962 | 0.790 | 0.716 | 0.704 | 0.539 | 0.357 |

Global-only gives a *standing* localized blob; `\eta_l\in[0.01,0.02]` roughly triples
the drift while keeping the active set coherent (overlap ≈0.7). Beyond that it
decoheres. This is the same role adaptation plays in Ermentrout & Kleinfeld, but
implemented as intrinsic excitability rather than a phase offset, so it cannot
spuriously rotate phase the way `use_adaptation`'s `z \leftarrow z - ia` can.

### The fire indicator must be hard in the forward pass

The homeostat needs a differentiable activity measure, but **any** smooth pointwise
indicator makes its target degenerate: a mean cannot distinguish "5% of sites at 1"
from "100% of sites at 0.05". Two measured failures:

- Emit magnitude `|z|/\sqrt{|z|^2+\theta^2}`: target met exactly, **zero** sites
  firing, zero transport.
- Sigmoid `\sigma((|z|-\theta)/(\beta\theta))` at `\beta=0.05`, started from
  `\theta_{\text{ref}}`: converges to `\theta=9.402`, soft-fire `=0.0500` (on
  target), **hard fire `=0.0000`**, `\mathrm{std}|z|=0.033` — a uniform field
  sitting 2.94 sigmoid-widths below threshold, whose tail integrates to the target.
  Shrinking `\beta` moves that attractor closer to `\theta` but never removes it.

The fix is a straight-through estimator — hard `|z|>\theta` forward, sigmoid
backward — the same idiom `moe_gate` uses. `\beta` then only sets the gradient scale
and reads physically as threshold jitter.

Two further calibration points, both measured: the target must be **2%, not 5%**
(at `target=0.10, \eta_g=0.15` the controller drives `\theta` below the flood
boundary and the sheet collapses), and a restoring prior pulling `\theta` toward
`\theta_{\text{ref}}` must **not** be added — it buys basin-robustness at `\lambda
=0.15` and permanently kills the sheet at `\lambda\ge 0.5`, where the correct
`\theta` is on the other side of the prior.

> **Scope.** Everything above was measured at the default `\lambda=0.15`, across
> `g\in[0.3,3]`, `N\in[48,96]`, and couplings `:dog`/`:aniso`/`:stencil`. At
> `\lambda\ge 0.5` the homeostat still regulates the rate to target but does so
> with a *synchronously bursting* sheet rather than a structured one. Hitting the
> target rate is necessary, not sufficient — check `\mathrm{std}|z|` too, which is
> what `wave_homeostat_trace` returns it for. Note also that the sheet has **no
> random initialization** (all parameters come from `init_*` constants), so
> seed-to-seed variation is not a meaningful robustness axis; vary the initial
> condition instead.

### Does `:spike` produce a traveling wave? Yes — and there are two of them

`wave_transport` reports `:traveling`, ratio 0.927, for the default spike sheet
— a verdict it could not produce at all before, since it is built on
`M = A + g_{\text{eff}}\hat W`. A seeded impulse (512², `demos/wave_spike_propagation.jl`)
then shows two distinct fronts:

| phase | what it is | speed |
|---|---|---|
| subthreshold, `t < 83` | concentric `\|z\|` ring, nothing has fired; the **linear** medium at `g/\theta` | `+0.882 / +0.914 / +0.942` for `:none`/`:global`/`:local`, vs band prediction **0.855** |
| suprathreshold, `t \ge 83` | **firing** ring: closed, thin, constant-speed | `+1.844`, fit residual 4.59 sites over 150 sites of travel |

The subthreshold ring is *identical* to a `:potential` sheet at the same
`g_{\text{eff}}`, and its ~3% agreement with the band is a direct validation of
the `g/\theta` linearization. The firing ring is 2.1× faster — a **nonlinear
ignition front**, not the linear group velocity, which is the expected
pushed-vs-pulled distinction for an excitable medium. Quoting one number as
"the" spike-mode wave speed conflates them.

**Sustained**, measured every 10 steps from `t = 100` to the wrap at `t = 177`:

| `t` | 100 | 110 | 120 | 130 | 140 | 150 | 160 | 170 |
|---|---|---|---|---|---|---|---|---|
| mean radius | 104.9 | 125.0 | 143.4 | 159.6 | 176.3 | 191.0 | 211.2 | 236.2 |
| radial `sd/mean` | 0.144 | 0.056 | 0.047 | 0.053 | 0.067 | 0.075 | 0.063 | 0.044 |
| sectors lit / 72 | 72 | 72 | 72 | 72 | 72 | 72 | 72 | 72 |
| firing | 3.4% | 3.4% | 3.2% | 3.5% | 3.6% | 4.0% | 3.0% | 3.5% |

Closed, thin and at steady firing rate for 130 sites of travel. Three
independent things have to hold for this and they fail differently — a ring can
stay closed while smearing into a disc, or stay thin while fragmenting into
arcs, so `ring_quality` in the demo reports all three.

**Homeostasis makes the ignition time sheet-size independent.** Because the
controller lowers `\theta` toward a field that has not reached it yet, first
spike lands at `t = 83` on both 256² and 512². Without it, ignition waits for
the field to grow to a fixed `\theta` and the delay scales with the sheet:
`t = 133` at 256², `t = 217` at 512².

> **Sheet size is load-bearing for this measurement.** The ring self-intersects
> once its radius passes `N/2`. On 128² that happens at about the moment firing
> starts, so the firing set is measured *after* it has wrapped and reads as
> disconnected arcs (17–20 of 36 sectors) with a ring speed of `+0.565` instead
> of `+0.88`. Both look exactly like real negative results. Derive the window
> from whichever front reaches `N/2` first — after ignition that is the firing
> front, not the `\|z\|` peak, and keying off the latter silently labels ~15
> wrapped steps as clean.

**Charging transient.** A real threshold means the sheet starts silent. From
`z=0` under unit-magnitude drive, `|z|` saturates at `1/(1-|A|)\approx 7.2`, which
is *below* the default `\theta\approx 10.6` — firing needs the recurrent
amplification to build, ~50 steps on a fresh sheet. Short rollouts therefore see no
spikes at all, and `\eta_l` in particular receives exactly zero gradient (correct:
refractoriness has nothing to modulate). Budget `L\gtrsim 60`, or lower
`init_theta_frac`.

---

## 4. What `dispersion()` computes

`dispersion(layer, ps, st)` returns `\hat W=\mathrm{fft2}(W)` and
`M=A+g\hat W`, `\kappa=k+g\hat W`. So:

- For **`:potential`** it is the *exact* dispersion (§2).
- For **`:spike`** it substitutes the *subthreshold* effective gain
  `g_{\text{eff}} = g/\theta` (§3-quater) and is then the exact dispersion of the
  linear medium the sheet reduces to for `|z|\ll\theta`. Above threshold the emit
  saturates and this stops applying; the relevant object there is the `g/r`
  Kuramoto linearization of §3, whose scale factor is not computable. Compare
  `exp(ps.log_theta)` against `emission_threshold(l, ps, st)` before trusting these
  numbers on a firing sheet.

The rest of the analysis surface, all exported from `src/wave.jl`:

| function | returns | use for |
|---|---|---|
| `dispersion` | `M`, `k_eff`, `W_hat`, `spectral_radius` | the raw per-mode map; criticality |
| `dispersion_diagnostics` | `q`, `growth`, `v_g`, `gvd`, `gain_curv` | 1-D band along one axis; GVD / packet spreading. **Axis slice — misses an off-axis critical mode** |
| `radial_band` | `q`, `gain`, `v_r`, `v_vec` | ring-averaged 2-D band; the correct velocity metric |
| `wave_transport` | `transport_ratio`, `verdict`, `delay_phase`, `q_star` | **the guard**: standing vs traveling, from the true 2-D critical mode |
| `matched_conduction_speed` | `c = 2σ_I/T` | the design rule of §3-bis; the layer's derived `init_log_speed` default |
| `emission_threshold` | `θ_ref = g·max_q\|Ŵ(q)\|` | the design rule of §3-quater; the layer's derived `init_log_theta` default (×`init_theta_frac`) |
| `wave_homeostat_trace` | `theta_g`, `fire`, `std_abs`, `theta_l_spread` | watch the homeostat. `θ` is rollout-local, so without this a controller stuck on the wrong branch is invisible |
| `zero_gvd_speed` | `c` at `β₂(q^\*) = 0` | the zero-dispersion transport regime |

A note on tiers at the matched operating point: the discrete recurrence
(`M = A + g\hat W`) and the ODE (`M = e^{(k+g\hat W)T}`) differ at
`O(g\,T)`, and because `k` and `\hat W` commute the exact per-period map is
`A\,e^{g\hat W T}`, so the gap is `\approx (1-A)\,g\hat W` per step. The matched
regime needs `g_\text{crit} \approx 0.069` instead of `0.026`, and that 2.7× gain
turns a negligible per-step gap into a visible one: field similarity at `t=60`
falls from 0.9997 (`c=40`, `g_\text{crit}=0.026`) to 0.56 (`c=6`,
`g_\text{crit}=0.069`). It tracks `g_\text{crit}`, not `c` — at `c=12`
(`g_\text{crit}=0.016`) it is 0.9995. **Do not assume long discrete and ODE
rollouts stay aligned here**; use `mode = :continuous` when reasoning about Tier 2.

---

## References

- Lattice dynamics / phonons: any solid-state text (Ashcroft–Mermin ch. 22).
- Sakaguchi & Kuramoto (1986), *A soluble active rotator model showing phase transitions via mutual entrainment* — the phase-frustrated Kuramoto model the spike sheet realizes.
- Muller, Chavane, Reynolds & Sejnowski (2018), *Cortical travelling waves: mechanisms and computational principles*, Nat. Rev. Neurosci. 19:255–268 ([PMC5933075](https://pmc.ncbi.nlm.nih.gov/articles/PMC5933075/)) — identifies distance-dependent conduction delay as "the key factor governing the generation and propagation" of mesoscopic cortical waves (not adaptation, not frequency gradients); reports unmyelinated horizontal fibres at 0.1–0.8 m/s, which at ~0.5 mm/site and a 25 ms carrier maps to `c ≈ 5–40` sites/period — the matched value sits at the typical end, the previous default of 40 at the very fastest.
- Ermentrout & Kleinfeld (2001), *Traveling electrical waves in cortex: insights from phase dynamics and speculation on a computational role*, Neuron 29:33–44 — coupled phase-oscillator route to the same phenomenology; the adaptation-driven mechanism that §3-quater's per-site threshold term plays the role of.
- Turrigiano (2011), *Too many cooks? Intrinsic and synaptic homeostatic mechanisms in cortical circuit refinement*, Annu. Rev. Neurosci. 34:89–103 — the biological two-timescale split §3-quater mirrors: global synaptic scaling for the activity set-point, local intrinsic excitability for the per-cell pattern.
- `src/wave.jl` (implementation), `docs/rf_wave_network_implementation.md` (design), `demos/wave_dispersion_derivation.ipynb` (verification), `demos/wave_dispersion.jl` (the shipped band-diagram demo).
