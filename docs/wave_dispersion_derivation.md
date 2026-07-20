# Dispersion of the PhasorWaveSheet

A lattice-dynamics ("phonon") derivation of how waves propagate in
`PhasorWaveSheet`, for both coupling modes:

- **`:potential`** (full-state coupling) — a *linear* lattice with an exact
  dispersion relation `κ(q) = k + g·Ŵ(q)`.
- **`:spike`** (unit-magnitude coupling, the default) — a *nonlinear*
  Sakaguchi–Kuramoto lattice; "dispersion" exists only as a **linearization**
  around a coherent reference, giving a phase-mode (Goldstone) band.

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

$$\text{`:potential`}:\ c_j = z_j, \qquad\qquad \text{`:spike`}:\ c_j = \frac{z_j}{|z_j|}.$$

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

## 4. What `dispersion()` computes

`dispersion(layer, ps, st)` returns `\hat W=\mathrm{fft2}(W)` and
`M=A+g\hat W`, `\kappa=k+g\hat W`. So:

- For **`:potential`** it is the *exact* dispersion (§2).
- For **`:spike`** it is the `g_{\text{eff}}=g/r` linearization (§3) — a correct
  local guide near a uniform-amplitude operating point (wavelength selection,
  speed, criticality), not a global exact relation.

---

## References

- Lattice dynamics / phonons: any solid-state text (Ashcroft–Mermin ch. 22).
- Sakaguchi & Kuramoto (1986), *A soluble active rotator model showing phase transitions via mutual entrainment* — the phase-frustrated Kuramoto model the spike sheet realizes.
- `src/wave.jl` (implementation), `docs/rf_wave_network_implementation.md` (design), `demos/wave_dispersion_derivation.ipynb` (verification).
