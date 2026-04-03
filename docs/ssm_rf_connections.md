# Formal Connections Between State-Space Models and Resonate-and-Fire Neuron Networks

## 1. The Starting Point: Two Parallel Systems

### 1.1 The SSM Framework

A continuous-time linear state-space model is defined by:

```
du/dt = A·u(t) + B·x(t)
y(t) = C·u(t) + D·x(t)
```

where `u(t)` is the hidden state, `x(t)` is the input, and `y(t)` is the output. The key insight of modern SSMs (S4, S4D, Mamba) is that this system admits three equivalent computational views:

- **Recurrent**: `u[n+1] = Ā·u[n] + B̄·x[n]`, `y[n] = C·u[n] + D·x[n]` (via zero-order-hold discretization)
- **Convolutional**: `y = K * x` where `K[n] = C·Āⁿ·B̄` is the causal impulse-response kernel
- **Continuous**: the original ODE, solved numerically

### 1.2 The R&F Neuron

A Resonate-and-Fire neuron has the ODE:

```
dz/dt = k·z(t) + w·I(t),    k = λ + iω
```

where `z(t) ∈ ℂ` is the complex membrane potential, `λ < 0` is the leakage (decay), `ω = 2π/T` is the angular frequency, and `I(t)` is the input current. This is already a 1D complex (equivalently 2D real) linear system — the same mathematical object as a diagonal SSM with a single complex eigenvalue.

### 1.3 The Biological Constraint

In SSMs, the state `u` feeds directly into the output equation `y = Cu + Dx`. But in a biologically-motivated R&F network, the internal complex potential `z` of a neuron is **not directly observable by other neurons**. Other neurons receive only **spikes** — discrete timing events that encode the **phase** of `z` at peak voltage. This means the natural output of an R&F neuron is not the raw potential `z(t)`, but rather the **phase** `θ = arg(z)/π ∈ [-1, 1]`.

This observation motivates the central question: can we formulate the R&F SSM entirely in terms of phases, preserving all three computational views?

---

## 2. From Potentials to Phases: The Phase-Domain SSM

### 2.1 Phase as the Natural Observable

The R&F neuron's complex potential can be decomposed as:

```
z(t) = r(t)·exp(iπθ(t))
```

where `r(t) = |z(t)|` is the amplitude and `θ(t)` is the phase. The neuron "fires" when `Im(z)` reaches a maximum (voltage peak), which occurs when the phase aligns with the natural oscillation. The key insight is that **phase is the information-carrying variable** — amplitude merely tracks signal strength and decays exponentially.

In the PhasorNetworks framework, phases live on the unit circle and can be extracted via:

```
θ = complex_to_angle(normalize_to_unit_circle(z))
```

This normalization discards amplitude and isolates the phase, which is exactly the quantity that downstream neurons can reconstruct from spike timing.

### 2.2 Phase Update Dynamics

For a single R&F oscillator receiving a complex input `I(t) = |I|·exp(iπφ)` (a signal at phase `φ`), the exact solution over one step `Δt` is:

```
z[n+1] = A·z[n] + B·I[n]
```

where `A = exp(kΔt)` and `B = (A - 1)/k`. We can decompose `A = |A|·exp(iπα)` where:

```
|A| = exp(λΔt)        (amplitude decay per step)
α = ωΔt/π             (phase rotation per step, in units of π)
```

On the unit circle (after normalization), the phase update becomes:

```
θ[n+1] = arg(A·exp(iπθ[n]) + B·exp(iπφ[n])) / π
```

This is **nonlinear in phase** due to the `arg` operation on a sum of complex numbers. This is fundamentally different from a standard SSM, where the state update is strictly linear. The nonlinearity arises because phases combine via **vector addition on the unit circle** (bundling), not scalar addition.

### 2.3 Connection to VSA Operations

This phase nonlinearity is not a defect — it is precisely the **bundling** operation from Vector Symbolic Architectures:

```
v_bundle(θ₁, θ₂) = arg(exp(iπθ₁) + exp(iπθ₂)) / π
```

Bundling is the constructive/destructive interference of unit phasors. When two phases are similar, they reinforce; when dissimilar, they partially cancel. This gives the R&F phase-SSM an inherent **similarity-weighted averaging** that linear SSMs lack.

Meanwhile, the **binding** operation is phase addition:

```
v_bind(θ₁, θ₂) = remap_phase(θ₁ + θ₂)
```

The natural rotation `A·z` in the R&F ODE is itself a binding operation — it binds the state phase `θ[n]` with the per-step rotation `α`. This means the R&F recurrence can be decomposed as:

```
θ[n+1] = v_bundle(v_bind(θ[n], α), v_bind(φ[n], β))
```

where `α` is the rotation from `A` and `β = arg(B)/π` is the input phase shift from the discretization gain.

**This is the core result**: the R&F recurrence is a composition of VSA binding (phase rotation) and bundling (interference), not a linear recurrence.

---

## 3. The Three Computational Views for R&F-SSMs

The R&F Phasor SSM admits three equivalent computational views. A critical distinction, discovered during implementation, is that the correct discretization depends on the **input type**: continuous signals use zero-order hold (ZOH), while phase/spike inputs use Dirac discretization.

### 3.1 Continuous View: The Two-Stage Coupled ODE

The continuous view is the most physically faithful. For a multi-channel SSM layer receiving spike inputs, the dynamics form a **two-stage coupled ODE**:

```
Stage 1 (input oscillators):    du/dt = k₀·u(t) + I(t)
Stage 2 (output oscillators):   dz_c/dt = k_c·z_c(t) + Σⱼ W[c,j]·u_j(t)
```

where:
- `k₀ = λ₀ + iω₀` is the **global** neuron constant (shared across all input channels, defined by `SpikingArgs`)
- `k_c = λ_c + iω_c` are the **per-channel** output eigenvalues (trainable)
- `I(t) = Σₛ E·κ(t - tₛ)` is the input current from spikes, with `κ` being the spike kernel (raised cosine) and `E` being the spike energy (integral of the kernel, `≈ 2·t_window`)
- `W` is the weight matrix projecting input channels to output channels

The input oscillators decode incoming spikes into complex potentials at the global frequency. The output oscillators, each tuned to its own `(λ_c, ω_c)`, integrate the weighted input potentials. Sampling `z_c` at period boundaries `t = n·T` yields the discrete output sequence.

This is implemented in `_forward_3d_spiking` via `oscillator_bank` with a combined state vector `[u; z]`.

### 3.2 Discrete View: Two Discretization Strategies

#### 3.2.1 Zero-Order Hold (ZOH) — for Continuous Inputs

When the input is a continuous signal sampled at regular intervals (e.g., pixel values encoded as complex numbers via `psk_encode`), the standard ZOH discretization applies:

```
z_c[n+1] = A_c·z_c[n] + B_c·(Σⱼ W[c,j]·x_j[n])
```

where `A_c = exp(k_c·Δt)` and `B_c = (A_c - 1)/k_c`. The input `x[n]` is treated as constant over each timestep.

This is the standard SSM recurrence. It operates on **complex-valued signals**, not phases, with phase extraction applied only at the readout boundary. Implemented in `PhasorDense`'s 3D complex dispatch.

#### 3.2.2 Dirac Discretization — for Phase/Spike Inputs

When the input is a **phase** from an upstream R&F layer, the phase encodes a spike time within the oscillation period. A phase `θ` corresponds to a spike at time `t_s = (θ/2 + 0.5)·T`, so the remaining propagation time is `dt = T·(0.5 - θ/2)`.

For the coupled two-stage system, the analytical solution for a single Dirac spike at time `t_s` arriving at input channel `j` is:

```
z_c(T) = W[c,j]·E · ∫_{t_s}^{T} exp(k_c·(T-τ)) · exp(k₀·(τ-t_s)) dτ
```

Evaluating the integral (product of two exponentials):

```
z_c(T) = W[c,j]·E · exp(k₀·dt)·(exp((k_c-k₀)·dt) - 1) / (k_c - k₀)
       = W[c,j]·E · (exp(k_c·dt) - exp(k₀·dt)) / (k_c - k₀)
```

where `dt = T - t_s = T·(0.5 - θ/2)` and `E` is the spike energy.

This formula captures two effects that ZOH misses:
1. **Sub-period timing**: different phases produce different propagation times `dt`, giving different decay and rotation amounts per output channel
2. **Input-output coupling**: the input oscillator at `k₀` and the output oscillator at `k_c` interact continuously, not at discrete boundaries

The per-output-channel Dirac encoding is:

```
dirac_enc[c,j,n] = E · (exp(k_c·dt_j[n]) - exp(k₀·dt_j[n])) / (k_c - k₀)
```

The full discrete recurrence then uses the output-stage kernel only:

```
H_c[n] = Σⱼ W[c,j] · dirac_enc[c,j,n]     (weighted Dirac-encoded input)
z_c[n+1] = exp(k_c·Δt) · z_c[n] + H_c[n]   (output stage recurrence)
```

Note: the Dirac kernel uses `K_c[n] = exp(k_c·n·Δt)` with **no ZOH gain factor** `(A-1)/k` — the gain is already absorbed into the Dirac encoding.

This is implemented in `dirac_encode` and `causal_conv_dirac`. The `PhasorDense` layer automatically selects Dirac for 3D Phase input (when `init_mode ≠ :default`) and ZOH for 3D complex input.

### 3.3 Convolutional View: FFT-Based Causal Convolution

Both discretizations produce a linear recurrence `z_c[n+1] = A_c·z_c[n] + H_c[n]`, which unrolls to a causal convolution:

```
z_c[n] = Σⱼ₌₀ⁿ K_c[n-j]·H_c[j]  =  (K_c * H_c)[n]
```

The kernel is `K_c[n] = exp(k_c·n·Δt)` for Dirac inputs (no `B` gain), or `K_c[n] = exp(k_c·n·Δt)·B_c` for ZOH inputs.

This convolution is implemented via FFT for O(C·L·log(L)·B) cost (`causal_conv_fft`), falling back to Toeplitz matrix multiplication for short sequences (`_causal_conv_toeplitz`). The `causal_conv` function auto-dispatches based on sequence length.

**Key property**: the convolution operates on **complex-valued signals** `H_c[n]`, not raw phases. Phases enter through the Dirac encoding (which produces complex values from phase inputs), and are extracted only at the output boundary via `complex_to_angle(normalize_to_unit_circle(Z))`.

#### Input Encoding Summary

| Input type | Encoding | Gain | Used by |
|-----------|----------|------|---------|
| Complex (continuous) | `H = W·x` | `B = (A-1)/k` (ZOH) | First layer, `psk_encode` data |
| Phase (from R&F layer) | `H_c = Σ_j W·dirac_enc[c,j]` | In encoding (Dirac) | Intermediate SSM layers |

### 3.4 Equivalence Between Views

The three views are exactly equivalent in the following sense:

1. **Continuous → Discrete (Dirac)**: As the spike kernel width `t_window → 0` and the ODE solver step `dt → 0`, the spiking ODE output converges to the Dirac discrete output. The Dirac formula is the analytical limit of the ODE integration.

2. **Discrete → Convolutional**: The causal convolution is an algebraic rearrangement of the recurrence — they compute the same values to floating-point precision.

3. **Convolutional (Toeplitz) → Convolutional (FFT)**: The FFT-based convolution computes the same linear convolution as the Toeplitz matrix multiply, via zero-padded circular convolution.

The remaining approximation gap (correlation ~0.5 between Dirac discrete and ODE spiking in tests) arises from:
- The spike kernel in the ODE having finite width (not a true Dirac delta)
- The ODE solver's temporal discretization
- Accumulated differences in how the continuous coupling vs discrete two-stage formula propagates over multiple periods

---

## 4. Multi-Channel Dynamics: The Role of the A Matrix

### 4.1 Per-Channel Eigenvalues as Diagonal SSM

In a standard SSM, the matrix `A` is `N×N` and couples all state dimensions. The S4D insight is that diagonalizing `A` decouples the state into `N` independent 1D complex systems:

```
zₙ[t+1] = aₙ·zₙ[t] + bₙ·xₙ[t],    aₙ = exp(kₙΔt)
```

This is **exactly** the R&F multi-channel architecture: each output channel `c` has its own complex eigenvalue `kc = λc + iωc`, defining an independent damped oscillator. The `PhasorDense` layer's `log_neg_lambda` and `omega` parameters are precisely the diagonal entries of the SSM's `A` matrix in eigenvalue form.

The weight matrix `W` (which maps input channels to output channels) plays the role of `B` in the SSM — it projects the input into the eigenspace of `A`. In the standard SSM notation:

```
A = diag(k₁, k₂, ..., k_C)          (diagonal, complex)
B = W                                  (in_dims × out_dims, real)
C = I                                  (identity — we read out all channels)
D = 0                                  (no skip connection)
```

### 4.2 Why Diagonal is Natural (Not a Limitation)

In the standard SSM literature, diagonalization of `A` is an *approximation* — the original HiPPO matrix is dense. But for R&F neurons, diagonal `A` is the **physically natural** structure: each neuron is an independent oscillator characterized by its own `(λ, ω)`. Cross-channel coupling happens through the weight matrix `W`, not through shared dynamics.

This means R&F networks are naturally S4D-like: they are already diagonalized, with no approximation required. The HiPPO-LegS initialization (implemented in `hippo_legs_diagonal`) provides a principled way to set the eigenvalues:

```
kₙ = -(n + 1/2) + iπ(n + 1/2)
```

This gives log-spaced decay rates and proportional frequencies, ensuring multi-timescale memory.

### 4.3 Biological Plausibility of Diagonal Structure

The diagonal structure has a biological interpretation: each oscillator is a **resonator** tuned to a specific frequency. The collection of oscillators forms a **filter bank** that decomposes the input signal into frequency components. This is analogous to:

- **Cochlear processing**: hair cells tuned to different frequencies
- **Cortical oscillations**: neural populations resonating at theta, alpha, beta, gamma bands
- **Place cells**: oscillatory interference models of spatial coding

The weight matrix `W` implements **synaptic connectivity** between input neurons and oscillator neurons, which is biologically standard.

---

## 5. Determining Leakage from the SSM Perspective

### 5.1 Leakage as Memory Timescale

In the SSM framework, the real part of each eigenvalue determines the **memory timescale** of that state dimension. For the R&F neuron:

```
λ = Re(k) = leakage
```

The decay constant `τ = -1/λ` gives the characteristic time over which the oscillator "remembers" past input. After `τ` time units, the contribution of a past input has decayed to `1/e ≈ 37%` of its original strength.

### 5.2 Optimal Leakage from HiPPO Theory

The HiPPO framework provides a principled answer to "what should the leakage be?" The LegS variant says: for a system with `N` oscillators that needs to remember a window of length `T`, the optimal eigenvalues are:

```
λₙ = -(n + 1/2)/T
```

This gives:
- Channel 0: `λ = -0.5/T` (long memory, slow decay)
- Channel N-1: `λ = -(N-0.5)/T` (short memory, fast decay)

The log-spaced variant (implemented in `hippo_legs_diagonal`) preserves this multi-timescale property while keeping all channels in a numerically trainable range.

### 5.3 Leakage-Frequency Coupling

In the HiPPO-LegS initialization, leakage and frequency are **coupled**: `ωₙ = π|λₙ|`. This means each oscillator completes roughly one full oscillation within its memory window — faster-decaying oscillators also oscillate faster. This coupling is physically natural: a resonator that rings down quickly must oscillate quickly if it is to complete at least one cycle before decaying.

However, this coupling is not mandatory. In the `PhasorDense` layer, `log_neg_lambda` and `omega` are independent trainable parameters, allowing the network to learn decoupled leakage-frequency relationships if the task demands it.

### 5.4 Practical Implications

The SSM perspective provides concrete guidance for setting leakage:

1. **Fixed leakage** (standard R&F): `λ = -0.2` for all channels. This gives `τ = 5` time units. All channels have the same memory length — the network is a bank of identical resonators at the same frequency, differentiated only by their input weights.

2. **HiPPO leakage** (SSM-informed R&F): Per-channel `λₙ` spanning orders of magnitude. This gives a multi-timescale memory that can simultaneously track recent fine detail and distant coarse structure.

3. **Trainable leakage**: Start from either (1) or (2), then let gradient descent adjust each `λc` independently. The SSM theory guarantees that the optimization landscape is well-behaved (the loss is smooth in `λ` because the kernel `K[n] = exp(kΔt)ⁿ·B` is smooth in `k`).

---

## 6. VSA Operations as SSM Transformations

### 6.1 Binding as State Rotation

The VSA binding operation `v_bind(θ₁, θ₂) = remap_phase(θ₁ + θ₂)` corresponds to complex multiplication: `exp(iπθ₁)·exp(iπθ₂) = exp(iπ(θ₁+θ₂))`. In the SSM framework, this is equivalent to a **state rotation** — multiplying the state by a unitary matrix.

In the R&F context, the per-step rotation `A = exp(kΔt)` is itself a binding with the natural phase increment `α = ωΔt/π`. This means **the SSM's temporal dynamics implement a sequence of bindings** with a fixed "time phase":

```
z[n] = bind(bind(...bind(bind(z[0], α), α)..., α), Σ inputs)
     = bind(z[0], n·α) + input contributions
```

The accumulated rotation `n·α` is the natural oscillator phase at step `n`. Input contributions are phase-shifted copies of the input, each rotated by the appropriate number of time steps.

### 6.2 Bundling as State Superposition

The VSA bundling operation `v_bundle(θ₁, θ₂) = arg(exp(iπθ₁) + exp(iπθ₂))/π` corresponds to complex addition followed by phase extraction. In the SSM recurrence, this happens at every step:

```
z[n+1] = A·z[n] + B·I[n]    (complex addition = bundling of rotated state and new input)
```

The causal convolution unrolls this as:

```
z[n] = Σⱼ K[n-j]·I[j]    (bundling of all past inputs, each at different decay/rotation)
```

This is a **temporal bundling**: the oscillator's state is a superposition of all past inputs, each "bound" with a time-dependent phase rotation and weighted by exponential decay.

### 6.3 The Readout Equation

In the standard SSM, the readout is `y = C·u + D·x`. In the R&F phase-SSM:

- **Phase readout** (`y = arg(z)/π`): Extracts the dominant phase from the temporal bundling. This is what `SSMReadout` does — it computes similarity between the bundled state phase and codebook prototypes. This is inherently nonlinear.

- **Similarity readout** (`y = cos(π(θ - θ_ref))`): Measures how well the state phase matches a reference. This is the `Codebook` layer's operation. It converts the circular phase variable into a scalar similarity score.

- **Complex readout** (`y = C·z`): A standard linear readout on the complex state, before phase extraction. This preserves the linear SSM structure but outputs complex values, not phases.

The choice of readout determines whether the overall system is linear (complex readout) or nonlinear (phase/similarity readout). The phase readout is the biologically natural choice (neurons communicate via spike timing = phase), while the complex readout preserves the SSM's linear structure for training.

---

## 7. Formal System Definition: The Phasor SSM

Combining the above, we define the **Phasor SSM** — a two-stage coupled diagonal state-space model that bridges R&F spiking networks and modern SSMs.

### 7.1 Parameters

- `W ∈ ℝ^{C_out × C_in}` — input projection (synaptic weights)
- `λ ∈ ℝ^{C_out}₋` — per-channel output decay rates (negative)
- `ω ∈ ℝ^{C_out}₊` — per-channel output angular frequencies
- `k₀ = λ₀ + iω₀ ∈ ℂ` — global input neuron constant
- `E ∈ ℝ₊` — spike energy (integral of spike kernel, `≈ 2·t_window`)
- `b ∈ ℂ^{C_out}` — bias (optional)

### 7.2 Derived Quantities

- `k_c = λ_c + iω_c` — complex eigenvalue for output channel `c`
- `A_c = exp(k_c·Δt)` — per-step output state transition
- `B_c = (A_c - 1)/k_c` — ZOH input gain (for continuous inputs only)

### 7.3 The Three Views

**Continuous/spiking form (two-stage coupled ODE):**
```
du/dt = k₀·u(t) + I(t)                    (input oscillators, global k₀)
dz_c/dt = k_c·z_c(t) + Σⱼ W[c,j]·u_j(t)  (output oscillators, per-channel k_c)
I(t) = Σₛ E·κ(t - tₛ)                     (spike current with kernel κ)
```

**Discrete form with Dirac discretization (for phase inputs):**

A phase input `θ_j` at step `n` represents a spike at time `t_s = (θ_j/2 + 0.5)·T` within period `n`. The remaining propagation time is `dt_j = T·(0.5 - θ_j/2)`. The coupled analytical solution gives the effective input to output channel `c`:

```
dirac_enc[c,j,n] = E · (exp(k_c·dt_j) - exp(k₀·dt_j)) / (k_c - k₀)
```

The output recurrence uses only the output-stage kernel (Dirac needs no ZOH gain):

```
H_c[n] = Σⱼ W[c,j] · dirac_enc[c,j,n]
z_c[n+1] = A_c · z_c[n] + H_c[n]
```

**Discrete form with ZOH (for continuous complex inputs):**
```
z_c[n+1] = A_c · z_c[n] + B_c · Σⱼ W[c,j] · x_j[n]
```

**Convolutional form (both discretizations):**
```
Z_c = K_c * H_c       where K_c[n] = A_cⁿ (Dirac) or A_cⁿ·B_c (ZOH)
Θ = complex_to_angle(normalize(Z))
```

### 7.4 The Dirac Encoding Derivation

For a Dirac spike `δ(t - t_s)` at input channel `j`, the coupled two-stage ODE has the exact solution:

**Stage 1**: `u_j(t) = exp(k₀·(t - t_s))` for `t > t_s` (input oscillator rings down)

**Stage 2**: The output oscillator receives `W[c,j]·u_j(t)` continuously:

```
z_c(T) = W[c,j] · ∫_{t_s}^{T} exp(k_c·(T-τ)) · exp(k₀·(τ-t_s)) dτ
```

Let `dt = T - t_s`. The integral evaluates to:

```
∫_{t_s}^{T} exp(k_c·(T-τ)) · exp(k₀·(τ-t_s)) dτ

  = exp(k₀·dt) · ∫_0^{dt} exp((k_c - k₀)·u) du       [substituting u = T-τ]

  = exp(k₀·dt) · (exp((k_c - k₀)·dt) - 1) / (k_c - k₀)

  = (exp(k_c·dt) - exp(k₀·dt)) / (k_c - k₀)
```

When `k_c = k₀` (same eigenvalue), L'Hôpital's rule gives `dt · exp(k_c·dt)`.

For the physical spiking system with spike energy `E` (integral of the kernel):

```
z_c(T) = W[c,j] · E · (exp(k_c·dt) - exp(k₀·dt)) / (k_c - k₀)
```

This is the exact analytical equivalent of the ODE integration — it captures the continuous coupling between input and output oscillators without temporal substeps.

### 7.5 Properties

1. **State propagation is linear** in complex space → convolutional and recurrent views are exact
2. **Phase readout is nonlinear** → equivalent to VSA bundling with similarity weighting
3. **Per-channel dynamics are independent** (diagonal A) → natural for oscillator banks
4. **Three equivalent views**: continuous (ODE), discrete (recurrence), convolutional (kernel)
5. **Two discretization strategies**: ZOH for continuous inputs, Dirac for phase/spike inputs
6. **VSA interpretation**: temporal dynamics = binding with time phase; state = temporal bundling of inputs; readout = similarity to codebook
7. **The Dirac encoding is per-output-channel** because it depends on `k_c`, unlike ZOH encoding which is channel-independent

### 7.6 What This System Can and Cannot Do

**Can do:**
- All three computational views (recurrent, convolutional, continuous/spiking)
- Analytically evaluate Dirac spike contributions via the coupled integral formula
- Learn per-channel dynamics (leakage and frequency) via gradient descent
- Use HiPPO-informed initialization for principled multi-timescale memory
- Express the computation as VSA operations (binding = rotation, bundling = superposition)
- Produce spike-based I/O for neuromorphic hardware
- Use FFT-based convolution for O(L log L) sequence processing

**Cannot do (inherent limitations):**
- Direct phase-space convolution (must work in complex space, extract phases after)
- Non-diagonal A matrix (oscillators are independent; coupling is through W only)
- Phase-linear state update (the `arg` readout introduces essential nonlinearity)
- Channel-independent Dirac encoding (unlike ZOH, the encoding depends on the output channel's eigenvalue)

---

## 8. Open Questions and Future Directions

### 8.1 Gated Phase Updates

Modern SSMs (Mamba, S6) introduce input-dependent gating: `A[n] = A(x[n])`. For R&F networks, this would mean **input-dependent leakage or frequency**: the neuron's resonant properties change based on what it receives. This is biologically plausible (neuromodulation changes membrane time constants) and could be implemented by making `λ` and `ω` functions of the input:

```
λ[n] = σ(Wλ·x[n]) · λ_base
ω[n] = σ(Wω·x[n]) · ω_base
```

This breaks the time-invariance that enables the convolutional view, but preserves the recurrent and continuous views.

### 8.2 Cross-Channel Phase Interaction

The current diagonal structure means channels interact only through the weight matrix W. An alternative is to allow **phase-domain cross-channel coupling** via VSA operations:

```
θ_out = v_bundle(W · v_bind(θ_in, θ_weight))
```

This would implement a dense layer in phase space, where binding replaces multiplication and bundling replaces addition. The challenge is that this operation is not equivalent to any linear SSM — it is a genuinely new computational primitive.

### 8.3 Convolutional Efficiency (Implemented)

FFT-based causal convolution is implemented in `causal_conv_fft`, achieving O(C·L·log(L)·B) cost. The `causal_conv` function auto-dispatches: FFT for L > 64, Toeplitz for shorter sequences. This enables sequential MNIST at L=784 and other long-range benchmarks.

The phasor kernel `K[n] = Aⁿ·B` also has a closed-form DFT: `K̂[f] = B / (1 - A·exp(-2πif/L))`, which could enable even more efficient frequency-domain computation in future work.

### 8.4 Structured Phase Initialization

Beyond HiPPO, there may be other principled initializations for the `(λ, ω)` pairs that exploit the phase structure. For example:

- **Harmonic series**: `ωc = c·ω_base` — musical harmonic analysis
- **Mel-spaced**: Log-frequency spacing matching auditory perception
- **Task-specific**: Initialize frequencies to match known periodicities in the data

---

## 9. Summary Table

| Concept | Standard SSM | R&F / Phasor SSM |
|---------|-------------|-------------------|
| State variable | `u ∈ ℝⁿ` or `ℂⁿ` | `z ∈ ℂᶜ` (complex potential) |
| State transition | `A ∈ ℝⁿˣⁿ` (dense or diagonal) | `diag(exp(kΔt))` (always diagonal) |
| Input projection | `B ∈ ℝⁿˣᵈ` | ZOH: `diag((A-1)/k)·W`; Dirac: `E·(exp(k_c·dt)-exp(k₀·dt))/(k_c-k₀)·W` |
| Output projection | `C ∈ ℝᵐˣⁿ` | `arg(·)/π` (nonlinear phase extraction) |
| Recurrent form | `u[n+1] = Āu[n] + B̄x[n]` | `z[n+1] = A·z[n] + H[n]` (H from encoding) |
| Convolutional form | `y = (CĀⁿB̄) * x` | `Z = (Aⁿ) * H` (Dirac) or `(AⁿB) * (Wx)` (ZOH) |
| Continuous form | `du/dt = Au + Bx` | `du/dt = k₀u + I(t)`, `dz/dt = k_c·z + Wu` |
| Eigenvalue meaning | Decay + oscillation | Leakage + resonant frequency |
| Addition of states | Linear sum | VSA bundling (interference) |
| Time evolution | Matrix exponential | Phase binding (rotation) |
| Readout | Linear projection | Cosine similarity to codebook |
| Discretization | ZOH, bilinear, etc. | ZOH (continuous input) or Dirac (phase/spike input) |
| Initialization | HiPPO-LegS, random | HiPPO-LegS, uniform, fixed |
| Gating | Input-dependent A (Mamba) | Input-dependent λ,ω (neuromodulation) |

---

## 10. Conclusion

The R&F neuron network is a **physically-grounded diagonal state-space model** where:

1. The **state** lives in complex space (membrane potentials), but the **observable** is phase (spike timing)
2. The **state transition** is multiplication by `exp(kΔt)` — a complex rotation with decay — which is equivalent to **VSA binding** with a time-phase
3. The **state accumulation** is complex addition — which is equivalent to **VSA bundling** (constructive/destructive interference)
4. The **readout** is phase extraction followed by codebook similarity — a nonlinear projection from the unit circle to classification logits

All three SSM computational views (recurrent, convolutional, continuous) apply to R&F networks, with the caveat that the **phase readout introduces nonlinearity** that prevents the convolutional view from operating directly on phases. The computation must proceed in complex space (where everything is linear) and extract phases only at the output boundary.

The SSM perspective provides three practical gifts to R&F networks:
- **HiPPO initialization** for principled multi-timescale memory (what should leakage be?)
- **Convolutional parallelism** for efficient training (avoiding sequential ODE integration)
- **Theoretical guarantees** on memory capacity and approximation quality

Conversely, the R&F perspective provides three gifts to SSMs:
- **Physical interpretability** (oscillator banks, spike timing, resonance)
- **Neuromorphic deployability** (direct mapping to spiking hardware)
- **VSA algebra** (binding and bundling as first-class operations on the state)

The Phasor SSM framework unifies these perspectives, showing that biological resonance, vector-symbolic computation, and modern sequence modeling are three views of the same underlying mathematical structure: **causal convolution with complex exponential kernels on the unit circle**.
