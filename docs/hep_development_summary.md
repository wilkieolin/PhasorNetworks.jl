# Adapting Holomorphic Equilibrium Propagation to Phasor Networks

## Development Summary

### 1. The Starting Point

Holomorphic EP (Laborieux & Zenke, NeurIPS 2022) trains energy-based networks
by extending the nudge parameter beta to the complex plane, then extracting
exact gradients as the first Fourier coefficient of energy gradients evaluated
on a contour in the complex beta-plane. The reference implementation uses
Hopfield-style networks with holomorphic activations (complex tanh).

PhasorNetworks.jl implements neural networks where information is encoded in
the phase of complex oscillators. The core equation is `dz/dt = kz + WI(t)`
where `k = lambda + i*omega`. This produces damped oscillations — the
amplitude decays while the phase encodes information.

The question: can hEP provide a biologically plausible, local learning rule
for phasor networks by exploiting the natural oscillatory dynamics?

### 2. Issues Encountered and Resolutions

#### Issue 1: Holomorphic Activation Function

**Problem**: The standard phasor activation `normalize_to_unit_circle` computes
`z/|z|`, which involves `|z| = sqrt(z*conj(z))` — conjugation breaks
holomorphicity. Without holomorphic activations, the contour integration
cannot extract exact gradients.

**Analysis**: We surveyed holomorphic alternatives:
- `holotanh(z) = tanh(z)`: bounded, preserves phase for moderate inputs,
  poles safely at `z = i*pi/2` (outside the unit disk)
- `z/(eps + sqrt(z*z))`: strong magnitude control but severe phase distortion
  from branch cuts
- Rational and polynomial approximations: either unbounded or have poles in
  the operating region

**Resolution**: Use `holotanh(z) = tanh(z)` as the inter-layer activation.
It provides soft saturation (prevents blowup) while preserving phase for
inputs with |z| < 1. No holomorphic function can exactly normalize to the
unit circle (Liouville's theorem: a bounded entire function is constant),
so exact normalization is fundamentally impossible. The practical solution
is to handle magnitude control through weight initialization scaling rather
than through the activation.

#### Issue 2: Forward Pass / Training Path Mismatch

**Problem**: PhasorDense's 2D complex dispatch computes `W*x + bias` with
NO activation — the activation parameter is only used by the 3D and Phase
dispatches. The model's Lux forward pass and the hEP equilibrium dynamics
were computing completely different functions, so the hEP gradients
optimized a different objective than the loss being monitored.

**Resolution**: Evaluate loss from the hEP free-phase equilibrium output
rather than from the Lux forward pass. This ensures the gradient computation
and loss measurement operate on the same function. The model's forward pass
is used only for inference after training.

#### Issue 3: DC Gain Attenuation

**Problem**: The oscillator's steady-state response to constant input is
`z* = -I/k`, with magnitude `|z*| = |I|/|k|`. With the default dynamics
(`lambda = -0.2, omega = 2*pi`), `|k| = 6.3`, so each layer attenuates
signals by a factor of ~6x. Two layers: 40x attenuation. States become
tiny, making Hebbian gradients vanishingly small.

**Analysis**: The attenuation is dominated by omega, not lambda. With
`omega = 0`, `|k| = 0.2` and the gain is 5x (amplification, not
attenuation). The omega term, which provides the oscillatory character
of the phasor neuron, also creates a high-pass filter that rejects the
DC (constant) component of the inter-layer drive.

**Resolution**: Set `omega = 0` during EP settling. The oscillation
frequency serves temporal encoding in the SSM/spiking pathway, not EP
convergence. With omega=0, the system reduces to exponentially decaying
integrators with gain `1/|lambda|`, which is well-conditioned for EP.

#### Issue 4: Oscillating States Don't Converge to Fixed Points

**Problem**: With omega > 0, the raw states z(t) rotate in the complex
plane and never reach a fixed point. The EP gradient theorem requires
evaluating energy gradients at equilibrium — but the "equilibrium" of
an oscillator is a periodic orbit, not a fixed point.

**Analysis**: We attempted demodulation — multiplying by `conj(ref(t))`
to extract the slowly-varying relative phase. This correctly identifies
the information-carrying component but doesn't fix the convergence issue
because the nonlinear inter-layer coupling (through holotanh) generates
harmonics and beat frequencies that prevent the demodulated states from
settling.

**Resolution**: The omega=0 setting eliminates the oscillation entirely
during EP, giving true fixed-point convergence. The demodulation
machinery (`_demodulate`) is retained for the readout layer, where it
converts the output layer's state to a relative-phase representation
for the interference-based cost function.

#### Issue 5: Missing Self-Energy in the Energy Function

**Problem**: The initial energy function used the Hopfield form
`Phi = sum <sigma(W*z_prev), z>` without a self-energy term. This means
the equilibrium condition `dPhi/dz = 0` gives
`sigma(W*z_prev) + feedback = 0`, which does NOT correspond to the
phasor ODE `dz/dt = kz + drive`. The oscillator's own dynamics were
not represented in the energy.

**Resolution**: Add the self-energy `(1/2)<z, Kz>` to the energy:

    Phi = sum_l [(1/2)<z_l, K_l*z_l> + <sigma(W_l*z_{l-1}+b_l), z_l>]
          - beta * C(z_L, y)

The gradient `dPhi/dz_l = K_l*z_l + sigma(W_l*z_{l-1}) + feedback`
now exactly matches the phasor ODE. This ensures the EP gradient
theorem holds: the Hebbian parameter gradient at the equilibrium of
this energy equals the loss gradient.

#### Issue 6: Gradient Sign Error

**Problem**: The hEP contour gradient had cosine similarity of -0.80
with the finite-difference gradient — correct direction but
wrong sign. The loss wouldn't decrease during training.

**Analysis**: The energy is `Phi = E - beta*C`, so `dPhi/dW` at
equilibrium gives the derivative of the energy w.r.t. parameters.
The EP theorem states this equals `dL/dW` (the loss gradient), but
with respect to the ENERGY, not the loss directly. Since the cost
enters with a minus sign in the energy, `dPhi/dW` actually gives
the negative of `dL/dW`.

**Resolution**: Negate the contour integration output before applying
the optimizer update. After this fix, cosine similarity with the
finite-difference gradient is +0.80 (W1) and +0.89 (W2).

#### Issue 7: Interference-Based Readout

**Problem**: The standard Codebook readout uses `cos(pi*(phase - code))`
averaged over features. This involves phase extraction (`angle(z)`)
and cosine — both non-holomorphic operations.

**Resolution**: `HolomorphicReadout` computes
`logit_c = (1/d) * sum(z .* conj(code_c))`. The conjugated codebook
entries are fixed constants (not functions of z), so the operation is
holomorphic in z. When z is on the unit circle, the real part of each
logit equals the standard codebook cosine similarity. This is physically
the same as oscillator interference: summing two oscillators and
measuring the result's magnitude.

### 3. Current Status

**Working**: With omega=0, SGD (lr=1.0), N=16 contour points:
- Gradient direction verified (cosine similarity 0.80-0.89 with FD)
- Training loss drops from 0.876 to 0.685 (chance = 0.693)
- Per-batch minima reach 0.594
- 464 tests pass

### 4. Open Issues: Connecting to Oscillatory Dynamics

The current implementation works with omega=0, which effectively
removes the oscillatory character of the phasor neurons during
training. Several issues need resolution to fully integrate hEP
with the oscillatory dynamics:

#### 4a. The Omega=0 Limitation

With omega=0, the network is equivalent to a real-valued network
with exponential decay integration and holomorphic tanh activation.
The complex structure is present but the oscillatory dynamics that
define phasor networks — phase encoding, frequency selectivity,
temporal integration — are not engaged during training. The trained
weights could in principle be transferred to an oscillatory network
for inference, but the training itself doesn't learn to exploit
phase relationships.

**Potential path**: Gradually increase omega during training
(curriculum-style), or train with small omega that provides some
phase structure without destabilizing the EP dynamics.

#### 4b. Inter-Layer Coupling with Oscillating States

When omega > 0, the raw states z(t) oscillate. The inter-layer
coupling `W * z_{l-1}` produces an oscillating drive. If all
layers share the same omega, the oscillations are coherent and the
relative phases are preserved through linear operations. But
holotanh is nonlinear — it generates harmonics of the carrier
frequency, which the downstream oscillator responds to differently
than the fundamental. This creates intermodulation products that
compound across layers and prevent convergence.

**Potential paths**:
- Use linear inter-layer coupling (no activation between layers),
  relying on the readout nonlinearity for classification power.
  Linear coupling preserves frequency content exactly.
- Use a weakly nonlinear activation (small `a` in holotanh) that
  keeps harmonics small relative to the fundamental.
- Demodulate before activation and remodulate after, so holotanh
  operates on slowly-varying envelopes rather than oscillating
  signals. This requires careful treatment of the energy function.

#### 4c. The DC Gain vs Resonant Gain

With omega > 0, the oscillator acts as a bandpass filter centered
at omega. The DC gain is `1/|k|` (small when omega >> |lambda|),
but the gain at the resonant frequency omega is `1/|lambda|` (large).
This means the oscillator strongly amplifies signals near its
natural frequency and attenuates everything else.

For EP with coherent coupling (all layers at the same omega), the
inter-layer drive IS at the resonant frequency, so the response
should be strong. The issue is that holotanh generates DC and
harmonic components that are strongly attenuated, creating an
effective low-rank coupling.

**Potential path**: The demodulate-activate-remodulate approach
would keep the inter-layer signal at the carrier frequency,
exploiting the resonant gain rather than fighting the DC attenuation.

#### 4d. Relationship to Physical Oscillatory Networks

In a physical implementation (neuromorphic hardware, analog
circuits), the oscillators run continuously and cannot be "set to
omega=0." The EP teaching signal would need to work with the
natural oscillatory dynamics. The holomorphic EP paper's key
insight — that a periodic teaching signal at frequency omega_teach
encodes the gradient in the first Fourier coefficient of the neural
response — maps naturally onto this scenario.

The current implementation uses discrete contour points (beta_n on
a circle) rather than a continuous oscillating beta(t). Connecting
these: in continuous time, `beta(t) = r*exp(i*omega_teach*t)` traces
the contour continuously, and the Fourier extraction happens via
temporal integration over one teaching period.

For this to work, the teaching frequency omega_teach must be
distinct from the carrier frequency omega, so the teaching response
can be spectrally separated from the natural oscillation — the
epicycle picture. The implementation would need to:
1. Run the coupled oscillator network continuously
2. Apply a periodic teaching signal at omega_teach
3. Extract the gradient as the component of the neural response at
   omega_teach, via bandpass filtering or correlation

This is the most physically motivated approach and would bring
the implementation closest to the original hEP vision, but it
requires solving the inter-layer coupling problem (4b) first.

#### 4e. Gradient Magnitude Scaling

The hEP gradient is ~30x smaller than the true gradient (as measured
by finite differences). This ratio may depend on the network size,
depth, weight scale, and dynamics parameters. Understanding and
controlling this ratio is important for practical training — it
determines the effective learning rate.

The magnitude gap likely stems from the 1/|k| gain in the
equilibrium: the states z* are O(1/|k|) times the input, so the
Hebbian product z*z_prev^T is O(1/|k|^2) smaller than the direct
gradient. Compensating this analytically (by scaling the hEP
gradient by |k|^2) could bridge the gap, but needs verification
that it doesn't affect the gradient direction.

### 5. Summary of Key Lessons

1. **The energy function must include the oscillator self-energy.**
   Without `(1/2)<z, Kz>`, the EP equilibrium doesn't correspond
   to the ODE steady state, and the gradient theorem fails.

2. **The sign convention matters.** EP energy Phi = E - beta*C
   means dPhi/dW is the NEGATIVE of the loss gradient.

3. **Inference and training paths must be aligned.** Computing loss
   from the Lux forward pass while training with hEP dynamics
   produces zero learning signal because they're different functions.

4. **Holomorphic normalization is impossible.** Use holotanh for
   soft saturation and handle magnitude via weight scaling.

5. **Omega=0 is a valid starting point.** The oscillatory dynamics
   are separable from the EP learning dynamics. Train the weights
   with EP at omega=0, then deploy with omega>0 for temporal
   processing. Bridging this gap is the next research challenge.
