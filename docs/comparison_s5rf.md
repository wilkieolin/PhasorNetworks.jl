# Comparison: PhasorNetworks SSM vs S5-RF (Huber et al., 2025)

arXiv:2504.00719 — "Scaling Up Resonate-and-Fire Networks for Fast Deep Learning"

## 1. Shared Foundation

Both approaches start from the same R&F neuron ODE:

| | S5-RF | PhasorNetworks |
|---|---|---|
| ODE | `dz/dt = (-b + iw)z + I(t)` | `dz/dt = (l + iw)z + I(t)` |
| Notation | b > 0 (decay), w (frequency) | l < 0 (leakage), w = 2pi/T |
| State | Complex z(t) | Complex z(t) |
| Relationship | b = -l (identical dynamics) | l = -b (identical dynamics) |

The continuous dynamics are mathematically identical. Both recognize the R&F neuron as a 1D complex (2D real) linear dynamical system — a diagonal SSM with a single complex eigenvalue k = -b + iw = l + iw.

## 2. Discretization: Dirac vs Zero-Order Hold

This is the most substantive technical difference.

### S5-RF: Dirac Discretization

S5-RF models inter-neuron communication as weighted Dirac delta spikes:

```
u(t) = sum_n delta(t - t_n) * u_n
```

Substituting into the ODE's integral solution and evaluating the delta:

```
x_k = exp(D*A) * x_{k-1} + B * u_k
```

Note: the B term has **no gain correction** — it is just `B`, not `(A-1)/k * B`. This is because a Dirac delta has infinite amplitude but zero width, so the integral of `exp(k*(t-tau)) * delta(tau - t_k)` evaluates to just `exp(0) * B = B` at the spike time.

They add a learnable scalar eta > 0:

```
A_bar = exp(eta * D * A)
B_bar = eta * B
```

This eta absorbs the effective "step size" and is learned per layer.

### PhasorNetworks: Zero-Order Hold Discretization

PhasorNetworks assumes piecewise-constant input over each timestep (standard ZOH):

```
z[n+1] = exp(k*Dt) * z[n] + (exp(k*Dt) - 1)/k * I[n]
```

So:
```
A = exp(k*Dt)
B = (A - 1) / k
```

The B term includes the integral correction `(A-1)/k` which accounts for the input being held constant over the interval, not instantaneous.

### Why This Matters

The Dirac discretization is physically correct for **spike-mediated communication**: a spike is an instantaneous event, so the input integral collapses to a point evaluation. The ZOH discretization is correct for **continuous current input**: the signal is held constant between samples.

In practice:
- S5-RF uses Dirac for intermediate spiking layers (where inputs are actual spikes) and ZOH for the input layer (where data is continuous)
- PhasorNetworks uses ZOH for its convolutional/discrete mode (where inputs are sampled complex signals) and the full ODE solver for its spiking mode (where spikes are converted to continuous currents via a kernel before integration)

**Key insight**: PhasorNetworks' spiking pathway actually implements something closer to S5-RF's Dirac discretization implicitly — when a spike is converted to a raised cosine current pulse and integrated by the ODE solver, the result approaches the Dirac evaluation as the pulse width approaches zero. The `spike_current` function with `t_window -> 0` converges to the Dirac limit.

### Connection to Our Analytic Phase Solution

Both discretizations exploit the same underlying fact: the R&F ODE has a closed-form solution. For PhasorNetworks, the analytic solution is explicit in `phasor_kernel`:

```
K[n] = A^n * B = exp(k*n*Dt) * (exp(k*Dt) - 1) / k
```

For S5-RF's Dirac case, the kernel simplifies to:

```
K[n] = A^n * B = exp(k*n*Dt) * B
```

The only difference is the gain term: `(exp(k*Dt) - 1)/k` vs `B` (or `eta*B` with the learnable scalar). For small `|k*Dt|`, these converge: `(exp(k*Dt) - 1)/k ≈ Dt`, so the ZOH gain is approximately `Dt * B`, while the Dirac gain is `eta * B`. In practice, the learnable eta in S5-RF can absorb this difference.

## 3. The Fundamental Semantic Difference: What Do Spikes Mean?

This is the most important conceptual distinction between the two approaches.

### S5-RF: Spikes as Thresholded Activations

S5-RF treats spikes as **binary activations** produced by thresholding the real part of the membrane potential:

```
s = H(Re(x) - xi)    (Heaviside function, xi = 1)
```

A spike fires whenever Re(z) > 1. The spike carries no information beyond "this neuron fired at this time." There is no prescribed relationship between spike timing and phase. The spike is a nonlinear activation function analogous to ReLU — it gates the output, creating sparsity, but the specific timing within a period has no semantic meaning.

S5-RF does not use refractory periods, so a neuron can fire multiple spikes per oscillation cycle. The natural oscillatory dynamics cause spiking to stop on its own as the potential decays.

### PhasorNetworks: Spikes as Phase Communicators

PhasorNetworks assigns **explicit semantic meaning** to spike timing:

```
spike_time = phase_to_time(theta, period)
theta = time_to_phase(spike_time, period)
```

A spike encodes the **phase** of the neuron's oscillation at the moment of peak voltage (Im(z) maximum). The spike time within the oscillation period directly maps to the phase value theta in [-1, 1]. This phase is the neuron's output — it represents the computed result in the phasor algebra.

Furthermore, **all neurons in a layer share the same resonant frequency** (in the standard R&F mode). This shared frequency acts as a clock: it defines the reference frame against which spike times are measured to extract phases. Without a shared frequency, spike timing would be ambiguous — you wouldn't know which cycle a spike belongs to.

### Implications

| Aspect | S5-RF | PhasorNetworks |
|--------|-------|----------------|
| Spike meaning | Binary gate (fired/not) | Phase value (when within cycle) |
| Information encoding | Rate/activation level | Temporal phase |
| Shared frequency | No (per-channel eigenvalues) | Yes (within a layer), No (across SSM channels) |
| Multi-spike per cycle | Yes (no refractory period) | No (one spike = one phase per cycle) |
| VSA operations | Not applicable | Binding = phase addition, Bundling = complex addition |
| Readout | Linear projection on spike counts/states | Phase extraction + codebook similarity |

S5-RF's approach is more flexible but less interpretable — spikes are just sparse nonlinear activations. PhasorNetworks' approach is more constrained but enables a rich algebraic structure (VSA operations) on the outputs.

## 4. SSM Architecture Comparison

### S5-RF Architecture

- Uses the S5 (Simplified State Spaces for Sequence Modeling) architecture: parallel prefix scan for recurrence
- Diagonalized HiPPO matrix A_N with N complex eigenvalues
- V_N^{-1} * B projects input into eigenspace (dropped in Dirac layers)
- Spiking nonlinearity (Heaviside on Re(x)) between layers
- Surrogate gradient (multi-Gaussian) for backprop through spikes
- Deep: up to 4 layers demonstrated
- JAX/Equinox implementation

### PhasorNetworks Architecture

- Uses causal convolution (Toeplitz matrix or FFT) for the convolutional view
- Per-channel eigenvalues (l_c + iw_c) — same mathematical object as diagonal SSM
- Weight matrix W projects input channels to output channels
- Normalization to unit circle + phase extraction between layers
- Standard AD through complex operations (Zygote), no surrogate gradients needed
- Demonstrated with 2 layers + readout
- Julia/Lux implementation

### Key Architectural Differences

1. **Parallel scan vs convolution**: S5-RF uses the parallel prefix scan from S5 for O(L log L) recurrent computation. PhasorNetworks uses FFT-based causal convolution for O(L log L) convolutional computation. Both achieve the same asymptotic complexity but through different algorithms.

2. **Surrogate gradients vs smooth AD**: S5-RF needs surrogate gradients because the Heaviside spiking function is non-differentiable. PhasorNetworks avoids this entirely in its convolutional/phase mode — `normalize_to_unit_circle` and `complex_to_angle` are smooth functions, so standard AD works. The spiking mode in PhasorNetworks also avoids surrogate gradients by operating through the ODE solver with adjoint sensitivity (BacksolveAdjoint + ZygoteVJP).

3. **Depth**: S5-RF demonstrates 4 layers. PhasorNetworks typically uses 2 SSM layers + readout. This is likely a matter of engineering effort rather than fundamental limitation.

## 5. HiPPO Initialization

### S5-RF

Derives R&F eigenvalues directly from the HiPPO-LegS framework:
- Real part fixed at -0.5 for all channels
- Imaginary parts from the HiPPO A_N matrix eigenvalues
- Claims HiPPO init is less critical when eta is learned (ablation shows random init works nearly as well)

### PhasorNetworks

Implements HiPPO-LegS via `hippo_legs_diagonal`:
- Log-spaced decay magnitudes from 0.5 to N-0.5
- Frequencies paired to decay: w_n = pi * |l_n|
- Both lambda and omega are trainable

The practical difference: S5-RF fixes Re(k) = -0.5 and learns only the frequencies, while PhasorNetworks makes both decay and frequency trainable from the start. S5-RF's learnable eta effectively modulates the overall timescale, which partially compensates for the fixed decay.

Our demo (`demos/long_range_demo.jl`) adds an additional insight: the HiPPO eigenvalues must be **scaled to match the sequence length** when using Dt=1 (sample-indexed time). We divide by L so the slowest channel's memory window spans the full sequence. S5-RF handles this via the learnable eta parameter, which serves the same purpose.

## 6. The Phase Perspective: What S5-RF Doesn't Do

S5-RF operates entirely in the complex potential domain. It never extracts phases, never maps spike times to phase values, and never performs VSA operations. The output of each layer is a binary spike train, not a phase-valued vector.

PhasorNetworks' unique contributions relative to S5-RF:

1. **Phase as first-class citizen**: The `Phase` type, `complex_to_angle`, `normalize_to_unit_circle`, and the entire phase algebra infrastructure. Phases are the computed values; complex potentials are implementation details.

2. **VSA operations on neural outputs**: `v_bind` (phase addition = complex multiplication) and `v_bundle` (complex addition + phase extraction) define an algebraic structure on the network's representations. This connects R&F networks to holographic reduced representations, Fourier holographic encoding, and the broader VSA literature.

3. **Codebook-based classification**: The `SSMReadout` and `Codebook` layers compare output phases against learned prototypes via cosine similarity. This is a fundamentally different classification mechanism from the linear projections used in S5-RF.

4. **Three execution modes**: The same mathematical model can run as floating-point phases (fast), ODE-integrated potentials (accurate), or event-driven spikes (neuromorphic hardware). S5-RF has only the recurrent spiking mode (plus the parallel scan optimization).

## 7. Performance Comparison

| Benchmark | S5-RF | PhasorNetworks (current demo) |
|-----------|-------|-------------------------------|
| sMNIST | 98.89% (128 hidden, 2 layers, 1.5h) | 20.64% (64 hidden, 2 layers, 3 epochs) |
| psMNIST | 95.29% | Not tested |
| SHD | 91.86% | Not tested |
| SSC | 78.8% (SOTA for recurrent SNNs) | Not tested |

The performance gap reflects engineering maturity, not fundamental capability:
- S5-RF trains for many epochs with cosine annealing; our demo runs 3 epochs with fixed LR
- S5-RF uses 128-512 hidden dims; our demo uses 64
- S5-RF uses parallel prefix scan; we use FFT convolution (same O(L log L) complexity)
- S5-RF has been optimized for these benchmarks; our demo is a proof-of-concept

The core math is the same diagonal complex SSM. Given equivalent training infrastructure, performance should converge.

## 8. Summary of Key Differences

| Dimension | S5-RF | PhasorNetworks |
|-----------|-------|----------------|
| **Discretization** | Dirac (correct for spikes) + learnable eta | ZOH (correct for continuous signals) |
| **Spike semantics** | Binary threshold activation | Phase-encoding temporal code |
| **Shared frequency** | No (per-channel eigenvalues) | Yes (within standard layers), No (SSM mode) |
| **Nonlinearity** | Heaviside threshold (needs surrogate gradient) | Unit circle normalization (smooth, AD-friendly) |
| **VSA algebra** | Not present | Binding, bundling, similarity as first-class ops |
| **Readout** | Linear projection | Codebook similarity (phase-based) |
| **Parallelization** | Prefix scan (S5-style) | FFT convolution |
| **HiPPO init** | Fixed decay -0.5, learnable frequencies | Trainable decay and frequency, sequence-length-scaled |
| **Execution modes** | Recurrent spiking only | Phase, spiking, ODE, convolutional |
| **Implementation** | JAX/Equinox | Julia/Lux |
| **Depth demonstrated** | 4 layers | 2 layers + readout |
| **Benchmark maturity** | Full SHD/SSC/sMNIST results | Proof-of-concept demos |

## 9. What Each Approach Can Learn from the Other

### What PhasorNetworks can adopt from S5-RF:
- **Dirac discretization** for spiking layers — more physically correct than ZOH when the input is actual spikes, and simpler (no `(A-1)/k` gain term)
- **Learnable eta** — elegant way to handle timescale adaptation without explicit sequence-length scaling
- **Deeper architectures** (4+ layers) with proper initialization
- **Parallel prefix scan** as an alternative to FFT convolution for the recurrent view

### What S5-RF could adopt from PhasorNetworks:
- **Phase semantics** — assigning meaning to spike timing enables VSA operations and connects to a rich algebraic theory
- **Multiple execution modes** — the same model as atemporal phases, ODE potentials, or event-driven spikes
- **Codebook readout** — phase-based classification may offer advantages over linear readout for certain tasks
- **Smooth activation** (unit circle projection) — eliminates the need for surrogate gradients entirely

## 10. Conclusion

S5-RF and PhasorNetworks independently arrived at the same core insight: the R&F neuron is a diagonal SSM, and SSM techniques (HiPPO initialization, efficient parallelization) can be applied to scale R&F networks. The two approaches diverge on what happens at the **output boundary** — S5-RF produces binary spikes (thresholded activations), while PhasorNetworks produces phases (temporal codes with algebraic structure).

The Dirac discretization from S5-RF and the ZOH discretization from PhasorNetworks are two valid discretizations of the same ODE, appropriate for different input types (spikes vs continuous signals). The analytic phase solution that PhasorNetworks exploits — where the kernel `K[n] = A^n * B` provides a closed-form mapping from input phases to output phases — applies equally under both discretizations, just with different gain terms.

The deepest difference is philosophical: S5-RF treats R&F neurons as efficient sparse activators (a hardware-friendly alternative to ReLU), while PhasorNetworks treats them as phase computers (implementing a vector-symbolic algebra through oscillatory dynamics). Both perspectives are valid; they optimize for different goals (sparse efficiency vs representational structure).
