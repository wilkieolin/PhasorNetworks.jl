# Holomorphic Equilibrium Propagation for PhasorNetworks.jl

## 0. Conceptual Introduction

### Backpropagation vs Equilibrium Propagation

**Backpropagation** is a bookkeeping algorithm. You run a network forward, get
a loss, then meticulously trace backward through every operation asking "how
much did *you* contribute to the error?" The forward pass transforms data; the
backward pass transforms error signals. They share the same weights but flow in
opposite directions through different computational graphs. The backward pass
requires storing all intermediate activations — nature doesn't have a "backward
mode."

**Equilibrium propagation** takes a different philosophy. Instead of separate
forward/backward computations, it uses a single dynamical process. You define
an energy function over all neurons simultaneously, and the network "computes"
by relaxing to a minimum of this energy — like a ball rolling to the bottom of
a bowl. This settling process *is* the forward pass.

To get gradients, you don't trace backward. You **nudge** the system: add a
small term to the energy that gently pushes the output toward the target. With
`beta = 0` (free phase), the system settles to its natural equilibrium — its
prediction. With small `beta > 0` (nudged phase), the equilibrium shifts
slightly toward the correct answer. The difference between where the system
settles in these two cases encodes the gradient direction.

### How the Nudge Propagates Through the Network

A natural question: doesn't information from the output still need to reach
hidden layers? Yes — but the mechanism differs fundamentally from backprop.

In EP, the update rule for a hidden neuron depends on **both** the layer below
and the layer above. For a 3-layer MLP (input `x`, hidden `h`, output `o`):

    h <- activation(W1 * x + W2^T * o)
    o <- activation(W2 * h)

The same weight `W2` sends activity forward (in the `o` update) and influence
backward (in the `h` update, via `W2^T`). This is the **weight symmetry**
requirement — symmetric connections guarantee the dynamics have a consistent
energy landscape (Lyapunov function) that always decreases, ensuring
convergence.

When you nudge the output, the output neurons shift, which changes the
`W2^T * o` term in the hidden layer update, which shifts `h`, which feeds back
into `o`, and so on until a new equilibrium. The perturbation propagates
backward through the **same settling dynamics** used for inference — there is
no separate backward pass. The network re-equilibrates, and during
re-equilibration, the nudge's influence naturally diffuses backward.

**Analogy**: In a room of people holding hands in a chain, backprop is someone
at the end whispering a message passed person-by-person to the start. EP is
someone at the end pulling the chain — everyone feels the tug and adjusts
simultaneously, and after some jostling, everyone reaches a new comfortable
stance. The same connections that transmit forces forward also transmit them
backward.

### The Small-Beta Problem

Standard EP has a catch: exact gradients require `beta -> 0`. At finite beta,
the nudge distorts the energy landscape enough to introduce bias (in the
statistical sense) into the gradient estimate.

- **Small beta**: Exact gradients, but tiny signal — the free and nudged
  equilibria are nearly identical (bad for noise and finite precision)
- **Large beta**: Strong signal, but biased gradients — the linear
  approximation breaks down, and this worsens with network depth

### Holomorphic EP: The Contour Trick

Holomorphic EP resolves this by extending beta into the **complex plane**. The
key insight comes from complex analysis: if a function is holomorphic
(complex-differentiable), its derivative at any point is completely determined
by its values on any circle around that point (the Cauchy integral formula).

Applied to EP: the gradient we want is `dE/d(beta)` at `beta = 0`. If the
energy is holomorphic in beta, then:

1. Choose a circle of radius `r` around `beta = 0` in the complex plane
2. Evaluate the energy gradient at N evenly-spaced points on this circle:
   `beta_n = r * exp(2*pi*i*n/N)`
3. The gradient at the center equals the **first Fourier coefficient** of
   these samples

This is exact for any radius `r`. Large nudge, strong signal, no bias. The
higher-order contamination that plagues real-valued EP gets distributed across
higher Fourier harmonics and is automatically filtered out.

### The Epicycle Picture

For oscillatory systems like phasor networks, the holomorphic extension has a
beautiful physical interpretation.

Each neuron's potential `z(t)` already traces a spiral in the complex plane —
oscillating at frequency `omega` while decaying at rate `lambda`. This is the
natural dynamics (the free phase). Now add the teaching signal: a perturbation
at a different frequency `omega_teach`. The potential traces an **epicycle** —
the original spiral with a secondary oscillation superimposed.

These components are spectrally separable:

- The natural dynamics live at frequency `omega`
- The teaching response lives at frequency `omega_teach`
- The **gradient** is encoded in the amplitude and phase of the response at
  the teaching frequency

This is the same principle as **lock-in amplification** in experimental
physics: multiply the signal by a reference at the known frequency, average
over time, and everything at other frequencies cancels out.

Reading the epicycle:

- The **radius** of the epicycle = gradient magnitude (how sensitive this
  neuron is to the output error)
- The **phase** of the epicycle = gradient direction (should the weight
  increase or decrease?)

Every neuron traces its own epicycle. Every weight's gradient can be read off
from the relationship between the epicycles of connected neurons. All of this
happens during a single forward simulation — no backward pass, no adjoint
equations, just oscillation and spectral analysis.

### Why Phasor Networks Are a Natural Fit

In a generic neural network, you must artificially complexify everything —
real activations become complex, real weights operate on complex states, and
you need special holomorphic activations. This works but feels forced.

In phasor networks, the fit is natural:

- The state `z` is already complex
- The dynamics are already oscillatory
- Information is already encoded in phase relationships
- Adding a teaching oscillation is just adding another input current to the
  oscillator bank — the same operation the network already performs
- The linear ODE `dz/dt = k*z + W*I(t)` is automatically holomorphic — no
  conjugation anywhere in the dynamics
- Superposition is exact in the linear system: the teaching frequency response
  doesn't interfere with the natural frequency response

The nonlinear activation between layers does generate some intermodulation
(responses at combination frequencies). But the contour integration with N
points handles this — higher harmonics map to higher Fourier coefficients and
are filtered out when extracting only the first. More contour points reject
more harmonics, at the cost of more forward evaluations.

---

## 1. Background: Equilibrium Propagation

### 1.1 Standard Equilibrium Propagation (EP)

Equilibrium propagation (Scellier & Bengio, 2017) is a biologically plausible
alternative to backpropagation for training energy-based neural networks. The
core idea is that gradient information can be extracted from the difference in
network behavior between two phases:

**Free phase**: The network settles to an equilibrium of its energy function
with no external teaching signal:

    s* = argmin_s E(s, theta)

where `s` denotes hidden neuron states and `theta` denotes parameters (weights,
biases).

**Nudged phase**: A teaching signal is applied by adding a cost term `C(s, y)`
scaled by a small factor `beta`:

    s_beta = argmin_s [E(s, theta) + beta * C(s, y)]

**Gradient theorem (Scellier & Bengio)**: In the limit `beta -> 0`:

    dL/dtheta = lim_{beta->0} (1/beta) * [dE/dtheta|_{s_beta} - dE/dtheta|_{s*}]

This provides a local learning rule: the parameter gradient is proportional to
the difference in energy gradients between nudged and free equilibria, divided
by the nudge strength.

### 1.2 Limitations of Standard EP

1. **Infinitesimal beta**: Exact gradients require `beta -> 0`, but small `beta`
   means tiny differences between the two phases, leading to poor signal-to-noise
   in finite-precision or noisy systems.
2. **Two separate phases**: Training requires distinct free and nudged temporal
   phases with precise timing.
3. **Scaling**: Finite `beta` introduces a systematic bias in the gradient estimate
   that worsens with network depth.

---

## 2. Holomorphic Equilibrium Propagation (hEP)

### 2.1 Key Insight: Complex Extension

Laborieux & Zenke (NeurIPS 2022, arXiv:2209.00530) resolve these limitations by
extending EP to the complex plane via holomorphic functions.

The central observation: if the energy function `E(s, theta)` and its equilibrium
map `s*(beta, theta)` are **holomorphic** (complex-differentiable) functions of
`beta`, then the Cauchy integral formula provides an exact expression for the
derivative at `beta = 0` from a contour integral at **finite** `|beta|`:

    dE/dtheta|_{beta=0} = (1/2*pi*i) * oint_{|beta|=r} [dE/dtheta|_{s*(beta)}] / beta^2 d(beta)

This contour integral can be evaluated numerically by sampling `N` equally-spaced
points on a circle of radius `r` in the complex `beta`-plane:

    beta_n = r * exp(2*pi*i*n / N),    n = 0, 1, ..., N-1

### 2.2 The Holomorphic Gradient Formula

The gradient is computed as the **first Fourier coefficient** of the energy
gradient evaluated at complex nudge values around the contour:

    dL/dtheta = (1/N) * sum_{n=0}^{N-1} [dE/dtheta|_{s*(beta_n)}] * exp(-2*pi*i*n / N)

Equivalently, this extracts the linear coefficient of the Laurent expansion. The
key properties:

- **Exact for any `r > 0`**: Unlike standard EP, this gives exact gradients
  even for large nudge amplitudes, as long as the equilibrium map is holomorphic
  inside the contour.
- **N = 2 suffices**: With just two evaluations (at `beta = +r` and `beta = -r`),
  we recover the symmetric finite-difference formula `(f(r) - f(-r)) / 2r`, but
  the complex contour formulation generalizes to handle higher-order corrections.
- **Noise robustness**: Averaging over N contour points provides natural denoising.

### 2.3 Holomorphic Activation Functions

For the energy function and equilibrium to be holomorphic in `beta`, the
activation functions must be complex-differentiable. The reference implementation
uses:

- **holotanh**: A complex-valued activation blending sigmoid components that is
  holomorphic (satisfies Cauchy-Riemann equations), unlike standard ReLU or
  tanh applied element-wise to real/imaginary parts.
- **Identity**: Trivially holomorphic.

The energy function takes the form:

    phi(s, theta, beta) = sum_l [sum(layer_l(s_l) * s_{l+1})] - beta * C(s, y)

where `C` is the cost (e.g. cross-entropy). When activations and cost are
holomorphic in the hidden states, the equilibrium map inherits holomorphicity
in `beta`.

### 2.4 Oscillatory Interpretation

A striking physical interpretation: when `beta` traces a circle in the complex
plane over time:

    beta(t) = r * exp(i * omega * t)

the network states oscillate at the teaching frequency `omega`. The gradient is
the **first Fourier coefficient** of these oscillations — extractable via a
simple temporal bandpass filter.

This eliminates the need for separate free/nudged phases entirely: the teaching
signal is always on, oscillating, and the gradient is continuously available
from the neural oscillation spectrum.

### 2.5 Training Algorithm

```
for each batch (x, y):
    # 1. Free phase: relax to equilibrium with beta=0
    s = init_neurons(x)
    for t in 1:T1:
        s = activation(grad_s(phi(s, theta, beta=0)))  # gradient ascent on energy

    s_free = s

    # 2. Complex contour evaluation
    for n in 0:N-1:
        beta_n = r * exp(2*pi*i*n / N)

        # Convert to complex-valued network
        s_n = to_complex(s_free)
        for t in 1:T2:
            s_n = activation(grad_s(phi(s_n, theta, beta_n)))  # holomorphic grad

        # Accumulate: dE/dtheta at this contour point
        grad_n = dE/dtheta|_{s_n, beta_n}
        grads += grad_n  (taking real part of weighted sum)

    # 3. Parameter update
    theta -= lr * (1/N) * real(grads)
```

The reference implementation (JAX) supports both MLP and CNN architectures and
achieves ImageNet 32x32 performance matching backpropagation.

---

## 3. PhasorNetworks.jl Architecture and Integration Points

### 3.1 The Unified SSM Equation

PhasorNetworks.jl is built on a single dynamical equation:

    dz_c/dt = k_c * z_c + W * I(t),    k_c = lambda_c + i*omega_c

This is a linear complex ODE with per-channel eigenvalue `k_c`. Crucially, this
is a **linear system**, which is automatically holomorphic in any parameter.

### 3.2 Computational Modes

The same dynamics are evaluated via four equivalent modes:

| Mode | Input Type | Implementation | File |
|------|-----------|----------------|------|
| 2D Complex | `(C, B)` | `W*x + bias` | network.jl:347 |
| 3D Complex | `(C, L, B)` | `W*x` then `causal_conv(K, H)` | network.jl:363 |
| 3D Phase | `(C, L, B)` | `causal_conv_dirac` | network.jl:408 |
| CurrentCall (ODE) | continuous | `solve(ODEProblem(dzdt, ...))` | network.jl:448 |

The discrete kernel connecting all modes:

    K[n] = A^n * B,    A = exp(k * dt),    B = (A - 1) / k

### 3.3 Current Training: Backpropagation Through Time

The existing `train()` function (network.jl:1574) uses standard automatic
differentiation:

```julia
lf = p -> loss(x, y, model, p, st)
lossval, gs = withgradient(lf, ps)
opt_state, ps = Optimisers.update(opt_state, ps, gs[1])
```

For ODE paths, this uses `BacksolveAdjoint` with `ZygoteVJP` for sensitivity
analysis through `DifferentialEquations.jl`. This is effective but:

- Requires storing/reconstructing the full forward trajectory
- Memory scales with sequence length
- Gradient computation is inherently sequential (adjoint ODE backward pass)
- Not biologically plausible

### 3.4 Energy Structure

The PhasorNetworks system has a natural energy interpretation. The linear ODE:

    dz/dt = k*z + W*I(t)

can be viewed as gradient flow on an energy landscape. For the autonomous part
(`I = 0`), the Lyapunov function is:

    V(z) = -Re(z^H * diag(k) * z)

since `lambda < 0` makes this a dissipative system that minimizes `V`. The
driven system (`I != 0`) reaches a steady state where the oscillators phase-lock
to the input — an equilibrium of the combined energy.

### 3.5 Readout and Loss

Two readout mechanisms exist:
- **Codebook** (network.jl:937): Stores prototype phase vectors per class;
  computes cosine similarity `cos(pi * (phase - code))` as logits.
- **SSMReadout** (ssm.jl:89): Temporal variant — computes codebook similarity
  at each timestep, then averages over a readout window.

Both produce scalar logits from phase states, serving as the cost function
`C(s, y)` needed for EP.

---

## 4. How hEP Maps onto PhasorNetworks

### 4.1 Natural Compatibility

PhasorNetworks is arguably a more natural fit for hEP than the Hopfield
networks used in the original paper, for several reasons:

1. **Already complex-valued**: The core representation is `z in C`. No need to
   "extend" to complex — the network is natively complex.

2. **Already oscillatory**: The eigenvalue `k = lambda + i*omega` produces
   natural oscillations. hEP's "teaching oscillation" interpretation maps
   directly onto modulating these existing oscillations.

3. **Linear dynamics are holomorphic**: The linear ODE `dz/dt = k*z + W*I(t)`
   is holomorphic in all parameters (`W`, `k`, `z`) automatically. The solution
   `z(t) = exp(k*t)*z(0) + integral(exp(k*(t-s))*W*I(s) ds)` is an entire
   function of the parameters.

4. **Equilibrium exists**: With `lambda < 0`, the system is dissipative and
   reaches a periodic steady state (phase-locked to input). This steady state
   is the "equilibrium" that EP requires.

### 4.2 The Phasor EP Energy Function

We can define an energy function for the phasor network that combines the
oscillator dynamics with a task term:

    Phi(z, theta, beta) = E_dynamics(z, theta) - beta * C(z, y)

where:
- `E_dynamics` captures the oscillator energy: for linear dynamics, this is
  `V(z) = -Re(sum(conj(z) .* (k .* z + W * I)))`, the Lyapunov function
  whose gradient flow gives the ODE.
- `C(z, y)` is the task cost, e.g. codebook cross-entropy:
  `C = -sum(y .* log_softmax(similarity(phase(z), codes)))`
- `beta in C` is the complex nudge parameter

### 4.3 The Holomorphic Gradient via Oscillation

In the phasor context, the hEP procedure becomes:

1. **Free phase**: Run the ODE `dz/dt = k*z + W*I(t)` to steady state (or
   for a fixed number of periods). This is exactly what `oscillator_bank`
   already does.

2. **Nudged oscillation**: Add a teaching current proportional to `beta * dC/dz`:

       dz/dt = k*z + W*I(t) + beta(t) * dC/dz

   where `beta(t) = r * exp(i*omega_teach*t)` oscillates at a teaching frequency.

3. **Gradient extraction**: The parameter gradient equals the first Fourier
   coefficient of `dE/dtheta` evaluated along the oscillating trajectory:

       dL/dtheta = (1/T) * integral_0^T [dE/dtheta|_{z(t)}] * exp(-i*omega_teach*t) dt

   which can be computed via a simple correlation/bandpass filter.

### 4.4 Connection Between Continuous and Discrete Modes

The equivalence between ODE and discrete (convolutional/FFT) modes in
PhasorNetworks means that hEP results proven in continuous time automatically
transfer to the discrete domain:

| Continuous (ODE) | Discrete (Conv/FFT) |
|-----------------|---------------------|
| `dz/dt = k*z + W*I(t)` | `z[n+1] = A*z[n] + B*W*I[n]` |
| Steady-state oscillation | Converged recurrence |
| Teaching current `beta(t)*dC/dz` | Teaching input `beta_n*dC/dz[n]` |
| Fourier coefficient | Discrete Fourier transform |
| Contour integral `oint f(beta)/beta^2 dbeta` | Sum over N roots of unity |

The discrete kernel `K[n] = A^n * B` is a rational function of `k = lambda + i*omega`,
hence holomorphic in the dynamics parameters. The causal convolution `z = K * I`
is linear in `W` and holomorphic in `k`, so the entire forward map is
holomorphic in parameters — exactly the condition hEP requires.

For the **FFT-based** causal convolution path, the gradient extraction via
Fourier coefficient is particularly natural: the FFT is already computing
frequency-domain representations, and extracting the teaching frequency
component is a single index lookup.

---

## 5. Proposed Implementation Plan

### 5.1 Phase 1: Minimal hEP for 2D Phase Networks

**Goal**: Prove the concept with the simplest path — 2D phase input (single-step,
no temporal dynamics).

**Architecture**:
```
Input (Phase) → PhasorDense → PhasorDense → Codebook → Loss
```

**Implementation**:
1. Define an energy function `phi(states, params, beta, x, y)` that sums
   layer-wise interaction energies plus `beta * cost`:
   ```julia
   function hep_energy(states, params, model, x, y, beta)
       # Layer-wise energy: sum of inner products between layer outputs
       phi = sum(real.(conj.(layer_out) .* next_state) for each layer pair)
       # Task cost (codebook similarity cross-entropy)
       logits = codebook_similarity(states[end], codes)
       cost = crossentropy(logits, y)
       return phi - beta * cost
   end
   ```

2. Equilibrium dynamics: iterate `s = activation(grad_s(phi))` for T steps.
   For 2D phase networks, this is a fixed-point iteration.

3. Contour integration: evaluate at `N` complex `beta` values, extract gradient.

**Key files to modify**:
- New file: `src/hep.jl` — energy function, equilibrium solver, contour gradient
- `src/network.jl` — add holomorphic activation option to PhasorDense
- `src/PhasorNetworks.jl` — include and export new functions

**Test**: Train on MNIST with 2-layer MLP, verify gradient agreement with
backpropagation (cosine similarity > 0.99).

### 5.2 Phase 2: hEP for 3D Temporal (SSM) Networks

**Goal**: Extend to the temporal/sequence domain using causal convolution.

**Architecture**:
```
PSK Encode → PhasorDense (SSM, 3D) → PhasorDense (SSM, 3D) → SSMReadout → Loss
```

**Implementation**:
1. The energy function for the temporal case must account for the causal
   structure. For each layer:
   ```
   E_layer = sum(real(conj(z_out) .* causal_conv(K, W * z_in)))
   ```
   where `K` is the phasor kernel.

2. Equilibrium is the fixed point of the discretized recurrence. Since the
   system is linear with `lambda < 0`, convergence is guaranteed.

3. The teaching signal modulates the cost term's strength along the contour.

**Test**: Train on FashionMNIST sequential (28-step), compare gradients and
convergence with BPTT.

### 5.3 Phase 3: hEP for Continuous ODE (Spiking) Networks

**Goal**: The most natural and biologically plausible mode — continuous-time
hEP using the oscillator bank ODE solver.

**Architecture**:
```
Phase → MakeSpiking → PhasorDense (ODE) → PhasorDense (ODE) → Codebook → Loss
```

**Implementation**:
1. Modify `oscillator_bank` to accept an additional teaching current:
   ```julia
   function dzdt_hep(u, p, t; teaching_fn, beta_fn)
       k = lambda + i*omega
       I = W * input_current(t)
       teaching = beta_fn(t) * teaching_fn(u, y)
       return k .* u .+ I .+ teaching
   end
   ```
   where `beta_fn(t) = r * exp(i * omega_teach * t)`.

2. Run the ODE for multiple teaching oscillation cycles.

3. Extract the gradient as the first Fourier coefficient of `dE/dtheta`
   over the oscillation trajectory.

4. **Key advantage**: This is a single forward pass — no adjoint ODE needed.
   The gradient comes from the oscillation spectrum, not from backpropagation
   through the solver.

**Relation to spiking**: In spiking mode, the teaching signal would manifest
as a periodic modulation of the output neuron's firing threshold or injected
current, synchronized at frequency `omega_teach`. The gradient is extracted
from the spike timing modulation — a plausible neural mechanism.

**Test**: Simple classification task, verify gradient quality vs adjoint method,
measure wall-clock speedup.

### 5.4 Phase 4: Holomorphic Activation Functions

The current activation `normalize_to_unit_circle` (projection onto the unit
circle) is **not** holomorphic — it involves `|z|` which uses complex
conjugation. For exact hEP gradients, we need holomorphic alternatives:

1. **Complex softmax normalization**:
   `sigma(z) = z / sum(exp(|z|))` — approximately holomorphic for bounded inputs.

2. **holotanh** (from the reference implementation):
   A complex-valued function built from sigmoid components that satisfies
   Cauchy-Riemann equations.

3. **Identity / linear**: Trivially holomorphic. May work for shallow networks.

4. **Complex polynomial**: `f(z) = z - z^3/3` — bounded, holomorphic, similar
   shape to tanh near origin.

**Note**: The linearity of the PhasorNetworks ODE is actually an advantage here.
Since the dynamics `dz/dt = k*z + W*I(t)` are already holomorphic, the main
concern is the activation applied between layers and in the readout.

---

## 6. Connections and Trade-offs

### 6.1 hEP vs Backpropagation Through Time (BPTT)

| Aspect | BPTT (current) | hEP (proposed) |
|--------|---------------|----------------|
| Memory | O(L * C) for adjoint | O(C) — single forward pass |
| Compute | Forward + backward ODE | N forward ODEs (contour points) |
| Gradient quality | Exact (up to solver error) | Exact (if holomorphic conditions met) |
| Biological plausibility | None | High — local, oscillation-based |
| Hardware mapping | GPU via AD | Neuromorphic via oscillation |

### 6.2 hEP vs Direct Loss Gradient (Discrete Mode)

In the discrete (3D complex) mode, PhasorNetworks currently computes gradients
via `Zygote.withgradient` through the causal convolution. hEP would instead:

1. Run the causal convolution to equilibrium (same forward pass)
2. Perturb with complex `beta` at N points (N additional forward passes)
3. Extract gradient from the Fourier structure

This is **slower** for the discrete mode (N extra forward passes vs one backward
pass) but provides:
- A gradient that doesn't require AD at all
- A path to training on analog/neuromorphic hardware
- A unifying framework across all computational modes

### 6.3 When Each Mode is Preferred

- **Discrete + AD**: Fastest for GPU training. Use for research/prototyping.
- **ODE + hEP**: Best for spiking/neuromorphic deployment. Single forward pass,
  local learning rule, directly maps to oscillatory hardware.
- **Discrete + hEP**: Useful as a validation bridge — verify hEP gradients match
  AD gradients in the discrete domain before moving to continuous.

---

## 7. Detailed First Experiment: hEP Gradient Verification

### 7.1 Setup

```julia
# Minimal 2-layer network
model = Chain(
    PhasorDense(784 => 128, holotanh; use_bias=true),
    PhasorDense(128 => 10, identity; use_bias=true),
    Codebook(10, 10)
)

# Initialize
rng = Random.default_rng()
ps, st = Lux.setup(rng, model)

# Single batch from MNIST
x = Phase.(rand(Float32, 784, 32) .* 2f0 .- 1f0)
y = Flux.onehotbatch(rand(0:9, 32), 0:9)
```

### 7.2 Backpropagation Gradient (Reference)

```julia
loss_fn(p) = codebook_loss(model, x, y, p, st)
_, bp_grads = Zygote.withgradient(loss_fn, ps)
```

### 7.3 hEP Gradient

```julia
function hep_gradient(model, ps, st, x, y; N=4, r=0.5f0)
    # Free phase: forward pass to equilibrium
    states_free = hep_forward(model, ps, st, x, beta=0f0)

    # Contour integration
    grads = zero_like(ps)
    for n in 0:N-1
        beta_n = r * exp(2f0im * pi * n / N)
        states_n = hep_forward(model, ps, st, x, beta=beta_n, y=y, init=states_free)
        grad_n = hep_energy_gradient(model, ps, states_n, x, y, beta_n)
        grads = grads .+ real.(grad_n .* exp(-2f0im * pi * n / N))
    end
    return grads ./ N
end
```

### 7.4 Verification

```julia
# Compare gradients
cos_sim = cosine_similarity(flatten(bp_grads), flatten(hep_grads))
@test cos_sim > 0.99

# Sweep beta amplitude
for r in [0.01, 0.1, 0.5, 1.0, 2.0]
    hep_g = hep_gradient(model, ps, st, x, y; r=r)
    sim = cosine_similarity(flatten(bp_grads), flatten(hep_g))
    println("r=$r: cosine_sim=$sim")
end
```

### 7.5 Training Loop

```julia
function train_hep(model, ps, st, train_loader; epochs=5, lr=0.001, N=4, r=0.5)
    opt_state = Optimisers.setup(Optimisers.Adam(lr), ps)

    for epoch in 1:epochs
        for (x, y) in train_loader
            grads = hep_gradient(model, ps, st, x, y; N=N, r=r)
            opt_state, ps = Optimisers.update(opt_state, ps, grads)
        end
    end
    return ps
end
```

---

## 8. Open Questions and Research Directions

1. **Activation holomorphicity**: How much does violating the holomorphic
   condition (via `normalize_to_unit_circle` or phase extraction) degrade
   gradient quality? The reference paper shows robustness to approximate
   holomorphicity — quantify this for phasor activations.

2. **Multi-layer equilibrium**: For deep networks, does the equilibrium
   propagation dynamics converge reliably? The linear ODE guarantees
   convergence of individual layers, but inter-layer coupling may introduce
   nonlinear dynamics.

3. **Teaching frequency selection**: What `omega_teach` works best? Should it
   be commensurate with the network's natural oscillation frequencies `omega_c`,
   or incommensurate to avoid interference?

4. **Spiking realization**: Can the teaching oscillation be implemented as a
   periodic modulation of spike thresholds? This would make hEP directly
   implementable in neuromorphic hardware.

5. **Codebook gradient**: The codebook readout involves `cos(pi * (phase - code))`
   which is not holomorphic (it uses real-valued phase extraction). Can we
   reformulate the cost in terms of complex inner products to maintain
   holomorphicity?

6. **Scaling**: How does hEP scale with the number of oscillator channels and
   sequence length? The contour integration adds a factor of N to forward pass
   cost, but eliminates the backward pass entirely.

---

## References

- Laborieux, A. & Zenke, F. (2022). "Holomorphic Equilibrium Propagation
  Computes Exact Gradients Through Finite Size Oscillations." NeurIPS 2022.
  arXiv:2209.00530.
- Scellier, B. & Bengio, Y. (2017). "Equilibrium Propagation: Bridging the Gap
  Between Energy-Based Models and Backpropagation." Frontiers in Computational
  Neuroscience.
- Reference implementation: https://github.com/Laborieux-Axel/holomorphic_eqprop
