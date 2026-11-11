# Implementation Comparison: S5-RF (JAX) vs PhasorNetworks (Julia)

Source: https://github.com/ThomasEHuber/s5-rf (commit at time of analysis)

## 1. Core SSM Computation

### S5-RF: Parallel Prefix Scan

The SSM recurrence `x[n] = A_bar * x[n-1] + B_bar * u[n]` is computed via
`jax.lax.associative_scan` — a parallel prefix scan that computes all N
timesteps in O(log N) sequential depth on GPU:

```python
def binary_operator(q_i, q_j):
    A_i, b_i = q_i
    A_j, b_j = q_j
    return A_j * A_i, A_j * b_i + b_j

_, xs = jax.lax.associative_scan(binary_operator, (Lambda_elements, Bu_elements))
```

The `binary_operator` implements the associative combining rule for linear
recurrences: `(A₂, b₂) ∘ (A₁, b₁) = (A₂A₁, A₂b₁ + b₂)`. This is the
S5 method from Linderman et al.

**Cost**: O(P·L) work, O(P·log(L)) depth, where P = state dimension, L = sequence
length. Fully parallelized on GPU — no sequential bottleneck.

### PhasorNetworks: FFT Causal Convolution

The same recurrence is computed by unrolling to a causal convolution and
applying it via FFT:

```julia
K = exp.(k_c .* T .* ns)          # kernel: (C_out, L)
Z = causal_conv_fft(K, H)         # FFT-based convolution
```

**Cost**: O(C·L·log(L)) work. Also fully parallelized, but requires zero-padding
to 2L (doubling memory) and three FFT calls (kernel, input, output).

### Comparison

| Aspect | S5-RF (prefix scan) | PhasorNetworks (FFT) |
|--------|--------------------|--------------------|
| Asymptotic cost | O(P·L) | O(C·L·log(L)) |
| Sequential depth | O(log L) | O(1) (FFT is parallel) |
| Memory | O(P·L) | O(C·2L) (zero-padded) |
| AD support | JAX associative_scan | Zygote through fft/ifft ChainRules |
| Recurrent mode | Native (the scan IS the recurrence) | Would need separate implementation |
| GPU efficiency | Very high (fused scan kernel) | High (cuFFT is well-optimized) |

The prefix scan is theoretically superior: O(P·L) vs O(C·L·log(L)), and it
naturally supports the recurrent mode (online inference, streaming). The FFT
approach is batch-only — it needs the full input sequence.

## 2. Discretization

### S5-RF

Three options, selected per layer:

```python
# ZOH (first layer, continuous input):
Lambda_bar = exp(Lambda * Delta)
B_bar = (1/Lambda * (Lambda_bar - I)) * B_tilde

# Dirac (intermediate layers, spike input):
Lambda_bar = exp(Lambda * Delta)
B_bar = B_tilde * Delta         # note: just B * Delta, no (A-1)/k correction

# Bilinear (alternative):
BL = 1 / (I - (Delta/2) * Lambda)
Lambda_bar = BL * (I + (Delta/2) * Lambda)
B_bar = (BL * Delta) * B_tilde
```

Key detail: `Delta` is a **learnable per-channel parameter** (`log_step`),
initialized uniformly in `[log(eta_min), log(eta_max)]` where
`eta_min=0.001, eta_max=0.1`. This learnable step size absorbs the timescale
matching that PhasorNetworks handles via explicit eigenvalue scaling.

For the Dirac case, `B_bar = B_tilde * Delta` is notably simple — the input
matrix B is just scaled by Delta. No `(A-1)/k` correction, no coupled integral.
This is because they model spikes as simple weighted Dirac impulses with
amplitude scaled by Delta.

### PhasorNetworks

Single-oscillator Dirac (our current approach):

```julia
dt = T * (0.5 - θ/2)                    # sub-period timing from phase
enc = exp.(k_c .* dt)                    # per-channel response
H[c,n] = Σ_j W[c,j] * enc[c,j,n]       # weighted sum
K[n] = exp(k_c * n * T)                 # causal kernel
Z = causal_conv(K, H)                   # convolve
```

Our Dirac encoding analytically evaluates `exp(k_c · dt)` at the exact
sub-period arrival time of each spike. S5-RF does NOT do this — their Dirac
B_bar is `B * Delta`, with no phase-dependent timing. All spikes within a
timestep are treated as arriving at the same instant.

This is a fundamental difference: PhasorNetworks encodes **phase information
through spike timing within the period**, while S5-RF encodes information
through **spike amplitude** (binary 0/1 via Heaviside threshold).

## 3. Network Architecture

### S5-RF

```
Input (L, H)
  → RFDense: real weight W → complex via Vinv @ W     (L, H) → (L, P)
  → RF: SSM scan + V projection + Heaviside spike     (L, P) → (L, P) binary spikes
  → [repeat for each layer, Dirac discretization for layers > 0]
  → RFDense: output projection                        (L, P) → (L, n_classes)
  → LI: leaky integrator pooling                      (L, n_classes) → (n_classes)
  → mean pooling + log_softmax                         → (n_classes)
```

Key design choices:
- **Separate Dense + RF layers**: `RFDense` does input projection (W @ x), `RF`
  does SSM dynamics. The SSM's B matrix is identity (or Vinv for first layer).
- **Heaviside spiking**: `cartesian_spike(x) = H(Re(x) - 1)` with multi-Gaussian
  surrogate gradient. Output is binary {0, 1}.
- **Skip connections**: between RF layers (when `apply_skip=True`).
- **Leaky integrator readout**: `LI` applies a learnable sigmoid-gated exponential
  moving average, then mean-pools over time.
- **V/Vinv projections**: First layer uses the HiPPO eigenvector matrix V to
  project from the diagonal basis to observation space. Subsequent layers use
  identity (V is dropped for Dirac layers, absorbed into learnable B).

### PhasorNetworks

```
Input (C_in, L, B) Phase
  → PhasorDense: Dirac encode + W mix + causal conv + normalize  → (D, L, B) Phase
  → PhasorDense: same                                             → (D, L, B) Phase
  → SSMReadout: cos similarity to codebook prototypes, pool       → (n_classes, B)
```

Key design choices:
- **Fused Dense + SSM**: `PhasorDense` combines weight matrix and SSM dynamics in
  one layer. The Dirac encoding fuses input projection with temporal integration.
- **Phase activation**: `normalize_to_unit_circle` + `complex_to_angle`, outputting
  Phase values. No surrogate gradients needed — activation is smooth.
- **Codebook readout**: `SSMReadout` uses cosine similarity against fixed random
  prototypes, not a learned linear projection.
- **No skip connections** in the demo (could be added).
- **No leaky integrator**: readout pools by averaging similarity over a time window.

## 4. HiPPO Initialization

### S5-RF

Constructs the full N×N HiPPO-LegS matrix, diagonalizes it to get eigenvalues
Lambda and eigenvectors V, Vinv. Supports `num_blocks` to create block-diagonal
structure (repeat the same block multiple times):

```python
Lambda, V, Vinv = init_A(block_size=neurons/blocks, num_blocks=blocks)
```

For sMNIST: `block_size=16, num_blocks=8` → 128 neurons with 8 copies of a
16-dimensional HiPPO block.

The real parts of Lambda are set to `mean(diag(S))` — a single shared decay
rate, not the per-channel HiPPO decay. This differs from standard S4D where
each eigenvalue has its own real part.

### PhasorNetworks

Uses `hippo_legs_diagonal` which directly computes the diagonal eigenvalues
without constructing the full matrix:

```julia
λ_mag = exp.(range(log(0.5), log(N-0.5); length=N))
λ = -λ_mag
ω = π .* λ_mag
```

Log-spaced decay magnitudes from 0.5 to N-0.5, paired with proportional
frequencies. No block structure — each channel is independent.

## 5. Training Configuration

### S5-RF (sMNIST config)

```
neurons=128, blocks=8, layers=2
discretization=zoh (first layer), dirac (subsequent)
batchsize=64, epochs=100
lr=0.008 (weights), lr_ssm=0.002 (Lambda, log_step)
optimizer=AdamW (weights, weight_decay=0.0001), Adam (SSM params)
scheduler=cosine decay to 1e-6
dropout=0.15
eta_min=0.001, eta_max=0.1 (learnable step size range)
loss=softmax cross-entropy
activation=cartesian_spike (Heaviside on Re(x)-1, multi-Gaussian surrogate)
```

### PhasorNetworks (current demo)

```
hidden=32-64, layers=2
discretization=Dirac (all layers, phase input)
batchsize=64, epochs=5-20
lr=0.001 (all params)
optimizer=Adam (all params, no weight decay)
scheduler=none (fixed LR)
dropout=none
loss=softmax cross-entropy (scaled similarities)
activation=normalize_to_unit_circle → complex_to_angle
```

## 6. Key Differences That Affect Performance

### a. Learnable Delta (step size)

S5-RF's `log_step` parameter allows each channel to learn its own effective
timescale. This is the main mechanism for adapting to sequence length — no
explicit eigenvalue scaling needed. PhasorNetworks currently scales eigenvalues
manually in the demo (`scale_dynamics_for_length`).

**Recommendation**: Add a learnable `log_step` parameter to PhasorDense, similar
to S5-RF. This would replace the manual scaling and allow per-channel timescale
adaptation during training.

### b. Differential Learning Rates

S5-RF uses **different learning rates** for SSM parameters (Lambda, Delta) vs
connection weights (B, V). Typically SSM LR is 2-4x lower than weight LR.
PhasorNetworks uses a single LR for everything.

**Recommendation**: Use `Optimisers.OptimiserChain` with parameter-group-specific
LRs. Lower LR for `log_neg_lambda` and `omega`.

### c. Cosine LR Schedule

S5-RF uses cosine annealing from initial LR down to 1e-6 over the full training.
PhasorNetworks uses fixed LR.

**Recommendation**: Add cosine schedule. `Optimisers.CosineAnnealing` or manual
scaling in the training loop.

### d. Weight Decay

S5-RF applies weight decay (0.0001) to connection weights only — NOT to SSM
parameters. PhasorNetworks has no weight decay.

**Recommendation**: Add weight decay to the weight matrix but not to
`log_neg_lambda` or `omega`.

### e. Dropout

S5-RF applies dropout (0.15-0.3) between layers. PhasorNetworks has no dropout
in the demo.

### f. Parallel Scan vs FFT

S5-RF's prefix scan is O(P·L) with O(log L) depth. PhasorNetworks' FFT is
O(C·L·log L). For long sequences, the scan is faster. For batch parallelism,
FFT may be competitive since it's embarrassingly parallel across batch elements.

**Recommendation**: Implementing `associative_scan` in Julia is feasible but
non-trivial for Zygote AD. The FFT approach is adequate for current needs.

### g. 4D Tensor in Dirac Encoding

S5-RF's Dirac discretization is `B_bar = B * Delta` — a simple scaling of the
input matrix. No per-channel phase-dependent encoding. PhasorNetworks computes
`exp(k_c · dt)` per channel, requiring the expensive channel-by-channel map.

The reason: S5-RF doesn't encode phase information in spike timing — spikes are
binary events. PhasorNetworks treats each input phase as a specific spike
arrival time within the period, requiring the per-channel `exp(k_c · dt)`.

This is a **design choice with performance consequences**. If we used S5-RF's
simpler Dirac (no sub-period timing), we'd lose the phase-to-time encoding but
gain a much faster input projection (standard matrix multiply instead of
per-channel exp + reduce).

## 7. Summary: What We Can Adopt

| Feature | Status | Impact | Effort |
|---------|--------|--------|--------|
| Learnable Delta (step size) | Not implemented | HIGH | Medium |
| Differential LR (SSM vs weights) | Not implemented | HIGH | Low |
| Cosine LR schedule | Not implemented | MEDIUM | Low |
| Weight decay (weights only) | Not implemented | MEDIUM | Low |
| Dropout between layers | Not in demo | LOW-MED | Low |
| Skip connections | Not in demo | LOW-MED | Low |
| Parallel prefix scan | Using FFT instead | LOW (FFT is adequate) | High |
| Block-diagonal HiPPO | Not implemented | LOW | Medium |
