# Brainstorm: Reducing the Computational Cost of Dirac Encoding

## The Bottleneck

The exact factored Dirac discretization requires a 4D tensor for Term 1:

```
p_c[c, j, n, b] = exp(k_c[c] · dt[j, n, b])
```

where `(c, j, n, b)` = (C_out, C_in, L, B). For a model with C_out=128, C_in=4, L=196, B=64, this is **6.4M complex entries** (~51MB), computed fresh for every forward pass and stored by Zygote for backprop.

Term 2 is cheap because `exp(k₀ · dt)` is channel-independent: just `(C_in, L, B)`.

The 4D tensor arises because each output channel c has a different eigenvalue k_c, so the phase-to-time encoding `exp(k_c · dt)` varies per channel. This is fundamental to the Dirac discretization — it's what makes it exact.

## Approach 1: Truncated Kernel Windows

**Idea**: Each channel's causal convolution kernel `K_c[n] = exp(k_c · n · T)` decays exponentially. Past cutoff lag `n_max(c) = log(ε) / (λ_c · T)`, contributions are negligible. Truncate the convolution to only process `n_max` lags per channel.

**Impact on 4D tensor**: None directly — the 4D tensor is the encoding, not the kernel. However, if we switch from FFT to direct windowed convolution for short-kernel channels, we compute `H1[c, n]` only for positions within the window of each output sample, reducing the effective work.

**Savings**: For fast-decaying channels (large |λ_c|), the window could be 10-20 steps instead of L=196+. With HiPPO, most channels are fast-decaying. Potential 5-10x speedup on the convolution step, but the 4D encoding itself is still computed for all positions.

**Complexity**: Requires per-channel variable-length convolution — harder to batch on GPU. Could group channels by similar window sizes.

## Approach 2: Taylor Expansion of exp(k_c · dt)

**Idea**: Approximate `exp(k_c · dt)` with a low-order polynomial:

```
exp(k_c · dt) ≈ 1 + k_c·dt + (k_c·dt)²/2 + ...
```

Each term is separable: `(k_c)^m · (dt)^m / m!`. The powers of `dt` are channel-independent (just `(C_in, L, B)`) and the powers of `k_c` are input-independent (just `(C_out,)`). So the weighted sum becomes:

```
H1_c[n] = Σ_j W[c,j] · Σ_m (k_c)^m / m! · (dt_j[n])^m
         = Σ_m (k_c)^m / m! · (Σ_j W[c,j] · dt_j[n]^m)
         = Σ_m (k_c)^m / m! · (W · dt^m)[c, n]
```

Each term in the sum is a **standard weight multiply** `W · dt^m` — no 4D tensor needed! The cost per term is O(C_out · C_in · L · B) as a matrix multiply, and we need M terms for accuracy.

**Savings**: Replaces one 4D tensor + reduction with M matrix multiplies. For M=3-4 terms, this could be 10-30x faster for large C_out, and uses O(M · C_in · L · B) memory instead of O(C_out · C_in · L · B).

**Accuracy**: For |k_c · dt| < 1, a 3rd-order Taylor gives <1% error. For our system, `dt ∈ [0, T]` and `|k_c|` is typically 0.01-0.5 (after scaling), so `|k_c · dt| ∈ [0, 0.5]` — well within the Taylor convergence radius. Could use more terms for channels with larger |k_c|.

**Risk**: The approximation error grows with |k_c| and dt. Fast-decaying channels (large |λ_c|) may need more terms. Could adaptively choose M per channel.

## Approach 3: Shared Encoding with Per-Channel Correction

**Idea**: Factor `exp(k_c · dt)` around a reference eigenvalue k_ref (e.g., the mean of k_c):

```
exp(k_c · dt) = exp(k_ref · dt) · exp((k_c - k_ref) · dt)
```

The first factor `exp(k_ref · dt)` is channel-independent → `(C_in, L, B)`, same as Term 2. The correction `exp(δk_c · dt)` has smaller argument `|δk_c · dt|` and can be Taylor-expanded with fewer terms:

```
exp(δk_c · dt) ≈ 1 + δk_c·dt + (δk_c·dt)²/2
```

**Savings**: One shared encoding (matrix multiply) + M small correction terms (also matrix multiplies). If k_c values are clustered, δk_c is small and M=1-2 terms suffice.

**Accuracy**: Depends on the spread of eigenvalues. For uniform init where all k_c are similar, this is very accurate. For HiPPO where eigenvalues span orders of magnitude, may need multiple reference points (clustering).

## Approach 4: Channel Grouping / Clustering

**Idea**: Group output channels with similar eigenvalues k_c into clusters. Within each cluster, use a shared reference encoding (Approach 3) with per-channel corrections. Channels with identical k_c (e.g., uniform init where all λ_c are the same) collapse to a single encoding.

**Implementation**: 
1. Cluster k_c values into G groups (e.g., G=4-8)
2. For each group g with reference k_g: compute shared encoding `exp(k_g · dt)` → `(C_in, L, B)`
3. Per-channel correction within group: `exp((k_c - k_g) · dt)` via 1-2 Taylor terms
4. Weight multiply per group

**Savings**: O(G · C_in · L · B) for shared encodings + O(G · M · C_in · L · B) for corrections, instead of O(C_out · C_in · L · B). If G << C_out, significant savings.

**Note**: For uniform init (all k_c identical), G=1 and this reduces to Term 2 — the 4D tensor disappears entirely!

## Approach 5: Diagonal Reformulation (Eliminate 4D Tensor)

**Idea**: Restructure so that the per-channel encoding doesn't require an explicit outer product. Instead of computing `H1[c,n] = Σ_j W[c,j] · exp(k_c · dt_j[n])`, reformulate as a sequence of operations that avoid materializing the 4D tensor.

For each output channel c independently:
```
H1[c, n] = W[c, :] · diag(exp(k_c · dt[:, n, b])) · ones(C_in)
         = Σ_j W[c,j] · exp(k_c · dt[j,n,b])
```

This is a **weighted sum** where the weights `W[c,j]` are fixed and the values `exp(k_c · dt[j,n,b])` vary. We can compute this channel-by-channel without the full 4D tensor:

```julia
H1 = zeros(ComplexF32, C_out, L, B)
for c in 1:C_out
    enc_c = exp.(k_c[c] .* dt)           # (C_in, L, B) — reused buffer
    H1[c, :, :] = sum(W[c, :] .* enc_c; dims=1)  # dot product per (n, b)
end
```

**Savings**: Peak memory is O(C_in · L · B) instead of O(C_out · C_in · L · B). Same total FLOPs but sequential over channels — worse GPU utilization.

**GPU variant**: Process channels in small groups (e.g., 8 at a time) to balance memory and parallelism. Or use `@tullio` / custom kernel for fused element-wise-multiply-reduce.

## Approach 6: Precomputed Kernel Table (Quantized Phases)

**Idea**: Phase values θ ∈ [-1, 1] map to dt ∈ [0, T]. If we quantize dt into Q discrete levels, we can precompute `exp(k_c · dt_q)` for each (c, q) pair as a lookup table of size `(C_out, Q)`. Then the encoding becomes a table lookup + weight multiply.

```
dt_quantized[j,n,b] = round(dt[j,n,b] / T * Q) 
enc[c,j,n,b] = table[c, dt_quantized[j,n,b]]
```

**Savings**: Table is O(C_out · Q), lookup is O(C_in · L · B) indices. No exp() computation in the forward pass.

**Accuracy**: With Q=256, the quantization error on dt is T/256 ≈ 0.004 for T=1. The error on exp(k_c · dt) is bounded by |k_c| · T/Q, which for typical |k_c| < 1 is sub-percent.

**Risk**: Table lookups with integer indices may not be Zygote-compatible (non-differentiable). Would need to use soft lookup (interpolation) or straight-through estimator for gradients. Also, on GPU, gather operations can be slow.

## Approach 7: Low-Rank Approximation

**Idea**: The 4D tensor `p_c[c,j,n,b] = exp(k_c · dt[j,n,b])` can be viewed as a matrix `M[c, (j,n,b)] = exp(k_c · dt)`. This matrix has rank structure because it's an outer product through an exponential. Approximate it with a rank-R decomposition:

```
M ≈ U · V'    where U is (C_out, R) and V is (C_in·L·B, R)
```

For R << C_out, this avoids materializing the full matrix.

**Challenge**: The standard SVD doesn't exploit the exponential structure. However, the exponential of an outer product `exp(a · b')` can be approximated by `Σ_r (a^r/r!) ⊗ (b^r)` — which is exactly the Taylor expansion (Approach 2). So this reduces to Approach 2 with R = M (Taylor order).

## Recommendation

**Short-term (easiest win)**: **Approach 2 (Taylor expansion)** with M=3-4 terms. Replaces the 4D tensor with M matrix multiplies, reducing memory by C_out/M ≈ 16-32x. Implementation is straightforward, Zygote-compatible, and the accuracy is controllable.

**Medium-term (best for production)**: **Approach 4 (channel grouping)** combined with Approach 3 (shared encoding + correction). For uniform init this is free (G=1); for HiPPO with diverse eigenvalues, G=4-8 groups would capture the spread.

**Long-term (maximum performance)**: **Approach 5 (diagonal reformulation)** with a custom fused GPU kernel that computes the per-channel exp + weight reduction without materializing the 4D tensor. This is the optimal solution but requires CUDA kernel development.

## Cost Comparison Summary

| Approach | Memory | FLOPs | Accuracy | Complexity |
|----------|--------|-------|----------|------------|
| Current (4D tensor) | O(C_out·C_in·L·B) | O(C_out·C_in·L·B) | Exact | Simple |
| 1. Truncated windows | O(C_out·C_in·L·B) | Reduced for fast channels | Exact (within ε) | Medium |
| 2. Taylor expansion | O(M·C_in·L·B) | O(M·C_out·C_in·L·B) matmul | ~1% for M=3 | Simple |
| 3. Shared + correction | O((1+M)·C_in·L·B) | O((1+M)·C_out·C_in) matmul | Depends on spread | Simple |
| 4. Channel grouping | O(G·C_in·L·B) | O(G·C_out·C_in) matmul | Good with G=4-8 | Medium |
| 5. Diagonal (loop) | O(C_in·L·B) | O(C_out·C_in·L·B) | Exact | Medium (GPU hard) |
| 6. Quantized table | O(C_out·Q) | O(C_in·L·B) lookup | ~1% for Q=256 | Hard (AD) |
| 7. Low-rank / Taylor | O(R·(C_out+C_in·L·B)) | O(R·C_out·C_in) | Same as Taylor | Same as 2 |
