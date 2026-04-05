# Training Bottleneck Analysis

## Forward Pass Profile (per batch)

For a model `PhasorDense(C_in→D) → PhasorDense(D→D) → SSMReadout(D→10)` with Phase input `(C_in, L, B)`:

### Layer 1: `_forward_3d_dirac` (PhasorDense, Phase→Phase)

1. **`Float32.(x)`** — Phase→Float32 conversion. O(C_in·L·B). Cheap.

2. **`dt = T·(0.5 - θ/2)`** — compute within-period time. O(C_in·L·B). Cheap.

3. **Diagonal encoding loop** (`map` over C_out channels):
   - `exp.(k_c[c:c] .* dt_flat)` — (C_in, L·B) exp per channel. **C_out × O(C_in·L·B) exp calls**.
   - `sum(w_row .* enc; dims=1)` — weighted sum. **C_out × O(C_in·L·B)**.
   - Total: **O(C_out · C_in · L · B)** element-wise exp + multiply. **This is the dominant cost.**

4. **`causal_conv(K, H)`** → dispatches to `causal_conv_fft` for L>64:
   - `zero(K)`, `zero(H)` — allocate zero pads. 2 allocations.
   - `cat(K, z_K)`, `cat(H, z_H)` — concatenate. 2 allocations + copies.
   - `fft(K_pad, 2)`, `fft(H_pad, 2)` — 2 FFTs. O(C_out·L·log(L)) + O(C_out·L·B·log(L)).
   - `reshape(K_f) .* H_f` — pointwise multiply. O(C_out·2L·B).
   - `ifft(Z_f, 2)` — 1 IFFT. O(C_out·2L·B·log(L)).
   - `Z_full[:, 1:L, :]` — slice. 1 allocation.
   - `ComplexF32.(Z)` — type cast. O(C_out·L·B). **Unnecessary if FFT already returns ComplexF32.**
   - Total: **O(C_out·L·log(L)·B)** compute, **6+ allocations**.

5. **`normalize_to_unit_circle(Z)`** — divide by magnitude. O(C_out·L·B). Moderate.

6. **`complex_to_angle(Y)`** — atan2 per element. O(C_out·L·B). Moderate.

### Layer 2: Same as Layer 1 but C_in = C_out = D.

### SSMReadout (Phase dispatch):

1. Slice readout window. Cheap.
2. `cos.(π .* (Float32.(p) .- Float32.(c)))` — **4D broadcast** (C, n_cls, W, B). For D=128, 10 classes, W=49 (readout_frac=0.25 of L=196), B=64: 128×10×49×64 = **4M elements**. Moderate.
3. Two `mean` reductions. Cheap.

## Identified Bottlenecks & Easy Wins

### 1. Sequential `map` over channels in Dirac encoding (HIGH IMPACT)

The `map(1:C_out)` loop processes channels one at a time. Each iteration launches separate GPU kernels for `exp`, `.*`, and `sum`. For C_out=128, that's ~384 kernel launches per layer per batch. GPU kernel launch overhead (~5-10μs each) adds up to ~2-4ms per layer.

**Fix**: Process channels in groups of G (e.g., G=8 or G=16). Each group materializes a (G, C_in, L·B) tensor — small enough to fit in memory but large enough for GPU efficiency.

```julia
G = min(8, C_out)
H_slices = map(1:G:C_out) do c0
    c1 = min(c0 + G - 1, C_out)
    k_grp = reshape(k_c[c0:c1], :, 1)                  # (G, 1)
    enc = exp.(k_grp .* reshape(dt_flat, 1, :))         # (G, C_in*L*B)
    # ... weight multiply and reshape
end
```

**Expected speedup**: 8-16x on the encoding step from reduced kernel launch overhead and better GPU occupancy.

### 2. Redundant allocations in `causal_conv_fft` (MEDIUM IMPACT)

Each call to `causal_conv_fft` allocates:
- `zero(K)` + `zero(H)` — two full-size zero arrays
- `cat(K, z_K)` + `cat(H, z_H)` — two concatenations (allocate + copy)
- `ComplexF32.(Z)` — full output copy for type cast

That's 6 allocations per FFT convolution, per layer, per batch. With two layers, 12 allocations per batch.

**Fix 1**: Pre-allocate padded buffers and copy in-place (requires Zygote workaround — use `ignore_derivatives` for the padding since zeros don't carry gradient).

**Fix 2**: The `ComplexF32.(Z)` cast at the end is likely unnecessary — FFTW on ComplexF32 input returns ComplexF32 output. Remove the cast and let it be a no-op:

```julia
# Instead of:
return ComplexF32.(Z)
# Just:
return Z
```

**Fix 3**: Use power-of-2 padding instead of 2L. FFT is fastest for power-of-2 sizes. `nextpow(2, 2L)` instead of `2L`.

### 3. `intensity_to_phase` called inside loss function (LOW-MEDIUM IMPACT)

`mnist_loss` calls `intensity_to_phase(x)` on every batch, every gradient step. This includes `permutedims` (memory copy) and `Phase.()` wrapping. Since the encoding is deterministic, it could be precomputed once in the DataLoader.

**Fix**: Encode the dataset before training:
```julia
x_train_phases = intensity_to_phase(reshape(x_train, L, C, :))
train_loader = DataLoader((x_train_phases, y_train); batchsize, shuffle=true)
```
Then `mnist_loss` becomes just `model(x, ps, st)` — no encoding step.

### 4. `GC.gc(false)` + `CUDA.reclaim()` every batch (LOW-MEDIUM IMPACT)

The training loop calls `GC.gc(false)` and `CUDA.reclaim()` after every single batch. This was added for ODE-heavy spiking workloads that accumulate large `sol.u` arrays. For the SSM convolutional path, this is unnecessary overhead.

**Fix**: Only call GC periodically (every N batches) or remove it for SSM training:
```julia
if step_count % 50 == 0
    GC.gc(false)
    CUDA.functional() && CUDA.reclaim()
end
```

### 5. `normalize_to_unit_circle` + `complex_to_angle` roundtrip (LOW IMPACT)

`_forward_3d_dirac` computes `complex_to_angle(normalize_to_unit_circle(Z))`. This does:
1. `abs.(Z)` — magnitudes
2. `Z ./ abs.(Z)` — normalize (with threshold check via `ifelse`)
3. `angle.(Z_norm)` — atan2
4. `Phase.(... / π)` — wrap

Steps 1-2 are unnecessary if we're immediately extracting the angle — `angle(z)` doesn't depend on magnitude. Could replace with a direct `complex_to_angle(Z)` that skips normalization:

```julia
# Instead of:
Y = a.activation(Z)           # normalize_to_unit_circle
return complex_to_angle(Y)
# Could do:
return complex_to_angle(Z)    # angle is magnitude-independent
```

However, the activation is configurable (users can set it to `identity` or `soft_normalize`), so this optimization only applies when activation is specifically `normalize_to_unit_circle` followed by `complex_to_angle`.

### 6. `reduce(vcat, H_slices)` after channel map (LOW IMPACT)

The `map` produces a Vector of (1, L, B) arrays, then `reduce(vcat, ...)` concatenates them. This allocates C_out intermediate arrays plus the final concatenation. Could use `stack` instead if available, or pre-allocate.

## Priority Summary

| # | Bottleneck | Impact | Effort | Recommendation |
|---|-----------|--------|--------|----------------|
| 1 | Sequential channel map | HIGH | Medium | Group channels (G=8-16) |
| 2 | FFT allocations | MEDIUM | Low | Remove `ComplexF32.()` cast, use pow2 padding |
| 3 | Repeated encoding | LOW-MED | Low | Precompute phases in DataLoader |
| 4 | Per-batch GC | LOW-MED | Trivial | Reduce GC frequency or skip for SSM |
| 5 | Unnecessary normalization | LOW | Low | Direct angle extraction |
| 6 | vcat after map | LOW | Low | Use `stack` or pre-allocate |
