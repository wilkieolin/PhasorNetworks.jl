#!/usr/bin/env julia
#
# scripts/budget.jl  —  GPU memory budget analyzer for PhasorNetworks demos
#
# Models the dominant peak-memory contributors for a Phasor model running
# under Zygote AD on a unified-memory GPU (DGX Spark GB10, ~121 GiB
# unified). Use BEFORE launching a training run to verify the chosen
# (D, L, B) settings will fit, since unified memory means GPU OOM cascades
# into host paging / system lockup rather than a clean process kill.
#
# The model captures the four costs that dominate at the sfmnist+attention
# scale:
#   1. PhasorResonant encoder (Complex 3D → Phase 3D, FFT causal_conv)
#   2. PhasorDense Phase 3D path (causal_conv_dirac with grouped-channel
#      tape pinned by Zygote)
#   3. attend(): fused similarity_outer (post-2026 fusion in vsa.jl) plus
#      the score·V batched_mul
#   4. LastStepDense readout, Adam optimizer state, per-batch GPU buffers
#
# The numbers are upper-bound estimates assuming Zygote pins all forward
# tensors needed by the backward — which is the worst case observed in
# practice when the GC has not run between batches.
#
# ---------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------
#   include("scripts/budget.jl")
#   budget_sfmnist(D=64, B=32, use_attention=true)        # one config
#   recommend_safe_sfmnist(use_attention=true)             # grid search
#
#   julia --project=scripts scripts/budget.jl              # default report

# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

const BYTES_F32  = 4
const BYTES_CF32 = 8
const GIB        = 2^30
const MIB        = 2^20

# Unified-memory ceilings on the DGX Spark GB10.
# Total physical: ~121 GiB. CLAUDE.md sets ~110 GiB as the safe ceiling
# (anything above starts paging to disk and locks the box). We carve a
# more conservative budget for *new* allocations on top of whatever the
# REPL / driver / OS already holds.
const TOTAL_UNIFIED_GIB    = 121.0f0
const SAFE_PEAK_GIB        = 40.0f0   # comfortable: leaves ~80 GiB slack
const MARGINAL_PEAK_GIB    = 70.0f0   # tight: leaves ~50 GiB slack
# Above MARGINAL_PEAK_GIB → RISKY (don't launch without watchdog).

# Causal-conv-dirac default group_size from src/kernels.jl:260.
const CCD_GROUP_SIZE = 8

# Empirical multiplier for FFT-based causal_conv intermediates kept alive
# during a forward pass + Zygote tape: zero-padded H/K, K_f, H_f, Z_f,
# Z_full are each (C, 2L, B). At peak they sum to ≈ 5× (C, L, B) ComplexF32.
const FFT_CONV_MULT = 5

# Backward-pass cotangent multiplier. Zygote allocates a cotangent buffer
# roughly mirroring each forward tensor it needs to differentiate through;
# at peak (during the backward sweep) the Zygote-tape contributors are
# effectively doubled. We use 1.6 rather than a strict 2.0 because
# leaf inputs and a few forward intermediates don't get cotangent buffers,
# but it's much closer to 2 than to 1.
const BACKWARD_FACTOR = 1.6f0

# Allocator + gradient-buffer slack on top of the modelled tensor live set.
# Covers: CUDA allocator pool caching freed blocks, GC lag between batches
# (often two batch-steps worth of intermediates alive simultaneously),
# MLDatasets / Lux compile-time intermediates, and per-batch CPU→GPU
# copies that haven't been reclaimed yet.
const SLACK_FACTOR = 1.50f0

# ---------------------------------------------------------------------
# Per-contributor estimates
# ---------------------------------------------------------------------

"""Bytes for a 3-D ComplexF32 tensor of given shape."""
cf32_3d(c, l, b) = BYTES_CF32 * c * l * b

"""Bytes for a 3-D Float32 tensor of given shape."""
f32_3d(a, b, c)  = BYTES_F32  * a * b * c

"""
    phasor_resonant_peak(C_in, D, L, B) -> bytes

PhasorResonant runs `phasor_kernel` (small) then `causal_conv_fft`
(L > 64 always for sfmnist), and outputs `(D, L, B)` ComplexF32 that gets
converted to Phase via `complex_to_angle`.

Pinned by Zygote tape across the full backward pass:
- input (C_in, L, B) ComplexF32                               (saved)
- intermediate H pre-conv (D, L, B) ComplexF32                (saved)
- causal_conv_fft transient — `FFT_CONV_MULT × (D, L, B)`     (transient peak)
- output (D, L, B) ComplexF32                                 (saved)
"""
function phasor_resonant_peak(C_in::Int, D::Int, L::Int, B::Int)
    input  = cf32_3d(C_in, L, B)
    out    = cf32_3d(D,    L, B)
    fft    = FFT_CONV_MULT * cf32_3d(D, L, B)
    return input + out + fft
end

"""
    phasor_dense_phase3d_peak(C_in, C_out, L, B; G=CCD_GROUP_SIZE) -> bytes

PhasorDense Phase 3D forward routes through `causal_conv_dirac`, which
iterates `c_start in 1:G:C_out` and at each step builds an
`(g, C_in, L*B)` ComplexF32 `enc` tensor. Zygote's `map`-based tape
pins all groups (worst case observed under tight training loops).

Plus the FFT causal_conv on the assembled H, plus the H output itself
held live for the Phase output.
"""
function phasor_dense_phase3d_peak(C_in::Int, C_out::Int, L::Int, B::Int;
                                    G::Int = CCD_GROUP_SIZE)
    g = min(G, C_out)
    n_groups = cld(C_out, g)
    # If all per-group encs pinned: total = n_groups * 8 * g * C_in * L * B
    enc_tape   = n_groups * BYTES_CF32 * g * C_in * L * B
    H_out      = cf32_3d(C_out, L, B)
    fft        = FFT_CONV_MULT * cf32_3d(C_out, L, B)
    return enc_tape + H_out + fft
end

"""
    fused_similarity_outer_peak(D, L, B) -> bytes

Forward (post-fusion in vsa.jl): Ar/Ai/Br/Bi reals (4× (D,L,B) Float32),
two batched_mul outputs (2× (L,L,B) Float32), final sim (L,L,B) Float32.
Backward (closed-form rrule): ḡc lift to ComplexF32 ((L,L,B)) +
AB_term/BA_term ((D,L,B) Cplx each) + dA, dB ((D,L,B) Cplx each).
"""
function fused_similarity_outer_peak(D::Int, L::Int, B::Int)
    fwd_reals  = 4 * f32_3d(D, L, B)
    fwd_gemms  = 2 * f32_3d(L, L, B)
    fwd_out    =     f32_3d(L, L, B)
    bwd_lift   = cf32_3d(L, L, B)         # ḡc = ComplexF32.(ḡ): doubles bytes from f32 to cf32
    bwd_grads  = 4 * cf32_3d(D, L, B)     # AB_term, BA_term, dA, dB
    return fwd_reals + fwd_gemms + fwd_out + bwd_lift + bwd_grads
end

"""
    attend_peak(D, L, B) -> bytes

attend(Q, K, V) does:
- similarity_outer(Q, K) — fused
- score_scale (in-place-ish on (L, L, B), small)
- angle_to_complex(V) → (D, L, B) Cplx held alive
- batched_mul(V_complex, scores) → (D, L, B) Cplx
- complex_to_angle(output) → (D, L, B) Phase (Float32 underneath)

Plus Zygote tape on each.
"""
function attend_peak(D::Int, L::Int, B::Int)
    sim       = fused_similarity_outer_peak(D, L, B)
    v_complex = cf32_3d(D, L, B)
    sv_out    = cf32_3d(D, L, B)
    sv_phase  = f32_3d(D, L, B)
    return sim + v_complex + sv_out + sv_phase
end

"""
    last_step_dense_peak(D, n_out, B) -> bytes

Slice the last timestep of (D, L, B) Phase, run a (n_out × D) GEMM.
Negligible relative to the per-step buffers.
"""
function last_step_dense_peak(D::Int, n_out::Int, B::Int)
    last_step = BYTES_F32 * D * B
    out       = BYTES_F32 * n_out * B
    return last_step + out
end

"""
    adam_state_bytes(param_count) -> bytes

Adam holds two Float32 moments (m, v) per trainable parameter, plus the
parameter tensor itself (also held by the optimiser update path). We
count `2 × param_count × 4` for Adam state on top of the params.
"""
adam_state_bytes(n_params::Int) = 2 * BYTES_F32 * n_params

"""
    sfmnist_param_count(D; use_attention) -> Int

Trainable parameter count for the sfmnist model:
- PhasorResonant(1 → D):   weight (D, 1) + log_neg_lambda (D,)
- 3× PhasorDense(D → D):   weight (D, D) + log_neg_lambda (D,) + bias (D,) per layer
- LastStepDense(D, 10):    weight (10, D) + bias (10)
"""
function sfmnist_param_count(D::Int; use_attention::Bool)
    encoder = D * 1 + D                                # (weight + log_neg_lambda)
    qkv     = use_attention ? 3 * (D * D + D + D) : 0   # 3 PhasorDense layers
    readout = 10 * D + 10
    return encoder + qkv + readout
end

# ---------------------------------------------------------------------
# Top-level budget
# ---------------------------------------------------------------------

"""
    budget_sfmnist(; D, B, L=784, use_attention) -> NamedTuple

Print a per-contributor breakdown and return the modelled peak. The
verdict (SAFE / MARGINAL / RISKY) is keyed off `SAFE_PEAK_GIB` /
`MARGINAL_PEAK_GIB`.
"""
function budget_sfmnist(; D::Int, B::Int, L::Int = 784,
                          use_attention::Bool = false,
                          verbose::Bool = true)
    enc_fwd  = phasor_resonant_peak(1, D, L, B)
    qkv_fwd  = use_attention ? 3 * phasor_dense_phase3d_peak(D, D, L, B) : 0
    att_fwd  = use_attention ? attend_peak(D, L, B) : 0
    head     = last_step_dense_peak(D, 10, B)
    n_pp     = sfmnist_param_count(D; use_attention = use_attention)
    adam     = adam_state_bytes(n_pp) + BYTES_F32 * n_pp     # state + params
    data     = cf32_3d(1, L, B) + BYTES_F32 * 10 * B         # input + onehot

    # Apply backward-pass cotangent multiplier to the Zygote-tape
    # contributors (encoder, Q/K/V, attention). Adam, params, head, and
    # data don't double during backward — they're either leaves or have
    # closed-form rrules that don't allocate a same-size cotangent.
    enc = round(Int, enc_fwd * BACKWARD_FACTOR)
    qkv = round(Int, qkv_fwd * BACKWARD_FACTOR)
    att = round(Int, att_fwd * BACKWARD_FACTOR)

    raw    = enc + qkv + att + head + adam + data
    peak   = raw * SLACK_FACTOR
    pgib   = peak / GIB
    verdict = pgib < SAFE_PEAK_GIB    ? :SAFE :
              pgib < MARGINAL_PEAK_GIB ? :MARGINAL : :RISKY

    if verbose
        println()
        println("==== budget: sfmnist", use_attention ? "+attention" : "",
                "  D=$D  L=$L  B=$B ====")
        rows = [
            ("PhasorResonant encoder (fwd+bwd)",       enc),
            ("3× PhasorDense Q/K/V tape (fwd+bwd)",    qkv),
            ("attend(): fused similarity_outer + sv",  att),
            ("LastStepDense readout",                  head),
            ("Adam state + params ($(n_pp) p)",        adam),
            ("Per-batch input + labels",               data),
        ]
        for (name, bytes) in rows
            mib = bytes / MIB
            tag = mib > 1024 ? string(round(mib/1024; digits=2), " GiB") :
                                 string(round(mib;     digits=1), " MiB")
            println(rpad(name, 42), lpad(tag, 14))
        end
        println(rpad("-- modelled live set",          42), lpad(string(round(raw/GIB;  digits=3), " GiB"), 14))
        println(rpad("-- × $(SLACK_FACTOR) slack ⇒ peak", 42), lpad(string(round(peak/GIB; digits=3), " GiB"), 14))
        println(rpad("-- verdict",                    42), lpad(string(verdict), 14))
        println("   thresholds: SAFE < $(SAFE_PEAK_GIB) GiB, MARGINAL < $(MARGINAL_PEAK_GIB) GiB,",
                " unified total ≈ $(TOTAL_UNIFIED_GIB) GiB")
    end

    return (D = D, B = B, L = L, use_attention = use_attention,
            peak_gib = pgib, verdict = verdict, breakdown_bytes = raw)
end

# ---------------------------------------------------------------------
# Grid search for safe configurations
# ---------------------------------------------------------------------

"""
    recommend_safe_sfmnist(; use_attention, L=784,
                           D_grid, B_grid) -> Vector{NamedTuple}

Walk the (D, B) grid and report which configurations land in SAFE /
MARGINAL / RISKY. Print as a table; return the SAFE rows. Pick a config
from the SAFE rows with the largest D·B you're willing to wait for.
"""
function recommend_safe_sfmnist(; use_attention::Bool,
                                  L::Int = 784,
                                  D_grid::AbstractVector{Int} = [32, 48, 64, 96, 128],
                                  B_grid::AbstractVector{Int} = [8, 16, 32, 64])
    rows = NamedTuple[]
    for D in D_grid, B in B_grid
        push!(rows, budget_sfmnist(D = D, B = B, L = L,
                                    use_attention = use_attention,
                                    verbose = false))
    end

    println()
    println("==== sfmnist", use_attention ? "+attention" : "",
            " grid (L=$L) — peak GiB / verdict ====")
    print(rpad("D \\ B", 8))
    for B in B_grid
        print(lpad("B=$B", 14))
    end
    println()
    for D in D_grid
        print(rpad("D=$D", 8))
        for B in B_grid
            r = first(filter(x -> x.D == D && x.B == B, rows))
            cell = string(round(r.peak_gib; digits=2), "/", first(string(r.verdict)))
            print(lpad(cell, 14))
        end
        println()
    end
    println("  legend: S=SAFE (<$(SAFE_PEAK_GIB) GiB), M=MARGINAL (<$(MARGINAL_PEAK_GIB) GiB), R=RISKY (≥$(MARGINAL_PEAK_GIB) GiB)")

    safe = filter(r -> r.verdict === :SAFE, rows)
    if !isempty(safe)
        # Pick the largest D·B safe config as the headline recommendation.
        sort!(safe, by = r -> -(r.D * r.B))
        top = safe[1]
        println()
        println("  → largest SAFE: D=$(top.D), B=$(top.B), peak ≈ $(round(top.peak_gib; digits=2)) GiB")
    end
    return safe
end

if abspath(PROGRAM_FILE) == @__FILE__
    println("# Without attention (default sfmnist):")
    recommend_safe_sfmnist(use_attention = false)
    println()
    println("# With attention (the configuration that previously locked up):")
    recommend_safe_sfmnist(use_attention = true)
end
