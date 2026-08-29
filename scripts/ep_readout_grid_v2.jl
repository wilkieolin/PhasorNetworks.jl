#!/usr/bin/env julia
#
# scripts/ep_readout_grid_v2.jl — readout frame × sampling rate grid for LockinEP
#
# Measures LockinEP gradient fidelity against centered StaticEP across:
#   - readout_frame ∈ {:co_rotating, :lab}
#   - sample_every ∈ {1, period_steps}  (every step vs one sample per probe period)
#   - carrier ∈ {nothing, 2π, 1.7, 2.9} (matching ep_readout_frame_check.jl)
#   - draw (multiple random seeds)
#
# Outputs CSV with columns for aggregation.
#
# Usage:
#   julia --project=. scripts/ep_readout_grid_v2.jl
#   julia --project=. -t 8 scripts/ep_readout_grid_v2.jl  (parallel draws)

using PhasorNetworks, Lux, LinearAlgebra, Printf, Statistics, CSV, DataFrames
using Random: Xoshiro
using Base.Threads: @threads, nthreads

const OUTDIR = joinpath(@__DIR__, "..", "results", "ep_readout_grid_v2")
isdir(OUTDIR) || mkpath(OUTDIR)
const OUTFILE = joinpath(OUTDIR, "grid.csv")

const N_DRAWS = 8
const DELTA = 0.005f0
const TOY_SEED_BASE = 42

function toy_chain(seed::Int = 42; scale = 0.4f0)
    ch = Chain(
        PhasorDense(4 => 8, normalize_to_unit_circle, use_bias=true),
        PhasorDense(8 => 2, normalize_to_unit_circle, use_bias=true)
    )
    ps, st = Lux.setup(Xoshiro(seed), ch)
    ps = (layer_1 = merge(ps.layer_1, (weight = scale .* ps.layer_1.weight,)),
          layer_2 = merge(ps.layer_2, (weight = scale .* ps.layer_2.weight,)))
    return ch, ps, st
end

function toy_input(seed::Int, B::Int = 3)
    rng = Xoshiro(seed)
    v = 2f0 .* rand(rng, Float32, 4, B) .- 1f0
    return Phase.(B == 1 ? vec(v) : v)
end

function toy_cost(seed::Int, B::Int = 3)
    rng = Xoshiro(seed)
    y = ComplexF32.(exp.(im .* π .* (2f0 .* rand(rng, Float32, 2, B) .- 1f0)))
    return SimilarityCost(B == 1 ? vec(y) : y)
end

cosine(a, b) = real(dot(vec(a), vec(b)) / (norm(vec(a)) * norm(vec(b)) + 1e-30))

# Reference: centered StaticEP at small β
ref_method = StaticEP(β=0.005f0, T_free=200, T_nudge=100, dt=0.5f0, centered=true)

# Lock-in parameters matching ep_readout_frame_check.jl
const LOCKIN_KW = (ε=0.02f0, ω_p=0.02f0, n_cycles=4, T_warmup_cycles=2, T_free=200, dt=0.5f0)

# Test configurations matching ep_readout_frame_check.jl arms
# Each config: (carrier_value, carrier_label, frame)
# carrier=nothing with frame=co_rotating = pure co-rotating (original LockinEP)
# carrier=ω with frame=lab = lab frame quantize (physical readout clock in lab)
# carrier=ω with frame=co_rotating = lab settle, co-rotating readout (carrier demodulated before quantize)
const CONFIGS = [
    (nothing,       "co_rotating", :co_rotating),  # original LockinEP
    (Float32(2π),   "lab_2pi",     :lab),          # commensurate lab
    (1.7f0,        "lab_1p7",     :lab),          # incommensurate lab
    (2.9f0,        "lab_2p9",     :lab),          # another incommensurate lab
    # Frame comparison at incommensurate carrier:
    (1.7f0,        "lab_1p7",     :co_rotating),  # lab settle, co-rotating readout
]

function run_one_draw(draw::Int)
    seed = TOY_SEED_BASE + draw
    ch, ps, st = toy_chain(seed)
    x = toy_input(seed + 100)
    y = toy_cost(seed + 200)
    cost = SimilarityCost(y.y)

    # Reference gradient
    gref, _ = ep_gradient(ref_method, ch, ps, st, x, cost)

    results = []
    period_steps = round(Int, 2π / (LOCKIN_KW.ω_p * LOCKIN_KW.dt))

    for (carrier, carrier_label, frame) in CONFIGS
        for se in (1, period_steps)
            m = LockinEP(; LOCKIN_KW..., carrier=carrier, readout_frame=frame, sample_every=se,
                           readout_δ=DELTA, readout_jitter=0f0)
            g, _ = ep_gradient(m, ch, ps, st, x, cost)

            cos_l1 = cosine(g.layer_1.weight, gref.layer_1.weight)
            cos_l2 = cosine(g.layer_2.weight, gref.layer_2.weight)
            re_l1 = norm(g.layer_1.weight .- gref.layer_1.weight) / norm(gref.layer_1.weight)

            push!(results, (
                draw = draw,
                carrier = carrier_label,
                frame = string(frame),
                sample_every = se,
                cos_l1 = cos_l1,
                cos_l2 = cos_l2,
                relerr_l1 = re_l1
            ))
        end
    end
    return results
end

function main()
    println("LockinEP readout frame × sampling grid")
    println("=====================================")
    println("Draws: $N_DRAWS")
    println("Δ (quantum): $DELTA turns")
    println("Lock-in params: ε=$(LOCKIN_KW.ε), ω_p=$(LOCKIN_KW.ω_p), n_cycles=$(LOCKIN_KW.n_cycles)")
    println("Output: $OUTFILE")
    println()

    # Check if file exists and has header
    has_header = isfile(OUTFILE) && filesize(OUTFILE) > 0

    all_results = []
    if nthreads() > 1
        println("Running on $(nthreads()) threads...")
        ch = Channel{Vector{NamedTuple}}(nthreads())
        @threads :dynamic for d in 1:N_DRAWS
            put!(ch, run_one_draw(d))
        end
        close(ch)
        for r in ch
            append!(all_results, r)
        end
    else
        println("Running single-threaded...")
        for d in 1:N_DRAWS
            append!(all_results, run_one_draw(d))
        end
    end

    # Write CSV
    df = DataFrame(all_results)
    CSV.write(OUTFILE, df; append=has_header)
    println("\nWrote $(length(all_results)) rows to $OUTFILE")

    # Summary
    println("\n--- Summary (median cos_l1 / cos_l2) ---")
    for carrier in unique(df.carrier), frame in unique(df.frame), se in unique(df.sample_every)
        subset = filter(r -> r.carrier == carrier && r.frame == frame && r.sample_every == se, all_results)
        isempty(subset) && continue
        med_l1 = median([r.cos_l1 for r in subset])
        med_l2 = median([r.cos_l2 for r in subset])
        min_l1 = minimum([r.cos_l1 for r in subset])
        min_l2 = minimum([r.cos_l2 for r in subset])
        @printf("  %-15s  %-15s  se=%-4d  med=%.4f/%.4f  min=%.4f/%.4f  n=%d\n",
                carrier, frame, se, med_l1, med_l2, min_l1, min_l2, length(subset))
    end

    # Key comparison: subsampling effect (co-rotating frame, carrier=nothing)
    println("\n--- Subsampling effect (co-rotating frame, carrier=nothing) ---")
    period_steps = round(Int, 2π / (LOCKIN_KW.ω_p * LOCKIN_KW.dt))
    for se in (1, period_steps)
        subset = filter(r -> r.carrier == "co_rotating" && r.frame == "co_rotating" && r.sample_every == se, all_results)
        med_l1 = median([r.cos_l1 for r in subset])
        med_l2 = median([r.cos_l2 for r in subset])
        @printf("  sample_every=%d: median cos_l1=%.4f, cos_l2=%.4f\n", se, med_l1, med_l2)
    end

    # Key comparison: frame effect at incommensurate carrier (lab_1p7)
    println("\n--- Frame effect at incommensurate carrier (lab_1p7) ---")
    for frame in ("co_rotating", "lab")
        subset = filter(r -> r.carrier == "lab_1p7" && r.frame == frame && r.sample_every == 1, all_results)
        isempty(subset) && continue
        med_l1 = median([r.cos_l1 for r in subset])
        med_l2 = median([r.cos_l2 for r in subset])
        @printf("  frame=%s: median cos_l1=%.4f, cos_l2=%.4f\n", frame, med_l1, med_l2)
    end

    # Compare with ep_readout_frame_check.jl expected results
    println("\n--- Comparison with ep_readout_frame_check.jl (δ=0.005, jitter=0) ---")
    println("Expected from original script:")
    println("  co_rotating (carrier=nothing):     cos_l1≈0.436, cos_l2≈0.825")
    println("  lab ω=2π (commensurate):           cos_l1≈0.452, cos_l2≈0.819")
    println("  lab ω=1.7 (incommensurate):        cos_l1≈0.998, cos_l2≈0.998")
    println("  lab ω=2.9 (incommensurate):        cos_l1≈0.998, cos_l2≈0.999")
    println("  lab ω=1.7, frame=co_rotating:      cos_l1≈0.436, cos_l2≈0.825 (matches co_rotating)")
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main()