#!/usr/bin/env julia
#
# scripts/ep_adiabatic_sweep.jl — map the adiabatic operating zone for LockinEP
#
# LockinEP's three knobs (ε, ω_p, n_cycles) trade gradient fidelity against
# cost, and the usable region moves with network width: the package defaults
# were tuned on a 4→8→2 chain and are NOT adiabatic at 784→256→64 (see
# results/ep_fashionmnist/FINDINGS.md §2). This maps the zone properly.
#
# The deliverable is a BOUNDARY — the cheapest configuration that still holds
# fidelity — not a single argmax. Cost is analytic:
#
#     steps(p) = T_free + (T_warmup_cycles + n_cycles) · round(2π / (ω_p·dt))
#
# so the real objective is: minimize steps subject to cos ≥ threshold.
#
# ---------------------------------------------------------------------
# Design notes that shaped this harness (measured, see the design doc)
# ---------------------------------------------------------------------
# * Dispatch is CPU threads, not GPU. A sweep evaluation is small and the
#   GPU settle is launch-latency-bound: GPU eval rate is FLAT from B=16 to
#   B=512 (~5/s) and concurrent streams add only 1.36x. CPU 10-way with
#   BLAS=1 does 14.6 evals/s. Set BLAS threads to 1 — the parallelism
#   belongs across configurations, not inside one gemm.
# * Estimator variance does NOT fall with batch size (measured flat-to-worse
#   from B=16 to B=1024), so replication over seeds is what buys precision,
#   not a bigger batch. Keep B small and spend the budget on replicates.
# * The reference gradient depends only on (weights, input, labels) — NOT on
#   any lock-in knob. It is computed once per replicate and reused across
#   every configuration, which is the single biggest saving in the sweep.
#
# ---------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------
#   julia --project=. -t 10 scripts/ep_adiabatic_sweep.jl variance
#   julia --project=. -t 10 scripts/ep_adiabatic_sweep.jl grid
#   julia --project=. -t 10 scripts/ep_adiabatic_sweep.jl report
#
# Both data-producing stages append to CSV as each point finishes and skip
# work already on disk, so they are killable and resumable.
#
# Env overrides: EPS_HID, EPS_DOUT, EPS_B, EPS_REPS, EPS_DT, EPS_SCALE,
#                EPS_TFREE, EPS_WARMUP, EPS_THRESH, EPS_OUT

using PhasorNetworks, Lux, LinearAlgebra, Printf, Statistics
using Random: Xoshiro

const OUT = get(ENV, "EPS_OUT",
                joinpath(@__DIR__, "..", "results", "ep_adiabatic"))

_envi(k, d) = parse(Int,     get(ENV, k, string(d)))
_envf(k, d) = parse(Float32, get(ENV, k, string(d)))

const HID     = _envi("EPS_HID",    256)
const DOUT    = _envi("EPS_DOUT",   64)
const NCLS    = 10
const B       = _envi("EPS_B",      32)     # small on purpose; see design notes
const REPS    = _envi("EPS_REPS",   8)
const DT      = _envf("EPS_DT",     0.5)    # fixed; see "why dt is not swept"
const SCALE   = _envf("EPS_SCALE",  0.4)
const T_FREE  = _envi("EPS_TFREE",  200)
const WARMUP  = _envi("EPS_WARMUP", 1)
const THRESH  = parse(Float64, get(ENV, "EPS_THRESH", "0.99"))

# ---------------------------------------------------------------------
# Resumable CSV (same pattern as scripts/temporal_scaling_sweep.jl)
# ---------------------------------------------------------------------
# Appending a row whose schema differs from the existing header writes
# headerless extra columns that later parse as garbage — silently, and in a
# way that looks like a real measurement. (This bit us once: a `ref_selfcos`
# column added in a second run read back as 0.0 for every row and nearly
# produced the opposite conclusion.) Validate instead.
function append_row(file::String, row::NamedTuple)
    isdir(dirname(file)) || mkpath(dirname(file))
    hdr = join(String.(keys(row)), ",")
    if isfile(file) && filesize(file) > 0
        existing = open(readline, file)
        if existing != hdr
            error("""
            CSV schema mismatch in $file
              on disk: $existing
              new row: $hdr
            Appending would misalign columns. Move or delete the old file,
            or write the new probe to its own file.""")
        end
    end
    open(file, "a") do io
        (isfile(file) && filesize(file) > 0) || println(io, hdr)
        println(io, join((string(getproperty(row, k)) for k in keys(row)), ","))
    end
end

function read_rows(file::String)
    isfile(file) || return Dict{String,String}[]
    lines = readlines(file)
    length(lines) <= 1 && return Dict{String,String}[]
    hdr = String.(split(lines[1], ','))
    [Dict(zip(hdr, String.(split(ln, ',')))) for ln in lines[2:end] if !isempty(strip(ln))]
end

function done_keys(file::String, keyfields::Tuple)
    done = Set{Tuple}()
    for r in read_rows(file)
        all(haskey(r, String(k)) for k in keyfields) || continue
        push!(done, Tuple(r[String(k)] for k in keyfields))
    end
    return done
end

_key(vals...) = Tuple(string(v) for v in vals)

# ---------------------------------------------------------------------
# Model + replicates
# ---------------------------------------------------------------------
function build_chain(seed::Int)
    chain = Chain(
        PhasorDense(784 => HID,  normalize_to_unit_circle, use_bias=true),
        PhasorDense(HID  => DOUT, normalize_to_unit_circle, use_bias=true),
    )
    ps, st = Lux.setup(Xoshiro(seed), chain)
    ps = (layer_1 = merge(ps.layer_1, (weight = SCALE .* ps.layer_1.weight,)),
          layer_2 = merge(ps.layer_2, (weight = SCALE .* ps.layer_2.weight,)))
    return chain, ps, st
end

make_codes(seed::Int) =
    ComplexF32.(angle_to_complex(orthogonal_codes(Xoshiro(seed), DOUT, NCLS)))

"Sparse image-like input matching FashionMNIST statistics (~80% zeros), mapped
through the half-plane encoding used by demos/ep_fashionmnist.jl."
function make_input(seed::Int, b::Int = B)
    rng = Xoshiro(seed)
    pix = rand(rng, Float32, 784, b)
    pix[rand(rng, Float32, 784, b) .< 0.8f0] .= 0f0
    μ = mean(pix; dims=1); σ = std(pix; dims=1) .+ 1f-6
    return Phase.(0.5f0 .* tanh.((pix .- μ) ./ σ))
end

make_cost(codes, seed::Int, b::Int = B) =
    CodebookCost(codes, rand(Xoshiro(seed + 5_000), 1:NCLS, b))

# ---------------------------------------------------------------------
# Cost model (analytic — no need to measure it)
# ---------------------------------------------------------------------
period_steps(ω_p, dt) = round(Int, 2π / (ω_p * dt))

lockin_steps(; ω_p, dt, n_cycles, T_warmup_cycles = WARMUP, T_free = T_FREE) =
    T_free + (T_warmup_cycles + n_cycles) * period_steps(ω_p, dt)

# ---------------------------------------------------------------------
# Fidelity of one lock-in configuration against a precomputed reference
# ---------------------------------------------------------------------
"Centered StaticEP at small β: the O(β) bias cancels, which matters because
`fd_gradient_phasor` needs n_params+1 settles and is unaffordable here."
reference_method(; β = 0.005f0, T_free = 400, T_nudge = 200) =
    StaticEP(β = β, T_free = T_free, T_nudge = T_nudge, dt = DT, centered = true)

function fidelity(g, ref)
    out = NamedTuple[]
    for k in keys(ref)
        haskey(ref[k], :weight) || continue
        a = vec(Array(g[k].weight)); b = vec(Array(ref[k].weight))
        push!(out, (layer = k,
                    cos = dot(a, b) / (norm(a) * norm(b) + 1e-12),
                    relerr = norm(a - b) / (norm(b) + 1e-12)))
    end
    return out
end

"Worst-layer cosine — the zone boundary should be set by the weakest layer,
not an average that hides a dead one."
worst_cos(f) = minimum(x -> x.cos, f)

# ---------------------------------------------------------------------
# Stage 0 — where does the estimator variance come from?
# ---------------------------------------------------------------------
#
# Measured: variance does not fall with batch size. Before mapping contours we
# need to know what DOES reduce it, otherwise the grid maps our own noise.
# Three candidates, each isolated here:
#   (1) demodulator selectivity  — scan n_cycles
#   (2) an unstable reference    — compare two references to each other
#   (3) draw-to-draw variation   — vary input seed vs weight seed separately
function stage_variance()
    file = joinpath(OUT, "variance.csv")
    done = done_keys(file, (:probe, :setting, :rep))
    codes = make_codes(8)
    @info "Stage 0: variance sources" file reps=REPS B

    # (1) n_cycles scan, weights fixed, input varying
    chain, ps, st = build_chain(7)
    for nc in (1, 2, 4, 8, 16), r in 1:REPS
        _key("n_cycles", nc, r) in done && continue
        x = make_input(100 + r); cost = make_cost(codes, 100 + r)
        ref, _ = ep_gradient(reference_method(), chain, ps, st, x, cost)
        g, _ = ep_gradient(LockinEP(ε=0.03f0, ω_p=0.02f0, n_cycles=nc,
                                    T_warmup_cycles=WARMUP, T_free=T_FREE, dt=DT),
                           chain, ps, st, x, cost)
        append_row(file, (probe="n_cycles", setting=nc, rep=r,
                          cos=worst_cos(fidelity(g, ref)),
                          steps=lockin_steps(ω_p=0.02f0, dt=DT, n_cycles=nc)))
    end

    # (2) reference self-consistency: two oracles, same input. If these
    #     disagree by as much as lock-in disagrees with them, the oracle is
    #     the noise source and no amount of replication fixes the map.
    for r in 1:REPS
        _key("ref_vs_ref", "beta", r) in done && continue
        x = make_input(100 + r); cost = make_cost(codes, 100 + r)
        r1, _ = ep_gradient(reference_method(β=0.005f0), chain, ps, st, x, cost)
        r2, _ = ep_gradient(reference_method(β=0.002f0, T_free=800, T_nudge=400),
                            chain, ps, st, x, cost)
        append_row(file, (probe="ref_vs_ref", setting="beta", rep=r,
                          cos=worst_cos(fidelity(r1, r2)), steps=0))
    end

    # (3) variance decomposition. Same lock-in config throughout; vary only
    #     the input draw, then only the weight draw.
    for r in 1:REPS
        if !(_key("vary_input", "fixed_w", r) in done)
            x = make_input(300 + r); cost = make_cost(codes, 300 + r)
            ref, _ = ep_gradient(reference_method(), chain, ps, st, x, cost)
            g, _ = ep_gradient(LockinEP(ε=0.03f0, ω_p=0.02f0, n_cycles=4,
                                        T_warmup_cycles=WARMUP, T_free=T_FREE, dt=DT),
                               chain, ps, st, x, cost)
            append_row(file, (probe="vary_input", setting="fixed_w", rep=r,
                              cos=worst_cos(fidelity(g, ref)), steps=0))
        end
        if !(_key("vary_weights", "fixed_x", r) in done)
            c2, p2, s2 = build_chain(400 + r)
            x = make_input(999); cost = make_cost(codes, 999)
            ref, _ = ep_gradient(reference_method(), c2, p2, s2, x, cost)
            g, _ = ep_gradient(LockinEP(ε=0.03f0, ω_p=0.02f0, n_cycles=4,
                                        T_warmup_cycles=WARMUP, T_free=T_FREE, dt=DT),
                               c2, p2, s2, x, cost)
            append_row(file, (probe="vary_weights", setting="fixed_x", rep=r,
                              cos=worst_cos(fidelity(g, ref)), steps=0))
        end
    end
    # (4) PAIRED probe. The per-replicate data shows a bimodal failure mode:
    #     most draws give cos 0.85-0.99, but occasional ones collapse to ~0
    #     (measured -0.034). This records reference-self-agreement and
    #     lock-in-vs-reference on the SAME draw, so the two can be
    #     correlated. If the reference stays self-consistent where lock-in
    #     collapses, the failure is specific to the demodulation; if both
    #     collapse together, the equilibrium itself is ill-conditioned for
    #     that input and no lock-in setting can rescue it.
    pfile = joinpath(OUT, "variance_paired.csv")
    pdone = done_keys(pfile, (:probe, :setting, :rep))
    for r in 1:(4 * REPS)
        _key("paired", "same_draw", r) in pdone && continue
        x = make_input(300 + r); cost = make_cost(codes, 300 + r)
        r1, _ = ep_gradient(reference_method(β=0.005f0), chain, ps, st, x, cost)
        r2, _ = ep_gradient(reference_method(β=0.002f0, T_free=800, T_nudge=400),
                            chain, ps, st, x, cost)
        g,  _ = ep_gradient(LockinEP(ε=0.03f0, ω_p=0.02f0, n_cycles=2,
                                     T_warmup_cycles=WARMUP, T_free=T_FREE, dt=DT),
                            chain, ps, st, x, cost)
        append_row(pfile, (probe="paired", setting="same_draw", rep=r,
                           cos=worst_cos(fidelity(g, r1)), steps=0,
                           ref_selfcos=worst_cos(fidelity(r1, r2))))
    end

    summarize_variance(file)
    summarize_paired(pfile)
end

"Correlate lock-in failure with reference self-consistency on the same draw."
function summarize_paired(file)
    rows = [r for r in read_rows(file)
            if get(r, "probe", "") == "paired" && haskey(r, "ref_selfcos")]
    isempty(rows) && return
    lk = [parse(Float64, r["cos"]) for r in rows]
    rf = [parse(Float64, r["ref_selfcos"]) for r in rows]
    bad = lk .< 0.9
    println("\n-- paired probe: is the reference also broken on bad draws? --")
    @printf("  n=%d   lock-in cos: median %.4f  p10 %.4f  min %.4f\n",
            length(lk), quantile_(lk, 0.5), quantile_(lk, 0.1), minimum(lk))
    @printf("         reference self-cos: median %.6f  min %.6f\n",
            quantile_(rf, 0.5), minimum(rf))
    @printf("  draws with lock-in cos < 0.9: %d/%d (%.0f%%)\n",
            count(bad), length(bad), 100count(bad)/length(bad))
    if any(bad)
        @printf("         reference self-cos on THOSE draws: min %.6f, median %.6f\n",
                minimum(rf[bad]), quantile_(rf[bad], 0.5))
        println("""
      If the reference stays ≈1 where lock-in collapses, the failure is in the
      demodulation and is a property of the operating point — map it.
      If the reference collapses too, the equilibrium is ill-conditioned for
      that input and the draw should be EXCLUDED, not averaged in.""")
    end
end

"Type-7 quantile without pulling in a dependency."
function quantile_(v::Vector{Float64}, p::Float64)
    isempty(v) && return NaN
    s = sort(v); n = length(s)
    n == 1 && return s[1]
    h = (n - 1) * p + 1
    lo = floor(Int, h); hi = min(lo + 1, n)
    return s[lo] + (h - lo) * (s[hi] - s[lo])
end

function summarize_variance(file)
    rows = read_rows(file)
    isempty(rows) && (@warn "no variance rows"; return)
    println("\n-- Stage 0: variance sources --")
    println("  probe          setting   n   mean cos    std cos   steps/eval")
    groups = Dict{Tuple{String,String},Vector{Float64}}()
    stepof = Dict{Tuple{String,String},String}()
    for r in rows
        k = (r["probe"], r["setting"])
        push!(get!(groups, k, Float64[]), parse(Float64, r["cos"]))
        stepof[k] = r["steps"]
    end
    for k in sort(collect(keys(groups)))
        v = groups[k]
        @printf("  %-13s %8s  %2d  %9.5f  %9.2e  %10s\n",
                k[1], k[2], length(v), mean(v), length(v) > 1 ? std(v) : NaN, stepof[k])
    end
    println("""
      Read: if `ref_vs_ref` cos is no closer to 1 than the n_cycles rows, the
      reference is the limiting noise source, not the lock-in estimator.
      If `vary_weights` spread >> `vary_input`, the zone is weight-dependent
      and the grid must average over weight draws too.""")
end

# ---------------------------------------------------------------------
# Stage 1 — cost-aware grid over (ε, ω_p, n_cycles), dt fixed
# ---------------------------------------------------------------------
#
# Why dt is not swept: cost depends on ω_p and dt only through the product
# (period_steps = 2π/(ω_p·dt)), but the PHYSICS depends on them separately —
# ω_p alone sets adiabaticity via ω_p/R_relax, while dt sets integration
# accuracy of the settle. Sweeping both would spend budget on a degenerate
# direction. Fix dt at the largest value the settle is stable at (0.5, the
# same step the static settle uses) and search ω_p.
#
# Why a grid and not Bayesian optimization: the deliverable is a contour, not
# an argmax (BO deliberately under-samples away from the optimum); the cost
# surface is known analytically so there is nothing to learn about it; and an
# evaluation is ~70 ms, so the whole grid is minutes. BO earns its complexity
# when evaluations cost minutes to hours.
const GRID_EPS      = (0.003f0, 0.01f0, 0.03f0, 0.1f0, 0.3f0)
const GRID_OMEGA    = (0.005f0, 0.01f0, 0.02f0, 0.05f0, 0.1f0, 0.2f0)
# Stage 0 measured mean cos essentially flat for n_cycles 1/2/4 (0.960/0.961/
# 0.961) and WORSE at 8 and 16 (0.958/0.930) — more demodulation cycles do not
# buy selectivity here, they just cost linearly more. So the grid includes 1.
const GRID_NCYCLES  = (1, 2, 4, 8)

function stage_grid()
    file = joinpath(OUT, "grid.csv")
    done = done_keys(file, (:eps, :omega_p, :n_cycles, :rep))
    codes = make_codes(8)
    chain, ps, st = build_chain(7)

    # Reference is independent of every lock-in knob, so compute it ONCE per
    # replicate and reuse across the whole grid. With 90 configs x REPS this
    # is the difference between REPS references and 90*REPS of them.
    @info "Stage 1: precomputing $(REPS) references (shared across all configs)"
    inputs = [(make_input(100 + r), make_cost(codes, 100 + r)) for r in 1:REPS]
    refs = Vector{Any}(undef, REPS)
    Threads.@threads for r in 1:REPS
        x, cost = inputs[r]
        refs[r], _ = ep_gradient(reference_method(), chain, ps, st, x, cost)
    end

    jobs = [(ε, ω, nc, r) for ε in GRID_EPS, ω in GRID_OMEGA,
                              nc in GRID_NCYCLES, r in 1:REPS]
    jobs = [j for j in vec(jobs) if !(_key(j[1], j[2], j[3], j[4]) in done)]
    # Cheapest first: cost varies ~40x across the grid, so this front-loads
    # coverage and leaves the expensive small-ω_p corner for last. A killed
    # run then still leaves a usable map.
    sort!(jobs, by = j -> lockin_steps(ω_p=j[2], dt=DT, n_cycles=j[3]))
    total_steps = sum(lockin_steps(ω_p=j[2], dt=DT, n_cycles=j[3]) for j in jobs; init=0)
    @info "Stage 1: $(length(jobs)) evaluations queued" total_settle_steps=total_steps

    lk = ReentrantLock(); ndone = Threads.Atomic{Int}(0)
    Threads.@threads for j in jobs
        ε, ω, nc, r = j
        x, cost = inputs[r]
        g, _ = ep_gradient(LockinEP(ε=ε, ω_p=ω, n_cycles=nc,
                                    T_warmup_cycles=WARMUP, T_free=T_FREE, dt=DT),
                           chain, ps, st, x, cost)
        f = fidelity(g, refs[r])
        row = (eps=ε, omega_p=ω, n_cycles=nc, rep=r,
               cos=worst_cos(f),
               cos_l1=f[1].cos, cos_l2=length(f) > 1 ? f[2].cos : NaN,
               relerr_l1=f[1].relerr,
               steps=lockin_steps(ω_p=ω, dt=DT, n_cycles=nc))
        lock(lk) do
            append_row(file, row)
            n = Threads.atomic_add!(ndone, 1) + 1
            n % 25 == 0 && @info "  $n/$(length(jobs)) done"
        end
    end
    report()
end

# ---------------------------------------------------------------------
# Stage 2 — read the zone off the grid
# ---------------------------------------------------------------------
function report()
    file = joinpath(OUT, "grid.csv")
    rows = read_rows(file)
    isempty(rows) && (@warn "no grid rows at $file — run the grid stage first"; return)

    agg = Dict{Tuple{String,String,String},Vector{Float64}}()
    steps = Dict{Tuple{String,String,String},Int}()
    for r in rows
        k = (r["eps"], r["omega_p"], r["n_cycles"])
        push!(get!(agg, k, Float64[]), parse(Float64, r["cos"]))
        steps[k] = parse(Int, r["steps"])
    end

    # Summaries are QUANTILE-based, not mean±stderr. The per-draw fidelity
    # distribution is bimodal, not Gaussian: most draws land at 0.85-0.99 and
    # occasional ones collapse to ~0 (measured -0.034). A mean over that is
    # not a meaningful centre and a Gaussian confidence bound is invalid.
    # What matters operationally is RELIABILITY — how often the estimator is
    # usable — so feasibility is gated on a lower quantile.
    println("\n-- fidelity map: worst-layer cos over draws, threshold $(THRESH) --")
    println("  eps      omega_p  cycles  n   median      p10        min    fail%  steps   feasible")
    ok = Tuple[]
    for k in sort(collect(keys(agg)), by = x -> (parse(Float64,x[1]), parse(Float64,x[2]), parse(Int,x[3])))
        v = agg[k]
        med = quantile_(v, 0.5); p10 = quantile_(v, 0.1)
        failfrac = count(<(0.9), v) / length(v)
        # Feasible = the 10th percentile clears the threshold, i.e. the
        # configuration is reliable across draws, not merely good on average.
        feas = length(v) >= 4 && p10 >= THRESH
        feas && push!(ok, (k..., steps[k], med))
        @printf("  %-8s %-8s %-6s %2d  %9.5f %9.5f %9.5f  %4.0f%%  %7d   %s\n",
                k[1], k[2], k[3], length(v), med, p10, minimum(v),
                100failfrac, steps[k], feas ? "yes" : "")
    end

    if isempty(ok)
        nmax = maximum(length(v) for v in values(agg))
        best_p10 = maximum(quantile_(v, 0.1) for v in values(agg))
        best_med = maximum(quantile_(v, 0.5) for v in values(agg))
        if nmax < 4
            @printf("\n  Only %d replicate(s) per point — too few for a p10. Re-run with EPS_REPS≥4.\n", nmax)
            @printf("  (best median cos so far: %.5f)\n", best_med)
        else
            @printf("\n  No configuration reaches p10 ≥ %.4f (best p10 %.5f, best median %.5f).\n",
                    THRESH, best_p10, best_med)
            println("  With a bimodal per-draw distribution this usually means occasional")
            println("  catastrophic draws, not a uniformly poor setting — check the fail%")
            println("  column and Stage 0's paired probe before loosening the threshold.")
        end
        return
    end
    sort!(ok, by = x -> x[4])
    println("\n-- cheapest feasible configurations (the operating zone's efficient frontier) --")
    println("  rank  eps      omega_p  cycles   steps   mean cos   speedup vs slowest feasible")
    slowest = maximum(x -> x[4], ok)
    for (i, o) in enumerate(ok[1:min(end, 8)])
        @printf("  %4d  %-8s %-8s %-6s %7d   %8.5f   %.1fx\n",
                i, o[1], o[2], o[3], o[4], o[5], slowest / o[4])
    end
    best = ok[1]
    @printf("\n  Recommended: ε=%s, ω_p=%s, n_cycles=%s, dt=%g  (%d steps/gradient)\n",
            best[1], best[2], best[3], DT, best[4])
end

# ---------------------------------------------------------------------
function main()
    stage = isempty(ARGS) ? "grid" : ARGS[1]
    BLAS.set_num_threads(1)   # parallelism belongs across configs, not inside gemm
    @info "ep_adiabatic_sweep" stage julia_threads=Threads.nthreads() blas=BLAS.get_num_threads() B REPS DT
    isdir(OUT) || mkpath(OUT)
    if stage == "variance"
        stage_variance()
    elseif stage == "grid"
        stage_grid()
    elseif stage == "report"
        report()
        summarize_variance(joinpath(OUT, "variance.csv"))
        summarize_paired(joinpath(OUT, "variance_paired.csv"))
    else
        error("unknown stage \"$stage\" — expected variance | grid | report")
    end
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main()
end
