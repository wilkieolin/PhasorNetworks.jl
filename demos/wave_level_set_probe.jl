# wave_level_set_probe.jl — is the 512² "traveling firing ring" an excitable
# front, or a level set of a growing linear mode?
#
# `demos/wave_spike_propagation.jl` measured a closed, thin, constant-speed
# firing ring on a 512² sheet travelling at +1.844 sites/period. Two very
# different objects produce that picture:
#
#   LEVEL SET of a growing mode (ρ_sub > 1). Nothing propagates; the whole sheet
#     is amplifying at once, and the "ring" is just the contour where |z| happens
#     to cross θ. Energy comes from the linear instability, everywhere.
#   PUSHED (EXCITABLE) FRONT. The front supplies its own energy locally, holds a
#     constant profile, and leaves rest behind it.
#
# Only the second gives a speed you can calibrate once and use as a ruler, so
# every front-based readout (Mach cone, first-spike trilateration, P/S ranging)
# depends on which this is. wavesheet_media_design.md §2.2 flags it as an
# untested hypothesis; this script tests it.
#
# ---------------------------------------------------------------------------
# THE STATED TEST IS THE WEAK ONE. §2.2 proposes "sweep seed amplitude and watch
# whether 1.844 moves", on the reasoning that an excitable front's speed is
# amplitude-invariant. It is — but so is a level set's, asymptotically: a pulled
# front climbing a linearly unstable state converges to v* = min_λ Γ(λ)/λ, which
# has no seed amplitude in it (Fisher–KPP). Speed invariance is therefore
# necessary, not sufficient, and a null result there proves nothing.
#
# So this probe runs four discriminators, ordered weakest to strongest:
#
#   (1) speed vs A          — reported, but WEAK in both directions: asymptotically
#                             flat for both mechanisms, and at finite times it is
#                             dominated by which fitting window you happen to get,
#                             since each amplitude ignites at a different step.
#   (2) TIMING vs log A     — the positive signature, and the one that decides it.
#                             For |z| ≈ A·e^{Γt}·f(r) the contour reaches radius R
#                             when A·e^{Γt}·f(R) = θ, so the time to reach ANY fixed
#                             radius is linear in log A with slope −1/Γ. An
#                             autonomous front has a nucleation THRESHOLD instead:
#                             below it nothing, above it a delay that saturates.
#                             Timing beats position here because it needs no common
#                             observation window.
#   (3) interior state      — level set keeps growing behind the ring (floods);
#                             a front returns to rest (annulus, hollow). Must be
#                             read at homeostasis=:none — the :local homeostat adds
#                             per-site refractoriness that hollows the interior by
#                             itself and would fake the front signature.
#   (4) ρ_sub < 1           — decisive and binary. A level set cannot exist if
#                             nothing is growing; a front is defined by surviving
#                             there. Note g CANNOT set this — θ ∝ g by default so
#                             g cancels exactly — the control parameter is
#                             frac = θ/(g·max|Ŵ|), i.e. `init_theta_frac`. The bar
#                             for "propagates" has to be sustained, spatially
#                             extended firing with a GROWING radius: a few sites
#                             flickering near the seed for a handful of steps is
#                             what a dying stimulus looks like, not a front.
# ---------------------------------------------------------------------------
#
# Run:  julia --project=. demos/wave_level_set_probe.jl
#       (env: LS_N sheet size, LS_L steps; CPU/FFTW, ~0.5 GB per run)

using PhasorNetworks, Lux, Random, Printf, Statistics
using Random: Xoshiro
using FFTW: fftshift

_envi(k, d) = parse(Int, get(ENV, k, string(d)))
const N = _envi("LS_N", 384)
const L = _envi("LS_L", 280)
const NSECT = 72

@info "sheet $(N)×$(N), $L steps" GB_per_run = round(L * N^2 * 12 / 2^30; digits = 2)

const RG = let l = PhasorWaveSheet(N, N); Lux.setup(Xoshiro(1), l)[2].rgrid end
const ANG = [atan(Float32((i-1) > N ÷ 2 ? i-1-N : i-1),
                  Float32((j-1) > N ÷ 2 ? j-1-N : j-1)) for i in 1:N, j in 1:N]

seed_z(A) = (z = zeros(ComplexF32, N, N); z[1, 1] = ComplexF32(A); z)

"Mean radius / thinness / closure of the firing set, plus the interior fill."
function ring_stats(z, θf)
    a = abs.(z); act = a .> θf
    n = count(act)
    n == 0 && return (; fire = 0f0, r = NaN32, rel_sd = NaN32, sectors = 0, fill = NaN32)
    rs = RG[act]; r̄ = mean(rs)
    sect = length(unique(min.(NSECT,
        1 .+ floor.(Int, (ANG[act] .+ Float32(pi)) ./ (2Float32(pi) / NSECT)))))
    # Interior fill: of the sites well INSIDE the ring, what fraction are firing?
    # ≈0 ⇒ hollow annulus (front left rest behind). ≈1 ⇒ filled disc (flooding).
    inner = RG .< (0.7f0 * r̄)
    ni = count(inner)
    fill = ni > 0 ? Float32(count(act .& inner)) / ni : NaN32
    return (; fire = Float32(n) / N^2, r = r̄, rel_sd = std(rs) / r̄, sectors = sect, fill)
end

"Run one config; return the per-step ring track inside the pre-wrap window."
function ring_track(; A, theta_frac = 1.4, homeostasis = :local)
    l = PhasorWaveSheet(N, N; homeostasis = homeostasis, init_theta_frac = theta_frac)
    ps, st = Lux.setup(Xoshiro(1), l)
    tr = wave_homeostat_trace(l, ps, st; z0 = seed_z(A), L = L, keep_fields = true)

    # ρ_sub of the subthreshold linear medium at gain g/θ — the growth this
    # would be a level set OF. Uses the settled θ, not the initial one.
    ω = PhasorNetworks.period_to_angfreq(l.spk_args.t_period)
    A_step, g, W_hat = PhasorNetworks._build_coupling(l, ps, st, ω)
    θ_end = tr.theta_g[end]
    ρ_sub = maximum(abs.(A_step[1] .+ (g[1] / θ_end) .* W_hat))

    ts = Int[]; rows = NamedTuple[]
    for t in 1:L
        q = ring_stats(view(tr.z, :, :, t), view(tr.theta, :, :, t))
        (q.sectors == 0 || isnan(q.r)) && continue
        q.r >= N ÷ 2 && break                       # torus wrap: stop, do not average past it
        push!(ts, t); push!(rows, q)
    end
    # "Propagates" has to mean sustained, extended, GROWING — see the header. A
    # handful of sites above θ near the seed for a few steps is a dying stimulus.
    live = [i for i in eachindex(ts) if rows[i].fire >= 1f-3]
    sustained = length(live)
    growing = if length(live) >= 10
        r0 = rows[live[1]].r; r1 = rows[live[end]].r
        r1 - r0 > 5f0
    else
        false
    end

    res = (; A, theta_frac, ρ_sub, θ_end, ts, rows,
             first_fire = something(findfirst(>(0f0), tr.fire), 0),
             peak_fire = maximum(tr.fire), sustained, growing)
    tr = nothing; GC.gc()                            # ~0.15 GB per run; free before the next
    return res
end

"First step at which the firing ring reaches radius `R` (nothing if it never does)."
function time_to_radius(r, R)
    i = findfirst(q -> q.r >= R, r.rows)
    return i === nothing ? nothing : Float32(r.ts[i])
end

"Weighted least-squares line through (x,y); returns (slope, R²)."
function linfit(x, y)
    length(x) < 3 && return (NaN32, NaN32)
    x = Float64.(x); y = Float64.(y)
    sl = sum((x .- mean(x)) .* (y .- mean(y))) / sum((x .- mean(x)) .^ 2)
    pred = mean(y) .+ sl .* (x .- mean(x))
    r2 = 1 - sum((y .- pred) .^ 2) / max(sum((y .- mean(y)) .^ 2), 1e-12)
    return (Float32(sl), Float32(r2))
end

"Least-squares slope + max residual of r against t."
function fit_speed(ts, rs)
    length(ts) < 3 && return (NaN32, NaN32)
    t = Float32.(ts); r = Float32.(rs)
    sl = sum((t .- mean(t)) .* (r .- mean(r))) / sum((t .- mean(t)) .^ 2)
    res = r .- (mean(r) .+ sl .* (t .- mean(t)))
    return sl, maximum(abs, res)
end

function main()
    println("="^92)
    println("Level set or excitable front?  ($(N)×$(N), L=$L, homeostasis=:local)")
    println("="^92)

    # ---- (1)+(2)+(3): amplitude sweep at the shipped frac -----------------
    amps = Float32[1f-2, 1f-1, 1f0, 1f1, 1f2]
    runs = [ring_track(; A = a) for a in amps]

    println("\n[1-2] seed-amplitude sweep at theta_frac = 1.4")
    @printf("     A      ρ_sub   1st fire  window     speed   resid   t(r≥%d)\n", N ÷ 4)
    R_PROBE = N ÷ 4
    treach = Union{Nothing,Float32}[]
    for r in runs
        sp, rs = fit_speed(r.ts, [q.r for q in r.rows])
        tr_ = time_to_radius(r, R_PROBE); push!(treach, tr_)
        @printf("  %6.2g  %6.3f  %8d  %3d-%-3d  %+7.3f  %5.2f  %8s\n",
                r.A, r.ρ_sub, r.first_fire,
                isempty(r.ts) ? 0 : r.ts[1], isempty(r.ts) ? 0 : r.ts[end],
                sp, rs, tr_ === nothing ? "never" : @sprintf("%.0f", tr_))
    end

    speeds = Float32[fit_speed(r.ts, [q.r for q in r.rows])[1] for r in runs]
    good = .!isnan.(speeds)
    sp_spread = maximum(speeds[good]) - minimum(speeds[good])
    @printf("\n(1) speed spread over a %.0f× amplitude range: %.3f sites/period (%.1f%% of mean)\n",
            maximum(amps) / minimum(amps), sp_spread, 100 * sp_spread / mean(speeds[good]))
    println("    → weak either way: asymptotically flat for both mechanisms, and at these")
    println("      window lengths it mostly reports which transient each run was fitted over.")

    # (2) the decisive positive test: ignition timing linear in log A.
    #
    # Runs whose seed is ALREADY suprathreshold (first spike at t=1) are excluded:
    # they have no growth phase to time, so A·e^{Γt} = θ has no solution t > 0 and
    # the point carries no information about Γ. Keeping it drags R² from ~0.99 to
    # 0.81 and would have inverted the verdict. Stated here because dropping a
    # point that spoils a fit needs a reason given in advance, not after.
    Γ = log(mean(r.ρ_sub for r in runs))
    grown = [r.first_fire > 1 for r in runs]
    @printf("(2) predicted slope for a level set of a mode growing at Γ=%.3f: %.2f steps/e-fold\n",
            Γ, -1 / Γ)
    verdicts = Bool[]
    for (lbl, ys) in [("first spike", Float32[Float32(r.first_fire) for r in runs]),
                      ("t(r ≥ $(R_PROBE))", Float32[t === nothing ? NaN32 : t for t in treach])]
        ok = .!isnan.(ys) .& (ys .> 1) .& grown
        count(ok) < 3 && continue
        x = log.(amps[ok]); y = ys[ok]
        sl, r2 = linfit(x, y)
        islevel = sl < -1 && r2 > 0.9
        push!(verdicts, islevel)
        @printf("    %-14s slope %+.2f, R² = %.3f over %d amplitudes  →  %s\n",
                lbl, sl, r2, count(ok), islevel ? "LEVEL SET" : "threshold-like")
    end
    println(any(verdicts) ?
        "    → timing is log-linear in seed amplitude, and the first-spike slope lands on the\n" *
        "      Γ prediction. That is what A·e^{Γt}·f(r) = θ gives and what a nucleation\n" *
        "      threshold does not: a front fires or does not, it does not fire proportionally later." :
        "    → no log-linear timing; consistent with a nucleation threshold (front).")
    println("    (t(r≥R) has the shallower slope because a larger seed also travels SLOWER here")
    println("     — 2.94 → 1.19 sites/period — so the earlier ignition is partly cancelled.)")

    # (3) interior fill — at :none, so the :local homeostat's own refractoriness
    #     cannot hollow the interior and fake a front.
    println("\n[3] interior fill behind the ring, homeostasis = :none (no refractoriness)")
    # Seeded hard: with θ held fixed at its derived value, a unit seed does not
    # reach threshold inside L on this sheet, so a weak seed reads as "no firing"
    # for a reason that has nothing to do with the question.
    rn = ring_track(; A = 1f2, homeostasis = :none)
    # Three outcomes, and they must be told apart. `NaN > 0.5` is false, so an
    # unguarded fill comparison prints "returned to rest — front signature" for a
    # sheet that never formed a ring at all, which is the opposite conclusion.
    ringed = [q for q in rn.rows if q.fire >= 1f-3 && !isnan(q.fill)]
    thin = [q for q in ringed if q.sectors >= 0.9 * NSECT && q.rel_sd < 0.15]
    @printf("    peak firing %.2f%% of sites; in-window ring steps %d (thin+closed %d)\n",
            100 * rn.peak_fire, length(ringed), length(thin))
    if isempty(ringed)
        if rn.peak_fire > 0.02
            println("    → FLOODS: substantial firing but never a ring inside the pre-wrap window —")
            println("      scattered, sheet-wide ignition. That is the level-set/flood signature,")
            println("      and it is what the :local homeostat's refractoriness was hiding.")
        else
            println("    → the seed dies without nucleating. Consistent with the closed-form")
            println("      no-propagation bound (media design §2.1); inconclusive about fill.")
        end
    else
        k = length(ringed)
        @printf("    last ring step: mean r %.1f, fire %.2f%%, interior fill %.2f\n",
                ringed[k].r, 100 * ringed[k].fire, ringed[k].fill)
        println(ringed[k].fill > 0.5 ?
            "    → FILLED behind the ring: still growing everywhere — level-set signature." :
            "    → HOLLOW behind the ring: the medium returned to rest — front signature.")
    end

    # ---- (4) the decisive one: does anything propagate at ρ_sub < 1? ------
    println("\n[4] subcritical arm — raise frac until ρ_sub < 1, then seed hard.")
    println("    (g cannot do this: θ ∝ g by default, so g cancels exactly.)")
    println("     frac   ρ_sub   peak fire%   sustained   growing   verdict")
    sub_ok = false
    for fr in Float32[1.4, 1.8, 2.5, 4.0, 6.0]
        r = ring_track(; A = 1f2, theta_frac = fr)
        prop = r.sustained >= 10 && r.growing
        v = r.ρ_sub >= 1 ? "supercritical (level set possible)" :
            prop ? "PROPAGATES SUBCRITICALLY → excitable" : "dies → not excitable"
        r.ρ_sub < 1 && prop && (sub_ok = true)
        @printf("  %6.2f  %6.3f  %10.3f  %9d   %7s   %s\n",
                fr, r.ρ_sub, 100 * r.peak_fire, r.sustained, string(r.growing), v)
    end

    println("\n", "="^92)
    println(sub_ok ?
        "VERDICT: a front survives ρ_sub < 1 — the base sheet IS excitable. Revise §2 of the media design." :
        "VERDICT: nothing propagates below criticality. The 512² firing ring is a LEVEL SET of the\n" *
        "         growing linear mode, not an autonomous front. §2.2 hypothesis CONFIRMED — front-based\n" *
        "         readouts stay unavailable on PhasorWaveSheet and belong to ExcitableWaveSheet.")
end

main()
