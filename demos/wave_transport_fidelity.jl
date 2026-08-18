# wave_transport_fidelity.jl — how far does a SYMBOL survive, not how far does
# a centroid move.
#
# Every propagation number in this repo is either a band-theory quantity
# (dispersion, v_g, GVD, gain curvature, transport ratio) or a centroid/spread
# measurement. None of them asks whether the thing being carried is still
# recoverable at the far end. Report §4.6 predicts the answer from the code's own
# spectrum — a CONCENTRATED code is broadband and shreds, an EXTENDED code tuned
# to q* is narrowband and translates rigidly — but that prediction was computed by
# hand for one figure and never checked against a measured survival distance.
#
# This benchmark measures it. Inject an FHRR hypervector as a spatial code,
# propagate, decode differentially against a payload-free control run (which
# cancels the propagation phase), and score with the ordinary VSA `similarity`.
#
#   fidelity(t) = similarity(decoded phases at step t, the injected hypervector)
#
# and report:
#   t½ — steps (= carrier periods) until fidelity falls to 0.5
#   d½ — DISTANCE the packet travelled by then, in sites
#
# ---------------------------------------------------------------------------
# WHY d½ AND NOT FIDELITY. Fidelity alone is a trap, and the `no-coupling`
# control arm exists to spring it: with g→0 nothing propagates and nothing
# dephases, so fidelity stays ≈1.00 forever and that arm WINS on fidelity while
# transporting the symbol precisely nowhere. The quantity a pipeline (or a lap of
# a looped transformer) can actually spend is distance-at-fidelity. Same failure
# mode as the four collected in wavesheet_media_design.md §6.7 — a summary
# statistic satisfied by the wrong mechanism.
# ---------------------------------------------------------------------------
#
# Axes crossed:
#   concentration — delta · box 2/4/8 · Gaussian σ=6 · Gabor σ=10 at q*
#   medium        — :dog matched-c · :dog c=40 · :aniso sharp · :aniso gentle-DC
#                   · :shift conveyor · no-coupling control
#   payload       — random hypervector (broadband) and a low-passed one, to
#                   separate "the envelope disperses" from "the payload itself is
#                   broadband along the other axis"
#
# Diffusive media are evaluated AT CRITICALITY (g ≈ 0.9·g_crit, bisected here).
# At the stock g the DC mode wins, q*→0, and every transport diagnostic reads
# :standing — a documented trap, see wavesheet-carrier-folds-out.
#
# Run:  julia --project=. demos/wave_transport_fidelity.jl
#       (env: TF_N sheet size, TF_L steps, TF_SEED, TF_SPIKE=1 to add :spike arms)
# Outputs a table + demos/wave_out/transport_fidelity.{png,csv}.

ENV["GKSwstype"] = "100"                     # headless GR

using PhasorNetworks, Lux, Random, Printf, Statistics
using Random: Xoshiro
using Plots

_envi(k, d) = parse(Int, get(ENV, k, string(d)))
const N     = _envi("TF_N", 64)              # sheet is N×N; columns carry the hypervector
const LMAX  = _envi("TF_L", 160)             # rollout length (steps = carrier periods)
const SEED  = _envi("TF_SEED", 11)
const SPIKE = _envi("TF_SPIKE", 0) == 1
const OUT   = joinpath(@__DIR__, "wave_out")

# ---------------------------------------------------------------------------
# Codes. Rows = the transport axis, so the ENVELOPE along rows sets the spectral
# occupancy that decides transport. Columns = the D=N hypervector components, so
# the PAYLOAD is orthogonal to the transport axis and the two can be varied
# independently — that separation is the whole point of the layout.
# ---------------------------------------------------------------------------

const CTR = N ÷ 2
_wrapd(a, b, n) = min(abs(a - b), n - abs(a - b))

"Row envelopes, ordered from maximally concentrated to maximally extended."
function envelopes(q_star)
    delta = Float32[i == CTR ? 1f0 : 0f0 for i in 1:N]
    box(w) = Float32[_wrapd(i, CTR, N) <= w ? 1f0 : 0f0 for i in 1:N]
    gauss(σ) = Float32[exp(-_wrapd(i, CTR, N)^2 / (2f0 * σ^2)) for i in 1:N]
    # The Gabor is the §4.6 "extended patch tuned to q*" — a smooth envelope
    # MODULATED at the band's own selected wavevector, so its spectrum is a thin
    # peak sitting on the gain maximum instead of spanning the whole band.
    gabor(σ) = ComplexF32[exp(-_wrapd(i, CTR, N)^2 / (2f0 * σ^2)) *
                          cis(q_star * (i - CTR)) for i in 1:N]
    return ["delta"      => ComplexF32.(delta),
            "box w=2"    => ComplexF32.(box(2)),
            "box w=4"    => ComplexF32.(box(4)),
            "box w=8"    => ComplexF32.(box(8)),
            "gauss σ=6"  => ComplexF32.(gauss(6.0)),
            "gabor σ=10" => gabor(10.0)]
end

"Random FHRR hypervector in [-1,1]^N; `smooth>0` low-passes it along columns."
function payload(rng; smooth::Int = 0)
    φ = 2f0 .* rand(rng, Float32, N) .- 1f0
    smooth == 0 && return φ
    z = cis.(Float32(pi) .* φ)                              # circular smoothing
    for _ in 1:smooth
        z = (circshift(z, 1) .+ 2f0 .* z .+ circshift(z, -1)) ./ 4f0
    end
    return Float32.(angle.(z) ./ Float32(pi))
end

"z0 = envelope(row) ⊙ exp(iπ·φ(col)); `carry=false` gives the payload-free control."
code_field(env, φ; carry::Bool = true) =
    ComplexF32[env[i] * (carry ? cis(Float32(pi) * φ[j]) : 1f0 + 0f0im)
               for i in 1:N, j in 1:N]

# ---------------------------------------------------------------------------
# Media. Diffusive arms are bisected to g ≈ 0.9·g_crit before use.
# ---------------------------------------------------------------------------

"Bisect log_g so the sheet's spectral radius hits `target`. Returns rescaled ps."
function at_criticality(l, ps, st; target = 0.9f0, iters = 48)
    ρ(lg) = dispersion(l, merge(ps, (; log_g = Float32[lg])), st).spectral_radius
    lo, hi = log(1f-6), log(1f3)
    ρ(lo) >= target && return ps                             # already supercritical at ~0 gain
    for _ in 1:iters
        mid = 0.5f0 * (lo + hi)
        ρ(mid) < target ? (lo = mid) : (hi = mid)
    end
    return merge(ps, (; log_g = Float32[0.5f0 * (lo + hi)]))
end

"Subthreshold spectral radius of a `:spike` sheet: the linear medium at gain g/θ."
function rho_sub(l, ps, st)
    ω = period_to_angfreq(l.spk_args.t_period)
    A_step, g, W_hat = PhasorNetworks._build_coupling(l, ps, st, ω)
    θ = exp(ps.log_theta[1])
    return maximum(abs.(A_step[1] .+ (g[1] / θ) .* W_hat))
end

"""
Bisect **log_theta** (not log_g) so a `:spike` sheet's ρ_sub hits `target`.

`at_criticality` is a no-op here and silently so. The derived default is
`θ = frac · g·max|Ŵ|`, so `ρ_sub = max|A + (g/θ)Ŵ|` has `g` cancel exactly — the
documented "g cannot stabilise the sheet in :spike" result. The dimensionless
control parameter is `frac = θ/(g·max|Ŵ|)`, i.e. θ.
"""
function at_criticality_spike(l, ps, st; target = 0.9f0, iters = 48)
    ρ(lt) = rho_sub(l, merge(ps, (; log_theta = Float32[lt])), st)
    lo, hi = log(1f-3), log(1f6)                             # small θ ⇒ huge gain ⇒ ρ big
    for _ in 1:iters
        mid = 0.5f0 * (lo + hi)
        ρ(mid) > target ? (lo = mid) : (hi = mid)            # ρ DEcreases in θ
    end
    return merge(ps, (; log_theta = Float32[0.5f0 * (lo + hi)]))
end

"Named media. Each returns (layer, ps, st) ready to roll."
function media(rng; transmit = :potential)
    mk(; kw...) = PhasorWaveSheet(N, N; transmit = transmit, kw...)
    out = Pair{String,Any}[]

    # (1) isotropic DoG at the derived matched conduction speed c = 2σ_I/T.
    for (nm, kw) in ["dog matched-c" => (;),
                     "dog c=40"      => (; init_log_speed = log(40.0))]
        l = mk(; kw...); p, s = Lux.setup(copy(rng), l)
        push!(out, nm => (l, at_criticality(l, p, s), s))
    end

    # (2) :aniso — a directed current on top of the hat. "sharp" is the selective
    #     band-pass filter; "gentle-DC" is the flattened-gain regime of §10.3.
    #
    #     β MUST be set AFTER the gain bisection. The drift is g·β, so bisecting g
    #     down to criticality silently crushes whatever β was set at construction:
    #     β=0.6 fixed up front measures v_g* = 0.007, ~20× below the report's 0.13,
    #     and the arm then reads "aniso does not transport" when what was actually
    #     measured is "aniso at this g does not transport". Solving β = v*/g puts
    #     both arms on report §10.3's operating point.
    for (nm, kw, vtgt) in [("aniso sharp",
                            (; init_log_speed = log(40.0)), 0.13f0),
                           ("aniso gentle-DC",
                            (; init_log_speed = log(40.0), init_A_exc = 1.0,
                               init_log_sigma_exc = log(3.0), init_B_inh = 0.10,
                               init_log_sigma_inh = log(6.0)), 0.09f0)]
        la = mk(; coupling = :aniso, init_beta_h = 0.0, kw...)
        pa, sa = Lux.setup(copy(rng), la)
        pa = at_criticality(la, pa, sa)
        pa = merge(pa, (; beta_h = Float32[vtgt / exp(pa.log_g[1])]))
        push!(out, nm => (la, pa, sa))
    end

    # (3) the shift conveyor: |Ŵ|≡1, exactly linear phase. NOT rescaled — a unit-
    #     gain phase ramp has no criticality reference (see `emission_threshold`).
    #
    #     A conveyor scoring fidelity 1.000 is NOT evidence — it is arithmetic.
    #     |Ŵ|≡1 with exactly linear phase is a unitary translation, and a
    #     translation cannot corrupt anything it carries. The s=0.7 arm was added
    #     to check whether s=1 was merely getting an axis-aligned freebie (an
    #     integer row-shift is a lattice permutation, and the payload varies along
    #     COLUMNS, so it cannot be mixed). It does not test that either: the shift
    #     acts only along rows, so for ANY row-kernel K,
    #         (K ⊛_h (env ⊙ p))[i,j] = (K ⊛_h env)[i,j] · p[j],
    #     and the differential decode cancels it exactly whether s is an integer or
    #     not. The conveyor's losslessness is a theorem, not a measurement; what the
    #     benchmark measures about it is only its SPEED (see the leak sweep).
    #
    #     s=0.7 is kept anyway, because it is the sharpest available check on the
    #     forecast: it is the one medium whose true answer is known analytically,
    #     and `transport_forecast` predicts t½ ≈ 1.6 there against a measured >100.
    for (nm, sh) in ["shift conveyor s=1" => 1.0, "shift conveyor s=0.7" => 0.7]
        ls = mk(; coupling = :shift, init_shift_h = sh, init_shift_w = 0.0,
                  init_log_neg_lambda = log(5.0), init_log_g = log(0.99))
        push!(out, nm => (ls, Lux.setup(copy(rng), ls)...))
    end

    # (4) THE CONTROL. g→0: no coupling, so nothing propagates and nothing
    #     dephases. Must score fidelity ≈ 1 and d½ = 0.
    lc = mk(; init_log_g = log(1f-6))
    push!(out, "no-coupling ctrl" => (lc, Lux.setup(copy(rng), lc)...))
    return out
end

# ---------------------------------------------------------------------------
# Rollout + differential decode.
# ---------------------------------------------------------------------------

"Circular row-centroid of |z|² (wrap-safe), in sites."
function crow(z)
    m = vec(sum(abs2.(z); dims = 2))
    R = sum(m .* cis.(2f0 * Float32(pi) .* (0:N-1) ./ N)) / (sum(m) + 1f-20)
    return mod(angle(R) * N / (2f0 * Float32(pi)), N)
end

"""
Fidelity and cumulative displacement per step.

The decode is DIFFERENTIAL: run the same medium twice, once with the payload and
once with a bare envelope, then read `z_A ⊙ conj(z_B)` in a window around the
control's packet. The control carries the identical propagation phase, so the
unbind cancels it and what is left is the payload — the same cancellation the
serial-pipeline demo uses. Reading run A alone instead would score the medium's
own phase ramp as payload corruption.
"""
function fidelity_curve(l, ps, st, env, φ; L = LMAX, win = 3, decode::Symbol = :differential)
    # A supercritical config blows up to Inf/NaN, and the centroid then rounds to
    # a non-representable Int and throws deep inside the readout. Detect it here
    # and return a NaN curve so the caller can report "diverged" as a result.
    zA = code_field(env, φ; carry = true)
    zB = code_field(env, φ; carry = false)
    # ALWAYS :discrete, never :deq. Two reasons: :deq throws outright once
    # homeostasis is on (it needs a fixed θ), and it is the parallel fixed-point
    # approximation rather than the sequential ground truth we want to measure.
    YA = wave_simulate(l, ps, st; z0 = zA, L = L, mode = :discrete)  # (N,N,L)
    YB = wave_simulate(l, ps, st; z0 = zB, L = L, mode = :discrete)

    fid = zeros(Float32, L); disp = zeros(Float32, L)
    c_prev = crow(zB); total = 0f0
    for t in 1:L
        a = @view YA[:, :, t]; b = @view YB[:, :, t]
        c = crow(b)
        (isfinite(c) && all(isfinite, a) && all(isfinite, b)) ||
            return fill(NaN32, L), disp
        # unwrap the circular centroid so displacement accumulates past the seam
        d = c - c_prev; d -= N * round(d / N); total += d; c_prev = c
        disp[t] = abs(total)

        rows = [mod(round(Int, c) + k - 1, N) + 1 for k in -win:win]
        if decode === :differential
            r = vec(sum(a[rows, :] .* conj.(b[rows, :]); dims = 1))  # (N,) per column
            φ̂ = Float32.(angle.(r) ./ Float32(pi))
            # `similarity` reduces dim 1, so on a pair of vectors it returns a
            # 0-dim array; `only` pulls the scalar out.
            fid[t] = all(isfinite, φ̂) ? Float32(only(similarity(φ̂, φ))) : 0f0
        else  # :blind
            r = vec(sum(a[rows, :]; dims = 1))                       # no control run
            φ̂ = Float32.(angle.(r) ./ Float32(pi))
            # Magnitude of the mean unbind phasor: coherence of (φ̂ − φ), which is
            # invariant to a global rotation. The differential decode removes the
            # propagation phase by subtracting a control run, and that is only
            # legitimate in a LINEAR medium — under :spike the two runs cross θ at
            # different times and (with homeostasis) settle at different θ, so the
            # control stops being the same channel. This is the decode that stays
            # valid, at the cost of being blind to a global offset.
            u = mean(cis.(Float32(pi) .* (φ̂ .- φ)))
            fid[t] = all(isfinite, φ̂) ? Float32(abs(u)) : 0f0
        end
    end
    return fid, disp
end

"First crossing of `fid` below `thr`, linearly interpolated. `nothing` if never."
function half_life(fid, thr = 0.5f0)
    i = findfirst(<(thr), fid)
    i === nothing && return nothing
    i == 1 && return 1f0
    f0, f1 = fid[i-1], fid[i]
    return Float32(i - 1) + (f0 - thr) / max(f0 - f1, 1f-9)
end

# ---------------------------------------------------------------------------

function main()
    rng = Xoshiro(SEED)
    mkpath(OUT)
    println("="^100)
    println("Transport fidelity — how far a SYMBOL survives  ($(N)×$(N) sheet, L=$(LMAX), seed $(SEED))")
    println("t½ = steps to 50% fidelity;  d½ = sites travelled by then;  pred = transport_forecast")
    println("="^100)

    rows = NamedTuple[]
    curves = Dict{String,Vector{Float32}}()
    transmits = SPIKE ? [:potential, :spike] : [:potential]

    for tr in transmits
        for (mname, (l, ps, st)) in media(Xoshiro(SEED); transmit = tr)
            dg = dispersion_diagnostics(l, ps, st; mode = tr)
            envs = envelopes(dg.q_star)
            @printf("\n[%s / %s]  q*=%.3f  v_g*=%+.3f  gvd*=%+.3f  ρ=%.3f\n",
                    mname, tr, dg.q_star, dg.v_g_star, dg.gvd_star,
                    dispersion(l, ps, st).spectral_radius)
            println("  code         payload     ℓ    Δv_g   lim    t½pred   t½meas   d½meas   fid@end")
            for (ename, env) in envs, (pname, sm) in ["random" => 0, "smooth" => 3]
                φ = payload(Xoshiro(SEED + 1), smooth = sm)
                fc = transport_forecast(l, ps, st, code_field(env, φ); mode = tr)
                fid, disp = fidelity_curve(l, ps, st, env, φ)
                th = half_life(fid)
                dh = th === nothing ? disp[end] : disp[clamp(ceil(Int, th), 1, LMAX)]
                key = "$(mname)|$(tr)|$(ename)|$(pname)"
                curves[key] = fid
                push!(rows, (; medium = mname, transmit = tr, code = ename, payload = pname,
                               ell = fc.ell, vg_spread = fc.vg_spread, limiter = fc.limiter,
                               t_half_pred = fc.t_half_pred,
                               t_half = th === nothing ? Float32(LMAX) : th,
                               censored = th === nothing, d_half = dh, fid_end = fid[end]))
                @printf("  %-11s  %-8s %5.2f %6.3f  %-5s %7.1f  %6s%s  %7.1f  %6.3f\n",
                        ename, pname, fc.ell, fc.vg_spread, String(fc.limiter),
                        min(fc.t_half_pred, 9999f0),
                        th === nothing ? ">$(LMAX)" : @sprintf("%.1f", th),
                        th === nothing ? " " : " ", dh, fid[end])
            end
        end
    end

    # ---- the gate: the control has top fidelity and zero transport -------
    ctrl = filter(r -> r.medium == "no-coupling ctrl", rows)
    real_media = filter(r -> r.medium != "no-coupling ctrl", rows)
    ctrl_fid = mean(r.fid_end for r in ctrl)
    ctrl_d = maximum(r.d_half for r in ctrl)
    best_d = maximum(r.d_half for r in real_media)
    println("\n", "="^100)
    @printf("CONTROL (no coupling): mean end-fidelity %.3f — the BEST of any arm; d½ = %.2f sites.\n",
            ctrl_fid, ctrl_d)
    @printf("Best real medium: d½ = %.1f sites. Fidelity alone would have ranked the control first.\n", best_d)
    gate = ctrl_d < 1f0 && best_d > 10f0 * max(ctrl_d, 1f-3)
    println(gate ? "GATE PASS ✓ — d½ separates transport from mere preservation." :
                   "GATE FAIL · — inspect the table above.")

    # ---- does the spectral-occupancy model predict survival? -------------
    # Only uncensored diffusive arms: the conveyor's prediction is +Inf and its
    # measurement is right-censored at L, so both would enter a correlation as
    # constants and manufacture agreement.
    _rank(v) = (p = sortperm(v); r = similar(v, Float64); r[p] = 1:length(v); r)
    spearman(a, b) = cor(_rank(a), _rank(b))
    cmp = filter(r -> !r.censored && isfinite(r.t_half_pred) && r.medium != "no-coupling ctrl", rows)
    if length(cmp) >= 3
        x = Float64[r.t_half_pred for r in cmp]; y = Float64[r.t_half for r in cmp]
        println("\n§4.6 check — does predicted t½ track measured t½?")
        @printf("  pooled (n=%d):  Pearson %+.3f   Spearman %+.3f\n",
                length(cmp), cor(x, y), spearman(x, y))
        for m in unique(r.medium for r in cmp)
            sub = filter(r -> r.medium == m, cmp)
            length(sub) < 4 && continue
            xa = Float64[r.t_half_pred for r in sub]; ya = Float64[r.t_half for r in sub]
            @printf("    %-17s n=%2d  Spearman %+.3f   pred %5.1f–%-5.1f  meas %5.1f–%-5.1f\n",
                    m, length(sub), spearman(xa, ya),
                    minimum(xa), maximum(xa), minimum(ya), maximum(ya))
        end
        # Does the model at least separate the two payload regimes — the part of
        # §4.6 that is about spectral content rather than geometry?
        #
        # Scored on end-fidelity over ALL diffusive rows, not on t½ over the
        # uncensored ones. Smoothing the payload is precisely what pushes a row
        # past L into censoring, so filtering censored rows out drops the smooth
        # arm's best results and inverts the comparison (measured 0.71× — smoothing
        # apparently HURTS — where the unfiltered answer is the opposite).
        allr = filter(r -> r.medium != "no-coupling ctrl" && !startswith(r.medium, "shift"), rows)
        pr = [r for r in allr if r.payload == "random"]
        psm = [r for r in allr if r.payload == "smooth"]
        if !isempty(pr) && !isempty(psm)
            fr, fs = mean(r.fid_end for r in pr), mean(r.fid_end for r in psm)
            nr = count(r -> r.censored, pr); ns = count(r -> r.censored, psm)
            @printf("  payload effect (all %d diffusive rows): mean end-fidelity %.3f → %.3f (%+.0f%%),\n",
                    length(allr), fr, fs, 100 * (fs - fr) / max(abs(fr), 1f-6))
            @printf("    still-above-50%%-at-L: %d/%d random vs %d/%d smooth\n",
                    nr, length(pr), ns, length(psm))
            println("    → smoothing the payload IS the lever §4.6 predicts, and it is the strongest")
            println("      one in the whole grid — stronger than any envelope shape.")
        end
        println(cor(x, y) > 0.5 ?
            "  → the spectral-occupancy model predicts survival. §4.6 stands as a quantitative claim." :
            "  → it does NOT, and the per-medium rows are the damning part: WITHIN a medium the\n" *
            "     forecast is ANTI-correlated with measurement, so it is not merely noisy. §4.6 is\n" *
            "     right that spectral content is the lever (see the payload effect below) and right\n" *
            "     about the ordering ACROSS media, but its occupancy-vs-band product does not\n" *
            "     predict how far a symbol survives. Treat the measurement as the authority and\n" *
            "     `transport_forecast` as a screening heuristic only.")
    end

    # ---- how much self-memory can a lap afford? --------------------------
    # §10.5 prices the conveyor honestly: to make the shift dominate, each site
    # must mostly forget its own state (A = e^{kT} ≈ 0), so the sheet stops being
    # a resonant medium and becomes a delay line. That trade-off has never been
    # measured, and it is the number a looped/recurrent-depth architecture needs:
    # a lap that keeps no self-memory cannot integrate anything.
    println("\nLeak sweep on the conveyor — what the ballistic regime costs in self-memory")
    println("  §10.5 prices the conveyor as needing A ≈ 0, i.e. no self-memory: the sheet stops")
    println("  being a resonant medium. A lap of a looped architecture that keeps no state")
    println("  cannot integrate anything, so this is the trade-off that decides Part 3.")
    println("  Fidelity stays 1.000 down the whole column — as it must, a shift is unitary.")
    println("  The cost of self-memory is therefore SPEED, not fidelity: read the d½ column.")
    println("  `g` is set to hold ρ ≈ 0.99 at every λ. It has to be: |Ŵ|≡1 for a shift, so")
    println("  ρ ≈ |A| + g, and holding g fixed at 0.99 makes the sheet ρ=1.85 by λ=0.15 —")
    println("  it diverges to NaN and the sweep reports a crash instead of a trade-off.")
    println("   λ        A       g      Δv_g    d½     fid@end   regime")
    φls = payload(Xoshiro(SEED + 1); smooth = 0)
    envls = envelopes(0f0)[4][2]                       # box w=8
    for λ in Float32[5.0, 2.0, 1.0, 0.5, 0.3, 0.15]
        Al = exp(-λ)                                   # |e^{kT}| at T=1
        gl = max(0.02f0, 0.99f0 - Al)
        ll = PhasorWaveSheet(N, N; coupling = :shift, transmit = :potential,
                             init_shift_h = 0.7, init_shift_w = 0.0,
                             init_log_neg_lambda = log(λ), init_log_g = log(gl))
        pl, sl_ = Lux.setup(Xoshiro(SEED), ll)
        fcl = transport_forecast(ll, pl, sl_, code_field(envls, φls); mode = :potential)
        fidl, displ = fidelity_curve(ll, pl, sl_, envls, φls)
        if !all(isfinite, fidl)
            @printf("  %5.2f  %7.4f  %6.3f       —       —        —     DIVERGED\n", λ, Al, gl)
            continue
        end
        thl = half_life(fidl)
        dhl = thl === nothing ? displ[end] : displ[clamp(ceil(Int, thl), 1, LMAX)]
        @printf("  %5.2f  %7.4f  %6.3f  %6.3f  %6.1f   %6.3f   %s\n",
                λ, Al, gl, fcl.vg_spread, dhl, fidl[end],
                Al < 0.1 ? "ballistic (no self-memory)" :
                Al < 0.6 ? "mixed" : "resonant (shift subdominant)")
    end

    # ---- outputs ---------------------------------------------------------
    open(joinpath(OUT, "transport_fidelity.csv"), "w") do io
        println(io, "medium,transmit,code,payload,ell,vg_spread,limiter,t_half_pred,t_half,censored,d_half,fid_end")
        for r in rows
            @printf(io, "%s,%s,%s,%s,%.4f,%.4f,%s,%.4f,%.4f,%d,%.4f,%.4f\n",
                    r.medium, r.transmit, r.code, r.payload, r.ell, r.vg_spread,
                    r.limiter, r.t_half_pred, r.t_half, r.censored, r.d_half, r.fid_end)
        end
    end

    # ---- :spike transmission with the homeostatic mechanisms --------------
    #
    # Everything above is :potential with homeostasis = :none — the LINEAR medium.
    # That is the regime the dispersion theory is exact in, and it is also not the
    # library default (`transmit = :spike`). Three things change here and each one
    # invalidates a piece of the machinery above:
    #
    #   1. `at_criticality` is a silent no-op: θ ∝ g by default so g cancels in
    #      ρ_sub. The control parameter is θ.
    #   2. `mode = :deq` throws once homeostasis is on. Use :discrete.
    #   3. The differential decode assumes a linear medium. Use the blind decode,
    #      and use it for the :potential reference row too so they compare.
    #
    # SEED AMPLITUDE IS IN UNITS OF θ, and this is load-bearing. θ is derived as
    # `1.4·g·max|Ŵ| ≈ 15` at these settings, so a unit-amplitude code sits three
    # orders of magnitude below threshold, nothing ever fires, and the run silently
    # measures the LINEAR medium at gain g/θ a second time — spike transmission
    # never happens. (First attempt did exactly that: fire% = 0.000 on every real
    # medium.) Sweeping amplitude in units of θ is what puts the sheet in the
    # regime the question is about.
    #
    # θ is left at the SHIPPED default (frac = 1.4) rather than bisected to a
    # target ρ_sub: that default is the library's actual operating point, which is
    # what "does it work in :spike" has to mean.
    #
    # Also reported: firing rate and std|z|. Rate alone is famously not enough — a
    # uniform subthreshold sheet hits any rate target exactly — so the structure
    # statistic travels with it.
    println("\n", "="^100)
    println(":spike transmission × homeostasis  (blind decode; :discrete rollout)")
    println("="^100)
    φsp = payload(Xoshiro(SEED + 1); smooth = 0)
    spike_rows = NamedTuple[]
    for (mlabel, kw) in ["dog matched-c" => (;),
                         "shift conveyor" => (; coupling = :shift, init_shift_h = 1.0,
                                                init_shift_w = 0.0,
                                                init_log_neg_lambda = log(5.0),
                                                init_log_g = log(0.99)),
                         "no-coupling ctrl" => (; init_log_g = log(1f-6))]
        for hm in (:none, :global, :local), amp in Float32[2, 20]
            l = PhasorWaveSheet(N, N; transmit = :spike, homeostasis = hm, kw...)
            ps, st = Lux.setup(Xoshiro(SEED), l)
            θ0 = exp(ps.log_theta[1])
            envs = envelopes(dispersion_diagnostics(l, ps, st; mode = :spike).q_star)
            env = (amp * θ0) .* envs[4][2]                          # box w=8, scaled to θ
            fid, disp = fidelity_curve(l, ps, st, env, φsp; decode = :blind)
            th = half_life(fid)
            dh = th === nothing ? disp[end] : disp[clamp(ceil(Int, th), 1, LMAX)]
            tr = wave_homeostat_trace(l, ps, st; z0 = code_field(env, φsp), L = LMAX)
            push!(spike_rows, (; medium = mlabel, homeostasis = hm, amp = amp,
                                 ρ_sub = rho_sub(l, ps, st), θ = θ0,
                                 fire = maximum(tr.fire), fire_mean = mean(tr.fire),
                                 structure = mean(tr.std_abs),
                                 t_half = th === nothing ? Float32(LMAX) : th,
                                 censored = th === nothing, d_half = dh, fid_end = fid[end]))
        end
    end
    println("  medium            homeo   seed  ρ_sub    θ     fire% pk/mean  std|z|    t½     d½   fid@end")
    for r in spike_rows
        @printf("  %-16s %-7s %4.0fθ %6.3f %7.2f %6.2f/%-5.2f %7.3f %6s %6.1f  %6.3f\n",
                r.medium, String(r.homeostasis), r.amp, r.ρ_sub, r.θ,
                100 * r.fire, 100 * r.fire_mean, r.structure,
                r.censored ? ">$(LMAX)" : @sprintf("%.1f", r.t_half), r.d_half, r.fid_end)
    end

    println("\n  Reading this table — three caveats, each of which inverts a row's meaning:")
    nofire = count(r -> r.fire < 1f-4, spike_rows)
    if nofire > 0
        @printf("  (a) %d/%d rows NEVER FIRED, so they measure the subthreshold linear medium at\n",
                nofire, length(spike_rows))
        println("      gain g/θ, not spike transmission. The conveyor is all of them: ballistic")
        println("      transport needs A≈0 (λ=5), which leaves |z|≈g≈1 against θ=1.39, permanently")
        println("      subthreshold. :shift and :spike are incompatible at the shipped leak.")
    end
    println("  (b) the no-coupling arm is NOT a control here. θ = 1.4·g·max|Ŵ| ∝ g, and the seed")
    println("      is set in units of θ, so lowering g rescales threshold, seed and coupling")
    println("      together and the normalised dynamics is unchanged — hence its t½/fid columns")
    println("      match the :dog sheet exactly. A real no-transport control in :spike has to pin")
    println("      θ independently of g. This is the g-cancellation, showing up as a broken arm.")
    println("  (c) peak firing ≈35% on the :dog rows is a FLOOD (ρ_sub = 1.08 > 1), not activity")
    println("      at the 2% homeostatic target. The homeostat moves end-fidelity but not this.")
    # :potential reference under the SAME blind decode, so the two are comparable.
    lref = PhasorWaveSheet(N, N; transmit = :potential)
    pref, sref = Lux.setup(Xoshiro(SEED), lref)
    pref = at_criticality(lref, pref, sref)
    envref = envelopes(dispersion_diagnostics(lref, pref, sref; mode = :potential).q_star)[4][2]
    fref, dref = fidelity_curve(lref, pref, sref, envref, φsp; decode = :blind)
    thref = half_life(fref)
    @printf("\n  :potential reference (dog matched-c, blind decode): t½ %s, fid@end %.3f\n",
            thref === nothing ? ">$(LMAX)" : @sprintf("%.1f", thref), fref[end])
    open(joinpath(OUT, "transport_fidelity_spike.csv"), "w") do io
        println(io, "medium,homeostasis,seed_amp_theta,rho_sub,theta,peak_fire,mean_fire,structure,t_half,censored,d_half,fid_end")
        for r in spike_rows
            @printf(io, "%s,%s,%.1f,%.4f,%.4f,%.5f,%.5f,%.5f,%.4f,%d,%.4f,%.4f\n",
                    r.medium, r.homeostasis, r.amp, r.ρ_sub, r.θ, r.fire, r.fire_mean,
                    r.structure, r.t_half, r.censored, r.d_half, r.fid_end)
        end
    end

    sel = filter(k -> occursin("|potential|", k) && endswith(k, "|random"), collect(keys(curves)))
    p1 = plot(; xlabel = "step (carrier periods)", ylabel = "fidelity  similarity(decoded, injected)",
                title = "Symbol survival by code concentration", legend = :outertopright,
                legendfontsize = 5, ylims = (-0.2, 1.05), size = (980, 520))
    for k in sort(sel)
        m, _, e, _ = split(k, "|")
        plot!(p1, curves[k]; label = "$(m) · $(e)", lw = 1.4)
    end
    hline!(p1, [0.5]; ls = :dash, c = :black, label = "50%")
    dif = filter(r -> !r.censored && isfinite(r.t_half_pred) && r.medium != "no-coupling ctrl", rows)
    p2 = scatter([r.t_half_pred for r in dif], [r.t_half for r in dif];
                 xlabel = "t½ predicted (transport_forecast)", ylabel = "t½ measured",
                 title = "§4.6: spectrum predicts survival", legend = false,
                 ms = 4, size = (520, 520))
    mx = maximum(vcat([r.t_half_pred for r in dif], [r.t_half for r in dif]); init = 1f0)
    plot!(p2, [0, mx], [0, mx]; ls = :dash, c = :grey)
    savefig(plot(p1, p2; layout = @layout([a{0.62w} b]), size = (1500, 520)),
            joinpath(OUT, "transport_fidelity.png"))
    println("\nwrote $(joinpath(OUT, "transport_fidelity.csv")) and .png")
    return rows
end

main()
