# Rung 1 of the superconducting-hardware ladder: device realism.
#
# Rung 0 characterised the method and left two gates open. This rung closes
# both, and finds a third thing on the way.
#
#   1. CURVATURE x JITTER. Rung 0 swept each alone. The κ-aligned stencil
#      assumes straight motion, so their combination was the named risk.
#   2. FABRICATION YIELD. Delay scatter, resonator-Q spread and coupling-gain
#      error drawn per device; what fraction of devices meets spec.
#   3. ENVELOPE WIDTH. Investigated here and largely a dead end: σ matters far
#      less than expected, and the "σ ≈ 0.75R" rule this study first inferred
#      does not generalise. Kept because the negative result is worth having.
#
# A warning encoded below as `max_curvature_for`: an arc must not lap its own
# path. Past one circumference the particle re-deposits at a different carrier
# phase, which destroys the ramp for ANY stencil. An earlier version of this
# study ran 150 sites of track around a 105-site circle and read the resulting
# 269% error as a curvature limit. It is not one.
#
# CPU, a few minutes. See docs/wavesheet_hardware_ladder.md for provenance.

using PhasorNetworks, Random, Lux, Printf

const T_PS  = 100.0
const DT_PS = 10.0
const ANG   = π / 8          # mid-bin: worst case for the readout
const SPEC  = 5.0            # a device passes at < 5% speed error

hdr(s) = (println(); println("="^78); println(s); println("="^78))
med(x) = (s = sort(x); n = length(s);
          isodd(n) ? s[(n + 1) ÷ 2] : 0.5 * (s[n ÷ 2] + s[n ÷ 2 + 1]))

bank(n, R, a, σ; ns = 24, na = 8) =
    hardware_bank(n, n; dt_lo_ps = 6, dt_hi_ps = 45, t_period_ps = T_PS,
                  n_speeds = ns, n_angles = na, margin = 0.9,
                  stencil_radius = R, stencil_aspect = a, init_log_sigma = log(σ))

# Longest run that keeps the arc under one lap. Without this the study measures
# self-overlap and calls it curvature.
max_curvature_for(cv, v, n) =
    max(3, ceil(Int, (cv == 0 ? 1.5n : min(1.5n, 0.8 * 2π / cv)) / v))

function trial(n, R, a, σ, cv, jit, seed; x0 = (20, 20), ds = 0.0, ls = 0.0, ge = 0.0)
    l = bank(n, R, a, σ); ps, st = Lux.setup(Xoshiro(0), l)
    v = speed_from_dt(DT_PS, T_PS); L = max_curvature_for(cv, v, n)
    sj, al = detector_noise(Xoshiro(1000 + seed), n, n;
                            jitter_fwhm_ps = jit, t_period_ps = T_PS)
    D = moving_drive(n, n, L; v, angle = ANG, x0, width = 0.8, substeps = 64,
                     curvature = cv, site_jitter = sj, site_alive = al)
    Z = (ds == 0 && ls == 0 && ge == 0) ?
        velocity_bank_run(l, ps, st, D) :
        velocity_bank_run_disordered(l, ps, st, D; delay_scatter = ds,
            loss_spread = ls, gain_error = ge, rng = Xoshiro(seed))
    any(!isfinite, Z) && return (Inf, Inf)
    sp, an, mk = decode_velocity(l, ps, Z; frac = 0.99)
    sel = findall(mk); isempty(sel) && return (Inf, Inf)
    se = abs(med([dt_from_speed(sp[i], T_PS) for i in sel]) - DT_PS) / DT_PS * 100
    # heading measured against the TRUE local tangent, which rotates along an arc
    hs = Float64[]
    for I in sel
        best = Inf; bt = 0.0
        for s in 0:0.5:(v * L)
            ah = cv == 0 ? s : sin(cv * s) / cv
            aw = cv == 0 ? 0.0 : (1 - cos(cv * s)) / cv
            ph = x0[1] + cos(ANG) * ah - sin(ANG) * aw
            pw = x0[2] + sin(ANG) * ah + cos(ANG) * aw
            d = hypot(I[1] - 1 - ph, I[2] - 1 - pw)
            d < best && (best = d; bt = ANG + cv * s)
        end
        best <= 2.0 && push!(hs, abs(rem(an[I] - bt, 2π, RoundNearest)))
    end
    return (se, isempty(hs) ? NaN : med(hs))
end

# ------------------------------------------------------ the lap warning --

hdr("WHY ARCS MUST BE CAPPED")
@printf("%8s %10s %15s %12s %8s\n", "κc", "radius", "circumference", "path run", "laps")
for cv in (0.01, 0.02, 0.04, 0.06, 0.10)
    circ = 2π / cv; path = 1.5 * 96
    @printf("%8.2f %8.0f s %13.0f s %10.0f s %7.2f  %s\n",
            cv, 1 / cv, circ, path, path / circ, path > circ ? "<-- would self-overlap" : "ok")
end
println("\nEvery arc below is capped at 0.8 laps. Uncapped, all stencils show a false")
println("cliff at κc = 0.06 (269% error even with no jitter and no anisotropy).")

# -------------------------------------------------- gate 1: curvature --

hdr("GATE 1 — CURVATURE x DETECTOR JITTER  (speed err % / heading err rad)")
println("κc is 1/radius in sites. Stencil half-length is 4-5 sites throughout.\n")
CFG = [(1.0, 1.0, 6.0, "iso R=1"), (4.0, 0.5, 3.0, "aniso 4/0.50"),
       (5.0, 0.4, 3.0, "aniso 5/0.40")]
curv_worst = 0.0
for (R, a, σ, nm) in CFG
    @printf("%14s", nm)
    for j in (0.0, 15.0, 25.0); @printf("%20s", string(Int(j)) * " ps"); end
    println()
    for cv in (0.0, 0.02, 0.04, 0.06, 0.10)
        @printf("  κc=%.2f r=%5s", cv, cv == 0 ? "inf" : string(round(Int, 1 / cv)))
        for j in (0.0, 15.0, 25.0)
            r = [trial(96, R, a, σ, cv, j, s) for s in 1:3]
            se = sum(x -> x[1], r) / 3; he = sum(x -> x[2], r) / 3
            startswith(nm, "aniso") && (global curv_worst = max(curv_worst, se))
            @printf("  %8.2f%% /%6.3f", se, he)
        end
        println()
    end
    println()
end

# ------------------------------------------------- envelope width find --

hdr("ENVELOPE WIDTH — mostly a dead end")
println("σ=6 is flat across an R=4 stencil, so taps enter at full weight rather than")
println("tapering, and tying σ to R ought to help. It does, but only for this one")
println("config and only in the clean column; under jitter the default is a wash or")
println("marginally better, and at R=1 σ has NO effect at all (one tap distance, so")
println("the envelope is a scalar that g_crit normalises away).\n")
@printf("%16s %6s %8s %13s %13s %13s\n", "config", "σ", "σ/R", "spd clean", "spd 25 ps", "hdg clean")
for (R, a, nm) in ((4.0, 0.5, "aniso 4/0.50"),)
    for σ in (1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
        c = [trial(96, R, a, σ, 0.0, 0.0, s; x0 = (3, 3)) for s in 1:2]
        n = [trial(96, R, a, σ, 0.0, 25.0, s; x0 = (3, 3)) for s in 1:3]
        @printf("%16s %6.1f %8.2f %12.2f%% %12.2f%% %13.4f\n", nm, σ, σ / R,
                sum(x -> x[1], c) / 2, sum(x -> x[1], n) / 3, sum(x -> x[2], c) / 2)
    end
end
println("\nAcross all six configs (see docs §2.5) the σ=0.75R and σ=6 columns differ by")
println("well under the seed-to-seed spread except for aniso 4/0.50 clean, where 3 beats")
println("6 by ~2x. An earlier pass generalised that single cell into a design rule; it")
println("does not hold. Leave σ at the default unless profiling this exact config.")

# ------------------------------------------- where inside the bin is worst --

hdr("WORST CASE IS NOT THE SAME PLACE FOR SPEED AND HEADING")
println("Everything above runs at exactly half a bin off-channel (π/8), on the")
println("assumption that is the worst case. It is — for SPEED. For HEADING the")
println("half-bin point is symmetric between two channels and therefore easy; the")
println("damage is at the quarter-bin offsets, where the fit is asymmetric.\n")
@printf("%12s %10s %14s %14s %14s\n",
        "heading", "frac of bin", "spd clean", "spd 25 ps", "hdg clean")
hdg_worst = 0.0
let R = 4.0, a = 0.5, σ = 3.0, dθ = 2π / 8
    for f in (0.0, 0.25, 0.5, 0.75, 1.0)
        ag = f * dθ
        l = bank(96, R, a, σ); ps, st = Lux.setup(Xoshiro(0), l)
        v = speed_from_dt(DT_PS, T_PS)
        function one(j, s)
            sj, al = detector_noise(Xoshiro(1000 + s), 96, 96;
                                    jitter_fwhm_ps = j, t_period_ps = T_PS)
            D = moving_drive(96, 96, ceil(Int, 1.5 * 96 / v); v, angle = ag, x0 = (3, 3),
                             width = 0.8, substeps = 64, site_jitter = sj, site_alive = al)
            sp, an, mk = decode_velocity(l, ps, velocity_bank_run(l, ps, st, D); frac = 0.99)
            sel = findall(mk)
            (abs(med([dt_from_speed(sp[i], T_PS) for i in sel]) - DT_PS) / DT_PS * 100,
             med([abs(rem(an[i] - ag, 2π, RoundNearest)) for i in sel]))
        end
        c = one(0.0, 1); n = [one(25.0, s) for s in 1:3]
        global hdg_worst = max(hdg_worst, c[2])
        @printf("%12.4f %10.2f %13.2f%% %13.2f%% %14.4f\n",
                ag, f, c[1], sum(x -> x[1], n) / 3, c[2])
    end
end
@printf("\nWorst clean heading error anywhere in the bin: %.4f rad (%.1f deg).\n",
        hdg_worst, hdg_worst * 180 / π)

# ------------------------------------------------------- gate 2: yield --

hdr("GATE 2 — FABRICATION YIELD  (device passes at < $(SPEC)% speed error)")
println("Three defects drawn per device: per-link delay scatter, per-site resonator-Q")
println("spread, and a per-device coupling-gain error. 24 devices per row, 64x64.\n")
const NDEV = 24
@printf("%11s %11s %10s %9s %12s %8s\n",
        "delay scat", "Q spread", "gain err", "jitter", "median err", "yield")
yield10 = 0.0; yield_real = 0.0
for (ds, ls, ge, jt) in ((0.0, 0.0, 0.0, 0.0), (0.05, 0.0, 0.0, 0.0),
                         (0.10, 0.0, 0.0, 0.0), (0.20, 0.0, 0.0, 0.0),
                         (0.0, 0.30, 0.0, 0.0), (0.0, 0.0, 0.10, 0.0),
                         (0.10, 0.10, 0.05, 0.0), (0.10, 0.10, 0.05, 15.0),
                         (0.10, 0.10, 0.05, 25.0), (0.05, 0.05, 0.03, 25.0))
    e = [trial(64, 4.0, 0.5, 3.0, 0.0, jt, s; x0 = (3, 3), ds, ls, ge)[1] for s in 1:NDEV]
    y = count(<(SPEC), e) / NDEV * 100
    (ds == 0.10 && ls == 0 && ge == 0 && jt == 0) && (global yield10 = y)
    (ds == 0.10 && ls == 0.10 && ge == 0.05 && jt == 25.0) && (global yield_real = y)
    @printf("%10.0f%% %10.0f%% %9.0f%% %7.0fps %11.2f%% %7.0f%%\n",
            ds * 100, ls * 100, ge * 100, jt, med(filter(isfinite, e)), y)
end

# ------------------------------------------------------------- gates --

hdr("RUNG 1 GATES")
gates = [("curvature x jitter, aniso, to radius 10 sites", curv_worst < 5.0,
          @sprintf("worst %.2f%%", curv_worst)),
         ("yield >= 80% at 10% delay scatter", yield10 >= 80,
          @sprintf("%.0f%%", yield10)),
         ("yield >= 80% on a fully realistic device", yield_real >= 80,
          @sprintf("%.0f%% at 10/10/5%% + 25 ps", yield_real))]
for (n, p, d) in gates
    @printf("  [%s] %-48s %s\n", p ? "PASS" : "FAIL", n, d)
end
@printf("\n%d/%d gates passed.\n", count(g -> g[2], gates), length(gates))
println()
println("Curvature turned out NOT to interact with jitter. The reason is structural:")
println("the readout is per-site and the stencil is only 4-5 sites long, so it sees a")
println("locally straight chord as long as its length is short against the radius.")
println("Design rule: stencil half-length <= radius/2.")
println()
println("The open item is now HEADING, not speed. Its error depends on where inside the")
@printf("angle bin the track falls — best at bin centre and at the half-bin symmetry\n")
@printf("point, worst at the quarter offsets (%.3f rad clean, above). Under 25 ps of\n", hdg_worst)
println("jitter it sits near 0.10 rad. Widening the envelope does not improve it, and")
println("isotropic is not a fix: it reaches 0.001 rad clean but degrades to 0.30 rad at")
println("25 ps, so a hybrid bank would buy nothing in the regime that matters.")
println()
println("Not established: WHY heading is capped this way. The natural explanation is")
println("Fourier reciprocity — a stencil elongated along the track is narrow in q-space")
println("radially and wide tangentially, i.e. sharp in speed and blunt in heading. An")
println("attempt to confirm it by measuring channel half-widths failed: the responses")
println("are broader than the channel grid, so the probe saturated. Treat the")
println("explanation as reasoning, not measurement.")
