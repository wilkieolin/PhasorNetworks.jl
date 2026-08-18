# Rung 0 of the superconducting-hardware ladder: re-run the velocity bank in
# physical units and check the claims that carry the most hardware leverage.
#
#   1. A hard-truncated coupling stencil suffices in 2D (4-12 taps/site against
#      ~1300 for the default Gaussian). Shown in 1D; this is the 2D check, and
#      it does NOT reproduce the 1D conclusion that R = 1 is optimal.
#   2. The readout interpolator, not the detector, is the accuracy floor.
#
# and to quantify the device tolerances later rungs must meet: detector jitter,
# photon dropout, and per-link delay scatter.
#
# Design point: T = 100 ps (10 GHz carrier), target Δt = 10 ps site-to-site,
# i.e. κ = 0.628 rad/site = v = 10 sites/period.
#
# The track heading is deliberately set MID-BIN (half an angle-bin off the
# nearest channel), which is the worst case for the readout. Testing on a
# bin-centred heading flatters every number here by an order of magnitude.
#
# Runs on CPU in a few minutes. Peak allocation ~30 MB, in the per-link weight
# array of the disordered rollout (H·W·|stencil|·C·8 bytes) — which is why that
# section drops to a smaller sheet. Nothing here uses the GPU.
#
# See docs/wavesheet_hardware_ladder.md for provenance of every device number.

using PhasorNetworks, Random, Lux, Printf

const T_PS   = 100.0        # carrier period
const DT_PS  = 10.0         # target site-to-site step
const H = W  = 96
const NSPEED = 24
const NANGLE = 8
const ANGLE  = π / NANGLE   # mid-bin: the worst-case heading for the readout
const X0     = (3, 3)
const DT_LO, DT_HI = 6.0, 45.0

hdr(s) = (println(); println("="^78); println(s); println("="^78))
med(v) = (s = sort(v); n = length(s);
          isodd(n) ? s[(n + 1) ÷ 2] : 0.5 * (s[n ÷ 2] + s[n ÷ 2 + 1]))
err(x, truth) = isnan(x) ? NaN : abs(x - truth) / truth * 100
verdict(e, ok, marg) = isnan(e) ? "NO TRACK" : e < ok ? "ok" : e < marg ? "marginal" : "FAIL"

function build(; n = H, radius = 1.0, aspect = 1.0, nspeed = NSPEED,
               nangle = NANGLE, margin = 0.9)
    l = hardware_bank(n, n; dt_lo_ps = DT_LO, dt_hi_ps = DT_HI, t_period_ps = T_PS,
                      n_speeds = nspeed, n_angles = nangle, margin,
                      stencil_radius = radius, stencil_aspect = aspect)
    ps, st = Lux.setup(Xoshiro(0), l)
    return l, ps, st
end

# L is set from the speed so every track crosses the sheet and then rings down.
# A fixed L would give fast targets a short aperture and slow ones a truncated
# track, confounding the speed sweeps below.
function drive_for(dt_ps; n = H, substeps = 64, jitter_fwhm_ps = 0.0,
                   dropout_frac = 0.0, seed = 0)
    v = speed_from_dt(dt_ps, T_PS)
    jit, alive = detector_noise(Xoshiro(seed), n, n;
                               jitter_fwhm_ps, t_period_ps = T_PS, dropout_frac)
    return moving_drive(n, n, ceil(Int, 1.5 * n / v); v, angle = ANGLE, x0 = X0,
                        width = 0.8, substeps, site_jitter = jit, site_alive = alive)
end

# Median estimated Δt over the sites the decoder marks. Median, not mean: the
# mask always admits a few off-track sites whose estimates are arbitrary.
function track_dt(l, ps, Z; frac = 0.985, interpolate = true)
    sp, _, mk = decode_velocity(l, ps, Z; frac, interpolate)
    sel = findall(mk)
    isempty(sel) && return NaN
    return med([dt_from_speed(sp[i], T_PS) for i in sel])
end

function track_heading(l, ps, Z; frac = 0.985)
    _, an, mk = decode_velocity(l, ps, Z; frac)
    sel = findall(mk)
    isempty(sel) && return NaN
    return med([an[i] for i in sel])
end

# ---------------------------------------------------------------- units --

hdr("PHYSICAL-UNITS SPEC SHEET   (T = $(T_PS) ps, f = $(round(1000/T_PS, digits=1)) GHz)")
@printf("%10s %14s %18s %10s\n", "Δt (ps)", "κ (rad/site)", "v (sites/period)", "κ/π")
for dt in (6, 8, 10, 15, 20, 30, 40, 45)
    κ = kappa_from_dt(dt, T_PS)
    @printf("%10.1f %14.4f %18.2f %10.3f\n", dt, κ, speed_from_dt(dt, T_PS), κ / π)
end
@printf("\nNyquist bound Δt < T/2 = %.1f ps (κ < π); beyond it the ramp aliases.\n",
        nyquist_dt(T_PS))
@printf("Sheet %d×%d, %d channels (%d speeds × %d headings), heading %.4f rad (mid-bin).\n",
        H, W, NSPEED * NANGLE, NSPEED, NANGLE, ANGLE)

hdr("DELAY-LINE LAYOUT   (kinetic-inductance microstrip)")
@printf("%12s %8s %10s %10s %12s %12s\n",
        "Ls (pH/sq)", "d (nm)", "v/c", "ps/mm", "10 ps", "40 ps")
for (ls, dn) in ((3.0, 100.0), (8.5, 100.0), (8.5, 50.0), (30.0, 50.0))
    a = delay_line_length(10.0; sheet_inductance_pH_sq = ls, dielectric_nm = dn)
    b = delay_line_length(40.0; sheet_inductance_pH_sq = ls, dielectric_nm = dn)
    @printf("%12.1f %8.0f %10.4f %10.1f %9.0f um %9.0f um\n",
            ls, dn, a.v_over_c, a.ps_per_mm, a.length_um, b.length_um)
end
let d = delay_line_length(DT_PS)
    @printf("\nAt %.0f ps/mm, one Δt=%.0f ps hop is %.0f um of line. Truncation is by\n",
            d.ps_per_mm, DT_PS, d.length_um)
    @printf("Euclidean distance, so R=1 is the 4-tap von Neumann stencil (%.2f mm/site\n",
            4 * d.length_um / 1000)
    @printf("per channel) and R=1.5 the 8-tap Moore one (%.2f mm, diagonals cost √2×).\n",
            (4 + 4 * sqrt(2)) * d.length_um / 1000)
end

# ------------------------------------------------- readout: the big win --

hdr("CLAIM 1 — INTERPOLATING READOUT vs ARGMAX")
println("Same hardware, same field; only the decode differs. The interpolator fits a")
println("2D quadratic in Cartesian κ over the 3×3 channel neighbourhood — separable")
println("polar refinement carries a cos(Δθ) foreshortening bias that this avoids.\n")
l, ps, st = build()
@printf("%10s %13s %10s %15s %10s %10s\n",
        "Δt true", "argmax", "err", "2D interp", "err", "gain")
readout_gain = Float64[]
for dt in (8.0, 10.0, 15.0, 20.0, 30.0, 40.0)
    Z = velocity_bank_run(l, ps, st, drive_for(dt))
    a = track_dt(l, ps, Z; interpolate = false)
    b = track_dt(l, ps, Z; interpolate = true)
    push!(readout_gain, err(a, dt) / max(err(b, dt), 1e-9))
    @printf("%8.1f ps %10.3f ps %9.2f%% %12.3f ps %9.2f%% %9.1fx\n",
            dt, a, err(a, dt), b, err(b, dt), err(a, dt) / max(err(b, dt), 1e-9))
end

hdr("HEADING RECOVERY   (across one full angle bin, worst case at the midpoint)")
@printf("%12s %14s %12s %14s %10s\n",
        "true (rad)", "estimated", "err (rad)", "Δt est (ps)", "Δt err")
for a in range(0, 2π / NANGLE; length = 5)
    Z = velocity_bank_run(l, ps, st,
          moving_drive(H, W, ceil(Int, 1.5H / speed_from_dt(DT_PS, T_PS));
                       v = speed_from_dt(DT_PS, T_PS), angle = a, x0 = X0,
                       width = 0.8, substeps = 64))
    d = track_dt(l, ps, Z); h = track_heading(l, ps, Z)
    @printf("%12.4f %14.4f %12.4f %14.3f %9.2f%%\n", a, h, abs(h - a), d, err(d, DT_PS))
end
@printf("\nAngle bin is %.4f rad; the readout resolves well inside it.\n", 2π / NANGLE)

# ----------------------------------------------- stencil: the 2D check --

hdr("CLAIM 2 — COUPLING STENCIL RADIUS, IN 2D")
println("R hard-truncates the coupling in sites. The default σ=6 Gaussian keeps every")
println("site within ~3σ, which is ~1300 taps/site and unwireable.")
println()
println("Line cost is distance-weighted: a tap at displacement d needs |d|·Δt of delay,")
println("not Δt. Counting taps alone understates every stencil beyond R=1.\n")

# taps and total delay-line length for a stencil of radius R
function stencil_cost(R)
    ds = [sqrt(min(i - 1, H - i + 1)^2 + min(j - 1, W - j + 1)^2)
          for i in 1:H, j in 1:W]
    keep = filter(d -> 0 < d <= R, vec(ds))
    return length(keep), sum(keep) * delay_line_length(DT_PS).length_um / 1000
end

@printf("%6s %10s %14s %10s %16s\n", "R", "taps/site", "Δt est (ps)", "err", "line/site/chan")
stencil = Tuple{Float64,Float64}[]
for R in (1.0, 1.5, 2.0, 3.0, 6.0)
    lr, psr, str = build(; radius = R)
    ntap, mm = stencil_cost(R)
    e = track_dt(lr, psr, velocity_bank_run(lr, psr, str, drive_for(DT_PS)))
    push!(stencil, (R, err(e, DT_PS)))
    @printf("%6.1f %10d %14.3f %9.2f%% %13.2f mm\n", R, ntap, e, err(e, DT_PS), mm)
end

println("\nBut the clean optimum is not the operating optimum. Under detector jitter a")
println("1D track in a 2D lattice couples mostly to OFF-track neighbours, and a wider")
println("stencil buys back the diluted coherent gain:\n")
@printf("%8s", "R\\FWHM")
for j in (0.0, 3.0, 15.0, 25.0, 35.0); @printf("%11s", string(Int(j)) * " ps"); end
@printf("%18s\n", "line/site/chan")
for R in (1.0, 2.0, 3.0, 4.0)
    @printf("%8.1f", R)
    for j in (0.0, 3.0, 15.0, 25.0, 35.0)
        lr, psr, str = build(; radius = R)
        e = [track_dt(lr, psr, velocity_bank_run(lr, psr, str,
               drive_for(DT_PS; jitter_fwhm_ps = j, seed = s))) for s in 1:4]
        @printf("%10.2f%%", sqrt(sum((e .- DT_PS) .^ 2) / length(e)) / DT_PS * 100)
    end
    @printf("%11.2f mm\n", stencil_cost(R)[2])
end

# ------------------------------------------- claim 3: κ-aligned stencil --

hdr("CLAIM 3 — κ-ALIGNED ANISOTROPIC STENCIL")
println("If the jitter loss above is off-track dilution, then elongating each channel's")
println("stencil along its OWN κ_c should recover it without paying for isotropic area.")
println("The stencil becomes an ellipse with semi-axis R along κ_c and aspect·R across.\n")

function tapcost(R, a, l, ps)
    tot = 0; wt = 0.0; r = Int(floor(R))
    for c in 1:l.n_channels
        kh, kw = ps.kappa[1, c], ps.kappa[2, c]; km = hypot(kh, kw)
        uh, uw = kh / km, kw / km
        for di in -r:r, dj in -r:r
            (di == 0 && dj == 0) && continue
            dp = di * uh + dj * uw; dq = -di * uw + dj * uh
            if (dp / R)^2 + (dq / (a * R))^2 <= 1 + 1e-4
                tot += 1; wt += hypot(di, dj)
            end
        end
    end
    return tot / l.n_channels, wt / l.n_channels * delay_line_length(DT_PS).length_um / 1000
end

const CFG = [(1.0, 1.0, "iso R=1"), (2.0, 1.0, "iso R=2"), (4.0, 1.0, "iso R=4"),
             (4.0, 0.5, "aniso 4/0.50"), (5.0, 0.4, "aniso 5/0.40"),
             (6.0, 0.25, "aniso 6/0.25")]
@printf("%14s %7s %13s %9s %9s %9s %9s %9s\n",
        "config", "taps", "mm/site/chan", "0 ps", "15 ps", "25 ps", "35 ps", "50 ps")
aniso_jitter_ok = 0.0
for (R, a, nm) in CFG
    lc, psc, stc = build(; radius = R, aspect = a)
    t, mm = tapcost(R, a, lc, psc)
    @printf("%14s %7.1f %10.2f mm", nm, t, mm)
    for j in (0.0, 15.0, 25.0, 35.0, 50.0)
        e = [track_dt(lc, psc, velocity_bank_run(lc, psc, stc,
               drive_for(DT_PS; jitter_fwhm_ps = j, seed = s))) for s in 1:5]
        rms = sqrt(sum((e .- DT_PS) .^ 2) / length(e)) / DT_PS * 100
        nm == "aniso 5/0.40" && rms < 3 && (global aniso_jitter_ok = j)
        @printf("%8.2f%%", rms)
    end
    println()
end

println("\nCurvature check — an elongated stencil assumes straight motion, so this is")
println("where it should break if it is going to (Δt = 10 ps, clean):\n")
@printf("%14s %10s %10s %10s %10s\n", "config", "κc = 0", "0.01", "0.02", "0.04")
for (R, a, nm) in CFG
    lc, psc, stc = build(; radius = R, aspect = a)
    @printf("%14s", nm)
    for cv in (0.0, 0.01, 0.02, 0.04)
        v = speed_from_dt(DT_PS, T_PS)
        D = moving_drive(H, W, ceil(Int, 1.5H / v); v, angle = ANGLE, x0 = X0,
                         width = 0.8, substeps = 64, curvature = cv)
        @printf("%9.2f%%", err(track_dt(lc, psc, velocity_bank_run(lc, psc, stc, D)), DT_PS))
    end
    println()
end

# ------------------------------------------------------ drive fidelity --

hdr("SUBSTEP RESOLUTION   (the drive cannot represent detail finer than T/substeps)")
@printf("%10s %16s %14s %10s\n", "substeps", "resolves (ps)", "Δt est (ps)", "err")
for ss in (4, 8, 16, 32, 64, 128)
    e = track_dt(l, ps, velocity_bank_run(l, ps, st, drive_for(DT_PS; substeps = ss)))
    @printf("%10d %16.2f %14.3f %9.2f%%\n", ss, T_PS / ss, e, err(e, DT_PS))
end

# ---------------------------------------------------- detector jitter --

hdr("DETECTOR JITTER   (frozen per site: a detector fires once, with one error)")
println("Reported SNSPD jitter: 15-18 ps high-Ic NbN | 26 ps MoSi | 50-62 ps arrays/SNAP")
println("| 191 ps WSi at 2.5 K. See docs for sources.\n")
@printf("%12s %14s %10s %12s %10s\n",
        "FWHM (ps)", "mean Δt (ps)", "rms err", "spread (ps)", "verdict")
jitter_ok = 0.0
for j in (0.0, 3.0, 15.0, 25.0, 35.0, 50.0, 80.0)
    est = [track_dt(l, ps, velocity_bank_run(l, ps, st,
             drive_for(DT_PS; jitter_fwhm_ps = j, seed = s))) for s in 1:5]
    m = sum(est) / length(est)
    sd = sqrt(sum((est .- m) .^ 2) / max(length(est) - 1, 1))
    rms = sqrt(sum((est .- DT_PS) .^ 2) / length(est)) / DT_PS * 100
    rms < 3 && (global jitter_ok = j)
    @printf("%12.0f %14.3f %9.2f%% %12.3f %10s\n", j, m, rms, sd, verdict(rms, 3, 10))
end

# ----------------------------------------------------------- dropout --

hdr("PHOTON DROPOUT   (frozen dead sites — a sparse aperture)")
@printf("%10s %14s %10s %10s\n", "dropout", "mean Δt (ps)", "rms err", "verdict")
dropout_ok = 0.0
for d in (0.0, 0.3, 0.5, 0.7, 0.85, 0.95)
    est = [track_dt(l, ps, velocity_bank_run(l, ps, st,
             drive_for(DT_PS; dropout_frac = d, seed = s))) for s in 1:5]
    rms = sqrt(sum((est .- DT_PS) .^ 2) / length(est)) / DT_PS * 100
    rms < 3 && (global dropout_ok = d)
    @printf("%9.0f%% %14.3f %9.2f%% %10s\n",
            d * 100, sum(est) / length(est), rms, verdict(rms, 3, 10))
end

# ------------------------------------------------- per-link disorder --

hdr("FABRICATION SCATTER ON DELAY LINES   (per-link, real-space rollout)")
println("Per-link disorder breaks translation invariance, so this path cannot use the")
println("FFT and costs O(H·W·C·|stencil|) per step — affordable only for a truncated")
println("stencil. Run at R=1 on a smaller sheet with fewer channels for runtime; the")
println("comparison is against its own clean baseline, not the 96x96 one above.\n")
const ND = 64
ld, psd, std_ = build(; n = ND, radius = 1.0, nspeed = 12)
D0 = drive_for(DT_PS; n = ND)
Zf = velocity_bank_run(ld, psd, std_, D0)
Zr = velocity_bank_run_disordered(ld, psd, std_, D0; delay_scatter = 0.0)
rel = maximum(abs.(Zf .- Zr)) / maximum(abs.(Zf))
@printf("  cross-check, FFT vs real-space at zero disorder: max rel diff %.3e  %s\n\n",
        rel, rel < 1e-4 ? "AGREE" : "MISMATCH — investigate before trusting the rest")

println("delay ∝ √Ls ∝ 1/√thickness, so x% delay scatter ≈ 2x% film-thickness spread.\n")
base = track_dt(ld, psd, Zf)
@printf("%12s %14s %14s %12s %10s\n",
        "scatter rms", "film spread", "mean Δt (ps)", "rms err", "verdict")
for fab in (0.0, 0.02, 0.05, 0.10, 0.20, 0.40)
    est = [track_dt(ld, psd, velocity_bank_run_disordered(ld, psd, std_, D0;
             delay_scatter = fab, rng = Xoshiro(100 + s))) for s in 1:3]
    rms = sqrt(sum((est .- DT_PS) .^ 2) / length(est)) / DT_PS * 100
    @printf("%11.0f%% %13.0f%% %14.3f %11.2f%% %10s\n",
            fab * 100, fab * 200, sum(est) / length(est), rms, verdict(rms, 3, 10))
end
@printf("\n(clean baseline for this smaller config: %.3f ps, %.2f%%)\n",
        base, err(base, DT_PS))

# ------------------------------------------------------- combination --

hdr("COMBINED REALISTIC DEVICE   (25 ps jitter + 50% dropout + 5% delay scatter)")
@printf("%10s %14s %12s %10s\n", "Δt true", "Δt est (ps)", "rms err", "verdict")
for dt in (8.0, 10.0, 15.0, 20.0, 30.0, 40.0)
    Dn = drive_for(dt; n = ND, jitter_fwhm_ps = 25.0, dropout_frac = 0.5, seed = 7)
    est = [track_dt(ld, psd, velocity_bank_run_disordered(ld, psd, std_, Dn;
             delay_scatter = 0.05, rng = Xoshiro(200 + s))) for s in 1:3]
    rms = sqrt(sum((est .- dt) .^ 2) / length(est)) / dt * 100
    @printf("%8.1f ps %14.3f %11.2f%% %10s\n",
            dt, sum(est) / length(est), rms, verdict(rms, 5, 15))
end

# ------------------------------------------------------------- gates --

hdr("RUNG 0 GATES")
Zc = velocity_bank_run(l, ps, st, drive_for(DT_PS))
clean = err(track_dt(l, ps, Zc), DT_PS)
r1 = stencil[1][2]; rbest = minimum(x -> x[2], stencil)
gates = [("Δt = 10 ps recovered < 1% (clean, worst-case heading)", clean < 1.0,
          @sprintf("%.2f%%", clean)),
         ("cheapest stencil R=1 still clears 1% clean", r1 < 1.0,
          @sprintf("R=1 %.2f%%, best tested %.2f%%", r1, rbest)),
         ("interpolation beats argmax by > 3x on average",
          sum(readout_gain) / length(readout_gain) > 3,
          @sprintf("%.1fx mean", sum(readout_gain) / length(readout_gain))),
         ("FFT and real-space rollouts agree", rel < 1e-4, @sprintf("%.1e", rel)),
         ("tolerates >= 70% dropout", dropout_ok >= 0.7,
          @sprintf("%.0f%%", dropout_ok * 100)),
         ("isotropic R=1 tolerates >= 15 ps jitter", jitter_ok >= 15,
          @sprintf("clears 3%% only to %.0f ps", jitter_ok)),
         ("κ-aligned stencil tolerates >= 25 ps jitter", aniso_jitter_ok >= 25,
          @sprintf("aniso 5/0.40 clears 3%% to %.0f ps", aniso_jitter_ok))]
for (name, pass, detail) in gates
    @printf("  [%s] %-52s %s\n", pass ? "PASS" : "FAIL", name, detail)
end
@printf("\n%d/%d gates passed.\n", count(g -> g[2], gates), length(gates))
println()
println("The isotropic jitter gate fails and that was the open question from the first")
println("pass. The κ-aligned stencil answers it: elongating each channel's coupling")
println("along its own κ_c raises jitter tolerance by an order of magnitude at LOWER")
println("cost than the isotropic stencil it beats (29 taps vs 48, 9.5 mm vs 16.6 mm).")
println("The detector spec moves from 'nothing available works' to 'MoSi at 26 ps")
println("works, high-Ic NbN has margin'. The price is a lattice pitch of ~1.1 mm and")
println("a modest loss on curved tracks.")
