# device_velocity_readout.jl — particle speed from the PERIMETER of a Mach sheet,
# against two-point time-of-flight and a fully instrumented track.
#
#     julia -t 16 --project=scripts demos/device_velocity_readout.jl
#     MODE=multi julia -t 16 --project=scripts demos/device_velocity_readout.jl
#
# THE QUESTION. A two-point TOF (detectors d = 10 cm apart, v = d/Δt) measures
# one speed along a trajectory it must already know, from one time difference.
# A sheet of SNSPD pixels feeding an R&F transponder lattice (DeviceSheet) turns
# any track into a Mach cone whose arrival sweeps along the sheet's side edges
# at exactly the particle's speed: for a planar cone front,
#
#     t_edge(s) = t₀ + s/v + β·ρ_edge,      β = √(1/u² − 1/v²)
#
# so a straight-line fit of first-spike time against position along the edge
# returns v, from 2N edge channels and for whatever track crossed the sheet. No
# pixel needs its own TDC; the lattice does the routing and smooths the
# detector jitter on the way out.
#
# THREE READOUTS of the same jittered particle (frozen per-site jitter, the
# `detector_noise` convention), over a 10 cm aperture:
#   2-pt TOF     — detectors at the two ends of the 10 cm baseline. Analytic:
#                  σv/v = √2·σt/Δt.
#   track array  — every hit pixel on the track timed by its own TDC, line fit.
#                  An upper bound: realising it for an arbitrary track needs a
#                  TDC on all N² pixels.
#   sheet edges  — first spikes on the two side columns only, line fit, with
#                  the lattice's deterministic bias (measured with zero jitter)
#                  calibrated out.
#
# MODES.
#   single — the default DeviceSheet: 10 ps of line per pitch, u ≈ 0.0206c.
#            Particles below u are subsonic and make no cone.
#   multi  — two interleaved media on a checkerboard. Each pixel feeds the
#            sublattice of its colour; links join same-colour sites only
#            (`G = 0` across colours), so the two are independent media
#            sharing one pixel plane. The slow one covers the bottom of the
#            range, where the fast one is subsonic. It is the fast medium
#            TIME-SCALED by s = 3 (`time_scaled`: 3.33 GHz tanks, 3× pulse,
#            refractory and line delay), an exact symmetry, so it is the same
#            excitable medium at u/3. Longer lines alone cannot do this: a
#            10 GHz cell's inputs then arrive spread over most of a carrier
#            period and cancel (a 50 ps/pitch sublattice does not propagate).
#            Because the sublattices interleave everywhere, this works for any
#            heading, and a subsonic fast front running ahead of the particle
#            cannot leak into the slow medium (as it would across a boundary
#            between side-by-side regions).
#
# Edge points are kept only where the edge is on the cone and not on the
# entry circle: s ≥ ρ_edge·tan α + margin, with α estimated in a first pass.
#
# Cost (800², K = 28): links 215 MB + history (pps·3/dt + 3)·2.6 MB; no
# frames are kept. The slowest runs (0.01c on the slow medium) take ~55k steps.

using PhasorNetworks, Printf, Statistics, Random
include(joinpath(@__DIR__, "..", "scripts", "gpu_backend.jl"))

const MODE   = Symbol(get(ENV, "MODE", "single"))
const N      = parse(Int, get(ENV, "VR_N", "800"))
const BASE   = parse(Int, get(ENV, "VR_BASE", "775"))         # track sites = 10 cm at 129 µm
const SEEDS  = parse(Int, get(ENV, "VR_SEEDS", "6"))
const JIT    = parse(Float64, get(ENV, "VR_JITTER", "25"))    # FWHM, ps
const WIDTH  = parse(Float64, get(ENV, "VR_WIDTH", "3"))
const VLIST  = parse.(Float64, split(get(ENV, "VR_V", MODE === :multi ?
                     "0.01,0.015,0.0206,0.043,0.1,0.3,1.0" : "0.043,0.1,0.3,1.0"), ","))
const PPS_FAST = parse(Float64, get(ENV, "VR_PPS_FAST", "10"))
const SLOW_S   = parse(Float64, get(ENV, "VR_SLOW_S", "3"))     # slow medium = time-scaled ×s
const AMP      = MODE === :multi ? 80.0 : 40.0                  # SNSPD pulse, µA (slow cells need more charge)
const G_SUB    = parse(Float64, get(ENV, "VR_G_SUB", "0.45"))
const ROW0   = 13
const P_UM   = 129.0                                          # pitch: 10 ps of NbTiN line
const C_SITES = 299.792458 / P_UM                             # c in sites/ps
const σT     = JIT / 2.355

backend, dev = load_gpu_backend()
cell = RFTransponder()

# ---- sheets ---------------------------------------------------------------
color(y, x) = iseven(y + x) ? 1 : 2                           # 1 = slow, 2 = fast (multi)

function build_sheet()
    MODE === :single && return DeviceSheet(cell, N, N; ps_per_site = PPS_FAST), [trues(N, N)], ["single"]
    slow = time_scaled(cell, SLOW_S)
    cells = [color(y, x) == 1 ? slow : cell for y in 1:N, x in 1:N]
    offs = stencil_offsets(3)
    K = length(offs)
    D = Array{Float32}(undef, N, N, K); G = zeros(Float32, N, N, K)
    for (k, (dy, dx)) in enumerate(offs), x in 1:N, y in 1:N
        r = sqrt(dy^2 + dx^2)
        D[y, x, k] = (color(y, x) == 1 ? SLOW_S * PPS_FAST : PPS_FAST) * r
        iseven(dy + dx) && (G[y, x, k] = G_SUB)              # same-colour links only
    end
    masks = [[color(y, x) == c for y in 1:N, x in 1:N] for c in 1:2]
    return DeviceSheet(cells, N, N; D, G), masks, ["slow", "fast"]
end
sheet, masks, names = build_sheet()
println(sheet)

# ---- [1] medium speeds, measured independently ------------------------------
println("\n[1] medium front speeds (radial seed, no particle)")
cen = (N ÷ 2, N ÷ 2)
rad = sheet_simulate(sheet, MODE === :multi ? 30000 : 9000; drive = seed_drive(N, N; radius = 3, amp = AMP),
                     device = dev)
U = map(eachindex(masks)) do c
    tf = ifelse.(masks[c], rad.t_first, Inf32)
    u = arrival_speed(tf; center = cen, rmin = 20, rmax = 150)[1]
    an = front_anisotropy(tf; center = cen, rmin = 20, rmax = 150)
    @printf("    %-6s u = %.5f sites/ps = %.4fc   diagonal/axis %.3f   fired %.1f%%\n",
            names[c], u, u / C_SITES, an.ratio, 100 * count(isfinite, tf) / count(masks[c]))
    flush(stdout)
    u
end

# ---- readouts ----------------------------------------------------------------
lin(x, y) = (mx = mean(x); my = mean(y);
             b = sum((x .- mx) .* (y .- my)) / sum((x .- mx) .^ 2); (b, my - b * mx))

function edge_speed(tf, mask, u)
    ρe = N ÷ 2 - 1
    pts(smin) = begin
        s_e = Float64[]; t_e = Float64[]
        for col in (1, N), y in 1:N
            s = y - ROW0; t = tf[y, col]
            (mask[y, col] && isfinite(t) && smin <= s <= BASE) || continue
            push!(s_e, s); push!(t_e, t)
        end
        s_e, t_e
    end
    s1, t1 = pts(BASE ÷ 2)                                    # pass 1: back half only
    length(s1) < 20 && return NaN
    v1 = 1 / lin(s1, t1)[1]
    v1 > 1.05u || return NaN                                  # subsonic: the edge sees the medium, not v
    smin = ρe * tan(asin(u / v1)) + 25
    smin < BASE - 60 || return NaN                            # cone never reaches the edge in the aperture
    s2, t2 = pts(round(Int, smin))
    length(s2) < 20 && return NaN
    return 1 / lin(s2, t2)[1]
end

function run_particle(v, seed, jit)
    dts = 1 / v
    d = particle_drive(N, N; dt_site = dts, x0 = (ROW0, N ÷ 2), width = WIDTH, amp = AMP,
                       jitter_fwhm_ps = jit, rng = Xoshiro(seed))
    d.t_hit[ROW0+BASE+1:end, :] .= Inf32                      # aperture ends at 10 cm
    s_h = Float64[]; t_h = Float64[]
    for x in 1:N, y in 1:N
        t = d.t_hit[y, x]; isfinite(t) && (push!(s_h, y - ROW0); push!(t_h, t))
    end
    v_track = 1 / lin(s_h, t_h)[1]
    umin = minimum(U)
    nst = round(Int, BASE * dts + (N ÷ 2) * sqrt(max(1 / umin^2 - 1 / v^2, 1 / umin^2 / 4)) + 1500)
    tr = sheet_simulate(sheet, nst; drive = d, device = dev)
    v_edge = [edge_speed(tr.t_first, masks[c], U[c]) for c in eachindex(masks)]
    return v_track, v_edge, nst
end

println("\n[2] jitter $(JIT) ps FWHM, track width $(WIDTH), aperture $(BASE) sites = $(round(BASE*P_UM/1e4; digits=1)) cm, $(SEEDS) seeds")
nl = length(masks)
hdr = join([@sprintf("%-26s", "edges[$(names[c])] σ / bias") for c in 1:nl], " | ")
@printf("%7s %8s | %-9s | %-11s | %s\n", "v/c", "Mach", "2-pt TOF", "track array", hdr)
results = []
for f in VLIST
    v = f * C_SITES
    _, cal, _ = run_particle(v, 0, 0.0)                       # zero jitter: lattice bias
    bias = [isfinite(cal[c]) ? cal[c] / v - 1 : NaN for c in 1:nl]
    vt = Float64[]; ve = [Float64[] for _ in 1:nl]; t0 = time()
    for seed in 1:SEEDS
        a, b, _ = run_particle(v, seed, JIT)
        push!(vt, a)
        for c in 1:nl
            push!(ve[c], b[c] / (1 + (isfinite(bias[c]) ? bias[c] : 0.0)))
        end
    end
    tof = sqrt(2) * σT / (BASE / v)
    cols = map(1:nl) do c
        ok = filter(isfinite, ve[c])
        length(ok) < max(2, SEEDS ÷ 2) ? @sprintf("%-26s", "— no cone (M=$(round(v/U[c]; digits=2)))") :
            @sprintf("%5.2f%%  cal %+5.2f%% → %+5.2f%%", 100std(ok) / v, 100bias[c], 100(mean(ok) / v - 1))
    end
    @printf("%7.4f %8s | %8.2f%% | %10.2f%% | %s   (%.0f s)\n", f,
            join([@sprintf("%.1f", v / U[c]) for c in 1:nl], "/"), 100tof, 100std(vt) / v,
            join(cols, " | "), time() - t0)
    flush(stdout)
    push!(results, (; f, tof, track = std(vt) / v, edges = [std(filter(isfinite, ve[c])) / v for c in 1:nl], bias))
end
