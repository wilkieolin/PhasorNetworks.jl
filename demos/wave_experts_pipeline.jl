# wave_experts_pipeline.jl — Rung 3(B): a wave-PROGRAMMED serial pipeline.
#
# The payoff of the whole dispersion analysis. On the ISOTROPIC sheet a payload
# spreads every way at once, so a spatial sequence of experts can't process it in
# order (§9 limitation). Here we run the sheet as the analytically-derived
# LOW-DISPERSION CONVEYOR (`:aniso`, gentle Mexican hat at marginality, DC carrier,
# advection drift β): a smooth payload packet injected at the top drifts COHERENTLY
# downward through E expert bands. Each ACTIVE expert stamps its bind V_e onto the
# passing packet; at the bottom the composition Σ_{active} V_e is recovered by a
# differential unbind (bound run − no-bind control cancels the propagation phase).
#
# VERIFY: (1) band-visit order 1→E (directed path); (2) recovered ≈ Σ active V_e;
# (3) causal (drop e removes V_e); (4) CONTRAST: isotropic :dog cannot pipeline.
#
# Run:  julia --project=. demos/wave_experts_pipeline.jl
#
using PhasorNetworks, Lux, Statistics, Printf
using Random: Xoshiro

# A 2-STAGE pipeline. Ballistic low-dispersion transport spans ~2 bands before the
# drift-vs-dispersion tradeoff (drift ∝ gβ, but low-dispersion + stability push g→0)
# lets the packet spread — so E=2 is the honest reach of this coupling. Scaling to
# E≥3 needs a shift-like (not central-difference) advection; see the closing note.
const H = 20; const W = 12; const E = 2
const BH = H ÷ E                                   # rows per band (=10)
const WIN = 3                                      # readout/bind half-window around the packet
const MAXSTEPS = 110
band_rows(e) = ((e - 1) * BH + 1):(e * BH)
band_of(row) = clamp(Int(cld(round(Int, row), BH)), 1, E)
const V = Float32[0.3, -0.5]                       # per-expert transforms (units of π)
_wrap(x) = x - 2f0 * round(x / 2f0)

mksheet(coupling; beta) = PhasorWaveSheet(H, W; coupling = coupling, transmit = :potential,
    init_A_exc = 1.0, init_log_sigma_exc = log(1.0), init_B_inh = 0.1, init_log_sigma_inh = log(1.6),
    init_beta_h = beta, init_log_neg_lambda = log(0.1), init_log_speed = log(40.0))

function tune_g!(l, ps, st; target = -0.03f0)      # marginality via dispersion diagnostics
    gstar(lg) = dispersion_diagnostics(l, merge(ps, (; log_g = Float32[lg])), st; mode = :potential).growth_star
    lo, hi = log(1f-5), log(20f0)
    for _ in 1:44
        m = (lo + hi) / 2
        if gstar(m) < target; lo = m; else; hi = m; end
    end
    return merge(ps, (; log_g = Float32[(lo + hi) / 2]))
end

crow(z) = (w = abs2.(z); sum((1:H) .* vec(sum(w; dims = 2))) / (sum(w) + 1f-12))
winrows(c) = max(1, round(Int, c) - WIN):min(H, round(Int, c) + WIN)
packet_phase(z, c) = angle(sum(@view z[winrows(c), :])) / Float32(pi)   # windowed on the packet

inject(φ0) = ComplexF32[exp(-((i - 3)^2 + (j - W ÷ 2)^2) / (2f0 * 3f0^2)) * cis(Float32(pi) * φ0)
                        for i in 1:H, j in 1:W]

# Step the conveyor one cycle at a time (coupling precomputed). When the packet
# centroid first enters band e, an active expert stamps V_e onto the packet window.
# Stop once the packet has reached band E (avoids torus wrap). Returns final packet
# phase + band-visit order.
function run_pipeline(l, ps, st, program::Set{Int}; φ0 = 0.3f0)
    ω = PhasorNetworks.period_to_angfreq(l.spk_args.t_period)
    A_step, g, W_hat = PhasorNetworks._build_coupling(l, ps, st, ω)
    z = reshape(inject(φ0), H, W, 1)
    drive0 = zeros(ComplexF32, H, W, 1)
    visited = Int[]; bound = Set{Int}(); c = crow(z[:, :, 1])
    for t in 1:MAXSTEPS
        z, _ = PhasorNetworks._wave_step(l, z, drive0, z, A_step, g, W_hat, 0f0, 0f0, false, false)
        c = crow(z[:, :, 1]); e = band_of(c)
        (isempty(visited) || visited[end] != e) && push!(visited, e)
        if e in program && !(e in bound)
            zb = collect(z); zb[winrows(c), :, 1] .*= cis(Float32(pi) * V[e]); z = zb; push!(bound, e)
        end
        e == E && length(visited) >= E && break     # reached the last band
    end
    return packet_phase(z[:, :, 1], c), visited
end

function recover(l, ps, st, program; φ0 = 0.3f0)
    φp, vis = run_pipeline(l, ps, st, program; φ0 = φ0)
    φc, _   = run_pipeline(l, ps, st, Set{Int}(); φ0 = φ0)
    return _wrap(φp - φc), vis
end
expected(program) = _wrap(sum(Float32[e in program ? V[e] : 0f0 for e in 1:E]))

function main()
    println("="^70)
    println("Rung 3(B) — wave-programmed serial pipeline ($(E) bands, $(H)×$(W))")
    println("transforms V = ", V, "  (units of π);  Σ = ", _wrap(sum(V)))
    println("="^70)
    la = mksheet(:aniso; beta = 40.0)
    pa, sa = Lux.setup(Xoshiro(1), la); pa = tune_g!(la, pa, sa)
    d = dispersion_diagnostics(la, pa, sa; mode = :potential)
    @printf("conveyor: g=%.4f  growth*=%.3f  gvd*=%.3f  |gain_curv*|=%.3f\n\n",
            exp(pa.log_g[1]), d.growth_star, d.gvd_star, abs(d.gain_curv_star))

    _, vis = recover(la, pa, sa, Set(1:E))
    ord_ok = length(unique(vis)) == E && vis == sort(unique(vis))
    @printf("[1] band-visit order: %s   %s\n", string(vis), ord_ok ? "✓ directed 1→E" : "· incomplete")

    println("\n[2] composition recovery (recovered ≈ Σ active V_e):")
    progs = [Set([1,2]), Set([1]), Set([2])]
    okc = true
    for p in progs
        r, _ = recover(la, pa, sa, p); ex = expected(p); err = abs(_wrap(r - ex)); okc &= err < 0.08
        @printf("   program %-12s recovered % .3f  expected % .3f  |err|=%.3f %s\n",
                string(sort(collect(p))), r, ex, err, err < 0.08 ? "✓" : "·")
    end

    println("\n[3] causal (Δrecovered when expert e dropped from {1..E} ≈ V_e):")
    rfull, _ = recover(la, pa, sa, Set(1:E)); oka = true
    for e in 1:E
        rdrop, _ = recover(la, pa, sa, setdiff(Set(1:E), Set([e])))
        Δ = _wrap(rfull - rdrop); err = abs(_wrap(Δ - V[e])); oka &= err < 0.08
        @printf("   drop e=%d: Δ=% .3f  V_e=% .3f  |err|=%.3f %s\n", e, Δ, V[e], err, err < 0.08 ? "✓" : "·")
    end

    println("\n[4] isotropic contrast (:dog, no drift — should FAIL to pipeline):")
    lo = mksheet(:dog; beta = 0.0); po, so = Lux.setup(Xoshiro(1), lo); po = tune_g!(lo, po, so)
    r_iso, vis_iso = recover(lo, po, so, Set(1:E)); ex_all = expected(Set(1:E))
    iso_err = abs(_wrap(r_iso - ex_all))
    @printf("   :dog band-visit: %s   recovered % .3f  vs full-composition % .3f  |err|=%.3f\n",
            string(vis_iso), r_iso, ex_all, iso_err)
    iso_fails = iso_err > 0.1 || length(unique(vis_iso)) < E

    ok = ord_ok && okc && oka && iso_fails
    println("\nRUNG 3(B) ", ok ? "PASS ✓ — directed serial composition on the conveyor; isotropic sheet cannot" :
                                 "INCOMPLETE — inspect metrics above")
end
main()
