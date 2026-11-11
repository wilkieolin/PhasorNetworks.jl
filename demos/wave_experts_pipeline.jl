# wave_experts_pipeline.jl — Rung 3(B): a wave-PROGRAMMED serial pipeline.
#
# The payoff of the dispersion analysis. On the ISOTROPIC sheet a payload spreads
# every way at once, so a spatial sequence of experts can't process it in order
# (§9 limitation). The diffusive :aniso conveyor reaches only ~2 bands before the
# drift↔dispersion trade-off (drift ∝ gβ, but low-dispersion + stability push g→0)
# lets the packet spread. The resolution is a SHIFT coupling: Ŵ_shift(q)=e^{-i q·s}
# has unit gain, linear phase and ZERO dispersion by construction (a translation),
# so with a strong leak (A≈0) the sheet is a ballistic delay-line that carries a
# packet coherently at ANY depth. Here that conveyor drives a deep E=4 pipeline.
#
# A smooth payload injected at the top DRIFTS at s sites/period through E expert
# bands; each ACTIVE expert stamps its bind V_e as the packet passes; the
# composition Σ_{active} V_e is recovered by a differential unbind (bound run −
# no-transport control cancels the propagation phase).
#
# VERIFY: (1) band-visit order 1→E (directed path); (2) recovered ≈ Σ active V_e;
# (3) causal (drop e removes V_e); (4) CONTRAST: with transport off (shift=0) the
# packet is frozen in band 1 and cannot pipeline — directed transport is necessary.
#
# Run:  julia --project=. demos/wave_experts_pipeline.jl
#
using PhasorNetworks, Lux, Statistics, Printf
using Random: Xoshiro

const H = 24; const W = 12; const E = 4
const BH = H ÷ E                                   # rows per band (=6)
const WIN = 3                                      # readout/bind half-window around the packet
const MAXSTEPS = 60
band_rows(e) = ((e - 1) * BH + 1):(e * BH)
band_of(row) = clamp(Int(cld(round(Int, row), BH)), 1, E)
const V = Float32[0.2, 0.3, -0.4, 0.5]            # per-expert transforms (units of π)
_wrap(x) = x - 2f0 * round(x / 2f0)

# Ballistic shift conveyor (A≈0 strong leak, unit-gain shift coupling). shift=0 ⇒
# transport off (the contrast).
conveyor(shift) = PhasorWaveSheet(H, W; coupling = :shift, transmit = :potential,
    init_shift_h = Float32(shift), init_shift_w = 0.0,
    init_log_neg_lambda = log(5.0), init_log_g = log(0.99))

crow(z) = (w = abs2.(z); sum((1:H) .* vec(sum(w; dims = 2))) / (sum(w) + 1f-12))
winrows(c) = max(1, round(Int, c) - WIN):min(H, round(Int, c) + WIN)
packet_phase(z, c) = angle(sum(@view z[winrows(c), :])) / Float32(pi)   # windowed on the packet
inject(φ0) = ComplexF32[exp(-((i - 3)^2 + (j - W ÷ 2)^2) / (2f0 * 3f0^2)) * cis(Float32(pi) * φ0)
                        for i in 1:H, j in 1:W]

# Step the conveyor one cycle at a time; when the packet centroid first enters band
# e, an active expert stamps V_e onto the packet window. Stop once band E is reached.
function run_pipeline(l, ps, st, program::Set{Int}; φ0 = 0.3f0)
    ω = PhasorNetworks.period_to_angfreq(l.spk_args.t_period)
    A_step, g, W_hat = PhasorNetworks._build_coupling(l, ps, st, ω)
    z = reshape(inject(φ0), H, W, 1); drive0 = zeros(ComplexF32, H, W, 1)
    visited = Int[]; bound = Set{Int}(); c = crow(z[:, :, 1])
    for t in 1:MAXSTEPS
        z, _ = PhasorNetworks._wave_step(l, z, drive0, z, A_step, g, W_hat, 0f0, 0f0, false, false)
        c = crow(z[:, :, 1]); e = band_of(c)
        (isempty(visited) || visited[end] != e) && push!(visited, e)
        if e in program && !(e in bound)
            zb = collect(z); zb[winrows(c), :, 1] .*= cis(Float32(pi) * V[e]); z = zb; push!(bound, e)
        end
        e == E && length(visited) >= E && break
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
    println("Rung 3(B) — wave-programmed serial pipeline on the SHIFT conveyor ($(E) bands, $(H)×$(W))")
    println("transforms V = ", V, "  (units of π);  Σ = ", _wrap(sum(V)))
    println("="^70)
    la = conveyor(1.0); pa, sa = Lux.setup(Xoshiro(1), la)
    d = dispersion_diagnostics(la, pa, sa; mode = :potential)
    @printf("conveyor: v_g*=%.3f  gvd*=%.3f  |gain_curv*|=%.3f  growth*=%.3f\n\n",
            d.v_g_star, d.gvd_star, abs(d.gain_curv_star), d.growth_star)

    _, vis = recover(la, pa, sa, Set(1:E))
    ord_ok = length(unique(vis)) == E && vis == sort(unique(vis))
    @printf("[1] band-visit order: %s   %s\n", string(vis), ord_ok ? "✓ directed 1→E" : "· incomplete")

    println("\n[2] composition recovery (recovered ≈ Σ active V_e):")
    progs = [Set(1:E), Set([1,3]), Set([2,4]), Set([1,2,4]), Set([3])]
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

    println("\n[4] contrast — transport OFF (shift=0): packet frozen, cannot pipeline:")
    lo = conveyor(0.0); po, so = Lux.setup(Xoshiro(1), lo)
    r_off, vis_off = recover(lo, po, so, Set(1:E)); ex_all = expected(Set(1:E))
    off_err = abs(_wrap(r_off - ex_all))
    @printf("   band-visit: %s   recovered % .3f  vs full-composition % .3f  |err|=%.3f\n",
            string(vis_off), r_off, ex_all, off_err)
    off_fails = off_err > 0.1 || length(unique(vis_off)) < E

    ok = ord_ok && okc && oka && off_fails
    println("\nRUNG 3(B) ", ok ? "PASS ✓ — deep (E=$E) directed serial composition on the shift conveyor" :
                                 "INCOMPLETE — inspect metrics above")
end
main()
