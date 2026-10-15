# E8 — the R&F transponder's domain of operation in frequency.
#
# (a) Sinusoidal tuning curve: the smallest amplitude of a smoothly ramped tone
#     that fires the cell, against frequency, by bisection; compared with the
#     steady-state closed form
#     A_th(f) = I_th·|1 − ω²LC + iωL/R| (loop-current gain of a current-driven
#     parallel RLC).
# (b) Two-pulse coincidence: the smallest common amplitude of two input pulses
#     (the cell's own output shape) that fires it, against their separation Δ.
#     This is the cell as a coincidence detector — and it is RESONANT: the
#     window repeats every carrier period, and pulses half a period apart
#     cancel. It is the microscopic origin of the coherence limit on line
#     delays (time_scaled docstring).
# (c) The time-scaled family f₀ ∈ {1, 2, 5, 10, 20} GHz: (a) on an absolute
#     frequency axis, i.e. the band the device family can cover.

include("common.jl")
using Plots

c = RFTransponder()
pe = phasor_equivalent(c)
f0 = resonance_frequency(c)                     # GHz

fires(cell, drive, dt, n) = !isempty(cell_simulate(cell, drive, dt, n).spike_time)
function bisect_amp(pred; lo = 0.01, hi = 2000.0, it = 40)
    pred(hi) || return NaN
    for _ in 1:it
        mid = sqrt(lo * hi); pred(mid) ? (hi = mid) : (lo = mid)
    end
    return hi
end

println("[a] sinusoidal tuning curve")
rows_a = NamedTuple[]
for r in range(0.5, 1.5; length = 41)
    f = r * f0 * 1e-3                           # 1/ps
    # raised-cosine turn-on over 3 ns (≈ 4.5 ringdown times): an abrupt tone also
    # rings the tank at f₀ and fires it below the steady-state threshold
    env(t) = t < 3000 ? 0.5 * (1 - cos(π * t / 3000)) : 1.0
    A = bisect_amp(a -> fires(c, t -> Float32(a * env(t) * sin(2π * f * t)), 0.5, 18000))
    A_step = bisect_amp(a -> fires(c, t -> Float32(a * sin(2π * f * t)), 0.5, 8000))
    ω = 2π * f
    Ath = c.I_th * abs(complex(1 - ω^2 * c.L * c.C, ω * c.L / c.R))
    push!(rows_a, (; f_over_f0 = r, f_GHz = r * f0, A_th_uA = A, A_th_theory_uA = Ath, A_th_abrupt_uA = A_step))
end
write_tsv("e8a_tuning_curve.tsv", rows_a)
fmin = rows_a[argmin([r.A_th_uA for r in rows_a])]
@printf("    minimum %.3f µA at %.2f GHz (theory I_th/Q = %.3f µA); half-power bandwidth f₀/Q = %.2f GHz\n",
        fmin.A_th_uA, fmin.f_GHz, c.I_th / quality_factor(c), f0 / quality_factor(c))

println("[b] two-pulse coincidence window")
τ = c.tau_out
pulse(t, t0) = (s = (t - t0) / τ; s > 0 ? s * exp(1 - s) : 0.0)
rows_b = NamedTuple[]
for Δ in 0:2:400
    A = bisect_amp(a -> fires(c, t -> Float32(a * (pulse(t, 20) + pulse(t, 20 + Δ))), 0.25, round(Int, (Δ + 400) / 0.25)))
    push!(rows_b, (; delta_ps = Δ, A_th_uA = A))
end
A1 = bisect_amp(a -> fires(c, t -> Float32(a * pulse(t, 20)), 0.25, 1600))
write_tsv("e8b_coincidence.tsv", rows_b)
@printf("    single pulse threshold %.2f µA; pair at Δ=0: %.2f (×%.2f); at Δ=T/2=%.0f ps: %.2f; at Δ=T: %.2f\n",
        A1, rows_b[1].A_th_uA, A1 / rows_b[1].A_th_uA, pe.t_period / 2,
        rows_b[findmin(abs.([r.delta_ps for r in rows_b] .- pe.t_period / 2))[2]].A_th_uA,
        rows_b[findmin(abs.([r.delta_ps for r in rows_b] .- pe.t_period))[2]].A_th_uA)

println("[c] time-scaled family")
rows_c = NamedTuple[]
for fg in (1.0, 2.0, 5.0, 10.0, 20.0)
    cs = time_scaled(c, f0 / fg)
    for r in range(0.5, 1.5; length = 21)
        f = r * fg * 1e-3
        ω = 2π * f
        Ath = cs.I_th * abs(complex(1 - ω^2 * cs.L * cs.C, ω * cs.L / cs.R))
        push!(rows_c, (; f0_GHz = fg, f_GHz = r * fg, A_th_uA = Ath, t_ref_ps = cs.t_ref, tau_out_ps = cs.tau_out))
    end
end
write_tsv("e8c_family.tsv", rows_c)

p1 = plot([r.f_GHz for r in rows_a], [r.A_th_theory_uA for r in rows_a]; lw = 2, lc = :gray, label = "closed form",
          yscale = :log10, xlabel = "drive frequency (GHz)", ylabel = "threshold amplitude (µA)",
          title = "sinusoidal tuning (Q = 21)")
scatter!(p1, [r.f_GHz for r in rows_a], [r.A_th_uA for r in rows_a]; ms = 3, mc = :steelblue, label = "simulated, ramped tone")
scatter!(p1, [r.f_GHz for r in rows_a], [r.A_th_abrupt_uA for r in rows_a]; ms = 2, mc = :orange, label = "simulated, abrupt tone")
p2 = plot([r.delta_ps for r in rows_b], [r.A_th_uA for r in rows_b]; lw = 2, lc = :crimson, label = "pair threshold",
          xlabel = "pulse separation Δ (ps)", ylabel = "amplitude to fire (µA)", title = "two-pulse coincidence")
hline!(p2, [A1]; lc = :gray, ls = :dash, label = "single pulse")
vline!(p2, pe.t_period .* (0.5:1:3.5); lc = :black, ls = :dot, alpha = 0.4, label = "Δ = (k+½)T")
p3 = plot(; xscale = :log10, yscale = :log10, xlabel = "drive frequency (GHz)", ylabel = "threshold (µA)",
          title = "time-scaled family", legend = :topright)
for fg in (1.0, 2.0, 5.0, 10.0, 20.0)
    rr = filter(r -> r.f0_GHz == fg, rows_c)
    plot!(p3, [r.f_GHz for r in rr], [r.A_th_uA for r in rr]; lw = 2, label = "f₀ = $(Int(fg)) GHz")
end
savefig(plot(p1, p2, p3; layout = (1, 3), size = (1350, 400), left_margin = 5Plots.mm, bottom_margin = 6Plots.mm),
        joinpath(STUDY_OUT, "e8_cell_tuning.png"))
println("    saved e8_cell_tuning.png")
