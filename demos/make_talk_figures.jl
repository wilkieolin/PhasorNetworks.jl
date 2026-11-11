ENV["GKSwstype"] = "100"
using PhasorNetworks, Lux, Random, Plots, Printf
using FFTW: fftshift

const OUTDIR = joinpath(@__DIR__, "wave_out")

# `init_log_speed` is omitted on purpose: the layer derives the matched
# conduction speed c = 2σ_I/T = 6 (delay phase φ = π), which is the regime where
# the band-selected mode actually carries group velocity. These figures used to
# pin c = 40 and g = 0.0261 — that is the weak-delay corner, where v_g(q*) → 0 and
# an impulse gives STANDING rings. g is now bisected to criticality per layer
# rather than quoted, since g_crit moves with c (0.026 → 0.069).
function make_layer(coupling, transmit; H=48, W=48)
    PhasorWaveSheet(H, W;
        coupling = coupling,
        transmit = transmit,
        saturating = false,
        init_log_neg_lambda = log(0.15),
        init_A_exc = 1.0,
        init_log_sigma_exc = log(1.5),
        init_B_inh = 0.25,
        init_log_sigma_inh = log(3.0))
end

"Setup a layer with its gain bisected to criticality. Returns (ps, st, g_crit)."
function setup_critical(l; seed = 1)
    ps, st = Lux.setup(Xoshiro(seed), l)
    sr(g) = dispersion(l, merge(ps, (log_g = Float32[log(g)],)), st).spectral_radius
    lo, hi = 1f-5, 1f3
    for _ in 1:50
        m = sqrt(lo * hi)
        sr(m) < 1 ? (lo = m) : (hi = m)
    end
    gc = Float32(sqrt(lo * hi))
    return merge(ps, (log_g = Float32[log(gc)],)), st, gc
end

function show_dispersion(l; name, st, ps)
    d = dispersion(l, ps, st)
    gr = fftshift(real.(d.k_eff))
    hm = heatmap(gr; title="growth rate $(name)  (sp_radius=$(round(d.spectral_radius,digits=3)))",
                 xlabel="qₓ", ylabel="q_y", aspect_ratio=1, c=:viridis)
    savefig(hm, joinpath(OUTDIR, "dispersion_$(name).png"))
end

rng = Xoshiro(1)

# Wavesheets 4: different kernels / coupling modes
for (coupling, label) in ((:dog, "dog"), (:stencil, "stencil"), (:aniso, "aniso"), (:shift, "shift"))
    l = make_layer(coupling, :potential)
    ps, st, gc = setup_critical(l)
    t = wave_transport(l, ps, st)
    show_dispersion(l; name=label, st=st, ps=ps)
    @printf("saved dispersion_%s.png  (g_crit=%.4f, transport ratio %.3f → %s)\n",
            label, gc, t.transport_ratio, t.verdict)
end

# Wavesheets 4: advection / particle-wake demonstration with aniso.
#
# NOTE c is pinned to 40 here. β's "+h" sign convention is only meaningful in the
# weak-delay regime: at the matched speed the DoG is a backward wave (v_g opposite
# to q), so β biases growth toward +q_h while the packet travels toward −h and the
# wake runs the other way. This figure is about advection in isolation.
println("\nMaking advection / particle wake demo figure...")
l_aniso = PhasorWaveSheet(48, 48; coupling=:aniso, transmit=:potential, saturating=false,
    init_beta_h=0.5, init_beta_w=0.0,
    init_log_neg_lambda=log(0.15), init_A_exc=1.0, init_log_sigma_exc=log(1.5),
    init_B_inh=0.25, init_log_sigma_inh=log(3.0), init_log_speed=log(40.0),
    init_log_g=log(0.0261))
ps_a, st_a = Lux.setup(rng, l_aniso)
seed = zeros(ComplexF32, 48, 48)
seed[1, 1] = 1.0f0 + 0.0f0im
traj_a = wave_simulate(l_aniso, ps_a, st_a; z0=seed, L=30)
anim = @animate for t in 1:30
    frame = fftshift(traj_a[:, :, t])
    mag = abs.(frame); mag ./= (maximum(mag) + 1f-12)
    heatmap(angle.(frame) .* mag; c = :twilight, clims = (-π, π),
            title = "advection (aniso, β_h=0.5)  step $(t)",
            aspect_ratio = 1, colorbar = false, axis = false)
end
gif(anim, joinpath(OUTDIR, "advection_particle_wake.gif"); fps = 10)
println("saved advection_particle_wake.gif")

# Criticality figure: spectral radius vs g for different coupling types
println("\nMaking criticality comparison figure...")
gs = [0.001 * 2.0^i for i in 0:12]
rc = plot(title="Criticality: spectral radius vs g", xlabel="g", ylabel="spectral radius",
          ylims=(0, 1.4), legend=:bottomright, lw=2)
for (coupling, label) in ((:dog, ":dog"), (:aniso, ":aniso"), (:shift, ":shift"))
    srs = Float32[]
    l0 = make_layer(coupling, :potential)
    ps0, st0 = Lux.setup(rng, l0)
    for gg in gs
        push!(srs, dispersion(l0, merge(ps0, (log_g = Float32[log(gg)],)), st0).spectral_radius)
    end
    plot!(rc, gs, srs; label=label, marker=:o, ms=3)
end
hline!([1.0]; label="critical (σ=1)", ls=:dash, c=:red)
savefig(rc, joinpath(OUTDIR, "criticality_spectral_radius.png"))
println("saved criticality_spectral_radius.png")

# Device / SNSPD excitation: how fast the sheet transports, vs operating point.
#
# This used to hardcode [0.150, 0.162, 0.168] — RMS-radius slopes copied out of
# wave_dispersion.jl. Those are bulk second moments of a concentric pulse, not a
# speed: they are window-dependent and about ⅓ of any level-set front speed. The
# transport rate the sheet actually has is the peak of the radial group-velocity
# band, max_q v_g(q) = max_q -dΩ/dq, which we now compute rather than quote.
println("\nMaking device / velocity extraction figure...")
lv = make_layer(:dog, :potential)
psv, stv, g_crit = setup_critical(lv)
gfac  = Float32[0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
vpeak = Float32[]
for f in gfac
    rb = radial_band(lv, merge(psv, (log_g = Float32[log(f * g_crit)],)), stv)
    push!(vpeak, maximum(abs.(rb.v_r)))     # abs: v_r is negative (backward wave)
end
sc = plot(gfac, vpeak; marker=:o, ms=5, lw=2, legend=false, c=:purple,
          title="Transport rate vs operating point (matched c)",
          xlabel="g / g_crit", ylabel="peak radial group speed  max|v_r|  (sites/period)")
vline!([1.0]; ls=:dash, c=:red)
annotate!([(1.02, minimum(vpeak), text("critical", 8, :left, :red))])
savefig(sc, joinpath(OUTDIR, "device_velocity_extraction.png"))
@printf("saved device_velocity_extraction.png  (peak |v_r|: %.3f -> %.3f sites/period over g/g_crit %.1f-%.1f)\n",
        first(vpeak), last(vpeak), first(gfac), last(gfac))

# ---------------------------------------------------------------------
# The headline result: the standing → traveling transition vs conduction speed.
# ---------------------------------------------------------------------
#
# Sweeps c and, at criticality for each c, reports how much of the band's
# available transport the band-SELECTED mode actually carries. The transition is
# controlled by the delay phase across the surround, φ = ω σ_I/c, and lands at
# φ = π (c = 2σ_I/T = 6). This is the figure that explains why the sheet used to
# make standing rings.
println("\nMaking standing → traveling transition figure...")
cs = exp.(range(log(2.5), log(80.0); length = 70))     # fine: the window is narrow
ratios = Float32[]; vstars = Float32[]
for c in cs
    l = PhasorWaveSheet(64, 64; transmit=:potential, saturating=false,
        init_log_neg_lambda=log(0.15), init_A_exc=1.0, init_log_sigma_exc=log(1.5),
        init_B_inh=0.25, init_log_sigma_inh=log(3.0), init_log_speed=log(c))
    ps_c, st_c, _ = setup_critical(l)
    t = wave_transport(l, ps_c, st_c)
    push!(ratios, t.transport_ratio); push!(vstars, abs(t.v_r_star))
end
xt = ([2.5, 4, 6, 10, 20, 40, 80], ["2.5", "4", "6", "10", "20", "40", "80"])
tp = plot(cs, ratios; lw=2.5, c=:darkgreen, xscale=:log10, xticks=xt,
          label="transport ratio  |v_r(q*)| / max|v_r|", legend=:topright,
          xlabel="conduction speed c (sites/period)", ylabel="transport ratio",
          title="Standing → traveling is set by the delay phase φ = ω σ_I / c",
          ylims=(-0.03, 1.12))
hline!(tp, [0.5]; ls=:dot, c=:gray, label="traveling / standing threshold")
vline!(tp, [6.0]; ls=:dash, c=:red, label="matched c = 2σ_I/T  (φ = π)")
vline!(tp, [40.0]; ls=:dash, c=:black, label="previous default c = 40  (φ = 0.15π)")
tv = plot(cs, vstars; lw=2.5, c=:purple, xscale=:log10, xticks=xt, legend=:topright,
          xlabel="conduction speed c (sites/period)", label="|v_r(q*)|",
          ylabel="speed of the selected mode\n(sites/period)",
          title="…and so is the speed the selected mode actually has")
vline!(tv, [6.0]; ls=:dash, c=:red, label="")
vline!(tv, [40.0]; ls=:dash, c=:black, label="")
trans = plot(tp, tv, layout=(2,1), size=(900, 660), leftmargin=6Plots.mm)
savefig(trans, joinpath(OUTDIR, "transport_transition.png"))
win = cs[ratios .>= 0.5]
@printf("saved transport_transition.png  (ratio %.3f at c=40 → %.3f at c=6)\n",
        ratios[argmin(abs.(cs .- 40))], ratios[argmin(abs.(cs .- 6))])
@printf("   traveling window (ratio ≥ 0.5): c ∈ [%.2f, %.2f]  = φ/π ∈ [%.2f, %.2f]  — a %.0f%% band, not a plateau\n",
        minimum(win), maximum(win), 2*3.0/maximum(win), 2*3.0/minimum(win),
        100*(maximum(win)-minimum(win))/6.0)

println("\nAll extra figures saved to $(OUTDIR)")
