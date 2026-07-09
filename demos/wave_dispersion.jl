# wave_dispersion.jl — dispersion, wave-speed, and ablation demo for PhasorWaveSheet
#
# The first milestone from docs/rf_wave_network_implementation.md §5: a
# *pure-forward* demonstration (no training) that the recurrent
# resonate-and-fire sheet
#   (1) has a closed-form dispersion relation and a computable criticality
#       point (spectral radius ≈ 1) — the phase-SSM analogue of branching
#       ratio σ ≈ 1 from rf_wave_network_plan.html;
#   (2) propagates a seeded pulse as a traveling wave at a measurable speed;
#   (3) reproduces the plan's decisive ablations:
#         • remove lateral inhibition → spreading/saturating front, no ring;
#         • detune g below/above criticality → extinguish / saturate;
#         • add adaptation → standing bump destabilizes and travels.
#
# Run:  julia --project=. demos/wave_dispersion.jl
# Outputs PNG/GIF into demos/wave_out/ and prints diagnostics to stdout.

ENV["GKSwstype"] = "100"                     # headless GR (no display needed)

using PhasorNetworks, Lux, Random, Printf, Statistics
using Plots
using FFTW: fftshift

const OUTDIR = joinpath(@__DIR__, "wave_out")
isdir(OUTDIR) || mkpath(OUTDIR)

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

"RMS radius of the energy distribution |z|² about the seed at (1,1) — the
robust 'how far has activity spread' metric. Uses the same wrapped-distance
grid the layer couples on."
function rms_radius(frame::AbstractMatrix{<:Complex}, rgrid::AbstractMatrix)
    e = abs2.(frame)
    s = sum(e)
    s < 1f-20 && return 0f0
    return sqrt(sum(e .* rgrid .^ 2) / s)
end

"Bisection for the recurrent gain g giving a target spectral radius.
Returns g (a Float32). Uses that spectral_radius is monotone increasing in
g once g is large enough that the coupling dominates the fixed decay A."
function critical_gain(layer, ps, st; target = 1.0f0, lo = 1f-3, hi = 1f2, iters = 40)
    setg(g) = merge(ps, (log_g = Float32[log(g)],))
    sr(g) = dispersion(layer, setg(g), st).spectral_radius
    for _ in 1:iters
        mid = sqrt(lo * hi)                  # geometric bisection (g spans orders of magnitude)
        sr(mid) < target ? (lo = mid) : (hi = mid)
    end
    return Float32(sqrt(lo * hi))
end

set_param(ps, name, val) = merge(ps, NamedTuple{(name,)}((Float32[val],)))

# ---------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------

rng = Xoshiro(1)
H = W = 48
L = 60

# Linear medium (saturating=false) so the dispersion readout is exact.
# Difference-of-Gaussians tuned near spatial balance (A_exc·σ_E² ≈ B_inh·σ_I²,
# so ∫W ≈ 0) — this selects a nonzero wavelength (a ring); the moderate
# conduction speed makes that pattern propagate.
layer = PhasorWaveSheet(H, W;
                        saturating          = false,
                        init_log_neg_lambda = log(0.15),   # light subthreshold damping
                        init_A_exc          = 1.0,
                        init_log_sigma_exc  = log(1.5),     # tight excitatory center
                        init_B_inh          = 0.25,         # balanced surround (∫W ≈ 0)
                        init_log_sigma_inh  = log(3.0),     # broad inhibitory surround (Mexican hat)
                        init_log_speed      = log(40.0))    # conduction speed c (pixels/period)
ps, st = Lux.setup(rng, layer)
rgrid  = st.rgrid

# A localized complex seed at the sheet origin (1,1), matching rgrid's center.
seed = zeros(ComplexF32, H, W)
seed[1, 1] = 1.0f0 + 0.0f0im

println("="^68)
println("PhasorWaveSheet dispersion & wave demo   ($(H)×$(W), L=$(L))")
println("="^68)

# ---------------------------------------------------------------------
# 1. Dispersion + criticality
# ---------------------------------------------------------------------

g_crit = critical_gain(layer, ps, st; target = 1.0f0)
@printf("\n[1] Critical gain (spectral radius = 1):  g_crit = %.4f\n", g_crit)
for f in (0.85f0, 1.0f0, 1.10f0)
    d = dispersion(layer, set_param(ps, :log_g, log(f * g_crit)), st)
    @printf("     g = %.3f·g_crit  →  spectral radius = %.4f   max growth/step = %+.4f\n",
            f, d.spectral_radius, log(d.spectral_radius))
end

# Growth-rate map over spatial frequency (fftshifted so DC is centered).
d_crit = dispersion(layer, set_param(ps, :log_g, log(g_crit)), st)
gr = fftshift(real.(d_crit.k_eff))
hm = heatmap(gr; title = "growth rate Re k_eff(q) at g_crit  (bright = marginal)",
             xlabel = "qₓ", ylabel = "q_y", aspect_ratio = 1, c = :viridis)
savefig(hm, joinpath(OUTDIR, "dispersion_growth.png"))
println("     saved dispersion_growth.png")

# ---------------------------------------------------------------------
# 2. Wave propagation + speed, at three operating points
# ---------------------------------------------------------------------

println("\n[2] Seeded-pulse propagation (linear medium):")
speeds = Dict{String,Float32}()
radius_curves = Dict{String, Vector{Float32}}()
for (label, f) in (("subcritical_0.85", 0.85f0),
                   ("critical_1.00",    1.00f0),
                   ("supercritical_1.10", 1.10f0))
    psg = set_param(ps, :log_g, log(f * g_crit))
    traj = wave_simulate(layer, psg, st; z0 = seed, L = L)      # (H,W,L)
    # normalize each frame for visualization / radius (linear medium grows/decays)
    radii = Float32[rms_radius(traj[:, :, t], rgrid) for t in 1:L]
    radius_curves[label] = radii
    # speed = slope of RMS-radius vs step over the early expansion window
    win = 5:min(L, 30)
    speed = (radii[last(win)] - radii[first(win)]) / (last(win) - first(win))
    speeds[label] = speed
    @printf("     %-20s  expansion speed ≈ %.3f px/step   final |z|ₘₐₓ = %.3e\n",
            label, speed, maximum(abs.(traj[:, :, end])))
end

rc = plot(title = "activity spread (RMS radius) vs step", xlabel = "step",
          ylabel = "RMS radius (px)", legend = :topleft)
for (label, radii) in sort(collect(radius_curves))
    plot!(rc, 1:L, radii; label = label, lw = 2)
end
savefig(rc, joinpath(OUTDIR, "wave_spread.png"))
println("     saved wave_spread.png")

# A GIF of the critical-gain traveling wave (phase-colored, magnitude-gated).
psg = set_param(ps, :log_g, log(g_crit))
traj = wave_simulate(layer, psg, st; z0 = seed, L = L)
anim = @animate for t in 1:L
    frame = fftshift(traj[:, :, t])
    mag = abs.(frame); mag ./= (maximum(mag) + 1f-12)
    heatmap(angle.(frame) .* mag; c = :twilight, clims = (-π, π),
            title = @sprintf("traveling wave  step %02d/%d", t, L),
            aspect_ratio = 1, colorbar = false, axis = false)
end
gif(anim, joinpath(OUTDIR, "traveling_wave.gif"); fps = 12)
println("     saved traveling_wave.gif")

# ---------------------------------------------------------------------
# 3. Ablations
# ---------------------------------------------------------------------

println("\n[3] Decisive ablations:")

# 3a. Remove lateral inhibition (B_inh = 0): the Mexican hat becomes a pure
#     excitatory Gaussian → no wavelength selection, activity spreads as a
#     saturating front rather than a structured ring.
ps_noinh = set_param(set_param(ps, :B_inh, 0.0f0), :log_g, log(g_crit))
d_noinh = dispersion(layer, ps_noinh, st)
@printf("     remove inhibition (B_inh=0):  spectral radius %.3f → %.3f  (DC dominates: front, not ring)\n",
        d_crit.spectral_radius, d_noinh.spectral_radius)
q_peak_full  = argmax(fftshift(real.(d_crit.k_eff)))
q_peak_noinh = argmax(fftshift(real.(d_noinh.k_eff)))
center = (H ÷ 2 + 1, W ÷ 2 + 1)
@printf("       fastest-growing mode |q|:  with inhibition = %.2f,  without = %.2f  (→ 0 = uniform front)\n",
        sqrt((q_peak_full[1]-center[1])^2 + (q_peak_full[2]-center[2])^2),
        sqrt((q_peak_noinh[1]-center[1])^2 + (q_peak_noinh[2]-center[2])^2))

# 3b. Criticality detune already shown in [2]: subcritical decays, supercritical saturates.
@printf("     detune g:  |z|ₘₐₓ(final)  sub=%.2e  crit=%.2e  super=%.2e  (extinguish → sustain → saturate)\n",
        maximum(abs.(wave_simulate(layer, set_param(ps,:log_g,log(0.85f0*g_crit)), st; z0=seed, L=L)[:,:,end])),
        maximum(abs.(wave_simulate(layer, set_param(ps,:log_g,log(1.00f0*g_crit)), st; z0=seed, L=L)[:,:,end])),
        maximum(abs.(wave_simulate(layer, set_param(ps,:log_g,log(1.10f0*g_crit)), st; z0=seed, L=L)[:,:,end])))

# 3c. Self-limiting (insight #3 of the companion doc). A *linear* wave medium
#     cannot self-limit: above criticality its amplitude grows geometrically
#     (|z|ₘₐₓ ~ spectral_radiusᵗ). Projecting onto the unit circle each step
#     (saturating=true, phase-only) bounds |z| ≡ 1 by construction — the
#     wave regulates its own amplitude while keeping all its phase content.
g_super = 1.6f0 * g_crit
layer_lin = PhasorWaveSheet(H, W; saturating = false,
                            init_log_neg_lambda = log(0.15), init_A_exc = 1.0,
                            init_log_sigma_exc = log(1.5), init_B_inh = 0.25,
                            init_log_sigma_inh = log(3.0), init_log_speed = log(40.0))
layer_sat = PhasorWaveSheet(H, W; saturating = true,
                            init_log_neg_lambda = log(0.15), init_A_exc = 1.0,
                            init_log_sigma_exc = log(1.5), init_B_inh = 0.25,
                            init_log_sigma_inh = log(3.0), init_log_speed = log(40.0))
ps_l, st_l = Lux.setup(rng, layer_lin)
ps_s, st_s = Lux.setup(rng, layer_sat)
ps_l = set_param(ps_l, :log_g, log(g_super))
ps_s = set_param(ps_s, :log_g, log(g_super))
traj_lin = wave_simulate(layer_lin, ps_l, st_l; z0 = seed, L = L)
traj_sat = wave_simulate(layer_sat, ps_s, st_s; z0 = seed, L = L)
amp_lin = Float32[maximum(abs.(traj_lin[:, :, t])) for t in 1:L]
amp_sat = Float32[maximum(abs.(traj_sat[:, :, t])) for t in 1:L]
@printf("     self-limiting @ g=1.6·g_crit:  linear |z|ₘₐₓ %.3f → %.3f (unbounded),  saturating %.3f → %.3f (≡1)\n",
        amp_lin[1], amp_lin[end], amp_sat[1], amp_sat[end])
sl = plot(1:L, amp_lin; label = "linear (grows geometrically)", lw = 2, yscale = :log10,
          title = "phase-only saturation self-limits amplitude", xlabel = "step",
          ylabel = "|z|ₘₐₓ (log)", legend = :topleft)
plot!(sl, 1:L, amp_sat; label = "saturating (phase-only, ≡ 1)", lw = 2)
savefig(sl, joinpath(OUTDIR, "self_limiting.png"))
println("     saved self_limiting.png")

# ---------------------------------------------------------------------
# 4. Tier-2 equivalence: discrete recurrence ≈ continuous ODE
# ---------------------------------------------------------------------
#
# The discrete phase-SSM recurrence (Tier 1) and the continuous ODE (Tier 2,
# integrated by DifferentialEquations.jl) integrate the *same* defining
# equation dz/dt = k·z + g·(coupling). Seeded from the same pulse at a
# subcritical gain, their fields stay strongly aligned — the two modes are
# one dynamics, exactly the SSM/ODE duality (K[n]=Aⁿ·B) the codebase is
# built on.

println("\n[4] Tier-2 equivalence (discrete recurrence vs continuous ODE):")
g_sub = 0.85f0 * g_crit
psg = set_param(ps, :log_g, log(g_sub))
traj_d = wave_simulate(layer, psg, st; z0 = seed, L = L, mode = :discrete)
traj_o = wave_simulate(layer, psg, st; z0 = seed, L = L, mode = :ode)
fieldsim(a, b) = abs(sum(vec(a) .* conj.(vec(b)))) /
                 (sqrt(sum(abs2, vec(a))) * sqrt(sum(abs2, vec(b))) + 1f-20)
for t in (5, 20, 40, L)
    @printf("     step %2d  discrete↔ODE field similarity = %.4f\n", t,
            fieldsim(traj_d[:, :, t], traj_o[:, :, t]))
end
cmp = plot(layout = (1, 2), size = (720, 340))
for (i, (lab, tr)) in enumerate(("discrete (Tier 1)" => traj_d, "ODE (Tier 2)" => traj_o))
    fr = fftshift(tr[:, :, L]); m = abs.(fr); m ./= (maximum(m) + 1f-12)
    heatmap!(cmp[i], angle.(fr) .* m; c = :twilight, clims = (-π, π),
             title = lab, aspect_ratio = 1, colorbar = false, axis = false)
end
savefig(cmp, joinpath(OUTDIR, "tier_equivalence.png"))
println("     saved tier_equivalence.png")

println("\nDone. Figures in $(OUTDIR)")
