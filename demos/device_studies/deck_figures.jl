# deck_figures — raw data for the explanatory figures of the slide deck.
#
# Writes into demos/wave_out/device_studies/deck/:
#   neuron_traces.tsv   single-cell responses (scenario, t, input, i/I_th, out)
#   neuron_spikes.tsv   spike times per scenario
#   *.f32               Float32 arrays, column-major, with a matching *.dims
#                       file (one size per line); NaN = never fired
#
# (a) The point neuron: sub- vs supra-threshold pulse, refractory period, and
#     resonant coincidence (pulse pairs a period vs half a period apart).
# (b) The medium: a single forced cell (dies), a 5-cell nucleus (ignites) and
#     two nuclei (fronts collide and annihilate), as firing + ringing frames.
# (c) Connectivity sets propagation: equal-time first-spike maps for three
#     line delays, an anisotropic medium and a delay interface.
# (d) Mach wakes at Mach 0.5, 1.5, 3, 10 (time since each cell fired).

include("common.jl")
backend, dev = load_gpu_backend()
const OUT = joinpath(STUDY_OUT, "deck"); isdir(OUT) || mkpath(OUT)
dump(name, A) = (write(joinpath(OUT, name * ".f32"), Float32.(A));
                 write(joinpath(OUT, name * ".dims"), join(size(A), '\n')))
cell = RFTransponder()

const PARTS = get(ENV, "DECK_PARTS", "abcdefg")

# ---------------------------------------------------------------- (a) neuron
if 'a' in PARTS
pulse(t, t0, a; τ = 10f0) = (s = (t - t0) / τ; s > 0 ? a * s * exp(1 - s) : 0f0)
T = 1000 / resonance_frequency(cell)                    # ps
scen = [
    ("sub",    [(50.0, 10.0)]),                                     # one weak pulse
    ("supra",  [(50.0, 20.0), (220.0, 20.0), (480.0, 20.0)]),        # fires, refractory, fires
    ("inphase", [(50.0, 12.0), (50.0 + T, 12.0)]),                   # pair one period apart
    ("antiphase", [(50.0, 12.0), (50.0 + T / 2, 12.0)]),             # pair half a period apart
    ("single12", [(50.0, 12.0)]),
]
rows = NamedTuple[]; srows = NamedTuple[]
for (name, ev) in scen
    drv(t) = Float32(sum(pulse(Float32(t), Float32(t0), Float32(a)) for (t0, a) in ev))
    tr = cell_simulate(cell, drv, 0.25, 3200; stride = 4)
    for k in eachindex(tr.t)
        push!(rows, (; scenario = name, t = tr.t[k], input = drv(tr.t[k]), i_rel = tr.i[k] / cell.I_th, out = tr.out[k]))
    end
    for ts in tr.spike_time
        push!(srows, (; scenario = name, t_spike = ts))
    end
    @printf("  %-9s spikes at %s ps\n", name, join(round.(tr.spike_time; digits = 1), ", "))
end
write_tsv("deck/neuron_traces.tsv", rows)
write_tsv("deck/neuron_spikes.tsv", isempty(srows) ? [(; scenario = "", t_spike = NaN)] : srows)
end

# ---------------------------------------------------------------- (b) medium
if 'b' in PARTS
N = 128
sh = DeviceSheet(cell, N, N)
for (name, drv, nst) in (
        ("single", seed_drive(N, N; radius = 0), 600),
        ("nucleus", seed_drive(N, N; radius = 1), 1800),
        ("collide", seed_drive(N, N; center = (N ÷ 2, N ÷ 4), radius = 2) +
                    seed_drive(N, N; center = (N ÷ 2, 3N ÷ 4), radius = 2), 1800))
    tr = sheet_simulate(sh, nst; drive = drv, frame_every = 20, keep_amplitude = true, device = dev)
    dump("med_$(name)_fire", tr.fire_frames); dump("med_$(name)_amp", tr.amp_frames)
    tf = copy(tr.t_first); tf[.!isfinite.(tf)] .= NaN; dump("med_$(name)_tfirst", tf)
    @printf("  medium %-8s fired %d sites, max spikes/site %d\n", name, count(isfinite, tr.t_first), maximum(tr.n_spikes))
end

end

# ------------------------------------------------- (c) connectivity → propagation
if 'c' in PARTS
N = 192
offs = stencil_offsets(3)
for (name, kw, nst) in (
        ("d5",  (; ps_per_site = 5),  2500),
        ("d10", (; ps_per_site = 10), 3500),
        ("d20", (; ps_per_site = 20), 5000),
        ("aniso", (; D = Float32[sqrt((10o[1])^2 + (25o[2])^2) for y in 1:N, x in 1:N, o in offs]), 6000))
    tr = sheet_simulate(DeviceSheet(cell, N, N; kw...), nst; drive = seed_drive(N, N; radius = 3), device = dev)
    tf = copy(tr.t_first); tf[.!isfinite.(tf)] .= NaN; dump("con_$(name)", tf)
    u = arrival_speed(tr.t_first; center = (N ÷ 2, N ÷ 2), rmin = 15, rmax = 80)[1]
    @printf("  connectivity %-6s u = %.4f sites/ps, all fired by %.0f ps\n", name, u, maximum(filter(isfinite, tr.t_first)))
end
# interface: left half 10 ps/pitch, right half 25; plane front at 35° incidence
θ1 = deg2rad(35.0); n = (sin(θ1), cos(θ1))
Db = Float32[(x > N ÷ 2 ? 25 : 10) * sqrt(o[1]^2 + o[2]^2) for y in 1:N, x in 1:N, o in offs]
P0 = (N ÷ 2 - round(Int, 50 * sin(θ1)), N ÷ 5)
th = Float32[abs((y - P0[1]) * n[1] + (x - P0[2]) * n[2]) <= 0.7 &&
             abs((y - P0[1]) * n[2] - (x - P0[2]) * n[1]) <= 45 && x < N ÷ 2 - 12 ? 0f0 : Inf32 for y in 1:N, x in 1:N]
tr = sheet_simulate(DeviceSheet(cell, N, N; D = Db), 7000; drive = EventDrive(th, 40), device = dev)
tf = copy(tr.t_first); tf[.!isfinite.(tf)] .= NaN; dump("con_interface", tf)
end

# ---------------------------------------------------------------- (d) Mach wakes
if 'd' in PARTS
N = 384; ROW0 = 10
sheet = DeviceSheet(cell, N, N)
u = medium_speed(sheet, dev; nsteps = 5000, rmax = 0.4N)
mrows = NamedTuple[]
for M in (0.5, 1.5, 3.0, 10.0)
    v = M * u
    nst = round(Int, 0.8 * (N - ROW0) / max(v, u))
    tr = sheet_simulate(sheet, nst; drive = particle_drive(N, N; dt_site = 1 / v, x0 = (ROW0, N ÷ 2), width = 3), device = dev)
    age = Float32(nst) .- tr.t_first
    dump(@sprintf("mach_%04.1f", M), ifelse.(isfinite.(age), age, NaN32))
    push!(mrows, (; mach = M, v_over_c = v / C_SITES, u_sites_ps = u, t_snap = nst, particle_row = ROW0 + v * nst))
end
write_tsv("deck/mach.tsv", mrows)
end

# ---------------------------------------------------------------- (e) edge readout
if 'e' in PARTS
u = 0.0477                                            # default medium, sites/ps (radial measurement)
H, W = 640, 160
v = 3u
tr = sheet_simulate(DeviceSheet(cell, H, W), round(Int, H / v + 2500);
                    drive = particle_drive(H, W; dt_site = 1 / v, x0 = (1, W ÷ 2), width = 3), device = dev)
tf = copy(tr.t_first); tf[.!isfinite.(tf)] .= NaN; dump("edge_axis", tf)
N = 384
tr = sheet_simulate(DeviceSheet(cell, N, N), 9000;
                    drive = particle_drive(N, N; dt_site = 1 / v, x0 = (1, 120), angle = deg2rad(30), width = 3), device = dev)
tf = copy(tr.t_first); tf[.!isfinite.(tf)] .= NaN; dump("edge_angle", tf)
write_tsv("deck/edge.tsv", [(; u, v, v_over_c = v / C_SITES, mach = v / u)])
end
println("done")

# ---------------------------------------------------------------- (f) slowing the wave
if 'f' in PARTS
N = 192
offs = stencil_offsets(3)
even = [iseven(y + x) for y in 1:N, x in 1:N]
frows = NamedTuple[]
for (name, c, pps, sub, G) in (("fast30", cell, 30, true, 0.45), ("fast70", cell, 70, true, 0.45),
                               ("slow30", time_scaled(cell, 3), 30, true, 0.45),
                               ("full30", cell, 30, false, 0.3), ("full70", cell, 70, false, 0.3),
                               ("full70slow", time_scaled(cell, 3), 70, false, 0.3))
    D = Float32[pps * sqrt(o[1]^2 + o[2]^2) for y in 1:N, x in 1:N, o in offs]
    Gm = Float32[(!sub || iseven(o[1] + o[2])) ? G : 0 for y in 1:N, x in 1:N, o in offs]
    m = sub ? even : trues(N, N)
    local tr = sheet_simulate(DeviceSheet(c, N, N; D, G = Gm), 16000;
                              drive = seed_drive(N, N; radius = 4, amp = 80), device = dev,
                              frame_every = startswith(name, "full70") ? 50 : 0)
    startswith(name, "full70") && dump("slow_$(name)_frames", tr.fire_frames)
    ns = ifelse.(m, tr.n_spikes, NaN32); dump("slow_$(name)_nspk", ns)
    local tf = ifelse.(m .& isfinite.(tr.t_first), tr.t_first, NaN32); dump("slow_$(name)_tfirst", tf)
    fired = count(isfinite, tf) / count(m)
    u = fired > 0.9 ? arrival_speed(ifelse.(m, tr.t_first, Inf32); center = (N ÷ 2, N ÷ 2), rmin = 15, rmax = 80)[1] : NaN
    mx = maximum(tr.n_spikes[m])
    @printf("  %-7s fired %5.1f%%  max spikes/cell %3d  mean %.2f  u %.4f sites/ps (%.4fc)\n", name, 100fired, mx,
            sum(tr.n_spikes[m]) / count(m), u, u / C_SITES)
    push!(frows, (; case = name, f0_GHz = resonance_frequency(c), pps, sublattice = sub, G, fired, max_spikes = mx, u, u_over_c = u / C_SITES))
end
write_tsv("deck/slowing.tsv", frows)
end

# ---------------------------------------------------------------- (g) Mach-cone animation
if 'g' in PARTS
N = 384; ROW0 = 10
u = 0.0477; v = 2.09u
local tr = sheet_simulate(DeviceSheet(cell, N, N), round(Int, (N - ROW0) / v + 600);
                          drive = particle_drive(N, N; dt_site = 1 / v, x0 = (ROW0, N ÷ 2), width = 3), device = dev)
tf = copy(tr.t_first); tf[.!isfinite.(tf)] .= NaN; dump("anim_tfirst", tf)
write_tsv("deck/anim.tsv", [(; u, v, mach = v / u, row0 = ROW0, v_over_c = v / C_SITES)])
end
