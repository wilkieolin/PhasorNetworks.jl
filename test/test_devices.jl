# Tests for devices.jl — the RFTransponder cell.
# Entry point: device_tests(), called from runtests.jl.
#
# Every check is against a closed form, not a stored trace: the free tank
# against the analytic RLC ringdown (the stepper is an exact ZOH propagator, so
# this holds to Float32 roundoff at any dt), the driven tank against the
# resonant gain Q·A, and the output pulse against the alpha function's charge.

const _DEV_CELL = RFTransponder()          # 10 GHz, Q = 21, I_th = 20 µA, t_ref = 300 ps

function test_device_construction()
    c = _DEV_CELL
    @test resonance_frequency(c) ≈ 10.0 rtol = 1e-5
    @test quality_factor(c) ≈ 21.0 rtol = 1e-5
    pe = phasor_equivalent(c)
    # λT = −π/(Q·√(1 − 1/4Q²)) — the phasor layers' default 0.15/period at Q = 21
    @test pe.leakage ≈ -π / (21 * sqrt(1 - 1 / (4 * 21^2))) rtol = 1e-5
    @test pe.t_period ≈ 100.0 rtol = 1e-3          # damped period, just above 1/f₀
    @test pe.t_period > 100.0
    @test_throws ArgumentError RFTransponder(Q = 0.4)
    @test occursin("RFTransponder", sprint(show, c))
end

# Free ringdown from (v, i) = (0, i0):  i(t) = i0·e^{−αt}(cos ωt + (α/ω) sin ωt),
# α = 1/(2RC). Exact discretisation ⇒ agreement to roundoff even at a coarse dt.
function test_device_exact_ringdown()
    c = _DEV_CELL
    pe = phasor_equivalent(c)
    α, ω = -pe.λ, pe.ω
    i0 = 5f0                                     # well below I_th
    for dt in (0.5, 5.0)
        n = round(Int, 600 / dt)
        s0 = merge(cell_state(c), (; i = fill(i0)))
        tr = cell_simulate(c, t -> 0f0, dt, n; s0 = s0)
        ia = @. i0 * exp(-α * tr.t) * (cos(ω * tr.t) + (α / ω) * sin(ω * tr.t))
        @test maximum(abs.(vec(tr.i) .- ia)) < 1e-4 * i0
        @test isempty(tr.spike_time)
    end
    # cell_phasor rotates counter-clockwise at ω (the phasor layers' convention)
    s0 = merge(cell_state(c), (; i = fill(i0)))
    tr = cell_simulate(c, t -> 0f0, 0.5, 800; s0 = s0)
    z = complex.(vec(tr.i), .-vec(tr.v) ./ sqrt(c.L / c.C))
    ϕ = unwrap_phase(angle.(z))
    slope = (ϕ[end] - ϕ[1]) / (tr.t[end] - tr.t[1])
    @test slope ≈ ω rtol = 2e-3
end

unwrap_phase(p) = (q = copy(p); for k in 2:length(q)
                       d = q[k] - q[k-1]; q[k:end] .-= 2π * round(d / 2π)
                   end; q)

# Current drive A·sin(ω₀t) at resonance: the loop current settles to |i| = Q·A.
function test_device_threshold_reset_refractory()
    c = _DEV_CELL
    ω0 = 2π * resonance_frequency(c) * 1e-3
    dt = 1.0
    # Q·A = 10.5 µA < I_th: settles, never fires, and reaches the predicted gain.
    sub = cell_simulate(c, t -> 0.5f0 * sin(ω0 * t), dt, 6000)
    @test isempty(sub.spike_time)
    @test maximum(abs.(sub.i[end-200:end])) ≈ 21 * 0.5 rtol = 0.03

    # Q·A = 42 µA > I_th: fires repeatedly.
    sup = cell_simulate(c, t -> 2f0 * sin(ω0 * t), dt, 6000)
    ts = sup.spike_time
    @test length(ts) >= 4
    isi = diff(ts)
    @test minimum(isi) >= c.t_ref                       # absolute refractory holds
    # After reset the tank rebuilds from zero: |i| ≈ QA(1−e^{−αt}) reaches I_th at
    # t = −ln(1 − I_th/QA)/α ≈ 433 ps, so ISI ≈ t_ref + 433. Loose: ringing phase.
    α = -phasor_equivalent(c).λ
    t_build = -log(1 - c.I_th / (21 * 2)) / α
    @test isapprox(mean(isi), c.t_ref + t_build; rtol = 0.15)
    # The loop is held at zero through the refractory window after each spike.
    k = round(Int, ts[2] / dt) + 1
    @test all(abs.(sup.i[k+1:k+round(Int, 0.9 * c.t_ref / dt)]) .== 0f0)
end

function test_device_resonance_selectivity()
    c = _DEV_CELL
    f0 = resonance_frequency(c) * 1e-3
    on  = cell_simulate(c, t -> 2f0 * sin(2π * f0 * t), 1.0, 3000)
    off = cell_simulate(c, t -> 2f0 * sin(2π * 0.7 * f0 * t), 1.0, 3000)
    @test !isempty(on.spike_time)
    @test isempty(off.spike_time)
    # off-resonance gain is QA/√(1 + Q²(r − 1/r)²) with r = 0.7 — about a quarter
    r = 0.7
    g_off = 21 * 2 / sqrt(1 + 21^2 * (r - 1 / r)^2) / r
    @test maximum(abs.(off.i[end-500:end])) ≈ g_off rtol = 0.1
end

# Spike times are interpolated inside the step, so they must not move with dt.
function test_device_spike_time_resolution()
    c = _DEV_CELL
    ω0 = 2π * resonance_frequency(c) * 1e-3
    first_spike(dt) = cell_simulate(c, t -> 2f0 * sin(ω0 * t), dt, round(Int, 1500 / dt)).spike_time[1]
    t_ref_sol = first_spike(0.05)
    @test abs(first_spike(0.25) - t_ref_sol) < 0.1
    @test abs(first_spike(2.0) - t_ref_sol) < 0.5          # 1/50 of a period, at 2 ps steps
end

# A cell started above threshold fires on its first step; its output is the
# alpha pulse I_out·(t/τ)e^{1−t/τ}: peak I_out at t = τ, charge I_out·e·τ.
function test_device_output_pulse()
    c = _DEV_CELL
    dt = 0.25
    s0 = merge(cell_state(c), (; i = fill(1.5f0 * c.I_th)))
    tr = cell_simulate(c, t -> 0f0, dt, round(Int, 400 / dt); s0 = s0)
    @test length(tr.spike_time) == 1
    @test tr.spike_time[1] ≈ 0f0 atol = 1e-6
    out = vec(tr.out)
    @test maximum(out) ≈ c.I_out rtol = 0.01
    @test tr.t[argmax(out)] ≈ c.tau_out atol = 2dt
    @test sum(out) * dt ≈ c.I_out * ℯ * c.tau_out rtol = 0.01
    @test cell_output(c, tr.state) == tr.state.o2
end

# Whole-array stepping: sites are independent, the step is pure, and a 2-D bank
# reproduces per-site scalar runs exactly.
function test_device_array_semantics()
    c = _DEV_CELL
    ω0 = 2π * resonance_frequency(c) * 1e-3
    amps = Float32[0.5 1.0 2.0; 3.0 0.0 1.5]
    bank = cell_simulate(c, t -> amps .* sin(ω0 * t), 1.0, 2000; dims = size(amps))
    @test size(bank.i) == (2, 3, 2000)
    for ci in CartesianIndices(amps)
        one = cell_simulate(c, t -> amps[ci] * sin(ω0 * t), 1.0, 2000)
        @test vec(bank.i[ci, :]) == vec(one.i)
        @test sort(bank.spike_time[bank.spike_index .== Ref(ci)]) == one.spike_time
    end
    stp = RFStepper(c, 1.0)
    s = merge(cell_state(c, 2, 3), (; i = fill(5f0, 2, 3)))
    snap = deepcopy(s)
    cell_step(stp, s, 1f0)
    @test s == snap                                      # no mutation
end

# time_scaled is an exact symmetry: the scaled cell under a scaled drive, stepped
# at a scaled dt, reproduces the original trajectory sample for sample.
function test_device_time_scaling()
    c = _DEV_CELL; sc = 3.0
    cs = time_scaled(c, sc)
    @test resonance_frequency(cs) ≈ resonance_frequency(c) / sc rtol = 1e-5
    @test quality_factor(cs) ≈ quality_factor(c) rtol = 1e-5
    ω0 = 2π * resonance_frequency(c) * 1e-3
    f(t) = 2f0 * sin(ω0 * t)
    a = cell_simulate(c, f, 1.0, 3000)
    b = cell_simulate(cs, t -> f(t / sc), sc, 3000)
    @test maximum(abs.(a.i .- b.i)) < 1e-3 * maximum(abs.(a.i))
    @test b.spike_time ≈ sc .* a.spike_time rtol = 1e-4
end

# A stepper built from a matrix of cells steps each site exactly as that cell's
# own scalar stepper would.
function test_device_heterogeneous_stepper()
    c1, c2 = _DEV_CELL, time_scaled(_DEV_CELL, 2.5)
    cells = [isodd(y + x) ? c1 : c2 for y in 1:3, x in 1:4]
    stp = RFStepper(cells, 1.0)
    @test stp.p11 isa Matrix{Float32} && size(stp.p11) == (3, 4)
    s = merge(cell_state(c1, 3, 4), (; i = fill(15f0, 3, 4), v = fill(30f0, 3, 4)))
    sh, fh, _ = cell_step(stp, s, 4f0)
    for (st1, cc) in ((RFStepper(c1, 1.0), c1), (RFStepper(c2, 1.0), c2))
        sx, fx, _ = cell_step(st1, s, 4f0)
        m = cells .== Ref(cc)
        @test sh.i[m] == sx.i[m] && sh.v[m] == sx.v[m] && fh[m] == fx[m]
    end
    @test stepper_to(RFStepper(c1, 1.0), identity) isa RFStepper{Float32}
end

# `force` makes an RF cell fire at a given sub-step time whatever its state.
function test_device_force()
    c = _DEV_CELL
    stp = RFStepper(c, 1.0)
    s = cell_state(c, 3)
    s1, f, tf = cell_step(stp, s, 0f0; force = [false, true, false], t_force = [0f0, 0.25f0, 0f0])
    @test f == [false, true, false]
    @test tf[2] == 0.25f0
    @test s1.ref[2] ≈ c.t_ref - 0.75f0
    @test s1.o1[2] > 0 && s1.o1[1] == 0
end

# ---- NTronTransponder (the spcnc reference device) -------------------------

const _NT_CELL = NTronTransponder()          # toy_example.py defaults

# One step of the loop is the exact solution of the reference equations with
# the nTron resistances frozen — checked against a Float64 matrix exponential
# in both eigen-regimes (real pair; complex pair near R_nw ≈ R_T ≈ R_norm).
function test_ntron_exact_loop_step()
    c = _NT_CELL
    for (Rnw, RT, dt) in ((0.0, 3.0, 5.0), (500.0, 3.0, 2.0), (100.0, 1003.0, 5.0), (1000.0, 1003.0, 1.0))
        (a11, a12, a21, a22), (c1, c2) = PhasorNetworks._ntron_system(
            Float64.((c.Lnw, c.Ls1, c.LT, c.Rs1))..., Rnw, RT, Float64(c.Ib1))
        A = [a11 a12; a21 a22]; cv = [c1, c2]
        x0 = [7.0, 2.0]
        xs = -(A \ cv)
        ref = xs .+ exp(A .* dt) * (x0 .- xs)
        y = PhasorNetworks._ntron_loop_step(Float32.((x0[1], x0[2], c.Lnw, c.Ls1, c.LT, c.Rs1, Rnw, RT, c.Ib1, dt))...)
        @test maximum(abs.(collect(y) .- ref)) < 1e-4
    end
    # rest (all bias in the channel, empty loop) is a fixed point; nothing fires
    tr = _ntron_run(c, t -> 0f0, 2.0, 20_000)
    @test maximum(abs.(tr.inw .- c.Ib1)) < 1e-3 && maximum(abs.(tr.iloop)) < 1e-4
    @test isempty(tr.out_t) && isempty(tr.in_t)
    @test loop_time_constant(c) ≈ 54_049 rtol = 1e-3      # slow mode of the quiescent loop, ps
    @test occursin("NTronTransponder", sprint(show, c))
    @test_throws ArgumentError NTronTransponder(RT = 0)
end

# Run one cell under a gate-current drive; returns traces and click times (ps).
function _ntron_run(c, drive, dt, n; force_at = Float32[])
    stp = NTronStepper(c, dt)
    s = cell_state(c, 1)
    inw = zeros(Float32, n); il = similar(inw); out = similar(inw); in_t = Float32[]; out_t = Float32[]
    for k in 0:n-1
        t0 = Float32(k * dt)
        fk = [any(t -> t0 <= t < t0 + dt, force_at)]
        tk = [sum(t - t0 for t in force_at if t0 <= t < t0 + dt; init = 0f0)]
        s, f, tf = cell_step(stp, s, Float32(drive(t0 + dt / 2)); force = fk, t_force = tk)
        s.age_in[1] <= dt && push!(in_t, t0 + dt - s.age_in[1])   # the input clock restarted this step
        f[1] && push!(out_t, t0 + tf[1])
        inw[k+1] = s.inw[1]; il[k+1] = s.iloop[1]; out[k+1] = s.out[1]
    end
    return (; inw, iloop = il, out, in_t, out_t, s)
end

gate_pulse(t0, amp; w = 2000) = t -> (t0 <= t < t0 + w) ? amp : 0f0

# The cell is a leaky integrator: a gate pulse above I1_sw − Ib1 clicks the
# input nTron once and parks current in the loop, which then leaks out with the
# loop's slow time constant; a pulse below that margin does nothing; a held
# gate re-clicks once per refractory period; clicks accumulate until the
# output nTron fires and dumps the loop.
function test_ntron_integrate_and_fire()
    c = NTronTransponder(Ib2 = 10)                      # output out of reach: watch the loop
    margin = c.I1_sw - c.Ib1
    sub = _ntron_run(c, gate_pulse(1000, margin - 0.5f0), 2.0, 10_000)
    @test isempty(sub.in_t) && maximum(abs.(sub.iloop)) < 1e-4
    one = _ntron_run(c, gate_pulse(1000, margin + 1f0), 2.0, 60_000)
    @test length(one.in_t) == 1 && isempty(one.out_t)
    @test 1000 <= one.in_t[1] < 1010                      # fires as the gate steps past the margin
    q = one.iloop
    k1, k2 = 10_000, 30_000                               # 20 ns and 60 ns, well after the click
    τ = 2 * (k2 - k1) / log(q[k1] / q[k2])
    @test q[k1] > 0.5f0
    @test τ ≈ loop_time_constant(c) rtol = 0.1            # leaks with the slow mode (plus the fast one's tail)
    # After a click the channel current is near zero and recovers over tens of
    # ns, so a gate just above the RESTING margin cannot re-trigger soon…
    slow = _ntron_run(c, t -> t >= 1000 ? margin + 1f0 : 0f0, 2.0, 6_000)
    @test length(slow.in_t) == 1
    # …while a gate above I1_sw on its own re-clicks once per refractory period.
    held = _ntron_run(c, t -> t >= 1000 ? c.I1_sw + 1f0 : 0f0, 2.0, 6_000)
    @test length(held.in_t) >= 3
    @test all(diff(held.in_t) .>= c.t_ref - 2)           # absolute refractory between clicks

    # integration to threshold: one click parks up to ≈1.45 µA in the loop, two
    # clicks 5 ns apart ≈2.06 µA (the second deposits less: the channel it diverts
    # has not recovered). With I2_sw − Ib2 = 1.75 µA, two fire the output and one does not.
    c2 = NTronTransponder(Ib2 = 20 - 1.75)
    g = c2.I1_sw + 1f0                                    # clicks whatever the channel current
    single = _ntron_run(c2, gate_pulse(1000, g), 2.0, 15_000)
    @test length(single.in_t) == 1 && isempty(single.out_t)
    double = _ntron_run(c2, t -> gate_pulse(1000, g)(t) + gate_pulse(6000, g)(t), 2.0, 15_000)
    @test length(double.in_t) == 2
    @test length(double.out_t) == 1 && double.out_t[1] > 6000
    k = round(Int, (double.out_t[1] + 1000) / 2)          # 1 ns after the output click
    @test double.iloop[k] < 0.5f0 * maximum(double.iloop) # the output nTron's hotspot dumped the loop
end

# The emitted pulse is the reference OutputPulse, i_b2·split·σ((s−t_lead)/μ)e^{−(s−t_lead)/τ},
# and a second pulse after the refractory period adds to the first's tail.
function test_ntron_output_pulse()
    c = _NT_CELL
    amp = c.Ib2 * c.split
    shape(s) = s < 0 ? 0.0 : exp(-(s - c.t_lead) / c.tau_out) / (1 + exp(-(s - c.t_lead) / c.mu_out))
    dt = 1.0
    t1, t2 = 1000.3f0, 5200.7f0
    n = 40_000                                            # 40 ns: both tails fully decayed
    tr = _ntron_run(NTronTransponder(I2_sw = 1e6), t -> 0f0, dt, n; force_at = [t1, t2])
    @test tr.out_t ≈ [t1, t2] atol = 1e-3
    ts = (1:n) .* dt
    ref = amp .* (shape.(ts .- t1) .+ shape.(ts .- t2))
    @test maximum(abs.(tr.out .- ref)) < 1e-3 * amp
    @test sum(tr.out) * dt ≈ 2 * amp * sum(shape.(0:0.1:60_000)) * 0.1 rtol = 1e-3
end

# A heterogeneous array steps each cell as its own scalar stepper does, and the
# array form of cell_state takes each cell's own bias.
function test_ntron_heterogeneous_stepper()
    c1, c2 = _NT_CELL, NTronTransponder(Ib1 = 12, I1_sw = 13.5, Rs1 = 3)
    cells = [isodd(i) ? c1 : c2 for i in 1:6]
    s = cell_state(cells)
    @test s.inw == Float32[c.Ib1 for c in cells]
    stp = NTronStepper(cells, 2.0)
    @test stp.Ib1 isa Vector{Float32}
    sh, fh, _ = cell_step(stp, s, 5f0)
    for cc in (c1, c2)
        sx, fx, _ = cell_step(NTronStepper(cc, 2.0), s, 5f0)
        m = cells .== Ref(cc)
        @test sh.inw[m] == sx.inw[m] && sh.iloop[m] == sx.iloop[m] && sh.age_in[m] == sx.age_in[m]
    end
    @test stepper_to(NTronStepper(c1, 1.0), identity) isa NTronStepper{Float32}
end

function device_tests()
    @testset "Device cells (NTronTransponder)" begin
        test_ntron_exact_loop_step()
        test_ntron_integrate_and_fire()
        test_ntron_output_pulse()
        test_ntron_heterogeneous_stepper()
    end
    @testset "Device cells (RFTransponder)" begin
        @info "Running device cell tests..."
        test_device_construction()
        test_device_exact_ringdown()
        test_device_threshold_reset_refractory()
        test_device_resonance_selectivity()
        test_device_spike_time_resolution()
        test_device_output_pulse()
        test_device_array_semantics()
        test_device_time_scaling()
        test_device_heterogeneous_stepper()
        test_device_force()
    end
end
