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

function device_tests()
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
    end
end
