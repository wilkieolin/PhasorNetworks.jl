# test/test_wave.jl
#
# Tests for PhasorWaveSheet — the recurrent resonate-and-fire sheet with
# delayed difference-of-Gaussians coupling (traveling waves). Run via
# wave_tests().
#
# Coverage:
#   1. Forward sanity (Phase 3D dispatch: shape, eltype, range, finiteness)
#   2. Dispersion / criticality (spectral radius monotone in g; a critical
#      gain giving spectral radius ≈ 1 exists and is findable)
#   3. Wave propagation (seeded pulse spreads: RMS radius increases)
#   4. Inhibition ablation (B_inh=0 shifts the fastest-growing mode toward DC)
#   5. Gradient flow through every trainable parameter (Zygote AD)
#   6. wave_simulate batch handling + autonomous vs driven
#   7. Emission threshold θ + homeostatic regulation (the :spike firing threshold)
#   8. Spectral occupancy → transport forecast (report §4.6: a code's own
#      spatial-frequency content decides how far it survives)

function wave_tests()
    @testset "PhasorWaveSheet" begin
        @info "Running PhasorWaveSheet tests..."
        test_wave_forward_sanity()
        test_wave_dispersion_criticality()
        test_wave_transport_regime()
        test_wave_propagation()
        test_wave_inhibition_ablation()
        test_wave_gradient_flow()
        test_wave_simulate_shapes()
        test_wave_scan_equivalence()
        test_wave_experts()
        test_wave_state_experts()
        test_wave_spike_deq()
        test_wave_expert_layer()
        test_wave_expert_decode()
        test_wave_route_features()
        test_wave_aniso_coupling()
        test_wave_shift_coupling()
        test_wave_spectral_transport()
        # ---- Tier 2: continuous ODE mode (needs the ODE stack from runtests.jl)
        test_wave_continuous_dispersion()
        test_wave_ode_equivalence()
        test_wave_currentcall()
        test_wave_chain_integration()
        test_wave_stencil_coupling()
        test_wave_spike_transmission()
        test_wave_emission_threshold()
        test_soliton_wave_sheet()
    end
end

# ---- SolitonWaveSheet: conservative (unitary) nonlinear wave sheet -----

function test_soliton_wave_sheet()
    @testset "SolitonWaveSheet (conservative + Kerr)" begin
        rng = Xoshiro(21)

        # (a) Phase-3D forward: shape, eltype, range, finiteness
        H = W = 10; L = 6; B = 2
        l = SolitonWaveSheet(H, W; init_log_D = log(1.0), init_beta = 0.2)
        ps, st = Lux.setup(rng, l)
        x = Phase.(2f0 .* rand(rng, Float32, H * W, L, B) .- 1f0)
        y, st_out = l(x, ps, st)
        @test size(y) == (H * W, L, B)
        @test eltype(y) === Phase
        @test all(isfinite, Float32.(y))
        @test all(-1f0 - 1f-4 .<= Float32.(y) .<= 1f0 + 1f-4)
        @test st_out === st

        # (b) autonomous recurrence is UNITARY: ‖z‖ conserved to ~machine precision
        Hn = Wn = 24
        ln = SolitonWaveSheet(Hn, Wn; init_log_D = log(1.0), init_beta = 0.4)
        pn, sn = Lux.setup(Xoshiro(3), ln)
        z0 = zeros(ComplexF32, Hn, Wn)
        for j in 1:Wn, i in 1:Hn
            z0[i, j] = ComplexF32(exp(-(((i-8)^2)+((j-12)^2))/(2*3f0^2))) * cis(0.5f0*(i-8))
        end
        traj = soliton_simulate(ln, pn, sn; z0 = z0, L = 40)
        @test size(traj) == (Hn, Wn, 40)
        n0 = sum(abs2, z0)
        ns = [sum(abs2, traj[:, :, t]) for t in 1:40]
        @test maximum(abs.(ns .- n0)) / n0 < 1f-3           # norm conserved

        # (c) Kerr self-trapping: β>0 arrests dispersion (narrower packet at distance)
        #     1-D stripe (uniform in y ⇒ stable 1-D soliton), stationary, matched β.
        Hs = 64; Ws = 4; Ls = 30; σ0 = 2f0
        seed = zeros(ComplexF32, Hs, Ws)
        for j in 1:Ws, i in 1:Hs
            seed[i, j] = ComplexF32(exp(-((i - Hs÷2)^2) / (2f0 * σ0^2)))   # peak |z|=1
        end
        width(β) = begin
            lb = SolitonWaveSheet(Hs, Ws; init_log_D = log(1.0), init_beta = β)
            pb, sb = Lux.setup(Xoshiro(5), lb)
            Y = soliton_simulate(lb, pb, sb; z0 = seed, L = Ls)
            I = vec(sum(abs2.(Y[:, :, Ls]); dims = 2))
            xs = Float32.(0:Hs-1); c = sum(xs .* I) / sum(I)
            sqrt(sum(((xs .- c) .^ 2) .* I) / sum(I))                        # RMS width
        end
        w_lin = width(0f0); w_kerr = width(0.25f0)
        @test w_kerr < w_lin                                # nonlinearity self-traps

        # (d) gradient flow through every trainable param
        xg = Phase.(2f0 .* rand(Xoshiro(7), Float32, H * W, L, B) .- 1f0)
        loss(p) = sum(abs2, Float32.(first(l(xg, p, st))))
        val, grads = Zygote.withgradient(loss, ps)
        g = grads[1]
        @test isfinite(val)
        for name in keys(ps)
            @test haskey(g, name) && all(isfinite, g[name])
        end
        @test any(any(abs.(g[name]) .> 0) for name in keys(ps))
    end
end

# ---- 1. Forward sanity ------------------------------------------------

function test_wave_forward_sanity()
    @testset "Phase 3D forward shape + range" begin
        rng = Xoshiro(0)
        H, W, L, B = 8, 8, 6, 3
        layer = PhasorWaveSheet(H, W)         # library defaults: transmit=:spike, no snap
        ps, st = Lux.setup(rng, layer)

        x = Phase.(2f0 .* rand(rng, Float32, H * W, L, B) .- 1f0)
        y, st_out = layer(x, ps, st)

        @test size(y) == (H * W, L, B)
        @test eltype(y) === Phase
        @test all(isfinite, Float32.(y))
        @test all(Float32.(y) .>= -1f0 .- 1f-4)
        @test all(Float32.(y) .<=  1f0 .+ 1f-4)
        @test st_out === st
    end
end

# ---- 2. Dispersion / criticality --------------------------------------

function test_wave_dispersion_criticality()
    @testset "dispersion + criticality knob" begin
        rng = Xoshiro(3)
        H = W = 16
        layer = PhasorWaveSheet(H, W; saturating = false)
        ps, st = Lux.setup(rng, layer)

        d = dispersion(layer, ps, st)
        @test size(d.M) == (H, W)
        @test size(d.k_eff) == (H, W)
        @test d.spectral_radius > 0
        @test all(isfinite, abs.(d.M))

        setg(g) = merge(ps, (log_g = Float32[log(g)],))
        sr(g) = dispersion(layer, setg(g), st).spectral_radius

        # Spectral radius grows with gain (coupling eventually dominates decay).
        @test sr(4f0) > sr(1f0) > sr(0.25f0)

        # A critical gain (spectral radius ≈ 1) exists and is findable by
        # geometric bisection — the phase-SSM branching-ratio-≈-1 point.
        lo, hi = 1f-3, 1f2
        for _ in 1:40
            mid = sqrt(lo * hi)
            sr(mid) < 1f0 ? (lo = mid) : (hi = mid)
        end
        g_crit = sqrt(lo * hi)
        @test isapprox(sr(g_crit), 1f0; atol = 1f-2)
    end
end

# ---- 3. Wave propagation ----------------------------------------------

function test_wave_transport_regime()
    @testset "transport diagnostic: standing vs traveling" begin
        H = W = 48
        # The design rule: c = 2σ_I/T puts the delay phase across the surround at π.
        @test matched_conduction_speed(3.0, 1.0) ≈ 6.0f0
        @test matched_conduction_speed(6.0, 1.0) ≈ 12.0f0
        @test matched_conduction_speed(3.0, 0.5) ≈ 12.0f0

        crit(l, ps, st) = begin
            sr(g) = dispersion(l, merge(ps, (log_g = Float32[log(g)],)), st).spectral_radius
            lo, hi = 1f-5, 1f3
            for _ in 1:50; m = sqrt(lo*hi); sr(m) < 1 ? (lo = m) : (hi = m); end
            merge(ps, (log_g = Float32[log(sqrt(lo*hi))],))
        end
        mk(c) = PhasorWaveSheet(H, W; transmit = :potential, saturating = false,
                                init_log_speed = log(c))

        # Weak delay (the old default): the gain peak sits on the phase-band
        # extremum, so the selected mode carries ~none of the band's transport.
        lw = mk(40.0); pw, sw = Lux.setup(Xoshiro(1), lw)
        tw = wave_transport(lw, crit(lw, pw, sw), sw)
        @test tw.delay_phase < 0.3 * pi
        @test tw.transport_ratio < 0.15
        @test tw.verdict === :standing

        # Matched delay (the derived default): selected mode ≈ fastest-transporting.
        lm = PhasorWaveSheet(H, W; transmit = :potential, saturating = false)
        pm, sm = Lux.setup(Xoshiro(1), lm)
        @test exp(pm.log_speed[1]) ≈ 6.0f0 rtol=1f-5           # derived, not 40
        tm = wave_transport(lm, crit(lm, pm, sm), sm)
        @test tm.delay_phase ≈ Float32(pi) rtol=1f-3
        @test tm.transport_ratio > 0.5
        @test tm.verdict === :traveling
        @test abs(tm.v_r_star) > 20 * abs(tw.v_r_star)          # ~0.72 vs ~0.002

        # radial_band: the vector average is identically zero at every |q| — the
        # trap the radial projection avoids — while v_r itself is not.
        rb = radial_band(lm, crit(lm, pm, sm), sm)
        @test length(rb.q) == length(rb.v_r) == length(rb.v_vec)
        @test maximum(rb.v_vec) < 1f-5
        @test maximum(abs.(rb.v_r)) > 0.5
        @test all(isfinite, rb.gain)
    end
end

function test_wave_propagation()
    @testset "seeded pulse spreads (traveling wave)" begin
        rng = Xoshiro(5)
        H = W = 24
        L = 30
        # linear-wave propagation physics → use potential coupling explicitly
        layer = PhasorWaveSheet(H, W; transmit = :potential, saturating = false,
                                init_A_exc = 1.0, init_B_inh = 0.25,
                                init_log_sigma_exc = log(1.5),
                                init_log_sigma_inh = log(3.0), init_log_speed = log(40.0))
        ps, st = Lux.setup(rng, layer)

        # Operate near criticality so the disturbance neither dies instantly
        # nor blows up before it can spread.
        setg(g) = merge(ps, (log_g = Float32[log(g)],))
        sr(g) = dispersion(layer, setg(g), st).spectral_radius
        lo, hi = 1f-3, 1f2
        for _ in 1:40
            mid = sqrt(lo * hi); sr(mid) < 1f0 ? (lo = mid) : (hi = mid)
        end
        g_crit = sqrt(lo * hi)

        seed = zeros(ComplexF32, H, W); seed[1, 1] = 1f0 + 0f0im
        traj = wave_simulate(layer, setg(g_crit), st; z0 = seed, L = L)
        @test size(traj) == (H, W, L)
        @test all(isfinite, traj)

        rgrid = st.rgrid
        rms(t) = sqrt(sum(abs2.(traj[:, :, t]) .* rgrid .^ 2) /
                      (sum(abs2.(traj[:, :, t])) + 1f-20))
        # Activity starts pinned at the seed (radius ≈ 0) and spreads outward.
        @test rms(1) < rms(L)
        @test rms(L) > 1f0
    end
end

# ---- 4. Inhibition ablation -------------------------------------------

function test_wave_inhibition_ablation()
    @testset "remove inhibition → fastest mode toward DC" begin
        rng = Xoshiro(7)
        H = W = 32
        # Spatially-balanced DoG (A_exc·σ_E² ≈ B_inh·σ_I²) + moderate delay:
        # selects a nonzero wavelength. Removing inhibition → pure excitatory
        # Gaussian → fastest growth collapses to DC (uniform front).
        layer = PhasorWaveSheet(H, W; saturating = false,
                                init_A_exc = 1.0, init_B_inh = 0.25,
                                init_log_sigma_exc = log(1.5),
                                init_log_sigma_inh = log(3.0),
                                init_log_speed = log(40.0))
        ps, st = Lux.setup(rng, layer)

        d_full  = dispersion(layer, ps, st)
        ps_noinh = merge(ps, (B_inh = Float32[0f0],))
        d_noinh = dispersion(layer, ps_noinh, st)

        # Distance of the fastest-growing spatial mode from DC (index (1,1)).
        function peak_q(d)
            gr = real.(d.k_eff)
            idx = argmax(gr)              # CartesianIndex over (H,W), origin at (1,1)
            di = idx[1] - 1; di = di > H ÷ 2 ? di - H : di
            dj = idx[2] - 1; dj = dj > W ÷ 2 ? dj - W : dj
            return sqrt(Float32(di)^2 + Float32(dj)^2)
        end

        # Mexican hat selects a nonzero wavelength; pure excitation peaks at DC.
        @test peak_q(d_full) > peak_q(d_noinh)
        @test peak_q(d_noinh) <= 1.5f0     # excitation-only → near-uniform front
    end
end

# ---- 5. Gradient flow -------------------------------------------------

function test_wave_gradient_flow()
    @testset "gradient flow through all params" begin
        rng = Xoshiro(11)
        H, W, L, B = 8, 8, 5, 2
        layer = PhasorWaveSheet(H, W; transmit = :potential, saturating = true, use_adaptation = true)
        ps, st = Lux.setup(rng, layer)
        x = Phase.(2f0 .* rand(rng, Float32, H * W, L, B) .- 1f0)

        loss(p) = begin
            y, _ = layer(x, p, st)
            sum(abs2, Float32.(y))
        end
        val, grads = Zygote.withgradient(loss, ps)
        g = grads[1]
        @test isfinite(val)
        for name in keys(ps)
            @test haskey(g, name)
            @test all(isfinite, g[name])
            # At least one parameter should get a nonzero gradient.
        end
        @test any(any(abs.(g[name]) .> 0) for name in keys(ps))
    end
end

# ---- 6. wave_simulate shapes ------------------------------------------

function test_wave_simulate_shapes()
    @testset "wave_simulate batch + driven" begin
        rng = Xoshiro(13)
        H = W = 10; L = 8; B = 3
        layer = PhasorWaveSheet(H, W; saturating = false)
        ps, st = Lux.setup(rng, layer)

        # Unbatched z0 → (H,W,L)
        z0 = zeros(ComplexF32, H, W); z0[1,1] = 1f0
        t1 = wave_simulate(layer, ps, st; z0 = z0, L = L)
        @test size(t1) == (H, W, L)

        # Batched z0 → (H,W,L,B)
        z0b = zeros(ComplexF32, H, W, B); z0b[1,1,:] .= 1f0
        t2 = wave_simulate(layer, ps, st; z0 = z0b, L = L)
        @test size(t2) == (H, W, L, B)

        # Driven (per-step complex drive), autonomous z0 = 0
        drive = zeros(ComplexF32, H, W, L); drive[H÷2, W÷2, 1] = 1f0
        t3 = wave_simulate(layer, ps, st; z0 = zeros(ComplexF32, H, W), L = L, drive = drive)
        @test size(t3) == (H, W, L)
        @test all(isfinite, t3)
    end
end

# ---- Tier 2: continuous ODE mode --------------------------------------

function test_wave_continuous_dispersion()
    @testset "continuous dispersion" begin
        rng = Xoshiro(2)
        H = W = 16
        layer = PhasorWaveSheet(H, W; saturating = false)
        ps, st = Lux.setup(rng, layer)

        dc = dispersion(layer, ps, st; mode = :continuous)
        dd = dispersion(layer, ps, st; mode = :discrete)
        @test size(dc.M) == (H, W)
        @test size(dc.k_eff) == (H, W)
        @test dc.spectral_radius > 0
        @test all(isfinite, abs.(dc.M))
        @test all(isfinite, real.(dc.k_eff))
        # Continuous eigenvalue is the exact operator k + g·Ŵ; discrete is the
        # operator-split log(A + g·Ŵ)/T. Different discretizations, both valid.
        @test dc.spectral_radius != dd.spectral_radius
        @test_throws ArgumentError dispersion(layer, ps, st; mode = :bogus)
    end
end

function test_wave_ode_equivalence()
    @testset "discrete ≈ ODE (same dynamics)" begin
        rng = Xoshiro(4)
        H = W = 16; L = 10
        # linear operator equivalence → potential coupling explicitly
        layer = PhasorWaveSheet(H, W; transmit = :potential, saturating = false,
                                init_A_exc = 1.0, init_B_inh = 0.25,
                                init_log_sigma_exc = log(1.5),
                                init_log_sigma_inh = log(3.0), init_log_speed = log(40.0))
        ps, st = Lux.setup(rng, layer)

        # Subcritical (continuous) so neither path diverges over the window.
        setg(g) = merge(ps, (log_g = Float32[log(g)],))
        srC(g) = dispersion(layer, setg(g), st; mode = :continuous).spectral_radius
        lo, hi = 1f-3, 1f2
        for _ in 1:40
            m = sqrt(lo * hi); srC(m) < 1f0 ? (lo = m) : (hi = m)
        end
        g_crit = sqrt(lo * hi)
        psg = setg(0.85f0 * g_crit)

        seed = zeros(ComplexF32, H, W); seed[1, 1] = 1f0
        td = wave_simulate(layer, psg, st; z0 = seed, L = L, mode = :discrete)
        to = wave_simulate(layer, psg, st; z0 = seed, L = L, mode = :ode)
        @test size(to) == (H, W, L)
        @test all(isfinite, to)

        # The operator-split discrete recurrence and the ODE integrate the same
        # dz/dt = k·z + g·(coupling); their fields stay strongly aligned.
        a = vec(td[:, :, 5]); b = vec(to[:, :, 5])
        corr = abs(sum(a .* conj.(b))) / (sqrt(sum(abs2, a)) * sqrt(sum(abs2, b)) + 1f-20)
        @test corr > 0.9
    end
end

function test_wave_currentcall()
    @testset "CurrentCall ODE dispatch + gradient" begin
        rng = Xoshiro(6)
        H = W = 8; B = 2
        layer = PhasorWaveSheet(H, W; transmit = :potential, saturating = false)
        ps, st = Lux.setup(rng, layer)

        drive = zeros(Float32, H * W, B); drive[H * W ÷ 2, :] .= 1f0
        cc = CurrentCall(LocalCurrent(t -> drive, (H * W, B), 0f0),
                         layer.spk_args, (0f0, 3f0))

        y, st_out = layer(cc, ps, st)
        @test size(y) == (H * W, 3, B)          # (H*W, L, B), L = 3 periods
        @test eltype(y) === Phase
        @test all(isfinite, Float32.(y))
        @test st_out === st

        # SpikingCall trampolines through CurrentCall.
        # Gradient through the ODE adjoint requires ComponentArray params.
        psc = ComponentArray(ps)
        loss(p) = begin
            yy, _ = layer(cc, p, st)
            sum(abs2, Float32.(yy))
        end
        val, grads = Zygote.withgradient(loss, psc)
        @test isfinite(val)
        @test all(isfinite, grads[1])
        @test any(abs.(grads[1]) .> 0)
    end
end

# ---- Chain integration (the FashionMNIST-demo composition) ------------
#
# Mirrors demos/wave_fashionmnist.jl on tiny random data: image → sheet drive
# → PhasorWaveSheet → last step → PhasorDense head → Codebook similarities.
# Confirms the sheet composes with WrappedFunction / PhasorDense / Codebook
# and that gradients reach both the wave coupling and the trainable head.

function test_wave_chain_integration()
    @testset "Chain integration (wave classifier)" begin
        rng = Xoshiro(21)
        H = W = 8; B = 4; L = 4
        model = Chain(
            WrappedFunction(x -> begin
                b = size(x, 3)
                ph = Phase.((2f0 .* x .- 1f0) .* 0.5f0)
                repeat(reshape(ph, H * W, 1, b), 1, L, 1)
            end),
            PhasorWaveSheet(H, W; init_log_g = log(0.02)),   # default spike mode
            WrappedFunction(x -> x[:, end, :]),
            PhasorDense(H * W => 16, normalize_to_unit_circle),
            Codebook(16 => 10; init_mode = :orthogonal),
        )
        ps, st = Lux.setup(rng, model)
        x = rand(Float32, H, W, B)

        y, _ = model(x, ps, st)
        @test size(y) == (10, B)
        @test all(isfinite, y)

        val, gs = Zygote.withgradient(p -> sum(abs2, first(model(x, p, st))), ps)
        @test isfinite(val)
        @test all(isfinite, gs[1].layer_2.log_g)         # wave coupling gets gradient
        @test any(abs.(gs[1].layer_4.weight) .> 0)       # trainable head gets gradient
    end
end

# ---- Learnable coupling stencil (bookmark 2) --------------------------

function test_wave_stencil_coupling()
    @testset "learnable stencil coupling" begin
        rng = Xoshiro(31)
        H = W = 12; L = 5; B = 3; R = 2
        layer = PhasorWaveSheet(H, W; coupling = :stencil, stencil_radius = R,
                                init_log_g = log(0.05))   # default spike mode
        ps, st = Lux.setup(rng, layer)

        # Param/state layout: free complex stencil + constant placement matrix.
        @test haskey(ps, :stencil_re) && haskey(ps, :stencil_im)
        @test size(ps.stencil_re) == (2R + 1, 2R + 1)
        @test !haskey(ps, :A_exc)                        # DoG scalars absent in :stencil
        @test haskey(st, :place) && size(st.place) == (H * W, (2R + 1)^2)

        # Forward + dispersion behave like the DoG mode.
        x = Phase.(2f0 .* rand(rng, Float32, H * W, L, B) .- 1f0)
        y, _ = layer(x, ps, st)
        @test size(y) == (H * W, L, B)
        @test all(isfinite, Float32.(y))
        d = dispersion(layer, ps, st)
        @test d.spectral_radius > 0 && all(isfinite, abs.(d.M))

        # Gradient reaches the full stencil (real capacity, not just 9 scalars).
        val, gs = Zygote.withgradient(p -> sum(abs2, Float32.(first(layer(x, p, st)))), ps)
        g = gs[1]
        @test isfinite(val)
        @test all(isfinite, g.stencil_re) && all(isfinite, g.stencil_im)
        @test any(abs.(g.stencil_re) .> 0) && any(abs.(g.stencil_im) .> 0)

        # Guard: stencil too large for the sheet.
        @test_throws ArgumentError PhasorWaveSheet(4, 4; coupling = :stencil, stencil_radius = 3)
    end
end

# ---- Spike transmission (BIBO stability without the snap) --------------
#
# transmit=:spike sends unit-magnitude z/|z| instead of the full potential, so
# the coupling drive is hard-bounded and the leaky-integrator state is
# BIBO-stable with NO state snap — while |z| survives (not phase-only). This is
# the physically-motivated alternative to saturating=true.

function test_wave_spike_transmission()
    @testset "spike transmission (BIBO, magnitude-preserving)" begin
        rng = Xoshiro(41)
        S = 16; L = 24; B = 2

        # Forward sanity, Phase 3D.
        layer = PhasorWaveSheet(S, S; transmit = :spike, saturating = false,
                                init_log_g = log(0.3))
        ps, st = Lux.setup(rng, layer)
        x = Phase.(2f0 .* rand(rng, Float32, S*S, L, B) .- 1f0)
        y, _ = layer(x, ps, st)
        @test size(y) == (S*S, L, B)
        @test eltype(y) === Phase
        @test all(isfinite, Float32.(y))

        # BIBO: at a gain that makes the *linear* (potential) sheet blow up, the
        # spike sheet stays bounded — and its magnitude varies cell-to-cell
        # (it is NOT phase-only).
        z0 = zeros(ComplexF32, S, S); z0[S÷2, S÷2] = 1f0
        ghi = log(1.0)                                  # way supercritical for the linear sheet
        spike = PhasorWaveSheet(S, S; transmit = :spike, saturating = false, init_log_g = ghi)
        pot   = PhasorWaveSheet(S, S; transmit = :potential, saturating = false, init_log_g = ghi)
        psp, ssp = Lux.setup(rng, spike); ppo, spo = Lux.setup(rng, pot)
        tsp = wave_simulate(spike, psp, ssp; z0 = z0, L = L)
        tpo = wave_simulate(pot,   ppo, spo; z0 = z0, L = L)
        @test all(isfinite, tsp)
        max_spike = maximum(abs.(tsp))
        max_pot   = maximum(abs.(tpo))
        @test max_spike < 1f3                           # bounded (no runaway)
        @test max_spike < max_pot                       # linear sheet explodes past it
        @test std(abs.(tsp[:, :, end])) > 1f-2          # magnitude varies (not phase-only)

        # Gradient flows to the coupling under spike transmission.
        val, gs = Zygote.withgradient(p -> sum(abs2, Float32.(first(layer(x, p, st)))), ps)
        @test isfinite(val)
        @test all(isfinite, gs[1].log_g) && any(abs.(gs[1].A_exc) .> 0)

        # Guard: invalid transmit mode.
        @test_throws ArgumentError PhasorWaveSheet(8, 8; transmit = :bogus)
    end
end

# ---- Parallel scan equivalence (§5.8 #1) ------------------------------
#
# The linear sheet (transmit=:potential, no snap/adaptation) is a diagonal SSM
# per spatial mode, so the sequential Buffer recurrence (`_wave_rollout`) and the
# parallel FFT-in-time forward (`_wave_rollout_scan`, exposed as
# wave_simulate mode=:scan) must produce identical trajectories. This pins that
# equivalence and checks gradients flow through the parallel path.

function test_wave_scan_equivalence()
    @testset "parallel scan ≈ discrete recurrence (linear sheet)" begin
        rng = Xoshiro(11)
        H = W = 16; L = 12; B = 3
        layer = PhasorWaveSheet(H, W; transmit = :potential, saturating = false,
                                init_A_exc = 1.0, init_B_inh = 0.25,
                                init_log_sigma_exc = log(1.5),
                                init_log_sigma_inh = log(3.0),
                                init_log_speed = log(40.0))
        ps, st = Lux.setup(rng, layer)

        # Subcritical g (discrete spectral radius < 1) so neither path diverges.
        setg(g) = merge(ps, (log_g = Float32[log(g)],))
        sr(g) = dispersion(layer, setg(g), st; mode = :discrete).spectral_radius
        lo, hi = 1f-3, 1f2
        for _ in 1:40
            m = sqrt(lo * hi); sr(m) < 1f0 ? (lo = m) : (hi = m)
        end
        psg = setg(0.7f0 * sqrt(lo * hi))

        relerr(a, b) = maximum(abs.(a .- b)) / (maximum(abs.(a)) + 1f-20)

        # (a) autonomous, batched: seed one impulse per batch element. The
        # parallel scan must match the sequential recurrence to FFT round-off.
        z0 = zeros(ComplexF32, H, W, B)
        for b in 1:B
            z0[rand(rng, 1:H), rand(rng, 1:W), b] = 1f0
        end
        td = wave_simulate(layer, psg, st; z0 = z0, L = L, mode = :discrete)
        ts = wave_simulate(layer, psg, st; z0 = z0, L = L, mode = :scan)
        @test size(ts) == (H, W, L, B)
        @test all(isfinite, ts)
        @test relerr(td, ts) < 1f-5

        # (b) driven.
        drive = 0.1f0 .* randn(rng, ComplexF32, H, W, L)
        z0s = zeros(ComplexF32, H, W)
        dd = wave_simulate(layer, psg, st; z0 = z0s, L = L, drive = drive, mode = :discrete)
        ds = wave_simulate(layer, psg, st; z0 = z0s, L = L, drive = drive, mode = :scan)
        @test size(ds) == (H, W, L)
        @test relerr(dd, ds) < 1f-5

        # (c) gradients flow through the parallel scan path.
        loss(p) = sum(abs2, abs.(wave_simulate(layer, p, st; z0 = z0, L = L, mode = :scan)))
        val, gs = Zygote.withgradient(loss, psg)
        @test isfinite(val)
        @test any(any(abs.(gs[1][name]) .> 0) for name in keys(psg))

        # (d) the scan refuses the nonlinear regimes.
        spikelayer = PhasorWaveSheet(H, W; transmit = :spike)
        psk, stk = Lux.setup(rng, spikelayer)
        @test_throws ArgumentError wave_simulate(spikelayer, psk, stk; z0 = z0s,
                                                 L = L, mode = :scan)
    end
end

# ---- Sparse experts: gate + input-conditioned bind (§5.3/§5.4/§5.8) ---
#
# Exercises the input-conditioned expert path: a straight-through top-1 router
# (`moe_gate`) with DeepSeek loss-free per-expert bias, the drive-modulating bind
# (`_apply_wave_experts`), and the end-to-end composition through the parallel
# scan with gradients reaching the bind phasors and router logits.

function test_wave_experts()
    @testset "sparse experts: gate + input-conditioned bind" begin
        rng = Xoshiro(23)
        E = 4; Lt = 6; Bt = 3

        # moe_gate: hard straight-through one-hot.
        logits = randn(rng, Float32, E, Lt, Bt)
        bias0 = zeros(Float32, E)
        gh = moe_gate(logits, bias0; hard = true)
        @test size(gh) == (E, Lt, Bt)
        @test all(isapprox.(sum(gh; dims = 1), 1f0; atol = 1f-4))          # one-hot per (n,b)
        @test all(x -> isapprox(x, 0f0; atol = 1f-4) || isapprox(x, 1f0; atol = 1f-4), gh)
        v, g = Zygote.withgradient(l -> sum(moe_gate(l, bias0; hard = true)), logits)
        @test g[1] !== nothing && all(isfinite, g[1])                      # grad via soft surrogate

        # soft gate is a softmax.
        gsoft = moe_gate(logits, bias0; hard = false)
        @test all(isapprox.(sum(gsoft; dims = 1), 1f0; atol = 1f-4))

        # bias steers selection only.
        lz = zeros(Float32, E, 1, 1); lz[1] = 0.1f0    # expert 1 has the max logit
        b = zeros(Float32, E); b[3] = 10f0             # but bias favours expert 3
        @test argmax(vec(moe_gate(lz, b; hard = true))) == 3

        # DeepSeek loss-free load-balancing update.
        gate_load = zeros(Float32, E, Lt, Bt); gate_load[1, :, :] .= 1f0   # expert 1 overused
        b1 = update_moe_bias(zeros(Float32, E), gate_load; rate = 1f-2)
        @test b1[1] < 0f0                              # overused → bias down
        @test all(b1[2:end] .> 0f0)                    # underused → bias up

        # drive modulation: off vs on.
        H = W = 8
        drive = randn(rng, ComplexF32, H, W, Lt, Bt)
        masks = zeros(Float32, H, W, E)
        for e in 1:E; masks[e, e, e] = 1f0; end        # disjoint single-site patches
        phis = ComplexF32.(cis.(Float32.(range(0.3, 1.2; length = E))))
        maskflat = reshape(masks, H * W, E)
        d_off = PhasorNetworks._apply_wave_experts(drive, maskflat, phis,
                                                   zeros(Float32, E, Lt, Bt))
        @test maximum(abs.(d_off .- drive)) < 1f-6     # gate off ⇒ unchanged
        d_on = PhasorNetworks._apply_wave_experts(drive, maskflat, phis,
                                                  ones(Float32, E, Lt, Bt))
        @test all(isapprox.(d_on[1, 1, :, :], drive[1, 1, :, :] .* phis[1]; atol = 1f-4))
        @test all(isapprox.(d_on[5, 5, :, :], drive[5, 5, :, :]; atol = 1f-6))  # unowned site

        # end-to-end: router → bind → parallel scan, gradients to expert params.
        layer = PhasorWaveSheet(H, W; transmit = :potential, saturating = false)
        ps, st = Lux.setup(rng, layer)
        setg(g) = merge(ps, (log_g = Float32[log(g)],))
        sr(g) = dispersion(layer, setg(g), st; mode = :discrete).spectral_radius
        lo, hi = 1f-3, 1f2
        for _ in 1:40
            m = sqrt(lo * hi); sr(m) < 1f0 ? (lo = m) : (hi = m)
        end
        psg = setg(0.7f0 * sqrt(lo * hi))
        z0 = zeros(ComplexF32, H, W, Bt)
        function eloss(θ)
            logit, phi = θ
            gate = moe_gate(logit, bias0; hard = true)
            d = PhasorNetworks._apply_wave_experts(drive, maskflat, phi, gate)
            Y = PhasorNetworks._wave_rollout_scan(layer, psg, st, z0, d, Lt)
            sum(abs2, abs.(Y))
        end
        θ0 = (randn(rng, Float32, E, Lt, Bt), phis)
        val, gr = Zygote.withgradient(eloss, θ0)
        @test isfinite(val)
        @test all(isfinite, gr[1][1])                  # ∂/∂logits
        @test all(isfinite, gr[1][2])                  # ∂/∂phis
        @test any(abs.(gr[1][2]) .> 0)                 # bind phasors receive gradient
    end
end

# ---- State-conditioned experts: chunked operator split (§5.4/§5.8 #4) --
#
# The event-based operator split: linear transport runs as a parallel scan within
# each chunk, a state-conditioned gate+bind fires at chunk boundaries. Anchors on
# the exactness guarantee (identity event ⇒ chunked == full scan for any chunk
# size) and checks a real state-conditioned expert trains through the feedback.

function test_wave_state_experts()
    @testset "state-conditioned experts: chunked operator split" begin
        rng = Xoshiro(29)
        H = W = 12; L = 12; B = 2; E = 3
        layer = PhasorWaveSheet(H, W; transmit = :potential, saturating = false)
        ps, st = Lux.setup(rng, layer)
        setg(g) = merge(ps, (log_g = Float32[log(g)],))
        sr(g) = dispersion(layer, setg(g), st; mode = :discrete).spectral_radius
        lo, hi = 1f-3, 1f2
        for _ in 1:40
            m = sqrt(lo * hi); sr(m) < 1f0 ? (lo = m) : (hi = m)
        end
        psg = setg(0.7f0 * sqrt(lo * hi))

        z0 = zeros(ComplexF32, H, W, B)
        for b in 1:B; z0[rand(rng, 1:H), rand(rng, 1:W), b] = 1f0; end
        drive = 0.05f0 .* randn(rng, ComplexF32, H, W, L, B)
        relerr(a, b) = maximum(abs.(a .- b)) / (maximum(abs.(a)) + 1f-20)

        # (a) operator-split exactness: identity event ⇒ chunked == full scan.
        ref = PhasorNetworks._wave_rollout_scan(layer, psg, st, z0, drive, L)
        for ch in (1, 3, 4, 5, L)
            ci = PhasorNetworks._wave_rollout_chunked(layer, psg, st, z0, drive, L;
                                                      chunk = ch, interact = (z, c) -> z)
            @test size(ci) == (H, W, L, B)
            @test relerr(ref, ci) < 1f-4
        end

        # State-conditioned gate+bind interact, built from moe_gate + state read.
        masks = zeros(Float32, H, W, E)
        for e in 1:E                                   # disjoint 2-site patches
            r = 2 * e; c = 2 * e
            masks[r, c, e] = 1f0; masks[r, c + 1, e] = 1f0
        end
        maskflat = reshape(masks, H * W, E)
        phis = ComplexF32.(cis.(Float32.(range(0.4, 1.3; length = E))))
        bias0 = zeros(Float32, E)
        w0 = ones(Float32, E)
        mk_interact(w, phi) = (z, c) -> begin
            r = PhasorNetworks._state_read(z, maskflat)                   # (E,B) complex
            gate = moe_gate(w .* abs.(r), bias0)                          # (E,B) state-cond.
            zb = PhasorNetworks._apply_wave_experts(reshape(z, H, W, 1, B),
                                                    maskflat, phi, reshape(gate, E, 1, B))
            reshape(zb, H, W, B)
        end

        # (b) nontrivial: the state-conditioned events perturb the linear field.
        traj = PhasorNetworks._wave_rollout_chunked(layer, psg, st, z0, drive, L;
                                                    chunk = 4, interact = mk_interact(w0, phis))
        @test all(isfinite, traj)
        @test relerr(ref, traj) > 1f-3

        # (c) gradients flow to router weights + bind phasors through the feedback.
        function loss(θ)
            w, phi = θ
            Y = PhasorNetworks._wave_rollout_chunked(layer, psg, st, z0, drive, L;
                                                     chunk = 4, interact = mk_interact(w, phi))
            sum(abs2, abs.(Y))
        end
        val, gr = Zygote.withgradient(loss, (w0, phis))
        @test isfinite(val)
        @test all(isfinite, gr[1][1]) && all(isfinite, gr[1][2])
        @test any(abs.(gr[1][2]) .> 0)                 # bind phasors get gradient
    end
end

# ---- Spike DEQ: phase-domain fixed-point trainer (§5.6) ---------------
#
# Spike mode as a fixed point s* = emit(linear_response(s*)): parallel sweeps
# with a pointwise emit between them. Anchors on exactness (n_sweeps == L equals
# the sequential spike rollout), checks settling for fewer sweeps, that the
# dirac-consistent emission runs and differs from the unit emit, and that
# gradients flow through the sweeps.

function test_wave_spike_deq()
    @testset "spike DEQ fixed-point trainer" begin
        rng = Xoshiro(31)
        H = W = 12; L = 8; B = 2
        layer = PhasorWaveSheet(H, W; transmit = :spike, saturating = false)
        ps, st = Lux.setup(rng, layer)
        # keep it subcritical-ish so the sequential rollout stays bounded
        setg(g) = merge(ps, (log_g = Float32[log(g)],))
        psg = setg(0.15f0)

        z0 = zeros(ComplexF32, H, W, B)
        for b in 1:B; z0[rand(rng, 1:H), rand(rng, 1:W), b] = 1f0; end
        drive = 0.05f0 .* randn(rng, ComplexF32, H, W, L, B)
        relerr(a, b) = maximum(abs.(a .- b)) / (maximum(abs.(a)) + 1f-20)

        # sequential spike rollout (the reference).
        ref = PhasorNetworks._wave_rollout(layer, psg, st, z0, drive, L)

        # (a) exactness: n_sweeps == L reproduces the sequential rollout.
        deqL = PhasorNetworks._wave_rollout_deq(layer, psg, st, z0, drive, L;
                                                n_sweeps = L, emit_mode = :unit)
        @test size(deqL) == (H, W, L, B)
        @test relerr(ref, deqL) < 1f-4

        # (b) settling: the causal fixed point converges monotonically in sweeps.
        e2 = relerr(ref, PhasorNetworks._wave_rollout_deq(layer, psg, st, z0, drive, L;
                                                          n_sweeps = 2, emit_mode = :unit))
        e5 = relerr(ref, PhasorNetworks._wave_rollout_deq(layer, psg, st, z0, drive, L;
                                                          n_sweeps = 5, emit_mode = :unit))
        @test e2 > e5 >= 0f0                            # more sweeps ⇒ closer

        # via the public API.
        wd = wave_simulate(layer, psg, st; z0 = z0, L = L, drive = drive,
                           mode = :deq, n_sweeps = L)
        @test relerr(ref, wd) < 1f-4

        # (c) dirac-consistent emission runs, is finite, and differs from unit.
        deqd = PhasorNetworks._wave_rollout_deq(layer, psg, st, z0, drive, L;
                                                n_sweeps = L, emit_mode = :dirac)
        @test all(isfinite, deqd)
        @test relerr(deqL, deqd) > 1f-3                 # sub-cycle leak ⇒ different dynamics

        # (d) gradients flow through the parallel sweeps.
        loss(p) = sum(abs2, abs.(PhasorNetworks._wave_rollout_deq(layer, p, st, z0, drive, L;
                                                                  n_sweeps = 4, emit_mode = :unit)))
        val, gs = Zygote.withgradient(loss, psg)
        @test isfinite(val)
        @test any(any(abs.(gs[1][name]) .> 0) for name in keys(psg))
    end
end

# ---- §6 prototype: WaveExpertSheet ------------------------------------
#
# The read-patch → gate → re-bind Lux layer trained on the discrete SSM. Checks
# both routing modes (input/state) on both substrates, that gradients reach the
# sheet coupling + router + bind phasors, the route_stats go/no-go readout, and
# the online DeepSeek bias plumbing.

function test_wave_expert_layer()
    @testset "WaveExpertSheet prototype (§6)" begin
        rng = Xoshiro(37)
        H = W = 8; L = 5; B = 3; E = 4
        mkx() = Phase.(2f0 .* rand(rng, Float32, H * W, L, B) .- 1f0)

        # input routing on both substrates: forward + gradients to all params.
        for tr in (:potential, :spike)
            layer = WaveExpertSheet(H, W; n_experts = E, routing = :input, transmit = tr,
                                    saturating = false, n_sweeps = L, init_log_g = log(0.1))
            ps, st = Lux.setup(rng, layer)
            x = mkx()
            y, st2 = layer(x, ps, st)
            @test size(y) == (H * W, L, B)
            @test eltype(y) === Phase
            @test all(isfinite, Float32.(y))
            @test all(-1f0 - 1f-4 .<= Float32.(y) .<= 1f0 + 1f-4)
            @test length(st2.route_bias) == E && all(isfinite, st2.route_bias)  # online balance
            gl(p) = sum(abs2, Float32.(first(layer(x, p, st))))
            v, g = Zygote.withgradient(gl, ps)
            @test isfinite(v)
            @test any(abs.(g[1].Wr) .> 0)                       # router learns
            @test any(abs.(g[1].bind_phase) .> 0)               # bind phasors learn
            @test any(any(abs.(g[1].sheet[name]) .> 0) for name in keys(ps.sheet))
        end

        # route_stats go/no-go readout.
        layer = WaveExpertSheet(H, W; n_experts = E, routing = :input,
                                transmit = :potential, init_log_g = log(0.1))
        ps, st = Lux.setup(rng, layer)
        x = mkx()
        rs = route_stats(layer, ps, st, x)
        @test length(rs.load) == E
        @test isapprox(sum(rs.load), 1f0; atol = 1f-4)          # soft occupancy is a distribution
        @test 0f0 <= rs.entropy <= log(Float32(E)) + 1f-3       # entropy in [0, log E]
        @test size(rs.gate) == (E, L, B)

        # state routing (potential substrate): forward + gradients.
        slayer = WaveExpertSheet(H, W; n_experts = E, routing = :state, chunk = 2,
                                 transmit = :potential, balance = false, init_log_g = log(0.1))
        sps, sst = Lux.setup(rng, slayer)
        ys, _ = slayer(x, sps, sst)
        @test size(ys) == (H * W, L, B)
        @test all(isfinite, Float32.(ys))
        vs, gs = Zygote.withgradient(p -> sum(abs2, Float32.(first(slayer(x, p, sst)))), sps)
        @test isfinite(vs)
        @test any(abs.(gs[1].bind_phase) .> 0)

        # state routing requires a linear substrate.
        @test_throws ArgumentError WaveExpertSheet(H, W; routing = :state, transmit = :spike)
    end
end

# ---- §6 step 3: decode check — stamped bind is recoverable -------------
#
# Confirms the expert's stamped bind φ_e is a recoverable VSA operation: unbinding
# the sheet output against a bind-free reference recovers the code (differential
# decode), robustly even under an incoherent carrier; while a blind decode (no
# reference) is coherence-limited — the square-law channel from the readout study.

function test_wave_expert_decode()
    @testset "decode check: stamped bind recoverable (§6 step 3)" begin
        rng = Xoshiro(43)
        H = W = 8; E = 6; L = 3; B = 1
        layer = WaveExpertSheet(H, W; n_experts = E, routing = :input,
                                transmit = :potential, init_log_g = log(0.1))
        ps, st = Lux.setup(rng, layer)
        codes = Float32.(collect(range(-1f0, 1f0, length = E + 1))[1:E])
        ps = merge(ps, (bind_phase = codes,))
        phis = ComplexF32.(cis.(Float32(pi) .* codes))
        acc(dec) = sum(e -> argmax([cos(Float32(pi) * (dec[e] - codes[ep])) for ep in 1:E]) == e, 1:E) / E

        function run(csigma)
            x = Phase.(clamp.(csigma .* randn(rng, Float32, H * W, L, B), -1f0, 1f0))
            drive = reshape(PhasorNetworks.angle_to_complex(x), H, W, L, B)
            z0 = zeros(ComplexF32, H, W, B)
            d2 = PhasorNetworks._apply_wave_experts(drive, st.masks, phis, ones(Float32, E, L, B))
            Y  = PhasorNetworks._wave_rollout_scan(layer.sheet, ps.sheet, st.sheet, z0, d2, L)
            Yr = PhasorNetworks._wave_rollout_scan(layer.sheet, ps.sheet, st.sheet, z0, drive, L)
            r  = vec(PhasorNetworks._patch_read(Y[:, :, end, :],  st.masks))
            r0 = vec(PhasorNetworks._patch_read(Yr[:, :, end, :], st.masks))
            return acc(angle.(r .* conj.(r0)) ./ Float32(pi)),      # differential (unbind ref)
                   acc(angle.(r) ./ Float32(pi))                    # blind (carrier assumed 0)
        end

        # coherent carrier: both decodes recover every expert.
        d0, b0 = run(0f0)
        @test d0 == 1f0
        @test b0 == 1f0
        # incoherent carrier: differential still recovers; blind is coherence-limited.
        dd, bb = run(1f0)
        @test dd >= 0.66f0                              # stamp recoverable via unbind
        @test bb <= dd + 1f-6                           # blind no better (needs coherence)

        # row-band tiling requires E ≤ H.
        @test_throws ArgumentError WaveExpertSheet(4, 4; n_experts = 8)
    end
end

# ---- §6 step 4: phase-domain routing features -------------------------
#
# The router summarises each patch by a phase-domain feature rather than raw
# amplitude. Checks the matched-filter ("is this my key?") read mechanism, that
# each of :coherence/:dispersion/:matched runs forward + differentiates to the
# right parameters, and that :matched is phase-selective (fires on a key match,
# is suppressed by the anti-phase).

function test_wave_route_features()
    @testset "phase-domain routing features (§6 step 4)" begin
        rng = Xoshiro(71)
        H = W = 8; L = 4; B = 2; E = 4
        mkx() = Phase.(2f0 .* rand(rng, Float32, H * W, L, B) .- 1f0)

        # (0) constructor validation.
        @test_throws ArgumentError WaveExpertSheet(H, W; route_feature = :bogus)

        # (1) matched-filter read mechanism (internal): unbinding a patch by its own
        #     key gives Re(ρ) = +1 on an exact phase match, −1 on the anti-phase.
        masks = PhasorNetworks._tile_masks(H, W, E)                       # (HW,E)
        key   = 2f0 .* rand(rng, Float32, H * W, E) .- 1f0               # (HW,E) per-site key
        fmatch = reshape(PhasorNetworks.angle_to_complex(key[:, 1]), H, W)  # field = expert-1 key
        ρm = PhasorNetworks._patch_read_keyed(fmatch, masks, key)         # (E,)
        @test real(ρm[1]) > 0.99f0                                        # exact match on patch 1
        ρa = PhasorNetworks._patch_read_keyed(-fmatch, masks, key)        # −f ≡ +π (anti-phase)
        @test real(ρa[1]) < -0.99f0                                       # anti-phase suppressed
        # key === nothing reduces to the plain pooled read.
        @test PhasorNetworks._patch_read_keyed(fmatch, masks, nothing) ≈
              PhasorNetworks._patch_read(fmatch, masks)

        # (2) each feature: bind_key iff :matched, forward + shape + finite,
        #     gradients to the router (and to the keys for :matched), route_stats OK.
        for feat in (:coherence, :dispersion, :matched)
            layer = WaveExpertSheet(H, W; n_experts = E, routing = :input,
                                    route_feature = feat, transmit = :potential,
                                    init_log_g = log(0.1))
            ps, st = Lux.setup(rng, layer)
            @test haskey(ps, :bind_key) == (feat === :matched)
            x = mkx()
            y, _ = layer(x, ps, st)
            @test size(y) == (H * W, L, B)
            @test all(isfinite, Float32.(y))
            @test all(-1f0 - 1f-4 .<= Float32.(y) .<= 1f0 + 1f-4)
            gl(p) = sum(abs2, Float32.(first(layer(x, p, st))))
            v, g = Zygote.withgradient(gl, ps)
            @test isfinite(v)
            @test any(abs.(g[1].Wr) .> 0)                                 # router learns
            feat === :matched && @test any(abs.(g[1].bind_key) .> 0)      # keys learn
            rs = route_stats(layer, ps, st, x)
            @test length(rs.load) == E && isapprox(sum(rs.load), 1f0; atol = 1f-4)
            @test 0f0 <= rs.entropy <= log(Float32(E)) + 1f-3
        end

        # (3) end-to-end selectivity: with keys frozen to the tile pattern, a patch
        #     matching expert e's key routes to e more than a random patch does.
        layer = WaveExpertSheet(H, W; n_experts = E, routing = :input,
                                route_feature = :matched, transmit = :potential,
                                hard = false, init_log_g = log(0.1))
        ps, st = Lux.setup(rng, layer)
        ps = merge(ps, (; Wr = Float32.([i == j for i in 1:E, j in 1:E])))  # identity mix
        # input phase = expert-2's key inside every site (matches patch 2 exactly).
        xk = Phase.(reshape(repeat(ps.bind_key[:, 2], 1, L * B), H * W, L, B))
        gk = route_stats(layer, ps, st, xk).gate                         # (E,L,B) soft
        @test argmax(vec(sum(gk; dims = (2, 3)))) == 2                    # patch-2 expert wins
    end
end

# ---- Anisotropic coupling: directed transport (:aniso) ----------------
#
# The isotropic DoG (:dog) is reflection-symmetric ⇒ waves spread both ways ⇒
# no net transport. `:aniso` adds a real antisymmetric advection stencil whose
# transform is Ŵ_adv(q) = −i(β_h·sin q_h + β_w·sin q_w) — purely imaginary, odd
# in q — giving a net group velocity v = g·β (a drifting packet) while staying
# FFT-diagonal. Checks: β=0 reduces to :dog exactly; the advection term is
# imaginary+odd; a seeded pulse drifts with the sign/magnitude of β; β is
# trainable.

function test_wave_aniso_coupling()
    @testset "anisotropic coupling: directed transport (:aniso)" begin
        H = W = 40
        ω = PhasorNetworks.period_to_angfreq(SpikingArgs().t_period)

        # (1) β=0 ⇒ :aniso W_hat is exactly the :dog W_hat.
        ld = PhasorWaveSheet(H, W; coupling = :dog, transmit = :potential)
        la0 = PhasorWaveSheet(H, W; coupling = :aniso, transmit = :potential,
                              init_beta_h = 0.0, init_beta_w = 0.0)
        pd, sd = Lux.setup(Xoshiro(1), ld)
        pa, sa = Lux.setup(Xoshiro(1), la0)
        _, _, Wd = PhasorNetworks._build_coupling(ld, pd, sd, ω)
        _, _, Wa = PhasorNetworks._build_coupling(la0, pa, sa, ω)
        @test maximum(abs.(Wa .- Wd)) < 1f-6

        # (2) advection term (the β-part) is purely imaginary and odd in q.
        la = PhasorWaveSheet(H, W; coupling = :aniso, transmit = :potential,
                             init_beta_h = 0.6, init_beta_w = 0.0)
        pa2, sa2 = Lux.setup(Xoshiro(1), la)
        _, _, Wa2 = PhasorNetworks._build_coupling(la, pa2, sa2, ω)
        adv = Wa2 .- Wd
        rev = adv[mod.(-(0:H-1), H) .+ 1, mod.(-(0:W-1), W) .+ 1]         # adv(−q)
        @test maximum(abs.(real.(adv))) < 1f-5                            # imaginary
        @test maximum(abs.(adv .+ rev)) < 1f-5                            # odd in q

        # (3) directed drift: seed a centered pulse, roll autonomously, track the
        #     row-centroid. Drift follows the sign of β_h, is ~0 at β=0, monotone.
        ctr = 20
        centroid_row(z) = (w = abs2.(z); sum((1:H) .* vec(sum(w; dims = 2))) / (sum(w) + 1f-12))
        # NOTE the explicit init_log_speed. The β sign convention ("+β_h drifts
        # toward +h") is only well-defined in the WEAK-DELAY regime, where the DoG
        # contributes no group velocity of its own and advection is the sole source
        # of drift. At the matched speed c = 2σ_I/T (now the layer default) the DoG
        # is a backward wave — v_g points opposite to q — so β still biases growth
        # toward +q_h but those modes travel toward −h, and the measured drift
        # inverts (verified: drift(+0.8) = −1.01 at c=6 vs +0.67 at c=40). Pinning c
        # keeps this test on the mechanism it is actually testing.
        function drift(beta)
            l = PhasorWaveSheet(H, W; coupling = :aniso, transmit = :potential,
                                init_beta_h = Float32(beta), init_beta_w = 0.0,
                                init_log_g = log(0.5), saturating = false,
                                init_log_speed = log(40.0))
            p, s = Lux.setup(Xoshiro(2), l)
            z0 = zeros(ComplexF32, H, W)
            for i in ctr-2:ctr+2, j in ctr-2:ctr+2
                z0[i, j] = exp(-((i - ctr)^2 + (j - ctr)^2) / 3f0)
            end
            Y = PhasorNetworks._wave_rollout(l, p, s, reshape(z0, H, W, 1), nothing, 8)
            return centroid_row(Y[:, :, end, 1]) - centroid_row(Y[:, :, 1, 1])
        end
        dp, dz, dn = drift(0.8), drift(0.0), drift(-0.8)
        @test dp > dz > dn                                                # monotone in β
        @test abs(dz) < 0.1                                               # ~no drift at β=0
        @test dp > 0.3 && dn < -0.3                                       # substantial, sign-controlled
        @test isapprox(dp, -dn; atol = 0.15)                             # symmetric ±β (advection)

        # (4) forward + β trainable.
        x = Phase.(2f0 .* rand(Xoshiro(4), Float32, H * W, 4, 2) .- 1f0)
        y, _ = la(x, pa2, sa2)
        @test size(y) == (H * W, 4, 2)
        @test all(isfinite, Float32.(y))
        gl(p) = sum(abs2, Float32.(first(la(x, p, sa2))))
        v, g = Zygote.withgradient(gl, pa2)
        @test isfinite(v)
        @test abs(g[1].beta_h[1]) > 0                                     # drift learns

        # (5) constructor rejects bad coupling.
        @test_throws ArgumentError PhasorWaveSheet(H, W; coupling = :bogus)
    end
end

# ---- Shift coupling: ballistic dispersion-free transport (:shift) -----
#
# A pure shift ramp Ŵ_shift(q)=e^{−i q·s} is a translation: unit gain (no gain
# narrowing), linear phase (constant v_g=s), zero GVD by construction. With a
# strong leak (A≈0) the sheet is a ballistic conveyor that carries a packet at any
# depth, breaking the drift↔dispersion trade-off of :aniso. Checks: flat, non-
# dispersive dispersion (v_g≈s, β₂≈0, gain_curv≈0, marginal); a bump translates
# with ~constant spread; the shift vector is trainable.

function test_wave_shift_coupling()
    @testset "shift coupling: ballistic transport (:shift)" begin
        H = W = 24
        l = PhasorWaveSheet(H, W; coupling = :shift, transmit = :potential,
                            init_shift_h = 1.0, init_shift_w = 0.0,
                            init_log_neg_lambda = log(5.0), init_log_g = log(0.99))
        p, s = Lux.setup(Xoshiro(1), l)
        @test haskey(p, :shift_h) && haskey(p, :shift_w)

        # (1) dispersion: unit-gain, linear-phase, zero-GVD, marginal.
        d = dispersion_diagnostics(l, p, s; mode = :potential)
        @test 0.8 < d.v_g_star < 1.15          # v_g ≈ shift = 1
        @test abs(d.gvd_star) < 0.05           # zero GVD
        @test abs(d.gain_curv_star) < 0.1      # flat gain (no narrowing)
        @test abs(d.growth_star) < 0.05        # marginal / stable

        # (2) ballistic rollout: a bump translates ~1 row/step with ~constant spread.
        crow(z) = (w = abs2.(z); sum((1:H) .* vec(sum(w; dims = 2))) / (sum(w) + 1f-12))
        spread(z) = (w = abs2.(z); m = crow(z);
                     sqrt(sum(((1:H) .- m) .^ 2 .* vec(sum(w; dims = 2))) / (sum(w) + 1f-12)))
        z0 = ComplexF32[exp(-((i - 4)^2 + (j - 12)^2) / 8f0) for i in 1:H, j in 1:W]
        Y = PhasorNetworks._wave_rollout(l, p, s, reshape(z0, H, W, 1), nothing, 12)
        c1 = crow(Y[:, :, 1, 1]); c2 = crow(Y[:, :, end, 1])
        @test c2 - c1 > 9                       # ballistic drift ≈ 11 rows over 12 steps
        @test abs(spread(Y[:, :, end, 1]) - spread(Y[:, :, 1, 1])) < 0.4   # spread ~constant

        # (3) shift is trainable; forward on Phase input is finite.
        x = Phase.(2f0 .* rand(Xoshiro(2), Float32, H * W, 4, 2) .- 1f0)
        y, _ = l(x, p, s)
        @test size(y) == (H * W, 4, 2) && all(isfinite, Float32.(y))
        vv, gg = Zygote.withgradient(pp -> sum(abs2, Float32.(first(l(x, pp, s)))), p)
        @test isfinite(vv) && abs(gg[1].shift_h[1]) > 0

        # (4) shift=0 ⇒ no transport (frozen packet).
        l0 = PhasorWaveSheet(H, W; coupling = :shift, transmit = :potential,
                             init_shift_h = 0.0, init_log_neg_lambda = log(5.0), init_log_g = log(0.99))
        p0, s0 = Lux.setup(Xoshiro(1), l0)
        Y0 = PhasorNetworks._wave_rollout(l0, p0, s0, reshape(z0, H, W, 1), nothing, 12)
        @test abs(crow(Y0[:, :, end, 1]) - crow(Y0[:, :, 1, 1])) < 0.5     # no drift
    end
end

# ---- The symbol's spectrum sets its transport (report §4.6) -----------
#
# `spectral_occupancy` / `transport_forecast` evaluate the band against a code's
# OWN spatial-frequency content, so the report's concentrated-vs-extended claim
# becomes computable rather than hand-drawn. The tests pin the reduction against
# a case with a known exact answer (a plane wave, where the forecast must return
# the band value at that single mode), then the monotone trend the claim rests
# on, then the two degenerate media where the answer is analytic.

function test_wave_spectral_transport()
    @testset "spectral occupancy → transport forecast (§4.6)" begin
        H = W = 48
        l = PhasorWaveSheet(H, W; transmit = :potential)
        p, s = Lux.setup(Xoshiro(1), l)
        d = dispersion_diagnostics(l, p, s; mode = :potential)

        # (1) EXACTNESS on a single mode. A pure plane wave at q_k occupies one
        #     bin, so the forecast must collapse to the band's own value there —
        #     zero spread, vg_mean = v_g(q_k). This is the reduction check: it
        #     fails loudly if the marginal, the q-grid ordering, or the FFT
        #     convention ever drift apart from `dispersion_diagnostics`.
        for k in (3, 5, 8)
            qk = 2f0 * Float32(pi) * k / H
            pw = ComplexF32[cis(qk * (i - 1)) for i in 1:H, j in 1:W]
            so = spectral_occupancy(pw)
            f = transport_forecast(l, p, s, pw; axis = :h, mode = :potential)
            idx = findfirst(q -> isapprox(q, qk; atol = 1f-4), d.q)
            @test isapprox(so.q_dom, qk; atol = 1f-4)
            @test so.q_width < 1f-4
            @test isapprox(f.vg_mean, d.v_g[idx]; atol = 1f-4)
            @test f.vg_spread < 1f-4
            # the 2-D path must agree on a code that only varies along one axis
            fb = transport_forecast(l, p, s, pw; axis = :both, mode = :potential)
            @test isapprox(fb.vg_mean, d.v_g[idx]; atol = 1f-3)
        end
        @test isapprox(sum(spectral_occupancy(randn(Xoshiro(3), Float32, H, W)).p), 1f0;
                       atol = 1f-5)                                   # p is a normalised pmf

        # (1b) ℓ vs w_x — the distinction the forecast turns on. A random FHRR
        #      payload filling the sheet has uniform intensity, so its ENVELOPE
        #      width is sheet-scale while its COHERENCE length is one site; a
        #      smooth field has both large. Scoring survival off w_x would
        #      predict the random payload is the most robust code there is.
        φr = 2f0 .* rand(Xoshiro(9), Float32, W) .- 1f0
        rnd = ComplexF32[cis(Float32(pi) * φr[j]) for i in 1:H, j in 1:W]
        sr = spectral_occupancy(rnd; axis = :w)
        @test sr.w_x > 0.2f0 * W                                      # envelope: sheet-wide
        @test sr.ell <= 2f0                                           # coherence: ~a site
        smoothf = ComplexF32[cis(Float32(pi) * cos(2f0 * Float32(pi) * j / W)) for i in 1:H, j in 1:W]
        @test spectral_occupancy(smoothf; axis = :w).ell > 3f0 * sr.ell

        # (2) THE CLAIM: concentrated ⇒ broadband ⇒ short coherence; extended ⇒
        #     narrowband ⇒ long. Monotone in the code's spatial width.
        ctr = H ÷ 2
        wrapd(a, b, n) = min(abs(a - b), n - abs(a - b))
        disc(r) = ComplexF32[wrapd(i, ctr, H)^2 + wrapd(j, ctr, W)^2 <= r^2 ? 1 : 0
                             for i in 1:H, j in 1:W]
        gabor(sg) = ComplexF32[exp(-((i - ctr)^2 + (j - ctr)^2) / (2f0 * sg^2)) *
                               cis(d.q_star * (i - ctr)) for i in 1:H, j in 1:W]
        delta = (c = zeros(ComplexF32, H, W); c[ctr, ctr] = 1; c)

        codes = [delta, disc(2), disc(4), disc(8), gabor(6.0), gabor(10.0)]
        occ = [spectral_occupancy(c) for c in codes]
        fc = [transport_forecast(l, p, s, c; axis = :h, mode = :potential) for c in codes]

        @test issorted([o.w_x for o in occ])                          # widening ladder
        @test issorted([o.q_width for o in occ]; rev = true)          # ⇒ narrowing spectrum
        @test issorted([f.vg_spread for f in fc]; rev = true)         # ⇒ tighter v_g spread
        @test issorted([f.t_half_pred for f in fc])                   # ⇒ longer survival
        # The contrast is large, not marginal — the whole point of §4.6.
        @test fc[1].vg_spread > 4f0 * fc[end].vg_spread
        @test fc[end].t_half_pred > 10f0 * fc[1].t_half_pred
        @test all(f -> f.limiter in (:gvd, :gain), fc)

        # (3) RIGID CONVEYOR: :shift has |Ŵ|≡1 and exactly linear phase, so both
        #     spreading channels vanish and any code survives indefinitely.
        ls = PhasorWaveSheet(H, W; coupling = :shift, transmit = :potential,
                             init_shift_h = 1.0, init_shift_w = 0.0,
                             init_log_neg_lambda = log(5.0), init_log_g = log(0.99))
        p2, s2 = Lux.setup(Xoshiro(1), ls)
        fs = transport_forecast(ls, p2, s2, disc(4); axis = :h, mode = :potential)
        @test fs.vg_spread < 1f-2 && fs.gamma_spread < 1f-2
        @test isapprox(fs.vg_mean, 1f0; atol = 0.05)                  # drift = the shift
        @test fs.t_half_pred > 100f0                                  # ≫ any diffusive medium
        @test fs.d_half_pred > 100f0

        # (4) ISOTROPIC ⇒ d_half_pred = 0 EXACTLY, and that is the right answer:
        #     a reflection-symmetric band is even, so v̄_g = 0 and the sheet
        #     carries a symbol nowhere however long it stays coherent. The
        #     coherence TIME is still finite and positive.
        fd = fc[4]                                                     # disc(8) on the :dog sheet
        @test abs(fd.vg_mean) < 1f-3
        @test fd.d_half_pred < 1f-2
        @test 0f0 < fd.t_half_pred < Inf32

        # (5) TWO-AXIS: a code whose payload varies across the OTHER axis must be
        #     scored as fragile, and `axis = :h` alone must miss it. This is the
        #     correction the measured benchmark forced — the single-axis forecast
        #     correlated with measured survival at r = +0.12 because it never
        #     looked at the axis the payload actually lived on.
        φrand = 2f0 .* rand(Xoshiro(5), Float32, W) .- 1f0
        smooth_env = Float32[exp(-wrapd(i, ctr, H)^2 / (2f0 * 10f0^2)) for i in 1:H]
        carry(φ) = ComplexF32[smooth_env[i] * cis(Float32(pi) * φ[j]) for i in 1:H, j in 1:W]
        f_h = transport_forecast(l, p, s, carry(φrand); axis = :h, mode = :potential)
        f_b = transport_forecast(l, p, s, carry(φrand); axis = :both, mode = :potential)
        @test f_b.t_half_pred < f_h.t_half_pred            # the column axis binds
        # the same envelope carrying a SMOOTH payload must survive longer
        φsm = Float32[cos(2f0 * Float32(pi) * j / W) for j in 1:W]
        f_sm = transport_forecast(l, p, s, carry(φsm); axis = :both, mode = :potential)
        @test f_sm.t_half_pred > f_b.t_half_pred

        # (6) argument validation: the code has to live on the sheet.
        @test_throws DimensionMismatch transport_forecast(l, p, s, zeros(ComplexF32, H + 2, W))
        @test_throws ArgumentError transport_forecast(l, p, s, disc(4); axis = :bogus)
        @test_throws ArgumentError spectral_occupancy(disc(4); axis = :bogus)
    end
end

# ---- Emission threshold + homeostasis ---------------------------------
#
# `:spike` transmission is `z/√(|z|²+θ²)`. θ used to be a hardcoded numerical
# guard (`√1e-8 = 1e-4`); it is really the sheet's FIRING THRESHOLD, and at that
# value it was ~5 orders of magnitude below the scale the coupling actually
# operates at, so every site above 1e-4 emitted a full spike. These tests pin
# the three things that fixes:
#
#   1. θ has a derived scale, `g·max|Ŵ|`, that tracks `g` exactly.
#   2. Below threshold the spike sheet IS the linear medium at gain `g/θ`, so
#      the whole `dispersion` toolchain becomes valid for it.
#   3. Homeostasis regulates θ to a target firing rate, which is what keeps the
#      sheet in the (only ≈1.3× wide) usable band as `g` moves.
#
# Note on strength: before this test existed, the whole suite passed unchanged
# with θ moved by five orders of magnitude — the spike-mode coverage was all
# shape/finiteness. These assertions are on dynamics.

function test_wave_emission_threshold()
    @testset "emission threshold + homeostasis" begin
        rng = Xoshiro(23)

        # ---- 1. θ_ref = g·max|Ŵ| scales EXACTLY with g -------------------
        base = PhasorWaveSheet(32, 32)
        p0, s0 = Lux.setup(rng, base)
        θ1 = emission_threshold(base, p0, s0)
        for f in (0.3f0, 3f0, 10f0)
            lg = PhasorWaveSheet(32, 32; init_log_g = log(f))
            pg, sg = Lux.setup(rng, lg)
            @test emission_threshold(lg, pg, sg) ≈ f * θ1 rtol=1f-4
        end

        # ---- 2. derived default = init_theta_frac × θ_ref -----------------
        @test exp(only(p0.log_theta)) ≈ 1.4f0 * θ1 rtol=1f-4
        lf = PhasorWaveSheet(32, 32; init_theta_frac = 2.0)
        pf, _ = Lux.setup(rng, lf)
        @test exp(only(pf.log_theta)) ≈ 2f0 * θ1 rtol=1f-4
        le = PhasorWaveSheet(32, 32; init_log_theta = log(3.0))
        pe, _ = Lux.setup(rng, le)
        @test exp(only(pe.log_theta)) ≈ 3f0 rtol=1f-5

        # :potential carries no threshold, so homeostasis has nothing to regulate.
        lp = PhasorWaveSheet(32, 32; transmit = :potential)
        pp, _ = Lux.setup(rng, lp)
        @test !haskey(pp, :log_theta)
        @test_throws ArgumentError PhasorWaveSheet(8, 8; homeostasis = :bogus)
        @test_throws ArgumentError PhasorWaveSheet(8, 8; transmit = :potential,
                                                   homeostasis = :global)

        # ---- 3. subthreshold linearization: spike at θ ≡ potential at g/θ --
        # For |z| ≪ θ the emit is z/θ, so the sheet IS the linear medium at
        # effective gain g/θ. Seed far below threshold and compare rollouts.
        θbig = 200f0
        lsp = PhasorWaveSheet(24, 24; transmit = :spike, init_log_theta = log(θbig))
        lpo = PhasorWaveSheet(24, 24; transmit = :potential,
                              init_log_g = log(1.0 / θbig))
        psp, ssp = Lux.setup(rng, lsp); ppo, spo = Lux.setup(rng, lpo)
        z0 = zeros(ComplexF32, 24, 24); z0[1,1] = 1f-3
        tsp = wave_simulate(lsp, psp, ssp; z0 = z0, L = 20)
        tpo = wave_simulate(lpo, ppo, spo; z0 = z0, L = 20)
        @test maximum(abs.(tsp .- tpo)) / maximum(abs.(tpo)) < 1f-3
        # ...and `dispersion` reports that effective gain, which is what makes
        # radial_band / wave_transport meaningful on a spike sheet at all.
        @test dispersion(lsp, psp, ssp).spectral_radius ≈
              dispersion(lpo, ppo, spo).spectral_radius rtol=1f-4

        # ---- 4. the legacy threshold floods; the derived one does not ------
        N = 48
        zi = zeros(ComplexF32, N, N); zi[1,1] = 1f0
        lold = PhasorWaveSheet(N, N; init_log_theta = log(1f-4))
        pold, sold = Lux.setup(rng, lold)
        Zold = wave_simulate(lold, pold, sold; z0 = zi, L = 60)
        @test mean(abs.(Zold[:, :, 60]) .> 1f-4) > 0.99     # whole sheet ignited

        lnew = PhasorWaveSheet(N, N)
        pnew, snew = Lux.setup(rng, lnew)
        Znew = wave_simulate(lnew, pnew, snew; z0 = zi, L = 400)
        θnew = exp(only(pnew.log_theta))
        @test mean(abs.(Znew[:, :, 400]) .> θnew) < 0.5     # sparse, not flooded
        # Structure survives. Collapse to a spatially uniform sheet is the
        # failure mode a rate-only check cannot see.
        @test std(abs.(Znew[:, :, 400])) > 0.5f0

        # ---- 5. :global regulates the rate and tracks g -------------------
        tr_g = Dict{Float32,Any}()
        for g in (0.1f0, 1f0, 10f0)
            lh = PhasorWaveSheet(N, N; homeostasis = :global, init_log_g = log(g))
            ph, sh = Lux.setup(Xoshiro(5), lh)
            tr_g[g] = wave_homeostat_trace(lh, ph, sh; z0 = zi, L = 500)
        end
        for (g, tr) in tr_g
            @test 0.005f0 < mean(tr.fire[401:500]) < 0.05f0        # target is 0.02
            @test 8f0 < tr.theta_g[end] / g < 16f0                 # θ ∝ g
            @test mean(tr.std_abs[401:500]) /
                  mean(tr.mean_abs[401:500]) > 0.2f0               # still structured
        end
        # A fixed θ cannot do this: it is calibrated at one g and the usable
        # band is only ≈1.3× wide.
        @test tr_g[10f0].theta_g[end] / tr_g[0.1f0].theta_g[end] > 50f0

        # ---- 6. :local builds refractoriness the global term cannot -------
        ll = PhasorWaveSheet(N, N; homeostasis = :local)
        pl, sl = Lux.setup(Xoshiro(5), ll)
        tr_l = wave_homeostat_trace(ll, pl, sl; z0 = zi, L = 500)
        @test tr_l.theta_l_spread[end] > 1.2f0                     # per-site spread built up
        @test tr_g[1f0].theta_l_spread[end] == 1f0                 # :global leaves it flat
        @test 0.005f0 < mean(tr_l.fire[401:500]) < 0.05f0          # still on target

        # ---- 7. gradients reach every new parameter -----------------------
        # L must be long enough for the sheet to charge past θ: below threshold
        # `fire ≡ 0`, so the homeostat parameters get exactly no signal. This is
        # correct (refractoriness has nothing to modulate) but it means short
        # rollouts on a fresh sheet will not train them.
        S = 24; L = 60
        lg2 = PhasorWaveSheet(S, S; homeostasis = :local)
        pg2, sg2 = Lux.setup(Xoshiro(3), lg2)
        x = Phase.(2f0 .* rand(Xoshiro(4), Float32, S*S, L, 2) .- 1f0)
        val, gs = Zygote.withgradient(p -> sum(abs2, Float32.(first(lg2(x, p, sg2)))), pg2)
        @test isfinite(val)
        for k in (:log_theta, :log_eta_g, :log_eta_l, :logit_target, :log_theta_beta)
            gk = getproperty(gs[1], k)
            @test all(isfinite, gk)
            @test any(abs.(gk) .> 0)
        end

        # ---- 8. the DEQ fixed point is incompatible with a moving θ -------
        ld = PhasorWaveSheet(16, 16; homeostasis = :global)
        pd, sd = Lux.setup(rng, ld)
        @test_throws ArgumentError wave_simulate(ld, pd, sd;
            z0 = zeros(ComplexF32, 16, 16), L = 6, mode = :deq)
    end
end
