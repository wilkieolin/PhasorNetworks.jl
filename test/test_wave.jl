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

function wave_tests()
    @testset "PhasorWaveSheet" begin
        @info "Running PhasorWaveSheet tests..."
        test_wave_forward_sanity()
        test_wave_dispersion_criticality()
        test_wave_propagation()
        test_wave_inhibition_ablation()
        test_wave_gradient_flow()
        test_wave_simulate_shapes()
        # ---- Tier 2: continuous ODE mode (needs the ODE stack from runtests.jl)
        test_wave_continuous_dispersion()
        test_wave_ode_equivalence()
        test_wave_currentcall()
        test_wave_chain_integration()
        test_wave_stencil_coupling()
    end
end

# ---- 1. Forward sanity ------------------------------------------------

function test_wave_forward_sanity()
    @testset "Phase 3D forward shape + range" begin
        rng = Xoshiro(0)
        H, W, L, B = 8, 8, 6, 3
        layer = PhasorWaveSheet(H, W; saturating = true)
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

function test_wave_propagation()
    @testset "seeded pulse spreads (traveling wave)" begin
        rng = Xoshiro(5)
        H = W = 24
        L = 30
        layer = PhasorWaveSheet(H, W; saturating = false,
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
        layer = PhasorWaveSheet(H, W; saturating = true, use_adaptation = true)
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
        layer = PhasorWaveSheet(H, W; saturating = false,
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
        layer = PhasorWaveSheet(H, W; saturating = false)
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
            PhasorWaveSheet(H, W; saturating = true, init_log_g = log(0.02)),
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
                                saturating = true, init_log_g = log(0.05))
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
