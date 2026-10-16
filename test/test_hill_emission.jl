# Hill emission (per-spike charge q = rⁿ/(rⁿ+θⁿ)) — opt-in charge carried by
# spikes (spiking path) and by `Charged` streams (discrete Dirac path).

function hill_emission_tests()
    @testset "Hill emission" begin
        hill_spike_charge_tests()
        hill_charged_dirac_tests()
    end
end

# ---- Step 1: SpikeTrain charges, spike_current, propagation ---------------

function hill_spike_charge_tests()
    @testset "SpikeTrain charges" begin
        sa = SpikingArgs(t_window = 0.01f0)
        rng = Xoshiro(3)
        n = 40
        shape = (4, 3)
        inds = [CartesianIndex(rand(rng, 1:4), rand(rng, 1:3)) for _ in 1:n]
        # include coincident spikes on one input to exercise accumulation
        inds[2] = inds[1]
        tms = rand(rng, Float32, n) .* 3f0
        tms[2] = tms[1]
        q = 0.1f0 .+ rand(rng, Float32, n)

        t0 = SpikeTrain(inds, tms, shape, 0f0)
        t1 = SpikeTrain(inds, tms, shape, 0f0, ones(Float32, n))
        tq = SpikeTrain(inds, tms, shape, 0f0, q)
        @test t0.charges === nothing
        @test t1.charges == ones(Float32, n)
        @test_throws DimensionMismatch SpikeTrain(inds, tms, shape, 0f0, ones(Float32, n - 1))

        # charges = ones ⇒ identical current; charges ⇒ Σ q·kernel
        for t in Float32.(range(0, 3; length = 301))
            c0 = PhasorNetworks.spike_current(t0, t, sa)
            @test PhasorNetworks.spike_current(t1, t, sa) == c0
            cq = PhasorNetworks.spike_current(tq, t, sa)
            ref = zeros(Float32, shape)
            k = PhasorNetworks.raised_cosine_kernel(tms, t, sa.t_window)
            for j in 1:n
                ref[inds[j]] += sa.spk_scale * q[j] * k[j]
            end
            @test cq ≈ ref atol = 1f-6
        end

        # propagation: delay, warmup shift, getindex, get_time, cat/vcat/stack, flatten
        d = PhasorNetworks.delay_train(tq, 0.5f0, 0.1f0)
        @test d.charges == q && d.times == tms .+ 0.5f0
        saw = SpikingArgs(warmup_periods = 1)
        w = PhasorNetworks._apply_warmup_shift(tq, saw)
        @test w.charges == q
        sub = tq[2:3, :]
        sel = [i[1] in 2:3 for i in inds]
        @test sub.charges == q[sel]
        g = PhasorNetworks.get_time(tq, (0.5f0, 2f0))
        gsel = (tms .> 0.5f0) .& (tms .< 2f0)
        @test g.charges == q[gsel]
        v = PhasorNetworks.vcat_trains(tq, t0)
        @test v.charges == vcat(q, ones(Float32, n))
        @test PhasorNetworks.vcat_trains(t0, t0).charges === nothing
        s = PhasorNetworks.stack_trains([tq, tq])
        @test s.charges == vcat(q, q)
        inds3 = [CartesianIndex(1 + (i[1] - 1) % 2, 1 + (i[1] - 1) ÷ 2, i[2]) for i in inds]
        f, _ = Lux.FlattenLayer()(SpikeTrain(inds3, tms, (2, 2, 3), 0f0, q), NamedTuple(), NamedTuple())
        @test f.charges == q && f.shape == (4, 3)

        # ssm_phases_to_train: charges follow the spike order (l outer, (c, b) inner)
        ph = Phase.(2f0 .* rand(rng, Float32, 3, 5, 2) .- 1f0)
        Q = rand(rng, Float32, 3, 5, 2)
        tr = PhasorNetworks.ssm_phases_to_train(ph; spk_args = sa, charges = Q)
        @test PhasorNetworks.ssm_phases_to_train(ph; spk_args = sa).charges === nothing
        lin = LinearIndices((3, 2))
        for (j, (idx, t)) in enumerate(zip(tr.indices, tr.times))
            l = floor(Int, t / sa.t_period) + 1
            @test tr.charges[j] == Q[idx[1], l, idx[2]]
        end
    end
end

# ---- Step 2: Charged stream, causal_conv_dirac charges, gradients ---------

function hill_charged_dirac_tests()
    @testset "Charged + causal_conv_dirac" begin
        rng = Xoshiro(11)
        Cin, Cout, L, B = 5, 6, 7, 3
        ph = Phase.(2f0 .* rand(rng, Float32, Cin, L, B) .- 1f0)
        W = randn(rng, Float32, Cout, Cin)
        λ = -0.1f0 .- 0.3f0 .* rand(rng, Float32, Cout)
        ω = fill(2f0 * Float32(π), Cout)
        q = 0.05f0 .+ rand(rng, Float32, Cin, L, B)

        Z0 = causal_conv_dirac(ph, W, λ, ω, 1f0)
        @test causal_conv_dirac(ph, W, λ, ω, 1f0; charges = ones(Float32, Cin, L, B)) == Z0
        Zq = causal_conv_dirac(ph, W, λ, ω, 1f0; charges = q)
        @test causal_conv_dirac(Charged(ph, q), W, λ, ω, 1f0) == Zq
        # linearity in the charges: Z(q) = Σ_j contributions scaled by q_j
        Zsum = causal_conv_dirac(ph, W, λ, ω, 1f0; charges = 2f0 .* q)
        @test Zsum ≈ 2f0 .* Zq rtol = 1f-5
        @test_throws DimensionMismatch Charged(ph, q[:, 1:2, :])
        @test_throws DimensionMismatch causal_conv_dirac(ph, W, λ, ω, 1f0; charges = q[:, 1:2, :])

        # finite-difference check of ∂/∂q and ∂/∂W through the charged path
        f(qq, WW) = sum(abs2, causal_conv_dirac(ph, WW, λ, ω, 1f0; charges = qq))
        gq, gW = Zygote.gradient(f, q, W)
        h = 1f-2
        for (i, j) in ((1, 1), (3, 7), (Cout * Cin, 2 * Cin * L + 4))
            e = zeros(Float32, size(q)); e[j] = h
            fd = (Float64(f(q .+ e, W)) - Float64(f(q .- e, W))) / (2h)
            @test isapprox(gq[j], fd; rtol = 2f-2)
            eW = zeros(Float32, size(W)); eW[i] = h
            fdW = (Float64(f(q, W .+ eW)) - Float64(f(q, W .- eW))) / (2h)
            @test isapprox(gW[i], fdW; rtol = 2f-2)
        end

        # hill_charge: value, limits and finite gradients at z = 0
        r = Float32[1f-3, 0.5f0, 1f0, 2f0, 50f0]
        θ = 0.7f0
        @test hill_charge(r, [log(θ)], 1) ≈ r ./ (r .+ θ) rtol = 1f-5
        @test hill_charge(r, [log(θ)], 4) ≈ r .^ 4 ./ (r .^ 4 .+ θ^4) rtol = 1f-4
        Zc = ComplexF32[0 0.3+0.4im; 1e-8im -2]
        Zc3 = reshape(Zc, 2, 2, 1)
        for n in (1, 3, 40)
            g = Zygote.gradient((z, lt) -> sum(PhasorNetworks._hill_emit(z, lt, n, true).q), Zc3, [log(θ)])
            @test all(isfinite, real.(g[1])) && all(isfinite, imag.(g[1])) && all(isfinite, g[2])
            @test g[1][1] == 0   # exactly zero gradient at the origin
        end
        # frozen θ: no gradient reaches log_theta
        gfz = Zygote.gradient(lt -> sum(PhasorNetworks._hill_emit(Zc3, lt, 1, false).q), [log(θ)])[1]
        @test gfz === nothing || all(iszero, gfz)
        # gradient of the phase through a Charged emission is finite at z = 0
        gp = Zygote.gradient(z -> sum(Float32.(PhasorNetworks._hill_emit(z, [0f0], 1, true).phase)), Zc3)[1]
        @test all(isfinite, real.(gp)) && all(isfinite, imag.(gp))
    end
end
