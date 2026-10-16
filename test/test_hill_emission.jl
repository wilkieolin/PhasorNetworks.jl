# Hill emission (per-spike charge q = rⁿ/(rⁿ+θⁿ)) — opt-in charge carried by
# spikes (spiking path) and by `Charged` streams (discrete Dirac path).

function hill_emission_tests()
    @testset "Hill emission" begin
        hill_spike_charge_tests()
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
