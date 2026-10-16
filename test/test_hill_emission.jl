# Hill emission (per-spike charge q = rⁿ/(rⁿ+θⁿ)) — opt-in charge carried by
# spikes (spiking path) and by `Charged` streams (discrete Dirac path).

function hill_emission_tests()
    @testset "Hill emission" begin
        hill_spike_charge_tests()
        hill_charged_dirac_tests()
        hill_dense_tests()
    end
end

const HILL_SPK = SpikingArgs(t_window = 0.01f0, threshold = 1f-3, solver = Tsit5(),
                             solver_args = Dict(:adaptive => false, :dt => 0.005,
                                                :sensealg => BacksolveAdjoint(; autojacvec = ZygoteVJP()),
                                                :save_start => true))

_hill_call(x::AbstractArray{<:Phase, 3}; charges = nothing, sa = HILL_SPK) =
    SpikingCall(PhasorNetworks.ssm_phases_to_train(x; spk_args = sa, charges = charges), sa,
                (0f0, Float32(size(x, 2)) * sa.t_period))
_hill_call(x::Charged; sa = HILL_SPK) = _hill_call(x.phase; charges = x.q, sa = sa)

# Discrete (Charged) vs spiking (decoded SpikingCall) agreement:
# circular phase agreement R, mean and 95th-percentile |Δq|.
function _hill_agreement(yd::Charged, ys::SpikingCall)
    dec = ssm_train_to_charged(ys)
    @test size(dec) == size(yd)
    R = abs(mean(cis.(Float32(π) .* (Float32.(dec.phase) .- Float32.(yd.phase)))))
    dq = vec(abs.(dec.q .- yd.q))
    return (; R, dq_mean = mean(dq), dq_p95 = quantile(dq, 0.95), dq_max = maximum(dq), dec)
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

# ---- Step 3: PhasorDense emission, both paths -----------------------------

function hill_dense_tests()
    @testset "PhasorDense emission" begin
        Din, D, L, B = 6, 5, 10, 4
        rng = Xoshiro(1)
        x = Phase.(2f0 .* rand(rng, Float32, Din, L, B) .- 1f0)
        qin = 0.2f0 .+ 0.8f0 .* rand(rng, Float32, Din, L, B)

        # off ⇒ identical layer, params and outputs (discrete and spiking)
        off = PhasorDense(Din => D; init_mode = :hippo, spk_args = HILL_SPK)
        offx = PhasorDense(Din => D; init_mode = :hippo, spk_args = HILL_SPK, emit_n = Inf)
        ps0, st0 = Lux.setup(Xoshiro(2), off)
        @test keys(ps0) == keys(first(Lux.setup(Xoshiro(2), offx)))
        @test !haskey(ps0, :log_theta)
        y0 = first(off(x, ps0, st0))
        @test y0 isa AbstractArray{<:Phase, 3}
        @test Float32.(first(offx(x, ps0, st0))) == Float32.(y0)
        s0 = first(off(_hill_call(x), ps0, st0))
        @test s0.train.charges === nothing
        @test first(offx(_hill_call(x), ps0, st0)).train.times == s0.train.times
        # an emission-off layer still honours charged input (amplitude q)
        yc = first(off(Charged(x, qin), ps0, st0))
        @test yc isa AbstractArray{<:Phase, 3}
        @test Float32.(first(off(Charged(x, ones(Float32, size(x))), ps0, st0))) == Float32.(y0)

        @testset "n = 1, init=$mode bias=$bias input=$inp" for (mode, bias) in
                ((:default, true), (:hippo, true), (:hippo, false)), inp in (:unit, :charged)
            lay = PhasorDense(Din => D; init_mode = mode, use_bias = bias, emit_n = 1,
                              init_log_theta = 0f0, spk_args = HILL_SPK)
            ps, st = Lux.setup(Xoshiro(2), lay)
            @test ps.log_theta == [0f0]
            xin = inp === :unit ? x : Charged(x, qin)
            yd, _ = lay(xin, ps, st)
            @test yd isa Charged
            # emission keeps the phase of the emission-off layer
            lay_off = PhasorDense(Din => D; init_mode = mode, use_bias = bias, spk_args = HILL_SPK)
            ps_off = Base.structdiff(ps, NamedTuple{(:log_theta,)})
            @test Float32.(yd.phase) == Float32.(first(lay_off(xin, ps_off, st)))
            # θ = 1 ⇒ q = r/(r+1): read back r and set θ at its median for the parity check
            r_disc = yd.q ./ (1f0 .- yd.q)
            ps = merge(ps, (log_theta = Float32[log(median(r_disc))],))
            yd, _ = lay(xin, ps, st)
            ys, _ = lay(_hill_call(xin), ps, st)
            @test ys isa SpikingCall && ys.train.charges !== nothing
            a = _hill_agreement(yd, ys)
            # Dirac→ODE gain: r_spk/r_disc (θ-independent) ≈ 1 ⇔ |u| ≈ g·|Z|, g = 2·t_window·spk_scale
            θ = exp(only(ps.log_theta))
            r_spk = θ .* a.dec.q ./ max.(1f0 .- a.dec.q, 1f-6)
            r_d = θ .* yd.q ./ (1f0 .- yd.q)
            gain_ratio = median(r_spk ./ r_d)
            @info "PhasorDense n=1 discrete vs spiking" mode bias inp a.R a.dq_mean a.dq_p95 a.dq_max gain_ratio measured_g = gain_ratio * PhasorNetworks._emit_gain(HILL_SPK)
            @test abs(gain_ratio - 1) < 0.005
            @test a.R > 0.99
            @test a.dq_mean < 0.01
            @test a.dq_p95 < 0.03
        end

        # gradients through the emitting layer (θ learnable and frozen)
        for learn in (true, false)
            lay = PhasorDense(Din => D; init_mode = :hippo, emit_n = 1, learn_theta = learn, spk_args = HILL_SPK)
            ps, st = Lux.setup(Xoshiro(4), lay)
            g = Zygote.gradient(p -> sum(first(lay(Charged(x, qin), p, st)).q), ps)[1]
            @test all(isfinite, g.weight)
            if learn
                @test g.log_theta !== nothing && all(isfinite, g.log_theta) && only(g.log_theta) < 0
            else
                @test g.log_theta === nothing || all(iszero, g.log_theta)
            end
        end
        @test_throws ArgumentError PhasorDense(2 => 2; emit_n = 0)
    end
end
