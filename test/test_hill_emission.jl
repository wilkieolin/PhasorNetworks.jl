# Hill emission (per-spike charge q = rⁿ/(rⁿ+θⁿ)) — opt-in charge carried by
# spikes (spiking path) and by `Charged` streams (discrete Dirac path).

function hill_emission_tests()
    @testset "Hill emission" begin
        hill_spike_charge_tests()
        hill_charged_dirac_tests()
        hill_dense_tests()
        hill_attention_tests()
        hill_block_tests()
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

# Charged-vs-Charged agreement (both discrete fields).
function _hill_agreement(yd::Charged, dec::Charged)
    @test size(dec) == size(yd)
    R = abs(mean(cis.(Float32(π) .* (Float32.(dec.phase) .- Float32.(yd.phase)))))
    dq = vec(abs.(dec.q .- yd.q))
    return (; R, dq_mean = mean(dq), dq_p95 = quantile(dq, 0.95), dq_max = maximum(dq), dec)
end

# Discrete (Charged) vs spiking (decoded SpikingCall) agreement:
# circular phase agreement R, mean and 95th-percentile |Δq|.
_hill_agreement(yd::Charged, ys::SpikingCall) = _hill_agreement(yd, ssm_train_to_charged(ys))

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

# ---- Step 4: PhasorLSA / PhasorLCA emission -------------------------------

function hill_attention_tests()
    @testset "Local attention emission ($name)" for (name, mk) in
            (("LSA", kw -> PhasorLSA(8 => 8, 2; spk_args = HILL_SPK, kw...)),
             ("LCA", kw -> PhasorLCA(8 => 8, 2, 4; spk_args = HILL_SPK, kw...)))
        D, L, B = 8, 6, 3
        rng = Xoshiro(21)
        x = Phase.(2f0 .* rand(rng, Float32, D, L, B) .- 1f0)
        qin = 0.2f0 .+ 0.8f0 .* rand(rng, Float32, D, L, B)

        off = mk((;)); offx = mk((; emit_n = Inf))
        ps0, st0 = Lux.setup(Xoshiro(22), off)
        @test !haskey(ps0, :log_theta)
        @test keys(first(Lux.setup(Xoshiro(22), offx))) == keys(ps0)
        @test Lux.parameterlength(off) == Lux.parameterlength(offx)
        y0 = first(off(x, ps0, st0))
        @test Float32.(first(offx(x, ps0, st0))) == Float32.(y0)
        # unit-charge spiking route unchanged (returns the complex field as before)
        @test first(off(_hill_call(x), ps0, st0)) isa AbstractArray{<:Complex, 3}

        lay = mk((; emit_n = 1, init_log_theta = 0f0))
        ps, st = Lux.setup(Xoshiro(22), lay)
        @test Lux.parameterlength(lay) == Lux.parameterlength(off) + 1
        for xin in (x, Charged(x, qin))
            yd = first(lay(xin, ps, st))
            @test yd isa Charged
            @test Float32.(yd.phase) == Float32.(first(off(xin, ps0, st0)))
            # θ at the median output amplitude
            r = yd.q ./ (1f0 .- yd.q)
            psθ = merge(ps, (log_theta = Float32[log(median(r))],))
            yd = first(lay(xin, psθ, st))
            ys = first(lay(_hill_call(xin), psθ, st))      # spiking dispatch → Charged field
            @test ys isa Charged
            a = _hill_agreement(yd, ys)
            # …and re-emitted onto a charged spike train (as inside a residual)
            sc = PhasorNetworks._as_spike_call(ys, _hill_call(x))
            @test sc.train.charges !== nothing
            b = _hill_agreement(yd, sc)
            inp = xin isa Charged ? :charged : :unit
            @info "$name n=1 discrete vs spiking" inp a.R a.dq_mean a.dq_p95 a.dq_max b.R b.dq_mean b.dq_p95 b.dq_max
            if inp === :charged
                # charged train → exact decode → same discrete arithmetic
                @test a.R > 1 - 1f-5 && a.dq_max < 1f-4
                @test b.R > 1 - 1f-5 && b.dq_max < 1f-4
            else
                # unit train → reconstruct_from_current (resonator) route; its
                # phase error is amplified by the attention nonlinearity
                # (measured: LSA R = 0.9990, mean|Δq| = 0.006, p95 = 0.036;
                #  LCA R = 0.9982, mean|Δq| = 0.003, p95 = 0.015)
                @test a.R > 0.995 && a.dq_mean < 0.01 && a.dq_p95 < 0.05
                @test b.R > 0.995 && b.dq_mean < 0.01 && b.dq_p95 < 0.05
            end
        end
        @test_throws ArgumentError lay(x[:, 1, :], ps, st)
        g = Zygote.gradient(p -> sum(first(lay(Charged(x, qin), p, st)).q), merge(ps, (; log_theta = [0f0])))[1]
        @test g.log_theta === nothing || all(iszero, g.log_theta)          # frozen θ
        @test all(isfinite, g.scale)
        lay_l = mk((; emit_n = 1, learn_theta = true))
        psl, stl = Lux.setup(Xoshiro(22), lay_l)
        gl = Zygote.gradient(p -> sum(first(lay_l(Charged(x, qin), p, stl)).q), psl)[1]
        @test all(isfinite, gl.log_theta) && only(gl.log_theta) < 0
    end
end

# ---- Step 5: residual relay, recenter, readout, transformer block ---------

function hill_block_tests()
    D, L, B, H, A = 8, 8, 3, 2, 4
    rng = Xoshiro(31)
    x = Phase.(2f0 .* rand(rng, Float32, D, L, B) .- 1f0)
    qin = 0.2f0 .+ 0.8f0 .* rand(rng, Float32, D, L, B)
    xc = Charged(x, qin)

    @testset "relays: combine / recenter / readout" begin
        b = Phase.(2f0 .* rand(Xoshiro(32), Float32, D, L, B) .- 1f0)
        qb = rand(Xoshiro(33), Float32, D, L, B)
        for g in (1f0, 0.37f0, -0.6f0)
            yd = PhasorNetworks._res_bind(xc, Float32[g], Charged(b, qb))
            @test yd isa Charged && yd.q == qin
            @test Float32.(yd.phase) == Float32.(v_bind(x, g .* (qb .* b)))
            ys = spike_phase_bind(_hill_call(xc), _hill_call(Charged(b, qb)); gain = g)
            a = _hill_agreement(yd, ys)
            @test a.R > 1 - 1f-5 && a.dq_max < 1f-6
            # unit-charge skip, charged branch → plain phase out, no charges on the train
            @test PhasorNetworks._res_bind(x, Float32[g], Charged(b, qb)) isa AbstractArray{<:Phase, 3}
            @test spike_phase_bind(_hill_call(x), _hill_call(Charged(b, qb)); gain = g).train.charges === nothing
        end
        # q_b = 1 ⇔ the original combine
        @test Float32.(PhasorNetworks._res_bind(x, Float32[0.4f0], Charged(b, ones(Float32, size(b))))) ==
              Float32.(v_bind(x, 0.4f0 .* b))
        yr, _ = PhaseRecenter()(xc, NamedTuple(), NamedTuple())
        @test yr isa Charged && yr.q == qin
        @test Float32.(yr.phase) == Float32.(first(PhaseRecenter()(x, NamedTuple(), NamedTuple())))
        a = _hill_agreement(yr, first(PhaseRecenter()(_hill_call(xc), NamedTuple(), NamedTuple())))
        @test a.R > 1 - 1f-5 && a.dq_max < 1f-6
        ro = SSMReadout(D => 3)
        pr, sr = Lux.setup(Xoshiro(34), ro)
        @test first(ro(xc, pr, sr)) == first(ro(x, pr, sr))
        @test_throws ArgumentError spike_phase_bundle(_hill_call(x), _hill_call(xc))
    end

    mk_attn(name; kw...) = name == "LSA" ? PhasorLSA(D => D, H; spk_args = HILL_SPK, kw...) :
                                           PhasorLCA(D => D, H, A; spk_args = HILL_SPK, kw...)
    mk_block(name; ffn = (;), attn = (;)) =
        PhasorTransformerBlock(D, mk_attn(name; attn...); branch_init_scale = 1f0, ffn...)
    α_set(ps, rng) = _hill_set_alphas(ps, () -> 0.05f0 + 0.2f0 * rand(rng, Float32))

    @testset "block emission off ⇒ identical ($name)" for name in ("LSA", "LCA")
        b0 = mk_block(name)
        bx = mk_block(name; ffn = (; ffn_emit_n = Inf), attn = (; emit_n = Inf))
        ps, st = Lux.setup(Xoshiro(41), b0)
        @test ps == first(Lux.setup(Xoshiro(41), bx))
        @test Float32.(first(b0(x, ps, st))) == Float32.(first(bx(x, ps, st)))
        y0 = first(b0(_hill_call(x), ps, st)); yx = first(bx(_hill_call(x), ps, st))
        @test y0.train.charges === nothing && yx.train.charges === nothing
        @test y0.train.times == yx.train.times && y0.train.indices == yx.train.indices
    end

    @testset "PhasorResidual(PhasorDense, n = 1) gate=$gate" for gate in (:none, :rezero)
        r = PhasorResidual(PhasorDense(D => D; init_mode = :hippo, emit_n = 1, spk_args = HILL_SPK); gate)
        ps, st = Lux.setup(Xoshiro(42), r)
        gate === :rezero && (ps = merge(ps, (alpha = Float32[0.45f0],)))
        amps = emission_amplitudes(r, ps, st, xc)
        @test size(amps.layer.r) == (D, L, B)
        ps = set_emission_theta(ps, amps; q = 0.5)
        @test ps.layer.log_theta ≈ [log(median(vec(amps.layer.r)))]
        yd = first(r(xc, ps, st))
        @test yd isa Charged && yd.q == qin
        a = _hill_agreement(yd, first(r(_hill_call(xc), ps, st)))
        @info "PhasorResidual(PhasorDense n=1) discrete vs spiking" gate a.R a.dq_max
        @test a.R > 0.999 && a.dq_max < 1f-6      # the skip charge passes through exactly
    end

    @testset "PhasorTransformerBlock n = 1 ($name, recenter=$rc)" for name in ("LSA", "LCA"), rc in (false, true)
        rc && name == "LCA" && continue
        for seed in 1:3
            rng = Xoshiro(500 + seed)
            blk = PhasorTransformerBlock(D, mk_attn(name; emit_n = 1); branch_init_scale = 1f0,
                                         ffn_emit_n = 1, recenter = rc)
            ps, st = Lux.setup(rng, blk)
            ps = α_set(ps, rng)
            amps = emission_amplitudes(blk, ps, st, xc)
            ps = set_emission_theta(ps, amps; q = 0.5)
            yd = first(blk(xc, ps, st))
            @test yd isa Charged && yd.q == qin
            ys = first(blk(_hill_call(xc), ps, st))
            @test ys.train.charges !== nothing
            a = _hill_agreement(yd, ys)
            @info "PhasorTransformerBlock n=1 discrete vs spiking" name rc seed a.R a.dq_max
            @test a.R > 0.98
            @test a.dq_max < 1f-6
        end
    end

    @testset "emission sites inside the block agree ($name)" for name in ("LSA", "LCA")
        rng = Xoshiro(600)
        blk = PhasorTransformerBlock(D, mk_attn(name; emit_n = 1); branch_init_scale = 1f0, ffn_emit_n = 1)
        ps, st = Lux.setup(rng, blk)
        ps = α_set(ps, rng)
        ps = set_emission_theta(ps, emission_amplitudes(blk, ps, st, xc); q = 0.5)
        # attention branch output (charged) — discrete vs spiking
        attn, pa, sa_ = blk.attn_res.layer, ps.attn_res.layer, st.attn_res.layer
        a1 = _hill_agreement(first(attn(xc, pa, sa_)), first(attn(_hill_call(xc), pa, sa_)))
        # FFN branch output on the attention-residual output h
        h = first(blk.attn_res(xc, ps.attn_res, st.attn_res))
        ffn, pf, sf = blk.ffn_res.layer, ps.ffn_res.layer, st.ffn_res.layer
        a2 = _hill_agreement(first(ffn(h, pf, sf)), first(ffn(_hill_call(h), pf, sf)))
        @info "block sites n=1 discrete vs spiking" name attn_R = a1.R attn_dq_mean = a1.dq_mean attn_dq_p95 = a1.dq_p95 ffn_R = a2.R ffn_dq_mean = a2.dq_mean ffn_dq_p95 = a2.dq_p95 ffn_dq_max = a2.dq_max
        @test a1.R > 1 - 1f-5 && a1.dq_max < 1f-4
        @test a2.R > 0.99 && a2.dq_mean < 0.01 && a2.dq_p95 < 0.03
    end

    @testset "probe + θ on Chain(encoder, ScanStack, readout)" begin
        enc = PhasorDense(5 => D; init_mode = :hippo, use_bias = false, emit_n = 1, spk_args = HILL_SPK)
        blk = PhasorTransformerBlock(D, mk_attn("LSA"; emit_n = 1); ffn_emit_n = 1)
        model = Chain(enc, ScanStack(blk, 2), SSMReadout(D => 3))
        base = Chain(PhasorDense(5 => D; init_mode = :hippo, use_bias = false, spk_args = HILL_SPK),
                     ScanStack(PhasorTransformerBlock(D, mk_attn("LSA")), 2), SSMReadout(D => 3))
        psb, stb = Lux.setup(Xoshiro(51), base)
        xin = Phase.(2f0 .* rand(Xoshiro(52), Float32, 5, L, B) .- 1f0)
        amps = emission_amplitudes(base, psb, stb, xin)
        @test amps.layer_1.r isa AbstractArray && amps.layer_3 === nothing
        @test length(amps.layer_2.blocks) == 2
        @test amps.layer_2.blocks[2].attn_res.layer.r isa AbstractArray
        @test amps.layer_2.blocks[2].ffn_res.layer.layer_2.r isa AbstractArray
        @test amps.layer_2.blocks[1].attn_res.layer isa NamedTuple{(:r,)}     # projections are not sites
        ps = set_emission_theta(psb, amps; q = 0.3)
        θ1 = only(ps.layer_2.blocks[1].ffn_res.layer.layer_1.log_theta)
        θ2 = only(ps.layer_2.blocks[2].ffn_res.layer.layer_1.log_theta)
        @test θ1 ≈ log(quantile(vec(amps.layer_2.blocks[1].ffn_res.layer.layer_1.r), 0.3))
        @test θ1 != θ2                                   # per-depth θ
        st = stb                                         # same (parameter-free) states
        @test keys(ps.layer_1) == keys(first(Lux.setup(Xoshiro(51), model)).layer_1)
        y, _ = model(xin, ps, st)
        @test size(y) == (3, B) && all(isfinite, y)
        # trains end to end (gradients finite, θ frozen)
        g = Zygote.gradient(p -> sum(first(model(xin, p, st))), ps)[1]
        @test all(isfinite, g.layer_1.weight)
        @test all(isfinite, g.layer_2.blocks[2].ffn_res.layer.layer_2.weight)
        # the probe also runs on the emitting model
        amps2 = emission_amplitudes(model, ps, st, xin)
        @test size(amps2.layer_2.blocks[1].attn_res.layer.r) == (D, L, B)
    end
end

_hill_set_alphas(ps::NamedTuple, f) =
    NamedTuple{keys(ps)}(map(k -> k === :alpha ? Float32[f()] : _hill_set_alphas(ps[k], f), keys(ps)))
_hill_set_alphas(ps::AbstractVector{<:NamedTuple}, f) = [_hill_set_alphas(p, f) for p in ps]
_hill_set_alphas(ps, f) = ps
