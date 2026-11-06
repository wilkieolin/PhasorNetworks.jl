# attend / PhasorAttention with Lq ≠ Lk (cross-attention).
#
# Before the fix, `attend` multiplied the values (D, Lk, B) by the
# (Lq, Lk, B) score array without transposing it, so it threw a
# DimensionMismatch whenever Lq ≠ Lk and, for Lq == Lk with q ≠ k,
# weighted value j for output i by sim(q_j, k_i) instead of sim(q_i, k_j).

using Zygote: withgradient

function attend_cross_tests()
    @testset "attend cross-attention" begin
        attend_cross_shapes()
        attend_reference_formula()
        attend_self_attention_unchanged()
        attend_ssm_cross_attention_unchanged()
        attend_cross_gradients()
        attend_spiking_cross()
    end
end

_rand_phase(rng, D, L, B) = Phase.(2f0 .* rand(rng, Float32, D, L, B) .- 1f0)

# Direct implementation of the score formula:
#   s[i,j,b]   = mean_d cos(π (q[d,i,b] − k[d,j,b]))
#   w[i,j,b]   = exp(scale · s[i,j,b]) / Lk
#   out[:,i,b] = angle(Σ_j w[i,j,b] · exp(iπ v[:,j,b])) / π
function _attend_reference(q, k, v, scale)
    qf, kf, vf = Float64.(q), Float64.(k), Float64.(v)
    D, Lq, B = size(qf); Lk = size(kf, 2)
    s = [sum(cos.(π .* (qf[:, i, b] .- kf[:, j, b]))) / D for i in 1:Lq, j in 1:Lk, b in 1:B]
    w = exp.(scale .* s) ./ Lk
    out = zeros(Float64, D, Lq, B)
    for b in 1:B, i in 1:Lq
        z = sum(w[i, j, b] .* exp.(im * π .* vf[:, j, b]) for j in 1:Lk)
        out[:, i, b] = angle.(z) ./ π
    end
    return out, w
end

# Phase distance on the circle (values in [-1, 1)).
_phase_dist(a, b) = abs.(mod.(Float64.(a) .- Float64.(b) .+ 1, 2) .- 1)

function attend_cross_shapes()
    @testset "shapes" begin
        rng = Xoshiro(11)
        D, B = 8, 2
        for (Lq, Lk) in ((1, 4), (3, 5), (5, 3), (4, 4))
            q, k, v = _rand_phase(rng, D, Lq, B), _rand_phase(rng, D, Lk, B), _rand_phase(rng, D, Lk, B)
            out, scores = attend(q, k, v; scale = [3f0])
            @test size(out) == (D, Lq, B)
            @test size(scores) == (Lq, Lk, B)
            @test eltype(out) <: Phase
            @test all(isfinite, Float32.(out))

            layer = PhasorAttention()
            ps, st = Lux.setup(rng, layer)
            y, st2 = layer(q, k, v, ps, st)
            @test size(y) == (D, Lq, B)
            @test size(st2.scores) == (Lq, Lk, B)
        end
        # keys and values must have the same length
        @test_throws AssertionError attend(_rand_phase(rng, D, 1, B), _rand_phase(rng, D, 4, B),
                                           _rand_phase(rng, D, 3, B))
    end
end

function attend_reference_formula()
    @testset "reference formula" begin
        rng = Xoshiro(12)
        D, B = 8, 2
        for (Lq, Lk) in ((1, 4), (3, 5), (5, 5))
            q, k, v = _rand_phase(rng, D, Lq, B), _rand_phase(rng, D, Lk, B), _rand_phase(rng, D, Lk, B)
            out, scores = attend(q, k, v; scale = [2.5f0])
            ref_out, ref_w = _attend_reference(q, k, v, 2.5)
            @test scores ≈ ref_w rtol = 1e-4
            @test maximum(_phase_dist(out, ref_out)) < 1e-4
        end
        # one query that equals key 2 exactly, with a large scale, reads out value 2
        q = _rand_phase(rng, D, 1, B); k = _rand_phase(rng, D, 4, B); v = _rand_phase(rng, D, 4, B)
        k[:, 2, :] .= q[:, 1, :]
        out, _ = attend(q, k, v; scale = [60f0])
        @test maximum(_phase_dist(out[:, 1, :], v[:, 2, :])) < 1e-2
    end
end

function attend_self_attention_unchanged()
    @testset "self-attention (q == k) unchanged" begin
        # Formula of the implementation before the fix (Lq == Lk only):
        # batched_mul(v, scores) with the untransposed (Lq, Lk, B) scores and
        # normalisation by size(scores, 1).
        legacy(q, k, v, scale) = begin
            s = similarity_outer(q, k, dims = 2)
            s = exp.(scale .* s) ./ size(s, 1)
            PhasorNetworks.complex_to_angle(PhasorNetworks.batched_mul(PhasorNetworks.angle_to_complex(v), s)), s
        end
        rng = Xoshiro(13)
        D, L, B = 8, 6, 3
        x, v = _rand_phase(rng, D, L, B), _rand_phase(rng, D, L, B)
        out, scores = attend(x, x, v; scale = [3f0])
        lout, lscores = legacy(x, x, v, [3f0])
        @test Float32.(out) == Float32.(lout)
        @test scores == lscores
        # q ≠ k: the old output equals the new one with q and k exchanged
        q, k = _rand_phase(rng, D, L, B), _rand_phase(rng, D, L, B)
        lout_qk, _ = legacy(q, k, v, [3f0])
        out_kq, _ = attend(k, q, v; scale = [3f0])
        @test Float32.(out_kq) == Float32.(lout_qk)
    end
end

function attend_ssm_cross_attention_unchanged()
    @testset "SSMCrossAttention unchanged" begin
        # SSMCrossAttention pools a length-L sequence onto n_keys prototypes:
        # out[:, n] = angle(Σ_l V_l exp(scale · sim(Q_l, keys_n)) / L).
        rng = Xoshiro(14)
        C, D, n_keys, L, B = 3, 8, 4, 6, 2
        layer = SSMCrossAttention(C => D, n_keys, identity)
        ps, st = Lux.setup(rng, layer)
        x = _rand_phase(rng, C, L, B)
        y, _ = layer(x, ps, st)
        @test size(y) == (D, n_keys, B)
        Q, _ = layer.q_proj(x, ps.q_proj, st.q_proj)
        V, _ = layer.v_proj(x, ps.v_proj, st.v_proj)
        K = repeat(reshape(ps.keys, D, n_keys, 1), 1, 1, B)
        ref, _ = _attend_reference(K, Q, V, Float64(ps.scale[1]))
        @test maximum(_phase_dist(y, ref)) < 1e-4
    end
end

function attend_cross_gradients()
    @testset "Zygote gradients, Lq ≠ Lk" begin
        rng = Xoshiro(15)
        D, Lq, Lk, B = 8, 1, 4, 2
        q, k, v = _rand_phase(rng, D, Lq, B), _rand_phase(rng, D, Lk, B), _rand_phase(rng, D, Lk, B)
        layer = PhasorAttention()
        ps, st = Lux.setup(rng, layer)
        target = _rand_phase(rng, D, Lq, B)
        loss(q, k, v, ps) = mean(1f0 .- cos.(Float32(π) .* (Float32.(layer(q, k, v, ps, st)[1]) .- Float32.(target))))
        l, grads = withgradient(loss, q, k, v, ps)
        gq, gk, gv, gps = grads
        @test isfinite(l)
        for g in (gq, gk, gv)
            @test g !== nothing
            @test size(g) == size(q) || size(g) == size(k)
            @test all(isfinite, Float32.(g))
            @test any(!iszero, Float32.(g))
        end
        @test gps.scale !== nothing && isfinite(gps.scale[1]) && gps.scale[1] != 0

        # central finite difference along one value entry and the scale
        h = 1f-2
        vp = copy(v); vp[3, 2, 1] = Phase(Float32(v[3, 2, 1]) + h)
        vm = copy(v); vm[3, 2, 1] = Phase(Float32(v[3, 2, 1]) - h)
        fd_v = (loss(q, k, vp, ps) - loss(q, k, vm, ps)) / (2h)
        @test Float32(gv[3, 2, 1]) ≈ fd_v rtol = 5e-2 atol = 1e-4
        fd_s = (loss(q, k, v, (scale = ps.scale .+ h,)) - loss(q, k, v, (scale = ps.scale .- h,))) / (2h)
        @test gps.scale[1] ≈ fd_s rtol = 5e-2 atol = 1e-4
    end
end

function attend_spiking_cross()
    @testset "spiking attend, Lq ≠ Lk" begin
        # The spiking method passed an unsupported `d_k` keyword to
        # score_scale and threw a MethodError for every shape.
        sa = SpikingArgs(t_window = 0.01, threshold = 0.001, solver = Tsit5(),
                         solver_args = Dict(:adaptive => false, :dt => 0.01, :save_start => true))
        rng = Xoshiro(16)
        D, B, reps = 8, 2, 6
        ts = (0.0f0, Float32(reps))
        for (Lq, Lk) in ((1, 4), (4, 4))
            q, k, v = _rand_phase(rng, D, Lq, B), _rand_phase(rng, D, Lk, B), _rand_phase(rng, D, Lk, B)
            tr(x) = phase_to_train(x, spk_args = sa, repeats = reps)
            out, scores = attend(tr(q), tr(k), tr(v); spk_args = sa, tspan = ts, scale = [3f0])
            ph = train_to_phase(out; spk_args = sa)
            @test size(ph)[1:3] == (D, Lq, B)
            @test size(scores[end]) == (Lq, Lk, B)
            ref, _ = attend(q, k, v; scale = [3f0])
            # decoded phase of the second-to-last cycle against the Phase path
            @test maximum(_phase_dist(ph[:, :, :, end-1], ref)) < 0.25
        end
    end
end
