# Tests for PhasorLSA and PhasorLCA (local attention layers).
# See docs/local_attention_derivation.tex for the formal spec.

function local_attention_tests()
    @testset "Local Attention" begin
        @info "Running local attention tests..."
        similarity_outer_heads_tests()
        phasor_lsa_tests()
        phasor_lca_tests()
        phasor_lsa_spiking_tests()
        phasor_lca_spiking_tests()
    end
end

# ----------------------------------------------------------------------
# Head-axis similarity primitive
# ----------------------------------------------------------------------

function similarity_outer_heads_tests()
    @testset "similarity_outer_heads" begin
        rng = Xoshiro(42)
        Dh, H, L, B, A = 8, 4, 6, 3, 12

        @testset "LSA shape (Dh,H,L,B) × same → (H,H,L,B)" begin
            q = Phase.(2f0 .* rand(rng, Float32, Dh, H, L, B) .- 1f0)
            k = Phase.(2f0 .* rand(rng, Float32, Dh, H, L, B) .- 1f0)
            s = similarity_outer_heads(q, k)
            @test size(s) == (H, H, L, B)
            @test eltype(s) == Float32
            @test all(isfinite, s)
            @test all(-1.01f0 .<= s .<= 1.01f0)   # similarity range
        end

        @testset "LCA shape (Dh,H,A) × (Dh,H,L,B) → (A,H,L,B)" begin
            q = Phase.(2f0 .* rand(rng, Float32, Dh, H, A) .- 1f0)
            k = Phase.(2f0 .* rand(rng, Float32, Dh, H, L, B) .- 1f0)
            s = similarity_outer_heads(q, k)
            @test size(s) == (A, H, L, B)
            @test eltype(s) == Float32
            @test all(isfinite, s)
            @test all(-1.01f0 .<= s .<= 1.01f0)
        end

        @testset "self-similarity is 1 on the diagonal (LSA)" begin
            q = Phase.(2f0 .* rand(rng, Float32, Dh, H, L, B) .- 1f0)
            s = similarity_outer_heads(q, q)            # (H, H, L, B)
            # Diagonal in (H, H) should be sim(q[:,h,l,b], q[:,h,l,b]) = 1.
            for h in 1:H, l in 1:L, b in 1:B
                @test isapprox(s[h, h, l, b], 1f0, atol=1f-4)
            end
        end

        @testset "matches per-slice similarity (LSA)" begin
            q = Phase.(2f0 .* rand(rng, Float32, Dh, H, L, B) .- 1f0)
            k = Phase.(2f0 .* rand(rng, Float32, Dh, H, L, B) .- 1f0)
            s = similarity_outer_heads(q, k)
            # Spot-check one (l, b) slice against the basic similarity function.
            # `similarity` on 1D inputs returns a 0-d array; use `only` to
            # extract the scalar.
            l_pick, b_pick = 2, 1
            for h in 1:H, hp in 1:H
                expected = only(similarity(q[:, h, l_pick, b_pick], k[:, hp, l_pick, b_pick]))
                @test isapprox(s[h, hp, l_pick, b_pick], expected; atol=1f-4)
            end
        end

        @testset "matches per-slice similarity (LCA)" begin
            anchors = Phase.(2f0 .* rand(rng, Float32, Dh, H, A) .- 1f0)
            k = Phase.(2f0 .* rand(rng, Float32, Dh, H, L, B) .- 1f0)
            s = similarity_outer_heads(anchors, k)
            l_pick, b_pick = 2, 1
            for a in 1:A, h in 1:H
                expected = only(similarity(anchors[:, h, a], k[:, h, l_pick, b_pick]))
                @test isapprox(s[a, h, l_pick, b_pick], expected; atol=1f-4)
            end
        end
    end
end

# ----------------------------------------------------------------------
# PhasorLSA
# ----------------------------------------------------------------------

function phasor_lsa_tests()
    @testset "PhasorLSA" begin
        rng = Xoshiro(42)
        C_in, d_model, n_heads = 8, 16, 4
        L, B = 6, 3

        layer = PhasorLSA(C_in => d_model, n_heads)
        ps, st = Lux.setup(rng, layer)

        @testset "parameter structure" begin
            @test haskey(ps, :q_proj)
            @test haskey(ps, :k_proj)
            @test haskey(ps, :v_proj)
            @test haskey(ps, :scale)
            @test size(ps.q_proj.weight) == (d_model, C_in)
            @test size(ps.q_proj.log_neg_lambda) == (d_model,)
            @test size(ps.k_proj.weight) == (d_model, C_in)
            @test size(ps.v_proj.weight) == (d_model, C_in)
            @test length(ps.scale) == 1
        end

        @testset "parameterlength" begin
            expected = 3 * (d_model * C_in + d_model) + 1
            @test Lux.parameterlength(layer) == expected
        end

        @testset "constructor rejects non-divisible (d_model, n_heads)" begin
            @test_throws AssertionError PhasorLSA(8 => 15, 4)
        end

        @testset "Phase 3D forward (primary path)" begin
            x = Phase.(2f0 .* rand(rng, Float32, C_in, L, B) .- 1f0)
            y, _ = layer(x, ps, st)
            @test size(y) == (d_model, L, B)
            @test eltype(y) <: Phase
            @test all(isfinite, Float32.(y))
            @test all(-1f0 .<= Float32.(y) .<= 1f0)
        end

        @testset "Phase 2D dispatch matches Phase 3D with L=1" begin
            x2 = Phase.(2f0 .* rand(rng, Float32, C_in, B) .- 1f0)
            x3 = reshape(x2, C_in, 1, B)
            y2, _ = layer(x2, ps, st)
            y3, _ = layer(x3, ps, st)
            @test size(y2) == (d_model, B)
            @test eltype(y2) <: Phase
            @test Float32.(y2) ≈ Float32.(dropdims(y3, dims=2)) atol=1f-5
        end

        @testset "Complex 3D back-compat trampoline" begin
            x = randn(rng, ComplexF32, C_in, L, B)
            y, _ = layer(x, ps, st)
            @test size(y) == (d_model, L, B)
            @test eltype(y) <: Complex
            @test all(isfinite, y)
            @test all(abs.(abs.(y) .- 1f0) .< 1f-5)
        end

        @testset "different inputs produce different outputs" begin
            x1 = Phase.(2f0 .* rand(rng, Float32, C_in, L, B) .- 1f0)
            x2 = Phase.(2f0 .* rand(rng, Float32, C_in, L, B) .- 1f0)
            y1, _ = layer(x1, ps, st)
            y2, _ = layer(x2, ps, st)
            @test Float32.(y1) != Float32.(y2)
        end

        @testset "gradient sanity" begin
            x = Phase.(2f0 .* rand(rng, Float32, C_in, L, B) .- 1f0)
            loss_fn = ps_ -> begin
                y, _ = layer(x, ps_, st)
                sum(abs2.(Float32.(y)))
            end
            val, grads = withgradient(loss_fn, ps)
            g = grads[1]
            @test isfinite(val)
            @test all(isfinite, g.q_proj.weight)
            @test all(isfinite, g.q_proj.log_neg_lambda)
            @test all(isfinite, g.k_proj.weight)
            @test all(isfinite, g.k_proj.log_neg_lambda)
            @test all(isfinite, g.v_proj.weight)
            @test all(isfinite, g.v_proj.log_neg_lambda)
            @test all(isfinite, g.scale)
            # The scale gradient should be non-zero: the loss depends on it
            # through the score weighting.
            @test abs(g.scale[1]) > 0f0
        end
    end
end

# ----------------------------------------------------------------------
# PhasorLCA
# ----------------------------------------------------------------------

function phasor_lca_tests()
    @testset "PhasorLCA" begin
        rng = Xoshiro(42)
        C_in, d_model, n_heads, n_anchors = 8, 16, 4, 12
        L, B = 6, 3

        layer = PhasorLCA(C_in => d_model, n_heads, n_anchors)
        ps, st = Lux.setup(rng, layer)

        @testset "parameter structure" begin
            @test haskey(ps, :k_proj)
            @test haskey(ps, :v_proj)
            @test haskey(ps, :anchors)
            @test haskey(ps, :scale)
            @test size(ps.k_proj.weight) == (d_model, C_in)
            @test size(ps.v_proj.weight) == (d_model, C_in)
            @test size(ps.anchors) == (d_model, n_anchors)
            @test eltype(ps.anchors) <: Phase
            @test length(ps.scale) == 1
        end

        @testset "parameterlength" begin
            expected = 2 * (d_model * C_in + d_model) + d_model * n_anchors + 1
            @test Lux.parameterlength(layer) == expected
        end

        @testset "constructor rejects non-divisible (d_model, n_heads)" begin
            @test_throws AssertionError PhasorLCA(8 => 15, 4, 8)
        end

        @testset "Phase 3D forward (primary path)" begin
            x = Phase.(2f0 .* rand(rng, Float32, C_in, L, B) .- 1f0)
            y, _ = layer(x, ps, st)
            @test size(y) == (d_model, L, B)
            @test eltype(y) <: Phase
            @test all(isfinite, Float32.(y))
            @test all(-1f0 .<= Float32.(y) .<= 1f0)
        end

        @testset "Phase 2D dispatch matches Phase 3D with L=1" begin
            x2 = Phase.(2f0 .* rand(rng, Float32, C_in, B) .- 1f0)
            x3 = reshape(x2, C_in, 1, B)
            y2, _ = layer(x2, ps, st)
            y3, _ = layer(x3, ps, st)
            @test size(y2) == (d_model, B)
            @test Float32.(y2) ≈ Float32.(dropdims(y3, dims=2)) atol=1f-5
        end

        @testset "Complex 3D back-compat trampoline" begin
            x = randn(rng, ComplexF32, C_in, L, B)
            y, _ = layer(x, ps, st)
            @test size(y) == (d_model, L, B)
            @test eltype(y) <: Complex
            @test all(isfinite, y)
            @test all(abs.(abs.(y) .- 1f0) .< 1f-5)
        end

        @testset "different inputs produce different outputs (non-degenerate)" begin
            # Critical regression test: under the default head-mix the
            # output phase must depend on the input (it's the V phase bound
            # with an attention-weighted anchor bundle, not just a rescale
            # of V). See PhasorLCA docstring "Design notes".
            x1 = Phase.(2f0 .* rand(rng, Float32, C_in, L, B) .- 1f0)
            x2 = Phase.(2f0 .* rand(rng, Float32, C_in, L, B) .- 1f0)
            y1, _ = layer(x1, ps, st)
            y2, _ = layer(x2, ps, st)
            @test Float32.(y1) != Float32.(y2)
            # Per-slice difference: not just a global offset.
            diffs = Float32.(y1) .- Float32.(y2)
            @test std(vec(diffs)) > 1f-3
        end

        @testset "output depends on anchors (Hopfield retrieval is live)" begin
            x = Phase.(2f0 .* rand(rng, Float32, C_in, L, B) .- 1f0)
            y_a, _ = layer(x, ps, st)
            # Swap to a completely different anchor bank.
            ps_b = merge(ps, (anchors = Phase.(2f0 .* rand(rng, Float32, d_model, n_anchors) .- 1f0),))
            y_b, _ = layer(x, ps_b, st)
            @test Float32.(y_a) != Float32.(y_b)
        end

        @testset "gradient sanity (including anchor gradient)" begin
            x = Phase.(2f0 .* rand(rng, Float32, C_in, L, B) .- 1f0)
            loss_fn = ps_ -> begin
                y, _ = layer(x, ps_, st)
                sum(abs2.(Float32.(y)))
            end
            val, grads = withgradient(loss_fn, ps)
            g = grads[1]
            @test isfinite(val)
            @test all(isfinite, g.k_proj.weight)
            @test all(isfinite, g.v_proj.weight)
            @test all(isfinite, g.anchors)
            @test all(isfinite, g.scale)
            # Anchor gradient should be non-zero: anchors are read by the
            # bundle in every forward call.
            @test maximum(abs.(g.anchors)) > 0f0
        end
    end
end

# ----------------------------------------------------------------------
# PhasorLSA spiking dispatch (SpikingCall + CurrentCall trampolines)
# ----------------------------------------------------------------------

function phasor_lsa_spiking_tests()
    @testset "PhasorLSA spiking dispatch" begin
        rng = Xoshiro(42)
        C_in, d_model, n_heads = 4, 8, 2
        L, B = 6, 2

        # Build a SpikingCall the same way the SSMSelfAttention spiking
        # dispatch tests do (see test_ssm.jl ssm_spiking_dispatch_tests).
        x_cmpx = randn(rng, ComplexF32, C_in, L, B)
        phases_in = complex_to_angle(normalize_to_unit_circle(x_cmpx))
        train = ssm_phases_to_train(phases_in, spk_args=spk_args)
        tspan_spk = (0.0f0, Float32(L) * spk_args.t_period)
        sc = SpikingCall(train, spk_args, tspan_spk)

        layer = PhasorLSA(C_in => d_model, n_heads)
        ps, st = Lux.setup(rng, layer)

        # The SpikingCall and CurrentCall paths flow through
        # `reconstruct_from_current` (which emits Complex 3D), then route
        # via the Complex-3D back-compat dispatch — mirroring the existing
        # SSMSelfAttention / SSMCrossAttention spiking convention. The
        # output type is therefore Complex 3D on the unit circle.
        @testset "SpikingCall dispatch returns Complex 3D on unit circle" begin
            y, _ = layer(sc, ps, st)
            @test size(y) == (d_model, L, B)
            @test eltype(y) <: Complex
            @test all(isfinite, y)
            @test all(abs.(abs.(y) .- 1f0) .< 1f-4)
        end

        @testset "CurrentCall dispatch returns Complex 3D on unit circle" begin
            cc = CurrentCall(sc)
            y, _ = layer(cc, ps, st)
            @test size(y) == (d_model, L, B)
            @test eltype(y) <: Complex
            @test all(isfinite, y)
        end

        @testset "Spiking output correlates with discrete Dirac output" begin
            # Discrete path: phases_in → 3D Phase dispatch, returns Phase.
            y_disc, _ = layer(phases_in, ps, st)
            # Spiking path: SpikingCall → reconstruct → Complex 3D, returns Complex.
            y_spk, _  = layer(sc, ps, st)
            # Convert both to a common Phase domain for the correlation.
            y_spk_phase = complex_to_angle(y_spk)
            c = cor_realvals(vec(Float32.(y_disc)), vec(Float32.(y_spk_phase)))
            # Same 0.3 floor used by ssm_spiking_correlation_tests.
            @test c > 0.3
        end
    end
end

# ----------------------------------------------------------------------
# PhasorLCA spiking dispatch
# ----------------------------------------------------------------------

function phasor_lca_spiking_tests()
    @testset "PhasorLCA spiking dispatch" begin
        rng = Xoshiro(42)
        C_in, d_model, n_heads, n_anchors = 4, 8, 2, 6
        L, B = 6, 2

        x_cmpx = randn(rng, ComplexF32, C_in, L, B)
        phases_in = complex_to_angle(normalize_to_unit_circle(x_cmpx))
        train = ssm_phases_to_train(phases_in, spk_args=spk_args)
        tspan_spk = (0.0f0, Float32(L) * spk_args.t_period)
        sc = SpikingCall(train, spk_args, tspan_spk)

        layer = PhasorLCA(C_in => d_model, n_heads, n_anchors)
        ps, st = Lux.setup(rng, layer)

        # Same convention as PhasorLSA spiking dispatch — the trampoline
        # returns Complex 3D on the unit circle.
        @testset "SpikingCall dispatch returns Complex 3D on unit circle" begin
            y, _ = layer(sc, ps, st)
            @test size(y) == (d_model, L, B)
            @test eltype(y) <: Complex
            @test all(isfinite, y)
            @test all(abs.(abs.(y) .- 1f0) .< 1f-4)
        end

        @testset "CurrentCall dispatch returns Complex 3D on unit circle" begin
            cc = CurrentCall(sc)
            y, _ = layer(cc, ps, st)
            @test size(y) == (d_model, L, B)
            @test eltype(y) <: Complex
            @test all(isfinite, y)
        end

        @testset "Spiking output correlates with discrete Dirac output" begin
            y_disc, _ = layer(phases_in, ps, st)
            y_spk, _  = layer(sc, ps, st)
            y_spk_phase = complex_to_angle(y_spk)
            c = cor_realvals(vec(Float32.(y_disc)), vec(Float32.(y_spk_phase)))
            @test c > 0.3
        end
    end
end

# ----------------------------------------------------------------------
# GPU dispatch (gated on CUDA.functional() in runtests.jl)
# ----------------------------------------------------------------------

function local_attention_gpu_tests()
    @testset "Local Attention GPU" begin
        @info "Running local attention GPU tests..."
        rng = Xoshiro(42)
        device = gpu_device()
        C_in, d_model, n_heads, n_anchors = 8, 16, 4, 10
        L, B = 6, 2

        @testset "PhasorLSA forward on GPU" begin
            layer = PhasorLSA(C_in => d_model, n_heads)
            ps, st = Lux.setup(rng, layer)
            ps = ps |> device
            st = st |> device
            x  = Phase.(2f0 .* rand(rng, Float32, C_in, L, B) .- 1f0) |> device

            y, _ = layer(x, ps, st)
            @test size(y) == (d_model, L, B)
            @test all(isfinite, Float32.(Array(y)))
        end

        @testset "PhasorLCA forward on GPU" begin
            layer = PhasorLCA(C_in => d_model, n_heads, n_anchors)
            ps, st = Lux.setup(rng, layer)
            ps = ps |> device
            st = st |> device
            x  = Phase.(2f0 .* rand(rng, Float32, C_in, L, B) .- 1f0) |> device

            y, _ = layer(x, ps, st)
            @test size(y) == (d_model, L, B)
            @test all(isfinite, Float32.(Array(y)))
        end
    end
end
