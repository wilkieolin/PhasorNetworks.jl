# Optional / heavier tests for the local attention layers, kept separate
# from the main `test_local_attention.jl` file because they are either
# expensive (FFT-path long-L forward) or test deeper analytical properties
# (Hopfield retrieval semantics) that go beyond the per-PR sanity tier.
#
# NOT wired into `test/runtests.jl`. Invoke manually with:
#
#     julia --project=.
#     # bootstrap globals as in runtests.jl (rng, spk_args, ...)
#     include("test/test_local_attention.jl")           # for shared helpers
#     include("test/test_local_attention_extras.jl")
#     local_attention_extras_tests()
#
# Covers companion-document items:
#   - §5.1 three-view equivalence (FFT branch of causal_conv_dirac, L > 64)
#   - §5.2 HD-VSA invariance — bind/unbind symmetry under LCA (score peak
#         at matching anchor + Hopfield bundle recovers matching anchor)
#
# Bookmarked / not yet implemented:
#   - §5.2 capacity sweep for LCA (varies A ∈ {16, 64, 256, 1024} on a
#     synthetic associative-recall task) — research-scale; belongs in
#     `scripts/` rather than `test/`.
#   - §5.3 comparative training (FashionMNIST drop-in swap, learned-weight
#     discrete↔spiking parity).
#   - §5.4 streaming inference benchmark, composition with selective SSM.

function local_attention_extras_tests()
    @testset "Local Attention Extras" begin
        @info "Running local attention extras (FFT path + §5.2 Hopfield)..."
        phasor_local_attention_fft_path_tests()
        phasor_lca_score_peak_tests()
        phasor_lca_hopfield_bundle_tests()
    end
end

# ----------------------------------------------------------------------
# §5.1 three-view equivalence — FFT branch
# ----------------------------------------------------------------------
#
# `causal_conv` (used inside the PhasorDense projections) dispatches to
# `causal_conv_fft` when L > 64 and to a Toeplitz multiply otherwise.
# The main test file uses L = 6, always exercising the Toeplitz path; this
# group runs at L = 128 to exercise the FFT path end-to-end.

function phasor_local_attention_fft_path_tests()
    @testset "FFT-path forward (L > 64)" begin
        rng = Xoshiro(42)
        C_in, d_model, n_heads, n_anchors = 8, 16, 4, 12
        L, B = 128, 2

        @testset "PhasorLSA long-L forward" begin
            layer = PhasorLSA(C_in => d_model, n_heads)
            ps, st = Lux.setup(rng, layer)
            x = Phase.(2f0 .* rand(rng, Float32, C_in, L, B) .- 1f0)
            y, _ = layer(x, ps, st)
            @test size(y) == (d_model, L, B)
            @test eltype(y) <: Phase
            @test all(isfinite, Float32.(y))
            @test all(-1f0 .<= Float32.(y) .<= 1f0)
        end

        @testset "PhasorLCA long-L forward" begin
            layer = PhasorLCA(C_in => d_model, n_heads, n_anchors)
            ps, st = Lux.setup(rng, layer)
            x = Phase.(2f0 .* rand(rng, Float32, C_in, L, B) .- 1f0)
            y, _ = layer(x, ps, st)
            @test size(y) == (d_model, L, B)
            @test eltype(y) <: Phase
            @test all(isfinite, Float32.(y))
            @test all(-1f0 .<= Float32.(y) .<= 1f0)
        end

        @testset "PhasorLSA long-L gradient stability" begin
            layer = PhasorLSA(C_in => d_model, n_heads)
            ps, st = Lux.setup(rng, layer)
            x = Phase.(2f0 .* rand(rng, Float32, C_in, L, B) .- 1f0)
            loss_fn = ps_ -> begin
                y, _ = layer(x, ps_, st)
                sum(abs2.(Float32.(y)))
            end
            val, grads = withgradient(loss_fn, ps)
            g = grads[1]
            @test isfinite(val)
            @test all(isfinite, g.q_proj.weight)
            @test all(isfinite, g.v_proj.weight)
            @test all(isfinite, g.scale)
        end
    end
end

# ----------------------------------------------------------------------
# §5.2 HD-VSA invariance — score peak at matching anchor
# ----------------------------------------------------------------------
#
# Tests the Hopfield "addressing" property at the level of the
# `similarity_outer_heads` primitive: when the query equals one of the
# stored anchors, the score should peak at that anchor's index, per head.

function phasor_lca_score_peak_tests()
    @testset "LCA score peaks at matching anchor" begin
        rng = Xoshiro(42)
        Dh, H, A = 16, 2, 8
        L, B = 1, 1

        # Random anchor bank — `random_symbols` from src/vsa.jl yields
        # Phase ∈ [-1, 1] with the expected chance-level pairwise
        # similarity ≈ 0 for D ≫ 1.
        anchors = Phase.(2f0 .* rand(rng, Float32, Dh, H, A) .- 1f0)

        for a_star in (1, 3, A)   # test multiple anchor indices
            @testset "query = anchors[:, :, $a_star]" begin
                K = reshape(anchors[:, :, a_star], Dh, H, L, B)
                scores = similarity_outer_heads(anchors, K)   # (A, H, L, B)

                # Per-head: argmax over a should be a_star, and the
                # peak score should be ≈ 1 (self-similarity).
                for h in 1:H
                    peak_idx = argmax(scores[:, h, 1, 1])
                    @test peak_idx == a_star
                    @test isapprox(scores[a_star, h, 1, 1], 1f0; atol=1f-4)
                end

                # Off-target scores should be well below 1 (chance level
                # ≈ 0 for random anchors at this dimension).
                for h in 1:H, a in 1:A
                    a == a_star && continue
                    @test scores[a, h, 1, 1] < 0.9f0
                end
            end
        end
    end
end

# ----------------------------------------------------------------------
# §5.2 HD-VSA invariance — Hopfield bundle recovers matching anchor
# ----------------------------------------------------------------------
#
# Tests Proposition 3 of the tex numerically: at high inverse temperature
# β, the softmax-weighted bundle of anchors (the retrieved memory `R` of
# `eq:lca-retrieve`) approximates the matching anchor. This is the
# associative-recall property that makes LCA useful as a content-
# addressable memory.

function phasor_lca_hopfield_bundle_tests()
    @testset "LCA Hopfield bundle recovers matching anchor" begin
        rng = Xoshiro(42)
        Dh, H, A = 32, 2, 16    # larger Dh sharpens the chance-level gap
        L, B = 1, 1
        β  = 10f0               # sharp softmax for clean retrieval

        anchors = Phase.(2f0 .* rand(rng, Float32, Dh, H, A) .- 1f0)

        for a_star in (2, 7, A)
            @testset "retrieve anchors[:, :, $a_star]" begin
                # Build the same bundle the layer would build, but
                # bypass the SSM projections (we test the retrieval
                # semantics in isolation).
                K = reshape(anchors[:, :, a_star], Dh, H, L, B)
                scores  = similarity_outer_heads(anchors, K)   # (A, H, L, B)
                weights = exp.(β .* scores) ./ Float32(A)      # (A, H, L, B)

                # Per-head bundle: Bundle[:, h] = Σ_a w[a, h] · e^{iπ A[:, h, a]}
                Ac = ComplexF32.(angle_to_complex(anchors))    # (Dh, H, A)
                bundle_h = [
                    sum(weights[a, h, 1, 1] .* Ac[:, h, a] for a in 1:A)
                    for h in 1:H
                ]
                bundle_phase_h = [
                    Float32.(complex_to_angle(bundle_h[h])) for h in 1:H
                ]

                # The bundle phase should be highly similar to the
                # matching anchor's phase. We use the Fourier-HRR
                # cosine-similarity averaged over channels (same
                # measure `similarity` uses).
                for h in 1:H
                    cos_sim = mean(cos.(Float32(pi) .*
                        (bundle_phase_h[h] .- Float32.(anchors[:, h, a_star]))))
                    @test cos_sim > 0.95f0   # near-perfect retrieval at β = 10
                end
            end
        end
    end
end
