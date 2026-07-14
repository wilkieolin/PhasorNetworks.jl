# Tests for MultiModePhasorDense — the SSM state-expansion layer (M timescale
# modes per output channel, block-diagonal complex mix-down). The key contracts:
#   • n_modes=1 (no bias) reduces exactly to the PhasorDense 3D discrete path
#   • correct parameter shapes / counts for M>1
#   • finite Zygote gradients through all leaves (weight, log_neg_lambda, C)
#   • wiring into PhasorTransformerBlock via ffn_n_modes
#   • the λ-range knob (hippo_tau_max) actually widens the init spectrum

using PhasorNetworks, Lux, Zygote
using Random: Xoshiro
using Statistics: mean

function multimode_tests()
    @testset "MultiModePhasorDense" begin
        @info "Running MultiModePhasorDense tests..."
        D, L, B = 8, 6, 4
        x = Phase.(2f0 .* rand(Xoshiro(1), Float32, D, L, B) .- 1f0)

        # (1) n_modes=1, no bias == PhasorDense 3D path (same rng ⇒ same W, λ).
        pd  = PhasorDense(D => D, normalize_to_unit_circle; use_bias = false, init_mode = :default)
        mm1 = MultiModePhasorDense(D => D, 1, normalize_to_unit_circle; use_bias = false, init_mode = :default)
        p1, s1 = Lux.setup(Xoshiro(42), pd)
        p2, s2 = Lux.setup(Xoshiro(42), mm1)
        y1, _ = pd(x, p1, s1)
        y2, _ = mm1(x, p2, s2)
        @test p1.weight == p2.weight
        @test p1.log_neg_lambda == p2.log_neg_lambda
        @test maximum(abs.(Float32.(y1) .- Float32.(y2))) < 1f-5

        # (2) shapes / param counts for M>1.
        M = 4
        mm = MultiModePhasorDense(D => D, M; init_mode = :hippo)
        pm, sm = Lux.setup(Xoshiro(7), mm)
        @test length(pm.log_neg_lambda) == D * M
        @test size(pm.C_real) == (D, M) && size(pm.C_imag) == (D, M)
        ym, _ = mm(x, pm, sm)
        @test size(ym) == (D, L, B)
        @test eltype(ym) <: Phase
        # 2D path
        y2d, _ = mm(x[:, 1, :], pm, sm)
        @test size(y2d) == (D, B)

        # (3) finite gradients through every leaf.
        g = Zygote.gradient(p -> sum(abs2, Float32.(first(mm(x, p, sm)))), pm)[1]
        @test all(isfinite, g.weight)
        @test all(isfinite, g.log_neg_lambda)
        @test all(isfinite, g.C_real) && all(isfinite, g.C_imag)

        # (4) wiring into PhasorTransformerBlock.
        blk = PhasorTransformerBlock(D, PhasorLSA(D => D, 2); ffn_n_modes = M, ffn_hippo_tau_max = 256f0)
        pb, sb = Lux.setup(Xoshiro(3), blk)
        yb, _ = blk(x, pb, sb)
        @test size(yb) == (D, L, B)
        # ffn_n_modes=1 block still builds and forwards (default path unchanged).
        blk1 = PhasorTransformerBlock(D, PhasorLSA(D => D, 2); ffn_n_modes = 1)
        pb1, sb1 = Lux.setup(Xoshiro(3), blk1)
        yb1, _ = blk1(x, pb1, sb1)
        @test size(yb1) == (D, L, B)

        # (5) λ-range knob widens the init spectrum: larger tau_max ⇒ slower
        # slowest mode (smaller min |λ| ⇒ more negative min log_neg_lambda).
        s_narrow, _ = Lux.setup(Xoshiro(5), MultiModePhasorDense(D => D, M; init_mode = :hippo, hippo_tau_max = 16f0))
        s_wide,   _ = Lux.setup(Xoshiro(5), MultiModePhasorDense(D => D, M; init_mode = :hippo, hippo_tau_max = 1024f0))
        @test minimum(s_wide.log_neg_lambda) < minimum(s_narrow.log_neg_lambda)
    end
end
