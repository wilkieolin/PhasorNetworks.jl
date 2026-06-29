# Tests for the phase-domain transformer stack:
#   PhaseRecenter (pre-norm), PhasorResidual (identity-at-init residual
#   wrapper), and PhasorTransformerBlock (residual(attn) + residual(FFN)).
# These are what make PhasorLSA/PhasorLCA stackable at depth.

# Recursively assert every numeric leaf of a gradient tree is finite.
function _grad_all_finite(g)
    ok = Ref(true)
    _gaf!(ok, g)
    return ok[]
end
_gaf!(ok, ::Nothing) = nothing
_gaf!(ok, g::NamedTuple) = (for k in keys(g); _gaf!(ok, getfield(g, k)); end)
_gaf!(ok, g::Tuple) = (for v in g; _gaf!(ok, v); end)
_gaf!(ok, g::AbstractArray{<:Number}) = (all(isfinite, Array(g)) || (ok[] = false))
_gaf!(ok, g::AbstractArray) = (for v in g; _gaf!(ok, v); end)
_gaf!(ok, g::Number) = (isfinite(g) || (ok[] = false))
_gaf!(ok, g) = nothing

function transformer_block_tests()
    @testset "Transformer Block" begin
        @info "Running transformer block tests..."
        phase_recenter_tests()
        phasor_residual_tests()
        phasor_transformer_block_tests()
    end
end

# ----------------------------------------------------------------------
# PhaseRecenter
# ----------------------------------------------------------------------

function phase_recenter_tests()
    @testset "PhaseRecenter" begin
        rng = Xoshiro(42)
        C, L, B = 8, 6, 3
        pr = PhaseRecenter()
        ps, st = Lux.setup(rng, pr)
        @test ps == NamedTuple()

        @testset "3D shape/type preserved + circular mean ≈ 0" begin
            x = Phase.(2f0 .* rand(rng, Float32, C, L, B) .- 1f0)
            y, _ = pr(x, ps, st)
            @test size(y) == (C, L, B)
            @test eltype(y) <: Phase
            @test all(-1f0 .<= Float32.(y) .<= 1f0)
            # after recentering, the per-(l,b) circular mean angle over
            # channels should be ~0
            m = complex_to_angle(sum(angle_to_complex(y), dims = 1))
            @test all(abs.(Float32.(m)) .< 1f-3)
        end

        @testset "2D shape preserved" begin
            x = Phase.(2f0 .* rand(rng, Float32, C, B) .- 1f0)
            y, _ = pr(x, ps, st)
            @test size(y) == (C, B)
            @test eltype(y) <: Phase
        end
    end
end

# ----------------------------------------------------------------------
# PhasorResidual
# ----------------------------------------------------------------------

function phasor_residual_tests()
    @testset "PhasorResidual" begin
        rng = Xoshiro(42)
        D, L, B, H = 16, 5, 3, 4
        x = Phase.(2f0 .* rand(rng, Float32, D, L, B) .- 1f0)

        @testset "gate=:none has no alpha; shape preserved" begin
            r = PhasorResidual(PhasorDense(D => D, normalize_to_unit_circle); gate = :none)
            ps, st = Lux.setup(rng, r)
            @test haskey(ps, :layer)
            @test !haskey(ps, :alpha)
            y, _ = r(x, ps, st)
            @test size(y) == (D, L, B)
            @test eltype(y) <: Phase
            @test Lux.parameterlength(r) == Lux.parameterlength(r.layer)
        end

        @testset "gate=:rezero adds alpha; alpha0=0 ⇒ exact identity (wraps attention)" begin
            r = PhasorResidual(PhasorLSA(D => D, H); gate = :rezero, alpha0 = 0f0)
            ps, st = Lux.setup(rng, r)
            @test haskey(ps, :alpha)
            @test length(ps.alpha) == 1
            @test Lux.parameterlength(r) == Lux.parameterlength(r.layer) + 1
            y, _ = r(x, ps, st)
            @test size(y) == (D, L, B)
            # α = 0 ⇒ the branch contributes nothing ⇒ output == input
            @test all(isapprox.(Float32.(y), Float32.(x); atol = 1f-5))
        end

        @testset "rejects bad gate" begin
            @test_throws ArgumentError PhasorResidual(PhasorDense(D => D); gate = :bogus)
        end
    end
end

# ----------------------------------------------------------------------
# PhasorTransformerBlock
# ----------------------------------------------------------------------

function phasor_transformer_block_tests()
    @testset "PhasorTransformerBlock" begin
        rng = Xoshiro(42)
        D, L, B, H, A = 16, 6, 3, 4, 8
        x3 = Phase.(2f0 .* rand(rng, Float32, D, L, B) .- 1f0)

        mk_lsa() = PhasorLSA(D => D, H)
        mk_lca() = PhasorLCA(D => D, H, A)

        @testset "forward shape/type — LSA & LCA, 3D and 2D" begin
            for mk in (mk_lsa, mk_lca)
                blk = PhasorTransformerBlock(D, mk())
                ps, st = Lux.setup(rng, blk)

                y3, _ = blk(x3, ps, st)
                @test size(y3) == (D, L, B)
                @test eltype(y3) <: Phase
                @test all(isfinite, Float32.(y3))
                @test all(-1f0 .<= Float32.(y3) .<= 1f0)

                x2 = Phase.(2f0 .* rand(rng, Float32, D, B) .- 1f0)
                y2, _ = blk(x2, ps, st)
                @test size(y2) == (D, B)
                @test eltype(y2) <: Phase
            end
        end

        @testset "parameterlength = attn_res + ffn_res, nonzero" begin
            blk = PhasorTransformerBlock(D, mk_lsa())
            @test Lux.parameterlength(blk) ==
                  Lux.parameterlength(blk.attn_res) + Lux.parameterlength(blk.ffn_res)
            @test Lux.parameterlength(blk) > 0
        end

        @testset "identity-at-init: gate=:rezero, alpha0=0, recenter=false ⇒ output==input" begin
            for mk in (mk_lsa, mk_lca)
                blk = PhasorTransformerBlock(D, mk();
                                             gate = :rezero, alpha0 = 0f0, recenter = false)
                ps, st = Lux.setup(rng, blk)
                y, _ = blk(x3, ps, st)
                @test all(isapprox.(Float32.(y), Float32.(x3); atol = 1f-5))
            end
        end

        @testset "recenter=true path runs (pre-norm in branch)" begin
            blk = PhasorTransformerBlock(D, mk_lsa(); gate = :rezero, recenter = true)
            ps, st = Lux.setup(rng, blk)
            y, _ = blk(x3, ps, st)
            @test size(y) == (D, L, B)
            @test eltype(y) <: Phase
        end

        @testset "gradients finite through a depth-4 stack" begin
            stack = Chain(ntuple(_ -> PhasorTransformerBlock(D, mk_lsa();
                                       gate = :rezero, alpha0 = 0.1f0), 4)...)
            ps, st = Lux.setup(rng, stack)
            loss(p) = sum(abs2, real.(angle_to_complex(first(stack(x3, p, st)))))
            l, gs = Zygote.withgradient(loss, ps)
            @test isfinite(l)
            @test _grad_all_finite(gs[1])
        end

        @testset "old vs new regime both construct & run" begin
            old = PhasorTransformerBlock(D, mk_lsa();
                                         gate = :none, branch_init_scale = 1f0, recenter = false)
            new = PhasorTransformerBlock(D, mk_lsa();
                                         gate = :rezero, branch_init_scale = 0.1f0, recenter = true)
            for blk in (old, new)
                ps, st = Lux.setup(rng, blk)
                y, _ = blk(x3, ps, st)
                @test size(y) == (D, L, B)
            end
        end
    end
end
