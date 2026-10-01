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
        scan_stack_tests()
        transformer_block_spiking_tests()
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

# ----------------------------------------------------------------------
# ScanStack (discrete)
# ----------------------------------------------------------------------

function scan_stack_tests()
    @testset "ScanStack" begin
        rng = Xoshiro(42)
        D, L, B, H = 8, 5, 2, 2
        x = Phase.(2f0 .* rand(rng, Float32, D, L, B) .- 1f0)
        blk = PhasorTransformerBlock(D, PhasorLSA(D => D, H); gate = :rezero)
        s = ScanStack(blk, 3)
        ps, st = Lux.setup(rng, s)
        @test length(ps.blocks) == 3
        @test Lux.parameterlength(s) == 3 * Lux.parameterlength(blk)
        y, _ = s(x, ps, st)
        # same as applying the blocks one after another
        h = x
        for i in 1:3
            h, _ = blk(h, ps.blocks[i], st.block)
        end
        @test Float32.(y) == Float32.(h)
        # checkpointed variant: identical forward, finite gradients
        sc = ScanStack(blk, 3; checkpoint = true)
        @test Float32.(first(sc(x, ps, st))) == Float32.(y)
        loss(p) = sum(abs2, real.(angle_to_complex(first(sc(x, p, st)))))
        l, gs = Zygote.withgradient(loss, ps)
        @test isfinite(l)
        @test _grad_all_finite(gs[1])
    end
end

# ----------------------------------------------------------------------
# Spiking dispatch: spike train in → spike train out, ≡ discrete
# ----------------------------------------------------------------------
#
# Equivalence metric: circular agreement between the spiking output phases
# (decoded from spike *times* with ssm_train_to_phases) and the discrete 3D
# output,  R = |mean(exp(iπ(φ_spk − φ_disc)))|  over every (channel, cycle,
# batch) output. R = 1 is exact agreement; 1 − R ≈ ½·mean(π·Δφ)² for small
# errors (R = 0.999 ↔ ≈0.045π rms).
#
# Where the gap comes from (measured, see the PR report): the spike-domain
# combine/recenter themselves are exact to Float32 (R ≥ 1 − 1e-6). All the
# disagreement comes from the ODE layers inside the branches (PhasorDense
# with the test spk_args dt=0.005, t_window=0.01: R ≈ 0.9996 per layer;
# reconstruct_from_current in PhasorLSA/LCA: R ≈ 0.99996), amplified by the
# network itself: the discrete network is ill-conditioned at a few elements
# (near-cancelling potentials, sub-threshold-silent branch neurons, and the
# ReZero α·φ_b discontinuity at the branch wrap φ_b = ±1), and with depth.
# The discrete stack perturbed by σ=0.01 input phase noise diverges just as
# fast as the spiking stack, so the depth tolerances below track the
# network's own sensitivity, not the spiking implementation.

_spk_tspan(L) = (0f0, Float32(L) * spk_args.t_period)
_spk_call(x3) = SpikingCall(ssm_phases_to_train(x3, spk_args = spk_args), spk_args,
                            _spk_tspan(size(x3, 2)))

# Circular agreement over entries that are non-silent on both sides.
function _circ_agreement(φ_spk, φ_disc)
    a, b = Float32.(φ_spk), Float32.(φ_disc)
    m = .!isnan.(a) .& .!isnan.(b)
    return abs(mean(cis.(Float32(π) .* (a[m] .- b[m]))))
end

# Overwrite every ReZero `alpha` leaf with a draw from f() (trained-like gates).
_set_alphas(ps::NamedTuple, f) =
    NamedTuple{keys(ps)}(map(k -> k === :alpha ? Float32[f()] : _set_alphas(ps[k], f), keys(ps)))
_set_alphas(ps::AbstractVector{<:NamedTuple}, f) = [_set_alphas(p, f) for p in ps]
_set_alphas(ps, f) = ps

function _spk_equivalence(model, ps, st, x3)
    y_disc, _ = model(x3, ps, st)
    y_spk, _ = model(_spk_call(x3), ps, st)
    @test y_spk isa SpikingCall
    φ = ssm_train_to_phases(y_spk)
    @test size(φ) == size(y_disc)
    return _circ_agreement(φ, y_disc), count(isnan, Float32.(φ))
end

function transformer_block_spiking_tests()
    @testset "Residual / transformer stack — spiking ≡ discrete" begin
        D, L, B, H, A = 8, 8, 3, 2, 4
        x3 = Phase.(2f0 .* rand(Xoshiro(11), Float32, D, L, B) .- 1f0)

        @testset "ssm_train_to_phases inverts ssm_phases_to_train" begin
            φ = ssm_train_to_phases(_spk_call(x3))
            @test size(φ) == (D, L, B)
            @test _circ_agreement(φ, x3) > 1 - 1f-6
        end

        @testset "spike_phase_bind ≡ v_bind(x, g·b) exactly (g = $g)" for g in (1f0, 0.37f0, -0.6f0, 0f0)
            b3 = Phase.(2f0 .* rand(Xoshiro(12), Float32, D, L, B) .- 1f0)
            y = spike_phase_bind(_spk_call(x3), _spk_call(b3); gain = g)
            @test y isa SpikingCall
            @test length(y.train.times) == D * L * B
            @test _circ_agreement(ssm_train_to_phases(y), v_bind(x3, g .* b3)) > 1 - 1f-6
        end

        @testset "spike_phase_recenter ≡ PhaseRecenter exactly" begin
            y_disc, _ = PhaseRecenter()(x3, NamedTuple(), NamedTuple())
            y_spk, _ = PhaseRecenter()(_spk_call(x3), NamedTuple(), NamedTuple())
            @test y_spk isa SpikingCall
            @test _circ_agreement(ssm_train_to_phases(y_spk), y_disc) > 1 - 1f-6
        end

        @testset "PhasorResidual(PhasorDense) gate=$gate" for gate in (:none, :rezero)
            r = PhasorResidual(PhasorDense(D => D; init_mode = :hippo); gate = gate)
            ps, st = Lux.setup(Xoshiro(13), r)
            gate === :rezero && (ps = merge(ps, (alpha = Float32[0.45f0],)))
            R, n_silent = _spk_equivalence(r, ps, st, x3)
            @info "PhasorResidual(PhasorDense) spiking ≡ discrete" gate R n_silent
            @test n_silent == 0
            @test R > 0.999          # measured ≈ 0.9997–0.9998
        end

        @testset "PhasorResidual gate=:rezero, α=0 ⇒ spikes pass through unchanged" begin
            r = PhasorResidual(PhasorLSA(D => D, H); gate = :rezero, alpha0 = 0f0)
            ps, st = Lux.setup(Xoshiro(14), r)
            sc = _spk_call(x3)
            y, _ = r(sc, ps, st)
            @test _circ_agreement(ssm_train_to_phases(y), x3) > 1 - 1f-6
            @test_throws ArgumentError r(CurrentCall(sc), ps, st)
        end

        # Trained-like regime: full-scale branch weights (branch_init_scale = 1,
        # not the 0.1 identity-at-init default) and ReZero gates drawn from
        # [0.05, 0.25] (learned α shrink with depth, see ResidualBlock docs).
        mk_block(attn; kw...) = PhasorTransformerBlock(D, attn; branch_init_scale = 1f0, kw...)
        α_draw(rng) = () -> 0.05f0 + 0.2f0 * rand(rng, Float32)

        @testset "PhasorTransformerBlock($name)" for (name, mk_attn) in
                (("LSA", () -> PhasorLSA(D => D, H)), ("LCA", () -> PhasorLCA(D => D, H, A)))
            for seed in 1:3
                rng = Xoshiro(100 + seed)
                blk = mk_block(mk_attn())
                ps, st = Lux.setup(rng, blk)
                ps = _set_alphas(ps, α_draw(rng))
                x = Phase.(2f0 .* rand(rng, Float32, D, L, B) .- 1f0)
                R, n_silent = _spk_equivalence(blk, ps, st, x)
                @info "PhasorTransformerBlock spiking ≡ discrete" name seed R n_silent
                @test R > 0.98       # measured min ≈ 0.986 over seeds
            end
        end

        @testset "PhasorTransformerBlock(LSA, recenter=true)" begin
            rng = Xoshiro(120)
            blk = mk_block(PhasorLSA(D => D, H); recenter = true)
            ps, st = Lux.setup(rng, blk)
            ps = _set_alphas(ps, α_draw(rng))
            R, n_silent = _spk_equivalence(blk, ps, st, x3)
            @info "PhasorTransformerBlock(recenter) spiking ≡ discrete" R n_silent
            @test R > 0.98
        end

        @testset "ScanStack depth $depth ($name)" for (name, mk_attn) in
                (("LSA", () -> PhasorLSA(D => D, H)), ("LCA", () -> PhasorLCA(D => D, H, A))),
                depth in 2:4
            rng = Xoshiro(200 + depth)
            stack = ScanStack(mk_block(mk_attn()), depth)
            ps, st = Lux.setup(rng, stack)
            ps = _set_alphas(ps, α_draw(rng))
            x = Phase.(2f0 .* rand(rng, Float32, D, L, B) .- 1f0)
            R, n_silent = _spk_equivalence(stack, ps, st, x)
            @info "ScanStack spiking ≡ discrete" name depth R n_silent
            # Measured here: R ≈ 0.987–0.995 (depth 2–4, LSA & LCA). A wider
            # sweep (4 seeds, D=8, L=8) gives min ≈ 0.965 / 0.94 / 0.94 at depth
            # 2 / 3 / 4, tracking the discrete stack's own sensitivity to input
            # phase noise; with α ∈ [0.3, 0.8] instead the network is chaotic
            # enough that depth-4 R drops to ≈ 0.4–0.6 *and* the discrete stack
            # perturbed by σ = 0.01 input noise drops just as far.
            @test R > 0.97
        end

        @testset "ScanStack(ResidualBlock) depth 4" begin
            rng = Xoshiro(210)
            stack = ScanStack(ResidualBlock((D, D); gate = :rezero, branch_init_scale = 1f0,
                                            init_mode = :hippo), 4)
            ps, st = Lux.setup(rng, stack)
            ps = _set_alphas(ps, α_draw(rng))
            R, n_silent = _spk_equivalence(stack, ps, st, x3)
            @info "ScanStack(ResidualBlock) spiking ≡ discrete" R n_silent
            @test R > 0.99
        end

        @testset "end-to-end: MakeSpikingSSM → PhasorDense → ScanStack(block) → SSMReadout" begin
            rng = Xoshiro(300)
            C_in, n_classes = 4, 5
            spiking_model = Chain(
                MakeSpikingSSM(spk_args),
                PhasorDense(C_in => D; init_mode = :hippo, return_type = SolutionType(:spiking)),
                ScanStack(mk_block(PhasorLSA(D => D, H)), 2),
                SSMReadout(D => n_classes))
            ps, st = Lux.setup(rng, spiking_model)
            ps = _set_alphas(ps, α_draw(rng))
            xc = normalize_to_unit_circle(randn(rng, ComplexF32, C_in, L, B))
            y_spk, _ = spiking_model(xc, ps, st)
            # same layers, same params, discrete 3D Phase input
            discrete_model = Chain(WrappedFunction(z -> complex_to_angle(normalize_to_unit_circle(z))),
                                   values(spiking_model.layers)[2:end]...)
            y_disc, _ = discrete_model(xc, ps, st)
            @test size(y_spk) == (n_classes, B)
            @test all(isfinite, y_spk)
            err = maximum(abs.(y_spk .- y_disc))
            @info "end-to-end readout spiking vs discrete" err
            @test err < 0.05
            @test [argmax(y_spk[:, b]) for b in 1:B] == [argmax(y_disc[:, b]) for b in 1:B]
        end
    end
end
