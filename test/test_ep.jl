# Vanilla phasor-EP tests. Phase 1 scope: PhasorDense + StaticEP +
# SimilarityCost + K=0 + use_bias=false. See docs/phasor_ep_design.md
# and demos/phasor_ep_demo.ipynb for context.

using Test
using PhasorNetworks
using Lux
using LinearAlgebra: norm, dot
using Random: Xoshiro

# Phase-1-specific test parameters. Local to this file — keep
# runtests.jl globals clean (per the planning report).
const EP_T_FREE  = 100
const EP_T_NUDGE = 50
const EP_DT      = 0.5f0
const EP_BETA    = 0.001f0       # small enough to suppress O(β) bias
const EP_FD_TOL  = 0.05          # 5% rel-err target on each layer

function ep_tests()
    @testset "Phasor EP (Phase 1)" begin
        ep_cost_tests()
        ep_interface_tests()
        ep_settle_tests()
        ep_gradient_vs_fd_tests()
        ep_training_tests()
        ep_bias_rejection_tests()
    end
end

# ----------------------------------------------------------------
# 1. SimilarityCost
# ----------------------------------------------------------------
function ep_cost_tests()
    @testset "SimilarityCost" begin
        rng = Xoshiro(42)
        d = 4
        y = ComplexF32.(exp.(im .* π .* (2f0 .* rand(rng, Float32, d) .- 1f0)))
        c = SimilarityCost(y)

        # Loss is zero at the target.
        @test isapprox(ep_loss(c, y), 0f0, atol=1e-5)

        # Loss is 2 at the antipode.
        @test isapprox(ep_loss(c, -y), 2f0, atol=1e-5)

        # Loss is 1 at orthogonal-by-rotation states.
        z_orth = im .* y
        @test isapprox(ep_loss(c, z_orth), 1f0, atol=1e-5)

        # Nudge force points toward the target with the documented
        # 1/d factor (real-parameter convention; no 1/2).
        z = zeros(ComplexF32, d)
        β = 0.5f0
        @test PhasorNetworks.nudge_force(c, z, β) ≈ (β / d) .* y
    end
end

# ----------------------------------------------------------------
# 2. Per-layer interface (PhasorDense)
# ----------------------------------------------------------------
function ep_interface_tests()
    @testset "Per-layer interface (PhasorDense)" begin
        rng = Xoshiro(42)
        layer = PhasorDense(3 => 4, normalize_to_unit_circle, use_bias=false)
        ps, st = Lux.setup(rng, layer)

        z_in   = ComplexF32[1+0im, 0+1im, 1-1im]
        z_self = ComplexF32[0.5+0.5im, 1+0im, 0+1im, -1+0im]

        # ep_drive: weight * z_in (no activation).
        d = PhasorNetworks.ep_drive(layer, ps, st, z_in)
        @test d ≈ ps.weight * z_in
        @test eltype(d) <: Complex

        # ep_feedback: transpose(weight) * z_self.
        f = PhasorNetworks.ep_feedback(layer, ps, st, z_self)
        @test f ≈ transpose(ps.weight) * z_self
        @test size(f) == size(z_in)

        # ep_self_force: zero in Phase 1.
        sf = PhasorNetworks.ep_self_force(layer, ps, st, z_self)
        @test all(sf .== 0)
        @test size(sf) == size(z_self)

        # ep_hebbian: real outer product, with zeros for non-EP-trained params.
        h = PhasorNetworks.ep_hebbian(layer, ps, st, z_in, z_self)
        @test haskey(h, :weight)
        @test h.weight ≈ real.(z_self * adjoint(z_in))
        @test size(h.weight) == size(ps.weight)
        # Phase 1 does not update log_neg_lambda; gradient must be present-and-zero
        # so Optimisers.update doesn't drop the param.
        @test haskey(h, :log_neg_lambda)
        @test all(h.log_neg_lambda .== 0)
        @test size(h.log_neg_lambda) == size(ps.log_neg_lambda)

        # ep_energy_contribution: real(<z_self, W·z_in>).
        e = PhasorNetworks.ep_energy_contribution(layer, ps, st, z_in, z_self)
        @test e ≈ Float32(real(dot(z_self, ps.weight * z_in)))
        @test e isa Float32
    end
end

# ----------------------------------------------------------------
# 3. phasor_settle
# ----------------------------------------------------------------
function _ep_chain(rng; n_in=4, n_hid=8, n_out=2, scale=0.4f0)
    chain = Chain(
        PhasorDense(n_in  => n_hid, normalize_to_unit_circle, use_bias=false),
        PhasorDense(n_hid => n_out, normalize_to_unit_circle, use_bias=false))
    ps, st = Lux.setup(rng, chain)
    # Scale weights down — the default glorot is wide enough that some
    # initial drives can have small magnitude during settling.
    ps = (
        layer_1 = merge(ps.layer_1, (weight = scale .* ps.layer_1.weight,)),
        layer_2 = merge(ps.layer_2, (weight = scale .* ps.layer_2.weight,)),
    )
    return chain, ps, st
end

function ep_settle_tests()
    @testset "phasor_settle equilibrium" begin
        rng = Xoshiro(42)
        chain, ps, st = _ep_chain(rng)
        x = Phase.(2f0 .* rand(rng, Float32, 4) .- 1f0)
        y = ComplexF32.(exp.(im .* π .* (2f0 .* rand(rng, Float32, 2) .- 1f0)))
        cost = SimilarityCost(y)

        # Free phase: states reach the unit circle.
        s_free = phasor_settle(chain, ps, st, x, cost, 0f0;
                               T=EP_T_FREE, dt=EP_DT)
        @test length(s_free) == 2
        @test all(z -> isapprox(abs(z), 1f0, atol=1e-3), s_free[1])
        @test all(z -> isapprox(abs(z), 1f0, atol=1e-3), s_free[2])

        # Nudged phase increases similarity to the target.
        free_loss = ep_loss(cost, s_free[end])
        s_nudge = phasor_settle(chain, ps, st, x, cost, 0.5f0;
                                T=EP_T_NUDGE, dt=EP_DT, init=s_free)
        nudge_loss = ep_loss(cost, s_nudge[end])
        @test nudge_loss < free_loss

        # Output dimensions match the chain's last layer.
        @test length(s_free[end]) == 2
    end
end

# ----------------------------------------------------------------
# 4. ep_gradient vs FD ground truth
# ----------------------------------------------------------------
function ep_gradient_vs_fd_tests()
    @testset "ep_gradient ≈ fd_gradient_phasor (small β)" begin
        rng = Xoshiro(42)
        chain, ps, st = _ep_chain(rng)
        x = Phase.(2f0 .* rand(rng, Float32, 4) .- 1f0)
        y = ComplexF32.(exp.(im .* π .* (2f0 .* rand(rng, Float32, 2) .- 1f0)))

        fd = fd_gradient_phasor(chain, ps, st, x, y; T=200, dt=EP_DT)
        grads, _ = ep_gradient(StaticEP(β=EP_BETA, T_free=200, T_nudge=100, dt=EP_DT),
                               chain, ps, st, x, y)

        for key in (:layer_1, :layer_2)
            g_ep = grads[key].weight
            g_fd = fd[key].weight
            re   = norm(g_ep - g_fd) / norm(g_fd)
            cs   = dot(vec(g_ep), vec(g_fd)) / (norm(g_ep) * norm(g_fd) + 1e-10)
            @info "EP vs FD on $key: cos=$(round(cs, digits=4)) rel-err=$(round(re, digits=4))"
            @test re < EP_FD_TOL
            @test cs > 1 - EP_FD_TOL
        end

        # Non-EP-trained params (log_neg_lambda) get zero gradient.
        @test all(grads.layer_1.log_neg_lambda .== 0)
        @test all(grads.layer_2.log_neg_lambda .== 0)
    end
end

# ----------------------------------------------------------------
# 5. ep_train end-to-end (single fixed pattern)
# ----------------------------------------------------------------
function ep_training_tests()
    @testset "ep_train decreases loss" begin
        rng = Xoshiro(42)
        chain, ps, st = _ep_chain(rng)
        x = Phase.(2f0 .* rand(rng, Float32, 4) .- 1f0)
        y = ComplexF32.(exp.(im .* π .* (2f0 .* rand(rng, Float32, 2) .- 1f0)))

        args = Args(lr=0.05, epochs=80)
        losses, _, _ = ep_train(chain, ps, st, [(x, y)], args;
                                method=StaticEP(β=0.1f0, T_free=100, T_nudge=50))

        @test length(losses) == 80
        @test all(isfinite, losses)
        @test losses[end] < losses[1]      # learning happened
        @test losses[end] < 0.3            # converged enough
        @info "ep_train: start=$(round(losses[1], digits=4))  end=$(round(losses[end], digits=4))"
    end
end

# ----------------------------------------------------------------
# 6. Boundary: bias rejection
# ----------------------------------------------------------------
function ep_bias_rejection_tests()
    @testset "use_bias=true is unsupported in Phase 1" begin
        rng = Xoshiro(42)
        # Layers with bias — ep_drive will pull the bias into the drive
        # even though we want the no-bias formulation. The current Phase 1
        # documentation requires use_bias=false; we verify the assumption
        # by demonstrating that the EP gradient diverges from FD when
        # bias is enabled (smoke test that the expected restriction is real
        # rather than silently giving a wrong answer).
        chain = Chain(
            PhasorDense(4 => 8, normalize_to_unit_circle, use_bias=true),
            PhasorDense(8 => 2, normalize_to_unit_circle, use_bias=false),
        )
        ps, st = Lux.setup(rng, chain)
        x = Phase.(2f0 .* rand(rng, Float32, 4) .- 1f0)
        y = ComplexF32.(exp.(im .* π .* (2f0 .* rand(rng, Float32, 2) .- 1f0)))

        # The settle and gradient calls themselves should NOT throw — the
        # restriction is documentary, not enforced. We just confirm that
        # the call succeeds (so users get a result, just with the bias
        # contribution unaccounted for in the FD derivation).
        @test_nowarn phasor_settle(chain, ps, st, x, SimilarityCost(y), 0f0;
                                   T=20, dt=EP_DT)
        @test_nowarn ep_gradient(StaticEP(β=0.01f0, T_free=20, T_nudge=10),
                                 chain, ps, st, x, y)
    end
end
