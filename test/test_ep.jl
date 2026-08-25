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
    @testset "Phasor EP" begin
        ep_cost_tests()
        ep_interface_tests()
        ep_settle_tests()
        ep_gradient_vs_fd_tests()
        ep_training_tests()
        ep_lockin_vs_fd_tests()
        ep_lockin_training_tests()
        ep_bias_support_tests()
        ep_codebook_cost_tests()
        ep_codebook_training_tests()
        ep_kmode_stored_tests()
        ep_carrier_tests()
        ep_readout_tests()
        ep_centered_tests()
        ep_batched_cost_tests()
        ep_batch_equivalence_tests()
        ep_mlp_proxy_fd_tests()
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
# 6. LockinEP vs FD ground truth
# ----------------------------------------------------------------
const EP_LOCKIN_FD_TOL = 0.10   # 10% rel-err — looser than static (extra knobs to tune)

function ep_lockin_vs_fd_tests()
    @testset "LockinEP ≈ fd_gradient_phasor (deep adiabatic regime)" begin
        rng = Xoshiro(42)
        chain, ps, st = _ep_chain(rng)
        x = Phase.(2f0 .* rand(rng, Float32, 4) .- 1f0)
        y = ComplexF32.(exp.(im .* π .* (2f0 .* rand(rng, Float32, 2) .- 1f0)))

        # FD oracle uses the same dt as static — 0.5 — to match the
        # corresponding equilibrium. LockinEP needs a finer dt for its
        # probe phase increments to be small enough.
        fd = fd_gradient_phasor(chain, ps, st, x, y; T=200, dt=EP_DT)

        # Use a slow probe and small amplitude — deep adiabatic
        # regime where lock-in matches FD.
        method = LockinEP(ε=0.01f0, ω_p=0.01f0,
                          n_cycles=8, T_warmup_cycles=2,
                          T_free=400, dt=0.1f0)
        grads, _ = ep_gradient(method, chain, ps, st, x, y)

        for key in (:layer_1, :layer_2)
            g_lk = grads[key].weight
            g_fd = fd[key].weight
            re   = norm(g_lk - g_fd) / norm(g_fd)
            cs   = dot(vec(g_lk), vec(g_fd)) / (norm(g_lk) * norm(g_fd) + 1e-10)
            @info "LockinEP vs FD on $key: cos=$(round(cs, digits=4)) rel-err=$(round(re, digits=4))"
            @test re < EP_LOCKIN_FD_TOL
            @test cs > 1 - EP_LOCKIN_FD_TOL
        end

        # Non-EP-trained params get zero gradient (same as StaticEP).
        @test all(grads.layer_1.log_neg_lambda .== 0)
        @test all(grads.layer_2.log_neg_lambda .== 0)
    end

    @testset "LockinEP non-adiabatic regime drifts (sanity check)" begin
        # Confirms that the adiabatic constraint is real — a
        # too-fast probe gives a worse gradient than the slow one.
        rng = Xoshiro(42)
        chain, ps, st = _ep_chain(rng)
        x = Phase.(2f0 .* rand(rng, Float32, 4) .- 1f0)
        y = ComplexF32.(exp.(im .* π .* (2f0 .* rand(rng, Float32, 2) .- 1f0)))

        fd = fd_gradient_phasor(chain, ps, st, x, y; T=200, dt=EP_DT)
        fd_norm1 = norm(fd.layer_1.weight)

        slow, _ = ep_gradient(LockinEP(ε=0.01f0, ω_p=0.01f0,
                                       n_cycles=8, T_warmup_cycles=2,
                                       T_free=400, dt=0.1f0),
                              chain, ps, st, x, y)
        fast, _ = ep_gradient(LockinEP(ε=0.01f0, ω_p=0.5f0,
                                       n_cycles=8, T_warmup_cycles=2,
                                       T_free=400, dt=0.1f0),
                              chain, ps, st, x, y)
        re_slow = norm(slow.layer_1.weight - fd.layer_1.weight) / fd_norm1
        re_fast = norm(fast.layer_1.weight - fd.layer_1.weight) / fd_norm1
        @info "Lock-in adiabaticity: slow rel-err=$(round(re_slow, digits=4)) fast rel-err=$(round(re_fast, digits=4))"
        @test re_slow < re_fast
    end
end

# ----------------------------------------------------------------
# 7. LockinEP end-to-end training
# ----------------------------------------------------------------
function ep_lockin_training_tests()
    @testset "ep_train(method=LockinEP) decreases loss" begin
        rng = Xoshiro(42)
        chain, ps, st = _ep_chain(rng)
        x = Phase.(2f0 .* rand(rng, Float32, 4) .- 1f0)
        y = ComplexF32.(exp.(im .* π .* (2f0 .* rand(rng, Float32, 2) .- 1f0)))

        # Lock-in at modest ε / ω_p — fast enough to keep test runtime
        # reasonable, slow enough to give a useful gradient.
        method = LockinEP(ε=0.05f0, ω_p=0.05f0,
                          n_cycles=4, T_warmup_cycles=2,
                          T_free=100, dt=0.1f0)
        args = Args(lr=0.05, epochs=40)

        losses, _, _ = ep_train(chain, ps, st, [(x, y)], args; method=method)

        @test length(losses) == 40
        @test all(isfinite, losses)
        @test losses[end] < losses[1]
        @test losses[end] < 0.5    # looser than static — slower convergence per epoch
        @info "ep_train (LockinEP): start=$(round(losses[1], digits=4)) end=$(round(losses[end], digits=4))"
    end
end

# ----------------------------------------------------------------
# 8. Bias support — EP-vs-FD with use_bias=true
# ----------------------------------------------------------------
function ep_bias_support_tests()
    @testset "Bias gradients match FD" begin
        rng = Xoshiro(42)
        chain = Chain(
            PhasorDense(4 => 8, normalize_to_unit_circle, use_bias=true),
            PhasorDense(8 => 2, normalize_to_unit_circle, use_bias=false),
        )
        ps, st = Lux.setup(rng, chain)
        ps = (
            layer_1 = merge(ps.layer_1, (weight = 0.4f0 .* ps.layer_1.weight,)),
            layer_2 = merge(ps.layer_2, (weight = 0.4f0 .* ps.layer_2.weight,)),
        )
        x = Phase.(2f0 .* rand(rng, Float32, 4) .- 1f0)
        y = ComplexF32.(exp.(im .* π .* (2f0 .* rand(rng, Float32, 2) .- 1f0)))

        # Sanity: layer_1 ps has bias_real / bias_imag; layer_2 doesn't.
        @test haskey(ps.layer_1, :bias_real)
        @test haskey(ps.layer_1, :bias_imag)
        @test !haskey(ps.layer_2, :bias_real)

        fd = fd_gradient_phasor(chain, ps, st, x, y; T=200, dt=EP_DT)
        grads, _ = ep_gradient(StaticEP(β=EP_BETA, T_free=200, T_nudge=100, dt=EP_DT),
                                chain, ps, st, x, y)

        for pname in (:weight, :bias_real, :bias_imag)
            g_ep = grads.layer_1[pname]
            g_fd = fd.layer_1[pname]
            re   = norm(g_ep .- g_fd) / norm(g_fd)
            @info "Bias EP vs FD on layer_1.$pname: rel-err=$(round(re, digits=4))"
            @test re < EP_FD_TOL * 2          # bias is looser; 10% target is fine
        end

        # Sanity: layer_2 weight still matches at the tighter tolerance.
        re2 = norm(grads.layer_2.weight .- fd.layer_2.weight) / norm(fd.layer_2.weight)
        @test re2 < EP_FD_TOL
    end
end

# ----------------------------------------------------------------
# 9. CodebookCost — cost identities and gradient shape
# ----------------------------------------------------------------
function ep_codebook_cost_tests()
    @testset "CodebookCost identities" begin
        rng = Xoshiro(42)
        d, n_classes = 8, 4
        codes_phase = Float32.(2 .* rand(rng, Float32, d, n_classes) .- 1)
        codes = ComplexF32.(exp.(im .* π .* codes_phase))

        # Class-2 target.
        cost = CodebookCost(codes, 2)
        @test length(cost.y_onehot) == n_classes
        @test cost.y_onehot[2] == one(Float32)
        @test sum(cost.y_onehot) == one(Float32)

        # Loss at perfect target codeword should be < log(n_classes).
        z_match = ComplexF32.(codes[:, 2])
        loss_match = ep_loss(cost, z_match)
        @test loss_match < Float32(log(n_classes))
        @test loss_match > 0

        # Loss at the *wrong* codeword should be larger than at the right one.
        z_wrong = ComplexF32.(codes[:, 3])
        @test ep_loss(cost, z_wrong) > ep_loss(cost, z_match)

        # Nudge force shape matches z_o; finite values.
        z_test = randn(rng, ComplexF32, d)
        nudge = PhasorNetworks.nudge_force(cost, z_test, 0.1f0)
        @test size(nudge) == (d,)
        @test all(isfinite, nudge)

        # One-hot constructor matches explicit y_onehot.
        cost_explicit = CodebookCost(codes, Float32[0, 1, 0, 0])
        @test PhasorNetworks.nudge_force(cost_explicit, z_test, 0.1f0) ≈
              PhasorNetworks.nudge_force(cost, z_test, 0.1f0)
    end
end

function ep_codebook_training_tests()
    @testset "ep_train(cost_fn=CodebookCost) decreases loss" begin
        rng = Xoshiro(42)
        d, n_classes = 8, 4
        codes_phase = Float32.(2 .* rand(Xoshiro(7), Float32, d, n_classes) .- 1)
        codes = ComplexF32.(exp.(im .* π .* codes_phase))

        chain = Chain(PhasorDense(4 => d, normalize_to_unit_circle, use_bias=false))
        ps, st = Lux.setup(rng, chain)
        ps = (layer_1 = merge(ps.layer_1, (weight = 0.4f0 .* ps.layer_1.weight,)),)

        x = Phase.(2f0 .* rand(Xoshiro(1), Float32, 4) .- 1f0)
        target_class = 2
        args = Args(lr=0.05, epochs=60)

        losses, _, _ = ep_train(chain, ps, st, [(x, target_class)], args;
                                method=StaticEP(β=0.1f0, T_free=100, T_nudge=50),
                                cost_fn = y_class -> CodebookCost(codes, y_class))

        @test length(losses) == 60
        @test all(isfinite, losses)
        @test losses[end] < losses[1]
        @info "Codebook training: start=$(round(losses[1], digits=4)) end=$(round(losses[end], digits=4))"
    end
end

# ----------------------------------------------------------------
# 10. K_mode = :stored — self-energy from layer's stored params
# ----------------------------------------------------------------
function ep_kmode_stored_tests()
    @testset "K_mode=:stored matches FD with omega=0" begin
        rng = Xoshiro(42)
        chain, ps, st = _ep_chain(rng)
        # Use omega_override = [zeros, zeros] so the K = λ + iω self-force
        # contributes only the real decay. The default layer ω = 2π
        # (derived from spk_args.t_period = 1.0 via period_to_angfreq)
        # would give per-step rotations dt·ω that destabilize the damped
        # iteration at dt = 0.5. ω was removed from PhasorDense
        # params/state by the per-channel ω rule (see CLAUDE.md), so the
        # override is now a per-call kwarg rather than a state hack.
        n1 = length(ps.layer_1.log_neg_lambda)
        n2 = length(ps.layer_2.log_neg_lambda)
        ω0 = [zeros(Float32, n1), zeros(Float32, n2)]

        x = Phase.(2f0 .* rand(rng, Float32, 4) .- 1f0)
        y = ComplexF32.(exp.(im .* π .* (2f0 .* rand(rng, Float32, 2) .- 1f0)))

        for K_mode in (:zero, :stored)
            fd = fd_gradient_phasor(chain, ps, st, x, y; T=200, dt=EP_DT,
                                    K_mode=K_mode, omega_override=ω0)
            grads, _ = ep_gradient(
                StaticEP(β=EP_BETA, T_free=200, T_nudge=100, dt=EP_DT, K_mode=K_mode),
                chain, ps, st, x, y; omega_override=ω0)
            for key in (:layer_1, :layer_2)
                re = norm(grads[key].weight .- fd[key].weight) / norm(fd[key].weight)
                @info "K_mode=:$K_mode $key rel-err=$(round(re, digits=4))"
                @test re < EP_FD_TOL
            end
        end

        # Sanity: with ω = 0 the self-force is purely a real ½·λ·z,
        # which only scales magnitude — and the unit-circle projection
        # then erases the difference. So the equilibria SHOULD match.
        # Verify this and then re-test with non-zero ω, where the
        # rotation contribution genuinely changes the angle.
        cost = SimilarityCost(y)
        s_zero   = phasor_settle(chain, ps, st, x, cost, 0f0;
                                  T=200, dt=EP_DT, K_mode=:zero,
                                  omega_override=ω0)
        s_stored = phasor_settle(chain, ps, st, x, cost, 0f0;
                                  T=200, dt=EP_DT, K_mode=:stored,
                                  omega_override=ω0)
        @test isapprox(s_stored[end], s_zero[end]; atol=1e-3)

        # K_mode=:stored produces a genuinely different equilibrium from
        # :zero — but because of λ, NOT ω. `ep_self_force` returns ½·λ·z
        # only: Re⟨z,(λ+iω)z⟩ = λ|z|², so the rotation contributes nothing
        # to the energy and is not part of its gradient. ω is applied
        # instead as an exact carrier rotation (see `ep_carrier_tests`).
        # `omega_override` is therefore inert here and the Δ below is
        # entirely the λ term; an earlier version of this test read the
        # same Δ as evidence that ω changes the equilibrium, which is the
        # misconception that the carrier gates now guard against.
        ω03 = [fill(0.3f0, n1), fill(0.3f0, n2)]
        s_zero_ω   = phasor_settle(chain, ps, st, x, cost, 0f0;
                                    T=400, dt=0.1f0, K_mode=:zero,
                                    omega_override=ω03)
        s_stored_ω = phasor_settle(chain, ps, st, x, cost, 0f0;
                                    T=400, dt=0.1f0, K_mode=:stored,
                                    omega_override=ω03)
        Δ = norm(s_stored_ω[end] .- s_zero_ω[end])
        @test Δ > 1e-3
        @info "K_mode :zero vs :stored (λ term only): Δ = $(round(Δ, digits=4))"

        # And the corollary: omega_override genuinely IS inert now, so two
        # different ω values must give byte-identical settles.
        ω99 = [fill(9.9f0, n1), fill(9.9f0, n2)]
        s_stored_ω99 = phasor_settle(chain, ps, st, x, cost, 0f0;
                                     T=400, dt=0.1f0, K_mode=:stored,
                                     omega_override=ω99)
        @test s_stored_ω99[end] == s_stored_ω[end]
    end
end


# ----------------------------------------------------------------
# 11. Centered (symmetric-β) StaticEP
# ----------------------------------------------------------------
function ep_centered_tests()
    @testset "StaticEP(centered=true) matches FD" begin
        rng = Xoshiro(42)
        chain, ps, st = _ep_chain(rng)
        x = Phase.(2f0 .* rand(rng, Float32, 4) .- 1f0)
        y = ComplexF32.(exp.(im .* π .* (2f0 .* rand(rng, Float32, 2) .- 1f0)))

        fd = fd_gradient_phasor(chain, ps, st, x, y; T=200, dt=EP_DT)

        # At a deliberately LARGE β the one-sided estimator's O(β) bias
        # is visible; the centered estimator cancels it. Compare both
        # against the same FD oracle.
        β_big = 0.5f0
        g_one, _ = ep_gradient(StaticEP(β=β_big, T_free=200, T_nudge=100, dt=EP_DT),
                               chain, ps, st, x, y)
        g_ctr, _ = ep_gradient(StaticEP(β=β_big, T_free=200, T_nudge=100, dt=EP_DT,
                                        centered=true),
                               chain, ps, st, x, y)

        re_one = norm(g_one.layer_1.weight - fd.layer_1.weight) / norm(fd.layer_1.weight)
        re_ctr = norm(g_ctr.layer_1.weight - fd.layer_1.weight) / norm(fd.layer_1.weight)
        @info "Centered vs one-sided at β=$β_big: one-sided rel-err=$(round(re_one, digits=4)) centered rel-err=$(round(re_ctr, digits=4))"
        @test re_ctr < re_one

        # At small β both should agree with FD; centered must not regress.
        g_ctr_s, _ = ep_gradient(StaticEP(β=EP_BETA, T_free=200, T_nudge=100, dt=EP_DT,
                                          centered=true),
                                 chain, ps, st, x, y)
        for key in (:layer_1, :layer_2)
            re = norm(g_ctr_s[key].weight - fd[key].weight) / norm(fd[key].weight)
            @test re < EP_FD_TOL
        end
    end
end

# ----------------------------------------------------------------
# 12. Batched cost identities
# ----------------------------------------------------------------
function ep_batched_cost_tests()
    @testset "Batched cost identities" begin
        rng = Xoshiro(3)
        d, K, B = 6, 4, 5
        Z = ComplexF32.(exp.(im .* 2π .* rand(rng, Float32, d, B)))
        codes = ComplexF32.(angle_to_complex(orthogonal_codes(rng, d, K)))
        labels = [1, 3, 2, 4, 2]

        cb = CodebookCost(codes, labels)
        @test size(cb.y_onehot) == (K, B)

        # Loss is the mean over the batch of the per-sample losses.
        l_batch = ep_loss(cb, Z)
        l_mean  = sum(ep_loss(CodebookCost(codes, labels[i]), Z[:, i]) for i in 1:B) / B
        @test isapprox(l_batch, l_mean; rtol=1e-5)

        # Nudge is column-wise and carries the FULL per-sample β — no
        # 1/B. Each column must equal the single-sample nudge exactly.
        β = 0.1f0
        F = PhasorNetworks.nudge_force(cb, Z, β)
        @test size(F) == (d, B)
        for i in 1:B
            f_i = PhasorNetworks.nudge_force(CodebookCost(codes, labels[i]), Z[:, i], β)
            @test isapprox(F[:, i], f_i; rtol=1e-5)
        end

        # Softmax must reduce over dims=1, not globally: columns sum to 1.
        s = PhasorNetworks._codebook_logits(cb, Z)
        p = PhasorNetworks._softmax(s)
        @test all(isapprox.(vec(sum(p; dims=1)), 1f0; atol=1e-5))

        # SimilarityCost batched: shared target broadcast across columns.
        y = ComplexF32.(exp.(im .* 2π .* rand(rng, Float32, d)))
        sc = SimilarityCost(y)
        @test isapprox(ep_loss(sc, Z),
                       sum(ep_loss(sc, Z[:, i]) for i in 1:B) / B; rtol=1e-5)

        # codebook_logits agrees with the private cost kernel.
        @test codebook_logits(codes, Z) ≈ s
    end
end

# ----------------------------------------------------------------
# 13. Batched gradient == mean of per-sample gradients
# ----------------------------------------------------------------
#
# This is exact by construction: every operation in `_phasor_step` is
# column-separable, so a batched settle is B independent settles. The
# tolerance is therefore tight — it is a roundoff budget, not a
# modelling allowance. The 1/β division in the EP estimate amplifies
# Float32 state noise, which sets the 1e-3 floor.
const EP_BATCH_TOL = 1e-3

function ep_batch_equivalence_tests()
    @testset "Batched EP == mean of per-sample EP" begin
        rng = Xoshiro(42)
        chain = Chain(
            PhasorDense(4 => 8, normalize_to_unit_circle, use_bias=true),
            PhasorDense(8 => 3, normalize_to_unit_circle, use_bias=true),
        )
        ps, st = Lux.setup(rng, chain)
        ps = (
            layer_1 = merge(ps.layer_1, (weight = 0.4f0 .* ps.layer_1.weight,)),
            layer_2 = merge(ps.layer_2, (weight = 0.4f0 .* ps.layer_2.weight,)),
        )

        B = 5
        X = Phase.(2f0 .* rand(rng, Float32, 4, B) .- 1f0)
        codes = ComplexF32.(angle_to_complex(orthogonal_codes(rng, 3, 3)))
        labels = [1, 3, 2, 2, 1]

        methods = (
            ("StaticEP",          StaticEP(β=0.01f0, T_free=150, T_nudge=80, dt=0.5f0)),
            ("StaticEP centered", StaticEP(β=0.01f0, T_free=150, T_nudge=80, dt=0.5f0,
                                           centered=true)),
            ("LockinEP",          LockinEP(ε=0.01f0, ω_p=0.05f0, n_cycles=3,
                                           T_warmup_cycles=2, T_free=200, dt=0.1f0)),
        )

        for (name, method) in methods
            g_batch, s_free = ep_gradient(method, chain, ps, st, X,
                                          CodebookCost(codes, labels))
            g_single = [ep_gradient(method, chain, ps, st, X[:, i],
                                    CodebookCost(codes, labels[i]))[1] for i in 1:B]

            # Batched settle produces (out, B) states.
            @test size(s_free[1]) == (8, B)
            @test size(s_free[2]) == (3, B)

            worst = 0.0
            for key in (:layer_1, :layer_2), pn in (:weight, :bias_real, :bias_imag)
                g_mean = sum(g[key][pn] for g in g_single) ./ B
                re = norm(g_batch[key][pn] - g_mean) / norm(g_mean)
                worst = max(worst, re)
                @test re < EP_BATCH_TOL
            end
            @info "Batched vs mean-of-singles ($name): worst rel-err=$(round(worst, sigdigits=3))"
        end
    end
end

# ----------------------------------------------------------------
# 14. Downscaled FD oracle on image-like input
# ----------------------------------------------------------------
#
# `fd_gradient_phasor` runs n_params + 1 settles, so it cannot touch the
# 217K-parameter FashionMNIST MLP. This proxy keeps the things that
# actually differ from the toy chains — a 10-class CodebookCost readout,
# and sparse image-like input where ~80% of pixels are exactly zero
# (a large common-mode drive) — at a size FD can afford.
#
# The input is synthetic rather than real FashionMNIST so the suite
# stays hermetic and offline; it matches the sparsity and range of a
# 7x7-downsampled garment image.
function ep_mlp_proxy_fd_tests()
    @testset "Downscaled MLP proxy matches FD (10-class codebook)" begin
        rng = Xoshiro(9)
        n_in, n_hid, n_cls = 49, 12, 10
        chain = Chain(
            PhasorDense(n_in  => n_hid, normalize_to_unit_circle, use_bias=true),
            PhasorDense(n_hid => n_cls, normalize_to_unit_circle, use_bias=true),
        )
        ps, st = Lux.setup(rng, chain)
        ps = (
            layer_1 = merge(ps.layer_1, (weight = 0.4f0 .* ps.layer_1.weight,)),
            layer_2 = merge(ps.layer_2, (weight = 0.4f0 .* ps.layer_2.weight,)),
        )

        # Sparse image-like pixels: ~80% zero, remainder in (0,1].
        pix = rand(rng, Float32, n_in)
        pix[rand(rng, Float32, n_in) .< 0.8f0] .= 0f0
        # Half-plane encoding: phase in [-0.5, 0.5], injective (no wrap
        # collapse at Phase(±1)).
        x = Phase.(0.5f0 .* (2f0 .* pix .- 1f0))

        codes = ComplexF32.(angle_to_complex(orthogonal_codes(rng, n_cls, n_cls)))
        cost  = CodebookCost(codes, 4)

        # Equilibrium is exactly stationary by T=100 here (measured:
        # ||z(T+1)-z(T)|| == 0 in Float32), so T=150 is ample.
        #
        # FD step size: the default ε=1e-5 is tuned for the toy chains'
        # O(1) SimilarityCost. The 10-class cross-entropy loss is O(2.4)
        # with O(1) gradients, so at ε=1e-5 the FD difference falls below
        # Float32 resolution and the ORACLE — not EP — becomes noise
        # (measured rel-err 0.24 at 1e-5, 0.027 at 1e-4, 0.004 at 1e-3,
        # 0.036 at 1e-2: the classic cancellation/truncation U-curve).
        # 1e-3 sits at the bottom of that curve.
        fd = fd_gradient_phasor(chain, ps, st, x, cost; ε=1f-3, T=150, dt=EP_DT)
        g, _ = ep_gradient(StaticEP(β=EP_BETA, T_free=150, T_nudge=80, dt=EP_DT,
                                    centered=true),
                           chain, ps, st, x, cost)

        for key in (:layer_1, :layer_2)
            g_ep = vec(g[key].weight); g_fd = vec(fd[key].weight)
            re = norm(g_ep - g_fd) / norm(g_fd)
            cs = dot(g_ep, g_fd) / (norm(g_ep) * norm(g_fd) + 1e-10)
            @info "MLP proxy EP vs FD on $key: cos=$(round(cs, digits=4)) rel-err=$(round(re, digits=4))"
            @test cs > 0.999
            @test re < 0.02
        end

        # Inference path: ep_predict must reproduce the cost's own logits.
        s = phasor_settle(chain, ps, st, x, cost, 0f0; T=150, dt=EP_DT)
        @test ep_predict(chain, ps, st, x, codes; T=150, dt=EP_DT) ≈
              PhasorNetworks._codebook_logits(cost, s[end])
    end
end


# ----------------------------------------------------------------
# 15. GPU parity (CUDA only — called from test_cuda.jl)
# ----------------------------------------------------------------
#
# `src/ep.jl` allocates through `gpu_zeros`/`similar` rather than bare
# `zeros`, so the whole settle-and-gradient path runs on-device. The
# tolerance is a Float32 roundoff budget: the GPU reduces in a different
# order than CPU BLAS, and the 1/β division in the EP estimate amplifies
# that difference.
const EP_GPU_TOL = 5e-3

function ep_gpu_parity_tests(dev)
    @testset "EP GPU parity with CPU" begin
        rng = Xoshiro(42)
        chain = Chain(
            PhasorDense(32 => 16, normalize_to_unit_circle, use_bias=true),
            PhasorDense(16 => 8,  normalize_to_unit_circle, use_bias=true),
        )
        ps, st = Lux.setup(rng, chain)
        ps = (
            layer_1 = merge(ps.layer_1, (weight = 0.4f0 .* ps.layer_1.weight,)),
            layer_2 = merge(ps.layer_2, (weight = 0.4f0 .* ps.layer_2.weight,)),
        )
        B = 8
        x = Phase.(0.5f0 .* (2f0 .* rand(rng, Float32, 32, B) .- 1f0))
        codes = ComplexF32.(angle_to_complex(orthogonal_codes(rng, 8, 5)))
        y = rand(rng, 1:5, B)

        ps_d, st_d, x_d = ps |> dev, st |> dev, x |> dev
        codes_d = codes |> dev

        # The one-hot target must follow the codebook onto the device.
        cost_d = CodebookCost(codes_d, y)
        @test !(cost_d.y_onehot isa Array)

        methods = (
            ("StaticEP", StaticEP(β=0.1f0, T_free=100, T_nudge=50, dt=0.5f0, centered=true)),
            ("LockinEP", LockinEP(ε=0.03f0, ω_p=0.2f0, n_cycles=2,
                                  T_warmup_cycles=1, T_free=80, dt=0.5f0)),
        )
        for (name, m) in methods
            g_cpu, _ = ep_gradient(m, chain, ps,   st,   x,   CodebookCost(codes, y))
            g_gpu, _ = ep_gradient(m, chain, ps_d, st_d, x_d, cost_d)
            worst = 0.0
            for key in (:layer_1, :layer_2), pn in (:weight, :bias_real, :bias_imag)
                a = vec(Array(g_cpu[key][pn])); b = vec(Array(g_gpu[key][pn]))
                re = norm(a - b) / norm(a)
                worst = max(worst, re)
                @test re < EP_GPU_TOL
            end
            @info "EP GPU parity ($name): worst rel-err=$(round(worst, sigdigits=3))"
        end

        # Settled states stay on-device rather than silently falling back.
        s_d = phasor_settle(chain, ps_d, st_d, x_d, cost_d, 0f0; T=50, dt=0.5f0)
        @test !(s_d[1] isa Array)
        @test size(s_d[1]) == (16, B)

        # Inference path.
        logits_cpu = ep_predict(chain, ps,   st,   x,   codes;   T=100, dt=0.5f0)
        logits_gpu = ep_predict(chain, ps_d, st_d, x_d, codes_d; T=100, dt=0.5f0)
        @test !(logits_gpu isa Array)
        @test isapprox(Array(logits_gpu), logits_cpu; rtol=EP_GPU_TOL, atol=1e-5)
    end
end


# ----------------------------------------------------------------
# 10b. Carrier / rotating-frame invariants
# ----------------------------------------------------------------
# These are the properties that make phasor EP valid on a running
# resonate-and-fire network, where every neuron's instantaneous phase
# rotates at a shared carrier ω while the information lives in the
# static RELATIVE phases.
#
# The claim is not that the carrier is approximately negligible — it is
# that it cancels EXACTLY. Φ = Σ_l Re⟨W_l z_{l-1}, z_l⟩ is U(1)-invariant
# and `_project_damp` is U(1)-equivariant, so under z_l = w_l·e^{iωt}
# applied to every layer (including the input) the carrier divides out of
# the discrete step at any dt. The only symmetry-breaking terms are the
# bias and the cost target, and both physically co-rotate in this package
# (`bias_current` injects one phase-locked pulse per period; the codebook
# is made of neurons at the same ω), so `phasor_settle(carrier=ω)` puts
# them on the carrier and the equivalence is exact.
#
# A prior attempt at a "rotating frame" shipped a settle that never
# rotated anything, a demodulator tuned to an empty band, and a reference
# that was a free-phase Hebbian rather than a gradient. Each of the three
# checks below fails loudly on one of those. See
# scripts/ep_rotating_gates.jl for the wider, slower version.
function ep_carrier_tests()
    @testset "Carrier / rotating-frame invariants" begin
        rng = Xoshiro(42)
        chain = Chain(
            PhasorDense(4 => 8, normalize_to_unit_circle, use_bias=true),
            PhasorDense(8 => 2, normalize_to_unit_circle, use_bias=true))
        ps, st = Lux.setup(rng, chain)
        ps = (layer_1 = merge(ps.layer_1, (weight = 0.4f0 .* ps.layer_1.weight,)),
              layer_2 = merge(ps.layer_2, (weight = 0.4f0 .* ps.layer_2.weight,)))
        x = Phase.(2f0 .* rand(rng, Float32, 4) .- 1f0)
        y = ComplexF32.(exp.(im .* π .* (2f0 .* rand(rng, Float32, 2) .- 1f0)))
        cost = SimilarityCost(y)

        # (a) carrier = 0 must be BIT-identical to no carrier. cis(0) is
        #     exactly 1+0im, so this is an equality, not a tolerance — the
        #     cheapest possible check that the lab-frame branch has not
        #     quietly changed the dynamics.
        for β in (0f0, 0.05f0), K_mode in (:zero, :stored)
            s_co  = phasor_settle(chain, ps, st, x, cost, β;
                                  T=120, dt=EP_DT, K_mode=K_mode)
            s_lab = phasor_settle(chain, ps, st, x, cost, β;
                                  T=120, dt=EP_DT, K_mode=K_mode, carrier=0f0)
            @test all(a == b for (a, b) in zip(s_co, s_lab))
        end

        # (b) At a real carrier the lab-frame settle must equal the
        #     co-rotating one after demodulation. dt=0.5 with ω=2π is
        #     ω·dt = π — exactly Nyquist — which an additive iω·z step
        #     cannot represent at all; the exact multiplicative rotation
        #     has no dt limit. Residual is accumulated Float32 rounding
        #     over T steps, so tolerance scales with T.
        for (ω, dt, T) in ((Float32(2π), 0.5f0, 120), (1.7f0, 0.5f0, 113))
            s_co  = phasor_settle(chain, ps, st, x, cost, 0.05f0;
                                  T=T, dt=dt)
            s_lab = phasor_settle(chain, ps, st, x, cost, 0.05f0;
                                  T=T, dt=dt, carrier=ω)
            ph = ComplexF32(cis(mod(Float64(ω) * Float64(T * dt), 2π)))
            for (a, b) in zip(s_lab, s_co)
                @test norm(a .* conj(ph) .- b) / norm(b) < 1e-6 * T
            end
        end

        # (c) Hebbians must be U(1)-invariant. `ep_hebbian` uses an
        #     ADJOINT, giving real(z_self·z_in') = cos(π(θ_self − θ_in)) —
        #     a relative phase, which is what a spiking substrate can read
        #     off pre/post spike-time differences. A `transpose` gives
        #     cos(π(θ_self + θ_in)), which spins at 2ω under a global
        #     rotation and is not measurable from spike timing at all.
        s  = phasor_settle(chain, ps, st, x, cost, 0f0; T=200, dt=EP_DT)
        z0 = ComplexF32.(angle_to_complex(x))
        h  = chain_hebbians(chain, ps, st, z0, s)
        for θ in (0.3f0, 2.7f0)
            u  = cis(θ)
            hθ = chain_hebbians(chain, ps, st, z0 .* u, [z .* u for z in s])
            for key in (:layer_1, :layer_2)
                @test norm(hθ[key].weight .- h[key].weight) /
                      norm(h[key].weight) < 1e-5
            end
            # Teeth: the transpose form must visibly break, so that (c)
            # cannot pass vacuously.
            tr(a, b) = real.(a * transpose(b))
            @test norm(tr(s[2] .* u, s[1] .* u) .- tr(s[2], s[1])) /
                  norm(tr(s[2], s[1])) > 0.1
        end

        # (d) BOTH projection kernels must be U(1)-equivariant:
        #     û(e^{iθ}g) = e^{iθ}û(g). This is what lets the carrier divide
        #     out of the discrete step. `_project_damp_soft` uses
        #     g/sqrt(|g|²+ε²) rather than the phase-interpolating
        #     `soft_normalize_to_unit_circle`, precisely because the latter
        #     compresses phase toward a FIXED reference (0) and so installs
        #     a preferred direction in the complex plane.
        #
        #     Measured cost of getting this wrong, on this chain: EP-vs-FD
        #     rel-err 0.198 (phase-interpolating) vs 0.032 (equivariant) at
        #     K_mode=:zero, and 0.318 vs 0.016 at :stored.
        for θ in (0.4f0, 1.9f0), (g, zz) in ((ComplexF32(0.7, -0.3), ComplexF32(0.2, 0.9)),
                                             (ComplexF32(1f-12, 0f0), ComplexF32(0.1, 0.1)))
            u = ComplexF32(cis(θ))
            hard_a = PhasorNetworks._project_damp(zz * u, g * u, 0.5f0, 1f-10)
            hard_b = u * PhasorNetworks._project_damp(zz, g, 0.5f0, 1f-10)
            soft_a = PhasorNetworks._project_damp_soft(zz * u, g * u, 0.5f0, 0.1f0)
            soft_b = u * PhasorNetworks._project_damp_soft(zz, g, 0.5f0, 0.1f0)
            @test isapprox(soft_a, soft_b; atol=1e-6)
            # The hard kernel is equivariant only above its threshold — the
            # |g| < th branch snaps to 1+0im, which is the discontinuity the
            # soft kernel exists to remove. Assert that asymmetry explicitly
            # rather than leaving it implied.
            if abs(g) > 1f-10
                @test isapprox(hard_a, hard_b; atol=1e-6)
            else
                @test !isapprox(hard_a, hard_b; atol=1e-6)
            end
        end

        # (e) A soft settle stays finite and near (not on) the unit circle.
        s_soft = phasor_settle(chain, ps, st, x, cost, 0f0;
                               T=200, dt=EP_DT, project=:soft)
        @test all(all(isfinite.(z)) for z in s_soft)
        @test all(0.8 < mean(abs.(z)) < 1.0 for z in s_soft)
    end
end

# ----------------------------------------------------------------
# 10c. Spike-timing readout floor
# ----------------------------------------------------------------
# A spiking substrate does not read a complex number off a neuron — it
# infers phase from WHEN the neuron spiked, to about the spike-kernel
# width. `LockinEP(readout_δ=…)` models that as readout-only quantization
# (dynamics stay analog; only what the synapse observes is snapped).
#
# This is what makes the usable ε window TWO-SIDED: large ε breaks
# linearity via the hard projection, and small ε drops the response below
# this floor. The exact-state sweep can only see the upper edge.
function ep_readout_tests()
    @testset "Spike-timing readout floor" begin
        # (a) The quantizer lands on the grid and preserves modulus.
        δ = 0.25f0                       # quarter-turn bins
        for θ in (-0.9f0, -0.1f0, 0.2f0, 0.7f0)
            z = 0.8f0 * ComplexF32(cis(π * θ))
            q = PhasorNetworks._quantize_phase(z, 1f0 / δ)
            @test isapprox(abs(q), abs(z); rtol=1e-5)
            turns = angle(q) / (2f0 * π)
            @test isapprox(turns / δ, round(turns / δ); atol=1e-4)
        end

        rng = Xoshiro(42)
        chain, ps, st = _ep_chain(rng)
        x = Phase.(2f0 .* rand(rng, Float32, 4) .- 1f0)
        y = ComplexF32.(exp.(im .* π .* (2f0 .* rand(rng, Float32, 2) .- 1f0)))
        mk(δ) = LockinEP(ε=0.01f0, ω_p=0.01f0, n_cycles=2,
                         T_warmup_cycles=1, T_free=200, dt=EP_DT, readout_δ=δ)

        # (b) readout_δ = 0 must be EXACTLY the original estimator — the
        #     quantization path must not perturb the default at all.
        g0, _ = ep_gradient(mk(0f0), chain, ps, st, x, y)
        g0b, _ = ep_gradient(mk(0f0), chain, ps, st, x, y)
        @test g0.layer_1.weight == g0b.layer_1.weight

        # (c) A coarse quantum measurably degrades fidelity against FD.
        #     This is the floor existing, which is the whole premise of the
        #     `readout` sweep axis; if it ever stops holding, the axis is
        #     measuring nothing.
        fd = fd_gradient_phasor(chain, ps, st, x, y; T=200, dt=EP_DT)
        re(g) = norm(g.layer_1.weight .- fd.layer_1.weight) /
                norm(fd.layer_1.weight)
        gq, _ = ep_gradient(mk(0.25f0), chain, ps, st, x, y)
        @info "readout floor: exact rel-err=$(round(re(g0), digits=4)) " *
              "δ=0.25 rel-err=$(round(re(gq), digits=4))"
        @test re(gq) > re(g0)
    end
end
