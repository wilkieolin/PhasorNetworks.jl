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

        # With non-zero ω (and a smaller dt to keep settling stable),
        # K_mode=:stored produces a genuinely different equilibrium.
        ω03 = [fill(0.3f0, n1), fill(0.3f0, n2)]
        s_zero_ω   = phasor_settle(chain, ps, st, x, cost, 0f0;
                                    T=400, dt=0.1f0, K_mode=:zero,
                                    omega_override=ω03)
        s_stored_ω = phasor_settle(chain, ps, st, x, cost, 0f0;
                                    T=400, dt=0.1f0, K_mode=:stored,
                                    omega_override=ω03)
        Δ = norm(s_stored_ω[end] .- s_zero_ω[end])
        @test Δ > 1e-3
        @info "K_mode :zero vs :stored with ω=0.3: Δ = $(round(Δ, digits=4))"
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
