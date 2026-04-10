# Holomorphic Equilibrium Propagation Tests
# Included and called from runtests.jl

using Test
using PhasorNetworks
using Statistics: mean
using Zygote: withgradient
using ComponentArrays

function hep_tests()
    @testset "Holomorphic EP Tests" begin
        holotanh_tests()
        holotanh_deriv_tests()
        hep_cost_tests()
        hep_phasor_kernel_tests()
        hep_equilibrium_tests()
        hep_energy_tests()
        hep_gradient_tests()
        holomorphic_readout_tests()
        hep_aligned_training_tests()
        hep_training_tests()
    end
end

# ----------------------------------------------------------------
# Activation tests
# ----------------------------------------------------------------

function holotanh_tests()
    @testset "holotanh activation" begin
        x_real = Float32.(randn(10, 4))
        y_real = holotanh(x_real)
        @test size(y_real) == (10, 4)
        @test all(abs.(y_real) .<= 1.0f0)
        @test y_real ≈ tanh.(x_real) atol=1e-5

        x_cmpx = ComplexF32.(randn(10, 4), randn(10, 4))
        y_cmpx = holotanh(x_cmpx)
        @test size(y_cmpx) == (10, 4)
        @test eltype(y_cmpx) <: Complex

        act = holotanh(2.0f0)
        @test act(x_real) ≈ holotanh(x_real; a=2.0f0) atol=1e-5
    end
end

function holotanh_deriv_tests()
    @testset "holotanh_deriv" begin
        x = Float32.(randn(10, 4))
        d = holotanh_deriv(x)
        @test size(d) == (10, 4)
        # tanh'(0) = 1
        @test holotanh_deriv(Float32[0.0])[1] ≈ 1.0f0 atol=1e-5
        # tanh'(x) = 1 - tanh(x)^2
        @test d ≈ 1.0f0 .- tanh.(x).^2 atol=1e-5
        # Works on complex
        xc = ComplexF32.(randn(5), randn(5))
        dc = holotanh_deriv(xc)
        @test eltype(dc) <: Complex
    end
end

# ----------------------------------------------------------------
# Cost function tests
# ----------------------------------------------------------------

function hep_cost_tests()
    @testset "holomorphic cost function" begin
        rng = Xoshiro(42)
        # Real logits
        z = Float32.(randn(rng, 3, 4))
        y = Float32.([1 0 0 1; 0 1 0 0; 0 0 1 0])
        c = hep_cost_xent(z, y)
        @test isfinite(c)
        @test real(c) > 0  # cross-entropy should be positive

        # Complex logits
        zc = ComplexF32.(randn(rng, 3, 4), randn(rng, 3, 4) * 0.1f0)
        cc = hep_cost_xent(zc, y)
        @test isfinite(real(cc))

        # Gradient
        g = hep_cost_xent_grad(z, y)
        @test size(g) == size(z)
        @test all(isfinite, g)
        # Gradient should sum to ~0 per sample (softmax constraint)
        col_sums = sum(g, dims=1)
        @test all(abs.(col_sums) .< 1f-5)

        # Verify gradient by finite difference
        eps = 1f-4
        g_fd = similar(z)
        c0 = real(hep_cost_xent(z, y))
        for i in eachindex(z)
            zp = copy(z)
            zp[i] += eps
            g_fd[i] = (real(hep_cost_xent(zp, y)) - c0) / eps
        end
        @test g ≈ g_fd atol=2e-2
    end
end

# ----------------------------------------------------------------
# Phasor kernel tests
# ----------------------------------------------------------------

function hep_phasor_kernel_tests()
    @testset "phasor kernel A, B" begin
        lnl = Float32.(log.([0.2, 0.5, 0.1]))  # log_neg_lambda
        omega = Float32.([2pi, 4pi, pi])
        A, B = PhasorNetworks._phasor_AB(lnl, omega, 1.0f0)

        @test length(A) == 3
        @test length(B) == 3
        # |A| < 1 (contractive)
        @test all(abs.(A) .< 1.0f0)
        # B is finite
        @test all(isfinite, B)

        # A = exp(k*dt), check manually for first channel
        k1 = -exp(lnl[1]) + im * omega[1]
        @test A[1] ≈ exp(k1) atol=1e-5
        @test B[1] ≈ (exp(k1) - 1) / k1 atol=1e-5
    end
end

# ----------------------------------------------------------------
# Equilibrium tests
# ----------------------------------------------------------------

function hep_equilibrium_tests()
    @testset "hep_equilibrium (coupled recurrence)" begin
        rng = Xoshiro(42)
        W1 = Float32.(randn(rng, 8, 2)) * 0.2f0
        W2 = Float32.(randn(rng, 3, 8)) * 0.2f0
        lnl1 = fill(Float32(log(0.2)), 8)
        lnl2 = fill(Float32(log(0.2)), 3)
        omega1 = fill(Float32(2pi), 8)
        omega2 = fill(Float32(2pi), 3)

        x = Float32.(randn(rng, 2, 4))
        y = Float32.([1 0 0 1; 0 1 0 0; 0 0 1 0])

        # Free phase
        states = hep_equilibrium(
            (W1, W2), (nothing, nothing),
            ((lnl1, omega1), (lnl2, omega2)),
            x, 0.0f0, y; T=80)

        @test length(states) == 2
        @test size(states[1]) == (8, 4)
        @test size(states[2]) == (3, 4)
        @test all(isfinite, states[1])
        @test all(isfinite, states[2])

        # Complex beta → complex states
        states_c = hep_equilibrium(
            (W1, W2), (nothing, nothing),
            ((lnl1, omega1), (lnl2, omega2)),
            x, 0.5f0 + 0.5f0im, y; T=80)
        @test eltype(states_c[1]) <: Complex

        # Convergence: more steps shouldn't change much
        states_80 = hep_equilibrium(
            (W1, W2), (nothing, nothing),
            ((lnl1, omega1), (lnl2, omega2)),
            x, 0.0f0, y; T=80)
        states_160 = hep_equilibrium(
            (W1, W2), (nothing, nothing),
            ((lnl1, omega1), (lnl2, omega2)),
            x, 0.0f0, y; T=160)
        diff = maximum(abs.(states_160[1] .- states_80[1]))
        @test diff < 0.5f0
    end
end

# ----------------------------------------------------------------
# Energy tests
# ----------------------------------------------------------------

function hep_energy_tests()
    @testset "hep_energy" begin
        rng = Xoshiro(42)
        W1 = Float32.(randn(rng, 8, 2)) * 0.2f0
        W2 = Float32.(randn(rng, 3, 8)) * 0.2f0
        x = Float32.(randn(rng, 2, 4))
        y = Float32.([1 0 0 1; 0 1 0 0; 0 0 1 0])

        s1 = holotanh(W1 * x)
        s2 = holotanh(W2 * s1)

        E0 = hep_energy((s1, s2), (W1, W2), (nothing, nothing), x, y, 0.0f0)
        @test isfinite(real(E0))

        E1 = hep_energy((s1, s2), (W1, W2), (nothing, nothing), x, y, 1.0f0)
        @test E1 != E0  # nudge changes energy

        # Complex beta
        Ec = hep_energy(
            (ComplexF32.(s1), ComplexF32.(s2)),
            (W1, W2), (nothing, nothing),
            ComplexF32.(x), y, 0.5f0 + 0.5f0im)
        @test isfinite(real(Ec))
    end
end

# ----------------------------------------------------------------
# Gradient tests
# ----------------------------------------------------------------

function hep_gradient_tests()
    @testset "hep_gradient (contour integration)" begin
        rng = Xoshiro(42)
        W1 = Float32.(randn(rng, 8, 2)) * 0.15f0
        W2 = Float32.(randn(rng, 3, 8)) * 0.15f0
        lnl1 = fill(Float32(log(0.3)), 8)
        lnl2 = fill(Float32(log(0.3)), 3)
        omega1 = fill(Float32(2pi), 8)
        omega2 = fill(Float32(2pi), 3)

        x = Float32.(randn(rng, 2, 8))
        y = Float32.(repeat([1 0 0; 0 1 0; 0 0 1][:, 1:3], 1, 1))
        # 3-class one-hot for 8 samples
        y = zeros(Float32, 3, 8)
        for i in 1:8
            y[mod1(i, 3), i] = 1.0f0
        end

        w_grads, b_grads = hep_gradient(
            (W1, W2), (nothing, nothing),
            ((lnl1, omega1), (lnl2, omega2)),
            x, y;
            N=4, r=0.3f0, T_free=80, T_nudge=30)

        @test length(w_grads) == 2
        @test size(w_grads[1]) == (8, 2)
        @test size(w_grads[2]) == (3, 8)
        @test all(isfinite, w_grads[1])
        @test all(isfinite, w_grads[2])
        @test maximum(abs.(w_grads[1])) > 1f-8
        @test maximum(abs.(w_grads[2])) > 1f-8

        # Stability across r values
        for r_val in [0.1f0, 0.5f0, 1.0f0]
            wg, _ = hep_gradient(
                (W1, W2), (nothing, nothing),
                ((lnl1, omega1), (lnl2, omega2)),
                x, y;
                N=4, r=r_val, T_free=80, T_nudge=30)
            @test all(isfinite, wg[1])
            @test all(isfinite, wg[2])
        end
    end
end

# ----------------------------------------------------------------
# HolomorphicReadout tests
# ----------------------------------------------------------------

function holomorphic_readout_tests()
    @testset "HolomorphicReadout" begin
        rng = Xoshiro(42)
        hr = HolomorphicReadout(8 => 3)
        ps_hr, st_hr = Lux.setup(rng, hr)

        # State should contain conjugated codes
        @test haskey(st_hr, :codes_conj)
        @test size(st_hr.codes_conj) == (8, 3)
        @test eltype(st_hr.codes_conj) <: Complex

        # Complex input → complex logits
        z_cmpx = ComplexF32.(randn(rng, 8, 4), randn(rng, 8, 4))
        logits_c, _ = hr(z_cmpx, ps_hr, st_hr)
        @test size(logits_c) == (3, 4)
        @test eltype(logits_c) <: Complex
        @test all(isfinite, logits_c)

        # Phase input → real logits (inference path)
        z_phase = Phase.(2f0 .* rand(rng, Float32, 8, 4) .- 1f0)
        logits_r, _ = hr(z_phase, ps_hr, st_hr)
        @test size(logits_r) == (3, 4)
        @test eltype(logits_r) <: Real
        @test all(isfinite, logits_r)

        # Logits should be in reasonable range (similarity-like)
        @test all(abs.(logits_r) .<= 1.0f0 + 0.1f0)

        # Perfect match: if z == conj(codes_conj) (i.e., the original code),
        # then z .* codes_conj = |code|^2 = 1 for unit-circle codes,
        # and the logit for the matching class should be ~1.
        z_match = conj.(st_hr.codes_conj[:, 1])  # recover original code
        # Reshape to (features, 1) batch
        z_match = reshape(z_match, :, 1)
        logits_match, _ = hr(z_match, ps_hr, st_hr)
        # First class logit should be ~1 (constructive interference)
        @test real(logits_match[1]) > 0.9f0
        # First class should dominate
        @test real(logits_match[1]) > real(logits_match[2])
        @test real(logits_match[1]) > real(logits_match[3])
    end
end

function hep_aligned_training_tests()
    @testset "hep_train aligned (HolomorphicReadout)" begin
        args = Args(lr=1e-3, epochs=3, batchsize=64)

        # Aligned model: holotanh activations + HolomorphicReadout
        # No normalize_to_unit_circle, no phase extraction during forward
        model = Chain(
            x -> ComplexF32.(x),          # real input → complex
            PhasorDense(2 => 32, holotanh, use_bias=false),
            PhasorDense(32 => 3, holotanh, use_bias=false),
            HolomorphicReadout(3 => 2)
        )
        ps, st = Lux.setup(args.rng, model)

        # Test forward pass works
        x_test = Float32.(randn(args.rng, 2, 4))
        y_pred, _ = model(x_test, ps, st)
        @test size(y_pred) == (2, 4)

        # Generate bullseye data — 2 classes
        train_loader = [bullseye_data(args.batchsize, args.rng) for _ in 1:20]

        losses, ps_trained, _ = hep_train(
            model, ps, st, train_loader, args;
            N=4, r=0.3f0, dt=1.0f0,
            T_free=60, T_nudge=20,
            activation=holotanh)

        @test length(losses) == 3 * 20
        @test all(isfinite, losses)

        n = length(losses)
        q = n ÷ 4
        early_avg = mean(losses[1:q])
        late_avg = mean(losses[end-q+1:end])
        @info "hEP aligned training: early_loss=$early_avg, late_loss=$late_avg"
        @test late_avg < early_avg + 0.5f0
    end
end

# ----------------------------------------------------------------
# Training tests (legacy, non-aligned)
# ----------------------------------------------------------------

function hep_training_tests()
    @testset "hep_train on bullseye" begin
        args = Args(lr=1e-3, epochs=3, batchsize=64)

        model = Chain(
            x -> Phase.(tanh_fast.(x)),
            x -> x,
            PhasorDense(2 => 32, normalize_to_unit_circle, use_bias=false),
            x -> x,
            PhasorDense(32 => 2, normalize_to_unit_circle, use_bias=false)
        )
        ps, st = Lux.setup(args.rng, model)

        train_loader = [bullseye_data(args.batchsize, args.rng) for _ in 1:20]

        losses, ps_trained, _ = hep_train(
            model, ps, st, train_loader, args;
            N=4, r=0.3f0, dt=1.0f0,
            T_free=60, T_nudge=20,
            activation=holotanh)

        @test length(losses) == 3 * 20
        @test all(isfinite, losses)

        n = length(losses)
        q = n ÷ 4
        early_avg = mean(losses[1:q])
        late_avg = mean(losses[end-q+1:end])
        @info "hEP training: early_loss=$early_avg, late_loss=$late_avg"
        # Loss should not diverge
        @test late_avg < early_avg + 0.5f0
    end
end
