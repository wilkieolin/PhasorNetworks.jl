# Holomorphic Equilibrium Propagation Tests

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
        hep_demodulation_tests()
        hep_equilibrium_tests()
        hep_energy_tests()
        hep_gradient_tests()
        holomorphic_readout_tests()
        hep_aligned_training_tests()
    end
end

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
        @test holotanh_deriv(Float32[0.0])[1] ≈ 1.0f0 atol=1e-5
        @test d ≈ 1.0f0 .- tanh.(x).^2 atol=1e-5

        xc = ComplexF32.(randn(5), randn(5))
        dc = holotanh_deriv(xc)
        @test eltype(dc) <: Complex
    end
end

function hep_cost_tests()
    @testset "holomorphic cost function" begin
        rng = Xoshiro(42)
        z = Float32.(randn(rng, 3, 4))
        y = Float32.([1 0 0 1; 0 1 0 0; 0 0 1 0])
        c = hep_cost_xent(z, y)
        @test isfinite(c)
        @test real(c) > 0

        zc = ComplexF32.(randn(rng, 3, 4), randn(rng, 3, 4) * 0.1f0)
        cc = hep_cost_xent(zc, y)
        @test isfinite(real(cc))

        g = hep_cost_xent_grad(z, y)
        @test size(g) == size(z)
        @test all(isfinite, g)
        col_sums = sum(g, dims=1)
        @test all(abs.(col_sums) .< 1f-5)

        # Finite difference verification
        eps = 1f-4
        g_fd = similar(z)
        c0 = real(hep_cost_xent(z, y))
        for i in eachindex(z)
            zp = copy(z); zp[i] += eps
            g_fd[i] = (real(hep_cost_xent(zp, y)) - c0) / eps
        end
        @test g ≈ g_fd atol=2e-2
    end
end

function hep_phasor_kernel_tests()
    @testset "phasor kernel A, B" begin
        lnl = Float32.(log.([0.2, 0.5, 0.1]))
        omega = Float32.([2pi, 4pi, pi])
        A, B = PhasorNetworks._phasor_AB(lnl, omega, 0.1f0)

        @test length(A) == 3
        @test length(B) == 3
        @test all(abs.(A) .< 1.0f0)
        @test all(isfinite, B)

        k1 = -exp(lnl[1]) + im * omega[1]
        @test A[1] ≈ exp(k1 * 0.1f0) atol=1e-5
        @test B[1] ≈ (exp(k1 * 0.1f0) - 1) / k1 atol=1e-5
    end
end

function hep_demodulation_tests()
    @testset "demodulation" begin
        rng = Xoshiro(42)
        omega = Float32.([2pi, 4pi])
        dt = 0.1f0

        # A pure carrier at omega should demodulate to a constant
        n = 10
        z_carrier = exp.(ComplexF32.(im .* omega .* (n * dt)))
        phi = PhasorNetworks._demodulate(z_carrier, ComplexF32.(omega), n, dt)
        @test all(abs.(phi .- 1.0f0) .< 1f-5)

        # A carrier with phase offset should preserve the offset
        offset = Float32.([0.3, -0.7])
        z_offset = exp.(ComplexF32.(im .* (omega .* (n * dt) .+ offset)))
        phi_offset = PhasorNetworks._demodulate(z_offset, ComplexF32.(omega), n, dt)
        @test all(abs.(abs.(phi_offset) .- 1.0f0) .< 1f-5)
        @test angle.(phi_offset) ≈ offset atol=1e-4

        # Demodulation should be holomorphic in z (linear operation)
        z1 = ComplexF32.(randn(rng, 4), randn(rng, 4))
        z2 = ComplexF32.(randn(rng, 4), randn(rng, 4))
        om = ComplexF32.(fill(Float32(2pi), 4))
        alpha = 0.5f0 + 0.3f0im
        lhs = PhasorNetworks._demodulate(alpha .* z1 .+ z2, om, 5, dt)
        rhs = alpha .* PhasorNetworks._demodulate(z1, om, 5, dt) .+ PhasorNetworks._demodulate(z2, om, 5, dt)
        @test lhs ≈ rhs atol=1e-5
    end
end

function hep_equilibrium_tests()
    @testset "hep_equilibrium (demodulated)" begin
        rng = Xoshiro(42)
        W1 = Float32.(randn(rng, 8, 2)) * 0.2f0
        W2 = Float32.(randn(rng, 3, 8)) * 0.2f0
        lnl1 = fill(Float32(log(0.2)), 8)
        lnl2 = fill(Float32(log(0.2)), 3)
        omega1 = fill(Float32(2pi), 8)
        omega2 = fill(Float32(2pi), 3)

        x = ComplexF32.(randn(rng, 2, 4))
        y = Float32.([1 0 0 1; 0 1 0 0; 0 0 1 0])

        states = hep_equilibrium(
            (W1, W2), (nothing, nothing),
            ((lnl1, omega1), (lnl2, omega2)),
            x, 0.0f0, y; T=80, dt=0.1f0)

        @test length(states) == 2
        @test size(states[1]) == (8, 4)
        @test size(states[2]) == (3, 4)
        @test all(isfinite, states[1])
        @test all(isfinite, states[2])

        # States should have non-trivial magnitude
        @test maximum(abs.(states[1])) > 1f-4
        @test maximum(abs.(states[2])) > 1f-4

        # Complex beta should work
        states_c = hep_equilibrium(
            (W1, W2), (nothing, nothing),
            ((lnl1, omega1), (lnl2, omega2)),
            x, 0.5f0 + 0.5f0im, y; T=80, dt=0.1f0)
        @test eltype(states_c[1]) <: Complex
        @test all(isfinite, states_c[1])
    end
end

function hep_energy_tests()
    @testset "hep_energy" begin
        rng = Xoshiro(42)
        W1 = Float32.(randn(rng, 8, 2)) * 0.2f0
        W2 = Float32.(randn(rng, 3, 8)) * 0.2f0
        omega1 = ComplexF32.(fill(Float32(2pi), 8))
        omega2 = ComplexF32.(fill(Float32(2pi), 3))

        x = ComplexF32.(randn(rng, 2, 4))
        y = Float32.([1 0 0 1; 0 1 0 0; 0 0 1 0])

        s1 = ComplexF32.(randn(rng, 8, 4)) * 0.1f0
        s2 = ComplexF32.(randn(rng, 3, 4)) * 0.1f0

        E0 = hep_energy((s1, s2), (W1, W2), (nothing, nothing),
                         (omega1, omega2), x, y, 0.0f0; dt=0.1f0)
        @test isfinite(real(E0))

        E1 = hep_energy((s1, s2), (W1, W2), (nothing, nothing),
                         (omega1, omega2), x, y, 1.0f0; dt=0.1f0)
        @test E1 != E0
    end
end

function hep_gradient_tests()
    @testset "hep_gradient (contour integration)" begin
        rng = Xoshiro(42)
        W1 = Float32.(randn(rng, 8, 2)) * 0.15f0
        W2 = Float32.(randn(rng, 3, 8)) * 0.15f0
        lnl1 = fill(Float32(log(0.3)), 8)
        lnl2 = fill(Float32(log(0.3)), 3)
        omega1 = fill(Float32(2pi), 8)
        omega2 = fill(Float32(2pi), 3)

        x = ComplexF32.(randn(rng, 2, 8))
        y = zeros(Float32, 3, 8)
        for i in 1:8; y[mod1(i, 3), i] = 1.0f0; end

        w_grads, b_grads = hep_gradient(
            (W1, W2), (nothing, nothing),
            ((lnl1, omega1), (lnl2, omega2)),
            x, y; N=4, r=0.3f0, T_free=80, T_nudge=30, dt=0.1f0)

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
                x, y; N=4, r=r_val, T_free=80, T_nudge=30, dt=0.1f0)
            @test all(isfinite, wg[1])
            @test all(isfinite, wg[2])
        end
    end
end

function holomorphic_readout_tests()
    @testset "HolomorphicReadout" begin
        rng = Xoshiro(42)
        hr = HolomorphicReadout(8 => 3)
        ps_hr, st_hr = Lux.setup(rng, hr)

        @test haskey(st_hr, :codes_conj)
        @test size(st_hr.codes_conj) == (8, 3)
        @test eltype(st_hr.codes_conj) <: Complex

        z_cmpx = ComplexF32.(randn(rng, 8, 4), randn(rng, 8, 4))
        logits_c, _ = hr(z_cmpx, ps_hr, st_hr)
        @test size(logits_c) == (3, 4)
        @test eltype(logits_c) <: Complex
        @test all(isfinite, logits_c)

        z_phase = Phase.(2f0 .* rand(rng, Float32, 8, 4) .- 1f0)
        logits_r, _ = hr(z_phase, ps_hr, st_hr)
        @test size(logits_r) == (3, 4)
        @test eltype(logits_r) <: Real
        @test all(isfinite, logits_r)
        @test all(abs.(logits_r) .<= 1.0f0 + 0.1f0)

        # Perfect match test
        z_match = conj.(st_hr.codes_conj[:, 1])
        z_match = reshape(z_match, :, 1)
        logits_match, _ = hr(z_match, ps_hr, st_hr)
        @test real(logits_match[1]) > 0.9f0
        @test real(logits_match[1]) > real(logits_match[2])
        @test real(logits_match[1]) > real(logits_match[3])
    end
end

function hep_aligned_training_tests()
    @testset "hep_train aligned (HolomorphicReadout)" begin
        args = Args(lr=1e-3, epochs=3, batchsize=64)

        model = Chain(
            x -> ComplexF32.(x),
            PhasorDense(2 => 32, holotanh, use_bias=false),
            PhasorDense(32 => 3, holotanh, use_bias=false),
            HolomorphicReadout(3 => 2)
        )
        ps, st = Lux.setup(args.rng, model)

        x_test = Float32.(randn(args.rng, 2, 4))
        y_pred, _ = model(x_test, ps, st)
        @test size(y_pred) == (2, 4)

        train_loader = [bullseye_data(args.batchsize, args.rng) for _ in 1:20]

        losses, ps_trained, _ = hep_train(
            model, ps, st, train_loader, args;
            N=4, r=0.3f0, dt=0.1f0,
            T_free=60, T_nudge=20)

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
