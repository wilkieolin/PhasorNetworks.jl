# Tests for ExcitableWaveSheet — the excitable fork of PhasorWaveSheet.
# Entry point: excitable_tests(), called from runtests.jl.
#
# The central claims under test are the four excitability criteria the base
# sheet fails, plus exact reduction to the base sheet at α = 0. Every threshold
# here is loose relative to the measured value (see src/excitable.jl's
# calibration table) so the suite catches regressions, not noise.

function _exc_seed(N, R, amp, θ)
    cen = N ÷ 2
    return ComplexF32[(i - cen)^2 + (j - cen)^2 <= R^2 ? amp * θ : 0
                      for i in 1:N, j in 1:N]
end

# α = 0 and κ_r = 0 must reproduce PhasorWaveSheet exactly. This is the contract
# that makes the layer a *fork* rather than a replacement: it pins the new terms
# to be strictly additive, so a regression here means the base dynamics moved.
function test_excitable_reduces_to_base()
    N = 48; rng = Xoshiro(11)
    l = ExcitableWaveSheet(N, N; alpha = 0.0, kappa_r = 0.0, theta_frac = 1.4)
    ps, st = Lux.setup(rng, l)
    base = PhasorWaveSheet(N, N; transmit = :spike, init_theta_frac = 1.4)
    pb, sb = Lux.setup(Xoshiro(11), base)
    θ = exp(pb.log_theta[1])
    z0 = _exc_seed(N, 2, 20f0, θ)
    Zb = wave_simulate(base, pb, sb; z0 = z0, L = 20)
    Ze = excitable_simulate(l, ps, st; z0 = z0, L = 20).z
    @test size(Ze) == size(Zb)
    @test maximum(abs.(Zb .- Ze)) / maximum(abs.(Zb)) < 1e-5
end

function test_excitable_emit_limits()
    θ = 2.0f0; α = 3.0f0; β = 0.15f0
    z = ComplexF32[0.001, 0.5, 1.0, 2.0, 20.0, 200.0] .* θ
    e = excitable_emit(z, θ, α, β)
    # Far below threshold the boost is ~off and the emit is the linear medium z/θ.
    # NOT exactly off: the sigmoid has a floor, so the residual boost is
    # 1 + α·σ(−1/β) ≈ 1.0076 at the defaults. That is a real (small) leak of
    # regeneration into the subthreshold band, and it means `excitable_regime`
    # slightly UNDERSTATES rho_sub. Pin the magnitude so it cannot grow silently.
    @test abs(e[1] - z[1] / θ) / abs(z[1] / θ) < 2e-2
    @test abs(excitable_emit(ComplexF32[1f-4 * θ], θ, α, β)[1]) / (1f-4 * θ) < 1 / θ * 1.02
    # Far above threshold it saturates at (1+α), not at 1 — that is the whole point.
    @test isapprox(abs(e[end]), 1 + α; rtol = 1e-3)
    # Monotone in |z|, and α = 0 recovers the base normalisation exactly.
    @test issorted(abs.(e))
    e0 = excitable_emit(z, θ, 0.0f0, β)
    @test maximum(abs.(e0 .- z ./ sqrt.(abs2.(z) .+ θ^2))) < 1e-6
end

function test_excitable_regime_diagnostics()
    N = 48; rng = Xoshiro(3)
    l = ExcitableWaveSheet(N, N)
    ps, st = Lux.setup(rng, l)
    r = excitable_regime(l, ps, st)
    @test r.rho_sub < 1                 # stable rest state
    @test r.quiescent
    @test r.regenerative_margin > 1.5   # room above the coherence loss (~0.61)
    @test r.can_propagate
    @test r.refractory_recovery > 10
    # The base sheet fails the propagation criterion at ITS default — this is the
    # measured fact that motivates the whole fork, so pin it.
    lb = ExcitableWaveSheet(N, N; alpha = 0.0, theta_frac = 1.4)
    pb, sb = Lux.setup(rng, lb)
    @test excitable_regime(lb, pb, sb).regenerative_margin < 1.25
end

# Criterion 1: quiescence. A noise seed must never fire.
function test_excitable_quiescent()
    N = 64; rng = Xoshiro(5)
    l = ExcitableWaveSheet(N, N)
    ps, st = Lux.setup(rng, l)
    z0 = ComplexF32.(0.1f0 .* randn(rng, N, N) .+ 0.1f0im .* randn(rng, N, N))
    tr = excitable_simulate(l, ps, st; z0 = z0, L = 100, keep_fields = false)
    @test maximum(tr.rate) == 0
end

# Criteria 2 + 3: a critical nucleus exists, and the front speed above it is a
# property of the medium rather than of the stimulus.
function test_excitable_front_and_nucleus()
    N = 128; rng = Xoshiro(11)
    l = ExcitableWaveSheet(N, N)
    ps, st = Lux.setup(rng, l)
    θ = excitable_regime(l, ps, st).theta
    # Sub-nucleus: a single site cannot ignite the medium.
    s_dead, _ = front_speed(excitable_simulate(l, ps, st;
                            z0 = _exc_seed(N, 0, 20f0, θ), L = 60).fire)
    @test isnan(s_dead)
    # Above the nucleus: a front, at a speed insensitive to stimulus amplitude.
    speeds = Float32[]
    for amp in Float32[3, 20, 100, 500]
        s, _ = front_speed(excitable_simulate(l, ps, st;
                           z0 = _exc_seed(N, 6, amp, θ), L = 60).fire)
        @test !isnan(s)
        @test s > 0.3
        push!(speeds, s)
    end
    # Measured spread is 5.7% over this 167× range; 25% is a regression guard.
    @test (maximum(speeds) - minimum(speeds)) / mean(speeds) < 0.25
end

# Criterion 4: the pulse is annular (it recovers behind the front) rather than a
# filling disc. This is what the refractory variable buys; α alone gives 0.75+.
function test_excitable_annular_pulse()
    N = 128; rng = Xoshiro(11); cen = N ÷ 2
    l = ExcitableWaveSheet(N, N)
    ps, st = Lux.setup(rng, l)
    θ = excitable_regime(l, ps, st).theta
    F = excitable_simulate(l, ps, st; z0 = _exc_seed(N, 6, 20f0, θ), L = 60).fire
    rg = Float32[sqrt(Float32((i - cen)^2 + (j - cen)^2)) for i in 1:N, j in 1:N]
    m = @view(F[:, :, 55]) .> 0f0
    @test count(m) > 20
    R = mean(rg[m])
    inner = rg .< 0.5f0 * R
    @test count(m .& inner) / count(inner) < 0.25    # measured ≈0.0
end

function test_excitable_shapes_and_batch()
    N = 32; rng = Xoshiro(7)
    l = ExcitableWaveSheet(N, N)
    ps, st = Lux.setup(rng, l)
    θ = excitable_regime(l, ps, st).theta
    tr = excitable_simulate(l, ps, st; z0 = _exc_seed(N, 3, 20f0, θ), L = 12)
    @test size(tr.z) == (N, N, 12) && size(tr.fire) == (N, N, 12) && size(tr.u) == (N, N, 12)
    z3 = repeat(_exc_seed(N, 3, 20f0, θ), 1, 1, 2)
    tb = excitable_simulate(l, ps, st; z0 = z3, L = 12)
    @test size(tb.z) == (N, N, 12, 2)
    @test tb.z[:, :, :, 1] ≈ tb.z[:, :, :, 2]        # independent, identical sheets
    @test all(0 .<= tr.u .<= 1 + 1e-5)               # refractory variable stays bounded
    tn = excitable_simulate(l, ps, st; z0 = _exc_seed(N, 3, 20f0, θ), L = 8, keep_fields = false)
    @test tn.z === nothing && length(tn.rate) == 8
    @test_throws DimensionMismatch excitable_simulate(l, ps, st; z0 = zeros(ComplexF32, N + 1, N), L = 3)
    @test_throws ArgumentError ExcitableWaveSheet(N, N; rho_u = 1.0)
    @test_throws ArgumentError ExcitableWaveSheet(N, N; alpha = -1)
end

function test_excitable_drive()
    N = 48; rng = Xoshiro(9)
    l = ExcitableWaveSheet(N, N)
    ps, st = Lux.setup(rng, l)
    θ = excitable_regime(l, ps, st).theta
    z0 = zeros(ComplexF32, N, N)
    D = zeros(ComplexF32, N, N, 20)
    D[N÷2-2:N÷2+2, N÷2-2:N÷2+2, 1:3] .= 30f0 * θ
    driven = excitable_simulate(l, ps, st; z0 = z0, L = 20, drive = D, keep_fields = false)
    quiet  = excitable_simulate(l, ps, st; z0 = z0, L = 20, keep_fields = false)
    @test maximum(quiet.rate) == 0            # no drive, no state ⇒ nothing at all
    @test maximum(driven.rate) > 0            # the drive path reaches the dynamics
end

function excitable_tests()
    @testset "ExcitableWaveSheet" begin
        @info "Running ExcitableWaveSheet tests..."
        test_excitable_reduces_to_base()
        test_excitable_emit_limits()
        test_excitable_regime_diagnostics()
        test_excitable_quiescent()
        test_excitable_front_and_nucleus()
        test_excitable_annular_pulse()
        test_excitable_shapes_and_batch()
        test_excitable_drive()
    end
end
