# Device specs: sub-spec defaults and validation, orbit functions, converters,
# and DeviceCard → DeviceSpec conversion.

function device_spec_tests()
    @testset "Device specs" begin
        device_spec_default_tests()
        device_spec_orbit_fn_tests()
        device_spec_amp_fn_tests()
        device_spec_converter_tests()
        device_spec_card_tests()
    end
end

function device_spec_default_tests()
    @testset "defaults are the ideal" begin
        d = IdealDevice()
        @test d isa DeviceSpec && is_ideal(d) && is_ideal(DeviceSpec())
        for T in (UnitSpec, SpreadSpec, OrbitSpec, NoiseSpec, EmitSpec, SynapseSpec, WireSpec, TrimSpec,
                  ClockSpec, ReaderSpec, CostSpec)
            @test is_ideal(T())
        end
        @test occursin("ideal", sprint(show, d))

        s = DeviceSpec(name = "spread", spread = SpreadSpec(sigma_omega = 0.02),
                       trim = TrimSpec(freq = :quantised, freq_bits = 4))
        @test !is_ideal(s) && s.spread.sigma_omega === 0.02f0
        @test occursin("spread, trim", sprint(show, s))
        @test !is_ideal(DeviceSpec(unit = UnitSpec(lambdaT = -0.2)))
        @test !is_ideal(DeviceSpec(orbit = OrbitSpec(S = speed_distortion(0.2))))
        @test !is_ideal(DeviceSpec(emit = EmitSpec(threshold = 0.05)))
        @test is_ideal(DeviceSpec(cost = CostSpec(area_um2 = 10.0)))      # bookkeeping only

        # Constructor checks.
        @test_throws ArgumentError UnitSpec(class = :X)
        @test_throws ArgumentError UnitSpec(lambdaT = 0.1)
        @test_throws ArgumentError UnitSpec(class = :L)                    # needs muT
        @test UnitSpec(class = :L, muT = 0.5).muT === 0.5f0
        @test_throws ArgumentError SpreadSpec(sigma_omega = -0.1)
        @test_throws ArgumentError EmitSpec(rule = :hill)
        @test_throws ArgumentError EmitSpec(burst = 0)
        @test_throws ArgumentError SynapseSpec(weight_per = :column)
        @test_throws ArgumentError SynapseSpec(dead_frac = 0.7, stuck_frac = 0.5)
        @test_throws ArgumentError TrimSpec(freq = :quantised)             # needs bits
        @test_throws ArgumentError TrimSpec(fll = :pll)
        @test_throws ArgumentError ClockSpec(decode = :local)              # needs ref_units
        @test_throws ArgumentError NoiseSpec(flicker_n = 0)
        @test_throws ArgumentError WireSpec(jitter = -1)
        @test_throws ArgumentError ReaderSpec(adc_bits = -2)
    end
end

function device_spec_orbit_fn_tests()
    @testset "orbit functions" begin
        one_r = FourierFn(1f0); one_c = FourierFn(1f0 + 0f0im)
        @test one_r(0.3f0) === 1f0 && one_c(0.3f0) === 1f0 + 0f0im
        @test_throws ArgumentError FourierFn{Float32, 2}((1f0 + 0im, 0f0 + 0im))

        # SD1's phenomenological axes, exactly.
        a, k, c = 0.3, 2, 0.4
        S = speed_distortion(a; k, phase = c)
        G = kick_distortion(0.2; k = 3, phase = -0.5)
        θs = range(-π, π; length = 41)
        @test all(t -> isapprox(S(t), (1 + a * cos(k * t + c)) / sqrt(1 - a^2); atol = 1e-6), θs)
        @test all(t -> isapprox(G(t), 1 + 0.2 * cos(3t - 0.5); atol = 1e-6), θs)
        @test S(0.1f0) isa Float32 && G(0.1f0) isa ComplexF32
        @test isbitstype(typeof(S)) && isbitstype(typeof(G))
        @test nharmonics(S) == 2 && nharmonics(G) == 3
        @test isapprox(mean_inverse_speed(S), 1; atol = 1e-6)
        @test speed_distortion(0) === FourierFn(1f0) && kick_distortion(0) === FourierFn(1f0 + 0f0im)
        @test_throws ArgumentError speed_distortion(1.0)

        # Differentiable in θ (Zygote vs central differences).
        θ = Float32[0.1, -1.2, 2.5]
        gz = Zygote.gradient(t -> sum(S.(t)) + sum(abs2.(G.(t))), θ)[1]
        h = 1f-3
        fd = [(S(t + h) + abs2(G(t + h)) - S(t - h) - abs2(G(t - h))) / (2h) for t in θ]
        @test isapprox(gz, fd; rtol = 2e-3)

        # TableFn is SD1's PTab rule.
        ptab(v, θ) = (n = length(v); x = (mod(θ + π, 2π)) / (2π) * n; i = floor(Int, x); f = x - i;
                      i = mod(i, n); (1 - f) * v[i + 1] + f * v[mod(i + 1, n) + 1])
        v = [1.0, 1.3, 0.8, 0.9, 1.1]
        tf = TableFn(v)
        @test all(t -> tf(t) == Float32(ptab(v, t)), Float32.(θs))
        tc = TableFn(ComplexF64.(v, 0.1 .* v))
        @test tc(0.2f0) isa ComplexF32

        # fourier_fit: exact for a trigonometric polynomial, at and between grid points.
        n = 32; grid = [-π + 2π * (j - 1) / n for j in 1:n]
        p(t) = 1 + cos(t) + 0.2 * sin(3t)
        f, res = fourier_fit(p.(grid); nharm = 4)
        @test f isa FourierFn{Float32} && nharmonics(f) == 4
        @test all(t -> isapprox(f(t), p(t); atol = 1e-5), θs)
        @test 0 < res < 0.05                                # vs linear interpolation of the samples
        fc, _ = fourier_fit(cis.(grid) .+ 0.5; nharm = 2)
        @test fc isa FourierFn{ComplexF32} && isapprox(fc(0.7), cis(0.7) + 0.5; atol = 1e-5)
        @test nharmonics(first(fourier_fit(p.(grid[1:8]); nharm = 10))) == 3    # capped at (n − 1) ÷ 2
        @test_throws ArgumentError fourier_fit([1.0, 2.0])
    end
end

function device_spec_amp_fn_tests()
    @testset "non-isochrony F (C2)" begin
        β = 0.11
        dlog(F) = (F(1e-5) - F(-1e-5)) / 2e-5
        for shape in (:loglinear, :linear, :quadratic)
            F = AmpFn(shape; beta = β)
            @test isapprox(F(0.0), 1; atol = 1e-7)
            @test isapprox(dlog(F), β; rtol = 1e-3)
            @test amp_beta(F) === Float32(β)
        end
        @test AmpFn(:none) === IsoAmp() && IsoAmp()(0.4f0) === 1f0 && amp_beta(IsoAmp()) == 0
        L = AmpFn(:loglinear; beta = β)
        @test L(5.0) ≈ 1 + β * log(3)                                        # clamped at ±ln 3
        @test AmpFn(:linear; beta = β)(log(2.0)) ≈ 1 + β
        @test AmpFn(:quadratic; beta = β)(log(2.0)) ≈ 1 + β / 2 * 3
        T = AmpFn(:table; table = ([-1.0, 0.0, 1.0], [0.9, 1.0, 1.2]))
        @test T isa TableAmp && T(-5.0) == 0.9f0 && T(5.0) == 1.2f0 && T(0.5) ≈ 1.1
        @test isapprox(amp_beta(T), 0.15; atol = 1e-6)
        @test_throws ArgumentError AmpFn(:table)
        @test_throws ArgumentError AmpFn(:cubic; beta = β)
        @test_throws ArgumentError TableAmp([1.0, 0.0], [1.0, 1.0])
    end
end

function device_spec_converter_tests()
    @testset "converters" begin
        # Stuart–Landau: ω(ρ) = ω₀ − νμ(ρ² − 1) ⇒ β = −2νμ/ω₀.
        ν, μT = 3.0, 0.6
        @test beta_from_nu(ν, μT) ≈ -2ν * μT / (2π)
        # E4 at its own κ = 10π, λT = −0.2 is the C2 β; other κ scale linearly.
        @test beta_from_e4(0.07) ≈ 0.07
        @test beta_from_e4(0.07; kappa = 5π) ≈ 0.035
        @test beta_from_iso(0.4, 2π; r_ref = 0.5) ≈ 0.2 / 2π

        # Phase noise: one D from each description, consistent with each other.
        f0, Δf = 1.0e8, 3.0e5
        DT = diffusion_from_linewidth(Δf, f0)
        @test DT ≈ π * Δf / f0
        off = 1.0e6
        Lc = 10 * log10(DT * f0 / (2π^2 * off^2))            # L(Δf) of white FM with this D
        @test diffusion_from_phase_noise(Lc, off, f0) ≈ DT
        σ = sqrt(2DT) / 2π                                    # period jitter σ_t/T for this D
        @test diffusion_from_jitter(σ) ≈ DT
    end
end

function device_spec_card_tests()
    spk = SpikingArgs()
    fx(name) = joinpath(DEVICE_FIXTURES, name)
    @testset "card → spec" begin
        ideal = device_spec(fx("ideal_R.toml"), spk)
        @test ideal.unit.lambdaT === -0.2f0 && ideal.unit.class === :R
        @test is_ideal(ideal.spread) && is_ideal(ideal.orbit) && is_ideal(ideal.noise)
        @test !is_ideal(ideal)                                  # the card imposes its λ
        @test is_ideal(device_spec(fx("ideal_R.toml"), spk; lambda = :layer))
        # Q alone gives λT = −π/Q.
        qd = Dict("schema" => DEVICE_CARD_SCHEMA,
                  "identity" => Dict("class" => Dict("value" => "R", "type" => "assumed")),
                  "unit" => Dict("f0_Hz" => Dict("value" => 2.0, "type" => "assumed"),
                                 "Q" => Dict("value" => 5.0, "type" => "assumed")))
        @test device_spec(parse_device_card(qd), spk).unit.lambdaT ≈ -π / 5

        s = device_spec(fx("spread_R.toml"), spk)
        @test s.name == "spread resonator (synthetic)" && endswith(s.source, "spread_R.toml")
        @test s.unit.lambdaT ≈ -1.0                              # −1000 /s at f₀ = 1 kHz
        @test s.spread.sigma_omega === 0.02f0 && s.spread.sigma_lambda === 0.1f0
        @test s.noise.diffusionT ≈ diffusion_from_jitter(0.01)
        @test s.unknown == ["noise.flicker_rel"] && s.noise.flicker_rel == 0
        @test ("spread.sigma_omega_rel" => :meas) in s.provenance
        @test ("orbit.G_table" => :sim) in s.provenance
        @test s.orbit.F isa LogLinearAmp && amp_beta(s.orbit.F) === 0.11f0
        @test s.orbit.G isa FourierFn{ComplexF32} && nharmonics(s.orbit.G) == 3
        @test any(n -> occursin("G_table: Fourier fit", n), s.notes)
        @test s.emit.threshold === 0.05f0 && s.emit.hill_n === 4f0 && s.emit.rule === :sampled
        st = device_spec(fx("spread_R.toml"), spk; tables = :table)
        @test st.orbit.G isa TableFn{ComplexF32} && st.orbit.G(Float32(-π / 2)) ≈ 0.9

        # Operating points.
        @test amp_beta(device_spec(fx("spread_R.toml"), spk; op = :small_signal).orbit.F) === 0.01f0
        hi = device_spec(fx("spread_R.toml"), spk; op = :above_threshold)
        @test hi.unit.class === :L && hi.unit.muT === 0.5f0 && hi.unit.lambdaT === nothing
        @test hi.emit.rule === :event && hi.spread.sigma_lambda == 0
        @test occursin("@ above_threshold", hi.name)

        # Unknowns, overrides.
        @test_throws ArgumentError device_spec(fx("spread_R.toml"), spk; on_unknown = :error)
        o = device_spec(fx("spread_R.toml"), spk; set = Dict("noise.flicker_rel" => 0.001,
                                                              "spread.sigma_omega_rel" => 0.05))
        @test o.noise.flicker_rel === 0.001f0 && o.spread.sigma_omega === 0.05f0 && isempty(o.unknown)
        @test ("spread.sigma_omega_rel" => :set) in o.provenance
        @test_throws ArgumentError device_spec(fx("spread_R.toml"), spk; set = Dict("unit.mu_per_s" => 1.0))

        # Class L.
        l = device_spec(fx("landau_L.toml"), spk)
        @test l.unit.class === :L && l.unit.muT ≈ 0.6 && l.unit.r0 == 1
        @test l.orbit.F isa QuadraticAmp && amp_beta(l.orbit.F) ≈ Float32(beta_from_nu(3.0, 0.6))
        @test l.noise.diffusionT ≈ diffusion_from_linewidth(3.0e5, 1.0e8)
        @test l.synapse.K_rel === 0.02f0 && l.emit.rule === :event
        @test_throws ArgumentError device_spec(fx("invalid.toml"), spk)        # class X

        # Raw Hill threshold → Dirac units (C3), class L without μ, phase noise without offset.
        base(sections) = parse_device_card(merge(Dict{String, Any}("schema" => DEVICE_CARD_SCHEMA), sections))
        cls(c) = Dict("class" => Dict("value" => c, "type" => "assumed"))
        raw = base(Dict("identity" => cls("R"),
                        "emission" => Dict("threshold" => Dict("value" => 0.004, "type" => "assumed"),
                                           "threshold_units" => Dict("value" => "raw", "type" => "assumed"))))
        @test device_spec(raw, spk).emit.threshold ≈ 0.004 / (2 * spk.t_window * spk.spk_scale)
        @test_throws ArgumentError device_spec(base(Dict("identity" => cls("L"))), spk)
        @test_throws ArgumentError device_spec(base(Dict("identity" => cls("R"),
            "noise" => Dict("linewidth_Hz" => Dict("value" => 1.0, "type" => "meas")))), spk)   # needs f0

        # Speed table rescaled to ⟨1/S⟩ = 1 (C4).
        mktempdir() do dir
            n = 16; grid = [-π + 2π * (j - 1) / n for j in 1:n]
            open(joinpath(dir, "S.csv"), "w") do io
                println(io, "theta,S")
                for t in grid; println(io, t, ",", 2 * (1 + 0.2cos(t))); end
            end
            card = parse_device_card(merge(Dict{String, Any}("schema" => DEVICE_CARD_SCHEMA),
                                           Dict("identity" => cls("R"),
                                                "orbit" => Dict("S_table" => Dict("file" => "S.csv", "type" => "sim"))));
                                     base_dir = dir)
            sp = device_spec(card, spk)
            @test isapprox(mean_inverse_speed(sp.orbit.S), 1; atol = 1e-5)
            @test any(n -> occursin("rescaled", n), sp.notes)
            open(joinpath(dir, "bad.csv"), "w") do io
                println(io, "theta,S"); println(io, "0.0,1.0"); println(io, "1.0,1.0"); println(io, "2.0,1.0")
            end
            bad = parse_device_card(Dict("schema" => DEVICE_CARD_SCHEMA, "identity" => cls("R"),
                                         "orbit" => Dict("S_table" => Dict("file" => "bad.csv", "type" => "sim")));
                                    base_dir = dir)
            @test_throws ArgumentError device_spec(bad, spk)
        end
    end
end
