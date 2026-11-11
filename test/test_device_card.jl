# Device card format: loading, operating points, value lookup, validation.

const DEVICE_FIXTURES = joinpath(@__DIR__, "fixtures", "devices")

function device_card_tests()
    @testset "Device cards" begin
        device_card_load_tests()
        device_card_value_tests()
        device_card_validation_tests()
        device_card_parse_error_tests()
    end
end

function device_card_load_tests()
    @testset "load" begin
        c = load_device_card(joinpath(DEVICE_FIXTURES, "spread_R.toml"))
        @test c isa DeviceCard
        @test c.schema == DEVICE_CARD_SCHEMA
        @test c.name == "spread resonator (synthetic)"
        @test card_class(c) === :R
        @test card_class(c; op = :above_threshold) === :L
        @test sort(collect(keys(c.operating_points))) == ["above_threshold", "small_signal"]

        v = card_value(c, "spread.sigma_omega_rel")
        @test v.value == 0.02 && v.kind === :meas && v.sd == 0.005
        @test v.cond == "n = 14" && v.src == "fixture" && v.units == "1" && v.tagged

        g = card_value(c, "orbit.G_table")
        @test g.value === nothing && isfile(g.file) && g.kind === :sim
        @test isabspath(g.file) && basename(g.file) == "orbit_G.csv"

        # Name falls back to the file name; bare values read as untagged assumptions.
        ideal = load_device_card(joinpath(DEVICE_FIXTURES, "ideal_R.toml"))
        @test ideal.name == "ideal resonator (synthetic)"
        id_name = card_value(ideal, "identity.name")
        @test id_name.kind === :assumed && !id_name.tagged

        @test occursin("spread resonator", sprint(show, c))
        @test_throws ArgumentError load_device_card(joinpath(DEVICE_FIXTURES, "no_such_card.toml"))
    end
end

function device_card_value_tests()
    @testset "card_value and operating points" begin
        c = load_device_card(joinpath(DEVICE_FIXTURES, "spread_R.toml"))
        @test card_value(c, "orbit.beta").value == 0.11
        @test card_value(c, "orbit.beta"; op = :small_signal).value == 0.01
        # Operating points override fields and keep the rest of the base card.
        @test card_value(c, "spread.sigma_omega_rel"; op = :small_signal).value == 0.02
        @test card_value(c, "unit.mu_per_s"; op = "above_threshold").value == 500.0
        @test_throws ArgumentError card_value(c, "orbit.beta"; op = :nonexistent)

        # Missing fields: default or KeyError.
        @test card_value(c, "coupling.K_rel"; default = 0.0) == 0.0
        @test_throws KeyError card_value(c, "coupling.K_rel")
        @test_throws ArgumentError card_value(c, "no_dot")

        # Unknown fields follow the on_unknown policy.
        @test_throws ArgumentError card_value(c, "noise.flicker_rel")
        @test card_value(c, "noise.flicker_rel"; on_unknown = :default, default = 0.0) == 0.0
        @test card_value(c, "noise.flicker_rel"; on_unknown = :value).kind === :unknown
        @test_throws ArgumentError card_value(c, "noise.flicker_rel"; on_unknown = :default)
        @test_throws ArgumentError card_value(c, "noise.flicker_rel"; on_unknown = :bogus)
    end
end

function device_card_validation_tests()
    @testset "validation" begin
        for name in ("ideal_R.toml", "spread_R.toml", "landau_L.toml")
            issues = validate_device_card(load_device_card(joinpath(DEVICE_FIXTURES, name)))
            @test isempty(filter(i -> i.level === :error, issues))
        end
        # ideal_R: only the untagged name; landau_L: fully clean.
        ideal_issues = validate_device_card(load_device_card(joinpath(DEVICE_FIXTURES, "ideal_R.toml")))
        @test [i.field for i in ideal_issues] == ["identity.name"]
        @test isempty(validate_device_card(load_device_card(joinpath(DEVICE_FIXTURES, "landau_L.toml"))))

        bad = load_device_card(joinpath(DEVICE_FIXTURES, "invalid.toml"))
        issues = validate_device_card(bad)
        has(level, field, text) = any(i -> i.level === level && i.field == field && occursin(text, i.message), issues)
        @test has(:error, "identity.class", "must be \"R\" or \"L\"")
        @test has(:warn, "unit.f0_Hz", "bare value")
        @test has(:error, "unit.lambda_per_s", "must be negative")
        @test has(:error, "spread.sigma_omega_rel", "≥ 0")
        @test has(:error, "spread.sigma_lambda_rel", "marked unknown but carries a value")
        @test has(:error, "orbit.G_table", "table file not found")
        @test has(:warn, "mystery", "section not in the card registry")
        @test has(:warn, "mystery.thing", "TK")

        # λ/Q consistency: λ = −ω₀/(2Q) within 5 %.
        d = Dict("schema" => DEVICE_CARD_SCHEMA,
                 "identity" => Dict("name" => Dict("value" => "q", "type" => "assumed"),
                                    "class" => Dict("value" => "R", "type" => "assumed")),
                 "unit" => Dict("f0_Hz" => Dict("value" => 1.0, "type" => "assumed"),
                                "Q" => Dict("value" => 10.0, "type" => "assumed"),
                                "lambda_per_s" => Dict("value" => -1.0, "type" => "assumed")))
        q_issues = validate_device_card(parse_device_card(d))
        @test any(i -> i.field == "unit.Q" && i.level === :error, q_issues)
        d["unit"]["lambda_per_s"]["value"] = -2π / 20
        @test isempty(validate_device_card(parse_device_card(d)))

        # Class-specific fields on the wrong class, and wrong units.
        d["unit"]["mu_per_s"] = Dict("value" => 1.0, "type" => "assumed")
        d["unit"]["f0_Hz"]["units"] = "kHz"
        w = validate_device_card(parse_device_card(d))
        @test any(i -> i.field == "unit.mu_per_s" && i.level === :warn && occursin("class L", i.message), w)
        @test any(i -> i.field == "unit.f0_Hz" && i.level === :error && occursin("units", i.message), w)

        # Operating-point issues are prefixed.
        spread = validate_device_card(load_device_card(joinpath(DEVICE_FIXTURES, "spread_R.toml")))
        @test all(i -> i.level === :warn, spread)
        @test any(i -> startswith(i.field, "op:above_threshold/"), spread)   # λ given while class L
    end
end

function device_card_parse_error_tests()
    @testset "structural errors" begin
        @test_throws ArgumentError parse_device_card(Dict("schema" => "other/9"))
        base = Dict("schema" => DEVICE_CARD_SCHEMA)
        @test_throws ArgumentError parse_device_card(merge(base,
            Dict("unit" => Dict("f0_Hz" => Dict("value" => 1.0, "type" => "guess")))))
        @test_throws ArgumentError parse_device_card(merge(base,
            Dict("unit" => Dict("f0_Hz" => Dict("value" => 1.0, "typo" => "meas")))))
        @test_throws ArgumentError parse_device_card(merge(base, Dict("unit" => 3)))
        # TK tag spellings from the H2 template are accepted.
        c = parse_device_card(merge(base,
            Dict("identity" => Dict("class" => Dict("value" => "R", "type" => "[TK]")),
                 "unit" => Dict("Q" => Dict("value" => 2.0, "type" => "[derived]")))))
        @test card_value(c, "identity.class").kind === :tk
        @test card_value(c, "unit.Q").kind === :derived
    end
end
