# Device cards: a machine-readable description of one hardware platform's
# phase unit (oscillator + emitter), in physical units, with every value tagged
# by where it comes from. Cards are data only; turning a card into simulation
# parameters is a separate step (see docs/devices.md).
#
# Format (TOML, schema "phasor-device-card/1"):
#
#     schema = "phasor-device-card/1"
#
#     [identity]
#     name  = "CMOS ring oscillator, 65 nm array"
#     class = "L"                                # "R" driven resonator, "L" limit cycle
#
#     [unit]
#     f0_Hz = { value = 1.0e8, type = "meas", cond = "65 nm, 1.0 V", src = "doi:..." }
#     Q     = { type = "unknown" }
#
#     [orbit]
#     G_table = { file = "ring_N3_tables.csv", type = "sim", src = "SD1 device_ring.jl" }
#
#     [operating_points.above_threshold]          # optional overrides, e.g. VO₂
#     identity = { class = "L" }
#     unit     = { f0_Hz = { value = 3.0e6, type = "meas" } }
#
# A field is either a bare value (read as `type = "assumed"`, flagged by
# validation as untagged) or an inline table with `value` (or `file` for a
# table stored as CSV next to the card), `type`, and optional `cond`, `src`,
# `sd` (one standard deviation), `units` (must match the registry), `note`.

"""
    DEVICE_CARD_SCHEMA

The schema string every device card must declare (`schema = "phasor-device-card/1"`).
"""
const DEVICE_CARD_SCHEMA = "phasor-device-card/1"

const _CARD_KINDS = (:meas, :sim, :vendor, :proj, :fit, :data, :derived, :assumed, :tk, :unknown)
const _CARD_KIND_ALIASES = Dict("[derived]" => :derived, "[tk]" => :tk, "tk" => :tk, "[TK]" => :tk,
                                "TK" => :tk)
const _CARD_VALUE_KEYS = ("value", "file", "type", "cond", "src", "sd", "units", "note")

"""
    DEVICE_CARD_FIELDS

Registry of known device-card fields: section ⇒ field ⇒
`(units, applies_to, description)`, with `applies_to` one of `:both`, `:R`,
`:L`. Sections: identity, unit, spread, noise, orbit, emission, coupling,
trim, clock, reader, cost. Fields outside the registry load normally and are
reported as warnings by [`validate_device_card`](@ref), so a new card can use
a field before the registry lists it.
"""
const DEVICE_CARD_FIELDS = Dict{String, Dict{String, Tuple{String, Symbol, String}}}(
    "identity" => Dict(
        "name"       => ("", :both, "human-readable device name"),
        "platform"   => ("", :both, "short platform key, e.g. cmos_ring"),
        "class"      => ("", :both, "\"R\" driven resonator or \"L\" limit-cycle oscillator"),
        "technology" => ("", :both, "technology node or device structure"),
        "maturity"   => ("", :both, "largest demonstrated array"),
    ),
    "unit" => Dict(
        "f0_Hz"           => ("Hz", :both, "natural frequency"),
        "tuning_range_Hz" => ("Hz", :both, "[low, high] tuning range"),
        "Q"               => ("1", :R, "quality factor; λ = −ω₀/(2Q)"),
        "lambda_per_s"    => ("1/s", :R, "decay rate λ (negative)"),
        "mu_per_s"        => ("1/s", :L, "limit-cycle growth rate μ (amplitude relaxes at 2μ)"),
        "r0"              => ("1", :L, "limit-cycle radius in simulation amplitude units"),
        "input_tau_s"     => ("s", :both, "input low-pass time constant"),
    ),
    "spread" => Dict(
        "sigma_omega_rel"     => ("1", :both, "device-to-device σ_ω/ω₀ before trim"),
        "sigma_lambda_rel"    => ("1", :R, "device-to-device σ_λ/|λ|"),
        "sigma_mu_rel"        => ("1", :L, "device-to-device σ_μ/μ"),
        "sigma_amp_rel"       => ("1", :both, "emission amplitude / gain spread σ_A/A"),
        "sigma_threshold_rel" => ("1", :both, "threshold spread"),
        "threshold_to_f_gain" => ("1", :L, "d ln f₀ / d ln V_th (relaxation oscillators)"),
        "sigma_delay_s"       => ("s", :both, "static per-unit (axon) delay spread"),
        "sigma_switch_rel"    => ("1", :L, "switching-delay spread"),
        "pelgrom_AVT_mVum"    => ("mV*um", :both, "Pelgrom mismatch coefficient"),
        "sigma_orbit_rel"     => ("1", :both, "device-to-device spread of orbit distortion"),
    ),
    "noise" => Dict(
        "jitter_rel"           => ("1", :both, "white period jitter σ_t/T per cycle"),
        "phase_noise_dBcHz"    => ("dBc/Hz", :both, "phase noise L(Δf) at offset_Hz"),
        "offset_Hz"            => ("Hz", :both, "offset frequency for phase_noise_dBcHz"),
        "linewidth_Hz"         => ("Hz", :L, "Lorentzian linewidth Δf (phase diffusion D = πΔf)"),
        "flicker_corner_Hz"    => ("Hz", :both, "1/f corner frequency"),
        "flicker_rel"          => ("1", :both, "RMS 1/f frequency wander relative to ω₀"),
        "drift_rel_per_s"      => ("1/s", :both, "linear frequency drift rate"),
        "temp_coeff_per_K"     => ("1/K", :both, "temperature coefficient of f₀"),
        "supply_pushing_per_V" => ("1/V", :both, "frequency pushing d ln f / dV"),
        "amp_noise_rel"        => ("1", :both, "cycle-to-cycle amplitude noise"),
        "clock_walk_rad"       => ("rad", :both, "common-mode rhythm walk ρ"),
    ),
    "orbit" => Dict(
        "beta"                 => ("1", :both, "non-isochrony β = d ln ω / d ln r"),
        "nu"                   => ("1", :L, "Stuart–Landau non-isochrony ν"),
        "F_shape"              => ("", :both, "loglinear | linear | quadratic | table"),
        "F_table"              => ("", :both, "CSV (x = ln(r/r_ref), F)"),
        "S_table"              => ("", :both, "CSV (theta, S) orbit speed"),
        "G_table"              => ("", :both, "CSV (theta, G_re, G_im) kick response"),
        "isf_table"            => ("", :L, "CSV (phi, Gamma) impulse sensitivity function"),
        "speed_distortion_rad" => ("rad", :both, "max orbit angle error vs uniform phase"),
        "prc_h2_rel"           => ("1", :both, "PRC 2nd harmonic relative to fundamental"),
        "prc_h3_rel"           => ("1", :both, "PRC 3rd harmonic relative to fundamental"),
        "prc_type"             => ("", :both, "PRC type, e.g. I, II, 0"),
        "amp_clamp"            => ("", :both, "amplitude limit: none | rails | threshold"),
        "r_ref"                => ("1", :both, "reference amplitude for β / F"),
    ),
    "emission" => Dict(
        "rule"            => ("", :both, "sampled | event | hill | phase_gated"),
        "event_phase_rad" => ("rad", :L, "orbit phase θ_e at which a class-L unit emits"),
        "threshold"       => ("1", :both, "emission threshold θ"),
        "threshold_units" => ("", :both, "raw | dirac (docs/devices.md C3)"),
        "hill_n"          => ("1", :both, "Hill sharpness n"),
        "pulse_width_rel" => ("1", :both, "pulse width / T"),
        "duty"            => ("1", :both, "duty cycle"),
        "burst"           => ("1", :both, "spikes per burst"),
        "kernel"          => ("", :both, "pulse kernel name or CSV"),
        "native_options"  => ("", :both, "native emission options (review §6.1 A–H)"),
    ),
    "coupling" => Dict(
        "K_Hz"              => ("Hz", :L, "coupling strength; Adler locking range 2K"),
        "K_rel"             => ("1", :L, "K/ω₀"),
        "coupling_phase_rad"=> ("rad", :both, "phase of the coupling"),
        "mechanism"         => ("", :both, "resistive | capacitive | ..."),
        "weight_bits"       => ("1", :both, "weight precision in bits"),
        "weight_per"        => ("", :both, "quantiser scale: tensor | row"),
        "sigma_weight_rel"  => ("1", :both, "multiplicative weight mismatch"),
        "dead_frac"         => ("1", :both, "fraction of dead units"),
        "weight_drift_nu"   => ("1", :both, "conductance drift exponent (G ∝ t^−ν)"),
        "gain_drift_per_s"  => ("1/s", :both, "emission-gain drift rate"),
        "link_delay_s"      => ("s", :both, "transmission delay per link"),
        "sigma_link_delay_s"=> ("s", :both, "per-link delay jitter"),
    ),
    "trim" => Dict(
        "freq_bits"        => ("1", :both, "frequency-trim bits"),
        "freq_range_sigma" => ("1", :both, "trim range in units of σ_ω"),
        "delay_bits"       => ("1", :both, "delay/phase-trim bits"),
        "phase_trim"       => ("", :both, "none | phase | full"),
        "actuator"         => ("", :both, "what is tuned: bias, capacitor, supply, register"),
        "fll"              => ("", :both, "none | push | nat"),
        "fll_gain_rel"     => ("1", :both, "FLL gain × T"),
        "homeostat_eta"    => ("1", :both, "threshold homeostat rate"),
        "lock_m"           => ("1", :both, "m-fold locking alphabet size"),
        "lock_K_rel"       => ("1", :both, "lock strength / ω₀"),
    ),
    "clock" => Dict(
        "global_clock"          => ("", :both, "how a shared reference is distributed"),
        "zm_m"                  => ("1", :both, "Z_m reference alphabet"),
        "shil_K_rel"            => ("1", :L, "sub-harmonic injection strength / ω₀"),
        "reader_clock_error_rel"=> ("1", :both, "reader clock frequency error δ"),
        "reader_jitter_rad"     => ("rad", :both, "reader sampling jitter σ_φ"),
    ),
    "reader" => Dict(
        "hebbian_sigma"       => ("1", :both, "Hebbian reader white noise"),
        "offset_in"           => ("1", :both, "reader input offset"),
        "offset_prod"         => ("1", :both, "multiplier output offset"),
        "adc_bits"            => ("1", :both, "reader ADC bits"),
        "ac_tau_s"            => ("s", :both, "AC-coupling time constant"),
        "readout_quant_turn"  => ("1", :both, "phase readout quantiser step (turns)"),
    ),
    "cost" => Dict(
        "energy_per_event_J" => ("J", :both, "energy to emit one spike to one receiver"),
        "energy_per_cycle_J" => ("J", :both, "energy per oscillation cycle"),
        "area_um2"           => ("um^2", :both, "area per site"),
        "eta_dB"             => ("dB", :both, "distance above the thermal floor"),
        "trim_area_um2"      => ("um^2", :both, "area of the trim circuit"),
    ),
)

"""
    CardValue

One field of a [`DeviceCard`](@ref): its `value` (a number, string, bool or
vector; `nothing` when `kind == :unknown` or the field is a table), its
provenance `kind` (`:meas`, `:sim`, `:vendor`, `:proj`, `:fit`, `:data`,
`:derived`, `:assumed`, `:tk` or `:unknown`, the tags of the H2 card
template), and the conditions `cond`, source `src`, one standard deviation
`sd`, `units`, `note`, and for table fields the absolute path `file` of the
CSV. `tagged` is `false` when the card gave a bare value without a `type`.
"""
struct CardValue
    value::Any
    kind::Symbol
    units::String
    cond::String
    src::String
    sd::Union{Nothing, Float64}
    file::Union{Nothing, String}
    note::String
    tagged::Bool
end

"""
    DeviceCard

A parsed device card: `schema`, `name`, `path` (source file, or `""`),
`sections` (section name ⇒ field name ⇒ [`CardValue`](@ref)) and
`operating_points` (name ⇒ section overrides, e.g. VO₂ below and above
threshold). Load with [`load_device_card`](@ref), check with
[`validate_device_card`](@ref), read fields with [`card_value`](@ref).
"""
struct DeviceCard
    schema::String
    name::String
    path::String
    sections::Dict{String, Dict{String, CardValue}}
    operating_points::Dict{String, Dict{String, Dict{String, CardValue}}}
end

"""
    CardIssue(level, field, message)

One finding of [`validate_device_card`](@ref). `level` is `:error` (the card
cannot be used as written) or `:warn` (usable, but check it); `field` is
`"section.name"`, prefixed by `"op:<name>/"` inside an operating point.
"""
struct CardIssue
    level::Symbol
    field::String
    message::String
end

Base.show(io::IO, v::CardValue) =
    print(io, "CardValue(", v.file === nothing ? repr(v.value) : "file=" * repr(basename(v.file)),
          ", :", v.kind, isempty(v.units) ? "" : ", " * v.units, ")")
Base.show(io::IO, c::DeviceCard) =
    print(io, "DeviceCard(\"", c.name, "\", class ", something(card_class(c), "?"), ", ",
          sum(length, values(c.sections); init = 0), " fields",
          isempty(c.operating_points) ? "" : ", operating points " * join(sort(collect(keys(c.operating_points))), "/"),
          ")")
Base.show(io::IO, i::CardIssue) = print(io, i.level, " ", i.field, ": ", i.message)

function _card_kind(t)
    t isa AbstractString || throw(ArgumentError("device card: `type` must be a string, got $(repr(t))"))
    haskey(_CARD_KIND_ALIASES, t) && return _CARD_KIND_ALIASES[t]
    k = Symbol(lowercase(strip(t, ['[', ']'])))
    k in _CARD_KINDS || throw(ArgumentError("device card: unknown value type \"$t\"; " *
                                            "expected one of $(join(_CARD_KINDS, ", "))"))
    return k
end

_registry_units(section, field) =
    haskey(DEVICE_CARD_FIELDS, section) && haskey(DEVICE_CARD_FIELDS[section], field) ?
        DEVICE_CARD_FIELDS[section][field][1] : ""

function _parse_card_value(section::String, field::String, raw, base_dir::String)
    units = _registry_units(section, field)
    if raw isa AbstractDict
        extra = setdiff(keys(raw), _CARD_VALUE_KEYS)
        isempty(extra) || throw(ArgumentError("device card: $section.$field has unknown keys " *
                                              join(sort(collect(extra)), ", ")))
        kind = haskey(raw, "type") ? _card_kind(raw["type"]) : :assumed
        file = haskey(raw, "file") ? normpath(joinpath(base_dir, raw["file"])) : nothing
        sd = haskey(raw, "sd") ? Float64(raw["sd"]) : nothing
        return CardValue(get(raw, "value", nothing), kind, string(get(raw, "units", units)),
                         string(get(raw, "cond", "")), string(get(raw, "src", "")), sd, file,
                         string(get(raw, "note", "")), haskey(raw, "type"))
    end
    return CardValue(raw, :assumed, units, "", "", nothing, nothing, "", false)
end

function _parse_card_sections(d::AbstractDict, base_dir::String; where::String)
    sections = Dict{String, Dict{String, CardValue}}()
    for (sec, body) in d
        sec in ("schema", "operating_points") && continue
        body isa AbstractDict || throw(ArgumentError("device card: $(where)[$sec] must be a table"))
        sections[sec] = Dict{String, CardValue}(f => _parse_card_value(sec, f, v, base_dir) for (f, v) in body)
    end
    return sections
end

"""
    parse_device_card(d::AbstractDict; base_dir = pwd(), path = "") -> DeviceCard

Build a [`DeviceCard`](@ref) from a parsed TOML dictionary. `base_dir`
resolves relative `file` entries. Throws `ArgumentError` on a structural
problem (wrong schema, unknown value type, unknown keys in a value table);
everything else is reported by [`validate_device_card`](@ref).
"""
function parse_device_card(d::AbstractDict; base_dir::AbstractString = pwd(), path::AbstractString = "")
    schema = get(d, "schema", "")
    schema == DEVICE_CARD_SCHEMA ||
        throw(ArgumentError("device card: schema must be \"$DEVICE_CARD_SCHEMA\", got $(repr(schema))"))
    base = String(base_dir)
    sections = _parse_card_sections(d, base; where = "")
    ops = Dict{String, Dict{String, Dict{String, CardValue}}}()
    for (name, body) in get(d, "operating_points", Dict())
        body isa AbstractDict || throw(ArgumentError("device card: operating_points.$name must be a table"))
        ops[name] = _parse_card_sections(body, base; where = "operating_points.$name.")
    end
    name_v = get(get(sections, "identity", Dict{String, CardValue}()), "name", nothing)
    name = name_v === nothing ? (isempty(path) ? "unnamed" : splitext(basename(path))[1]) : string(name_v.value)
    return DeviceCard(schema, name, String(path), sections, ops)
end

"""
    load_device_card(path) -> DeviceCard

Read a device card from a TOML file. Relative `file` entries (CSV tables)
resolve against the card's directory. See [`parse_device_card`](@ref) for
what is checked while reading and [`validate_device_card`](@ref) for the
rest.
"""
function load_device_card(path::AbstractString)
    isfile(path) || throw(ArgumentError("device card not found: $path"))
    return parse_device_card(TOML.parsefile(path); base_dir = dirname(abspath(path)), path = abspath(path))
end

function _merged_sections(card::DeviceCard, op)
    op === nothing && return card.sections
    haskey(card.operating_points, string(op)) ||
        throw(ArgumentError("device card \"$(card.name)\" has no operating point \"$op\"; " *
                            "available: $(join(sort(collect(keys(card.operating_points))), ", "))"))
    merged = Dict{String, Dict{String, CardValue}}(s => copy(f) for (s, f) in card.sections)
    for (s, fields) in card.operating_points[string(op)]
        merge!(get!(merged, s, Dict{String, CardValue}()), fields)
    end
    return merged
end

"""
    card_class(card; op = nothing) -> Union{Symbol, Nothing}

The device class, `:R` (driven resonator) or `:L` (limit-cycle oscillator),
at operating point `op` (the base card when `nothing`). Returns `nothing`
when the card does not state it.
"""
function card_class(card::DeviceCard; op = nothing)
    v = get(get(_merged_sections(card, op), "identity", Dict{String, CardValue}()), "class", nothing)
    (v === nothing || v.value === nothing) && return nothing
    return Symbol(uppercase(string(v.value)))
end

"""
    card_value(card, field; op = nothing, default = nothing, on_unknown = :error)

Look up `field` (`"section.name"`) at operating point `op`. Returns the
[`CardValue`](@ref). A missing field returns `default` when one is given and
throws otherwise. A field marked `unknown` follows `on_unknown`: `:error`
throws, `:default` returns `default` (which must be given), `:value` returns
the `CardValue` as is.
"""
function card_value(card::DeviceCard, field::AbstractString; op = nothing, default = nothing,
                    on_unknown::Symbol = :error)
    on_unknown in (:error, :default, :value) ||
        throw(ArgumentError("on_unknown must be :error, :default or :value"))
    parts = split(field, '.'; limit = 2)
    length(parts) == 2 || throw(ArgumentError("card field must be \"section.name\", got \"$field\""))
    sec, name = String(parts[1]), String(parts[2])
    v = get(get(_merged_sections(card, op), sec, Dict{String, CardValue}()), name, nothing)
    if v === nothing
        default === nothing && throw(KeyError("device card \"$(card.name)\" has no field $field"))
        return default
    end
    if v.kind === :unknown
        on_unknown === :value && return v
        if on_unknown === :default
            default === nothing && throw(ArgumentError("on_unknown = :default needs a `default`"))
            return default
        end
        throw(ArgumentError("device card \"$(card.name)\": $field is marked unknown " *
                            "(pass on_unknown = :default with a default to substitute a value)"))
    end
    return v
end

_isnum(x) = x isa Real && !(x isa Bool)

function _validate_class_fields!(issues, sections, prefix, class)
    class === nothing && return
    for (sec, fields) in sections, (f, v) in fields
        haskey(DEVICE_CARD_FIELDS, sec) && haskey(DEVICE_CARD_FIELDS[sec], f) || continue
        app = DEVICE_CARD_FIELDS[sec][f][2]
        app !== :both && app !== class && v.kind !== :unknown &&
            push!(issues, CardIssue(:warn, prefix * sec * "." * f, "applies to class $app, card is class $class"))
    end
end

function _validate_sections!(issues, sections, prefix)
    for (sec, fields) in sections
        known_sec = haskey(DEVICE_CARD_FIELDS, sec)
        known_sec || push!(issues, CardIssue(:warn, prefix * sec, "section not in the card registry"))
        for (f, v) in fields
            key = prefix * sec * "." * f
            if known_sec && !haskey(DEVICE_CARD_FIELDS[sec], f)
                push!(issues, CardIssue(:warn, key, "field not in the card registry"))
            end
            v.tagged || push!(issues, CardIssue(:warn, key, "bare value without a `type`; read as assumed"))
            if v.kind === :unknown
                (v.value === nothing && v.file === nothing) ||
                    push!(issues, CardIssue(:error, key, "marked unknown but carries a value"))
            elseif v.value === nothing && v.file === nothing
                push!(issues, CardIssue(:error, key, "no `value` or `file` (mark it type = \"unknown\")"))
            end
            v.kind === :tk && push!(issues, CardIssue(:warn, key, "value not yet checked against a source (TK)"))
            reg = _registry_units(sec, f)
            if !isempty(reg) && !isempty(v.units) && v.units != reg
                push!(issues, CardIssue(:error, key, "units \"$(v.units)\" differ from the registry's \"$reg\""))
            end
            if v.file !== nothing && !isfile(v.file)
                push!(issues, CardIssue(:error, key, "table file not found: $(v.file)"))
            end
            if (startswith(f, "sigma_") || endswith(f, "_bits") || f in ("jitter_rel", "linewidth_Hz", "f0_Hz",
                                                                       "Q", "dead_frac")) &&
               _isnum(v.value) && v.value < 0
                push!(issues, CardIssue(:error, key, "must be ≥ 0, got $(v.value)"))
            end
            if f == "dead_frac" && _isnum(v.value) && v.value > 1
                push!(issues, CardIssue(:error, key, "must be ≤ 1, got $(v.value)"))
            end
        end
    end
end

function _validate_physics!(issues, sections, prefix, class)
    get_num(sec, f) = (v = get(get(sections, sec, Dict{String, CardValue}()), f, nothing);
                       v !== nothing && _isnum(v.value) ? Float64(v.value) : nothing)
    f0 = get_num("unit", "f0_Hz"); Q = get_num("unit", "Q"); λ = get_num("unit", "lambda_per_s")
    if λ !== nothing && λ >= 0
        push!(issues, CardIssue(:error, prefix * "unit.lambda_per_s", "decay rate must be negative, got $λ"))
    end
    if f0 !== nothing && Q !== nothing && λ !== nothing && Q > 0
        λ_from_Q = -2π * f0 / (2Q)
        rel = abs(λ - λ_from_Q) / abs(λ_from_Q)
        rel > 0.05 && push!(issues, CardIssue(:error, prefix * "unit.Q",
                                              "λ = $λ /s disagrees with −ω₀/(2Q) = $(round(λ_from_Q, sigdigits = 4)) /s " *
                                              "($(round(100rel, digits = 1)) %)"))
    end
    if class === :R && Q === nothing && λ === nothing
        push!(issues, CardIssue(:warn, prefix * "unit", "class R card gives neither Q nor lambda_per_s"))
    elseif class === :L && get_num("unit", "mu_per_s") === nothing &&
           !haskey(get(sections, "unit", Dict{String, CardValue}()), "mu_per_s")
        push!(issues, CardIssue(:warn, prefix * "unit", "class L card gives no mu_per_s"))
    end
    thr_units = get(get(sections, "emission", Dict{String, CardValue}()), "threshold_units", nothing)
    if thr_units !== nothing && !(string(thr_units.value) in ("raw", "dirac"))
        push!(issues, CardIssue(:error, prefix * "emission.threshold_units", "must be \"raw\" or \"dirac\""))
    end
end

"""
    validate_device_card(card) -> Vector{CardIssue}

Check a [`DeviceCard`](@ref) and return every [`CardIssue`](@ref) found,
errors and warnings, for the base card and for each operating point merged
over it. An empty vector means the card is clean. Checks: the class is
`"R"` or `"L"`; unknown values carry no number and other values carry one;
units match the registry ([`DEVICE_CARD_FIELDS`](@ref)); table files exist;
spreads, bits and rates are non-negative; λ is negative and agrees with
`−ω₀/(2Q)` within 5 % when both are given; class-specific fields match the
class. Fields outside the registry, untagged values and `TK` values are
warnings.
"""
function validate_device_card(card::DeviceCard)
    issues = CardIssue[]
    for (prefix, op) in Iterators.flatten(((("", nothing),), (("op:$k/", k) for k in sort(collect(keys(card.operating_points))))))
        sections = _merged_sections(card, op)
        cls_v = get(get(sections, "identity", Dict{String, CardValue}()), "class", nothing)
        class = nothing
        if cls_v === nothing
            push!(issues, CardIssue(:error, prefix * "identity.class", "missing; must be \"R\" or \"L\""))
        else
            c = Symbol(uppercase(string(cls_v.value)))
            c in (:R, :L) ? (class = c) :
                push!(issues, CardIssue(:error, prefix * "identity.class", "must be \"R\" or \"L\", got $(repr(cls_v.value))"))
        end
        # Per-field checks run once, on the fields each level defines (the base card,
        # then each operating point's overrides); class and physics checks run on
        # the merged card, so an operating point that switches class is checked
        # against every field it inherits.
        _validate_sections!(issues, op === nothing ? sections : card.operating_points[op], prefix)
        _validate_class_fields!(issues, sections, prefix, class)
        _validate_physics!(issues, sections, prefix, class)
    end
    return issues
end
