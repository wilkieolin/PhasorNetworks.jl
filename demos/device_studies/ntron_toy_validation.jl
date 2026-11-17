# ntron_toy_validation.jl — NTronTransponder + DeviceReservoir vs the reference toy network
#
# Replays the seven networks of transponder_model's CI job `toy-tests`
# (src/tests/toy_example.py, two sensors → 6 transponders) on NTronTransponder
# cells in a DeviceReservoir and compares every input- and output-nTron click
# time with the reference's. Reference data: ntron_toy_reference.csv, written by
#
#     python3 demos/device_studies/ntron_toy_reference.py \
#         <transponder_model>/src/tests/toy_example.py demos/device_studies/ntron_toy_reference.json
#
# Mapping: the toy's line delay d puts a pulse's half-rise at click + d; ours
# is at click + t_lead + D, so D = d − t_lead. Fan-out splitting is
# `fanout = :split` (gain 1/out-degree). Sensors are forced to fire at the
# times the reference's sensors did (0.99 ns, not 1.00: its event match has a
# 10 ps window).
#
#     julia --project=scripts demos/device_studies/ntron_toy_validation.jl [dt_ps ...]

using PhasorNetworks, CSV, Printf

const REF = joinpath(@__DIR__, "ntron_toy_reference.csv")
# toy_example.py constants for the CI runs (BASE in ntron_toy_reference.py)
const I1SW, I2SW, SPLIT = 13.5, 24.0, 0.3

function load_reference(path)
    rows = CSV.File(path)
    cfg = Dict{String, Any}()
    for r in rows
        c = get!(() -> Dict(:delay => Tuple{Int,Int,Float64}[], :bias1 => Dict{Int,Float64}(),
                            :bias2 => Dict{Int,Float64}(), :input => Dict{Int,Vector{Float64}}(),
                            :output => Dict{Int,Vector{Float64}}()), cfg, String(r.config))
        k = Symbol(r.kind)
        if k === :delay
            push!(c[:delay], (r.a, r.b, r.value))
        elseif k in (:bias1, :bias2)
            c[k][r.a] = r.value
        elseif k in (:input, :output)
            push!(get!(c[k], r.a, Float64[]), r.value)
        end
    end
    return cfg
end

function replay(c; dt = 1.0, t_end_ns = 100.0)
    N = length(c[:bias1])
    cells = [NTronTransponder(Ib1 = c[:bias1][i], I1_sw = I1SW, Ib2 = c[:bias2][i], I2_sw = I2SW,
                              split = SPLIT) for i in 1:N]
    lead = cells[1].t_lead
    src = [e[1] for e in c[:delay]]; dst = [e[2] for e in c[:delay]]
    od = zeros(Int, N); foreach(i -> od[i] += 1, src)
    res = DeviceReservoir(cells, N, src, dst; D = [e[3] * 1000 - lead for e in c[:delay]],
                          G = [1 / od[i] for i in src], dt = dt)
    sensors = [i for i in 1:N if !(i in dst)]
    fc = Int[]; ft = Float64[]
    for i in sensors, t in get(c[:output], i, Float64[])
        push!(fc, i); push!(ft, t * 1000)
    end
    nsteps = round(Int, t_end_ns * 1000 / dt)
    out = reservoir_simulate(res, nsteps; forced = (fc, ft), probe = 1:N)
    # input clicks: age_in drops when the input nTron fires; the click was age_in ago
    A = out.probes.age_in
    inp = Dict(i => Float64[] for i in 1:N)
    for i in 1:N, k in 2:size(A, 2)
        A[i, k] < A[i, k-1] && push!(inp[i], (out.probe_t[k] - A[i, k]) / 1000)
    end
    outp = Dict(i => Float64.(out.spike_time[out.spike_index .== i]) ./ 1000 for i in 1:N)
    return (; inp, outp, sensors, N)
end

fmt(v) = isempty(v) ? "—" : join((@sprintf("%.3f", x) for x in v), ", ")

function compare(name, c, r)
    worst = 0.0; agree = true
    println("\n", name)
    for (kind, ref, ours) in ((:input, c[:input], r.inp), (:output, c[:output], r.outp))
        for i in 1:r.N
            i in r.sensors && continue
            a = get(ref, i, Float64[]); b = ours[i]
            same = length(a) == length(b)
            d = same && !isempty(a) ? maximum(abs.(a .- b)) : 0.0
            agree &= same; worst = max(worst, d)
            @printf("  node %d %-6s ref [%s]  ours [%s]%s\n", i, kind, fmt(a), fmt(b),
                    same ? (isempty(a) ? "" : @sprintf("  Δ≤%.0f ps", 1000d)) : "  COUNT MISMATCH")
        end
    end
    return agree, worst
end

function main(dts)
    cfg = load_reference(REF)
    names = sort(collect(keys(cfg)))
    for dt in dts
        println("\n===== dt = $dt ps =====")
        summary = []
        for name in names
            r = replay(cfg[name]; dt)
            push!(summary, (name, compare(name, cfg[name], r)...))
        end
        println("\nsummary, dt = $dt ps:")
        for (name, ok, w) in summary
            @printf("  %-28s %s  max |Δt| = %4.0f ps\n", name, ok ? "same clicks" : "DIFFERENT  ", 1000w)
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(isempty(ARGS) ? [1.0, 5.0] : parse.(Float64, ARGS))
end
