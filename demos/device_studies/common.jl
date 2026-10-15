# common.jl — shared helpers for the device-sheet studies (E1–E9).
#
# Every study writes a tab-separated table and figures into
# demos/wave_out/device_studies/ so the report can be built from them. Run the
# studies from the repository root with
#
#     julia -t 16 --project=scripts demos/device_studies/eN_*.jl
#
# Geometry and units follow demos/device_velocity_readout.jl: sites at a
# 129 µm pitch (10 ps of NbTiN line), time in ps, speeds in sites/ps.

ENV["GKSwstype"] = "100"
using PhasorNetworks, Printf, Statistics, Random
include(joinpath(@__DIR__, "..", "..", "scripts", "gpu_backend.jl"))

const STUDY_OUT = joinpath(@__DIR__, "..", "wave_out", "device_studies")
isdir(STUDY_OUT) || mkpath(STUDY_OUT)
const P_UM    = 129.0
const C_SITES = 299.792458 / P_UM                 # c in sites/ps

"""Write rows (a vector of NamedTuples) as a TSV with a header."""
function write_tsv(name, rows)
    path = joinpath(STUDY_OUT, name)
    open(path, "w") do io
        println(io, join(string.(keys(rows[1])), '\t'))
        for r in rows
            println(io, join([x isa AbstractFloat ? @sprintf("%.6g", x) : string(x) for x in values(r)], '\t'))
        end
    end
    println("    wrote ", path)
end

lin(x, y) = (mx = mean(x); my = mean(y);
             b = sum((x .- mx) .* (y .- my)) / sum((x .- mx) .^ 2); (b, my - b * mx))

"""Radial front speed of `sheet` (sites/ps) from a centred seed."""
function medium_speed(sheet, dev; nsteps, rmax, amp = 40.0, mask = nothing)
    N = size(sheet)[1]; cen = (N ÷ 2, N ÷ 2)
    tf = sheet_simulate(sheet, nsteps; drive = seed_drive(N, N; radius = 3, amp), device = dev).t_first
    mask === nothing || (tf = ifelse.(mask, tf, Inf32))
    return arrival_speed(tf; center = cen, rmin = 15, rmax)[1]
end

"""
    edge_speed_sides(tf, u; row0, base, mask = trues) -> v or NaN

Speed from the two side columns of a sheet whose track runs down the centre
column (see device_velocity_readout.jl): line fit of first-spike time against
row over the part of the edge that lies on the cone.
"""
function edge_speed_sides(tf, u; row0, base, mask = trues(size(tf)))
    N = size(tf, 1); ρe = size(tf, 2) ÷ 2 - 1
    pts(smin) = begin
        s_e = Float64[]; t_e = Float64[]
        for col in (1, size(tf, 2)), y in 1:N
            s = y - row0; t = tf[y, col]
            (mask[y, col] && isfinite(t) && smin <= s <= base) || continue
            push!(s_e, s); push!(t_e, t)
        end
        s_e, t_e
    end
    s1, t1 = pts(base ÷ 2)
    length(s1) < 20 && return NaN
    v1 = 1 / lin(s1, t1)[1]
    v1 > 1.05u || return NaN
    smin = ρe * tan(asin(u / v1)) + 25
    smin < base - 60 || return NaN
    s2, t2 = pts(round(Int, smin))
    length(s2) < 20 && return NaN
    return 1 / lin(s2, t2)[1]
end

# ---- Perimeter (all four edges) track fit -------------------------------------

"""Perimeter sites of an H×W sheet as (row, col) tuples, each once."""
perimeter(H, W) = unique(vcat([(1, x) for x in 1:W], [(H, x) for x in 1:W],
                              [(y, 1) for y in 1:H], [(y, W) for y in 1:H]))

# Distance along heading û from e to the sheet boundary (the track's exit).
function exit_length(e, θ, H, W)
    uy, ux = cos(θ), sin(θ)
    ts = Float64[]
    uy > 1e-12  && push!(ts, (H - e[1]) / uy)
    uy < -1e-12 && push!(ts, (1 - e[1]) / uy)
    ux > 1e-12  && push!(ts, (W - e[2]) / ux)
    ux < -1e-12 && push!(ts, (1 - e[2]) / ux)
    return max(minimum(ts), 0.0)
end

"""
    huygens_time(p, e, θ, v, u, s_exit) -> first-arrival time minus t₀

Exact first arrival at `p` in a uniform medium of speed `u` from a source
moving at `v` from `e` along heading `θ` (0 = increasing row) for `s_exit`
sites: minimise s'/v + |p − e − s'û|/u over s' ∈ [0, s_exit]. The minimiser is
s_p − ρ_p·tan α when the particle is supersonic, clamped to the track.
"""
function huygens_time(p, e, θ, v, u, s_exit)
    uy, ux = cos(θ), sin(θ)
    dy, dx = p[1] - e[1], p[2] - e[2]
    sp = dy * uy + dx * ux
    ρp = abs(-dy * ux + dx * uy)
    sc = v > u ? clamp(sp - ρp * tan(asin(u / v)), 0.0, s_exit) : 0.0
    return sc / v + sqrt((sp - sc)^2 + ρp^2) / u
end

function _nelder_mead(f, x0, step; iters = 600, tol = 1e-10)
    n = length(x0)
    S = [copy(x0)]
    for i in 1:n
        x = copy(x0); x[i] += step[i]; push!(S, x)
    end
    F = map(f, S)
    for _ in 1:iters
        o = sortperm(F); S = S[o]; F = F[o]
        (F[end] - F[1]) < tol * (abs(F[1]) + tol) && break
        c = sum(S[1:n]) / n
        xr = c .+ (c .- S[end]); fr = f(xr)
        if fr < F[1]
            xe = c .+ 2 .* (c .- S[end]); fe = f(xe)
            fe < fr ? (S[end] = xe; F[end] = fe) : (S[end] = xr; F[end] = fr)
        elseif fr < F[n]
            S[end] = xr; F[end] = fr
        else
            xc = c .+ 0.5 .* (S[end] .- c); fc = f(xc)
            if fc < F[end]
                S[end] = xc; F[end] = fc
            else
                for i in 2:n+1
                    S[i] = S[1] .+ 0.5 .* (S[i] .- S[1]); F[i] = f(S[i])
                end
            end
        end
    end
    i = argmin(F)
    return S[i], F[i]
end

"""
    perimeter_fit(tf, u) -> (; v, θ, e, t0, rms, n)

Recover a straight track — entry point `e` on the boundary, heading `θ`, speed
`v` — from first-spike times on the sheet's perimeter only, by least squares on
[`huygens_time`](@ref) with `t₀` eliminated in closed form. Coarse grid over
(entry, heading, log v), then Nelder–Mead. `u` is the calibrated medium speed.
"""
function perimeter_fit(tf, u; vgrid = exp.(range(log(1.02u), log(5.0), length = 25)))
    H, W = size(tf)
    P = [p for p in perimeter(H, W) if isfinite(tf[p...])]
    T = Float64[tf[p...] for p in P]
    cost(e, θ, v) = begin
        (v > 0 && 1 <= e[1] <= H && 1 <= e[2] <= W) || return Inf
        se = exit_length(e, θ, H, W)
        m = [huygens_time(p, e, θ, v, u, se) for p in P]
        r = T .- m; r .-= mean(r)
        sum(abs2, r)
    end
    # entry estimate: the earliest perimeter spike is near where the track enters
    i0 = argmin(T); e0 = Float64.(P[i0])
    best = (Inf, e0, 0.0, 1.0)
    for θ in range(-π, π; length = 49)[1:48], v in vgrid
        c = cost(e0, θ, v)
        c < best[1] && (best = (c, e0, θ, v))
    end
    _, e0, θ0, v0 = best
    f(x) = cost((x[1], x[2]), x[3], exp(x[4]))
    x, _ = _nelder_mead(f, [e0[1], e0[2], θ0, log(v0)], [3.0, 3.0, 0.05, 0.05])
    x, c = _nelder_mead(f, x, [1.0, 1.0, 0.01, 0.01])
    e = (x[1], x[2]); θ = x[3]; v = exp(x[4])
    se = exit_length(e, θ, H, W)
    m = [huygens_time(p, e, θ, v, u, se) for p in P]
    t0 = mean(T .- m)
    return (; v, θ = rem2pi(θ, RoundNearest), e, t0, rms = sqrt(c / length(P)), n = length(P))
end
