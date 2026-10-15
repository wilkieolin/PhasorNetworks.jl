# device_sheet.jl — a 2-D sheet of device cells coupled by real delay lines
#
# The continuous-time counterpart of PhasorWaveSheet / ExcitableWaveSheet. Each
# site is an `AbstractDeviceCell` (devices.jl); each coupling is a physical
# kinetic-inductance line that carries the source cell's *output current* to
# the target after a real delay. Nothing is carrier-locked and nothing is a
# phase factor: a pulse emitted at t arrives at t + D.
#
# THE LINK ARRAYS. Every cell has K incoming lines, one per stencil offset
# (dy_k, dx_k). Their delays and gains are stored PER LINK, at the receiver:
#
#     D[y,x,k]  delay (ps) of the line from (y+dy_k, x+dx_k) into (y,x)
#     G[y,x,k]  gain of that line (fraction of the source's output current)
#
#     I_in[y,x](t) = Σ_k G[y,x,k] · out[y+dy_k, x+dx_k](t − D[y,x,k])
#
# Per link, not per offset, on purpose: the delays are a processing resource.
# `D[:,:,k] = d₀ + |offset_k|·ps_per_site` everywhere is the uniform medium (the
# Mach-cone baseline); anything else — regions tuned to different speeds,
# heading-aligned links, fabrication scatter, a trained delay map — is the same
# arrays with different contents. That breaks translation invariance, so there
# is no FFT path; coupling is a K-tap gather per cell per step, which is cheap
# at the stencil sizes a real layout can afford.
#
# Stored at the RECEIVER so the update is a gather: each cell reads its K
# inputs and writes only itself — no atomics, the same kernel on CPU threads
# and GPU.
#
# THE HISTORY. Outputs are written each step into a ring buffer (H,W,NB) and
# read back at fractional delay by linear interpolation. Each link's delay is
# split once, at construction, into whole steps `qi` and a fraction `qf` of
# `D/dt − ½`; the read is then
#
#     out(t_n + dt/2 − D) = qf·S[n−qi−1] + (1−qf)·S[n−qi]
#
# which never forms `n − D/dt` in Float32 (at n ~ 10⁴ that would cost ~10⁻³ of
# a step). The `+dt/2` is the midpoint rule: the cell holds its input across a
# step, so it should see the input at the step's centre. Causality needs
# D ≥ dt/2; the constructor enforces D ≥ dt.
#
# WHY THIS MEDIUM IS QUIESCENT BY CONSTRUCTION. In the phasor sheets the
# subthreshold field couples linearly, so rest is stable only if the spectral
# radius ρ_sub < 1 — the constraint that drove ExcitableWaveSheet's whole
# calibration. Here a cell drives its lines only when it FIRES (`out` is the
# emitted pulse, zero otherwise), so subthreshold activity does not propagate
# at all. Excitability reduces to two numbers: how many coincident input
# pulses it takes to fire a cell (sets the critical nucleus) and the refractory
# time (sets whether colliding fronts annihilate).
#
# CALIBRATION (2026-10-15; default RFTransponder — 10 GHz, Q = 21, I_th = 20 µA,
# t_ref = 300 ps, 20 µA/10 ps output pulse; open boundary; dt = 1 ps). Seed
# radius r is a forced nucleus (seed_drive, 40 µA); u from first-spike times at
# r ∈ [10, 40]; "aniso" is diagonal/axis speed.
#
#     R  gain   r=0   r=1   r=2   u (sites/ps)  aniso
#     2  0.30    —     —     —        —           —      nothing propagates
#     2  0.45    —    ✓     ✓      0.0339       0.992
#     3  0.20    —     —    ✓      0.0416       1.001
#     3  0.30    —    ✓     ✓      0.0474       0.998   <-- DEFAULT
#     3  0.70    —    ✓     ✓      0.0569       0.995
#
# At the default, on 128²: u = 0.04752 identical to 5 digits over drive
# amplitude 25–400 µA and seed radius 1–4; two colliding fronts, and one front
# meeting its own wrapped copy on a torus, both leave every site fired exactly
# once and the sheet at rest; random single-site hits at 0.2/1/3% density fire
# only the hit sites. The window is wide — compare ExcitableWaveSheet's, where
# one knob a notch either way floods or collapses — because nothing couples
# below threshold.
#
# The line delay is the speed knob: u ≈ 1/(ps_per_site + ℓ) with a hop latency
# ℓ ≈ 10–14 ps/site (the tank's rise to threshold plus the pulse peak):
#     ps_per_site   2      5      10     20     40
#     u           0.086  0.066  0.048  0.031  0.018
# so the Mach number of a particle crossing at 10 ps/site (0.1 sites/ps) is set
# by layout: ≈ 2.1 at the default.
#
# Speed: CPU 4.7 / 19 ms per step at 256² / 512² (16 threads); GPU (GB10)
# 0.76 / 1.8 ms, agreeing with CPU to 1e-4 ps in first-spike time.
#
# Memory: state 5·H·W, link arrays 3·H·W·K (D, G, qf; qi is Int32), history
# H·W·NB with NB = max(D)/dt + 3, all Float32. At 640², K = 28, D ≤ 300 ps,
# dt = 1 ps: 8 + 138 + 495 MB.

"""
    stencil_offsets(radius) -> Vector{NTuple{2,Int}}

All non-zero `(dy, dx)` with `√(dy² + dx²) ≤ radius`. Radius 1 is the 4-tap
cross, 2 is 12 taps, 3 is 28. Larger discs give rounder fronts (a 4-tap cross
propagates a diamond); see [`front_anisotropy`](@ref).
"""
stencil_offsets(R::Real) = [(dy, dx) for dy in -floor(Int, R):floor(Int, R)
                                     for dx in -floor(Int, R):floor(Int, R)
                                     if 0 < dy^2 + dx^2 <= R^2]

"""
    DeviceSheet(cell, H, W; dt = 1, radius = 3, ps_per_site = 10, d0 = 0,
                gain = 0.3, D = nothing, G = nothing, boundary = :open)

An `H×W` sheet of `cell`s coupled by delay lines (see the file header for the
link-array convention). Units are the device system: ps, µA.

# Keyword arguments
- `dt = 1` — step, ps. The cell tests put spike-time error at 0.08 ps for 2 ps
  steps and 1.3 ps for 5 ps; keep `dt ≤ 2` when timing at the 10 ps scale
  matters. Every delay must be ≥ `dt`.
- `radius = 3` — stencil radius; `K = length(stencil_offsets(radius))`.
- `ps_per_site = 10`, `d0 = 0` — the default uniform delay map
  `D = d0 + ps_per_site·|offset|`. `ps_per_site` is the delay of a straight line
  spanning one lattice pitch: it cannot be below pitch × the line's ps/mm
  (`delay_line_length`), only above it (meandering).
- `gain = 0.3` — default uniform `G`: the fraction of a source's output current
  delivered by each line. With the default cell a single line at gain 1 drives
  a quiescent cell to ≈1.2·I_th, so 0.3 needs ~3–4 roughly coincident inputs.
- `D`, `G` — explicit `(H, W, K)` arrays, overriding the defaults. Use
  [`uniform_delays`](@ref) to start from the uniform map and edit.
- `boundary = :open` — lines that would leave the sheet do not exist. `:periodic`
  wraps to a torus (fronts then meet their own wrapped copies).
"""
struct DeviceSheet{C<:AbstractDeviceCell}
    cell::C
    H::Int
    W::Int
    dt::Float32
    offsets::Matrix{Int32}        # (2, K): rows dy, dx
    D::Array{Float32,3}           # (H,W,K) ps
    G::Array{Float32,3}           # (H,W,K)
    qi::Array{Int32,3}            # whole steps of D/dt − ½
    qf::Array{Float32,3}          # fractional part
    NB::Int                       # ring-buffer depth
    periodic::Bool
end

"""
    uniform_delays(H, W, offsets; ps_per_site = 10, d0 = 0) -> (H, W, K)

The uniform medium: every link's delay is `d0 + ps_per_site·|offset|`.
"""
function uniform_delays(H::Integer, W::Integer, offs; ps_per_site::Real = 10, d0::Real = 0)
    D = Array{Float32}(undef, H, W, length(offs))
    for (k, (dy, dx)) in enumerate(offs)
        D[:, :, k] .= Float32(d0 + ps_per_site * sqrt(dy^2 + dx^2))
    end
    return D
end

function DeviceSheet(cell::AbstractDeviceCell, H::Integer, W::Integer;
                     dt::Real = 1.0, radius::Real = 3, ps_per_site::Real = 10.0,
                     d0::Real = 0.0, gain::Real = 0.3, D = nothing, G = nothing,
                     boundary::Symbol = :open)
    boundary in (:open, :periodic) || throw(ArgumentError("boundary must be :open or :periodic"))
    offs = stencil_offsets(radius)
    K = length(offs)
    K > 0 || throw(ArgumentError("radius $radius gives an empty stencil"))
    D = D === nothing ? uniform_delays(H, W, offs; ps_per_site, d0) : Float32.(D)
    G = G === nothing ? fill(Float32(gain), H, W, K) : Float32.(G)
    size(D) == (H, W, K) || throw(DimensionMismatch("D must be ($H, $W, $K), got $(size(D))"))
    size(G) == (H, W, K) || throw(DimensionMismatch("G must be ($H, $W, $K), got $(size(G))"))
    minimum(D) >= dt || throw(ArgumentError("every delay must be ≥ dt = $dt ps; min(D) = $(minimum(D))"))
    q  = Float64.(D) ./ dt .- 0.5
    qi = floor.(Int32, q)
    qf = Float32.(q .- qi)
    NB = Int(maximum(qi)) + 3
    O  = Int32[o[j] for j in 1:2, o in offs]
    return DeviceSheet(cell, Int(H), Int(W), Float32(dt), O, D, G, qi, qf, NB, boundary === :periodic)
end

Base.size(s::DeviceSheet) = (s.H, s.W)
n_links(s::DeviceSheet) = size(s.offsets, 2)

function Base.show(io::IO, s::DeviceSheet)
    print(io, "DeviceSheet($(s.H)×$(s.W), K=$(n_links(s)), dt=$(s.dt) ps, D∈[",
          round(minimum(s.D); digits = 1), ", ", round(maximum(s.D); digits = 1), "] ps, ",
          s.periodic ? "periodic" : "open", "; ", s.cell, ")")
end

# ---- Coupling gather -----------------------------------------------------

@kernel function _sheet_gather_kernel!(Iin, @Const(buf), @Const(G), @Const(qi), @Const(qf),
                                       @Const(offs), n::Int, NB::Int, H::Int, W::Int,
                                       K::Int, periodic::Bool)
    y, x = @index(Global, NTuple)
    acc = 0f0
    for k in 1:K
        sy = y + offs[1, k]
        sx = x + offs[2, k]
        if periodic
            sy = mod1(sy, H); sx = mod1(sx, W)
        end
        if 1 <= sy <= H && 1 <= sx <= W
            m1 = n - qi[y, x, k]                       # newer sample
            f  = qf[y, x, k]
            v  = f * buf[sy, sx, mod(m1 - 1, NB) + 1] + (1f0 - f) * buf[sy, sx, mod(m1, NB) + 1]
            acc += G[y, x, k] * v
        end
    end
    @inbounds Iin[y, x] = acc
end

"""
    sheet_input!(Iin, sheet, buf, n, links)

Coupled input current at step `n` (from `t_n` to `t_n + dt`), read from the
history `buf` at each link's delay. `links` is `(; G, qi, qf, offs)` already on
the device of `buf`.
"""
function sheet_input!(Iin, sheet::DeviceSheet, buf, n::Integer, links)
    backend = get_backend(Iin)
    _sheet_gather_kernel!(backend)(Iin, buf, links.G, links.qi, links.qf, links.offs,
                                   Int(n), sheet.NB, sheet.H, sheet.W, n_links(sheet),
                                   sheet.periodic; ndrange = (sheet.H, sheet.W))
    return Iin
end

# ---- External drive: detector events --------------------------------------

"""
    EventDrive(t_hit, amp; tau = 10)

External current from one detector event per site: a site with finite
`t_hit[y,x]` (ps) receives the alpha pulse `amp·(s)·e^{1−s}`, `s = (t − t_hit)/τ`,
peaking at `amp` (µA) a time `τ` after the hit. `Inf` means the site is never
hit. Callable: `drive(t)` returns the `(H, W)` current at time `t`.

Sum several with `+`. Build with [`seed_drive`](@ref) or
[`particle_drive`](@ref).
"""
struct EventDrive
    t_hit::Matrix{Float32}
    amp::Matrix{Float32}
    tau::Float32
end

EventDrive(t_hit, amp; tau::Real = 10.0) =
    EventDrive(Float32.(t_hit), amp isa Number ? fill(Float32(amp), size(t_hit)) : Float32.(amp), Float32(tau))

function (d::EventDrive)(t::Real)
    s = (Float32(t) .- d.t_hit) ./ d.tau
    return ifelse.(s .> 0f0, d.amp .* s .* exp.(1f0 .- s), 0f0)
end

struct SumDrive
    parts::Vector{EventDrive}
end
Base.:+(a::EventDrive, b::EventDrive) = SumDrive([a, b])
Base.:+(a::SumDrive, b::EventDrive) = SumDrive([a.parts; b])
(d::SumDrive)(t::Real) = sum(p(t) for p in d.parts)

"""
    seed_drive(H, W; center = (H÷2, W÷2), radius = 2, t0 = 0, amp = 40, tau = 10)

Every site within `radius` of `center` is hit at `t0`. With the default cell a
40 µA, 10 ps pulse fires a quiescent cell on its own (≈1.2 µA of loop current
per µA of peak drive), so this is a forced nucleus of the given radius.
"""
function seed_drive(H::Integer, W::Integer; center = (H ÷ 2, W ÷ 2), radius::Real = 2,
                    t0::Real = 0, amp::Real = 40, tau::Real = 10)
    cy, cx = center
    th = Float32[(y - cy)^2 + (x - cx)^2 <= radius^2 ? t0 : Inf32 for y in 1:H, x in 1:W]
    return EventDrive(th, amp; tau)
end

"""
    particle_drive(H, W; dt_site, angle = 0, x0 = (1, W÷2), t0 = 0, width = 1,
                   amp = 40, tau = 10, jitter_fwhm_ps = 0, dropout_frac = 0, rng)

A particle crossing the sheet in a straight line from `x0` along `angle`
(radians; 0 = increasing row, the `moving_drive` convention), passing one
lattice pitch every `dt_site` ps — so its speed is `1/dt_site` sites/ps. Each
site within `width/2` of the track is hit once, when the particle passes its
closest point; sites behind `x0` are not hit.

`width` must exceed the sheet's critical nucleus or the track fires but
launches no lateral front (the lesson of `demos/wave_mach_cone.jl` (1)).

Detector imperfections are frozen per event, as in [`detector_noise`](@ref):
one Gaussian timing error per site (`jitter_fwhm_ps`, FWHM) and a fraction
`dropout_frac` of sites that register nothing.
"""
function particle_drive(H::Integer, W::Integer; dt_site::Real, angle::Real = 0.0,
                        x0 = (1, W ÷ 2), t0::Real = 0, width::Real = 1.0,
                        amp::Real = 40, tau::Real = 10, jitter_fwhm_ps::Real = 0,
                        dropout_frac::Real = 0, rng::AbstractRNG = Random.default_rng())
    dt_site > 0 || throw(ArgumentError("dt_site must be > 0"))
    0 <= dropout_frac < 1 || throw(ArgumentError("dropout_frac must lie in [0,1)"))
    uy, ux = cos(angle), sin(angle)
    σ = jitter_fwhm_ps / 2.355
    th = fill(Inf32, H, W)
    for x in 1:W, y in 1:H
        ry, rx = y - x0[1], x - x0[2]
        along = ry * uy + rx * ux
        perp  = abs(-ry * ux + rx * uy)
        (along >= 0 && perp <= width / 2) || continue
        dropout_frac > 0 && rand(rng) < dropout_frac && continue
        th[y, x] = Float32(t0 + along * dt_site + (σ > 0 ? σ * randn(rng) : 0.0))
    end
    return EventDrive(th, amp; tau)
end

# ---- Rollout ---------------------------------------------------------------

"""
    sheet_simulate(sheet, nsteps; drive = nothing, s0 = nothing,
                   frame_every = 0, keep_amplitude = false, device = cpu_device())
        -> (; t_first, n_spikes, fire_frames, amp_frames, frame_t, rate, state)

Roll the sheet out for `nsteps` steps of `sheet.dt`.

`drive(t)` returns the external current `(H, W)` in µA at time `t` (an
[`EventDrive`](@ref), a sum of them, or any function); it is sampled at each
step's midpoint and added to the coupled input.

Always returned:
- `t_first` — `(H, W)` time of each site's first spike, ps (`Inf32` if none).
  The cleanest record of a front: its gradient is the local slowness.
- `n_spikes` — `(H, W)` spike count.
- `rate` — `(nsteps,)` fraction of the sheet firing in each step.

With `frame_every = m > 0`, also `fire_frames` `(H, W, F)` — 1 where a site
fired at any point in that m-step window (the instantaneous wavefront) — and,
with `keep_amplitude`, `amp_frames` `(H, W, F)` = `|cell_phasor|/I_th` at each
frame's end. `frame_t` holds the frame end times. Frames cost `4·H·W` bytes each.

`device` places the rollout (e.g. `gpu_device()`); results come back on CPU.
"""
function sheet_simulate(sheet::DeviceSheet, nsteps::Integer; drive = nothing, s0 = nothing,
                        frame_every::Integer = 0, keep_amplitude::Bool = false,
                        device = cpu_device())
    H, W, dt, NB = sheet.H, sheet.W, sheet.dt, sheet.NB
    cell = sheet.cell
    stp = cell_stepper(cell, dt)
    s = device(s0 === nothing ? cell_state(cell, H, W) : s0)
    links = device((; G = sheet.G, qi = sheet.qi, qf = sheet.qf, offs = sheet.offsets))
    buf = device(zeros(Float32, H, W, NB))
    buf[:, :, 1] .= cell_output(cell, s)               # sample m = 0 (slot mod(0,NB)+1)
    Iin = device(zeros(Float32, H, W))
    t_first = device(fill(Inf32, H, W))
    nsp = device(zeros(Float32, H, W))
    nF = frame_every > 0 ? nsteps ÷ frame_every : 0
    Ff = Array{Float32}(undef, H, W, nF)
    Af = keep_amplitude ? Array{Float32}(undef, H, W, nF) : nothing
    ft = Vector{Float32}(undef, nF)
    win = device(zeros(Float32, H, W))
    rate = Vector{Float32}(undef, nsteps)
    for n in 0:nsteps-1
        tn = n * dt
        sheet_input!(Iin, sheet, buf, n, links)
        I = drive === nothing ? Iin : Iin .+ device(drive(tn + dt / 2))
        s, fire, tf = cell_step(stp, s, I)
        buf[:, :, mod(n + 1, NB) + 1] .= cell_output(cell, s)
        t_first .= ifelse.(fire .& isinf.(t_first), tn .+ tf, t_first)
        nsp .+= fire
        rate[n+1] = sum(fire) / Float32(H * W)
        if nF > 0
            win .= max.(win, fire)
            if (n + 1) % frame_every == 0
                k = (n + 1) ÷ frame_every
                Ff[:, :, k] .= Array(win); win .= 0f0
                keep_amplitude && (Af[:, :, k] .= Array(abs.(cell_phasor(cell, s)) ./ cell.I_th))
                ft[k] = (n + 1) * dt
            end
        end
    end
    return (; t_first = Array(t_first), n_spikes = Array(nsp), fire_frames = Ff,
              amp_frames = Af, frame_t = ft, rate, state = map(Array, s))
end

# ---- Front diagnostics -----------------------------------------------------

"""
    arrival_speed(t_first; center, rmin, rmax, sector = nothing) -> (speed, n)

Front speed in sites/ps from a first-spike map: least-squares slope of radius
against first-spike time over sites with `rmin ≤ r ≤ rmax` from `center`.
Excluding the near field matters — the nucleus fires on the drive, not the
medium, and a small front is slowed by its own curvature (`v = v_plane − D·κ`).

`sector = (θ, halfwidth)` restricts to directions within `halfwidth` radians of
`θ` (0 = increasing row). Returns `(NaN32, n)` with fewer than 10 sites.
"""
function arrival_speed(t_first::AbstractMatrix; center, rmin::Real, rmax::Real, sector = nothing)
    cy, cx = center
    ts = Float64[]; rs = Float64[]
    for x in axes(t_first, 2), y in axes(t_first, 1)
        t = t_first[y, x]; isfinite(t) || continue
        dy, dx = y - cy, x - cx
        r = sqrt(dy^2 + dx^2)
        rmin <= r <= rmax || continue
        if sector !== nothing
            θ, hw = sector
            abs(rem2pi(atan(dx, dy) - θ, RoundNearest)) <= hw || continue
        end
        push!(ts, t); push!(rs, r)
    end
    length(ts) < 10 && return (NaN32, length(ts))
    μt, μr = mean(ts), mean(rs)
    return (Float32(sum((ts .- μt) .* (rs .- μr)) / sum((ts .- μt) .^ 2)), length(ts))
end

"""
    front_anisotropy(t_first; center, rmin, rmax) -> (axis, diagonal, ratio)

Front speed along the lattice axes versus the diagonals (±22.5° sectors),
from [`arrival_speed`](@ref). A round front has `ratio ≈ 1`; the 4-tap cross
gives a diamond. This matters for the Mach cone, whose half-angle is read off
the front speed in one specific direction.
"""
function front_anisotropy(t_first::AbstractMatrix; center, rmin::Real, rmax::Real)
    hw = π / 8
    ax = mean(filter(isfinite, [arrival_speed(t_first; center, rmin, rmax, sector = (θ, hw))[1]
                                for θ in (0, π / 2, π, -π / 2)]))
    dg = mean(filter(isfinite, [arrival_speed(t_first; center, rmin, rmax, sector = (θ, hw))[1]
                                for θ in (π / 4, 3π / 4, -π / 4, -3π / 4)]))
    return (; axis = Float32(ax), diagonal = Float32(dg), ratio = Float32(dg / ax))
end

"""
    link_lengths(sheet; kwargs...) -> (; per_link_um, per_site_mm, total_m)

Physical kinetic-inductance line needed to realise the sheet's delay map,
through [`delay_line_length`](@ref) (`kwargs` forwarded: sheet inductance,
dielectric, ε_r). Links that leave an open sheet are not counted.
"""
function link_lengths(sheet::DeviceSheet; kwargs...)
    um_per_ps = delay_line_length(1.0; kwargs...).length_um
    L = sheet.D .* Float32(um_per_ps)
    if !sheet.periodic
        for k in 1:n_links(sheet), x in 1:sheet.W, y in 1:sheet.H
            sy, sx = y + sheet.offsets[1, k], x + sheet.offsets[2, k]
            (1 <= sy <= sheet.H && 1 <= sx <= sheet.W) || (L[y, x, k] = 0f0)
        end
    end
    per_site = dropdims(sum(L; dims = 3); dims = 3) ./ 1000
    return (; per_link_um = L, per_site_mm = per_site, total_m = sum(per_site) / 1000)
end
