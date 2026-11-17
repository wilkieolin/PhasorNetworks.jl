# devices.jl — physical device cells for continuous-time spiking sheets
#
# The phasor layers (PhasorWaveSheet, ExcitableWaveSheet) step once per carrier
# period and carry conduction delay as a carrier phase e^{-iωr/c}. That is exact
# only for carrier-locked phasors. A superconducting device emits real current
# pulses, and a kinetic-inductance line delays them by real time — so a sheet of
# devices has to be stepped in physical time, with delays read out of a history
# buffer. This file is the cell half of that: the device model, stepped on
# whole arrays (any shape, CPU or GPU), with the threshold / reset / refractory
# events applied between steps rather than inside an ODE right-hand side.
#
# UNITS. Everything is in a self-consistent picosecond system, chosen so that
# Float32 stays well-conditioned and no 1e-12 factors appear anywhere:
#
#     time ps   current µA   voltage µV   inductance pH   capacitance pF   resistance Ω
#
# It closes: µV = pH·µA/ps, µA = pF·µV/ps, Ω·pF = ps, pH/Ω = ps, pH·pF = ps².
# Frequencies are therefore in 1/ps; 10 GHz = 0.01 ps⁻¹.
#
# THE R&F TRANSPONDER. A superconducting loop (inductance L) shunted by a
# capacitance C and a damping resistance R, driven by current injected at the
# tank node. A nanowire in the loop switches when the loop current reaches its
# switching current I_th; the hotspot dumps the loop (reset), emits an output
# current pulse into the outgoing lines, and stays resistive for its recovery
# time t_ref (absolute refractory). The subthreshold tank is linear:
#
#     C dv/dt = I − v/R − i        L di/dt = v
#
# so it is integrated EXACTLY over a step by a 2×2 matrix exponential with the
# input held across the step — unconditionally stable at any dt, and the same
# object as the phasor layers' A_step = exp((λ+iω)T). The phasor it is
# equivalent to has
#
#     λ = −1/(2RC),   ω = √(1/LC − λ²),   Q = R·√(C/L) = ω₀RC
#
# (parallel-RLC Q. SpikingNetworks.jl applied this formula to a SERIES tank,
# where the right one is √(L/C)/R; its reported Q was off by ~200×.)
#
# At the hardware ladder's design point — 10 GHz, Q ≈ 21 — that is λT = π/Q =
# 0.15 per period, exactly the `leakage` the phasor layers already default to.
# `phasor_equivalent` makes this mapping explicit so a device sheet and a phasor
# sheet can be configured to the same operating point and compared.
#
# WHY NOT SpikingNetworks.jl's TRANSPONDER CODE. Measured 2026-10-15 before it
# was retired: driven to 10× threshold its R&F cell never entered refractory
# (`ref_timer` is set on a local, not on `dz`), its coupling never delivered a
# pulse (emission is gated on an integrated state equalling `t` to 1e-12), and
# its IaF cell ignored its input argument. All three also mutated model state
# inside the ODE right-hand side. Only the circuit topology is carried over.

"""
    AbstractDeviceCell

A physical neuron/resonator model stepped in picoseconds on whole arrays.
A concrete cell implements

- `cell_state(cell, dims...)` — zeroed state as a NamedTuple of arrays,
- `cell_stepper(cell, dt)` — precomputed per-step constants, and
  `cell_step(stepper, s, I)` — advance one step with input current `I` held
  across it; returns `(s_next, fire, t_fire)`,
- `cell_output(cell, s)` — the current the cell is driving into its outputs,

and may implement `cell_phasor(cell, s)` for a complex readout.
"""
abstract type AbstractDeviceCell end

"""
    RFTransponder(; f0_GHz = 10, Q = 21, L = 100, I_th = 20, t_ref = 300,
                    I_out = 20, tau_out = 10)

Resonate-and-fire superconducting transponder. Units: ps, µA, µV, pH, pF, Ω
(see the file header).

The tank is specified by its resonance and quality factor; `L` (pH) fixes the
impedance level, and `C`, `R` follow:

    C = 1/(ω₀² L),   R = Q·√(L/C).

# Keyword arguments
- `f0_GHz = 10` — undamped tank resonance ω₀/2π. The hardware ladder's carrier.
- `Q = 21` — loaded quality factor. 21 ⇔ λT = 0.15/period, the phasor layers'
  default leakage. Superconducting resonators natively reach 10³–10⁶ and would
  need *spoiling* down to this.
- `L = 100` — loop inductance in pH.
- `I_th = 20` — nanowire switching current, µA, compared against `|i_L|`.
- `t_ref = 300` — hotspot recovery time, ps (absolute refractory: the loop is
  held reset). This, times the sheet's front speed, is the refractory
  wavelength, which must exceed the sheet's half-width for colliding fronts to
  annihilate (see `src/excitable.jl`).
- `I_out = 20` — peak of the emitted output current pulse, µA.
- `tau_out = 10` — output pulse time constant, ps. The pulse is an alpha
  function `I_out·(t/τ)·e^{1−t/τ}`, peaking at `t = τ`, carrying charge
  `I_out·e·τ`.
"""
struct RFTransponder <: AbstractDeviceCell
    L::Float32
    C::Float32
    R::Float32
    I_th::Float32
    t_ref::Float32
    I_out::Float32
    tau_out::Float32
end

function RFTransponder(; f0_GHz::Real = 10.0, Q::Real = 21.0, L::Real = 100.0,
                       I_th::Real = 20.0, t_ref::Real = 300.0,
                       I_out::Real = 20.0, tau_out::Real = 10.0)
    f0_GHz > 0 || throw(ArgumentError("f0_GHz must be > 0, got $f0_GHz"))
    Q > 0.5   || throw(ArgumentError("Q must exceed 0.5 (underdamped) for a resonator, got $Q"))
    L > 0 && I_th > 0 && tau_out > 0 && t_ref >= 0 ||
        throw(ArgumentError("need L, I_th, tau_out > 0 and t_ref ≥ 0"))
    ω0 = 2π * f0_GHz * 1e-3                          # rad/ps
    C  = 1 / (ω0^2 * L)
    R  = Q * sqrt(L / C)
    return RFTransponder(L, C, R, I_th, t_ref, I_out, tau_out)
end

function Base.show(io::IO, c::RFTransponder)
    pe = phasor_equivalent(c)
    print(io, "RFTransponder(f₀=", round(resonance_frequency(c); sigdigits = 4), " GHz, Q=",
          round(quality_factor(c); sigdigits = 3), ", λT=", round(-pe.leakage; sigdigits = 3),
          ", I_th=", c.I_th, " µA, t_ref=", c.t_ref, " ps)")
end

"""
    resonance_frequency(cell::RFTransponder) -> GHz

Undamped tank resonance `1/(2π√(LC))`. The ringing frequency is the slightly
lower damped value — see [`phasor_equivalent`](@ref).
"""
resonance_frequency(c::RFTransponder) = 1e3 / (2π * sqrt(Float64(c.L) * c.C))

"""
    quality_factor(cell::RFTransponder)

Parallel-RLC quality factor `R·√(C/L)`.
"""
quality_factor(c::RFTransponder) = Float64(c.R) * sqrt(Float64(c.C) / c.L)

"""
    phasor_equivalent(cell) -> (; leakage, t_period, λ, ω)

The linear phasor neuron this cell's subthreshold tank is equivalent to:
`dz/dt = (λ + iω)z`, with `λ = −1/(2RC)` and `ω` the damped ringing frequency.
`t_period = 2π/ω` in ps, and `leakage = λ·t_period` — the dimensionless
per-period value that `SpikingArgs(leakage=…)` and the wave sheets use.
"""
function phasor_equivalent(c::RFTransponder)
    λ  = -1 / (2 * Float64(c.R) * c.C)
    ω2 = 1 / (Float64(c.L) * c.C) - λ^2
    ω2 > 0 || throw(ArgumentError("tank is overdamped (Q ≤ 0.5); there is no phasor equivalent"))
    ω = sqrt(ω2)
    T = 2π / ω
    return (; leakage = λ * T, t_period = T, λ, ω)
end

# Exact zero-order-hold discretisation of the tank, x = (v, i):
#   ẋ = A x + B I,   A = [−1/RC  −1/C; 1/L  0],   B = [1/C, 0]
#   x⁺ = Φ x + Γ I,  Φ = e^{A dt},  Γ = A⁻¹(Φ − 1)B.
# Computed in Float64 (A is stiff-ish in Float32 at small dt) and rounded once.
function _tank_propagator(c::RFTransponder, dt::Real)
    R, L, C = Float64(c.R), Float64(c.L), Float64(c.C)
    A = [-1/(R*C) -1/C; 1/L 0.0]
    Φ = exp(A .* Float64(dt))
    Γ = A \ ((Φ - [1.0 0.0; 0.0 1.0]) * [1/C, 0.0])
    return Float32.(Φ), Float32.(Γ)
end

"""
    cell_state(cell::RFTransponder, dims...) -> (; v, i, ref, o1, o2)

Zeroed state on CPU. `v` tank voltage (µV), `i` loop current (µA), `ref`
remaining refractory time (ps), `o1, o2` the two stages of the output-pulse
filter (`o2` is the emitted current). Move to a device with `gdev(s)` /
`map(gdev, s)` as with any other state NamedTuple.
"""
function cell_state(::RFTransponder, dims::Integer...)
    z() = zeros(Float32, dims...)
    return (; v = z(), i = z(), ref = z(), o1 = z(), o2 = z())
end
cell_state(cells::AbstractArray{RFTransponder}) = cell_state(first(cells), size(cells)...)

"""
    cell_output(cell::RFTransponder, s) -> array (µA)

Current the cell is presently driving into its output lines.
"""
cell_output(::RFTransponder, s) = s.o2

"""
    cell_phasor(cell::RFTransponder, s) -> complex array (µA)

Energy-normalised complex tank amplitude `z = i − i·v/Z₀`, `Z₀ = √(L/C)`.
`|z|²·L/2` is the stored energy, and for the free tank `z` rotates as
`e^{(λ+iω)t}` — counter-clockwise, the phasor layers' convention — so `angle(z)`
is directly comparable to a phasor neuron's phase and `|z|/I_th` to `|z|/θ`.
"""
cell_phasor(c::RFTransponder, s) = complex.(s.i, .-s.v ./ sqrt(c.L / c.C))

"""
    RFStepper(cell, dt)
    RFStepper(cells::AbstractArray{RFTransponder}, dt)

Precomputed per-step constants for [`cell_step`](@ref): the exact tank
propagator, the threshold, refractory and output-filter constants. Build once
per (cell, dt).

Given an array of cells — a heterogeneous sheet `(H, W)` or reservoir `(N,)` —
every constant becomes an array of that shape instead of a scalar. `cell_step` is broadcast-only, so it runs
unchanged on either; the propagator is computed once per distinct cell.
"""
struct RFStepper{T}
    dt::Float32
    p11::T; p12::T; p21::T; p22::T     # tank propagator Φ
    g1::T; g2::T                       # input gain Γ
    I_th::T
    t_ref::T
    tau::T                             # output pulse τ
    e_out::T                           # e^{−dt/τ}
    kick::T                            # alpha-pulse amplitude I_out·e
    z0::T                              # √(L/C), for cell_phasor
end

"""
    cell_stepper(cell, dt)

Per-step constants for `cell_step` — the hook a sheet uses to stay agnostic of
the cell type. For `RFTransponder`s (one, or a matrix of them) this is an
[`RFStepper`](@ref).
"""
cell_stepper(c::RFTransponder, dt::Real) = RFStepper(c, dt)
cell_stepper(c::AbstractArray{RFTransponder}, dt::Real) = RFStepper(c, dt)

function _rf_constants(c::RFTransponder, dt::Real)
    Φ, Γ = _tank_propagator(c, dt)
    return (Φ[1, 1], Φ[1, 2], Φ[2, 1], Φ[2, 2], Γ[1], Γ[2], c.I_th, c.t_ref, c.tau_out,
            Float32(exp(-dt / c.tau_out)), Float32(c.I_out * ℯ), Float32(sqrt(c.L / c.C)))
end

function RFStepper(c::RFTransponder, dt::Real)
    dt > 0 || throw(ArgumentError("dt must be > 0"))
    return RFStepper{Float32}(Float32(dt), _rf_constants(c, dt)...)
end

function RFStepper(cells::AbstractArray{RFTransponder}, dt::Real)
    dt > 0 || throw(ArgumentError("dt must be > 0"))
    cache = Dict{RFTransponder, NTuple{12, Float32}}()
    k = [get!(() -> _rf_constants(c, dt), cache, c) for c in cells]
    field(j) = Float32[t[j] for t in k]
    return RFStepper{Array{Float32, ndims(cells)}}(Float32(dt), (field(j) for j in 1:12)...)
end

"""
    stepper_to(stp::RFStepper, device) -> RFStepper

Move a heterogeneous stepper's arrays onto `device` (scalars are left alone).
"""
function stepper_to(stp::RFStepper, device)
    f = [getfield(stp, n) for n in fieldnames(RFStepper)][2:end]
    stp.p11 isa Number && return stp
    moved = map(device, f)
    return RFStepper{typeof(moved[1])}(stp.dt, moved...)
end

"""
    cell_step(stp::RFStepper, s, I) -> (s_next, fire, t_fire)

Advance every cell one step of `stp.dt` with input current `I` (µA, same shape
as the state, or a scalar) held across the step. For second-order accuracy in
a time-varying drive, pass the input evaluated at the step MIDPOINT.

Order of operations, and why:
1. The tank evolves exactly (linear, ZOH). Refractory cells are held at zero —
   the hotspot is resistive and the loop cannot store current.
2. A non-refractory cell whose `|i|` reached `I_th` fires. The crossing time
   is linearly interpolated inside the step (`t_fire ∈ [0, dt]` from the step
   start; `NaN32`-free, 0 where no spike), so spike timing is not quantised to
   `dt` — this matters for a delay-coded sheet where 10 ps is one site.
3. A firing cell is reset to zero and its refractory clock started from the
   crossing time, and its output filter receives an alpha-pulse kick also
   aligned to the crossing time.

`force` (Bool array or scalar) makes a cell fire this step whatever its state —
a detector-driven sensor cell — at `t_force` ps from the step start (ignored
for a cell that also crossed threshold on its own). Every cell type accepts
these two keywords.

Pure: returns a new state, never mutates `s`. Broadcast-only, so it runs
unchanged on GPU arrays and on heterogeneous (per-site constant) steppers.
"""
function cell_step(stp::RFStepper, s, I; force = false, t_force = 0f0)
    dt = stp.dt
    live = s.ref .<= 0f0
    v1 = stp.p11 .* s.v .+ stp.p12 .* s.i .+ stp.g1 .* I
    i1 = stp.p21 .* s.v .+ stp.p22 .* s.i .+ stp.g2 .* I
    v1 = ifelse.(live, v1, 0f0)
    i1 = ifelse.(live, i1, 0f0)

    a0, a1 = abs.(s.i), abs.(i1)
    fire_th = live .& (a1 .>= stp.I_th)
    fire = fire_th .| force
    # fraction of the step at which |i| crossed I_th (a0 < I_th ≤ a1 on a fire)
    frac = clamp.((stp.I_th .- a0) ./ max.(a1 .- a0, 1f-12), 0f0, 1f0)
    t_fire = ifelse.(fire_th, frac .* dt, ifelse.(fire, t_force, 0f0))
    t_since = dt .- t_fire                                   # time since the crossing

    v2   = ifelse.(fire, 0f0, v1)
    i2   = ifelse.(fire, 0f0, i1)
    ref2 = ifelse.(fire, stp.t_ref .- t_since, max.(s.ref .- dt, 0f0))

    # Output alpha filter: o1' = −o1/τ, o2' = (o1 − o2)/τ, exact over the step,
    # plus an impulse at the crossing propagated forward by `t_since`.
    τ  = stp.tau
    o2 = stp.e_out .* (s.o2 .+ (dt ./ τ) .* s.o1)
    o1 = stp.e_out .* s.o1
    ek = exp.(.-t_since ./ τ)
    o1 = o1 .+ ifelse.(fire, stp.kick .* ek, 0f0)
    o2 = o2 .+ ifelse.(fire, stp.kick .* (t_since ./ τ) .* ek, 0f0)

    return (; v = v2, i = i2, ref = ref2, o1, o2), fire, t_fire
end

"""
    cell_phasor(stp::RFStepper, s)

[`cell_phasor`](@ref) using the stepper's (possibly per-site) `√(L/C)`.
"""
cell_phasor(stp::RFStepper, s) = complex.(s.i, .-s.v ./ stp.z0)

"""
    time_scaled(cell::RFTransponder, s) -> RFTransponder

The same cell running `s` times slower: `f₀/s`, `L·s` and `C·s` (so `Z₀`, `Q`
and every current are unchanged), output pulse `τ·s`, refractory `t_ref·s`.

This is an EXACT symmetry of the cell, and of a sheet whose line delays are
also scaled by `s`: every state trajectory maps onto the original under
`t → s·t`, so the scaled medium has the same nucleus, the same round fronts and
front speed `u/s`. It is how to slow a resonant medium down.

Lengthening the lines alone fails in two ways (measured on a 10 GHz
checkerboard sublattice, demos/device_studies/e5b_nucleation.jl). Nucleation:
inputs arrive spread by ≈ 2·ps_per_site across a radius-3 stencil, and pulses
half a carrier period apart do not add (the two-pulse coincidence window of
e8_cell_tuning.jl), so small seeds die — at 50 ps/pitch every seed up to
radius 20 failed or misbehaved. Recovery: once a large source does ignite it
(a line source at 70 ps/pitch propagates at 0.0090 sites/ps against a
plane-front prediction of 0.0086), the slow front's refractory zone u·t_ref is
only 3–6 sites, comparable to the stencil's reach, and excitation re-enters:
sites fired 64–80 times and the sheet never returned to rest. Time scaling
fixes both because it scales t_ref with the delays, keeping the input spread
per carrier period and u·t_ref per stencil radius fixed.
"""
function time_scaled(c::RFTransponder, s::Real)
    s > 0 || throw(ArgumentError("s must be > 0"))
    return RFTransponder(c.L * s, c.C * s, c.R, c.I_th, c.t_ref * s, c.I_out, c.tau_out * s)
end

"""
    cell_simulate(cell, drive, dt, nsteps; dims = (), s0 = nothing, stride = 1)
        -> (; t, v, i, out, spike_index, spike_time, state)

Run uncoupled cells for `nsteps` steps of `dt` ps. `drive(t)` returns the input
current (µA) at time `t` ps — a scalar, or an array of shape `dims` — and is
sampled at each step's midpoint.

Traces `v`, `i`, `out` are recorded every `stride` steps with shape
`(dims..., nsteps ÷ stride)`; `t` holds the matching times (step ends). Spike
times are exact crossing times (see [`cell_step`](@ref)), returned as a vector
of `CartesianIndex` into `dims` plus a vector of times in ps.

This is the single-cell / bank characterisation tool; sheets with coupling go
through the sheet engine.
"""
function cell_simulate(cell::RFTransponder, drive, dt::Real, nsteps::Integer;
                       dims::Tuple = (), s0 = nothing, stride::Integer = 1)
    stp = RFStepper(cell, dt)
    # A single cell runs as a length-1 vector: broadcasting over 0-dim arrays
    # returns scalars, which would silently change the state's type mid-run.
    scalar = isempty(dims)
    dims = scalar ? (1,) : dims
    s = s0 === nothing ? cell_state(cell, dims...) :
        map(x -> x isa Number ? fill(Float32(x), dims) : reshape(Array{Float32}(x), dims), s0)
    nrec = nsteps ÷ stride
    V = Array{Float32}(undef, dims..., nrec)
    Ii = similar(V); O = similar(V)
    ts = Vector{Float32}(undef, nrec)
    sp_idx = CartesianIndex{length(dims)}[]
    sp_t = Float32[]
    colons = ntuple(_ -> Colon(), length(dims))
    for n in 1:nsteps
        t0 = Float32((n - 1) * dt)
        s, fire, tf = cell_step(stp, s, drive(t0 + 0.5f0 * Float32(dt)))
        if any(fire)
            f = Array(fire); tfa = Array(tf)
            for ci in CartesianIndices(f)[f]
                push!(sp_idx, ci); push!(sp_t, t0 + tfa[ci])
            end
        end
        if n % stride == 0
            k = n ÷ stride
            V[colons..., k] .= Array(s.v); Ii[colons..., k] .= Array(s.i)
            O[colons..., k] .= Array(s.o2); ts[k] = Float32(n * dt)
        end
    end
    if scalar
        V, Ii, O = vec(V), vec(Ii), vec(O)
    end
    return (; t = ts, v = V, i = Ii, out = O, spike_index = sp_idx, spike_time = sp_t, state = s)
end

# =============================================================================
# THE nTRON LOOP TRANSPONDER — the device the spcnc group is building
# =============================================================================
#
# Port of the reference model in gitlab.com/spcnc/transponder_model
# (src/tests/toy_example.py, doc/main.tex). Unlike RFTransponder there is no
# capacitor and nothing resonates: it is a two-nTron LEAKY INTEGRATOR.
#
#   bias i_b1 ─┬─ input nTron Q₁ channel (L_nw, R_nw(t)) ─ gnd
#              ├─ shunt R_s1 + L_s1 ─ gnd
#              └─ loop  L_T + R_T(t) ─ gate of output nTron Q₂ (biased at i_b2)
#
# At rest the whole bias flows through Q₁'s superconducting channel and the
# loop is empty. An input pulse on Q₁'s gate that takes i_nw + i_gate past Q₁'s
# switching current makes the channel resistive (R_nw ramps to R_norm for a few
# ns), diverting part of the bias into the loop, where it is STORED and leaks
# out with τ ≈ L_T/R_T (≈ 50 ns at the defaults) — each input click deposits a
# quantum of loop current. When i_loop + i_b2 reaches Q₂'s switching current,
# Q₂ fires: its hotspot puts R_norm into the loop (dumping it, τ ≈ 150 ps) and
# it emits an output current pulse, split between its fan-out lines.
#
# The two loop currents x = (i_nw, i_loop) obey the reference model's equations
# (main.tex §Transponder Model, verbatim in toy_example.py):
#
#     ẋ = A(R_nw, R_T) x + c,   den = L_nw(L_s1 − L_T) − L_s1 L_T
#     A = [L_T(R_nw+R_s1) − L_s1 R_nw    L_T R_s1 + L_s1 R_T          ] / den
#         [L_nw R_s1 − L_s1 R_nw          L_nw R_s1 + (L_nw + L_s1) R_T]
#     c = −i_b1 [L_T R_s1, L_nw R_s1] / den
#
# With R_nw, R_T held at their step-midpoint values the system is linear, so a
# step is exact: x⁺ = x* + e^{A dt}(x − x*), x* = −A⁻¹c, with the closed-form
# 2×2 exponential. (A's eigenvalues are real over most of the range but become
# a complex pair when R_nw ≈ R_T ≈ R_norm, so both branches are needed. The
# fastest mode is ≈15 ps: dt ≤ 5 ps resolves the switching ramps.)
#
# What is carried over from the reference AS IS, including its simplifications:
# - nTron switching is a prescribed resistance pulse (linear rise/hold/fall),
#   not the electrothermal hotspot model (that lives in transponder_python/
#   three_transponder.py, the LTspice replica, and is not ported).
# - The input-gate current only TRIGGERS Q₁; it is not injected into the loop
#   equations (main.tex: "the simple model above only injects current into the
#   leaky current loop").
# - Refractory (both nTrons) is the reference's single `ref_period`; a pulse
#   that outlasts it is cut off when the nTron re-arms, as in the reference.
# - The output pulse is the reference's OutputPulse, σ((s−t_lead)/μ)·e^{−(s−t_lead)/τ},
#   with amplitude i_b2·split divided by the fan-out (the reservoir's
#   `fanout = :split`). The reference places a pulse's half-rise point at
#   click + line delay; here it is at click + t_lead + D, so a reference delay
#   d maps to D = d − t_lead.
#
# Two pulses from the same cell overlap only after its refractory period, by
# which time the sigmoid has saturated (σ(10) = 1 − 5·10⁻⁵ at the defaults), so
# the older pulse is carried on as a pure exponential `tail`. That is exact to
# that factor and keeps the state fixed-size.

"""
    NTronTransponder(; Ib1 = 16, I1_sw = 20, Ib2 = 16, I2_sw = 20, split = 0.1,
                       Ls1_nH = 5, Rs1 = 2, LT_nH = 150, RT = 3, Lnw_nH = 10, Rnorm = 1000,
                       t_ref_ns = 3, in_pulse_ns = (1, 1, 0.5), out_pulse_ns = (1, 2, 0.5),
                       tau_out_ns = 1, mu_out_ns = 0.2, t_lead_ns = 1)

Two-nTron leaky current-loop transponder — the spcnc reference device
(gitlab.com/spcnc/transponder_model). Defaults are `toy_example.py`'s.
Currents in µA, resistances in Ω; inductances and times are taken in nH / ns
for readability and stored in the device system (pH, ps).

# Keyword arguments
- `Ib1`, `I1_sw` — input nTron bias and switching current. It fires when
  `i_nw + i_gate > I1_sw`, so at rest it needs `I1_sw − Ib1` of gate current.
- `Ib2`, `I2_sw` — output nTron bias and switching current. It fires when
  `i_loop + Ib2 > I2_sw`.
- `split` — fraction of `Ib2` the output pulse delivers to its lines in total
  (the rest goes into the output shunt). Peak per line is `Ib2·split·G`, with
  `G = 1/fan-out` under `fanout = :split`.
- `Ls1_nH`, `Rs1` — input shunt; `LT_nH`, `RT` — the storage loop
  (leak `τ ≈ LT/RT`); `Lnw_nH` — input nTron channel inductance; `Rnorm` — an
  nTron's normal-state resistance.
- `t_ref_ns` — refractory period of each nTron.
- `in_pulse_ns`, `out_pulse_ns` — `(rise, hold, fall)` of the input nTron's
  channel resistance and the output nTron's loop resistance when they fire.
- `tau_out_ns`, `mu_out_ns`, `t_lead_ns` — output pulse shape (decay, edge
  width, half-rise time after the click).
"""
struct NTronTransponder <: AbstractDeviceCell
    Ib1::Float32; I1_sw::Float32; Ib2::Float32; I2_sw::Float32; split::Float32
    Ls1::Float32; Rs1::Float32; LT::Float32; RT::Float32; Lnw::Float32; Rnorm::Float32
    t_ref::Float32
    rise_in::Float32; hold_in::Float32; fall_in::Float32
    rise_out::Float32; hold_out::Float32; fall_out::Float32
    tau_out::Float32; mu_out::Float32; t_lead::Float32
end

function NTronTransponder(; Ib1::Real = 16, I1_sw::Real = 20, Ib2::Real = 16, I2_sw::Real = 20,
                          split::Real = 0.1, Ls1_nH::Real = 5, Rs1::Real = 2, LT_nH::Real = 150,
                          RT::Real = 3, Lnw_nH::Real = 10, Rnorm::Real = 1000, t_ref_ns::Real = 3,
                          in_pulse_ns = (1, 1, 0.5), out_pulse_ns = (1, 2, 0.5),
                          tau_out_ns::Real = 1, mu_out_ns::Real = 0.2, t_lead_ns::Real = 1)
    Rs1 > 0 && RT > 0 || throw(ArgumentError("Rs1 and RT must be > 0 (the loop must leak)"))
    Ls1_nH > 0 && LT_nH > 0 && Lnw_nH > 0 || throw(ArgumentError("inductances must be > 0"))
    tau_out_ns > 0 && mu_out_ns > 0 || throw(ArgumentError("tau_out_ns, mu_out_ns must be > 0"))
    length(in_pulse_ns) == 3 && length(out_pulse_ns) == 3 ||
        throw(ArgumentError("in_pulse_ns / out_pulse_ns are (rise, hold, fall)"))
    all(>(0), (in_pulse_ns[1], in_pulse_ns[3], out_pulse_ns[1], out_pulse_ns[3])) ||
        throw(ArgumentError("pulse rise and fall times must be > 0"))
    ns, nH = 1000, 1000                                   # → ps, pH
    return NTronTransponder(Ib1, I1_sw, Ib2, I2_sw, split,
                            Ls1_nH * nH, Rs1, LT_nH * nH, RT, Lnw_nH * nH, Rnorm, t_ref_ns * ns,
                            (in_pulse_ns .* ns)..., (out_pulse_ns .* ns)...,
                            tau_out_ns * ns, mu_out_ns * ns, t_lead_ns * ns)
end

function Base.show(io::IO, c::NTronTransponder)
    print(io, "NTronTransponder(Ib1/I1sw=", c.Ib1, "/", c.I1_sw, " µA, Ib2/I2sw=", c.Ib2, "/",
          c.I2_sw, " µA, τ_loop≈", round(loop_time_constant(c) / 1000; sigdigits = 3),
          " ns, t_ref=", c.t_ref / 1000, " ns)")
end

"""
    loop_time_constant(cell::NTronTransponder) -> ps

Slow eigenmode of the quiescent loop (`R_nw = 0`): how long stored loop
current — the integrated input — persists. ≈ `L_T/R_T` (54 ns at the defaults).
"""
function loop_time_constant(c::NTronTransponder)
    (a, b, cc, d), _ = _ntron_system(Float64.((c.Lnw, c.Ls1, c.LT, c.Rs1, 0, c.RT, c.Ib1))...)
    q = ((a - d) / 2)^2 + b * cc
    return -1 / ((a + d) / 2 + (q > 0 ? sqrt(q) : 0.0))   # the eigenvalue nearest zero
end

# A (row-major tuple a11, a12, a21, a22) and c of the reference equations.
@inline function _ntron_system(Lnw, Ls1, LT, Rs1, Rnw, RT, i1)
    den = Lnw * (Ls1 - LT) - Ls1 * LT
    a11 = (LT * (Rnw + Rs1) - Ls1 * Rnw) / den
    a12 = (LT * Rs1 + Ls1 * RT) / den
    a21 = (Lnw * Rs1 - Ls1 * Rnw) / den
    a22 = (Lnw * Rs1 + (Lnw + Ls1) * RT) / den
    c1 = -i1 * LT * Rs1 / den
    c2 = -i1 * Lnw * Rs1 / den
    return (a11, a12, a21, a22), (c1, c2)
end

# Exact step of ẋ = A x + c over dt: x⁺ = x* + e^{A dt}(x − x*), x* = −A⁻¹c.
# e^{A dt} = e^{m dt}[C·I + S·(A − m I)], m = tr/2, q = ((a−d)/2)² + bc:
# q > 0 → C = cosh(r dt), S = sinh(r dt)/r; q < 0 → cos / sin; q → 0 → S → dt.
@inline function _ntron_loop_step(x1, x2, Lnw, Ls1, LT, Rs1, Rnw, RT, i1, dt)
    (a, b, c, d), (c1, c2) = _ntron_system(Lnw, Ls1, LT, Rs1, Rnw, RT, i1)
    det = a * d - b * c
    xs1 = -(d * c1 - b * c2) / det
    xs2 = -(-c * c1 + a * c2) / det
    m = (a + d) / 2
    h = (a - d) / 2
    q = h * h + b * c
    r = sqrt(abs(q))
    rt = r * dt
    C = ifelse(q >= 0, cosh(rt), cos(rt))
    S = ifelse(rt < 1f-4, dt * (1 + ifelse(q >= 0, 1, -1) * rt * rt / 6),
               ifelse(q >= 0, sinh(rt), sin(rt)) / max(r, 1f-30))
    em = exp(m * dt)
    e11 = em * (C + S * h); e12 = em * S * b
    e21 = em * S * c;       e22 = em * (C - S * h)
    y1, y2 = x1 - xs1, x2 - xs2
    return xs1 + e11 * y1 + e12 * y2, xs2 + e21 * y1 + e22 * y2
end

# Reference LinearPulse: 0 → 1 over `rise`, hold, 1 → 0 over `fall`.
@inline _lin_pulse(s, rise, hold, fall) =
    ifelse(s <= 0, 0f0, ifelse(s < rise, s / rise, ifelse(s <= rise + hold, 1f0,
           ifelse(s < rise + hold + fall, (rise + hold + fall - s) / fall, 0f0))))

# Reference OutputPulse shape, unit peak scale, as a function of time since the click.
@inline function _ntron_out_shape(s, tau, mu, lead)
    u = s - lead
    return ifelse(s < 0, 0f0, exp(-u / tau) / (1 + exp(-u / mu)))
end

"""
    cell_state(cell::NTronTransponder, dims...) -> (; inw, iloop, age_in, age_out, tail, out)
    cell_state(cells::AbstractArray{NTronTransponder})

Rest state: `inw = Ib1` (all bias in the input channel), empty loop. `age_in`,
`age_out` are the times since each nTron last fired (ps; start "long ago"),
`tail` the decaying remainder of earlier output pulses, `out` the emitted
current. The array form takes each cell's own `Ib1`.
"""
function cell_state(c::NTronTransponder, dims::Integer...)
    z() = zeros(Float32, dims...)
    return (; inw = fill(c.Ib1, dims...), iloop = z(), age_in = fill(_NT_LONG_AGO, dims...),
            age_out = fill(_NT_LONG_AGO, dims...), tail = z(), out = z())
end
function cell_state(cells::AbstractArray{NTronTransponder})
    s = cell_state(first(cells), size(cells)...)
    return merge(s, (; inw = Float32[c.Ib1 for c in cells]))
end

const _NT_LONG_AGO = 1f9                 # ps; "never fired" (output shape is exactly 0 there)

cell_output(::NTronTransponder, s) = s.out

"""
    NTronStepper(cell, dt)
    NTronStepper(cells::AbstractArray{NTronTransponder}, dt)

Per-step constants for `cell_step` on [`NTronTransponder`](@ref)s: the cell's
parameters, as scalars or (for a heterogeneous array) arrays of the array's
shape. Unlike `RFStepper` nothing is precomputed — the loop's propagator
depends on the instantaneous nTron resistances, so it is formed each step.
"""
struct NTronStepper{T}
    dt::Float32
    Ib1::T; I1_sw::T; Ib2::T; I2_sw::T; amp::T
    Ls1::T; Rs1::T; LT::T; RT::T; Lnw::T; Rnorm::T
    t_ref::T
    rise_in::T; hold_in::T; fall_in::T
    rise_out::T; hold_out::T; fall_out::T
    tau_out::T; mu_out::T; t_lead::T
    e_out::T                                        # e^{−dt/τ_out}
end

_ntron_constants(c::NTronTransponder, dt) =
    (c.Ib1, c.I1_sw, c.Ib2, c.I2_sw, c.Ib2 * c.split, c.Ls1, c.Rs1, c.LT, c.RT, c.Lnw, c.Rnorm,
     c.t_ref, c.rise_in, c.hold_in, c.fall_in, c.rise_out, c.hold_out, c.fall_out,
     c.tau_out, c.mu_out, c.t_lead, Float32(exp(-dt / c.tau_out)))

function NTronStepper(c::NTronTransponder, dt::Real)
    dt > 0 || throw(ArgumentError("dt must be > 0"))
    return NTronStepper{Float32}(Float32(dt), _ntron_constants(c, dt)...)
end

function NTronStepper(cells::AbstractArray{NTronTransponder}, dt::Real)
    dt > 0 || throw(ArgumentError("dt must be > 0"))
    k = [_ntron_constants(c, dt) for c in cells]
    field(j) = Float32[t[j] for t in k]
    return NTronStepper{Array{Float32, ndims(cells)}}(Float32(dt), (field(j) for j in 1:22)...)
end

cell_stepper(c::NTronTransponder, dt::Real) = NTronStepper(c, dt)
cell_stepper(c::AbstractArray{NTronTransponder}, dt::Real) = NTronStepper(c, dt)

function stepper_to(stp::NTronStepper, device)
    stp.Ib1 isa Number && return stp
    moved = map(device, [getfield(stp, n) for n in fieldnames(NTronStepper)][2:end])
    return NTronStepper{typeof(moved[1])}(stp.dt, moved...)
end

"""
    cell_step(stp::NTronStepper, s, I; force = false, t_force = 0) -> (s_next, fire, t_fire)

Advance [`NTronTransponder`](@ref)s one step. `I` is the current on each input
nTron's gate (µA, held across the step). `fire` / `t_fire` report OUTPUT clicks
— emitted spikes — with sub-step crossing times; input clicks show up as
`age_in` resetting.

1. The nTron resistances are evaluated at the step midpoint (R_nw from the
   input nTron's pulse, R_T + the output nTron's pulse), each only while that
   nTron is refractory, and the loop currents are stepped exactly.
2. Input click: an armed input nTron whose `i_nw + I` passed `I1_sw`.
3. Output click: an armed output nTron whose `i_loop + Ib2` passed `I2_sw`, or
   any cell in `force` (a detector-driven sensor, firing at `t_force`).
4. On an output click the previous pulse joins the exponential `tail` and a new
   pulse starts from the crossing time.
"""
function cell_step(stp::NTronStepper, s, I; force = false, t_force = 0f0)
    dt = stp.dt
    hdt = dt / 2
    refr_in  = s.age_in  .< stp.t_ref
    refr_out = s.age_out .< stp.t_ref
    Rnw = ifelse.(refr_in, stp.Rnorm .* _lin_pulse.(s.age_in .+ hdt, stp.rise_in, stp.hold_in, stp.fall_in), 0f0)
    RTt = stp.RT .+ ifelse.(refr_out, stp.Rnorm .* _lin_pulse.(s.age_out .+ hdt, stp.rise_out,
                                                             stp.hold_out, stp.fall_out), 0f0)
    x = _ntron_loop_step.(s.inw, s.iloop, stp.Lnw, stp.Ls1, stp.LT, stp.Rs1, Rnw, RTt, stp.Ib1, dt)
    inw1 = first.(x)
    iloop1 = last.(x)
    age_in1  = min.(s.age_in  .+ dt, _NT_LONG_AGO)
    age_out1 = min.(s.age_out .+ dt, _NT_LONG_AGO)

    # input click (resets the input nTron's clock; drives the loop from next step)
    g0, g1 = s.inw .+ I, inw1 .+ I
    fire_in = .!refr_in .& (g1 .> stp.I1_sw)
    fin = clamp.((stp.I1_sw .- g0) ./ max.(g1 .- g0, 1f-12), 0f0, 1f0)
    age_in2 = ifelse.(fire_in, dt .- fin .* dt, age_in1)

    # output click
    h0, h1 = s.iloop .+ stp.Ib2, iloop1 .+ stp.Ib2
    fire_th = .!refr_out .& (h1 .> stp.I2_sw)
    fire = fire_th .| force
    fout = clamp.((stp.I2_sw .- h0) ./ max.(h1 .- h0, 1f-12), 0f0, 1f0)
    t_fire = ifelse.(fire_th, fout .* dt, ifelse.(fire, t_force, 0f0))
    since = dt .- t_fire

    old = stp.amp .* _ntron_out_shape.(age_out1, stp.tau_out, stp.mu_out, stp.t_lead)
    tail = s.tail .* stp.e_out .+ ifelse.(fire, old, 0f0)
    age_out2 = ifelse.(fire, since, age_out1)
    out = tail .+ stp.amp .* _ntron_out_shape.(age_out2, stp.tau_out, stp.mu_out, stp.t_lead)

    return (; inw = inw1, iloop = iloop1, age_in = age_in2, age_out = age_out2, tail, out),
           fire, ifelse.(fire, t_fire, 0f0)
end
