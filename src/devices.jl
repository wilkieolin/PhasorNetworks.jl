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

Precomputed per-step constants for [`cell_step`](@ref): the exact tank
propagator and the output-filter decay. Build once per (cell, dt).
"""
struct RFStepper
    cell::RFTransponder
    dt::Float32
    Φ::Matrix{Float32}
    Γ::Vector{Float32}
    e_out::Float32          # e^{−dt/τ_out}
    kick::Float32           # alpha-pulse amplitude: I_out·e
end

"""
    cell_stepper(cell, dt)

Per-step constants for `cell_step` — the hook a sheet uses to stay agnostic of
the cell type. For an `RFTransponder` this is an [`RFStepper`](@ref).
"""
cell_stepper(c::RFTransponder, dt::Real) = RFStepper(c, dt)

function RFStepper(c::RFTransponder, dt::Real)
    dt > 0 || throw(ArgumentError("dt must be > 0"))
    Φ, Γ = _tank_propagator(c, dt)
    return RFStepper(c, Float32(dt), Φ, Γ, Float32(exp(-dt / c.tau_out)), Float32(c.I_out * ℯ))
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

Pure: returns a new state, never mutates `s`. Broadcast-only, so it runs
unchanged on GPU arrays.
"""
function cell_step(stp::RFStepper, s, I)
    c, dt, Φ, Γ = stp.cell, stp.dt, stp.Φ, stp.Γ
    live = s.ref .<= 0f0
    v1 = Φ[1, 1] .* s.v .+ Φ[1, 2] .* s.i .+ Γ[1] .* I
    i1 = Φ[2, 1] .* s.v .+ Φ[2, 2] .* s.i .+ Γ[2] .* I
    v1 = ifelse.(live, v1, 0f0)
    i1 = ifelse.(live, i1, 0f0)

    a0, a1 = abs.(s.i), abs.(i1)
    fire = live .& (a1 .>= c.I_th)
    # fraction of the step at which |i| crossed I_th (a0 < I_th ≤ a1 on a fire)
    frac = clamp.((c.I_th .- a0) ./ max.(a1 .- a0, 1f-12), 0f0, 1f0)
    t_fire = ifelse.(fire, frac .* dt, 0f0)
    t_since = dt .- t_fire                                   # time since the crossing

    v2   = ifelse.(fire, 0f0, v1)
    i2   = ifelse.(fire, 0f0, i1)
    ref2 = ifelse.(fire, c.t_ref .- t_since, max.(s.ref .- dt, 0f0))

    # Output alpha filter: o1' = −o1/τ, o2' = (o1 − o2)/τ, exact over the step,
    # plus an impulse at the crossing propagated forward by `t_since`.
    τ  = c.tau_out
    o2 = stp.e_out .* (s.o2 .+ (dt / τ) .* s.o1)
    o1 = stp.e_out .* s.o1
    ek = exp.(.-t_since ./ τ)
    o1 = o1 .+ ifelse.(fire, stp.kick .* ek, 0f0)
    o2 = o2 .+ ifelse.(fire, stp.kick .* (t_since ./ τ) .* ek, 0f0)

    return (; v = v2, i = i2, ref = ref2, o1, o2), fire, t_fire
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
