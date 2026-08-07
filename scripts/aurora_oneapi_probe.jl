#!/usr/bin/env julia
# aurora_oneapi_probe.jl
#
# Fail-fast capability probe for running the CONTINUOUS-TIME (`:current` mode)
# phasor forward on an Aurora PVC tile via oneAPI.jl.  Every stage is wrapped so
# the whole report prints in one node session even when an early stage fails.
#
# Run on a PVC compute node (after `module load frameworks` and instantiating the
# oneAPI-enabled project):
#     julia --project=. scripts/aurora_oneapi_probe.jl
#
# Decision gate (see the Aurora scope):
#   * Stage 4 (Tsit5 on oneArray) is make-or-break for Path A (Julia-on-PVC).
#     PASS  -> Path A viable; build the batched full-scale Aurora driver.
#     FAIL  -> fall back to Path B (fixed-step RK4 in phasor_torch on XPU).
#   * Stage 7 (FFT) is informational only — the continuous path is ODE-only and
#     does NOT need it.  It gates the DISCRETE path, not this test.

using Printf

const RESULTS = Tuple{String,Bool,String}[]

function stage(name::String, critical::Bool, f::Function)
    tag = critical ? "[CRITICAL]" : "[info]    "
    print(@sprintf("%-11s %-42s ... ", tag, name))
    try
        f()
        println("PASS")
        push!(RESULTS, (name, true, ""))
    catch err
        println("FAIL")
        msg = sprint(showerror, err)
        # keep the report readable; full trace still on stderr
        @error "stage failed" name exception=(err, catch_backtrace())
        push!(RESULTS, (name, false, first(split(msg, '\n'))))
    end
end

# ---------------------------------------------------------------------------
println("="^72)
println("Aurora oneAPI capability probe — continuous-time (:current) forward")
println("="^72)

using oneAPI
using PhasorNetworks
using OrdinaryDiffEq: ODEProblem, solve, Tsit5
using NNlib: batched_mul
using KernelAbstractions
using AbstractFFTs: fft          # dispatches to oneMKL FFT on oneArray (Stage 7)

# Inference-only solver args with the MEMORY-SAFE overrides baked in
# (save_everystep=false + saveat).  The library default retains full dense
# trajectories — that is what locked the Spark; never omit these on a real run.
const T = 1.0f0
solver_args(saveat) = Dict(
    :dt => 0.1f0, :adaptive => false, :save_start => true,
    :save_everystep => false, :saveat => saveat,
)

# ---- Stage 0: device functional ------------------------------------------
stage("0. oneAPI.functional()", true) do
    @assert oneAPI.functional() "oneAPI reports not functional on this node"
    dev = PhasorNetworks.select_device(:oneapi)
    println()
    oneAPI.versioninfo()
    print("             device selector -> $(typeof(dev))   ")
end

# ---- Stage 1: complex64 alloc + broadcast (the ODE RHS elementwise) -------
stage("1. oneArray{ComplexF32} broadcast", true) do
    uh = rand(ComplexF32, 256, 64); kh = rand(ComplexF32, 256); Ih = rand(ComplexF32, 256, 64)
    r = oneArray(kh) .* oneArray(uh) .+ oneArray(Ih)   # the dz/dt elementwise core
    @assert eltype(r) == ComplexF32
    @assert isapprox(Array(r), kh .* uh .+ Ih; rtol = 1f-4)
end

# ---- Stage 2: batched ComplexF32 GEMM (oneMKL ext path) ------------------
stage("2. batched_mul ComplexF32 (oneMKL)", true) do
    W = oneArray(rand(ComplexF32, 256, 128, 8))   # (out,in,B)
    x = oneArray(rand(ComplexF32, 128, 4, 8))     # (in,L,B)
    y = batched_mul(W, x)                          # -> (out,L,B)
    @assert size(y) == (256, 4, 8)
    KernelAbstractions.synchronize(get_backend(y))
end

# ---- Stage 3: a real KA kernel from the package on device ----------------
stage("3. KA kernel (potential_to_phase)", true) do
    pot = oneArray(rand(ComplexF32, 32, 16))
    ts  = collect(0.0f0:T:(4*T))
    ph  = PhasorNetworks.potential_to_phase(pot, ts; spk_args = SpikingArgs(t_period = T))
    KernelAbstractions.synchronize(get_backend(pot))
    @assert ph !== nothing
end

# ---- Stage 4: Tsit5 ODE solve on oneArray  *** MAKE OR BREAK *** ----------
# dz/dt = k*z + W*I(t), state (out, B), driven by a ZOH current — the exact
# math every continuous layer runs, minimal form (no layer construction).
stage("4. Tsit5 solve on oneArray", true) do
    out, inp, B, L = 64, 32, 8, 10
    ω = Float32(2π / T)
    W    = oneArray(rand(ComplexF32, out, inp))
    Idr  = oneArray(rand(ComplexF32, inp, B))                       # constant ZOH drive
    kvec = oneArray(ComplexF32.(-rand(Float32, out) .+ im * ω))     # per-channel k = λ + iω
    u0   = oneAPI.zeros(ComplexF32, out, B)
    f(u, p, t) = kvec .* u .+ W * Idr
    prob = ODEProblem(f, u0, (0.0f0, Float32(L) * T))
    saveat = collect(T:T:(Float32(L) * T))
    sol = solve(prob, Tsit5(); solver_args(saveat)...)
    yT = Array(sol(Float32(L) * T))
    @assert size(yT) == (out, B)
    @assert all(isfinite, yT)
end

# ---- Stage 5: similarity_outer on oneArray (attention inner product) ------
stage("5. similarity_outer on oneArray", true) do
    A = oneArray(rand(ComplexF32, 64, 4, 8))
    Bm = oneArray(rand(ComplexF32, 64, 4, 8))
    s = PhasorNetworks.similarity_outer(A, Bm; dims = 1)
    KernelAbstractions.synchronize(get_backend(A))
    @assert s !== nothing
end

# ---- Stage 6: oscillator_bank(u0, dzdt) — the package solver entry --------
# Same call every continuous layer funnels through; confirms the library's own
# wrapper (not just raw OrdinaryDiffEq) runs on device with saveat.
stage("6. oscillator_bank(u0, dzdt) device", true) do
    out, B = 64, 8
    kk = ComplexF32(-0.2f0 + 1im * Float32(2π / T))
    u0 = oneAPI.zeros(ComplexF32, out, B)
    drive = oneArray(rand(ComplexF32, out, B))
    dzdt(u, p, t) = kk .* u .+ drive
    spk = SpikingArgs(t_period = T, solver_args = solver_args(collect(T:T:10T)))
    sol = PhasorNetworks.oscillator_bank(u0, dzdt; tspan = (0.0f0, 10.0f0), spk_args = spk)
    @assert all(isfinite, Array(sol(10.0f0)))
end

# ---- Stage 7: FFT on oneArray (DISCRETE path only — informational) -------
stage("7. fft on oneArray (discrete-only)", false) do
    x = oneArray(rand(ComplexF32, 512, 8))
    y = fft(x, 1)
    @assert size(y) == size(x)
end

# ---------------------------------------------------------------------------
println("\n" * "="^72)
println("SUMMARY")
println("="^72)
crit_ok = true
for (name, ok, msg) in RESULTS
    println(@sprintf("  %-4s %s%s", ok ? "PASS" : "FAIL", name, ok ? "" : "   -> $msg"))
end
# Stage 4 is the gate
gate = findfirst(r -> startswith(r[1], "4."), RESULTS)
if gate !== nothing && RESULTS[gate][2]
    println("\n>>> Stage 4 PASS: Tsit5-on-oneArray works. PATH A (Julia-on-PVC) VIABLE.")
    println("    Next: batched full-scale continuous driver + PBS submit script.")
else
    println("\n>>> Stage 4 FAIL: fall back to PATH B (fixed-step RK4 in phasor_torch/XPU).")
end
