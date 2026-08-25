#!/usr/bin/env julia
#
# scripts/ep_rotating_gates.jl — blocking correctness gates for rotating
# (resonate-and-fire) phasor EP.
#
# WHY THIS FILE EXISTS
# --------------------
# An earlier attempt at a "rotating frame" EP extension produced a full
# 120-cell sweep (results/ep_adiabatic_rot/) in which every cell failed,
# and the failure was reported as physics. It was not: the rotating settle
# was a no-op, the demodulator was tuned to a band with no signal, the grid
# never read its own sweep variables, and the reference was a free-phase
# Hebbian rather than a gradient.
#
# Every one of those would have been caught by one of the four gates below,
# each of which costs seconds. So: no rotating sweep runs until these pass.
#
# THE CLAIM BEING GATED
# ---------------------
# For a chain with a single shared carrier ω, the rotating problem IS the
# static problem. Φ = Σ_l Re⟨W_l z_{l-1}, z_l⟩ is U(1)-invariant and the
# hard projection is U(1)-equivariant, so under z_l = w_l·e^{iωt} the
# carrier cancels identically — exactly, at any dt, not adiabatically.
# The only symmetry-breaking terms are the bias and the cost target, and
# in this package both physically co-rotate (see `phasor_settle`'s
# docstring). So the correct rotating extension must reproduce the
# co-rotating result to machine precision, and a sweep in the lab frame
# buys nothing.
#
#   julia --project=. scripts/ep_rotating_gates.jl
#
# Exit code is nonzero if any gate fails.

using PhasorNetworks, Lux, LinearAlgebra, Printf
using Random: Xoshiro

const TWO_PI = Float32(2π)

# ---------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------
const FAILURES = String[]

function gate(name::String, ok::Bool, detail::String)
    @printf("  %-4s %-52s %s\n", ok ? "PASS" : "FAIL", name, detail)
    ok || push!(FAILURES, name)
    return ok
end

section(t) = (@printf("\n%s\n%s\n", t, "-"^length(t)))

# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------
# Toy width, chosen so `fd_gradient_phasor` (n_params + 1 settles) is
# affordable: 4→8→2 with bias is 4*8+8 + 8*2+2*2 = 66 parameters.
function toy_chain(seed::Int = 42; t_period = 1.0f0, scale = 0.4f0)
    chain = Chain(
        PhasorDense(4 => 8, normalize_to_unit_circle, use_bias=true,
                    spk_args=SpikingArgs(t_period=t_period)),
        PhasorDense(8 => 2, normalize_to_unit_circle, use_bias=true,
                    spk_args=SpikingArgs(t_period=t_period)),
    )
    ps, st = Lux.setup(Xoshiro(seed), chain)
    # Downscale as test/test_ep.jl does: the default glorot is wide enough
    # that some initial drives have small magnitude during settling, and a
    # near-zero drive lands on `_project_damp`'s discontinuity (|g| < th
    # snaps to 1+0im) rather than on the dynamics these gates are testing.
    ps = (layer_1 = merge(ps.layer_1, (weight = scale .* ps.layer_1.weight,)),
          layer_2 = merge(ps.layer_2, (weight = scale .* ps.layer_2.weight,)))
    return chain, ps, st
end

# B = 1 yields a plain `(d,)` vector — the original single-sample
# interface — rather than a `(d,1)` matrix, so gates that compare against
# the proven single-sample FD path exercise exactly that code path.
function toy_input(seed::Int = 7, B::Int = 3)
    rng = Xoshiro(seed)
    v = 2f0 .* rand(rng, Float32, 4, B) .- 1f0
    return Phase.(B == 1 ? vec(v) : v)
end

function toy_cost(seed::Int = 11, B::Int = 3)
    rng = Xoshiro(seed)
    y = ComplexF32.(exp.(im .* π .* (2f0 .* rand(rng, Float32, 2, B) .- 1f0)))
    return SimilarityCost(B == 1 ? vec(y) : y)
end

cosine(a, b) = dot(vec(a), vec(b)) / (norm(vec(a)) * norm(vec(b)) + 1e-30)
relerr(a, b) = norm(vec(a) .- vec(b)) / (norm(vec(b)) + 1e-30)

# Demodulate lab-frame states back to the co-rotating frame at time `t`.
# Same Float64 argument reduction as `_phasor_step`, so the gate measures
# the frames rather than its own rounding.
_carrier_phase(ω, t) = ComplexF32(cis(mod(Float64(ω) * Float64(t), 2π)))
demod(states, ω, t) = [z .* conj(_carrier_phase(ω, t)) for z in states]

# ---------------------------------------------------------------------
# Gate A (Stage -1) — the corrected self-force still matches true FD.
# ---------------------------------------------------------------------
# `ep_self_force` used to return ½(λ + iω)z. It now returns ½λz, because
# Re⟨z,(λ+iω)z⟩ = λ|z|²: the rotation contributes nothing to the energy.
# FD differentiates the loss at the settle's own fixed point, so it is the
# oracle for whether force and energy now agree. This also exercises the
# concrete win: `:stored` at the layer's DEFAULT ω = 2π and dt = 0.5,
# which the old additive-iω force could not settle at all.
function gate_fd()
    section("Gate A (Stage -1): StaticEP vs fd_gradient_phasor at toy width")
    chain, ps, st = toy_chain()
    x    = toy_input(7, 1)          # single sample: see note above
    cost = toy_cost(11, 1)

    # A forward difference at ε on an O(1) Float32 loss carries roundoff
    # ~eps/ε in the gradient: 6e-3 absolute at the default ε=1e-5. Where a
    # parameter's true gradient is itself small, that is a large RELATIVE
    # error — in the oracle, not in EP. So FD is evaluated at two ε and the
    # spread between them is reported as the oracle's own noise floor; a
    # disagreement no bigger than that floor is not evidence against EP.
    for K_mode in (:zero, :stored)
        fd_a = fd_gradient_phasor(chain, ps, st, x, cost;
                                  ε=1f-5, T=200, dt=0.5f0, K_mode=K_mode)
        fd_b = fd_gradient_phasor(chain, ps, st, x, cost;
                                  ε=1f-4, T=200, dt=0.5f0, K_mode=K_mode)
        g, _ = ep_gradient(StaticEP(β=0.001f0, T_free=200, T_nudge=100,
                                    dt=0.5f0, K_mode=K_mode, centered=true),
                           chain, ps, st, x, cost)
        for key in (:layer_1, :layer_2), pname in (:weight, :bias_real, :bias_imag)
            floor_ = relerr(fd_a[key][pname], fd_b[key][pname])   # FD vs itself
            re     = relerr(g[key][pname], fd_b[key][pname])
            ok     = re < max(0.05, 2 * floor_)
            gate("A  K_mode=:$K_mode $key.$pname", ok,
                 @sprintf("rel-err=%.4g (FD noise floor %.4g)", re, floor_))
        end
    end
end

# ---------------------------------------------------------------------
# Gate B — ω → 0 continuity.
# ---------------------------------------------------------------------
# carrier=0 must reproduce the co-rotating path EXACTLY. cis(0) is exactly
# 1+0im, so this is a bit-for-bit assertion, not a tolerance. It is the
# cheapest possible check that the lab-frame branch has not silently
# changed the dynamics — and it is the check whose absence let a no-op
# "rotating frame" pass for a working one.
function gate_zero_carrier()
    section("Gate B (Stage 0): carrier=0 reproduces the co-rotating settle")
    chain, ps, st = toy_chain()
    x    = toy_input()
    cost = toy_cost()

    for K_mode in (:zero, :stored), β in (0f0, 0.05f0)
        s_co  = phasor_settle(chain, ps, st, x, cost, β;
                              T=120, dt=0.5f0, K_mode=K_mode)
        s_lab = phasor_settle(chain, ps, st, x, cost, β;
                              T=120, dt=0.5f0, K_mode=K_mode, carrier=0f0)
        d = maximum(maximum(abs.(a .- b)) for (a, b) in zip(s_co, s_lab))
        gate("B  K_mode=:$K_mode β=$β", d == 0,
             @sprintf("max|Δ|=%.3g (exact 0 required)", d))
    end
end

# ---------------------------------------------------------------------
# Gate C — Frame A ≡ Frame B at a real carrier.
# ---------------------------------------------------------------------
# The substantive claim. Deliberately includes dt = 0.5 at ω = 2π, where
# ω·dt = π — exactly Nyquist. A discretized rotation (additive iω·z) is
# unusable there; an exact multiplicative one has no dt limit at all, and
# that difference is the whole reason the carrier is applied as cis(ω·dt).
# Incommensurate (ω, dt, T) triples are included so the demodulation is a
# real rotation rather than an accidental multiple of 2π.
function gate_frame_equivalence()
    section("Gate C (Stage 0): lab frame ≡ co-rotating frame")
    chain, ps, st = toy_chain()
    x    = toy_input()
    cost = toy_cost()

    cases = [(TWO_PI, 0.5f0,  120, "ω=2π  dt=0.5 (ω·dt=π, Nyquist)"),
             (TWO_PI, 0.37f0, 137, "ω=2π  dt=0.37 (incommensurate)"),
             (1.7f0,  0.5f0,  113, "ω=1.7 dt=0.5 (incommensurate)"),
             (TWO_PI, 0.05f0, 400, "ω=2π  dt=0.05 (well-resolved)")]

    # The lab frame applies `rot` once per step, so its state accumulates
    # ~T·eps of Float32 rounding that the co-rotating frame never incurs.
    # The tolerance therefore scales with T. That the residual tracks T
    # (rather than ω, dt, or ω·dt) is itself the evidence that it is
    # rounding and not a frame error — and it is a standing practical
    # argument for doing the real work in the co-rotating frame.
    for (ω, dt, T, label) in cases, K_mode in (:zero, :stored)
        β = 0.05f0
        s_co  = phasor_settle(chain, ps, st, x, cost, β;
                              T=T, dt=dt, K_mode=K_mode)
        s_lab = phasor_settle(chain, ps, st, x, cost, β;
                              T=T, dt=dt, K_mode=K_mode, carrier=ω)
        s_dm  = demod(s_lab, ω, Float32(T * dt))
        d   = maximum(relerr(a, b) for (a, b) in zip(s_dm, s_co))
        tol = max(1e-6, 1e-6 * T)          # accumulated-rounding budget
        # Direction must be exact regardless — a genuine frame error would
        # rotate the state, which shows up here long before it shows up in
        # the magnitude.
        # 1 - 1e-6, not tighter: Float32 eps is 1.19e-7, so a few ulp of
        # cosine deviation is the representational floor, not a signal.
        cs  = minimum(real(cosine(a, b)) for (a, b) in zip(s_dm, s_co))
        gate("C  $label K=:$K_mode", d < tol && cs > 1 - 1e-6,
             @sprintf("rel-err=%.3g (tol %.1g)  cos=%.9f", d, tol, cs))
    end
end

# ---------------------------------------------------------------------
# Gate D — U(1) invariance of the Hebbians.
# ---------------------------------------------------------------------
# `ep_hebbian` uses real(z_self·z_in') — an ADJOINT. In the co-rotating
# frame that is cos(π(θ_self − θ_in)): a relative phase, which is what a
# spiking substrate can measure from pre/post spike-time differences, and
# which is invariant under a global rotation.
#
# Using `transpose` instead gives cos(π(θ_self + θ_in)), which oscillates
# at 2ω under a global rotation and is not a relative-phase quantity at
# all. That substitution is exactly what the earlier prototype's reference
# did (via `unrotate_solution`, which returns conj(w)).
#
# So this gate asserts the adjoint form is invariant AND that the
# transpose form is not — the second half proves the gate has teeth
# rather than passing vacuously.
function gate_u1_invariance()
    section("Gate D (Stage 0): Hebbians are U(1)-invariant")
    chain, ps, st = toy_chain()
    x    = toy_input()
    cost = toy_cost()

    s  = phasor_settle(chain, ps, st, x, cost, 0f0; T=200, dt=0.5f0)
    z0 = ComplexF32.(angle_to_complex(x))
    h  = chain_hebbians(chain, ps, st, z0, s)

    worst_adj = 0.0
    for θ in (0.3f0, 1.1f0, 2.7f0, -0.8f0)
        u  = cis(θ)
        hθ = chain_hebbians(chain, ps, st, z0 .* u, [z .* u for z in s])
        for key in (:layer_1, :layer_2)
            worst_adj = max(worst_adj, relerr(hθ[key].weight, h[key].weight))
        end
    end
    gate("D  adjoint Hebbian invariant under global e^{iθ}",
         worst_adj < 1e-5, @sprintf("worst rel-err=%.3g", worst_adj))

    # Teeth: the transpose form must visibly BREAK under the same rotation.
    transpose_hebb(z_self, z_in) = real.(z_self * transpose(z_in))
    base = transpose_hebb(s[2], s[1])
    worst_tr = 0.0
    for θ in (0.3f0, 1.1f0, 2.7f0, -0.8f0)
        u = cis(θ)
        worst_tr = max(worst_tr,
                       relerr(transpose_hebb(s[2] .* u, s[1] .* u), base))
    end
    gate("D  transpose Hebbian DOES break (gate has teeth)",
         worst_tr > 0.1, @sprintf("worst rel-err=%.3g (want ≫0)", worst_tr))
end

# ---------------------------------------------------------------------
# Gate E — end-to-end: the lab-frame GRADIENT equals the co-rotating one.
# ---------------------------------------------------------------------
# Gates B–D are about states and Hebbians. This is the statement that
# actually matters for training: run the whole StaticEP estimator in the
# lab frame — free settle, then a nudged settle warm-started from it and
# resuming at the correct absolute time — demodulate, and compare
# gradients. `t0` threading is load-bearing here; getting it wrong shows
# up as a rotation between the two Hebbian snapshots.
function gate_gradient_equivalence()
    section("Gate E (Stage 0): lab-frame gradient ≡ co-rotating gradient")
    chain, ps, st = toy_chain()
    x    = toy_input()
    cost = toy_cost()
    z0   = ComplexF32.(angle_to_complex(x))

    β, T_free, T_nudge, dt = 0.005f0, 200, 100, 0.5f0

    for (ω, label) in ((TWO_PI, "ω=2π"), (1.7f0, "ω=1.7"))
        g_ref, _ = ep_gradient(StaticEP(β=β, T_free=T_free, T_nudge=T_nudge,
                                        dt=dt, centered=true),
                               chain, ps, st, x, cost)

        # Same estimator, driven by hand so the carrier and t0 can be threaded.
        sf = phasor_settle(chain, ps, st, x, cost, 0f0;
                           T=T_free, dt=dt, carrier=ω, t0=0f0)
        tf = Float32(T_free * dt)
        sp = phasor_settle(chain, ps, st, x, cost,  β; T=T_nudge, dt=dt,
                           init=sf, carrier=ω, t0=tf)
        sn = phasor_settle(chain, ps, st, x, cost, -β; T=T_nudge, dt=dt,
                           init=sf, carrier=ω, t0=tf)
        tn = Float32((T_free + T_nudge) * dt)

        hp = chain_hebbians(chain, ps, st, z0, demod(sp, ω, tn))
        hn = chain_hebbians(chain, ps, st, z0, demod(sn, ω, tn))
        g_lab = PhasorNetworks._ep_diff_gradient(ps, hn, hp, 2f0 * β)

        for key in (:layer_1, :layer_2)
            re = relerr(g_lab[key].weight, g_ref[key].weight)
            cs = real(cosine(g_lab[key].weight, g_ref[key].weight))
            gate("E  $label $key.weight", re < 1e-4,
                 @sprintf("rel-err=%.3g cos=%.8f", re, cs))
        end
    end
end

# ---------------------------------------------------------------------
function main()
    println("Rotating-EP correctness gates")
    println("=============================")
    gate_fd()
    gate_zero_carrier()
    gate_frame_equivalence()
    gate_u1_invariance()
    gate_gradient_equivalence()

    println()
    if isempty(FAILURES)
        println("All gates passed. A rotating sweep is now meaningful to run.")
    else
        println("FAILED GATES ($(length(FAILURES))):")
        foreach(f -> println("  - $f"), FAILURES)
        println("\nDo NOT run a sweep until these pass.")
        exit(1)
    end
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main()
