# wave_velocity_bank.jl — velocity-tuned delay banks on the phasor sheet
#
# Reads 2-D velocity paths out of a resonate-and-fire sheet without asking the
# sheet to be a wave medium — by matched filtering rather than by reading a wake.
#
# (The original justification here said the sheet is "a poor medium: v_p ~0.08
# sites/period, and Im Λ ≡ 0 in :spike". Both have since moved: v_p is 0.747 at
# the matched conduction speed evaluated at criticality, and the Im Λ ≡ 0 result
# is scoped to the saturated branch — a threshold-gated spike sheet has a full
# subthreshold band. See the header of src/velocity_bank.jl. The matched filter
# is still preferred, because it has no Mach floor, needs no criticality, and
# gives a dense per-site field — see the CURVED TRACK section below, which a
# global wake FFT cannot do.)
#
# The mechanism: ω·T = 2π exactly, so a transiting particle writes its sub-cycle
# arrival phase into the sheet and the deposited field is a plane wave whose
# wavevector IS the velocity, κ = (2π/v)·û. A kernel with a delay linear in
# displacement — free, by §3.1 — is the matched filter for that wavevector.
#
# Run:  julia --project=. demos/wave_velocity_bank.jl

using PhasorNetworks, Lux, Random, Printf, Statistics

const N   = 128
const RNG = Xoshiro(0)

speeds = Float32[2.5, 3, 4, 5, 6, 8, 10, 13]
angles = Float32.(range(0, 2π; length = 17)[1:16])

bank = PhasorVelocityBank(N, N; speeds = speeds, angles = angles)
ps, st = Lux.setup(RNG, bank)
A, g, M = velocity_coupling(bank, ps, st)
V = channel_velocities(bank, ps)

println("="^76)
println(bank)
@printf("A = %.6f (arg = %.1e — carrier folds out, ωT = 2π)\n", real(A), angle(A))
# g is PER CHANNEL — (1,1,C), not a scalar. Each channel's kernel has its own
# ΣG and hence its own closed-form critical gain (1−A)/ΣG, so one number here
# would be wrong even if it printed (it did not: this line threw).
@printf("g ∈ [%.4e, %.4e] over %d channels   spectral radius ρ = max|M| = %.6f\n",
        minimum(g), maximum(g), length(g), maximum(abs.(M)))
@printf("max|Im M| = %.2e  → M real positive: no band, no dispersion, closed-form stability\n",
        maximum(abs.(imag.(M))))
@printf("channels: %d   speed range [%.1f, %.1f] sites/period\n",
        bank.n_channels, minimum(sqrt.(sum(abs2, V; dims = 1))),
        maximum(sqrt.(sum(abs2, V; dims = 1))))

track_L(v) = clamp(round(Int, 100 / v) + 8, 12, 64)

function estimate(v, ang; curvature = 0.0, width = 0.8, noise = 0.0, seed = 1)
    L = track_L(v)
    D = moving_drive(N, N, L; v = v, angle = ang, width = width, curvature = curvature)
    if noise > 0
        rng = Xoshiro(seed)
        D = D .+ ComplexF32(noise) .* (randn(rng, ComplexF32, size(D)))
    end
    Z = velocity_bank_run(bank, ps, st, D)
    e = dropdims(sum(abs2, Z; dims = (1, 2, 4)); dims = (1, 2, 4))   # per-channel energy
    c = argmax(e)
    return sqrt(V[1, c]^2 + V[2, c]^2), atan(V[2, c], V[1, c]), L
end

wrap(x) = mod(x + π, 2π) - π

println("\n" * "="^76)
println("STRAIGHT TRACKS — argmax over the bank")
@printf("%7s %8s %5s | %7s %9s | %8s %9s\n", "v true", "ang", "L", "v hat", "ang hat", "v err", "ang err")
nok = 0; ntot = 0
for (v, a) in [(2.5, 0.0), (3.0, 0.39), (4.0, 1.18), (5.0, -0.79),
               (6.0, 2.36), (8.0, 3.53), (10.0, 0.79), (13.0, 4.71)]
    vh, ah, L = estimate(v, a)
    ae = wrap(ah - a); ve = 100 * (vh - v) / v
    global nok += (abs(ve) < 15 && abs(ae) < 0.4); global ntot += 1
    @printf("%7.1f %8.2f %5d | %7.1f %9.2f | %7.1f%% %9.3f\n", v, a, L, vh, ah, ve, ae)
end
@printf("  %d/%d within (15%% speed, 0.4 rad = 1 angular bin)\n", nok, ntot)

println("\n" * "="^76)
println("RESOLUTION vs integration length — predicted δv/v ≈ 1/(2L)")
fine = Float32[3.4, 3.7, 3.85, 4.0, 4.15, 4.3, 4.6]
fb = PhasorVelocityBank(N, N; speeds = fine, angles = Float32[1.18])
fps, fst = Lux.setup(RNG, fb)
for L in (8, 16, 32, 48)
    D = moving_drive(N, N, L; v = 4.0, angle = 1.18, width = 0.8)
    Z = velocity_bank_run(fb, fps, fst, D)
    e = dropdims(sum(abs2, Z; dims = (1, 2, 4)); dims = (1, 2, 4))
    e ./= maximum(e)
    half = fine[e .> 0.5]
    @printf("  L=%3d: peak v_c=%.2f  FWHM=[%.2f, %.2f]  δv/v=%.3f   1/(2L)=%.3f\n",
            L, fine[argmax(e)], minimum(half), maximum(half),
            (maximum(half) - minimum(half)) / 4.0, 1 / (2L))
end

println("\n" * "="^76)
println("NOISE — i.i.d. complex noise on the drive at every site, every step")
for nz in (0.0, 0.005, 0.01, 0.02, 0.03, 0.05)
    vh, ah, _ = estimate(4.0, 1.18; noise = nz)
    @printf("  noise=%-6.3f → v=%5.1f  ang=%6.2f   %s\n", nz, vh, ah,
            abs(vh - 4) < 1e-3 ? "OK" : "FAIL")
end

println("\n" * "="^76)
println("FOOTPRINT — a blunt particle low-passes its own ramp by exp(−κ²w²/2)")
for w in (0.8, 1.5, 2.5)
    vh, _, _ = estimate(3.0, 0.0; width = w)
    @printf("  width=%.1f (attenuation %.3f) → v=%.1f  %s\n",
            w, exp(-(2π / 3)^2 * w^2 / 2), vh, abs(vh - 3) < 1e-3 ? "OK" : "FAIL")
end

println("\n" * "="^76)
println("CURVED TRACK — dense per-site velocity field (a single global FFT cannot do this)")
x0 = (N ÷ 5, N ÷ 5)
for κ in (0.0, 0.015, 0.03)
    D = moving_drive(N, N, 40; v = 5.0, angle = 0.0, width = 0.8, curvature = κ)
    Z = velocity_bank_run(bank, ps, st, D)
    sp, an, mask = decode_velocity(bank, ps, Z; frac = 0.98)
    # true heading at each labelled site: nearest point on the arc, tangent there
    arc = range(0, 5.0 * 40; length = 4000)
    pts = κ == 0 ? [(x0[1] + s, float(x0[2])) for s in arc] :
          [(x0[1] + sin(κ * s) / κ, x0[2] + (1 - cos(κ * s)) / κ) for s in arc]
    errs = Float64[]
    for I in findall(mask)
        i, j = I[1] - 1, I[2] - 1
        k = argmin([(p[1] - i)^2 + (p[2] - j)^2 for p in pts])
        push!(errs, abs(wrap(an[I] - κ * arc[k])))
    end
    @printf("  curvature=%.3f (turns %.2f rad): %4d sites | speed median %.2f (true 5.0) | ",
            κ, κ * 5.0 * 40, count(mask), median(sp[mask]))
    @printf("heading err median %.3f rad, 90th pct %.3f\n",
            median(errs), sort(errs)[max(1, ceil(Int, 0.9 * length(errs)))])
end
println("="^76)
