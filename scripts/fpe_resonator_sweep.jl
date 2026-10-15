# Full-size runs of the residue-code experiments behind `docs/fpe_resonator.md`
# (ARC A3-min; predictions of grid_code_addressing.md §3(e)).
#
#     CUDA_VISIBLE_DEVICES= julia --project=. scripts/fpe_resonator_sweep.jl [outdir]
#
# CPU only; needs nothing beyond the package environment (the experiment
# functions live in test/test_fpe_resonator.jl and are included here). Writes
# one CSV per experiment into `results/fpe_resonator/` (default) and prints a
# summary. Runtime ≈ 5 min on a laptop core.

using PhasorNetworks, Lux, Statistics, Printf
using Random: Xoshiro
using LinearAlgebra: BLAS
include(joinpath(@__DIR__, "..", "test", "test_fpe_resonator.jl"))

const OUT = length(ARGS) >= 1 ? ARGS[1] : joinpath(@__DIR__, "..", "results", "fpe_resonator")
mkpath(OUT)

function write_csv(path, rows::AbstractVector)
    isempty(rows) && return
    keys_ = collect(keys(rows[1]))
    open(path, "w") do io
        println(io, join(string.(keys_), ","))
        for r in rows
            println(io, join([string(getfield(r, k)) for k in keys_], ","))
        end
    end
    println("wrote $(path) ($(length(rows)) rows)")
end

fmt(x) = x isa AbstractFloat ? @sprintf("%.3g", x) : string(x)

# ---------------------------------------------------------------------------
# (i) Range law
# ---------------------------------------------------------------------------
println("\n(i) range law")
families = ([13], [11, 13], [9, 11, 13], [7, 9, 11, 13], [5, 7, 9, 11, 13], [4, 5, 7, 9, 11, 13])
rows_i = NamedTuple[]
for D in (256, 1024), σ in (0.3, 0.6, 1.0), f in families
    r = range_law(Xoshiro(11); moduli = f, D = D, σ = σ, n_addr = 500)
    push!(rows_i, (D = D, sigma = σ, K = length(f), moduli = join(f, "·"), max_m = maximum(f),
                   product = r.product, range = r.range, acc_at_product = r.acc_at_product,
                   acc_at_2product = r.acc_at_2product, K_eff = r.K_eff, K_eff_over_K = r.K_eff / length(f)))
    @printf("  D=%4d σ=%.1f K=%d ∏m=%6d  range=%6d  acc(∏m)=%.3f  K_eff=%.2f (%.2f K)\n",
            D, σ, length(f), r.product, r.range, r.acc_at_product, r.K_eff, r.K_eff / length(f))
end
write_csv(joinpath(OUT, "range_law.csv"), rows_i)

# ---------------------------------------------------------------------------
# (ii) Resonator vs codebook: capacity at D = 1024
# ---------------------------------------------------------------------------
println("\n(ii) resonator vs codebook")
sets = ([7, 11, 13], [13, 17, 19], [23, 29, 31], [41, 43, 47], [53, 59, 61], [67, 71, 73],
        [79, 83, 89], [97, 101, 103], [127, 131, 137],
        [97, 101], [251, 257], [499, 503],
        [7, 11, 13, 17], [13, 17, 19, 23], [23, 29, 31, 37])
rows_ii = NamedTuple[]
for (D, σ, proj) in ((1024, 0.3, :abs), (1024, 0.0, :abs), (1024, 0.6, :abs), (512, 0.3, :abs),
                     (2048, 0.3, :abs), (1024, 0.3, :linear), (1024, 0.3, :codeword))
    for m in sets
        (proj !== :abs && prod(m) > 10^5) && continue
        r = resonator_vs_codebook(Xoshiro(21); moduli = m, D = D, σ = σ, n_addr = 200,
                                  iterations = 200, projection = proj)
        push!(rows_ii, (D = D, sigma = σ, projection = proj, K = length(m), moduli = join(m, "·"),
                        product = r.product, storage = r.storage, resonator_acc = r.acc,
                        mean_iterations = r.mean_iterations, converged = r.converged,
                        direct_codebook_acc = r.direct_codebook_acc,
                        flat_codebook_addresses = r.flat_codebook_addresses,
                        flat_codebook_acc = r.flat_codebook_acc, ratio = r.ratio))
        @printf("  D=%4d σ=%.1f %-8s K=%d ∏m=%8d Σm=%4d  res acc=%.3f it=%5.1f  direct=%.2f  flat=%.2f  ratio=%7.1f\n",
                D, σ, proj, length(m), r.product, r.storage, r.acc, r.mean_iterations,
                r.direct_codebook_acc, r.flat_codebook_acc, r.ratio)
    end
end
write_csv(joinpath(OUT, "resonator_vs_codebook.csv"), rows_ii)

# ---------------------------------------------------------------------------
# (iii) Hex three-phase vs square two-phase
# ---------------------------------------------------------------------------
println("\n(iii) hex vs square")
rows_iii = NamedTuple[]
for m in (3, 5, 7, 9, 11, 13), σ in (0.1, 0.2)
    for r in hex_vs_square(Xoshiro(31); m = m, D = 1536, σ = σ, n_pos = 20000)
        push!(rows_iii, (m = m, sigma = σ, code = r.code, axes = r.axes,
                         states_per_period = r.states_per_period, predicted = r.axes == 3 ? 3m^2 : m^2,
                         kymn_count = r.axes == 3 ? 3m^2 - 3m + 1 : m^2,
                         states_per_area = r.states_per_area, bits_per_area = log2(r.states_per_area),
                         noise_flip_rate = r.noise_flip_rate, mse = r.mse, anisotropy = r.anisotropy))
        @printf("  m=%2d σ=%.1f %-7s states/period=%4d (3m²-3m+1=%4d) density=%.3f  flip=%.3f  mse=%.4f  aniso=%.2f\n",
                m, σ, r.code, r.states_per_period, r.axes == 3 ? 3m^2 - 3m + 1 : m^2,
                r.states_per_area, r.noise_flip_rate, r.mse, r.anisotropy)
    end
end
write_csv(joinpath(OUT, "hex_vs_square.csv"), rows_iii)

# ---------------------------------------------------------------------------
# (iv) Error-correction margin
# ---------------------------------------------------------------------------
println("\n(iv) error-correction margin (2^16 addresses)")
rows_iv = NamedTuple[]
for moduli in ([13, 15, 16, 17, 19], [29, 31, 32, 33]), R in (1, 4, 16, 64), σ in 0.05:0.05:0.5
    r = margin_trial(Xoshiro(41); moduli = moduli, σ = σ, replicas = R, n_trials = 400)
    push!(rows_iv, (moduli = join(moduli, "·"), product = prod(moduli), K = length(moduli),
                    replicas = R, sigma = σ, residue_perchannel_err = r.residue_perchannel_err,
                    residue_nearest_codeword_err = r.residue_nearest_codeword_err,
                    mary_phase_err = r.mary_phase_err, mary_address_err = r.mary_address_err))
end
for moduli in ("13·15·16·17·19", "29·31·32·33"), R in (1, 4, 16, 64)
    sel = filter(r -> r.moduli == moduli && r.replicas == R, rows_iv)
    first_ok(f) = (i = findlast(r -> f(r) <= 0.01, sel); i === nothing ? NaN : sel[i].sigma)
    @printf("  %-15s R=%2d  σ(err≤1%%): residue NC %.2f | residue per-channel %.2f | 16-ary phase %.2f | 16-ary address %.2f\n",
            moduli, R, first_ok(r -> r.residue_nearest_codeword_err), first_ok(r -> r.residue_perchannel_err),
            first_ok(r -> r.mary_phase_err), first_ok(r -> r.mary_address_err))
end
write_csv(joinpath(OUT, "margin.csv"), rows_iv)

# Vector form of the same code: resonator at D = 1024 vs σ.
println("\n(iv-b) vector form, resonator, moduli 13·15·16·17·19, addresses < 2^16")
rows_ivb = NamedTuple[]
for D in (1024, 2048)
    moduli = [13, 15, 16, 17, 19]
    enc = FPEEncoder(1 => D; moduli = moduli)
    _, st = Lux.setup(Xoshiro(51), enc)
    res = Resonator(enc, st; iterations = 200)
    rps, rst = Lux.setup(Xoshiro(52), res)
    for σ in (0.3, 0.6, 0.9, 1.2, 1.5)
        rng = Xoshiro(53)
        xs = rand(rng, 0:2^16-1, 200)
        θ, _ = enc(Float32.(xs), NamedTuple(), st)
        out, _ = res(add_phase_noise(rng, θ, σ), rps, rst)
        acc = mean(resonator_decode(out.indices, moduli) .== xs)
        push!(rows_ivb, (D = D, sigma = σ, acc = acc, mean_iterations = mean(out.iterations),
                         converged = mean(out.converged)))
        @printf("  D=%d σ=%.1f  acc=%.3f  iterations=%.1f\n", D, σ, acc, mean(out.iterations))
    end
end
write_csv(joinpath(OUT, "margin_vector_form.csv"), rows_ivb)

# ---------------------------------------------------------------------------
# (v) Round trip at D = 1024
# ---------------------------------------------------------------------------
println("\n(v) round trip, D=1024")
let rng = Xoshiro(61), D = 1024
    a = random_symbols(rng, (D, 64)); b = random_symbols(rng, (D, 64))
    e_vsa = maximum(abs.(arc_error(Float32.(v_unbind(v_bind(a, b), b)) .- Float32.(a))))
    enc = FPEEncoder(1 => D; moduli = [13, 15, 16, 17, 19], random_offset = true)
    _, st = Lux.setup(rng, enc)
    x = Float32.(rand(rng, 0:2^15, 64)); y = Float32.(rand(rng, 0:2^15, 64))
    ex, _ = enc(x, NamedTuple(), st); ey, _ = enc(y, NamedTuple(), st); exy, _ = enc(x .+ y, NamedTuple(), st)
    bound = v_bind(v_unbind(ex, Phase.(st.offset)), ey)
    e_hom = maximum(abs.(arc_error(Float32.(bound) .- Float32.(exy))))
    spk = SpikingArgs(); ms = MakeSpiking(spk, 2)
    cx, _ = ms(ex, NamedTuple(), NamedTuple()); cy, _ = ms(ey, NamedTuple(), NamedTuple())
    e_spk = maximum(abs.(arc_error(train_to_phase(cx)[:, :, end] .- Float32.(ex))))
    e_spk_bind = maximum(abs.(arc_error(train_to_phase(spike_phase_bind(cx, cy))[:, :, end] .- Float32.(v_bind(ex, ey)))))
    rows_v = [(quantity = "unbind(bind(a,b),b) vs a", max_arc_error = e_vsa),
              (quantity = "bind(enc(x),enc(y)) vs enc(x+y)", max_arc_error = e_hom),
              (quantity = "spike-time round trip", max_arc_error = e_spk),
              (quantity = "spike_phase_bind vs v_bind", max_arc_error = e_spk_bind)]
    foreach(r -> @printf("  %-34s %.2e\n", r.quantity, r.max_arc_error), rows_v)
    write_csv(joinpath(OUT, "roundtrip.csv"), rows_v)
end
println("\ndone")
