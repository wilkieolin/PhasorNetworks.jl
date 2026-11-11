# Tests for the FPE encoder, the resonator network and the residue-code
# predictions of ARC note `grid_code_addressing.md` §3(e) (A3-min).
#
# The experiment functions in the first half are shared with
# `scripts/fpe_resonator_sweep.jl`, which includes this file and runs them at
# full size (D = 1024, more trials) to produce the numbers in
# `docs/fpe_resonator.md`. The testset at the bottom runs them small.

using Test
using Lux
using PhasorNetworks
using Random: Xoshiro, AbstractRNG
using Statistics: mean
using LinearAlgebra: norm

const _fpe_rp = PhasorNetworks.remap_phase

# ---------------------------------------------------------------------------
# Noise and decoding helpers
# ---------------------------------------------------------------------------

"""
    vonmises_noise(rng, σ_rad, dims...) -> Array{Float32}

Zero-mean von Mises samples with concentration `κ = 1/σ²` (circular std ≈ σ
for σ ≲ 0.5 rad), returned in units of π (Best & Fisher 1979).
"""
function vonmises_noise(rng::AbstractRNG, σ_rad::Real, dims::Integer...)
    σ_rad <= 0 && return zeros(Float32, dims...)
    κ = 1.0 / σ_rad^2
    τ = 1 + sqrt(1 + 4κ^2)
    ρ = (τ - sqrt(2τ)) / (2κ)
    r = (1 + ρ^2) / (2ρ)
    out = Array{Float32}(undef, dims...)
    for i in eachindex(out)
        θ = 0.0
        while true
            u1, u2, u3 = rand(rng), rand(rng), rand(rng)
            z = cos(π * u1)
            f = (1 + r * z) / (r + z)
            c = κ * (r - f)
            if c * (2 - c) - u2 > 0 || log(c / u2) + 1 - c >= 0
                θ = sign(u3 - 0.5) * acos(clamp(f, -1, 1))
                break
            end
        end
        out[i] = Float32(θ / π)
    end
    return out
end

add_phase_noise(rng::AbstractRNG, θ::AbstractArray, σ_rad::Real) =
    _fpe_rp(Float32.(θ) .+ vonmises_noise(rng, σ_rad, size(θ)...))

# Nearest codeword (max cosine similarity) of every column of `θ` in `codebook`.
function nearest_codeword(θ::AbstractMatrix, codebook::AbstractMatrix)
    Z = angle_to_complex(Float32.(θ))
    C = angle_to_complex(Float32.(codebook))
    S = real.(C' * Z)
    return [argmax(view(S, :, b)) for b in 1:size(S, 2)]
end

# Per-block codebook decode of a `:blocks`-layout residue encoder: residue of
# each modulus from its own block, then CRT.
function blockwise_decode(enc::FPEEncoder, st, θ::AbstractMatrix)
    K = length(enc.moduli)
    res = zeros(Int, K, size(θ, 2))
    for k in 1:K
        blk = st.blocks[k]+1:st.blocks[k+1]
        cb = fpe_codebook(enc, st, k)[blk, :]
        res[k, :] = nearest_codeword(θ[blk, :], cb) .- 1
    end
    return [crt_decode(res[:, b], enc.moduli) for b in 1:size(θ, 2)]
end

resonator_decode(indices::AbstractMatrix{Int}, moduli) =
    [crt_decode(indices[:, b] .- 1, moduli) for b in 1:size(indices, 2)]

# ---------------------------------------------------------------------------
# (i) Range law: unique range of a K-modulus residue code
# ---------------------------------------------------------------------------

"""
    range_law(rng; moduli, D, σ, n_addr = 300, steps = 20, acc_min = 0.99)

Blocks-layout residue code, von Mises noise `σ` rad per component, per-modulus
codebook decode + CRT. Addresses are drawn uniformly from `[0, X)` for `X` on a
geometric grid around `∏ m` (`steps` per decade); the measured unique range
`R` is the largest `X` with decode accuracy ≥ `acc_min`. Returns
`(range = R, product = ∏m, acc_at_product, acc_at_2product, K_eff = log R / log max m)`.
"""
function range_law(rng::AbstractRNG; moduli, D::Int, σ::Real, n_addr::Int = 300,
                   steps::Int = 20, acc_min::Real = 0.99)
    enc = FPEEncoder(1 => D; mode = :residue, moduli = moduli, layout = :blocks)
    _, st = Lux.setup(rng, enc)
    M = prod(moduli)
    accuracy(X) = begin
        xs = rand(rng, 0:X-1, n_addr)
        θ, _ = enc(Float32.(xs), NamedTuple(), st)
        dec = blockwise_decode(enc, st, add_phase_noise(rng, θ, σ))
        mean(dec .== xs)
    end
    R = 0
    for j in -steps:steps
        X = max(2, round(Int, M * 10.0^(j / steps)))
        accuracy(X) >= acc_min && (R = X)
    end
    return (range = R, product = M, acc_at_product = accuracy(M),
            acc_at_2product = accuracy(2M), K_eff = log(max(R, 1)) / log(maximum(moduli)))
end

# ---------------------------------------------------------------------------
# (ii) Resonator vs codebook decoding of bound addresses
# ---------------------------------------------------------------------------

"""
    resonator_vs_codebook(rng; moduli, D, σ, n_addr = 100, iterations = 100, projection = :abs)

Bound-layout residue code (one dense address vector). Returns the resonator's
address accuracy, mean iterations and convergence fraction; the accuracy of
decoding each modulus directly against its codebook *without* unbinding the
other factors (what a `Codebook` layer alone can do on a bound vector); the
storage of the resonator (`Σ m_k` codewords); and the number of addresses a
flat codebook of the same storage can hold (`Σ m_k`, each a stored vector),
with its decode accuracy under the same noise.
"""
function resonator_vs_codebook(rng::AbstractRNG; moduli, D::Int, σ::Real, n_addr::Int = 100,
                               iterations::Int = 100, projection::Symbol = :abs)
    enc = FPEEncoder(1 => D; mode = :residue, moduli = moduli, layout = :bound)
    _, st = Lux.setup(rng, enc)
    M = prod(moduli)
    xs = rand(rng, 0:M-1, n_addr)
    θ, _ = enc(Float32.(xs), NamedTuple(), st)
    θn = add_phase_noise(rng, θ, σ)

    res = Resonator(enc, st; iterations = iterations, projection = projection)
    rps, rst = Lux.setup(rng, res)
    out, _ = res(θn, rps, rst)
    dec = resonator_decode(out.indices, moduli)

    # Direct per-modulus codebook decode of the bound vector (no unbinding).
    cbs = fpe_codebooks(enc, st)
    direct = [nearest_codeword(θn, cb) .- 1 for cb in cbs]
    direct_ok = [all(direct[k][b] == mod(xs[b], moduli[k]) for k in eachindex(moduli)) for b in 1:n_addr]

    # Flat codebook with the resonator's storage: Σ m_k random address vectors.
    storage = sum(moduli)
    flat_addr = rand(rng, 0:M-1, storage)
    flat_cb, _ = enc(Float32.(flat_addr), NamedTuple(), st)
    q = rand(rng, 1:storage, n_addr)
    flat_dec = nearest_codeword(add_phase_noise(rng, flat_cb[:, q], σ), flat_cb)

    return (product = M, acc = mean(dec .== xs), mean_iterations = mean(out.iterations),
            converged = mean(out.converged), direct_codebook_acc = mean(direct_ok),
            storage = storage, flat_codebook_addresses = storage,
            flat_codebook_acc = mean(flat_dec .== q),
            ratio = M * mean(dec .== xs) / storage)
end

# ---------------------------------------------------------------------------
# (iii) Hexagonal three-phase vs Cartesian two-phase 2-D code
# ---------------------------------------------------------------------------

# Wrap a per-axis level error to (−m/2, m/2].
_wrap_level(e, m) = e - m * round(e / m)

# Positions uniformly inside one period cell of the code: [0, m)² for the
# square code, {m(u b₁ + v b₂)} for the hex code with b_i·k_j = δ_ij.
function _period_positions(rng, mode::Symbol, m::Int, n::Int)
    u = rand(rng, Float32, 2, n)
    if mode === :hex
        B = Float32[1 0; 1/sqrt(3f0) 2/sqrt(3f0)]      # columns b₁, b₂
        return Float32(m) .* (B * u)
    else
        return Float32(m) .* u
    end
end

"""
    hex_vs_square(rng; m, D, σ, n_pos = 4000, n_bins = 12)

Continuous 2-D positions encoded by unit-exponent residue blocks (`ω = 2π/m`
per axis), one block per axis: two Cartesian axes vs the three hex axes
`k₁ + k₂ + k₃ = 0`. Each axis phase is decoded to the nearest of `m` levels.
Returns, per code, the number of distinct decoded states per period cell, the
state density per unit area, the probability that noise `σ` changes the
decoded state, and the anisotropy of the quantisation error (max/min of the
mean squared least-squares position error over `n_bins` direction bins).
"""
function hex_vs_square(rng::AbstractRNG; m::Int, D::Int, σ::Real, n_pos::Int = 4000, n_bins::Int = 12)
    results = NamedTuple[]
    for mode in (:residue, :hex)
        enc = FPEEncoder(2 => D; mode = mode, moduli = [m], layout = :blocks, exponent = :unit)
        _, st = Lux.setup(rng, enc)
        A = size(st.axes, 1)
        p = _period_positions(rng, mode, m, n_pos)
        θ, _ = enc(p, NamedTuple(), st)
        θn = add_phase_noise(rng, θ, σ)
        q = zeros(Int, A, n_pos); qn = zeros(Int, A, n_pos)
        for a in 1:A
            blk = st.blocks[a]+1:st.blocks[a+1]
            cb = fpe_codebook(enc, st, 1, a)[blk, :]
            q[a, :] = nearest_codeword(θ[blk, :], cb) .- 1
            qn[a, :] = nearest_codeword(θn[blk, :], cb) .- 1
        end
        states = length(unique(eachcol(q)))
        period_area = mode === :hex ? (2 / sqrt(3)) * m^2 : Float64(m^2)
        # Least-squares position error from the wrapped per-axis level errors:
        # e = (2/3) Σ_i ε_i k_i for the tight hex frame, e = (ε_x, ε_y) for axes.
        proj = st.axes * p                                     # (A, n) true projections
        ε = _wrap_level.(Float32.(q) .- proj, m)                # (A, n)
        e = (A == 3 ? (2f0 / 3f0) : 1f0) .* (st.axes' * ε)      # (2, n)
        ang = atan.(e[2, :], e[1, :])
        bins = clamp.(floor.(Int, (ang .+ π) ./ (2π) .* n_bins) .+ 1, 1, n_bins)
        mse = [mean(sum(abs2, e[:, bins .== b], dims = 1)) for b in 1:n_bins]
        push!(results, (code = mode === :hex ? :hex3 : :square2, axes = A, states_per_period = states,
                        states_per_area = states / period_area,
                        noise_flip_rate = mean(any(qn .!= q, dims = 1)),
                        mse = mean(sum(abs2, e, dims = 1)), anisotropy = maximum(mse) / minimum(mse)))
    end
    return results
end

# ---------------------------------------------------------------------------
# (iv) Error-correction margin (channel form: one phase per modulus)
# ---------------------------------------------------------------------------

# Phases (rad) of the residue code of `xs` on the K-torus, (K, n).
_residue_phases(xs, moduli) = Float32[2π * mod(x, m) / m for m in moduli, x in xs]

# Circular mean over `R` noisy replicas of each phase (rad): the decoder of a
# channel carried by R phasors sees one averaged phase.
function _noisy_replicated(rng, Φ::AbstractMatrix, σ, R::Int)
    R == 1 && return Φ .+ Float32(π) .* vonmises_noise(rng, σ, size(Φ)...)
    acc = zeros(ComplexF32, size(Φ))
    for _ in 1:R
        acc .+= cis.(Φ .+ Float32(π) .* vonmises_noise(rng, σ, size(Φ)...))
    end
    return angle.(acc)
end

"""
    margin_trial(rng; moduli, n_addr = 2^16, σ, n_trials = 400, digits = 4, base = 16, replicas = 1)

Comparison at phase noise `σ` rad per phasor, with each phase carried by
`replicas` independently noisy phasors (`1` = channel form, one oscillator per
modulus; `D/K` ≈ vector form):
- residue code with `K = length(moduli)` phases, addresses `0:n_addr-1` inside
  `∏ m`; decoded (a) per channel to the nearest level + CRT (no use of the
  address set) and (b) to the nearest *valid* codeword over the `n_addr`
  addresses in the wrapped-phase metric (nearest-codeword cleanup);
- a `base`-ary phase per digit (`digits` phases hold `base^digits` addresses);
  per-phase error rate and address error rate.
Returns the four error rates.
"""
function margin_trial(rng::AbstractRNG; moduli, n_addr::Int = 2^16, σ::Real, n_trials::Int = 400,
                      digits::Int = 4, base::Int = 16, replicas::Int = 1)
    K = length(moduli)
    xs = rand(rng, 0:n_addr-1, n_trials)
    O = _noisy_replicated(rng, _residue_phases(xs, moduli), σ, replicas)   # (K, T)
    # (a) per-channel rounding + CRT
    res = [mod(round(Int, O[k, t] * moduli[k] / (2π)), moduli[k]) for k in 1:K, t in 1:n_trials]
    dec_a = [crt_decode(res[:, t], moduli) for t in 1:n_trials]
    # (b) nearest valid codeword, brute force in chunks
    valid = _residue_phases(0:n_addr-1, moduli)                 # (K, N)
    dec_b = zeros(Int, n_trials)
    chunk = 64
    for t0 in 1:chunk:n_trials
        ts = t0:min(t0 + chunk - 1, n_trials)
        d2 = zeros(Float32, n_addr, length(ts))
        for k in 1:K
            Δ = valid[k, :] .- O[k, ts]'
            d2 .+= rem2pi.(Δ, RoundNearest) .^ 2
        end
        dec_b[ts] = [argmin(view(d2, :, j)) - 1 for j in 1:length(ts)]
    end
    # base-ary digits
    xd = rand(rng, 0:base^digits-1, n_trials)
    digit = [mod(x ÷ base^(j - 1), base) for j in 1:digits, x in xd]
    Od = _noisy_replicated(rng, Float32(2π) .* digit ./ base, σ, replicas)
    dd = mod.(round.(Int, Od .* base ./ (2π)), base)
    return (sigma = σ, replicas = replicas, residue_perchannel_err = mean(dec_a .!= xs),
            residue_nearest_codeword_err = mean(dec_b .!= xs),
            mary_phase_err = mean(dd .!= digit), mary_address_err = mean(any(dd .!= digit, dims = 1)))
end

# ---------------------------------------------------------------------------
# Testset
# ---------------------------------------------------------------------------

function fpe_resonator_tests()
    @testset "FPE encoder and resonator" begin
        @info "Running FPE / resonator tests..."
        t_start = time()

        @testset "FPEEncoder construction and shapes" begin
            rng = Xoshiro(1)
            enc = FPEEncoder(1 => 64; moduli = [7, 11, 13])
            @test enc.mode === :residue && enc.layout === :bound
            ps, st = Lux.setup(rng, enc)
            @test isempty(ps)
            @test size(st.weight) == (64, 1) && size(st.omega) == (64, 1, 3)
            y, _ = enc(Float32.(0:9), ps, st)
            @test size(y) == (64, 10) && eltype(y) == Phase
            @test all(-1f0 .<= Float32.(y) .<= 1f0)
            y3, _ = enc(rand(Float32, 1, 5, 2), ps, st)
            @test size(y3) == (64, 5, 2)
            # residue table: ω = 2π/m per modulus
            tab = fpe_table(enc, st)
            @test [r.modulus for r in tab] == [7, 11, 13]
            @test all(r.omega_rad ≈ 2π / r.modulus for r in tab)
            # blocks layout partitions D
            encb = FPEEncoder(1 => 64; moduli = [7, 11, 13], layout = :blocks)
            _, stb = Lux.setup(rng, encb)
            @test stb.blocks == [0, 21, 43, 64]
            @test all(iszero, stb.omega[1:21, 1, 2:3]) && all(!iszero, stb.omega[1:21, 1, 1])
            # uniform mode: sinc kernel — similar nearby, orthogonal far away
            encu = FPEEncoder(1 => 2048; mode = :uniform, bandwidth = 2f0)
            _, stu = Lux.setup(rng, encu)
            yu, _ = encu(Float32[0, 0.1, 10], ps, stu)
            s = similarity_outer(yu, yu; dims = 2)
            @test s[1, 2] > 0.9 && abs(s[1, 3]) < 0.1
            # hex mode: three axes summing to zero, needs in_dims == 2
            ench = FPEEncoder(2 => 96; mode = :hex, moduli = [5], layout = :blocks, exponent = :unit)
            _, sth = Lux.setup(rng, ench)
            @test size(sth.axes) == (3, 2) && all(abs.(sum(sth.axes, dims = 1)) .< 1f-6)
            @test_throws ArgumentError FPEEncoder(1 => 8; mode = :hex, moduli = [5])
            @test_throws ArgumentError FPEEncoder(1 => 8; mode = :residue)
            @test_throws ArgumentError FPEEncoder(1 => 8; moduli = [5], layout = :diagonal)
            # codebooks reproduce the encoder: bind of selected columns == output
            cbs = fpe_codebooks(enc, st)
            x = 100
            cols = [cbs[k][:, mod(x, enc.moduli[k]) + 1] for k in 1:3]
            yx, _ = enc(Float32[x], ps, st)
            @test maximum(abs.(arc_error(Float32.(reduce(v_bind, cols)) .- Float32.(yx)))) < 1f-4
            @test crt_decode([mod(x, m) for m in enc.moduli], enc.moduli) == x
        end

        @testset "(v) bind / unbind round trip" begin
            rng = Xoshiro(2)
            a = random_symbols(rng, (256, 8)); b = random_symbols(rng, (256, 8))
            @test maximum(abs.(arc_error(Float32.(v_unbind(v_bind(a, b), b)) .- Float32.(a)))) < 1f-5
            enc = FPEEncoder(1 => 256; moduli = [7, 11, 13], random_offset = true)
            ps, st = Lux.setup(rng, enc)
            ex, _ = enc(Float32[3, 20, 500], ps, st)
            ey, _ = enc(Float32[5, 17, 999], ps, st)
            exy, _ = enc(Float32[8, 37, 1499], ps, st)
            # FPE is a homomorphism: enc(x + y) = bind(enc(x), enc(y)) (offset enters once)
            bound = v_bind(v_unbind(ex, Phase.(st.offset)), ey)
            @test maximum(abs.(arc_error(Float32.(bound) .- Float32.(exy)))) < 1f-4
            @test maximum(abs.(arc_error(Float32.(v_unbind(bound, ey)) .- Float32.(v_unbind(ex, Phase.(st.offset)))))) < 1f-5

            # Spiking path: phase → spike time → phase is exact, and the spiking
            # bind (spike delay) reproduces the phase bind cycle by cycle.
            spk = SpikingArgs()
            ms = MakeSpiking(spk, 2)
            call_x, _ = ms(ex, NamedTuple(), NamedTuple())
            call_y, _ = ms(ey, NamedTuple(), NamedTuple())
            @test call_x isa SpikingCall
            px = train_to_phase(call_x)
            @test size(px) == (256, 3, 2)
            @test maximum(abs.(arc_error(px[:, :, end] .- Float32.(ex)))) < 1f-4
            call_xy = spike_phase_bind(call_x, call_y)
            pxy = train_to_phase(call_xy)
            @test maximum(abs.(arc_error(pxy[:, :, end] .- Float32.(v_bind(ex, ey))))) < 1f-3
            # Resonator reads the spike train and recovers the same factors
            res = Resonator(enc, st; iterations = 50)
            rps, rst = Lux.setup(Xoshiro(3), res)
            o_phase, _ = res(ex, rps, rst)
            o_spike, _ = res(call_x, rps, rst)
            @test o_phase.indices == o_spike.indices
            @test resonator_decode(o_phase.indices, enc.moduli) == [3, 20, 500]
            @test all(o_phase.converged) && all(o_phase.similarity .> 0.99f0)
        end

        @testset "(i) range law" begin
            rng = Xoshiro(4)
            families = ([13], [11, 13], [9, 11, 13], [7, 9, 11, 13], [5, 7, 9, 11, 13], [4, 5, 7, 9, 11, 13])
            rows = [range_law(rng; moduli = f, D = 192, σ = 0.3, n_addr = 200) for f in families]
            for (f, r) in zip(families, rows)
                @info "range law" moduli = f range = r.range product = r.product K_eff = round(r.K_eff, digits = 2) K_eff_over_K = round(r.K_eff / length(f), digits = 2)
                @test r.range == r.product
                @test r.acc_at_product >= 0.99 && r.acc_at_2product <= 0.6
            end
            ranges = [r.range for r in rows]
            @test all(ranges[2:end] ./ ranges[1:end-1] .>= 4)          # exponential in K
            @test all(0.75 .<= [r.K_eff / length(f) for (f, r) in zip(families, rows)] .<= 1.0)
        end

        @testset "(ii) resonator vs codebook" begin
            rng = Xoshiro(5)
            r = resonator_vs_codebook(rng; moduli = [11, 13, 17], D = 256, σ = 0.3, n_addr = 100)
            @info "resonator vs codebook (D=256)" r.product acc = r.acc iters = r.mean_iterations direct = r.direct_codebook_acc storage = r.storage ratio = round(r.ratio, digits = 1)
            @test r.acc >= 0.95 && r.converged >= 0.95
            @test r.direct_codebook_acc < 0.2
            @test r.flat_codebook_acc >= 0.95 && r.flat_codebook_addresses == r.storage
            @test r.ratio > 10
            # the hard codeword snap from a random start does not find the factors
            rc = resonator_vs_codebook(Xoshiro(5); moduli = [11, 13, 17], D = 256, σ = 0.3, n_addr = 50,
                                       projection = :codeword)
            @test rc.acc < 0.5
            rl = resonator_vs_codebook(Xoshiro(5); moduli = [7, 11, 13], D = 256, σ = 0.3, n_addr = 50,
                                       projection = :linear)
            @info "projection variants" codeword_acc = rc.acc linear_acc = rl.acc
        end

        @testset "(iii) hex three-phase vs square two-phase" begin
            rng = Xoshiro(6)
            m = 5
            rows = hex_vs_square(rng; m = m, D = 96, σ = 0.15, n_pos = 3000)
            sq, hx = rows
            @info "hex vs square (m=5)" square_states = sq.states_per_period hex_states = hx.states_per_period square_density = round(sq.states_per_area, digits = 3) hex_density = round(hx.states_per_area, digits = 3) square_aniso = round(sq.anisotropy, digits = 2) hex_aniso = round(hx.anisotropy, digits = 2) square_flip = sq.noise_flip_rate hex_flip = hx.noise_flip_rate
            @test sq.states_per_period == m^2
            @test hx.states_per_period == 3m^2                          # > 3m² − 3m + 1
            @test hx.states_per_area > 2 * sq.states_per_area
            @test hx.mse < sq.mse
        end

        @testset "(iv) error-correction margin" begin
            rng = Xoshiro(7)
            moduli = [13, 15, 16, 17, 19]                               # ∏ = 1 007 760
            @test prod(moduli) > 10^6
            rows = [margin_trial(rng; moduli = moduli, σ = σ, replicas = R, n_trials = 200)
                    for R in (1, 16), σ in (0.2, 0.3)]
            for r in rows
                @info "margin" r...
            end
            r1_02, r16_02, r1_03, r16_03 = rows
            # single 16-ary phase (one phasor per digit) fails at σ ≈ 0.2 rad
            @test r1_02.mary_phase_err > 0.25
            # nearest-codeword cleanup beats per-channel rounding at every σ
            @test all(r.residue_nearest_codeword_err <= r.residue_perchannel_err for r in rows)
            # channel form: the residue code does NOT survive 0.3 rad (prediction falsified)
            @test r1_03.residue_nearest_codeword_err > 0.5
            # with 16 phasors per channel it does, where the 16-ary code still errs
            @test r16_03.residue_nearest_codeword_err < 0.05
            @test r16_03.residue_nearest_codeword_err <= r16_03.mary_address_err
        end

        @info "FPE / resonator tests done in $(round(time() - t_start, digits = 1)) s"
    end
end
