###
### Fractional power encoding (FPE) and the resonator network
###
### A phasor's phase is a residue mod 2π. Writing a scalar x into D phasors as
### θ_j = ω_j·x (mod 2π) is fractional power encoding (Plate 1995; Frady et al.
### 2021). With ω_j = 2π r_j / m_k (roots of unity) a block of components
### stores x mod m_k exactly, and K blocks with pairwise-coprime moduli form a
### K-modulus residue code (Kymn et al. 2024) whose unique range is ∏ m_k.
### Binding (`v_bind`, phase addition) is carry-free addition of the encoded
### scalars; a bound address is factored back into its residues by the
### resonator network (Frady, Kent, Olshausen & Sommer 2020).
###
### Ground rule: only phases leave these layers. The encoder (real x → phase) is
### the one off-torus operation and sits at the input boundary; the resonator's
### codebook projection is a readout of the same kind as `Codebook`.
###
### The iterated unbind / similarity / cleanup loop was first written in demo
### form in `demos/03_resonator.ipynb` (ported from the older phasor_julia
### demos); `Resonator(projection = :abs)` lifts that loop into the library,
### `:linear` is the projection of Frady et al. 2020 / Kymn et al. 2024 as
### written, `:codeword` the hard-snap baseline.

# Three unit vectors at 120°: k₁ + k₂ + k₃ = 0. Rows are the axes, columns the
# 2-D input coordinates.
const HEX_AXES = Float32[1.0 0.0; -0.5 (sqrt(3) / 2); -0.5 (-sqrt(3) / 2)]

"""
    FPEEncoder(in_dims => out_dims; mode = :residue, moduli = Int[],
               bandwidth = 1f0, exponent = :random, layout = :bound,
               random_offset = false) <: Lux.AbstractLuxLayer

Fractional-power encoder: real coordinates `x ∈ ℝ^in_dims` ↦ `D = out_dims`
phases `θ_j = Σ_a Ω[j, a] x_a + ψ_j (mod 2π)`. Output is a `Phase` array in the
library's `[-1, 1]` (units of π) convention, so every downstream phase-domain
layer (`v_bind`, `Codebook`, `PhasorDense`, `MakeSpiking` → spike times)
accepts it directly.

# Modes
- `:uniform` — `Ω[j, a] ~ U(−B, B)` rad per unit input (`bandwidth = B`).
  Similarity kernel between two inputs is `∏_a sinc(B Δx_a / π)`: similar for
  nearby inputs, orthogonal beyond `≈ π / B` (Frady et al. 2021).
- `:residue` — `K = length(moduli)` root-of-unity bases. Component `j` of
  modulus block `k` gets `Ω = 2π r_j / m_k` with `r_j = 1` (`exponent = :unit`,
  smooth periodic kernel `cos(2π Δx / m_k)`) or `r_j` drawn uniformly from
  `{0, …, m_k − 1}` (`exponent = :random`, Kymn et al. 2024: each residue is a
  quasi-orthogonal vector, delta kernel; components with `r_j = 0` are
  constant for that modulus). The block stores exactly `x mod m_k`; with pairwise
  coprime moduli the code is unique over `∏ m_k` integers.
- `:hex` — `:residue` over the three 2-D axes `k₁, k₂, k₃` at 120°
  (`k₁ + k₂ + k₃ = 0`, [`HEX_AXES`](@ref)); requires `in_dims == 2`. The
  three axis phases satisfy `φ₁ + φ₂ + φ₃ ≡ 0`.

# Layout (how moduli and axes share the `D` components)
- `:bound` — every component carries every (modulus, axis) factor:
  `θ_j = Σ_{k,a} 2π r_{jka} x_a / m_k`. The output is the *bind* of the per-factor
  encodings (`∏_{k,a} z_{ka}(x_a)`), one dense pseudo-random address vector.
  Decoding needs the [`Resonator`](@ref) (vector form, Kymn).
- `:blocks` — the `D` components are partitioned into `K × A` blocks, one per
  (modulus, axis); each block is a separate FPE of one factor. Decodable block
  by block with a `Codebook` (channel / concatenation form, Fiete).

`random_offset = true` adds a fixed random phase `ψ_j` per component (assigned
to factor `(k = 1, a = 1)` in [`fpe_codebook`](@ref) so that the bind of the
codebook entries still reproduces the encoder output exactly).

# Residue mapping (moduli ↔ ω ↔ block)
For `moduli = [m_1, …, m_K]` and `:blocks` layout, modulus `m_k` occupies
components `blocks[k]+1 : blocks[k+1]` (near-equal partition of `D`), with
`ω = 2π r_j / m_k` rad per unit input, i.e. `2 r_j / m_k` in the library's
units of π. Query the table with [`fpe_table`](@ref).

# Shapes
Input `(in_dims, B)` (or a vector of `B` scalars when `in_dims == 1`, or
`(in_dims, L, B)` for a sequence); output `(D, B)` / `(D, L, B)` `Phase`.

# Spiking path
Phase → spike time is the library's encoder convention: follow the layer with
`MakeSpiking(spk_args, repeats)`; the spike train carries `θ_j` as the spike
time within each carrier cycle, and `v_bind` on two such trains is a spike
delay ([`spike_phase_bind`](@ref)).

# State (no trainable parameters)
- `weight::Matrix{Float32}` `(D, in_dims)` — total frequency `Ω` in units of π
- `omega::Array{Float32, 3}` `(D, A, K)` — per-(axis, modulus) frequency
- `axes::Matrix{Float32}` `(A, in_dims)` — identity or [`HEX_AXES`](@ref)
- `offset::Vector{Float32}` `(D,)` — `ψ` in units of π (zeros unless
  `random_offset`)
- `blocks::Vector{Int}` — block boundaries for `:blocks` layout (`[0, D]` for
  `:bound`)
"""
struct FPEEncoder <: Lux.AbstractLuxLayer
    in_dims::Int
    out_dims::Int
    mode::Symbol
    moduli::Vector{Int}
    bandwidth::Float32
    exponent::Symbol
    layout::Symbol
    random_offset::Bool
end

function FPEEncoder(dims::Pair{<:Int, <:Int};
                    mode::Symbol = :residue,
                    moduli::AbstractVector{<:Integer} = Int[],
                    bandwidth::Real = 1f0,
                    exponent::Symbol = :random,
                    layout::Symbol = :bound,
                    random_offset::Bool = false)
    in_dims, out_dims = dims
    mode in (:uniform, :residue, :hex) ||
        throw(ArgumentError("FPEEncoder mode must be :uniform, :residue or :hex, got :$mode"))
    exponent in (:unit, :random) ||
        throw(ArgumentError("FPEEncoder exponent must be :unit or :random, got :$exponent"))
    layout in (:bound, :blocks) ||
        throw(ArgumentError("FPEEncoder layout must be :bound or :blocks, got :$layout"))
    in_dims >= 1 || throw(ArgumentError("in_dims must be ≥ 1"))
    if mode === :hex
        in_dims == 2 || throw(ArgumentError("FPEEncoder :hex mode needs in_dims == 2, got $in_dims"))
    end
    if mode !== :uniform
        isempty(moduli) && throw(ArgumentError("FPEEncoder :$mode mode needs `moduli`"))
        all(m -> m >= 2, moduli) || throw(ArgumentError("moduli must be ≥ 2, got $moduli"))
        for i in eachindex(moduli), j in i+1:length(moduli)
            gcd(moduli[i], moduli[j]) == 1 ||
                @warn "moduli $(moduli[i]) and $(moduli[j]) are not coprime; the unique range is lcm, not the product"
        end
    end
    return FPEEncoder(in_dims, out_dims, mode, Int.(collect(moduli)), Float32(bandwidth),
                      exponent, layout, random_offset)
end

function Base.show(io::IO, e::FPEEncoder)
    print(io, "FPEEncoder($(e.in_dims) => $(e.out_dims); mode=:$(e.mode)")
    e.mode === :uniform ? print(io, ", bandwidth=$(e.bandwidth)") :
        print(io, ", moduli=$(e.moduli), exponent=:$(e.exponent)")
    print(io, ", layout=:$(e.layout))")
end

_fpe_axes(e::FPEEncoder) = e.mode === :hex ? HEX_AXES : Matrix{Float32}(I, e.in_dims, e.in_dims)
_fpe_n_axes(e::FPEEncoder) = size(_fpe_axes(e), 1)
_fpe_n_moduli(e::FPEEncoder) = e.mode === :uniform ? 1 : length(e.moduli)
_fpe_n_factors(e::FPEEncoder) = _fpe_n_moduli(e) * _fpe_n_axes(e)

# Random root-of-unity exponent r ∈ {0, …, m−1} (Kymn et al. 2024). Drawing
# from all m residues makes E[e^{2πi r Δ/m}] = 0 for every Δ ≢ 0 (mod m), so
# the m codewords of a modulus are orthogonal in expectation. Restricting r to
# the units of ℤ_m would not: for even m every unit is odd and codeword
# l + m/2 is exactly −c_l (which the `:abs` resonator projection cannot
# separate), and for composite m such as 15 codewords 5 apart have
# similarity −1/2.
_random_root(rng::AbstractRNG, m::Int) = rand(rng, 0:m-1)

Lux.initialparameters(::AbstractRNG, ::FPEEncoder) = NamedTuple()

function Lux.initialstates(rng::AbstractRNG, e::FPEEncoder)
    D = e.out_dims
    axes = _fpe_axes(e)
    A = size(axes, 1)
    K = _fpe_n_moduli(e)
    omega = zeros(Float32, D, A, K)
    # Block partition: one block per (modulus, axis) for :blocks, the whole
    # vector for :bound. Block f = k + K*(a-1) (modulus index fastest).
    n_blocks = e.layout === :blocks ? K * A : 1
    blocks = round.(Int, range(0, D, length = n_blocks + 1))
    for a in 1:A, k in 1:K
        f = k + K * (a - 1)
        rng_j = e.layout === :blocks ? (blocks[f]+1:blocks[f+1]) : (1:D)
        for j in rng_j
            if e.mode === :uniform
                # ω ~ U(−B, B) rad per unit input, stored in units of π
                omega[j, a, k] = (2f0 * rand(rng, Float32) - 1f0) * e.bandwidth / pi_f32
            else
                m = e.moduli[k]
                r = e.exponent === :unit ? 1 : _random_root(rng, m)
                omega[j, a, k] = 2f0 * Float32(r) / Float32(m)
            end
        end
    end
    offset = e.random_offset ? (2f0 .* rand(rng, Float32, D) .- 1f0) : zeros(Float32, D)
    weight = reshape(sum(omega, dims = 3), D, A) * axes      # (D, in_dims), units of π
    return (weight = weight, omega = omega, axes = axes, offset = offset, blocks = blocks)
end

function (e::FPEEncoder)(x::AbstractVector{<:Real}, ps::LuxParams, st::NamedTuple)
    e.in_dims == 1 || throw(DimensionMismatch(
        "FPEEncoder with in_dims=$(e.in_dims) needs an (in_dims, batch) input; got a vector"))
    return e(reshape(x, 1, :), ps, st)
end

function (e::FPEEncoder)(x::AbstractArray{<:Real}, ps::LuxParams, st::NamedTuple)
    size(x, 1) == e.in_dims || throw(DimensionMismatch(
        "FPEEncoder expects $(e.in_dims) input coordinates, got $(size(x, 1))"))
    xf = Float32.(reshape(x, e.in_dims, :))
    if e.mode === :uniform
        θ = st.weight * xf
    else
        # Reduce each axis projection mod m_k before multiplying by 2r/m_k:
        # exact for every real x (2r(x − n m)/m ≡ 2rx/m mod 2) and keeps the
        # Float32 phase accurate for large inputs.
        proj = st.axes * xf                                   # (A, N)
        D, A, K = size(st.omega)
        θ = zeros(Float32, D, size(xf, 2))
        for k in 1:K
            θ = θ .+ reshape(st.omega[:, :, k], D, A) * mod.(proj, Float32(e.moduli[k]))
        end
    end
    y = remap_phase(θ .+ st.offset)
    return reshape(y, e.out_dims, size(x)[2:end]...), st
end

"""
    fpe_codebook(e::FPEEncoder, st, k = 1, a = 1; levels = 0:m_k-1) -> Matrix{Phase}

`(D, length(levels))` codebook of factor `(modulus k, axis a)`: column `i` is the
encoder's contribution for input level `levels[i]` on that factor alone
(zero on all other factors). Factor `(1, 1)` carries the random offset `ψ`,
so for integer inputs `v_bind` over all factors' selected columns reproduces
the encoder output exactly. For `:uniform` mode pass continuous `levels`.
"""
function fpe_codebook(e::FPEEncoder, st::NamedTuple, k::Int = 1, a::Int = 1;
                      levels = e.mode === :uniform ? (0:0) : (0:e.moduli[k]-1))
    1 <= k <= _fpe_n_moduli(e) || throw(ArgumentError("modulus index $k out of range"))
    1 <= a <= _fpe_n_axes(e) || throw(ArgumentError("axis index $a out of range"))
    ω = st.omega[:, a, k]                                   # (D,)
    θ = ω * Float32.(collect(levels))'                        # (D, n_levels)
    if k == 1 && a == 1
        θ = θ .+ st.offset
    end
    return remap_phase(θ)
end

"""
    fpe_codebooks(e::FPEEncoder, st) -> Vector{Matrix{Phase}}

All `K × A` factor codebooks, ordered modulus-fastest (`f = k + K*(a-1)`), as
the [`Resonator`](@ref) expects. Residue modes only.
"""
function fpe_codebooks(e::FPEEncoder, st::NamedTuple)
    e.mode === :uniform && throw(ArgumentError("fpe_codebooks needs a residue-mode encoder"))
    K = _fpe_n_moduli(e)
    return [fpe_codebook(e, st, k, a) for a in 1:_fpe_n_axes(e) for k in 1:K]
end

"""
    fpe_table(e::FPEEncoder, st) -> Vector{NamedTuple}

The residue mapping table: one row per factor with `modulus`, `axis`,
`omega_rad` (`2π/m`, the base frequency in rad per unit input), `block`
(component range for `:blocks` layout; `1:D` for `:bound`) and `exponent`.
"""
function fpe_table(e::FPEEncoder, st::NamedTuple)
    K = _fpe_n_moduli(e)
    rows = NamedTuple[]
    for a in 1:_fpe_n_axes(e), k in 1:K
        f = k + K * (a - 1)
        block = e.layout === :blocks ? (st.blocks[f]+1:st.blocks[f+1]) : (1:e.out_dims)
        m = e.mode === :uniform ? NaN : e.moduli[k]
        ω = e.mode === :uniform ? e.bandwidth : 2f0 * pi_f32 / Float32(m)
        push!(rows, (factor = f, modulus = m, axis = a, omega_rad = ω, block = block,
                     exponent = e.mode === :uniform ? :uniform : e.exponent))
    end
    return rows
end

"""
    crt_decode(residues, moduli) -> Int

Chinese-remainder reconstruction: the unique `x ∈ [0, ∏ m_k)` with
`x ≡ residues[k] (mod moduli[k])` for pairwise coprime moduli. Residues are
0-based (as the codebook column index minus one).
"""
function crt_decode(residues::AbstractVector{<:Integer}, moduli::AbstractVector{<:Integer})
    length(residues) == length(moduli) || throw(DimensionMismatch("residues and moduli differ in length"))
    M = prod(Int, moduli)
    x = 0
    for (r, m) in zip(residues, moduli)
        Mk = M ÷ m
        x = mod(x + mod(r, m) * Mk * invmod(Mk, m), M)
    end
    return x
end

###
### Resonator network
###

"""
    Resonator(codebooks; iterations = 50, projection = :abs, tol = 1f-3, sequential = true)
    Resonator(e::FPEEncoder, st; kwargs...)

Resonator network (Frady, Kent, Olshausen & Sommer 2020): factor a composite
phasor vector `s = ⊙_f c_f` into one codeword `c_f` per codebook
`codebooks[f]::AbstractMatrix{<:Phase}` of shape `(D, m_f)`. Each factor
estimate `x̂_f` is refined by unbinding the current estimates of all other
factors from `s`, projecting onto its codebook by similarity, and returning
to the torus:

- `projection = :abs` (default) — `x̂_f ← ∠ C_f |C_fᴴ ê_f|`, where
  `ê_f = s ⊘ ⊙_{g≠f} x̂_g`: every codeword is bundled with weight equal to the
  *magnitude* of its complex similarity to the unbound estimate, then each
  component returns to the unit circle. The magnitude is invariant to the
  global phase the other factors' estimates carry during the search, which
  is why this cleanup (the loop of `demos/03_resonator.ipynb`) reaches far
  larger products ∏ m_f than the linear projection below.
- `projection = :linear` — `x̂_f ← ∠ C_f (C_fᴴ ê_f)`: the complex
  codebook-subspace projection followed by the phasor non-linearity, the
  update of Frady et al. 2020 (eq. 7, phasor variant) and Kymn et al. 2024.
- `projection = :codeword` — snap to the single best codeword each step
  (coordinate ascent on the reconstruction similarity; the baseline that gets
  stuck in spurious fixed points from a random start).

Updates are sequential (Gauss–Seidel, each factor sees the latest estimates
of the others) or synchronous (`sequential = false`). The loop runs at most
`iterations` sweeps and stops when every component of every estimate moves
by less than `tol` (chordal distance) over a full sweep.

# Forward
`(r::Resonator)(s::AbstractArray{<:Phase}, ps, st)` with `s` of shape `(D,)`
or `(D, B)` returns

    (indices = (F, B) Int matrix, converged = BitVector(B), iterations = Vector{Int}(B),
     similarity = Vector{Float32}(B))

and the updated state (its RNG advances: every factor starts from a random
phasor vector, because the superposition of a root-of-unity codebook is
zero). `indices[f, b]` is the codeword of factor `f` with the largest
`|⟨c, ê_f⟩|` at the last sweep;
`similarity` is the cosine similarity between `s` and the bind of the selected
codewords (1 for an exact factorisation).

A `SpikingCall` input is read out per cycle: the last carrier cycle's spike
times become phases (`train_to_phase`) and the phase path runs; see
`docs/fpe_resonator.md` for what a cycle-by-cycle spiking loop still needs.
Inference only (not differentiable).

# State
- `codes::Vector{Matrix{ComplexF32}}` — unit phasors of each codebook
- `rng` — initial-guess RNG
"""
struct Resonator <: Lux.AbstractLuxLayer
    codebooks::Vector{Matrix{Phase}}
    iterations::Int
    projection::Symbol
    tol::Float32
    sequential::Bool
end

function Resonator(codebooks::AbstractVector{<:AbstractMatrix};
                   iterations::Int = 50, projection::Symbol = :abs,
                   tol::Real = 1f-3, sequential::Bool = true)
    isempty(codebooks) && throw(ArgumentError("Resonator needs at least one codebook"))
    projection in (:abs, :linear, :codeword) ||
        throw(ArgumentError("Resonator projection must be :abs, :linear or :codeword, got :$projection"))
    iterations >= 1 || throw(ArgumentError("iterations must be ≥ 1"))
    D = size(codebooks[1], 1)
    all(cb -> size(cb, 1) == D, codebooks) ||
        throw(DimensionMismatch("all codebooks must share the vector dimension D=$D"))
    cbs = [Matrix{Phase}(Phase.(Float32.(cb))) for cb in codebooks]
    return Resonator(cbs, iterations, projection, Float32(tol), sequential)
end

Resonator(e::FPEEncoder, st::NamedTuple; kwargs...) = Resonator(fpe_codebooks(e, st); kwargs...)

function Base.show(io::IO, r::Resonator)
    sizes = [size(cb, 2) for cb in r.codebooks]
    print(io, "Resonator(D=$(size(r.codebooks[1], 1)), codebooks=$(sizes); iterations=$(r.iterations), projection=:$(r.projection))")
end

Lux.initialparameters(::AbstractRNG, ::Resonator) = NamedTuple()

function Lux.initialstates(rng::AbstractRNG, r::Resonator)
    codes = [ComplexF32.(angle_to_complex(cb)) for cb in r.codebooks]
    return (codes = codes, rng = Lux.replicate(rng))
end

# Complex similarity of every codeword (columns of C, unit phasors) with every
# column of Z: (m, B) = Cᴴ Z / D. Its real part is the library's cosine
# similarity `mean(cos(θ_c − θ_z))`; its magnitude is that similarity maximised
# over a global phase rotation of Z.
_codebook_similarity(C::AbstractMatrix{<:Complex}, Z::AbstractMatrix{<:Complex}) =
    (C' * Z) ./ Float32(size(C, 1))

_argmax_cols(S::AbstractMatrix{<:Real}) = [argmax(view(S, :, b)) for b in 1:size(S, 2)]

# Per-component return to the unit circle; a zero component falls back to the
# best codeword's phase so the estimate never leaves the torus.
function _to_torus(Y::AbstractMatrix{<:Complex}, fallback::AbstractMatrix{<:Complex})
    mag = abs.(Y)
    return ifelse.(mag .> 0f0, Y ./ max.(mag, eps(Float32)), fallback)
end

function _project(r::Resonator, C::AbstractMatrix{<:Complex}, S::AbstractMatrix{<:Complex},
                  idx::AbstractVector{Int})
    best = C[:, idx]
    if r.projection === :codeword
        return best
    elseif r.projection === :linear
        return _to_torus(C * S, best)
    else
        return _to_torus(C * abs.(S), best)
    end
end

function (r::Resonator)(s::AbstractVector{<:Phase}, ps::LuxParams, st::NamedTuple)
    return r(reshape(s, :, 1), ps, st)
end

function (r::Resonator)(s::AbstractMatrix{<:Phase}, ps::LuxParams, st::NamedTuple)
    codes = st.codes
    F = length(codes)
    D, B = size(s)
    D == size(codes[1], 1) || throw(DimensionMismatch(
        "Resonator codebooks have D=$(size(codes[1], 1)), input has D=$D"))
    rng = Lux.replicate(st.rng)
    Z = ComplexF32.(angle_to_complex(s))                   # (D, B)

    # Random phasor start for every factor; P = bind of all estimates.
    X = [ComplexF32.(angle_to_complex(random_symbols(rng, (D, B)))) for _ in 1:F]
    idx = [ones(Int, B) for _ in 1:F]
    P = reduce((a, b) -> a .* b, X)

    converged = falses(B)
    n_iter = fill(r.iterations, B)
    for it in 1:r.iterations
        moved = zeros(Float32, B)
        if r.sequential
            for f in 1:F
                others = P .* conj.(X[f])                   # bind of the other factors
                est = Z .* conj.(others)                    # unbind them from s
                S = _codebook_similarity(codes[f], est)
                idx[f] = _argmax_cols(abs.(S))
                Xf = _project(r, codes[f], S, idx[f])
                moved = max.(moved, vec(maximum(abs.(Xf .- X[f]), dims = 1)))
                X[f] = Xf
                P = others .* Xf
            end
        else
            new_X = similar(X)
            for f in 1:F
                est = Z .* conj.(P .* conj.(X[f]))
                S = _codebook_similarity(codes[f], est)
                idx[f] = _argmax_cols(abs.(S))
                new_X[f] = _project(r, codes[f], S, idx[f])
                moved = max.(moved, vec(maximum(abs.(new_X[f] .- X[f]), dims = 1)))
            end
            X = new_X
            P = reduce((a, b) -> a .* b, X)
        end
        still = moved .< r.tol
        newly = still .& .!converged
        n_iter[newly] .= it
        converged .|= still
        all(converged) && break
    end

    # Reconstruction quality from the selected codewords.
    R = reduce((a, b) -> a .* b, [codes[f][:, idx[f]] for f in 1:F])
    sim = vec(real.(sum(conj.(R) .* Z, dims = 1))) ./ Float32(D)
    indices = reduce(vcat, [reshape(idx[f], 1, B) for f in 1:F])
    out = (indices = indices, converged = converged, iterations = n_iter, similarity = sim)
    return out, (codes = codes, rng = rng)
end

function (r::Resonator)(x::SpikingCall, ps::LuxParams, st::NamedTuple)
    phases = train_to_phase(x)                             # (shape..., cycles)
    phases === missing && throw(ArgumentError("Resonator: empty spike train"))
    last = selectdim(phases, ndims(phases), size(phases, ndims(phases)))
    θ = Phase.(ifelse.(isnan.(last), 0f0, Float32.(last)))   # silent neuron → phase 0
    return r(reshape(θ, size(θ, 1), :), ps, st)
end
