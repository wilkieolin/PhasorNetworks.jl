# device_reservoir.jl — device cells on an arbitrary graph of delay lines
#
# DeviceSheet's sibling for reservoir-style networks: N cells (any
# AbstractDeviceCell, one type or a per-cell vector) connected by an ARBITRARY
# directed graph of kinetic-inductance lines — random, long-range, recurrent,
# whatever the edge list says. Each line still carries its source's output
# current to its target after a real, individually set delay:
#
#     I_in[i](t) = Σ_{l: dst(l) = i} G[l] · out[src(l)](t − D[l])
#
# STORAGE. The edges are held in compressed-sparse-row form KEYED BY RECEIVER
# (`rowptr`, then per-link `src`, `D`, `G`), so the update is the same gather as
# the sheet's: one thread per cell sums its own incoming links and writes only
# itself — no atomics, one KernelAbstractions kernel for CPU threads and GPU.
# Link order inside a row is irrelevant; the constructor sorts by (dst, src).
#
# HISTORY. Outputs go into an (N, NB) ring buffer — one column per step, so the
# per-step write is contiguous — read back at fractional delay with exactly
# DeviceSheet's split of D/dt − ½ into whole steps qi and fraction qf (see the
# device_sheet.jl header for why that is exact in Float32 and causal for
# D ≥ dt). NB = max(D)/dt + 3.
#
# MEMORY. Ring buffer 4·N·NB bytes; links 16 bytes each (src, qi, qf, G; D is
# kept on the host only); state ~6·4·N bytes. The buffer dominates at long
# delays: nTron networks run at ns delays, so 10⁵ cells with D ≤ 10 ns at
# dt = 1 ps is 4 GB of history — use dt = 2–5 ps there (the nTron's fastest
# mode is ≈15 ps). `reservoir_simulate` refuses to allocate more than
# `max_bytes` (default 16 GiB) on a unified-memory machine.
#
# SPIKES. Spike times are recorded on the device into an (N, C) chunk and
# copied to the host every C steps, so there is no per-step synchronisation;
# rate, first-spike times and counts are computed on the host afterwards.

"""
    DeviceReservoir(cell, N, src, dst; D, G = 1, dt = 1)

`N` device cells connected by directed delay lines `src[l] → dst[l]`
(1-based indices) with per-link delay `D[l]` (ps, each ≥ `dt`) and gain
`G[l]` (fraction of the source's output current; scalars broadcast). `cell`
is one `AbstractDeviceCell` for all, or a length-`N` vector of them.

Duplicate edges are kept (two parallel lines). See [`random_reservoir`](@ref)
for random graphs and [`reservoir_simulate`](@ref) to run it.
"""
struct DeviceReservoir{C}
    cell::C                       # one cell, or an (N,) vector of cells
    N::Int
    dt::Float32
    rowptr::Vector{Int32}         # (N+1,): links of receiver i are rowptr[i]:rowptr[i+1]-1
    src::Vector{Int32}            # (L,) source cell of each link
    D::Vector{Float32}            # (L,) delay, ps
    G::Vector{Float32}            # (L,) gain
    qi::Vector{Int32}             # whole steps of D/dt − ½
    qf::Vector{Float32}           # fractional part
    NB::Int                       # ring-buffer depth
end

function DeviceReservoir(cell::Union{AbstractDeviceCell, AbstractVector{<:AbstractDeviceCell}},
                         N::Integer, src::AbstractVector{<:Integer}, dst::AbstractVector{<:Integer};
                         D, G = 1.0f0, dt::Real = 1.0)
    N > 0 || throw(ArgumentError("N must be > 0"))
    dt > 0 || throw(ArgumentError("dt must be > 0"))
    cell isa AbstractVector && length(cell) != N &&
        throw(DimensionMismatch("cell vector must have length $N, got $(length(cell))"))
    L = length(src)
    length(dst) == L || throw(DimensionMismatch("src and dst lengths differ"))
    Dv = D isa Real ? fill(Float32(D), L) : Float32.(collect(D))
    Gv = G isa Real ? fill(Float32(G), L) : Float32.(collect(G))
    length(Dv) == L && length(Gv) == L || throw(DimensionMismatch("D and G must have one entry per link"))
    L == 0 || (all(i -> 1 <= i <= N, src) && all(i -> 1 <= i <= N, dst)) ||
        throw(ArgumentError("link endpoints must lie in 1:$N"))
    L == 0 || minimum(Dv) >= dt ||
        throw(ArgumentError("every delay must be ≥ dt = $dt ps; min(D) = $(minimum(Dv))"))
    p = sortperm(collect(zip(dst, src)))
    s, d, Dv, Gv = Int32.(src[p]), Int32.(dst[p]), Dv[p], Gv[p]
    rowptr = zeros(Int32, N + 1)
    rowptr[1] = 1
    for i in d
        rowptr[i+1] += 1
    end
    cumsum!(rowptr, rowptr)
    q  = Float64.(Dv) ./ dt .- 0.5
    qi = floor.(Int32, q)
    qf = Float32.(q .- qi)
    NB = (L == 0 ? 0 : Int(maximum(qi))) + 3
    return DeviceReservoir(cell, Int(N), Float32(dt), rowptr, s, Dv, Gv, qi, qf, NB)
end

n_links(r::DeviceReservoir) = length(r.src)
Base.length(r::DeviceReservoir) = r.N

"""
    in_degree(res) / out_degree(res) -> Vector{Int}

Number of lines into / out of each cell.
"""
in_degree(r::DeviceReservoir) = Int.(diff(r.rowptr))
out_degree(r::DeviceReservoir) = (o = zeros(Int, r.N); for j in r.src; o[j] += 1; end; o)

"""
    reservoir_bytes(res; chunk = nothing) -> Int

Device memory `reservoir_simulate` will allocate (history buffer, links, state,
spike chunk), in bytes.
"""
function reservoir_bytes(r::DeviceReservoir; chunk = nothing)
    C = chunk === nothing ? _spike_chunk(r.N) : chunk
    return 4 * r.N * r.NB + 16 * n_links(r) + 4 * (r.N + 1) + 4 * 8 * r.N + 4 * r.N * C
end

_spike_chunk(N) = clamp(2^24 ÷ (4 * N), 1, 512)          # ≤ 16 MB per chunk

function Base.show(io::IO, r::DeviceReservoir)
    L = n_links(r)
    print(io, "DeviceReservoir(N=", r.N, ", L=", L, " links (⟨k_in⟩=",
          round(L / r.N; sigdigits = 3), "), dt=", r.dt, " ps, D∈[",
          L == 0 ? "–" : round(minimum(r.D); digits = 1), ", ",
          L == 0 ? "–" : round(maximum(r.D); digits = 1), "] ps; ",
          r.cell isa AbstractVector ? "$(length(unique(r.cell))) cell types" : r.cell, ")")
end

"""
    random_reservoir(cell, N; p = nothing, k_in = nothing, delay = (100, 1000),
                     gain = 1, fanout = :none, self_loops = false,
                     rng = Random.default_rng(), dt = 1) -> DeviceReservoir

A random directed graph of delay lines on `N` cells. Give exactly one of
- `p` — each ordered pair is linked independently with probability `p`
  (Erdős–Rényi), or
- `k_in` — every cell receives exactly `k_in` lines from distinct random sources.

`delay` (ps) is a scalar, a `(lo, hi)` uniform range, or a function
`(rng, i, j) -> D` of source `i` and target `j` (e.g. distance on a layout).
`gain` is a scalar, a `(lo, hi)` uniform range, or a function `(rng, i, j)`.
`fanout = :split` divides each link's gain by its source's out-degree — the
nTron transponder's output current is shared between its lines, so this is
the physical choice for [`NTronTransponder`](@ref) networks.
"""
function random_reservoir(cell, N::Integer; p = nothing, k_in = nothing, delay = (100, 1000),
                          gain = 1, fanout::Symbol = :none, self_loops::Bool = false,
                          rng::AbstractRNG = Random.default_rng(), dt::Real = 1.0)
    (p === nothing) != (k_in === nothing) || throw(ArgumentError("give exactly one of p or k_in"))
    fanout in (:none, :split) || throw(ArgumentError("fanout must be :none or :split"))
    src, dst = Int32[], Int32[]
    if p !== nothing
        0 <= p <= 1 || throw(ArgumentError("p must be in [0, 1]"))
        for j in 1:N, i in 1:N
            (i != j || self_loops) && rand(rng) < p && (push!(src, i); push!(dst, j))
        end
    else
        nsrc = self_loops ? N : N - 1
        0 <= k_in <= nsrc || throw(ArgumentError("k_in must be in 0:$nsrc"))
        for j in 1:N
            pool = self_loops ? collect(1:N) : [i for i in 1:N if i != j]
            for i in Random.randperm(rng, length(pool))[1:k_in]
                push!(src, pool[i]); push!(dst, j)
            end
        end
    end
    draw(spec, i, j) = spec isa Function ? spec(rng, i, j) :
                       spec isa Tuple ? spec[1] + (spec[2] - spec[1]) * rand(rng) : spec
    D = Float32[draw(delay, i, j) for (i, j) in zip(src, dst)]
    G = Float32[draw(gain, i, j) for (i, j) in zip(src, dst)]
    if fanout === :split
        od = zeros(Int, N)
        for i in src
            od[i] += 1
        end
        G ./= od[src]
    end
    return DeviceReservoir(cell, N, src, dst; D, G, dt)
end

# ---- Coupling gather -----------------------------------------------------

@kernel function _reservoir_gather_kernel!(Iin, @Const(buf), @Const(rowptr), @Const(src),
                                           @Const(G), @Const(qi), @Const(qf), n::Int, NB::Int)
    i = @index(Global)
    acc = 0f0
    for l in rowptr[i]:(rowptr[i+1] - Int32(1))
        j  = src[l]
        m1 = n - qi[l]                                   # newer sample
        f  = qf[l]
        acc += G[l] * (f * buf[j, mod(m1 - 1, NB) + 1] + (1f0 - f) * buf[j, mod(m1, NB) + 1])
    end
    @inbounds Iin[i] = acc
end

"""
    reservoir_input!(Iin, res, buf, n, links)

Coupled input current at step `n`, gathered from the history `buf` over each
cell's incoming links. `links` is `(; rowptr, src, G, qi, qf)` on `buf`'s device.
"""
function reservoir_input!(Iin, r::DeviceReservoir, buf, n::Integer, links)
    backend = get_backend(Iin)
    _reservoir_gather_kernel!(backend)(Iin, buf, links.rowptr, links.src, links.G, links.qi,
                                       links.qf, Int(n), r.NB; ndrange = r.N)
    return Iin
end

# ---- Rollout ---------------------------------------------------------------

"""
    reservoir_simulate(res, nsteps; drive = nothing, forced = nothing, s0 = nothing,
                       probe = Int[], probe_every = 1, device = cpu_device(),
                       max_bytes = 2^34)
        -> (; spike_index, spike_time, t_first, n_spikes, rate, probe_t, probes, state)

Roll the reservoir out for `nsteps` steps of `res.dt` ps.

Inputs (either or both):
- `drive(t)` — external input current `(N,)` in µA at time `t`, sampled at
  step midpoints and added to the coupled input (for an nTron cell this is
  input-gate current).
- `forced = (cells, times)` — make `cells[k]` fire at `times[k]` (ps)
  regardless of its state: detector-driven sensor cells, as in the reference
  toy network. Each event lands in the step containing its time, at its exact
  sub-step offset.

Returns, on the host:
- `spike_index`, `spike_time` — every spike (cell, ps), in time order.
- `t_first` `(N,)` first-spike time (`Inf32` if none), `n_spikes` `(N,)`,
  `rate` `(nsteps,)` fraction of cells firing per step.
- with `probe` (cell indices): `probes`, a NamedTuple of `(length(probe), F)`
  matrices — every state field of those cells, every `probe_every` steps (each
  record synchronises the device) — and their times `probe_t` (step ends).
- `state` — the final state.
"""
function reservoir_simulate(r::DeviceReservoir, nsteps::Integer; drive = nothing, forced = nothing,
                            s0 = nothing, probe = Int[], probe_every::Integer = 1,
                            device = cpu_device(), max_bytes::Integer = 2^34)
    N, dt, NB = r.N, r.dt, r.NB
    C = _spike_chunk(N)
    need = reservoir_bytes(r; chunk = C)
    need <= max_bytes || throw(ArgumentError(
        "reservoir needs $(round(need / 2^30; digits = 2)) GiB (history $(N)×$(NB)); " *
        "raise dt, shorten delays, or pass a larger max_bytes deliberately"))
    cell = r.cell
    stp = stepper_to(cell_stepper(cell, dt), device)
    c1 = cell isa AbstractVector ? first(cell) : cell
    s = device(s0 === nothing ? (cell isa AbstractVector ? cell_state(cell) : cell_state(cell, N)) : s0)
    links = device((; rowptr = r.rowptr, src = r.src, G = r.G, qi = r.qi, qf = r.qf))
    buf = device(zeros(Float32, N, NB))
    buf[:, 1] .= cell_output(c1, s)
    Iin = device(zeros(Float32, N))
    TF = device(fill(-1f0, N, C))

    # forced events, bucketed by step
    fev = Dict{Int, Vector{Tuple{Int, Float32}}}()
    if forced !== nothing
        fc, ft = forced
        length(fc) == length(ft) || throw(DimensionMismatch("forced cells and times differ in length"))
        for (c, t) in zip(fc, ft)
            1 <= c <= N || throw(ArgumentError("forced cell $c outside 1:$N"))
            n = floor(Int, t / dt)
            0 <= n < nsteps && push!(get!(fev, n, Tuple{Int, Float32}[]), (c, Float32(t - n * dt)))
        end
    end
    fmask_h = zeros(Bool, N); ftime_h = zeros(Float32, N)

    pidx = collect(Int, probe)
    nP = isempty(pidx) ? 0 : nsteps ÷ probe_every
    pidx_d = device(pidx)
    probes = nothing
    probe_t = Vector{Float32}(undef, nP)

    sp_i, sp_t = Int32[], Float32[]
    flush!(c0, ncols) = begin
        h = Array(TF)
        for k in 1:ncols, i in 1:N
            @inbounds h[i, k] >= 0 && (push!(sp_i, i); push!(sp_t, h[i, k]))
        end
        TF .= -1f0
    end

    col = 0
    for n in 0:nsteps-1
        tn = n * dt
        reservoir_input!(Iin, r, buf, n, links)
        I = drive === nothing ? Iin : Iin .+ device(drive(tn + dt / 2))
        if haskey(fev, n)
            fill!(fmask_h, false)
            for (c, τ) in fev[n]
                fmask_h[c] = true; ftime_h[c] = τ
            end
            s, fire, tf = cell_step(stp, s, I; force = device(fmask_h), t_force = device(ftime_h))
        else
            s, fire, tf = cell_step(stp, s, I)
        end
        buf[:, mod(n + 1, NB) + 1] .= cell_output(c1, s)
        col += 1
        view(TF, :, col) .= ifelse.(fire, Float32(tn) .+ tf, -1f0)
        if col == C
            flush!(n, col); col = 0
        end
        if nP > 0 && (n + 1) % probe_every == 0
            k = (n + 1) ÷ probe_every
            rec = map(x -> Array(x[pidx_d]), s)
            probes === nothing && (probes = map(v -> Matrix{Float32}(undef, length(pidx), nP), rec))
            foreach((P, v) -> (P[:, k] .= v), probes, rec)
            probe_t[k] = (n + 1) * dt
        end
    end
    col > 0 && flush!(nsteps - 1, col)

    p = sortperm(sp_t)                                    # chunks are in step order; this orders within a step
    sp_i, sp_t = sp_i[p], sp_t[p]
    t_first = fill(Inf32, N); nsp = zeros(Int, N); rate = zeros(Float32, nsteps)
    for (i, t) in zip(sp_i, sp_t)
        nsp[i] += 1
        t < t_first[i] && (t_first[i] = t)
        rate[clamp(floor(Int, t / dt) + 1, 1, nsteps)] += 1f0 / N
    end
    return (; spike_index = sp_i, spike_time = sp_t, t_first, n_spikes = nsp, rate,
              probe_t, probes, state = map(Array, s))
end
