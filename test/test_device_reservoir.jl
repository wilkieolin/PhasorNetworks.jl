# Tests for device_reservoir.jl — DeviceReservoir, random graphs, rollout.
# Entry point: device_reservoir_tests(), called from runtests.jl.
#
# The gather is pinned exactly (a linear ramp in the history must come back at
# each link's own fractional delay); the engine is cross-checked against
# DeviceSheet by building a reservoir from a sheet's own links; and the nTron
# path is checked against the reference toy network (transponder_model's CI
# job `toy-tests`), whose click times are hard-coded below (from the
# reference harness in the sibling experiments dir ../reservoir_scaleup).

const _RS_CELL = RFTransponder()

function test_reservoir_construction()
    src = [1, 3, 2, 1, 3]; dst = [2, 2, 3, 3, 1]
    r = DeviceReservoir(_RS_CELL, 3, src, dst; D = [10, 20, 30, 40, 50], G = 0.5)
    @test n_links(r) == 5 && length(r) == 3
    @test in_degree(r) == [1, 2, 2] && out_degree(r) == [2, 1, 2]
    @test r.rowptr == Int32[1, 2, 4, 6]
    # each receiver's links are its own, with their delays carried along
    for i in 1:3, l in r.rowptr[i]:r.rowptr[i+1]-1
        k = findfirst(m -> src[m] == r.src[l] && dst[m] == i, eachindex(src))
        @test r.D[l] == [10, 20, 30, 40, 50][k]
    end
    @test r.NB == floor(Int, 50 - 0.5) + 3
    @test occursin("DeviceReservoir", sprint(show, r))
    @test_throws ArgumentError DeviceReservoir(_RS_CELL, 3, [1], [2]; D = 0.5)       # D < dt
    @test_throws ArgumentError DeviceReservoir(_RS_CELL, 3, [1], [4]; D = 10)        # endpoint
    @test_throws DimensionMismatch DeviceReservoir([_RS_CELL], 3, [1], [2]; D = 10)
    @test_throws ArgumentError reservoir_simulate(r, 10; max_bytes = 10)
    @test reservoir_bytes(r) > 4 * 3 * r.NB
end

# A linear ramp in the history is reproduced exactly by linear interpolation, so
# every receiver must get Σ G·(a_src·(t_n + dt/2 − D) + b_src) over its links.
function test_reservoir_link_convention()
    rng = Xoshiro(3)
    N, L = 9, 40
    src = rand(rng, 1:N, L); dst = rand(rng, 1:N, L)
    D = 1 .+ 30 .* rand(rng, L); G = rand(rng, L)
    r = DeviceReservoir(_RS_CELL, N, src, dst; D, G, dt = 1.0)
    a = rand(rng, Float32, N); b = rand(rng, Float32, N)
    n = 60
    buf = zeros(Float32, N, r.NB)
    for m in (n - r.NB + 1):n
        buf[:, mod(m, r.NB) + 1] .= a .* m .+ b
    end
    links = (; rowptr = r.rowptr, src = r.src, G = r.G, qi = r.qi, qf = r.qf)
    Iin = reservoir_input!(zeros(Float32, N), r, buf, n, links)
    ref = zeros(N)
    for l in 1:L
        ref[dst[l]] += G[l] * (a[src[l]] * (n + 0.5 - D[l]) + b[src[l]])
    end
    @test maximum(abs.(Iin .- ref)) < 1e-4 * maximum(abs.(ref))
end

# The same medium two ways: a reservoir built from a periodic DeviceSheet's own
# links (receiver (y,x) ← source (y+dy, x+dx), linear index y + (x−1)H) must
# light up the same sites at the same times.
function test_reservoir_matches_sheet()
    H = W = 20
    rng = Xoshiro(11)
    sh0 = DeviceSheet(_RS_CELL, H, W; radius = 2, boundary = :periodic)
    K = n_links(sh0)
    D = sh0.D .* (0.8f0 .+ 0.4f0 .* rand(rng, Float32, H, W, K))
    G = fill(0.45f0, H, W, K)
    sh = DeviceSheet(_RS_CELL, H, W; radius = 2, D, G, boundary = :periodic)
    li(y, x) = y + (x - 1) * H
    src = Int[]; dst = Int[]; Dl = Float32[]; Gl = Float32[]
    for x in 1:W, y in 1:H, k in 1:K
        push!(src, li(mod1(y + sh.offsets[1, k], H), mod1(x + sh.offsets[2, k], W)))
        push!(dst, li(y, x)); push!(Dl, D[y, x, k]); push!(Gl, G[y, x, k])
    end
    r = DeviceReservoir(_RS_CELL, H * W, src, dst; D = Dl, G = Gl)
    drv = seed_drive(H, W; radius = 1, amp = 60)
    ts = sheet_simulate(sh, 800; drive = drv)
    tr = reservoir_simulate(r, 800; drive = t -> vec(drv(t)))
    @test isfinite.(vec(ts.t_first)) == isfinite.(tr.t_first)
    @test count(isfinite, tr.t_first) > H * W ÷ 2                    # a front actually crossed it
    fin = isfinite.(tr.t_first)
    @test maximum(abs.(vec(ts.t_first)[fin] .- tr.t_first[fin])) < 1e-2
    @test vec(ts.n_spikes) == tr.n_spikes
    @test issorted(tr.spike_time) && length(tr.spike_time) == sum(tr.n_spikes)
    @test sum(tr.rate) * H * W ≈ length(tr.spike_time)
end

function test_random_reservoir()
    N = 200
    r = random_reservoir(_RS_CELL, N; k_in = 7, delay = (20, 80), gain = (0.1, 0.3), rng = Xoshiro(1))
    @test all(==(7), in_degree(r))
    @test all(i -> !any(r.src[r.rowptr[i]:r.rowptr[i+1]-1] .== i), 1:N)        # no self-loops
    @test all(l -> 20 <= l <= 80, r.D) && all(g -> 0.1 <= g <= 0.3, r.G)
    for i in 1:N                                                               # distinct sources per cell
        s = r.src[r.rowptr[i]:r.rowptr[i+1]-1]
        @test length(unique(s)) == length(s)
    end
    rp = random_reservoir(_RS_CELL, N; p = 0.05, delay = 50, rng = Xoshiro(2))
    @test n_links(rp) ≈ 0.05 * N * (N - 1) rtol = 0.1
    @test all(==(50f0), rp.D)
    # fan-out splitting: each source's lines share its output current
    rs = random_reservoir(_RS_CELL, N; p = 0.05, gain = 1, fanout = :split, rng = Xoshiro(2))
    tot = zeros(N); foreach((j, g) -> tot[j] += g, rs.src, rs.G)
    @test all(i -> out_degree(rs)[i] == 0 || isapprox(tot[i], 1; atol = 1e-5), 1:N)   # Float32 gains
    # reproducible, and distance-dependent delays through a function
    pos = rand(Xoshiro(9), 2, N)
    rd = random_reservoir(_RS_CELL, N; k_in = 3, rng = Xoshiro(4),
                          delay = (rng, i, j) -> 10 + 500 * hypot(pos[1, i] - pos[1, j], pos[2, i] - pos[2, j]))
    rd2 = random_reservoir(_RS_CELL, N; k_in = 3, rng = Xoshiro(4),
                           delay = (rng, i, j) -> 10 + 500 * hypot(pos[1, i] - pos[1, j], pos[2, i] - pos[2, j]))
    @test rd.src == rd2.src && rd.D == rd2.D
    @test_throws ArgumentError random_reservoir(_RS_CELL, N)
    @test_throws ArgumentError random_reservoir(_RS_CELL, N; p = 0.1, k_in = 2)
end

# The reference toy network: sensors 1, 2 → transponders 3…8 (toy_example.py's
# delay matrix), I1_sw = 13.5, I2_sw = 24, split = 0.3. Reference OUTPUT click
# times (ns) for two of its CI configurations: at dt = 15 ns node 6 is silent;
# raising node 3's and node 4's output biases fires it. Our timing differs by
# ≤ 57 ps at node 5/6, where the loop current crosses threshold at a grazing
# slope (≈0.05 µA/ns, so the two models' 0.1% current difference is ~25 ps) —
# ../reservoir_scaleup/ntron_toy_validation/ has the full comparison.
const _TOY_LINKS = [(1, 3, 5.0), (1, 7, 9.0), (2, 4, 5.0), (2, 8, 5.0), (3, 5, 5.0), (4, 5, 5.0), (5, 6, 5.0)]
const _TOY_CASES = [
    (ib1 = Dict(3 => 13.0, 5 => 11.0), ib2 = Dict(5 => 22.6),
     out = Dict(3 => [10.657], 4 => [27.891], 5 => Float64[], 6 => Float64[], 7 => [16.908], 8 => [27.891])),
    (ib1 = Dict(3 => 13.0, 5 => 11.0), ib2 = Dict(5 => 22.6, 3 => 23.4, 4 => 22.95),
     out = Dict(3 => [6.636], 4 => [29.846], 5 => [36.716], 6 => [48.380], 7 => [16.908], 8 => [27.891])),
]

function test_reservoir_ntron_toy_network(; dt = 2.0)
    for case in _TOY_CASES
        cells = [NTronTransponder(Ib1 = get(case.ib1, i, 12.0), I1_sw = 13.5, Ib2 = get(case.ib2, i, 23.0),
                                  I2_sw = 24, split = 0.3) for i in 1:8]
        src = first.(_TOY_LINKS); dst = getindex.(_TOY_LINKS, 2)
        od = zeros(Int, 8); foreach(i -> od[i] += 1, src)
        r = DeviceReservoir(cells, 8, src, dst; D = [1000 * l[3] - cells[1].t_lead for l in _TOY_LINKS],
                            G = [1 / od[i] for i in src], dt)
        out = reservoir_simulate(r, round(Int, 100_000 / dt); forced = ([1, 2], [990.0, 15_990.0]))
        @test out.spike_time[out.spike_index .== 1] ≈ [990f0] atol = 1e-2
        for (i, ref) in case.out
            ours = out.spike_time[out.spike_index .== i] ./ 1000
            @test length(ours) == length(ref)
            length(ours) == length(ref) && @test all(abs.(ours .- ref) .< 0.1)
        end
    end
end

# Probes record every state field of the chosen cells; forced spikes outside the
# run are dropped; a random nTron reservoir with fan-out splitting runs end to end.
function test_reservoir_probes_and_random_ntron()
    N = 300
    r = random_reservoir(NTronTransponder(), N; k_in = 4, delay = (2000, 8000), fanout = :split,
                         rng = Xoshiro(7), dt = 5.0)
    out = reservoir_simulate(r, 4_000; forced = (collect(1:20), fill(500.0, 20)),
                             drive = t -> fill(3.9f0, N), probe = [1, 50], probe_every = 10)
    @test keys(out.probes) == (:inw, :iloop, :age_in, :age_out, :tail, :out)
    @test size(out.probes.iloop) == (2, 400) && out.probe_t[end] == 20_000f0
    @test 1 in out.spike_index
    @test all(isfinite, out.t_first[1:20])
    @test issorted(out.spike_time)
    @test out.probes.out[1, 30] > 0                                        # cell 1 was made to emit
    late = reservoir_simulate(r, 10; forced = ([1], [1e9]))
    @test isempty(late.spike_time)
end

function device_reservoir_tests()
    @testset "DeviceReservoir" begin
        @info "Running DeviceReservoir tests..."
        test_reservoir_construction()
        test_reservoir_link_convention()
        test_reservoir_matches_sheet()
        test_random_reservoir()
        test_reservoir_ntron_toy_network()
        test_reservoir_probes_and_random_ntron()
    end
end
