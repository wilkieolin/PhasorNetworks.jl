# Tests for device_sheet.jl — DeviceSheet, per-link delay lines, detector drives.
# Entry point: device_sheet_tests(), called from runtests.jl.
#
# Two groups. The link tests pin the delay-line convention exactly: a known
# history read through `sheet_input!` must come back at each link's own
# fractional delay, and editing ONE link's delay must shift everything
# downstream of it by exactly that amount. The medium tests are the four
# excitability criteria `src/excitable.jl` uses for the phasor sheet; measured
# values (128², defaults) are in the comments and the thresholds are loose.

const _SH_CELL = RFTransponder()

# A linear ramp in the history is reproduced exactly by linear interpolation, so
# every link must return G·(t_n + dt/2 − D) for its own D — integer or not.
function test_sheet_link_convention()
    H, W = 7, 7
    sh = DeviceSheet(_SH_CELL, H, W; radius = 2, dt = 1.0)
    K = n_links(sh)
    rng = Xoshiro(5)
    D = 1.0f0 .+ 20f0 .* rand(rng, Float32, H, W, K)      # arbitrary, fractional, ≥ dt
    G = rand(rng, Float32, H, W, K)
    sh = DeviceSheet(_SH_CELL, H, W; radius = 2, dt = 1.0, D = D, G = G)
    # history: source (y,x) emitted out(t) = a[y,x]·t + b[y,x] at every stored sample
    a = rand(rng, Float32, H, W); b = rand(rng, Float32, H, W)
    n = 40
    buf = zeros(Float32, H, W, sh.NB)
    for m in (n - sh.NB + 1):n
        buf[:, :, mod(m, sh.NB) + 1] .= a .* (m * sh.dt) .+ b
    end
    links = (; G = sh.G, qi = sh.qi, qf = sh.qf, offs = sh.offsets)
    Iin = sheet_input!(zeros(Float32, H, W), sh, buf, n, links)
    t_mid = (n + 0.5f0) * sh.dt
    ref = zeros(Float64, H, W)
    for x in 1:W, y in 1:H, k in 1:K
        sy, sx = y + sh.offsets[1, k], x + sh.offsets[2, k]
        (1 <= sy <= H && 1 <= sx <= W) || continue           # open boundary: no line
        ref[y, x] += G[y, x, k] * (a[sy, sx] * (t_mid - D[y, x, k]) + b[sy, sx])
    end
    @test maximum(abs.(Iin .- ref)) < 1e-4 * maximum(abs.(ref))
    # periodic wraps instead of dropping
    shp = DeviceSheet(_SH_CELL, H, W; radius = 2, D = D, G = G, boundary = :periodic)
    Ip = sheet_input!(zeros(Float32, H, W), shp, buf, n, (; G = shp.G, qi = shp.qi, qf = shp.qf, offs = shp.offsets))
    @test sum(Ip) > sum(Iin)
    @test_throws ArgumentError DeviceSheet(_SH_CELL, H, W; D = fill(0.5f0, H, W, length(stencil_offsets(3))))
    @test_throws DimensionMismatch DeviceSheet(_SH_CELL, H, W; G = ones(Float32, H, W, 2))
end

# A 1×N chain with radius-1 links and gain 1 (one input fires a cell): the
# front relays cell to cell. Lengthening the single line into cell j by δ ps
# must delay every cell from j on by exactly δ, and nothing before it.
function test_sheet_per_link_delay()
    Nc = 24
    base = DeviceSheet(_SH_CELL, 1, Nc; radius = 1, gain = 1.0)
    drive = EventDrive(Float32[x == 1 ? 0f0 : Inf32 for y in 1:1, x in 1:Nc], 40)
    t0 = sheet_simulate(base, 1500; drive).t_first
    @test all(isfinite, t0)
    @test all(diff(vec(t0)) .> 0)                         # strictly relayed left → right
    j, δ = 12, 7.3f0
    D = copy(base.D)
    kL = findfirst(k -> base.offsets[:, k] == [0, -1], 1:n_links(base))   # line from the left
    D[1, j, kL] += δ
    t1 = sheet_simulate(DeviceSheet(_SH_CELL, 1, Nc; radius = 1, gain = 1.0, D = D), 1500; drive).t_first
    @test maximum(abs.(t1[1, 1:j-1] .- t0[1, 1:j-1])) < 1e-3
    @test maximum(abs.(t1[1, j:end] .- t0[1, j:end] .- δ)) < 0.1
end

# (1) Quiescence + critical nucleus: a single fired cell does not ignite the
# sheet, a radius-1 seed does, and isolated random hits never nucleate.
function test_sheet_nucleus_and_quiescence()
    N = 64
    sh = DeviceSheet(_SH_CELL, N, N)
    one = sheet_simulate(sh, 600; drive = seed_drive(N, N; radius = 0))
    @test count(isfinite, one.t_first) == 1
    five = sheet_simulate(sh, 1600; drive = seed_drive(N, N; radius = 1))
    @test count(isfinite, five.t_first) == N^2
    rng = Xoshiro(3)
    th = Float32[rand(rng) < 0.01 ? 1000 * rand(rng) : Inf32 for y in 1:N, x in 1:N]
    noisy = sheet_simulate(sh, 1600; drive = EventDrive(th, 40))
    @test count(isfinite, noisy.t_first) == count(isfinite, th)   # measured: 0 extra at 1% and 3%
end

# (2) Front speed is a property of the medium: invariant to drive amplitude and
# seed size (measured 0.04752 sites/ps at every amp 25–400 µA, seed r 1–4), and
# round with the radius-3 stencil (diagonal/axis measured 0.998).
function test_sheet_speed_invariance()
    N = 80; cen = (N ÷ 2, N ÷ 2)
    sh = DeviceSheet(_SH_CELL, N, N)
    us = Float32[]
    for (amp, r) in ((25, 4), (40, 1), (400, 2))
        tr = sheet_simulate(sh, 1400; drive = seed_drive(N, N; radius = r, amp = amp))
        push!(us, arrival_speed(tr.t_first; center = cen, rmin = 10, rmax = 34)[1])
    end
    @test all(isfinite, us)
    @test (maximum(us) - minimum(us)) / mean(us) < 0.01
    @test 0.035 < mean(us) < 0.06
    tr = sheet_simulate(sh, 1400; drive = seed_drive(N, N; radius = 2))
    an = front_anisotropy(tr.t_first; center = cen, rmin = 10, rmax = 34)
    @test 0.95 < an.ratio < 1.05
end

# (3) Colliding fronts annihilate and the sheet returns to rest — including a
# front meeting its own wrapped copy on a torus. Every site fires exactly once.
function test_sheet_annihilation()
    N = 64
    d2 = seed_drive(N, N; center = (N ÷ 2, N ÷ 4), radius = 2) +
         seed_drive(N, N; center = (N ÷ 2, 3N ÷ 4), radius = 2)
    tr = sheet_simulate(DeviceSheet(_SH_CELL, N, N), 2000; drive = d2)
    @test count(isfinite, tr.t_first) == N^2
    @test maximum(tr.n_spikes) == 1
    @test all(iszero, tr.rate[end-300:end])
    trp = sheet_simulate(DeviceSheet(_SH_CELL, N, N; boundary = :periodic), 2500;
                         drive = seed_drive(N, N; radius = 2))
    @test maximum(trp.n_spikes) == 1
    @test all(iszero, trp.rate[end-300:end])
end

# (4) The delay knob sets the medium speed: u ≈ 1/(ps_per_site + hop latency),
# monotone, and always below the line-only bound 1/ps_per_site.
function test_sheet_delay_sets_speed()
    N = 64; cen = (N ÷ 2, N ÷ 2)
    us = map((5, 20)) do pps
        tr = sheet_simulate(DeviceSheet(_SH_CELL, N, N; ps_per_site = pps), 2400;
                            drive = seed_drive(N, N; radius = 2))
        arrival_speed(tr.t_first; center = cen, rmin = 8, rmax = 26)[1]
    end
    @test us[1] > us[2]
    @test us[1] < 1 / 5 && us[2] < 1 / 20
end

function test_sheet_particle_drive_and_frames()
    N = 48
    d = particle_drive(N, N; dt_site = 10, angle = 0.0, x0 = (1, N ÷ 2), width = 3)
    hit = isfinite.(d.t_hit)
    @test all(hit[:, N ÷ 2 - 1:N ÷ 2 + 1])               # 3-wide track along increasing row
    @test !any(hit[:, 1:N ÷ 2 - 2])
    @test d.t_hit[11, N ÷ 2] ≈ 100f0                      # 10 rows at 10 ps/site
    @test d(d.t_hit[11, N ÷ 2] + d.tau)[11, N ÷ 2] ≈ d.amp[11, N ÷ 2]   # alpha peak at τ
    dj = particle_drive(N, N; dt_site = 10, width = 3, jitter_fwhm_ps = 20, dropout_frac = 0.5,
                        rng = Xoshiro(1))
    @test 0.3 < count(isfinite, dj.t_hit) / count(hit) < 0.7
    tr = sheet_simulate(DeviceSheet(_SH_CELL, N, N), 600; drive = d, frame_every = 50,
                        keep_amplitude = true)
    @test size(tr.fire_frames) == (N, N, 12) && size(tr.amp_frames) == (N, N, 12)
    @test tr.frame_t[end] ≈ 600f0
    @test all(sum(tr.fire_frames; dims = 3) .>= (tr.n_spikes .> 0))
    ll = link_lengths(DeviceSheet(_SH_CELL, N, N))
    @test ll.total_m > 0 && size(ll.per_site_mm) == (N, N)
    @test maximum(ll.per_site_mm) > minimum(ll.per_site_mm)  # edge sites have fewer lines
end

function device_sheet_tests()
    @testset "DeviceSheet" begin
        @info "Running DeviceSheet tests..."
        test_sheet_link_convention()
        test_sheet_per_link_delay()
        test_sheet_nucleus_and_quiescence()
        test_sheet_speed_invariance()
        test_sheet_annihilation()
        test_sheet_delay_sets_speed()
        test_sheet_particle_drive_and_frames()
    end
end
