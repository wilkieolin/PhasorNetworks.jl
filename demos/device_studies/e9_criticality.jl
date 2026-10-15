# E9 — pilot: criticality of the wavesheet, damage, and a local repair rule.
#
# WHY THERE IS NOISE. Without it the sheet is all-or-none: below the
# propagation threshold a single spike fires no neighbour (a cell needs ~4
# coincident inputs at G ≤ 0.3), so every avalanche has size 1, and above it
# a nucleus sweeps the whole sheet. The probability that one spike fires a
# neighbour jumps from 0 to 1 and there is no critical point to find (measured
# first: ⟨s⟩ = 1.0 at G = 0.08 and 0.10 under sparse single-site seeds). A
# branching ratio becomes continuous only when cells sit noisily below
# threshold, as cortical neurons do. Here each tank receives a white
# background current (a stand-in for thermal or bias noise) scaled so its loop
# current rings at ~0.28·I_th rms; the rare noise-driven crossings are the
# spontaneous seeds.
#
# Measurement. Avalanche sizes need the activity to fall silent between
# cascades, which a noisy sheet rarely does (first noisy run: ~370 spontaneous
# spikes per ns sheet-wide, so 2–13 "avalanches" per run). The branching ratio
# is therefore taken from the activity series itself: with A_k = spikes in 25 ps
# bin k (about one hop), A_{k+1} = m·A_k + h for a driven branching process, and
# the regression slope of A_{k+1} on A_k estimates m even when cascades overlap
# (the multistep-regression idea of Wilting & Priesemann, at lag 1, fully
# sampled). Swept over line gain G at two background levels.
#
# Damage and repair. At the lowest gain whose noisy sheet sustains its own
# activity, 30% of cells are killed (no firing, no relaying). Survivors then
# adapt their input sensitivity toward the intact sheet's activity, block by
# block (8×8). Scaling a cell's input gains by a is exactly a threshold change
# I_th → I_th/a for a linear subthreshold tank — in hardware, the nanowire
# bias current.

include("common.jl")
using Plots
using Statistics: cov, var
backend, dev = load_gpu_backend()

const N = 160; const BIN = 20
cell = RFTransponder()
offs = stencil_offsets(3); K = length(offs)

# ---- noise calibration: per-step current σ giving i_rms = 0.28·I_th --------
function noise_sigma(TARGET)
    quiet = RFTransponder(cell.L, cell.C, cell.R, 1f9, cell.t_ref, cell.I_out, cell.tau_out)
    rng = Xoshiro(1)
    tr = cell_simulate(quiet, t -> randn(rng, Float32, 32, 32), 1.0, 6000; dims = (32, 32), stride = 10)
    irms = sqrt(mean(abs2, tr.i[:, :, 200:end]))
    return TARGET * cell.I_th / irms
end
const LEVELS = (0.20, 0.28)
const SIG = Dict(l => noise_sigma(l) for l in LEVELS)
for l in LEVELS
    @printf("background %.2f I_th rms: per-step σ = %.2f µA\n", l, SIG[l])
end

struct NoiseDrive
    rng::Xoshiro
    σ::Float32
    alive::BitMatrix
end
(d::NoiseDrive)(t) = ifelse.(d.alive, d.σ .* randn(d.rng, Float32, N, N), 0f0)

gain_array(g; alive = trues(N, N), scale = ones(Float32, N, N)) = begin
    Gm = zeros(Float32, N, N, K)
    for (k, (dy, dx)) in enumerate(offs), x in 1:N, y in 1:N
        sy, sx = y + dy, x + dx
        (1 <= sy <= N && 1 <= sx <= N && alive[y, x] && alive[sy, sx]) || continue
        Gm[y, x, k] = g * scale[y, x]
    end
    Gm
end

function avalanches(rate)
    A = [sum(@view rate[i:min(i + BIN - 1, end)]) * N^2 for i in 1:BIN:length(rate)]
    sizes = Float64[]; cur = 0.0
    for a in A
        if a > 0.5
            cur += a
        elseif cur > 0
            push!(sizes, cur); cur = 0.0
        end
    end
    sizes
end

function branching(rate; bin = 25)
    A = [sum(@view rate[i:min(i + bin - 1, end)]) * N^2 for i in 1:bin:length(rate)-bin]
    x, y = A[1:end-1], A[2:end]
    var(x) < 1e-9 && return NaN
    return cov(x, y) / var(x)
end

function measure(Gm; alive = trues(N, N), T = 120_000, seed = 1, level = 0.28)
    tr = sheet_simulate(DeviceSheet(cell, N, N; G = Gm), T; drive = NoiseDrive(Xoshiro(seed), SIG[level], BitMatrix(alive)), device = dev)
    s = avalanches(tr.rate)
    isempty(s) && (s = [0.0])
    (; sizes = s, mean_s = mean(s), n_av = length(s), max_s = maximum(s), m = branching(tr.rate),
       rate_per_cell_GHz = 1000 * sum(tr.n_spikes) / count(alive) / T, n_spikes = tr.n_spikes)
end

println("[1] intact sheet vs line gain")
rows = NamedTuple[]; dists = Dict{Float64, Vector{Float64}}()
for level in LEVELS, g in (0.0, 0.10, 0.15, 0.18, 0.20, 0.22, 0.24, 0.26, 0.30, 0.40)
    r = measure(gain_array(g); level)
    level == 0.20 && (dists[g] = r.sizes)
    @printf("    noise %.2f  G %.2f:  m̂ %.3f  rate %.4f GHz/cell  (%d silent-separated avalanches, ⟨s⟩ %.1f, max %.0f)\n",
            level, g, r.m, r.rate_per_cell_GHz, r.n_av, r.mean_s, r.max_s)
    flush(stdout)
    push!(rows, (; noise = level, G = g, m_est = r.m, rate_GHz = r.rate_per_cell_GHz, n_av = r.n_av, mean_s = r.mean_s, max_s = r.max_s))
end
write_tsv("e9_gain_sweep.tsv", rows)

# operating point: the lowest gain in the self-sustained state at noise 0.28 —
# the state damage destroys and repair must restore
hi = filter(r -> r.noise == 0.28 && r.rate_GHz > 0.3, rows)
g_op = minimum(r.G for r in hi)
println("\n[2] damage and repair at G = $g_op")
rng = Xoshiro(9)
alive = rand(rng, N, N) .> 0.30
nb = N ÷ 8
blocksum(A) = [sum(A[(i-1)*8+1:i*8, (j-1)*8+1:j*8]) for i in 1:nb, j in 1:nb]
intact = measure(gain_array(g_op); seed = 11)
target = blocksum(intact.n_spikes) ./ 64                          # spikes per cell per block, intact
damaged = measure(gain_array(g_op; alive); alive, seed = 12)
@printf("    intact   m̂ %.3f  rate %.4f GHz/cell\n", intact.m, intact.rate_per_cell_GHz)
@printf("    damaged  m̂ %.3f  rate %.4f GHz/cell  [%.0f%% dead]\n", damaged.m, damaged.rate_per_cell_GHz, 100(1 - mean(alive)))
alive_b = blocksum(Float32.(alive))
scale = ones(Float32, N, N)
hist = NamedTuple[]
for ep in 1:16
    r = measure(gain_array(g_op; alive, scale); alive, T = 40_000, seed = 100 + ep)
    per_cell = blocksum(r.n_spikes) ./ max.(alive_b, 1) .* (120_000 / 40_000)   # same time base as target
    err = (target .- per_cell) ./ max.(target, 1e-3)
    for j in 1:N, i in 1:N
        scale[i, j] = clamp(scale[i, j] * exp(0.15 * clamp(err[cld(i, 8), cld(j, 8)], -1, 1)), 0.5f0, 3f0)
    end
    push!(hist, (; epoch = ep, mean_scale = mean(scale[alive]), m_est = r.m, rate_rel = sum(per_cell) / sum(target)))
    @printf("    epoch %2d: survivor sensitivity ×%.3f  m̂ %.3f  activity %.2f of intact\n",
            ep, hist[end].mean_scale, r.m, hist[end].rate_rel)
    flush(stdout)
end
repaired = measure(gain_array(g_op; alive, scale); alive, seed = 13)
@printf("    repaired m̂ %.3f  rate %.4f GHz/cell\n", repaired.m, repaired.rate_per_cell_GHz)
write_tsv("e9_repair.tsv", [(; state = "intact", G = g_op, mean_s = intact.mean_s, m_est = intact.m, max_s = intact.max_s, rate_GHz = intact.rate_per_cell_GHz),
                            (; state = "damaged", G = g_op, mean_s = damaged.mean_s, m_est = damaged.m, max_s = damaged.max_s, rate_GHz = damaged.rate_per_cell_GHz),
                            (; state = "repaired", G = g_op, mean_s = repaired.mean_s, m_est = repaired.m, max_s = repaired.max_s, rate_GHz = repaired.rate_per_cell_GHz)])
write_tsv("e9_repair_history.tsv", hist)
# avalanche-size CCDFs for the report
ccdf(s) = (x = sort(s); (x, 1 .- (0:length(x)-1) ./ length(x)))
cc = NamedTuple[]
for (lab, s) in vcat([("G=$(g)", dists[g]) for g in sort(collect(keys(dists)))],
                     [("intact", intact.sizes), ("damaged", damaged.sizes), ("repaired", repaired.sizes)])
    x, y = ccdf(s)
    idx = unique(round.(Int, exp.(range(0, log(length(x)), length = min(60, length(x))))))
    for i in idx
        push!(cc, (; series = lab, s = x[i], p = y[i]))
    end
end
write_tsv("e9_ccdf.tsv", cc)
println("    done")
