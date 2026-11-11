# scripts/e1_settle_convergence.jl — E1 follow-up: are the trained-arm failures
# settle-convergence failures?
#
# The main E1 sweep found that on the trained trajectory, EP gradient fidelity
# is not a monotone function of ‖W‖ — it is perfect at ‖W₁‖ = 32, 41, 69 and 89
# and catastrophic at 56 and 75, where even the centered estimator inverts
# (cos = -0.059, -0.882). Those two snapshots are also the only ones whose free
# settle had not converged at T = 400:
#
#     ‖W₁‖     56      75   |  every other snapshot
#     resid   3e-6    1e-4  |  <= 2e-7
#     drift   4e-5    3e-3  |  <= 1e-5
#
# Two candidate explanations, with different consequences for the paper:
#
#   (a) the settle is short. T=400 was fixed for the whole sweep, but ‖W‖ grows
#       during training and slows the coupling Jacobian's slowest mode, so a T
#       that was ample at epoch 2 need not be at epoch 8. If so, the failures
#       are an artefact of the probe, the residual is a sufficient diagnostic,
#       and the hardware statement is "give the network time to equilibrate".
#
#   (b) the failure is specific to the fixed 16-sample probe batch. Every point
#       in the main sweep used X[:, 1:16], so a parameter×batch interaction is
#       entirely unseparated from a parameter effect.
#
# This script varies T and the batch independently over the saved snapshots and
# settles which it is. If cos recovers with T at fixed batch, (a). If it varies
# across batches at fixed T, (b). Both are possible.
#
# No retraining: reads snapshots_*.jls written by ep_trained_vs_rescaled.jl.
#
# Run: julia --project=. -t auto scripts/e1_settle_convergence.jl

using PhasorNetworks, Lux, LinearAlgebra, Random, Statistics, Printf, Serialization
using Random: Xoshiro

const DIR = joinpath(@__DIR__, "..", "results", "ep_trained_vs_rescaled")

_envi(k, d) = parse(Int,     get(ENV, k, string(d)))
_envf(k, d) = parse(Float32, get(ENV, k, string(d)))

const HID   = _envi("E1_HID",   256)
const DOUT  = _envi("E1_DOUT",  64)
const SEED  = _envi("E1_SEED",  7)
const SCALE = _envf("E1_SCALE", 0.4)
const DT    = _envf("E1_DT",    0.5)
const K_FD  = _envi("E1C_K_FD", 64)     # smaller than the main sweep: we are
                                        # detecting cos 0 -> 1, not measuring 1e-3
const DIR_SEED = _envi("E1_DIR_SEED", 90210)
const PROBE_B  = _envi("E1_PROBE_B", 16)
const N_BATCH  = _envi("E1C_N_BATCH", 3)              # disjoint probe batches
const T_LADDER = [400, 800, 1600, 3200]
const FD_EPS   = _envf("E1_FD_EPS", 0.03)
# Probing all 20 snapshots at four settle lengths costs ~1 h and most of it is
# spent re-confirming cells that were already clean. Default to the two failures
# and two bracketing controls per trajectory; E1C_EPOCHS overrides.
const EPOCH_FILTER = let e = get(ENV, "E1C_EPOCHS", "")
    isempty(e) ? Int[] : parse.(Int, split(e, ","))
end
const BETA     = _envf("E1C_BETA",  0.1)

# ---- shared with the main script ----------------------------------------
function encode_phase(imgs::AbstractArray{Float32,3})
    flat = reshape(imgs, :, size(imgs, 3))
    μ = mean(flat; dims=1); σ = std(flat; dims=1) .+ 1f-6
    return Phase.(0.5f0 .* tanh.((flat .- μ) ./ σ))
end
make_codes(rng) = ComplexF32.(angle_to_complex(orthogonal_codes(rng, DOUT, 10)))
direction(k, n) = (d = randn(Xoshiro(DIR_SEED + k), Float32, n); d ./= norm(d); d)
cos_sim(a, b) = dot(a, b) / (norm(a) * norm(b) + 1e-12)

function build_chain(rng)
    chain = Chain(PhasorDense(784 => HID,  normalize_to_unit_circle, use_bias=true),
                  PhasorDense(HID => DOUT, normalize_to_unit_circle, use_bias=true))
    ps, st = Lux.setup(rng, chain)
    return chain, ps, st
end

function fd_directional(chain, ps, st, x, cost, K; eps, T)
    W = ps.layer_1.weight; n = length(W); W0 = copy(W); out = zeros(Float32, K)
    loss_at() = ep_loss(cost, phasor_settle(chain, ps, st, x, cost, 0f0; T=T, dt=DT)[end])
    for k in 1:K
        d = reshape(direction(k, n), size(W))
        @. W = W0 + eps * d; Lp = loss_at()
        @. W = W0 - eps * d; Lm = loss_at()
        out[k] = (Lp - Lm) / (2eps)
    end
    W .= W0; return out
end

function stationarity(chain, ps, st, x, cost; T)
    a = phasor_settle(chain, ps, st, x, cost, 0f0; T=T,  dt=DT)[end]
    b = phasor_settle(chain, ps, st, x, cost, 0f0; T=T+1, dt=DT)[end]
    c = phasor_settle(chain, ps, st, x, cost, 0f0; T=2T, dt=DT)[end]
    return norm(b - a)/sqrt(length(a)), norm(c - a)/sqrt(length(a))
end

function main()
    tr = fashion_mnist_data(:train)
    X = encode_phase(Float32.(tr.features[:, :, 1:(N_BATCH * PROBE_B)]))
    y = Int.(tr.targets[1:(N_BATCH * PROBE_B)]) .+ 1
    codes = make_codes(Xoshiro(SEED + 1))
    chain, _, st = build_chain(Xoshiro(SEED))

    rows = NamedTuple[]
    for runname in ("nodecay", "wd1e-4")
        f = joinpath(DIR, "snapshots_$runname.jls")
        isfile(f) || (@warn "missing $f — skipping"; continue)
        snaps = deserialize(f)
        # nodecay failures were epochs 8 (‖W₁‖=55.9) and 14 (75.1); 4 and 20
        # bracket them as controls. Same epochs used for wd1e-4 so the two
        # trajectories are compared at matched training progress, not matched ‖W‖.
        want = isempty(EPOCH_FILTER) ? [4, 8, 14, 20] : EPOCH_FILTER
        snaps = [s for s in snaps if s[1] in want]
        @printf("\n=== %s: %d snapshots (epochs %s) ===\n",
                runname, length(snaps), join((s[1] for s in snaps), ","))
        for (epoch, ps, acc) in snaps
            n1 = norm(ps.layer_1.weight); n = length(ps.layer_1.weight)
            @printf("\n-- epoch %d | ‖W₁‖=%.1f | acc=%.4f --\n", epoch, n1, acc)
            @printf("   %-6s %-6s %10s %10s %10s %10s\n",
                    "batch", "T", "cos_1side", "cos_cent", "resid", "drift")
            for b in 1:N_BATCH
                sl = ((b-1)*PROBE_B + 1):(b*PROBE_B)
                x = X[:, sl]; cost = CodebookCost(codes, y[sl])
                for T in T_LADDER
                    resid, drift = stationarity(chain, ps, st, x, cost; T=T)
                    ref = fd_directional(chain, ps, st, x, cost, K_FD; eps=FD_EPS, T=T)
                    cs = map((false, true)) do cen
                        m = StaticEP(β=BETA, T_free=T, T_nudge=T÷2, dt=DT, centered=cen)
                        g, _ = ep_gradient(m, chain, ps, st, x, cost)
                        gv = vec(g.layer_1.weight)
                        cos_sim(Float32[dot(gv, direction(k, n)) for k in 1:K_FD], ref)
                    end
                    @printf("   %-6d %-6d %10.4f %10.4f %10.2e %10.2e\n",
                            b, T, cs[1], cs[2], resid, drift)
                    flush(stdout)
                    push!(rows, (; run=runname, epoch, w1_norm=n1, acc, batch=b, T,
                                   cos_onesided=cs[1], cos_centered=cs[2],
                                   settle_resid=resid, settle_drift=drift,
                                   k_fd=K_FD, beta=BETA))
                end
            end
        end
    end

    path = joinpath(DIR, "settle_convergence.csv")
    open(path, "w") do io
        println(io, join(string.(keys(rows[1])), ","))
        for r in rows
            println(io, join([v isa AbstractFloat ? @sprintf("%.6g", v) : string(v)
                              for v in values(r)], ","))
        end
    end
    @info "wrote $path ($(length(rows)) rows)"
end

main()
