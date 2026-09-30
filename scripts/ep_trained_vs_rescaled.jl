# scripts/ep_trained_vs_rescaled.jl — E1: does the ‖W‖ decorrelation survive training?
#
# The problem this settles
# ------------------------
# `results/ep_fashionmnist/gradient_fidelity_vs_weightnorm.csv` says the EP
# gradient decorrelates hard past ‖W₁‖ ≈ 15: one-sided cos(L1) falls 0.995 →
# 0.104 between ‖W₁‖ = 7.9 and 31.4, with relative error scaling as 1/β — the
# signature of the free and nudged settles landing in *different* fixed points.
#
# But the headline FashionMNIST run trains at ‖W₁‖ ≈ 20–29 for its whole length
# and still reaches 0.83. Both statements cannot be load-bearing. The caveat in
# FINDINGS.md §3 is that the probe used *randomly initialized matrices rescaled
# to the stated norm*, and that trained weights of the same norm may behave
# better. That is a hypothesis, not a measurement. This script measures it.
#
# Design
# ------
# 1. Train with StaticEP, snapshotting parameters every epoch. Two trajectories:
#      wd=1e-4  → ‖W₁‖ plateaus near 25, accuracy holds     (the good run)
#      wd=0     → ‖W₁‖ grows 27 → 109, accuracy peaks then decays
#    The no-decay run is the decisive one: it sweeps the whole norm range under
#    test *and* exhibits the accuracy decay the decorrelation is meant to explain.
#
# 2. At each snapshot, build a matched RESCALED control — the run's own random
#    init, rescaled so ‖W₁‖ and ‖W₂‖ equal the snapshot's. Same norms, no
#    training structure. Trained vs rescaled at matched norm is the comparison.
#
# 3. Score both against subset finite differences.
#
# Why subset FD and not centered StaticEP
# ---------------------------------------
# The existing probe scores EP against centered StaticEP at small β. That is
# fine for calibrating lock-in, but it is circular *here*: basin hopping is a
# property of the settle, so it contaminates the centered estimator too, and a
# reference that shares the failure mode cannot detect it. Full FD needs
# n_params+1 settles (217K) and is unaffordable — but we do not need the full
# gradient. A random subset of K entries of layer_1.weight gives an unbiased
# view of the direction, costs 2K settles, and is genuinely independent of EP.
#
# Central differences at ε=1e-3: FINDINGS.md records that fd_gradient_phasor's
# default ε=1e-5 is under-conditioned against a Float32 10-class cross-entropy
# (EP-vs-FD rel-err 0.24 at 1e-5, 0.004 at 1e-3) — below ~1e-4 the *oracle*
# becomes the noisy party. A three-step bracket makes that visible per probe.
#
# Cost / hardware
# ---------------
# CPU only; src/ep.jl's settle at B=16 is a few hundred KB of working set and
# no CUDA path is touched. Nothing here needs a GPU memory cap.
#
# Run:
#   julia --project=. -t auto scripts/ep_trained_vs_rescaled.jl
# Smoke test (~2 min, verifies the whole path before committing to the real run):
#   E1_SMOKE=1 julia --project=. -t auto scripts/ep_trained_vs_rescaled.jl

using PhasorNetworks, Lux, Optimisers, MLUtils, LinearAlgebra
using Random, Statistics, Printf, Serialization
using Random: Xoshiro

const OUTDIR = joinpath(@__DIR__, "..", "results", "ep_trained_vs_rescaled")
isdir(OUTDIR) || mkpath(OUTDIR)

_envi(k, d) = parse(Int,     get(ENV, k, string(d)))
_envf(k, d) = parse(Float32, get(ENV, k, string(d)))
_envb(k, d) = get(ENV, k, string(d)) in ("1", "true", "yes")

const SMOKE = _envb("E1_SMOKE", false)

# ---- config --------------------------------------------------------------
const N_TRAIN = _envi("E1_N_TRAIN", SMOKE ? 2000 : 60000)
const N_TEST  = _envi("E1_N_TEST",  SMOKE ? 1000 : 10000)
const BATCH   = _envi("E1_BATCH",   128)
const HID     = _envi("E1_HID",     256)
const DOUT    = _envi("E1_DOUT",    64)
const SEED    = _envi("E1_SEED",    7)
const SCALE   = _envf("E1_SCALE",   0.4)
const EPOCHS  = _envi("E1_EPOCHS",  SMOKE ? 2 : 20)
const LR      = parse(Float64, get(ENV, "E1_LR", "0.003"))

# Settle geometry, matched to demos/ep_fashionmnist.jl so the snapshots are the
# same objects the headline run produces.
const T_FREE  = _envi("E1_T_FREE",  200)
const T_NUDGE = _envi("E1_T_NUDGE", 100)
const DT      = _envf("E1_DT",      0.5)

# Probe settings
const K_FD      = _envi("E1_K_FD",    SMOKE ? 24 : 256)   # FD subset size
const FD_T      = _envi("E1_FD_T",    400)                # longer settle for the oracle
# Bracket of directional step sizes. The centre one is the reference; the
# other two are its error bar. A bracket beats a single step because the two
# failure modes pull in opposite directions -- too small is swamped by Float32
# resolution, too large leaves the linear regime (and, here, can hop the
# settle's basin) -- so agreement across the bracket rules out both at once.
const FD_EPSILONS = Float32[
    _envf("E1_FD_EPS_LO", 0.01),
    _envf("E1_FD_EPS",    0.03),   # reference
    _envf("E1_FD_EPS_HI", 0.10),
]
const FD_REF_I  = 2
# Below this agreement between the reference and BOTH neighbours, the loss is
# not locally linear at any usable step and no cosine against it means anything.
const FD_TRUST  = _envf("E1_FD_TRUST", 0.95)
const DIR_SEED  = _envi("E1_DIR_SEED", 90210)
const PROBE_B   = _envi("E1_PROBE_B", 16)                 # batch for the probe
const EVERY     = _envi("E1_EVERY",   SMOKE ? 1 : 2)      # probe every Nth epoch
const BETAS     = Float32[0.3, 0.1, 0.03, 0.01, 0.003]
const LOCKIN_EPS = _envf("E1_LOCKIN_EPS", 0.03)
const LOCKIN_WP  = _envf("E1_LOCKIN_WP",  0.02)

# ---- data / model (mirrors demos/ep_fashionmnist.jl) ----------------------
function encode_phase(imgs::AbstractArray{Float32,3})
    flat = reshape(imgs, :, size(imgs, 3))
    μ = mean(flat; dims=1); σ = std(flat; dims=1) .+ 1f-6
    return Phase.(0.5f0 .* tanh.((flat .- μ) ./ σ))
end

function load_data()
    tr = fashion_mnist_data(:train); te = fashion_mnist_data(:test)
    ntr = min(N_TRAIN, length(tr.targets)); nte = min(N_TEST, length(te.targets))
    return (encode_phase(Float32.(tr.features[:, :, 1:ntr])), Int.(tr.targets[1:ntr]) .+ 1),
           (encode_phase(Float32.(te.features[:, :, 1:nte])), Int.(te.targets[1:nte]) .+ 1)
end

function build_chain(rng)
    chain = Chain(
        PhasorDense(784 => HID,  normalize_to_unit_circle, use_bias=true),
        PhasorDense(HID => DOUT, normalize_to_unit_circle, use_bias=true),
    )
    ps, st = Lux.setup(rng, chain)
    ps = (layer_1 = merge(ps.layer_1, (weight = SCALE .* ps.layer_1.weight,)),
          layer_2 = merge(ps.layer_2, (weight = SCALE .* ps.layer_2.weight,)))
    return chain, ps, st
end

make_codes(rng) = ComplexF32.(angle_to_complex(orthogonal_codes(rng, DOUT, 10)))
weight_norms(ps) = (norm(ps.layer_1.weight), norm(ps.layer_2.weight))

function accuracy(chain, ps, st, codes, X, y; batch=512)
    correct = 0
    for i in 1:batch:length(y)
        e = min(i + batch - 1, length(y))
        logits = ep_predict(chain, ps, st, X[:, i:e], codes; T=T_FREE, dt=DT)
        correct += sum([argmax(view(logits, :, b)) for b in 1:(e-i+1)] .== y[i:e])
    end
    return correct / length(y)
end

# ---- the oracle: directional central finite differences ------------------
#
# Coordinate FD on a single entry of a 784x256 matrix is badly conditioned: the
# loss is a Float32 cross-entropy near 1.7, so machine resolution is ~1e-7,
# while one weight moves it by ~1e-6. Measured effect (first version of this
# script): FD at 1e-3 vs 3e-4 agreed only to cos 0.94 even where the settle
# residual was 1e-9, i.e. the oracle was the noisy party, exactly as
# FINDINGS.md warns for fd_gradient_phasor's default epsilon.
#
# Perturbing along a dense random UNIT direction moves every entry at once, so
# dL is larger by ~sqrt(n) and the same epsilon sits far above resolution. The
# K directional derivatives <grad, d_k> are also a better statistic than K
# coordinates: by Johnson-Lindenstrauss the cosine between the two projection
# vectors estimates the cosine between the full gradients, which is the
# quantity actually in question, rather than the cosine on a coordinate slice.
#
# Directions are regenerated from a per-index seed rather than stored: at
# n = 200704 a K = 256 basis would be 205 MB held live for no reason.
direction(k::Int, n::Int) = (d = randn(Xoshiro(DIR_SEED + k), Float32, n);
                             d ./= norm(d); d)

"""
    fd_directional(chain, ps, st, x, cost, K; eps, T) -> Vector{Float32}

`K` central-difference directional derivatives of the settled loss with respect
to `layer_1.weight`, along fixed pseudo-random unit directions. Mutates the
weight in place and restores it, so no per-direction deepcopy.
"""
function fd_directional(chain, ps, st, x, cost, K::Int; eps::Float32, T::Int)
    W = ps.layer_1.weight
    n = length(W)
    W0 = copy(W)
    out = zeros(Float32, K)
    loss_at() = ep_loss(cost, phasor_settle(chain, ps, st, x, cost, 0f0;
                                            T=T, dt=DT)[end])
    for k in 1:K
        d = reshape(direction(k, n), size(W))
        @. W = W0 + eps * d; Lp = loss_at()
        @. W = W0 - eps * d; Lm = loss_at()
        out[k] = (Lp - Lm) / (2eps)
    end
    W .= W0
    return out
end

"Project a gradient's layer_1.weight onto the same K directions."
function project_dirs(g, K::Int, n::Int)
    gv = vec(g.layer_1.weight)
    return Float32[dot(gv, direction(k, n)) for k in 1:K]
end

"""
Report the raw loss-difference magnitude at each epsilon. If this is not
comfortably above Float32 resolution on a loss of order 1, the oracle is noise
and every cosine downstream is meaningless. Printed once per run.
"""
function fd_conditioning(chain, ps, st, x, cost; T::Int, K::Int = 8)
    W = ps.layer_1.weight; n = length(W); W0 = copy(W)
    loss_at() = ep_loss(cost, phasor_settle(chain, ps, st, x, cost, 0f0;
                                            T=T, dt=DT)[end])
    println("  -- FD conditioning (|L(+eps d) - L(-eps d)|, Float32 resolution ~1e-7) --")
    for eps in FD_EPSILONS
        ds = Float32[]
        for k in 1:K
            d = reshape(direction(k, n), size(W))
            @. W = W0 + eps * d; Lp = loss_at()
            @. W = W0 - eps * d; Lm = loss_at()
            push!(ds, abs(Lp - Lm))
        end
        W .= W0
        @printf("     eps=%-8.4g  median |dL| = %.3e   min = %.3e\n",
                eps, median(ds), minimum(ds))
    end
    flush(stdout)
end

cos_sim(a, b) = dot(a, b) / (norm(a) * norm(b) + 1e-12)
rel_err(a, b) = norm(a .- b) / (norm(b) + 1e-12)

# ---- matched rescaled control -------------------------------------------
#
# The run's own random init, rescaled so both layer norms match the snapshot.
# Same norms, none of the structure training put there — this is exactly the
# construction FINDINGS.md flags as possibly overstating the effect.
function rescaled_like(ps_init, n1::Real, n2::Real)
    w1 = ps_init.layer_1.weight; w2 = ps_init.layer_2.weight
    return (layer_1 = merge(ps_init.layer_1, (weight = Float32(n1 / norm(w1)) .* w1,)),
            layer_2 = merge(ps_init.layer_2, (weight = Float32(n2 / norm(w2)) .* w2,)))
end

# ---- settle stationarity at the probed parameters -----------------------
#
# The FD oracle assumes the free settle has reached a fixed point. If it has
# not, both the oracle and EP are reading a moving target, and a low cosine
# says nothing about the estimator. Reported per probe point so a bad cell can
# be identified as un-converged rather than mis-estimated.
function settle_stationarity(chain, ps, st, x, cost; T::Int)
    a = phasor_settle(chain, ps, st, x, cost, 0f0; T=T,     dt=DT)[end]
    b = phasor_settle(chain, ps, st, x, cost, 0f0; T=T+1,   dt=DT)[end]
    c = phasor_settle(chain, ps, st, x, cost, 0f0; T=2T,    dt=DT)[end]
    return (norm(b - a) / sqrt(length(a)),    # step-to-step residual
            norm(c - a) / sqrt(length(a)))    # T vs 2T drift
end

# ---- probe one parameter set --------------------------------------------
function probe!(rows, chain, ps, st, codes, X, y; tag, run, epoch, acc)
    x    = X[:, 1:PROBE_B]
    cost = CodebookCost(codes, y[1:PROBE_B])
    n1, n2 = weight_norms(ps)
    n = length(ps.layer_1.weight)

    resid, drift = settle_stationarity(chain, ps, st, x, cost; T=FD_T)

    refs = [fd_directional(chain, ps, st, x, cost, K_FD; eps=e, T=FD_T)
            for e in FD_EPSILONS]
    ref     = refs[FD_REF_I]
    agree_lo = cos_sim(ref, refs[FD_REF_I - 1])   # step halved  -> linear regime?
    agree_hi = cos_sim(ref, refs[FD_REF_I + 1])   # step tripled -> local curvature
    # Trust is decided by the SMALL-step end only. Convergence as the step
    # shrinks is what certifies the reference sits in the linear regime; the
    # large step is expected to disagree wherever the loss has curvature, and
    # treating that as oracle failure (an earlier version of this script did)
    # wrongly discards the points of greatest interest. agree_hi is kept as a
    # measurement of local roughness, not as a gate.
    fd_ok = agree_lo >= FD_TRUST

    @printf("    [oracle] agree lo=%+.4f hi=%+.4f  |ref|=%.3e  resid=%.2e  %s\n",
            agree_lo, agree_hi, norm(ref), resid, fd_ok ? "TRUSTED" : "UNTRUSTED")

    estimators = Tuple{String,Float32,Any}[]
    for β in BETAS
        push!(estimators, ("static_onesided", β,
              StaticEP(β=β, T_free=T_FREE, T_nudge=T_NUDGE, dt=DT, centered=false)))
        push!(estimators, ("static_centered", β,
              StaticEP(β=β, T_free=T_FREE, T_nudge=T_NUDGE, dt=DT, centered=true)))
    end
    push!(estimators, ("lockin", LOCKIN_EPS,
          LockinEP(ε=LOCKIN_EPS, ω_p=LOCKIN_WP, n_cycles=4, T_warmup_cycles=2,
                   T_free=T_FREE, dt=DT)))

    for (name, β, m) in estimators
        g, _ = ep_gradient(m, chain, ps, st, x, cost)
        gsub = project_dirs(g, K_FD, n)
        push!(rows, (; run, tag, epoch, acc, w1_norm=n1, w2_norm=n2,
                       estimator=name, beta=β,
                       cos_fd=cos_sim(gsub, ref), relerr_fd=rel_err(gsub, ref),
                       fd_agree_lo=agree_lo, fd_agree_hi=agree_hi, fd_trusted=fd_ok,
                       fd_ref_norm=norm(ref),
                       settle_resid=resid, settle_drift=drift,
                       k_fd=K_FD, probe_b=PROBE_B))
        @printf("    %-16s β=%-6.3f  cos=%+.4f  relerr=%9.3f\n",
                name, β, rows[end].cos_fd, rows[end].relerr_fd)
        flush(stdout)
    end
    return nothing
end

function write_csv(path, rows)
    open(path, "w") do io
        println(io, join(string.(keys(rows[1])), ","))
        for r in rows
            println(io, join([v isa AbstractFloat ? @sprintf("%.6g", v) : string(v)
                              for v in values(r)], ","))
        end
    end
    @info "wrote $path ($(length(rows)) rows)"
end

# ---- main ----------------------------------------------------------------
function main()
    @printf("E1: trained vs rescaled weights at matched norm%s\n", SMOKE ? "  [SMOKE]" : "")
    @printf("  %d train / %d test | 784→%d→%d | %d epochs\n", N_TRAIN, N_TEST, HID, DOUT, EPOCHS)
    @printf("  oracle: %d directions, FD bracket [%s] (ref %g), settle T=%d\n",
            K_FD, join((@sprintf("%g", e) for e in FD_EPSILONS), ", "),
            FD_EPSILONS[FD_REF_I], FD_T)
    @printf("  BLAS threads %d | probe batch %d | every %d epoch(s)\n\n",
            BLAS.get_num_threads(), PROBE_B, EVERY)

    rng = Xoshiro(SEED)
    (Xtr, ytr), (Xte, yte) = load_data()
    codes = make_codes(Xoshiro(SEED + 1))
    chain, ps_init, st = build_chain(rng)

    # Directions are fixed by DIR_SEED and regenerated identically at every
    # probe point, so cosines are comparable across snapshots rather than each
    # measuring a different random subspace.
    fd_conditioning(chain, ps_init, st,
                    Xtr[:, 1:PROBE_B], CodebookCost(codes, ytr[1:PROBE_B]); T=FD_T)

    rows = NamedTuple[]
    method = StaticEP(β=0.1f0, T_free=T_FREE, T_nudge=T_NUDGE, dt=DT, centered=true)

    for (runname, wd) in (("nodecay", 0.0), ("wd1e-4", 1e-4))
        @printf("\n=== training %s (wd=%g) ===\n", runname, wd)
        loader = MLUtils.DataLoader((Xtr, ytr); batchsize=BATCH, shuffle=true)
        args = PhasorNetworks.Args(lr=LR, epochs=EPOCHS, weight_decay=wd)
        snaps = Tuple{Int,Any,Float64}[]
        t0 = time()
        cb = function (epoch, ps_now, st_now, epoch_loss)
            a = accuracy(chain, ps_now, st_now, codes, Xte, yte)
            n1, n2 = weight_norms(ps_now)
            @printf("[%s] epoch %2d/%d  loss=%.4f  acc=%.4f  |W1|=%.1f |W2|=%.1f  (%.0f s)\n",
                    runname, epoch, EPOCHS, epoch_loss, a, n1, n2, time() - t0)
            flush(stdout)
            epoch % EVERY == 0 && push!(snaps, (epoch, deepcopy(ps_now), a))
        end
        ep_train(chain, deepcopy(ps_init), st, loader, args;
                 method=method, cost_fn = yb -> CodebookCost(codes, yb),
                 optimiser=Optimisers.Adam, callback=cb)
        serialize(joinpath(OUTDIR, "snapshots_$runname.jls"), snaps)

        for (epoch, ps_t, a) in snaps
            n1, n2 = weight_norms(ps_t)
            @printf("\n  -- %s epoch %d | ‖W₁‖=%.1f ‖W₂‖=%.1f | acc=%.4f --\n",
                    runname, epoch, n1, n2, a)
            println("  [trained]")
            probe!(rows, chain, ps_t, st, codes, Xtr, ytr;
                   tag="trained", run=runname, epoch=epoch, acc=a)
            println("  [rescaled control: random init at matched norms]")
            ps_r = rescaled_like(ps_init, n1, n2)
            a_r = accuracy(chain, ps_r, st, codes, Xte, yte)
            probe!(rows, chain, ps_r, st, codes, Xtr, ytr;
                   tag="rescaled", run=runname, epoch=epoch, acc=a_r)

            # Third control, and the one that actually separates the variables:
            # the FIRST snapshot's weights — trained structure, small norm —
            # rescaled up to this snapshot's norms. If it tracks `trained`, norm
            # is not the driver; if it tracks `rescaled`, norm is. The random
            # control alone cannot distinguish these, because it differs from
            # the trained snapshot in both structure and accuracy.
            if epoch != first(snaps)[1]
                ps_tr = rescaled_like(first(snaps)[2], n1, n2)
                a_tr = accuracy(chain, ps_tr, st, codes, Xte, yte)
                probe!(rows, chain, ps_tr, st, codes, Xtr, ytr;
                       tag="trained_rescaled", run=runname, epoch=epoch, acc=a_tr)
            end
            write_csv(joinpath(OUTDIR, "trained_vs_rescaled.csv"), rows)  # incremental
        end
    end

    write_csv(joinpath(OUTDIR, "trained_vs_rescaled.csv"), rows)
    println("\ndone.")
end

main()
