# ep_fashionmnist.jl — phasor Equilibrium Propagation on FashionMNIST
#
# First scale-up of vanilla phasor EP (src/ep.jl) past toy problems. Prior
# validation was a single fixed pattern and the 4-corner XOR in
# demos/lockin_demo.ipynb §7; EP had never been run on a dataset.
#
# Model:  784 → 256 → 64, two PhasorDense layers with complex bias,
#         normalize_to_unit_circle activation, K_mode = :zero.
# Readout: CodebookCost over 64×10 orthogonal phase codewords —
#         logits s_c = (1/d)·Re⟨code_c, z_out⟩, softmax cross-entropy.
#
# Both gradient estimators are run at full scale from the same init:
#   StaticEP — two-phase finite difference (free + nudged equilibrium)
#   LockinEP — one settle plus one driven trajectory, β(t) = ε·cos(ω_p t)
#              demodulated at ω_p. This is the hardware-relevant estimator
#              (a synchronous demodulator is an analog primitive).
#
# Everything here is CPU: src/ep.jl allocates host arrays throughout.
# Peak footprint is a few hundred MB, dominated by the encoded dataset.
#
# Run:   julia --project=. -t auto demos/ep_fashionmnist.jl
# Quick: EP_N_TRAIN=4000 EP_N_TEST=1000 EP_EPOCHS=3 EP_RUN_LOCKIN=0 \
#          julia --project=. -t auto demos/ep_fashionmnist.jl
# Calibrate lock-in knobs only (no training):
#        EP_MODE=calibrate julia --project=. -t auto demos/ep_fashionmnist.jl

ENV["GKSwstype"] = "100"

using PhasorNetworks, Lux, Optimisers, MLUtils, LinearAlgebra
using Random, Statistics, Printf
using Random: Xoshiro
using Plots

const OUTDIR = joinpath(@__DIR__, "ep_out")
isdir(OUTDIR) || mkpath(OUTDIR)

# ---- config (env-overridable) -------------------------------------------
_envi(k, d) = parse(Int,     get(ENV, k, string(d)))
_envf(k, d) = parse(Float32, get(ENV, k, string(d)))
_envb(k, d) = get(ENV, k, string(d)) in ("1", "true", "yes")

const N_TRAIN = _envi("EP_N_TRAIN", 60000)
const N_TEST  = _envi("EP_N_TEST",  10000)
const BATCH   = _envi("EP_BATCH",   128)
const HID     = _envi("EP_HID",     256)
const DOUT    = _envi("EP_DOUT",    64)     # readout width (codeword length)
const SEED    = _envi("EP_SEED",    7)
const SCALE   = _envf("EP_SCALE",   0.4)    # weight init scale, per the demos

const STATIC_EPOCHS = _envi("EP_EPOCHS",        20)
# lr default goes with the optimizer default below (Adam). For plain
# Descent the workable range is ~0.05, but see sweep_optimisers: Descent
# tops out around 0.50 accuracy at this scale no matter the lr.
const STATIC_LR     = parse(Float64, get(ENV, "EP_LR", "0.003"))  # Args.lr is Float64
const STATIC_BETA   = _envf("EP_BETA",          0.1)
# T_free=100 was tuned on a width-8 toy chain and is NOT converged at
# width 256: measured stationarity residual ||z(T+1)-z(T)||/sqrt(N) is
# 4.8e-5 at T=100 but 9.8e-7 at T=200. Since the EP gradient is a
# difference of Hebbians divided by β, a residual that size leaks
# straight into the gradient. The convergence table is printed on every
# run — check it if you change the width.
const T_FREE        = _envi("EP_T_FREE",        200)
const T_NUDGE       = _envi("EP_T_NUDGE",       100)
const DT            = _envf("EP_DT",            0.5)
const CENTERED      = _envb("EP_CENTERED",      true)
# With normalize_to_unit_circle the states are scale-invariant, so nothing
# in the loss bounds ‖W‖: measured growth is |W1| 26.7 → 108.9 over 20
# epochs, roughly linear and unbounded. That matters because EP's gradient
# fidelity collapses at large ‖W‖ (see the StaticEP docstring — the free
# and nudged settles stop sharing a fixed point), which is the likely
# driver of the accuracy peak-then-decay seen without decay.
#
# But be careful about the causal claim: weight decay at 1e-4 improves
# accuracy (peak 0.771→0.793, final 0.721→0.775 on a 10K/8-epoch probe)
# while leaving the weight-norm trajectory nearly unchanged (13.9→30.4 vs
# 13.1→27.1), and 1e-3 bounds ‖W‖ far more (→15.7) yet does WORSE (0.699).
# So 1e-4 is not helping by bounding ‖W‖; the mechanism is unresolved.
# Use `EP_CENTERED=1` for the actual large-‖W‖ mitigation.
const WEIGHT_DECAY  = parse(Float64, get(ENV, "EP_WD", "0.0001"))

const RUN_LOCKIN     = _envb("EP_RUN_LOCKIN",    true)
const LOCKIN_EPOCHS  = _envi("EP_LOCKIN_EPOCHS", 10)
# Measured at width 256 (EP_MODE=calibrate), larger ε is BETTER, which is
# the opposite of the width-8 guidance in lockin_demo §4: at this width the
# demodulator's noise floor dominates over the O(ε²) nonlinearity, so the
# probe wants to be big, not small. ε=0.03 and ε=0.1 score the same
# (cos 0.998); 0.03 is used as it sits well inside the documented
# bifurcation-safe range.
const LOCKIN_EPS     = _envf("EP_LOCKIN_EPS",    0.03)
# Lock-in defaults are set by the MEASURED relaxation rate at this width,
# not by the toy-chain defaults. See check_convergence: R_relax ≈ 0.134
# /time-unit at 784→256→64, so adiabaticity needs ω_p ≲ 0.01, roughly 5x
# slower than the package default of 0.05.
#
# The trick is that ω_p and dt trade off for free: the cost is
# period_steps = 2π/(ω_p·dt), so running at dt=0.5 rather than 0.1 buys a
# 5x slower probe *in time units* at identical step count. dt=0.5 is the
# same step the static settle uses, and the probe increment ω_p·dt =
# 0.005 rad/step is far from aliasing.
const LOCKIN_WP      = _envf("EP_LOCKIN_WP",     0.02)
const LOCKIN_CYCLES  = _envi("EP_LOCKIN_CYCLES", 4)
const LOCKIN_TFREE   = _envi("EP_LOCKIN_TFREE",  200)
const LOCKIN_DT      = _envf("EP_LOCKIN_DT",     0.5)

const MODE = Symbol(get(ENV, "EP_MODE", "train"))   # :train | :calibrate | :sweep

# Plain SGD is ep_train's own default (and what the XOR demo used), but it
# is the wrong choice at this scale. Measured on a 10K subset, 5 epochs
# (EP_MODE=sweep): Descent reaches 0.476/0.462/0.493 at lr 0.05/0.02/0.005
# and oscillates, while Adam reaches 0.763/0.798/0.772 at lr
# 0.01/0.003/0.001. Adam at 3e-3 it is.
const OPTIMISER = let o = get(ENV, "EP_OPT", "adam")
    o == "adam"    ? Optimisers.Adam    :
    o == "descent" ? Optimisers.Descent :
    error("EP_OPT must be \"descent\" or \"adam\", got \"$o\"")
end

# ---- pixels → phase ------------------------------------------------------
#
# Phases live in [-1, 1] in units of π and the unit circle identifies
# Phase(1) with Phase(-1) (both are -1+0i), so a full-range map is NOT
# injective — the two extremes collapse onto each other. The repo's usual
# static path, `Phase.(tanh.(LayerNorm(x)))`, saturates *toward* exactly
# that collapse point.
#
# We instead standardize per image and map through 0.5·tanh, landing in
# [-0.5, 0.5]: the right half-plane, injective end to end, and zero-centered.
# Standardizing also matters because ~80% of FashionMNIST pixels are exactly
# zero — without it every background pixel drives the same 1+0i and the
# common mode swamps the signal.
function encode_phase(imgs::AbstractArray{Float32,3})
    N = size(imgs, 3)
    flat = reshape(imgs, :, N)                       # (784, N)
    μ = mean(flat; dims=1)
    σ = std(flat; dims=1) .+ 1f-6
    return Phase.(0.5f0 .* tanh.((flat .- μ) ./ σ))
end

# Alternative kept for ablation: the {0°,90°} first-quadrant arc used by the
# XOR demo. Injective, but real W is axis-preserving so a first-quadrant-only
# input tends to polarize the hidden layer (lockin_demo §7.1) — the complex
# bias is what breaks that.
encode_phase_arc(imgs::AbstractArray{Float32,3}) =
    Phase.(reshape(imgs, :, size(imgs, 3)) .* 0.5f0)

function load_data()
    tr = fashion_mnist_data(:train)
    te = fashion_mnist_data(:test)
    ntr = min(N_TRAIN, length(tr.targets))
    nte = min(N_TEST,  length(te.targets))
    Xtr = encode_phase(Float32.(tr.features[:, :, 1:ntr]))
    Xte = encode_phase(Float32.(te.features[:, :, 1:nte]))
    # 1-based labels: CodebookCost indexes columns of the codebook, and
    # predict/evaluate_accuracy are 1-based, while the raw targets are 0..9.
    ytr = Int.(tr.targets[1:ntr]) .+ 1
    yte = Int.(te.targets[1:nte]) .+ 1
    return (Xtr, ytr), (Xte, yte)
end

# ---- model ---------------------------------------------------------------
#
# use_bias=true is load-bearing, not cosmetic: with real W and an entrywise
# unit projection the map is axis-preserving, so without a complex bias whole
# input classes stay locked to one axis of the complex plane and cannot be
# routed to a shared output class (lockin_demo §7.1).
function build_chain(rng)
    chain = Chain(
        PhasorDense(784  => HID,  normalize_to_unit_circle, use_bias=true),
        PhasorDense(HID  => DOUT, normalize_to_unit_circle, use_bias=true),
    )
    ps, st = Lux.setup(rng, chain)
    ps = (layer_1 = merge(ps.layer_1, (weight = SCALE .* ps.layer_1.weight,)),
          layer_2 = merge(ps.layer_2, (weight = SCALE .* ps.layer_2.weight,)))
    return chain, ps, st
end

# Orthogonal codewords beat random ones badly at this width: measured max
# inter-class crosstalk under CodebookCost's own logit kernel is 0.028 for
# orthogonal at d=64 vs 0.200 for random. (d=80 would be exactly 0, since
# 10 | 80.)
make_codes(rng) = ComplexF32.(angle_to_complex(orthogonal_codes(rng, DOUT, 10)))

function accuracy(chain, ps, st, codes, X, y; batch=512, T=T_FREE, dt=DT)
    correct = 0
    for i in 1:batch:length(y)
        e = min(i + batch - 1, length(y))
        logits = ep_predict(chain, ps, st, X[:, i:e], codes; T=T, dt=dt)
        pred = [argmax(view(logits, :, b)) for b in 1:(e - i + 1)]
        correct += sum(pred .== y[i:e])
    end
    return correct / length(y)
end

# Stationarity residual of the free settle at the CURRENT parameters.
#
# The whole EP gradient assumes the settle reached equilibrium. That was
# checked once at init — but ||W|| grows during training, which slows the
# coupling Jacobian's slowest mode, so a T_free that was ample at epoch 0
# can be inadequate by epoch 10. Tracking this per epoch tells you whether
# an accuracy collapse is an optimizer problem or an un-converged-settle
# problem.
function settle_residual(chain, ps, st, codes, X; n=32, T=T_FREE, dt=DT)
    x = X[:, 1:n]
    cost = CodebookCost(codes, ones(Int, n))
    a = phasor_settle(chain, ps, st, x, cost, 0f0; T=T,   dt=dt)[end]
    b = phasor_settle(chain, ps, st, x, cost, 0f0; T=T+1, dt=dt)[end]
    return norm(b - a) / sqrt(length(a))
end

weight_norms(ps) = (norm(ps.layer_1.weight), norm(ps.layer_2.weight))

# ---- settling-convergence check -----------------------------------------
#
# T_free=100 was tuned on a width-8 toy chain. Confirm the equilibrium is
# actually stationary at width 256 before trusting any gradient built on it.
function check_convergence(chain, ps, st, codes, X)
    x = X[:, 1:32]
    cost = CodebookCost(codes, ones(Int, 32))
    println("\n-- settling convergence at 784→$(HID)→$(DOUT) --")
    Ts, res = Int[], Float64[]
    for T in (25, 50, 100, 200, 400)
        a = phasor_settle(chain, ps, st, x, cost, 0f0; T=T,   dt=DT)[end]
        b = phasor_settle(chain, ps, st, x, cost, 0f0; T=T+1, dt=DT)[end]
        r = norm(b - a) / sqrt(length(a))
        @printf("  T=%-4d  ||z(T+1)-z(T)||/sqrt(N) = %.3e\n", T, r)
        push!(Ts, T); push!(res, r)
    end

    # Fit the relaxation rate from the residual decay. This is the number
    # that actually governs lock-in adiabaticity (ω_p ≪ R_relax), and it
    # is NOT ~1: the damped iteration z ← (1-dt)z + dt·unit(grad) would
    # relax at rate 1 only if the drive were independent of z, but the
    # inter-layer feedback puts eigenvalues of the coupling Jacobian close
    # to 1, and the slowest mode sets the rate. Measured ≈0.134 at width
    # 256 — so the toy-chain ω_p of 0.05 is NOT adiabatic here.
    keep = findall(r -> r > 1e-8, res)
    if length(keep) >= 3
        t = Float64.(Ts[keep]) .* Float64(DT)      # time units, not steps
        y = log.(res[keep]); n = length(t)
        R = -((n*sum(t .* y) - sum(t)*sum(y)) / (n*sum(t .^ 2) - sum(t)^2))
        @printf("  fitted R_relax = %.4f /time-unit  →  adiabatic lock-in wants ω_p ≲ %.3f\n",
                R, R / 10)
        @printf("  configured ω_p = %.3f (dt=%.2f) → ω_p/R_relax = %.2f, lag cos(φ) ≈ %.3f\n",
                LOCKIN_WP, LOCKIN_DT, LOCKIN_WP / R, cos(atan(LOCKIN_WP / R)))
    end
end

# ---- lock-in calibration -------------------------------------------------
#
# fd_gradient_phasor needs n_params+1 settles — impossible at 217K params. So
# calibrate LockinEP against CENTERED StaticEP at small β instead: the
# centered estimator's O(β) bias cancels, leaving O(β²), which makes it a
# usable oracle where FD is not. Pick the largest ω_p that still holds
# cosine ≳ 0.99 — every doubling of ω_p halves the lock-in bill.
function calibrate_lockin(chain, ps, st, codes, X, y)
    x = X[:, 1:16]
    cost = CodebookCost(codes, y[1:16])
    ref, _ = ep_gradient(StaticEP(β=0.005f0, T_free=300, T_nudge=150, dt=DT,
                                  centered=true), chain, ps, st, x, cost)
    println("\n-- lock-in calibration vs centered StaticEP (β=0.005) --")
    println("   ε        ω_p    steps/grad   cos(L1)  cos(L2)  rel-err(L1)")
    rows = []
    for ε in (0.01f0, 0.03f0, 0.1f0), ωp in (0.005f0, 0.01f0, 0.02f0, 0.05f0)
        # dt=0.5 throughout: ω_p and dt trade off as 2π/(ω_p·dt), so the
        # coarser step buys a slower probe in time units for free.
        m = LockinEP(ε=ε, ω_p=ωp, n_cycles=LOCKIN_CYCLES, T_warmup_cycles=2,
                     T_free=LOCKIN_TFREE, dt=LOCKIN_DT)
        period = round(Int, 2π / (ωp * LOCKIN_DT))
        steps  = LOCKIN_TFREE + (2 + LOCKIN_CYCLES) * period
        g, _ = ep_gradient(m, chain, ps, st, x, cost)
        cs = map((:layer_1, :layer_2)) do k
            a = vec(g[k].weight); b = vec(ref[k].weight)
            dot(a, b) / (norm(a) * norm(b) + 1e-10)
        end
        re = norm(g.layer_1.weight - ref.layer_1.weight) / norm(ref.layer_1.weight)
        @printf("  %5.3f  %5.3f  %9d   %7.4f  %7.4f  %10.4f\n",
                ε, ωp, steps, cs[1], cs[2], re)
        push!(rows, (ε=ε, ωp=ωp, steps=steps, cos1=cs[1], cos2=cs[2], relerr=re))
    end
    return rows
end

# ---- optimizer / learning-rate sweep -------------------------------------
#
# ep_train's default is plain Optimisers.Descent at whatever lr you pass,
# which is what the XOR demo used with 4 training patterns. At 217K
# parameters on 60K images that default is NOT stable: measured test
# accuracy over 10 full epochs went 0.664, 0.424, 0.445, 0.663, 0.689,
# 0.641, 0.657, 0.680, 0.695, 0.641 with the loss flat at ~1.94 — the
# classic too-large-step oscillation. Run this before committing to a
# long job.
function sweep_optimisers(chain, ps0, st0, codes, (Xtr, ytr), (Xte, yte);
                          epochs = 5)
    method = StaticEP(β=STATIC_BETA, T_free=T_FREE, T_nudge=T_NUDGE, dt=DT,
                      centered=CENTERED)
    configs = [(Optimisers.Descent, lr) for lr in (0.05, 0.02, 0.005)]
    append!(configs, [(Optimisers.Adam, lr) for lr in (0.01, 0.003, 0.001)])

    println("\n-- optimizer sweep ($(length(ytr)) train, $epochs epochs each) --")
    println("  optimiser   lr       accs by epoch                     final  best")
    best = nothing
    for (opt, lr) in configs
        _, ps_o, st_o = (nothing, ps0, st0)
        loader = MLUtils.DataLoader((Xtr, ytr); batchsize=BATCH, shuffle=true)
        args = PhasorNetworks.Args(lr=lr, epochs=epochs,
                                   weight_decay=WEIGHT_DECAY)
        accs = Float64[]
        cb = (e, p, stt, l) -> push!(accs, accuracy(chain, p, stt, codes, Xte, yte))
        _, ps_o, st_o = ep_train(chain, ps0, st0, loader, args;
                                 method=method,
                                 cost_fn=yb -> CodebookCost(codes, yb),
                                 optimiser=opt, callback=cb)
        @printf("  %-10s  %-7.4f  %-32s  %.4f  %.4f\n",
                nameof(opt), lr, join(map(a -> @sprintf("%.3f", a), accs), " "),
                accs[end], maximum(accs))
        if best === nothing || accs[end] > best[3]
            best = (opt, lr, accs[end])
        end
    end
    @printf("\n  best by final accuracy: %s lr=%.4f (%.4f)\n",
            nameof(best[1]), best[2], best[3])
    return best
end

# ---- training ------------------------------------------------------------
function run_method(name, method, epochs, chain, ps, st, codes,
                    (Xtr, ytr), (Xte, yte))
    loader = MLUtils.DataLoader((Xtr, ytr); batchsize=BATCH, shuffle=true)
    args = PhasorNetworks.Args(lr=STATIC_LR, epochs=epochs,
                               weight_decay=WEIGHT_DECAY)
    accs = Float64[]
    epoch_losses = Float32[]
    t0 = time()
    resids = Float64[]
    best_acc, best_ps = -1.0, ps
    cb = function (epoch, ps_now, st_now, epoch_loss)
        a = accuracy(chain, ps_now, st_now, codes, Xte, yte)
        r = settle_residual(chain, ps_now, st_now, codes, Xtr)
        w1, w2 = weight_norms(ps_now)
        push!(accs, a); push!(epoch_losses, epoch_loss); push!(resids, r)
        if a > best_acc
            best_acc = a
            best_ps = deepcopy(ps_now)
        end
        @printf("[%s] epoch %2d/%d  loss=%.4f  test-acc=%.4f  settle-resid=%.2e  |W1|=%.1f |W2|=%.1f  (%.1f s)\n",
                name, epoch, epochs, epoch_loss, a, r, w1, w2, time() - t0)
        flush(stdout)
    end
    losses, ps_out, st_out = ep_train(chain, ps, st, loader, args;
                                      method    = method,
                                      cost_fn   = yb -> CodebookCost(codes, yb),
                                      optimiser = OPTIMISER,
                                      callback  = cb)
    return (; name, losses, epoch_losses, accs, resids, ps=ps_out, st=st_out,
              best_ps, best_acc, seconds=time() - t0)
end

function main()
    @printf("PhasorNetworks EP on FashionMNIST | %d train / %d test | batch %d\n",
            N_TRAIN, N_TEST, BATCH)
    @printf("model 784→%d→%d (%d params) | opt %s lr=%g wd=%g | BLAS threads %d\n",
            HID, DOUT, 784*HID + HID*DOUT + 2*(HID + DOUT),
            nameof(OPTIMISER), STATIC_LR, WEIGHT_DECAY, BLAS.get_num_threads())

    rng = Xoshiro(SEED)
    (Xtr, ytr), (Xte, yte) = load_data()
    codes = make_codes(Xoshiro(SEED + 1))
    chain, ps0, st0 = build_chain(rng)

    check_convergence(chain, ps0, st0, codes, Xtr)
    @printf("\nchance accuracy = %.4f | init accuracy = %.4f\n",
            1/10, accuracy(chain, ps0, st0, codes, Xte, yte))

    if MODE === :calibrate
        calibrate_lockin(chain, ps0, st0, codes, Xtr, ytr)
        return
    elseif MODE === :sweep
        sweep_optimisers(chain, ps0, st0, codes, (Xtr, ytr), (Xte, yte))
        return
    end

    results = Any[]

    static = StaticEP(β=STATIC_BETA, T_free=T_FREE, T_nudge=T_NUDGE, dt=DT,
                      centered=CENTERED)
    push!(results, run_method("StaticEP", static, STATIC_EPOCHS,
                              chain, ps0, st0, codes, (Xtr, ytr), (Xte, yte)))

    if RUN_LOCKIN
        lockin = LockinEP(ε=LOCKIN_EPS, ω_p=LOCKIN_WP, n_cycles=LOCKIN_CYCLES,
                          T_warmup_cycles=2, T_free=LOCKIN_TFREE, dt=LOCKIN_DT)
        period = round(Int, 2π / (LOCKIN_WP * LOCKIN_DT))
        @printf("\nLockinEP: %d steps/gradient (period=%d steps)\n",
                LOCKIN_TFREE + (2 + LOCKIN_CYCLES) * period, period)
        # Same init as StaticEP so the comparison is apples-to-apples.
        push!(results, run_method("LockinEP", lockin, LOCKIN_EPOCHS,
                                  chain, ps0, st0, codes, (Xtr, ytr), (Xte, yte)))
    end

    println("\n== summary ==")
    for r in results
        @printf("%-10s  final loss=%.4f  best test-acc=%.4f  final=%.4f  %.1f min\n",
                r.name, r.epoch_losses[end], maximum(r.accs), r.accs[end],
                r.seconds / 60)
    end

    plt = plot(xlabel="epoch", ylabel="test accuracy",
               title="Phasor EP on FashionMNIST (784→$HID→$DOUT)", legend=:bottomright)
    for r in results
        plot!(plt, 1:length(r.accs), r.accs, lw=2, marker=:circle, label=r.name)
    end
    hline!(plt, [0.1], ls=:dash, c=:gray, label="chance")
    savefig(plt, joinpath(OUTDIR, "ep_fashionmnist_accuracy.png"))

    plt2 = plot(xlabel="epoch", ylabel="mean codebook CE loss",
                title="Phasor EP training loss", legend=:topright)
    for r in results
        plot!(plt2, 1:length(r.epoch_losses), r.epoch_losses, lw=2, label=r.name)
    end
    savefig(plt2, joinpath(OUTDIR, "ep_fashionmnist_loss.png"))
    println("\nwrote plots to $OUTDIR")
end

main()
