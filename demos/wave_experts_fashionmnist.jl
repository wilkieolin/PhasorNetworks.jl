# wave_experts_fashionmnist.jl — WaveExpertSheet FashionMNIST classifier
#
# Wires the §6 prototype (WaveExpertSheet, docs/wavesheet_experts_design.md) into
# a real task. The image is injected as a spatial drive; a sparsely-gated bank of
# experts reads patches of the drive, routes (top-1, DeepSeek loss-free bias), and
# stamps a learned bind phasor onto the injected wavefront; the linear sheet
# propagates it (parallel scan); a similarity readout (PhasorDense → Codebook)
# classifies the final phase field. Trained end-to-end on the DISCRETE SSM.
#
# It answers two things on a real task:
#   1. does adding the expert bank help over the plain wave sheet (same head)?
#   2. does the gate stay non-collapsed while training on real data — the §6
#      go/no-go beyond the synthetic probe (Open Q §7.1)?
#
# The router bias is balanced out-of-graph via route_stats + update_moe_bias
# (cheap: no rollout), which also gives the per-epoch gate entropy / load trace.
#
# Quick CPU run by default; scale via env (WAVE_N_TRAIN, WAVE_EPOCHS, ...).
# Run:  julia --project=. demos/wave_experts_fashionmnist.jl

ENV["GKSwstype"] = "100"

using PhasorNetworks, Lux, Zygote, Optimisers, OneHotArrays, ComponentArrays
using Random, Statistics, Printf
using Plots

const OUTDIR = joinpath(@__DIR__, "wave_out")
isdir(OUTDIR) || mkpath(OUTDIR)

_envi(k, d) = parse(Int, get(ENV, k, string(d)))
_envf(k, d) = parse(Float32, get(ENV, k, string(d)))
const N_TRAIN = _envi("WAVE_N_TRAIN", 6000)
const N_TEST  = _envi("WAVE_N_TEST",  2000)
const BATCH   = _envi("WAVE_BATCH",   128)
const EPOCHS  = _envi("WAVE_EPOCHS",  5)
const L_STEPS = _envi("WAVE_L",       5)
const HID     = _envi("WAVE_HID",     64)
const LR      = _envf("WAVE_LR",      3f-3)
const SHEET   = 28
const NEXP    = _envi("WAVE_NEXP",    7)      # experts (row-bands); needs ≤ SHEET
const GINIT   = _envf("WAVE_GINIT",   0.05f0) # subcritical coupling (linear scan)
const RF      = Symbol(get(ENV, "WAVE_RF", "coherence"))  # :coherence|:dispersion|:matched

# ---- image → sheet drive (phase, constant over L) --------------------
drive_encode(x, L) = begin
    ph = Phase.((2f0 .* x .- 1f0) .* 0.5f0)
    repeat(reshape(ph, size(x,1)*size(x,2), 1, size(x,3)), 1, L, 1)   # (784,L,B)
end

# ---- readout head (shared by both models) ----------------------------
make_head(seed) = begin
    h = PhasorDense(SHEET^2 => HID, normalize_to_unit_circle)
    b = Codebook(HID => 10; init_mode = :orthogonal)
    ph, sh = Lux.setup(Xoshiro(seed), h)
    pb, sb = Lux.setup(Xoshiro(seed + 1), b)
    return (h, b), (; head = ph, book = pb), (; head = sh, book = sb)
end
apply_head(heads, yfield, ps, st) = begin
    h, b = heads
    yh, _ = h(yfield, ps.head, st.head)
    s, _  = b(yh, ps.book, st.book)
    return s
end

# ---- models: expert sheet vs plain sheet (same head) -----------------
const EXP = WaveExpertSheet(SHEET, SHEET; n_experts = NEXP, routing = :input,
                            route_feature = RF, transmit = :potential, saturating = false,
                            init_log_g = log(GINIT), balance = false)
const PLAIN = PhasorWaveSheet(SHEET, SHEET; transmit = :potential,
                              saturating = false, init_log_g = log(GINIT))

nparams(ps) = length(ComponentArray(ps))

# ---- data ------------------------------------------------------------
function load_subset(split, n, rng)
    d = fashion_mnist_data(split)
    idx = randperm(rng, length(d.targets))[1:n]
    return d.features[:, :, idx], d.targets[idx]
end
function minibatches(X, y, batch, rng; shuffle = true)
    n = length(y); order = shuffle ? randperm(rng, n) : collect(1:n)
    return (( X[:, :, order[i:min(i+batch-1, n)]], y[order[i:min(i+batch-1, n)]] )
            for i in 1:batch:n)
end

# ---- train / eval ----------------------------------------------------
function accuracy(fwd, ps, st, X, y)
    correct = 0; total = 0
    for (xb, yb) in minibatches(X, y, 512, Random.default_rng(); shuffle = false)
        ŷ = fwd(xb, ps, st)
        yoh = Float32.(onehotbatch(yb, 0:9))
        c, t = evaluate_accuracy(ŷ, yoh, :similarity)
        correct += c[1]; total += t
    end
    return correct / total
end

# Expert model: forward + out-of-graph router-bias balancing via route_stats.
function train_expert!(ps, st, Xtr, ytr, Xte, yte)
    fwd(xb, p, s) = begin
        ye, _ = EXP(drive_encode(xb, L_STEPS), p.wave, s.wave)
        apply_head((HEAD_LAYERS), ye[:, end, :], p, s)
    end
    opt = Optimisers.setup(Optimisers.Adam(LR), ps)
    rng = Xoshiro(123)
    accs = Float32[]; ents = Float32[]
    @printf("  [wave+experts] params = %d  (experts=%d)\n", nparams(ps), NEXP)
    for epoch in 1:EPOCHS
        elosses = Float32[]
        for (xb, yb) in minibatches(Xtr, ytr, BATCH, rng)
            yoh = Float32.(onehotbatch(yb, 0:9))
            loss, gs = Zygote.withgradient(p -> mean(evaluate_loss(fwd(xb, p, st), yoh, :similarity)), ps)
            opt, ps = Optimisers.update(opt, ps, gs[1])
            push!(elosses, loss)
            # cheap out-of-graph load balancing (no rollout)
            rs = route_stats(EXP, ps.wave, st.wave, drive_encode(xb, L_STEPS))
            newbias = update_moe_bias(st.wave.route_bias, rs.gate; rate = 5f-2)
            st = merge(st, (wave = merge(st.wave, (route_bias = newbias,)),))
        end
        acc = accuracy(fwd, ps, st, Xte, yte)
        rs = route_stats(EXP, ps.wave, st.wave, drive_encode(Xte[:, :, 1:min(256, N_TEST)], L_STEPS))
        push!(accs, acc); push!(ents, rs.entropy)
        @printf("  [wave+experts] epoch %d/%d  loss %.4f  acc %.3f  gate-ent %.3f/%.3f  load %s\n",
                epoch, EPOCHS, mean(elosses), acc, rs.entropy, log(Float32(NEXP)),
                string(round.(rs.load, digits = 2)))
    end
    return ps, st, accs, ents
end

function train_plain!(ps, st, Xtr, ytr, Xte, yte)
    fwd(xb, p, s) = begin
        yw, _ = PLAIN(drive_encode(xb, L_STEPS), p.wave, s.wave)
        apply_head((HEAD_LAYERS), yw[:, end, :], p, s)
    end
    opt = Optimisers.setup(Optimisers.Adam(LR), ps)
    rng = Xoshiro(123); accs = Float32[]
    @printf("  [wave-only]    params = %d\n", nparams(ps))
    for epoch in 1:EPOCHS
        elosses = Float32[]
        for (xb, yb) in minibatches(Xtr, ytr, BATCH, rng)
            yoh = Float32.(onehotbatch(yb, 0:9))
            loss, gs = Zygote.withgradient(p -> mean(evaluate_loss(fwd(xb, p, st), yoh, :similarity)), ps)
            opt, ps = Optimisers.update(opt, ps, gs[1]); push!(elosses, loss)
        end
        acc = accuracy(fwd, ps, st, Xte, yte); push!(accs, acc)
        @printf("  [wave-only]    epoch %d/%d  loss %.4f  acc %.3f\n", epoch, EPOCHS, mean(elosses), acc)
    end
    return ps, st, accs
end

# ---------------------------------------------------------------------
println("="^68)
println("WaveExpertSheet FashionMNIST — $(N_TRAIN)/$(N_TEST), $(EPOCHS) ep, L=$(L_STEPS), $(NEXP) experts, route=:$(RF)")
println("="^68)
rng = Xoshiro(42)
Xtr, ytr = load_subset(:train, N_TRAIN, rng)
Xte, yte = load_subset(:test,  N_TEST,  rng)

# shared head layers (same init for both models for a fair comparison)
const HEAD_LAYERS, HEAD_PS, HEAD_ST = make_head(7)

println("\n[1] wave + experts:")
pe0, se0 = Lux.setup(Xoshiro(1), EXP)
pe = merge(HEAD_PS, (wave = pe0,)); se = merge(HEAD_ST, (wave = se0,))
pe, se, acc_e, ent_e = train_expert!(pe, se, Xtr, ytr, Xte, yte)

println("\n[2] wave only (no experts, same head):")
pp0, sp0 = Lux.setup(Xoshiro(1), PLAIN)
pp = merge(HEAD_PS, (wave = pp0,)); sp = merge(HEAD_ST, (wave = sp0,))
pp, sp, acc_p = train_plain!(pp, sp, Xtr, ytr, Xte, yte)

println("\n[3] Results (test accuracy, chance = 0.100):")
@printf("     %-20s %6.3f   (%d params)\n", "wave + experts", acc_e[end], nparams(pe))
@printf("     %-20s %6.3f   (%d params)\n", "wave only",      acc_p[end], nparams(pp))
@printf("     experts vs plain: %+.3f acc\n", acc_e[end] - acc_p[end])
@printf("     final gate entropy %.3f / %.3f (max) — %s\n", ent_e[end], log(Float32(NEXP)),
        ent_e[end] > 0.6f0 * log(Float32(NEXP)) ? "no collapse" : "COLLAPSED")

# ---- figures ---------------------------------------------------------
cur = plot(1:EPOCHS, acc_e; lw = 2, marker = :o, label = "wave + experts",
           legend = :bottomright, title = "FashionMNIST test accuracy",
           xlabel = "epoch", ylabel = "accuracy")
plot!(cur, 1:EPOCHS, acc_p; lw = 2, marker = :o, label = "wave only")
hline!(cur, [0.1]; ls = :dash, c = :gray, label = "chance")
savefig(cur, joinpath(OUTDIR, "fmnist_experts_accuracy.png"))
entp = plot(1:EPOCHS, ent_e; lw = 2, marker = :o, label = "gate entropy",
            title = "Gate entropy (go/no-go)", xlabel = "epoch", ylabel = "entropy (nats)")
hline!(entp, [log(NEXP)]; ls = :dash, c = :gray, label = "max = ln $(NEXP)")
savefig(entp, joinpath(OUTDIR, "fmnist_experts_entropy.png"))
println("\nSaved fmnist_experts_accuracy.png, fmnist_experts_entropy.png in $(OUTDIR)")
