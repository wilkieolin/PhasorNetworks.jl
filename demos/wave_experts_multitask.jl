# wave_experts_multitask.jl — Rung 2 of the MoE-patch demonstrator ladder.
#
# Rung 1 proved emergent key-routing + causal specialisation on a clean synthetic
# codebook. Rung 2 asks: does it survive REAL, high-entropy inputs?
#
# Multi-task FashionMNIST. Each example is (image, task_id t). K binary sub-tasks
# give the SAME image different labels, so the network cannot ignore the task — it
# must route on it. The task is delivered as a strong, full-range **phase tag**
# broadcast into a strip of columns that every expert's row-band can read (real
# routing signal, unlike FashionMNIST-10's thin amplitude phase). The classifier
# head reads only the IMAGE columns — the tag strip is sliced off — so task
# information reaches the head ONLY through the expert bank: routing must work, and
# each expert must mark its task, or the head cannot disambiguate.
#
# EMERGENT, then verified:
#   • routing confusion tag→expert forms a permutation (discovered from random keys)
#   • zeroing keys collapses routing toward chance (keys load-bearing)
#   • CAUSAL ablation: disable expert perm(t) ⇒ ONLY task t degrades
#
# Run:  julia --project=. demos/wave_experts_multitask.jl
#
using PhasorNetworks, Lux, Zygote, Optimisers, Random, Statistics, Printf
using Random: Xoshiro; using Optimisers: Adam; using Zygote: withgradient

_envi(k, d) = parse(Int, get(ENV, k, string(d)))
const SHEET = 28
const E     = 4                       # experts = tasks
const TAGW  = 14                      # tag strip width (cols 1:TAGW, all rows) — must
                                      # dominate the routing read, so the image (rest of
                                      # the band) doesn't blur the key match
const IMGW  = SHEET - TAGW            # image columns the head sees
const L     = 4
const HID   = 64
const GINIT = 1f-3                    # ~0 coupling ⇒ sheet passes the mark through
const N_TRAIN = _envi("MT_N_TRAIN", 5000)
const N_TEST  = _envi("MT_N_TEST",  2000)
const BATCH   = _envi("MT_BATCH",   128)
const EPOCHS  = _envi("MT_EPOCHS",  8)
const LR      = 3f-3

# K binary sub-tasks over FashionMNIST classes (0..9). Distinct functions of the
# same image ⇒ the label depends on the task, forcing task routing.
# Four binary one-vs-rest questions over a DISJOINT PARTITION of the 10 classes.
# Disjointness is the key: the positive sets are mutually exclusive, so if two tasks
# were routed to the same expert (same mark), the head would face the same
# (image, mark) input demanding opposite answers — a hard contradiction. That forces
# each tag onto its own expert (no merge/collapse) and keeps the positives balanced.
# Each disjoint group also contains a DISTINCTIVE anchor class (footwear, a top,
# trouser, bag) so the head can actually solve every task above baseline — otherwise
# a near-baseline task (e.g. the mutually-confusable coat/dress pair) barely uses its
# expert's mark and its ablation looks "weak" even though routing is correct.
const TASK_SETS = (Set([5, 7, 9]),      # t1: footwear (sandal/sneaker/boot)  30%
                   Set([0, 2, 6]),      # t2: top / pullover / shirt          30%
                   Set([1, 3]),         # t3: trouser (anchor) + dress        20%
                   Set([8, 4]))         # t4: bag (anchor) + coat             20%
task_label(t, cls) = cls in TASK_SETS[t] ? 1 : 0

# Fixed task tags: full-range phase pattern per task, over TAGW columns, constant
# across rows so every expert's band sees the identical tag tile.
const CB  = Xoshiro(2024)
const TAG = [2f0 .* rand(CB, Float32, TAGW) .- 1f0 for _ in 1:E]   # each (TAGW,)

# image (28×28×B) + task ids → phase drive (SHEET,SHEET,L,B) with tag overlaid
function build_drive(imgs, tasks)
    B = size(imgs, 3)
    ph = (2f0 .* imgs .- 1f0) .* 0.5f0                     # (28,28,B) thin image phase
    field = copy(ph)
    for b in 1:B, c in 1:TAGW
        field[:, c, b] .= TAG[tasks[b]][c]                # tag strip (cols 1:TAGW), all rows
    end
    return Phase.(repeat(reshape(field, SHEET * SHEET, 1, B), 1, L, 1))
end

const EXP = WaveExpertSheet(SHEET, SHEET; n_experts = E, routing = :input,
                            route_feature = :matched, transmit = :potential,
                            init_log_g = log(GINIT), balance = false)

# head reads ONLY the image columns of the output field (tag strip sliced off).
function image_cols(yfield_flat)  # (SHEET*SHEET, B) phases → (SHEET*IMGW, B)
    B = size(yfield_flat, 2)
    f = reshape(yfield_flat, SHEET, SHEET, B)
    return reshape(f[:, (TAGW + 1):SHEET, :], SHEET * IMGW, B)
end

# ---- data ------------------------------------------------------------
function subset(split, n, rng)
    d = fashion_mnist_data(split)
    idx = randperm(rng, length(d.targets))[1:n]
    return d.features[:, :, idx], d.targets[idx]
end
# expand each image into K task-examples (image repeated, one per task)
function task_batch(imgs, cls, rng)
    B = length(cls)
    tasks = rand(rng, 1:E, B)
    y = [task_label(tasks[b], cls[b]) for b in 1:B]
    return imgs, tasks, y
end

# ---- model + head ----------------------------------------------------
# `drive` is prebuilt OUTSIDE the gradient (it depends on data, not params, and
# build_drive mutates arrays — not Zygote-safe).
function fwd(heads, drive, ps, st)
    ye, _ = EXP(drive, ps.wave, st.wave)
    yimg  = image_cols(Float32.(ye[:, end, :]))           # (SHEET*IMGW, B) phases
    h, = heads
    yh, _ = h(Phase.(yimg), ps.head, st.head)             # PhasorDense
    feat = vcat(cospi.(Float32.(yh)), sinpi.(Float32.(yh)))
    return ps.W2 * feat                                    # (2, B) logits
end
celoss(logits, y) = begin
    lp = logits .- log.(sum(exp.(logits); dims = 1) .+ 1f-12)
    -mean(lp[y[b] + 1, b] for b in 1:length(y))
end

# ---- metrics ---------------------------------------------------------
fired(ps, st, imgs, tasks) = begin
    g = route_stats(EXP, ps.wave, st.wave, build_drive(imgs, tasks)).gate
    B = size(g, 3); [argmax(vec(sum(g[:, :, b]; dims = 2))) for b in 1:B]
end
function per_task_acc(heads, ps, st, imgs, cls; abl = 0, perm = nothing)
    accs = zeros(E)
    psx = ps
    if abl > 0
        bp = copy(ps.wave.bind_phase); bp[abl] = 0f0
        psx = merge(ps, (; wave = merge(ps.wave, (; bind_phase = bp))))
    end
    for t in 1:E
        tasks = fill(t, length(cls))
        y = [task_label(t, c) for c in cls]
        lg = fwd(heads, build_drive(imgs, tasks), psx, st)
        pred = [argmax(lg[:, b]) - 1 for b in 1:length(y)]
        accs[t] = mean(pred .== y)
    end
    return accs
end

# ---------------------------------------------------------------------
function main()
    println("="^70)
    println("Rung 2 — multi-task FashionMNIST ($(E) tasks, tag cols=1:$(TAGW), head sees cols $(TAGW+1):$(SHEET))")
    println("="^70)
    rng = Xoshiro(42)
    Xtr, ctr = subset(:train, N_TRAIN, rng)
    Xte, cte = subset(:test,  N_TEST,  rng)

    h = PhasorDense(SHEET * IMGW => HID, normalize_to_unit_circle)
    ph, sh = Lux.setup(Xoshiro(7), h)
    pe, se = Lux.setup(Xoshiro(1), EXP)
    W2 = 0.1f0 .* randn(Xoshiro(3), Float32, 2, 2 * HID)
    ps = (; wave = pe, head = ph, W2 = W2); st = (; wave = se, head = sh)
    heads = (h,)

    opt = Optimisers.setup(Adam(LR), ps)
    trng = Xoshiro(123)
    for epoch in 1:EPOCHS
        losses = Float32[]
        order = randperm(trng, N_TRAIN)
        for i in 1:BATCH:N_TRAIN
            idx = order[i:min(i + BATCH - 1, N_TRAIN)]
            imgs = Xtr[:, :, idx]; cls = ctr[idx]
            _, tasks, y = task_batch(imgs, cls, trng)
            drive = build_drive(imgs, tasks)
            l, g = withgradient(p -> celoss(fwd(heads, drive, p, st), y), ps)
            opt, ps = Optimisers.update(opt, ps, g[1])
            push!(losses, l)
            rs = route_stats(EXP, ps.wave, st.wave, build_drive(imgs, tasks))
            st = merge(st, (; wave = merge(st.wave,
                          (; route_bias = update_moe_bias(st.wave.route_bias, rs.gate; rate = 3f-2)))))
        end
        accs = per_task_acc(heads, ps, st, Xte, cte)
        @printf("epoch %d/%d  loss %.4f  per-task acc %s  mean %.3f\n",
                epoch, EPOCHS, mean(losses), string(round.(accs, digits = 3)), mean(accs))
    end

    # ---- verification ----
    println("\n[routing confusion  rows=task tag, cols=fired expert]")
    tsk = repeat(1:E, inner = N_TEST)
    imgs_rep = cat((Xte for _ in 1:E)...; dims = 3)
    pred = fired(ps, st, imgs_rep, tsk)
    conf = zeros(Int, E, E)
    for k in 1:length(tsk); conf[tsk[k], pred[k]] += 1; end
    for t in 1:E; println("  tag $t → ", conf[t, :]); end
    perm = [argmax(conf[t, :]) for t in 1:E]
    is_perm = length(unique(perm)) == E
    route_acc = mean(pred[k] == perm[tsk[k]] for k in 1:length(tsk))
    @printf("  permutation %s (bijection: %s)  routing acc %.3f\n",
            string(perm), is_perm ? "YES" : "no", route_acc)

    ps0 = merge(ps, (; wave = merge(ps.wave, (; bind_key = zero(ps.wave.bind_key)))))
    pred0 = fired(ps0, st, imgs_rep, tsk)
    @printf("  routing w/ keys=0: %.3f  (chance %.2f ⇒ keys load-bearing)\n",
            mean(pred0[k] == perm[tsk[k]] for k in 1:length(tsk)), 1 / E)

    println("\n[causal ablation — disable one expert's bind, watch per-task accuracy]")
    base = per_task_acc(heads, ps, st, Xte, cte)
    @printf("  baseline per-task acc: %s\n", string(round.(base, digits = 3)))
    localized = true
    for t in 1:E
        j = perm[t]
        acc = per_task_acc(heads, ps, st, Xte, cte; abl = j)
        drops = base .- acc
        own = drops[t]; other = maximum(drops[setdiff(1:E, t)])
        @printf("  ablate expert %d (tag %d): Δacc task %d = %+.3f | worst other = %+.3f  %s\n",
                j, t, t, own, other, own > other + 0.02 ? "✓ localised" : "· weak")
        own > other + 0.02 || (localized = false)
    end

    ok = is_perm && route_acc > 0.9 && mean(base) > 0.7 && localized
    println("\nRUNG 2 ", ok ? "PASS ✓ — emergent task routing + causal specialisation on real data" :
                            "INCOMPLETE — inspect metrics above")
end
main()
