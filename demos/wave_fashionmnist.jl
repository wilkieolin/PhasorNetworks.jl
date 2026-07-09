# wave_fashionmnist.jl — a wave-based FashionMNIST classifier
#
# Demo 2 from docs/rf_wave_network_implementation.md §5: train a classifier
# whose spatial computation is done by a recurrent PhasorWaveSheet, feeding
# the standard similarity readout (Codebook) + similarity_loss pipeline.
#
# This exercises constraints (c) and (d) together on a real task:
#   (c) the wave sheet trains end-to-end through the discrete phase-SSM path
#       (Zygote AD through the Buffer/FFT recurrence);
#   (d) computation happens by *wave propagation + interference* on a 28×28
#       sheet — the image is injected as a spatial drive, the wave mixes it
#       over L steps, and a similarity readout classifies the resulting phase
#       field.
#
# We compare against a matched baseline with the SAME trainable head but NO
# wave sheet, to isolate what the (≈9-parameter) wave layer contributes.
#
# This is a first-pass demo tuned for a quick CPU run (a subset, few epochs),
# not a leaderboard entry. Scale the consts up for a serious run.
#
# Run:  julia --project=. demos/wave_fashionmnist.jl

ENV["GKSwstype"] = "100"

using PhasorNetworks, Lux, Zygote, Optimisers, OneHotArrays, ComponentArrays
using Random, Statistics, Printf
using Plots

const OUTDIR = joinpath(@__DIR__, "wave_out")
isdir(OUTDIR) || mkpath(OUTDIR)

# ---- config (env-overridable, e.g. WAVE_N_TRAIN=60000 WAVE_EPOCHS=12) ----
_envi(k, d) = parse(Int, get(ENV, k, string(d)))
_envf(k, d) = parse(Float32, get(ENV, k, string(d)))
const N_TRAIN = _envi("WAVE_N_TRAIN", 20000)
const N_TEST  = _envi("WAVE_N_TEST",  5000)
const BATCH   = _envi("WAVE_BATCH",   128)
const EPOCHS  = _envi("WAVE_EPOCHS",  8)
const L_STEPS = _envi("WAVE_L",       5)     # wave propagation steps
const HID     = _envi("WAVE_HID",     64)    # readout head width
const LR      = _envf("WAVE_LR",      3f-3)
const SHEET   = 28                            # 28×28 sheet = one cell per pixel
const STENCIL_R = _envi("WAVE_STENCIL_R", 3)  # learnable-stencil radius

# ---- image → sheet drive ---------------------------------------------
# Intensity v∈[0,1] → phase (v-0.5)∈[-0.5,0.5] (injective under exp(iπθ)),
# broadcast over L timesteps as a constant spatial drive.
function drive_encode(x, L)
    B = size(x, 3)
    ph = Phase.((2f0 .* x .- 1f0) .* 0.5f0)                     # (28,28,B)
    return repeat(reshape(ph, size(x, 1) * size(x, 2), 1, B), 1, L, 1)  # (784,L,B)
end
# Baseline encoder: same phase map, flattened, no time axis.
flatten_phase(x) = Phase.((2f0 .* reshape(x, size(x,1)*size(x,2), size(x,3)) .- 1f0) .* 0.5f0)

# ---- models ----------------------------------------------------------
function wave_model()
    Chain(
        WrappedFunction(x -> drive_encode(x, L_STEPS)),
        PhasorWaveSheet(SHEET, SHEET; saturating = true, init_log_g = log(0.02)),
        WrappedFunction(x -> x[:, end, :]),                    # last-step phase field (784,B)
        PhasorDense(SHEET^2 => HID, normalize_to_unit_circle),
        Codebook(HID => 10; init_mode = :orthogonal),
    )
end
# Same architecture as wave_model, but the wave sheet's coupling is a free
# learnable stencil (bookmark 2) instead of the 9-scalar DoG — i.e. real
# trainable capacity inside the wave layer itself.
function wave_stencil_model()
    Chain(
        WrappedFunction(x -> drive_encode(x, L_STEPS)),
        PhasorWaveSheet(SHEET, SHEET; coupling = :stencil, stencil_radius = STENCIL_R,
                        saturating = true, init_log_g = log(0.02)),
        WrappedFunction(x -> x[:, end, :]),
        PhasorDense(SHEET^2 => HID, normalize_to_unit_circle),
        Codebook(HID => 10; init_mode = :orthogonal),
    )
end
function baseline_model()
    Chain(
        WrappedFunction(flatten_phase),
        PhasorDense(SHEET^2 => HID, normalize_to_unit_circle),
        Codebook(HID => 10; init_mode = :orthogonal),
    )
end
# PhasorConv stack (the repo's proven conv classifier structure) — a different
# spatial inductive bias to compare the wave sheet against.
function conv_model()
    Chain(
        WrappedFunction(x -> Phase.((2f0 .* reshape(x, SHEET, SHEET, 1, size(x, 3)) .- 1f0) .* 0.5f0)),
        PhasorConv((16, 16), 1 => 3, soft_angle),   # (28,28,1,B) → (13,13,3,B) Phase
        PhasorConv((8, 8), 3 => 1, soft_angle),     # → (6,6,1,B) Phase
        FlattenLayer(),                             # → (36, B) Phase
        PhasorDense(36 => HID, normalize_to_unit_circle),
        Codebook(HID => 10; init_mode = :orthogonal),
    )
end

nparams(ps) = length(ComponentArray(ps))

# ---- data ------------------------------------------------------------
function load_subset(split, n, rng)
    d = fashion_mnist_data(split)
    idx = randperm(rng, length(d.targets))[1:n]
    return d.features[:, :, idx], d.targets[idx]
end

function minibatches(X, y, batch, rng; shuffle = true)
    n = length(y)
    order = shuffle ? randperm(rng, n) : collect(1:n)
    return (( X[:, :, order[i:min(i+batch-1, n)]],
              y[order[i:min(i+batch-1, n)]] ) for i in 1:batch:n)
end

# ---- train / eval ----------------------------------------------------
function accuracy(model, ps, st, X, y)
    correct = 0; total = 0
    for (xb, yb) in minibatches(X, y, 512, Random.default_rng(); shuffle = false)
        ŷ, _ = model(xb, ps, st)
        yoh = Float32.(onehotbatch(yb, 0:9))
        c, t = evaluate_accuracy(ŷ, yoh, :similarity)
        correct += c[1]; total += t
    end
    return correct / total
end

function train_model!(model, ps, st, Xtr, ytr, Xte, yte; label = "model")
    opt = Optimisers.setup(Optimisers.Adam(LR), ps)
    rng = Xoshiro(123)
    curve_loss = Float32[]; curve_acc = Float32[]
    @printf("  [%s] params = %d\n", label, nparams(ps))
    for epoch in 1:EPOCHS
        elosses = Float32[]
        for (xb, yb) in minibatches(Xtr, ytr, BATCH, rng)
            yoh = Float32.(onehotbatch(yb, 0:9))
            loss, gs = Zygote.withgradient(ps) do p
                ŷ, _ = model(xb, p, st)
                mean(evaluate_loss(ŷ, yoh, :similarity))
            end
            opt, ps = Optimisers.update(opt, ps, gs[1])
            push!(elosses, loss)
        end
        acc = accuracy(model, ps, st, Xte, yte)
        push!(curve_loss, mean(elosses)); push!(curve_acc, acc)
        @printf("  [%s] epoch %d/%d  train loss %.4f   test acc %.3f\n",
                label, epoch, EPOCHS, mean(elosses), acc)
    end
    return ps, st, curve_loss, curve_acc
end

# ---------------------------------------------------------------------
println("="^68)
println("Wave-based FashionMNIST classifier")
println("  subset $(N_TRAIN)/$(N_TEST), batch $(BATCH), $(EPOCHS) epochs, L=$(L_STEPS) wave steps")
println("="^68)

rng = Xoshiro(42)
Xtr, ytr = load_subset(:train, N_TRAIN, rng)
Xte, yte = load_subset(:test,  N_TEST,  rng)

println("\n[1] Wave model — DoG coupling (9 scalars):")
mw = wave_model(); psw, stw = Lux.setup(Xoshiro(1), mw)
psw, stw, lw, aw = train_model!(mw, psw, stw, Xtr, ytr, Xte, yte; label = "wave-dog")

println("\n[2] Wave model — learnable stencil coupling (R=$(STENCIL_R)):")
ms = wave_stencil_model(); pss, sts = Lux.setup(Xoshiro(1), ms)
pss, sts, ls, as = train_model!(ms, pss, sts, Xtr, ytr, Xte, yte; label = "wave-sten")

println("\n[3] Dense baseline (same head, no wave sheet):")
mb = baseline_model(); psb, stb = Lux.setup(Xoshiro(1), mb)
psb, stb, lb, ab = train_model!(mb, psb, stb, Xtr, ytr, Xte, yte; label = "dense")

println("\n[4] PhasorConv stack (different spatial bias):")
mc = conv_model(); psc, stc = Lux.setup(Xoshiro(1), mc)
psc, stc, lc, ac = train_model!(mc, psc, stc, Xtr, ytr, Xte, yte; label = "conv")

println("\n[5] Results (test accuracy, chance = 0.100):")
@printf("     %-22s %6.3f   (%d params)\n", "wave — DoG coupling",     aw[end], nparams(psw))
@printf("     %-22s %6.3f   (%d params)\n", "wave — stencil (R=$(STENCIL_R))", as[end], nparams(pss))
@printf("     %-22s %6.3f   (%d params)\n", "dense baseline",          ab[end], nparams(psb))
@printf("     %-22s %6.3f   (%d params)\n", "PhasorConv stack",        ac[end], nparams(psc))
@printf("     learnable stencil vs DoG coupling: %+.3f acc for +%d coupling params\n",
        as[end] - aw[end], nparams(pss) - nparams(psw))

# ---- figures ---------------------------------------------------------
cur = plot(1:EPOCHS, aw; lw = 2, marker = :o, label = "wave — DoG", legend = :bottomright,
           title = "FashionMNIST test accuracy", xlabel = "epoch", ylabel = "accuracy")
plot!(cur, 1:EPOCHS, as; lw = 2, marker = :o, label = "wave — stencil")
plot!(cur, 1:EPOCHS, ab; lw = 2, marker = :o, label = "dense baseline")
plot!(cur, 1:EPOCHS, ac; lw = 2, marker = :o, label = "PhasorConv stack")
hline!(cur, [0.1]; ls = :dash, c = :gray, label = "chance")
savefig(cur, joinpath(OUTDIR, "fmnist_accuracy.png"))
println("     saved fmnist_accuracy.png")

# Visualize the wave field mixing one test image over the L propagation steps.
sample = Xte[:, :, 1:1]
drive = drive_encode(sample, L_STEPS)
field, _ = mw[2](drive, psw.layer_2, stw.layer_2)           # PhasorWaveSheet output (784,L,1) Phase
field = reshape(Float32.(field), SHEET, SHEET, L_STEPS)
snaps = plot(layout = (1, L_STEPS + 1), size = (180 * (L_STEPS + 1), 200))
heatmap!(snaps[1], sample[:, :, 1]; c = :grays, title = "input", aspect_ratio = 1,
         colorbar = false, axis = false, yflip = true)
for t in 1:L_STEPS
    heatmap!(snaps[t + 1], field[:, :, t]; c = :twilight, clims = (-1, 1),
             title = "wave t$t", aspect_ratio = 1, colorbar = false, axis = false, yflip = true)
end
savefig(snaps, joinpath(OUTDIR, "fmnist_wavefield.png"))
println("     saved fmnist_wavefield.png")

println("\nDone. Figures in $(OUTDIR)")
