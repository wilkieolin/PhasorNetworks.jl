# wave_attribution.jl — where does the learning happen: the wave sheet or the head?
#
# The FashionMNIST wave classifier is  drive → PhasorWaveSheet → head(PhasorDense
# 784→64) → Codebook. The head has ~50k params; the wave coupling has ~7 (DoG).
# So: is the accuracy mostly the head's, with the wave sheet a fixed prior — or
# does *training* the wave sheet matter?
#
# Three conditions, identical head/data/seed, decompose it:
#   1. dense           — head only, NO wave sheet          (head-alone ceiling)
#   2. wave-frozen      — wave sheet coupling FROZEN at init + trained head
#                        (head + the wave sheet's *fixed* transform)
#   3. wave-trained     — everything trained               (full model)
#
# Reading:
#   dense → wave-frozen : does the wave sheet's fixed propagation help the head?
#   wave-frozen → wave-trained : does *training* the coupling add anything more?
#
# Run:  julia --project=. demos/wave_attribution.jl
#       (env-overridable: WAVE_N_TRAIN, WAVE_N_TEST, WAVE_EPOCHS, WAVE_L, WAVE_HID)

ENV["GKSwstype"] = "100"
using PhasorNetworks, Lux, Zygote, Optimisers, OneHotArrays, ComponentArrays
using Random, Statistics, Printf

_envi(k, d) = parse(Int, get(ENV, k, string(d)))
const N_TRAIN = _envi("WAVE_N_TRAIN", 8000)
const N_TEST  = _envi("WAVE_N_TEST",  2000)
const BATCH   = 128
const EPOCHS  = _envi("WAVE_EPOCHS", 6)
const L_STEPS = _envi("WAVE_L", 5)
const HID     = _envi("WAVE_HID", 64)
const LR      = 3f-3
const SHEET   = 28

drive_encode(x, L) = (b = size(x, 3);
    repeat(reshape(Phase.((2f0 .* x .- 1f0) .* 0.5f0), SHEET^2, 1, b), 1, L, 1))
flatten_phase(x) = Phase.((2f0 .* reshape(x, SHEET^2, size(x, 3)) .- 1f0) .* 0.5f0)

wave_model() = Chain(
    WrappedFunction(x -> drive_encode(x, L_STEPS)),
    # pinned to potential coupling to reproduce the documented attribution result
    PhasorWaveSheet(SHEET, SHEET; transmit = :potential, saturating = true, init_log_g = log(0.02)),
    WrappedFunction(x -> x[:, end, :]),
    PhasorDense(SHEET^2 => HID, normalize_to_unit_circle),
    Codebook(HID => 10; init_mode = :orthogonal),
)
dense_model() = Chain(
    WrappedFunction(flatten_phase),
    PhasorDense(SHEET^2 => HID, normalize_to_unit_circle),
    Codebook(HID => 10; init_mode = :orthogonal),
)

nparams(ps) = length(ComponentArray(ps))
# Recursively zero every array leaf of a gradient tree (NamedTuple/Tuple of
# arrays), so a frozen sublayer receives a zero gradient and never updates.
zero_tree(g::AbstractArray) = zero(g)
zero_tree(g::NamedTuple)    = map(zero_tree, g)
zero_tree(g::Tuple)         = map(zero_tree, g)
zero_tree(g)                = g

function load_subset(split, n, rng)
    d = fashion_mnist_data(split)
    idx = randperm(rng, length(d.targets))[1:n]
    return d.features[:, :, idx], d.targets[idx]
end
function minibatches(X, y, batch, rng; shuffle = true)
    n = length(y); order = shuffle ? randperm(rng, n) : collect(1:n)
    return ((X[:, :, order[i:min(i+batch-1, n)]], y[order[i:min(i+batch-1, n)]]) for i in 1:batch:n)
end
function accuracy(model, ps, st, X, y)
    correct = 0; total = 0
    for (xb, yb) in minibatches(X, y, 512, Random.default_rng(); shuffle = false)
        ŷ, _ = model(xb, ps, st)
        c, t = evaluate_accuracy(ŷ, Float32.(onehotbatch(yb, 0:9)), :similarity)
        correct += c[1]; total += t
    end
    return correct / total
end

# freeze_layer: name of a Chain sublayer whose gradient is zeroed each step
# (so its params never move), or nothing to train everything.
function train_model!(model, ps, st, Xtr, ytr, Xte, yte; label, freeze_layer = nothing)
    opt = Optimisers.setup(Optimisers.Adam(LR), ps)
    rng = Xoshiro(123)
    @printf("  [%s] params = %d%s\n", label, nparams(ps),
            freeze_layer === nothing ? "" : "  (frozen: $(freeze_layer))")
    local acc = 0f0
    for epoch in 1:EPOCHS
        el = Float32[]
        for (xb, yb) in minibatches(Xtr, ytr, BATCH, rng)
            yoh = Float32.(onehotbatch(yb, 0:9))
            loss, gs = Zygote.withgradient(ps) do p
                ŷ, _ = model(xb, p, st)
                mean(evaluate_loss(ŷ, yoh, :similarity))
            end
            g = gs[1]
            if freeze_layer !== nothing
                g = merge(g, NamedTuple{(freeze_layer,)}((zero_tree(getproperty(g, freeze_layer)),)))
            end
            opt, ps = Optimisers.update(opt, ps, g)
            push!(el, loss)
        end
        acc = accuracy(model, ps, st, Xte, yte)
        @printf("  [%s] epoch %d/%d  train loss %.4f   test acc %.3f\n", label, epoch, EPOCHS, mean(el), acc)
    end
    return acc
end

println("="^68)
println("Wave-sheet vs head attribution   ($(N_TRAIN)/$(N_TEST), $(EPOCHS) ep, L=$(L_STEPS))")
println("="^68)

rng = Xoshiro(42)
Xtr, ytr = load_subset(:train, N_TRAIN, rng)
Xte, yte = load_subset(:test, N_TEST, rng)

println("\n[1] dense (head only, no wave sheet):")
md = dense_model(); pd, sd = Lux.setup(Xoshiro(1), md)
a_dense = train_model!(md, pd, sd, Xtr, ytr, Xte, yte; label = "dense")

println("\n[2] wave, coupling FROZEN at init (only head trains):")
mf = wave_model(); pf, sf = Lux.setup(Xoshiro(1), mf)
a_frozen = train_model!(mf, pf, sf, Xtr, ytr, Xte, yte; label = "wave-frozen", freeze_layer = :layer_2)

println("\n[3] wave, everything trained:")
mt = wave_model(); pt, stt = Lux.setup(Xoshiro(1), mt)
a_train = train_model!(mt, pt, stt, Xtr, ytr, Xte, yte; label = "wave-trained")

println("\n[4] Attribution (test accuracy):")
@printf("     head only (dense)            %.3f\n", a_dense)
@printf("     head + FIXED wave transform  %.3f   (Δ vs head = %+.3f)\n", a_frozen, a_frozen - a_dense)
@printf("     head + TRAINED wave          %.3f   (Δ vs frozen = %+.3f)\n", a_train, a_train - a_frozen)
println("\n  Reading:")
println("   • Δ(frozen − dense)   = benefit of the wave sheet's *fixed* propagation/mixing.")
println("   • Δ(trained − frozen) = benefit of actually *training* the ~7 coupling params.")
println("   • dense as a fraction of full = how much the head alone already achieves.")
@printf("   • head-alone is %.1f%% of the full wave model's accuracy.\n", 100 * a_dense / a_train)
