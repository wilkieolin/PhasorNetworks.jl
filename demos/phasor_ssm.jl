#=
Discrete State Space Model for Phasor Networks — Demo
=====================================================

PSK-encode FashionMNIST as a complex time series and classify it
using a stack of PhasorSSM layers with a Codebook readout and similarity loss.

All SSM primitives (PhasorSSM, SSMReadout, phasor_kernel, causal_conv,
hippo_legs_diagonal, psk_encode) are imported from PhasorNetworks.
=#

using PhasorNetworks
using Lux, Random, Optimisers, Zygote, Statistics
using MLUtils, OneHotArrays
using CUDA, LuxCUDA
using ArgParse

# ================================================================
# Model + Loss
# ================================================================

function create_model(; D_hidden=128, n_classes=10, C_in=28, init=:uniform)
    model = Chain(
        PhasorSSM(C_in => D_hidden, normalize_to_unit_circle; init),
        PhasorSSM(D_hidden => D_hidden, identity; init),
        SSMReadout(D_hidden => n_classes),
    )
    return model
end

"""
Loss function matching PhasorNetworks.train interface: loss(x, y, model, ps, st).
PSK-encodes images, forward-passes through the model (SSM → readout → Codebook),
and computes similarity_loss against one-hot targets.
"""
function ssm_loss(x, y, model, ps, st)
    x_enc = psk_encode(x)
    sims, _ = model(x_enc, ps, st)       # (n_classes, batch) similarities
    return mean(similarity_loss(sims, y))  # scalar
end

# ================================================================
# Evaluation
# ================================================================

function evaluate(model, ps, st, loader, device)
    correct = 0
    total = 0
    for (x, y) in loader
        x_dev = x |> device
        y_dev = y |> device
        x_enc = psk_encode(x_dev)
        sims, _ = model(x_enc, ps, st)
        preds = argmax(sims; dims=1)
        truth = argmax(y_dev; dims=1)
        correct += sum(preds .== truth)
        total += size(x, 3)
    end
    return correct / total
end

# ================================================================
# Main
# ================================================================

function parse_args()
    s = ArgParseSettings(description="PhasorSSM: Discrete state-space phasor network on FashionMNIST")
    @add_arg_table! s begin
        "--epochs"
            help = "number of training epochs"
            arg_type = Int
            default = 5
        "--batchsize"
            help = "training batch size"
            arg_type = Int
            default = 128
        "--lr"
            help = "learning rate"
            arg_type = Float64
            default = 3e-4
        "--hidden"
            help = "hidden dimension (number of oscillators per layer)"
            arg_type = Int
            default = 128
        "--seed"
            help = "random seed"
            arg_type = Int
            default = 42
        "--no-cuda"
            help = "disable CUDA even if available"
            action = :store_true
        "--init"
            help = "parameter initialization: uniform or hippo"
            arg_type = String
            default = "uniform"
    end
    return ArgParse.parse_args(s)
end

function main()
    parsed = parse_args()
    n_epochs  = parsed["epochs"]
    batchsize = parsed["batchsize"]
    lr        = Float64(parsed["lr"])
    D_hidden  = parsed["hidden"]
    seed      = parsed["seed"]
    no_cuda   = parsed["no-cuda"]
    init_mode = Symbol(parsed["init"])

    rng = Xoshiro(seed)

    println("Loading FashionMNIST...")
    train_data = fashion_mnist_data(:train)
    test_data  = fashion_mnist_data(:test)

    x_train = Float32.(train_data.features)                   # 28 × 28 × 60000
    y_train = Float32.(onehotbatch(train_data.targets, 0:9))  # 10 × 60000
    x_test  = Float32.(test_data.features)
    y_test  = Float32.(onehotbatch(test_data.targets, 0:9))

    train_loader = DataLoader((x_train, y_train); batchsize, shuffle=true)
    test_loader  = DataLoader((x_test,  y_test);  batchsize)

    println("Building model (D_hidden=$D_hidden, init=$init_mode)...")
    model = create_model(; D_hidden, n_classes=10, C_in=28, init=init_mode)
    ps, st = Lux.setup(rng, model)

    _count(nt::NamedTuple) = isempty(nt) ? 0 : sum(_count(v) for v in values(nt))
    _count(x::AbstractArray) = length(x)
    _count(_) = 0
    n_params = _count(ps)
    println("  Trainable parameters: $n_params")

    # Move to GPU if available
    use_cuda = !no_cuda && CUDA.functional()
    device = use_cuda ? gpu_device() : cpu_device()
    ps = ps |> device
    st = st |> device
    println("  Device: $(use_cuda ? "CUDA GPU" : "CPU")")

    args = Args(epochs=1, batchsize=batchsize, lr=lr, use_cuda=use_cuda)

    for epoch in 1:n_epochs
        # Train one epoch using PhasorNetworks.train
        losses, ps, st = train(model, ps, st, train_loader, ssm_loss, args)
        avg_loss = mean(losses)

        # Evaluate on test set
        acc = evaluate(model, ps, st, test_loader, device)

        println("Epoch $epoch/$n_epochs  avg_loss=$(round(avg_loss; digits=4))  test_acc=$(round(acc; digits=4))")
    end

    return ps, st
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
