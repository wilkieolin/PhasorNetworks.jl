#=
SSM Encoding × Initialization Experiment
==========================================

Compares 4 conditions on FashionMNIST:
  1. uniform init + complex PSK encoding
  2. hippo   init + complex PSK encoding
  3. uniform init + impulse encoding (real-valued temporal current)
  4. hippo   init + impulse encoding

Complex PSK: pixel v → constant complex phasor exp(iπ(2v-1)) at each step.
  The phase information is present at every time step simultaneously.

Impulse encoding: pixel v → phase θ = (2v-1) → spike time within a period,
  then a von-Mises-shaped real current pulse at that time.  The network
  must integrate the temporally-shifted impulse to recover the phase —
  the same mechanism the ODE system uses with spike trains.  This tests
  temporal memory: information arrives as a brief pulse at a specific time,
  and the SSM must remember it through subsequent steps.

Usage:
  julia --project demos/ssm_experiment.jl [--epochs 20] [--hidden 128]
=#

using PhasorNetworks
using Lux, LuxCore, Random, Optimisers, Zygote, Statistics
using MLDatasets, MLUtils, OneHotArrays
using NNlib: batched_mul
using ChainRulesCore: ignore_derivatives
using CUDA, LuxCUDA
using Plots
using ArgParse

# ================================================================
# Import core SSM definitions (without running main)
# ================================================================

include("phasor_ssm.jl")

# ================================================================
# Impulse Encoding
# ================================================================

"""
    impulse_encode(images; substeps=4) -> ComplexF32 array (C × L × B)

Encode pixel values as temporally-shifted real-valued impulse currents.

Each row of the image gets `substeps` discrete time steps (one "period").
The pixel's phase determines WHERE within those substeps the impulse fires:
  pixel v ∈ [0,1] → phase θ = 2v-1 → spike time t = (θ+1)/2 · T

A von-Mises-shaped pulse centered at the spike time produces a real current.
Total sequence length L = H × substeps (e.g. 28 × 4 = 112).

This genuinely tests temporal memory: the phase information arrives as a
brief pulse at a specific sub-step, and the SSM must integrate and remember
it through subsequent steps to produce the correct output phase.

Compare with `psk_encode`: there the phase is directly encoded as a complex
value exp(iπθ) at every step — no temporal memory is required, just
instantaneous readout.
"""
function impulse_encode(images::AbstractArray{<:Real, 3}; substeps::Int=4)
    # Encoding has no trainable params — stop Zygote from tracing
    return ignore_derivatives() do
        _impulse_encode_impl(images; substeps)
    end
end

function _impulse_encode_impl(images::AbstractArray{<:Real, 3}; substeps::Int=4)
    H, W, B = size(images)
    C = W
    T = Float32(substeps)
    t_sigma = T * 0.1f0
    kappa = 1f0 / (1f0 - cos(2f0 * Float32(π) * t_sigma / T))

    phases = 2f0 .* images .- 1f0                     # H × W × B → [-1,1]
    phases_ct = permutedims(phases, (2, 1, 3))          # C × H × B

    # spike_times[c, row, b] = position in [0, T) within the row's period
    spike_times = (phases_ct .+ 1f0) ./ 2f0 .* T       # C × H × B

    # Build substep time indices on the same device as input
    ts_dev = similar(phases, Float32, substeps)
    copyto!(ts_dev, Float32.(1:substeps))
    ts_sub = reshape(ts_dev, 1, substeps, 1)              # 1 × substeps × 1

    slices = map(1:H) do r
        t_spike = spike_times[:, r:r, :]                # C × 1 × B
        dt = ts_sub .- t_spike                          # C × substeps × B
        exp.(kappa .* (cos.(2f0 * Float32(π) .* dt ./ T) .- 1f0))
    end
    signal = cat(slices...; dims=2)                     # C × L × B

    return complex.(signal, zero(signal))
end

# ================================================================
# Experiment Runner
# ================================================================

function run_experiment(; n_epochs=20, batchsize=128, lr=3e-4, D_hidden=128,
                         seed=42, substeps=4)
    println("Loading FashionMNIST...")
    train_data = FashionMNIST(split=:train)
    test_data  = FashionMNIST(split=:test)
    x_train = Float32.(train_data.features)
    y_train = Float32.(onehotbatch(train_data.targets, 0:9))
    x_test  = Float32.(test_data.features)
    y_test  = Float32.(onehotbatch(test_data.targets, 0:9))

    train_loader = DataLoader((x_train, y_train); batchsize, shuffle=true)
    test_loader  = DataLoader((x_test,  y_test);  batchsize)

    use_cuda = CUDA.functional()
    device = use_cuda ? gpu_device() : cpu_device()
    println("Device: $(use_cuda ? "CUDA GPU" : "CPU")\n")

    conditions = [
        ("uniform+complex",  :uniform, :complex),
        ("hippo+complex",    :hippo,   :complex),
        ("uniform+impulse",  :uniform, :impulse),
        ("hippo+impulse",    :hippo,   :impulse),
    ]

    results = Dict{String, @NamedTuple{losses::Vector{Float64}, accs::Vector{Float64}}}()

    for (name, init_mode, encoding) in conditions
        println("="^60)
        println("  $name  (init=$init_mode, encoding=$encoding)")
        println("="^60)

        rng = Xoshiro(seed)

        model = Chain(
            PhasorSSM(28 => D_hidden, normalize_to_unit_circle; init=init_mode),
            PhasorSSM(D_hidden => D_hidden, identity; init=init_mode),
            SSMReadout(0.25f0),
            Codebook(D_hidden => 10),
        )

        ps, st = Lux.setup(rng, model)
        ps = ps |> device
        st = st |> device

        encode_fn = if encoding == :complex
            x -> psk_encode(x)
        else
            x -> impulse_encode(x; substeps)
        end

        loss_fn = function(x, y, model, ps, st)
            x_enc = encode_fn(x)
            sims, _ = model(x_enc, ps, st)
            return mean(similarity_loss(sims, y))
        end

        eval_fn = function(loader)
            correct = 0; total = 0
            for (x, y) in loader
                x_dev = x |> device
                y_dev = y |> device
                x_enc = encode_fn(x_dev)
                sims, _ = model(x_enc, ps, st)
                correct += sum(argmax(sims; dims=1) .== argmax(y_dev; dims=1))
                total += size(x, 3)
            end
            return correct / total
        end

        args = Args(epochs=1, batchsize=batchsize, lr=lr, use_cuda=use_cuda)
        all_losses = Float64[]
        accs = Float64[]

        for epoch in 1:n_epochs
            losses, ps, st = train(model, ps, st, train_loader, loss_fn, args)
            append!(all_losses, losses)
            acc = eval_fn(test_loader)
            push!(accs, acc)
            println("  Epoch $epoch/$n_epochs  loss=$(round(mean(losses); digits=4))  acc=$(round(acc; digits=4))")
        end

        results[name] = (losses=all_losses, accs=accs)
    end

    return results
end

# ================================================================
# Plotting
# ================================================================

function plot_results(results, n_epochs)
    colors = Dict(
        "uniform+complex"  => :blue,
        "hippo+complex"    => :red,
        "uniform+impulse"  => :cyan,
        "hippo+impulse"    => :orange,
    )
    order = ["uniform+complex", "hippo+complex", "uniform+impulse", "hippo+impulse"]

    # Accuracy curves
    p_acc = plot(title="Test Accuracy", xlabel="epoch", ylabel="accuracy (%)",
                 legend=:bottomright)
    for name in order
        haskey(results, name) || continue
        res = results[name]
        plot!(p_acc, 1:n_epochs, res.accs .* 100,
              label=name, lw=2, color=colors[name], marker=:circle, ms=3)
    end
    hline!(p_acc, [10.0], color=:gray, ls=:dash, alpha=0.4, label="chance")

    # Smoothed loss
    p_loss = plot(title="Training Loss (smoothed)", xlabel="step", ylabel="loss",
                  legend=:topright)
    for name in order
        haskey(results, name) || continue
        res = results[name]
        w = min(50, length(res.losses) ÷ max(n_epochs, 1))
        w = max(w, 1)
        smoothed = [mean(res.losses[max(1,i-w+1):i]) for i in 1:length(res.losses)]
        plot!(p_loss, smoothed, label=name, lw=1.5, color=colors[name], alpha=0.8)
    end

    # Final accuracy bar
    names_present = filter(n -> haskey(results, n), order)
    final_accs = [results[n].accs[end] * 100 for n in names_present]
    p_bar = bar(names_present, final_accs,
                title="Final Accuracy (epoch $n_epochs)",
                ylabel="accuracy (%)", label=nothing,
                color=[colors[n] for n in names_present],
                xrotation=15)

    p = plot(p_acc, p_loss, p_bar;
             layout=@layout([a b; c{0.4h}]),
             size=(1000, 800),
             plot_title="PhasorSSM: Init × Encoding")

    savefig(p, "ssm_experiment_results.png")
    println("\nPlot saved to ssm_experiment_results.png")
    return p
end

# ================================================================
# Main
# ================================================================

function experiment_main()
    s = ArgParseSettings(description="PhasorSSM Init × Encoding experiment")
    @add_arg_table! s begin
        "--epochs"
            help = "number of training epochs per condition"
            arg_type = Int
            default = 20
        "--batchsize"
            help = "training batch size"
            arg_type = Int
            default = 128
        "--lr"
            help = "learning rate"
            arg_type = Float64
            default = 3e-4
        "--hidden"
            help = "hidden dimension"
            arg_type = Int
            default = 128
        "--seed"
            help = "random seed"
            arg_type = Int
            default = 42
        "--substeps"
            help = "substeps per row for impulse encoding (L = 28 × substeps)"
            arg_type = Int
            default = 4
    end
    parsed = ArgParse.parse_args(s)

    results = run_experiment(;
        n_epochs=parsed["epochs"],
        batchsize=parsed["batchsize"],
        lr=parsed["lr"],
        D_hidden=parsed["hidden"],
        seed=parsed["seed"],
        substeps=parsed["substeps"],
    )

    plot_results(results, parsed["epochs"])

    println("\n" * "="^60)
    println("  SUMMARY")
    println("="^60)
    for name in ["uniform+complex", "hippo+complex", "uniform+impulse", "hippo+impulse"]
        haskey(results, name) || continue
        println("  $name:  $(round(results[name].accs[end] * 100; digits=2))%")
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    experiment_main()
end
