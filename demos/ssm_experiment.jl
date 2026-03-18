#=
SSM Encoding × Initialization × Architecture Experiment
=========================================================

Compares conditions on FashionMNIST across three axes:

  Init:      uniform | hippo
  Encoding:  complex PSK | impulse
  Model:     base (SSM only) | attention (SSM + SSMSelfAttention)

Base model:      PhasorSSM → PhasorSSM → SSMReadout
Attention model: PhasorSSM → SSMSelfAttention → PhasorSSM → SSMReadout

Complex PSK: pixel v → constant complex phasor exp(iπ(2v-1)) at each step.
  The phase information is present at every time step simultaneously.

Impulse encoding: pixel v → phase θ = (2v-1) → spike time within a period,
  then a von-Mises-shaped real current pulse at that time.  The network
  must integrate the temporally-shifted impulse to recover the phase —
  the same mechanism the ODE system uses with spike trains.  This tests
  temporal memory: information arrives as a brief pulse at a specific time,
  and the SSM must remember it through subsequent steps.

Usage:
  julia --project demos/ssm_experiment.jl [--epochs 20] [--hidden 128] [--model both]
=#

using PhasorNetworks
using Lux, Random, Optimisers, Zygote, Statistics
using MLDatasets, MLUtils, OneHotArrays
using CUDA, LuxCUDA
using Plots
using ArgParse

# ================================================================
# Experiment Runner
# ================================================================

function build_model(; C_in=28, D_hidden=128, n_classes=10, init_mode=:uniform,
                       model_type=:base)
    if model_type == :attention
        return Chain(
            PhasorSSM(C_in => D_hidden, normalize_to_unit_circle; init=init_mode),
            SSMSelfAttention(D_hidden => D_hidden, normalize_to_unit_circle),
            PhasorSSM(D_hidden => D_hidden, identity; init=init_mode),
            SSMReadout(D_hidden => n_classes),
        )
    else
        return Chain(
            PhasorSSM(C_in => D_hidden, normalize_to_unit_circle; init=init_mode),
            PhasorSSM(D_hidden => D_hidden, identity; init=init_mode),
            SSMReadout(D_hidden => n_classes),
        )
    end
end

function run_experiment(; n_epochs=20, batchsize=128, lr=3e-4, D_hidden=128,
                         seed=42, substeps=4, model_filter="both")
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

    model_types = if model_filter == "both"
        [:base, :attention]
    elseif model_filter == "attention"
        [:attention]
    else
        [:base]
    end

    conditions = Tuple{String, Symbol, Symbol, Symbol}[]
    for mt in model_types
        suffix = mt == :attention ? "+attn" : ""
        for (init_mode, encoding) in [(:uniform, :complex), (:hippo, :complex),
                                       (:uniform, :impulse), (:hippo, :impulse)]
            name = "$(init_mode)+$(encoding)$(suffix)"
            push!(conditions, (name, init_mode, encoding, mt))
        end
    end

    results = Dict{String, @NamedTuple{losses::Vector{Float64}, accs::Vector{Float64}}}()

    for (name, init_mode, encoding, model_type) in conditions
        println("="^60)
        println("  $name  (init=$init_mode, encoding=$encoding, model=$model_type)")
        println("="^60)

        rng = Xoshiro(seed)

        model = build_model(; D_hidden, init_mode, model_type)

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
    # Colors by init+encoding pair; linestyle by model type
    base_colors = Dict(
        "uniform+complex"  => :blue,
        "hippo+complex"    => :red,
        "uniform+impulse"  => :cyan,
        "hippo+impulse"    => :orange,
    )
    get_color(name) = base_colors[replace(name, "+attn" => "")]
    get_style(name) = endswith(name, "+attn") ? :dash : :solid
    get_marker(name) = endswith(name, "+attn") ? :diamond : :circle

    # Stable ordering: base conditions first, then attention
    order = [
        "uniform+complex",  "hippo+complex",  "uniform+impulse",  "hippo+impulse",
        "uniform+complex+attn", "hippo+complex+attn", "uniform+impulse+attn", "hippo+impulse+attn",
    ]

    # Accuracy curves
    p_acc = plot(title="Test Accuracy", xlabel="epoch", ylabel="accuracy (%)",
                 legend=:bottomright)
    for name in order
        haskey(results, name) || continue
        res = results[name]
        plot!(p_acc, 1:n_epochs, res.accs .* 100,
              label=name, lw=2, color=get_color(name),
              ls=get_style(name), marker=get_marker(name), ms=3)
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
        plot!(p_loss, smoothed, label=name, lw=1.5,
              color=get_color(name), ls=get_style(name), alpha=0.8)
    end

    # Final accuracy bar
    names_present = filter(n -> haskey(results, n), order)
    final_accs = [results[n].accs[end] * 100 for n in names_present]
    p_bar = bar(names_present, final_accs,
                title="Final Accuracy (epoch $n_epochs)",
                ylabel="accuracy (%)", label=nothing,
                color=[get_color(n) for n in names_present],
                xrotation=20)

    p = plot(p_acc, p_loss, p_bar;
             layout=@layout([a b; c{0.4h}]),
             size=(1100, 850),
             plot_title="PhasorSSM: Init × Encoding × Architecture")

    savefig(p, "ssm_experiment_results.png")
    println("\nPlot saved to ssm_experiment_results.png")
    return p
end

# ================================================================
# Main
# ================================================================

function experiment_main()
    s = ArgParseSettings(description="PhasorSSM Init × Encoding × Architecture experiment")
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
        "--model"
            help = "model type: base, attention, or both"
            arg_type = String
            default = "both"
    end
    parsed = ArgParse.parse_args(s)

    results = run_experiment(;
        n_epochs=parsed["epochs"],
        batchsize=parsed["batchsize"],
        lr=parsed["lr"],
        D_hidden=parsed["hidden"],
        seed=parsed["seed"],
        substeps=parsed["substeps"],
        model_filter=parsed["model"],
    )

    plot_results(results, parsed["epochs"])

    println("\n" * "="^60)
    println("  SUMMARY")
    println("="^60)
    all_names = [
        "uniform+complex", "hippo+complex", "uniform+impulse", "hippo+impulse",
        "uniform+complex+attn", "hippo+complex+attn", "uniform+impulse+attn", "hippo+impulse+attn",
    ]
    for name in all_names
        haskey(results, name) || continue
        println("  $name:  $(round(results[name].accs[end] * 100; digits=2))%")
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    experiment_main()
end
