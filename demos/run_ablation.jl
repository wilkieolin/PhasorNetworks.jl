#=
Ablation Experiment Runner + Plotting
=====================================

Runs the ablation experiments and generates result plots.

Usage:
  julia --project=. demos/run_ablation.jl
  julia --project=. demos/run_ablation.jl --epochs 20 --hidden 64
  julia --project=. demos/run_ablation.jl --lr-ssm 1e-4 --cosine-schedule
=#

include("long_range_demo.jl")
using Plots
using JLD2

# ================================================================
# Configuration
# ================================================================

function parse_ablation_args()
    s = ArgParseSettings(description="Run ablation experiments with plotting")
    @add_arg_table! s begin
        "--epochs"
            arg_type = Int
            default = 15
        "--batchsize"
            arg_type = Int
            default = 128
        "--lr"
            arg_type = Float64
            default = 3e-4
        "--lr-ssm"
            arg_type = Float64
            default = 1e-4
        "--hidden"
            arg_type = Int
            default = 64
        "--seed"
            arg_type = Int
            default = 42
        "--no-cuda"
            action = :store_true
        "--copy-length"
            help = "sequence length for copying task"
            arg_type = Int
            default = 128
        "--weight-decay"
            arg_type = Float64
            default = 0.0
        "--cosine-schedule"
            action = :store_true
        "--gc-interval"
            arg_type = Int
            default = 50
        "--output"
            help = "output directory for plots and data"
            arg_type = String
            default = "ablation_results"
    end
    return ArgParse.parse_args(s)
end

# ================================================================
# Smoothing utility
# ================================================================

function smooth(x::Vector, window::Int)
    [mean(x[max(1, i - window + 1):i]) for i in 1:length(x)]
end

# ================================================================
# Plotting
# ================================================================

function plot_ablation_losses(results::Dict, title_str::String; window::Int=20)
    p = plot(title=title_str, xlabel="training step", ylabel="loss",
             legend=:topright, lw=2, size=(700, 400))
    for (name, res) in sort(collect(results); by=first)
        plot!(p, smooth(res.losses, window), label=name)
    end
    return p
end

function plot_ablation_accs(results::Dict, title_str::String)
    p = plot(title=title_str, xlabel="epoch", ylabel="test accuracy (%)",
             legend=:bottomright, lw=2, marker=:circle, ms=4, size=(700, 400))
    for (name, res) in sort(collect(results); by=first)
        plot!(p, 1:length(res.accs), res.accs .* 100, label=name)
    end
    hline!(p, [10.0], color=:gray, ls=:dash, alpha=0.4, label="chance")
    return p
end

function plot_final_accuracy_bar(results::Dict, title_str::String)
    names = sort(collect(keys(results)))
    accs = [results[n].accs[end] * 100 for n in names]

    p = bar(names, accs, title=title_str,
            ylabel="test accuracy (%)", legend=false,
            bar_width=0.6, color=[:steelblue, :coral, :steelblue, :coral],
            size=(600, 400), xrotation=15)
    hline!(p, [10.0], color=:gray, ls=:dash, alpha=0.4)
    return p
end

# ================================================================
# Main
# ================================================================

function run()
    args = parse_ablation_args()
    n_epochs   = args["epochs"]
    batchsize  = args["batchsize"]
    lr         = Float64(args["lr"])
    lr_ssm     = Float64(args["lr-ssm"])
    D_hidden   = args["hidden"]
    seed       = args["seed"]
    use_cuda   = !args["no-cuda"] && CUDA.functional()
    L_copy     = args["copy-length"]
    wd         = Float64(args["weight-decay"])
    cosine     = args["cosine-schedule"]
    gc_int     = args["gc-interval"]
    outdir     = args["output"]

    mkpath(outdir)

    device = use_cuda ? gpu_device() : cpu_device()
    println("Device: $(use_cuda ? "CUDA GPU" : "CPU")")
    println("Config: hidden=$D_hidden, lr=$lr, lr_ssm=$lr_ssm, epochs=$n_epochs, batch=$batchsize")
    println("Output: $outdir/")

    n_classes = 10
    D_token = 16
    n_train = 10000
    n_test = 2000
    pps = 28

    all_results = Dict{String, Any}()

    common_kwargs = (; D_hidden, n_classes, n_epochs, batchsize, lr, seed,
                       use_cuda, lr_ssm, weight_decay=wd,
                       cosine_schedule=cosine, gc_interval=gc_int)

    # ---- FashionMNIST ablation: STFT vs no STFT ----
    println("\n" * "="^64)
    println("  FashionMNIST: PhasorSTFT ablation")
    println("="^64)

    train_loader, test_loader, fmnist_L, fmnist_C = load_sequential_fmnist(; batchsize, pixels_per_step=pps)

    fmnist_results = Dict{String, NamedTuple}()
    #                   (label,                stft,  attention)
    fmnist_configs = [("STFT + Attn + Dense", true,  true),
                      ("STFT + Dense",        true,  false),
                      ("Dense only",          false, false)]
    for (name, stft, attn) in fmnist_configs
        fmnist_results[name] = run_ablation(name, train_loader, test_loader,
                                             mnist_loss, intensity_to_phase;
                                             C_in=fmnist_C,
                                             use_stft=stft, use_attention=attn,
                                             readout_frac=0.25f0,
                                             common_kwargs...)
    end
    all_results["fmnist"] = fmnist_results

    # ---- Copying ablation: attention vs no attention ----
    println("\n" * "="^64)
    println("  Selective Copying (L=$L_copy): Attention ablation")
    println("="^64)

    # Attention computes an (L x L) score matrix; for long sequences this
    # requires a smaller batch size to fit in GPU memory during backprop.
    copy_batchsize = min(batchsize, max(16, 4096 ÷ L_copy))
    println("  Using batch size $copy_batchsize for copying (L=$L_copy with attention)")

    codebook = generate_copying_codebook(Xoshiro(seed), n_classes, D_token)
    x_train, y_train = generate_copying_data(Xoshiro(seed + 1000), codebook, L_copy, n_train)
    x_test, y_test   = generate_copying_data(Xoshiro(seed + 2000), codebook, L_copy, n_test)
    copy_train = DataLoader((x_train, y_train); batchsize=copy_batchsize, shuffle=true)
    copy_test  = DataLoader((x_test,  y_test);  batchsize=copy_batchsize)

    copy_kwargs = merge(common_kwargs, (batchsize=copy_batchsize,))
    copy_results = Dict{String, NamedTuple}()
    #                  (label,                  stft,  attention)
    copy_configs = [("STFT + Attn + Dense",   true,  true),
                    ("Dense + Attention",      false, true),
                    ("Dense only",            false, false)]
    for (name, stft, attn) in copy_configs
        copy_results[name] = run_ablation(name, copy_train, copy_test,
                                           copying_loss, identity;
                                           C_in=D_token,
                                           use_stft=stft, use_attention=attn,
                                           readout_frac=0.1f0,
                                           copy_kwargs...)
    end
    all_results["copying"] = copy_results

    # ---- Save raw results ----
    results_file = joinpath(outdir, "ablation_data.jld2")
    jldsave(results_file;
            fmnist_results=Dict(k => (losses=v.losses, accs=v.accs) for (k, v) in fmnist_results),
            copy_results=Dict(k => (losses=v.losses, accs=v.accs) for (k, v) in copy_results),
            config=Dict("epochs" => n_epochs, "hidden" => D_hidden, "lr" => lr,
                         "lr_ssm" => lr_ssm, "seed" => seed, "copy_length" => L_copy))
    println("\nResults saved to $results_file")

    # ---- Plot ----
    println("\nGenerating plots...")

    # FashionMNIST
    p_fm_loss = plot_ablation_losses(fmnist_results, "FashionMNIST: Training Loss")
    p_fm_acc  = plot_ablation_accs(fmnist_results, "FashionMNIST: Test Accuracy")
    p_fm = plot(p_fm_loss, p_fm_acc; layout=(1, 2), size=(1200, 400),
                plot_title="FashionMNIST STFT Ablation")
    savefig(p_fm, joinpath(outdir, "fmnist_ablation.png"))

    # Copying
    p_cp_loss = plot_ablation_losses(copy_results, "Copying (L=$L_copy): Training Loss")
    p_cp_acc  = plot_ablation_accs(copy_results, "Copying (L=$L_copy): Test Accuracy")
    p_cp = plot(p_cp_loss, p_cp_acc; layout=(1, 2), size=(1200, 400),
                plot_title="Selective Copying Attention Ablation")
    savefig(p_cp, joinpath(outdir, "copying_ablation.png"))

    # Combined final accuracy bar chart
    combined = merge(
        Dict("FashionMNIST: $k" => v for (k, v) in fmnist_results),
        Dict("Copying: $k" => v for (k, v) in copy_results))
    p_bar = plot_final_accuracy_bar(combined, "Ablation: Final Test Accuracy")
    savefig(p_bar, joinpath(outdir, "ablation_final_accuracy.png"))

    println("\nPlots saved to $outdir/")

    # ---- Print summary table ----
    println("\n" * "="^56)
    println("  ABLATION SUMMARY")
    println("="^56)
    @printf("  %-30s | %-14s\n", "Configuration", "Final Accuracy")
    println("  " * "-"^30 * "-|-" * "-"^14)
    for (task, results) in [("FashionMNIST", fmnist_results), ("Copying", copy_results)]
        for (name, res) in sort(collect(results); by=first)
            @printf("  %-30s | %12.2f%%\n", "$task: $name", res.accs[end] * 100)
        end
    end
    println("="^56)

    println("\nDone.")
end

if abspath(PROGRAM_FILE) == @__FILE__
    run()
end
