#=
Long-Range Dependency Benchmarks for Phasor SSMs
=================================================

Validates that HiPPO-initialized Phasor SSMs capture long-range dependencies
by running two benchmarks with three initialization conditions:

  1. Selective Copying (synthetic) — recall a phase pattern placed at position 0
     after a long blank gap. Tested at multiple sequence lengths.
  2. Sequential FashionMNIST (column-by-column) — classify garments processed as
     28-step sequences with 28 channels per timestep (one column per step).

Three conditions:
  - hippo:        HiPPO-LegS multi-timescale initialization, scaled for sequence length
  - uniform:      uniformly spread frequencies, moderate decay scaled for sequence length
  - short-memory: deliberately high leakage relative to sequence, cannot retain info

Ablation mode (--benchmark ablation):
  Tests the contribution of individual architectural components:
  - FashionMNIST: compares models with and without the ResonantSTFT input layer
  - Copying: compares models with and without SSMSelfAttention

Key insight: PhasorDense uses Δt=1 in its convolutional mode, so the kernel
magnitude at lag n is |K[n]| = exp(λ·n).  For a signal at position 0 to survive
to position L, we need |λ| ~ O(1/L).  The HiPPO eigenvalues define the *shape*
of the multi-timescale memory (log-spaced decay rates); this script scales them
so the memory window matches the target sequence length.

Usage:
  julia --project=. demos/long_range_demo.jl --benchmark both --epochs 10
  julia --project=. demos/long_range_demo.jl --benchmark copying --copy-lengths 128,256,512
  julia --project=. demos/long_range_demo.jl --benchmark fmnist --epochs 15
  julia --project=. demos/long_range_demo.jl --benchmark ablation --epochs 15
  julia --project=. demos/long_range_demo.jl --benchmark fmnist --epochs 20 --lr 0.008 --lr-ssm 0.002 --weight-decay 0.0001 --cosine-schedule
=#

using PhasorNetworks
using Lux, Random, Optimisers, Zygote, Statistics
using MLDatasets, MLUtils, OneHotArrays
using CUDA, LuxCUDA
using ArgParse
using Printf

# ================================================================
# 1. ArgParse
# ================================================================

function parse_args()
    s = ArgParseSettings(description="Long-range dependency benchmarks for Phasor SSMs")
    @add_arg_table! s begin
        "--epochs"
            help = "number of training epochs"
            arg_type = Int
            default = 10
        "--batchsize"
            help = "training batch size"
            arg_type = Int
            default = 64
        "--lr"
            help = "learning rate"
            arg_type = Float64
            default = 3e-4
        "--hidden"
            help = "hidden dimension (number of oscillators per layer)"
            arg_type = Int
            default = 64
        "--seed"
            help = "random seed"
            arg_type = Int
            default = 42
        "--no-cuda"
            help = "disable CUDA even if available"
            action = :store_true
        "--benchmark"
            help = "which benchmark to run: copying, fmnist, both, or ablation"
            arg_type = String
            default = "both"
        "--d-token"
            help = "token dimension for selective copying"
            arg_type = Int
            default = 16
        "--n-train"
            help = "number of training samples for selective copying"
            arg_type = Int
            default = 10000
        "--n-test"
            help = "number of test samples for selective copying"
            arg_type = Int
            default = 2000
        "--copy-lengths"
            help = "comma-separated sequence lengths for copying task"
            arg_type = String
            default = "128,256,512"
        "--pixels-per-step"
            help = "pixels grouped per timestep for sequential FashionMNIST (28=column-wise, 1=full 784)"
            arg_type = Int
            default = 28
        "--lr-ssm"
            help = "learning rate for SSM dynamics params (log_neg_lambda, omega). 0 = use --lr for all."
            arg_type = Float64
            default = 0.0
        "--weight-decay"
            help = "L2 weight decay on connection weights only (not SSM params)"
            arg_type = Float64
            default = 0.0
        "--cosine-schedule"
            help = "enable cosine LR annealing"
            action = :store_true
        "--gc-interval"
            help = "GC every N batches (0 = every batch). Set higher for SSM workloads."
            arg_type = Int
            default = 50
    end
    return ArgParse.parse_args(s)
end

# ================================================================
# 2. Data Generation — Selective Copying
# ================================================================

"""
    generate_copying_codebook(rng, n_classes, D_token) -> ComplexF32 matrix

Generate a fixed codebook of unit-complex class prototypes, shared across
train and test splits.
"""
function generate_copying_codebook(rng::AbstractRNG, n_classes::Int, D_token::Int)
    codebook_phases = random_symbols(rng, (D_token, n_classes))
    return angle_to_complex(codebook_phases)
end

"""
    generate_copying_data(rng, codebook, seq_len, n_samples; noise_scale)

Generate synthetic selective-copying data from a shared codebook.

Each sample is a (D_token × seq_len) complex sequence where:
- Position 1: the target class's codebook vector (the signal to remember)
- Positions 2 through seq_len: random complex noise at `noise_scale` amplitude

The noise creates superposition interference — the model must retain the initial
signal's phase despite accumulated noise.  With short memory (fast decay), the
initial signal is overwhelmed by noise.  With long memory (slow decay, e.g.
HiPPO), the initial signal persists and can be read out.

Returns (x, y) where x is (D_token × seq_len × n_samples) ComplexF32 and
y is (n_classes × n_samples) Float32 one-hot.
"""
function generate_copying_data(rng::AbstractRNG, codebook::AbstractMatrix{<:Complex},
                               seq_len::Int, n_samples::Int;
                               noise_scale::Float32=0.5f0)
    D_token, n_classes = size(codebook)
    codebook_phases = complex_to_angle(codebook)

    labels = rand(rng, 0:n_classes-1, n_samples)

    # Generate as Phase arrays so PhasorDense uses the Dirac path
    x = zeros(Float32, D_token, seq_len, n_samples)
    for i in 1:n_samples
        x[:, 1, i] .= Float32.(codebook_phases[:, labels[i] + 1])
    end

    # Add random phase noise at positions 2:end to create interference
    if noise_scale > 0f0
        noise_phases = 2f0 .* rand(rng, Float32, D_token, seq_len - 1, n_samples) .- 1f0
        x[:, 2:end, :] .= noise_phases
    end

    y = Float32.(onehotbatch(labels, 0:n_classes-1))
    return Phase.(x), y
end

# ================================================================
# 3. Data Loading — Sequential MNIST
# ================================================================

"""
    load_sequential_fmnist(; batchsize, pixels_per_step)

Load FashionMNIST and reshape for sequential processing.

Groups `pixels_per_step` pixels into each timestep as separate channels.
With pixels_per_step=28: (28, 28, N) → L=28, C=28 (column-wise, fast)
With pixels_per_step=1:  (784, 1, N) → L=784, C=1 (pixel-wise, slow)

The raw data is returned as (L × C × N) Float32.

Returns (train_loader, test_loader, L, C_in).
"""
function load_sequential_fmnist(; batchsize::Int=64, pixels_per_step::Int=28)
    println("Loading FashionMNIST...")
    train_data = FashionMNIST(split=:train)
    test_data  = FashionMNIST(split=:test)

    n_pixels = 784
    @assert n_pixels % pixels_per_step == 0 "784 must be divisible by pixels_per_step"
    L = n_pixels ÷ pixels_per_step
    C_in = pixels_per_step

    # Flatten to pixel sequences: (28,28,N) -> (L, C, N)
    x_train = reshape(Float32.(train_data.features), L, C_in, :)
    y_train = Float32.(onehotbatch(train_data.targets, 0:9))
    x_test  = reshape(Float32.(test_data.features), L, C_in, :)
    y_test  = Float32.(onehotbatch(test_data.targets, 0:9))

    train_loader = DataLoader((x_train, y_train); batchsize, shuffle=true)
    test_loader  = DataLoader((x_test,  y_test);  batchsize)

    println("  Sequence length L=$L, channels C=$C_in (pixels_per_step=$pixels_per_step)")
    return train_loader, test_loader, L, C_in
end

# ================================================================
# 4. Model Construction
# ================================================================

# Convert Phase arrays to complex (for attention layers); pass complex through unchanged
_to_complex(x::AbstractArray{<:Phase}) = angle_to_complex(x)
_to_complex(x::AbstractArray{<:Complex}) = x

"""
    create_model(; C_in, D_hidden, n_classes, init_mode, readout_frac)

Build a two-layer Phasor SSM with codebook readout.
"""
function create_model(; C_in::Int, D_hidden::Int=64, n_classes::Int=10,
                        init_mode::Symbol=:uniform, readout_frac::Float32=0.25f0)
    return Chain(
        PhasorDense(C_in => D_hidden, normalize_to_unit_circle;
                    init_mode=init_mode, use_bias=false),
        PhasorDense(D_hidden => D_hidden, identity;
                    init_mode=init_mode, use_bias=false),
        SSMReadout(D_hidden => n_classes; readout_frac=readout_frac),
    )
end

"""
    create_ablation_model(; C_in, D_hidden, n_classes, use_stft, use_attention, readout_frac)

Build a model with optional ResonantSTFT and SSMSelfAttention for ablation testing.
"""
function create_ablation_model(; C_in::Int, D_hidden::Int=64, n_classes::Int=10,
                                 use_stft::Bool=true, use_attention::Bool=true,
                                 readout_frac::Float32=0.25f0)
    layers = []

    # Input layer: ResonantSTFT (trainable omega, outputs complex) or PhasorDense
    if use_stft
        push!(layers, ResonantSTFT(C_in => D_hidden, normalize_to_unit_circle))
    else
        push!(layers, PhasorDense(C_in => D_hidden, normalize_to_unit_circle; use_bias=false))
    end

    # Optional attention (requires complex input; Phase output from prior layer
    # must be converted to complex on the unit circle)
    if use_attention
        push!(layers, Lux.WrappedFunction(_to_complex))
        push!(layers, SSMSelfAttention(D_hidden => D_hidden, normalize_to_unit_circle))
    end

    # Second SSM layer + readout
    push!(layers, PhasorDense(D_hidden => D_hidden, identity; use_bias=false))
    push!(layers, SSMReadout(D_hidden => n_classes; readout_frac=readout_frac))

    return Chain(Tuple(layers)...)
end

"""
    scale_dynamics_for_length(ps, st, seq_len, D_hidden, init_mode)

Scale per-channel decay rates (λ) and frequencies (ω) so the SSM's memory
window matches the target sequence length.

With Δt=1 (hardcoded in PhasorDense's 3D dispatch), the kernel magnitude at
lag n is |K[n]| = exp(λ·n).  For the signal at position 0 to have meaningful
influence at position L, we need |λ| ≈ O(1/L).

- **HiPPO**: takes the standard HiPPO-LegS eigenvalue *shape* (log-spaced
  decay magnitudes from 0.5 to N-0.5) and rescales so the slowest channel's
  memory window spans ~2L steps.  This preserves the multi-timescale structure
  while matching the sequence length.
- **Uniform**: all channels get |λ| = 2/L (signal decays to ~13% at position L).
  Frequencies spread across [0.1, 1.5] rad/step.

Must be called BEFORE moving to GPU.
"""
# Helper: apply a function to all layers that have log_neg_lambda in params
function _map_ssm_layers(fn_ps, fn_st, ps::NamedTuple, st::NamedTuple)
    ps_dict = Dict(pairs(ps))
    st_dict = Dict(pairs(st))
    for k in keys(ps)
        layer_ps = ps[k]
        layer_st = get(st, k, NamedTuple())
        if layer_ps isa NamedTuple && haskey(layer_ps, :log_neg_lambda)
            ps_dict[k] = fn_ps(layer_ps)
            if layer_st isa NamedTuple
                st_dict[k] = fn_st(layer_st)
            end
        end
    end
    return NamedTuple(ps_dict), NamedTuple(st_dict)
end

function scale_dynamics_for_length(ps::NamedTuple, st::NamedTuple,
                                   seq_len::Int, D_hidden::Int, init_mode::Symbol)
    if init_mode == :hippo
        λ_base, ω_base = hippo_legs_diagonal(D_hidden)
        scale = Float32(seq_len)
        λ_scaled = λ_base ./ scale
        ω_scaled = ω_base ./ scale
        log_neg_lambda = Float32.(log.(-λ_scaled))
        omega = ω_scaled
    else  # :uniform
        λ_val = 2f0 / Float32(seq_len)
        log_neg_lambda = fill(Float32(log(λ_val)), D_hidden)
        omega = Float32.(collect(range(0.1f0, 1.5f0; length=D_hidden)))
    end

    return _map_ssm_layers(
        lp -> merge(lp, (log_neg_lambda = log_neg_lambda,)),
        ls -> haskey(ls, :omega) ? merge(ls, (omega = omega,)) : ls,
        ps, st)
end

"""
    force_short_memory(ps, st, seq_len; memory_frac)

Override parameters to create a short-memory baseline relative to sequence length.
Sets |λ| = 1/(memory_frac*L) so memory decays within ~memory_frac*L steps.
Must be called BEFORE moving to GPU.
"""
function force_short_memory(ps::NamedTuple, st::NamedTuple, seq_len::Int;
                            memory_frac::Float32=0.05f0)
    decay_val = 1f0 / (memory_frac * Float32(seq_len))
    log_val = Float32(log(decay_val))
    omega_val = 1.0f0

    return _map_ssm_layers(
        lp -> merge(lp, (log_neg_lambda = fill(log_val, size(lp.log_neg_lambda)),)),
        ls -> haskey(ls, :omega) ? merge(ls, (omega = fill(omega_val, size(ls.omega)),)) : ls,
        ps, st)
end

# ================================================================
# 5. Loss Functions
# ================================================================

# Cross-entropy on similarity logits — works at any hidden dimension,
# unlike similarity_loss which requires high-dimensional (~512+) codebooks
# for the random prototypes to be approximately orthogonal.

function _softmax_ce(sims, y)
    # sims: (n_classes, B) similarity scores in [-1, 1]
    # y: (n_classes, B) one-hot targets
    # Scale similarities to logit range and apply cross-entropy
    logits = 5f0 .* sims  # scale factor to sharpen softmax
    # Numerically stable log-softmax
    logits_max = maximum(logits; dims=1)
    shifted = logits .- logits_max
    log_sum_exp = log.(sum(exp.(shifted); dims=1))
    log_probs = shifted .- log_sum_exp
    # Cross-entropy: -Σ y · log(p)
    return -mean(sum(y .* log_probs; dims=1))
end

function copying_loss(x, y, model, ps, st)
    sims, _ = model(x, ps, st)
    return _softmax_ce(sims, y)
end

function mnist_loss(x, y, model, ps, st)
    x_enc = intensity_to_phase(x)
    sims, _ = model(x_enc, ps, st)
    return _softmax_ce(sims, y)
end

"""
    intensity_to_phase(images) -> Phase array (C × L × B)

Encode pixel intensities as phases: 0 (black) → 0 (0°), 1 (white) → 0.5 (90°).
Returns Phase arrays so PhasorDense uses the Dirac discretization path.
"""
function intensity_to_phase(images::AbstractArray{<:Real, 3})
    H, W, B = size(images)
    phases = Float32.(images) .* 0.5f0           # [0,1] → [0, 0.5]
    phases_ct = permutedims(phases, (2, 1, 3))   # channels × time × batch
    return Phase.(phases_ct)
end

# ================================================================
# 6. Evaluation
# ================================================================

function evaluate(model, ps, st, loader, device; encode_fn=identity)
    correct = 0
    total = 0
    for (x, y) in loader
        x_dev = x |> device
        y_dev = y |> device
        x_enc = encode_fn(x_dev)
        sims, _ = model(x_enc, ps, st)
        preds = argmax(sims; dims=1)
        truth = argmax(y_dev; dims=1)
        correct += sum(preds .== truth)
        total += size(y, 2)
    end
    return correct / total
end

# ================================================================
# 7. Experiment Runner
# ================================================================

_count(nt::NamedTuple) = isempty(nt) ? 0 : sum(_count(v) for v in values(nt))
_count(x::AbstractArray) = length(x)
_count(_) = 0

"""
    run_conditions(name, train_loader, test_loader, loss_fn, encode_fn; kwargs...)

Run all three initialization conditions (hippo, uniform, short-memory) for a
single benchmark configuration.  Returns a Dict mapping condition name to
(losses, accs) named tuples.

The `seq_len` parameter controls eigenvalue scaling — decay rates are set so
the SSM's memory window is appropriate for the target sequence length.
"""
function run_conditions(name::String, train_loader, test_loader,
                        loss_fn::Function, encode_fn::Function;
                        C_in::Int, D_hidden::Int, n_classes::Int,
                        n_epochs::Int, batchsize::Int, lr::Float64,
                        seed::Int, use_cuda::Bool, readout_frac::Float32,
                        seq_len::Int, lr_ssm::Float64=0.0,
                        weight_decay::Float64=0.0, cosine_schedule::Bool=false,
                        gc_interval::Int=50)

    device = use_cuda ? gpu_device() : cpu_device()

    conditions = [
        ("hippo",        :hippo,   false),
        ("uniform",      :uniform, false),
        ("short-memory", :uniform, true),
    ]

    results = Dict{String, NamedTuple{(:losses, :accs), Tuple{Vector{Float64}, Vector{Float64}}}}()

    for (cond_name, init_mode, force_short) in conditions
        println("\n--- [$name] condition: $cond_name ---")
        rng = Xoshiro(seed)
        model = create_model(; C_in, D_hidden, n_classes, init_mode, readout_frac)
        ps, st = Lux.setup(rng, model)

        if force_short
            ps, st = force_short_memory(ps, st, seq_len)
        else
            ps, st = scale_dynamics_for_length(ps, st, seq_len, D_hidden, init_mode)
        end

        ps = ps |> device
        st = st |> device

        n_params = _count(ps)
        λ_1 = -exp.(Array(ps.layer_1.log_neg_lambda))
        @printf("  Parameters: %d  |  λ range: [%.4f, %.4f]\n",
                n_params, minimum(λ_1), maximum(λ_1))

        args = Args(epochs=1, batchsize=batchsize, lr=lr, use_cuda=use_cuda,
                    rng=Xoshiro(seed), lr_ssm=lr_ssm, weight_decay=weight_decay,
                    cosine_schedule=cosine_schedule, gc_interval=gc_interval)

        all_losses = Float64[]
        accs = Float64[]

        for epoch in 1:n_epochs
            losses, ps, st = train(model, ps, st, train_loader, loss_fn, args)
            append!(all_losses, Float64.(losses))
            acc = evaluate(model, ps, st, test_loader, device; encode_fn)
            push!(accs, acc)
            @printf("  [%s/%s] Epoch %2d/%d  loss=%.4f  acc=%.4f\n",
                    name, cond_name, epoch, n_epochs, mean(losses), acc)
        end

        results[cond_name] = (losses=all_losses, accs=accs)
    end

    return results
end

# ================================================================
# 8. Results Display
# ================================================================

function print_copying_table(results_by_length::Dict, lengths::Vector{Int})
    println("\n" * "="^64)
    println("  SELECTIVE COPYING RESULTS")
    println("="^64)
    @printf("  %-12s | %-12s | %-12s | %-12s\n",
            "Seq Length", "hippo", "uniform", "short-memory")
    println("  " * "-"^12 * "-|-" * "-"^12 * "-|-" * "-"^12 * "-|-" * "-"^12)
    for L in sort(lengths)
        if !haskey(results_by_length, L)
            continue
        end
        res = results_by_length[L]
        @printf("  %-12d | %10.2f%%  | %10.2f%%  | %10.2f%%\n",
                L,
                get(res, "hippo", (accs=[0.0],)).accs[end] * 100,
                get(res, "uniform", (accs=[0.0],)).accs[end] * 100,
                get(res, "short-memory", (accs=[0.0],)).accs[end] * 100)
    end
    println("="^64)
end

function print_fmnist_table(results::Dict)
    println("\n" * "="^48)
    println("  SEQUENTIAL FASHIONMNIST RESULTS")
    println("="^48)
    @printf("  %-14s | %-14s\n", "Condition", "Final Accuracy")
    println("  " * "-"^14 * "-|-" * "-"^14)
    for cond in ["hippo", "uniform", "short-memory"]
        if haskey(results, cond)
            @printf("  %-14s | %12.2f%%\n", cond, results[cond].accs[end] * 100)
        end
    end
    println("="^48)
end

function print_ablation_table(results::Dict)
    println("\n" * "="^56)
    println("  ABLATION RESULTS")
    println("="^56)
    @printf("  %-30s | %-14s\n", "Configuration", "Final Accuracy")
    println("  " * "-"^30 * "-|-" * "-"^14)
    for (name, res) in sort(collect(results); by=first)
        @printf("  %-30s | %12.2f%%\n", name, res.accs[end] * 100)
    end
    println("="^56)
end

# ================================================================
# 9. Ablation Runner
# ================================================================

"""
    run_ablation(name, train_loader, test_loader, loss_fn, encode_fn; kwargs...)

Run a single ablation configuration: build a model with the specified
use_stft and use_attention flags, train it, and return results.
"""
function run_ablation(name::String, train_loader, test_loader,
                      loss_fn::Function, encode_fn::Function;
                      C_in::Int, D_hidden::Int, n_classes::Int,
                      n_epochs::Int, batchsize::Int, lr::Float64,
                      seed::Int, use_cuda::Bool, readout_frac::Float32,
                      use_stft::Bool, use_attention::Bool,
                      lr_ssm::Float64=0.0, weight_decay::Float64=0.0,
                      cosine_schedule::Bool=false, gc_interval::Int=50)

    device = use_cuda ? gpu_device() : cpu_device()
    rng = Xoshiro(seed)

    model = create_ablation_model(; C_in, D_hidden, n_classes,
                                    use_stft, use_attention, readout_frac)
    ps, st = Lux.setup(rng, model)
    ps = ps |> device
    st = st |> device

    n_params = _count(ps)
    println("  [$name] Parameters: $n_params, STFT=$use_stft, Attention=$use_attention")

    args = Args(epochs=1, batchsize=batchsize, lr=lr, use_cuda=use_cuda,
                rng=Xoshiro(seed), lr_ssm=lr_ssm, weight_decay=weight_decay,
                cosine_schedule=cosine_schedule, gc_interval=gc_interval)

    all_losses = Float64[]
    accs = Float64[]

    for epoch in 1:n_epochs
        losses, ps, st = train(model, ps, st, train_loader, loss_fn, args)
        append!(all_losses, Float64.(losses))
        acc = evaluate(model, ps, st, test_loader, device; encode_fn)
        push!(accs, acc)
        @printf("  [%s] Epoch %2d/%d  loss=%.4f  acc=%.4f\n",
                name, epoch, n_epochs, mean(losses), acc)
    end

    return (losses=all_losses, accs=accs)
end

# ================================================================
# 10. Main
# ================================================================

function main()
    parsed = parse_args()
    n_epochs   = parsed["epochs"]
    batchsize  = parsed["batchsize"]
    lr         = Float64(parsed["lr"])
    D_hidden   = parsed["hidden"]
    seed       = parsed["seed"]
    no_cuda    = parsed["no-cuda"]
    benchmark  = parsed["benchmark"]
    D_token    = parsed["d-token"]
    n_train    = parsed["n-train"]
    n_test     = parsed["n-test"]
    copy_lens  = parse.(Int, split(parsed["copy-lengths"], ","))

    pps        = parsed["pixels-per-step"]
    lr_ssm     = Float64(parsed["lr-ssm"])
    wd         = Float64(parsed["weight-decay"])
    cosine     = parsed["cosine-schedule"]
    gc_int     = parsed["gc-interval"]

    use_cuda = !no_cuda && CUDA.functional()
    println("Device: $(use_cuda ? "CUDA GPU" : "CPU")")
    println("Hidden dim: $D_hidden, LR: $lr, Batch: $batchsize, Epochs: $n_epochs")
    lr_ssm > 0 && println("SSM LR: $lr_ssm")
    wd > 0 && println("Weight decay: $wd")
    cosine && println("Cosine LR schedule enabled")
    gc_int > 0 && println("GC interval: $gc_int batches")

    n_classes = 10

    # ---- Selective Copying ----
    if benchmark in ("copying", "both")
        println("\n" * "#"^64)
        println("  BENCHMARK: SELECTIVE COPYING")
        println("#"^64)
        println("  D_token=$D_token, n_classes=$n_classes, n_train=$n_train")

        codebook = generate_copying_codebook(Xoshiro(seed), n_classes, D_token)
        copying_results = Dict{Int, Dict}()

        for L in copy_lens
            println("\n>>> Sequence length L=$L <<<")

            x_train, y_train = generate_copying_data(Xoshiro(seed + 1000), codebook, L, n_train)
            x_test, y_test   = generate_copying_data(Xoshiro(seed + 2000), codebook, L, n_test)

            train_loader = DataLoader((x_train, y_train); batchsize, shuffle=true)
            test_loader  = DataLoader((x_test,  y_test);  batchsize)

            results = run_conditions("copy-L=$L", train_loader, test_loader,
                                     copying_loss, identity;
                                     C_in=D_token, D_hidden, n_classes,
                                     n_epochs, batchsize, lr, seed, use_cuda,
                                     readout_frac=0.1f0, seq_len=L,
                                     lr_ssm, weight_decay=wd,
                                     cosine_schedule=cosine, gc_interval=gc_int)
            copying_results[L] = results
        end

        print_copying_table(copying_results, copy_lens)
    end

    # ---- Sequential FashionMNIST ----
    if benchmark in ("fmnist", "both")
        println("\n" * "#"^64)
        println("  BENCHMARK: SEQUENTIAL FASHIONMNIST")
        println("#"^64)

        train_loader, test_loader, fmnist_L, fmnist_C = load_sequential_fmnist(; batchsize, pixels_per_step=pps)

        fmnist_results = run_conditions("sFashionMNIST-L=$fmnist_L", train_loader, test_loader,
                                        mnist_loss, intensity_to_phase;
                                        C_in=fmnist_C, D_hidden, n_classes=10,
                                        n_epochs, batchsize, lr, seed, use_cuda,
                                        readout_frac=0.25f0, seq_len=fmnist_L,
                                        lr_ssm, weight_decay=wd,
                                        cosine_schedule=cosine, gc_interval=gc_int)

        print_fmnist_table(fmnist_results)
    end

    # ---- Ablation ----
    if benchmark == "ablation"
        ablation_results = Dict{String, NamedTuple}()
        common_kwargs = (; D_hidden, n_classes=10, n_epochs, batchsize, lr, seed,
                           use_cuda, lr_ssm, weight_decay=wd,
                           cosine_schedule=cosine, gc_interval=gc_int)

        # --- FashionMNIST ablation: STFT vs no STFT ---
        println("\n" * "#"^64)
        println("  ABLATION: ResonantSTFT on FashionMNIST")
        println("#"^64)

        train_loader, test_loader, fmnist_L, fmnist_C = load_sequential_fmnist(; batchsize, pixels_per_step=pps)

        for (name, stft) in [("fmnist+STFT", true), ("fmnist-no-STFT", false)]
            ablation_results[name] = run_ablation(name, train_loader, test_loader,
                                                   mnist_loss, intensity_to_phase;
                                                   C_in=fmnist_C,
                                                   use_stft=stft, use_attention=false,
                                                   readout_frac=0.25f0,
                                                   common_kwargs...)
        end

        # --- Copying ablation: attention vs no attention ---
        println("\n" * "#"^64)
        println("  ABLATION: SSMSelfAttention on Selective Copying")
        println("#"^64)

        L_copy = copy_lens[end]  # use longest copying length
        codebook = generate_copying_codebook(Xoshiro(seed), n_classes, D_token)
        x_train, y_train = generate_copying_data(Xoshiro(seed + 1000), codebook, L_copy, n_train)
        x_test, y_test   = generate_copying_data(Xoshiro(seed + 2000), codebook, L_copy, n_test)
        train_loader = DataLoader((x_train, y_train); batchsize, shuffle=true)
        test_loader  = DataLoader((x_test,  y_test);  batchsize)

        for (name, attn) in [("copy+attention", true), ("copy-no-attention", false)]
            ablation_results[name] = run_ablation(name, train_loader, test_loader,
                                                   copying_loss, identity;
                                                   C_in=D_token,
                                                   use_stft=false, use_attention=attn,
                                                   readout_frac=0.1f0,
                                                   common_kwargs...)
        end

        print_ablation_table(ablation_results)
    end

    println("\nDone.")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
