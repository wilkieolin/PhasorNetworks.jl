#=
Long-Range Dependency Benchmarks for Phasor SSMs
=================================================

Validates that HiPPO-initialized Phasor SSMs capture long-range dependencies
by running two benchmarks with three initialization conditions:

  1. Selective Copying (synthetic) — recall a phase pattern placed at position 0
     after a long blank gap. Tested at multiple sequence lengths.
  2. Sequential MNIST (pixel-by-pixel) — classify MNIST digits processed as
     784-step sequences with 1 channel per timestep.

Three conditions:
  - hippo:        HiPPO-LegS multi-timescale initialization, scaled for sequence length
  - uniform:      uniformly spread frequencies, moderate decay scaled for sequence length
  - short-memory: deliberately high leakage relative to sequence, cannot retain info

Key insight: PhasorDense uses Δt=1 in its convolutional mode, so the kernel
magnitude at lag n is |K[n]| = exp(λ·n).  For a signal at position 0 to survive
to position L, we need |λ| ~ O(1/L).  The HiPPO eigenvalues define the *shape*
of the multi-timescale memory (log-spaced decay rates); this script scales them
so the memory window matches the target sequence length.

Usage:
  julia --project=. demos/long_range_demo.jl --benchmark both --epochs 10
  julia --project=. demos/long_range_demo.jl --benchmark copying --copy-lengths 128,256,512
  julia --project=. demos/long_range_demo.jl --benchmark mnist --epochs 15
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
            help = "which benchmark to run: copying, mnist, or both"
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
            help = "pixels grouped per timestep for sequential MNIST (1=full 784, 4=196 steps)"
            arg_type = Int
            default = 4
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

    labels = rand(rng, 0:n_classes-1, n_samples)
    x = zeros(ComplexF32, D_token, seq_len, n_samples)
    for i in 1:n_samples
        x[:, 1, i] .= codebook[:, labels[i] + 1]
    end

    # Add random complex noise at positions 2:end to create interference
    if noise_scale > 0f0
        noise_phases = Phase.(2f0 .* rand(rng, Float32, D_token, seq_len - 1, n_samples) .- 1f0)
        noise = noise_scale .* angle_to_complex(noise_phases)
        x[:, 2:end, :] .= noise
    end

    y = Float32.(onehotbatch(labels, 0:n_classes-1))
    return x, y
end

# ================================================================
# 3. Data Loading — Sequential MNIST
# ================================================================

"""
    load_sequential_mnist(; batchsize, pixels_per_step)

Load MNIST and reshape for sequential processing.

Groups `pixels_per_step` pixels into each timestep as separate channels.
With pixels_per_step=1: (784, 1, N) → L=784, C=1 (classic sMNIST, very slow)
With pixels_per_step=4: (196, 4, N) → L=196, C=4 (practical default)

The raw data is returned as (L × C × N) Float32 for use with `psk_encode`,
which transposes to (C × L × N) complex.

Returns (train_loader, test_loader, L, C_in).
"""
function load_sequential_mnist(; batchsize::Int=64, pixels_per_step::Int=4)
    println("Loading MNIST...")
    train_data = MNIST(split=:train)
    test_data  = MNIST(split=:test)

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
function scale_dynamics_for_length(ps::NamedTuple, st::NamedTuple,
                                   seq_len::Int, D_hidden::Int, init_mode::Symbol)
    if init_mode == :hippo
        λ_base, ω_base = hippo_legs_diagonal(D_hidden)
        # HiPPO gives λ ∈ [-0.5, -(N-0.5)] with log spacing.
        # Scale so the slowest channel (|λ|=0.5) has memory window = 2*seq_len:
        #   |λ_scaled| = |λ_base| / seq_len  →  slowest: 0.5/L, fastest: (N-0.5)/L
        #   At lag L: exp(-0.5) ≈ 0.61 (slowest), exp(-(N-0.5)) ≈ 0 (fastest)
        scale = Float32(seq_len)
        λ_scaled = λ_base ./ scale
        ω_scaled = ω_base ./ scale
        log_neg_lambda = Float32.(log.(-λ_scaled))
        omega = ω_scaled
    else  # :uniform
        # Moderate decay: exp(-2/L * L) = exp(-2) ≈ 0.13 at full sequence length
        λ_val = 2f0 / Float32(seq_len)
        log_neg_lambda = fill(Float32(log(λ_val)), D_hidden)
        omega = Float32.(collect(range(0.1f0, 1.5f0; length=D_hidden)))
    end

    ps_new = merge(ps, (
        layer_1 = merge(ps.layer_1, (log_neg_lambda = log_neg_lambda,)),
        layer_2 = merge(ps.layer_2, (log_neg_lambda = log_neg_lambda,)),
    ))
    st_new = merge(st, (
        layer_1 = merge(st.layer_1, (omega = omega,)),
        layer_2 = merge(st.layer_2, (omega = omega,)),
    ))

    return ps_new, st_new
end

"""
    force_short_memory(ps, st, seq_len; memory_frac)

Override parameters to create a short-memory baseline relative to sequence length.
Sets |λ| = 20/L so memory decays within ~L/20 steps (5% of sequence).
Must be called BEFORE moving to GPU.
"""
function force_short_memory(ps::NamedTuple, st::NamedTuple, seq_len::Int;
                            memory_frac::Float32=0.05f0)
    # Memory decays to exp(-1) within memory_frac * seq_len steps
    # |λ| = 1 / (memory_frac * L)
    decay_val = 1f0 / (memory_frac * Float32(seq_len))
    log_val = Float32(log(decay_val))

    ps_new = merge(ps, (
        layer_1 = merge(ps.layer_1, (
            log_neg_lambda = fill(log_val, size(ps.layer_1.log_neg_lambda)),
        )),
        layer_2 = merge(ps.layer_2, (
            log_neg_lambda = fill(log_val, size(ps.layer_2.log_neg_lambda)),
        )),
    ))

    omega_val = 1.0f0
    st_new = merge(st, (
        layer_1 = merge(st.layer_1, (
            omega = fill(omega_val, size(st.layer_1.omega)),
        )),
        layer_2 = merge(st.layer_2, (
            omega = fill(omega_val, size(st.layer_2.omega)),
        )),
    ))

    return ps_new, st_new
end

# ================================================================
# 5. Loss Functions
# ================================================================

function copying_loss(x, y, model, ps, st)
    sims, _ = model(x, ps, st)
    return mean(similarity_loss(sims, y))
end

function mnist_loss(x, y, model, ps, st)
    x_enc = psk_encode(x)
    sims, _ = model(x_enc, ps, st)
    return mean(similarity_loss(sims, y))
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
                        seq_len::Int)

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
                    rng=Xoshiro(seed))

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

function print_mnist_table(results::Dict)
    println("\n" * "="^48)
    println("  SEQUENTIAL MNIST RESULTS")
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

# ================================================================
# 9. Main
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

    use_cuda = !no_cuda && CUDA.functional()
    println("Device: $(use_cuda ? "CUDA GPU" : "CPU")")
    println("Hidden dim: $D_hidden, LR: $lr, Batch: $batchsize, Epochs: $n_epochs")

    n_classes = 10

    # ---- Selective Copying ----
    if benchmark in ("copying", "both")
        println("\n" * "#"^64)
        println("  BENCHMARK: SELECTIVE COPYING")
        println("#"^64)
        println("  D_token=$D_token, n_classes=$n_classes, n_train=$n_train")

        # Generate codebook once — shared across all sequence lengths, train/test splits
        codebook = generate_copying_codebook(Xoshiro(seed), n_classes, D_token)
        copying_results = Dict{Int, Dict}()

        for L in copy_lens
            println("\n>>> Sequence length L=$L <<<")

            # Generate train/test data from the shared codebook
            x_train, y_train = generate_copying_data(Xoshiro(seed + 1000), codebook, L, n_train)
            x_test, y_test   = generate_copying_data(Xoshiro(seed + 2000), codebook, L, n_test)

            train_loader = DataLoader((x_train, y_train); batchsize, shuffle=true)
            test_loader  = DataLoader((x_test,  y_test);  batchsize)

            results = run_conditions("copy-L=$L", train_loader, test_loader,
                                     copying_loss, identity;
                                     C_in=D_token, D_hidden, n_classes,
                                     n_epochs, batchsize, lr, seed, use_cuda,
                                     readout_frac=0.1f0, seq_len=L)
            copying_results[L] = results
        end

        print_copying_table(copying_results, copy_lens)
    end

    # ---- Sequential MNIST ----
    if benchmark in ("mnist", "both")
        println("\n" * "#"^64)
        println("  BENCHMARK: SEQUENTIAL MNIST")
        println("#"^64)

        train_loader, test_loader, mnist_L, mnist_C = load_sequential_mnist(; batchsize, pixels_per_step=pps)

        mnist_results = run_conditions("sMNIST-L=$mnist_L", train_loader, test_loader,
                                       mnist_loss, psk_encode;
                                       C_in=mnist_C, D_hidden, n_classes=10,
                                       n_epochs, batchsize, lr, seed, use_cuda,
                                       readout_frac=0.25f0, seq_len=mnist_L)

        print_mnist_table(mnist_results)
    end

    println("\nDone.")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
