#=
local_attention_compare.jl — §5.3 of docs/local_attention_companion.md
=======================================================================

Comparative training run on column-wise sequential FashionMNIST that
swaps the attention slot of a small Phasor SSM chain between four
variants and reports, per variant:

  - n_params
  - test accuracy (discrete dispatch — Phase 3D Dirac path)
  - test accuracy (spiking dispatch — input PhasorDense via the ODE
    pathway, downstream layers in discrete mode; see notes below)
  - the discrete-vs-spiking accuracy gap
  - per-batch training wall time (mean across epochs)
  - peak GPU memory used (delta CUDA.used_memory before/after train,
    `n/a` on CPU)

Chain (one base, with a single swappable slot):

    PhasorDense(C_in => D)             # input encoder
    [ ATTENTION_SLOT ]                  # variant
    PhasorDense(D => D)                 # body
    SSMReadout(D => n_classes)          # readout

Attention variants:

  - none         (skip the slot — baseline)
  - ssm_self     (SSMSelfAttention; existing across-time attention)
  - local_self   (PhasorLSA; new across-head attention)
  - local_cross  (PhasorLCA; new Hopfield retrieval + V binding)

SSMCrossAttention is intentionally *not* compared — it pools over time
(output shape (D, n_keys, B)) and would require a different downstream
chain. PhasorLCA stands in for the LCA-shaped slot since it preserves
the time axis.

Spiking eval — design note:
  The chain's first layer is PhasorDense. Its SpikingCall dispatch
  routes through CurrentCall, whose default `return_type=:phase`
  returns a Vector-of-Phase-arrays (one per ODE save point), not a
  3D Phase tensor. To produce a (D, L, B) Phase array that the rest
  of the chain can consume, we build a *parallel* input PhasorDense
  with `return_type=:potential` (sharing parameters with the trained
  layer), run it on the SpikingCall, then call `sample_phases_at_periods`
  to sample at period boundaries. The downstream layers (PhasorLSA /
  PhasorLCA / SSMSelfAttention / body PhasorDense / SSMReadout) then
  run in their discrete Phase 3D dispatch.

  This matches `test/test_ssm.jl` ssm_spiking_correlation_tests — the
  established convention for spiking↔discrete parity checking in the
  test suite. A "fully spiking" variant where every layer's output is
  itself a SpikingCall is an open research item, not §5.3 scope.

  Both eval branches stay on the training device. `ssm_phases_to_train`
  has a CuArray dispatch in `src/gpu.jl` that mirrors the CPU body
  via fused broadcast + permute, so spiking eval runs on GPU when
  training was on GPU.

Usage:
  julia --project=. scripts/local_attention_compare.jl
  julia --project=. scripts/local_attention_compare.jl --epochs 5
  julia --project=. scripts/local_attention_compare.jl \
      --variants none,local_self --epochs 1 --hidden 16 --no-cuda
=#

using PhasorNetworks
using Lux, Random, Optimisers, Zygote, Statistics
using MLUtils, OneHotArrays
using CUDA, LuxCUDA
using ArgParse
using Printf
using Base: @kwdef

# ================================================================
# 1. ArgParse
# ================================================================

function parse_args()
    s = ArgParseSettings(description="Attention-variant comparison for PhasorLSA/PhasorLCA")
    @add_arg_table! s begin
        "--epochs"
            help = "number of training epochs"
            arg_type = Int
            default = 5
        "--batchsize"
            help = "training batch size"
            arg_type = Int
            default = 64
        "--lr"
            help = "learning rate"
            arg_type = Float64
            default = 3e-4
        "--hidden"
            help = "hidden dimension D"
            arg_type = Int
            default = 64
        "--n-heads"
            help = "number of heads (PhasorLSA / PhasorLCA only)"
            arg_type = Int
            default = 4
        "--n-anchors"
            help = "anchor count (PhasorLCA only)"
            arg_type = Int
            default = 32
        "--pixels-per-step"
            help = "pixels per timestep; 28=column-wise (L=28), 1=pixel-wise (L=784)"
            arg_type = Int
            default = 28
        "--seed"
            help = "random seed"
            arg_type = Int
            default = 42
        "--no-cuda"
            help = "disable CUDA even if available"
            action = :store_true
        "--variants"
            help = "comma-separated variants to run: none, ssm_self, local_self, local_cross"
            arg_type = String
            default = "none,ssm_self,local_self,local_cross"
    end
    return ArgParse.parse_args(s)
end

# ================================================================
# 2. Args (mirrors test/runtests.jl)
# ================================================================

@kwdef mutable struct Args
    lr::Float64 = 3e-4
    lr_ssm::Float64 = 0.0
    weight_decay::Float64 = 0.0
    cosine_schedule::Bool = false
    lr_min::Float64 = 1e-6
    gc_interval::Int = 50
    batchsize::Int = 64
    epochs::Int = 1
    use_cuda::Bool = false
    rng::Xoshiro = Xoshiro(42)
end

# ================================================================
# 3. Data loading (copied from demos/long_range_demo.jl:198-219)
# ================================================================

function load_sequential_fmnist(; batchsize::Int=64, pixels_per_step::Int=28)
    println("Loading FashionMNIST...")
    train_data = fashion_mnist_data(:train)
    test_data  = fashion_mnist_data(:test)

    n_pixels = 784
    @assert n_pixels % pixels_per_step == 0 "784 must be divisible by pixels_per_step"
    L = n_pixels ÷ pixels_per_step
    C_in = pixels_per_step

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
# 4. Input encoding + loss (copied from demos/long_range_demo.jl)
# ================================================================

function intensity_to_phase(images::AbstractArray{<:Real, 3})
    H, W, B = size(images)
    phases = Float32.(images) .* 0.5f0
    phases_ct = permutedims(phases, (2, 1, 3))   # channels × time × batch
    return Phase.(phases_ct)
end

function _softmax_ce(sims, y)
    logits = 5f0 .* sims
    logits_max = maximum(logits; dims=1)
    shifted = logits .- logits_max
    log_sum_exp = log.(sum(exp.(shifted); dims=1))
    log_probs = shifted .- log_sum_exp
    return -mean(sum(y .* log_probs; dims=1))
end

function mnist_loss(x, y, model, ps, st)
    x_enc = intensity_to_phase(x)
    sims, _ = model(x_enc, ps, st)
    return _softmax_ce(sims, y)
end

# ================================================================
# 5. SSM dynamics scaling (copied from demos/long_range_demo.jl)
#
#    Note: the helper only descends one layer deep into the chain's
#    parameter tree, so the per-channel λ inside the Q/K/V projections
#    of PhasorLSA/PhasorLCA is NOT rescaled — those keep their default
#    :hippo init. For L = 28 (the default) this is fine; for very long
#    sequences a recursive descent would be needed.
# ================================================================

function _map_ssm_layers(fn_ps, fn_st, ps::NamedTuple, st::NamedTuple)
    ps_keys = keys(ps)
    st_keys = keys(st)
    new_ps_vals = Tuple(
        let layer_ps = ps[k]
            (layer_ps isa NamedTuple && haskey(layer_ps, :log_neg_lambda)) ?
                fn_ps(layer_ps) : layer_ps
        end for k in ps_keys)
    new_st_vals = Tuple(
        let layer_ps = get(ps, k, NamedTuple()), layer_st = st[k]
            (layer_ps isa NamedTuple && haskey(layer_ps, :log_neg_lambda) &&
             layer_st isa NamedTuple) ?
                fn_st(layer_st) : layer_st
        end for k in st_keys)
    return NamedTuple{ps_keys}(new_ps_vals), NamedTuple{st_keys}(new_st_vals)
end

function scale_dynamics_for_length(ps::NamedTuple, st::NamedTuple,
                                   seq_len::Int, D_hidden::Int)
    # :hippo path only (we use :hippo init across all variants).
    λ_base, ω_base = hippo_legs_diagonal(D_hidden)
    scale = Float32(seq_len)
    λ_scaled = λ_base ./ scale
    ω_scaled = ω_base ./ scale
    log_neg_lambda = Float32.(log.(-λ_scaled))
    omega = ω_scaled

    return _map_ssm_layers(
        lp -> merge(lp, (log_neg_lambda = log_neg_lambda,)),
        ls -> haskey(ls, :omega) ? merge(ls, (omega = omega,)) : ls,
        ps, st)
end

# Parameter counter (copied from demos/long_range_demo.jl).
_count(nt::NamedTuple) = isempty(nt) ? 0 : sum(_count(v) for v in values(nt))
_count(x::AbstractArray) = length(x)
_count(_) = 0

# ================================================================
# 6. Chain factory — one chain, four attention-slot variants
# ================================================================

function build_chain(variant::Symbol;
                     C_in::Int, D_hidden::Int, n_classes::Int,
                     n_heads::Int=4, n_anchors::Int=32,
                     init_mode::Symbol=:hippo,
                     spk_args::SpikingArgs=SpikingArgs())
    input_layer = PhasorDense(C_in => D_hidden, normalize_to_unit_circle;
                              init_mode=init_mode, use_bias=false, spk_args=spk_args)
    body_layer  = PhasorDense(D_hidden => D_hidden, identity;
                              init_mode=init_mode, use_bias=false, spk_args=spk_args)
    readout     = SSMReadout(D_hidden => n_classes; readout_frac=0.25f0)

    attn = if variant === :none
        nothing
    elseif variant === :ssm_self
        SSMSelfAttention(D_hidden => D_hidden, normalize_to_unit_circle)
    elseif variant === :local_self
        PhasorLSA(D_hidden => D_hidden, n_heads;
                  init_mode=init_mode, spk_args=spk_args)
    elseif variant === :local_cross
        PhasorLCA(D_hidden => D_hidden, n_heads, n_anchors;
                  init_mode=init_mode, spk_args=spk_args)
    else
        error("Unknown variant: $variant")
    end

    layers = attn === nothing ?
        (input_layer, body_layer, readout) :
        (input_layer, attn, body_layer, readout)
    return Chain(layers...)
end

# ================================================================
# 7. Discrete evaluation (copied from demos/long_range_demo.jl:410-424)
# ================================================================

function evaluate_discrete(model, ps, st, loader, device)
    correct = 0
    total = 0
    for (x, y) in loader
        x_dev = x |> device
        y_dev = y |> device
        x_enc = intensity_to_phase(x_dev)
        sims, _ = model(x_enc, ps, st)
        preds = argmax(sims; dims=1)
        truth = argmax(y_dev; dims=1)
        correct += sum(preds .== truth)
        total += size(y, 2)
    end
    return correct / total
end

# ================================================================
# 8. Spiking evaluation — first layer ODE + tail in discrete dispatch
#
# Architecture choice: see the design note at the top of this file.
# ================================================================

function evaluate_spiking(model::Chain, ps::NamedTuple, st::NamedTuple,
                          loader, spk_args::SpikingArgs, device,
                          C_in::Int, D_hidden::Int, init_mode::Symbol)
    # Build a return_type=:potential variant of the trained input
    # PhasorDense. Its parameter tree matches the trained `ps.layer_1`
    # exactly (same in_dims, out_dims, use_bias, init_mode).
    spiking_input = PhasorDense(C_in => D_hidden, normalize_to_unit_circle;
                                init_mode = init_mode,
                                use_bias = false,
                                spk_args = spk_args,
                                return_type = SolutionType(:potential))

    ps_input = ps.layer_1
    st_input = st.layer_1

    # Tail = layers 2..end. Iterate manually so we don't have to repack
    # the parameter NamedTuple.
    tail_keys = collect(keys(ps))[2:end]   # :layer_2, :layer_3, ...

    correct = 0
    total = 0
    for (x, y) in loader
        x_dev = x |> device
        y_dev = y |> device

        x_phase = intensity_to_phase(x_dev)
        L = size(x_phase, 2)
        train_spk = ssm_phases_to_train(x_phase; spk_args=spk_args)
        tspan = (0.0f0, Float32(L) * spk_args.t_period)
        sc = SpikingCall(train_spk, spk_args, tspan)

        # First layer (input PhasorDense) via the ODE pathway.
        sol, _ = spiking_input(sc, ps_input, st_input)

        # Sample at period boundaries to recover a (D_hidden, L, B) Phase
        # tensor. `unrotate=true` derotates the carrier so the result is
        # in the static phase frame, matching what the discrete Dirac
        # path produces.
        z = sample_phases_at_periods(sol, L, spk_args;
                                     activation = normalize_to_unit_circle,
                                     unrotate = true)

        # Tail layers in discrete Phase 3D dispatch.
        intermediate = z
        for k in tail_keys
            layer = model.layers[k]
            intermediate, _ = layer(intermediate, ps[k], st[k])
        end
        sims = intermediate

        preds = argmax(sims; dims=1)
        truth = argmax(y_dev; dims=1)
        correct += sum(preds .== truth)
        total += size(y, 2)
    end
    return correct / total
end

# ================================================================
# 9. Wall-time + GPU memory helpers
# ================================================================

# Returns CUDA-resident bytes converted to MiB, or `missing` on CPU.
function snapshot_gpu_mem_mib(use_cuda::Bool)
    use_cuda || return missing
    # CUDA.used_memory() returns the total reserved bytes (process-wide).
    return round(CUDA.used_memory() / 1024^2; digits=1)
end

# ================================================================
# 10. Variant runner
# ================================================================

function run_attention_variant(variant::Symbol, train_loader, test_loader;
                               C_in::Int, D_hidden::Int, n_classes::Int,
                               n_heads::Int, n_anchors::Int,
                               n_epochs::Int, batchsize::Int, lr::Float64,
                               seed::Int, use_cuda::Bool, seq_len::Int,
                               init_mode::Symbol=:hippo,
                               spk_args::SpikingArgs=SpikingArgs())
    device = use_cuda ? gpu_device() : cpu_device()
    rng = Xoshiro(seed)

    println("\n--- variant: $variant ---"); flush(stdout)
    model = build_chain(variant;
                        C_in, D_hidden, n_classes,
                        n_heads, n_anchors, init_mode, spk_args)
    ps, st = Lux.setup(rng, model)
    ps, st = scale_dynamics_for_length(ps, st, seq_len, D_hidden)
    ps = ps |> device
    st = st |> device

    n_params = _count(ps)
    @printf("  parameters: %d\n", n_params); flush(stdout)

    args = Args(epochs=1, batchsize=batchsize, lr=lr,
                use_cuda=use_cuda, rng=Xoshiro(seed))

    # Baseline GPU mem snapshot before training. CUDA.used_memory()
    # reports the steady-state CUDA reservation; we report the delta
    # after the first epoch is finished so the one-shot allocations
    # (cuFFT plans, ODE workspaces) are amortized into the "peak"
    # number rather than the baseline.
    use_cuda && CUDA.reclaim()
    mem_before = snapshot_gpu_mem_mib(use_cuda)

    all_losses = Float64[]
    wall_times = Float64[]
    for epoch in 1:n_epochs
        t0 = time()
        losses, ps, st = train(model, ps, st, train_loader, mnist_loss, args)
        elapsed = time() - t0
        push!(wall_times, elapsed)
        append!(all_losses, Float64.(losses))
        @printf("  epoch %2d/%d  loss=%.4f  wall=%.2fs\n",
                epoch, n_epochs, mean(losses), elapsed)
        flush(stdout)
    end
    mem_after = snapshot_gpu_mem_mib(use_cuda)
    mem_delta = (use_cuda && !ismissing(mem_after) && !ismissing(mem_before)) ?
                (mem_after - mem_before) : missing

    n_batches = length(train_loader)
    wall_per_batch_ms = (mean(wall_times) / n_batches) * 1000

    @printf("  evaluating discrete...\n"); flush(stdout)
    disc_acc = evaluate_discrete(model, ps, st, test_loader, device)

    @printf("  evaluating spiking (ODE first layer + discrete tail)...\n"); flush(stdout)
    spk_acc = evaluate_spiking(model, ps, st, test_loader, spk_args, device,
                               C_in, D_hidden, init_mode)

    @printf("  -> disc=%.2f%%  spk=%.2f%%  gap=%.2f%%  wall/batch=%.2fms  mem_delta=%s\n",
            disc_acc*100, spk_acc*100, (disc_acc - spk_acc)*100,
            wall_per_batch_ms,
            ismissing(mem_delta) ? "n/a" : @sprintf("%.1fMiB", mem_delta))
    flush(stdout)

    return (variant=variant, n_params=n_params,
            losses=all_losses,
            wall_per_batch_ms=wall_per_batch_ms,
            mem_mib=mem_delta,
            disc_acc=disc_acc,
            spk_acc=spk_acc,
            gap=disc_acc - spk_acc)
end

# ================================================================
# 11. Table printer
# ================================================================

function print_compare_table(results::Vector)
    println("\n" * "="^96)
    println("  ATTENTION VARIANT COMPARISON — Sequential FashionMNIST")
    println("="^96)
    @printf("  %-12s | %8s | %9s | %9s | %8s | %12s | %12s\n",
            "Variant", "params", "disc_acc", "spk_acc", "gap", "wall/batch", "peak_mem")
    println("  " * "-"^94)
    for r in results
        mem_str = ismissing(r.mem_mib) ? "    n/a    " :
                  @sprintf("%7.1f MiB ", r.mem_mib)
        @printf("  %-12s | %8d | %8.2f%% | %8.2f%% | %7.2f%% | %9.2f ms | %s\n",
                String(r.variant), r.n_params,
                r.disc_acc*100, r.spk_acc*100, r.gap*100,
                r.wall_per_batch_ms, mem_str)
    end
    println("="^96)
end

# ================================================================
# 12. Main
# ================================================================

function main()
    parsed = parse_args()
    n_epochs   = parsed["epochs"]
    batchsize  = parsed["batchsize"]
    lr         = Float64(parsed["lr"])
    D_hidden   = parsed["hidden"]
    n_heads    = parsed["n-heads"]
    n_anchors  = parsed["n-anchors"]
    pps        = parsed["pixels-per-step"]
    seed       = parsed["seed"]
    no_cuda    = parsed["no-cuda"]

    variant_strs = strip.(split(parsed["variants"], ","))
    variants = [Symbol(s) for s in variant_strs]
    for v in variants
        v in (:none, :ssm_self, :local_self, :local_cross) ||
            error("Unknown variant: $v (use none, ssm_self, local_self, or local_cross)")
    end

    use_cuda = !no_cuda && CUDA.functional()
    println("Device: $(use_cuda ? "CUDA GPU" : "CPU")")
    println("Config: D=$D_hidden, epochs=$n_epochs, batchsize=$batchsize, lr=$lr")
    println("        n_heads=$n_heads, n_anchors=$n_anchors, pixels_per_step=$pps")
    println("        variants=$(variant_strs)")

    n_classes = 10
    spk_args  = SpikingArgs()

    # Auto-scale batchsize when L gets large (mirrors long_range_demo.jl).
    L_pre = 784 ÷ pps
    fmnist_bs = min(batchsize, max(16, 4096 ÷ L_pre))
    if fmnist_bs < batchsize
        @printf("  Auto-scaling batchsize from %d to %d for L=%d\n",
                batchsize, fmnist_bs, L_pre)
    end

    train_loader, test_loader, L, C_in =
        load_sequential_fmnist(; batchsize=fmnist_bs, pixels_per_step=pps)

    results = []
    for v in variants
        push!(results,
              run_attention_variant(v, train_loader, test_loader;
                                    C_in, D_hidden, n_classes,
                                    n_heads, n_anchors,
                                    n_epochs, batchsize=fmnist_bs, lr,
                                    seed, use_cuda, seq_len=L,
                                    init_mode=:hippo, spk_args=spk_args))
    end

    print_compare_table(results)
    println("\nDone.")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
