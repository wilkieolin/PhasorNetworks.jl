#=
Discrete State Space Model for Phasor Networks
================================================

Instead of solving a continuous ODE (dz/dt = kz + I(t)), we discretize the
Resonate-and-Fire neuron into a linear recurrence:

    z[n+1] = A · z[n] + B · I[n]

where A = exp(k·Δt), B = (A-1)/k, k = λ + iω.

Unrolling gives a causal convolution: z[n] = Σⱼ K[n-j]·I[j], K[n] = Aⁿ·B,
computable in parallel via matrix multiply (small L) or FFT (large L).

This is the same insight behind S4/S4D/Mamba: diagonal complex state spaces
are equivalent to complex convolution kernels.

Demo: PSK-encode FashionMNIST as a complex time series and classify it
using a stack of PhasorSSM layers with a Codebook readout and similarity loss.
=#

using PhasorNetworks
using Lux, LuxCore, Random, Optimisers, Zygote, Statistics
using MLDatasets, MLUtils, OneHotArrays
using NNlib: batched_mul
using CUDA, LuxCUDA
using ArgParse

# ================================================================
# 1. Discrete Phasor Kernel
# ================================================================

"""
    phasor_kernel(λ, ω, Δt, L) -> ComplexF32 matrix (C × L)

Compute the causal impulse-response kernel for C damped oscillators over L
discrete time steps.  This is the discrete equivalent of what `oscillator_bank`
computes continuously: given a single unit impulse at time 0, how does the
oscillator ring down over subsequent steps?

# Arguments
- `λ::AbstractVector{Float32}` — **Decay rates** (length C, one per oscillator).
  Same physical meaning as `spk_args.leakage` in the spiking system: controls
  how fast the membrane potential spirals inward.  Must be negative for
  stability (e.g. -0.1).  More negative = faster decay = shorter memory.
  In the continuous ODE this appears as:  dz/dt = (λ + iω)z + I(t).

- `ω::AbstractVector{Float32}` — **Angular frequencies** (length C, rad/step).
  Same role as `2π / spk_args.t_period` in the spiking system: controls how
  fast the oscillator rotates in the complex plane.  In the continuous ODE
  the neuron constant is k = λ + iω; here ω is angular frequency directly
  rather than derived from a period.

- `Δt::Real` — **Time step** between consecutive input samples.  Analogous to
  the `dt` solver argument in `oscillator_bank`.  We set Δt=1 here because
  our "time" axis is just sample indices (row 1, row 2, ..., row 28 of the
  image).  If your samples were 0.02s apart (like the ODE demo), you'd set
  Δt=0.02.

- `L::Int` — **Sequence length** (number of time steps).  For FashionMNIST
  with rows-as-time, L=28.

# Returns
A `C × L` complex matrix where K[c, n] is the response of oscillator c at
lag n.  Entry K[c, 0] (column 1) is the immediate response to the current
input; K[c, 27] (column 28) is the lingering contribution from 27 steps ago.

# Connection to the continuous ODE
In `oscillator_bank`, the neuron constant is:

    k = neuron_constant(leakage, t_period) = leakage + i·(2π / t_period)

The ODE `dz/dt = k·z + I(t)` has exact solution over one step Δt:

    z[n+1] = exp(k·Δt) · z[n] + B · I[n]

where `A = exp(k·Δt)` and `B = (A - 1) / k` (zero-order-hold discretization).
Unrolling this recurrence gives z[n] = Σⱼ Aⁿ⁻ʲ · B · I[j], which is a
causal convolution with kernel K[n] = Aⁿ · B.

So this function precomputes that entire kernel in one shot, replacing the
sequential ODE solve with a single matrix that can be applied in parallel.
"""
function phasor_kernel(λ::AbstractVector, ω::AbstractVector, Δt::Real, L::Int)
    k = ComplexF32.(λ .+ im .* ω)            # C eigenvalues
    # Build range on same device as k (GPU-safe)
    ns_cpu = Float32.(0:L-1)
    ns = reshape(typeof(real.(k))(ns_cpu), 1, L) # 1 × L, same device as params
    A_powers = exp.(k .* Δt .* ns)            # C × L:  A^n
    B_gain = (exp.(k .* Δt) .- 1f0) ./ k     # C × 1:  input gain
    return A_powers .* B_gain                  # C × L
end

# ================================================================
# 2. Causal Convolution via Lower-Triangular Toeplitz Matrix
# ================================================================

"""
    causal_conv(K, H) -> ComplexF32 array (C × L × B)

Apply the precomputed impulse-response kernel to a batch of input signals.
This replaces the role of `DifferentialEquations.solve()` in the spiking
system — instead of stepping through time sequentially, we compute all
time steps at once via matrix multiplication.

# Arguments
- `K::AbstractMatrix{<:Complex}` — **Kernel matrix** (C × L), as returned by
  `phasor_kernel`.  Each row is one oscillator's impulse response over L lags.
  Think of it as: "if I inject a unit current at time 0, what is the membrane
  potential at times 0, 1, 2, ..., L-1?"

- `H::AbstractArray{<:Complex, 3}` — **Input signal** (C × L × B).
  C = number of output channels (oscillators), L = time steps, B = batch size.
  This is the weighted input current at each time step — the discrete
  equivalent of the continuous `W · x(t)` that drives the ODE.

# Returns
`Z` (C × L × B): the complex membrane potential at each time step, for each
oscillator, for each batch element.  Z[:, end, :] is the "final state" —
the equivalent of reading `sol.u[end]` from the ODE solver.

# How it works
The causal convolution Z[c,t] = Σⱼ₌₀ᵗ K[c, t-j] · H[c, j] is equivalent to
multiplying H by a lower-triangular Toeplitz matrix:

    ┌ K[1]  0     0    0   ┐   ┌ H[1] ┐
    │ K[2]  K[1]  0    0   │   │ H[2] │
    │ K[3]  K[2]  K[1] 0   │ × │ H[3] │
    └ K[4]  K[3]  K[2] K[1]┘   └ H[4] ┘

Row t of the result sums K[t-j+1]·H[j] for j=1..t, which is exactly the
state of the oscillator at time t given all inputs up to time t (causality
means we never look at future inputs).

Uses `NNlib.batched_mul` for GPU-friendly parallel computation.  Cost is
O(C·L²·B); for very long sequences (L > 1000) an FFT-based approach would
be faster at O(C·L·log(L)·B).
"""
function causal_conv(K::AbstractMatrix{<:Complex}, H::AbstractArray{<:Complex, 3})
    C, L, B = size(H)

    # Pad kernel with one zero column so out-of-bounds indices map to 0
    # Use zero(K[:, 1:1]) for Zygote-safe, device-safe zero allocation
    K_pad = cat(K, zero(K[:, 1:1]); dims=2)  # C × (L+1)

    # Lower-triangular index matrix: T[i,j] = K[:, i-j+1] when i≥j, else K[:, L+1]=0
    idx = [i >= j ? i - j + 1 : L + 1 for i in 1:L, j in 1:L]  # L × L (constant, CPU)
    T = K_pad[:, idx]                                      # C × L × L

    # batched_mul wants (M,K,batch) × (K,N,batch) → (M,N,batch), batch = C
    T_perm = permutedims(T, (2, 3, 1))  # L × L × C
    H_perm = permutedims(H, (2, 3, 1))  # L × B × C
    Z_perm = batched_mul(T_perm, H_perm) # L × B × C
    return permutedims(Z_perm, (3, 1, 2)) # C × L × B
end

# ================================================================
# 3. PhasorSSM Layer  (Lux-compatible)
# ================================================================

"""
    PhasorSSM(in_dims => out_dims, activation; init_omega_range=(0.2, 2.5))

Discrete phasor state-space layer — the SSM equivalent of `PhasorDense`.

Where `PhasorDense` feeds weighted input into `oscillator_bank` (a
continuous ODE solver), `PhasorSSM` does the same integration as a
precomputed convolution kernel applied via matrix multiply.  The two
produce equivalent dynamics; the difference is computational:
- ODE: sequential, expensive, but handles continuous/irregular time
- SSM: parallel, fast, but requires uniformly-sampled discrete input

# Constructor arguments
- `in_dims => out_dims` — Channel dimensions (same as `PhasorDense(in => out)`).
  `in_dims` input channels are mixed by a weight matrix into `out_dims`
  oscillators, each with its own decay and frequency.

- `activation` — Applied after temporal integration.  Typically
  `normalize_to_unit_circle` (project to |z|=1) between layers, or
  `identity` for the final layer (preserve magnitude for readout).

- `init_omega_range` — Initial spread of angular frequencies across the
  `out_dims` oscillators.  Default `(0.2, 2.5)` spans roughly 0.03 to
  0.4 Hz (cycles per step), so the oscillators cover a range of
  timescales.  Comparable to assigning different `t_period` values to
  different neurons in the spiking system.

# Trainable parameters (in `ps`)
- `weight` (Float32, out × in) — Channel mixing matrix.  Same role as the
  weight matrix in `PhasorDense`: linearly combines input channels before
  feeding them to the oscillators.

- `log_neg_lambda` (Float32, out) — Log-space decay rates.  The actual
  decay is `λ = -exp(log_neg_lambda)`, which is always negative
  (guaranteeing stability).  This is the same physical quantity as
  `spk_args.leakage` — it controls how quickly each oscillator's state
  decays.  Initialized to `log(0.1)` → λ ≈ -0.1, similar to the default
  `leakage = -0.2` in `SpikingArgs`.  More negative = faster forgetting.

- `omega` (Float32, out) — Angular frequencies in radians per time step.
  Related to `spk_args.t_period` by ω = 2π/T.  Each oscillator gets its
  own frequency, initialized as a linspace across `init_omega_range`.
  In the spiking system all neurons typically share one period; here,
  giving each its own frequency lets the layer capture multiple timescales
  simultaneously (like a filter bank).
"""
struct PhasorSSM <: LuxCore.AbstractLuxLayer
    in_dims::Int
    out_dims::Int
    activation::Function
    omega_lo::Float32
    omega_hi::Float32
    init_mode::Symbol   # :uniform or :hippo
end

function PhasorSSM(dims::Pair{Int,Int}, act=normalize_to_unit_circle;
                   init_omega_range=(0.2f0, 2.5f0), init=:uniform)
    @assert init in (:uniform, :hippo) "init must be :uniform or :hippo"
    return PhasorSSM(dims.first, dims.second, act,
                     Float32(init_omega_range[1]), Float32(init_omega_range[2]),
                     init)
end

"""
    hippo_legs_diagonal(N; clip_decay=nothing) -> (λ, ω)

HiPPO-LegS (Legendre Scaled) diagonal initialization from S4D.

The HiPPO framework defines optimal state matrices for online function
approximation.  The LegS variant projects input history onto a scaled
Legendre polynomial basis, giving a principled multi-timescale memory.

Diagonalizing the N×N HiPPO-LegS matrix yields N complex eigenvalues:

    k_n = -(n + 1/2) + iπ(n + 1/2),   n = 0, 1, ..., N-1

This gives each oscillator:
- **Decay rate** λ_n = -(n + 1/2): low-index channels decay slowly (long
  memory), high-index channels decay fast (local features).  This linear
  spacing is key — uniform decay wastes capacity by making all channels
  redundant at the same timescale.
- **Frequency** ω_n = π(n + 1/2): paired with the decay so that each
  channel completes roughly one full oscillation within its memory window
  (1/|λ_n| steps).  This is the Nyquist-optimal frequency for each
  timescale.

# Keyword arguments
- `clip_decay`: Maximum magnitude of λ.  Raw HiPPO gives λ up to -(N-0.5),
  which for N=128 means channel 127 decays to exp(-127.5) ≈ 0 in one step.
  Clipping to e.g. `clip_decay=5.0` (memory ≈ L/5 steps) keeps all channels
  useful for short sequences.  When `nothing` (default), eigenvalues are
  log-spaced from 0.5 to N-0.5 across N channels, preserving the HiPPO
  multi-timescale structure while keeping all channels in a trainable range.

Returns raw (λ, ω) vectors — caller must map to the log-parameterization.
"""
function hippo_legs_diagonal(N::Int; clip_decay::Union{Nothing, Real}=nothing)
    if clip_decay === nothing
        # Log-spaced variant: span from λ=-0.5 to λ=-N+0.5 on a log scale.
        # Preserves HiPPO's multi-timescale property (slow + fast channels)
        # while keeping high-index channels in a useful range.
        λ_mag = Float32.(exp.(range(log(0.5), log(N - 0.5); length=N)))
    else
        # Linear HiPPO with hard clip
        ns = Float32.(0:N-1)
        λ_mag = min.(ns .+ 0.5f0, Float32(clip_decay))
    end
    λ = -λ_mag
    # Frequency paired to decay: one oscillation per memory window
    ω = Float32(π) .* λ_mag
    return λ, ω
end

function Lux.initialparameters(rng::AbstractRNG, l::PhasorSSM)
    W = Float32.(randn(rng, l.out_dims, l.in_dims)) ./ sqrt(Float32(l.in_dims))

    if l.init_mode == :hippo
        λ_init, ω_init = hippo_legs_diagonal(l.out_dims)
        # Map λ to log-parameterization: λ = -exp(v), so v = log(-λ) = log(n + 0.5)
        log_neg_lambda = log.(-λ_init)
        omega = ω_init
    else  # :uniform
        log_neg_lambda = fill(Float32(log(0.1)), l.out_dims)
        omega = Float32.(collect(range(l.omega_lo, l.omega_hi; length=l.out_dims)))
    end
    return (weight=W, log_neg_lambda=log_neg_lambda, omega=omega)
end

Lux.initialstates(::AbstractRNG, ::PhasorSSM) = NamedTuple()
Lux.parameterlength(l::PhasorSSM) = l.out_dims * l.in_dims + 2 * l.out_dims

function (l::PhasorSSM)(x::AbstractArray{<:Complex, 3}, ps, st)
    # x: (C_in × L × B) — complex input signal, one value per channel per time step
    C_in, L, B = size(x)

    # Step 1: Build the impulse-response kernel from oscillator parameters.
    # Equivalent to setting up the neuron constants in oscillator_bank:
    #   k = neuron_constant(leakage, t_period) = λ + iω
    λ = -exp.(ps.log_neg_lambda)                # (out,)  leakage (always negative)
    ω = ps.omega                                 # (out,)  angular frequency
    K = phasor_kernel(λ, ω, 1f0, L)             # out × L: precomputed kernel

    # Step 2: Apply weight matrix to mix input channels.
    # Equivalent to: layer.layer(x(t), params.layer, state.layer) in PhasorDense,
    # but done for all time steps at once instead of inside the ODE.
    xr = reshape(x, C_in, L * B)
    Hr = complex.(ps.weight * real.(xr), ps.weight * imag.(xr))
    H  = reshape(Hr, l.out_dims, L, B)          # out × L × B: weighted input

    # Step 3: Temporal integration via causal convolution.
    # Replaces: solve(ODEProblem(dzdt, u0, tspan), solver; ...)
    # The kernel K encodes the full dynamics (resonance + decay), so
    # convolving it with the input produces the same trajectory.
    Z = causal_conv(K, H)                        # out × L × B: membrane potential

    # Step 4: Activation (e.g. project to unit circle).
    # Same as layer.activation(sol.u[end]) in PhasorDense.
    Y = l.activation(Z)
    return Y, st
end

# ================================================================
# 4. SSM Readout Layer
# ================================================================

"""
    SSMReadout(readout_frac=0.25)

Bridges the 3D complex SSM output to 2D real phases for the Codebook layer.
Averages the last `readout_frac` fraction of time steps, normalizes to the
unit circle, and converts to phase angles in [-1, 1].

Input:  (C × L × B) complex  (membrane potentials over time)
Output: (C × B)     real     (phase angles, ready for Codebook)
"""
struct SSMReadout <: LuxCore.AbstractLuxLayer
    readout_frac::Float32
end

SSMReadout() = SSMReadout(0.25f0)

Lux.initialparameters(::AbstractRNG, ::SSMReadout) = NamedTuple()
Lux.initialstates(::AbstractRNG, ::SSMReadout) = NamedTuple()

function (l::SSMReadout)(z::AbstractArray{<:Complex, 3}, ps, st)
    L = size(z, 2)
    t0 = max(1, L - max(1, round(Int, L * l.readout_frac)) + 1)
    z_avg  = mean(z[:, t0:L, :]; dims=2)[:, 1, :]   # C × B
    z_norm = normalize_to_unit_circle(z_avg)
    return complex_to_angle(z_norm), st               # C × B real phases
end

# ================================================================
# 5. PSK Encoding
# ================================================================

"""
    psk_encode(images; n_repeats=1) -> ComplexF32 array (C × L × B)

Phase-shift-key encode grayscale images as complex time series.
Columns = channels (C=28), rows = time steps (L=28×n_repeats).
Pixel value v∈[0,1] → phase θ = 2v-1 ∈ [-1,1] (π-radians) → exp(iπθ).
"""
function psk_encode(images::AbstractArray{<:Real, 3}; n_repeats::Int=1)
    H, W, B = size(images)
    phases = 2f0 .* images .- 1f0                       # [0,1] → [-1,1]
    phases_ct = permutedims(phases, (2, 1, 3))           # channels × time × batch
    if n_repeats > 1
        phases_ct = repeat(phases_ct, 1, n_repeats, 1)
    end
    return angle_to_complex(phases_ct)
end

# ================================================================
# 6. Model + Loss
# ================================================================

function create_model(; D_hidden=128, n_classes=10, C_in=28, init=:uniform)
    model = Chain(
        PhasorSSM(C_in => D_hidden, normalize_to_unit_circle; init),
        PhasorSSM(D_hidden => D_hidden, identity; init),
        SSMReadout(0.25f0),
        Codebook(D_hidden => n_classes),
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
# 7. Evaluation
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
# 8. Main
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
    train_data = FashionMNIST(split=:train)
    test_data  = FashionMNIST(split=:test)

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
