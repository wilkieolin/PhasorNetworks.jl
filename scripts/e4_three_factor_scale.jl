#!/usr/bin/env julia
# E4: ThreeFactorLockin Identity at Scale (FashionMNIST 784→256→64)
# Tests ThreeFactorLockin gradient equivalence to LockinEP with ≥100 random inputs, all layers

using Pkg
function find_repo_root(start_dir::String = pwd())
    dir = start_dir
    while !(isfile(joinpath(dir, "Project.toml")) && isdir(joinpath(dir, ".git")))
        parent = dirname(dir)
        if parent == dir
            error("Repository root not found from $(start_dir)")
        end
        dir = parent
    end
    return dir
end

repo_root = find_repo_root(@__DIR__)
cd(repo_root)
Pkg.activate(joinpath(repo_root, "scripts"))

using PhasorNetworks, Lux, LinearAlgebra, Statistics, Random, Optimisers, CSV, DataFrames, Printf, CUDA
using Random: Xoshiro
using PhasorNetworks: AbstractEPMethod, LockinEP, SimilarityCost, AbstractEPCost, ep_gradient, phasor_settle, chain_hebbians, _phase_input_to_complex, _weight_cache, _input_drive, _phasor_step, _pad_dynamics_zeros, _zero_grad, gpu_zeros

const SEED = 42
const N_INPUTS = 100  # ≥100 random inputs
const BATCHSIZE = 128
const HID = 256
const DOUT = 64
const SCALE = 0.4f0
const USE_CUDA = CUDA.functional()

const OUT = joinpath(repo_root, "results", "e4_three_factor_scale")
mkpath(OUT)

cdev = cpu_device()
gdev = gpu_device()
dev = USE_CUDA ? gdev : cdev

# Provenance
const GITREV = try
    rev = strip(read(`git -C $(repo_root) rev-parse --short HEAD`, String))
    d   = read(`git -C $(repo_root) diff HEAD -- src`, String)
    isempty(strip(d)) ? rev : rev * "-d" * string(hash(d), base = 16)[1:8]
catch
    "unknown"
end

# ---- ThreeFactorLockin (from ep_three_factor_stdp.jl) ----
Base.@kwdef struct ThreeFactorLockin
    ε::Float32          = 0.05f0
    ω_p::Float32        = 0.05f0
    n_cycles::Int       = 4
    T_warmup_cycles::Int = 2
    T_free::Int         = 100
    dt::Float32         = 0.1f0
    K_mode::Symbol      = :zero
    project::Symbol     = :hard
end

function three_factor_gradient(m::ThreeFactorLockin, chain::Lux.Chain, ps, st, x,
                               cost::AbstractEPCost;
                               omega_override::Union{Nothing, Vector} = nothing)
    # 1. Free settle
    s_free = phasor_settle(chain, ps, st, x, cost, 0f0;
                           T=m.T_free, dt=m.dt, K_mode=m.K_mode,
                           omega_override=omega_override, project=m.project)
    t_free_end = Float32(m.T_free * m.dt)

    # 2. DC Hebbian (for DC subtraction)
    h_dc = chain_hebbians(chain, ps, st, x, s_free)

    # 3. Warmup
    layer_keys = collect(keys(ps))
    z0 = _phase_input_to_complex(x)
    period_steps = round(Int, 2π / (m.ω_p * m.dt))
    T_warmup = m.T_warmup_cycles * period_steps
    T_lockin = m.n_cycles * period_steps

    states = [copy(s) for s in s_free]
    cache = _weight_cache(chain, ps, layer_keys)
    drive0 = _input_drive(chain, ps, st, layer_keys, z0; cache=cache)

    # Warmup
    for t in 1:T_warmup
        β_t = m.ε * cos(m.ω_p * t * m.dt)
        t_now = t_free_end + Float32((t - 1) * m.dt)
        states = _phasor_step(chain, ps, st, layer_keys, z0, cost,
                              β_t, m.dt, states; K_mode=m.K_mode,
                              omega_override=omega_override, drive0=drive0,
                              cache=cache, carriers=nothing, t_now=t_now,
                              project=m.project, soft_ε=0.1f0)
    end

    # 4. Three-factor accumulation
    t_warm_end = t_free_end + Float32(T_warmup * m.dt)

    # Eligibility accumulators
    E_W = Dict{Symbol, Any}()
    E_b = Dict{Symbol, Any}()
    for (l, key) in enumerate(layer_keys)
        haskey(ps[key], :weight) || continue
        E_W[key] = gpu_zeros(ps[key].weight, ComplexF32, size(ps[key].weight)...)
        if haskey(ps[key], :bias_real)
            E_b[key] = gpu_zeros(states[l], ComplexF32, size(states[l])...)
        end
    end

    # Integration with explicit three-factor rule
    for t in 1:T_lockin
        β_t = m.ε * cos(m.ω_p * t * m.dt)
        t_now = t_warm_end + Float32((t - 1) * m.dt)
        states = _phasor_step(chain, ps, st, layer_keys, z0, cost,
                              β_t, m.dt, states; K_mode=m.K_mode,
                              omega_override=omega_override, drive0=drive0,
                              cache=cache, carriers=nothing, t_now=t_now,
                              project=m.project, soft_ε=0.1f0)

        demod = ComplexF32(exp(-im * m.ω_p * Float32(t) * m.dt))

        for l in 1:length(layer_keys)
            key = layer_keys[l]
            haskey(ps[key], :weight) || continue

            z_l = states[l]
            z_in = (l == 1) ? z0 : states[l-1]

            # Eligibility trace: Hebbian outer product (adjoint for complex)
            elig_W = z_l * adjoint(z_in)
            elig_b = z_l

            # DC-subtracted eligibility
            elig_W_dc = elig_W .- h_dc[key].weight
            elig_b_dc = elig_b .- ComplexF32.(h_dc[key].bias_real .+ 1f0im .* h_dc[key].bias_imag)

            # Three-factor: accumulate eligibility × demodulation
            E_W[key] .+= elig_W_dc .* demod
            if haskey(E_b, key)
                E_b[key] .+= elig_b_dc .* demod
            end
        end
    end

    # 5. Convert to real-parameter gradients
    norm_factor = Float32(T_lockin) * m.ε
    grads = NamedTuple()

    for key in layer_keys
        if haskey(ps[key], :weight)
            entry = (weight = -2f0 .* real.(E_W[key]) ./ norm_factor,)
            if haskey(ps[key], :bias_real)
                entry = merge(entry, (
                    bias_real = -2f0 .* real.(E_b[key]) ./ norm_factor,
                    bias_imag = -2f0 .* imag.(E_b[key]) ./ norm_factor,
                ))
            end
            entry = _pad_dynamics_zeros(entry, ps[key])
            grads = merge(grads, NamedTuple{(key,)}((entry,)))
        else
            grads = merge(grads, NamedTuple{(key,)}((_zero_grad(ps[key]),)))
        end
    end

    return grads, s_free
end

# Metrics
cosine(a, b) = real(dot(vec(a), vec(b)) / (norm(vec(a)) * norm(vec(b)) + 1e-30))
rel_err(a, b) = norm(a - b) / (norm(b) + 1e-30)

# CSV output
csv_file = joinpath(OUT, "e4_three_factor_$(GITREV).csv")
isfile(csv_file) || CSV.write(csv_file, DataFrame(
    gitrev=String[],
    input_idx=Int[],
    layer=String[],
    param=String[],
    cos_fd_lk=Float32[],
    cos_fd_3f=Float32[],
    cos_lk_3f=Float32[],
    rel_err_lk_3f=Float32[],
    rel_err_fd_lk=Float32[],
    rel_err_fd_3f=Float32[]
))

# Build FashionMNIST-scale chain
println("=== Building FashionMNIST-scale chain (784→256→64) ===")
rng = Xoshiro(SEED)
chain = Chain(
    PhasorDense(784 => HID, normalize_to_unit_circle, use_bias=true),
    PhasorDense(HID => DOUT, normalize_to_unit_circle, use_bias=true)
)
ps, st = Lux.setup(rng, chain)
ps = (layer_1 = merge(ps.layer_1, (weight = SCALE .* ps.layer_1.weight,)),
      layer_2 = merge(ps.layer_2, (weight = SCALE .* ps.layer_2.weight,)))

if USE_CUDA
    chain = chain |> gdev
    ps = ps |> gdev
    st = st |> gdev
end

# LockinEP and ThreeFactorLockin methods
m_lockin = LockinEP(ε=0.05f0, ω_p=0.05f0, n_cycles=4, T_warmup_cycles=2, T_free=100, dt=0.1f0, K_mode=:zero)
m_3f = ThreeFactorLockin(ε=0.05f0, ω_p=0.05f0, n_cycles=4, T_warmup_cycles=2, T_free=100, dt=0.1f0, K_mode=:zero)

# Generate random inputs
println("=== Generating $N_INPUTS random inputs ===")
inputs = [Phase.(rand(rng, Float32, 784) .* 2 .- 1) for _ in 1:N_INPUTS]
targets = [ComplexF32.(exp.(im .* π .* (rand(rng, Float32, DOUT) .* 2 .- 1))) for _ in 1:N_INPUTS]

# Process each input
println("=== Processing $N_INPUTS inputs ===")
for idx in 1:N_INPUTS
    if idx % 10 == 0
        println("  Input $idx/$N_INPUTS")
    end
    
    x = inputs[idx] |> dev
    y = targets[idx] |> dev
    cost = SimilarityCost(y)
    
    # LockinEP gradient
    g_lk, _ = ep_gradient(m_lockin, chain, ps, st, x, cost)
    
    # ThreeFactorLockin gradient
    g_3f, _ = three_factor_gradient(m_3f, chain, ps, st, x, cost)
    
    # Compare per layer, per parameter
    for key in (:layer_1, :layer_2)
        for pname in (:weight, :bias_real, :bias_imag)
            haskey(g_lk[key], pname) || continue
            haskey(g_3f[key], pname) || continue
            
            lk = g_lk[key][pname]
            f3 = g_3f[key][pname]
            
            cos_lk_3f = cosine(lk, f3)
            re_lk_3f = rel_err(lk, f3)
            
            row = DataFrame(
                gitrev = GITREV,
                input_idx = idx,
                layer = string(key),
                param = string(pname),
                cos_fd_lk = Float32(NaN),  # FD not computed at scale
                cos_fd_3f = Float32(NaN),
                cos_lk_3f = Float32(cos_lk_3f),
                rel_err_lk_3f = Float32(re_lk_3f),
                rel_err_fd_lk = Float32(NaN),
                rel_err_fd_3f = Float32(NaN)
            )
            CSV.write(csv_file, row, append=true)
        end
    end
end

# Summary statistics
println("\n=== Summary ===")
df = CSV.read(csv_file, DataFrame)

for layer in ["layer_1", "layer_2"]
    sub = filter(r -> r.layer == layer, df)
    for param in ["weight", "bias_real", "bias_imag"]
        sub2 = filter(r -> r.param == param, sub)
        if nrow(sub2) > 0
            cos_mean = mean(sub2.cos_lk_3f)
            cos_min = minimum(sub2.cos_lk_3f)
            cos_max = maximum(sub2.cos_lk_3f)
            re_mean = mean(sub2.rel_err_lk_3f)
            re_max = maximum(sub2.rel_err_lk_3f)
            @printf("%s.%s: cos=%.6f (min=%.6f, max=%.6f), rel-err=%.6e (max=%.6e)\n",
                layer, param, cos_mean, cos_min, cos_max, re_mean, re_max)
        end
    end
end

println("\nResults: $csv_file")