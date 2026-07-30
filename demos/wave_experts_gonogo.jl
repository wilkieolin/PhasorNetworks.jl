# §6 go/no-go probe for the WaveExpertSheet prototype (docs/wavesheet_experts_design.md).
#
# Central question: when a discrete top-1 router (straight-through gate) reads from
# and writes into the wave substrate, does routing stay trainable — or collapse to
# one always-on expert? Synthetic task: one row-band patch is phase-coherent, the
# label is which; the router should route to that patch's expert.
#
# Result (seed 7): loss→~0, accuracy→100%, per-expert load stays spread
# (load-entropy ~1.2 of max 1.386) — NO collapse. The gate gradient survives the
# discrete-SSM rollout. Run: julia --project=. demos/wave_experts_gonogo.jl
#
using PhasorNetworks, Lux, Random, Zygote, FFTW, Statistics, Optimisers
using Random: Xoshiro

function main()
    rng = Xoshiro(7)
    H = W = 8; L = 5; E = 4; B = 32
    RF = Symbol(get(ENV, "WAVE_RF", "coherence"))   # :coherence | :dispersion | :matched
    println("route_feature = :$RF")
    layer = WaveExpertSheet(H, W; n_experts=E, routing=:input, transmit=:potential,
                            route_feature=RF, init_log_g = log(0.15), balance=false, hard=true)
    ps, st = Lux.setup(rng, layer)
    Wout = 0.1f0 .* randn(rng, Float32, E, 2*H*W)
    θ = (wave = ps, Wout = Wout)

    function make_batch(rng)
        x = 2f0 .* rand(rng, Float32, H*W, L, B) .- 1f0
        y = rand(rng, 1:E, B)
        xg = reshape(x, H, W, L, B)
        for b in 1:B
            p = y[b]
            rows = [i for i in 1:H if min(E, 1+((i-1)*E)÷H) == p]
            xg[rows, :, :, b] .= 0.3f0
        end
        return Phase.(reshape(xg, H*W, L, B)), y
    end
    onehot(y) = Float32.(reduce(hcat, [ (1:E).==yi for yi in y ]))

    function loss(θ, st, x, yoh)
        yph, _ = layer(x, θ.wave, st)
        zf = Float32.(yph[:, end, :])
        feat = vcat(cospi.(zf), sinpi.(zf))
        lg = θ.Wout * feat
        lp = lg .- log.(sum(exp.(lg); dims=1) .+ 1f-12)
        return -mean(sum(yoh .* lp; dims=1))
    end

    opt = Optimisers.setup(Optimisers.Adam(0.02f0), θ)
    loadent(v) = (p = v ./ (sum(v)+1f-12); -sum(p .* log.(p .+ 1f-12)))
    println("step |  loss  | acc  | load(per-expert)         | load_ent (max=$(round(log(E),digits=3)))")
    for step in 1:60
        x, y = make_batch(rng); yoh = onehot(y)
        rs = route_stats(layer, θ.wave, st, x)
        st = merge(st, (route_bias = update_moe_bias(st.route_bias, rs.gate; rate=0.05f0),))
        l, g = Zygote.withgradient(t -> loss(t, st, x, yoh), θ)
        opt, θ = Optimisers.update(opt, θ, g[1])
        if step % 10 == 0 || step == 1
            yph, _ = layer(x, θ.wave, st); zf = Float32.(yph[:,end,:])
            lg = θ.Wout * vcat(cospi.(zf), sinpi.(zf))
            acc = mean(map(b -> argmax(lg[:,b]) == y[b], 1:B))
            println(lpad(step,4), " | ", round(l,digits=4), " | ", round(acc,digits=3),
                    " | ", round.(rs.load,digits=3), " | ", round(loadent(rs.load),digits=3))
        end
    end
end
main()
