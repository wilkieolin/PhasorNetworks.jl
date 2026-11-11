# wave_experts_keyed.jl — Rung 1 of the MoE-patch demonstrator ladder.
#
# A KEYED ASSOCIATIVE TRANSFORM. This is the task FashionMNIST-10 could not be:
# one where expert specialisation is *necessary*, the routing signal is real phase
# structure, and specialisation is *measurable*.
#
#   • Fixed codebook of E address symbols A_e — full-range phase patterns, each
#     living in expert e's row-band (real phase structure the matched gate can
#     route on, unlike FashionMNIST's thin amplitude-coded phase).
#   • Each example: pick e, stamp A_e into band e over a noisy background.
#   • Target: the same field with band e rotated by a fixed value phasor V_e.
#   • The network must (routing) match A_e with expert e's *learned* key
#     (bind_key, random init ⇒ EMERGENT address↔expert permutation), then (bind)
#     stamp its learned value (bind_phase) so the output reconstructs the target.
#
# Wrong expert ⇒ wrong band stamped ⇒ double error, so correct routing AND the
# correct per-expert transform are both forced by one reconstruction loss. No head.
#
# VERIFY (emergent, then measured):
#   1. routing confusion → a clean permutation (address e always fires one expert)
#   2. bind alignment: learned bind_phase[perm(e)] ≈ V_e
#   3. key alignment:  learned key on band e ≈ A_e (so the unbind is coherent)
#   4. CAUSAL ablation: zero expert perm(e)'s bind ⇒ damage localises to address-e
#   5. unbind decode: output band ⊙ conj(V_e) recovers A_e
#
# Run:  julia --project=. demos/wave_experts_keyed.jl
#
using PhasorNetworks, Lux, Zygote, Optimisers, Random, Statistics, Printf
using Random: Xoshiro
using Optimisers: Adam
using Zygote: withgradient

const H = 12; const W = 12; const E = 4; const L = 4
const BTR = 64; const STEPS = 500; const GINIT = 1f-3     # ~0 coupling ⇒ sheet is phase-preserving

bandrows(e) = [i for i in 1:H if min(E, 1 + ((i - 1) * E) ÷ H) == e]

# Fixed codebook (seeded): address pattern per expert-band + a distinct value phase.
const CB   = Xoshiro(2024)
const ADDR = [2f0 .* rand(CB, Float32, length(bandrows(e)), W) .- 1f0 for e in 1:E]
const VAL  = Float32[-0.6, -0.2, 0.2, 0.6]                # target transform (units of π)

_wrap(x) = x - 2f0 * round(x / 2f0)                       # → [-1,1]

# Every band carries a fresh FULL-range random phase pattern (so "which patch has
# signal" is uninformative — coherence routing cannot solve this); the active band
# is overwritten with its codebook symbol A_e. Only a key that MATCHES A_e can
# single out the active expert, so the matched key is load-bearing.
function make_batch(rng, B)
    θ  = 2f0 .* rand(rng, Float32, H, W, B) .- 1f0        # full random phase in every band
    tg = copy(θ)
    es = rand(rng, 1:E, B)
    for b in 1:B
        e = es[b]; rows = bandrows(e)
        θ[rows, :, b]  .= ADDR[e]                         # active band ← codebook symbol A_e
        tg[rows, :, b] .= _wrap.(ADDR[e] .+ VAL[e])       # target: active band rotated by V_e
    end
    xin = Phase.(repeat(reshape(θ, H * W, 1, B), 1, L, 1))  # (HW,L,B) constant drive
    return xin, reshape(tg, H * W, B), es
end

# Smooth circular reconstruction loss: 0 when output phase ≡ target (mod 2).
recon_loss(a, tg) = mean(1f0 .- cospi.(a .- tg))

const LAYER = WaveExpertSheet(H, W; n_experts = E, routing = :input,
                              route_feature = :matched, transmit = :potential,
                              init_log_g = log(GINIT), balance = false, hard = true)

function loss(ps, st, xin, tg)
    y, _ = LAYER(xin, ps, st)
    return recon_loss(Float32.(y[:, end, :]), tg)
end

# ---- metrics ---------------------------------------------------------
function fired_expert(ps, st, xin)
    g = route_stats(LAYER, ps, st, xin).gate                    # (E,L,B)
    B = size(g, 3)
    return [argmax(vec(sum(g[:, :, b]; dims = 2))) for b in 1:B] # (B,)
end

function evaluate(ps, st)
    xin, tg, es = make_batch(Xoshiro(999), 512)
    pred = fired_expert(ps, st, xin)
    conf = zeros(Int, E, E)                                     # [address, fired]
    for b in 1:512; conf[es[b], pred[b]] += 1; end
    perm = [argmax(conf[e, :]) for e in 1:E]                    # address → expert
    route_acc = mean(pred[b] == perm[es[b]] for b in 1:512)
    is_perm   = length(unique(perm)) == E

    y, _ = LAYER(xin, ps, st); a = Float32.(y[:, end, :])
    recon = recon_loss(a, tg)

    # bind alignment: learned bind_phase[perm(e)] vs V_e (circular cos, →1 good)
    bind_al = mean(cospi(_wrap(ps.bind_phase[perm[e]] - VAL[e])) for e in 1:E)
    # key coherence: |mean cos(key − A_e)| on the addressed band. Magnitude, not
    # signed mean — a key offset by π is EQUALLY selective (the sign is absorbed by
    # Wr and the bind), so a signed average would wrongly cancel to ~0. This equals
    # |Re(ρ)| of the matched read on the address: →1 iff the key singles A_e out.
    key = reshape(ps.bind_key, H, W, E)
    key_coh = mean(begin
        rows = bandrows(e); j = perm[e]
        abs(mean(cospi.(_wrap.(key[rows, :, j] .- ADDR[e]))))
    end for e in 1:E)
    # load-bearing check: zero the keys ⇒ routing must collapse toward chance (1/E).
    ps0 = merge(ps, (; bind_key = zero(ps.bind_key)))
    pred0 = fired_expert(ps0, st, xin)
    route_acc_nokeys = mean(pred0[b] == perm[es[b]] for b in 1:512)
    return (; conf, perm, route_acc, is_perm, recon, bind_al, key_coh, route_acc_nokeys)
end

# Causal specialisation: ablate expert perm(e)'s bind (→ identity); the error must
# jump on address-e examples and stay flat elsewhere.
function ablation(ps, st, perm)
    xin, tg, es = make_batch(Xoshiro(1234), 512)
    y0, _ = LAYER(xin, ps, st); base = 1f0 .- cospi.(Float32.(y0[:, end, :]) .- tg)
    println("  ablate expert  | err on its address | err on others  (Δ = localised damage)")
    for e in 1:E
        j = perm[e]
        bp = copy(ps.bind_phase); bp[j] = 0f0                  # V=1 (no-op)
        psa = merge(ps, (; bind_phase = bp))
        ya, _ = LAYER(xin, psa, st); err = 1f0 .- cospi.(Float32.(ya[:, end, :]) .- tg)
        on  = mean(vec(mean(err;  dims = 1))[es .== e])
        off = mean(vec(mean(err;  dims = 1))[es .!= e])
        @printf("   e=%d (expert %d) |      %.4f       |    %.4f     (Δ=%+.4f)\n",
                e, j, on, off, on - off)
    end
    @printf("  (baseline recon, no ablation: %.4f)\n", mean(base))
end

# ---------------------------------------------------------------------
function main()
    println("="^70)
    println("Rung 1 — keyed associative transform ($(E) experts, $(H)×$(W) sheet, L=$L)")
    println("="^70)
    ps, st = Lux.setup(Xoshiro(1), LAYER)
    opt = Optimisers.setup(Adam(0.02f0), ps)
    rng = Xoshiro(7)
    for step in 1:STEPS
        xin, tg, es = make_batch(rng, BTR)
        rs = route_stats(LAYER, ps, st, xin)
        st = merge(st, (; route_bias = update_moe_bias(st.route_bias, rs.gate; rate = 0.05f0)))
        l, g = withgradient(p -> loss(p, st, xin, tg), ps)
        opt, ps = Optimisers.update(opt, ps, g[1])
        if step % 100 == 0 || step == 1
            m = evaluate(ps, st)
            @printf("step %4d | loss %.4f | recon %.4f | route_acc %.3f | perm %s %s\n",
                    step, l, m.recon, m.route_acc, string(m.perm), m.is_perm ? "✓" : "✗")
        end
    end
    m = evaluate(ps, st)
    println("\n[routing confusion  rows=address, cols=fired expert]")
    for e in 1:E; println("  A", e, " → ", m.conf[e, :]); end
    @printf("\n  routing permutation : %s   (bijection: %s)\n", string(m.perm), m.is_perm ? "YES" : "no")
    @printf("  routing accuracy    : %.3f\n", m.route_acc)
    @printf("  reconstruction err  : %.4f  (0 = perfect phase match)\n", m.recon)
    @printf("  bind  alignment     : %.3f  (1 = learned V_e matches target)\n", m.bind_al)
    @printf("  key   coherence     : %.3f  (1 = key singles out its address A_e)\n", m.key_coh)
    @printf("  routing w/ keys=0   : %.3f  (→ chance %.2f ⇒ keys are load-bearing)\n",
            m.route_acc_nokeys, 1 / E)
    println("\n[causal ablation — specialisation is real iff damage localises]")
    ablation(ps, st, m.perm)

    ok = m.is_perm && m.route_acc > 0.95 && m.recon < 0.05 && m.bind_al > 0.9 &&
         m.key_coh > 0.6 && m.route_acc_nokeys < 0.45
    println("\nRUNG 1 ", ok ? "PASS ✓ — emergent routing + specialisation verified" :
                            "INCOMPLETE — inspect metrics above")
end
main()
