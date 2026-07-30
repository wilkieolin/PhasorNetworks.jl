# §6 step 3 — decode check for the WaveExpertSheet prototype
# (docs/wavesheet_experts_design.md).
#
# Confirms the expert's stamped bind φ_e is a recoverable VSA operation and maps
# where recovery breaks. Force every expert to stamp its distinct code onto a
# carrier, propagate L cycles, then pooled-read each patch and identify the code
# by nearest cosine similarity to the codebook.
#
# Findings (seeds 1-6):
#   * differential decode (unbind vs a bind-free reference run, r·conj(r0)) is
#     robustly recoverable: 100% up to E=64 with a coherent carrier, ~92-99% even
#     at a fully incoherent carrier (σ=1). The stamp survives propagation.
#   * blind decode (no reference) is coherence-limited: 100% at σ=0 → 0% at σ=1 —
#     the square-law/coherence channel from the readout study.
#   * row-band tiling needs E ≤ H (else empty patches); enforced by the constructor.
#   * the per-expert bind is a 1-D scalar phase → no superposition capacity: codes
#     decode independently on disjoint patches but cannot be bundled at one site
#     and factored. High-D per-site codes + a resonator network are the path to
#     superposition capacity (Open Q §7.2), not yet implemented.
#
# Run: julia --project=. demos/wave_experts_decode.jl
using PhasorNetworks, Lux, Random, Statistics
using Random: Xoshiro
using PhasorNetworks: _apply_wave_experts, _wave_rollout_scan, _patch_read, angle_to_complex

function decode(; H, W, E, L, gfrac=0.7f0, csigma=0f0, mode=:diff, seed=1)
    rng = Xoshiro(seed)
    layer = WaveExpertSheet(H, W; n_experts=E, routing=:input, transmit=:potential)
    ps, st = Lux.setup(rng, layer)
    codes = Float32.(collect(range(-1f0,1f0,length=E+1))[1:E])
    ps = merge(ps, (bind_phase=codes,))
    srd(g)=dispersion(layer.sheet, merge(ps.sheet,(log_g=Float32[log(g)],)), st.sheet; mode=:discrete).spectral_radius
    lo,hi=1f-3,1f2; for _ in 1:40; m=sqrt(lo*hi); srd(m) < 1f0 ? (lo = m) : (hi = m); end
    ps = merge(ps,(sheet=merge(ps.sheet,(log_g=Float32[log(gfrac*sqrt(lo*hi))],)),))
    phis = ComplexF32.(cis.(Float32(pi).*codes)); B=1
    x = Phase.(clamp.(csigma.*randn(rng,Float32,H*W,L,B),-1f0,1f0))
    drive = reshape(angle_to_complex(x),H,W,L,B)
    d2 = _apply_wave_experts(drive, st.masks, phis, ones(Float32,E,L,B))
    Y = _wave_rollout_scan(layer.sheet, ps.sheet, st.sheet, zeros(ComplexF32,H,W,B), d2, L)
    r = vec(_patch_read(Y[:,:,end,:], st.masks))
    dec = if mode==:diff
        Yr = _wave_rollout_scan(layer.sheet, ps.sheet, st.sheet, zeros(ComplexF32,H,W,B), drive, L)
        angle.(r .* conj.(vec(_patch_read(Yr[:,:,end,:], st.masks)))) ./ Float32(pi)
    else  # blind: assume carrier phase 0
        angle.(r) ./ Float32(pi)
    end
    # count only non-empty patches (guard tiling degeneracy)
    valid = [e for e in 1:E if sum(st.masks[:,e]) > 0]
    ok = sum(e -> argmax([cos(Float32(pi)*(dec[e]-codes[ep])) for ep in valid])==findfirst(==(e),valid), valid)
    return ok/length(valid), length(valid)
end

println("=== differential decode, E=H (bands non-degenerate), coherent σ=0 ===")
for E in (4,8,16,32,64)
    a=[decode(H=E,W=E,E=E,L=4,csigma=0f0,mode=:diff,seed=s)[1] for s in 1:5]
    println("E=",lpad(E,2)," H=W=",E,"  acc=",round(mean(a),digits=3))
end
println("=== differential decode, incoherent σ=1.0, E=H ===")
for E in (4,8,16,32)
    a=[decode(H=E,W=E,E=E,L=4,csigma=1f0,mode=:diff,seed=s)[1] for s in 1:5]
    println("E=",lpad(E,2),"  acc=",round(mean(a),digits=3))
end
println("=== BLIND decode vs carrier incoherence σ (H=W=16, E=8) ===")
for cs in (0f0,0.25f0,0.5f0,1f0)
    a=[decode(H=16,W=16,E=8,L=4,csigma=cs,mode=:blind,seed=s)[1] for s in 1:6]
    println("σ=",rpad(cs,4)," blind-acc=",round(mean(a),digits=3))
end
