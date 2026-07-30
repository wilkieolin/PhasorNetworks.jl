# wave_xor_reservoir.jl — "computing in a bucket" with spiking waves
#
# The classic reservoir-computing / liquid-state-machine demonstration
# (Fernando & Sojakka 2003, "Pattern Recognition in a Bucket"; Maass et al. 2002),
# with the R&F PhasorWaveSheet as the reservoir and spiking activity as the waves.
#
# XOR is not linearly separable in the input, so the nonlinearity must come from
# the medium. Two bits are injected as phase-encoded drives at two sites; the
# spiking waves propagate and INTERFERE; the sheet is FIXED (coupling untrained),
# and only a linear (ridge) readout on the interference intensity |z|^2 is fit.
#
# Result (seed 7): raw bits 0.74 | intensity before waves meet (L=1) 0.49 |
# reservoir (Re,Im) 0.54 | reservoir |z|^2 1.00. The separable feature is the
# square-law interference intensity, and only after propagation (waves meeting).
#
# Run: julia --project=. demos/wave_xor_reservoir.jl
using PhasorNetworks, Lux, Random, Statistics, LinearAlgebra
using Random: Xoshiro

function main()
    rng = Xoshiro(7)
    H=W=12; L=8; N=1200
    # FIXED reservoir: R&F sheet, spiking waves, coupling NOT trained
    sheet = PhasorWaveSheet(H, W; transmit=:spike, saturating=false, init_log_g=log(0.6))
    ps, st = Lux.setup(Xoshiro(1), sheet)

    # two input sites; bit -> phase (0 or 1 in Phase units => phasor +1 / -1)
    pA=(3,6); pB=(9,6); rad=1
    function encode(bits) # bits :: (2,N)
        drive = zeros(ComplexF32, H, W, L, size(bits,2))
        for i in 1:size(bits,2)
            θa = bits[1,i]*1f0 + 0.05f0*randn(rng,Float32)
            θb = bits[2,i]*1f0 + 0.05f0*randn(rng,Float32)
            for dx in -rad:rad, dy in -rad:rad
                drive[pA[1]+dx, pA[2]+dy, :, i] .= cis(Float32(pi)*θa)
                drive[pB[1]+dx, pB[2]+dy, :, i] .= cis(Float32(pi)*θb)
            end
        end
        return drive
    end

    bits = rand(rng, 0:1, 2, N)
    y = Float32.(xor.(bits[1,:], bits[2,:]))                     # XOR label
    drive = encode(bits)
    z0 = zeros(ComplexF32, H, W, N)
    Y = wave_simulate(sheet, ps, st; z0=z0, L=L, drive=drive)   # (H,W,L,N) complex, spiking waves

    intens = reshape(abs2.(Y), H*W*L, N)                         # |z|^2 reservoir features
    reim   = reshape(vcat(reshape(real.(Y),H*W*L,N), reshape(imag.(Y),H*W*L,N)), 2*H*W*L, N)
    l1     = reshape(abs2.(Y[:,:,1,:]), H*W, N)                  # intensity BEFORE waves meet (L=1)
    raw    = Float32.(bits)                                      # raw input bits

    # ridge linear readout + accuracy (80/20 split)
    tr = 1:round(Int,0.8N); te = round(Int,0.8N)+1:N
    function ridge_acc(X)
        Xa = vcat(X, ones(Float32,1,size(X,2)))                 # bias row
        A = Xa[:,tr]; b = y[tr]
        w = (A*A' + 1f0*I) \ (A*b)
        ŷ = (w' * vcat(X[:,te], ones(Float32,1,length(te))))'
        mean((vec(ŷ).>0.5f0) .== (y[te].>0.5f0))
    end
    println("XOR via reservoir computing (R&F spiking wave sheet, FIXED coupling)")
    println("  linear on raw input bits              : ", round(ridge_acc(raw),digits=3))
    println("  linear on field intensity, L=1 (no mix): ", round(ridge_acc(l1),digits=3))
    println("  linear on reservoir (Re,Im), L=$L      : ", round(ridge_acc(reim),digits=3))
    println("  linear on reservoir |z|^2, L=$L        : ", round(ridge_acc(intens),digits=3))
end
main()
