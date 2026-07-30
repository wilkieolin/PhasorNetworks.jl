# wave_spiking_video.jl — render propagating spiking activity on the wave sheet,
# driven by a FashionMNIST image, with COLOUR = phase (cyclic hue) and brightness
# = amplitude. Produces an mp4 (and a gif).
#
# The sheet runs in :spike mode. We roll the discrete per-cycle SSM to get the
# slowly-varying phasor envelope Z[t] at each cycle, then reconstruct the smooth
# oscillating field between cycles: the physical potential is Re(Z(t) e^{iωt}), so
# the INSTANTANEOUS phase is arg(Z) + (carrier advance). Interpolating Z across
# cycles and sweeping the carrier gives smooth traveling waves; hue encodes that
# instantaneous phase, so wavefronts appear as moving colour bands.
#
# Run:  julia --project=. demos/wave_spiking_video.jl
#   env: SPK_L (cycles, 30), SPK_SUB (sub-frames/cycle, 6), SPK_IMG (image idx, 7)
#
ENV["GKSwstype"] = "100"
using PhasorNetworks, Lux, Plots, Statistics
using Random: Xoshiro
gr()

const OUTDIR = joinpath(@__DIR__, "wave_out"); isdir(OUTDIR) || mkpath(OUTDIR)
_envi(k,d) = parse(Int, get(ENV,k,string(d)))
const H = 28; const W = 28
const L   = _envi("SPK_L", 30)      # carrier cycles
const SUB = _envi("SPK_SUB", 6)     # reconstructed sub-frames per cycle
const IMG = _envi("SPK_IMG", 7)

# ---- image → phase drive --------------------------------------------
d = fashion_mnist_data(:test)
img = d.features[:, :, IMG]
zimg = ComplexF32.(angle_to_complex(Float32.((2f0 .* img .- 1f0) .* 0.5f0)))   # (28,28) unit phasors

# ---- sheet: spiking, Mexican-hat coupling (traveling waves) ----------
l = PhasorWaveSheet(H, W; coupling = :dog, transmit = :spike,
                    init_log_g = log(0.8), init_log_neg_lambda = log(0.05),
                    init_A_exc = 1.0, init_log_sigma_exc = log(1.5),
                    init_B_inh = 0.5, init_log_sigma_inh = log(3.0))
ps, st = Lux.setup(Xoshiro(0), l)
ω = PhasorNetworks.period_to_angfreq(l.spk_args.t_period)

# continuous image drive → sustained spiking activity
drive = repeat(reshape(zimg, H, W, 1, 1), 1, 1, L, 1)
Z = PhasorNetworks._wave_rollout(l, ps, st, reshape(zero(zimg), H, W, 1), drive, L)[:, :, :, 1]  # (H,W,L)

_wrap(x) = x - 2f0 * round(x / 2f0)

println("rendering $(H)×$(W) sheet, $L cycles × $SUB sub-frames = $((L-1)*SUB) frames …")
anim = @animate for t in 1:(L-1), k in 0:(SUB-1)
    frac = Float32(k) / SUB
    Zc = (1f0 - frac) .* Z[:, :, t] .+ frac .* Z[:, :, t+1]           # interpolate envelope
    iphase = _wrap.(Float32.(angle.(Zc)) ./ Float32(pi) .+ 2f0 * frac) # + carrier advance (units of π)
    mag = abs.(Zc) ./ (maximum(abs.(Zc)) + 1f-6)                       # per-frame brightness
    disp = [mag[i, j] > 0.15f0 ? iphase[i, j] : NaN32 for i in 1:H, j in 1:W]
    heatmap(disp; c = :hsv, clims = (-1, 1), aspect_ratio = 1, framestyle = :none,
            legend = false, colorbar = false, size = (360, 380), background = :black,
            title = "spiking phase field — cycle $t", titlefontcolor = :white, titlefontsize = 9)
end

mp4(anim, joinpath(OUTDIR, "fmnist_spiking_phase.mp4"), fps = 18)
gif(anim, joinpath(OUTDIR, "fmnist_spiking_phase.gif"), fps = 18)
println("saved fmnist_spiking_phase.mp4 / .gif in $OUTDIR")
