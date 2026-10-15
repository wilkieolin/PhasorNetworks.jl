# E5b — is a "failure to propagate" a property of the medium or of the seed?
# The plane-front theory (E5) finds plane-front solutions in two configurations
# where small radial seeds died: R=3 at G=0.15, and the checkerboard sublattice
# at 50 and 70 ps/pitch. Drive each with seeds of growing radius and with a
# full line source (a planar front from the start).
include("common.jl")
backend, dev = load_gpu_backend()
c = RFTransponder(); N = 256
color(y, x) = iseven(y + x)
function sub_sheet(pps; G = 0.45)
    offs = stencil_offsets(3); K = length(offs)
    D = Float32[pps * sqrt(o[1]^2 + o[2]^2) for y in 1:N, x in 1:N, o in offs]
    Gm = Float32[iseven(o[1] + o[2]) ? G : 0 for y in 1:N, x in 1:N, o in offs]
    DeviceSheet(c, N, N; D, G = Gm)
end
line_drive() = EventDrive(Float32[y == 20 ? 0f0 : Inf32 for y in 1:N, x in 1:N], 40)
cases = [("R=3 G=0.15, 10 ps", DeviceSheet(c, N, N; gain = 0.15), trues(N, N), 0.0373),
         ("R=3 G=0.12, 10 ps", DeviceSheet(c, N, N; gain = 0.12), trues(N, N), NaN),
         ("checker G=0.45, 30 ps", sub_sheet(30), [color(y, x) for y in 1:N, x in 1:N], NaN),
         ("checker G=0.45, 50 ps", sub_sheet(50), [color(y, x) for y in 1:N, x in 1:N], 0.0114),
         ("checker G=0.45, 70 ps", sub_sheet(70), [color(y, x) for y in 1:N, x in 1:N], 0.0086)]
rows = NamedTuple[]
for (name, sh, m, upred) in cases
    nst = 26000
    for (src, d) in [("seed r=4", seed_drive(N, N; radius = 4)), ("seed r=10", seed_drive(N, N; radius = 10)),
                     ("seed r=20", seed_drive(N, N; radius = 20)), ("line", line_drive())]
        tr = sheet_simulate(sh, nst; drive = d, device = dev)
        tf = ifelse.(m, tr.t_first, Inf32)
        fired = count(isfinite, tf) / count(m)
        u = NaN
        if src == "line" && fired > 0.5
            ss = Float64[]; tt = Float64[]
            for x in 30:N-30, y in 60:N-20
                isfinite(tf[y, x]) && (push!(ss, y); push!(tt, tf[y, x]))
            end
            length(ss) > 20 && (u = 1 / lin(tt, ss)[1] |> inv |> inv; u = lin(tt, ss)[1])
        end
        refire = maximum(tr.n_spikes[m])
        @printf("    %-22s %-9s fired %5.1f%%  max spikes %d  u_line %.4f (plane theory %.4f)\n", name, src, 100fired, refire, u, upred)
        push!(rows, (; config = name, source = src, fired, max_spikes = refire, u_line = u, u_plane_theory = upred))
    end
end
write_tsv("e5b_nucleation.tsv", rows)
