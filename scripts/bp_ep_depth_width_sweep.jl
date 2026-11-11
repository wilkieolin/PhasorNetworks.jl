# Run with: julia --project=scripts scripts/bp_ep_depth_width_sweep.jl
#!/usr/bin/env julia
# scripts/bp_ep_depth_width_sweep.jl — how the BP→EP correspondence scales.
#
# Trains a chain with backpropagation at each depth and width, settles it as an
# EP network, and records the constants of Supplementary Proof B. Reuses
# `run_verification` from verify_bp_ep_bound.jl so there is one implementation
# of the measurement.
#
# The point of the sweep is that every previously reported depth trend came
# from UNTRAINED networks, and training moved the two-layer deviation by a
# factor of 2.4 — so the untrained trend cannot be extrapolated.

include(joinpath(@__DIR__, "verify_bp_ep_bound.jl"))

using CSV, DataFrames, Printf

const SWEEP_OUT = joinpath(repo_root, "results", "bp_ep_depth_width_sweep")
mkpath(SWEEP_OUT)

# Depth at fixed hidden width, then width at fixed depth. [784, 256, 64] is the
# architecture of the main text and appears in both families.
const CONFIGS = [
    (family = "depth", dims = [784, 256, 64]),
    (family = "depth", dims = [784, 256, 256, 64]),
    (family = "depth", dims = [784, 256, 256, 256, 64]),
    (family = "width", dims = [784, 64, 64]),
    (family = "width", dims = [784, 128, 64]),
    (family = "width", dims = [784, 512, 64]),
]

println("=== BP→EP depth and width sweep (hard projection) ===")
println("seed=$SEED  init scale=$SCALE  T_settle=$T_SETTLE  dt=$DT  n_test=$NTEST")
println("$(length(CONFIGS)) configurations\n")

rows = NamedTuple[]
for (i, cfg) in enumerate(CONFIGS)
    @printf("[%d/%d] %s  %s\n", i, length(CONFIGS), cfg.family, join(cfg.dims, "->"))
    flush(stdout)
    r = run_verification(cfg.dims)
    report(r)
    push!(rows, (family = cfg.family,
                 arch = join(cfg.dims, "->"),
                 depth = r.depth,
                 hidden = cfg.dims[2],
                 settle_resid = r.settle_resid,
                 rho_first = r.rho[1],
                 rho_last = r.rho[end],
                 rho_ratio = r.rho_ratio,
                 mu_min = minimum(r.mu_min),
                 mu_median_min = minimum(r.mu_median),
                 rho_G_min = r.rho_G_min,
                 rho_G_median = r.rho_G_median,
                 out_dev_mean = r.out_dev_mean,
                 out_dev_max = r.out_dev_max,
                 phase_removed = r.phase_removed,
                 acc_bp = r.acc_bp,
                 acc_ep = r.acc_ep,
                 drop = r.drop,
                 agreement = r.agreement,
                 margin_median = r.margin_median,
                 shift_mean = r.shift_mean,
                 bound = r.bound))
    CSV.write(joinpath(SWEEP_OUT, "per_sample_$(join(cfg.dims, "x")).csv"), r.per_sample)
    CSV.write(joinpath(SWEEP_OUT, "sweep_$(SEED).csv"), DataFrame(rows))
end

df = DataFrame(rows)
println("\n\n=== SUMMARY ===")
println("depth family (hidden 256):")
for r in eachrow(df[df.family .== "depth", :])
    @printf("  L=%d %-22s  resid %.1e  rho(G) med %6.2f  out dev %.3f  BP %.3f EP %.3f drop %+.4f  bound %.3f\n",
            r.depth, r.arch, r.settle_resid, r.rho_G_median, r.out_dev_mean,
            r.acc_bp, r.acc_ep, r.drop, r.bound)
end
println("width family (depth 2):")
for r in eachrow(df[df.family .== "width", :])
    @printf("  h=%-4d %-20s  resid %.1e  rho(G) med %6.2f  out dev %.3f  BP %.3f EP %.3f drop %+.4f  bound %.3f\n",
            r.hidden, r.arch, r.settle_resid, r.rho_G_median, r.out_dev_mean,
            r.acc_bp, r.acc_ep, r.drop, r.bound)
end
println("\nwrote $(SWEEP_OUT)/sweep_$(SEED).csv")
