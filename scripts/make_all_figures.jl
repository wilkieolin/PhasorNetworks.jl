#!/usr/bin/env julia
# Master script: regenerate every figure the manuscript includes, then copy the
# PDFs into the manuscript repository.
#
# figures/ is gitignored in this repository, so nothing here is under version
# control. The manuscript keeps its own tracked copy -- see MANUSCRIPT_FIG_DIR
# below -- because otherwise the paper cannot be rebuilt from a fresh clone.

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

const MANUSCRIPT_FIG_DIR = get(ENV, "LIEP_MANUSCRIPT_FIGS",
                               joinpath(homedir(), "Documents", "ICRC-LIEP-manuscript", "figures"))

println("=== Generating publication figures ===\n")
mkpath("figures")
include("figure_style.jl")

# The five main-text figures, in manuscript order. Each reads CSVs under
# results/ and computes nothing -- see the note on retired scripts below.
const FIGURES = [
    ("Figure 1 - gradient validity",      "make_figure1_gradient_validity.jl", "figure1_gradient_validity"),
    ("Figure 2 - adiabatic basin",        "make_figure2_basin.jl",             "figure2_basin"),
    ("Figure 3 - spike-timing readout",   "make_figure3_readout.jl",           "figure3_readout"),
    ("Figure 4 - impairment recovery",    "make_figure4_recovery.jl",          "figure4_recovery"),
    ("Figure 5 - three-factor rule",      "make_figure5_threefactor.jl",       "figure5_threefactor"),
]

for (i, (title, script, _)) in enumerate(FIGURES)
    println("$i/$(length(FIGURES)): $title")
    include(script)
    println()
end

# ---------------------------------------------------------------- publish
mkpath(MANUSCRIPT_FIG_DIR)
println("Copying PDFs to $MANUSCRIPT_FIG_DIR")
for (_, _, stem) in FIGURES
    src = joinpath("figures", stem * ".pdf")
    if isfile(src)
        cp(src, joinpath(MANUSCRIPT_FIG_DIR, stem * ".pdf"); force = true)
        println("  $stem.pdf")
    else
        @warn "missing $src -- the manuscript will not build"
    end
end

# ---------------------------------------------------------------- geometry check
# Every figure is authored at FULL_W = 370.6 pt == \textwidth of the one-column
# sn-jnl class, and placed with \includegraphics[width=\textwidth]{...} so the
# scale factor is exactly 1.000 and the 7/8/9 pt fonts render at nominal size.
println("\nPage-size check (target width 371 pt):")
for (_, _, stem) in FIGURES
    f = joinpath("figures", stem * ".pdf")
    isfile(f) || continue
    out = try read(`pdfinfo $f`, String) catch; "" end
    m = match(r"Page size:\s+([0-9.]+) x ([0-9.]+)", out)
    if m !== nothing
        w = parse(Float64, m[1])
        flag = abs(w - 371) <= 1.5 ? "ok" : "WRONG WIDTH"
        println("  $stem: $(m[1]) x $(m[2]) pt  [$flag]")
    end
end

# Retired, deliberately not run:
#   make_figure1_fashionmnist.jl  -- panels (b),(c) superseded by E1; see figures/SUPERSEDED.txt
#   make_figure2_adiabatic.jl     -- replaced by make_figure2_basin.jl
#   make_figure3_finetune.jl      -- replaced by make_figure4_recovery.jl
#   make_figure4_threefactor.jl   -- replaced by make_figure5_threefactor.jl; it also
#                                    re-ran the experiment at figure time
#   make_figure_3factor.jl        -- diagnostic scratch, superseded
#   make_supplementary.jl         -- re-runs the detuning sweep at figure time, and that
#                                    sweep sits on a broken dw = 0 baseline
println("\n=== done ===")
