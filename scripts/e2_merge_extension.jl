#!/usr/bin/env julia
# Merge the sharded extended-severity runs into the main sweep CSV.
#
# Each shard wrote results/e2_extend/<cell>/analog_finetune_<gitrev>.csv, and the
# spark2 shards' files are fetched into results/e2_extend_spark2/ first. This
# script concatenates them into results/ep_analog_finetune/analog_finetune_<gitrev>.csv.
#
# DEDUPLICATION IS NOT OPTIONAL. The sweep is not bit-reproducible -- BP
# pretraining depends on BLAS thread count, so every invocation yields a slightly
# different pretrained network. Two rows sharing (pretrain, type, param, rep)
# are therefore the SAME damage draw measured against DIFFERENT weights, not two
# independent samples; counting both would understate the spread and inflate n.
# The first occurrence wins, and the main CSV is read first so existing rows are
# never displaced by a re-run.
#
#   julia --project=scripts scripts/e2_merge_extension.jl [--apply]
#
# Without --apply it reports what would change and writes nothing.

using Pkg
function find_repo_root(start_dir::String = pwd())
    dir = start_dir
    while !(isfile(joinpath(dir, "Project.toml")) && isdir(joinpath(dir, ".git")))
        parent = dirname(dir); parent == dir && error("no repo root"); dir = parent
    end
    return dir
end
repo_root = find_repo_root(@__DIR__); cd(repo_root); Pkg.activate(joinpath(repo_root, "scripts"))
using CSV, DataFrames, Printf

const APPLY = "--apply" in ARGS
const MAIN = joinpath(repo_root, "results", "ep_analog_finetune")
const KEY = [:pretrain, :impairment_type, :param, :rep]

main_files = filter(f -> occursin(r"^analog_finetune_[0-9a-f]+\.csv$", basename(f)),
                    readdir(MAIN; join = true))
isempty(main_files) && error("no main sweep CSV found")
target = last(sort(main_files))
base = CSV.read(target, DataFrame)
@printf("main   %-52s %4d rows\n", basename(target), nrow(base))

shards = DataFrame[]
for root in ("e2_extend", "e2_extend_spark2")
    d = joinpath(repo_root, "results", root)
    isdir(d) || continue
    for cell in sort(readdir(d))
        for f in readdir(joinpath(d, cell); join = true)
            endswith(f, ".csv") || continue
            df = CSV.read(f, DataFrame)
            nrow(df) == 0 && continue
            @printf("shard  %-52s %4d rows\n", "$root/$cell", nrow(df))
            push!(shards, df)
        end
    end
end
isempty(shards) && (println("\nno shard rows to merge"); exit(0))

allshards = vcat(shards...; cols = :union)
merged = vcat(base, allshards; cols = :union)
before = nrow(merged)
unique!(merged, KEY)                       # first occurrence wins; base was first
@printf("\nmerged %d rows, %d dropped as duplicate %s\n",
        nrow(merged), before - nrow(merged), string(KEY))

# Base wins ties, so a shard row whose key already exists is discarded. Say which,
# rather than letting it happen silently: the retained row may come from a
# different pretraining run than the rest of its cell, and the reader should be
# able to see that and decide.
clash = semijoin(unique(allshards, KEY), base, on = KEY)
if nrow(clash) > 0
    # On a repeat merge most clashes are simply rows folded in last time, which is
    # not interesting. Summarise per cell, and only spell out rows whose values
    # actually differ from the retained copy -- those are the ones where the
    # tie-break changed the data rather than just re-confirming it.
    cmp = innerjoin(select(clash, KEY..., :acc_tuned => :shard_tuned),
                    select(base, KEY..., :acc_tuned => :base_tuned), on = KEY)
    differing = cmp[.!isapprox.(cmp.shard_tuned, cmp.base_tuned; atol = 1e-6), :]
    @printf("\n%d shard rows already present in the main CSV (base wins ties)", nrow(clash))
    if nrow(differing) == 0
        println(" -- all identical, nothing changed.")
    else
        println(":")
        for r in eachrow(sort(differing, [:impairment_type, :param, :rep]))
            @printf("  %-12s %-6s rep %-2d  base acc_tuned %.4f, shard %.4f -- kept base\n",
                    r.impairment_type, string(r.param), r.rep, r.base_tuned, r.shard_tuned)
        end
        println("  (delete those rows from the main CSV first if you want the shard's instead)")
    end
end

new = antijoin(merged, base, on = KEY)
if nrow(new) > 0
    println("\nnew cells:")
    for g in groupby(sort(new, [:impairment_type, :param]), [:impairment_type, :param])
        @printf("  %-12s %-6s n=%d\n", first(g.impairment_type), string(first(g.param)), nrow(g))
    end
end

if APPLY
    cp(target, target * ".bak"; force = true)
    CSV.write(target, merged)
    @printf("\nwrote %s (backup at %s.bak)\n", basename(target), basename(target))
else
    println("\ndry run -- pass --apply to write")
end
