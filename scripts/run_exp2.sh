#!/usr/bin/env bash
# Self-healing, process-chunked launcher for Exp 2 (LSA/LCA residual trained sweep).
#
# Why chunking: a single long-lived Julia process accumulates memory across
# configs (leak/fragmentation) and stalls into GC-thrash after ~100+ configs.
# Each invocation here does at most CHUNK new configs then exits cleanly; the
# loop spawns a fresh process (full memory reset) and resume=true continues.
#
# Memory failsafes:
#   - JULIA_HARD_MEMORY_LIMIT: Julia throws a catchable OOM instead of OS SIGKILL.
#   - --heap-size-hint=48G: GC targets ~48 GB, collecting transient garbage early.

set -u
export JULIA_HARD_MEMORY_LIMIT=100000000000   # ~100 GB (box has ~127 GB)

OUT=results/lsa_lca_residual/exp2_final
SUM="$OUT/sweep_summary.csv"
GAP="$OUT/spiking_gap.csv"
CHUNK=20                                        # new configs per fresh process
EXPECTED=181                                    # header + 3 kinds × 4 treat × 5 depth × 3 seed
GAP_EXPECTED=4                                  # header + 3 spiking-gap rows

for i in $(seq 1 25); do
    sl=0; [ -f "$SUM" ] && sl=$(wc -l < "$SUM")
    gl=0; [ -f "$GAP" ] && gl=$(wc -l < "$GAP")
    if [ "$sl" -ge "$EXPECTED" ] && [ "$gl" -ge "$GAP_EXPECTED" ]; then
        echo "=== all done: $((sl-1)) configs, $((gl-1)) spiking rows ==="
        break
    fi
    echo "=== chunk $i: $((sl>0?sl-1:0)) configs done, spawning fresh process ==="
    julia --heap-size-hint=48G --project=. -e "include(\"scripts/lsa_lca_residual_sweep.jl\"); main_lsa_lca_sweep(; depths=(1,2,4,8,16), seeds=1:3, epochs=8, D=64, batchsize=32, checkpoint=true, init_probe=false, use_cuda=true, spiking_depth=8, best_treatment=:rezero, resume=true, max_configs=$CHUNK, outdir=\"$OUT\")" \
        || echo "=== chunk $i exited non-zero; will resume ==="
done
echo "=== DRIVER DONE ==="
