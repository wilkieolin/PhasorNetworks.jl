#!/bin/bash
# Fetch the extended-severity shard CSVs from the companion machine.
#
# spark2 runs its shards out of ~/code/PhasorNetworks-liep (a separate checkout
# pinned to the same commit, so appended rows carry the right gitrev). This
# copies their per-cell CSVs into results/e2_extend_spark2/, where
# scripts/e2_merge_extension.jl picks them up.
#
# Safe to run while the shards are still going -- each cell's CSV is appended
# row by row, so a partial fetch simply has fewer reps.
#
#   scripts/e2_fetch_spark2.sh && julia --project=. scripts/e2_merge_extension.jl
set -euo pipefail
REMOTE=${1:-spark2}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
DEST="$ROOT/results/e2_extend_spark2"
mkdir -p "$DEST"
rsync -az --include='*/' --include='*.csv' --exclude='*' \
      "$REMOTE:~/code/PhasorNetworks-liep/results/e2_extend/" "$DEST/"

# E8 baseline arms run on the companion machine land here too
mkdir -p "$ROOT/results/e8_baselines_spark2"
rsync -az --include='*.csv' --exclude='*' \
      "$REMOTE:~/code/PhasorNetworks-liep/results/e8_baselines/" \
      "$ROOT/results/e8_baselines_spark2/" 2>/dev/null || true

# point-B lock-in shards
mkdir -p "$ROOT/results/e6_basin_training"
rsync -az --include='*pointB*.csv' --exclude='*' \
      "$REMOTE:~/code/PhasorNetworks-liep/results/e6_basin_training/" \
      "$ROOT/results/e6_basin_training/" 2>/dev/null || true
echo "fetched into results/e2_extend_spark2:"
for d in "$DEST"/*/; do
  f=$(ls "$d"*.csv 2>/dev/null | head -1) || continue
  [ -n "$f" ] && printf "  %-20s %d reps\n" "$(basename "$d")" "$(( $(wc -l < "$f") - 1 ))"
done
