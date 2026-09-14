#!/bin/sh
#
# Profile selected benchmarks with native frames.
# Usage: scripts/profile-gaps.sh [ducky|duckdb]
# Output: .mew-traces/<engine>/<name>/
set -eu

engine="${1:-ducky}"
case "$engine" in
    ducky | duckdb) ;;
    *)
        echo "usage: $0 [ducky|duckdb]" >&2
        exit 2
        ;;
esac

bench="benchmarks/bench_${engine}.py"
outroot=".mew-traces/${engine}"

# name | pattern | iterations
gaps='
connect|^bench_connect$|2500
executemany|bench_executemany\[n=100000\]|2
arrow_reader|bench_select_arrow_reader\[n=100000\]|35000
iter_batches|bench_select_iter_batches\[n=1000000\]|2500
'

echo "$gaps" | while IFS='|' read -r name pattern iters; do
    [ -z "$name" ] && continue
    echo ">> $engine :: $name  (-k '$pattern', --iterations $iters)"
    uv run --group bench mew profile "$bench" \
        -k "$pattern" --iterations "$iters" --time-limit 60s \
        -o "$outroot/$name" ||
        echo "   skipped/failed: $name" >&2
done

echo "Done. Traces under $outroot/ (open in Instruments, or re-run with --open)."
