#!/bin/sh
#
# Record native-frame profiles for the benchmarks where ducky trails duckdb,
# one isolated case per benchmark. Uses `mew profile` (xctrace on macOS,
# py-spy/perf on Linux) — the out-of-process native profiler, not `mew run
# --sample` (which is Python-only).
#
# Usage:
#   scripts/profile-gaps.sh [ducky|duckdb]
#
#   engine   Which bench file to profile (default: ducky). Run with `duckdb`
#            too for the A/B: same case on both engines, diff the native frames
#            to see *why* ducky's path costs more (e.g. connect, executemany).
#
# Artifacts land in .mew-traces/<engine>/<name>/. Open in Instruments (macOS)
# or pass --open by editing the command below.
#
# --iterations is sized per case so the ~1 kHz sampler gets enough hits — slow
# bodies (executemany ~12 s/iter) need a handful, µs-fast ones need many.
#
# IMPORTANT: an out-of-process native sampler records the *whole* process; it
# cannot honour mew's in-process state.pause(). So per-iteration setup shows up
# in the trace. The cases below are chosen so that's a non-issue — their setup
# is either one-time (select_* call _populate() once before the loop, amortized
# over thousands of iters) or dwarfed by the body (executemany's body is ~12 s
# vs ~15 ms setup). bench_bulk_insert_arrow is deliberately omitted: its body
# (~18 ms) is per-iteration ducky.connect() (~6 ms) plus the INSERT, so its
# trace is ~30% connect frames. Profile that one with a standalone driver that
# hoists connect()/register_arrow() out of the loop, or read only the INSERT
# subtree in the viewer.
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

# name | -k regex (pins one parametrized case) | --iterations (~15-25 s of samples)
# Only cases whose body dominates the native trace (see the setup note above).
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
