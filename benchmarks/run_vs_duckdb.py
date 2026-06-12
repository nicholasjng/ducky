"""ducky vs duckdb-python: side-by-side runtime + memory comparison.

ducky and the upstream ``duckdb`` module both statically link DuckDB and can't
share an interpreter, so each runs in its own subprocess. mew's variant runner
(`mew run --variant`) does this directly — interleaving repetitions so drift
doesn't bias one engine, forwarding the profilers to each child, and tagging
each row with its variant and per-engine context. `mew compare --by variant`
then pivots the single result file (it matches on the function name across
variants automatically) and annotates each column with its engine version, so
the (usually different) embedded DuckDB builds are visible.

Usage:
    uv run --group bench python benchmarks/run_vs_duckdb.py
    uv run --group bench python benchmarks/run_vs_duckdb.py --memory
    uv run --group bench python benchmarks/run_vs_duckdb.py --tag ml
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

BENCH_DIR = Path(__file__).parent
# duckdb first so it's the comparison baseline (speedup reads ducky-vs-duckdb).
VARIANTS = {"duckdb": BENCH_DIR / "bench_duckdb.py", "ducky": BENCH_DIR / "bench_ducky.py"}
# allocations_per_iteration is the per-call, cross-engine-comparable count;
# peak_bytes is an iteration-independent high-water mark. total_allocations is
# deliberately omitted — it's a single-pass count dominated by first-call costs.
MEMORY_METRICS = ("memory.peak_bytes", "memory.allocations_per_iteration")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--memory", action="store_true", help="add a memray pass (per variant)")
    parser.add_argument(
        "-t", "--tag", default=None, help="filter to a tag (select, insert, ml, udf, ...)"
    )
    parser.add_argument(
        "-k",
        "--pattern",
        default=None,
        help="filter to benchmarks whose name contains this substring",
    )
    parser.add_argument(
        "-r",
        "--repetitions",
        type=int,
        default=1,
        help="process-level samples per (benchmark x variant), interleaved across variants; "
        ">1 surfaces per-engine stddev so outliers are visible",
    )
    args = parser.parse_args()
    filter_args = (["-t", args.tag] if args.tag else []) + (
        ["-k", args.pattern] if args.pattern else []
    )
    run_args = ["--profile-memory"] if args.memory else []
    if args.repetitions > 1:
        run_args += ["--repetitions", str(args.repetitions)]
    metrics = ("real_time", *MEMORY_METRICS) if args.memory else ("real_time",)

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "vs.jsonl"
        subprocess.run(
            [
                "mew",
                "run",
                *(f"--variant={name}={path}" for name, path in VARIANTS.items()),
                *run_args,
                "-o",
                str(out),
                *filter_args,
            ],
            check=True,
        )
        rc = 0
        for metric in metrics:
            rc = (
                subprocess.run(
                    [
                        "mew",
                        "compare",
                        str(out),
                        "--by",
                        "variant",
                        "--baseline",
                        "duckdb",
                        "--metric",
                        metric,
                        "--stddev",
                    ],
                    check=False,
                ).returncode
                or rc
            )
        return rc


if __name__ == "__main__":
    sys.exit(main())
