"""Compare ducky and duckdb-python benchmarks in separate processes."""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

BENCH_DIR = Path(__file__).parent
SUITES = {"duckdb": BENCH_DIR / "bench_duckdb.py", "ducky": BENCH_DIR / "bench_ducky.py"}
MEMORY_METRICS = ("memory.peak_bytes", "memory.allocations_per_iteration")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--memory", action="store_true", help="add a memray pass per engine")
    parser.add_argument(
        "-t", "--tag", default=None, help="filter to a tag (select, insert, ml, udf, ...)"
    )
    parser.add_argument(
        "-k",
        "--pattern",
        default=None,
        help="run benchmarks whose name matches this regex",
    )
    parser.add_argument(
        "-r",
        "--repetitions",
        type=int,
        default=1,
        help="samples per benchmark and engine; >1 enables stddev columns",
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
        results: dict[str, Path] = {}
        for name, suite in SUITES.items():
            out = Path(tmp) / f"{name}.json"
            subprocess.run(
                ["mew", "run", str(suite), *run_args, "-o", str(out), *filter_args],
                check=True,
            )
            results[name] = out
        rc = 0
        for metric in metrics:
            rc = (
                subprocess.run(
                    [
                        "mew",
                        "compare",
                        str(results["ducky"]),
                        str(results["duckdb"]),
                        "--key",
                        "func",
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
