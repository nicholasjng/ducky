"""Compare ducky and duckdb-python benchmarks in separate processes.

Each engine runs ``bench_engine.py`` in its own process, with
``DUCKY_BENCH_ENGINE`` picking the engine. The engines take turns for
``--rounds`` rounds, each appending to that engine's result file, so drift in
machine load spreads over both; ``mew compare`` then pools each file's rounds
and compares ducky against duckdb-python as the baseline.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

SUITE = Path(__file__).parent / "bench_engine.py"
ENGINES = ("duckdb", "ducky")  # baseline first
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
        help="samples per benchmark, engine and round",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=1,
        help="alternate the engines this many times; rounds pool as repetitions",
    )
    args = parser.parse_args()
    filter_args = (["-t", args.tag] if args.tag else []) + (
        ["-k", args.pattern] if args.pattern else []
    )
    run_args = ["--repetitions", str(args.repetitions)] if args.repetitions > 1 else []
    metrics = ("real_time", *MEMORY_METRICS) if args.memory else ("real_time",)

    with tempfile.TemporaryDirectory() as tmp:
        results = {name: Path(tmp) / f"{name}.jsonl" for name in ENGINES}
        for _ in range(args.rounds):
            for name in ENGINES:
                profile_args = []
                if args.memory:
                    profile_dir = Path(tmp) / f"profiles.{name}"
                    profile_args = [
                        "--memory-profiler",
                        "memray",
                        "--profile-dir",
                        str(profile_dir),
                    ]
                subprocess.run(
                    [
                        "mew",
                        "run",
                        str(SUITE),
                        *run_args,
                        *profile_args,
                        *filter_args,
                        "--append",
                        "-o",
                        str(results[name]),
                    ],
                    check=True,
                    env={**os.environ, "DUCKY_BENCH_ENGINE": name},
                )
        rc = 0
        for metric in metrics:
            compare = ["mew", "compare", "--metric", metric, *map(str, results.values())]
            rc = subprocess.run(compare, check=False).returncode or rc
        return rc


if __name__ == "__main__":
    sys.exit(main())
