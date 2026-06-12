"""ducky vs duckdb-python: end-to-end runtime + memory comparison driver.

ducky and the upstream ``duckdb`` Python module both statically link DuckDB
and cannot coexist in one interpreter, so each engine runs in its own
``mew run`` subprocess. ``mew compare --key func`` then matches the suites by
function name (the file prefixes differ) and prints the side-by-side table,
including each file's context line — the bench modules record their engine
version via ``mew.set_context``, which also surfaces that the two engines
usually embed *different DuckDB versions* (ducky builds the ext/duckdb
submodule pin, the pip wheel is a release), so part of any gap can be the
engine rather than the bindings.

Usage:
    uv run --group bench python benchmarks/run_vs_duckdb.py
    uv run --group bench python benchmarks/run_vs_duckdb.py --no-memory
    uv run --group bench python benchmarks/run_vs_duckdb.py --tag ml
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

BENCH_DIR = Path(__file__).parent
DUCKY_FILE = BENCH_DIR / "bench_ducky.py"
DUCKDB_FILE = BENCH_DIR / "bench_duckdb.py"


def _run_mew(bench_file: Path, out_path: Path, *, profile_memory: bool, tag: str | None) -> None:
    cmd = ["mew", "run", str(bench_file), "-o", str(out_path)]
    if profile_memory:
        cmd.append("--profile-memory")
    if tag:
        cmd.extend(["-t", tag])
    label = "timing+memory" if profile_memory else "timing"
    print(f"\n--- {label}: {bench_file.name} ---", flush=True)
    subprocess.run(cmd, check=True)


def _compare(baseline: Path, head: Path, metric: str | None = None) -> int:
    cmd = ["mew", "compare", str(baseline), str(head), "--key", "func"]
    if metric:
        cmd.extend(["--metric", metric])
    return subprocess.run(cmd, check=False).returncode


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-memory", action="store_true", help="skip the memray memory pass")
    parser.add_argument(
        "-t", "--tag", default=None, help="filter to a single tag (select, insert, ml, udf, ...)"
    )
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        ducky_out = Path(tmp) / "ducky.jsonl"
        duckdb_out = Path(tmp) / "duckdb.jsonl"
        _run_mew(DUCKY_FILE, ducky_out, profile_memory=not args.no_memory, tag=args.tag)
        _run_mew(DUCKDB_FILE, duckdb_out, profile_memory=not args.no_memory, tag=args.tag)

        # duckdb is the baseline, so "speedup" reads as ducky-relative-to-duckdb.
        rc = _compare(duckdb_out, ducky_out)
        if not args.no_memory:
            for metric in ("memory.peak_bytes", "memory.total_allocations"):
                rc = _compare(duckdb_out, ducky_out, metric) or rc
    return rc


if __name__ == "__main__":
    sys.exit(main())
