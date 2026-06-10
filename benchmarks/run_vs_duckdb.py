"""ducky vs duckdb-python: end-to-end runtime + memory comparison driver.

ducky and the upstream ``duckdb`` Python module both statically link DuckDB
and cannot coexist in one interpreter, so each engine runs in its own
``mew run`` subprocess. We then merge the two JSONL streams and print a
side-by-side table.

Usage:
    uv run --group bench python benchmarks/run_vs_duckdb.py
    uv run --group bench python benchmarks/run_vs_duckdb.py --no-memory
    uv run --group bench python benchmarks/run_vs_duckdb.py --tag ml
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

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


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            # The first row is the context; runs have a `name` field.
            if "name" in d:
                rows.append(d)
    return rows


def _key(bench: dict[str, Any]) -> tuple[str, str]:
    name = bench["name"].split("::")[-1].split("/")[0]
    if name.startswith("bench_"):
        name = name[len("bench_") :]
    return name, bench["label"]


def _fmt_time(ns: float) -> str:
    if ns < 1_000:
        return f"{ns:.0f} ns"
    if ns < 1_000_000:
        return f"{ns / 1_000:.2f} µs"
    if ns < 1_000_000_000:
        return f"{ns / 1_000_000:.2f} ms"
    return f"{ns / 1_000_000_000:.2f} s"


def _fmt_bytes(n: int | None) -> str:
    if n is None:
        return "—"
    if n < 1024:
        return f"{n} B"
    if n < 1024**2:
        return f"{n / 1024:.1f} KiB"
    if n < 1024**3:
        return f"{n / 1024**2:.1f} MiB"
    return f"{n / 1024**3:.2f} GiB"


def _print_runtime_table(ducky_runs: list[dict], duckdb_runs: list[dict]) -> None:
    dby = {_key(b): b for b in ducky_runs if not b.get("skipped")}
    uby = {_key(b): b for b in duckdb_runs if not b.get("skipped")}
    keys = sorted(set(dby) | set(uby))
    print("\n=== Runtime (Google Benchmark, real_time) ===\n")
    header = f"{'workload':<22} {'label':<12} {'ducky':>12} {'duckdb':>12} {'speedup':>10}"
    print(header)
    print("-" * len(header))
    for k in keys:
        d, u = dby.get(k), uby.get(k)
        dt = d["real_time"] if d else None
        ut = u["real_time"] if u else None
        speedup = (ut / dt) if (dt and ut) else float("nan")
        print(
            f"{k[0]:<22} {k[1]:<12} "
            f"{(_fmt_time(dt) if dt else '—'):>12} "
            f"{(_fmt_time(ut) if ut else '—'):>12} "
            f"{(f'{speedup:.2f}×' if speedup == speedup else '—'):>10}"
        )


def _print_memory_table(ducky_runs: list[dict], duckdb_runs: list[dict]) -> None:
    dby = {_key(b): b for b in ducky_runs if b.get("memory")}
    uby = {_key(b): b for b in duckdb_runs if b.get("memory")}
    keys = sorted(set(dby) | set(uby))
    if not keys:
        return
    print("\n=== Memory (memray peak heap + total allocations) ===\n")
    header = (
        f"{'workload':<22} {'label':<12} "
        f"{'ducky peak':>12} {'ducky allocs':>14} "
        f"{'duckdb peak':>12} {'duckdb allocs':>14} "
        f"{'peak Δ':>9} {'alloc Δ':>9}"
    )
    print(header)
    print("-" * len(header))
    for k in keys:
        dm = (dby.get(k) or {}).get("memory") or {}
        um = (uby.get(k) or {}).get("memory") or {}
        dp, up = dm.get("peak_bytes"), um.get("peak_bytes")
        da, ua = dm.get("total_allocations"), um.get("total_allocations")
        pr = (dp / up) if (dp and up) else float("nan")
        ar = (da / ua) if (da and ua) else float("nan")
        print(
            f"{k[0]:<22} {k[1]:<12} "
            f"{_fmt_bytes(dp):>12} "
            f"{(f'{da:,}' if da is not None else '—'):>14} "
            f"{_fmt_bytes(up):>12} "
            f"{(f'{ua:,}' if ua is not None else '—'):>14} "
            f"{(f'{pr:.2f}×' if pr == pr else '—'):>9} "
            f"{(f'{ar:.2f}×' if ar == ar else '—'):>9}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-memory", action="store_true",
                        help="skip the memray memory pass")
    parser.add_argument("-t", "--tag", default=None,
                        help="filter to a single tag (select, insert, ml, ...)")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        ducky_out = Path(tmp) / "ducky.jsonl"
        duckdb_out = Path(tmp) / "duckdb.jsonl"
        _run_mew(DUCKY_FILE, ducky_out, profile_memory=not args.no_memory, tag=args.tag)
        _run_mew(DUCKDB_FILE, duckdb_out, profile_memory=not args.no_memory, tag=args.tag)
        ducky_runs = _load_jsonl(ducky_out)
        duckdb_runs = _load_jsonl(duckdb_out)

    _print_runtime_table(ducky_runs, duckdb_runs)
    if not args.no_memory:
        _print_memory_table(ducky_runs, duckdb_runs)


if __name__ == "__main__":
    sys.exit(main())
