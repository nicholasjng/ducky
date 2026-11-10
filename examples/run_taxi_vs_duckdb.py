# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Compare ducky and duckdb-python on NYC Yellow Taxi data."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

EXAMPLES_DIR = Path(__file__).parent
SCRIPTS = {
    "ducky": EXAMPLES_DIR / "taxi_ducky.py",
    "duckdb-python": EXAMPLES_DIR / "taxi_duckdb.py",
}
_RESULT_RE = re.compile(
    r"^RESULT engine=(?P<engine>\S+) n=(?P<n>\d+) load_s=(?P<load_s>[\d.]+) "
    r"train_s=(?P<train_s>[\d.]+) val_mse=(?P<val_mse>[\d.]+) val_mae=(?P<val_mae>[\d.]+)"
)


def _run(name: str, script: Path, months: list[str]) -> dict[str, str]:
    proc = subprocess.run(
        ["uv", "run", str(script), *months],
        capture_output=True,
        text=True,
        check=False,
        cwd=EXAMPLES_DIR,
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        raise RuntimeError(f"{name} ({script.name}) exited with code {proc.returncode}")
    for line in proc.stdout.splitlines():
        if m := _RESULT_RE.match(line):
            return m.groupdict()
    sys.stderr.write(proc.stdout)
    raise RuntimeError(f"{name} ({script.name}) printed no RESULT line")


def main() -> None:
    months = sys.argv[1:] if len(sys.argv) > 1 else ["2023-01"]
    print(
        f"Loading + training on {', '.join(months)}, once per engine (separate subprocesses)...\n"
    )

    results = {name: _run(name, script, months) for name, script in SCRIPTS.items()}

    header = (
        f"{'engine':<15}{'rows':>10}{'load (s)':>10}{'train (s)':>11}{'total (s)':>11}"
        f"{'val MAE (min)':>15}"
    )
    print(header)
    print("-" * len(header))
    for name, r in results.items():
        load_s, train_s = float(r["load_s"]), float(r["train_s"])
        print(
            f"{name:<15}{int(r['n']):>10,}{load_s:>10.3f}{train_s:>11.3f}"
            f"{load_s + train_s:>11.3f}{float(r['val_mae']):>15.3f}"
        )

    ducky_load = float(results["ducky"]["load_s"])
    other_load = float(results["duckdb-python"]["load_s"])
    faster = "ducky" if ducky_load < other_load else "duckdb-python"
    ratio = max(ducky_load, other_load) / min(ducky_load, other_load)
    print(f"\nData loading: {faster} was {ratio:.2f}x faster on this run.")
    print(
        "(Both engines reach equivalent val MAE; training itself is identical code — "
        "see _taxi_common.py.)"
    )


if __name__ == "__main__":
    main()
