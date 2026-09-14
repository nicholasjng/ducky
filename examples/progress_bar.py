"""Show query progress with tqdm or the stderr fallback."""

from __future__ import annotations

import ducky


def main() -> None:
    # threads >= 2 lets DuckDB's scheduler report meaningful progress.
    con = ducky.connect(threads=4)

    # Everything inside the block is tracked; the bar finalizes to 100% on exit.
    with ducky.progress_bar(con, desc="counting"):
        row = con.execute(
            "SELECT count(*) FROM range(2_000_000_000) t(i) WHERE i % 7 = 0"
        ).fetchone()
    # A COUNT(*) always yields exactly one row; assert narrows tuple | None.
    assert row is not None
    (count,) = row

    print(f"matched {count:,} rows")

    # Force the stderr fallback bar (no tqdm) for comparison.
    with ducky.progress_bar(con, desc="counting", use_tqdm=False):
        con.execute("SELECT count(*) FROM range(2_000_000_000) t(i) WHERE i % 11 = 0").fetchall()


if __name__ == "__main__":
    main()
