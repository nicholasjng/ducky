"""Query-profiling helpers: a context manager that toggles DuckDB's profiling
settings, and a pretty-printer for the nested dict returned by
:meth:`Connection.get_profiling_info`.

`Connection.get_profiling_info` is the raw, programmatic access — but it only
returns anything once `enable_profiling` is set, and the dict is dense
(operator subtree, memory aggregate, IO aggregate, optimizer-rule timings,
parser/binder/planner phases, query summary). This module wraps both halves:

    >>> import ducky
    >>> con = ducky.connect()
    >>> with ducky.profile(con) as p:
    ...     con.execute("SELECT count(*) FROM range(1_000_000)").fetchall()
    >>> print(p)  # or format_profiling_info(p.info) for the same string
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from ._core import Connection

ProfilingMode = Literal["standard", "detailed"]


class ProfileResult:
    """Container yielded by :func:`profile`.

    The most recent profiling tree is snapshotted on context-manager exit (just
    before the connection's profiling settings are restored), so callers don't
    have to remember to call ``con.get_profiling_info()`` inside the block.

    Stringifying the result renders it via :func:`format_profiling_info`, so
    ``print(p)`` after the block is the one-liner display.

    Once the ``with`` block exits, the result drops its reference to the
    connection — the dict in ``.info`` is the only thing it pins, so a
    long-lived ``ProfileResult`` won't keep a transient connection open.
    """

    __slots__ = ("_con", "info")

    def __init__(self, con: Connection) -> None:
        self._con: Connection | None = con
        self.info: dict[str, Any] | None = None

    def refresh(self) -> dict[str, Any] | None:
        """Re-read the profiling tree from the connection right now.

        Useful inside the ``with`` block to grab snapshots between queries (the
        connection only retains the *most recent* query's profile). On exit the
        context manager calls this once more so ``.info`` reflects the last
        query that ran, then releases the connection reference; calling
        ``refresh()`` after the block raises.
        """
        if self._con is None:
            raise RuntimeError(
                "ProfileResult.refresh() can only be called inside the "
                "`with ducky.profile(con):` block"
            )
        self.info = self._con.get_profiling_info()
        return self.info

    def __str__(self) -> str:
        return format_profiling_info(self.info)

    def __repr__(self) -> str:
        return f"ProfileResult(info={'set' if self.info is not None else 'None'})"


def _current_setting(con: Connection, name: str) -> str | None:
    return con.execute(f"SELECT current_setting('{name}')").fetchitem()


def _restore(con: Connection, name: str, prev: str | None) -> None:
    # Unset → RESET back to default; otherwise re-apply the original literal
    # (single-quoted; DuckDB profiling settings are simple enum / mode strings
    # with no embedded quotes, so no escaping is needed).
    if prev is None:
        con.execute(f"RESET {name}")
    else:
        con.execute(f"SET {name}='{prev}'")


@contextlib.contextmanager
def profile(con: Connection, *, mode: ProfilingMode = "standard") -> Iterator[ProfileResult]:
    """Enable DuckDB query profiling for the duration of the block.

    Sets ``enable_profiling='no_output'`` (so DuckDB collects a profiling tree
    without also printing one) and, when ``mode='detailed'``, also sets
    ``profiling_mode='detailed'`` to surface per-operator counters like
    ``cpu_time``. Both settings are restored to whatever they were before the
    block on exit, even if the block raises.

    The yielded :class:`ProfileResult` snapshots ``con.get_profiling_info()``
    just before the settings are restored, so the most recent query's
    profiling tree survives the context::

        with ducky.profile(con) as p:
            con.execute("...").fetchall()
        print(p)  # uses format_profiling_info under the hood

    Use ``p.refresh()`` inside the block to capture intermediate snapshots if
    you run several queries (the connection only retains the *latest* query's
    profile). The ``ProfileResult`` does not pin the connection beyond the
    block — its only field after exit is the (already-detached) ``.info`` dict.

    Parameters
    ----------
    con:
        The connection to enable profiling on.
    mode:
        ``'standard'`` (default) collects DuckDB's standard metric set;
        ``'detailed'`` adds extra per-operator counters.
    """
    prev_enabled = _current_setting(con, "enable_profiling")
    prev_mode = _current_setting(con, "profiling_mode")
    con.execute("SET enable_profiling='no_output'")
    if mode == "detailed":
        con.execute("SET profiling_mode='detailed'")
    result = ProfileResult(con)
    try:
        yield result
    finally:
        # Snapshot the profiling tree *before* restoring settings — the
        # connection's stored profile is cleared as soon as enable_profiling
        # goes back to its prior value. Then drop the connection reference so
        # holding on to the result doesn't keep the connection alive.
        # If the block left the connection unreadable, .info stays at its last
        # value; restoring settings still has to happen.
        with contextlib.suppress(Exception):
            result.refresh()
        result._con = None
        _restore(con, "profiling_mode", prev_mode)
        _restore(con, "enable_profiling", prev_enabled)


# ── Pretty-printing ──────────────────────────────────────────────────────────


def _fmt_duration(seconds: float) -> str:
    if seconds < 1e-6:
        return f"{seconds * 1e9:.0f}ns"
    if seconds < 1e-3:
        return f"{seconds * 1e6:.1f}µs"
    if seconds < 1:
        return f"{seconds * 1e3:.2f}ms"
    return f"{seconds:.2f}s"


def _try_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _try_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def format_profiling_info(
    info: dict[str, Any] | None,
    *,
    extra_info_width: int = 80,
) -> str:
    """Render the profiling tree from :meth:`Connection.get_profiling_info`.

    Returns a multi-line string with three sections, in order:

    * a SQL header (truncated to one line) plus total / CPU time and the
      total number of intermediate rows, if the query-summary node is
      present;
    * the operator subtree, one operator per line, with ``time=`` and
      ``rows=`` aligned and the operator's ``extra_info`` (table name,
      projection list, group keys, etc.) wrapped on a continuation line;
    * nothing else — phase / optimizer-rule timings live in sibling
      subtrees of the raw dict and are intentionally left out of the
      default rendering to keep the output focused on the plan.

    Passing ``None`` (the value returned when ``enable_profiling`` isn't
    set) produces a single hint line instead of raising.
    """
    if info is None:
        return "(profiling not enabled — wrap queries in `with ducky.profile(con): ...`)"

    lines: list[str] = []

    # Query summary lives in a sibling node identified by its `sql` metric.
    summary = next((c for c in info["children"] if "sql" in c["metrics"]), None)
    if summary is not None:
        s = summary["metrics"]
        sql = " ".join(s.get("sql", "").split())
        if sql:
            if len(sql) > 100:
                sql = sql[:97] + "..."
            lines.append(f"SQL: {sql}")
        bits: list[str] = []
        total = _try_float(s.get("total_time"))
        if total is not None:
            bits.append(f"total={_fmt_duration(total)}")
        cpu = _try_float(s.get("cpu_time"))
        if cpu is not None:
            bits.append(f"cpu={_fmt_duration(cpu)}")
        irows = _try_int(s.get("total_intermediate_rows"))
        if irows is not None:
            bits.append(f"intermediate_rows={irows:,}")
        if bits:
            lines.append("  ".join(bits))
        lines.append("")

    # Operator subtree: the first (and only) child whose own metrics carry a `type`.
    op_tree = next((c for c in info["children"] if "type" in c["metrics"]), None)
    if op_tree is None:
        lines.append("(no operator subtree in profiling info)")
        return "\n".join(lines)

    # Two-pass render so the timing/rows columns line up.
    rows: list[tuple[str, str | None, str | None, str | None]] = []

    def walk(node: dict[str, Any], depth: int) -> None:
        m = node["metrics"]
        label = "  " * depth + m.get("type", "?")
        rows.append((label, m.get("timing"), m.get("intermediate_rows"), m.get("extra_info")))
        for child in node["children"]:
            walk(child, depth + 1)

    walk(op_tree, 0)

    name_width = max(len(r[0]) for r in rows)
    for label, timing, irows_s, extra in rows:
        parts = [label.ljust(name_width)]
        tf = _try_float(timing)
        parts.append(f"  time={_fmt_duration(tf):>8s}" if tf is not None else " " * 14)
        ni = _try_int(irows_s)
        parts.append(f"  rows={ni:>10,}" if ni is not None else "")
        lines.append("".join(parts).rstrip())
        # `extra_info` arrives as a `{Key=value, Key=value, ...}` string with
        # embedded newlines inside multi-value entries. Collapse and clip.
        if extra and extra not in ("{}", ""):
            collapsed = " ".join(extra.split())
            if collapsed.startswith("{") and collapsed.endswith("}"):
                collapsed = collapsed[1:-1]
            if len(collapsed) > extra_info_width:
                collapsed = collapsed[: extra_info_width - 3] + "..."
            lines.append(" " * (name_width + 4) + collapsed)

    return "\n".join(lines)
