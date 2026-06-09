"""Query-profiling helpers.

Three layers on top of :meth:`Connection.get_profiling_info`:

* :func:`profile` — context manager that toggles DuckDB's profiling settings
  and snapshots the resulting tree.
* :func:`format_profiling_info` — pretty-printer for the nested dict.
* :class:`ProfileConfig` + :func:`jsonl_profile_sink` — always-on sinks for
  training loops / dataset queries the call site doesn't own. Wired up
  programmatically or via the ``DUCKY_PROFILE_*`` environment variables.
"""

from __future__ import annotations

import contextlib
import json
import os
import threading
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any, Literal, NamedTuple

if TYPE_CHECKING:
    from ._core import Connection

ProfilingMode = Literal["standard", "detailed"]

ProfileSink = Callable[[str, dict[str, Any]], None]


class ProfileResult:
    """Container yielded by :func:`profile`.

    On context-manager exit the most recent profiling tree is snapshotted into ``.info`` (just before settings are restored), and the connection reference is dropped so a stashed result doesn't pin the connection.
    ``str(p)`` renders via :func:`format_profiling_info`.

    Attributes
    ----------
    info : dict | None
        The profiling tree from the last query in the block, or ``None`` if nothing was captured.
    """

    __slots__ = ("_con", "info")

    def __init__(self, con: Connection) -> None:
        self._con: Connection | None = con
        self.info: dict[str, Any] | None = None

    def refresh(self) -> dict[str, Any] | None:
        """Re-read the profiling tree from the connection.

        Useful inside the ``with`` block to capture snapshots between queries — the connection only retains the *most recent* query's profile.
        Raises ``RuntimeError`` outside the block.
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
    # DuckDB profiling settings are enum/mode strings with no quotes — safe to
    # interpolate directly.
    if prev is None:
        con.execute(f"RESET {name}")
    else:
        con.execute(f"SET {name}='{prev}'")


@contextlib.contextmanager
def profile(con: Connection, *, mode: ProfilingMode = "standard") -> Iterator[ProfileResult]:
    """Enable DuckDB query profiling for the duration of the block.

    On exit, the most recent query's profiling tree is snapshotted into the yielded :class:`ProfileResult` and DuckDB's settings are restored.
    Use ``p.refresh()`` inside the block to capture intermediate snapshots.

    Parameters
    ----------
    con : Connection
        The connection to enable profiling on.
    mode : {'standard', 'detailed'}, default 'standard'
        ``'detailed'`` adds extra per-operator counters (``cpu_time``, …).

    Examples
    --------
    >>> with ducky.profile(con) as p:
    ...     con.execute("SELECT count(*) FROM range(1_000_000)").fetchall()
    >>> print(p)
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
        # Snapshot before restoring settings: the connection's stored profile
        # is cleared as soon as enable_profiling goes back to its prior value.
        with contextlib.suppress(Exception):
            result.refresh()
        result._con = None
        _restore(con, "profiling_mode", prev_mode)
        _restore(con, "enable_profiling", prev_enabled)


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
    """Render a profiling tree as a multi-line operator-plan view.

    Renders the SQL + total/CPU summary (if present) followed by the operator subtree, one operator per line with aligned ``time=`` / ``rows=`` columns and a continuation line for each operator's ``extra_info``.
    Phase and optimizer-rule timings are intentionally omitted.

    Parameters
    ----------
    info : dict or None
        The dict returned by :meth:`Connection.get_profiling_info`.
        ``None`` produces a single hint line instead of raising.
    extra_info_width : int, default 80
        Maximum characters of an operator's ``extra_info`` to print before clipping.
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


def jsonl_profile_sink(
    path: str | os.PathLike[str],
    *,
    append: bool = True,
    with_info: bool = True,
) -> ProfileSink:
    """Return a profile sink that appends one JSON line per query to ``path``.

    Each record is ``{"ts": <UTC ISO-8601>, "sql": ..., "info": {...}}`` (or ``"summary": {...}`` when ``with_info=False``).
    The file is line-buffered so writes survive a crash; writes are lock-serialized across threads.

    Parameters
    ----------
    path : str or path-like
        Output file.
    append : bool, default True
        Open in append mode; pass ``False`` to truncate.
    with_info : bool, default True
        Include the full profiling tree.
        Set ``False`` to write only the SQL plus a compact ``{total_time, cpu_time, total_intermediate_rows}`` summary.
    """
    # Long-lived handle, owned by the sink closure (closed on process exit).
    f: IO[str] = Path(path).open("a" if append else "w", buffering=1)  # noqa: SIM115
    lock = threading.Lock()

    def _summary(info: dict[str, Any]) -> dict[str, Any]:
        for child in info.get("children", []):
            if "sql" in child["metrics"]:
                m = child["metrics"]
                return {
                    k: m[k] for k in ("total_time", "cpu_time", "total_intermediate_rows") if k in m
                }
        return {}

    def sink(sql: str, info: dict[str, Any]) -> None:
        record: dict[str, Any] = {
            "ts": datetime.now(UTC).isoformat(timespec="microseconds"),
            "sql": sql,
        }
        if with_info:
            record["info"] = info
        else:
            record["summary"] = _summary(info)
        line = json.dumps(record, default=str)
        with lock:
            f.write(line + "\n")

    return sink


class ProfileConfig(NamedTuple):
    """Configuration bundle for an always-on profile sink.

    Holds the sink callable plus the ``sample`` and ``mode`` arguments :meth:`Connection.set_profile_sink` accepts.
    Pure data — construct directly for a programmatic config or call :meth:`from_env`.

    Examples
    --------
    >>> cfg = ducky.ProfileConfig(my_sink, sample=10, mode="detailed")
    >>> con.set_profile_sink(cfg.sink, sample=cfg.sample, mode=cfg.mode)
    """

    sink: ProfileSink
    sample: int = 1
    mode: str = "standard"

    @classmethod
    def from_env(cls) -> ProfileConfig | None:
        """Build a config from the ``DUCKY_PROFILE_*`` environment.

        Returns ``None`` when ``DUCKY_PROFILE_DIR`` is unset; otherwise a :class:`ProfileConfig` whose sink writes to ``{DUCKY_PROFILE_DIR}/profile-{pid}.jsonl`` (one file per process, shared across connections; lock-serialized).

        Environment Variables
        ---------------------
        ``DUCKY_PROFILE_DIR``
            Output directory (created if missing); required.
        ``DUCKY_PROFILE_MODE``
            ``'standard'`` (default) or ``'detailed'``.
        ``DUCKY_PROFILE_SAMPLE``
            Integer N; sink fires every Nth query.
            Default 1.
        ``DUCKY_PROFILE_NO_INFO``
            Any non-empty value writes a compact ``summary`` instead of the full tree.
        """
        directory_env = os.environ.get("DUCKY_PROFILE_DIR")
        if not directory_env:
            return None
        directory = Path(directory_env)
        with_info = not os.environ.get("DUCKY_PROFILE_NO_INFO")
        key = (directory, with_info)
        with _env_sink_lock:
            sink = _env_sink_cache.get(key)
            if sink is None:
                directory.mkdir(parents=True, exist_ok=True)
                path = directory / f"profile-{os.getpid()}.jsonl"
                sink = jsonl_profile_sink(path, with_info=with_info)
                _env_sink_cache[key] = sink
        return cls(
            sink=sink,
            sample=int(os.environ.get("DUCKY_PROFILE_SAMPLE", "1")),
            mode=os.environ.get("DUCKY_PROFILE_MODE", "standard"),
        )


# Process-wide cache so every connection appends to the same file.
_env_sink_lock = threading.Lock()
_env_sink_cache: dict[tuple[Path, bool], ProfileSink] = {}
