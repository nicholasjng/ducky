"""Query profiling helpers."""

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
    """Profile captured by :func:`profile`; ``info`` holds the latest tree."""

    __slots__ = ("_con", "info")

    def __init__(self, con: Connection) -> None:
        self._con: Connection | None = con
        self.info: dict[str, Any] | None = None

    def refresh(self) -> dict[str, Any] | None:
        """Capture the latest query profile while inside the context."""
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
    if prev is None:
        con.execute(f"RESET {name}")
    else:
        con.execute(f"SET {name}='{prev}'")


@contextlib.contextmanager
def profile(con: Connection, *, mode: ProfilingMode = "standard") -> Iterator[ProfileResult]:
    """Profile queries in the block and restore prior settings on exit."""
    prev_enabled = _current_setting(con, "enable_profiling")
    prev_mode = _current_setting(con, "profiling_mode")
    con.execute("SET enable_profiling='no_output'")
    if mode == "detailed":
        con.execute("SET profiling_mode='detailed'")
    result = ProfileResult(con)
    try:
        yield result
    finally:
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
    """Render a profiling tree as a multi-line operator plan."""
    if info is None:
        return "(profiling not enabled — wrap queries in `with ducky.profile(con): ...`)"

    lines: list[str] = []

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

    op_tree = next((c for c in info["children"] if "type" in c["metrics"]), None)
    if op_tree is None:
        lines.append("(no operator subtree in profiling info)")
        return "\n".join(lines)

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
    """Write one JSON line per query, optionally with the full profile tree."""
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
    """Profile sink, sampling interval, and profiling mode."""

    sink: ProfileSink
    sample: int = 1
    mode: str = "standard"

    @classmethod
    def from_env(cls) -> ProfileConfig | None:
        """Build configuration from the ``DUCKY_PROFILE_*`` environment."""
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


_env_sink_lock = threading.Lock()
_env_sink_cache: dict[tuple[Path, bool], ProfileSink] = {}
