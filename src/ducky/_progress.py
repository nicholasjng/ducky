"""Live progress bar for long-running queries.

See :func:`progress_bar`.
Renders via ``tqdm`` when installed; falls back to a minimal stderr bar otherwise (the import never hard-fails on missing tqdm).
"""

from __future__ import annotations

import contextlib
import sys
from collections.abc import Iterator
from typing import IO, TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from ._core import Connection

# Poll the progress snapshot this often (seconds) while a query is in flight.
_DEFAULT_INTERVAL = 0.1


class _Bar(Protocol):
    """Minimal surface both bar backends implement."""

    def update_to(self, pct: float) -> None: ...
    def close(self) -> None: ...


class _AsciiBar:
    """Fallback bar written to a text stream: `desc [####    ]  42.0%`."""

    def __init__(self, desc: str, out: IO[str], width: int = 40) -> None:
        self._desc = desc
        self._out = out
        self._width = width
        self._last = -1  # last rendered integer percent; skip redundant redraws

    def update_to(self, pct: float) -> None:
        # DuckDB reports -1.0 until it has an estimate; clamp into [0, 100].
        pct = 0.0 if pct < 0 else min(pct, 100.0)
        if int(pct) == self._last:
            return
        self._last = int(pct)
        filled = int(self._width * pct / 100.0)
        bar = "#" * filled + " " * (self._width - filled)
        self._out.write(f"\r{self._desc} [{bar}] {pct:5.1f}%")
        self._out.flush()

    def close(self) -> None:
        self.update_to(100.0)
        self._out.write("\n")
        self._out.flush()


class _TqdmBar:
    """tqdm-backed bar driven by absolute percentage (total=100)."""

    def __init__(self, tqdm_cls: type, desc: str, out: IO[str]) -> None:
        self._bar = tqdm_cls(total=100, desc=desc, file=out, unit="%")

    def update_to(self, pct: float) -> None:
        if pct < 0:  # no estimate yet — leave the bar at its current position
            return
        self._bar.n = min(pct, 100.0)
        self._bar.refresh()

    def close(self) -> None:
        self._bar.n = 100
        self._bar.refresh()
        self._bar.close()


def _make_bar(desc: str, out: IO[str], use_tqdm: bool) -> _Bar:
    if use_tqdm:
        try:
            from tqdm import tqdm
        except ImportError:
            pass
        else:
            return _TqdmBar(tqdm, desc, out)
    return _AsciiBar(desc, out)


def _bool_setting(con: Connection, name: str) -> bool:
    row = con.execute(f"SELECT current_setting('{name}')").fetchone()
    return bool(row[0]) if row is not None else False


def _set_bool(con: Connection, name: str, value: bool) -> None:
    con.execute(f"SET {name}={'true' if value else 'false'}")


@contextlib.contextmanager
def progress_bar(
    con: Connection,
    *,
    desc: str = "query",
    interval: float = _DEFAULT_INTERVAL,
    out: IO[str] | None = None,
    use_tqdm: bool = True,
) -> Iterator[Connection]:
    """Render a live progress bar for queries inside the block.

    Enables DuckDB progress tracking, polls :meth:`Connection.progress` from a background thread, and restores the connection's progress settings on exit.

    Parameters
    ----------
    con : Connection
        The connection whose queries should be tracked.
    desc : str, default 'query'
        Label shown to the left of the bar.
    interval : float, default 0.1
        Seconds between progress polls.
    out : file-like, optional
        Text stream to draw on.
        Defaults to ``sys.stderr``.
    use_tqdm : bool, default True
        Render through ``tqdm`` when installed; set ``False`` to force the stderr fallback.

    Yields
    ------
    Connection
        The same connection, so ``with progress_bar(con) as c:`` works.

    Examples
    --------
    >>> with ducky.progress_bar(con):
    ...     con.execute("SELECT count(*) FROM range(1_000_000_000) WHERE i % 7 = 0")
    """
    # Imported lazily so importing ducky doesn't pull in threading machinery.
    import threading

    out = sys.stderr if out is None else out

    # Snapshot the settings we touch so we can restore them afterwards.
    prev_enabled = _bool_setting(con, "enable_progress_bar")
    prev_print = _bool_setting(con, "enable_progress_bar_print")
    _set_bool(con, "enable_progress_bar", True)
    _set_bool(con, "enable_progress_bar_print", False)

    bar = _make_bar(desc, out, use_tqdm)
    stop = threading.Event()

    def _poll() -> None:
        while not stop.is_set():
            try:
                pct, _rows, _total = con.progress()
            except Exception:
                break
            bar.update_to(pct)
            stop.wait(interval)

    thread = threading.Thread(target=_poll, name="ducky-progress", daemon=True)
    thread.start()
    try:
        yield con
    finally:
        stop.set()
        thread.join()
        bar.close()
        _set_bool(con, "enable_progress_bar", prev_enabled)
        _set_bool(con, "enable_progress_bar_print", prev_print)
