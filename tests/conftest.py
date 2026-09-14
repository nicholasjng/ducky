import ctypes
import os
import sys

import pytest

import ducky


@pytest.fixture
def con() -> ducky.Connection:
    """A fresh in-memory connection, isolated per test."""
    return ducky.connect()


def _deep_executable() -> str:
    """Return the framework interpreter needed for ASAN preloading on macOS."""
    dyld = ctypes.cdll.LoadLibrary("/usr/lib/system/libdyld.dylib")
    size = ctypes.c_ulong(1024)
    buf = ctypes.create_string_buffer(b"\000", size.value)
    dyld._NSGetExecutablePath(ctypes.byref(buf), ctypes.byref(size))
    return buf.value.decode()


@pytest.fixture
def subprocess_python() -> tuple[str, dict[str, str]]:
    """Return an executable and environment suitable for child test processes."""
    preload = os.environ.get("DUCKY_ASAN_PRELOAD")
    if not preload:
        return sys.executable, dict(os.environ)
    env = dict(os.environ)
    env["DYLD_INSERT_LIBRARIES"] = preload
    env["__PYVENV_LAUNCHER__"] = sys.executable
    return _deep_executable(), env
