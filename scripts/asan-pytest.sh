#!/usr/bin/env bash
#
# Run pytest under AddressSanitizer on macOS.
#
# Why this script exists: on macOS, dyld silently drops DYLD_INSERT_LIBRARIES
# when launching Homebrew's bin/python3.13 (the shim that .venv/bin/python
# symlinks to), so the standard "DYLD_INSERT_LIBRARIES=… uv run pytest" pattern
# does not preload the ASAN runtime and the test process aborts at the first
# import of an ASAN-built extension with "Interceptors are not working".
#
# What works instead is the same trick nanobind uses for stubgen (PR #1000):
#   1. Resolve the *real* CPython binary at Python.app/Contents/MacOS/Python
#      via dyld's _NSGetExecutablePath (Homebrew's bin/python3.13 is a shim that
#      re-execs to it; the deep binary preserves DYLD_INSERT_LIBRARIES).
#   2. Exec it directly from bash with DYLD_INSERT_LIBRARIES as a per-command
#      assignment — bash's `exec` builtin goes straight to execve(2), so the
#      env var lands on the child without any /bin/sh -c trampoline to strip it.
#   3. Set __PYVENV_LAUNCHER__ to the venv's bin/python so Python re-activates
#      the venv: sys.executable / sys.prefix point at .venv, the editable
#      install's .pth is processed, and `import ducky` finds the ASAN-built
#      _core.so in build/asan via the scikit-build editable finder.
#
# Pre-reqs:
#   - The uv-managed venv is activated in the current shell (so VIRTUAL_ENV is
#     set), e.g. via `source .venv/bin/activate` or whatever your shell hook
#     does on `cd`.
#   - The ASAN editable install is current:
#         DUCKY_ASAN=1 uv sync --all-groups --reinstall-package=ducky
#     A plain `uv sync` afterwards re-points the editable install at the
#     Release build/ dir; re-run the DUCKY_ASAN sync to switch back.
#
# Usage: scripts/asan-pytest.sh [pytest args]

set -euo pipefail

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  echo "scripts/asan-pytest.sh: VIRTUAL_ENV is unset — activate the venv first" >&2
  exit 1
fi
VENV_PY="$VIRTUAL_ENV/bin/python"
if [[ ! -x "$VENV_PY" ]]; then
  echo "scripts/asan-pytest.sh: $VENV_PY not found — is VIRTUAL_ENV correct?" >&2
  exit 1
fi

# Resolve the underlying CPython binary via _NSGetExecutablePath, the same way
# nanobind/cmake/darwin-python-path.py does it. The venv python is a symlink
# (or thin wrapper) over the Homebrew Python.framework binary; this returns
# the real Mach-O at .../Python.app/Contents/MacOS/Python.
DEEP_PY=$("$VENV_PY" - <<'PY'
import ctypes
dyld = ctypes.cdll.LoadLibrary('/usr/lib/system/libdyld.dylib')
n = ctypes.c_ulong(1024)
buf = ctypes.create_string_buffer(b'\000', n.value)
dyld._NSGetExecutablePath(ctypes.byref(buf), ctypes.byref(n))
print(buf.value.decode())
PY
)

ASAN_LIB=$(clang -print-file-name=libclang_rt.asan_osx_dynamic.dylib)
if [[ ! -f "$ASAN_LIB" ]]; then
  echo "scripts/asan-pytest.sh: couldn't locate libclang_rt.asan_osx_dynamic.dylib (got '$ASAN_LIB')" >&2
  exit 1
fi

DYLD_INSERT_LIBRARIES="$ASAN_LIB" \
ASAN_OPTIONS="${ASAN_OPTIONS:-detect_leaks=0:halt_on_error=1}" \
__PYVENV_LAUNCHER__="$VENV_PY" \
  "$DEEP_PY" -m pytest "$@"
