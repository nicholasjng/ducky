#!/usr/bin/env bash
#
# Run the ASAN build's tests on macOS.
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

# Resolve the framework interpreter used by dyld.
DEEP_PY=$("$VENV_PY" - <<'PY'
import ctypes
dyld = ctypes.cdll.LoadLibrary('/usr/lib/system/libdyld.dylib')
n = ctypes.c_ulong(1024)
buf = ctypes.create_string_buffer(b'\000', n.value)
dyld._NSGetExecutablePath(ctypes.byref(buf), ctypes.byref(n))
print(buf.value.decode())
PY
)

ASAN_LIB="$(clang -print-file-name=libclang_rt.asan_osx_dynamic.dylib)"

# Preserve the preload path for child test processes.
DYLD_INSERT_LIBRARIES="$ASAN_LIB" \
DUCKY_ASAN_PRELOAD="$ASAN_LIB" \
ASAN_OPTIONS="${ASAN_OPTIONS:-detect_leaks=0:halt_on_error=1}" \
__PYVENV_LAUNCHER__="$VENV_PY" \
  "$DEEP_PY" -m pytest --capture no "$@"
