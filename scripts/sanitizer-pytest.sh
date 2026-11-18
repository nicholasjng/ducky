#!/usr/bin/env bash
#
# Run pytest under ASAN, UBSAN, or TSAN on macOS.
# SIP strips DYLD_INSERT_LIBRARIES when a protected binary, such as a system Python launcher, runs in between, so this starts the interpreter behind the venv's python directly.
#
# Prerequisites: activate the uv-managed venv (VIRTUAL_ENV set) holding the sanitizer's editable install:
#     DUCKY_ASAN=1 uv sync --all-groups --reinstall-package=ducky
#     DUCKY_UBSAN=1 uv sync --all-groups --reinstall-package=ducky
#     DUCKY_TSAN=1 uv sync --all-groups --reinstall-package=ducky
#
# Usage: scripts/sanitizer-pytest.sh asan|ubsan|tsan [pytest args]

set -euo pipefail

me="scripts/sanitizer-pytest.sh"
case "${1:-}" in
  asan)
    options_var=ASAN_OPTIONS
    options="detect_leaks=0:halt_on_error=1:detect_stack_use_after_return=1:strict_string_checks=1"
    ;;
  ubsan) options_var=UBSAN_OPTIONS options="halt_on_error=1:print_stacktrace=1" ;;
  tsan) options_var=TSAN_OPTIONS options="halt_on_error=1:print_full_thread_history=1" ;;
  *)
    echo "usage: $me asan|ubsan|tsan [pytest args]" >&2
    exit 2
    ;;
esac
sanitizer=$1
shift

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  echo "$me: VIRTUAL_ENV is unset — activate the venv first" >&2
  exit 1
fi
VENV_PY="$VIRTUAL_ENV/bin/python"
if [[ ! -x "$VENV_PY" ]]; then
  echo "$me: $VENV_PY not found — is VIRTUAL_ENV correct?" >&2
  exit 1
fi

# Give ASAN visibility into individual Python allocations.
# Free-threaded Python requires mimalloc, so it keeps it, as does any caller override.
if [[ "$sanitizer" == asan && -z "${PYTHONMALLOC+x}" ]]; then
  if "$VENV_PY" -c 'import sys, sysconfig; sys.exit(bool(sysconfig.get_config_var("Py_GIL_DISABLED")))'; then
    export PYTHONMALLOC=malloc
  fi
fi

# whoami.py, taken from https://jonasdevlieghere.com/post/sanitizing-python-modules/.
DEEP_PY=$("$VENV_PY" - <<'PY'
import ctypes
dyld = ctypes.cdll.LoadLibrary('/usr/lib/system/libdyld.dylib')
n = ctypes.c_ulong(1024)
buf = ctypes.create_string_buffer(b'\000', n.value)
dyld._NSGetExecutablePath(ctypes.byref(buf), ctypes.byref(n))
print(buf.value.decode())
PY
)

RUNTIME="$(clang -print-file-name="libclang_rt.${sanitizer}_osx_dynamic.dylib")"

# At startup the runtime strips itself from DYLD_INSERT_LIBRARIES, and strip_env=0 keeps it for child interpreters.
# DUCKY_SANITIZER_PRELOAD tells the tests' subprocess_python fixture which runtime to preload into the children it starts.
# The caller's own options replace the defaults above and, coming later, can turn strip_env back on.
export "$options_var=strip_env=0:${!options_var:-$options}"
DYLD_INSERT_LIBRARIES="$RUNTIME" \
DUCKY_SANITIZER_PRELOAD="$RUNTIME" \
__PYVENV_LAUNCHER__="$VENV_PY" \
  "$DEEP_PY" -m pytest --capture no "$@"
