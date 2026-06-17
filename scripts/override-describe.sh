#!/bin/sh
# picky post-update hook: stamp a static OVERRIDE_GIT_DESCRIBE for DuckDB's
# build. Uses only HEAD (no tags), so it works in a shallow checkout.
set -eu
ver="v1.5.4"                                   # the DuckDB version you track
printf '%s-0-g%s\n' "$ver" "$(git rev-parse --short HEAD)" \
    > "$PICKY_ROOT/cmake/duckdb-describe.txt"
