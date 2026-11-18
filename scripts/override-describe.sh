#!/bin/sh
# picky post-update hook. DuckDB's build needs a version, which a shallow clone
# cannot describe. Reading the release line from the checkout keeps pin bumps
# from leaving a stale version behind. The dev iteration is unknown, so it is
# zero, while the embedded commit hash stays exact.
set -eu
release="$(cat scripts/ci/release_version.txt)"
printf 'v%s.0-dev0\n' "$release" > "$PICKY_ROOT/cmake/duckdb-describe.txt"
