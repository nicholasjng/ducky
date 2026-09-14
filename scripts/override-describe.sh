#!/bin/sh
# picky post-update hook: stamp a valid development version for DuckDB's build.
# Read the release line from the sparse checkout so pin bumps cannot leave a
# stale hard-coded version behind. A shallow clone cannot derive DuckDB's full
# dev iteration, so use zero; the separately embedded commit hash remains exact.
set -eu
release="$(cat scripts/ci/release_version.txt)"
printf 'v%s.0-dev0\n' "$release" > "$PICKY_ROOT/cmake/duckdb-describe.txt"
