#!/bin/sh
#
# Initialize the DuckDB submodule as a sparse, blobless checkout.
#
# DuckDB's full working tree is ~280 MB; ducky only needs ~50 MB of it (source,
# third_party, the essential extensions, and a handful of CMake glue files).
# This script configures the sparse checkout *before* the first checkout, so the
# unused paths are never written to disk — and, by fetching with
# `--filter=blob:none`, their blobs are never downloaded either.
#
# `git submodule update` has no `--no-checkout`, so we build the submodule's
# git dir ourselves (`git init --separate-git-dir` into `.git/modules/<path>`
# where `git submodule` expects it), configure sparsity, then check out the
# pinned revision — fetching it shallow and blobless first if needed.
#
# Every step is idempotent: fresh clone, half-finished run, or existing full
# checkout all converge to the same state. Keep the path list below in sync
# with DEVELOPMENT.md.
#
# Usage:
#   scripts/init-duckdb.sh
set -eu

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

submodule="ext/duckdb"
# Place the git dir where `git submodule` expects it.
gitdir="$repo_root/$(git rev-parse --git-path "modules/$submodule")"

# Sparse-checkout patterns (non-cone; leading '/' anchors at the submodule
# root). Anything not listed stays unwritten.
sparse_patterns='/CMakeLists.txt
/DuckDBConfig.cmake.in
/DuckDBConfigVersion.cmake.in
/LICENSE
/src/
/third_party/
/scripts/
/tools/CMakeLists.txt
/tools/utils/
/extension/CMakeLists.txt
/extension/*.cmake
/extension/*.in
/extension/loader/
/extension/core_functions/
/extension/parquet/
/extension/json/
/.github/config/extensions/httpfs.cmake
/.github/patches/extensions/httpfs/'

echo ">> Registering submodule..."
git submodule init "$submodule"
url="$(git config "submodule.$submodule.url")"
# Gitlink revision the superproject pins this submodule to.
sha="$(git ls-files -s "$submodule" | awk '{print $2}')"

# Create the git dir + worktree link only if it isn't linked yet.
if [ ! -e "$submodule/.git" ]; then
    echo ">> Creating git dir..."
    mkdir -p "$(dirname "$gitdir")" "$submodule"
    git init -q --separate-git-dir "$gitdir" "$submodule"
fi

echo ">> Configuring remote and sparse checkout..."
if git -C "$submodule" remote get-url origin >/dev/null 2>&1; then
    git -C "$submodule" remote set-url origin "$url"
else
    git -C "$submodule" remote add origin "$url"
fi
git -C "$submodule" config extensions.partialClone origin
git -C "$submodule" config remote.origin.promisor true
git -C "$submodule" config remote.origin.partialclonefilter blob:none
git -C "$submodule" config core.sparseCheckout true
git -C "$submodule" config core.sparseCheckoutCone false
printf '%s\n' "$sparse_patterns" >"$gitdir/info/sparse-checkout"

if ! git -C "$submodule" cat-file -e "$sha^{commit}" 2>/dev/null; then
    echo ">> Fetching pinned revision $sha (shallow, blobless)..."
    git -C "$submodule" fetch -q --filter=blob:none --depth 1 origin "$sha"
fi

echo ">> Checking out $sha..."
# -f repopulates the tree even when the index is already at this commit but
# files are missing (e.g. a wiped or half-built tree).
git -C "$submodule" checkout -q -f --detach "$sha"
# Trim paths a pre-existing full checkout may have left outside the sparse set.
git -C "$submodule" sparse-checkout reapply

echo ">> Done. DuckDB working tree: $(du -sh "$submodule" | cut -f1)"
