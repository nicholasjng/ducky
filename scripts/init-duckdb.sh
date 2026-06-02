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
# `git submodule update` has no `--no-checkout`, so there is no one-liner that
# sets sparsity before materializing the tree. Instead we build the submodule's
# git dir ourselves (`git init --separate-git-dir` into the standard
# `.git/modules/<path>` location, exactly where `git submodule` expects it),
# configure sparsity, then check out the pinned revision — fetching it shallow
# and blobless first if the git dir doesn't already have it.
#
# Every step is idempotent and the flow converges to the same end state (pinned
# revision, sparse working tree) from a fresh clone, a half-finished run, or an
# existing full checkout. Keep the path list below in sync with DEVELOPMENT.md.
#
# Usage:
#   scripts/init-duckdb.sh
set -eu

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

submodule="ext/duckdb"
# `git submodule` keeps each submodule's git dir under the superproject's
# modules/ directory; place ours there so submodule commands keep working.
gitdir="$repo_root/$(git rev-parse --git-path "modules/$submodule")"

# Paths we keep inside the DuckDB working tree, one gitignore-style pattern per
# line (non-cone mode; a leading '/' anchors each pattern at the submodule
# root). Anything not listed here stays unwritten.
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
# The recorded gitlink revision the superproject pins this submodule to.
sha="$(git ls-files -s "$submodule" | awk '{print $2}')"

# Create the git dir + worktree link if they aren't already there. `git init`
# is safe to repeat (it reinitializes), but we only need it when the worktree
# isn't linked yet — avoid touching an already-working checkout.
if [ ! -e "$submodule/.git" ]; then
    echo ">> Creating git dir..."
    mkdir -p "$(dirname "$gitdir")" "$submodule"
    git init -q --separate-git-dir "$gitdir" "$submodule"
fi

echo ">> Configuring remote and sparse checkout..."
# Idempotent remote setup: the git dir may already carry an 'origin'.
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

# Fetch the pinned revision only if the git dir doesn't already have it.
if ! git -C "$submodule" cat-file -e "$sha^{commit}" 2>/dev/null; then
    echo ">> Fetching pinned revision $sha (shallow, blobless)..."
    git -C "$submodule" fetch -q --filter=blob:none --depth 1 origin "$sha"
fi

echo ">> Checking out $sha..."
# -f resets the working tree to the pinned revision (like `git submodule
# update --force`) and repopulates it even when the index already points at
# this commit but the files are missing (e.g. a wiped or half-built tree).
git -C "$submodule" checkout -q -f --detach "$sha"
# Trim any paths a pre-existing full checkout left outside the sparse set.
git -C "$submodule" sparse-checkout reapply

echo ">> Done. DuckDB working tree: $(du -sh "$submodule" | cut -f1)"
