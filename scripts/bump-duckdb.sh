#!/bin/sh
#
# Bump the vendored DuckDB submodule, or re-apply ducky's patch stack.
#
# Patches in patches/ (NNNN-slug.patch) are a working-tree overlay over a
# pristine upstream commit, applied in lexical order from a clean checkout.
#
# Usage:
#   scripts/bump-duckdb.sh [<ref>] [--no-patches]
#
#   <ref>          Commit SHA, tag, or branch. Moves the pin and refreshes
#                  OVERRIDE_GIT_DESCRIBE. Omit to re-apply patches/ at the
#                  current pin (run this after init-duckdb.sh).
#   --no-patches   Skip the patch stack (build against pristine upstream).
#
# Examples:
#   scripts/bump-duckdb.sh v1.6.3
#   scripts/bump-duckdb.sh
set -eu

usage() {
    awk 'NR < 3 { next } /^#/ { sub(/^# ?/, ""); print; next } { exit }' "$0"
}

ref=""
patches=1
while [ $# -gt 0 ]; do
    case "$1" in
        -h | --help) usage; exit 0 ;;
        --no-patches) patches=0; shift ;;
        --*) echo "error: unknown option '$1'" >&2; usage >&2; exit 2 ;;
        *)
            [ -z "$ref" ] || { echo "error: multiple refs given" >&2; exit 2; }
            ref="$1"; shift ;;
    esac
done

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"
submodule="ext/duckdb"
cmake="CMakeLists.txt"
patch_dir="patches"

[ -e "$submodule/.git" ] || {
    echo "error: $submodule is not checked out — run scripts/init-duckdb.sh first" >&2
    exit 1
}

if [ -n "$ref" ]; then
    bump=1
else
    bump=0
    ref="$(git -C "$submodule" rev-parse HEAD)"
fi

old_sha="$(git -C "$submodule" rev-parse HEAD)"
old_desc="$(awk -F'"' '/set\(OVERRIDE_GIT_DESCRIBE "/ {print $2; exit}' "$cmake")"

# checkout -f below discards the working tree; warn if there are changes not
# accounted for by patches/, so an unexported hand-edit isn't lost silently.
if [ -n "$(git -C "$submodule" status --porcelain)" ]; then
    echo ">> Submodule working tree is dirty; resetting to $ref + patches/."
    echo "   (Export hand-edits into $patch_dir/ first — see DEVELOPMENT.md.)"
fi

if [ "$bump" -eq 1 ]; then
    # `git describe` needs the tag/commit graph; the checkout is shallow.
    if [ "$(git -C "$submodule" rev-parse --is-shallow-repository)" = "true" ]; then
        unshallow="--unshallow"
    else
        unshallow=""
    fi
    echo ">> Fetching DuckDB history + tags (blobless)..."
    # shellcheck disable=SC2086
    git -C "$submodule" fetch --tags --filter=blob:none $unshallow origin \
        '+refs/heads/*:refs/remotes/origin/*'
fi

echo ">> Checking out $ref..."
git -C "$submodule" checkout -q -f --detach "$ref"
git -C "$submodule" sparse-checkout reapply 2>/dev/null || true

new_sha="$(git -C "$submodule" rev-parse HEAD)"

# --3way so patches survive context drift across a bump; any failure is fatal.
applied=0
if [ "$patches" -eq 1 ] && [ -d "$patch_dir" ]; then
    echo ">> Applying patch stack from $patch_dir/..."
    failed=""
    for p in "$patch_dir"/*.patch; do
        [ -e "$p" ] || break
        name="$(basename "$p")"
        if git -C "$submodule" apply --3way --check "$repo_root/$p" 2>/dev/null; then
            git -C "$submodule" apply --3way "$repo_root/$p"
            echo "   applied  $name"
            applied=$((applied + 1))
        else
            echo "   FAILED   $name" >&2
            failed="$failed $name"
        fi
    done
    if [ -n "$failed" ]; then
        echo "error: patch(es) did not apply against $new_sha:$failed" >&2
        echo "       Refresh or retire them (see DEVELOPMENT.md), then re-run." >&2
        exit 1
    fi
fi

describe=""
if [ "$bump" -eq 1 ]; then
    describe="$(git -C "$submodule" describe --tags --long --match 'v[0-9]*' 2>/dev/null || true)"
    [ -n "$describe" ] || {
        echo "error: could not derive a version describe — were tags fetched?" >&2
        exit 1
    }
    awk -v val="$describe" '
        /set\(OVERRIDE_GIT_DESCRIBE "/ && !done { sub(/"[^"]*"/, "\"" val "\""); done = 1 }
        { print }
    ' "$cmake" >"$cmake.tmp" && mv "$cmake.tmp" "$cmake"
fi

echo
if [ "$bump" -eq 1 ]; then
    echo "Bumped: $old_sha -> $new_sha"
    echo "        ${old_desc:-<none>} -> $describe"
else
    echo "Refreshed at $new_sha (pin unchanged)."
fi
echo "  patches applied: $applied$([ "$patches" -eq 0 ] && echo ' (--no-patches)')"
echo
echo "Next: uv sync --all-groups --reinstall-package=ducky && uv run --no-sync pytest -q"
[ "$bump" -eq 1 ] && echo "      then record the gitlink + $cmake (jj describe / git add)."
exit 0
