#!/usr/bin/env sh
# Checks src/ggml_type.rs against ggml's own type table: type IDs, names,
# block sizes, block byte sizes, the quantized flag and the removed IDs, for
# every type. Downloads include/ggml.h, src/ggml.c and src/ggml-common.h at a
# ggml commit, compiles ggml's type_traits initializer (without its function
# pointers) into scripts/ggml-type-table.c, and compares the result with
# `gguf-inspect types`. Needs curl, a C compiler and cargo.
#
# The initializer is copied out of ggml.c line by line, so a reformatting of
# that table makes this script fail (to compile or to match), never pass.
#
# usage: scripts/check-ggml-type-table.sh [GGML_COMMIT]
set -eu

commit="${1:-a05c9922613c6a2230dbb1d0de679519ecafb4f0}"
here="$(cd "$(dirname "$0")" && pwd)"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

base="https://raw.githubusercontent.com/ggml-org/ggml/${commit}"
curl -fsSL "$base/include/ggml.h" -o "$work/ggml.h"
curl -fsSL "$base/src/ggml-common.h" -o "$work/ggml-common.h"
curl -fsSL "$base/src/ggml.c" -o "$work/ggml.c"

awk '/^static const struct ggml_type_traits type_traits\[GGML_TYPE_COUNT\] = \{/ { inside = 1; next }
     inside && /^};/ { exit }
     inside' "$work/ggml.c" |
    grep -v -e '\.to_float' -e '\.from_float' >"$work/traits.inc"
test -s "$work/traits.inc" || { echo "type_traits table not found in ggml.c" >&2; exit 1; }

cc -std=c11 -Wall -Werror -I "$work" -o "$work/table" "$here/ggml-type-table.c"
"$work/table" >"$work/ggml.txt"

cargo run --quiet --bin gguf-inspect -- types >"$work/types.txt"
{
    awk 'NR > 1 && NF == 6 { print $1, $2, $3, $4, $6 }' "$work/types.txt"
    sed -n 's/^removed ids: //p' "$work/types.txt" | tr ',' '\n' | tr -d ' ' |
        awk 'NF { print $1, "removed" }'
} | sort -n >"$work/ours.txt"

if diff -u "$work/ggml.txt" "$work/ours.txt"; then
    known=$(grep -vc removed "$work/ours.txt")
    removed=$(grep -c removed "$work/ours.txt")
    echo "ok: ${known} types and ${removed} removed IDs match ggml ${commit}"
else
    echo "mismatch between ggml ${commit} (-) and gguf-inspect (+)" >&2
    exit 1
fi
