#!/usr/bin/env sh
# Checks the block sizes and block byte sizes in src/ggml_type.rs against
# ggml's own block structs: downloads src/ggml-common.h at a ggml commit,
# compiles scripts/ggml-type-sizes.c against it, and compares the output with
# `gguf-inspect types`. Needs curl, a C compiler and cargo.
#
# usage: scripts/check-ggml-type-sizes.sh [GGML_COMMIT]
set -eu

commit="${1:-a05c9922613c6a2230dbb1d0de679519ecafb4f0}"
here="$(cd "$(dirname "$0")" && pwd)"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

curl -fsSL "https://raw.githubusercontent.com/ggml-org/ggml/${commit}/src/ggml-common.h" \
    -o "$work/ggml-common.h"
cc -std=c11 -Wall -I "$work" -o "$work/sizes" "$here/ggml-type-sizes.c"
"$work/sizes" | sort >"$work/ggml.txt"

cargo run --quiet --bin gguf-inspect -- types |
    awk 'NR > 1 && $6 == "yes" { print $2, $3, $4 }' | sort >"$work/ours.txt"

if diff -u "$work/ggml.txt" "$work/ours.txt"; then
    echo "ok: $(wc -l <"$work/ours.txt" | tr -d ' ') quantized types match ggml ${commit}"
else
    echo "mismatch between ggml ${commit} (-) and gguf-inspect (+)" >&2
    exit 1
fi
