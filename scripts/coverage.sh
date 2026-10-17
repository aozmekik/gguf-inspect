#!/usr/bin/env bash
# Line and region coverage of src/ with LLVM source-based coverage, without
# cargo-llvm-cov. Needs llvm-profdata and llvm-cov from the same LLVM major
# version as rustc (`rustc -vV | grep LLVM`); point LLVM_BIN at them.
#
# usage: LLVM_BIN=/opt/homebrew/opt/llvm/bin scripts/coverage.sh
set -euo pipefail

llvm_bin="${LLVM_BIN:-$(dirname "$(command -v llvm-cov)")}"
root="$(cd "$(dirname "$0")/.." && pwd)"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

cd "$root"
export RUSTFLAGS="-C instrument-coverage"
export LLVM_PROFILE_FILE="$work/profiles/%p-%m.profraw"
export CARGO_TARGET_DIR="$work/target"

cargo test --workspace --quiet >/dev/null
"$llvm_bin/llvm-profdata" merge -sparse "$work"/profiles/*.profraw -o "$work/merged.profdata"

objects=()
while IFS= read -r exe; do
    objects+=(--object "$exe")
done < <(cargo test --workspace --no-run --message-format=json 2>/dev/null |
    sed -n 's/.*"executable":"\([^"]*\)".*/\1/p')

"$llvm_bin/llvm-cov" report --instr-profile="$work/merged.profdata" "${objects[@]}" \
    --ignore-filename-regex='(\.cargo/registry|/rustc/|/lib/rustlib/|/tests/|/testkit/|/benches/|/examples/)'
