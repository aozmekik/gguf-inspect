# Fuzzing

One [cargo-fuzz](https://github.com/rust-fuzz/cargo-fuzz) target, `parse`,
feeds arbitrary bytes to `GgufFile::from_bytes` and then exercises
everything built on a successful parse (validation, model summary, KV-cache
estimates, diff). Any panic or allocation failure is a bug.

## With cargo-fuzz (nightly)

The standard way; it also enables AddressSanitizer.

```sh
rustup toolchain install nightly
cargo install cargo-fuzz

# Seed corpus from the synthetic writer (run from the repository root).
mkdir -p fuzz/corpus/parse
cargo run --example synth -- --header-only fuzz/corpus/parse/tiny-header.gguf
cargo run --example synth -- --every-type fuzz/corpus/parse/every-type.gguf

cargo +nightly fuzz run parse fuzz/corpus/parse -- -max_len=16384
```

`fuzz/corpus/` and `fuzz/artifacts/` are git-ignored.

## On stable, without a sanitizer

The development machine for this project had no nightly toolchain, so the
target was built with the coverage flags cargo-fuzz passes, minus
`-Zsanitizer` (nightly only). Overflow checks and debug assertions are on.
`--target` must be given explicitly so the flags do not apply to build
scripts.

```sh
cd fuzz
RUSTFLAGS="-Cpasses=sancov-module \
  -Cllvm-args=-sanitizer-coverage-level=4 \
  -Cllvm-args=-sanitizer-coverage-inline-8bit-counters \
  -Cllvm-args=-sanitizer-coverage-pc-table \
  -Cllvm-args=-sanitizer-coverage-trace-compares \
  --cfg fuzzing -Cdebug-assertions -Coverflow-checks" \
  cargo build --release --target "$(rustc -vV | sed -n 's/^host: //p')" --bin parse
./target/*/release/parse corpus/parse -max_len=16384 -max_total_time=300
```

Result of that run on 2026-10-18 at commit `4003852` (Apple M1 Pro, 16 GB,
Rust 1.99.1, seeded with the two files above, machine shared with other build
jobs): 13,510,653 executions in 301 s, no crashes, peak RSS 32 MB. An earlier
300 s run on commit `fd28152` executed 3,925,632 inputs, also without crashes.
