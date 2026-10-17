# Contributing

Thanks for taking a look. Issues and pull requests are welcome; for larger
changes, please open an issue first to discuss the approach.

## Setup

Rust 1.88 or newer (stable). Nothing else is needed for building and testing.
The fuzzing and type-table checks below need extra tools, listed with them.

## Build, test, lint

```sh
cargo build --workspace
cargo test --workspace                       # unit, integration, snapshot, property and doc tests
cargo test -p gguf-inspect --lib --no-default-features   # library without CLI dependencies
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo clippy -p gguf-inspect --lib --no-default-features -- -D warnings
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps
```

The lint configuration lives in `Cargo.toml` (`[workspace.lints]`, clippy
`pedantic`) and `clippy.toml`. `rust-version = "1.88"` lets clippy flag
standard-library APIs that are newer than the MSRV.

## Coverage

```sh
LLVM_BIN=/path/to/llvm/bin scripts/coverage.sh
```

The script uses rustc's built-in source-based coverage, so `llvm-profdata`
and `llvm-cov` must come from the same LLVM major version as `rustc -vV`.

## Snapshot tests

CLI output is checked with [insta](https://insta.rs) snapshots in
`tests/snapshots/`. After an intentional output change:

```sh
INSTA_UPDATE=always cargo test --test cli
git diff tests/snapshots   # review every change before committing
```

## Test data

Never commit real model files. Tests, benchmarks and examples generate
synthetic files with `gguf-testkit` (`testkit/`); `*.gguf` is git-ignored.

```sh
cargo run --example synth -- tiny.gguf
cargo run --example synth -- --llama3-8b big.gguf   # sparse, 4.82 GiB apparent size
```

## Benchmarks

```sh
cargo bench --bench parse
cargo build --release
hyperfine --warmup 5 -N 'target/release/gguf-inspect info big.gguf'
```

Report the hardware, toolchain and machine load next to any number you quote.

## Fuzzing

See [fuzz/README.md](fuzz/README.md). The standard workflow needs a nightly
toolchain and `cargo install cargo-fuzz`:

```sh
cargo +nightly fuzz run parse
```

## ggml type table

When ggml adds or changes a tensor type, update `src/ggml_type.rs` and run:

```sh
scripts/check-ggml-type-sizes.sh <ggml-commit>   # needs curl and a C compiler
```

## Commits

Conventional Commits (`feat:`, `fix:`, `test:`, `docs:`, `ci:`, ...). Add an
entry to `CHANGELOG.md` under `[Unreleased]` for user-visible changes.
