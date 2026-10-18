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

Two test suites run the binary many times: `tests/cli.rs` (snapshots) and
`tests/cli_properties.rs` (property tests on hostile files). One more is
opt-in, for real model files that are not part of the repository:

```sh
GGUF_INSPECT_REAL_MODELS=/models/a.gguf:/models/b.gguf \
    cargo test --release --test real_models -- --ignored --nocapture
```

## API conventions

- Every public item has a doc comment; examples in docs are doctests.
- Public structs with public fields and public enums are
  `#[non_exhaustive]`, so fields and variants can be added in minor
  releases. The exceptions are enums whose cases are complete by
  construction and that callers should match exhaustively: `PerLayer`
  (uniform or per-layer), `MetadataChange` and `TensorChange` (added,
  removed, changed).
- Strings from the file never reach the terminal unescaped: text output goes
  through `escape::text` (table cells do this themselves) or
  `escape::push_quoted`, and JSON through `escape::JsonFormatter`.
- The library does not print, has no dependencies and no `unsafe`.

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
The one exception is `tests/fixtures/gguf-py/`: three small synthetic files
written by gguf-py, llama.cpp's reference writer. To regenerate them (CI
checks that the output does not change):

```sh
python3 -m venv .venv
.venv/bin/pip install -r tests/fixtures/gguf-py/requirements.txt
.venv/bin/python -I tests/fixtures/gguf-py/generate.py tests/fixtures/gguf-py
```

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

The `Fuzz` workflow runs it for 20 minutes every week.

## ggml type table

When ggml adds or changes a tensor type, update `src/ggml_type.rs` and run
the check that CI runs (it compares every type ID, name, size and removed ID
with ggml's own `type_traits` table):

```sh
scripts/check-ggml-type-table.sh <ggml-commit>   # needs curl and a C compiler
```

## Commits

Conventional Commits (`feat:`, `fix:`, `test:`, `docs:`, `ci:`, ...); `perf:`
is for changes that make something faster or smaller, while adding a
benchmark is `test:`. Add an entry to `CHANGELOG.md` under `[Unreleased]`
for user-visible changes.
