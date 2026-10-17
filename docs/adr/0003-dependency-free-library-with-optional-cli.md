# ADR 0003: Dependency-free library, CLI behind a default feature

- Status: accepted
- Date: 2026-10-18

## Context

The project is both a library (to embed in model servers, download managers
and tooling) and a CLI. The CLI needs argument parsing (clap) and JSON output
(serde, serde_json). Library users who parse untrusted files care about
auditability and compile times, and should not have to take those
dependencies. At the same time, `cargo install --git <url> gguf-inspect`
should install the CLI without feature flags.

## Options considered

1. **One package, always with the CLI dependencies.** Simplest manifest;
   every library user compiles clap and serde.
2. **Two packages in one workspace** (a `gguf-inspect` library and a separate
   CLI package). Clean separation, but two crate names to publish and
   version, and a third crate name to explain.
3. **One package with a `cli` feature that is on by default.** The binary
   target declares `required-features = ["cli"]`; clap, serde, serde_json and
   unicode-width are optional dependencies enabled by that feature.

## Decision

Option 3. Library users depend on the crate with `default-features = false`
and get zero dependencies. The JSON output formats are defined in the binary
(`src/bin/gguf-inspect/`), not by deriving `Serialize` on library types, so
the library's public types do not commit to a serialization format.

## Consequences

- `cargo install --git https://github.com/aozmekik/gguf-inspect gguf-inspect`
  installs the CLI (the package name is needed only because the fuzzing
  crate in `fuzz/` also has binaries); the library alone builds with no
  third-party code.
- CI runs clippy and the unit tests with `--no-default-features` to keep the
  library honest.
- Library users who want JSON write their own `Serialize` wrappers. An
  optional `serde` feature on the library could be added later without
  breaking anything.
- Integration tests that run the binary are declared with
  `required-features = ["cli"]`.
