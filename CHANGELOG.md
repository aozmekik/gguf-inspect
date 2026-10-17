# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-10-18

### Added

- Library: `GgufFile` parses the header of GGUF v2 and v3 files (all 13
  metadata value types including nested arrays, tensor infos,
  `general.alignment`, data offset) through a bounds-checked buffered reader,
  without reading tensor data. Big-endian files are detected and rejected.
- Errors (`Error`, `ErrorKind`) with byte offsets and the metadata entry or
  tensor being read; configurable allocation caps (`Limits`).
- `GgufFile::validate` reports data-section issues (truncated files,
  misaligned or overlapping tensors) and missing required keys.
- ggml type table (`GgmlType`) for 35 types, checked against ggml commit
  `a05c992`.
- `ModelInfo` (well-known hyperparameters), `kv_cache::estimate` (GQA and
  per-layer head counts), `diff::diff` (header comparison).
- CLI `gguf-inspect` with `info`, `metadata`, `tensors`, `kv-cache`, `diff`
  and `types`, each with `--json`.
- Test-only synthetic GGUF writer (`gguf-testkit`), property, malformed-input
  and CLI snapshot tests, a cargo-fuzz target, a criterion benchmark, and
  examples `synth` and `summary`.

[Unreleased]: https://github.com/aozmekik/gguf-inspect/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/aozmekik/gguf-inspect/releases/tag/v0.1.0
