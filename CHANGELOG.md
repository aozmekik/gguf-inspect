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
  without reading tensor data. Big-endian files and non-regular files (pipes,
  devices) are detected and rejected.
- Errors (`Error`, `ErrorKind`) with byte offsets and the metadata entry or
  tensor being read; configurable allocation caps (`Limits`), with array
  nesting never deeper than 64 levels.
- `GgufFile::validate` reports data-section issues (truncated files,
  misaligned or overlapping tensors) and missing required keys, each kind
  once with a count.
- ggml type table (`GgmlType`) for 35 types, checked against ggml commit
  `a05c992` by a script CI runs.
- `ModelInfo` (well-known hyperparameters, `general.file_type` names from the
  spec and llama.cpp), `kv_cache::estimate` (GQA, per-layer head counts,
  multi-head latent attention and recurrent models, following llama.cpp
  `547208d`), `diff::diff` (header comparison) and
  `MetadataArray::first_difference`.
- CLI `gguf-inspect` with `info`, `metadata`, `tensors`, `kv-cache`, `diff`
  and `types`, each with `--json` (documented in `docs/json.md`). File-derived
  strings are escaped in text and JSON output.
- Test-only synthetic GGUF writer (`gguf-testkit`); fixtures written by
  gguf-py; property, malformed-input, CLI snapshot and CLI property tests; an
  opt-in test for real models; a cargo-fuzz target; a criterion benchmark;
  examples `synth` and `summary`.

[Unreleased]: https://github.com/aozmekik/gguf-inspect/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/aozmekik/gguf-inspect/releases/tag/v0.1.0
