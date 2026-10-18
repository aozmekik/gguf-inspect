# ADR 0002: Fail on header problems, report data-section problems

- Status: accepted
- Date: 2026-10-18

## Context

An inspector sees malformed files more often than a loader does: partial
downloads, files from experimental converters, files written by hand. Two
failure modes are both bad:

- refusing to show anything about a file that is mostly fine (for example a
  model whose download stopped at 60 %), and
- showing confident but wrong numbers (a parameter count or a size computed
  from data the parser silently guessed at).

ggml's own reader (`src/gguf.cpp`) is the de facto reference for what loaders
accept, and the GGUF specification (`docs/gguf.md`) states several "must"
rules that `gguf.cpp` does not enforce.

## Options considered

1. **Strict everywhere**, rejecting the file on any deviation, including
   tensor offsets that are misaligned or point past the end of the file.
   Partial downloads become unreadable, although their headers are complete.
2. **Lenient everywhere**: lossy UTF-8, unknown tensor types kept as opaque
   IDs, bools accepted if non-zero. Every derived number (sizes, parameter
   counts, KV-cache estimates) then has to carry an "unknown" state, and some
   would be wrong without saying so.
3. **Split by what the problem affects.** Anything that makes the header
   unreadable or its meaning ambiguous is an `Error` with a byte offset.
   Anything that only concerns where tensor data lies, or which keys are
   present, is an `Issue` returned by `GgufFile::validate()`; the header is
   still fully inspectable.

## Decision

Option 3.

Errors: truncation inside the header, bad magic, big-endian byte order,
versions other than 2 and 3, unknown value type tags, bools other than 0/1
(the spec calls anything else invalid), invalid UTF-8 in keys, strings and
tensor names (the spec says UTF-8), empty or duplicate keys, duplicate tensor
names, `general.alignment` that is not a power-of-two `u32`, tensor types
whose storage size is unknown (removed or newer than this crate), shapes with
a dimension above `i64::MAX` or whose dimensions overflow `i64` when
multiplied in order (as `gguf.cpp` checks), byte sizes that overflow `u64`,
first dimensions that are not a whole number of blocks, and declared sizes
over the configured limits. Paths that are not regular files (pipes,
devices) are rejected up front, because declared sizes are checked against
the file size.

Issues: tensor data extending past the end of the file, offsets not aligned,
overlapping tensors, an alignment that is not a multiple of 8, a missing
`general.architecture`, and quantized tensors without
`general.quantization_version`. Each kind is reported once, with a count and
the first affected tensor, so a hostile file cannot produce a million
warnings.

Where this differs from `gguf.cpp`: bools are checked strictly (`gguf.cpp`
treats any non-zero byte as true), strings must be valid UTF-8 (`gguf.cpp`
does not check), nested arrays are supported (the spec allows them; `gguf.cpp`
rejects them) up to 8 levels by default and never more than 64, tensor names may be 64 bytes long as the spec says
(`gguf.cpp` stops at 63 because it stores a NUL terminator), and tensor
offsets need not be packed back to back (`gguf.cpp` requires each offset to
equal the running total; that is not checked here, because inspection does
not depend on it). Duplicates, alignment rules,
dimension limits and unknown types are handled as in `gguf.cpp`.

## Consequences

- `gguf-inspect info` works on partial downloads and says how much is missing.
- Every number the tool prints is computed from validated fields only.
- A file with an exotic but spec-conforming feature (nested arrays) can be
  inspected even though llama.cpp would refuse to load it; one with invalid
  UTF-8 in a string is rejected even though llama.cpp might load it. Both are
  deliberate and documented in the README.
- New ggml types need a crate update before files using them can be read;
  the error message names the unknown type ID.
