# ADR 0001: Read the header with a bounded streaming reader, not mmap

- Status: accepted
- Date: 2026-10-18

## Context

GGUF models are routinely 4-100 GB, but everything `gguf-inspect` needs (magic,
version, metadata, tensor infos) sits in a header that is usually a few
megabytes, most of it the tokenizer vocabulary. The tool must:

- never touch tensor data, so cost does not grow with model size;
- treat every file as untrusted: a header can declare a 2^64-byte string or a
  2^64-element array, and the parser must not allocate on the strength of it;
- work the same on Linux, macOS and Windows;
- ideally contain no `unsafe` code, since its whole job is reading hostile input.

## Options considered

1. **Read the whole file into memory.** Simplest parsing over a `&[u8]`, but
   reads gigabytes to use megabytes. Rejected.
2. **Memory-map the file (for example with `memmap2`) and parse the mapped
   slice.** Fast, and strings could be borrowed (`&str`) instead of copied.
   But creating a map is `unsafe` in Rust because another process can modify
   or truncate the file while it is mapped, which is undefined behaviour and,
   in practice, `SIGBUS` on Unix. Borrowed results would also tie every parsed
   value to the lifetime of the map. Only the header pages would be touched,
   so the I/O advantage over a buffered reader is small.
3. **A buffered streaming reader over any `Read`, with the total length known
   up front.** Every field is read through a small reader that tracks its
   position and checks each read, and each declared length, against the
   bytes left in the file before allocating. Parsed values are owned.

## Decision

Option 3. `GgufFile::open` wraps the file in a 256 KiB `BufReader` and passes
its length from `metadata()`; `GgufFile::from_reader` accepts any `Read` plus a
length, and `GgufFile::from_bytes` parses a slice. The crate is
`#![forbid(unsafe_code)]` (via the workspace lint table).

Declared counts are checked against the remaining bytes divided by the
smallest possible encoding of one element (1 byte for a `u8`, 8 for a string's
length prefix, 12 for a nested array's type tag and length), so a count can
never exceed what the file could actually hold. Fixed-width arrays are then
read in full, so their allocation matches bytes that exist. Vectors of
variable-size elements (strings, nested arrays, tensor infos) reserve at most
65,536 slots up front and grow as elements are parsed, so nesting cannot
multiply reservations. Configurable caps (`Limits`) bound
implausible-but-consistent files on top.

## Consequences

- No `unsafe`, no platform-specific code, no dependency for I/O.
- `GgufFile` owns its data, is `Send + Sync`, and has no lifetime parameter.
- Every vocabulary token is a separate `String` allocation. Measured on the
  development machine, a header with a 150,000-entry vocabulary parses in
  about 6 ms (see the README for the exact command and conditions), which is
  well below process start-up and terminal output for a CLI.
- Memory use is proportional to the header size, never to declared lengths.
- Reading tensor *data* is out of scope. If it is ever added, mmap or
  positioned reads would be reconsidered for that part only.
