# Benchmarks

All numbers on this page were measured on 2026-10-18 on the development
machine: Apple M1 Pro (8 cores), 16 GB RAM, macOS 26, Rust 1.99.1 (Homebrew),
release profile with thin LTO. Every input is synthetic; nothing here was
measured on a real model.

The machine was shared with other build jobs. Load averages are given next
to each measurement; with load above the core count, results vary from run
to run, so treat absolute numbers as indicative.

## Header parsing (criterion)

```sh
cargo bench --bench parse
```

`benches/parse.rs`, criterion 0.8.2, mean and 95% confidence interval; load
average about 8 (1-minute):

| Input | Header bytes | Time | Throughput |
|---|---:|---:|---:|
| 150,000-entry vocab + scores + token types, from memory | 3.79 MB | 5.90 ms [5.87, 5.94] | 612 MiB/s |
| 8B Llama-3-shaped header (128,256-entry vocab, 291 tensors), from memory | 3.24 MB | 4.43 ms [4.25, 4.73] | 698 MiB/s |
| same, `GgufFile::open` on a file (page cache warm) | 3.24 MB | 5.21 ms [5.19, 5.23] | 593 MiB/s |

### Before and after the review fixes

The fixes for the first review (escaping, sorted indexes instead of hash
maps, streamed listings) touched the parser, so the benchmark was run for
commit `8d094e5` (before) and the current code, alternating three times
while the load average moved between 10 and 33. Under that much load the
means of a single binary varied by up to 1.8x, so the comparison uses the
fastest and the median per-iteration sample of each run (from criterion's
`sample.json`), which other load can only make slower:

| Input | Before: min / median | After: min / median |
|---|---:|---:|
| vocab, from memory | 5.49-5.62 / 5.68-5.81 ms | 5.51-5.54 / 5.77-5.84 ms |
| 8B header, from memory | 4.06-4.12 / 4.13-4.22 ms | 4.07-4.12 / 4.17-4.33 ms |
| 8B header, from a file | 5.03-5.07 / 5.14-5.24 ms | 5.09-5.16 / 5.29-5.36 ms |

The ranges overlap or differ by about 2%: no measurable change.

## End to end (hyperfine)

```sh
cargo build --release
cargo run --release --example synth -- --llama3-8b llama3-8b-shape.gguf
hyperfine --warmup 5 --runs 50 -N \
  'target/release/gguf-inspect info llama3-8b-shape.gguf' \
  'target/release/gguf-inspect tensors llama3-8b-shape.gguf' \
  'target/release/gguf-inspect metadata llama3-8b-shape.gguf --json' \
  'target/release/gguf-inspect kv-cache llama3-8b-shape.gguf --ctx 8192' \
  'target/release/gguf-inspect types'
```

hyperfine 2.0.0, which also records peak resident memory per run. The file
is sparse: 5,175,662,304 bytes apparent size, 3.1 MB on disk. Load average
about 8:

| Command | Wall time (mean ± σ) | Peak RSS (median, max) |
|---|---:|---:|
| `info llama3-8b-shape.gguf` | 8.7 ms ± 0.5 ms | 9.3 MB, 10.4 MB |
| `tensors llama3-8b-shape.gguf` | 10.0 ms ± 1.1 ms | 9.5 MB, 10.6 MB |
| `metadata llama3-8b-shape.gguf --json` | 9.2 ms ± 1.5 ms | 9.3 MB, 10.4 MB |
| `kv-cache llama3-8b-shape.gguf --ctx 8192` | 8.9 ms ± 1.0 ms | 9.3 MB, 10.4 MB |
| `types` (no file: process start-up) | 2.6 ms ± 0.2 ms | 2.1 MB, 2.2 MB |

MB is 10^6 bytes.

Cost depends on the header, not on the file size. In a separate run (load
average 11), `tensors` took 15.2 ms ± 4.0 ms on the 4.82 GiB file and
14.2 ms ± 3.3 ms on a 3,241,696-byte copy holding only its header
(`head -c 3241696`).

## Worst-case memory

Headers made of many tiny records use the most memory per byte of file. Two
adversarial but valid files, written by `python3 -I
scripts/worst-case-headers.py <dir>`: 2^20 metadata
entries with one-byte values (22,020,120 bytes), and 2^20 scalar f32 tensor
infos at offset 0 (33,554,561 bytes). Peak RSS from `/usr/bin/time -l` on the
release build:

| Command | Before the review fixes (`8d094e5`) | Now |
|---|---:|---:|
| `metadata many-kv.gguf` | 421.8 MB | 94.8 MB |
| `metadata many-kv.gguf --json` | 381.3 MB | 94.8 MB |
| `info many-kv.gguf` | 270.3 MB | 94.8 MB |
| `tensors many-tensors.gguf` | 580.9 MB | 181.0 MB |
| `tensors many-tensors.gguf --json` | 398.9 MB | 180.8 MB |
| `info many-tensors.gguf` | 496.9 MB, 1,048,589 lines | 181.0 MB, 15 lines |

Before the fixes, `info many-tensors.gguf` printed one warning per
overlapping pair of tensors; it now prints one warning per kind. Both
binaries were run back to back with the same files.

Peak heap while parsing the 3.24 MB header of the 8B-shaped model, measured
with dhat's counting allocator in `tests/memory.rs`: 5,390,813 bytes.
