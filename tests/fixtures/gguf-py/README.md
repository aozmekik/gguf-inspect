# Fixtures written by gguf-py

Three small GGUF files written by [gguf-py](https://github.com/ggml-org/llama.cpp/tree/master/gguf-py)
(the `gguf` package on PyPI, version 0.19.0), the writer that llama.cpp's
`convert_hf_to_gguf.py` uses for real models. The rest of the test suite uses
this repository's own writer (`testkit/`), which shares types with the parser;
these files check the parser against an independent implementation.

The contents are synthetic: made-up metadata, a five-token vocabulary and
tiny tensors (`arange`, ones, zeros). No part of any real model is included.

| File | What it is |
|---|---|
| `little-endian.gguf` | GGUF v3: every scalar value type, typed and nested arrays, f32/f16/q8_0/i32/f64 tensors including a 0-dimensional one |
| `aligned-64.gguf` | The same with `general.alignment = 64` |
| `big-endian.gguf` | The same written big-endian, which gguf-inspect rejects |

`tests/gguf_py.rs` checks every value, shape, type and offset, and reads the
tensor bytes at the offsets gguf-inspect computes. `tests/cli.rs` snapshots
the CLI output on them.

## Regenerating

```sh
python3 -m venv .venv
.venv/bin/pip install -r tests/fixtures/gguf-py/requirements.txt
.venv/bin/python -I tests/fixtures/gguf-py/generate.py tests/fixtures/gguf-py
git diff --exit-code tests/fixtures/gguf-py   # the output is deterministic
```

The files were generated on 2026-10-18 with Python 3.14.5 on macOS (Apple
M1 Pro); CI regenerates them on Linux and checks that the bytes match.
