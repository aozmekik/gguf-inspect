# ADR 0004: KV-cache estimates follow llama.cpp's allocation

- Status: accepted
- Date: 2026-10-18

## Context

`gguf-inspect kv-cache` answers "how much memory will the KV cache need at
this context length?". The GGUF specification names the hyperparameters
(`block_count`, `attention.head_count_kv`, `attention.key_length`, ...) but
not how a runtime turns them into a cache. The textbook formula,
`layers × tokens × head_count_kv × head_dim × 2 × bytes per element`, is
right for multi-head and grouped-query attention and wrong elsewhere:

- **Multi-head latent attention** (DeepSeek-V2/V3, Kimi K2). Current
  converters store the compressed latent sizes in `key_length`/`value_length`
  (576 and 512 for DeepSeek-V3, one KV head) and the decompressed head sizes
  in `key_length_mla`/`value_length_mla`. llama.cpp caches the latent as K
  and has no V cache, so the textbook formula is about 1.9x too high.
- **Recurrent models** (Mamba, RWKV) have `head_count = 0`; the formula
  divides by zero. Hybrid models list 0 KV heads for their recurrent layers.
- **Sequences.** llama.cpp's `-c` is the total for all `-np` sequences, and
  it pads the context to a multiple of 256.
- **Sliding-window layers** may be cached for fewer tokens, depending on
  runtime settings.

An estimate that silently uses the wrong model for any of these is worse
than none: ADR 0002 rules out "confident but wrong numbers".

## Options considered

1. **The textbook formula only**, documented as such. Simple, but wrong by
   about 2x for the MLA models users are most likely to ask about, and an
   error for recurrent models.
2. **Follow llama.cpp's allocation** at a pinned commit, the reference
   runtime for GGUF, citing the source lines; flag with notes what is not
   modelled.
3. **Model several runtimes** (llama.cpp, vLLM, MLX, ...). Their cache
   layouts and options differ and most do not read GGUF hyperparameters the
   same way; the tool would need a runtime flag and many more tests than it
   could honestly maintain.

## Decision

Option 2, against llama.cpp commit
[`547208d`](https://github.com/ggml-org/llama.cpp/tree/547208dca250ceb3e4c8615598d043f2b602a527):

- MLA is detected as llama.cpp does (`llama_hparams::is_mla`: both `*_mla`
  keys non-zero), and then only K is counted (`has_v = !is_mla` in
  `src/llama-kv-cache.cpp`). Files in the older layout, with
  `kv_lora_rank` but no `*_mla` keys, keep full K and V, which is what
  llama.cpp allocates for them, and get a note saying so.
- `head_count = 0` gives head sizes of 0 and an empty cache, as in
  `src/llama-model.cpp`, with notes for the recurrent state and the layers
  without KV heads.
- A per-layer list must have one entry per layer, as llama.cpp's
  `get_key_or_arr` requires ("wrong array length").
- `--sequences N` caches `--ctx` tokens per sequence; the help and README map
  llama.cpp's `-c C -np N` to `--ctx C/N --sequences N`.

Beyond llama.cpp, a key that is present but unusable (negative, a float, a
nested array) is an error naming the key and what was found, instead of a
fallback to a default.

## Consequences

- The DeepSeek-V3-shaped test file is estimated at 549 MiB at 8,192 tokens,
  the size llama.cpp allocates, instead of 1.01 GiB.
- The estimate is tied to one runtime. When llama.cpp changes its cache, the
  pinned commit and the tests need updating; the module docs say which
  source lines were followed.
- Not modelled, with a note when a file uses them: sliding-window caches
  (the estimate is then an upper bound), recurrent state, compute buffers and
  llama.cpp's padding of the context to 256 tokens.
- Stricter than llama.cpp for unusable keys, which it may accept or reject
  with a less specific message.
