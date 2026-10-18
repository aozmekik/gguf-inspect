# JSON output

Every command accepts `--json` and then prints one pretty-printed JSON value
on stdout. The shapes below are those of v0.1.0; they may still change before
1.0, and changes will be listed in the [changelog](../CHANGELOG.md).

Conventions used everywhere:

- Names follow the file: metadata keys and tensor names are reported as
  stored. Strings are escaped as JSON requires, plus DEL, C1 control
  characters and invisible format characters (bidi overrides, zero-width
  characters), which are written as `\uXXXX`. A JSON parser returns the
  exact string from the file.
- `ggml_type` is always a ggml tensor type name (`"q4_K"`); `type` is
  always a metadata value type (`"u32"`, `"string"`, `"array"`).
- Absent optional values are `null`. Non-finite floats (NaN, infinity) have
  no JSON form and are written as `null`.
- Per-layer hyperparameters (`head_count`, `head_count_kv`,
  `feed_forward_length`) are a number when the file stores one value for all
  layers, and an array with one entry per layer when it stores a list.
- Unless `metadata --full` is given, arrays are cut to their first 8
  elements at every level of nesting. `len` gives the full length of the
  top-level array, and `"truncated": true` marks values where anything was
  cut. Strings are never cut.
- Warnings and notes are objects with a stable `kind` and a human-readable
  `message`; match on `kind`.

## `info --json`

```json
{
  "file": "tiny.gguf",
  "file_size": 1059232,
  "gguf_version": 3,
  "alignment": 32,
  "header_size": 13712,
  "data_offset": 13728,
  "metadata_count": 19,
  "tensor_count": 21,
  "parameter_count": 1443072,
  "tensor_data_size": 1045504,
  "model": {
    "architecture": "llama",
    "name": "Synthetic tiny",
    "file_type": { "id": 15, "name": "MOSTLY_Q4_K_M" },
    "quantization_version": 2,
    "context_length": 2048,
    "embedding_length": 256,
    "block_count": 2,
    "feed_forward_length": 512,
    "head_count": 4,
    "head_count_kv": 2,
    "key_length": null,
    "value_length": null,
    "expert_count": null,
    "expert_used_count": null,
    "rope": {
      "dimension_count": 64,
      "freq_base": 500000.0,
      "scaling_type": null,
      "scaling_factor": null,
      "original_context_length": null
    },
    "tokenizer": { "model": "gpt2", "vocab_size": 512 }
  },
  "quantization": [
    { "ggml_type": "q4_K", "tensors": 11, "elements": 983040, "bytes": 552960, "bits_per_weight": 4.5 }
  ],
  "warnings": []
}
```

(`quantization` abridged to its first entry; it lists every ggml type used,
largest share of bytes first. All examples here are real output on the
synthetic files of the test suite, reformatted more compactly.) `model` holds the well-known keys, `null` when
missing or mistyped; `context_length` there is the training context
(`[arch].context_length`). `file_type.name` is `null` for IDs this version
does not know.

Warning kinds: `missing_architecture`, `missing_quantization_version`,
`alignment_not_multiple_of_8`, `misaligned_tensors`, `data_out_of_bounds`,
`overlapping_tensors`. Each kind appears at most once.

## `metadata --json`

An array of entries in file order:

```json
[
  {
    "key": "tokenizer.ggml.tokens",
    "type": "array",
    "element_type": "string",
    "len": 512,
    "truncated": true,
    "value": ["tok_0", "tok_1", "tok_2", "tok_3", "tok_4", "tok_5", "tok_6", "tok_7"]
  },
  { "key": "tokenizer.ggml.bos_token_id", "type": "u32", "value": 0 }
]
```

`element_type` and `len` are present for arrays only, `truncated` only when
true. Nested arrays are JSON arrays of arrays; the element type of inner
arrays is not reported.

## `tensors --json`

An array in the order chosen by `--sort`:

```json
[
  {
    "name": "token_embd.weight",
    "ggml_type": "q4_K",
    "dims": [256, 512],
    "elements": 131072,
    "bytes": 73728,
    "offset": 0,
    "absolute_offset": 13728
  }
]
```

`dims` is in ggml order, innermost (contiguous) dimension first. `offset` is
relative to the data section as stored; `absolute_offset` is from the start
of the file.

## `kv-cache --json`

```json
{
  "file": "tiny.gguf",
  "architecture": "llama",
  "tokens_per_sequence": 1000,
  "sequences": 1,
  "cache_type": "f16",
  "block_count": 2,
  "head_count": 4,
  "head_count_kv": 2,
  "key_length": 64,
  "value_length": 64,
  "head_dim_source": "derived",
  "value_cache": true,
  "k_bytes_per_token": 512,
  "v_bytes_per_token": 512,
  "bytes_per_token": 1024,
  "total_bytes": 1024000,
  "notes": []
}
```

`*_bytes_per_token` are summed over layers. `head_dim_source` is `explicit`
(both head sizes in the file), `derived` (at least one computed as
`embedding_length / head_count`) or `unused` (no layer has KV heads).
`value_cache` is `false` for multi-head latent attention, where only the
compressed latent is cached, as K.

Note kinds: `sliding_window`, `latent_attention`,
`uncompressed_latent_attention`, `recurrent_state`, `layers_without_kv`.

## `diff --json`

From `diff tiny.gguf tiny-q5.gguf` in the CLI tests, abridged to one entry
of each list:

```json
{
  "old_file": "tiny.gguf",
  "new_file": "tiny-q5.gguf",
  "identical": false,
  "version": { "old": 3, "new": 2 },
  "alignment": null,
  "metadata": [
    {
      "change": "changed",
      "key": "general.name",
      "old": { "type": "string", "value": "Synthetic tiny" },
      "new": { "type": "string", "value": "Synthetic tiny, requantized" }
    }
  ],
  "tensors": [
    {
      "change": "changed",
      "name": "blk.0.attn_q.weight",
      "old": { "ggml_type": "q4_K", "dims": [256, 256] },
      "new": { "ggml_type": "q5_K", "dims": [256, 256] }
    }
  ]
}
```

`change` is `added` (only `new` present), `removed` (only `old`) or
`changed`. `version` and `alignment` are `null` when equal. Metadata values
carry `element_type`, `len` and `truncated` like `metadata --json`. The exit
status is 1 when the headers differ, as for text output.

## `types --json`

```json
[
  { "id": 0, "name": "f32", "block_size": 1, "type_size": 4, "bits_per_weight": 32.0, "quantized": false }
]
```

One object per known ggml type, in ID order. `type_size` is the size of one
block in bytes.
