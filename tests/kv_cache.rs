//! KV-cache estimates against hand-computed values.

use gguf_inspect::kv_cache::{HeadDimSource, KvCacheConfig, KvCacheError, KvCacheNote, estimate};
use gguf_inspect::{GgmlType, GgufFile, MetadataArray, MetadataValue};
use gguf_testkit::{GgufBuilder, SyntheticModel};

fn file(entries: &[(&str, MetadataValue)]) -> GgufFile {
    let builder = entries
        .iter()
        .fold(GgufBuilder::new(), |b, (k, v)| b.kv(k, v));
    GgufFile::from_bytes(&builder.build()).expect("valid synthetic file")
}

fn llama(extra: &[(&str, MetadataValue)]) -> GgufFile {
    let mut entries = vec![
        (
            "general.architecture",
            MetadataValue::String("llama".into()),
        ),
        ("llama.block_count", MetadataValue::U32(32)),
        ("llama.embedding_length", MetadataValue::U32(4096)),
        ("llama.attention.head_count", MetadataValue::U32(32)),
    ];
    entries.extend(extra.iter().cloned());
    file(&entries)
}

#[test]
fn multi_head_attention_f16() {
    // No head_count_kv: every head has its own K/V. head_dim = 4096 / 32 = 128.
    let e = estimate(&llama(&[]), &KvCacheConfig::new(4096)).unwrap();
    assert_eq!(e.key_length, 128);
    assert_eq!(e.value_length, 128);
    assert_eq!(e.head_dim_source, HeadDimSource::Derived);
    // 32 layers × 32 heads × 128 × 2 bytes, for K and for V.
    assert_eq!(e.k_bytes_per_token, 32 * 32 * 128 * 2);
    assert_eq!(e.bytes_per_token(), 524_288);
    assert_eq!(e.total_bytes, 4096 * 524_288); // 2 GiB
    assert_eq!(e.notes, []);
}

#[test]
fn grouped_query_attention_divides_the_cache() {
    let gqa = llama(&[("llama.attention.head_count_kv", MetadataValue::U32(8))]);
    let e = estimate(&gqa, &KvCacheConfig::new(8192)).unwrap();
    // 8 KV heads instead of 32: a quarter of the MHA size.
    assert_eq!(e.bytes_per_token(), 524_288 / 4);
    assert_eq!(e.total_bytes, 8192 * 131_072); // 1 GiB, as for Llama-3-8B at 8k context
}

#[test]
fn quantized_cache_types_use_ggml_row_sizes() {
    let gqa = llama(&[("llama.attention.head_count_kv", MetadataValue::U32(8))]);
    let config = KvCacheConfig::new(1000).with_sequences(3);
    let q8 = estimate(&gqa, &config.with_cache_type(GgmlType::Q8_0)).unwrap();
    // Row of 8 × 128 = 1024 elements = 32 blocks of 34 bytes.
    assert_eq!(q8.k_bytes_per_token, 32 * 1088);
    assert_eq!(q8.total_bytes, 3 * 1000 * 32 * 2 * 1088);
    let q4 = estimate(&gqa, &config.with_cache_type(GgmlType::Q4_0)).unwrap();
    assert_eq!(q4.k_bytes_per_token, 32 * 32 * 18);
}

#[test]
fn explicit_head_sizes_win() {
    let f = llama(&[
        ("llama.attention.head_count_kv", MetadataValue::U32(4)),
        ("llama.attention.key_length", MetadataValue::U32(192)),
        ("llama.attention.value_length", MetadataValue::U32(128)),
    ]);
    let e = estimate(&f, &KvCacheConfig::new(1)).unwrap();
    assert_eq!(e.head_dim_source, HeadDimSource::Explicit);
    assert_eq!(e.k_bytes_per_token, 32 * 4 * 192 * 2);
    assert_eq!(e.v_bytes_per_token, 32 * 4 * 128 * 2);
}

#[test]
fn per_layer_kv_heads_are_summed() {
    let f = file(&[
        (
            "general.architecture",
            MetadataValue::String("hybrid".into()),
        ),
        ("hybrid.block_count", MetadataValue::U32(4)),
        ("hybrid.attention.key_length", MetadataValue::U32(64)),
        ("hybrid.attention.value_length", MetadataValue::U32(64)),
        (
            "hybrid.attention.head_count_kv",
            MetadataValue::Array(MetadataArray::I32(vec![0, 8, 0, 4])),
        ),
        ("hybrid.ssm.state_size", MetadataValue::U32(16)),
    ]);
    let e = estimate(&f, &KvCacheConfig::new(10)).unwrap();
    assert_eq!(e.k_bytes_per_token, (8 + 4) * 64 * 2);
    assert_eq!(e.total_bytes, 10 * 2 * 12 * 64 * 2);
    assert_eq!(
        e.notes,
        [KvCacheNote::RecurrentState, KvCacheNote::LayersWithoutKv(2)]
    );
}

#[test]
fn notes_flag_unmodelled_attention() {
    let f = llama(&[
        ("llama.attention.sliding_window", MetadataValue::U32(4096)),
        ("llama.attention.kv_lora_rank", MetadataValue::U32(512)),
    ]);
    let e = estimate(&f, &KvCacheConfig::new(1)).unwrap();
    assert_eq!(
        e.notes,
        [
            KvCacheNote::SlidingWindow(4096),
            KvCacheNote::UncompressedLatentAttention(512)
        ]
    );
}

#[test]
fn synthetic_llama3_shape() {
    let model = SyntheticModel::tiny();
    let gguf = GgufFile::from_bytes(&model.builder().build()).unwrap();
    let e = estimate(&gguf, &KvCacheConfig::new(model.context_length)).unwrap();
    let head_dim = model.embedding_length / model.head_count;
    assert_eq!(
        e.total_bytes,
        model.context_length * model.block_count * 2 * model.head_count_kv * head_dim * 2
    );
}

#[test]
fn latent_attention_caches_the_latent_as_k_only() {
    // DeepSeek-V3-shaped, as written by current converters: key_length and
    // value_length describe the compressed latent (kv_lora_rank 512 + 64 rope
    // dimensions), with one KV head. llama.cpp allocates no V cache.
    let f = file(&[
        (
            "general.architecture",
            MetadataValue::String("deepseek2".into()),
        ),
        ("deepseek2.block_count", MetadataValue::U32(61)),
        ("deepseek2.embedding_length", MetadataValue::U32(7168)),
        ("deepseek2.attention.head_count", MetadataValue::U32(128)),
        ("deepseek2.attention.head_count_kv", MetadataValue::U32(1)),
        ("deepseek2.attention.key_length", MetadataValue::U32(576)),
        ("deepseek2.attention.value_length", MetadataValue::U32(512)),
        (
            "deepseek2.attention.key_length_mla",
            MetadataValue::U32(192),
        ),
        (
            "deepseek2.attention.value_length_mla",
            MetadataValue::U32(128),
        ),
        ("deepseek2.attention.kv_lora_rank", MetadataValue::U32(512)),
    ]);
    let e = estimate(&f, &KvCacheConfig::new(8192)).unwrap();
    assert!(!e.has_value_cache);
    assert_eq!(e.k_bytes_per_token, 61 * 576 * 2);
    assert_eq!(e.v_bytes_per_token, 0);
    assert_eq!(e.total_bytes, 575_668_224); // 61 x 8192 x 1152 B
    assert_eq!(
        e.notes,
        [KvCacheNote::LatentAttention {
            key_length_mla: 192,
            value_length_mla: 128
        }]
    );
}

#[test]
fn older_latent_attention_files_cache_full_k_and_v() {
    // The older DeepSeek-V2 layout has no *_mla keys; llama.cpp then treats the
    // model as multi-head attention with per-head K (192) and V (128).
    let f = file(&[
        (
            "general.architecture",
            MetadataValue::String("deepseek2".into()),
        ),
        ("deepseek2.block_count", MetadataValue::U32(2)),
        ("deepseek2.attention.head_count", MetadataValue::U32(16)),
        ("deepseek2.attention.key_length", MetadataValue::U32(192)),
        ("deepseek2.attention.value_length", MetadataValue::U32(128)),
        ("deepseek2.attention.kv_lora_rank", MetadataValue::U32(512)),
    ]);
    let e = estimate(&f, &KvCacheConfig::new(1)).unwrap();
    assert!(e.has_value_cache);
    assert_eq!(e.k_bytes_per_token, 2 * 16 * 192 * 2);
    assert_eq!(e.v_bytes_per_token, 2 * 16 * 128 * 2);
    assert_eq!(e.notes, [KvCacheNote::UncompressedLatentAttention(512)]);
}

#[test]
fn recurrent_models_have_no_kv_cache() {
    // Mamba as written by llama.cpp's converter: head_count 0, no head_count_kv.
    let f = file(&[
        (
            "general.architecture",
            MetadataValue::String("mamba".into()),
        ),
        ("mamba.block_count", MetadataValue::U32(64)),
        ("mamba.embedding_length", MetadataValue::U32(2560)),
        ("mamba.attention.head_count", MetadataValue::U32(0)),
        ("mamba.ssm.state_size", MetadataValue::U32(16)),
    ]);
    let e = estimate(&f, &KvCacheConfig::new(8192)).unwrap();
    assert_eq!(e.total_bytes, 0);
    assert_eq!((e.key_length, e.value_length), (0, 0));
    assert_eq!(e.head_dim_source, HeadDimSource::Unused);
    assert_eq!(
        e.notes,
        [
            KvCacheNote::RecurrentState,
            KvCacheNote::LayersWithoutKv(64)
        ]
    );
}

#[test]
fn hybrid_head_count_lists_derive_from_attention_layers() {
    let f = file(&[
        ("general.architecture", MetadataValue::String("h".into())),
        ("h.block_count", MetadataValue::U32(4)),
        ("h.embedding_length", MetadataValue::U32(4096)),
        (
            "h.attention.head_count",
            MetadataValue::Array(MetadataArray::U32(vec![0, 32, 0, 32])),
        ),
        (
            "h.attention.head_count_kv",
            MetadataValue::Array(MetadataArray::U32(vec![0, 8, 0, 8])),
        ),
    ]);
    let e = estimate(&f, &KvCacheConfig::new(1)).unwrap();
    assert_eq!(e.key_length, 128);
    assert_eq!(e.k_bytes_per_token, 2 * 8 * 128 * 2);
    assert_eq!(e.notes, [KvCacheNote::LayersWithoutKv(2)]);
}

#[test]
fn zero_layers_give_an_empty_cache() {
    let f = file(&[
        ("general.architecture", MetadataValue::String("x".into())),
        ("x.block_count", MetadataValue::U32(0)),
        ("x.attention.head_count", MetadataValue::U32(8)),
        ("x.embedding_length", MetadataValue::U32(512)),
    ]);
    let e = estimate(&f, &KvCacheConfig::new(4096)).unwrap();
    assert_eq!((e.key_length, e.total_bytes), (64, 0));
}

#[test]
fn errors() {
    let config = KvCacheConfig::new(1);
    assert_eq!(
        estimate(&file(&[]), &config),
        Err(KvCacheError::MissingArchitecture)
    );
    let no_layers = file(&[("general.architecture", MetadataValue::String("x".into()))]);
    assert_eq!(
        estimate(&no_layers, &config),
        Err(KvCacheError::MissingKey("x.block_count".into()))
    );

    let zero_heads = file(&[
        ("general.architecture", MetadataValue::String("x".into())),
        ("x.block_count", MetadataValue::U32(1)),
        ("x.embedding_length", MetadataValue::U32(64)),
        ("x.attention.head_count", MetadataValue::U32(0)),
        ("x.attention.head_count_kv", MetadataValue::U32(8)),
    ]);
    assert_eq!(
        estimate(&zero_heads, &config),
        Err(KvCacheError::ZeroHeads("x.attention.head_count".into()))
    );

    let mismatch = llama(&[(
        "llama.attention.head_count_kv",
        MetadataValue::Array(MetadataArray::U32(vec![8, 4])),
    )]);
    assert!(matches!(
        estimate(&mismatch, &config),
        Err(KvCacheError::LayerCountMismatch {
            expected: 32,
            found: 2,
            ..
        })
    ));

    let varying = file(&[
        ("general.architecture", MetadataValue::String("x".into())),
        ("x.block_count", MetadataValue::U32(2)),
        ("x.embedding_length", MetadataValue::U32(4096)),
        (
            "x.attention.head_count",
            MetadataValue::Array(MetadataArray::U32(vec![32, 16])),
        ),
    ]);
    assert_eq!(
        estimate(&varying, &config),
        Err(KvCacheError::VaryingHeadCount(
            "x.attention.head_count".into()
        ))
    );

    let tiny_embedding = file(&[
        ("general.architecture", MetadataValue::String("x".into())),
        ("x.block_count", MetadataValue::U32(2)),
        ("x.embedding_length", MetadataValue::U32(64)),
        ("x.attention.head_count", MetadataValue::U32(128)),
    ]);
    assert_eq!(
        estimate(&tiny_embedding, &config),
        Err(KvCacheError::ZeroHeadSize {
            embedding_length: 64,
            head_count: 128
        })
    );

    let odd = llama(&[("llama.attention.head_count_kv", MetadataValue::U32(1))]);
    // A 128-element row is not a whole number of 256-element q4_K blocks.
    assert_eq!(
        estimate(&odd, &config.with_cache_type(GgmlType::Q4_K)),
        Err(KvCacheError::PartialBlock {
            cache_type: GgmlType::Q4_K,
            row_elements: 128
        })
    );

    let huge = KvCacheConfig::new(u64::MAX);
    assert_eq!(estimate(&llama(&[]), &huge), Err(KvCacheError::Overflow));

    for err in [
        KvCacheError::MissingArchitecture,
        KvCacheError::VaryingHeadCount("k".into()),
        KvCacheError::Overflow,
    ] {
        assert_ne!(err.to_string(), "");
    }
}

/// A key that is present but unusable is an error, never replaced by a default.
#[test]
fn present_but_invalid_keys_are_errors() {
    let invalid = |key: &str, value: MetadataValue, found: &str| {
        let full = format!("llama.{key}");
        let f = file(&[
            (
                "general.architecture",
                MetadataValue::String("llama".into()),
            ),
            ("llama.block_count", MetadataValue::U32(32)),
            ("llama.embedding_length", MetadataValue::U32(4096)),
            ("llama.attention.head_count", MetadataValue::U32(32)),
            (full.as_str(), value),
        ]);
        let Err(KvCacheError::InvalidKey {
            key: k, found: f, ..
        }) = estimate(&f, &KvCacheConfig::new(1))
        else {
            panic!("{full}: expected InvalidKey");
        };
        assert_eq!((k.as_str(), f.as_str()), (full.as_str(), found));
    };
    invalid("attention.head_count_kv", MetadataValue::I32(-8), "i32 -8");
    invalid("attention.head_count_kv", MetadataValue::F32(8.0), "f32 8");
    invalid(
        "attention.head_count_kv",
        MetadataValue::Array(MetadataArray::Array(vec![MetadataArray::U32(vec![8])])),
        "array[array]",
    );
    invalid(
        "attention.head_count_kv",
        MetadataValue::Array(MetadataArray::I32(vec![8, -8])),
        "array[i32] with a negative entry",
    );
    invalid(
        "attention.key_length",
        MetadataValue::String("128".into()),
        "a string",
    );
    invalid(
        "attention.value_length",
        MetadataValue::Array(MetadataArray::U32(vec![128])),
        "array[u32]",
    );
    invalid(
        "attention.key_length_mla",
        MetadataValue::Bool(true),
        "bool true",
    );

    // block_count itself, and a uniform-looking list of the wrong length.
    let f = file(&[
        (
            "general.architecture",
            MetadataValue::String("llama".into()),
        ),
        ("llama.block_count", MetadataValue::I64(-1)),
    ]);
    assert!(matches!(
        estimate(&f, &KvCacheConfig::new(1)),
        Err(KvCacheError::InvalidKey { .. })
    ));
    let same_values = llama(&[(
        "llama.attention.head_count_kv",
        MetadataValue::Array(MetadataArray::U32(vec![8, 8])),
    )]);
    assert_eq!(
        estimate(&same_values, &KvCacheConfig::new(1)),
        Err(KvCacheError::LayerCountMismatch {
            key: "llama.attention.head_count_kv".into(),
            expected: 32,
            found: 2
        })
    );
}
