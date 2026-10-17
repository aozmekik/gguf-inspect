//! Header parsing throughput on synthetic files.
//!
//! Run with `cargo bench --bench parse`. Inputs are generated, not real models:
//!
//! - `vocab_150k`: a 150 000-entry token array plus matching scores and token
//!   types, the bulk of a real header.
//! - `llama3_8b_shape`: the full header of an 8B Llama-3-shaped model
//!   (128 256-entry vocabulary, 291 tensor infos), parsed from memory and
//!   from a sparse file on disk (page cache warm after the first iteration).

use std::hint::black_box;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use gguf_inspect::{GgmlType, GgufFile, MetadataArray, MetadataValue};
use gguf_testkit::{GgufBuilder, SyntheticModel};

fn vocab_header(entries: usize) -> Vec<u8> {
    let tokens: Vec<String> = (0..entries).map(|i| format!("tok_{i}")).collect();
    #[allow(clippy::cast_precision_loss)]
    let scores: Vec<f32> = (0..entries).map(|i| 0.0 - i as f32).collect();
    GgufBuilder::new()
        .kv("general.architecture", &MetadataValue::String("llama".into()))
        .kv("tokenizer.ggml.model", &MetadataValue::String("gpt2".into()))
        .kv(
            "tokenizer.ggml.tokens",
            &MetadataValue::Array(MetadataArray::String(tokens)),
        )
        .kv(
            "tokenizer.ggml.scores",
            &MetadataValue::Array(MetadataArray::F32(scores)),
        )
        .kv(
            "tokenizer.ggml.token_type",
            &MetadataValue::Array(MetadataArray::I32(vec![1; entries])),
        )
        .tensor("token_embd.weight", &[4096, entries as u64], GgmlType::Q4_K)
        .header()
}

fn parse(c: &mut Criterion) {
    let mut group = c.benchmark_group("parse");

    let vocab = vocab_header(150_000);
    group.throughput(Throughput::Bytes(vocab.len() as u64));
    group.bench_function("vocab_150k/from_bytes", |b| {
        b.iter(|| GgufFile::from_bytes(black_box(&vocab)).expect("valid"));
    });

    let builder = SyntheticModel::llama3_8b_like().builder();
    let header = builder.header();
    group.throughput(Throughput::Bytes(header.len() as u64));
    group.bench_function("llama3_8b_shape/from_bytes", |b| {
        b.iter(|| GgufFile::from_bytes(black_box(&header)).expect("valid"));
    });

    let dir = tempfile::tempdir().expect("temp dir");
    let path = dir.path().join("llama3-8b-shape.gguf");
    builder.write_sparse(&path).expect("write sparse file");
    group.bench_function("llama3_8b_shape/open_file", |b| {
        b.iter(|| GgufFile::open(black_box(&path)).expect("valid"));
    });

    group.finish();
}

criterion_group!(benches, parse);
criterion_main!(benches);
