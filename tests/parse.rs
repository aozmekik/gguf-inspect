//! Parsing well-formed synthetic files.

use gguf_inspect::{
    GgmlType, GgufFile, GgufVersion, Issue, MetadataArray, MetadataValue, ModelInfo, TypeStats,
    ValueType,
};
use gguf_testkit::{GgufBuilder, SyntheticModel, TensorSpec, every_value_type};

fn with_every_type(builder: GgufBuilder) -> GgufBuilder {
    every_value_type()
        .iter()
        .fold(builder, |b, (key, value)| b.kv(key, value))
}

#[test]
fn every_metadata_type_in_both_versions() {
    for (version, expected) in [(2, GgufVersion::V2), (3, GgufVersion::V3)] {
        let bytes = with_every_type(GgufBuilder::new().version(version)).build();
        let gguf = GgufFile::from_bytes(&bytes).unwrap();
        assert_eq!(gguf.version(), expected);
        let parsed: Vec<_> = gguf
            .metadata()
            .iter()
            .map(|(k, v)| (k.to_owned(), v.clone()))
            .collect();
        let expected: Vec<_> = every_value_type()
            .into_iter()
            .map(|(k, v)| (k.to_owned(), v))
            .collect();
        assert_eq!(parsed, expected);
    }
}

#[test]
fn covers_all_thirteen_value_types() {
    let mut seen: Vec<ValueType> = every_value_type()
        .iter()
        .map(|(_, v)| v.value_type())
        .collect();
    seen.sort();
    seen.dedup();
    assert_eq!(seen, ValueType::ALL);
}

#[test]
fn metadata_offsets_point_at_entries() {
    let builder = with_every_type(GgufBuilder::new());
    let layout = builder.layout();
    let gguf = GgufFile::from_bytes(&builder.build()).unwrap();
    for ((key, _), &offset) in gguf.metadata().iter().zip(&layout.kv_offsets) {
        assert_eq!(gguf.metadata().offset_of(key), Some(offset), "{key}");
    }
}

#[test]
fn alignment_variants() {
    for alignment in [None, Some(1), Some(8), Some(32), Some(64), Some(4096)] {
        let mut builder = GgufBuilder::new();
        if let Some(a) = alignment {
            builder = builder.alignment(a);
        }
        builder = builder
            .tensor("a", &[3], GgmlType::F32)
            .tensor("b", &[32, 2], GgmlType::Q4_0)
            .tensor("c", &[5], GgmlType::I8);
        let layout = builder.layout();
        let gguf = GgufFile::from_bytes(&builder.build()).unwrap();

        let a = alignment.unwrap_or(32);
        assert_eq!(gguf.alignment(), a);
        assert_eq!(gguf.header_size(), layout.header_size);
        assert_eq!(gguf.data_offset(), layout.data_offset);
        assert_eq!(gguf.data_offset() % u64::from(a), 0);
        for (t, &offset) in gguf.tensors().iter().zip(&layout.tensor_offsets) {
            assert_eq!(t.offset(), offset);
            assert_eq!(t.absolute_offset(), layout.data_offset + offset);
        }
        let issues = gguf.validate();
        let expect_issue = alignment.is_some_and(|a| a % 8 != 0);
        assert_eq!(
            issues.contains(&Issue::AlignmentNotMultipleOf8(a)),
            expect_issue,
            "{issues:?}"
        );
    }
}

#[test]
fn tensor_infos() {
    let bytes = GgufBuilder::new()
        .kv("general.architecture", &MetadataValue::String("x".into()))
        .tensor("scalar", &[], GgmlType::F32)
        .tensor("matrix", &[256, 3], GgmlType::Q4_K)
        .tensor("four_d", &[2, 3, 4, 5], GgmlType::BF16)
        .tensor("empty", &[0, 7], GgmlType::F16)
        .build();
    let gguf = GgufFile::from_bytes(&bytes).unwrap();

    let matrix = gguf.tensor("matrix").unwrap();
    assert_eq!(matrix.dims(), [256, 3]);
    assert_eq!(matrix.ggml_type(), GgmlType::Q4_K);
    assert_eq!(matrix.n_elements(), 768);
    assert_eq!(matrix.n_bytes(), 3 * 144);

    assert_eq!(gguf.tensor("scalar").unwrap().n_elements(), 1);
    assert_eq!(gguf.tensor("four_d").unwrap().n_bytes(), 120 * 2);
    assert_eq!(gguf.tensor("empty").unwrap().n_bytes(), 0);
    assert!(gguf.tensor("missing").is_none());

    assert_eq!(gguf.parameter_count(), 1 + 768 + 120);
    assert_eq!(gguf.tensor_data_size(), 4 + 432 + 240);
    assert_eq!(gguf.file_size(), bytes.len() as u64);
    // Quantized tensors without general.quantization_version.
    assert_eq!(gguf.validate(), [Issue::MissingQuantizationVersion]);
}

#[test]
fn type_breakdown_sorts_by_bytes() {
    let bytes = GgufBuilder::new()
        .tensor("a", &[32], GgmlType::F32)
        .tensor("b", &[256, 4], GgmlType::Q6_K)
        .tensor("c", &[8], GgmlType::F32)
        .build();
    let gguf = GgufFile::from_bytes(&bytes).unwrap();
    assert_eq!(
        gguf.type_breakdown(),
        [
            TypeStats {
                ggml_type: GgmlType::Q6_K,
                tensors: 1,
                elements: 1024,
                bytes: 840,
            },
            TypeStats {
                ggml_type: GgmlType::F32,
                tensors: 2,
                elements: 40,
                bytes: 160,
            },
        ]
    );
}

#[test]
fn synthetic_llama_model_summary() {
    let model = SyntheticModel::tiny();
    let gguf = GgufFile::from_bytes(&model.builder().build()).unwrap();
    assert_eq!(gguf.validate(), []);
    assert_eq!(gguf.parameter_count(), model.parameter_count());
    assert_eq!(gguf.tensors().len() as u64, 3 + 9 * model.block_count);

    let info = ModelInfo::from_file(&gguf);
    assert_eq!(info.architecture.as_deref(), Some("llama"));
    assert_eq!(info.name.as_deref(), Some(model.name.as_str()));
    assert_eq!(info.context_length, Some(model.context_length));
    assert_eq!(info.embedding_length, Some(model.embedding_length));
    assert_eq!(info.block_count, Some(model.block_count));
    assert_eq!(
        info.head_count_kv.and_then(|h| h.uniform()),
        Some(model.head_count_kv)
    );
    assert_eq!(info.rope.freq_base, Some(500_000.0));
    assert_eq!(info.tokenizer.model.as_deref(), Some("gpt2"));
    assert_eq!(info.tokenizer.vocab_size, Some(model.vocab_size));
    assert_eq!(info.file_type, Some(15));
}

#[test]
fn reads_from_disk_and_stops_at_the_header() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("sparse.gguf");
    let builder = GgufBuilder::new()
        .kv(
            "general.architecture",
            &MetadataValue::String("llama".into()),
        )
        .tensor("big", &[1 << 18, 1 << 10], GgmlType::F16); // 512 MiB, never written
    let layout = builder.write_sparse(&path).unwrap();
    let gguf = GgufFile::open(&path).unwrap();
    assert_eq!(gguf.file_size(), layout.file_size);
    assert_eq!(gguf.file_size(), layout.data_offset + (512 << 20));
    assert_eq!(gguf.validate(), []);
}

#[test]
fn validate_reports_data_section_problems() {
    let spec = |name: &str, offset| TensorSpec {
        name: name.into(),
        dims: vec![64],
        type_id: GgmlType::F32.id(),
        offset: Some(offset),
    };
    let builder = GgufBuilder::new()
        .kv("general.architecture", &MetadataValue::String("x".into()))
        .tensor_spec(spec("first", 0))
        .tensor_spec(spec("overlaps", 128))
        .tensor_spec(spec("misaligned", 300));
    let layout = builder.layout();
    let mut bytes = builder.build();
    bytes.truncate(usize::try_from(layout.data_offset).unwrap() + 100);
    let gguf = GgufFile::from_bytes(&bytes).unwrap();
    let issues = gguf.validate();
    assert!(issues.contains(&Issue::MisalignedTensor {
        tensor: "misaligned".into(),
        offset: 300,
        alignment: 32,
    }));
    assert!(issues.contains(&Issue::OverlappingTensors {
        first: "first".into(),
        second: "overlaps".into(),
    }));
    assert!(issues.contains(&Issue::DataOutOfBounds {
        tensors: 3,
        first: "first".into(),
        data_end: layout.data_offset + 300 + 256,
        file_size: bytes.len() as u64,
    }));
    for issue in &issues {
        assert_ne!(issue.to_string(), "");
    }
}

#[test]
fn per_layer_head_counts_are_exposed() {
    let bytes = GgufBuilder::new()
        .kv(
            "general.architecture",
            &MetadataValue::String("hybrid".into()),
        )
        .kv(
            "hybrid.attention.head_count_kv",
            &MetadataValue::Array(MetadataArray::U32(vec![0, 8, 0, 8])),
        )
        .build();
    let info = ModelInfo::from_file(&GgufFile::from_bytes(&bytes).unwrap());
    assert_eq!(info.head_count_kv.unwrap().range(), (0, 8));
}
