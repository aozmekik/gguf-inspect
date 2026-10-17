//! Header diffs.

use gguf_inspect::diff::{MetadataChange, TensorChange, diff};
use gguf_inspect::{GgmlType, GgufFile, GgufVersion, MetadataArray, MetadataValue};
use gguf_testkit::{GgufBuilder, SyntheticModel};

fn parse(builder: &GgufBuilder) -> GgufFile {
    GgufFile::from_bytes(&builder.build()).expect("valid synthetic file")
}

#[test]
fn identical_files_have_an_empty_diff() {
    let a = parse(&SyntheticModel::tiny().builder());
    let b = parse(&SyntheticModel::tiny().builder());
    let d = diff(&a, &b);
    assert!(d.is_empty(), "{d:?}");
}

#[test]
fn reports_added_removed_and_changed_entries_in_order() {
    let old = parse(
        &GgufBuilder::new()
            .version(2)
            .kv("kept", &MetadataValue::U32(1))
            .kv("removed", &MetadataValue::String("x".into()))
            .kv("retyped", &MetadataValue::U32(7))
            .kv(
                "vocab",
                &MetadataValue::Array(MetadataArray::String(vec!["a".into()])),
            )
            .tensor("same", &[32], GgmlType::F32)
            .tensor("requantized", &[256, 2], GgmlType::F16)
            .tensor("reshaped", &[4, 4], GgmlType::F32)
            .tensor("gone", &[1], GgmlType::F32),
    );
    let new = parse(
        &GgufBuilder::new()
            .alignment(64)
            .kv("kept", &MetadataValue::U32(1))
            .kv("retyped", &MetadataValue::U64(7))
            .kv(
                "vocab",
                &MetadataValue::Array(MetadataArray::String(vec!["a".into(), "b".into()])),
            )
            .kv("added", &MetadataValue::Bool(true))
            .tensor("new", &[1], GgmlType::F32)
            .tensor("same", &[32], GgmlType::F32)
            .tensor("requantized", &[256, 2], GgmlType::Q4_K)
            .tensor("reshaped", &[2, 8], GgmlType::F32),
    );
    let d = diff(&old, &new);
    assert_eq!(d.version, Some((GgufVersion::V2, GgufVersion::V3)));
    assert_eq!(d.alignment, Some((32, 64)));

    let keys: Vec<_> = d.metadata.iter().map(MetadataChange::key).collect();
    assert_eq!(
        keys,
        ["removed", "retyped", "vocab", "general.alignment", "added"]
    );
    assert!(matches!(
        d.metadata[1],
        MetadataChange::Changed {
            old: MetadataValue::U32(7),
            new: MetadataValue::U64(7),
            ..
        }
    ));
    assert!(matches!(d.metadata[0], MetadataChange::Removed { .. }));
    assert!(matches!(d.metadata[4], MetadataChange::Added { .. }));

    let names: Vec<_> = d.tensors.iter().map(TensorChange::name).collect();
    assert_eq!(names, ["requantized", "reshaped", "gone", "new"]);
    assert!(matches!(d.tensors[2], TensorChange::Removed(_)));
    assert!(matches!(d.tensors[3], TensorChange::Added(_)));
    let TensorChange::Changed { old, new } = d.tensors[0] else {
        panic!("expected a change");
    };
    assert_eq!(
        (old.ggml_type(), new.ggml_type()),
        (GgmlType::F16, GgmlType::Q4_K)
    );
}

#[test]
fn offsets_alone_do_not_count_as_changes() {
    let a = parse(&GgufBuilder::new().kv("a", &MetadataValue::U8(1)).tensor(
        "t",
        &[8],
        GgmlType::F32,
    ));
    let b = parse(
        &GgufBuilder::new()
            .kv("a", &MetadataValue::U8(1))
            .alignment(4096)
            .tensor("x", &[8], GgmlType::F32)
            .tensor("t", &[8], GgmlType::F32),
    );
    assert_ne!(a.tensors()[0].offset(), b.tensor("t").unwrap().offset());
    let d = diff(&a, &b);
    assert_eq!(d.tensors.len(), 1);
    assert_eq!(d.tensors[0].name(), "x");
}
