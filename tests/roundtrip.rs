//! Property tests: whatever the test writer produces, the parser reads back.

use gguf_inspect::{GgmlType, GgufFile, Issue, MetadataArray, MetadataValue};
use gguf_testkit::GgufBuilder;
use proptest::collection::vec;
use proptest::prelude::*;

fn finite_f32() -> impl Strategy<Value = f32> {
    any::<f32>().prop_filter("NaN != NaN", |f| !f.is_nan())
}

fn finite_f64() -> impl Strategy<Value = f64> {
    any::<f64>().prop_filter("NaN != NaN", |f| !f.is_nan())
}

fn array() -> impl Strategy<Value = MetadataArray> {
    let n = 0..6;
    let leaf = prop_oneof![
        vec(any::<u8>(), n.clone()).prop_map(MetadataArray::U8),
        vec(any::<i8>(), n.clone()).prop_map(MetadataArray::I8),
        vec(any::<u16>(), n.clone()).prop_map(MetadataArray::U16),
        vec(any::<i16>(), n.clone()).prop_map(MetadataArray::I16),
        vec(any::<u32>(), n.clone()).prop_map(MetadataArray::U32),
        vec(any::<i32>(), n.clone()).prop_map(MetadataArray::I32),
        vec(finite_f32(), n.clone()).prop_map(MetadataArray::F32),
        vec(any::<bool>(), n.clone()).prop_map(MetadataArray::Bool),
        vec(".{0,12}", n.clone()).prop_map(MetadataArray::String),
        vec(any::<u64>(), n.clone()).prop_map(MetadataArray::U64),
        vec(any::<i64>(), n.clone()).prop_map(MetadataArray::I64),
        vec(finite_f64(), n).prop_map(MetadataArray::F64),
    ];
    leaf.prop_recursive(3, 24, 4, |inner| {
        vec(inner, 0..4).prop_map(MetadataArray::Array)
    })
}

fn value() -> impl Strategy<Value = MetadataValue> {
    prop_oneof![
        any::<u8>().prop_map(MetadataValue::U8),
        any::<i8>().prop_map(MetadataValue::I8),
        any::<u16>().prop_map(MetadataValue::U16),
        any::<i16>().prop_map(MetadataValue::I16),
        any::<u32>().prop_map(MetadataValue::U32),
        any::<i32>().prop_map(MetadataValue::I32),
        finite_f32().prop_map(MetadataValue::F32),
        any::<bool>().prop_map(MetadataValue::Bool),
        ".{0,40}".prop_map(MetadataValue::String),
        array().prop_map(MetadataValue::Array),
        any::<u64>().prop_map(MetadataValue::U64),
        any::<i64>().prop_map(MetadataValue::I64),
        finite_f64().prop_map(MetadataValue::F64),
    ]
}

#[derive(Debug, Clone)]
struct Tensor {
    dims: Vec<u64>,
    ggml_type: GgmlType,
}

fn tensor() -> impl Strategy<Value = Tensor> {
    (
        prop::sample::select(GgmlType::ALL),
        0_u64..3,
        vec(0_u64..4, 0..4),
    )
        .prop_map(|(ggml_type, blocks, rest)| {
            let mut dims = vec![blocks * ggml_type.block_size()];
            dims.extend(rest);
            Tensor { dims, ggml_type }
        })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    #[test]
    fn write_then_parse_round_trips(
        version in prop::sample::select(vec![2_u32, 3]),
        alignment in prop::option::of(prop::sample::select(vec![1_u32, 2, 8, 16, 32, 64, 256, 4096])),
        entries in vec(("[a-z_]{1,8}(\\.[a-z0-9_]{1,8}){0,3}", value()), 0..12),
        tensors in vec(tensor(), 0..8),
    ) {
        let mut builder = GgufBuilder::new().version(version);
        let mut expected = Vec::new();
        if let Some(a) = alignment {
            builder = builder.alignment(a);
            expected.push(("general.alignment".to_owned(), MetadataValue::U32(a)));
        }
        for (i, (key, value)) in entries.into_iter().enumerate() {
            let key = format!("k{i}.{key}"); // unique
            builder = builder.kv(&key, &value);
            expected.push((key, value));
        }
        for (i, t) in tensors.iter().enumerate() {
            builder = builder.tensor(&format!("t{i}"), &t.dims, t.ggml_type);
        }
        let layout = builder.layout();
        let bytes = builder.build();
        let gguf = GgufFile::from_bytes(&bytes).unwrap();

        prop_assert_eq!(gguf.version().number(), version);
        prop_assert_eq!(gguf.alignment(), alignment.unwrap_or(32));
        let parsed: Vec<_> = gguf.metadata().iter().map(|(k, v)| (k.to_owned(), v.clone())).collect();
        prop_assert_eq!(parsed, expected);

        prop_assert_eq!(gguf.tensors().len(), tensors.len());
        for ((parsed, written), &offset) in gguf.tensors().iter().zip(&tensors).zip(&layout.tensor_offsets) {
            prop_assert_eq!(parsed.dims(), written.dims.as_slice());
            prop_assert_eq!(parsed.ggml_type(), written.ggml_type);
            prop_assert_eq!(parsed.offset(), offset);
        }
        prop_assert_eq!(gguf.header_size(), layout.header_size);
        prop_assert_eq!(gguf.data_offset(), layout.data_offset);
        prop_assert_eq!(gguf.file_size(), bytes.len() as u64);

        let layout_issues: Vec<_> = gguf.validate().into_iter().filter(|i| matches!(
            i,
            Issue::DataOutOfBounds { .. } | Issue::OverlappingTensors { .. } | Issue::MisalignedTensors { .. }
        )).collect();
        prop_assert!(layout_issues.is_empty(), "{:?}", layout_issues);
    }
}

#[test]
fn nan_and_negative_zero_survive_bit_for_bit() {
    let bits = [f32::NAN.to_bits(), 0x7fc0_1234, (-0.0_f32).to_bits()];
    let values: Vec<f32> = bits.iter().map(|&b| f32::from_bits(b)).collect();
    let bytes = GgufBuilder::new()
        .kv("f", &MetadataValue::Array(MetadataArray::F32(values)))
        .kv(
            "d",
            &MetadataValue::F64(f64::from_bits(0x7ff8_dead_beef_0001)),
        )
        .build();
    let gguf = GgufFile::from_bytes(&bytes).unwrap();
    let Some(MetadataValue::Array(MetadataArray::F32(parsed))) = gguf.metadata().get("f") else {
        panic!("expected an f32 array");
    };
    assert_eq!(parsed.iter().map(|f| f.to_bits()).collect::<Vec<_>>(), bits);
    let Some(MetadataValue::F64(d)) = gguf.metadata().get("d") else {
        panic!("expected an f64");
    };
    assert_eq!(d.to_bits(), 0x7ff8_dead_beef_0001);
}
