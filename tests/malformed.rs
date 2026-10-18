//! Malformed input must produce precise errors, never panics or huge allocations.

use gguf_inspect::{ErrorKind, GgmlType, GgufFile, Limits, MetadataArray, MetadataValue};
use gguf_testkit::{
    Endian, GgufBuilder, SyntheticModel, TensorSpec, every_value_type, put_u32, put_u64,
};
use proptest::prelude::*;

/// A small file that exercises every value type and several tensors.
fn sample() -> GgufBuilder {
    every_value_type()
        .iter()
        .fold(GgufBuilder::new().alignment(64), |b, (k, v)| b.kv(k, v))
        .tensor("token_embd.weight", &[256, 4], GgmlType::Q4_K)
        .tensor("norm", &[256], GgmlType::F32)
        .tensor("scalar", &[], GgmlType::I32)
}

fn parse(bytes: &[u8]) -> gguf_inspect::Result<GgufFile> {
    GgufFile::from_bytes(bytes)
}

fn header(version: u32, tensors: u64, kvs: u64) -> Vec<u8> {
    let mut b = b"GGUF".to_vec();
    put_u32(&mut b, version, Endian::Little);
    put_u64(&mut b, tensors, Endian::Little);
    put_u64(&mut b, kvs, Endian::Little);
    b
}

/// The type tag and value bytes of one entry.
fn typed(type_id: u32, payload: &[u8]) -> Vec<u8> {
    let mut b = type_id.to_le_bytes().to_vec();
    b.extend_from_slice(payload);
    b
}

#[test]
fn truncation_at_every_header_offset_is_an_error() {
    let builder = sample();
    let layout = builder.layout();
    let bytes = builder.build();
    for len in 0..usize::try_from(layout.header_size).unwrap() {
        let err = parse(&bytes[..len]).expect_err("truncated header must not parse");
        let offset = err.offset().expect("truncation errors carry an offset");
        assert!(
            offset <= len as u64,
            "len {len}: offset {offset} past the end: {err}"
        );
        assert!(
            matches!(
                err.kind(),
                ErrorKind::UnexpectedEof { .. } | ErrorKind::LengthExceedsFile { .. }
            ),
            "len {len}: {err}"
        );
    }
}

#[test]
fn truncation_in_the_data_section_is_reported_by_validate() {
    let builder = sample();
    let layout = builder.layout();
    let bytes = builder.build();
    for len in usize::try_from(layout.header_size).unwrap()..bytes.len() {
        let gguf = parse(&bytes[..len]).expect("header is complete");
        assert!(
            gguf.validate()
                .iter()
                .any(|i| matches!(i, gguf_inspect::Issue::DataOutOfBounds { .. })),
            "len {len}"
        );
    }
    assert!(
        !parse(&bytes)
            .unwrap()
            .validate()
            .iter()
            .any(|i| matches!(i, gguf_inspect::Issue::DataOutOfBounds { .. }))
    );
}

#[test]
fn bad_magic() {
    let mut bytes = sample().build();
    bytes[..4].copy_from_slice(b"GGML");
    let err = parse(&bytes).unwrap_err();
    assert!(matches!(err.kind(), ErrorKind::BadMagic(m) if m == b"GGML"));
    assert_eq!(err.offset(), Some(0));
    assert_eq!(
        err.to_string(),
        "not a GGUF file: expected magic \"GGUF\", found \"GGML\" at byte offset 0"
    );
    assert!(parse(b"").is_err());
    assert!(parse(&[0xff; 3]).is_err());
}

#[test]
fn big_endian_files_are_detected() {
    for version in [2, 3] {
        let bytes = SyntheticModel::tiny()
            .builder()
            .endian(Endian::Big)
            .version(version)
            .build();
        let err = parse(&bytes).unwrap_err();
        assert!(
            matches!(err.kind(), ErrorKind::BigEndian { version: v } if *v == version),
            "{err}"
        );
        assert_eq!(err.offset(), Some(4));
    }
}

#[test]
fn unsupported_versions() {
    for version in [0, 1, 4, 99] {
        let err = parse(&GgufBuilder::new().version(version).build()).unwrap_err();
        assert!(matches!(err.kind(), ErrorKind::UnsupportedVersion(v) if *v == version));
    }
}

#[test]
fn absurd_counts_are_rejected_before_allocating() {
    for (tensors, kvs) in [(u64::MAX, 0), (0, u64::MAX), (1 << 19, 0), (0, 1 << 19)] {
        let err = parse(&header(3, tensors, kvs)).unwrap_err();
        assert!(
            matches!(
                err.kind(),
                ErrorKind::LimitExceeded { .. } | ErrorKind::LengthExceedsFile { .. }
            ),
            "{err}"
        );
        let expected_offset = if tensors > 0 { 8 } else { 16 };
        assert_eq!(err.offset(), Some(expected_offset), "{err}");
    }
}

#[test]
fn absurd_lengths_are_rejected() {
    let cases: Vec<(&str, Vec<u8>)> = vec![
        ("string", typed(8, &u64::MAX.to_le_bytes())),
        ("u8 array", {
            let mut p = 0_u32.to_le_bytes().to_vec();
            p.extend_from_slice(&u64::MAX.to_le_bytes());
            typed(9, &p)
        }),
        ("string array", {
            let mut p = 8_u32.to_le_bytes().to_vec();
            p.extend_from_slice(&(1_u64 << 40).to_le_bytes());
            typed(9, &p)
        }),
        ("u32 array", {
            let mut p = 4_u32.to_le_bytes().to_vec();
            p.extend_from_slice(&(1_u64 << 20).to_le_bytes());
            typed(9, &p)
        }),
        ("nested array", {
            let mut p = 9_u32.to_le_bytes().to_vec();
            p.extend_from_slice(&(1_u64 << 30).to_le_bytes());
            typed(9, &p)
        }),
    ];
    for (name, payload) in cases {
        let bytes = GgufBuilder::new().raw_kv(b"k", payload).build();
        let err = parse(&bytes).unwrap_err();
        assert!(
            matches!(
                err.kind(),
                ErrorKind::LengthExceedsFile { .. } | ErrorKind::LimitExceeded { .. }
            ),
            "{name}: {err}"
        );
        // The length field follows: header (24) + key length (8) + key (1) + tag (4).
        let len_offset = if name == "string" { 37 } else { 41 };
        assert_eq!(err.offset(), Some(len_offset), "{name}: {err}");
        assert_eq!(err.context(), Some("metadata entry 0 `k`"), "{name}");
    }
}

#[test]
fn huge_key_lengths() {
    let mut bytes = header(3, 0, 1);
    bytes.extend_from_slice(&65_536_u64.to_le_bytes());
    bytes.extend(std::iter::repeat_n(b'a', 65_536));
    let err = parse(&bytes).unwrap_err();
    assert!(matches!(
        err.kind(),
        ErrorKind::LimitExceeded {
            value: 65_536,
            limit: 65_535,
            ..
        }
    ));
}

#[test]
fn configured_limits_apply() {
    let bytes = SyntheticModel::tiny().builder().build();
    let mut limits = Limits::default();
    limits.max_array_len = 100;
    let err = GgufFile::from_reader(bytes.as_slice(), bytes.len() as u64, &limits).unwrap_err();
    assert!(matches!(
        err.kind(),
        ErrorKind::LimitExceeded {
            what: "array length",
            value: 512,
            limit: 100
        }
    ));
    assert_eq!(
        err.context(),
        Some("metadata entry 14 `tokenizer.ggml.tokens`")
    );

    let mut limits = Limits::default();
    limits.max_string_len = 3;
    assert!(GgufFile::from_reader(bytes.as_slice(), bytes.len() as u64, &limits).is_err());

    let mut limits = Limits::default();
    limits.max_tensors = 2;
    let err = GgufFile::from_reader(bytes.as_slice(), bytes.len() as u64, &limits).unwrap_err();
    assert_eq!(err.offset(), Some(8));
}

#[test]
fn invalid_value_types() {
    for tag in [13, 0xFFFF_FFFF] {
        let bytes = GgufBuilder::new().raw_kv(b"k", typed(tag, &[])).build();
        let err = parse(&bytes).unwrap_err();
        assert!(matches!(err.kind(), ErrorKind::InvalidValueType(t) if *t == tag));
        assert_eq!(err.offset(), Some(33));
    }
    // Invalid element type inside an array.
    let mut payload = 42_u32.to_le_bytes().to_vec();
    payload.extend_from_slice(&0_u64.to_le_bytes());
    let bytes = GgufBuilder::new().raw_kv(b"k", typed(9, &payload)).build();
    let err = parse(&bytes).unwrap_err();
    assert!(matches!(err.kind(), ErrorKind::InvalidValueType(42)));
    assert_eq!(err.offset(), Some(37));
}

#[test]
fn invalid_bools() {
    let bytes = GgufBuilder::new().raw_kv(b"k", typed(7, &[2])).build();
    let err = parse(&bytes).unwrap_err();
    assert!(matches!(err.kind(), ErrorKind::InvalidBool(2)));
    assert_eq!(err.offset(), Some(37));

    let mut payload = 7_u32.to_le_bytes().to_vec();
    payload.extend_from_slice(&3_u64.to_le_bytes());
    payload.extend_from_slice(&[0, 1, 255]);
    let bytes = GgufBuilder::new().raw_kv(b"k", typed(9, &payload)).build();
    let err = parse(&bytes).unwrap_err();
    assert!(matches!(err.kind(), ErrorKind::InvalidBool(255)));
    assert_eq!(err.offset(), Some(37 + 12 + 2));
}

#[test]
fn invalid_utf8() {
    let bytes = GgufBuilder::new()
        .raw_kv(&[0xff, 0xfe], typed(0, &[1]))
        .build();
    let err = parse(&bytes).unwrap_err();
    assert!(matches!(err.kind(), ErrorKind::InvalidUtf8));
    assert_eq!(err.offset(), Some(32));

    // The offset points at the first invalid byte, after the valid prefix.
    let mut payload = 4_u64.to_le_bytes().to_vec();
    payload.extend_from_slice(b"ab\xc3\x28");
    let bytes = GgufBuilder::new().raw_kv(b"k", typed(8, &payload)).build();
    let err = parse(&bytes).unwrap_err();
    assert!(matches!(err.kind(), ErrorKind::InvalidUtf8));
    assert_eq!(err.offset(), Some(24 + 8 + 1 + 4 + 8 + 2));
}

#[test]
fn empty_and_duplicate_keys() {
    let bytes = GgufBuilder::new().raw_kv(b"", typed(0, &[1])).build();
    assert!(matches!(
        parse(&bytes).unwrap_err().kind(),
        ErrorKind::EmptyKey
    ));

    let builder = GgufBuilder::new()
        .kv("a", &MetadataValue::U8(1))
        .kv("a", &MetadataValue::U8(2));
    let err = parse(&builder.build()).unwrap_err();
    assert!(matches!(err.kind(), ErrorKind::DuplicateKey(k) if k == "a"));
    assert_eq!(err.offset(), Some(builder.layout().kv_offsets[1]));
}

#[test]
fn alignment_errors() {
    for bad in [0, 3, 48] {
        let builder = GgufBuilder::new()
            .kv("x", &MetadataValue::U8(0))
            .kv("general.alignment", &MetadataValue::U32(bad));
        let err = parse(&builder.build()).unwrap_err();
        assert!(matches!(err.kind(), ErrorKind::InvalidAlignment(a) if *a == bad));
        assert_eq!(err.offset(), Some(builder.layout().kv_offsets[1]));
    }
    let bytes = GgufBuilder::new()
        .kv("general.alignment", &MetadataValue::U64(32))
        .build();
    assert!(matches!(
        parse(&bytes).unwrap_err().kind(),
        ErrorKind::AlignmentType(gguf_inspect::ValueType::U64)
    ));
}

fn one_tensor(spec: TensorSpec) -> (Vec<u8>, u64) {
    let builder = GgufBuilder::new().tensor_spec(spec);
    let at = builder.layout().tensor_info_offsets[0];
    (builder.header(), at)
}

fn spec(name: &str, dims: Vec<u64>, type_id: u32) -> TensorSpec {
    TensorSpec {
        name: name.into(),
        dims,
        type_id,
        offset: None,
    }
}

#[test]
fn tensor_errors() {
    for id in [4, 31, 43, 1000] {
        let (bytes, at) = one_tensor(spec("t", vec![32], id));
        let err = parse(&bytes).unwrap_err();
        assert!(matches!(err.kind(), ErrorKind::UnknownGgmlType(t) if *t == id));
        assert_eq!(err.offset(), Some(at + 8 + 1 + 4 + 8));
        assert_eq!(err.context(), Some("tensor info 0 `t`"));
    }

    let (bytes, at) = one_tensor(spec("t", vec![1; 5], GgmlType::F32.id()));
    let err = parse(&bytes).unwrap_err();
    assert!(matches!(
        err.kind(),
        ErrorKind::TooManyDimensions { n_dims: 5, max: 4 }
    ));
    assert_eq!(err.offset(), Some(at + 9));

    let (bytes, at) = one_tensor(spec("t", vec![1 << 40, 1 << 40], GgmlType::F32.id()));
    let err = parse(&bytes).unwrap_err();
    assert!(matches!(err.kind(), ErrorKind::ElementCountOverflow(_)));
    assert_eq!(err.offset(), Some(at + 13));

    let (bytes, _) = one_tensor(spec("t", vec![100], GgmlType::Q4_K.id()));
    assert!(matches!(
        parse(&bytes).unwrap_err().kind(),
        ErrorKind::PartialBlock {
            ggml_type: GgmlType::Q4_K,
            ne0: 100
        }
    ));

    let long = "x".repeat(65);
    let (bytes, _) = one_tensor(spec(&long, vec![1], GgmlType::F32.id()));
    assert!(matches!(
        parse(&bytes).unwrap_err().kind(),
        ErrorKind::LimitExceeded { limit: 64, .. }
    ));
    let ok = "x".repeat(64);
    assert!(parse(&one_tensor(spec(&ok, vec![1], GgmlType::F32.id())).0).is_ok());

    let builder =
        GgufBuilder::new()
            .tensor("t", &[1], GgmlType::F32)
            .tensor("t", &[1], GgmlType::F32);
    let err = parse(&builder.build()).unwrap_err();
    assert!(matches!(err.kind(), ErrorKind::DuplicateTensorName(n) if n == "t"));
    assert_eq!(err.offset(), Some(builder.layout().tensor_info_offsets[1]));

    let mut absurd_offset = spec("t", vec![1], GgmlType::F32.id());
    absurd_offset.offset = Some(u64::MAX - 1);
    let (bytes, at) = one_tensor(absurd_offset);
    let err = parse(&bytes).unwrap_err();
    assert!(matches!(err.kind(), ErrorKind::OffsetOverflow));
    assert_eq!(err.offset(), Some(at + 8 + 1 + 4 + 8 + 4));
}

#[test]
fn deeply_nested_arrays_hit_the_depth_limit() {
    // 10 000 levels of array-of-array: must fail cleanly, not overflow the stack.
    let depth = 10_000;
    let mut payload = Vec::new();
    for _ in 0..depth {
        payload.extend_from_slice(&9_u32.to_le_bytes());
        payload.extend_from_slice(&1_u64.to_le_bytes());
    }
    payload.extend_from_slice(&0_u32.to_le_bytes());
    payload.extend_from_slice(&0_u64.to_le_bytes());
    let bytes = GgufBuilder::new().raw_kv(b"k", typed(9, &payload)).build();
    let err = parse(&bytes).unwrap_err();
    assert!(matches!(err.kind(), ErrorKind::NestingTooDeep { limit: 8 }));

    // Exactly at the limit is fine.
    let mut inner = MetadataArray::U8(vec![1]);
    for _ in 1..8 {
        inner = MetadataArray::Array(vec![inner]);
    }
    let bytes = GgufBuilder::new()
        .kv("k", &MetadataValue::Array(inner))
        .build();
    assert!(parse(&bytes).is_ok());
}

/// `array[array[...]]` nested `depth` levels deep around one `u8`.
fn nested(depth: u32) -> Vec<u8> {
    let mut payload = Vec::new();
    for _ in 1..depth {
        payload.extend_from_slice(&9_u32.to_le_bytes());
        payload.extend_from_slice(&1_u64.to_le_bytes());
    }
    payload.extend_from_slice(&0_u32.to_le_bytes());
    payload.extend_from_slice(&1_u64.to_le_bytes());
    payload.push(7);
    GgufBuilder::new().raw_kv(b"k", typed(9, &payload)).build()
}

#[test]
fn raising_the_depth_limit_cannot_overflow_the_stack() {
    // Without the ceiling, a debug build overflowed a 2 MiB stack at about
    // 400 levels. Run on a small stack to show the ceiling holds with room
    // to spare, including for clone, comparison and drop of the result.
    let mut limits = Limits::default();
    limits.max_array_depth = u32::MAX;
    let deep = nested(100_000);
    let at_ceiling = nested(Limits::MAX_ARRAY_DEPTH);
    let small_stack = std::thread::Builder::new().stack_size(1 << 20);
    small_stack
        .spawn(move || {
            let err =
                GgufFile::from_reader(deep.as_slice(), deep.len() as u64, &limits).unwrap_err();
            assert!(matches!(
                err.kind(),
                ErrorKind::NestingTooDeep { limit: 64 }
            ));

            let gguf =
                GgufFile::from_reader(at_ceiling.as_slice(), at_ceiling.len() as u64, &limits)
                    .unwrap();
            let value = gguf.metadata().get("k").unwrap().clone();
            assert!(value.bitwise_eq(gguf.metadata().get("k").unwrap()));
        })
        .unwrap()
        .join()
        .unwrap();
}

#[test]
fn errors_display_offset_and_context() {
    let bytes = GgufBuilder::new()
        .kv("general.name", &MetadataValue::String("x".into()))
        .raw_kv(b"flag", typed(7, &[9]))
        .build();
    let message = parse(&bytes).unwrap_err().to_string();
    assert!(message.starts_with("invalid bool value 9 (must be 0 or 1) at byte offset "));
    assert!(
        message.ends_with("(in metadata entry 1 `flag`)"),
        "{message}"
    );
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(512))]

    /// Overwriting random header bytes must never panic.
    #[test]
    fn random_mutations_never_panic(
        edits in proptest::collection::vec((any::<prop::sample::Index>(), any::<u8>()), 1..16)
    ) {
        let builder = sample();
        let mut bytes = builder.build();
        let header = usize::try_from(builder.layout().header_size).unwrap();
        for (index, byte) in edits {
            bytes[index.index(header)] = byte;
        }
        if let Ok(gguf) = parse(&bytes) {
            let _ = gguf.validate();
            let _ = gguf_inspect::ModelInfo::from_file(&gguf);
            let _ = gguf.type_breakdown();
        }
    }

    /// Arbitrary bytes after a valid magic and version must never panic.
    #[test]
    fn random_bodies_never_panic(body in proptest::collection::vec(any::<u8>(), 0..512)) {
        let mut bytes = b"GGUF\x03\0\0\0".to_vec();
        bytes.extend_from_slice(&body);
        let _ = parse(&bytes);
    }
}
