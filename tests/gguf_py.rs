//! Files written by gguf-py, llama.cpp's reference GGUF writer, rather than by
//! this crate's own test writer. See `tests/fixtures/gguf-py/README.md` for how
//! they were made; everything checked here is what `generate.py` wrote.

use std::path::PathBuf;

use gguf_inspect::{ErrorKind, GgmlType, GgufFile, GgufVersion, MetadataArray, MetadataValue};

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/gguf-py")
        .join(name)
}

/// The metadata `generate.py` writes, in order, after `general.architecture`
/// (and `general.alignment`, when set).
fn expected_metadata() -> Vec<(&'static str, MetadataValue)> {
    use MetadataValue as V;
    let strings = |v: &[&str]| MetadataArray::String(v.iter().map(|s| (*s).to_owned()).collect());
    vec![
        (
            "general.name",
            V::String("Synthetic gguf-py fixture".into()),
        ),
        ("general.quantization_version", V::U32(2)),
        ("general.file_type", V::U32(7)),
        ("llama.context_length", V::U32(4096)),
        ("llama.embedding_length", V::U32(32)),
        ("llama.block_count", V::U32(2)),
        ("llama.attention.head_count", V::U32(4)),
        ("llama.attention.head_count_kv", V::U32(2)),
        ("llama.rope.freq_base", V::F32(10000.0)),
        ("test.u8", V::U8(200)),
        ("test.i8", V::I8(-100)),
        ("test.u16", V::U16(60_000)),
        ("test.i16", V::I16(-30_000)),
        ("test.u32", V::U32(4_000_000_000)),
        ("test.i32", V::I32(-2_000_000_000)),
        ("test.f32", V::F32(0.1)),
        ("test.bool", V::Bool(true)),
        ("test.string", V::String("héllo, wörld".into())),
        ("test.u64", V::U64(u64::MAX)),
        ("test.i64", V::I64(i64::MIN)),
        ("test.f64", V::F64(-1.5e300)),
        ("tokenizer.ggml.model", V::String("llama".into())),
        (
            "tokenizer.ggml.tokens",
            V::Array(strings(&["<unk>", "<s>", "</s>", "\u{2581}a", "\u{2581}b"])),
        ),
        (
            "tokenizer.ggml.scores",
            V::Array(MetadataArray::F32(vec![0.0, 0.0, 0.0, -1.0, -2.0])),
        ),
        (
            "tokenizer.ggml.token_type",
            V::Array(MetadataArray::I32(vec![2, 3, 3, 1, 1])),
        ),
        (
            "test.array.u16",
            V::Array(MetadataArray::U16(vec![1, 2, 65_535])),
        ),
        (
            "test.array.f64",
            V::Array(MetadataArray::F64(vec![0.5, -0.25])),
        ),
        (
            "test.array.bools",
            V::Array(MetadataArray::Bool(vec![true, false, true])),
        ),
        (
            "test.array.nested",
            V::Array(MetadataArray::Array(vec![
                MetadataArray::I32(vec![1, 2]),
                MetadataArray::I32(vec![3]),
                strings(&["x", "y"]),
            ])),
        ),
    ]
}

/// The raw bytes of `n` values of `W` bytes each, starting at `offset`.
fn read_le<const W: usize>(bytes: &[u8], offset: u64, n: usize) -> Vec<[u8; W]> {
    let start = usize::try_from(offset).expect("offset fits in usize");
    bytes[start..start + n * W].as_chunks::<W>().0.to_vec()
}

fn check(name: &str, alignment: u32, relative_offsets: [u64; 5]) {
    let path = fixture(name);
    let bytes = std::fs::read(&path).expect("fixture is readable");
    let gguf = GgufFile::open(&path).expect("fixture parses");
    assert_eq!(gguf.version(), GgufVersion::V3);
    assert_eq!(gguf.alignment(), alignment);
    assert_eq!(gguf.file_size(), bytes.len() as u64);
    assert_eq!(gguf.data_offset() % u64::from(alignment), 0);
    assert_eq!(gguf.validate(), []);

    let mut expected = vec![(
        "general.architecture",
        MetadataValue::String("llama".into()),
    )];
    if alignment != 32 {
        expected.push(("general.alignment", MetadataValue::U32(alignment)));
    }
    expected.extend(expected_metadata());
    let parsed: Vec<_> = gguf.metadata().iter().collect();
    assert_eq!(parsed.len(), expected.len());
    for ((key, value), (want_key, want)) in parsed.iter().zip(&expected) {
        assert_eq!(key, want_key);
        assert!(value.bitwise_eq(want), "{key}: {value:?} != {want:?}");
    }

    // gguf-py writes numpy shapes reversed, innermost dimension first.
    let tensors: Vec<_> = gguf
        .tensors()
        .iter()
        .map(|t| (t.name(), t.dims().to_vec(), t.ggml_type(), t.offset()))
        .collect();
    assert_eq!(
        tensors,
        [
            (
                "token_embd.weight",
                vec![32, 5],
                GgmlType::F32,
                relative_offsets[0]
            ),
            (
                "blk.0.attn_norm.weight",
                vec![32],
                GgmlType::F16,
                relative_offsets[1]
            ),
            (
                "blk.0.attn_q.weight",
                vec![32, 2],
                GgmlType::Q8_0,
                relative_offsets[2]
            ),
            (
                "blk.1.ffn_down.weight",
                vec![2, 3],
                GgmlType::I32,
                relative_offsets[3]
            ),
            ("rope_freqs", vec![], GgmlType::F64, relative_offsets[4]),
        ]
    );
    assert_eq!(gguf.parameter_count(), 160 + 32 + 64 + 6 + 1);

    // The data gguf-py wrote is where the computed offsets say it is.
    let at = |name: &str| gguf.tensor(name).expect("tensor exists").absolute_offset();
    // Compared bit for bit: these are the exact values gguf-py wrote.
    let embd: Vec<u32> = read_le::<4>(&bytes, at("token_embd.weight"), 160)
        .into_iter()
        .map(u32::from_le_bytes)
        .collect();
    let arange: Vec<u32> = (0..160_u16).map(|i| f32::from(i).to_bits()).collect();
    assert_eq!(embd, arange);
    let norm = read_le::<2>(&bytes, at("blk.0.attn_norm.weight"), 32);
    assert!(
        norm.iter().all(|&h| u16::from_le_bytes(h) == 0x3c00),
        "f16 1.0"
    );
    let down: Vec<i32> = read_le::<4>(&bytes, at("blk.1.ffn_down.weight"), 6)
        .into_iter()
        .map(i32::from_le_bytes)
        .collect();
    assert_eq!(down, [0, 1, 2, 3, 4, 5]);
    let rope = read_le::<8>(&bytes, at("rope_freqs"), 1);
    assert_eq!(u64::from_le_bytes(rope[0]), 1.5_f64.to_bits());
    // gguf-py pads after every tensor, the last one included.
    let last_end = gguf.tensors().last().expect("tensors").data_range().end;
    assert_eq!(
        last_end.next_multiple_of(u64::from(alignment)),
        gguf.file_size()
    );
}

#[test]
fn reads_what_gguf_py_writes() {
    // Offsets advance by each tensor's size rounded up to the alignment.
    check("little-endian.gguf", 32, [0, 640, 704, 800, 832]);
}

#[test]
fn reads_gguf_py_files_with_a_custom_alignment() {
    check("aligned-64.gguf", 64, [0, 640, 704, 832, 896]);
}

#[test]
fn rejects_gguf_py_big_endian_files() {
    let err = GgufFile::open(fixture("big-endian.gguf")).unwrap_err();
    assert!(
        matches!(err.kind(), ErrorKind::BigEndian { version: 3 }),
        "{err}"
    );
    assert_eq!(err.offset(), Some(4));
}
