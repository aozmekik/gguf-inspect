//! Peak heap use while parsing, measured with dhat's counting allocator.
//!
//! The allocator is process-global, so this file holds a single test.

use gguf_inspect::{ErrorKind, GgufFile};
use gguf_testkit::{GgufBuilder, SyntheticModel};

#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;

const MIB: u64 = 1 << 20;

/// Parses `bytes` and returns the result with the peak heap bytes allocated
/// during parsing (the input itself is allocated before profiling starts).
fn peak_heap(bytes: &[u8]) -> (gguf_inspect::Result<GgufFile>, u64) {
    let profiler = dhat::Profiler::builder().testing().build();
    let result = GgufFile::from_bytes(bytes);
    let peak = dhat::HeapStats::get().max_bytes as u64;
    drop(profiler);
    (result, peak)
}

/// An entry whose value is an array nested `levels` deep, each level
/// declaring as many elements as the rest of the file could hold, followed by
/// an invalid element type and `padding` filler bytes.
fn nested_declarations(levels: usize, padding: usize) -> Vec<u8> {
    let declared = (padding / 12) as u64;
    let mut value = 9_u32.to_le_bytes().to_vec(); // value type: array
    for _ in 0..levels {
        value.extend_from_slice(&9_u32.to_le_bytes()); // element type: array
        value.extend_from_slice(&declared.to_le_bytes());
    }
    value.extend_from_slice(&99_u32.to_le_bytes()); // invalid element type
    value.resize(value.len() + padding, 0);
    GgufBuilder::new().raw_kv(b"k", value).build()
}

/// A string array declaring as many strings as the file could hold, whose
/// first string is not valid UTF-8.
fn string_declarations(padding: usize) -> Vec<u8> {
    let mut value = 9_u32.to_le_bytes().to_vec();
    value.extend_from_slice(&8_u32.to_le_bytes()); // element type: string
    value.extend_from_slice(&((padding / 8) as u64).to_le_bytes());
    value.extend_from_slice(&1_u64.to_le_bytes());
    value.push(0xff);
    value.resize(value.len() + padding, 0);
    GgufBuilder::new().raw_kv(b"k", value).build()
}

#[test]
fn peak_heap_follows_bytes_parsed_not_counts_declared() {
    // Seven levels of 1M-element declarations in a 12 MiB file. Reserving
    // what each level declares would take 7 x 1M x 32 bytes, about 224 MiB.
    let bytes = nested_declarations(7, 12 << 20);
    let (result, peak) = peak_heap(&bytes);
    assert!(matches!(
        result.unwrap_err().kind(),
        ErrorKind::InvalidValueType(99)
    ));
    assert!(peak < 16 * MIB, "peak heap {peak} bytes");

    // 1.5M declared strings, the first one invalid: no 36 MiB reservation.
    let bytes = string_declarations(12 << 20);
    let (result, peak) = peak_heap(&bytes);
    assert!(matches!(result.unwrap_err().kind(), ErrorKind::InvalidUtf8));
    assert!(peak < 4 * MIB, "peak heap {peak} bytes");

    // A real-sized header: the 8B-shaped synthetic model (3.24 MB).
    let header = SyntheticModel::llama3_8b_like().builder().header();
    let (result, peak) = peak_heap(&header);
    let gguf = result.unwrap();
    assert_eq!(gguf.tensors().len(), 291);
    let header_len = header.len() as u64;
    println!("8B-shaped header: {header_len} bytes, peak heap while parsing {peak} bytes");
    assert!(peak < 4 * header_len, "peak heap {peak} bytes");
}
