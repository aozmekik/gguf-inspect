//! Feeds arbitrary bytes to the parser and to everything built on its output.
//! Any panic, overflow, or out-of-memory abort is a bug.

#![no_main]

use gguf_inspect::kv_cache::{KvCacheConfig, estimate};
use gguf_inspect::{GgmlType, GgufFile, ModelInfo};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let Ok(file) = GgufFile::from_bytes(data) else {
        return;
    };
    let _ = file.validate();
    let _ = file.type_breakdown();
    let _ = file.parameter_count();
    let _ = ModelInfo::from_file(&file);
    for cache_type in [GgmlType::F16, GgmlType::Q8_0] {
        let config = KvCacheConfig::new(4096).with_cache_type(cache_type);
        let _ = estimate(&file, &config);
    }
    assert!(gguf_inspect::diff::diff(&file, &file).is_empty());
});
