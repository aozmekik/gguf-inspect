//! Library tour: open a GGUF file and print a short summary.
//!
//! ```text
//! cargo run --example summary -- model.gguf
//! cargo run --example summary            # uses a small synthetic model
//! ```

use std::process::ExitCode;

use gguf_inspect::kv_cache::{KvCacheConfig, estimate};
use gguf_inspect::{GgmlType, GgufFile, ModelInfo};

fn main() -> ExitCode {
    let parsed = if let Some(path) = std::env::args().nth(1) {
        GgufFile::open(&path)
    } else {
        println!("no file given; using a synthetic model");
        GgufFile::from_bytes(&gguf_testkit::SyntheticModel::tiny().builder().build())
    };
    let gguf = match parsed {
        Ok(gguf) => gguf,
        Err(err) => {
            eprintln!("summary: {err}");
            return ExitCode::FAILURE;
        }
    };

    let info = ModelInfo::from_file(&gguf);
    println!(
        "{} ({}), GGUF v{}",
        info.name.as_deref().unwrap_or("unnamed"),
        info.architecture
            .as_deref()
            .unwrap_or("unknown architecture"),
        gguf.version()
    );
    println!(
        "{} tensors, {} parameters, {} bytes of tensor data",
        gguf.tensors().len(),
        gguf.parameter_count(),
        gguf.tensor_data_size()
    );
    for stats in gguf.type_breakdown() {
        println!(
            "  {:>6}: {:>4} tensors, {:>12} bytes",
            stats.ggml_type.name(),
            stats.tensors,
            stats.bytes
        );
    }

    let config = KvCacheConfig::new(4096).with_cache_type(GgmlType::Q8_0);
    match estimate(&gguf, &config) {
        Ok(kv) => println!("KV cache at 4096 tokens (q8_0): {} bytes", kv.total_bytes),
        Err(err) => println!("KV cache: cannot estimate ({err})"),
    }
    for issue in gguf.validate() {
        println!("warning: {issue}");
    }
    ExitCode::SUCCESS
}
