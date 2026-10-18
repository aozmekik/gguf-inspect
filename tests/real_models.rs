//! Opt-in checks on real model files, which are not part of the repository
//! (the test suite itself uses synthetic files only).
//!
//! ```text
//! GGUF_INSPECT_REAL_MODELS=/models/a.gguf:/models/b.gguf \
//!     cargo test --release --test real_models -- --ignored --nocapture
//! ```
//!
//! Paths are separated as in `PATH` (`:` on Unix, `;` on Windows). Every
//! file must parse, every subcommand must succeed on it and print only
//! terminal-safe text, and a summary of each file is printed.

use std::path::Path;
use std::process::Command;

use gguf_inspect::kv_cache::{KvCacheConfig, estimate};
use gguf_inspect::{GgufFile, Issue, ModelInfo};

fn run(args: &[&str], path: &Path) -> (i32, String) {
    let output = Command::new(env!("CARGO_BIN_EXE_gguf-inspect"))
        .args(args)
        .arg(path)
        .output()
        .expect("binary runs");
    let text = String::from_utf8(output.stdout).expect("UTF-8 output")
        + &String::from_utf8(output.stderr).expect("UTF-8 output");
    assert!(
        !text.chars().any(|c| c.is_control() && c != '\n'),
        "{args:?} {}: control characters in output",
        path.display()
    );
    (output.status.code().expect("exit code"), text)
}

#[test]
#[ignore = "needs GGUF_INSPECT_REAL_MODELS; see the module docs"]
fn real_models_parse_and_every_command_succeeds() {
    let paths = std::env::var_os("GGUF_INSPECT_REAL_MODELS")
        .expect("set GGUF_INSPECT_REAL_MODELS to one or more .gguf paths");
    for path in std::env::split_paths(&paths) {
        let gguf = GgufFile::open(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        let info = ModelInfo::from_file(&gguf);
        let issues = gguf.validate();
        println!(
            "{}: GGUF v{}, {} ({}), {} tensors, {} parameters",
            path.display(),
            gguf.version(),
            info.name.as_deref().unwrap_or("unnamed"),
            info.architecture.as_deref().unwrap_or("no architecture"),
            gguf.tensors().len(),
            gguf.parameter_count()
        );
        match estimate(&gguf, &KvCacheConfig::new(4096)) {
            Ok(kv) => println!("  KV cache at 4,096 tokens (f16): {} bytes", kv.total_bytes),
            Err(err) => println!("  KV cache: {err}"),
        }
        for issue in &issues {
            println!("  warning: {issue}");
        }
        assert!(
            !issues
                .iter()
                .any(|i| matches!(i, Issue::DataOutOfBounds { .. })),
            "{}: tensor data past the end of the file",
            path.display()
        );
        let total: u64 = gguf.type_breakdown().iter().map(|s| s.bytes).sum();
        assert_eq!(total, gguf.tensor_data_size());

        for args in [
            &["info"][..],
            &["info", "--json"],
            &["metadata"],
            &["metadata", "--json"],
            &["tensors", "--json"],
        ] {
            let (code, text) = run(args, &path);
            assert_eq!(code, 0, "{args:?} {}: {text}", path.display());
        }
        // Architectures without attention keys (vision projectors, say) exit 2.
        let (code, text) = run(&["kv-cache", "--ctx", "4096"], &path);
        assert!(code == 0 || code == 2, "{}: {text}", path.display());
    }
}
