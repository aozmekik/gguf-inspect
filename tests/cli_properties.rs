//! Property tests of the `gguf-inspect` binary itself: whatever the file
//! holds, every subcommand exits with 0, 1 or 2, never panics, prints only
//! terminal-safe text and, with `--json`, valid JSON.
//!
//! The cargo-fuzz target covers the library in-process; this covers the
//! rendering layer end to end, at the cost of one process per command.

use std::path::Path;
use std::process::Command;

use gguf_inspect::{GgmlType, MetadataArray, MetadataValue};
use gguf_testkit::{GgufBuilder, every_value_type};
use proptest::prelude::*;

const COMMANDS: &[&[&str]] = &[
    &["info"],
    &["info", "--json"],
    &["metadata", "--full"],
    &["metadata", "--json"],
    &["tensors", "--sort", "size"],
    &["tensors", "--json"],
    &["kv-cache", "--ctx", "64"],
    &["kv-cache", "--ctx", "64", "--json"],
];

/// No control characters except the newlines ending lines, and none of the
/// invisible format characters the CLI escapes.
fn is_terminal_safe(text: &str) -> bool {
    !text.chars().any(|c| {
        (c.is_control() && c != '\n')
            || matches!(
                c,
                '\u{200B}'..='\u{200F}' | '\u{2028}'..='\u{202E}' | '\u{2066}'..='\u{2069}' | '\u{FEFF}'
            )
    })
}

/// Runs every command on `file` (and `diff` against `other`) and checks the
/// properties; returns a description of the first violation.
fn check_commands(dir: &Path, file: &str, other: &str) -> Result<(), String> {
    let diff: &[&str] = &["diff", other, file];
    for args in COMMANDS.iter().copied().chain([diff]) {
        let mut cmd = Command::new(env!("CARGO_BIN_EXE_gguf-inspect"));
        cmd.args(args).current_dir(dir);
        if args[0] != "diff" {
            cmd.arg(file);
        }
        let output = cmd.output().map_err(|e| format!("{args:?}: {e}"))?;
        let stdout = String::from_utf8(output.stdout).map_err(|e| format!("{args:?}: {e}"))?;
        let stderr = String::from_utf8(output.stderr).map_err(|e| format!("{args:?}: {e}"))?;
        let code = output.status.code().unwrap_or(-1);
        // 1 is `diff` finding differences; 101 would be a panic.
        if !(0..=2).contains(&code) || stderr.contains("panicked") {
            return Err(format!("{args:?} exited with {code}: {stderr}"));
        }
        if !is_terminal_safe(&stdout) || !is_terminal_safe(&stderr) {
            return Err(format!("{args:?}: unsafe output {stdout:?} {stderr:?}"));
        }
        if code == 0 && args.contains(&"--json") {
            serde_json::from_str::<serde_json::Value>(&stdout)
                .map_err(|e| format!("{args:?}: invalid JSON ({e}): {stdout}"))?;
        }
    }
    Ok(())
}

/// A small file with every value type, KV-cache keys, strings worth escaping
/// and a few tensors.
fn sample() -> GgufBuilder {
    every_value_type()
        .iter()
        .fold(GgufBuilder::new(), |b, (k, v)| b.kv(k, v))
        .kv(
            "general.architecture",
            &MetadataValue::String("llama".into()),
        )
        .kv("general.name", &MetadataValue::String("n\x1b[0m".into()))
        .kv("llama.block_count", &MetadataValue::U32(2))
        .kv("llama.embedding_length", &MetadataValue::U32(64))
        .kv("llama.attention.head_count", &MetadataValue::U32(4))
        .kv(
            "llama.attention.head_count_kv",
            &MetadataValue::Array(MetadataArray::U32(vec![2, 0])),
        )
        .tensor("token_embd.weight", &[64, 4], GgmlType::Q8_0)
        .tensor("blk.0.attn_norm.weight", &[64], GgmlType::F32)
        .tensor("rope_freqs", &[], GgmlType::F32)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(48))]

    /// Random byte edits in the header.
    #[test]
    fn mutated_headers(
        edits in proptest::collection::vec((any::<prop::sample::Index>(), any::<u8>()), 1..12)
    ) {
        let dir = tempfile::tempdir().expect("temp dir");
        let builder = sample();
        let original = builder.build();
        let header = usize::try_from(builder.layout().header_size).expect("small header");
        let mut bytes = original.clone();
        for (index, byte) in edits {
            bytes[index.index(header)] = byte;
        }
        std::fs::write(dir.path().join("original.gguf"), original).expect("write");
        std::fs::write(dir.path().join("mutated.gguf"), bytes).expect("write");
        let result = check_commands(dir.path(), "mutated.gguf", "original.gguf");
        prop_assert!(result.is_ok(), "{}", result.unwrap_err());
    }

    /// Keys, values, the architecture and tensor names drawn from all of
    /// Unicode, control and format characters included.
    #[test]
    fn arbitrary_strings(
        arch in ".{0,12}",
        entries in proptest::collection::vec((".{1,16}", ".{0,24}"), 0..6),
        tensor_names in proptest::collection::vec(".{0,16}", 0..4),
    ) {
        let dir = tempfile::tempdir().expect("temp dir");
        let mut builder = GgufBuilder::new()
            .kv("general.architecture", &MetadataValue::String(arch.clone()))
            .kv(&format!("{arch}.block_count"), &MetadataValue::U32(1))
            .kv(&format!("{arch}.attention.head_count"), &MetadataValue::U32(1))
            .kv(&format!("{arch}.embedding_length"), &MetadataValue::U32(8));
        for (key, value) in &entries {
            builder = builder
                .kv(key, &MetadataValue::String(value.clone()))
                .kv(
                    &format!("{key}.list"),
                    &MetadataValue::Array(MetadataArray::String(vec![value.clone(); 2])),
                );
        }
        for name in &tensor_names {
            builder = builder.tensor(name, &[4], GgmlType::F32);
        }
        // Duplicate keys or names make some files invalid: also fine, as long
        // as the error message is safe.
        std::fs::write(dir.path().join("strings.gguf"), builder.build()).expect("write");
        std::fs::write(dir.path().join("sample.gguf"), sample().build()).expect("write");
        let result = check_commands(dir.path(), "strings.gguf", "sample.gguf");
        prop_assert!(result.is_ok(), "{}", result.unwrap_err());
    }
}
