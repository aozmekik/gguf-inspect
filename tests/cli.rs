//! Snapshot tests of the `gguf-inspect` binary on synthetic files.
//!
//! Each test writes its inputs into a fresh temporary directory and runs the
//! binary there with relative paths, so output is identical on every platform.
//! Update snapshots with `INSTA_UPDATE=always cargo test --test cli` and review
//! the diff.

use std::path::Path;
use std::process::Command;

use gguf_inspect::{GgmlType, MetadataArray, MetadataValue};
use gguf_testkit::{GgufBuilder, SyntheticModel, every_value_type};
use insta::assert_snapshot;

struct Run {
    stdout: String,
    stderr: String,
    code: i32,
}

fn run(dir: &Path, args: &[&str]) -> Run {
    let output = Command::new(env!("CARGO_BIN_EXE_gguf-inspect"))
        .args(args)
        .current_dir(dir)
        .output()
        .expect("binary runs");
    Run {
        stdout: String::from_utf8(output.stdout).expect("UTF-8 stdout"),
        stderr: String::from_utf8(output.stderr).expect("UTF-8 stderr"),
        code: output.status.code().expect("exit code"),
    }
}

/// Runs a command that must succeed and returns its stdout.
fn ok(dir: &Path, args: &[&str]) -> String {
    let r = run(dir, args);
    assert_eq!(r.code, 0, "{args:?} failed: {}", r.stderr);
    assert_eq!(r.stderr, "");
    r.stdout
}

/// A directory with the synthetic files the tests use.
fn fixtures() -> tempfile::TempDir {
    let dir = tempfile::tempdir().expect("temp dir");
    let write = |name: &str, bytes: Vec<u8>| {
        std::fs::write(dir.path().join(name), bytes).expect("write fixture");
    };
    write("tiny.gguf", SyntheticModel::tiny().builder().build());

    let mut changed = SyntheticModel::tiny();
    changed.name = "Synthetic tiny, requantized".into();
    changed.context_length = 4096;
    changed.weight_type = GgmlType::Q5_K;
    write("tiny-q5.gguf", changed.builder().version(2).build());

    let every = every_value_type()
        .iter()
        .fold(GgufBuilder::new(), |b, (k, v)| b.kv(k, v))
        .kv(
            "test.long_string",
            &MetadataValue::String("{% for message in messages %}".repeat(4)),
        )
        .kv(
            "test.long_array",
            &MetadataValue::Array(MetadataArray::U16((0..20).collect())),
        );
    write("every-type.gguf", every.build());

    // Every kind of change the diff can report, against every-type.gguf.
    let mut changed_array: Vec<u16> = (0..20).collect();
    changed_array[13] = 1300;
    let edited = every_value_type()
        .iter()
        .filter(|(k, _)| *k != "test.i8")
        .fold(GgufBuilder::new().alignment(64), |b, (k, v)| match *k {
            "test.u16" => b.kv(k, &MetadataValue::U32(60_000)),
            "test.array.strings" => b.kv(
                k,
                &MetadataValue::Array(MetadataArray::String(vec![
                    "a".into(),
                    String::new(),
                    "c".into(),
                ])),
            ),
            _ => b.kv(k, v),
        })
        .kv(
            "test.long_string",
            &MetadataValue::String("{% for message in messages %}".repeat(4)),
        )
        .kv(
            "test.long_array",
            &MetadataValue::Array(MetadataArray::U16(changed_array)),
        )
        .kv("test.added", &MetadataValue::Bool(false))
        .tensor("added.weight", &[32], GgmlType::F32);
    write("every-type-edited.gguf", edited.build());

    for (name, builder) in attention_variants() {
        write(name, builder.build());
    }

    // Header 24 bytes, key 8 + 12, type 4, string length at 48, text at 56..65.
    let mut truncated = GgufBuilder::new()
        .kv("general.name", &MetadataValue::String("Synthetic".into()))
        .build();
    truncated.truncate(60);
    write("truncated.gguf", truncated);
    let mut partial = SyntheticModel::tiny().builder().build();
    partial.truncate(partial.len() / 2);
    write("partial.gguf", partial);
    write("hostile.gguf", hostile().build());
    write(
        "hostile-duplicate.gguf",
        GgufBuilder::new()
            .kv("dup\x1b[2J", &MetadataValue::U8(1))
            .kv("dup\x1b[2J", &MetadataValue::U8(2))
            .build(),
    );
    write("not-gguf.gguf", b"GGML not a gguf file".to_vec());
    write(
        "big-endian.gguf",
        GgufBuilder::new().endian(gguf_testkit::Endian::Big).build(),
    );
    dir
}

/// Models whose KV cache is not plain grouped-query attention: hybrid
/// per-layer KV heads, DeepSeek-V3-shaped multi-head latent attention and a
/// Mamba-style recurrent model, laid out as llama.cpp's converters write them.
fn attention_variants() -> Vec<(&'static str, GgufBuilder)> {
    let hybrid = GgufBuilder::new()
        .kv(
            "general.architecture",
            &MetadataValue::String("hybrid".into()),
        )
        .kv("hybrid.block_count", &MetadataValue::U32(4))
        .kv("hybrid.embedding_length", &MetadataValue::U32(1024))
        .kv("hybrid.attention.head_count", &MetadataValue::U32(16))
        .kv(
            "hybrid.attention.head_count_kv",
            &MetadataValue::Array(MetadataArray::U32(vec![0, 4, 0, 4])),
        )
        .kv("hybrid.attention.sliding_window", &MetadataValue::U32(512))
        .kv("hybrid.ssm.state_size", &MetadataValue::U32(16));

    let u32_entries = |arch: &str, entries: &[(&str, u32)]| {
        entries.iter().fold(
            GgufBuilder::new().kv("general.architecture", &MetadataValue::String(arch.into())),
            |b, (k, v)| b.kv(&format!("{arch}.{k}"), &MetadataValue::U32(*v)),
        )
    };
    let deepseek = u32_entries(
        "deepseek2",
        &[
            ("block_count", 61),
            ("embedding_length", 7168),
            ("attention.head_count", 128),
            ("attention.head_count_kv", 1),
            ("attention.key_length", 576),
            ("attention.value_length", 512),
            ("attention.key_length_mla", 192),
            ("attention.value_length_mla", 128),
            ("attention.kv_lora_rank", 512),
        ],
    );
    let mamba = u32_entries(
        "mamba",
        &[
            ("block_count", 64),
            ("embedding_length", 2560),
            ("attention.head_count", 0),
            ("ssm.state_size", 16),
        ],
    );
    vec![
        ("hybrid.gguf", hybrid),
        ("deepseek.gguf", deepseek),
        ("mamba.gguf", mamba),
    ]
}

/// Terminal escape sequences, a forged row, a bidi override and a C1
/// control in every kind of string the CLI prints.
const HOSTILE_ARCH: &str = "llama\x1b[31mRED\x1b[0m";

fn hostile() -> GgufBuilder {
    let arch = |key: &str| format!("{HOSTILE_ARCH}.{key}");
    GgufBuilder::new()
        .kv(
            "general.architecture",
            &MetadataValue::String(HOSTILE_ARCH.into()),
        )
        .kv(
            "general.name",
            &MetadataValue::String("evil\x1b]0;pwned-title\x07\x1b[2J".into()),
        )
        .kv("evil\x1b[31m.key\nfake.key  u32", &MetadataValue::U32(1))
        .kv(
            "tokenizer.chat_template",
            &MetadataValue::String("{% if message['role'] == 'user' %}\u{202e}cba".into()),
        )
        .kv(
            "general.c1",
            &MetadataValue::Array(MetadataArray::String(vec!["\u{9b}31m\r".into()])),
        )
        .kv(&arch("block_count"), &MetadataValue::U32(1))
        .kv(&arch("attention.head_count"), &MetadataValue::U32(1))
        .kv(&arch("attention.key_length"), &MetadataValue::U32(32))
        .kv(&arch("attention.value_length"), &MetadataValue::U32(32))
        .tensor("t\x1b[1mBOLD\n", &[8], GgmlType::F32)
}

/// No control or invisible format characters except the newlines that end lines.
fn assert_terminal_safe(output: &str) {
    let unsafe_char = output.chars().find(|&c| {
        (c.is_control() && c != '\n')
            || matches!(c, '\u{200B}'..='\u{200F}' | '\u{2028}'..='\u{202E}' | '\u{2066}'..='\u{2069}')
    });
    assert_eq!(unsafe_char, None, "{output:?}");
}

#[test]
fn file_strings_cannot_inject_terminal_sequences() {
    let dir = fixtures();
    for (name, args) in [
        ("hostile_info", &["info", "hostile.gguf"][..]),
        ("hostile_metadata", &["metadata", "hostile.gguf", "--full"]),
        ("hostile_tensors", &["tensors", "hostile.gguf"]),
        (
            "hostile_kv_cache",
            &["kv-cache", "hostile.gguf", "--ctx", "16"],
        ),
        ("hostile_diff", &["diff", "tiny.gguf", "hostile.gguf"]),
        (
            "hostile_metadata_json",
            &["metadata", "hostile.gguf", "--json"],
        ),
    ] {
        let r = run(dir.path(), args);
        assert!(r.code == 0 || r.code == 1, "{args:?}: {}", r.stderr);
        assert_terminal_safe(&r.stdout);
        assert_terminal_safe(&r.stderr);
        assert_snapshot!(name, r.stdout);
    }

    // JSON escapes are standard, so the parsed value is exactly what the file holds.
    let json = ok(dir.path(), &["metadata", "hostile.gguf", "--json"]);
    let parsed: serde_json::Value = serde_json::from_str(&json).expect("valid JSON");
    assert_eq!(parsed[0]["value"], HOSTILE_ARCH);
    assert_eq!(parsed[4]["value"][0], "\u{9b}31m\r");

    let r = run(dir.path(), &["info", "hostile-duplicate.gguf"]);
    assert_eq!(r.code, 2);
    assert_terminal_safe(&r.stderr);
    assert_eq!(
        r.stderr,
        "gguf-inspect: error: hostile-duplicate.gguf: duplicate metadata key `dup\\u{1b}[2J` \
         at byte offset 44 (in metadata entry 1)\n"
    );
}

#[test]
fn info() {
    let dir = fixtures();
    assert_snapshot!("info", ok(dir.path(), &["info", "tiny.gguf"]));
    assert_snapshot!(
        "info_json",
        ok(dir.path(), &["info", "tiny.gguf", "--json"])
    );
}

#[test]
fn info_reports_partial_downloads() {
    let dir = fixtures();
    assert_snapshot!("info_partial", ok(dir.path(), &["info", "partial.gguf"]));
}

#[test]
fn metadata() {
    let dir = fixtures();
    assert_snapshot!("metadata", ok(dir.path(), &["metadata", "tiny.gguf"]));
    assert_snapshot!(
        "metadata_every_type",
        ok(dir.path(), &["metadata", "every-type.gguf"])
    );
    assert_snapshot!(
        "metadata_every_type_full",
        ok(dir.path(), &["metadata", "every-type.gguf", "--full"])
    );
    assert_snapshot!(
        "metadata_filter_json",
        ok(
            dir.path(),
            &["metadata", "tiny.gguf", "--filter", "tokenizer.", "--json"]
        )
    );
    assert_snapshot!(
        "metadata_every_type_json",
        ok(dir.path(), &["metadata", "every-type.gguf", "--json"])
    );
    assert_eq!(
        ok(dir.path(), &["metadata", "tiny.gguf", "--filter", "nope."]),
        "no metadata keys start with `nope.`\n"
    );
}

#[test]
fn full_json_keeps_whole_arrays() {
    let dir = fixtures();
    let json = ok(
        dir.path(),
        &[
            "metadata",
            "tiny.gguf",
            "--json",
            "--full",
            "--filter",
            "tokenizer.ggml.tokens",
        ],
    );
    let parsed: serde_json::Value = serde_json::from_str(&json).expect("valid JSON");
    assert_eq!(parsed[0]["len"], 512);
    assert_eq!(parsed[0]["value"].as_array().map(Vec::len), Some(512));
    assert!(parsed[0].get("truncated").is_none());
}

#[test]
fn tensors() {
    let dir = fixtures();
    assert_snapshot!("tensors", ok(dir.path(), &["tensors", "tiny.gguf"]));
    assert_snapshot!(
        "tensors_by_size",
        ok(dir.path(), &["tensors", "tiny.gguf", "--sort", "size"])
    );
    let json = ok(
        dir.path(),
        &["tensors", "tiny.gguf", "--json", "--sort", "name"],
    );
    let parsed: serde_json::Value = serde_json::from_str(&json).expect("valid JSON");
    let names: Vec<&str> = parsed
        .as_array()
        .expect("array")
        .iter()
        .filter_map(|t| t["name"].as_str())
        .collect();
    let mut sorted = names.clone();
    sorted.sort_unstable();
    assert_eq!(names, sorted);
    assert_snapshot!(
        "tensors_json_first",
        serde_json::to_string_pretty(&parsed[0]).expect("json")
    );
}

#[test]
fn kv_cache() {
    let dir = fixtures();
    assert_snapshot!(
        "kv_cache_f16",
        ok(dir.path(), &["kv-cache", "tiny.gguf", "--ctx", "2048"])
    );
    let sequences = ok(
        dir.path(),
        &[
            "kv-cache",
            "tiny.gguf",
            "--ctx",
            "4096",
            "--cache-type",
            "q8_0",
            "--sequences",
            "4",
        ],
    );
    assert_snapshot!("kv_cache_q8_0_sequences", sequences);
    // llama.cpp users know the flag as --parallel (-np).
    assert_eq!(
        ok(
            dir.path(),
            &[
                "kv-cache",
                "tiny.gguf",
                "--ctx",
                "4096",
                "--cache-type",
                "q8_0",
                "--parallel",
                "4",
            ],
        ),
        sequences
    );
    assert_snapshot!(
        "kv_cache_json",
        ok(
            dir.path(),
            &["kv-cache", "tiny.gguf", "--ctx", "1000", "--json"]
        )
    );
    assert_snapshot!(
        "kv_cache_per_layer",
        ok(dir.path(), &["kv-cache", "hybrid.gguf", "--ctx", "8192"])
    );
    assert_snapshot!(
        "kv_cache_latent_attention",
        ok(dir.path(), &["kv-cache", "deepseek.gguf", "--ctx", "8192"])
    );
    assert_snapshot!(
        "kv_cache_recurrent",
        ok(dir.path(), &["kv-cache", "mamba.gguf", "--ctx", "8192"])
    );
}

#[test]
fn kv_cache_rejects_bad_arguments() {
    let dir = fixtures();
    for args in [
        &["kv-cache", "tiny.gguf"][..],
        &["kv-cache", "tiny.gguf", "--ctx", "0"],
        &[
            "kv-cache",
            "tiny.gguf",
            "--ctx",
            "8",
            "--cache-type",
            "q4_K",
        ],
    ] {
        let r = run(dir.path(), args);
        assert_eq!(r.code, 2, "{args:?}");
        assert!(r.stderr.contains("error"), "{args:?}: {}", r.stderr);
    }
    let r = run(dir.path(), &["kv-cache", "every-type.gguf", "--ctx", "8"]);
    assert_eq!(r.code, 2);
    assert_eq!(
        r.stderr,
        "gguf-inspect: error: every-type.gguf: general.architecture is missing\n"
    );
}

#[test]
fn diff() {
    let dir = fixtures();
    let r = run(dir.path(), &["diff", "tiny.gguf", "tiny-q5.gguf"]);
    assert_eq!(r.code, 1, "differences exit with 1");
    assert_snapshot!("diff", r.stdout);

    let r = run(dir.path(), &["diff", "tiny.gguf", "tiny-q5.gguf", "--json"]);
    assert_eq!(r.code, 1);
    assert_snapshot!("diff_json", r.stdout);

    assert_snapshot!(
        "diff_identical",
        ok(dir.path(), &["diff", "tiny.gguf", "tiny.gguf"])
    );

    let r = run(
        dir.path(),
        &["diff", "every-type.gguf", "every-type-edited.gguf"],
    );
    assert_eq!(r.code, 1);
    assert_snapshot!("diff_every_kind", r.stdout);
    let r = run(
        dir.path(),
        &[
            "diff",
            "every-type-edited.gguf",
            "every-type.gguf",
            "--json",
        ],
    );
    assert_eq!(r.code, 1);
    assert_snapshot!("diff_every_kind_json", r.stdout);
}

#[test]
fn types() {
    let dir = fixtures();
    assert_snapshot!("types", ok(dir.path(), &["types"]));
    let json = ok(dir.path(), &["types", "--json"]);
    let parsed: serde_json::Value = serde_json::from_str(&json).expect("valid JSON");
    assert_eq!(parsed.as_array().map(Vec::len), Some(GgmlType::ALL.len()));
}

#[test]
fn errors_name_the_file_and_exit_with_2() {
    let dir = fixtures();
    for (file, expected) in [
        (
            "truncated.gguf",
            "gguf-inspect: error: truncated.gguf: string length 9 needs at least 9 bytes but only 4 remain at byte offset 48 (in metadata entry 0 `general.name`)\n",
        ),
        (
            "not-gguf.gguf",
            "gguf-inspect: error: not-gguf.gguf: not a GGUF file: expected magic \"GGUF\", found \"GGML\" at byte offset 0\n",
        ),
        (
            "big-endian.gguf",
            "gguf-inspect: error: big-endian.gguf: big-endian GGUF files are not supported (the version field reads as version 3 only when byte-swapped) at byte offset 4\n",
        ),
    ] {
        let r = run(dir.path(), &["info", file]);
        assert_eq!(r.code, 2, "{file}");
        assert_eq!(r.stdout, "");
        assert_eq!(r.stderr, expected);
    }
    // OS error text differs between platforms; check the stable part.
    let r = run(dir.path(), &["tensors", "missing.gguf"]);
    assert_eq!(r.code, 2);
    assert!(
        r.stderr
            .starts_with("gguf-inspect: error: missing.gguf: I/O error: ")
    );
}

#[test]
fn usage_errors_exit_with_2() {
    let dir = fixtures();
    assert_eq!(run(dir.path(), &[]).code, 2);
    assert_eq!(run(dir.path(), &["frobnicate"]).code, 2);
    let r = run(dir.path(), &["--version"]);
    assert_eq!(r.code, 0);
    assert_eq!(
        r.stdout,
        format!("gguf-inspect {}\n", env!("CARGO_PKG_VERSION"))
    );
}
