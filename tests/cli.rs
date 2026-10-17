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
    write("hybrid.gguf", hybrid.build());

    // Header 24 bytes, key 8 + 12, type 4, string length at 48, text at 56..65.
    let mut truncated = GgufBuilder::new()
        .kv("general.name", &MetadataValue::String("Synthetic".into()))
        .build();
    truncated.truncate(60);
    write("truncated.gguf", truncated);
    let mut partial = SyntheticModel::tiny().builder().build();
    partial.truncate(partial.len() / 2);
    write("partial.gguf", partial);
    write("not-gguf.gguf", b"GGML not a gguf file".to_vec());
    write(
        "big-endian.gguf",
        GgufBuilder::new().endian(gguf_testkit::Endian::Big).build(),
    );
    dir
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
    assert_snapshot!(
        "kv_cache_q8_0_batch",
        ok(
            dir.path(),
            &[
                "kv-cache",
                "tiny.gguf",
                "--ctx",
                "4096",
                "--cache-type",
                "q8_0",
                "--batch",
                "4"
            ]
        )
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
