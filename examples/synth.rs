//! Writes a synthetic GGUF file for trying the CLI without downloading a model.
//!
//! ```text
//! cargo run --example synth -- tiny.gguf                 # small model, data included
//! cargo run --example synth -- --llama3-8b big.gguf      # 8B-shaped, sparse (4.82 GiB apparent)
//! cargo run --example synth -- --header-only seed.gguf   # small model header, no data
//! cargo run --example synth -- --every-type types.gguf   # one entry of each metadata type
//! ```
//!
//! The files are synthetic: generated vocabulary strings and zero (or sparse)
//! tensor data. The `--llama3-8b` header has the tensor shapes and vocabulary
//! size of an 8B Llama-3 model with a simplified Q4_K_M-style type mix, which
//! makes it a realistically sized benchmark input.

use std::path::PathBuf;
use std::process::ExitCode;

use gguf_testkit::{GgufBuilder, SyntheticModel, every_value_type};

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Tiny,
    Llama3_8b,
    HeaderOnly,
    EveryType,
}

fn main() -> ExitCode {
    let mut kind = Kind::Tiny;
    let mut path = None;
    for arg in std::env::args().skip(1) {
        match arg.as_str() {
            "--llama3-8b" => kind = Kind::Llama3_8b,
            "--header-only" => kind = Kind::HeaderOnly,
            "--every-type" => kind = Kind::EveryType,
            _ if path.is_none() && !arg.starts_with('-') => path = Some(PathBuf::from(arg)),
            _ => return usage(),
        }
    }
    let Some(path) = path else {
        return usage();
    };

    let (label, result) = match kind {
        Kind::Llama3_8b => {
            let model = SyntheticModel::llama3_8b_like();
            let written = model.builder().write_sparse(&path).map(|l| l.file_size);
            (model.name, written)
        }
        Kind::Tiny | Kind::HeaderOnly => {
            let model = SyntheticModel::tiny();
            let builder = model.builder();
            let bytes = if kind == Kind::Tiny {
                builder.build()
            } else {
                builder.header()
            };
            (model.name, write(&path, &bytes))
        }
        Kind::EveryType => {
            let bytes = every_value_type()
                .iter()
                .fold(GgufBuilder::new(), |b, (key, value)| b.kv(key, value))
                .build();
            ("every metadata value type".to_owned(), write(&path, &bytes))
        }
    };
    match result {
        Ok(size) => {
            println!("wrote {} ({size} bytes): {label}", path.display());
            ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("synth: {}: {err}", path.display());
            ExitCode::FAILURE
        }
    }
}

fn write(path: &std::path::Path, bytes: &[u8]) -> std::io::Result<u64> {
    std::fs::write(path, bytes).map(|()| bytes.len() as u64)
}

fn usage() -> ExitCode {
    eprintln!("usage: synth [--llama3-8b | --header-only | --every-type] OUTPUT.gguf");
    ExitCode::from(2)
}
