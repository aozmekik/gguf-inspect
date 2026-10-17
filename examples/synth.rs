//! Writes a synthetic GGUF file for trying the CLI without downloading a model.
//!
//! ```text
//! cargo run --example synth -- tiny.gguf            # small, fully written
//! cargo run --example synth -- --llama3-8b big.gguf # 8B-shaped, sparse (~4.6 GiB apparent size)
//! ```
//!
//! The files are synthetic: generated vocabulary strings and zero (or sparse)
//! tensor data. The `--llama3-8b` header has the tensor shapes and vocabulary
//! size of an 8B Llama-3 model with a simplified Q4_K_M-style type mix, which
//! makes it a realistically sized benchmark input.

use std::path::PathBuf;
use std::process::ExitCode;

use gguf_testkit::SyntheticModel;

fn main() -> ExitCode {
    let mut big = false;
    let mut path = None;
    for arg in std::env::args().skip(1) {
        match arg.as_str() {
            "--llama3-8b" => big = true,
            _ if path.is_none() && !arg.starts_with('-') => path = Some(PathBuf::from(arg)),
            _ => return usage(),
        }
    }
    let Some(path) = path else {
        return usage();
    };

    let model = if big {
        SyntheticModel::llama3_8b_like()
    } else {
        SyntheticModel::tiny()
    };
    let builder = model.builder();
    let result = if big {
        builder.write_sparse(&path).map(|layout| layout.file_size)
    } else {
        let bytes = builder.build();
        std::fs::write(&path, &bytes).map(|()| bytes.len() as u64)
    };
    match result {
        Ok(size) => {
            println!("wrote {} ({size} bytes): {}", path.display(), model.name);
            ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("synth: {}: {err}", path.display());
            ExitCode::FAILURE
        }
    }
}

fn usage() -> ExitCode {
    eprintln!("usage: synth [--llama3-8b] OUTPUT.gguf");
    ExitCode::from(2)
}
