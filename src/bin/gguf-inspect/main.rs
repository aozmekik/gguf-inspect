//! `gguf-inspect`: inspect GGUF model files without loading tensor data.
//!
//! Exit status: 0 on success, 1 when `diff` finds differences, 2 on errors.

mod commands;
mod escape;
mod format;
mod values;

use std::fmt;
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::{Parser, Subcommand, ValueEnum};
use gguf_inspect::GgmlType;

/// Inspect GGUF model files (llama.cpp / ggml) without loading tensor data.
#[derive(Debug, Parser)]
#[command(name = "gguf-inspect", version, about, long_about = None)]
#[command(after_help = "Exit status: 0 on success, 1 when `diff` finds differences, 2 on errors.")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Summarize a model: architecture, parameters, quantization, hyperparameters.
    Info {
        /// GGUF file to read.
        file: PathBuf,
        /// Print JSON instead of text.
        #[arg(long)]
        json: bool,
    },
    /// List metadata key-value pairs.
    Metadata {
        /// GGUF file to read.
        file: PathBuf,
        /// Print JSON instead of text.
        #[arg(long)]
        json: bool,
        /// Only show keys starting with this prefix, for example `tokenizer.`.
        #[arg(long, value_name = "PREFIX")]
        filter: Option<String>,
        /// Show long arrays (such as the vocabulary) and strings in full.
        #[arg(long)]
        full: bool,
    },
    /// List tensors with their type, shape, size and offset.
    Tensors {
        /// GGUF file to read.
        file: PathBuf,
        /// Print JSON instead of text.
        #[arg(long)]
        json: bool,
        /// Sort order.
        #[arg(long, value_enum, default_value_t = TensorSort::File)]
        sort: TensorSort,
    },
    /// Estimate KV-cache memory for a context length.
    KvCache {
        /// GGUF file to read.
        file: PathBuf,
        /// Tokens cached per sequence.
        #[arg(long, value_name = "N", value_parser = clap::value_parser!(u64).range(1..))]
        ctx: u64,
        /// Storage type of cached keys and values.
        #[arg(long, value_enum, default_value_t = CacheType::F16)]
        cache_type: CacheType,
        /// Sequences cached at once, each holding --ctx tokens. llama.cpp's
        /// -c is the total for all -np sequences: `-c 8192 -np 4` is
        /// `--ctx 2048 --sequences 4`.
        #[arg(
            long,
            visible_alias = "parallel",
            value_name = "N",
            default_value_t = 1,
            value_parser = clap::value_parser!(u64).range(1..)
        )]
        sequences: u64,
        /// Print JSON instead of text.
        #[arg(long)]
        json: bool,
    },
    /// Compare the metadata and tensor infos of two files.
    Diff {
        /// First (old) file.
        a: PathBuf,
        /// Second (new) file.
        b: PathBuf,
        /// Print JSON instead of text.
        #[arg(long)]
        json: bool,
    },
    /// List the ggml tensor types this tool knows, with their block layout.
    Types {
        /// Print JSON instead of text.
        #[arg(long)]
        json: bool,
    },
}

/// Tensor list order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum TensorSort {
    /// Order of the tensor infos in the file.
    File,
    /// Largest data size first.
    Size,
    /// By name.
    Name,
    /// By data offset.
    Offset,
}

/// KV-cache storage types accepted by llama.cpp's `--cache-type-k/-v`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum CacheType {
    #[value(name = "f32")]
    F32,
    #[value(name = "f16")]
    F16,
    #[value(name = "bf16")]
    Bf16,
    #[value(name = "q8_0")]
    Q8_0,
    #[value(name = "q5_1")]
    Q5_1,
    #[value(name = "q5_0")]
    Q5_0,
    #[value(name = "q4_1")]
    Q4_1,
    #[value(name = "q4_0")]
    Q4_0,
    #[value(name = "iq4_nl")]
    Iq4Nl,
}

impl CacheType {
    fn ggml(self) -> GgmlType {
        match self {
            Self::F32 => GgmlType::F32,
            Self::F16 => GgmlType::F16,
            Self::Bf16 => GgmlType::BF16,
            Self::Q8_0 => GgmlType::Q8_0,
            Self::Q5_1 => GgmlType::Q5_1,
            Self::Q5_0 => GgmlType::Q5_0,
            Self::Q4_1 => GgmlType::Q4_1,
            Self::Q4_0 => GgmlType::Q4_0,
            Self::Iq4Nl => GgmlType::IQ4_NL,
        }
    }
}

/// Why a command failed.
#[derive(Debug)]
pub(crate) enum Failure {
    /// Writing output failed (a closed pipe is not reported).
    Output(io::Error),
    /// Anything else, already formatted.
    Message(String),
}

impl Failure {
    pub(crate) fn with_path(path: &Path, err: impl fmt::Display) -> Self {
        Self::Message(format!("{}: {err}", path.display()))
    }
}

impl From<io::Error> for Failure {
    fn from(err: io::Error) -> Self {
        Self::Output(err)
    }
}

impl From<serde_json::Error> for Failure {
    fn from(err: serde_json::Error) -> Self {
        Self::Output(err.into())
    }
}

/// Outcome of a successful command.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Outcome {
    Ok,
    Differences,
}

/// Opens `path`, turning parse errors into messages that name the file.
pub(crate) fn open(path: &Path) -> Result<gguf_inspect::GgufFile, Failure> {
    gguf_inspect::GgufFile::open(path).map_err(|e| Failure::with_path(path, e))
}

fn run(cli: Cli, out: &mut dyn Write) -> Result<Outcome, Failure> {
    match cli.command {
        Command::Info { file, json } => commands::info(&file, json, out),
        Command::Metadata {
            file,
            json,
            filter,
            full,
        } => commands::metadata(&file, json, filter.as_deref(), full, out),
        Command::Tensors { file, json, sort } => commands::tensors(&file, json, sort, out),
        Command::KvCache {
            file,
            ctx,
            cache_type,
            sequences,
            json,
        } => commands::kv_cache(&file, ctx, cache_type.ggml(), sequences, json, out),
        Command::Diff { a, b, json } => commands::diff(&a, &b, json, out),
        Command::Types { json } => commands::types(json, out),
    }
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let mut out = BufWriter::new(io::stdout().lock());
    let result = run(cli, &mut out).and_then(|outcome| {
        out.flush()?;
        Ok(outcome)
    });
    match result {
        Ok(Outcome::Ok) => ExitCode::SUCCESS,
        Ok(Outcome::Differences) => ExitCode::from(1),
        // `gguf-inspect tensors model.gguf | head` closes the pipe early.
        Err(Failure::Output(e)) if e.kind() == io::ErrorKind::BrokenPipe => ExitCode::SUCCESS,
        Err(err) => {
            let message = match err {
                Failure::Output(e) => format!("writing output: {e}"),
                Failure::Message(m) => m,
            };
            // Messages can quote keys and tensor names from the file.
            let message = escape::text(&message);
            // Nothing sensible is left to do if stderr is gone too.
            let _ = writeln!(io::stderr(), "gguf-inspect: error: {message}");
            ExitCode::from(2)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    #[test]
    fn cli_definition_is_valid() {
        Cli::command().debug_assert();
    }

    #[test]
    fn cache_types_map_to_matching_ggml_names() {
        for cache in CacheType::value_variants() {
            let name = cache.to_possible_value().unwrap();
            assert_eq!(cache.ggml().name().to_lowercase(), name.get_name());
        }
    }
}
