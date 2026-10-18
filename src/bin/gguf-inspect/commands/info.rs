//! `gguf-inspect info`

use std::io::Write;
use std::path::Path;

use gguf_inspect::model::{PerLayer, file_type_name};
use gguf_inspect::{GgufFile, ModelInfo, TypeStats};
use serde::Serialize;

use super::{PerLayerJson, write_json};
use crate::format::{Table, attention, bytes, count, percent, thousands};
use crate::{Failure, Outcome, escape, open};

pub(crate) fn info(path: &Path, json: bool, out: &mut dyn Write) -> Result<Outcome, Failure> {
    let gguf = open(path)?;
    let model = ModelInfo::from_file(&gguf);
    let breakdown = gguf.type_breakdown();
    let warnings: Vec<String> = gguf.validate().iter().map(ToString::to_string).collect();
    if json {
        write_json(
            out,
            &InfoJson::new(path, &gguf, &model, &breakdown, warnings),
        )?;
    } else {
        render(out, path, &gguf, &model, &breakdown, &warnings)?;
    }
    Ok(Outcome::Ok)
}

fn render(
    out: &mut dyn Write,
    path: &Path,
    gguf: &GgufFile,
    model: &ModelInfo,
    breakdown: &[TypeStats],
    warnings: &[String],
) -> Result<(), Failure> {
    let mut file = Table::plain();
    file.row(["file".to_owned(), path.display().to_string()]);
    file.row([
        "size".to_owned(),
        format!(
            "{} ({} bytes)",
            bytes(gguf.file_size()),
            thousands(gguf.file_size())
        ),
    ]);
    file.row([
        "format".to_owned(),
        format!(
            "GGUF v{}, alignment {}, data offset {}",
            gguf.version(),
            gguf.alignment(),
            thousands(gguf.data_offset())
        ),
    ]);
    file.row(["metadata", &format!("{} entries", gguf.metadata().len())]);
    file.row(["tensors", &thousands(gguf.tensors().len() as u64)]);
    file.render(out)?;
    writeln!(out)?;

    let mut m = Table::plain();
    let mut opt = |label: &str, value: Option<String>| {
        if let Some(v) = value {
            m.row([label.to_owned(), v]);
        }
    };
    opt("architecture", model.architecture.clone());
    opt("name", model.name.clone());
    let params = gguf.parameter_count();
    opt(
        "parameters",
        Some(format!("{} ({})", count(params), thousands(params))),
    );
    opt(
        "file type",
        model.file_type.map(|id| match file_type_name(id) {
            Some(name) => format!("{name} ({id})"),
            None => id.to_string(),
        }),
    );
    opt("context length", model.context_length.map(thousands));
    opt("embedding", model.embedding_length.map(thousands));
    opt("layers", model.block_count.map(thousands));
    opt(
        "feed forward",
        model.feed_forward_length.as_ref().map(per_layer),
    );
    opt("heads", model.head_count.as_ref().map(per_layer));
    opt(
        "kv heads",
        model
            .head_count_kv
            .as_ref()
            .map(|kv| describe_kv_heads(kv, model.head_count.as_ref())),
    );
    opt("head size", head_size(model));
    opt(
        "experts",
        model.expert_count.map(|n| match model.expert_used_count {
            Some(used) => format!("{n} ({used} used per token)"),
            None => n.to_string(),
        }),
    );
    opt("rope", rope(model));
    opt("tokenizer", tokenizer(model));
    m.render(out)?;

    if !breakdown.is_empty() {
        writeln!(out)?;
        writeln!(out, "quantization")?;
        let total = gguf.tensor_data_size();
        let mut t = Table::with_header(&["TYPE", "TENSORS", "PARAMETERS", "BYTES", "SHARE", "BPW"])
            .right(&[1, 2, 3, 4, 5])
            .indent(2);
        for s in breakdown {
            t.row([
                s.ggml_type.name().to_owned(),
                s.tensors.to_string(),
                count(s.elements),
                bytes(s.bytes),
                percent(s.bytes, total),
                format!("{:.2}", s.ggml_type.bits_per_element()),
            ]);
        }
        t.render(out)?;
    }

    if !warnings.is_empty() {
        writeln!(out)?;
        writeln!(out, "warnings")?;
        for w in warnings {
            writeln!(out, "  - {}", escape::text(w))?;
        }
    }
    Ok(())
}

/// Single values with thousands separators; per-layer lists as a range.
fn per_layer(value: &PerLayer) -> String {
    match value {
        PerLayer::Uniform(v) => thousands(*v),
        list @ PerLayer::List(_) => list.to_string(),
    }
}

fn describe_kv_heads(kv: &PerLayer, heads: Option<&PerLayer>) -> String {
    match attention(kv, heads) {
        Some(kind) => format!("{} ({kind})", per_layer(kv)),
        None => per_layer(kv),
    }
}

fn head_size(model: &ModelInfo) -> Option<String> {
    match (model.key_length, model.value_length) {
        (None, None) => None,
        (k, v) => {
            let show = |x: Option<u64>| x.map_or_else(|| "derived".to_owned(), |x| x.to_string());
            Some(format!("key {}, value {}", show(k), show(v)))
        }
    }
}

fn rope(model: &ModelInfo) -> Option<String> {
    let r = &model.rope;
    let mut parts = Vec::new();
    if let Some(d) = r.dimension_count {
        parts.push(format!("dimensions {d}"));
    }
    if let Some(base) = r.freq_base {
        parts.push(format!("freq base {base}"));
    }
    if let Some(kind) = &r.scaling_type {
        parts.push(format!("scaling {kind}"));
    }
    if let Some(factor) = r.scaling_factor {
        parts.push(format!("factor {factor}"));
    }
    if let Some(ctx) = r.original_context_length {
        parts.push(format!("original context {}", thousands(ctx)));
    }
    (!parts.is_empty()).then(|| parts.join(", "))
}

fn tokenizer(model: &ModelInfo) -> Option<String> {
    let t = &model.tokenizer;
    match (&t.model, t.vocab_size) {
        (None, None) => None,
        (Some(m), Some(v)) => Some(format!("{m}, vocabulary {}", thousands(v))),
        (Some(m), None) => Some(m.clone()),
        (None, Some(v)) => Some(format!("vocabulary {}", thousands(v))),
    }
}

#[derive(Serialize)]
struct FileTypeJson {
    id: u64,
    name: Option<&'static str>,
}

#[derive(Serialize)]
struct RopeJson<'a> {
    dimension_count: Option<u64>,
    freq_base: Option<f64>,
    scaling_type: Option<&'a str>,
    scaling_factor: Option<f64>,
    original_context_length: Option<u64>,
}

#[derive(Serialize)]
struct TokenizerJson<'a> {
    model: Option<&'a str>,
    vocab_size: Option<u64>,
}

#[derive(Serialize)]
struct ModelJson<'a> {
    architecture: Option<&'a str>,
    name: Option<&'a str>,
    file_type: Option<FileTypeJson>,
    quantization_version: Option<u64>,
    context_length: Option<u64>,
    embedding_length: Option<u64>,
    block_count: Option<u64>,
    feed_forward_length: Option<PerLayerJson<'a>>,
    head_count: Option<PerLayerJson<'a>>,
    head_count_kv: Option<PerLayerJson<'a>>,
    key_length: Option<u64>,
    value_length: Option<u64>,
    expert_count: Option<u64>,
    expert_used_count: Option<u64>,
    rope: RopeJson<'a>,
    tokenizer: TokenizerJson<'a>,
}

#[derive(Serialize)]
struct QuantJson {
    ggml_type: &'static str,
    tensors: usize,
    elements: u64,
    bytes: u64,
    bits_per_weight: f64,
}

#[derive(Serialize)]
struct InfoJson<'a> {
    file: String,
    file_size: u64,
    gguf_version: u32,
    alignment: u32,
    header_size: u64,
    data_offset: u64,
    metadata_count: usize,
    tensor_count: usize,
    parameter_count: u64,
    tensor_data_size: u64,
    model: ModelJson<'a>,
    quantization: Vec<QuantJson>,
    warnings: Vec<String>,
}

impl<'a> InfoJson<'a> {
    fn new(
        path: &Path,
        gguf: &GgufFile,
        m: &'a ModelInfo,
        breakdown: &[TypeStats],
        warnings: Vec<String>,
    ) -> Self {
        Self {
            file: path.display().to_string(),
            file_size: gguf.file_size(),
            gguf_version: gguf.version().number(),
            alignment: gguf.alignment(),
            header_size: gguf.header_size(),
            data_offset: gguf.data_offset(),
            metadata_count: gguf.metadata().len(),
            tensor_count: gguf.tensors().len(),
            parameter_count: gguf.parameter_count(),
            tensor_data_size: gguf.tensor_data_size(),
            model: ModelJson {
                architecture: m.architecture.as_deref(),
                name: m.name.as_deref(),
                file_type: m.file_type.map(|id| FileTypeJson {
                    id,
                    name: file_type_name(id),
                }),
                quantization_version: m.quantization_version,
                context_length: m.context_length,
                embedding_length: m.embedding_length,
                block_count: m.block_count,
                feed_forward_length: m.feed_forward_length.as_ref().map(Into::into),
                head_count: m.head_count.as_ref().map(Into::into),
                head_count_kv: m.head_count_kv.as_ref().map(Into::into),
                key_length: m.key_length,
                value_length: m.value_length,
                expert_count: m.expert_count,
                expert_used_count: m.expert_used_count,
                rope: RopeJson {
                    dimension_count: m.rope.dimension_count,
                    freq_base: m.rope.freq_base,
                    scaling_type: m.rope.scaling_type.as_deref(),
                    scaling_factor: m.rope.scaling_factor,
                    original_context_length: m.rope.original_context_length,
                },
                tokenizer: TokenizerJson {
                    model: m.tokenizer.model.as_deref(),
                    vocab_size: m.tokenizer.vocab_size,
                },
            },
            quantization: breakdown
                .iter()
                .map(|s| QuantJson {
                    ggml_type: s.ggml_type.name(),
                    tensors: s.tensors,
                    elements: s.elements,
                    bytes: s.bytes,
                    bits_per_weight: s.ggml_type.bits_per_element(),
                })
                .collect(),
            warnings,
        }
    }
}
