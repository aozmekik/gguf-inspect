//! `gguf-inspect kv-cache`

use std::io::Write;
use std::path::Path;

use gguf_inspect::kv_cache::{HeadDimSource, KvCacheConfig, KvCacheEstimate, estimate};
use gguf_inspect::model::PerLayer;
use gguf_inspect::{GgmlType, ModelInfo};
use serde::Serialize;

use super::write_json;
use crate::format::{Table, bytes, thousands};
use crate::{Failure, Outcome, escape, open};

#[derive(Serialize)]
#[serde(untagged)]
enum Heads<'a> {
    Uniform(u64),
    PerLayer(&'a [u64]),
}

impl<'a> From<&'a PerLayer> for Heads<'a> {
    fn from(p: &'a PerLayer) -> Self {
        match p {
            PerLayer::Uniform(v) => Self::Uniform(*v),
            PerLayer::Varying(v) => Self::PerLayer(v),
        }
    }
}

#[derive(Serialize)]
struct KvJson<'a> {
    file: String,
    architecture: Option<&'a str>,
    context_length: u64,
    batch: u64,
    cache_type: &'static str,
    block_count: u64,
    head_count: Option<Heads<'a>>,
    head_count_kv: Heads<'a>,
    key_length: u64,
    value_length: u64,
    head_dim_source: &'static str,
    k_bytes_per_token: u64,
    v_bytes_per_token: u64,
    bytes_per_token: u64,
    total_bytes: u64,
    notes: Vec<String>,
}

pub(crate) fn kv_cache(
    path: &Path,
    ctx: u64,
    cache_type: GgmlType,
    batch: u64,
    json: bool,
    out: &mut dyn Write,
) -> Result<Outcome, Failure> {
    let gguf = open(path)?;
    let config = KvCacheConfig::new(ctx)
        .with_cache_type(cache_type)
        .with_sequences(batch);
    let e = estimate(&gguf, &config).map_err(|err| Failure::with_path(path, err))?;
    let model = ModelInfo::from_file(&gguf);

    if json {
        write_json(
            out,
            &KvJson {
                file: path.display().to_string(),
                architecture: model.architecture.as_deref(),
                context_length: ctx,
                batch,
                cache_type: cache_type.name(),
                block_count: e.block_count,
                head_count: e.head_count.as_ref().map(Into::into),
                head_count_kv: (&e.head_count_kv).into(),
                key_length: e.key_length,
                value_length: e.value_length,
                head_dim_source: match e.head_dim_source {
                    HeadDimSource::Explicit => "explicit",
                    HeadDimSource::Derived => "derived",
                },
                k_bytes_per_token: e.k_bytes_per_token,
                v_bytes_per_token: e.v_bytes_per_token,
                bytes_per_token: e.bytes_per_token(),
                total_bytes: e.total_bytes,
                notes: e.notes.iter().map(ToString::to_string).collect(),
            },
        )?;
        return Ok(Outcome::Ok);
    }
    render(out, path, &model, &e)?;
    Ok(Outcome::Ok)
}

fn render(
    out: &mut dyn Write,
    path: &Path,
    model: &ModelInfo,
    estimate: &KvCacheEstimate,
) -> Result<(), Failure> {
    let arch = model.architecture.as_deref().unwrap_or("?");
    writeln!(
        out,
        "KV cache estimate for {} ({})",
        escape::text(&path.display().to_string()),
        escape::text(arch)
    )?;
    writeln!(out)?;
    inputs(model, estimate).render(out)?;
    writeln!(out)?;
    writeln!(out, "formula")?;
    formula(out, estimate)?;
    if !estimate.notes.is_empty() {
        writeln!(out)?;
        writeln!(out, "notes")?;
        for note in &estimate.notes {
            writeln!(out, "  - {}", escape::text(&note.to_string()))?;
        }
    }
    Ok(())
}

/// The hyperparameters that enter the formula, with where they came from.
fn inputs(model: &ModelInfo, e: &KvCacheEstimate) -> Table {
    let arch = model.architecture.as_deref().unwrap_or("?");
    let source = |explicit: Option<u64>, key: &str| match (
        explicit,
        model.embedding_length,
        &model.head_count,
    ) {
        (Some(_), ..) => format!("from {arch}.{key}"),
        (None, Some(embd), Some(heads)) => format!("embedding_length {embd} / head_count {heads}"),
        _ => "derived".to_owned(),
    };
    let gqa = match (
        e.head_count_kv.uniform(),
        e.head_count.as_ref().and_then(PerLayer::uniform),
    ) {
        (Some(k), Some(h)) if k > 0 && k < h && h % k == 0 => {
            format!("grouped-query attention, {} heads per KV head", h / k)
        }
        (Some(k), Some(h)) if k == h => "one KV head per head".to_owned(),
        _ => String::new(),
    };
    let ty = e.config.cache_type;
    let layout = if ty.block_size() == 1 {
        format!("{} bytes per element", ty.type_size())
    } else {
        format!("{} bytes per {} elements", ty.type_size(), ty.block_size())
    };

    let mut table = Table::plain().indent(2);
    table.row(["block_count", &e.block_count.to_string(), ""]);
    if let Some(h) = &e.head_count {
        table.row(["head_count", &h.to_string(), ""]);
    }
    table.row(["head_count_kv".to_owned(), e.head_count_kv.to_string(), gqa]);
    table.row([
        "key_length".to_owned(),
        e.key_length.to_string(),
        source(model.key_length, "attention.key_length"),
    ]);
    table.row([
        "value_length".to_owned(),
        e.value_length.to_string(),
        source(model.value_length, "attention.value_length"),
    ]);
    table.row(["cache type".to_owned(), ty.name().to_owned(), layout]);
    table.row(["context", &thousands(e.config.context_length), "tokens"]);
    let sequences = if e.config.sequences == 1 {
        "sequence"
    } else {
        "sequences"
    };
    table.row(["batch", &thousands(e.config.sequences), sequences]);
    table
}

/// The formula with the file's numbers substituted.
fn formula(out: &mut dyn Write, e: &KvCacheEstimate) -> Result<(), Failure> {
    let ty = e.config.cache_type;
    let ctx = thousands(e.config.context_length);
    let batch = thousands(e.config.sequences);
    if let Some(kv) = e.head_count_kv.uniform() {
        let layers = e.block_count.max(1);
        let row = |dim: u64, bytes_per_layer: u64| {
            format!(
                "row_size({ty}, {kv} x {dim} = {}) = {} B",
                thousands(kv * dim),
                thousands(bytes_per_layer)
            )
        };
        let k = row(e.key_length, e.k_bytes_per_token / layers);
        let v = row(e.value_length, e.v_bytes_per_token / layers);
        writeln!(out, "  K per layer and token = {k}")?;
        writeln!(out, "  V per layer and token = {v}")?;
        writeln!(out, "  total = batch x ctx x block_count x (K + V)")?;
        writeln!(
            out,
            "        = {batch} x {ctx} x {} x {} B",
            e.block_count,
            thousands(e.bytes_per_token() / layers)
        )?;
    } else {
        for (name, dim, bytes) in [
            ("K", e.key_length, e.k_bytes_per_token),
            ("V", e.value_length, e.v_bytes_per_token),
        ] {
            writeln!(
                out,
                "  {name} per token = sum over layers of row_size({ty}, head_count_kv[layer] x {dim}) = {} B",
                thousands(bytes)
            )?;
        }
        writeln!(out, "  total = batch x ctx x (K + V)")?;
        writeln!(
            out,
            "        = {batch} x {ctx} x {} B",
            thousands(e.bytes_per_token())
        )?;
    }
    writeln!(
        out,
        "        = {} B ({})",
        thousands(e.total_bytes),
        bytes(e.total_bytes)
    )?;
    Ok(())
}
