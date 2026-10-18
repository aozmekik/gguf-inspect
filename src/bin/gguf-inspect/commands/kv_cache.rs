//! `gguf-inspect kv-cache`

use std::io::Write;
use std::path::Path;

use gguf_inspect::kv_cache::{
    HeadDimSource, KvCacheConfig, KvCacheEstimate, KvCacheNote, estimate,
};
use gguf_inspect::{GgmlType, ModelInfo};
use serde::Serialize;

use super::{PerLayerJson, write_json};
use crate::format::{Table, attention, bytes, thousands};
use crate::{Failure, Outcome, escape, open};

#[derive(Serialize)]
struct NoteJson {
    kind: &'static str,
    message: String,
}

impl From<&KvCacheNote> for NoteJson {
    fn from(note: &KvCacheNote) -> Self {
        let kind = match note {
            KvCacheNote::SlidingWindow(_) => "sliding_window",
            KvCacheNote::LatentAttention { .. } => "latent_attention",
            KvCacheNote::UncompressedLatentAttention(_) => "uncompressed_latent_attention",
            KvCacheNote::RecurrentState => "recurrent_state",
            KvCacheNote::LayersWithoutKv(_) => "layers_without_kv",
            _ => "other",
        };
        Self {
            kind,
            message: note.to_string(),
        }
    }
}

#[derive(Serialize)]
struct KvJson<'a> {
    file: String,
    architecture: Option<&'a str>,
    tokens_per_sequence: u64,
    sequences: u64,
    cache_type: &'static str,
    block_count: u64,
    head_count: Option<PerLayerJson<'a>>,
    head_count_kv: PerLayerJson<'a>,
    key_length: u64,
    value_length: u64,
    head_dim_source: &'static str,
    value_cache: bool,
    k_bytes_per_token: u64,
    v_bytes_per_token: u64,
    bytes_per_token: u64,
    total_bytes: u64,
    notes: Vec<NoteJson>,
}

pub(crate) fn kv_cache(
    path: &Path,
    ctx: u64,
    cache_type: GgmlType,
    sequences: u64,
    json: bool,
    out: &mut dyn Write,
) -> Result<Outcome, Failure> {
    let gguf = open(path)?;
    let config = KvCacheConfig::new(ctx)
        .with_cache_type(cache_type)
        .with_sequences(sequences);
    let e = estimate(&gguf, &config).map_err(|err| Failure::with_path(path, &err))?;
    let model = ModelInfo::from_file(&gguf);

    if json {
        write_json(
            out,
            &KvJson {
                file: path.display().to_string(),
                architecture: model.architecture.as_deref(),
                tokens_per_sequence: ctx,
                sequences,
                cache_type: cache_type.name(),
                block_count: e.block_count,
                head_count: e.head_count.as_ref().map(Into::into),
                head_count_kv: (&e.head_count_kv).into(),
                key_length: e.key_length,
                value_length: e.value_length,
                head_dim_source: match e.head_dim_source {
                    HeadDimSource::Explicit => "explicit",
                    HeadDimSource::Derived => "derived",
                    HeadDimSource::Unused => "unused",
                    _ => "other",
                },
                value_cache: e.has_value_cache,
                k_bytes_per_token: e.k_bytes_per_token,
                v_bytes_per_token: e.v_bytes_per_token,
                bytes_per_token: e.bytes_per_token(),
                total_bytes: e.total_bytes,
                notes: e.notes.iter().map(Into::into).collect(),
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
    let source = |explicit: Option<u64>, key: &str| match (explicit, e.head_dim_source) {
        (Some(_), _) => format!("from {arch}.{key}"),
        (None, HeadDimSource::Unused) => "unused: no layer has KV heads".to_owned(),
        (None, _) => match (model.embedding_length, &model.head_count) {
            (Some(embd), Some(heads)) => format!("embedding_length {embd} / head_count {heads}"),
            _ => "derived".to_owned(),
        },
    };
    let ty = e.config.cache_type;
    let layout = if ty.block_size() == 1 {
        format!("{} bytes per element", ty.type_size())
    } else {
        format!("{} bytes per {} elements", ty.type_size(), ty.block_size())
    };
    let mut value_source = source(model.value_length, "attention.value_length");
    if !e.has_value_cache {
        value_source.push_str("; not cached (multi-head latent attention)");
    }

    let mut table = Table::plain().indent(2);
    table.row(["block_count", &e.block_count.to_string(), ""]);
    if let Some(h) = &e.head_count {
        table.row(["head_count", &h.to_string(), ""]);
    }
    table.row([
        "head_count_kv".to_owned(),
        e.head_count_kv.to_string(),
        attention(&e.head_count_kv, e.head_count.as_ref()).unwrap_or_default(),
    ]);
    table.row([
        "key_length".to_owned(),
        e.key_length.to_string(),
        source(model.key_length, "attention.key_length"),
    ]);
    table.row([
        "value_length".to_owned(),
        e.value_length.to_string(),
        value_source,
    ]);
    table.row(["cache type".to_owned(), ty.name().to_owned(), layout]);
    table.row([
        "context",
        &thousands(e.config.context_length),
        "tokens per sequence",
    ]);
    table.row(["sequences", &thousands(e.config.sequences), ""]);
    table
}

/// The formula with the file's numbers substituted.
fn formula(out: &mut dyn Write, e: &KvCacheEstimate) -> Result<(), Failure> {
    let ty = e.config.cache_type;
    let ctx = thousands(e.config.context_length);
    let sequences = thousands(e.config.sequences);
    if let Some(kv) = e.head_count_kv.uniform() {
        // The estimate has checked that these rows are whole blocks and fit.
        let row = |dim: u64| {
            let elements = kv.saturating_mul(dim);
            let row_bytes = ty.row_size(elements).unwrap_or_default();
            (
                format!(
                    "row_size({ty}, {kv} x {dim} = {}) = {} B",
                    thousands(elements),
                    thousands(row_bytes)
                ),
                row_bytes,
            )
        };
        let (k, k_bytes) = row(e.key_length);
        writeln!(out, "  K per layer and token = {k}")?;
        let v_bytes = if e.has_value_cache {
            let (v, v_bytes) = row(e.value_length);
            writeln!(out, "  V per layer and token = {v}")?;
            v_bytes
        } else {
            writeln!(
                out,
                "  V per layer and token = 0 B (no V cache with multi-head latent attention)"
            )?;
            0
        };
        writeln!(out, "  total = sequences x ctx x block_count x (K + V)")?;
        writeln!(
            out,
            "        = {sequences} x {ctx} x {} x {} B",
            thousands(e.block_count),
            thousands(k_bytes.saturating_add(v_bytes))
        )?;
    } else {
        writeln!(
            out,
            "  K per token = sum over layers of row_size({ty}, head_count_kv[layer] x {}) = {} B",
            e.key_length,
            thousands(e.k_bytes_per_token)
        )?;
        if e.has_value_cache {
            writeln!(
                out,
                "  V per token = sum over layers of row_size({ty}, head_count_kv[layer] x {}) = {} B",
                e.value_length,
                thousands(e.v_bytes_per_token)
            )?;
        } else {
            writeln!(
                out,
                "  V per token = 0 B (no V cache with multi-head latent attention)"
            )?;
        }
        writeln!(out, "  total = sequences x ctx x (K + V)")?;
        writeln!(
            out,
            "        = {sequences} x {ctx} x {} B",
            thousands(e.bytes_per_token())
        )?;
    }
    let total = thousands(e.total_bytes);
    if e.total_bytes < 1024 {
        writeln!(out, "        = {total} B")?;
    } else {
        writeln!(out, "        = {total} B ({})", bytes(e.total_bytes))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_note_has_a_kind() {
        let notes = [
            KvCacheNote::SlidingWindow(1),
            KvCacheNote::LatentAttention {
                key_length_mla: 1,
                value_length_mla: 1,
            },
            KvCacheNote::UncompressedLatentAttention(1),
            KvCacheNote::RecurrentState,
            KvCacheNote::LayersWithoutKv(1),
        ];
        for note in &notes {
            assert_ne!(NoteJson::from(note).kind, "other", "{note}");
        }
    }
}
