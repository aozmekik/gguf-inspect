//! `gguf-inspect diff`

use std::io::Write;
use std::path::Path;

use gguf_inspect::diff::{FileDiff, MetadataChange, TensorChange};
use gguf_inspect::{MetadataValue, TensorInfo};
use serde::Serialize;
use serde_json::Value as Json;

use super::write_json;
use crate::format::{Table, shape, thousands};
use crate::values::{PREVIEW_ITEMS, Preview, render, to_json, type_name};
use crate::{Failure, Outcome, escape, open};

pub(crate) fn diff(
    a: &Path,
    b: &Path,
    json: bool,
    out: &mut dyn Write,
) -> Result<Outcome, Failure> {
    let old = open(a)?;
    let new = open(b)?;
    let d = gguf_inspect::diff::diff(&old, &new);
    if json {
        write_json(out, &to_json_report(a, b, &d))?;
    } else {
        render_text(out, a, b, &d)?;
    }
    Ok(if d.is_empty() {
        Outcome::Ok
    } else {
        Outcome::Differences
    })
}

fn render_text(
    out: &mut dyn Write,
    old_path: &Path,
    new_path: &Path,
    changes: &FileDiff<'_>,
) -> Result<(), Failure> {
    writeln!(out, "--- {}", escape::text(&old_path.display().to_string()))?;
    writeln!(out, "+++ {}", escape::text(&new_path.display().to_string()))?;
    if changes.is_empty() {
        writeln!(out)?;
        writeln!(out, "headers are identical (metadata and tensor infos)")?;
        return Ok(());
    }

    if changes.version.is_some() || changes.alignment.is_some() {
        writeln!(out)?;
        writeln!(out, "header")?;
        let mut table = Table::plain().indent(2);
        if let Some((old, new)) = changes.version {
            table.row(["version".to_owned(), format!("{old} -> {new}")]);
        }
        if let Some((old, new)) = changes.alignment {
            table.row(["alignment".to_owned(), format!("{old} -> {new}")]);
        }
        table.render(out)?;
    }

    if !changes.metadata.is_empty() {
        writeln!(out)?;
        writeln!(out, "metadata: {} difference(s)", changes.metadata.len())?;
        let mut table = Table::plain().indent(2);
        for change in &changes.metadata {
            table.row(match *change {
                MetadataChange::Removed { key, value } => [
                    "-".to_owned(),
                    key.to_owned(),
                    type_name(value),
                    short(value),
                ],
                MetadataChange::Added { key, value } => [
                    "+".to_owned(),
                    key.to_owned(),
                    type_name(value),
                    short(value),
                ],
                MetadataChange::Changed { key, old, new } => {
                    let (from, to) = (type_name(old), type_name(new));
                    let types = if from == to {
                        from
                    } else {
                        format!("{from} -> {to}")
                    };
                    [
                        "~".to_owned(),
                        key.to_owned(),
                        types,
                        describe_change(old, new),
                    ]
                }
            });
        }
        table.render(out)?;
    }

    if !changes.tensors.is_empty() {
        writeln!(out)?;
        writeln!(out, "tensors: {} difference(s)", changes.tensors.len())?;
        let describe = |t: &TensorInfo| format!("{} {}", t.ggml_type(), shape(t.dims()));
        let mut table = Table::plain().indent(2);
        for change in &changes.tensors {
            table.row(match *change {
                TensorChange::Removed(tensor) => {
                    ["-".to_owned(), tensor.name().to_owned(), describe(tensor)]
                }
                TensorChange::Added(tensor) => {
                    ["+".to_owned(), tensor.name().to_owned(), describe(tensor)]
                }
                TensorChange::Changed { old, new } => [
                    "~".to_owned(),
                    old.name().to_owned(),
                    format!("{} -> {}", describe(old), describe(new)),
                ],
            });
        }
        table.render(out)?;
    }
    Ok(())
}

fn short(value: &MetadataValue) -> String {
    render(value, Preview::SHORT).0
}

/// Explains how a value changed; for arrays, where.
fn describe_change(old: &MetadataValue, new: &MetadataValue) -> String {
    if let (Some(x), Some(y)) = (old.as_array(), new.as_array())
        && x.element_type() == y.element_type()
    {
        let lengths = || {
            format!(
                "length {} -> {}",
                thousands(x.len() as u64),
                thousands(y.len() as u64)
            )
        };
        let element = |a: &gguf_inspect::MetadataArray, i: usize| {
            a.get(i).map(|v| short(&v)).unwrap_or_default()
        };
        return match x.first_difference(y) {
            Some(i) if i < x.len() && i < y.len() => format!(
                "first difference at index {}: {} -> {}{}",
                thousands(i as u64),
                element(x, i),
                element(y, i),
                if x.len() == y.len() {
                    String::new()
                } else {
                    format!("; {}", lengths())
                }
            ),
            Some(_) => format!("{} (shared elements identical)", lengths()),
            // Unreachable for arrays the diff reported, kept for totality.
            None => "identical".to_owned(),
        };
    }
    format!("{} -> {}", short(old), short(new))
}

#[derive(Serialize)]
struct Pair<T> {
    old: T,
    new: T,
}

#[derive(Serialize)]
struct ValueJson {
    #[serde(rename = "type")]
    value_type: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    element_type: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    len: Option<usize>,
    value: Json,
}

impl From<&MetadataValue> for ValueJson {
    fn from(value: &MetadataValue) -> Self {
        let array = value.as_array();
        Self {
            value_type: value.value_type().name(),
            element_type: array.map(|a| a.element_type().name()),
            len: array.map(gguf_inspect::MetadataArray::len),
            value: to_json(value, Some(PREVIEW_ITEMS)),
        }
    }
}

#[derive(Serialize)]
struct TensorJson<'a> {
    #[serde(rename = "type")]
    ggml_type: &'static str,
    dims: &'a [u64],
}

impl<'a> From<&'a TensorInfo> for TensorJson<'a> {
    fn from(t: &'a TensorInfo) -> Self {
        Self {
            ggml_type: t.ggml_type().name(),
            dims: t.dims(),
        }
    }
}

#[derive(Serialize)]
struct ChangeJson<'a, T> {
    change: &'static str,
    #[serde(rename = "key")]
    name: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    old: Option<T>,
    #[serde(skip_serializing_if = "Option::is_none")]
    new: Option<T>,
}

#[derive(Serialize)]
struct DiffJson<'a> {
    old_file: String,
    new_file: String,
    identical: bool,
    version: Option<Pair<u32>>,
    alignment: Option<Pair<u32>>,
    metadata: Vec<ChangeJson<'a, ValueJson>>,
    tensors: Vec<TensorChangeJson<'a>>,
}

/// Like [`ChangeJson`] but keyed by `name`.
#[derive(Serialize)]
struct TensorChangeJson<'a> {
    change: &'static str,
    name: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    old: Option<TensorJson<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    new: Option<TensorJson<'a>>,
}

fn to_json_report<'a>(old_path: &Path, new_path: &Path, changes: &FileDiff<'a>) -> DiffJson<'a> {
    let metadata = changes
        .metadata
        .iter()
        .map(|c| match *c {
            MetadataChange::Removed { key, value } => ChangeJson {
                change: "removed",
                name: key,
                old: Some(value.into()),
                new: None,
            },
            MetadataChange::Added { key, value } => ChangeJson {
                change: "added",
                name: key,
                old: None,
                new: Some(value.into()),
            },
            MetadataChange::Changed { key, old, new } => ChangeJson {
                change: "changed",
                name: key,
                old: Some(old.into()),
                new: Some(new.into()),
            },
        })
        .collect();
    let tensors = changes
        .tensors
        .iter()
        .map(|c| match *c {
            TensorChange::Removed(t) => TensorChangeJson {
                change: "removed",
                name: t.name(),
                old: Some(t.into()),
                new: None,
            },
            TensorChange::Added(t) => TensorChangeJson {
                change: "added",
                name: t.name(),
                old: None,
                new: Some(t.into()),
            },
            TensorChange::Changed { old, new } => TensorChangeJson {
                change: "changed",
                name: old.name(),
                old: Some(old.into()),
                new: Some(new.into()),
            },
        })
        .collect();
    DiffJson {
        old_file: old_path.display().to_string(),
        new_file: new_path.display().to_string(),
        identical: changes.is_empty(),
        version: changes.version.map(|(old, new)| Pair {
            old: old.number(),
            new: new.number(),
        }),
        alignment: changes.alignment.map(|(old, new)| Pair { old, new }),
        metadata,
        tensors,
    }
}
