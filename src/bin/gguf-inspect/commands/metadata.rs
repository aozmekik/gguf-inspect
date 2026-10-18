//! `gguf-inspect metadata`

use std::io::Write;
use std::path::Path;

use gguf_inspect::MetadataValue;
use serde::Serialize;
use serde_json::Value as Json;

use super::write_json;
use crate::format::Table;
use crate::values::{PREVIEW_ITEMS, Preview, render, to_json, type_name};
use crate::{Failure, Outcome, escape, open};

#[derive(Serialize)]
struct EntryJson<'a> {
    key: &'a str,
    #[serde(rename = "type")]
    value_type: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    element_type: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    len: Option<usize>,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    truncated: bool,
    value: Json,
}

pub(crate) fn metadata(
    path: &Path,
    json: bool,
    filter: Option<&str>,
    full: bool,
    out: &mut dyn Write,
) -> Result<Outcome, Failure> {
    let gguf = open(path)?;
    let entries = gguf
        .metadata()
        .iter()
        .filter(|(key, _)| filter.is_none_or(|prefix| key.starts_with(prefix)));

    if json {
        let max_items = (!full).then_some(PREVIEW_ITEMS);
        let rows: Vec<EntryJson<'_>> = entries
            .map(|(key, value)| {
                let array = value.as_array();
                EntryJson {
                    key,
                    value_type: value.value_type().name(),
                    element_type: array.map(|a| a.element_type().name()),
                    len: array.map(gguf_inspect::MetadataArray::len),
                    truncated: array.is_some_and(|a| max_items.is_some_and(|m| a.len() > m)),
                    value: to_json(value, max_items),
                }
            })
            .collect();
        write_json(out, &rows)?;
        return Ok(Outcome::Ok);
    }

    let preview = if full { Preview::FULL } else { Preview::SHORT };
    let mut table = Table::with_header(&["KEY", "TYPE", "VALUE"]);
    let mut any_truncated = false;
    for (key, value) in entries {
        let (text, truncated) = render(value, preview);
        any_truncated |= truncated;
        table.row([key.to_owned(), type_name(value), with_len(value, text)]);
    }
    if table.is_empty() {
        if let Some(prefix) = filter {
            writeln!(
                out,
                "no metadata keys start with `{}`",
                escape::text(prefix)
            )?;
        } else {
            writeln!(out, "no metadata")?;
        }
        return Ok(Outcome::Ok);
    }
    table.render(out)?;
    if any_truncated {
        writeln!(out)?;
        writeln!(
            out,
            "long arrays and strings are truncated; use --full to show them"
        )?;
    }
    Ok(Outcome::Ok)
}

/// Appends the element count to array renderings that hide elements.
fn with_len(value: &MetadataValue, text: String) -> String {
    match value.as_array() {
        Some(a) if a.len() > PREVIEW_ITEMS => {
            format!(
                "{text} ({} items)",
                crate::format::thousands(a.len() as u64)
            )
        }
        _ => text,
    }
}
