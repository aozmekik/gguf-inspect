//! One module per subcommand. Each renders either aligned text or JSON.

mod diff;
mod info;
mod kv_cache;
mod metadata;
mod tensors;
mod types;

use std::io::Write;

use gguf_inspect::model::PerLayer;
use serde::Serialize;

pub(crate) use diff::diff;
pub(crate) use info::info;
pub(crate) use kv_cache::kv_cache;
pub(crate) use metadata::metadata;
pub(crate) use tensors::tensors;
pub(crate) use types::types;

use crate::Failure;
use crate::escape::JsonFormatter;

/// Writes `value` as pretty-printed, terminal-safe JSON followed by a newline.
fn write_json(out: &mut dyn Write, value: &impl Serialize) -> Result<(), Failure> {
    let mut serializer = serde_json::Serializer::with_formatter(&mut *out, JsonFormatter::new());
    value.serialize(&mut serializer)?;
    writeln!(out)?;
    Ok(())
}

/// A per-layer hyperparameter in JSON: a number when one value applies to
/// every layer, an array (one entry per layer) when the file stores a list.
#[derive(Serialize)]
#[serde(untagged)]
enum PerLayerJson<'a> {
    Uniform(u64),
    List(&'a [u64]),
}

impl<'a> From<&'a PerLayer> for PerLayerJson<'a> {
    fn from(p: &'a PerLayer) -> Self {
        match p {
            PerLayer::Uniform(v) => Self::Uniform(*v),
            PerLayer::List(v) => Self::List(v),
        }
    }
}
