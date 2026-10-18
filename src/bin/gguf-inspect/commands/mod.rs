//! One module per subcommand. Each renders either aligned text or JSON.

mod diff;
mod info;
mod kv_cache;
mod metadata;
mod tensors;
mod types;

use std::io::Write;

use gguf_inspect::model::PerLayer;
use serde::{Serialize, Serializer};

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

/// A JSON array produced from an iterator as it is written, so a listing of
/// a million entries is never collected in memory.
struct JsonSeq<F>(F);

impl<F, I> Serialize for JsonSeq<F>
where
    F: Fn() -> I,
    I: IntoIterator,
    I::Item: Serialize,
{
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_seq((self.0)())
    }
}

/// `1 tensor`, `2 tensors`.
fn plural(n: u64, one: &str, many: &str) -> String {
    let word = if n == 1 { one } else { many };
    format!("{} {word}", crate::format::thousands(n))
}
