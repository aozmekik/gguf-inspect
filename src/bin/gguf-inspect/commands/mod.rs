//! One module per subcommand. Each renders either aligned text or JSON.

mod diff;
mod info;
mod kv_cache;
mod metadata;
mod tensors;
mod types;

use std::io::Write;

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
