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

/// Writes `value` as pretty-printed JSON followed by a newline.
fn write_json(out: &mut dyn Write, value: &impl Serialize) -> Result<(), Failure> {
    serde_json::to_writer_pretty(&mut *out, value)?;
    writeln!(out)?;
    Ok(())
}
