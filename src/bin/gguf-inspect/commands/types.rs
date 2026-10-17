//! `gguf-inspect types`

use std::io::Write;

use gguf_inspect::GgmlType;
use serde::Serialize;

use super::write_json;
use crate::format::Table;
use crate::{Failure, Outcome};

#[derive(Serialize)]
struct TypeJson {
    id: u32,
    name: &'static str,
    block_size: u64,
    type_size: u64,
    bits_per_weight: f64,
    quantized: bool,
}

pub(crate) fn types(json: bool, out: &mut dyn Write) -> Result<Outcome, Failure> {
    if json {
        let rows: Vec<_> = GgmlType::ALL
            .iter()
            .map(|t| TypeJson {
                id: t.id(),
                name: t.name(),
                block_size: t.block_size(),
                type_size: t.type_size(),
                bits_per_weight: t.bits_per_element(),
                quantized: t.is_quantized(),
            })
            .collect();
        write_json(out, &rows)?;
        return Ok(Outcome::Ok);
    }
    let mut table = Table::with_header(&["ID", "NAME", "BLOCK", "BYTES", "BPW", "QUANTIZED"])
        .right(&[0, 2, 3, 4]);
    for t in GgmlType::ALL {
        table.row([
            t.id().to_string(),
            t.name().to_owned(),
            t.block_size().to_string(),
            t.type_size().to_string(),
            format!("{:.4}", t.bits_per_element()),
            if t.is_quantized() { "yes" } else { "no" }.to_owned(),
        ]);
    }
    table.render(out)?;
    Ok(Outcome::Ok)
}
