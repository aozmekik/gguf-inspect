//! `gguf-inspect tensors`

use std::io::Write;
use std::path::Path;

use gguf_inspect::TensorInfo;
use serde::Serialize;

use super::write_json;
use crate::format::{Table, bytes, count, shape, thousands};
use crate::{Failure, Outcome, TensorSort, open};

#[derive(Serialize)]
struct TensorJson<'a> {
    name: &'a str,
    #[serde(rename = "type")]
    ggml_type: &'static str,
    dims: &'a [u64],
    elements: u64,
    bytes: u64,
    offset: u64,
    absolute_offset: u64,
}

pub(crate) fn tensors(
    path: &Path,
    json: bool,
    sort: TensorSort,
    out: &mut dyn Write,
) -> Result<Outcome, Failure> {
    let gguf = open(path)?;
    let mut list: Vec<&TensorInfo> = gguf.tensors().iter().collect();
    match sort {
        TensorSort::File => {}
        TensorSort::Size => list.sort_by_key(|t| std::cmp::Reverse(t.n_bytes())),
        TensorSort::Name => list.sort_by(|a, b| a.name().cmp(b.name())),
        TensorSort::Offset => list.sort_by_key(|t| t.absolute_offset()),
    }

    if json {
        let rows: Vec<_> = list
            .iter()
            .map(|t| TensorJson {
                name: t.name(),
                ggml_type: t.ggml_type().name(),
                dims: t.dims(),
                elements: t.n_elements(),
                bytes: t.n_bytes(),
                offset: t.offset(),
                absolute_offset: t.absolute_offset(),
            })
            .collect();
        write_json(out, &rows)?;
        return Ok(Outcome::Ok);
    }

    if list.is_empty() {
        writeln!(out, "no tensors")?;
        return Ok(Outcome::Ok);
    }
    let mut table = Table::with_header(&["NAME", "TYPE", "SHAPE", "ELEMENTS", "BYTES", "OFFSET"])
        .right(&[3, 4, 5]);
    for t in &list {
        table.row([
            t.name().to_owned(),
            t.ggml_type().name().to_owned(),
            shape(t.dims()),
            thousands(t.n_elements()),
            bytes(t.n_bytes()),
            thousands(t.absolute_offset()),
        ]);
    }
    table.render(out)?;
    writeln!(out)?;
    writeln!(
        out,
        "{} tensors, {} parameters, {} of tensor data",
        thousands(list.len() as u64),
        count(gguf.parameter_count()),
        bytes(gguf.tensor_data_size())
    )?;
    Ok(Outcome::Ok)
}
