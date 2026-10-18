//! Number formatting and aligned plain-text tables.

use std::io::{self, Write};

use gguf_inspect::model::PerLayer;
use unicode_width::UnicodeWidthStr;

use crate::escape;

/// `1234567` -> `"1,234,567"`.
pub(crate) fn thousands(n: u64) -> String {
    let digits = n.to_string();
    let mut out = String::with_capacity(digits.len() + digits.len() / 3);
    for (i, c) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(c);
    }
    out
}

/// Byte counts in binary units: `"512 B"`, `"4.58 GiB"`.
#[allow(clippy::cast_precision_loss)] // display only
pub(crate) fn bytes(n: u64) -> String {
    const UNITS: [&str; 6] = ["KiB", "MiB", "GiB", "TiB", "PiB", "EiB"];
    if n < 1024 {
        return format!("{n} B");
    }
    let mut value = n as f64 / 1024.0;
    let mut unit = 0;
    while value >= 1024.0 && unit + 1 < UNITS.len() {
        value /= 1024.0;
        unit += 1;
    }
    format!("{value:.2} {}", UNITS[unit])
}

/// Large counts with a decimal suffix: `"8.03 B"`, `"525.34 M"`, `"512"`.
#[allow(clippy::cast_precision_loss)] // display only
pub(crate) fn count(n: u64) -> String {
    const UNITS: [(f64, &str); 4] = [(1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")];
    let value = n as f64;
    UNITS.iter().find(|(scale, _)| value >= *scale).map_or_else(
        || n.to_string(),
        |(scale, suffix)| format!("{:.2} {suffix}", value / scale),
    )
}

/// `part / total` as a percentage with one decimal.
#[allow(clippy::cast_precision_loss)] // display only
pub(crate) fn percent(part: u64, total: u64) -> String {
    if total == 0 {
        return "-".to_owned();
    }
    format!("{:.1}%", part as f64 * 100.0 / total as f64)
}

/// A shape in ggml order, `[4096, 32000]`.
pub(crate) fn shape(dims: &[u64]) -> String {
    let inner: Vec<String> = dims.iter().map(u64::to_string).collect();
    format!("[{}]", inner.join(", "))
}

/// The attention variant implied by the KV head count, for example
/// `"grouped-query attention, 4 heads per KV head"`.
pub(crate) fn attention(kv: &PerLayer, heads: Option<&PerLayer>) -> Option<String> {
    let heads = heads.and_then(PerLayer::uniform);
    let Some(kv) = kv.uniform() else {
        // Hybrid models list 0 KV heads for their recurrent layers.
        let PerLayer::List(list) = kv else {
            return None;
        };
        let with_kv = list.iter().filter(|&&h| h > 0).count();
        return (with_kv < list.len())
            .then(|| format!("KV heads in {with_kv} of {} layers", list.len()));
    };
    match (kv, heads?) {
        (0, 0) => Some("no attention heads".to_owned()),
        (k, h) if k == h => Some("multi-head attention".to_owned()),
        (1, h) if h > 1 => Some("multi-query attention".to_owned()),
        (k, h) if k > 0 && k < h && h % k == 0 => Some(format!(
            "grouped-query attention, {} heads per KV head",
            h / k
        )),
        _ => None,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Align {
    Left,
    Right,
}

/// The layout of a table: columns separated by two spaces and padded by
/// display width. Every cell is passed through [`escape::text`], so a cell
/// taken from the file cannot break the layout or send control sequences to
/// the terminal.
///
/// Long listings use it directly in two passes over their data, measuring
/// every row and then writing every row, so no row is kept in memory;
/// [`Table`] buffers rows for the small tables.
#[derive(Debug, Clone, Default)]
pub(crate) struct Columns {
    header: Option<Vec<String>>,
    align: Vec<Align>,
    widths: Vec<usize>,
    indent: usize,
}

impl Columns {
    pub(crate) fn with_header(header: &[&str]) -> Self {
        let mut columns = Self {
            header: Some(header.iter().map(|h| (*h).to_owned()).collect()),
            ..Self::default()
        };
        columns.measure(header);
        columns
    }

    /// Right-aligns the given column indices (numbers).
    pub(crate) fn right(mut self, columns: &[usize]) -> Self {
        for &c in columns {
            if self.align.len() <= c {
                self.align.resize(c + 1, Align::Left);
            }
            self.align[c] = Align::Right;
        }
        self
    }

    /// Widens the columns to fit `cells`. A left-aligned last column is
    /// never padded, so it need not be measured.
    pub(crate) fn measure<S: AsRef<str>>(&mut self, cells: &[S]) {
        if self.widths.len() < cells.len() {
            self.widths.resize(cells.len(), 0);
        }
        for (width, cell) in self.widths.iter_mut().zip(cells) {
            *width = (*width).max(escape::text(cell.as_ref()).width());
        }
    }

    pub(crate) fn write_header(&self, out: &mut dyn Write) -> io::Result<()> {
        match &self.header {
            Some(header) => self.write_row(out, header),
            None => Ok(()),
        }
    }

    pub(crate) fn write_row<S: AsRef<str>>(
        &self,
        out: &mut dyn Write,
        cells: &[S],
    ) -> io::Result<()> {
        let mut line = " ".repeat(self.indent);
        for (i, cell) in cells.iter().enumerate() {
            let cell = escape::text(cell.as_ref());
            let pad = self
                .widths
                .get(i)
                .copied()
                .unwrap_or(0)
                .saturating_sub(cell.width());
            if i > 0 {
                line.push_str("  ");
            }
            match self.align.get(i).copied().unwrap_or(Align::Left) {
                Align::Right => {
                    line.push_str(&" ".repeat(pad));
                    line.push_str(&cell);
                }
                Align::Left => {
                    line.push_str(&cell);
                    if i + 1 < cells.len() {
                        line.push_str(&" ".repeat(pad));
                    }
                }
            }
        }
        writeln!(out, "{}", line.trim_end())
    }
}

/// A small table whose rows are kept until it is rendered.
#[derive(Debug, Default)]
pub(crate) struct Table {
    columns: Columns,
    rows: Vec<Vec<String>>,
}

impl Table {
    pub(crate) fn with_header(header: &[&str]) -> Self {
        Self {
            columns: Columns::with_header(header),
            rows: Vec::new(),
        }
    }

    pub(crate) fn plain() -> Self {
        Self::default()
    }

    /// Right-aligns the given column indices (numbers).
    pub(crate) fn right(mut self, columns: &[usize]) -> Self {
        self.columns = self.columns.right(columns);
        self
    }

    pub(crate) fn indent(mut self, spaces: usize) -> Self {
        self.columns.indent = spaces;
        self
    }

    pub(crate) fn row<I, S>(&mut self, cells: I)
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let row: Vec<String> = cells.into_iter().map(Into::into).collect();
        self.columns.measure(&row);
        self.rows.push(row);
    }

    pub(crate) fn render(&self, out: &mut dyn Write) -> io::Result<()> {
        self.columns.write_header(out)?;
        for row in &self.rows {
            self.columns.write_row(out, row)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn number_formats() {
        assert_eq!(thousands(0), "0");
        assert_eq!(thousands(999), "999");
        assert_eq!(thousands(1000), "1,000");
        assert_eq!(thousands(8_030_261_248), "8,030,261,248");
        assert_eq!(bytes(1023), "1023 B");
        assert_eq!(bytes(1536), "1.50 KiB");
        assert_eq!(bytes(1 << 30), "1.00 GiB");
        assert_eq!(count(999), "999");
        assert_eq!(count(8_030_261_248), "8.03 B");
        assert_eq!(percent(1, 3), "33.3%");
        assert_eq!(percent(1, 0), "-");
        assert_eq!(shape(&[4096, 32]), "[4096, 32]");
        assert_eq!(shape(&[]), "[]");
    }

    #[test]
    fn attention_variants() {
        let u = PerLayer::Uniform;
        let describe = |kv: u64, heads: u64| attention(&u(kv), Some(&u(heads)));
        assert_eq!(describe(32, 32).as_deref(), Some("multi-head attention"));
        assert_eq!(describe(1, 128).as_deref(), Some("multi-query attention"));
        assert_eq!(
            describe(8, 32).as_deref(),
            Some("grouped-query attention, 4 heads per KV head")
        );
        assert_eq!(describe(0, 0).as_deref(), Some("no attention heads"));
        assert_eq!(describe(3, 32), None);
        assert_eq!(describe(1, 1).as_deref(), Some("multi-head attention"));
        let hybrid = PerLayer::List(vec![0, 8, 0, 8]);
        assert_eq!(
            attention(&hybrid, Some(&u(32))).as_deref(),
            Some("KV heads in 2 of 4 layers")
        );
        assert_eq!(attention(&PerLayer::List(vec![2, 4]), None), None);
        assert_eq!(attention(&u(8), None), None);
    }

    #[test]
    fn table_cells_cannot_add_rows_or_controls() {
        let mut t = Table::plain();
        t.row(["key\nfake.key", "\x1b[2Jv"]);
        let mut out = Vec::new();
        t.render(&mut out).unwrap();
        assert_eq!(
            String::from_utf8(out).unwrap(),
            "key\\nfake.key  \\u{1b}[2Jv\n"
        );
    }

    #[test]
    fn columns_render_in_two_passes() {
        let rows = [["a", "1", "x"], ["bbb", "22", "a long last column"]];
        let mut columns = Columns::with_header(&["K", "N", "V"]).right(&[1]);
        for row in &rows {
            columns.measure(&[row[0], row[1], ""]);
        }
        let mut out = Vec::new();
        columns.write_header(&mut out).unwrap();
        for row in &rows {
            columns.write_row(&mut out, row).unwrap();
        }
        assert_eq!(
            String::from_utf8(out).unwrap(),
            "K     N  V\na     1  x\nbbb  22  a long last column\n"
        );
    }

    #[test]
    fn table_alignment_uses_display_width() {
        let mut t = Table::with_header(&["NAME", "N"]).right(&[1]);
        t.row(["wörld", "5"]);
        t.row(["a", "123"]);
        let mut out = Vec::new();
        t.render(&mut out).unwrap();
        assert_eq!(
            String::from_utf8(out).unwrap(),
            "NAME     N\nwörld    5\na      123\n"
        );
    }
}
