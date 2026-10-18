//! Number formatting and aligned plain-text tables.

use std::borrow::Cow;
use std::io::{self, Write};

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Align {
    Left,
    Right,
}

/// Columns separated by two spaces, padded by display width. Every cell is
/// passed through [`escape::text`], so a cell taken from the file cannot
/// break the layout or send control sequences to the terminal.
#[derive(Debug, Default)]
pub(crate) struct Table {
    header: Option<Vec<String>>,
    align: Vec<Align>,
    rows: Vec<Vec<String>>,
    indent: usize,
}

impl Table {
    pub(crate) fn with_header(header: &[&str]) -> Self {
        Self {
            header: Some(header.iter().map(|h| (*h).to_owned()).collect()),
            ..Self::default()
        }
    }

    pub(crate) fn plain() -> Self {
        Self::default()
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

    pub(crate) fn indent(mut self, spaces: usize) -> Self {
        self.indent = spaces;
        self
    }

    pub(crate) fn row<I, S>(&mut self, cells: I)
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.rows.push(
            cells
                .into_iter()
                .map(|cell| {
                    let cell: String = cell.into();
                    match escape::text(&cell) {
                        Cow::Borrowed(_) => cell,
                        Cow::Owned(escaped) => escaped,
                    }
                })
                .collect(),
        );
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    pub(crate) fn render(&self, out: &mut dyn Write) -> io::Result<()> {
        let all_rows = || self.header.iter().chain(&self.rows);
        let columns = all_rows().map(Vec::len).max().unwrap_or(0);
        let mut widths = vec![0; columns];
        for row in all_rows() {
            for (i, cell) in row.iter().enumerate() {
                widths[i] = widths[i].max(cell.width());
            }
        }
        for row in all_rows() {
            let mut line = " ".repeat(self.indent);
            for (i, cell) in row.iter().enumerate() {
                let pad = widths[i] - cell.width();
                let last = i + 1 == row.len();
                if i > 0 {
                    line.push_str("  ");
                }
                match self.align.get(i).copied().unwrap_or(Align::Left) {
                    Align::Right => {
                        line.push_str(&" ".repeat(pad));
                        line.push_str(cell);
                    }
                    Align::Left => {
                        line.push_str(cell);
                        if !last {
                            line.push_str(&" ".repeat(pad));
                        }
                    }
                }
            }
            writeln!(out, "{}", line.trim_end())?;
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
