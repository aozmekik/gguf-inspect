//! Escaping of file-derived strings before they reach a terminal.
//!
//! Keys, tensor names and string values come from an untrusted file. Printed
//! raw, they could carry terminal control sequences (colours, cursor moves,
//! OSC sequences that retitle the window or write the clipboard), carriage
//! returns that overwrite earlier output, newlines that forge extra table
//! rows, or bidirectional overrides that reorder what is shown. Text output
//! passes every such string through [`text`] or [`push_quoted`]; JSON output
//! goes through [`JsonFormatter`].

use std::borrow::Cow;
use std::fmt::Write as _;
use std::io;

use serde_json::ser::{Formatter, PrettyFormatter};

/// Whether `c` must not reach the terminal as is: control characters (C0,
/// DEL and C1) and invisible format characters that hide or reorder text.
pub(crate) fn is_unsafe(c: char) -> bool {
    c.is_control()
        || matches!(
            c,
            '\u{00AD}' // soft hyphen
                | '\u{061C}' // Arabic letter mark
                | '\u{180E}' // Mongolian vowel separator
                | '\u{200B}'..='\u{200F}' // zero-width space and joiners, LRM, RLM
                | '\u{2028}'..='\u{202E}' // line and paragraph separators, bidi embeddings and overrides
                | '\u{2060}'..='\u{206F}' // word joiner, invisible operators, bidi isolates
                | '\u{FEFF}' // zero-width no-break space
                | '\u{FFF9}'..='\u{FFFB}' // interlinear annotation
                | '\u{E0000}'..='\u{E007F}' // tag characters
        )
}

fn push_escape(out: &mut String, c: char) {
    match c {
        '\n' => out.push_str("\\n"),
        '\r' => out.push_str("\\r"),
        '\t' => out.push_str("\\t"),
        _ => {
            let _ = write!(out, "\\u{{{:x}}}", u32::from(c));
        }
    }
}

/// `s` with unsafe characters written as escapes (`\n`, `\u{1b}`), for
/// unquoted text such as keys, tensor names and messages. Backslashes are kept
/// as they are, so a literal `\n` in the input and an escaped newline look the
/// same; [`push_quoted`] is unambiguous.
pub(crate) fn text(s: &str) -> Cow<'_, str> {
    if !s.chars().any(is_unsafe) {
        return Cow::Borrowed(s);
    }
    let mut out = String::with_capacity(s.len() + 8);
    for c in s.chars() {
        if is_unsafe(c) {
            push_escape(&mut out, c);
        } else {
            out.push(c);
        }
    }
    Cow::Owned(out)
}

/// Appends `s` as a double-quoted literal. Unsafe characters, `"` and `\`
/// are escaped; everything else, including `'` and non-ASCII text, is kept,
/// so templates such as `{% if message['role'] == 'user' %}` read as written.
pub(crate) fn push_quoted(out: &mut String, s: &str) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            c if is_unsafe(c) => push_escape(out, c),
            c => out.push(c),
        }
    }
    out.push('"');
}

/// `serde_json`'s pretty printer, but also escaping the characters it writes
/// raw: DEL, C1 controls and the format characters of [`is_unsafe`]
/// (`serde_json` escapes only C0 controls, `"` and `\`). The escapes are
/// standard JSON, so parsed values are unchanged.
pub(crate) struct JsonFormatter(PrettyFormatter<'static>);

impl JsonFormatter {
    pub(crate) fn new() -> Self {
        Self(PrettyFormatter::new())
    }
}

impl Formatter for JsonFormatter {
    fn write_string_fragment<W: ?Sized + io::Write>(
        &mut self,
        writer: &mut W,
        fragment: &str,
    ) -> io::Result<()> {
        let mut start = 0;
        for (i, c) in fragment.char_indices() {
            if is_unsafe(c) {
                writer.write_all(&fragment.as_bytes()[start..i])?;
                let mut units = [0_u16; 2];
                for unit in c.encode_utf16(&mut units) {
                    write!(writer, "\\u{unit:04x}")?;
                }
                start = i + c.len_utf8();
            }
        }
        writer.write_all(&fragment.as_bytes()[start..])
    }

    fn begin_array<W: ?Sized + io::Write>(&mut self, writer: &mut W) -> io::Result<()> {
        self.0.begin_array(writer)
    }

    fn end_array<W: ?Sized + io::Write>(&mut self, writer: &mut W) -> io::Result<()> {
        self.0.end_array(writer)
    }

    fn begin_array_value<W: ?Sized + io::Write>(
        &mut self,
        writer: &mut W,
        first: bool,
    ) -> io::Result<()> {
        self.0.begin_array_value(writer, first)
    }

    fn end_array_value<W: ?Sized + io::Write>(&mut self, writer: &mut W) -> io::Result<()> {
        self.0.end_array_value(writer)
    }

    fn begin_object<W: ?Sized + io::Write>(&mut self, writer: &mut W) -> io::Result<()> {
        self.0.begin_object(writer)
    }

    fn end_object<W: ?Sized + io::Write>(&mut self, writer: &mut W) -> io::Result<()> {
        self.0.end_object(writer)
    }

    fn begin_object_key<W: ?Sized + io::Write>(
        &mut self,
        writer: &mut W,
        first: bool,
    ) -> io::Result<()> {
        self.0.begin_object_key(writer, first)
    }

    fn begin_object_value<W: ?Sized + io::Write>(&mut self, writer: &mut W) -> io::Result<()> {
        self.0.begin_object_value(writer)
    }

    fn end_object_value<W: ?Sized + io::Write>(&mut self, writer: &mut W) -> io::Result<()> {
        self.0.end_object_value(writer)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Serialize;

    fn quoted(s: &str) -> String {
        let mut out = String::new();
        push_quoted(&mut out, s);
        out
    }

    #[test]
    fn plain_text_is_borrowed() {
        assert!(matches!(text("blk.0.attn_q.weight"), Cow::Borrowed(_)));
        assert!(matches!(text("héllo, 世界 👋"), Cow::Borrowed(_)));
    }

    #[test]
    fn controls_and_format_characters_are_escaped() {
        assert_eq!(text("a\x1b[31mb"), "a\\u{1b}[31mb");
        assert_eq!(text("row\nfake\trow\r"), "row\\nfake\\trow\\r");
        assert_eq!(text("\x7f\u{9b}\0"), "\\u{7f}\\u{9b}\\u{0}");
        assert_eq!(text("abc\u{202e}fed"), "abc\\u{202e}fed");
        assert_eq!(
            text("x\u{200b}\u{feff}\u{e0041}"),
            "x\\u{200b}\\u{feff}\\u{e0041}"
        );
        // Combining marks and other printable non-ASCII text are kept.
        assert_eq!(text("e\u{301}"), "e\u{301}");
    }

    #[test]
    fn quoted_strings_keep_single_quotes() {
        assert_eq!(
            quoted("{% if message['role'] == 'user' %}"),
            "\"{% if message['role'] == 'user' %}\""
        );
        assert_eq!(quoted(r#"a"b\c"#), r#""a\"b\\c""#);
        assert_eq!(quoted("\x1b]0;title\x07"), "\"\\u{1b}]0;title\\u{7}\"");
    }

    #[test]
    fn json_escapes_what_serde_json_leaves_raw() {
        #[derive(Serialize)]
        struct S<'a> {
            v: &'a str,
        }
        let value = "a\x1b\x7f\u{9b}\u{202e}\u{e0041}é'";
        let mut out = Vec::new();
        let mut ser = serde_json::Serializer::with_formatter(&mut out, JsonFormatter::new());
        S { v: value }.serialize(&mut ser).unwrap();
        let json = String::from_utf8(out).unwrap();
        assert_eq!(
            json,
            "{\n  \"v\": \"a\\u001b\\u007f\\u009b\\u202e\\udb40\\udc41é'\"\n}"
        );
        assert!(!json.chars().any(|c| is_unsafe(c) && c != '\n'));
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed["v"], value, "escapes round-trip");
    }
}
