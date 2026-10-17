//! Rendering metadata values for humans and for JSON.

use std::fmt::Write as _;

use gguf_inspect::{MetadataArray, MetadataValue};
use serde_json::{Number, Value as Json};

/// Array elements shown before truncating, unless `--full` is given.
pub(crate) const PREVIEW_ITEMS: usize = 8;
/// String characters shown before truncating in human output.
pub(crate) const PREVIEW_CHARS: usize = 64;

/// How much of a value to show.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Preview {
    pub(crate) items: Option<usize>,
    pub(crate) chars: Option<usize>,
}

impl Preview {
    pub(crate) const FULL: Self = Self {
        items: None,
        chars: None,
    };
    pub(crate) const SHORT: Self = Self {
        items: Some(PREVIEW_ITEMS),
        chars: Some(PREVIEW_CHARS),
    };
}

/// `"u32"`, `"string"`, `"array[string]"`, `"array[array]"`.
pub(crate) fn type_name(value: &MetadataValue) -> String {
    match value {
        MetadataValue::Array(a) => format!("array[{}]", a.element_type()),
        other => other.value_type().to_string(),
    }
}

/// Human rendering. Returns the text and whether anything was cut short.
pub(crate) fn render(value: &MetadataValue, preview: Preview) -> (String, bool) {
    let mut out = String::new();
    let truncated = render_into(&mut out, value, preview);
    (out, truncated)
}

fn render_into(out: &mut String, value: &MetadataValue, preview: Preview) -> bool {
    match value {
        MetadataValue::String(s) => render_string(out, s, preview.chars),
        MetadataValue::Array(a) => render_array(out, a, preview),
        // Debug formatting gives the shortest round-trip form and switches
        // to exponent notation for very large or small values.
        MetadataValue::F32(v) => push(out, format_args!("{v:?}")),
        MetadataValue::F64(v) => push(out, format_args!("{v:?}")),
        MetadataValue::Bool(b) => {
            out.push_str(if *b { "true" } else { "false" });
            false
        }
        other => {
            let n = other.as_i64().map_or_else(
                || other.as_u64().map_or_else(String::new, |v| v.to_string()),
                |v| v.to_string(),
            );
            out.push_str(&n);
            false
        }
    }
}

/// Appends formatted text; returns `false` (nothing truncated) for use in match arms.
fn push(out: &mut String, args: std::fmt::Arguments<'_>) -> bool {
    out.write_fmt(args)
        .expect("writing to a String cannot fail");
    false
}

fn render_string(out: &mut String, s: &str, max_chars: Option<usize>) -> bool {
    let shown: String = match max_chars {
        Some(max) if s.chars().count() > max => s.chars().take(max).collect(),
        _ => s.to_owned(),
    };
    let truncated = shown.len() < s.len();
    out.push('"');
    out.push_str(&shown.escape_debug().to_string());
    out.push('"');
    if truncated {
        push(out, format_args!("... ({} bytes)", s.len()));
    }
    truncated
}

fn render_array(out: &mut String, array: &MetadataArray, preview: Preview) -> bool {
    let len = array.len();
    let shown = preview.items.map_or(len, |max| len.min(max));
    let mut truncated = shown < len;
    out.push('[');
    for (i, item) in array.iter().take(shown).enumerate() {
        if i > 0 {
            out.push_str(", ");
        }
        truncated |= render_into(out, &item, preview);
    }
    if shown < len {
        if shown > 0 {
            out.push_str(", ");
        }
        push(
            out,
            format_args!(
                "... {} more",
                crate::format::thousands((len - shown) as u64)
            ),
        );
    }
    out.push(']');
    truncated
}

/// JSON for a float, using the shortest decimal that round-trips its own
/// width (so an `f32` 0.1 becomes `0.1`, not `0.10000000149011612`).
/// Non-finite values have no JSON representation and become `null`.
fn float(shortest: &str) -> Json {
    shortest
        .parse::<f64>()
        .ok()
        .and_then(Number::from_f64)
        .map_or(Json::Null, Json::Number)
}

/// JSON for a value; arrays keep at most `max_items` elements at the top level.
pub(crate) fn to_json(value: &MetadataValue, max_items: Option<usize>) -> Json {
    match value {
        MetadataValue::String(s) => Json::String(s.clone()),
        MetadataValue::Bool(b) => Json::Bool(*b),
        MetadataValue::F32(v) => float(&v.to_string()),
        MetadataValue::F64(v) => float(&v.to_string()),
        MetadataValue::Array(a) => array_to_json(a, max_items),
        other => other
            .as_u64()
            .map(Json::from)
            .or_else(|| other.as_i64().map(Json::from))
            .unwrap_or(Json::Null),
    }
}

fn array_to_json(array: &MetadataArray, max_items: Option<usize>) -> Json {
    let shown = max_items.map_or(array.len(), |m| array.len().min(m));
    Json::Array(
        array
            .iter()
            .take(shown)
            .map(|item| to_json(&item, None))
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn short(v: &MetadataValue) -> String {
        render(v, Preview::SHORT).0
    }

    #[test]
    fn scalars() {
        assert_eq!(short(&MetadataValue::I8(-3)), "-3");
        assert_eq!(short(&MetadataValue::U64(u64::MAX)), u64::MAX.to_string());
        assert_eq!(short(&MetadataValue::F32(0.1)), "0.1");
        assert_eq!(short(&MetadataValue::F64(1e300)), "1e300");
        assert_eq!(short(&MetadataValue::Bool(false)), "false");
        assert_eq!(
            short(&MetadataValue::String("a\nb\"".into())),
            r#""a\nb\"""#
        );
    }

    #[test]
    fn long_strings_and_arrays_are_truncated() {
        let long = MetadataValue::String("x".repeat(100));
        let (text, truncated) = render(&long, Preview::SHORT);
        assert!(truncated);
        assert!(text.ends_with("... (100 bytes)"));
        assert_eq!(render(&long, Preview::FULL).0.len(), 102);

        let vocab = MetadataValue::Array(MetadataArray::U32((0..1000).collect()));
        assert_eq!(short(&vocab), "[0, 1, 2, 3, 4, 5, 6, 7, ... 992 more]");
        let empty = MetadataValue::Array(MetadataArray::Bool(vec![]));
        assert_eq!(render(&empty, Preview::SHORT), ("[]".to_owned(), false));
    }

    #[test]
    fn nested_arrays() {
        let v = MetadataValue::Array(MetadataArray::Array(vec![
            MetadataArray::I32(vec![1, -2]),
            MetadataArray::String(vec!["x".into()]),
        ]));
        assert_eq!(short(&v), r#"[[1, -2], ["x"]]"#);
        assert_eq!(type_name(&v), "array[array]");
        assert_eq!(to_json(&v, None).to_string(), r#"[[1,-2],["x"]]"#);
    }

    #[test]
    fn json_numbers() {
        assert_eq!(to_json(&MetadataValue::F32(0.1), None), Json::from(0.1));
        assert_eq!(to_json(&MetadataValue::F32(f32::NAN), None), Json::Null);
        assert_eq!(to_json(&MetadataValue::I64(-5), None), Json::from(-5));
        assert_eq!(
            to_json(&MetadataValue::U64(u64::MAX), None),
            Json::from(u64::MAX)
        );
        let arr = MetadataValue::Array(MetadataArray::U8(vec![1, 2, 3]));
        assert_eq!(to_json(&arr, Some(2)).to_string(), "[1,2]");
    }
}
