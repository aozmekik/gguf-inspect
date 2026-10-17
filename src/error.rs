//! Error type returned by the parser.

use std::fmt;
use std::io;

use crate::GgmlType;
use crate::metadata::ValueType;

/// Convenience alias for results produced by this crate.
pub type Result<T, E = Error> = std::result::Result<T, E>;

/// A parse failure, with the byte offset where it was detected and, when known,
/// which metadata entry or tensor was being read.
///
/// ```
/// use gguf_inspect::{ErrorKind, GgufFile};
///
/// let err = GgufFile::from_bytes(b"GGML\x03\0\0\0").unwrap_err();
/// assert!(matches!(err.kind(), ErrorKind::BadMagic(_)));
/// assert_eq!(err.offset(), Some(0));
/// ```
#[derive(Debug)]
pub struct Error {
    kind: ErrorKind,
    offset: Option<u64>,
    context: Option<String>,
}

/// What went wrong while parsing a GGUF file.
#[derive(Debug)]
#[non_exhaustive]
pub enum ErrorKind {
    /// Opening or reading the file failed.
    Io(io::Error),
    /// The file ended in the middle of a fixed-size field.
    UnexpectedEof {
        /// Bytes the field needs.
        needed: u64,
        /// Bytes left in the file.
        available: u64,
    },
    /// The first four bytes are not `GGUF`.
    BadMagic([u8; 4]),
    /// The version field is byte-swapped, which means the file was written big-endian.
    BigEndian {
        /// The version number after swapping the bytes back.
        version: u32,
    },
    /// The format version is not 2 or 3.
    UnsupportedVersion(u32),
    /// A metadata value type tag is not one of the 13 defined types.
    InvalidValueType(u32),
    /// A bool byte is neither 0 nor 1.
    InvalidBool(u8),
    /// A string is not valid UTF-8.
    InvalidUtf8,
    /// A declared length or count cannot fit in the bytes left in the file.
    LengthExceedsFile {
        /// What the length describes, such as `"string length"`.
        what: &'static str,
        /// The value read from the file.
        declared: u64,
        /// The fewest bytes such a value would occupy.
        min_bytes: u64,
        /// Bytes left in the file.
        available: u64,
    },
    /// A declared length or count exceeds a configured [`Limits`](crate::Limits) cap
    /// or a limit set by the GGUF specification.
    LimitExceeded {
        /// What the value describes, such as `"tensor count"`.
        what: &'static str,
        /// The value read from the file.
        value: u64,
        /// The largest accepted value.
        limit: u64,
    },
    /// Arrays are nested deeper than [`Limits::max_array_depth`](crate::Limits::max_array_depth).
    NestingTooDeep {
        /// The configured depth limit.
        limit: u32,
    },
    /// A metadata key is the empty string.
    EmptyKey,
    /// The same metadata key appears twice.
    DuplicateKey(String),
    /// The same tensor name appears twice.
    DuplicateTensorName(String),
    /// A tensor has more dimensions than allowed.
    TooManyDimensions {
        /// Dimensions declared by the tensor.
        n_dims: u32,
        /// The configured maximum.
        max: u32,
    },
    /// A tensor uses a ggml type ID that is unknown or has been removed from ggml.
    UnknownGgmlType(u32),
    /// A tensor's element count does not fit in `i64`, the type ggml uses for shapes.
    TensorShapeOverflow(Vec<u64>),
    /// A tensor's first dimension is not a whole number of quantization blocks.
    PartialBlock {
        /// The tensor's element type.
        ggml_type: GgmlType,
        /// The tensor's first dimension.
        ne0: u64,
    },
    /// `general.alignment` is present but not a `u32`.
    AlignmentType(ValueType),
    /// `general.alignment` is zero or not a power of two.
    InvalidAlignment(u32),
    /// A tensor's absolute data offset does not fit in `u64`.
    OffsetOverflow,
}

impl Error {
    pub(crate) fn new(kind: ErrorKind, offset: u64) -> Self {
        Self {
            kind,
            offset: Some(offset),
            context: None,
        }
    }

    /// Attaches a description of what was being parsed, unless a more specific
    /// one is already present.
    pub(crate) fn with_context(mut self, context: impl FnOnce() -> String) -> Self {
        if self.context.is_none() {
            self.context = Some(context());
        }
        self
    }

    /// What went wrong.
    pub fn kind(&self) -> &ErrorKind {
        &self.kind
    }

    /// Byte offset from the start of the file where the problem was detected,
    /// if the error relates to a position in the file.
    pub fn offset(&self) -> Option<u64> {
        self.offset
    }

    /// The metadata entry or tensor being read, for example
    /// ``metadata entry 3 `general.name` ``.
    pub fn context(&self) -> Option<&str> {
        self.context.as_deref()
    }
}

impl From<io::Error> for Error {
    fn from(err: io::Error) -> Self {
        Self {
            kind: ErrorKind::Io(err),
            offset: None,
            context: None,
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.kind, f)?;
        if let Some(offset) = self.offset {
            write!(f, " at byte offset {offset}")?;
        }
        if let Some(context) = &self.context {
            write!(f, " (in {context})")?;
        }
        Ok(())
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match &self.kind {
            ErrorKind::Io(err) => Some(err),
            _ => None,
        }
    }
}

impl fmt::Display for ErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(err) => write!(f, "I/O error: {err}"),
            Self::UnexpectedEof { needed, available } => write!(
                f,
                "unexpected end of file: needed {needed} bytes but only {available} remain"
            ),
            Self::BadMagic(found) => write!(
                f,
                "not a GGUF file: expected magic \"GGUF\", found \"{}\"",
                found.escape_ascii()
            ),
            Self::BigEndian { version } => write!(
                f,
                "big-endian GGUF files are not supported (the version field reads as \
                 version {version} only when byte-swapped)"
            ),
            Self::UnsupportedVersion(1) => f.write_str(
                "GGUF version 1 is not supported; it used 32-bit lengths (supported: 2, 3)",
            ),
            Self::UnsupportedVersion(v) => {
                write!(f, "unsupported GGUF version {v} (supported: 2, 3)")
            }
            Self::InvalidValueType(t) => write!(f, "invalid metadata value type {t}"),
            Self::InvalidBool(b) => write!(f, "invalid bool value {b} (must be 0 or 1)"),
            Self::InvalidUtf8 => f.write_str("string is not valid UTF-8"),
            Self::LengthExceedsFile {
                what,
                declared,
                min_bytes,
                available,
            } => write!(
                f,
                "{what} {declared} needs at least {min_bytes} bytes but only {available} remain"
            ),
            Self::LimitExceeded { what, value, limit } => {
                write!(f, "{what} {value} exceeds the limit of {limit}")
            }
            Self::NestingTooDeep { limit } => {
                write!(f, "arrays are nested more than {limit} levels deep")
            }
            Self::EmptyKey => f.write_str("metadata key is empty"),
            Self::DuplicateKey(key) => write!(f, "duplicate metadata key `{key}`"),
            Self::DuplicateTensorName(name) => write!(f, "duplicate tensor name `{name}`"),
            Self::TooManyDimensions { n_dims, max } => {
                write!(f, "tensor has {n_dims} dimensions (at most {max} allowed)")
            }
            Self::UnknownGgmlType(id) if GgmlType::is_removed_id(*id) => {
                write!(f, "ggml type {id} has been removed from ggml")
            }
            Self::UnknownGgmlType(id) => write!(f, "unknown ggml type {id}"),
            Self::TensorShapeOverflow(dims) => {
                write!(f, "tensor shape {dims:?} has more than i64::MAX elements")
            }
            Self::PartialBlock { ggml_type, ne0 } => write!(
                f,
                "first dimension {ne0} is not a multiple of the {ggml_type} block size {}",
                ggml_type.block_size()
            ),
            Self::AlignmentType(t) => write!(f, "general.alignment must be a u32, found {t}"),
            Self::InvalidAlignment(a) => {
                write!(f, "general.alignment {a} is not a power of two")
            }
            Self::OffsetOverflow => f.write_str("tensor data offset overflows u64"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_includes_offset_and_context() {
        let err = Error::new(ErrorKind::InvalidBool(7), 42)
            .with_context(|| "metadata entry 0 `a.b`".to_owned())
            .with_context(|| "ignored, already set".to_owned());
        assert_eq!(
            err.to_string(),
            "invalid bool value 7 (must be 0 or 1) at byte offset 42 (in metadata entry 0 `a.b`)"
        );
        assert_eq!(err.context(), Some("metadata entry 0 `a.b`"));
    }

    #[test]
    fn io_errors_expose_their_source() {
        let err = Error::from(io::Error::other("boom"));
        assert_eq!(err.offset(), None);
        assert!(std::error::Error::source(&err).is_some());
        assert_eq!(err.to_string(), "I/O error: boom");
    }

    #[test]
    fn every_kind_has_a_message() {
        let cases: Vec<(ErrorKind, &str)> = vec![
            (
                ErrorKind::UnexpectedEof {
                    needed: 8,
                    available: 3,
                },
                "unexpected end of file: needed 8 bytes but only 3 remain",
            ),
            (
                ErrorKind::BadMagic(*b"\0\x01ab"),
                r#"not a GGUF file: expected magic "GGUF", found "\x00\x01ab""#,
            ),
            (
                ErrorKind::BigEndian { version: 3 },
                "big-endian GGUF files are not supported (the version field reads as version 3 \
                 only when byte-swapped)",
            ),
            (
                ErrorKind::UnsupportedVersion(1),
                "GGUF version 1 is not supported; it used 32-bit lengths (supported: 2, 3)",
            ),
            (
                ErrorKind::UnsupportedVersion(9),
                "unsupported GGUF version 9 (supported: 2, 3)",
            ),
            (
                ErrorKind::InvalidValueType(13),
                "invalid metadata value type 13",
            ),
            (ErrorKind::InvalidUtf8, "string is not valid UTF-8"),
            (
                ErrorKind::LengthExceedsFile {
                    what: "array length",
                    declared: 10,
                    min_bytes: 80,
                    available: 5,
                },
                "array length 10 needs at least 80 bytes but only 5 remain",
            ),
            (
                ErrorKind::LimitExceeded {
                    what: "tensor count",
                    value: 9,
                    limit: 8,
                },
                "tensor count 9 exceeds the limit of 8",
            ),
            (
                ErrorKind::NestingTooDeep { limit: 8 },
                "arrays are nested more than 8 levels deep",
            ),
            (ErrorKind::EmptyKey, "metadata key is empty"),
            (
                ErrorKind::DuplicateKey("a".into()),
                "duplicate metadata key `a`",
            ),
            (
                ErrorKind::DuplicateTensorName("t".into()),
                "duplicate tensor name `t`",
            ),
            (
                ErrorKind::TooManyDimensions { n_dims: 5, max: 4 },
                "tensor has 5 dimensions (at most 4 allowed)",
            ),
            (
                ErrorKind::TensorShapeOverflow(vec![1, 2]),
                "tensor shape [1, 2] has more than i64::MAX elements",
            ),
            (
                ErrorKind::PartialBlock {
                    ggml_type: GgmlType::Q4_K,
                    ne0: 100,
                },
                "first dimension 100 is not a multiple of the q4_K block size 256",
            ),
            (
                ErrorKind::AlignmentType(ValueType::U64),
                "general.alignment must be a u32, found u64",
            ),
            (
                ErrorKind::InvalidAlignment(48),
                "general.alignment 48 is not a power of two",
            ),
            (
                ErrorKind::OffsetOverflow,
                "tensor data offset overflows u64",
            ),
        ];
        for (kind, expected) in cases {
            assert_eq!(kind.to_string(), expected);
        }
    }

    #[test]
    fn removed_types_are_named_as_such() {
        assert_eq!(
            ErrorKind::UnknownGgmlType(31).to_string(),
            "ggml type 31 has been removed from ggml"
        );
        assert_eq!(
            ErrorKind::UnknownGgmlType(99).to_string(),
            "unknown ggml type 99"
        );
    }
}
