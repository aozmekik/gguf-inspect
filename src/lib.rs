//! Read the header of a [GGUF] file (the model format of llama.cpp and ggml)
//! without loading tensor data.
//!
//! The parser reads only the header region through a buffered reader: magic,
//! version, metadata key-value pairs and tensor infos. It treats input as
//! untrusted. Every declared length and count is checked against the bytes
//! left in the file and against configurable [`Limits`] before anything is
//! allocated, malformed input yields an [`Error`] with the byte offset of the
//! offending field, and the crate contains no `unsafe` code.
//!
//! Supported: GGUF versions 2 and 3, little-endian. Big-endian files are
//! detected and rejected with [`ErrorKind::BigEndian`].
//!
//! # Example
//!
//! ```
//! use gguf_inspect::{GgufFile, MetadataValue};
//!
//! // A minimal file: version 3, no tensors, one metadata entry.
//! let mut bytes = b"GGUF".to_vec();
//! bytes.extend_from_slice(&3u32.to_le_bytes()); // version
//! bytes.extend_from_slice(&0u64.to_le_bytes()); // tensor count
//! bytes.extend_from_slice(&1u64.to_le_bytes()); // metadata entry count
//! let key = b"general.architecture";
//! bytes.extend_from_slice(&(key.len() as u64).to_le_bytes());
//! bytes.extend_from_slice(key);
//! bytes.extend_from_slice(&8u32.to_le_bytes()); // value type: string
//! bytes.extend_from_slice(&5u64.to_le_bytes());
//! bytes.extend_from_slice(b"llama");
//!
//! let gguf = GgufFile::from_bytes(&bytes)?;
//! assert_eq!(gguf.architecture(), Some("llama"));
//! assert_eq!(
//!     gguf.metadata().get("general.architecture"),
//!     Some(&MetadataValue::String("llama".into()))
//! );
//!
//! // Truncating the file anywhere inside the header is an error, never a panic.
//! let err = GgufFile::from_bytes(&bytes[..bytes.len() - 1]).unwrap_err();
//! assert_eq!(err.offset(), Some(56));
//! # Ok::<(), gguf_inspect::Error>(())
//! ```
//!
//! Real files are opened with [`GgufFile::open`]. Higher-level views live in
//! [`model`] (well-known hyperparameters), [`kv_cache`] (KV-cache sizing) and
//! [`diff`] (comparing two files).
//!
//! [GGUF]: https://github.com/ggml-org/ggml/blob/a05c9922613c6a2230dbb1d0de679519ecafb4f0/docs/gguf.md

// A library has no business printing.
#![deny(clippy::print_stdout, clippy::print_stderr)]

pub mod diff;
mod error;
mod file;
mod ggml_type;
pub mod kv_cache;
mod limits;
mod metadata;
pub mod model;
mod reader;
mod tensor;

pub use error::{Error, ErrorKind, Result};
pub use file::{
    ALIGNMENT_KEY, DEFAULT_ALIGNMENT, GgufFile, GgufVersion, Issue, MAGIC, MAX_KEY_LEN,
    MAX_TENSOR_NAME_LEN, TypeStats,
};
pub use ggml_type::{GgmlType, ParseGgmlTypeError};
pub use limits::Limits;
pub use metadata::{Metadata, MetadataArray, MetadataValue, ValueType};
pub use model::ModelInfo;
pub use tensor::TensorInfo;

// Compile the Rust examples in README.md as doctests.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
struct ReadmeDoctests;
