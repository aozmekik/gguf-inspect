//! Caps on declared sizes, applied while parsing untrusted files.

/// Upper bounds on sizes and counts declared inside a file.
///
/// Every declared length is first checked against the bytes actually left in
/// the file, so a small file can never trigger a large allocation. These caps
/// are a second line of defence for large but implausible files, and keep
/// worst-case memory use predictable. The defaults are far above anything seen
/// in published models (vocabularies have around 32 000-260 000 entries;
/// models have a few hundred to a few thousand tensors).
///
/// ```
/// use gguf_inspect::Limits;
///
/// let mut limits = Limits::default();
/// limits.max_array_len = 1 << 20; // reject vocabularies over a million entries
/// assert_eq!(limits.max_dims, 4);
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct Limits {
    /// Longest accepted metadata string value, in bytes. Default: 256 MiB
    /// (`tokenizer.huggingface.json` can embed a whole `tokenizer.json`).
    pub max_string_len: u64,
    /// Most elements accepted in one metadata array. Default: 2^24.
    pub max_array_len: u64,
    /// Deepest accepted array nesting; a top-level array has depth 1. Default: 8.
    pub max_array_depth: u32,
    /// Most metadata entries accepted. Default: 2^20.
    pub max_metadata_entries: u64,
    /// Most tensors accepted. Default: 2^20.
    pub max_tensors: u64,
    /// Most dimensions accepted per tensor. Default: 4, ggml's `GGML_MAX_DIMS`.
    pub max_dims: u32,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            max_string_len: 256 << 20,
            max_array_len: 1 << 24,
            max_array_depth: 8,
            max_metadata_entries: 1 << 20,
            max_tensors: 1 << 20,
            max_dims: 4,
        }
    }
}
