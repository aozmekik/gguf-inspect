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
/// # Memory use
///
/// Parsed values are owned, so memory grows with what the header holds, not
/// with what it declares. Besides the bytes of its strings and arrays, each
/// metadata entry and tensor info has a fixed cost that is larger than its
/// encoding: headers made of many tiny records use the most memory per byte.
/// Measured with `/usr/bin/time -l` on the release build of the CLI (Apple
/// M1 Pro): a 22 MB header of 2^20 one-byte entries peaks at 95 MB (about
/// 90 bytes per entry) and a 33.5 MB header of 2^20 scalar tensor infos at
/// 181 MB (about 170 bytes per tensor). The defaults cap both counts at 2^20.
/// A realistic header, the 3.24 MB one of the 8B-shaped synthetic model,
/// peaks at 5.4 MB of heap while parsing (`tests/memory.rs`).
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
    /// Deepest accepted array nesting; a top-level array has depth 1.
    /// Default: 8. Values above [`Limits::MAX_ARRAY_DEPTH`] (64) are treated
    /// as 64: nested arrays are parsed, compared and dropped recursively, and
    /// the ceiling keeps a hostile file from exhausting the stack.
    pub max_array_depth: u32,
    /// Most metadata entries accepted. Default: 2^20.
    pub max_metadata_entries: u64,
    /// Most tensors accepted. Default: 2^20.
    pub max_tensors: u64,
    /// Most dimensions accepted per tensor. Default: 4, ggml's `GGML_MAX_DIMS`.
    pub max_dims: u32,
}

impl Limits {
    /// The highest nesting depth ever accepted, whatever
    /// [`max_array_depth`](Self::max_array_depth) says. ggml's own reader
    /// rejects nested arrays altogether.
    pub const MAX_ARRAY_DEPTH: u32 = 64;

    /// The nesting depth actually enforced.
    pub(crate) fn array_depth(&self) -> u32 {
        self.max_array_depth.min(Self::MAX_ARRAY_DEPTH)
    }
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
