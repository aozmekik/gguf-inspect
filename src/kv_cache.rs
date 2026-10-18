//! KV-cache memory estimates, following how llama.cpp sizes its cache.
//!
//! For every layer, the cache holds one key row and one value row per token:
//!
//! ```text
//! K row = head_count_kv[layer] × key_length   elements
//! V row = head_count_kv[layer] × value_length elements
//! bytes = sequences × context × Σ_layers (row_size(K row) + row_size(V row))
//! ```
//!
//! `row_size` is the ggml storage size of a row in the cache type (2 bytes per
//! element for `f16`, 34 bytes per 32 elements for `q8_0`, ...). Grouped-query
//! attention enters through `head_count_kv`, which defaults to `head_count`.
//! When the file does not give `key_length`/`value_length` they default to
//! `embedding_length / head_count`, as the spec prescribes.
//!
//! Two cases follow llama.cpp ([commit `547208d`]) rather than the formula
//! above:
//!
//! - **Multi-head latent attention.** When `attention.key_length_mla` and
//!   `attention.value_length_mla` are both set, `key_length` and
//!   `value_length` describe the compressed latent, which llama.cpp caches as
//!   K alone: there is no V cache (`src/llama-kv-cache.cpp`,
//!   `has_v = !is_mla`).
//! - **No attention.** When `head_count` is 0 (Mamba, RWKV and other
//!   recurrent models), llama.cpp sets the head sizes to 0
//!   (`src/llama-model.cpp`), so the KV cache is empty. The recurrent state
//!   is not estimated.
//!
//! Sliding-window layers and recurrent state are not modelled;
//! [`KvCacheEstimate::notes`] flags files that use them. A key that is present
//! but unusable (a negative head count, a float, an array of the wrong length)
//! is an error, never silently replaced by a default.
//!
//! [commit `547208d`]: https://github.com/ggml-org/llama.cpp/tree/547208dca250ceb3e4c8615598d043f2b602a527

use std::fmt;

use crate::metadata::{Metadata, MetadataValue};
use crate::model::PerLayer;
use crate::{GgmlType, GgufFile};

/// What to size the cache for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct KvCacheConfig {
    /// Tokens cached per sequence.
    pub context_length: u64,
    /// Storage type of cached keys and values.
    pub cache_type: GgmlType,
    /// Sequences cached at once, each holding `context_length` tokens.
    pub sequences: u64,
}

impl KvCacheConfig {
    /// One sequence of `context_length` tokens with an `f16` cache.
    pub fn new(context_length: u64) -> Self {
        Self {
            context_length,
            cache_type: GgmlType::F16,
            sequences: 1,
        }
    }

    /// Sets the cache storage type.
    #[must_use]
    pub fn with_cache_type(mut self, cache_type: GgmlType) -> Self {
        self.cache_type = cache_type;
        self
    }

    /// Sets the number of sequences.
    #[must_use]
    pub fn with_sequences(mut self, sequences: u64) -> Self {
        self.sequences = sequences;
        self
    }
}

/// Where the per-head key and value sizes came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum HeadDimSource {
    /// `attention.key_length` and `attention.value_length` are in the file.
    Explicit,
    /// At least one was derived as `embedding_length / head_count`.
    Derived,
    /// No layer has KV heads, so the sizes do not matter. Those the file does
    /// not give are reported as 0.
    Unused,
}

/// Model features the estimate does not model, or models in a way worth
/// pointing out.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum KvCacheNote {
    /// `attention.sliding_window` is set: runtimes may cache fewer tokens for
    /// sliding-window layers, so the estimate is an upper bound.
    SlidingWindow(u64),
    /// `attention.key_length_mla` and `attention.value_length_mla` are set
    /// (multi-head latent attention): the compressed latent is cached as K and
    /// there is no V cache, as in llama.cpp.
    LatentAttention {
        /// `attention.key_length_mla`: the decompressed key head size.
        key_length_mla: u64,
        /// `attention.value_length_mla`: the decompressed value head size.
        value_length_mla: u64,
    },
    /// `attention.kv_lora_rank` is set but the `*_mla` keys are not: an older
    /// layout of multi-head latent attention, for which llama.cpp caches full
    /// per-head keys and values, as estimated here.
    UncompressedLatentAttention(u64),
    /// `ssm.*` keys are present: recurrent state is not included.
    RecurrentState,
    /// Some layers have no KV heads and contribute nothing.
    LayersWithoutKv(u64),
}

impl fmt::Display for KvCacheNote {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SlidingWindow(w) => write!(
                f,
                "sliding-window attention (window {w}): runtimes may cache fewer tokens \
                 for those layers; this is an upper bound"
            ),
            Self::LatentAttention {
                key_length_mla,
                value_length_mla,
            } => write!(
                f,
                "multi-head latent attention (key_length_mla {key_length_mla}, \
                 value_length_mla {value_length_mla}): the compressed latent is cached as K \
                 and there is no V cache, as in llama.cpp"
            ),
            Self::UncompressedLatentAttention(rank) => write!(
                f,
                "multi-head latent attention (kv_lora_rank {rank}) in the older layout \
                 without key_length_mla/value_length_mla: llama.cpp caches full per-head K \
                 and V for such files, as estimated here"
            ),
            Self::RecurrentState => f.write_str("recurrent (SSM) state is not included"),
            Self::LayersWithoutKv(1) => f.write_str("1 layer has no KV heads"),
            Self::LayersWithoutKv(n) => write!(f, "{n} layers have no KV heads"),
        }
    }
}

/// The result of [`estimate`].
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct KvCacheEstimate {
    /// The configuration estimated for.
    pub config: KvCacheConfig,
    /// `[arch].block_count`.
    pub block_count: u64,
    /// `[arch].attention.head_count`, if present.
    pub head_count: Option<PerLayer>,
    /// `[arch].attention.head_count_kv`, falling back to `head_count`.
    pub head_count_kv: PerLayer,
    /// Key elements per KV head.
    pub key_length: u64,
    /// Value elements per KV head. Not cached when
    /// [`has_value_cache`](Self::has_value_cache) is `false`.
    pub value_length: u64,
    /// Whether the head sizes were read or derived.
    pub head_dim_source: HeadDimSource,
    /// Whether values are cached. `false` for multi-head latent attention,
    /// where the latent is cached as K only.
    pub has_value_cache: bool,
    /// Key bytes per token, summed over layers.
    pub k_bytes_per_token: u64,
    /// Value bytes per token, summed over layers.
    pub v_bytes_per_token: u64,
    /// Total cache size in bytes.
    pub total_bytes: u64,
    /// Features worth pointing out, see [`KvCacheNote`].
    pub notes: Vec<KvCacheNote>,
}

impl KvCacheEstimate {
    /// Key plus value bytes per token, summed over layers.
    pub fn bytes_per_token(&self) -> u64 {
        self.k_bytes_per_token + self.v_bytes_per_token
    }
}

/// Why an estimate could not be made.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum KvCacheError {
    /// `general.architecture` is missing, so `[arch].*` keys cannot be resolved.
    MissingArchitecture,
    /// A required key is missing.
    MissingKey(String),
    /// A key is present, but its value cannot be used.
    InvalidKey {
        /// The key.
        key: String,
        /// What the value must be.
        expected: &'static str,
        /// What it is, for example `"i32 -8"` or `"array[f32]"`.
        found: String,
    },
    /// Head sizes must be derived, but `head_count` (the key named here) is 0
    /// while some layers have KV heads.
    ZeroHeads(String),
    /// Head sizes must be derived, but the per-layer `head_count` list (the
    /// key named here) differs between attention layers.
    VaryingHeadCount(String),
    /// `embedding_length / head_count` is 0.
    ZeroHeadSize {
        /// `[arch].embedding_length`.
        embedding_length: u64,
        /// The head count divided by.
        head_count: u64,
    },
    /// A per-layer list has a different length than `block_count`.
    LayerCountMismatch {
        /// The per-layer key.
        key: String,
        /// `block_count`.
        expected: u64,
        /// Length of the list.
        found: usize,
    },
    /// A K or V row is not a whole number of blocks of the cache type.
    PartialBlock {
        /// The cache type.
        cache_type: GgmlType,
        /// Elements in the row.
        row_elements: u64,
    },
    /// The size does not fit in `u64`.
    Overflow,
}

impl fmt::Display for KvCacheError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingArchitecture => f.write_str("general.architecture is missing"),
            Self::MissingKey(key) => write!(f, "required key `{key}` is missing"),
            Self::InvalidKey {
                key,
                expected,
                found,
            } => write!(f, "`{key}` must be {expected}, found {found}"),
            Self::ZeroHeads(key) => write!(
                f,
                "`{key}` is 0 but some layers have KV heads, so key_length and value_length \
                 cannot be derived"
            ),
            Self::VaryingHeadCount(key) => write!(
                f,
                "`{key}` differs between layers and the file does not set both key_length and \
                 value_length, so they cannot be derived"
            ),
            Self::ZeroHeadSize {
                embedding_length,
                head_count,
            } => write!(
                f,
                "embedding_length {embedding_length} / head_count {head_count} gives a head \
                 size of 0"
            ),
            Self::LayerCountMismatch {
                key,
                expected,
                found,
            } => write!(
                f,
                "`{key}` has {found} entries but block_count is {expected}"
            ),
            Self::PartialBlock {
                cache_type,
                row_elements,
            } => write!(
                f,
                "a row of {row_elements} elements is not a multiple of the {cache_type} block \
                 size {}",
                cache_type.block_size()
            ),
            Self::Overflow => f.write_str("the cache size overflows u64"),
        }
    }
}

impl std::error::Error for KvCacheError {}

/// Estimates the KV-cache size of `file` for `config`.
///
/// ```
/// use gguf_inspect::kv_cache::{KvCacheConfig, estimate};
/// use gguf_inspect::GgufFile;
/// # fn entry(b: &mut Vec<u8>, key: &str, value: u32) {
/// #     b.extend_from_slice(&(key.len() as u64).to_le_bytes());
/// #     b.extend_from_slice(key.as_bytes());
/// #     b.extend_from_slice(&4u32.to_le_bytes()); // u32
/// #     b.extend_from_slice(&value.to_le_bytes());
/// # }
/// # let mut bytes = b"GGUF".to_vec();
/// # bytes.extend_from_slice(&3u32.to_le_bytes());
/// # bytes.extend_from_slice(&0u64.to_le_bytes());
/// # bytes.extend_from_slice(&5u64.to_le_bytes());
/// # let arch = b"general.architecture";
/// # bytes.extend_from_slice(&(arch.len() as u64).to_le_bytes());
/// # bytes.extend_from_slice(arch);
/// # bytes.extend_from_slice(&8u32.to_le_bytes());
/// # bytes.extend_from_slice(&5u64.to_le_bytes());
/// # bytes.extend_from_slice(b"llama");
/// # entry(&mut bytes, "llama.block_count", 32);
/// # entry(&mut bytes, "llama.embedding_length", 4096);
/// # entry(&mut bytes, "llama.attention.head_count", 32);
/// # entry(&mut bytes, "llama.attention.head_count_kv", 8);
/// // An 8B Llama-3-shaped header: 32 layers, 32 heads, 8 KV heads, n_embd 4096.
/// let gguf = GgufFile::from_bytes(&bytes)?;
/// let kv = estimate(&gguf, &KvCacheConfig::new(8192))?;
/// // 32 layers × (K + V) × 8 KV heads × 128 × 2 bytes per f16 element.
/// assert_eq!(kv.bytes_per_token(), 32 * 2 * 8 * 128 * 2);
/// assert_eq!(kv.total_bytes, 1 << 30); // 1 GiB at 8,192 tokens
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// # Errors
///
/// Returns an error if the hyperparameters needed are missing, mistyped or
/// inconsistent.
pub fn estimate(file: &GgufFile, config: &KvCacheConfig) -> Result<KvCacheEstimate, KvCacheError> {
    let prefix = file
        .architecture()
        .ok_or(KvCacheError::MissingArchitecture)?;
    let keys = Keys {
        m: file.metadata(),
        prefix,
    };

    let block_count = keys.required_u64("block_count")?;
    let head_count = keys.per_layer("attention.head_count", block_count)?;
    let head_count_kv = match keys.per_layer("attention.head_count_kv", block_count)? {
        Some(kv) => kv,
        // Without head_count_kv every head has its own keys and values.
        None => head_count
            .clone()
            .ok_or_else(|| KvCacheError::MissingKey(keys.name("attention.head_count")))?,
    };
    let has_kv_heads = head_count_kv.range().is_some_and(|(_, max)| max > 0);

    let explicit_k = keys.u64("attention.key_length")?;
    let explicit_v = keys.u64("attention.value_length")?;
    let (key_length, value_length, head_dim_source) = match (explicit_k, explicit_v) {
        (Some(k), Some(v)) => (k, v, HeadDimSource::Explicit),
        _ if !has_kv_heads => (
            explicit_k.unwrap_or(0),
            explicit_v.unwrap_or(0),
            HeadDimSource::Unused,
        ),
        _ => {
            let derived = derived_head_size(&keys, head_count.as_ref())?;
            (
                explicit_k.unwrap_or(derived),
                explicit_v.unwrap_or(derived),
                HeadDimSource::Derived,
            )
        }
    };

    let latent = latent_attention(&keys)?;
    let has_value_cache = latent.is_none();
    let sums = LayerSums::compute(
        &head_count_kv,
        block_count,
        key_length,
        if has_value_cache { value_length } else { 0 },
        config.cache_type,
    )?;
    let total_bytes = sums
        .k_bytes
        .checked_add(sums.v_bytes)
        .and_then(|b| b.checked_mul(config.context_length))
        .and_then(|b| b.checked_mul(config.sequences))
        .ok_or(KvCacheError::Overflow)?;

    Ok(KvCacheEstimate {
        config: *config,
        block_count,
        head_count,
        head_count_kv,
        key_length,
        value_length,
        head_dim_source,
        has_value_cache,
        k_bytes_per_token: sums.k_bytes,
        v_bytes_per_token: sums.v_bytes,
        total_bytes,
        notes: notes(&keys, latent, sums.empty_layers),
    })
}

const INTEGER: &str = "a non-negative integer";
const INTEGER_OR_LIST: &str = "a non-negative integer or a per-layer array of them";

/// Strict lookups of `[arch].key`: absent is `None`, present but unusable is
/// an error.
struct Keys<'a> {
    m: &'a Metadata,
    prefix: &'a str,
}

impl Keys<'_> {
    fn name(&self, key: &str) -> String {
        format!("{}.{key}", self.prefix)
    }

    fn get(&self, key: &str) -> Option<&MetadataValue> {
        self.m.get(&self.name(key))
    }

    fn invalid(&self, key: &str, expected: &'static str, value: &MetadataValue) -> KvCacheError {
        KvCacheError::InvalidKey {
            key: self.name(key),
            expected,
            found: describe(value),
        }
    }

    fn u64(&self, key: &str) -> Result<Option<u64>, KvCacheError> {
        let Some(value) = self.get(key) else {
            return Ok(None);
        };
        value
            .as_u64()
            .map(Some)
            .ok_or_else(|| self.invalid(key, INTEGER, value))
    }

    fn required_u64(&self, key: &str) -> Result<u64, KvCacheError> {
        self.u64(key)?
            .ok_or_else(|| KvCacheError::MissingKey(self.name(key)))
    }

    /// A per-layer value; a list must have one entry per layer.
    fn per_layer(&self, key: &str, block_count: u64) -> Result<Option<PerLayer>, KvCacheError> {
        let Some(value) = self.get(key) else {
            return Ok(None);
        };
        let parsed =
            PerLayer::from_value(value).ok_or_else(|| self.invalid(key, INTEGER_OR_LIST, value))?;
        match parsed.list_len() {
            Some(found) if found as u64 != block_count => Err(KvCacheError::LayerCountMismatch {
                key: self.name(key),
                expected: block_count,
                found,
            }),
            _ => Ok(Some(parsed)),
        }
    }

    /// Lenient lookup for keys that only add a note.
    fn note_u64(&self, key: &str) -> Option<u64> {
        self.get(key)?.as_u64()
    }
}

/// A short description of a value that cannot be used, for error messages.
/// It never includes string contents.
fn describe(value: &MetadataValue) -> String {
    match value {
        MetadataValue::String(_) => "a string".to_owned(),
        MetadataValue::Bool(b) => format!("bool {b}"),
        MetadataValue::F32(v) => format!("f32 {v}"),
        MetadataValue::F64(v) => format!("f64 {v}"),
        MetadataValue::Array(a) if a.element_type().is_integer() && a.to_u64_vec().is_none() => {
            format!("array[{}] with a negative entry", a.element_type())
        }
        MetadataValue::Array(a) => format!("array[{}]", a.element_type()),
        other => match other.as_i64() {
            Some(v) => format!("{} {v}", other.value_type()),
            None => other.value_type().to_string(),
        },
    }
}

/// `embedding_length / head_count`. In a per-layer `head_count` list, layers
/// with 0 heads (the recurrent layers of hybrid models) are skipped.
fn derived_head_size(keys: &Keys<'_>, head_count: Option<&PerLayer>) -> Result<u64, KvCacheError> {
    let key = keys.name("attention.head_count");
    let heads = match head_count {
        None => return Err(KvCacheError::MissingKey(key)),
        Some(PerLayer::Uniform(heads)) => *heads,
        Some(PerLayer::List(list)) => {
            let mut attention_layers = list.iter().copied().filter(|&h| h > 0);
            let first = attention_layers.next().unwrap_or(0);
            if attention_layers.any(|h| h != first) {
                return Err(KvCacheError::VaryingHeadCount(key));
            }
            first
        }
    };
    if heads == 0 {
        return Err(KvCacheError::ZeroHeads(key));
    }
    let embedding_length = keys.required_u64("embedding_length")?;
    match embedding_length / heads {
        0 => Err(KvCacheError::ZeroHeadSize {
            embedding_length,
            head_count: heads,
        }),
        size => Ok(size),
    }
}

/// `(key_length_mla, value_length_mla)` if both are set and non-zero, which
/// is how llama.cpp decides that a model uses multi-head latent attention.
fn latent_attention(keys: &Keys<'_>) -> Result<Option<(u64, u64)>, KvCacheError> {
    let k = keys.u64("attention.key_length_mla")?;
    let v = keys.u64("attention.value_length_mla")?;
    Ok(match (k, v) {
        (Some(k), Some(v)) if k > 0 && v > 0 => Some((k, v)),
        _ => None,
    })
}

/// Per-token K and V bytes summed over layers.
struct LayerSums {
    k_bytes: u64,
    v_bytes: u64,
    empty_layers: u64,
}

impl LayerSums {
    fn compute(
        head_count_kv: &PerLayer,
        block_count: u64,
        key_length: u64,
        value_length: u64,
        cache_type: GgmlType,
    ) -> Result<Self, KvCacheError> {
        let row_bytes = |heads: u64, head_dim: u64| -> Result<u64, KvCacheError> {
            let row_elements = heads.checked_mul(head_dim).ok_or(KvCacheError::Overflow)?;
            cache_type
                .row_size(row_elements)
                .ok_or(KvCacheError::PartialBlock {
                    cache_type,
                    row_elements,
                })
        };
        let add = |acc: u64, x: u64| acc.checked_add(x).ok_or(KvCacheError::Overflow);
        match head_count_kv {
            PerLayer::Uniform(heads) => {
                let times_layers =
                    |b: u64| b.checked_mul(block_count).ok_or(KvCacheError::Overflow);
                Ok(Self {
                    k_bytes: times_layers(row_bytes(*heads, key_length)?)?,
                    v_bytes: times_layers(row_bytes(*heads, value_length)?)?,
                    empty_layers: if *heads == 0 { block_count } else { 0 },
                })
            }
            PerLayer::List(per_layer) => {
                let mut sums = Self {
                    k_bytes: 0,
                    v_bytes: 0,
                    empty_layers: 0,
                };
                for &heads in per_layer {
                    sums.k_bytes = add(sums.k_bytes, row_bytes(heads, key_length)?)?;
                    sums.v_bytes = add(sums.v_bytes, row_bytes(heads, value_length)?)?;
                    sums.empty_layers += u64::from(heads == 0);
                }
                Ok(sums)
            }
        }
    }
}

fn notes(keys: &Keys<'_>, latent: Option<(u64, u64)>, empty_layers: u64) -> Vec<KvCacheNote> {
    let mut notes = Vec::new();
    if let Some(window) = keys.note_u64("attention.sliding_window") {
        notes.push(KvCacheNote::SlidingWindow(window));
    }
    match latent {
        Some((key_length_mla, value_length_mla)) => notes.push(KvCacheNote::LatentAttention {
            key_length_mla,
            value_length_mla,
        }),
        None => {
            if let Some(rank) = keys.note_u64("attention.kv_lora_rank") {
                notes.push(KvCacheNote::UncompressedLatentAttention(rank));
            }
        }
    }
    let ssm_prefix = keys.name("ssm.");
    if keys.m.iter().any(|(k, _)| k.starts_with(&ssm_prefix)) {
        notes.push(KvCacheNote::RecurrentState);
    }
    if empty_layers > 0 {
        notes.push(KvCacheNote::LayersWithoutKv(empty_layers));
    }
    notes
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MetadataArray;

    #[test]
    fn messages() {
        let notes = [
            (
                KvCacheNote::SlidingWindow(4096),
                "sliding-window attention (window 4096)",
            ),
            (
                KvCacheNote::LatentAttention {
                    key_length_mla: 192,
                    value_length_mla: 128,
                },
                "multi-head latent attention (key_length_mla 192, value_length_mla 128)",
            ),
            (
                KvCacheNote::UncompressedLatentAttention(512),
                "multi-head latent attention (kv_lora_rank 512) in the older layout",
            ),
            (
                KvCacheNote::RecurrentState,
                "recurrent (SSM) state is not included",
            ),
            (KvCacheNote::LayersWithoutKv(1), "1 layer has no KV heads"),
            (KvCacheNote::LayersWithoutKv(2), "2 layers have no KV heads"),
        ];
        for (note, prefix) in notes {
            assert!(note.to_string().starts_with(prefix), "{note}");
        }
        let errors = [
            (
                KvCacheError::MissingKey("llama.block_count".into()),
                "required key `llama.block_count` is missing",
            ),
            (
                KvCacheError::InvalidKey {
                    key: "x.attention.head_count_kv".into(),
                    expected: INTEGER_OR_LIST,
                    found: "i32 -8".into(),
                },
                "`x.attention.head_count_kv` must be a non-negative integer or a per-layer array \
                 of them, found i32 -8",
            ),
            (
                KvCacheError::ZeroHeads("mamba.attention.head_count".into()),
                "`mamba.attention.head_count` is 0 but some layers have KV heads, so key_length \
                 and value_length cannot be derived",
            ),
            (
                KvCacheError::ZeroHeadSize {
                    embedding_length: 64,
                    head_count: 128,
                },
                "embedding_length 64 / head_count 128 gives a head size of 0",
            ),
            (
                KvCacheError::LayerCountMismatch {
                    key: "x.attention.head_count_kv".into(),
                    expected: 32,
                    found: 2,
                },
                "`x.attention.head_count_kv` has 2 entries but block_count is 32",
            ),
            (
                KvCacheError::PartialBlock {
                    cache_type: GgmlType::Q8_0,
                    row_elements: 40,
                },
                "a row of 40 elements is not a multiple of the q8_0 block size 32",
            ),
            (KvCacheError::Overflow, "the cache size overflows u64"),
        ];
        for (err, expected) in errors {
            assert_eq!(err.to_string(), expected);
        }
        assert!(
            KvCacheError::VaryingHeadCount("x.attention.head_count".into())
                .to_string()
                .starts_with("`x.attention.head_count` differs between layers")
        );
    }

    #[test]
    fn invalid_values_are_described_without_their_contents() {
        let cases = [
            (MetadataValue::I32(-8), "i32 -8"),
            (MetadataValue::F32(8.0), "f32 8"),
            (MetadataValue::Bool(true), "bool true"),
            (MetadataValue::String("\x1b[2J".into()), "a string"),
            (
                MetadataValue::Array(MetadataArray::I64(vec![8, -1])),
                "array[i64] with a negative entry",
            ),
            (
                MetadataValue::Array(MetadataArray::U32(vec![128])),
                "array[u32]",
            ),
            (
                MetadataValue::Array(MetadataArray::F32(vec![8.0])),
                "array[f32]",
            ),
            (
                MetadataValue::Array(MetadataArray::Array(vec![])),
                "array[array]",
            ),
        ];
        for (value, expected) in cases {
            assert_eq!(describe(&value), expected);
        }
    }

    #[test]
    fn config_builders() {
        let config = KvCacheConfig::new(10)
            .with_cache_type(GgmlType::Q4_0)
            .with_sequences(3);
        assert_eq!(
            (config.context_length, config.cache_type, config.sequences),
            (10, GgmlType::Q4_0, 3)
        );
    }
}
