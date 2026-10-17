//! KV-cache memory estimates for transformer models.
//!
//! For every layer the cache holds one key row and one value row per token:
//!
//! ```text
//! K row = head_count_kv × key_length   elements
//! V row = head_count_kv × value_length elements
//! bytes = sequences × context × Σ_layers (row_size(K row) + row_size(V row))
//! ```
//!
//! `row_size` is the ggml storage size of a row in the cache type (2 bytes per
//! element for `f16`, 34 bytes per 32 elements for `q8_0`, ...). Grouped-query
//! attention enters through `head_count_kv`; when the file does not give
//! `key_length`/`value_length` they default to `embedding_length / head_count`,
//! as the spec prescribes.
//!
//! The estimate assumes standard multi-head or grouped-query attention in every
//! layer. Sliding-window layers, multi-head latent attention and recurrent
//! state are not modelled; [`KvCacheEstimate::notes`] flags files that use them.

use std::fmt;

use crate::model::{Arch, PerLayer};
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
pub enum HeadDimSource {
    /// `attention.key_length` and `attention.value_length` are in the file.
    Explicit,
    /// At least one was derived as `embedding_length / head_count`.
    Derived,
}

/// Model features the estimate does not account for.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum KvCacheNote {
    /// `attention.sliding_window` is set: runtimes may cache fewer tokens for
    /// sliding-window layers, so the estimate is an upper bound.
    SlidingWindow(u64),
    /// `attention.kv_lora_rank` is set (multi-head latent attention): runtimes
    /// may cache compressed latents instead, so the estimate may be far off.
    LatentAttention(u64),
    /// `ssm.*` keys are present: recurrent state is not included.
    RecurrentState,
    /// Some layers have zero KV heads and contribute nothing.
    LayersWithoutKv(usize),
}

impl fmt::Display for KvCacheNote {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SlidingWindow(w) => write!(
                f,
                "sliding-window attention (window {w}): runtimes may cache fewer tokens \
                 for those layers; this is an upper bound"
            ),
            Self::LatentAttention(r) => write!(
                f,
                "multi-head latent attention (kv_lora_rank {r}): runtimes may cache \
                 compressed latents; this estimate assumes full K/V and may be far off"
            ),
            Self::RecurrentState => f.write_str("recurrent (SSM) state is not included"),
            Self::LayersWithoutKv(n) => write!(f, "{n} layer(s) have no KV heads"),
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
    /// Value elements per KV head.
    pub value_length: u64,
    /// Whether the head sizes were read or derived.
    pub head_dim_source: HeadDimSource,
    /// Key bytes per token, summed over layers.
    pub k_bytes_per_token: u64,
    /// Value bytes per token, summed over layers.
    pub v_bytes_per_token: u64,
    /// Total cache size in bytes.
    pub total_bytes: u64,
    /// Features the estimate does not model.
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
    /// A required key is missing or has the wrong type.
    MissingKey(String),
    /// `head_count` is zero, so head sizes cannot be derived.
    ZeroHeads,
    /// Head sizes must be derived but `head_count` varies per layer.
    VaryingHeadCount,
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
            Self::MissingKey(key) => write!(f, "required key `{key}` is missing or not an integer"),
            Self::ZeroHeads => f.write_str("attention.head_count is zero"),
            Self::VaryingHeadCount => f.write_str(
                "attention.head_count varies per layer and key_length/value_length are not set",
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
/// # Errors
///
/// Returns an error if the hyperparameters needed are missing or inconsistent.
pub fn estimate(file: &GgufFile, config: &KvCacheConfig) -> Result<KvCacheEstimate, KvCacheError> {
    let prefix = file
        .architecture()
        .ok_or(KvCacheError::MissingArchitecture)?;
    let arch = Arch {
        m: file.metadata(),
        prefix: Some(prefix),
    };
    let missing = |key: &str| KvCacheError::MissingKey(format!("{prefix}.{key}"));

    let block_count = arch
        .u64("block_count")
        .ok_or_else(|| missing("block_count"))?;
    let head_count = arch.per_layer("attention.head_count");
    let head_count_kv = arch
        .per_layer("attention.head_count_kv")
        .or_else(|| head_count.clone())
        .ok_or_else(|| missing("attention.head_count"))?;
    if let PerLayer::Varying(values) = &head_count_kv
        && values.len() as u64 != block_count
    {
        return Err(KvCacheError::LayerCountMismatch {
            key: format!("{prefix}.attention.head_count_kv"),
            expected: block_count,
            found: values.len(),
        });
    }

    let (key_length, value_length, head_dim_source) = head_dims(&arch, head_count.as_ref())?;
    let sums = LayerSums::compute(
        &head_count_kv,
        block_count,
        key_length,
        value_length,
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
        k_bytes_per_token: sums.k_bytes,
        v_bytes_per_token: sums.v_bytes,
        total_bytes,
        notes: notes(&arch, prefix, sums.empty_layers),
    })
}

/// Resolves the key and value head sizes, deriving missing ones as
/// `embedding_length / head_count`.
fn head_dims(
    arch: &Arch<'_>,
    head_count: Option<&PerLayer>,
) -> Result<(u64, u64, HeadDimSource), KvCacheError> {
    let missing = |key: &str| {
        arch.key(key)
            .map_or(KvCacheError::MissingArchitecture, KvCacheError::MissingKey)
    };
    let derived = || -> Result<u64, KvCacheError> {
        let heads = head_count
            .ok_or_else(|| missing("attention.head_count"))?
            .uniform()
            .ok_or(KvCacheError::VaryingHeadCount)?;
        let n_embd = arch
            .u64("embedding_length")
            .ok_or_else(|| missing("embedding_length"))?;
        n_embd.checked_div(heads).ok_or(KvCacheError::ZeroHeads)
    };
    let explicit_k = arch.u64("attention.key_length");
    let explicit_v = arch.u64("attention.value_length");
    let source = if explicit_k.is_some() && explicit_v.is_some() {
        HeadDimSource::Explicit
    } else {
        HeadDimSource::Derived
    };
    Ok((
        explicit_k.map_or_else(derived, Ok)?,
        explicit_v.map_or_else(derived, Ok)?,
        source,
    ))
}

/// Per-token K and V bytes summed over layers.
struct LayerSums {
    k_bytes: u64,
    v_bytes: u64,
    empty_layers: usize,
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
                    empty_layers: if *heads == 0 {
                        usize::try_from(block_count).unwrap_or(usize::MAX)
                    } else {
                        0
                    },
                })
            }
            PerLayer::Varying(per_layer) => {
                let mut sums = Self {
                    k_bytes: 0,
                    v_bytes: 0,
                    empty_layers: 0,
                };
                for &heads in per_layer {
                    sums.k_bytes = add(sums.k_bytes, row_bytes(heads, key_length)?)?;
                    sums.v_bytes = add(sums.v_bytes, row_bytes(heads, value_length)?)?;
                    sums.empty_layers += usize::from(heads == 0);
                }
                Ok(sums)
            }
        }
    }
}

fn notes(arch: &Arch<'_>, prefix: &str, empty_layers: usize) -> Vec<KvCacheNote> {
    let mut notes = Vec::new();
    if let Some(window) = arch.u64("attention.sliding_window") {
        notes.push(KvCacheNote::SlidingWindow(window));
    }
    if let Some(rank) = arch.u64("attention.kv_lora_rank") {
        notes.push(KvCacheNote::LatentAttention(rank));
    }
    let ssm_prefix = format!("{prefix}.ssm.");
    if arch.m.iter().any(|(k, _)| k.starts_with(&ssm_prefix)) {
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

    #[test]
    fn messages() {
        let notes = [
            (
                KvCacheNote::SlidingWindow(4096),
                "sliding-window attention (window 4096)",
            ),
            (
                KvCacheNote::LatentAttention(512),
                "multi-head latent attention (kv_lora_rank 512)",
            ),
            (
                KvCacheNote::RecurrentState,
                "recurrent (SSM) state is not included",
            ),
            (
                KvCacheNote::LayersWithoutKv(2),
                "2 layer(s) have no KV heads",
            ),
        ];
        for (note, prefix) in notes {
            assert!(note.to_string().starts_with(prefix), "{note}");
        }
        let errors = [
            (
                KvCacheError::MissingKey("llama.block_count".into()),
                "required key `llama.block_count` is missing or not an integer",
            ),
            (KvCacheError::ZeroHeads, "attention.head_count is zero"),
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
