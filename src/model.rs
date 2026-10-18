//! Well-known model hyperparameters, resolved from metadata.
//!
//! Architecture-specific keys are spelled `[arch].key` in the spec, for example
//! `llama.context_length`; [`ModelInfo`] looks them up under the file's
//! `general.architecture`.

use std::fmt;

use crate::GgufFile;
use crate::metadata::{Metadata, MetadataValue};

/// A hyperparameter given either once for all layers or once per layer.
///
/// Most models store head counts as one integer; some (for example models with
/// hybrid attention/recurrent layers) store an array with one entry per layer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PerLayer {
    /// One integer that applies to every layer.
    Uniform(u64),
    /// An array with one entry per layer, as stored in the file. The entries
    /// may all be equal; the array is kept so that its length can be checked
    /// against `block_count`.
    List(Vec<u64>),
}

impl PerLayer {
    /// Interprets an integer or an array of integers, all non-negative.
    /// Returns `None` for anything else, including negative numbers, floats
    /// and nested arrays.
    ///
    /// ```
    /// use gguf_inspect::model::PerLayer;
    /// use gguf_inspect::{MetadataArray, MetadataValue};
    ///
    /// assert_eq!(PerLayer::from_value(&MetadataValue::U32(8)), Some(PerLayer::Uniform(8)));
    /// let list = MetadataValue::Array(MetadataArray::I32(vec![0, 8]));
    /// assert_eq!(PerLayer::from_value(&list), Some(PerLayer::List(vec![0, 8])));
    /// assert_eq!(PerLayer::from_value(&MetadataValue::I32(-8)), None);
    /// ```
    pub fn from_value(value: &MetadataValue) -> Option<Self> {
        if let Some(v) = value.as_u64() {
            return Some(Self::Uniform(v));
        }
        value.as_array()?.to_u64_vec().map(Self::List)
    }

    /// The value for `layer`, or `None` past the end of a per-layer list.
    pub fn get(&self, layer: usize) -> Option<u64> {
        match self {
            Self::Uniform(v) => Some(*v),
            Self::List(values) => values.get(layer).copied(),
        }
    }

    /// The value, if every layer has the same one: a uniform value, or a
    /// non-empty list whose entries are all equal.
    pub fn uniform(&self) -> Option<u64> {
        match self {
            Self::Uniform(v) => Some(*v),
            Self::List(values) => {
                let (first, rest) = values.split_first()?;
                rest.iter().all(|v| v == first).then_some(*first)
            }
        }
    }

    /// Number of entries, if this is a per-layer list.
    pub fn list_len(&self) -> Option<usize> {
        match self {
            Self::Uniform(_) => None,
            Self::List(values) => Some(values.len()),
        }
    }

    /// Smallest and largest value, or `None` for an empty list.
    pub fn range(&self) -> Option<(u64, u64)> {
        match self {
            Self::Uniform(v) => Some((*v, *v)),
            Self::List(values) => {
                Some((values.iter().copied().min()?, values.iter().copied().max()?))
            }
        }
    }
}

/// `8`, `8 (32 per-layer entries)` or `0-8 (4 per-layer entries)`.
impl fmt::Display for PerLayer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Self::List(values) = self else {
            return write!(f, "{}", self.uniform().unwrap_or_default());
        };
        let entries = match values.len() {
            1 => "1 per-layer entry".to_owned(),
            n => format!("{n} per-layer entries"),
        };
        match self.range() {
            None => write!(f, "empty list"),
            Some((min, max)) if min == max => write!(f, "{min} ({entries})"),
            Some((min, max)) => write!(f, "{min}-{max} ({entries})"),
        }
    }
}

/// RoPE parameters (`[arch].rope.*`).
#[derive(Debug, Clone, Default, PartialEq)]
#[non_exhaustive]
pub struct RopeInfo {
    /// `rope.dimension_count`: rotary dimensions per head.
    pub dimension_count: Option<u64>,
    /// `rope.freq_base`: base frequency. `f32` values are converted via their
    /// shortest decimal form (0.1 stays 0.1).
    pub freq_base: Option<f64>,
    /// `rope.scaling.type`: `none`, `linear`, `yarn`, ...
    pub scaling_type: Option<String>,
    /// `rope.scaling.factor`, or the legacy `rope.scale_linear`.
    pub scaling_factor: Option<f64>,
    /// `rope.scaling.original_context_length`.
    pub original_context_length: Option<u64>,
}

impl RopeInfo {
    /// Whether no RoPE key is present.
    pub fn is_empty(&self) -> bool {
        *self == Self::default()
    }
}

/// Embedded tokenizer summary (`tokenizer.ggml.*`).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct TokenizerInfo {
    /// `tokenizer.ggml.model`, for example `llama` or `gpt2`.
    pub model: Option<String>,
    /// Length of `tokenizer.ggml.tokens`, or `[arch].vocab_size` if there is no token list.
    pub vocab_size: Option<u64>,
}

/// Commonly needed hyperparameters of a model.
///
/// ```
/// use gguf_inspect::{GgufFile, ModelInfo};
///
/// # let mut bytes = b"GGUF".to_vec();
/// # bytes.extend_from_slice(&3u32.to_le_bytes());
/// # bytes.extend_from_slice(&0u64.to_le_bytes());
/// # bytes.extend_from_slice(&0u64.to_le_bytes());
/// let gguf = GgufFile::from_bytes(&bytes)?;
/// let info = ModelInfo::from_file(&gguf);
/// assert_eq!(info.architecture, None);
/// # Ok::<(), gguf_inspect::Error>(())
/// ```
#[derive(Debug, Clone, Default, PartialEq)]
#[non_exhaustive]
pub struct ModelInfo {
    /// `general.architecture`.
    pub architecture: Option<String>,
    /// `general.name`.
    pub name: Option<String>,
    /// `general.file_type`: the type of most tensors, see [`file_type_name`].
    pub file_type: Option<u64>,
    /// `general.quantization_version`.
    pub quantization_version: Option<u64>,
    /// `[arch].context_length`: training context length (`n_ctx`).
    pub context_length: Option<u64>,
    /// `[arch].embedding_length` (`n_embd`).
    pub embedding_length: Option<u64>,
    /// `[arch].block_count`: number of layers.
    pub block_count: Option<u64>,
    /// `[arch].feed_forward_length` (`n_ff`).
    pub feed_forward_length: Option<PerLayer>,
    /// `[arch].attention.head_count` (`n_head`).
    pub head_count: Option<PerLayer>,
    /// `[arch].attention.head_count_kv`; fewer KV heads than heads means
    /// grouped-query attention.
    pub head_count_kv: Option<PerLayer>,
    /// `[arch].attention.key_length`: key head size, if not `n_embd / n_head`.
    pub key_length: Option<u64>,
    /// `[arch].attention.value_length`: value head size, if not `n_embd / n_head`.
    pub value_length: Option<u64>,
    /// `[arch].expert_count` (mixture-of-experts models).
    pub expert_count: Option<u64>,
    /// `[arch].expert_used_count` (mixture-of-experts models).
    pub expert_used_count: Option<u64>,
    /// RoPE parameters.
    pub rope: RopeInfo,
    /// Tokenizer summary.
    pub tokenizer: TokenizerInfo,
}

impl ModelInfo {
    /// Collects the well-known keys present in `file`. Missing or mistyped
    /// keys are left as `None`.
    pub fn from_file(file: &GgufFile) -> Self {
        Self::from_metadata(file.metadata())
    }

    /// Like [`from_file`](Self::from_file), from metadata alone.
    pub fn from_metadata(m: &Metadata) -> Self {
        let architecture = m.get_str("general.architecture").map(str::to_owned);
        let arch = Arch {
            m,
            prefix: architecture.as_deref(),
        };
        let vocab_size = m
            .get("tokenizer.ggml.tokens")
            .and_then(MetadataValue::as_array)
            .map(|a| a.len() as u64)
            .or_else(|| arch.u64("vocab_size"));
        Self {
            name: m.get_str("general.name").map(str::to_owned),
            file_type: m.get_u64("general.file_type"),
            quantization_version: m.get_u64("general.quantization_version"),
            context_length: arch.u64("context_length"),
            embedding_length: arch.u64("embedding_length"),
            block_count: arch.u64("block_count"),
            feed_forward_length: arch.per_layer("feed_forward_length"),
            head_count: arch.per_layer("attention.head_count"),
            head_count_kv: arch.per_layer("attention.head_count_kv"),
            key_length: arch.u64("attention.key_length"),
            value_length: arch.u64("attention.value_length"),
            expert_count: arch.u64("expert_count"),
            expert_used_count: arch.u64("expert_used_count"),
            rope: RopeInfo {
                dimension_count: arch.u64("rope.dimension_count"),
                freq_base: arch.f64("rope.freq_base"),
                scaling_type: arch.str("rope.scaling.type"),
                scaling_factor: arch
                    .f64("rope.scaling.factor")
                    .or_else(|| arch.f64("rope.scale_linear")),
                original_context_length: arch.u64("rope.scaling.original_context_length"),
            },
            tokenizer: TokenizerInfo {
                model: m.get_str("tokenizer.ggml.model").map(str::to_owned),
                vocab_size,
            },
            architecture,
        }
    }
}

/// Looks up `[arch].key` for one architecture, leniently: a missing or
/// mistyped key is `None`.
struct Arch<'a> {
    m: &'a Metadata,
    prefix: Option<&'a str>,
}

impl<'a> Arch<'a> {
    fn get(&self, key: &str) -> Option<&'a MetadataValue> {
        self.m.get(&format!("{}.{key}", self.prefix?))
    }

    fn u64(&self, key: &str) -> Option<u64> {
        self.get(key)?.as_u64()
    }

    /// Floats widened through their shortest decimal form, so an `f32`
    /// 0.1 reads as 0.1 rather than 0.10000000149011612.
    fn f64(&self, key: &str) -> Option<f64> {
        match self.get(key)? {
            MetadataValue::F32(v) => v.to_string().parse().ok(),
            other => other.as_f64(),
        }
    }

    fn str(&self, key: &str) -> Option<String> {
        self.get(key)?.as_str().map(str::to_owned)
    }

    fn per_layer(&self, key: &str) -> Option<PerLayer> {
        PerLayer::from_value(self.get(key)?)
    }
}

/// Name of a `general.file_type` value, as listed in the GGUF specification
/// (IDs 0-18). Newer IDs defined only by llama.cpp return `None`.
///
/// ```
/// assert_eq!(gguf_inspect::model::file_type_name(15), Some("MOSTLY_Q4_K_M"));
/// assert_eq!(gguf_inspect::model::file_type_name(99), None);
/// ```
pub fn file_type_name(id: u64) -> Option<&'static str> {
    const NAMES: [&str; 19] = [
        "ALL_F32",
        "MOSTLY_F16",
        "MOSTLY_Q4_0",
        "MOSTLY_Q4_1",
        "MOSTLY_Q4_1_SOME_F16",
        "MOSTLY_Q4_2",
        "MOSTLY_Q4_3",
        "MOSTLY_Q8_0",
        "MOSTLY_Q5_0",
        "MOSTLY_Q5_1",
        "MOSTLY_Q2_K",
        "MOSTLY_Q3_K_S",
        "MOSTLY_Q3_K_M",
        "MOSTLY_Q3_K_L",
        "MOSTLY_Q4_K_S",
        "MOSTLY_Q4_K_M",
        "MOSTLY_Q5_K_S",
        "MOSTLY_Q5_K_M",
        "MOSTLY_Q6_K",
    ];
    NAMES.get(usize::try_from(id).ok()?).copied()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metadata::MetadataArray;

    fn metadata(entries: Vec<(&str, MetadataValue)>) -> Metadata {
        let mut m = Metadata::default();
        for (k, v) in entries {
            assert!(m.insert(k.to_owned(), v, 0));
        }
        m
    }

    #[test]
    fn resolves_architecture_keys() {
        let m = metadata(vec![
            (
                "general.architecture",
                MetadataValue::String("llama".into()),
            ),
            ("llama.context_length", MetadataValue::U32(4096)),
            ("llama.attention.head_count", MetadataValue::U64(32)),
            ("llama.attention.head_count_kv", MetadataValue::U32(8)),
            ("llama.rope.freq_base", MetadataValue::F32(500_000.0)),
            ("llama.rope.scale_linear", MetadataValue::F32(2.0)),
            (
                "tokenizer.ggml.tokens",
                MetadataValue::Array(MetadataArray::String(vec!["a".into(), "b".into()])),
            ),
            ("mistral.context_length", MetadataValue::U32(1)),
        ]);
        let info = ModelInfo::from_metadata(&m);
        assert_eq!(info.architecture.as_deref(), Some("llama"));
        assert_eq!(info.context_length, Some(4096));
        assert_eq!(info.head_count, Some(PerLayer::Uniform(32)));
        assert_eq!(info.head_count_kv, Some(PerLayer::Uniform(8)));
        assert_eq!(info.rope.freq_base, Some(500_000.0));
        assert_eq!(info.rope.scaling_factor, Some(2.0));
        assert_eq!(info.tokenizer.vocab_size, Some(2));
        assert_eq!(info.block_count, None);
    }

    #[test]
    fn without_architecture_only_general_keys_resolve() {
        let m = metadata(vec![
            ("general.name", MetadataValue::String("x".into())),
            ("llama.context_length", MetadataValue::U32(4096)),
        ]);
        let info = ModelInfo::from_metadata(&m);
        assert_eq!(info.name.as_deref(), Some("x"));
        assert_eq!(info.context_length, None);
        assert!(info.rope.is_empty());
    }

    #[test]
    fn per_layer_values() {
        let arr = |v: Vec<i32>| MetadataValue::Array(MetadataArray::I32(v));
        let same = PerLayer::from_value(&arr(vec![8, 8])).unwrap();
        assert_eq!(same, PerLayer::List(vec![8, 8]), "the list length is kept");
        assert_eq!(same.uniform(), Some(8));
        assert_eq!(same.list_len(), Some(2));
        assert_eq!(same.to_string(), "8 (2 per-layer entries)");

        let varying = PerLayer::from_value(&arr(vec![0, 8, 0, 8])).unwrap();
        assert_eq!(varying.range(), Some((0, 8)));
        assert_eq!(varying.get(1), Some(8));
        assert_eq!(varying.get(4), None);
        assert_eq!(varying.uniform(), None);
        assert_eq!(varying.to_string(), "0-8 (4 per-layer entries)");
        assert_eq!(PerLayer::List(vec![3]).to_string(), "3 (1 per-layer entry)");

        let empty = PerLayer::from_value(&arr(vec![])).unwrap();
        assert_eq!((empty.range(), empty.uniform()), (None, None));
        assert_eq!(empty.to_string(), "empty list");

        let uniform = PerLayer::Uniform(32);
        assert_eq!((uniform.get(99), uniform.list_len()), (Some(32), None));
        assert_eq!(uniform.to_string(), "32");

        assert_eq!(PerLayer::from_value(&arr(vec![-1])), None);
        assert_eq!(PerLayer::from_value(&MetadataValue::F32(8.0)), None);
        assert_eq!(
            PerLayer::from_value(&MetadataValue::String("8".into())),
            None
        );
        let nested = MetadataValue::Array(MetadataArray::Array(vec![MetadataArray::U8(vec![8])]));
        assert_eq!(PerLayer::from_value(&nested), None);
    }

    #[test]
    fn file_type_names_follow_the_spec() {
        assert_eq!(file_type_name(0), Some("ALL_F32"));
        assert_eq!(file_type_name(18), Some("MOSTLY_Q6_K"));
        assert_eq!(file_type_name(19), None);
        assert_eq!(file_type_name(u64::MAX), None);
    }
}
