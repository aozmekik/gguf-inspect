//! Structural comparison of two GGUF headers.

use crate::metadata::{MetadataArray, MetadataValue};
use crate::{GgufFile, GgufVersion, TensorInfo};

/// Differences between two files, borrowing from both.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct FileDiff<'a> {
    /// `(old, new)` format versions, if they differ.
    pub version: Option<(GgufVersion, GgufVersion)>,
    /// `(old, new)` alignments, if they differ.
    pub alignment: Option<(u32, u32)>,
    /// Metadata changes: removed and changed keys in the first file's order,
    /// then added keys in the second file's order.
    pub metadata: Vec<MetadataChange<'a>>,
    /// Tensor changes, ordered the same way. Tensors are matched by name and
    /// compared by shape and type; data offsets are ignored because any
    /// metadata change shifts them.
    pub tensors: Vec<TensorChange<'a>>,
}

impl FileDiff<'_> {
    /// Whether the two headers are equivalent.
    pub fn is_empty(&self) -> bool {
        self.version.is_none()
            && self.alignment.is_none()
            && self.metadata.is_empty()
            && self.tensors.is_empty()
    }
}

/// A metadata key that differs between the two files.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MetadataChange<'a> {
    /// Only in the second file.
    Added {
        /// The key.
        key: &'a str,
        /// Its value in the second file.
        value: &'a MetadataValue,
    },
    /// Only in the first file.
    Removed {
        /// The key.
        key: &'a str,
        /// Its value in the first file.
        value: &'a MetadataValue,
    },
    /// In both, with a different type or value.
    Changed {
        /// The key.
        key: &'a str,
        /// Value in the first file.
        old: &'a MetadataValue,
        /// Value in the second file.
        new: &'a MetadataValue,
    },
}

impl MetadataChange<'_> {
    /// The key concerned.
    pub fn key(&self) -> &str {
        match self {
            Self::Added { key, .. } | Self::Removed { key, .. } | Self::Changed { key, .. } => key,
        }
    }
}

/// A tensor that differs between the two files.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TensorChange<'a> {
    /// Only in the second file.
    Added(&'a TensorInfo),
    /// Only in the first file.
    Removed(&'a TensorInfo),
    /// In both, with a different shape or type.
    Changed {
        /// The tensor in the first file.
        old: &'a TensorInfo,
        /// The tensor in the second file.
        new: &'a TensorInfo,
    },
}

impl TensorChange<'_> {
    /// The tensor name concerned.
    pub fn name(&self) -> &str {
        match self {
            Self::Added(t) | Self::Removed(t) | Self::Changed { old: t, .. } => t.name(),
        }
    }
}

/// Compares the headers of `old` and `new`.
///
/// Values are compared exactly: floats bit for bit, so `NaN` equals itself and
/// `0.0` differs from `-0.0`.
pub fn diff<'a>(old: &'a GgufFile, new: &'a GgufFile) -> FileDiff<'a> {
    let mut metadata = Vec::new();
    for (key, value) in old.metadata().iter() {
        match new.metadata().get(key) {
            None => metadata.push(MetadataChange::Removed { key, value }),
            Some(other) if !identical(value, other) => metadata.push(MetadataChange::Changed {
                key,
                old: value,
                new: other,
            }),
            Some(_) => {}
        }
    }
    for (key, value) in new.metadata().iter() {
        if !old.metadata().contains_key(key) {
            metadata.push(MetadataChange::Added { key, value });
        }
    }

    let mut tensors = Vec::new();
    for t in old.tensors() {
        match new.tensor(t.name()) {
            None => tensors.push(TensorChange::Removed(t)),
            Some(n) if n.dims() != t.dims() || n.ggml_type() != t.ggml_type() => {
                tensors.push(TensorChange::Changed { old: t, new: n });
            }
            Some(_) => {}
        }
    }
    for t in new.tensors() {
        if old.tensor(t.name()).is_none() {
            tensors.push(TensorChange::Added(t));
        }
    }

    FileDiff {
        version: (old.version() != new.version()).then_some((old.version(), new.version())),
        alignment: (old.alignment() != new.alignment())
            .then_some((old.alignment(), new.alignment())),
        metadata,
        tensors,
    }
}

/// Exact equality, comparing floats by their bits.
fn identical(a: &MetadataValue, b: &MetadataValue) -> bool {
    match (a, b) {
        (MetadataValue::F32(x), MetadataValue::F32(y)) => x.to_bits() == y.to_bits(),
        (MetadataValue::F64(x), MetadataValue::F64(y)) => x.to_bits() == y.to_bits(),
        (MetadataValue::Array(x), MetadataValue::Array(y)) => identical_arrays(x, y),
        _ => a == b,
    }
}

fn identical_arrays(a: &MetadataArray, b: &MetadataArray) -> bool {
    match (a, b) {
        (MetadataArray::F32(x), MetadataArray::F32(y)) => x
            .iter()
            .map(|v| v.to_bits())
            .eq(y.iter().map(|v| v.to_bits())),
        (MetadataArray::F64(x), MetadataArray::F64(y)) => x
            .iter()
            .map(|v| v.to_bits())
            .eq(y.iter().map(|v| v.to_bits())),
        (MetadataArray::Array(x), MetadataArray::Array(y)) => {
            x.len() == y.len() && x.iter().zip(y).all(|(p, q)| identical_arrays(p, q))
        }
        _ => a == b,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn floats_compare_by_bits() {
        let nan = MetadataValue::F32(f32::NAN);
        assert!(identical(&nan, &nan));
        assert!(!identical(
            &MetadataValue::F64(0.0),
            &MetadataValue::F64(-0.0)
        ));
        let arr = |v: Vec<f32>| MetadataValue::Array(MetadataArray::F32(v));
        assert!(identical(
            &arr(vec![f32::NAN, 1.0]),
            &arr(vec![f32::NAN, 1.0])
        ));
        assert!(!identical(&arr(vec![1.0]), &arr(vec![1.0, 2.0])));
    }

    #[test]
    fn types_matter() {
        assert!(!identical(&MetadataValue::U32(1), &MetadataValue::U64(1)));
        let nested = |v: Vec<MetadataArray>| MetadataValue::Array(MetadataArray::Array(v));
        assert!(identical(
            &nested(vec![MetadataArray::F64(vec![f64::NAN])]),
            &nested(vec![MetadataArray::F64(vec![f64::NAN])])
        ));
        assert!(!identical(
            &nested(vec![MetadataArray::U8(vec![])]),
            &nested(vec![MetadataArray::I8(vec![])])
        ));
    }
}
