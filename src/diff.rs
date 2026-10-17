//! Structural comparison of two GGUF headers.

use crate::{GgufFile, GgufVersion, MetadataValue, TensorInfo};

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
            Some(other) if !value.bitwise_eq(other) => metadata.push(MetadataChange::Changed {
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

#[cfg(test)]
mod tests {
    use crate::{MetadataArray, MetadataValue};

    #[test]
    fn floats_compare_by_bits() {
        let nan = MetadataValue::F32(f32::NAN);
        assert!(nan.bitwise_eq(&nan));
        assert!(!MetadataValue::F64(0.0).bitwise_eq(&MetadataValue::F64(-0.0)));
        let arr = |v: Vec<f32>| MetadataValue::Array(MetadataArray::F32(v));
        assert!(arr(vec![f32::NAN, 1.0]).bitwise_eq(&arr(vec![f32::NAN, 1.0])));
        assert!(!arr(vec![1.0]).bitwise_eq(&arr(vec![1.0, 2.0])));
    }

    #[test]
    fn types_matter() {
        assert!(!MetadataValue::U32(1).bitwise_eq(&MetadataValue::U64(1)));
        let nested = |v: Vec<MetadataArray>| MetadataValue::Array(MetadataArray::Array(v));
        assert!(
            nested(vec![MetadataArray::F64(vec![f64::NAN])])
                .bitwise_eq(&nested(vec![MetadataArray::F64(vec![f64::NAN])]))
        );
        assert!(
            !nested(vec![MetadataArray::U8(vec![])])
                .bitwise_eq(&nested(vec![MetadataArray::I8(vec![])]))
        );
    }
}
