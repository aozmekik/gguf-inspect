//! Tensor info records.

use std::ops::Range;

use crate::GgmlType;
use crate::error::ErrorKind;

/// Describes one tensor: its name, shape, element type and where its data lives.
///
/// Element and byte counts are computed and overflow-checked when the file is
/// parsed, so the accessors are infallible.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorInfo {
    name: String,
    dims: Vec<u64>,
    ggml_type: GgmlType,
    offset: u64,
    absolute_offset: u64,
    n_elements: u64,
    n_bytes: u64,
}

impl TensorInfo {
    /// Validates a shape the way ggml's `gguf.cpp` does: every dimension
    /// must fit in `i64`, multiplying the dimensions in order must never
    /// exceed `i64::MAX` (so `[2^31, 2^32, 0]` is rejected although it has
    /// no elements), the first dimension must be a whole number of
    /// quantization blocks, and the byte size must fit in `u64`.
    pub(crate) fn new(
        name: String,
        dims: Vec<u64>,
        ggml_type: GgmlType,
        offset: u64,
    ) -> Result<Self, ErrorKind> {
        const MAX: u64 = i64::MAX.unsigned_abs();
        if dims.iter().any(|&d| d > MAX) {
            return Err(ErrorKind::DimensionOverflow(dims));
        }
        let n_elements = dims
            .iter()
            .try_fold(1_u64, |acc, &d| acc.checked_mul(d).filter(|&n| n <= MAX));
        let Some(n_elements) = n_elements else {
            return Err(ErrorKind::ElementCountOverflow(dims));
        };
        // A zero-dimensional tensor is a scalar: one element, ne0 = 1.
        let ne0 = dims.first().copied().unwrap_or(1);
        if ne0 % ggml_type.block_size() != 0 {
            return Err(ErrorKind::PartialBlock { ggml_type, ne0 });
        }
        // n_elements is a multiple of ne0, hence of the block size.
        let Some(n_bytes) =
            (n_elements / ggml_type.block_size()).checked_mul(ggml_type.type_size())
        else {
            return Err(ErrorKind::ByteSizeOverflow { dims, ggml_type });
        };
        Ok(Self {
            name,
            dims,
            ggml_type,
            offset,
            absolute_offset: 0,
            n_elements,
            n_bytes,
        })
    }

    pub(crate) fn set_absolute_offset(&mut self, absolute: u64) {
        self.absolute_offset = absolute;
    }

    /// The tensor's name, for example `blk.0.attn_q.weight`.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The shape in ggml order: `dims()[0]` is the innermost (contiguous) dimension.
    pub fn dims(&self) -> &[u64] {
        &self.dims
    }

    /// The element type.
    pub fn ggml_type(&self) -> GgmlType {
        self.ggml_type
    }

    /// Data offset relative to the start of the data section, as stored in the file.
    pub fn offset(&self) -> u64 {
        self.offset
    }

    /// Data offset from the start of the file.
    pub fn absolute_offset(&self) -> u64 {
        self.absolute_offset
    }

    /// Number of elements (the product of the dimensions).
    pub fn n_elements(&self) -> u64 {
        self.n_elements
    }

    /// Bytes the tensor's data occupies.
    pub fn n_bytes(&self) -> u64 {
        self.n_bytes
    }

    /// Absolute byte range of the tensor's data. The end saturates at
    /// `u64::MAX` for adversarial offsets.
    pub fn data_range(&self) -> Range<u64> {
        self.absolute_offset..self.absolute_offset.saturating_add(self.n_bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn computes_elements_and_bytes() {
        let t = TensorInfo::new("w".into(), vec![4096, 11008], GgmlType::Q4_K, 0).unwrap();
        assert_eq!(t.n_elements(), 4096 * 11008);
        assert_eq!(t.n_bytes(), 4096 * 11008 / 256 * 144);
        assert_eq!(t.data_range(), 0..t.n_bytes());
    }

    #[test]
    fn scalar_and_empty_shapes() {
        let scalar = TensorInfo::new("s".into(), vec![], GgmlType::F32, 0).unwrap();
        assert_eq!((scalar.n_elements(), scalar.n_bytes()), (1, 4));
        let empty = TensorInfo::new("e".into(), vec![0, 7], GgmlType::Q8_0, 0).unwrap();
        assert_eq!((empty.n_elements(), empty.n_bytes()), (0, 0));
        assert!(matches!(
            TensorInfo::new("q".into(), vec![], GgmlType::Q8_0, 0),
            Err(ErrorKind::PartialBlock { ne0: 1, .. })
        ));
    }

    #[test]
    fn rejects_overflowing_shapes() {
        let check = |dims: Vec<u64>, ty: GgmlType| TensorInfo::new("x".into(), dims, ty, 0);
        let big = i64::MAX.unsigned_abs() + 1;
        for dims in [vec![u64::MAX], vec![big, 0], vec![0, big]] {
            assert!(matches!(
                check(dims, GgmlType::F32),
                Err(ErrorKind::DimensionOverflow(_))
            ));
        }
        for dims in [vec![1 << 32, 1 << 32], vec![1 << 31, 1 << 32, 0]] {
            assert!(matches!(
                check(dims, GgmlType::F32),
                Err(ErrorKind::ElementCountOverflow(_))
            ));
        }
        // A zero early on keeps the running product at 0, as in ggml.
        let empty = check(vec![0, 1 << 40, 1 << 40], GgmlType::F32).unwrap();
        assert_eq!(empty.n_elements(), 0);
        // Fits in i64 as elements but not as bytes.
        assert!(matches!(
            check(vec![1 << 62], GgmlType::F64),
            Err(ErrorKind::ByteSizeOverflow {
                ggml_type: GgmlType::F64,
                ..
            })
        ));
    }

    #[test]
    fn rejects_partial_blocks() {
        assert!(matches!(
            TensorInfo::new("x".into(), vec![100, 2], GgmlType::Q4_0, 0),
            Err(ErrorKind::PartialBlock { ne0: 100, .. })
        ));
    }
}
