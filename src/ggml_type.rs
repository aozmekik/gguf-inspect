//! The ggml tensor element types and their storage layout.
//!
//! Type IDs follow `enum ggml_type` in `include/ggml.h`; names, block sizes and
//! block byte sizes follow the `type_traits` table in `src/ggml.c` together with
//! the block structs in `src/ggml-common.h`. All three were checked against
//! [ggml commit `a05c992`](https://github.com/ggml-org/ggml/tree/a05c9922613c6a2230dbb1d0de679519ecafb4f0)
//! (2026-10-15); `scripts/check-ggml-type-sizes.sh` repeats the check by
//! compiling the ggml header and comparing `sizeof` of every block struct.

use std::fmt;
use std::str::FromStr;

/// Storage layout of one [`GgmlType`]: values are packed in blocks of
/// `block_size` elements, each block occupying `type_size` bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Traits {
    name: &'static str,
    block_size: u32,
    type_size: u32,
    quantized: bool,
}

const fn plain(name: &'static str, type_size: u32) -> Traits {
    Traits {
        name,
        block_size: 1,
        type_size,
        quantized: false,
    }
}

const fn quant(name: &'static str, block_size: u32, type_size: u32) -> Traits {
    Traits {
        name,
        block_size,
        type_size,
        quantized: true,
    }
}

/// Generates the enum, its ID conversion and its trait lookup from one table so
/// the three can never drift apart.
macro_rules! ggml_types {
    ($( $(#[$doc:meta])* $variant:ident = $id:literal => $traits:expr, )+) => {
        /// A ggml tensor element type, as stored in a GGUF tensor info record.
        ///
        /// IDs that ggml has removed (4, 5, 31-33 and 36-38) and IDs newer than this
        /// crate are rejected when parsing, because their storage size is unknown.
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
        #[non_exhaustive]
        #[allow(non_camel_case_types)]
        pub enum GgmlType {
            $( $(#[$doc])* $variant = $id, )+
        }

        impl GgmlType {
            /// Every type this crate knows, in ascending ID order.
            pub const ALL: &'static [GgmlType] = &[$(GgmlType::$variant),+];

            /// Looks up a type by its numeric ID.
            ///
            /// Returns `None` for IDs that ggml removed or that this crate does not know.
            pub const fn from_id(id: u32) -> Option<Self> {
                match id {
                    $( $id => Some(GgmlType::$variant), )+
                    _ => None,
                }
            }

            const fn traits(self) -> Traits {
                match self {
                    $( GgmlType::$variant => $traits, )+
                }
            }
        }
    };
}

ggml_types! {
    /// 32-bit IEEE 754 float.
    F32 = 0 => plain("f32", 4),
    /// 16-bit IEEE 754 half float.
    F16 = 1 => plain("f16", 2),
    /// 4-bit, 32-element blocks with one f16 scale.
    Q4_0 = 2 => quant("q4_0", 32, 18),
    /// 4-bit, 32-element blocks with f16 scale and minimum.
    Q4_1 = 3 => quant("q4_1", 32, 20),
    /// 5-bit, 32-element blocks with one f16 scale.
    Q5_0 = 6 => quant("q5_0", 32, 22),
    /// 5-bit, 32-element blocks with f16 scale and minimum.
    Q5_1 = 7 => quant("q5_1", 32, 24),
    /// 8-bit, 32-element blocks with one f16 scale.
    Q8_0 = 8 => quant("q8_0", 32, 34),
    /// 8-bit, 32-element blocks with f16 scale and sum.
    Q8_1 = 9 => quant("q8_1", 32, 36),
    /// 2-bit k-quant, 256-element super-blocks.
    Q2_K = 10 => quant("q2_K", 256, 84),
    /// 3-bit k-quant, 256-element super-blocks.
    Q3_K = 11 => quant("q3_K", 256, 110),
    /// 4-bit k-quant, 256-element super-blocks.
    Q4_K = 12 => quant("q4_K", 256, 144),
    /// 5-bit k-quant, 256-element super-blocks.
    Q5_K = 13 => quant("q5_K", 256, 176),
    /// 6-bit k-quant, 256-element super-blocks.
    Q6_K = 14 => quant("q6_K", 256, 210),
    /// 8-bit k-quant (used for intermediate results), 256-element super-blocks.
    Q8_K = 15 => quant("q8_K", 256, 292),
    /// 2.06 bits per weight i-quant.
    IQ2_XXS = 16 => quant("iq2_xxs", 256, 66),
    /// 2.31 bits per weight i-quant.
    IQ2_XS = 17 => quant("iq2_xs", 256, 74),
    /// 3.06 bits per weight i-quant.
    IQ3_XXS = 18 => quant("iq3_xxs", 256, 98),
    /// 1.56 bits per weight i-quant.
    IQ1_S = 19 => quant("iq1_s", 256, 50),
    /// 4.5 bits per weight non-linear quant, 32-element blocks.
    IQ4_NL = 20 => quant("iq4_nl", 32, 18),
    /// 3.44 bits per weight i-quant.
    IQ3_S = 21 => quant("iq3_s", 256, 110),
    /// 2.56 bits per weight i-quant.
    IQ2_S = 22 => quant("iq2_s", 256, 82),
    /// 4.25 bits per weight non-linear quant, 256-element super-blocks.
    IQ4_XS = 23 => quant("iq4_xs", 256, 136),
    /// 8-bit signed integer.
    I8 = 24 => plain("i8", 1),
    /// 16-bit signed integer.
    I16 = 25 => plain("i16", 2),
    /// 32-bit signed integer.
    I32 = 26 => plain("i32", 4),
    /// 64-bit signed integer.
    I64 = 27 => plain("i64", 8),
    /// 64-bit IEEE 754 float.
    F64 = 28 => plain("f64", 8),
    /// 1.75 bits per weight i-quant.
    IQ1_M = 29 => quant("iq1_m", 256, 56),
    /// bfloat16.
    BF16 = 30 => plain("bf16", 2),
    /// Ternary quant, 1.69 bits per weight.
    TQ1_0 = 34 => quant("tq1_0", 256, 54),
    /// Ternary quant, 2.06 bits per weight.
    TQ2_0 = 35 => quant("tq2_0", 256, 66),
    /// OCP microscaling FP4, 32-element blocks with an E8M0 scale.
    MXFP4 = 39 => quant("mxfp4", 32, 17),
    /// NVIDIA FP4, 64-element blocks with four E4M3 scales.
    NVFP4 = 40 => quant("nvfp4", 64, 36),
    /// 1-bit, 128-element blocks with one f16 scale.
    Q1_0 = 41 => quant("q1_0", 128, 18),
    /// 2-bit, 64-element blocks with one f16 scale.
    Q2_0 = 42 => quant("q2_0", 64, 18),
}

/// Type IDs that existed in ggml but whose support has been removed.
const REMOVED_IDS: [u32; 8] = [4, 5, 31, 32, 33, 36, 37, 38];

impl GgmlType {
    /// The numeric ID stored in GGUF files.
    pub const fn id(self) -> u32 {
        self as u32
    }

    /// The canonical lower-case name used by ggml, for example `"q4_K"`.
    pub const fn name(self) -> &'static str {
        self.traits().name
    }

    /// Number of elements in one block (1 for unquantized types).
    pub const fn block_size(self) -> u64 {
        self.traits().block_size as u64
    }

    /// Number of bytes one block occupies.
    pub const fn type_size(self) -> u64 {
        self.traits().type_size as u64
    }

    /// Whether ggml marks this type as quantized.
    pub const fn is_quantized(self) -> bool {
        self.traits().quantized
    }

    /// Bytes needed to store a row of `elements` values, or `None` if `elements`
    /// is not a multiple of [`block_size`](Self::block_size) or the size overflows.
    ///
    /// ```
    /// use gguf_inspect::GgmlType;
    ///
    /// assert_eq!(GgmlType::F16.row_size(4096), Some(8192));
    /// assert_eq!(GgmlType::Q8_0.row_size(4096), Some(4096 / 32 * 34));
    /// assert_eq!(GgmlType::Q4_K.row_size(100), None); // not a whole number of blocks
    /// ```
    pub const fn row_size(self, elements: u64) -> Option<u64> {
        let block = self.block_size();
        if !elements.is_multiple_of(block) {
            return None;
        }
        (elements / block).checked_mul(self.type_size())
    }

    /// Average storage cost in bits per element.
    #[allow(clippy::cast_precision_loss)] // both operands are tiny constants
    pub fn bits_per_element(self) -> f64 {
        (self.type_size() * 8) as f64 / self.block_size() as f64
    }

    /// Whether `id` names a type ggml used to have but removed.
    pub(crate) fn is_removed_id(id: u32) -> bool {
        REMOVED_IDS.contains(&id)
    }
}

impl fmt::Display for GgmlType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// Error returned when parsing an unknown ggml type name.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseGgmlTypeError(String);

impl fmt::Display for ParseGgmlTypeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "unknown ggml type `{}`", self.0)
    }
}

impl std::error::Error for ParseGgmlTypeError {}

impl FromStr for GgmlType {
    type Err = ParseGgmlTypeError;

    /// Parses a type name case-insensitively, so `"Q4_K"`, `"q4_k"` and `"q4_K"`
    /// all name [`GgmlType::Q4_K`].
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::ALL
            .iter()
            .copied()
            .find(|t| t.name().eq_ignore_ascii_case(s))
            .ok_or_else(|| ParseGgmlTypeError(s.to_owned()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ids_round_trip_and_are_sorted() {
        for pair in GgmlType::ALL.windows(2) {
            assert!(pair[0].id() < pair[1].id());
        }
        for &t in GgmlType::ALL {
            assert_eq!(GgmlType::from_id(t.id()), Some(t));
        }
    }

    #[test]
    fn removed_and_unknown_ids_are_rejected() {
        for id in REMOVED_IDS {
            assert_eq!(GgmlType::from_id(id), None);
            assert!(GgmlType::is_removed_id(id));
        }
        assert_eq!(GgmlType::from_id(43), None);
        assert_eq!(GgmlType::from_id(u32::MAX), None);
    }

    #[test]
    fn names_parse_case_insensitively() {
        for &t in GgmlType::ALL {
            assert_eq!(t.name().parse::<GgmlType>(), Ok(t));
            assert_eq!(t.name().to_uppercase().parse::<GgmlType>(), Ok(t));
        }
        assert!("q4_2".parse::<GgmlType>().is_err());
    }

    #[test]
    fn layout_matches_ggml_reference_values() {
        // Spot checks against values printed by scripts/check-ggml-type-sizes.sh.
        let expect = [
            (GgmlType::F32, 1, 4),
            (GgmlType::BF16, 1, 2),
            (GgmlType::Q4_0, 32, 18),
            (GgmlType::Q8_0, 32, 34),
            (GgmlType::Q4_K, 256, 144),
            (GgmlType::Q6_K, 256, 210),
            (GgmlType::IQ4_XS, 256, 136),
            (GgmlType::TQ1_0, 256, 54),
            (GgmlType::MXFP4, 32, 17),
            (GgmlType::Q1_0, 128, 18),
        ];
        for (t, block, size) in expect {
            assert_eq!((t.block_size(), t.type_size()), (block, size), "{t}");
        }
    }

    #[test]
    fn quantized_flag_matches_block_size() {
        for &t in GgmlType::ALL {
            assert_eq!(t.is_quantized(), t.block_size() > 1, "{t}");
        }
    }

    #[test]
    fn row_size_rejects_partial_blocks_and_overflow() {
        assert_eq!(GgmlType::Q4_0.row_size(64), Some(36));
        assert_eq!(GgmlType::Q4_0.row_size(33), None);
        assert_eq!(GgmlType::F64.row_size(u64::MAX), None);
        assert_eq!(GgmlType::F32.row_size(0), Some(0));
    }

    #[test]
    fn bits_per_element() {
        assert!((GgmlType::Q4_K.bits_per_element() - 4.5).abs() < 1e-12);
        assert!((GgmlType::F16.bits_per_element() - 16.0).abs() < 1e-12);
    }
}
