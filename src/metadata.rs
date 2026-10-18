//! Metadata key-value pairs and their typed values.

use std::fmt;

use crate::name_index::NameIndex;

/// The type tag of a metadata value (`gguf_metadata_value_type` in the spec).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[non_exhaustive]
pub enum ValueType {
    /// `uint8`
    U8 = 0,
    /// `int8`
    I8 = 1,
    /// `uint16`
    U16 = 2,
    /// `int16`
    I16 = 3,
    /// `uint32`
    U32 = 4,
    /// `int32`
    I32 = 5,
    /// `float32`
    F32 = 6,
    /// `bool`, one byte that must be 0 or 1
    Bool = 7,
    /// length-prefixed UTF-8 string
    String = 8,
    /// typed, length-prefixed array; elements may themselves be arrays
    Array = 9,
    /// `uint64`
    U64 = 10,
    /// `int64`
    I64 = 11,
    /// `float64`
    F64 = 12,
}

impl ValueType {
    /// Every value type, in ID order.
    pub const ALL: [ValueType; 13] = [
        Self::U8,
        Self::I8,
        Self::U16,
        Self::I16,
        Self::U32,
        Self::I32,
        Self::F32,
        Self::Bool,
        Self::String,
        Self::Array,
        Self::U64,
        Self::I64,
        Self::F64,
    ];

    /// Looks up a type by its numeric tag.
    pub const fn from_id(id: u32) -> Option<Self> {
        Some(match id {
            0 => Self::U8,
            1 => Self::I8,
            2 => Self::U16,
            3 => Self::I16,
            4 => Self::U32,
            5 => Self::I32,
            6 => Self::F32,
            7 => Self::Bool,
            8 => Self::String,
            9 => Self::Array,
            10 => Self::U64,
            11 => Self::I64,
            12 => Self::F64,
            _ => return None,
        })
    }

    /// The numeric tag stored in the file.
    pub const fn id(self) -> u32 {
        self as u32
    }

    /// Short lower-case name, for example `"u32"` or `"string"`.
    pub const fn name(self) -> &'static str {
        match self {
            Self::U8 => "u8",
            Self::I8 => "i8",
            Self::U16 => "u16",
            Self::I16 => "i16",
            Self::U32 => "u32",
            Self::I32 => "i32",
            Self::F32 => "f32",
            Self::Bool => "bool",
            Self::String => "string",
            Self::Array => "array",
            Self::U64 => "u64",
            Self::I64 => "i64",
            Self::F64 => "f64",
        }
    }

    /// Whether this is one of the eight integer types.
    pub const fn is_integer(self) -> bool {
        matches!(
            self,
            Self::U8
                | Self::I8
                | Self::U16
                | Self::I16
                | Self::U32
                | Self::I32
                | Self::U64
                | Self::I64
        )
    }

    /// The fewest bytes a value of this type can occupy in a file: its fixed
    /// width, or the size of its length prefix (plus element type tag for arrays).
    pub(crate) const fn min_encoded_size(self) -> u64 {
        match self {
            Self::U8 | Self::I8 | Self::Bool => 1,
            Self::U16 | Self::I16 => 2,
            Self::U32 | Self::I32 | Self::F32 => 4,
            Self::U64 | Self::I64 | Self::F64 | Self::String => 8,
            Self::Array => 12,
        }
    }
}

impl fmt::Display for ValueType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// A single metadata value.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum MetadataValue {
    /// `uint8`
    U8(u8),
    /// `int8`
    I8(i8),
    /// `uint16`
    U16(u16),
    /// `int16`
    I16(i16),
    /// `uint32`
    U32(u32),
    /// `int32`
    I32(i32),
    /// `float32`
    F32(f32),
    /// `bool`
    Bool(bool),
    /// UTF-8 string
    String(String),
    /// Homogeneous array
    Array(MetadataArray),
    /// `uint64`
    U64(u64),
    /// `int64`
    I64(i64),
    /// `float64`
    F64(f64),
}

impl MetadataValue {
    /// The value's type tag.
    pub const fn value_type(&self) -> ValueType {
        match self {
            Self::U8(_) => ValueType::U8,
            Self::I8(_) => ValueType::I8,
            Self::U16(_) => ValueType::U16,
            Self::I16(_) => ValueType::I16,
            Self::U32(_) => ValueType::U32,
            Self::I32(_) => ValueType::I32,
            Self::F32(_) => ValueType::F32,
            Self::Bool(_) => ValueType::Bool,
            Self::String(_) => ValueType::String,
            Self::Array(_) => ValueType::Array,
            Self::U64(_) => ValueType::U64,
            Self::I64(_) => ValueType::I64,
            Self::F64(_) => ValueType::F64,
        }
    }

    /// The value as `u64` if it is an integer of any width that is not negative.
    ///
    /// The spec recommends `uint64` for counts but notes that some writers use
    /// `uint32`, so readers should accept any integer type.
    pub fn as_u64(&self) -> Option<u64> {
        match *self {
            Self::U8(v) => Some(v.into()),
            Self::U16(v) => Some(v.into()),
            Self::U32(v) => Some(v.into()),
            Self::U64(v) => Some(v),
            Self::I8(v) => u64::try_from(v).ok(),
            Self::I16(v) => u64::try_from(v).ok(),
            Self::I32(v) => u64::try_from(v).ok(),
            Self::I64(v) => u64::try_from(v).ok(),
            _ => None,
        }
    }

    /// The value as `i64` if it is an integer that fits.
    pub fn as_i64(&self) -> Option<i64> {
        match *self {
            Self::U8(v) => Some(v.into()),
            Self::U16(v) => Some(v.into()),
            Self::U32(v) => Some(v.into()),
            Self::U64(v) => i64::try_from(v).ok(),
            Self::I8(v) => Some(v.into()),
            Self::I16(v) => Some(v.into()),
            Self::I32(v) => Some(v.into()),
            Self::I64(v) => Some(v),
            _ => None,
        }
    }

    /// The value as `f64` if it is a float or an integer (64-bit integers may
    /// lose precision).
    #[allow(clippy::cast_precision_loss)]
    pub fn as_f64(&self) -> Option<f64> {
        match *self {
            Self::F32(v) => Some(v.into()),
            Self::F64(v) => Some(v),
            Self::U64(v) => Some(v as f64),
            Self::I64(v) => Some(v as f64),
            _ => self.as_i64().map(|v| v as f64),
        }
    }

    /// The value as a string slice if it is a string.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(s) => Some(s),
            _ => None,
        }
    }

    /// The value if it is a bool.
    pub const fn as_bool(&self) -> Option<bool> {
        match *self {
            Self::Bool(b) => Some(b),
            _ => None,
        }
    }

    /// The value if it is an array.
    pub const fn as_array(&self) -> Option<&MetadataArray> {
        match self {
            Self::Array(a) => Some(a),
            _ => None,
        }
    }

    /// Exact equality: same type and value, with floats compared by bit
    /// pattern. Unlike `==`, a `NaN` equals itself and `0.0` differs from `-0.0`.
    ///
    /// ```
    /// use gguf_inspect::MetadataValue;
    ///
    /// let nan = MetadataValue::F32(f32::NAN);
    /// assert!(nan != nan.clone());
    /// assert!(nan.bitwise_eq(&nan));
    /// assert!(!MetadataValue::F64(0.0).bitwise_eq(&MetadataValue::F64(-0.0)));
    /// ```
    pub fn bitwise_eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::F32(x), Self::F32(y)) => x.to_bits() == y.to_bits(),
            (Self::F64(x), Self::F64(y)) => x.to_bits() == y.to_bits(),
            (Self::Array(x), Self::Array(y)) => x.bitwise_eq(y),
            _ => self == other,
        }
    }
}

/// A homogeneous metadata array.
///
/// Elements are stored in typed vectors, so a 150 000-entry vocabulary costs
/// one `String` per token rather than one enum per token, and empty arrays keep
/// their element type. Nested arrays (allowed by the spec) are
/// [`MetadataArray::Array`]; each inner array has its own element type.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
#[allow(missing_docs)] // variants mirror ValueType one to one
pub enum MetadataArray {
    U8(Vec<u8>),
    I8(Vec<i8>),
    U16(Vec<u16>),
    I16(Vec<i16>),
    U32(Vec<u32>),
    I32(Vec<i32>),
    F32(Vec<f32>),
    Bool(Vec<bool>),
    String(Vec<String>),
    Array(Vec<MetadataArray>),
    U64(Vec<u64>),
    I64(Vec<i64>),
    F64(Vec<f64>),
}

/// Applies `$body` to the vector inside any [`MetadataArray`] variant.
macro_rules! with_vec {
    ($array:expr, $v:ident => $body:expr) => {
        match $array {
            MetadataArray::U8($v) => $body,
            MetadataArray::I8($v) => $body,
            MetadataArray::U16($v) => $body,
            MetadataArray::I16($v) => $body,
            MetadataArray::U32($v) => $body,
            MetadataArray::I32($v) => $body,
            MetadataArray::F32($v) => $body,
            MetadataArray::Bool($v) => $body,
            MetadataArray::String($v) => $body,
            MetadataArray::Array($v) => $body,
            MetadataArray::U64($v) => $body,
            MetadataArray::I64($v) => $body,
            MetadataArray::F64($v) => $body,
        }
    };
}

impl MetadataArray {
    /// The type of every element.
    pub const fn element_type(&self) -> ValueType {
        match self {
            Self::U8(_) => ValueType::U8,
            Self::I8(_) => ValueType::I8,
            Self::U16(_) => ValueType::U16,
            Self::I16(_) => ValueType::I16,
            Self::U32(_) => ValueType::U32,
            Self::I32(_) => ValueType::I32,
            Self::F32(_) => ValueType::F32,
            Self::Bool(_) => ValueType::Bool,
            Self::String(_) => ValueType::String,
            Self::Array(_) => ValueType::Array,
            Self::U64(_) => ValueType::U64,
            Self::I64(_) => ValueType::I64,
            Self::F64(_) => ValueType::F64,
        }
    }

    /// Number of elements.
    pub fn len(&self) -> usize {
        with_vec!(self, v => v.len())
    }

    /// Whether the array has no elements.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// A copy of the element at `index`.
    ///
    /// Strings and nested arrays are cloned. For borrowed access, match on
    /// the variant (`MetadataArray::String(tokens)` gives `&Vec<String>`) or
    /// use [`as_strings`](Self::as_strings).
    pub fn get(&self, index: usize) -> Option<MetadataValue> {
        Some(match self {
            Self::U8(v) => MetadataValue::U8(*v.get(index)?),
            Self::I8(v) => MetadataValue::I8(*v.get(index)?),
            Self::U16(v) => MetadataValue::U16(*v.get(index)?),
            Self::I16(v) => MetadataValue::I16(*v.get(index)?),
            Self::U32(v) => MetadataValue::U32(*v.get(index)?),
            Self::I32(v) => MetadataValue::I32(*v.get(index)?),
            Self::F32(v) => MetadataValue::F32(*v.get(index)?),
            Self::Bool(v) => MetadataValue::Bool(*v.get(index)?),
            Self::String(v) => MetadataValue::String(v.get(index)?.clone()),
            Self::Array(v) => MetadataValue::Array(v.get(index)?.clone()),
            Self::U64(v) => MetadataValue::U64(*v.get(index)?),
            Self::I64(v) => MetadataValue::I64(*v.get(index)?),
            Self::F64(v) => MetadataValue::F64(*v.get(index)?),
        })
    }

    /// Iterates over copies of the elements; see [`get`](Self::get) for
    /// borrowed access.
    pub fn iter(&self) -> impl Iterator<Item = MetadataValue> + '_ {
        (0..self.len()).map_while(|i| self.get(i))
    }

    /// The elements as `u64` if the array holds integers that are all non-negative.
    pub fn to_u64_vec(&self) -> Option<Vec<u64>> {
        fn conv<T: Copy>(v: &[T], f: impl Fn(T) -> Option<u64>) -> Option<Vec<u64>> {
            v.iter().map(|&x| f(x)).collect()
        }
        match self {
            Self::U8(v) => conv(v, |x| Some(x.into())),
            Self::U16(v) => conv(v, |x| Some(x.into())),
            Self::U32(v) => conv(v, |x| Some(x.into())),
            Self::U64(v) => Some(v.clone()),
            Self::I8(v) => conv(v, |x| u64::try_from(x).ok()),
            Self::I16(v) => conv(v, |x| u64::try_from(x).ok()),
            Self::I32(v) => conv(v, |x| u64::try_from(x).ok()),
            Self::I64(v) => conv(v, |x| u64::try_from(x).ok()),
            _ => None,
        }
    }

    /// Exact equality with floats compared by bit pattern; see
    /// [`MetadataValue::bitwise_eq`].
    pub fn bitwise_eq(&self, other: &Self) -> bool {
        self.first_difference(other).is_none()
    }

    /// The index of the first element where the arrays differ, compared as
    /// by [`bitwise_eq`](Self::bitwise_eq), or the length of the shorter
    /// array if it is a prefix of the longer one. `None` if they are equal.
    /// Arrays of different element types differ at index 0, even when empty.
    /// Nothing is copied.
    ///
    /// ```
    /// use gguf_inspect::MetadataArray;
    ///
    /// let a = MetadataArray::U32(vec![1, 2, 3]);
    /// assert_eq!(a.first_difference(&MetadataArray::U32(vec![1, 9, 3])), Some(1));
    /// assert_eq!(a.first_difference(&MetadataArray::U32(vec![1, 2])), Some(2));
    /// assert_eq!(a.first_difference(&a.clone()), None);
    /// ```
    pub fn first_difference(&self, other: &Self) -> Option<usize> {
        fn first<T>(a: &[T], b: &[T], eq: impl Fn(&T, &T) -> bool) -> Option<usize> {
            a.iter()
                .zip(b)
                .position(|(x, y)| !eq(x, y))
                .or_else(|| (a.len() != b.len()).then(|| a.len().min(b.len())))
        }
        match (self, other) {
            (Self::U8(a), Self::U8(b)) => first(a, b, PartialEq::eq),
            (Self::I8(a), Self::I8(b)) => first(a, b, PartialEq::eq),
            (Self::U16(a), Self::U16(b)) => first(a, b, PartialEq::eq),
            (Self::I16(a), Self::I16(b)) => first(a, b, PartialEq::eq),
            (Self::U32(a), Self::U32(b)) => first(a, b, PartialEq::eq),
            (Self::I32(a), Self::I32(b)) => first(a, b, PartialEq::eq),
            (Self::U64(a), Self::U64(b)) => first(a, b, PartialEq::eq),
            (Self::I64(a), Self::I64(b)) => first(a, b, PartialEq::eq),
            (Self::Bool(a), Self::Bool(b)) => first(a, b, PartialEq::eq),
            (Self::String(a), Self::String(b)) => first(a, b, PartialEq::eq),
            (Self::F32(a), Self::F32(b)) => first(a, b, |x, y| x.to_bits() == y.to_bits()),
            (Self::F64(a), Self::F64(b)) => first(a, b, |x, y| x.to_bits() == y.to_bits()),
            (Self::Array(a), Self::Array(b)) => first(a, b, Self::bitwise_eq),
            _ => Some(0),
        }
    }

    /// The elements if this is an array of strings.
    pub fn as_strings(&self) -> Option<&[String]> {
        match self {
            Self::String(v) => Some(v),
            _ => None,
        }
    }
}

/// The metadata section of a GGUF file: key-value pairs in file order, with
/// lookup by key in `O(log n)`. Keys are unique (duplicates are rejected when
/// parsing) and stored once.
#[derive(Debug, Clone, Default)]
pub struct Metadata {
    entries: Vec<Entry>,
    index: NameIndex,
}

#[derive(Debug, Clone, PartialEq)]
struct Entry {
    key: String,
    value: MetadataValue,
    offset: u64,
}

/// The first entry, in file order, whose key repeats an earlier one.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct DuplicateKey {
    pub(crate) position: usize,
    pub(crate) key: String,
    pub(crate) offset: u64,
}

impl Metadata {
    /// Builds the metadata from entries in file order, each with the byte
    /// offset where it starts.
    pub(crate) fn from_entries(
        entries: impl IntoIterator<Item = (String, MetadataValue, u64)>,
    ) -> Result<Self, DuplicateKey> {
        let mut entries: Vec<Entry> = entries
            .into_iter()
            .map(|(key, value, offset)| Entry { key, value, offset })
            .collect();
        match NameIndex::build(entries.len(), |i| &entries[i].key) {
            Ok(index) => Ok(Self { entries, index }),
            Err(position) => {
                let Entry { key, offset, .. } = entries.swap_remove(position);
                Err(DuplicateKey {
                    position,
                    key,
                    offset,
                })
            }
        }
    }

    /// Appends an entry that starts at byte `offset`. Returns `false`, leaving
    /// the metadata unchanged, if the key already exists.
    #[cfg(test)]
    pub(crate) fn insert(&mut self, key: String, value: MetadataValue, offset: u64) -> bool {
        if self.contains_key(&key) {
            return false;
        }
        self.entries.push(Entry { key, value, offset });
        let entries = &self.entries;
        self.index.insert(entries.len() - 1, |i| &entries[i].key)
    }

    fn position(&self, key: &str) -> Option<usize> {
        self.index.find(key, |i| &self.entries[i].key)
    }

    /// Number of entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether there are no entries.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// The value stored under `key`.
    pub fn get(&self, key: &str) -> Option<&MetadataValue> {
        self.position(key).map(|i| &self.entries[i].value)
    }

    /// Whether `key` is present.
    pub fn contains_key(&self, key: &str) -> bool {
        self.position(key).is_some()
    }

    /// The string stored under `key`, if it is a string.
    pub fn get_str(&self, key: &str) -> Option<&str> {
        self.get(key)?.as_str()
    }

    /// The non-negative integer stored under `key`, whatever its width.
    pub fn get_u64(&self, key: &str) -> Option<u64> {
        self.get(key)?.as_u64()
    }

    /// The number stored under `key`, converted to `f64`.
    pub fn get_f64(&self, key: &str) -> Option<f64> {
        self.get(key)?.as_f64()
    }

    /// Byte offset of the start of the entry for `key` (its key string).
    pub fn offset_of(&self, key: &str) -> Option<u64> {
        self.position(key).map(|i| self.entries[i].offset)
    }

    /// Iterates over entries in file order.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = (&str, &MetadataValue)> {
        self.entries.iter().map(|e| (e.key.as_str(), &e.value))
    }
}

/// Two metadata sections are equal if they hold the same entries in the same
/// order; byte offsets are ignored.
impl PartialEq for Metadata {
    fn eq(&self, other: &Self) -> bool {
        self.iter().eq(other.iter())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn in_memory_sizes_stay_small() {
        // Each array element costs at most this many bytes of `Vec` storage,
        // which bounds memory use relative to the bytes read from the file.
        assert!(std::mem::size_of::<MetadataArray>() <= 32);
        assert!(std::mem::size_of::<MetadataValue>() <= 32);
        assert_eq!(
            std::mem::size_of::<String>(),
            3 * std::mem::size_of::<usize>()
        );
    }

    fn one_of_each_array() -> Vec<MetadataArray> {
        vec![
            MetadataArray::U8(vec![1]),
            MetadataArray::I8(vec![-1]),
            MetadataArray::U16(vec![1]),
            MetadataArray::I16(vec![-1]),
            MetadataArray::U32(vec![1]),
            MetadataArray::I32(vec![-1]),
            MetadataArray::F32(vec![0.5]),
            MetadataArray::Bool(vec![true]),
            MetadataArray::String(vec!["s".into()]),
            MetadataArray::Array(vec![MetadataArray::U8(vec![])]),
            MetadataArray::U64(vec![1]),
            MetadataArray::I64(vec![-1]),
            MetadataArray::F64(vec![0.5]),
        ]
    }

    #[test]
    fn element_types_and_values_agree_for_every_variant() {
        let arrays = one_of_each_array();
        let types: Vec<ValueType> = arrays.iter().map(MetadataArray::element_type).collect();
        assert_eq!(types, ValueType::ALL);
        for array in &arrays {
            let first = array.get(0).unwrap();
            assert_eq!(first.value_type(), array.element_type());
            assert_eq!(array.get(1), None);
            assert!(array.bitwise_eq(array));
            let value = MetadataValue::Array(array.clone());
            assert_eq!(value.value_type(), ValueType::Array);
            assert_eq!(value.as_array(), Some(array));
            assert_eq!(value.as_str(), None);
            assert_eq!(value.as_bool(), None);
            assert_eq!(value.as_u64(), None);
        }
        assert_eq!(
            arrays[8].as_strings().map(<[String]>::len),
            Some(1),
            "string array"
        );
        assert_eq!(arrays[0].as_strings(), None);
        let names: Vec<&str> = ValueType::ALL.iter().map(|t| t.name()).collect();
        assert_eq!(
            names.join(" "),
            "u8 i8 u16 i16 u32 i32 f32 bool string array u64 i64 f64"
        );
    }

    #[test]
    fn value_type_ids_round_trip() {
        for t in ValueType::ALL {
            assert_eq!(ValueType::from_id(t.id()), Some(t));
        }
        let integers: Vec<&str> = ValueType::ALL
            .iter()
            .filter(|t| t.is_integer())
            .map(|t| t.name())
            .collect();
        assert_eq!(
            integers,
            ["u8", "i8", "u16", "i16", "u32", "i32", "u64", "i64"]
        );
        assert_eq!(ValueType::from_id(13), None);
    }

    #[test]
    fn integer_accessors_accept_every_width() {
        assert_eq!(MetadataValue::U8(7).as_u64(), Some(7));
        assert_eq!(MetadataValue::I32(7).as_u64(), Some(7));
        assert_eq!(MetadataValue::I64(-1).as_u64(), None);
        assert_eq!(MetadataValue::U64(u64::MAX).as_i64(), None);
        assert_eq!(MetadataValue::I16(-3).as_i64(), Some(-3));
        assert_eq!(MetadataValue::F32(1.5).as_u64(), None);
        assert_eq!(MetadataValue::String("x".into()).as_u64(), None);
    }

    #[test]
    fn float_accessor_converts_integers() {
        assert_eq!(MetadataValue::F32(0.5).as_f64(), Some(0.5));
        assert_eq!(MetadataValue::U32(10_000).as_f64(), Some(10_000.0));
        assert_eq!(MetadataValue::Bool(true).as_f64(), None);
    }

    #[test]
    fn array_helpers() {
        let a = MetadataArray::I32(vec![1, 2, 3]);
        assert_eq!(a.len(), 3);
        assert_eq!(a.element_type(), ValueType::I32);
        assert_eq!(a.get(1), Some(MetadataValue::I32(2)));
        assert_eq!(a.get(3), None);
        assert_eq!(a.to_u64_vec(), Some(vec![1, 2, 3]));
        assert_eq!(MetadataArray::I8(vec![-1]).to_u64_vec(), None);
        assert_eq!(a.iter().count(), 3);

        let nested = MetadataArray::Array(vec![MetadataArray::String(vec![])]);
        assert_eq!(
            nested.get(0),
            Some(MetadataValue::Array(MetadataArray::String(vec![])))
        );
        assert!(MetadataArray::F64(vec![]).is_empty());
    }

    #[test]
    fn first_difference_of_every_variant() {
        for array in one_of_each_array() {
            assert_eq!(array.first_difference(&array.clone()), None);
            // A different value or a different element type: index 0 either way.
            let other = MetadataArray::F64(vec![9.0]);
            assert_eq!(array.first_difference(&other), Some(0));
        }
        let nan = |bits: u32| MetadataArray::F32(vec![1.0, f32::from_bits(bits)]);
        assert_eq!(nan(0x7fc0_0000).first_difference(&nan(0x7fc0_0000)), None);
        assert_eq!(
            nan(0x7fc0_0000).first_difference(&nan(0x7fc0_0001)),
            Some(1)
        );
        let strings =
            |v: &[&str]| MetadataArray::String(v.iter().map(|s| (*s).to_owned()).collect());
        assert_eq!(
            strings(&["a", "b"]).first_difference(&strings(&["a", "c"])),
            Some(1)
        );
        assert_eq!(
            strings(&["a"]).first_difference(&strings(&["a", "b"])),
            Some(1)
        );
        assert_eq!(
            MetadataArray::U8(vec![]).first_difference(&MetadataArray::I8(vec![])),
            Some(0),
            "element types differ"
        );
    }

    #[test]
    fn metadata_preserves_order_and_rejects_duplicates() {
        let mut m = Metadata::default();
        assert!(m.insert("b".into(), MetadataValue::U8(1), 24));
        assert!(m.insert("a".into(), MetadataValue::String("x".into()), 40));
        assert!(!m.insert("b".into(), MetadataValue::U8(2), 60));
        assert_eq!(m.len(), 2);
        assert_eq!(m.iter().map(|(k, _)| k).collect::<Vec<_>>(), ["b", "a"]);
        assert_eq!(m.get_u64("b"), Some(1));
        assert_eq!(m.get_str("a"), Some("x"));
        assert_eq!(m.offset_of("a"), Some(40));
        assert!(!m.contains_key("c"));
    }

    #[test]
    fn from_entries_reports_the_first_repeated_key() {
        let entry = |k: &str, offset| (k.to_owned(), MetadataValue::U8(0), offset);
        let m = Metadata::from_entries([entry("z", 24), entry("a", 37)]).unwrap();
        assert_eq!(m.iter().map(|(k, _)| k).collect::<Vec<_>>(), ["z", "a"]);
        assert_eq!(m.offset_of("z"), Some(24));
        let err = Metadata::from_entries([entry("a", 0), entry("b", 1), entry("a", 2)]);
        assert_eq!(
            err.unwrap_err(),
            DuplicateKey {
                position: 2,
                key: "a".into(),
                offset: 2
            }
        );
    }
}
