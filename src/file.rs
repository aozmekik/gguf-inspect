//! The parsed header of a GGUF file and the parser that produces it.

use std::collections::{BTreeMap, HashMap};
use std::fmt;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

use crate::error::{Error, ErrorKind, Result};
use crate::metadata::{Metadata, MetadataArray, MetadataValue, ValueType};
use crate::reader::Reader;
use crate::{GgmlType, Limits, TensorInfo};

/// The magic bytes every GGUF file starts with.
pub const MAGIC: [u8; 4] = *b"GGUF";

/// Metadata key that sets the data alignment.
pub const ALIGNMENT_KEY: &str = "general.alignment";

/// Alignment assumed when `general.alignment` is absent.
pub const DEFAULT_ALIGNMENT: u32 = 32;

/// Longest metadata key the spec allows, in bytes.
pub const MAX_KEY_LEN: u64 = 65_535;

/// Longest tensor name the spec allows, in bytes.
pub const MAX_TENSOR_NAME_LEN: u64 = 64;

/// Smallest possible metadata entry: empty-key length, type tag, one-byte value.
const MIN_KV_BYTES: u64 = 8 + 4 + 1;
/// Smallest possible tensor info: name length, `n_dims`, type, offset.
const MIN_TENSOR_INFO_BYTES: u64 = 8 + 4 + 4 + 8;

/// Read buffer for [`GgufFile::open`]. Headers of real models are a few
/// megabytes at most, mostly the tokenizer vocabulary.
const READ_BUFFER_BYTES: usize = 256 * 1024;

/// A supported GGUF format version.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[non_exhaustive]
pub enum GgufVersion {
    /// Version 2: lengths and counts widened to 64 bits.
    V2 = 2,
    /// Version 3: identical layout to version 2; adds big-endian files.
    V3 = 3,
}

impl GgufVersion {
    /// The version number stored in the file.
    pub const fn number(self) -> u32 {
        self as u32
    }
}

impl fmt::Display for GgufVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.number())
    }
}

/// The header of a GGUF file: version, metadata and tensor infos.
///
/// Parsing reads only the header region; tensor data is never touched, so
/// opening a multi-gigabyte model costs as much as reading its metadata.
///
/// ```no_run
/// use gguf_inspect::GgufFile;
///
/// let gguf = GgufFile::open("model.gguf")?;
/// println!("{} tensors, {} parameters", gguf.tensors().len(), gguf.parameter_count());
/// if let Some(arch) = gguf.metadata().get_str("general.architecture") {
///     println!("architecture: {arch}");
/// }
/// # Ok::<(), gguf_inspect::Error>(())
/// ```
#[derive(Debug, Clone)]
pub struct GgufFile {
    version: GgufVersion,
    metadata: Metadata,
    tensors: Vec<TensorInfo>,
    tensor_index: HashMap<String, usize>,
    alignment: u32,
    header_size: u64,
    data_offset: u64,
    file_size: u64,
}

impl GgufFile {
    /// Opens and parses the file at `path` with default [`Limits`].
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read or its header is malformed.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        Self::open_with_limits(path, &Limits::default())
    }

    /// Opens and parses the file at `path` with custom [`Limits`].
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read or its header is malformed.
    pub fn open_with_limits(path: impl AsRef<Path>, limits: &Limits) -> Result<Self> {
        let file = File::open(path)?;
        let len = file.metadata()?.len();
        Self::from_reader(
            BufReader::with_capacity(READ_BUFFER_BYTES, file),
            len,
            limits,
        )
    }

    /// Parses a file held in memory, with default [`Limits`].
    ///
    /// ```
    /// use gguf_inspect::{GgufFile, GgufVersion};
    ///
    /// // Magic, version 3, zero tensors, zero metadata entries.
    /// let mut bytes = b"GGUF".to_vec();
    /// bytes.extend_from_slice(&3u32.to_le_bytes());
    /// bytes.extend_from_slice(&0u64.to_le_bytes());
    /// bytes.extend_from_slice(&0u64.to_le_bytes());
    ///
    /// let gguf = GgufFile::from_bytes(&bytes)?;
    /// assert_eq!(gguf.version(), GgufVersion::V3);
    /// assert_eq!(gguf.alignment(), 32);
    /// assert!(gguf.metadata().is_empty());
    /// # Ok::<(), gguf_inspect::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if the header is malformed or truncated.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        Self::from_reader(bytes, bytes.len() as u64, &Limits::default())
    }

    /// Parses a file from `reader`, which must yield `len` bytes: the length is
    /// what declared sizes are validated against.
    ///
    /// Only the header is consumed. Wrap unbuffered readers in a
    /// [`BufReader`]; the parser issues many small reads.
    ///
    /// # Errors
    ///
    /// Returns an error if reading fails or the header is malformed.
    pub fn from_reader(reader: impl Read, len: u64, limits: &Limits) -> Result<Self> {
        Parser {
            r: Reader::new(reader, len),
            limits,
        }
        .parse(len)
    }

    /// The format version.
    pub fn version(&self) -> GgufVersion {
        self.version
    }

    /// The metadata entries.
    pub fn metadata(&self) -> &Metadata {
        &self.metadata
    }

    /// The tensor infos, in file order.
    pub fn tensors(&self) -> &[TensorInfo] {
        &self.tensors
    }

    /// The tensor called `name`.
    pub fn tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.tensor_index.get(name).map(|&i| &self.tensors[i])
    }

    /// Data alignment: `general.alignment`, or 32 if absent.
    pub fn alignment(&self) -> u32 {
        self.alignment
    }

    /// Bytes from the start of the file to the end of the last tensor info.
    pub fn header_size(&self) -> u64 {
        self.header_size
    }

    /// Offset of the data section: [`header_size`](Self::header_size) rounded
    /// up to the alignment. Tensor offsets are relative to it.
    pub fn data_offset(&self) -> u64 {
        self.data_offset
    }

    /// Size of the whole file in bytes, as given when parsing.
    pub fn file_size(&self) -> u64 {
        self.file_size
    }

    /// `general.architecture`, if present.
    pub fn architecture(&self) -> Option<&str> {
        self.metadata.get_str("general.architecture")
    }

    /// Total number of tensor elements (parameters), saturating at `u64::MAX`.
    pub fn parameter_count(&self) -> u64 {
        self.tensors
            .iter()
            .fold(0, |acc: u64, t| acc.saturating_add(t.n_elements()))
    }

    /// Total bytes of tensor data, excluding alignment padding, saturating at `u64::MAX`.
    pub fn tensor_data_size(&self) -> u64 {
        self.tensors
            .iter()
            .fold(0, |acc: u64, t| acc.saturating_add(t.n_bytes()))
    }

    /// Tensor counts, elements and bytes per ggml type, largest share of bytes first.
    pub fn type_breakdown(&self) -> Vec<TypeStats> {
        let mut by_type: BTreeMap<GgmlType, TypeStats> = BTreeMap::new();
        for t in &self.tensors {
            let entry = by_type.entry(t.ggml_type()).or_insert(TypeStats {
                ggml_type: t.ggml_type(),
                tensors: 0,
                elements: 0,
                bytes: 0,
            });
            entry.tensors += 1;
            entry.elements = entry.elements.saturating_add(t.n_elements());
            entry.bytes = entry.bytes.saturating_add(t.n_bytes());
        }
        let mut stats: Vec<_> = by_type.into_values().collect();
        stats.sort_by(|a, b| b.bytes.cmp(&a.bytes).then(a.ggml_type.cmp(&b.ggml_type)));
        stats
    }

    /// Checks the parts of the file that do not stop the header from being
    /// read: tensor data placement and keys the spec requires.
    ///
    /// A partially downloaded model, for example, parses fine but reports
    /// [`Issue::DataOutOfBounds`].
    pub fn validate(&self) -> Vec<Issue> {
        let mut issues = Vec::new();
        if self.architecture().is_none() {
            issues.push(Issue::MissingArchitecture);
        }
        let quantized = self.tensors.iter().any(|t| t.ggml_type().is_quantized());
        if quantized && !self.metadata.contains_key("general.quantization_version") {
            issues.push(Issue::MissingQuantizationVersion);
        }
        if !self.alignment.is_multiple_of(8) {
            issues.push(Issue::AlignmentNotMultipleOf8(self.alignment));
        }
        for t in &self.tensors {
            if !t.offset().is_multiple_of(u64::from(self.alignment)) {
                issues.push(Issue::MisalignedTensor {
                    tensor: t.name().to_owned(),
                    offset: t.offset(),
                    alignment: self.alignment,
                });
            }
        }

        let beyond: Vec<&TensorInfo> = self
            .tensors
            .iter()
            .filter(|t| t.data_range().end > self.file_size)
            .collect();
        if let Some(first) = beyond.first() {
            let data_end = beyond.iter().map(|t| t.data_range().end).max().unwrap_or(0);
            issues.push(Issue::DataOutOfBounds {
                tensors: beyond.len(),
                first: first.name().to_owned(),
                data_end,
                file_size: self.file_size,
            });
        }

        let mut by_offset: Vec<&TensorInfo> =
            self.tensors.iter().filter(|t| t.n_bytes() > 0).collect();
        by_offset.sort_by_key(|t| t.absolute_offset());
        for pair in by_offset.windows(2) {
            if pair[0].data_range().end > pair[1].absolute_offset() {
                issues.push(Issue::OverlappingTensors {
                    first: pair[0].name().to_owned(),
                    second: pair[1].name().to_owned(),
                });
            }
        }
        issues
    }
}

/// Aggregate size of all tensors of one ggml type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TypeStats {
    /// The element type.
    pub ggml_type: GgmlType,
    /// Number of tensors of this type.
    pub tensors: usize,
    /// Total elements across those tensors.
    pub elements: u64,
    /// Total data bytes across those tensors.
    pub bytes: u64,
}

/// A problem found by [`GgufFile::validate`]. None of these stop the header
/// from being inspected, but loaders such as llama.cpp may reject the file.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum Issue {
    /// `general.architecture` (required by the spec) is missing.
    MissingArchitecture,
    /// Some tensors are quantized but `general.quantization_version`
    /// (required in that case) is missing.
    MissingQuantizationVersion,
    /// The spec requires the alignment to be a multiple of 8.
    AlignmentNotMultipleOf8(u32),
    /// A tensor's offset is not a multiple of the alignment.
    MisalignedTensor {
        /// Tensor name.
        tensor: String,
        /// Offset relative to the data section.
        offset: u64,
        /// The file's alignment.
        alignment: u32,
    },
    /// Some tensor data lies past the end of the file, usually because the
    /// file is truncated.
    DataOutOfBounds {
        /// Number of tensors affected.
        tensors: usize,
        /// The first affected tensor in file order.
        first: String,
        /// Where the furthest tensor's data would end.
        data_end: u64,
        /// The actual file size.
        file_size: u64,
    },
    /// Two tensors' data ranges overlap.
    OverlappingTensors {
        /// The tensor that starts first.
        first: String,
        /// The tensor that starts inside it.
        second: String,
    },
}

impl fmt::Display for Issue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingArchitecture => {
                f.write_str("required key general.architecture is missing")
            }
            Self::MissingQuantizationVersion => f.write_str(
                "tensors are quantized but required key general.quantization_version is missing",
            ),
            Self::AlignmentNotMultipleOf8(a) => {
                write!(
                    f,
                    "alignment {a} is not a multiple of 8 as the spec requires"
                )
            }
            Self::MisalignedTensor {
                tensor,
                offset,
                alignment,
            } => write!(
                f,
                "tensor `{tensor}` offset {offset} is not a multiple of the alignment {alignment}"
            ),
            Self::DataOutOfBounds {
                tensors,
                first,
                data_end,
                file_size,
            } => write!(
                f,
                "{tensors} tensor(s) starting with `{first}` extend past the end of the file \
                 (data ends at byte {data_end}, file is {file_size} bytes); the file may be truncated"
            ),
            Self::OverlappingTensors { first, second } => {
                write!(f, "data of tensors `{first}` and `{second}` overlaps")
            }
        }
    }
}

struct Parser<'l, R> {
    r: Reader<R>,
    limits: &'l Limits,
}

impl<R: Read> Parser<'_, R> {
    fn parse(mut self, file_size: u64) -> Result<GgufFile> {
        let magic = self.r.array::<4>()?;
        if magic != MAGIC {
            return Err(Error::new(ErrorKind::BadMagic(magic), 0));
        }
        let version = self.version()?;

        let tensor_count_at = self.r.position();
        let tensor_count = self.r.u64()?;
        let kv_count_at = self.r.position();
        let kv_count = self.r.u64()?;
        check_cap(
            "tensor count",
            tensor_count,
            self.limits.max_tensors,
            tensor_count_at,
        )?;
        check_cap(
            "metadata entry count",
            kv_count,
            self.limits.max_metadata_entries,
            kv_count_at,
        )?;
        self.r
            .check_count("metadata entry count", kv_count, MIN_KV_BYTES, kv_count_at)?;
        self.r.check_count(
            "tensor count",
            tensor_count,
            MIN_TENSOR_INFO_BYTES,
            tensor_count_at,
        )?;

        let mut metadata = Metadata::default();
        for i in 0..kv_count {
            self.metadata_entry(i, &mut metadata)?;
        }
        let alignment = alignment(&metadata)?;

        // Re-check now that the metadata has been consumed.
        let n_tensors = self.r.check_count(
            "tensor count",
            tensor_count,
            MIN_TENSOR_INFO_BYTES,
            tensor_count_at,
        )?;
        let mut tensors = Vec::with_capacity(n_tensors);
        let mut offset_fields = Vec::with_capacity(n_tensors);
        let mut tensor_index = HashMap::with_capacity(n_tensors);
        for i in 0..n_tensors {
            let (tensor, offset_at) = self.tensor_info(i, &tensor_index)?;
            tensor_index.insert(tensor.name().to_owned(), i);
            tensors.push(tensor);
            offset_fields.push(offset_at);
        }

        let header_size = self.r.position();
        let data_offset = align_up(header_size, alignment)
            .ok_or_else(|| Error::new(ErrorKind::OffsetOverflow, header_size))?;
        for (tensor, &at) in tensors.iter_mut().zip(&offset_fields) {
            let absolute = data_offset.checked_add(tensor.offset()).ok_or_else(|| {
                Error::new(ErrorKind::OffsetOverflow, at)
                    .with_context(|| format!("tensor info `{}`", tensor.name()))
            })?;
            tensor.set_absolute_offset(absolute);
        }

        Ok(GgufFile {
            version,
            metadata,
            tensors,
            tensor_index,
            alignment,
            header_size,
            data_offset,
            file_size,
        })
    }

    fn version(&mut self) -> Result<GgufVersion> {
        let at = self.r.position();
        let raw = self.r.u32()?;
        let kind = match raw {
            2 => return Ok(GgufVersion::V2),
            3 => return Ok(GgufVersion::V3),
            // A big-endian writer stores e.g. 3 as 00 00 00 03, which reads as
            // 0x0300_0000 here. Real versions never get anywhere near 2^16.
            _ if raw != 0 && raw.trailing_zeros() >= 16 => ErrorKind::BigEndian {
                version: raw.swap_bytes(),
            },
            other => ErrorKind::UnsupportedVersion(other),
        };
        Err(Error::new(kind, at))
    }

    fn metadata_entry(&mut self, index: u64, metadata: &mut Metadata) -> Result<()> {
        let at = self.r.position();
        let entry = || format!("metadata entry {index}");
        let key = self
            .r
            .string("key length", MAX_KEY_LEN)
            .map_err(|e| e.with_context(entry))?;
        if key.is_empty() {
            return Err(Error::new(ErrorKind::EmptyKey, at).with_context(entry));
        }
        if metadata.contains_key(&key) {
            return Err(Error::new(ErrorKind::DuplicateKey(key), at).with_context(entry));
        }
        let value = self
            .value_type()
            .and_then(|ty| self.value(ty, 0))
            .map_err(|e| e.with_context(|| format!("metadata entry {index} `{key}`")))?;
        let inserted = metadata.insert(key, value, at);
        debug_assert!(inserted, "duplicates are rejected above");
        Ok(())
    }

    fn value_type(&mut self) -> Result<ValueType> {
        let at = self.r.position();
        let id = self.r.u32()?;
        ValueType::from_id(id).ok_or_else(|| Error::new(ErrorKind::InvalidValueType(id), at))
    }

    fn value(&mut self, ty: ValueType, depth: u32) -> Result<MetadataValue> {
        let r = &mut self.r;
        Ok(match ty {
            ValueType::U8 => MetadataValue::U8(r.u8()?),
            ValueType::I8 => MetadataValue::I8(r.i8()?),
            ValueType::U16 => MetadataValue::U16(r.u16()?),
            ValueType::I16 => MetadataValue::I16(r.i16()?),
            ValueType::U32 => MetadataValue::U32(r.u32()?),
            ValueType::I32 => MetadataValue::I32(r.i32()?),
            ValueType::F32 => MetadataValue::F32(r.f32()?),
            ValueType::Bool => MetadataValue::Bool(r.bool()?),
            ValueType::U64 => MetadataValue::U64(r.u64()?),
            ValueType::I64 => MetadataValue::I64(r.i64()?),
            ValueType::F64 => MetadataValue::F64(r.f64()?),
            ValueType::String => {
                MetadataValue::String(r.string("string length", self.limits.max_string_len)?)
            }
            ValueType::Array => MetadataValue::Array(self.array(depth + 1)?),
        })
    }

    fn array(&mut self, depth: u32) -> Result<MetadataArray> {
        let at = self.r.position();
        if depth > self.limits.max_array_depth {
            return Err(Error::new(
                ErrorKind::NestingTooDeep {
                    limit: self.limits.max_array_depth,
                },
                at,
            ));
        }
        let ty = self.value_type()?;
        let len_at = self.r.position();
        let len = self.r.u64()?;
        check_cap("array length", len, self.limits.max_array_len, len_at)?;
        let n = self
            .r
            .check_count("array length", len, ty.min_encoded_size(), len_at)?;
        let r = &mut self.r;
        Ok(match ty {
            ValueType::U8 => MetadataArray::U8(r.fixed_vec(n, u8::from_le_bytes)?),
            ValueType::I8 => MetadataArray::I8(r.fixed_vec(n, i8::from_le_bytes)?),
            ValueType::U16 => MetadataArray::U16(r.fixed_vec(n, u16::from_le_bytes)?),
            ValueType::I16 => MetadataArray::I16(r.fixed_vec(n, i16::from_le_bytes)?),
            ValueType::U32 => MetadataArray::U32(r.fixed_vec(n, u32::from_le_bytes)?),
            ValueType::I32 => MetadataArray::I32(r.fixed_vec(n, i32::from_le_bytes)?),
            ValueType::F32 => MetadataArray::F32(r.fixed_vec(n, f32::from_le_bytes)?),
            ValueType::U64 => MetadataArray::U64(r.fixed_vec(n, u64::from_le_bytes)?),
            ValueType::I64 => MetadataArray::I64(r.fixed_vec(n, i64::from_le_bytes)?),
            ValueType::F64 => MetadataArray::F64(r.fixed_vec(n, f64::from_le_bytes)?),
            ValueType::Bool => MetadataArray::Bool(r.bool_vec(n)?),
            ValueType::String => {
                let mut items = Vec::with_capacity(n);
                for _ in 0..n {
                    items.push(r.string("string length", self.limits.max_string_len)?);
                }
                MetadataArray::String(items)
            }
            ValueType::Array => {
                let mut items = Vec::with_capacity(n);
                for _ in 0..n {
                    items.push(self.array(depth + 1)?);
                }
                MetadataArray::Array(items)
            }
        })
    }

    /// Reads one tensor info record and returns it with the offset of its
    /// `offset` field.
    fn tensor_info(
        &mut self,
        index: usize,
        seen: &HashMap<String, usize>,
    ) -> Result<(TensorInfo, u64)> {
        let at = self.r.position();
        let record = || format!("tensor info {index}");
        let name = self
            .r
            .string("tensor name length", MAX_TENSOR_NAME_LEN)
            .map_err(|e| e.with_context(record))?;
        if seen.contains_key(&name) {
            return Err(Error::new(ErrorKind::DuplicateTensorName(name), at).with_context(record));
        }
        let context = |e: Error| e.with_context(|| format!("tensor info {index} `{name}`"));

        let n_dims_at = self.r.position();
        let n_dims = self.r.u32().map_err(context)?;
        if n_dims > self.limits.max_dims {
            return Err(context(Error::new(
                ErrorKind::TooManyDimensions {
                    n_dims,
                    max: self.limits.max_dims,
                },
                n_dims_at,
            )));
        }
        let dims_at = self.r.position();
        let dims = (0..n_dims)
            .map(|_| self.r.u64())
            .collect::<Result<Vec<_>>>()
            .map_err(context)?;
        let type_at = self.r.position();
        let type_id = self.r.u32().map_err(context)?;
        let ggml_type = GgmlType::from_id(type_id)
            .ok_or_else(|| context(Error::new(ErrorKind::UnknownGgmlType(type_id), type_at)))?;
        let offset_at = self.r.position();
        let offset = self.r.u64().map_err(context)?;

        let tensor = TensorInfo::new(name.clone(), dims, ggml_type, offset)
            .map_err(|kind| context(Error::new(kind, dims_at)))?;
        Ok((tensor, offset_at))
    }
}

fn check_cap(what: &'static str, value: u64, limit: u64, at: u64) -> Result<()> {
    if value > limit {
        return Err(Error::new(
            ErrorKind::LimitExceeded { what, value, limit },
            at,
        ));
    }
    Ok(())
}

/// Reads and validates `general.alignment`, defaulting to 32.
fn alignment(metadata: &Metadata) -> Result<u32> {
    let Some(value) = metadata.get(ALIGNMENT_KEY) else {
        return Ok(DEFAULT_ALIGNMENT);
    };
    let at = metadata.offset_of(ALIGNMENT_KEY).unwrap_or(0);
    match *value {
        MetadataValue::U32(a) if a.is_power_of_two() => Ok(a),
        MetadataValue::U32(a) => Err(Error::new(ErrorKind::InvalidAlignment(a), at)),
        ref other => Err(Error::new(ErrorKind::AlignmentType(other.value_type()), at)),
    }
}

/// Rounds `offset` up to a multiple of `alignment` (the spec's `align_offset`).
fn align_up(offset: u64, alignment: u32) -> Option<u64> {
    let alignment = u64::from(alignment);
    offset.checked_next_multiple_of(alignment)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn align_up_matches_spec_formula() {
        let spec = |o: u64, a: u64| o + (a - (o % a)) % a;
        for a in [1_u32, 8, 32, 64, 4096] {
            for o in 0..200 {
                assert_eq!(align_up(o, a), Some(spec(o, u64::from(a))));
            }
        }
        assert_eq!(align_up(u64::MAX, 32), None);
    }

    #[test]
    fn version_detection() {
        let header = |v: u32| {
            let mut b = MAGIC.to_vec();
            b.extend_from_slice(&v.to_le_bytes());
            b.extend_from_slice(&[0; 16]);
            b
        };
        assert_eq!(
            GgufFile::from_bytes(&header(2)).unwrap().version(),
            GgufVersion::V2
        );
        let be = GgufFile::from_bytes(&header(3_u32.swap_bytes())).unwrap_err();
        assert!(matches!(be.kind(), ErrorKind::BigEndian { version: 3 }));
        assert_eq!(be.offset(), Some(4));
        for v in [0, 1, 4, 0xFFFF_FFFF] {
            let err = GgufFile::from_bytes(&header(v)).unwrap_err();
            assert!(
                matches!(err.kind(), ErrorKind::UnsupportedVersion(x) if *x == v),
                "{v}: {err}"
            );
        }
    }
}
