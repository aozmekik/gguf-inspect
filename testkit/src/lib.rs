//! Test-only writer for synthetic GGUF files.
//!
//! Used by the `gguf-inspect` tests, benchmarks, examples and fuzz seed
//! corpus. It is deliberately small and permissive: it writes whatever it is
//! told, including invalid versions, big-endian byte order and misplaced
//! tensors, so that tests can produce malformed files on purpose.
//!
//! Nothing here is a real model. [`SyntheticModel`] produces headers shaped
//! like a llama-architecture model with made-up vocabulary strings and
//! zero-filled (or sparse) tensor data.

use std::fs::File;
use std::io::{self, BufWriter, Seek, SeekFrom, Write};
use std::path::Path;

use gguf_inspect::{GgmlType, MetadataArray, MetadataValue};

/// Byte order to write multi-byte fields in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Endian {
    /// Little-endian, the only byte order `gguf-inspect` accepts.
    #[default]
    Little,
    /// Big-endian, as written by `gguf-py` with `endianess=BIG`.
    Big,
}

/// One tensor info record to write.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorSpec {
    /// Tensor name.
    pub name: String,
    /// Shape, innermost dimension first.
    pub dims: Vec<u64>,
    /// Raw ggml type ID.
    pub type_id: u32,
    /// Explicit data offset; `None` places the tensor after the previous one,
    /// aligned.
    pub offset: Option<u64>,
}

/// Where each part of a built file starts.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Layout {
    /// Offset of each metadata entry (its key length field).
    pub kv_offsets: Vec<u64>,
    /// Offset of each tensor info record.
    pub tensor_info_offsets: Vec<u64>,
    /// End of the last tensor info.
    pub header_size: u64,
    /// Start of the data section.
    pub data_offset: u64,
    /// Data offset of each tensor, relative to the data section.
    pub tensor_offsets: Vec<u64>,
    /// Size of the complete file, data included.
    pub file_size: u64,
}

/// Builds GGUF files in memory or on disk.
///
/// ```
/// use gguf_inspect::{GgmlType, GgufFile, MetadataValue};
/// use gguf_testkit::GgufBuilder;
///
/// let bytes = GgufBuilder::new()
///     .kv("general.architecture", &MetadataValue::String("llama".into()))
///     .tensor("w", &[64, 2], GgmlType::Q8_0)
///     .build();
/// let gguf = GgufFile::from_bytes(&bytes).unwrap();
/// assert_eq!(gguf.tensors()[0].n_bytes(), 2 * 2 * 34);
/// ```
#[derive(Debug, Clone)]
pub struct GgufBuilder {
    version: u32,
    endian: Endian,
    alignment: u32,
    kvs: Vec<(Vec<u8>, Vec<u8>)>,
    tensors: Vec<TensorSpec>,
}

impl Default for GgufBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl GgufBuilder {
    /// An empty version-3 little-endian file with the default alignment of 32.
    pub fn new() -> Self {
        Self {
            version: 3,
            endian: Endian::Little,
            alignment: 32,
            kvs: Vec::new(),
            tensors: Vec::new(),
        }
    }

    /// Sets the version field (any value, valid or not).
    #[must_use]
    pub fn version(mut self, version: u32) -> Self {
        self.version = version;
        self
    }

    /// Sets the byte order.
    #[must_use]
    pub fn endian(mut self, endian: Endian) -> Self {
        self.endian = endian;
        self
    }

    /// Adds `general.alignment` as a `u32` and lays tensors out with it.
    #[must_use]
    pub fn alignment(mut self, alignment: u32) -> Self {
        self.alignment = alignment;
        self.kv("general.alignment", &MetadataValue::U32(alignment))
    }

    /// Adds a metadata entry.
    #[must_use]
    pub fn kv(mut self, key: &str, value: &MetadataValue) -> Self {
        let mut encoded = Vec::new();
        put_u32(&mut encoded, value.value_type().id(), self.endian);
        encode_value(&mut encoded, value, self.endian);
        self.kvs.push((key.as_bytes().to_vec(), encoded));
        self
    }

    /// Adds a metadata entry with a raw key and raw bytes after the key,
    /// starting with the type tag.
    #[must_use]
    pub fn raw_kv(mut self, key: &[u8], type_and_value: Vec<u8>) -> Self {
        self.kvs.push((key.to_vec(), type_and_value));
        self
    }

    /// Adds a tensor placed after the previous one.
    #[must_use]
    pub fn tensor(self, name: &str, dims: &[u64], ggml_type: GgmlType) -> Self {
        self.tensor_spec(TensorSpec {
            name: name.to_owned(),
            dims: dims.to_vec(),
            type_id: ggml_type.id(),
            offset: None,
        })
    }

    /// Adds an arbitrary tensor info record.
    #[must_use]
    pub fn tensor_spec(mut self, spec: TensorSpec) -> Self {
        self.tensors.push(spec);
        self
    }

    /// Computes where everything goes without writing anything.
    pub fn layout(&self) -> Layout {
        let mut layout = Layout::default();
        let mut pos = 4 + 4 + 8 + 8;
        for (key, rest) in &self.kvs {
            layout.kv_offsets.push(pos);
            pos += 8 + key.len() as u64 + rest.len() as u64;
        }
        let mut next = 0_u64;
        for t in &self.tensors {
            layout.tensor_info_offsets.push(pos);
            pos += 8 + t.name.len() as u64 + 4 + 8 * t.dims.len() as u64 + 4 + 8;
            let offset = t.offset.unwrap_or(next);
            layout.tensor_offsets.push(offset);
            next = align(offset.saturating_add(data_size(t)), self.alignment);
        }
        layout.header_size = pos;
        layout.data_offset = align(pos, self.alignment);
        layout.file_size = layout.data_offset.saturating_add(
            self.tensors
                .iter()
                .zip(&layout.tensor_offsets)
                .map(|(t, &o)| o.saturating_add(data_size(t)))
                .max()
                .unwrap_or(0),
        );
        layout
    }

    /// The header and alignment padding, without tensor data.
    pub fn header(&self) -> Vec<u8> {
        let layout = self.layout();
        let e = self.endian;
        let mut out = Vec::with_capacity(usize::try_from(layout.data_offset).unwrap_or(0));
        out.extend_from_slice(b"GGUF");
        put_u32(&mut out, self.version, e);
        put_u64(&mut out, self.tensors.len() as u64, e);
        put_u64(&mut out, self.kvs.len() as u64, e);
        for (key, rest) in &self.kvs {
            put_u64(&mut out, key.len() as u64, e);
            out.extend_from_slice(key);
            out.extend_from_slice(rest);
        }
        for (t, &offset) in self.tensors.iter().zip(&layout.tensor_offsets) {
            put_str(&mut out, &t.name, e);
            put_u32(&mut out, u32::try_from(t.dims.len()).unwrap_or(u32::MAX), e);
            for &d in &t.dims {
                put_u64(&mut out, d, e);
            }
            put_u32(&mut out, t.type_id, e);
            put_u64(&mut out, offset, e);
        }
        out.resize(usize::try_from(layout.data_offset).unwrap_or(out.len()), 0);
        out
    }

    /// The complete file with zero-filled tensor data.
    ///
    /// # Panics
    ///
    /// If the tensor data does not fit in memory.
    pub fn build(&self) -> Vec<u8> {
        let mut out = self.header();
        let size = usize::try_from(self.layout().file_size).expect("file fits in memory");
        out.resize(size, 0);
        out
    }

    /// Writes the header to `path` and extends the file to its full size
    /// without writing tensor data, which yields a sparse file on file systems
    /// that support them (APFS, ext4, NTFS).
    ///
    /// # Errors
    ///
    /// Returns any I/O error from creating or writing the file.
    pub fn write_sparse(&self, path: &Path) -> io::Result<Layout> {
        let layout = self.layout();
        let mut file = BufWriter::new(File::create(path)?);
        file.write_all(&self.header())?;
        let mut file = file.into_inner().map_err(io::IntoInnerError::into_error)?;
        file.seek(SeekFrom::Start(0))?;
        file.set_len(layout.file_size.max(layout.data_offset))?;
        Ok(layout)
    }
}

/// Bytes of tensor data a spec occupies; 0 for unknown types or overflowing shapes.
fn data_size(t: &TensorSpec) -> u64 {
    let Some(ty) = GgmlType::from_id(t.type_id) else {
        return 0;
    };
    let elements = t
        .dims
        .iter()
        .try_fold(1_u64, |acc, &d| acc.checked_mul(d))
        .unwrap_or(0);
    (elements / ty.block_size()).saturating_mul(ty.type_size())
}

fn align(offset: u64, alignment: u32) -> u64 {
    let a = u64::from(alignment.max(1));
    offset.checked_next_multiple_of(a).unwrap_or(u64::MAX)
}

/// Appends a `u32` in byte order `e`.
pub fn put_u32(out: &mut Vec<u8>, v: u32, e: Endian) {
    out.extend_from_slice(&match e {
        Endian::Little => v.to_le_bytes(),
        Endian::Big => v.to_be_bytes(),
    });
}

/// Appends a `u64` in byte order `e`.
pub fn put_u64(out: &mut Vec<u8>, v: u64, e: Endian) {
    out.extend_from_slice(&match e {
        Endian::Little => v.to_le_bytes(),
        Endian::Big => v.to_be_bytes(),
    });
}

/// Appends a length-prefixed string.
pub fn put_str(out: &mut Vec<u8>, s: &str, e: Endian) {
    put_u64(out, s.len() as u64, e);
    out.extend_from_slice(s.as_bytes());
}

macro_rules! put_num {
    ($out:expr, $v:expr, $e:expr) => {
        $out.extend_from_slice(&match $e {
            Endian::Little => $v.to_le_bytes(),
            Endian::Big => $v.to_be_bytes(),
        })
    };
}

/// Appends a value without its type tag.
///
/// # Panics
///
/// On value variants added to `gguf-inspect` after this function was written.
pub fn encode_value(out: &mut Vec<u8>, value: &MetadataValue, e: Endian) {
    match value {
        MetadataValue::U8(v) => out.push(*v),
        MetadataValue::I8(v) => put_num!(out, v, e),
        MetadataValue::U16(v) => put_num!(out, v, e),
        MetadataValue::I16(v) => put_num!(out, v, e),
        MetadataValue::U32(v) => put_num!(out, v, e),
        MetadataValue::I32(v) => put_num!(out, v, e),
        MetadataValue::F32(v) => put_num!(out, v, e),
        MetadataValue::Bool(v) => out.push(u8::from(*v)),
        MetadataValue::String(s) => put_str(out, s, e),
        MetadataValue::Array(a) => encode_array(out, a, e),
        MetadataValue::U64(v) => put_num!(out, v, e),
        MetadataValue::I64(v) => put_num!(out, v, e),
        MetadataValue::F64(v) => put_num!(out, v, e),
        other => panic!("testkit cannot encode {other:?}"),
    }
}

/// Appends an array: element type tag, length, elements.
///
/// # Panics
///
/// On array variants added to `gguf-inspect` after this function was written.
pub fn encode_array(out: &mut Vec<u8>, array: &MetadataArray, e: Endian) {
    put_u32(out, array.element_type().id(), e);
    put_u64(out, array.len() as u64, e);
    match array {
        MetadataArray::String(v) => v.iter().for_each(|s| put_str(out, s, e)),
        MetadataArray::Array(v) => v.iter().for_each(|a| encode_array(out, a, e)),
        MetadataArray::U8(v) => out.extend_from_slice(v),
        MetadataArray::Bool(v) => out.extend(v.iter().map(|&b| u8::from(b))),
        MetadataArray::I8(v) => v.iter().for_each(|x| put_num!(out, x, e)),
        MetadataArray::U16(v) => v.iter().for_each(|x| put_num!(out, x, e)),
        MetadataArray::I16(v) => v.iter().for_each(|x| put_num!(out, x, e)),
        MetadataArray::U32(v) => v.iter().for_each(|x| put_num!(out, x, e)),
        MetadataArray::I32(v) => v.iter().for_each(|x| put_num!(out, x, e)),
        MetadataArray::F32(v) => v.iter().for_each(|x| put_num!(out, x, e)),
        MetadataArray::U64(v) => v.iter().for_each(|x| put_num!(out, x, e)),
        MetadataArray::I64(v) => v.iter().for_each(|x| put_num!(out, x, e)),
        MetadataArray::F64(v) => v.iter().for_each(|x| put_num!(out, x, e)),
        other => panic!("testkit cannot encode {other:?}"),
    }
}

/// One metadata entry of every value type, including nested and empty arrays.
pub fn every_value_type() -> Vec<(&'static str, MetadataValue)> {
    use MetadataValue as V;
    vec![
        ("test.u8", V::U8(200)),
        ("test.i8", V::I8(-100)),
        ("test.u16", V::U16(60_000)),
        ("test.i16", V::I16(-30_000)),
        ("test.u32", V::U32(4_000_000_000)),
        ("test.i32", V::I32(-2_000_000_000)),
        ("test.f32", V::F32(0.1)),
        ("test.bool", V::Bool(true)),
        ("test.string", V::String("héllo, wörld".into())),
        ("test.u64", V::U64(u64::MAX)),
        ("test.i64", V::I64(i64::MIN)),
        ("test.f64", V::F64(-1.5e300)),
        (
            "test.array.strings",
            V::Array(MetadataArray::String(vec!["a".into(), String::new()])),
        ),
        (
            "test.array.bools",
            V::Array(MetadataArray::Bool(vec![true, false])),
        ),
        ("test.array.empty_f64", V::Array(MetadataArray::F64(vec![]))),
        (
            "test.array.nested",
            V::Array(MetadataArray::Array(vec![
                MetadataArray::I32(vec![1, -2]),
                MetadataArray::String(vec!["x".into()]),
                MetadataArray::Array(vec![MetadataArray::U8(vec![7])]),
            ])),
        ),
    ]
}

/// Hyperparameters of a synthetic llama-architecture model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SyntheticModel {
    /// `general.name`.
    pub name: String,
    /// Vocabulary size; tokens are generated strings like `tok_123`.
    pub vocab_size: u64,
    /// `llama.embedding_length`.
    pub embedding_length: u64,
    /// `llama.block_count`.
    pub block_count: u64,
    /// `llama.attention.head_count`.
    pub head_count: u64,
    /// `llama.attention.head_count_kv`.
    pub head_count_kv: u64,
    /// `llama.feed_forward_length`.
    pub feed_forward_length: u64,
    /// `llama.context_length`.
    pub context_length: u64,
    /// Type of the large weight matrices.
    pub weight_type: GgmlType,
    /// Type of `output.weight` and `attn_v`/`ffn_down`.
    pub sensitive_type: GgmlType,
}

impl SyntheticModel {
    /// Shaped like an 8B-parameter Llama-3 model (vocab 128 256, 32 layers,
    /// 32 heads, 8 KV heads), with `q4_K` weights and `q6_K` for `attn_v`,
    /// `ffn_down` and `output`: a simplified `Q4_K_M`-style mix.
    pub fn llama3_8b_like() -> Self {
        Self {
            name: "Synthetic Llama-3-8B-shaped".into(),
            vocab_size: 128_256,
            embedding_length: 4096,
            block_count: 32,
            head_count: 32,
            head_count_kv: 8,
            feed_forward_length: 14_336,
            context_length: 8192,
            weight_type: GgmlType::Q4_K,
            sensitive_type: GgmlType::Q6_K,
        }
    }

    /// A tiny model for fast tests: vocab 512, 2 layers, 4 heads, 2 KV heads.
    pub fn tiny() -> Self {
        Self {
            name: "Synthetic tiny".into(),
            vocab_size: 512,
            embedding_length: 256,
            block_count: 2,
            head_count: 4,
            head_count_kv: 2,
            feed_forward_length: 512,
            context_length: 2048,
            weight_type: GgmlType::Q4_K,
            sensitive_type: GgmlType::Q8_0,
        }
    }

    /// The builder for this model's header.
    ///
    /// # Panics
    ///
    /// If a hyperparameter does not fit in `u32` or the vocabulary does not
    /// fit in memory.
    pub fn builder(&self) -> GgufBuilder {
        use MetadataValue as V;
        let u32v = |x: u64| V::U32(u32::try_from(x).expect("hyperparameter fits in u32"));
        let head_dim = self.embedding_length / self.head_count;
        let kv_dim = head_dim * self.head_count_kv;
        let vocab = usize::try_from(self.vocab_size).expect("vocab fits in memory");
        let tokens: Vec<String> = (0..vocab).map(|i| format!("tok_{i}")).collect();
        #[allow(clippy::cast_precision_loss)]
        let scores: Vec<f32> = (0..vocab).map(|i| 0.0 - i as f32).collect();
        let token_types: Vec<i32> = (0..vocab).map(|i| if i < 3 { 3 } else { 1 }).collect();

        let metadata = [
            ("general.architecture", V::String("llama".into())),
            ("general.name", V::String(self.name.clone())),
            ("general.file_type", V::U32(15)),
            ("general.quantization_version", V::U32(2)),
            ("llama.context_length", u32v(self.context_length)),
            ("llama.embedding_length", u32v(self.embedding_length)),
            ("llama.block_count", u32v(self.block_count)),
            ("llama.feed_forward_length", u32v(self.feed_forward_length)),
            ("llama.attention.head_count", u32v(self.head_count)),
            ("llama.attention.head_count_kv", u32v(self.head_count_kv)),
            ("llama.attention.layer_norm_rms_epsilon", V::F32(1e-5)),
            ("llama.rope.dimension_count", u32v(head_dim)),
            ("llama.rope.freq_base", V::F32(500_000.0)),
            ("tokenizer.ggml.model", V::String("gpt2".into())),
            (
                "tokenizer.ggml.tokens",
                V::Array(MetadataArray::String(tokens)),
            ),
            (
                "tokenizer.ggml.scores",
                V::Array(MetadataArray::F32(scores)),
            ),
            (
                "tokenizer.ggml.token_type",
                V::Array(MetadataArray::I32(token_types)),
            ),
            ("tokenizer.ggml.bos_token_id", V::U32(0)),
            ("tokenizer.ggml.eos_token_id", V::U32(1)),
        ];
        let mut builder = metadata
            .iter()
            .fold(GgufBuilder::new(), |b, (key, value)| b.kv(key, value));

        let embd = self.embedding_length;
        let ff = self.feed_forward_length;
        let (weights, sensitive) = (self.weight_type, self.sensitive_type);
        builder = builder.tensor("token_embd.weight", &[embd, self.vocab_size], weights);
        for layer in 0..self.block_count {
            let name = |s: &str| format!("blk.{layer}.{s}");
            builder = builder
                .tensor(&name("attn_norm.weight"), &[embd], GgmlType::F32)
                .tensor(&name("attn_q.weight"), &[embd, embd], weights)
                .tensor(&name("attn_k.weight"), &[embd, kv_dim], weights)
                .tensor(&name("attn_v.weight"), &[embd, kv_dim], sensitive)
                .tensor(&name("attn_output.weight"), &[embd, embd], weights)
                .tensor(&name("ffn_norm.weight"), &[embd], GgmlType::F32)
                .tensor(&name("ffn_gate.weight"), &[embd, ff], weights)
                .tensor(&name("ffn_up.weight"), &[embd, ff], weights)
                .tensor(&name("ffn_down.weight"), &[ff, embd], sensitive);
        }
        builder
            .tensor("output_norm.weight", &[embd], GgmlType::F32)
            .tensor("output.weight", &[embd, self.vocab_size], sensitive)
    }

    /// Total parameter count implied by the tensor shapes.
    pub fn parameter_count(&self) -> u64 {
        let embd = self.embedding_length;
        let kv_dim = embd / self.head_count * self.head_count_kv;
        let per_layer =
            2 * embd + 2 * embd * embd + 2 * embd * kv_dim + 3 * embd * self.feed_forward_length;
        2 * embd * self.vocab_size + embd + self.block_count * per_layer
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_matches_bytes() {
        let b = GgufBuilder::new()
            .alignment(64)
            .kv("a", &MetadataValue::String("xyz".into()))
            .tensor("t", &[32], GgmlType::Q4_0)
            .tensor("u", &[3], GgmlType::F32);
        let layout = b.layout();
        let bytes = b.build();
        assert_eq!(bytes.len() as u64, layout.file_size);
        assert_eq!(layout.data_offset % 64, 0);
        assert_eq!(layout.tensor_offsets, vec![0, 64]);
        assert_eq!(layout.file_size, layout.data_offset + 64 + 12);
        assert_eq!(&bytes[..4], b"GGUF");
    }

    #[test]
    fn big_endian_swaps_fields() {
        let bytes = GgufBuilder::new().endian(Endian::Big).header();
        assert_eq!(&bytes[4..8], &[0, 0, 0, 3]);
    }
}
