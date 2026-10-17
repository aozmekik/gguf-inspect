//! A bounds-checked little-endian reader over any [`Read`] of known length.
//!
//! Every read is checked against the bytes left before touching the
//! underlying reader, and every declared length is checked before anything is
//! allocated for it. Errors carry the offset of the offending field.

use std::io::Read;

use crate::error::{Error, ErrorKind, Result};

/// Elements decoded per chunk when reading fixed-width arrays.
const CHUNK_BYTES: usize = 8 * 1024;

pub(crate) struct Reader<R> {
    inner: R,
    pos: u64,
    len: u64,
}

macro_rules! read_le {
    ($($name:ident -> $ty:ty),+ $(,)?) => {
        $(
            pub(crate) fn $name(&mut self) -> Result<$ty> {
                Ok(<$ty>::from_le_bytes(self.array()?))
            }
        )+
    };
}

impl<R: Read> Reader<R> {
    pub(crate) fn new(inner: R, len: u64) -> Self {
        Self { inner, pos: 0, len }
    }

    /// Offset of the next byte to be read.
    pub(crate) fn position(&self) -> u64 {
        self.pos
    }

    pub(crate) fn remaining(&self) -> u64 {
        self.len.saturating_sub(self.pos)
    }

    fn fill(&mut self, buf: &mut [u8]) -> Result<()> {
        let needed = buf.len() as u64;
        let available = self.remaining();
        if needed > available {
            return Err(Error::new(
                ErrorKind::UnexpectedEof { needed, available },
                self.pos,
            ));
        }
        self.inner
            .read_exact(buf)
            .map_err(|e| Error::new(ErrorKind::Io(e), self.pos))?;
        self.pos += needed;
        Ok(())
    }

    pub(crate) fn array<const N: usize>(&mut self) -> Result<[u8; N]> {
        let mut buf = [0; N];
        self.fill(&mut buf)?;
        Ok(buf)
    }

    read_le! {
        u8 -> u8, i8 -> i8, u16 -> u16, i16 -> i16, u32 -> u32, i32 -> i32,
        u64 -> u64, i64 -> i64, f32 -> f32, f64 -> f64,
    }

    pub(crate) fn bool(&mut self) -> Result<bool> {
        let at = self.pos;
        decode_bool(self.u8()?, at)
    }

    /// Checks that `count` items of at least `min_size` bytes each can fit in
    /// the rest of the file, so that `count` is safe to use as a capacity.
    /// `at` is the offset of the field that declared `count`.
    pub(crate) fn check_count(
        &self,
        what: &'static str,
        count: u64,
        min_size: u64,
        at: u64,
    ) -> Result<usize> {
        let min_bytes = count.saturating_mul(min_size);
        let available = self.remaining();
        if min_bytes > available {
            return Err(Error::new(
                ErrorKind::LengthExceedsFile {
                    what,
                    declared: count,
                    min_bytes,
                    available,
                },
                at,
            ));
        }
        // Only fails on targets where usize is narrower than the file.
        usize::try_from(count).map_err(|_| {
            Error::new(
                ErrorKind::LimitExceeded {
                    what,
                    value: count,
                    limit: usize::MAX as u64,
                },
                at,
            )
        })
    }

    /// Reads a length-prefixed UTF-8 string of at most `max_len` bytes.
    pub(crate) fn string(&mut self, what: &'static str, max_len: u64) -> Result<String> {
        let at = self.pos;
        let len = self.u64()?;
        if len > max_len {
            return Err(Error::new(
                ErrorKind::LimitExceeded {
                    what,
                    value: len,
                    limit: max_len,
                },
                at,
            ));
        }
        let n = self.check_count(what, len, 1, at)?;
        let mut buf = vec![0; n];
        self.fill(&mut buf)?;
        String::from_utf8(buf).map_err(|_| Error::new(ErrorKind::InvalidUtf8, at + 8))
    }

    /// Reads `n` fixed-width values. The caller must have validated `n` with
    /// [`check_count`](Self::check_count).
    pub(crate) fn fixed_vec<T, const N: usize>(
        &mut self,
        n: usize,
        decode: fn([u8; N]) -> T,
    ) -> Result<Vec<T>> {
        let mut out = Vec::with_capacity(n);
        let mut chunk = [0_u8; CHUNK_BYTES];
        let per_chunk = CHUNK_BYTES / N;
        while out.len() < n {
            let take = (n - out.len()).min(per_chunk);
            let bytes = &mut chunk[..take * N];
            self.fill(bytes)?;
            let (values, _) = bytes.as_chunks::<N>();
            out.extend(values.iter().map(|&b| decode(b)));
        }
        Ok(out)
    }

    /// Reads `n` bools, rejecting bytes other than 0 and 1.
    pub(crate) fn bool_vec(&mut self, n: usize) -> Result<Vec<bool>> {
        let mut out = Vec::with_capacity(n);
        let mut chunk = [0_u8; CHUNK_BYTES];
        while out.len() < n {
            let take = (n - out.len()).min(CHUNK_BYTES);
            let start = self.pos;
            let bytes = &mut chunk[..take];
            self.fill(bytes)?;
            for (i, &b) in bytes.iter().enumerate() {
                out.push(decode_bool(b, start + i as u64)?);
            }
        }
        Ok(out)
    }
}

fn decode_bool(byte: u8, at: u64) -> Result<bool> {
    match byte {
        0 => Ok(false),
        1 => Ok(true),
        b => Err(Error::new(ErrorKind::InvalidBool(b), at)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reader(bytes: &[u8]) -> Reader<&[u8]> {
        Reader::new(bytes, bytes.len() as u64)
    }

    #[test]
    fn reads_little_endian_scalars() {
        let mut r = reader(&[1, 0, 2, 0, 0, 0, 0xff]);
        assert_eq!(r.u16().unwrap(), 1);
        assert_eq!(r.u32().unwrap(), 2);
        assert_eq!(r.i8().unwrap(), -1);
        assert_eq!(r.position(), 7);
        assert_eq!(r.remaining(), 0);
    }

    #[test]
    fn eof_reports_position_and_sizes() {
        let mut r = reader(&[0, 0, 0]);
        let err = r.u32().unwrap_err();
        assert_eq!(err.offset(), Some(0));
        assert!(matches!(
            err.kind(),
            ErrorKind::UnexpectedEof {
                needed: 4,
                available: 3
            }
        ));
    }

    #[test]
    fn declared_length_is_checked_before_allocating() {
        let mut bytes = u64::MAX.to_le_bytes().to_vec();
        bytes.extend_from_slice(b"abc");
        let err = reader(&bytes)
            .string("string length", u64::MAX)
            .unwrap_err();
        assert!(matches!(
            err.kind(),
            ErrorKind::LengthExceedsFile { available: 3, .. }
        ));
    }

    #[test]
    fn string_cap_is_enforced() {
        let mut bytes = 5_u64.to_le_bytes().to_vec();
        bytes.extend_from_slice(b"hello");
        let err = reader(&bytes).string("key length", 4).unwrap_err();
        assert!(matches!(
            err.kind(),
            ErrorKind::LimitExceeded {
                value: 5,
                limit: 4,
                ..
            }
        ));
        assert_eq!(reader(&bytes).string("key length", 5).unwrap(), "hello");
    }

    #[test]
    fn invalid_utf8_points_at_the_string_bytes() {
        let mut bytes = 2_u64.to_le_bytes().to_vec();
        bytes.extend_from_slice(&[0xc3, 0x28]);
        let err = reader(&bytes).string("s", 10).unwrap_err();
        assert!(matches!(err.kind(), ErrorKind::InvalidUtf8));
        assert_eq!(err.offset(), Some(8));
    }

    #[test]
    fn fixed_vec_spans_chunks() {
        let n = CHUNK_BYTES / 4 * 3 + 5;
        let bytes: Vec<u8> = (0..u32::try_from(n).unwrap())
            .flat_map(u32::to_le_bytes)
            .collect();
        let mut r = reader(&bytes);
        let values = r.fixed_vec(n, u32::from_le_bytes).unwrap();
        assert_eq!(values.len(), n);
        assert!(values.iter().enumerate().all(|(i, &v)| v as usize == i));
    }

    #[test]
    fn bool_vec_rejects_other_bytes_with_exact_offset() {
        let mut r = reader(&[1, 0, 1, 2]);
        let err = r.bool_vec(4).unwrap_err();
        assert!(matches!(err.kind(), ErrorKind::InvalidBool(2)));
        assert_eq!(err.offset(), Some(3));
    }
}
