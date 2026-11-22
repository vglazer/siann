//! How list vectors are stored: as f32; as float16 bits (`u16`), half the memory traffic; or as bytes
//! (`u8`), a quarter, for vectors whose coordinates are integers from 0 to 255 (SIFT's). Either way,
//! distances are computed in f32 from the stored values, widened exactly.

use super::kernels::L;
use super::metric::Metric;

/// A storage type for list vectors.
pub trait Elem: Copy + Send + Sync {
    /// The value of padding slots.
    const PAD: Self;
    /// A block as f32: the block itself for f32; for float16, widened into `buf`. Widening a whole
    /// block first, in one flat loop, then running the f32 code on it was 1.3-1.9 times faster on
    /// SIFT1M than widening inside the distance loops, where the compiler gathered the halves one lane
    /// at a time (BENCHMARKS.md, "List storage").
    fn as_f32<'a>(blk: &'a [[Self; L]], buf: &'a mut Vec<[f32; L]>) -> &'a [[f32; L]];
    /// One stored value as f32, for converting in registers inside a distance loop.
    fn lane(x: Self) -> f32;
    /// Whether single-query scans convert in registers (`lane`) rather than through a buffer
    /// (`as_f32`). Bytes convert in two vector instructions and gain; float16's bit operations make
    /// the compiler load the halves one lane at a time, so it keeps the buffer.
    const DIRECT: bool;
    /// For byte lists: the query as bytes, if every coordinate is an integer from 0 to 255 and `d` is
    /// small enough for the integer distances to be exact as float32 (d * 255^2 below 2^24, so they
    /// rank exactly, and equal the float path's, whose partial sums of integers are exact too);
    /// otherwise None, and the float path is used. None for other storages.
    fn integer_query(_q: &[f32]) -> Option<Vec<u8>> {
        None
    }
    /// Whether lists also keep their vectors row by row (each vector's coordinates contiguous),
    /// for the integer path.
    const ROW_MAJOR: bool = false;
    /// For byte lists with an integer query: one vector's distance in integer arithmetic.
    fn int_dist<M: Metric>(_q: &[u8], _x: &[Self]) -> u32 {
        unreachable!("integer distances are only for byte lists")
    }
}

impl Elem for f32 {
    /// Infinity: padding vectors never enter a top-k.
    const PAD: f32 = f32::INFINITY;
    const DIRECT: bool = true;
    #[inline(always)]
    fn as_f32<'a>(blk: &'a [[f32; L]], _buf: &'a mut Vec<[f32; L]>) -> &'a [[f32; L]] {
        blk
    }
    #[inline(always)]
    fn lane(x: f32) -> f32 {
        x
    }
}

impl Elem for u8 {
    /// Zero: list scans skip padding slots by their missing ids.
    const PAD: u8 = 0;
    const DIRECT: bool = true;
    #[inline]
    fn as_f32<'a>(blk: &'a [[u8; L]], buf: &'a mut Vec<[f32; L]>) -> &'a [[f32; L]] {
        buf.resize(blk.len(), [0.0; L]);
        for (o, &h) in buf.as_flattened_mut().iter_mut().zip(blk.as_flattened()) {
            *o = f32::from(h);
        }
        buf
    }
    #[inline(always)]
    fn lane(x: u8) -> f32 {
        f32::from(x)
    }
    fn integer_query(q: &[f32]) -> Option<Vec<u8>> {
        // L2 sums reach d * 255^2, exact in float32 (and so in the top-k) below 2^24: d <= 258.
        if q.len() > 258
            || !q
                .iter()
                .all(|&v| (0.0..=255.0).contains(&v) && v.fract() == 0.0)
        {
            return None;
        }
        Some(q.iter().map(|&v| v as u8).collect())
    }
    /// Byte lists keep a row-major copy for integer queries: in the blocked layout a dimension's 8
    /// bytes fill half a register, and the integer kernel there was no faster than converting to
    /// float; row by row, 16 of a vector's bytes fill one, and the sum is one dot product.
    const ROW_MAJOR: bool = true;
    #[inline]
    fn int_dist<M: Metric>(q: &[u8], x: &[u8]) -> u32 {
        M::dist_int(q, x)
    }
}

/// 2^112: the difference between float32's exponent bias (127) and float16's (15).
const HALF_SCALE: f32 = f32::from_bits((127 + 112) << 23);

/// A finite float16 (its bits) as f32, exactly: its sign, exponent and mantissa moved into f32's
/// positions read as an f32 2^112 times too small, normal or subnormal, so one exact multiply
/// finishes it. Branch-free, so the compiler vectorizes it. Infinity and NaN (exponent 31) would come
/// out finite, so lists reject them when built.
#[inline(always)]
pub fn widen_half(h: u16) -> f32 {
    let h = u32::from(h);
    f32::from_bits(((h & 0x8000) << 16) | ((h & 0x7fff) << 13)) * HALF_SCALE
}

impl Elem for u16 {
    /// Zero: float16 has no padding value that widens to infinity here, so list scans skip padding
    /// slots by their missing ids instead.
    const PAD: u16 = 0;
    const DIRECT: bool = false;
    #[inline]
    fn as_f32<'a>(blk: &'a [[u16; L]], buf: &'a mut Vec<[f32; L]>) -> &'a [[f32; L]] {
        buf.resize(blk.len(), [0.0; L]);
        // Flat slices: over nested [u16; L] arrays the compiler loads the halves one lane at a time.
        for (o, &h) in buf.as_flattened_mut().iter_mut().zip(blk.as_flattened()) {
            *o = widen_half(h);
        }
        buf
    }
    #[inline(always)]
    fn lane(x: u16) -> f32 {
        widen_half(x)
    }
}
