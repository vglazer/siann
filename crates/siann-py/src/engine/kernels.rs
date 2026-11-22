//! The blocked layout and the distance loops over one block, shared by exact search and the IVF, and the
//! top-k they feed.
//!
//! Vectors are stored in blocks of `L`, dimension by dimension (`blocks[b * d + j][l]` is dimension
//! `j` of vector `b * L + l`). The inner loop computes queries against one block, keeping one
//! running distance per (query, vector) pair: a sum for L1 and L2, a maximum for L-infinity. They are
//! independent, so the compiler can map them onto NEON (Apple's vector instructions) without
//! reordering any floating-point additions. Padding vectors hold infinity and never enter a top-k.

use super::elem::Elem;
use super::metric::Metric;

/// Vectors per block.
pub const L: usize = 8;
/// Queries per register block in exact search.
pub const QB: usize = 4;
/// Blocks per cache tile in exact search: 128 blocks of 8 vectors of 128 dimensions is 512 KB.
pub const TILE: usize = 128;

/// The distances between `QB` queries (interleaved: `qq[j][a]` is dimension `j` of query `a`) and
/// the `L` vectors of one block.
#[inline]
pub fn block_dists<M: Metric>(qq: &[[f32; QB]], blk: &[[f32; L]]) -> [[f32; L]; QB] {
    let mut acc = [[0.0f32; L]; QB];
    for (q, x) in qq.iter().zip(blk) {
        for a in 0..QB {
            for l in 0..L {
                acc[a][l] = M::combine(acc[a][l], q[a], x[l]);
            }
        }
    }
    acc
}

/// The distances between one query and the `L` vectors of one block. With one query there are only
/// `L` running distances, so each is split four ways by dimension, to keep the operations from
/// waiting on each other; the four parts are joined at the end.
#[inline]
pub fn block_dists_one<M: Metric>(q: &[f32], blk: &[[f32; L]]) -> [f32; L] {
    let mut acc = [[0.0f32; L]; 4];
    let (q4, q_rest) = q.as_chunks::<4>();
    let (b4, b_rest) = blk.as_chunks::<4>();
    for (qs, xs) in q4.iter().zip(b4) {
        for p in 0..4 {
            for l in 0..L {
                acc[p][l] = M::combine(acc[p][l], qs[p], xs[p][l]);
            }
        }
    }
    for (&qv, x) in q_rest.iter().zip(b_rest) {
        for l in 0..L {
            acc[0][l] = M::combine(acc[0][l], qv, x[l]);
        }
    }
    let mut out = [0.0f32; L];
    for l in 0..L {
        out[l] = M::join(M::join(acc[0][l], acc[1][l]), M::join(acc[2][l], acc[3][l]));
    }
    out
}

/// `block_dists_one` reading the block in its stored type, converting each value in registers inside
/// the loop instead of widening the block into a buffer first.
#[inline]
pub fn block_dists_one_direct<M: Metric, T: Elem>(q: &[f32], blk: &[[T; L]]) -> [f32; L] {
    let mut acc = [[0.0f32; L]; 4];
    let (q4, q_rest) = q.as_chunks::<4>();
    let (b4, b_rest) = blk.as_chunks::<4>();
    for (qs, xs) in q4.iter().zip(b4) {
        for p in 0..4 {
            for l in 0..L {
                acc[p][l] = M::combine(acc[p][l], qs[p], T::lane(xs[p][l]));
            }
        }
    }
    for (&qv, x) in q_rest.iter().zip(b_rest) {
        for l in 0..L {
            acc[0][l] = M::combine(acc[0][l], qv, T::lane(x[l]));
        }
    }
    let mut out = [0.0f32; L];
    for l in 0..L {
        out[l] = M::join(M::join(acc[0][l], acc[1][l]), M::join(acc[2][l], acc[3][l]));
    }
    out
}

/// The k smallest distances seen so far, in increasing order; equal distances by increasing id, so that
/// the result does not depend on the order in which vectors are pushed.
pub struct TopK {
    pub dist: Vec<f32>,
    pub idx: Vec<u32>,
}

impl TopK {
    pub fn new(k: usize) -> Self {
        TopK {
            dist: vec![f32::INFINITY; k],
            idx: vec![u32::MAX; k],
        }
    }

    #[inline]
    pub fn push(&mut self, v: f32, i: u32) {
        let k = self.dist.len();
        let last = self.dist[k - 1];
        if v > last || (v == last && i >= self.idx[k - 1]) {
            return;
        }
        let mut p = k - 1;
        while p > 0 && (self.dist[p - 1] > v || (self.dist[p - 1] == v && self.idx[p - 1] > i)) {
            self.dist[p] = self.dist[p - 1];
            self.idx[p] = self.idx[p - 1];
            p -= 1;
        }
        self.dist[p] = v;
        self.idx[p] = i;
    }
}

/// Vectors (rows, each `d` long) in blocks of `L`, dimension by dimension, padded with `T::PAD`.
pub fn to_blocks<'a, T: Elem + 'a>(
    rows: impl ExactSizeIterator<Item = &'a [T]>,
    d: usize,
) -> Vec<[T; L]> {
    let nb = rows.len().div_ceil(L);
    let mut blocks = vec![[T::PAD; L]; nb * d];
    for (i, row) in rows.enumerate() {
        let (b, l) = (i / L, i % L);
        for (j, &v) in row.iter().enumerate() {
            blocks[b * d + j][l] = v;
        }
    }
    blocks
}
