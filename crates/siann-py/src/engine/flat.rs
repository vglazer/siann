//! Exact search: `QB` queries at a time against each block; each thread scans the base in tiles of `TILE`
//! blocks, reused for all of its queries while in cache.

use rayon::prelude::*;

use super::Result;
use super::kernels::{L, QB, TILE, TopK, block_dists, to_blocks};
use super::metric::{Kind, Metric, with_metric};

/// Each query of a chunk's k nearest among blocks `b0..b1` of `blocks` (one `TopK` per query).
fn flat_range<M: Metric>(
    blocks: &[[f32; L]],
    d: usize,
    n: usize,
    qc: &[f32],
    k: usize,
    b0: usize,
    b1: usize,
) -> Vec<TopK> {
    let m = qc.len() / d;
    let groups = m.div_ceil(QB);
    // The queries, interleaved by groups of QB; padding queries are zero and their results dropped.
    let mut qq = vec![[0.0f32; QB]; groups * d];
    for (i, row) in qc.chunks_exact(d).enumerate() {
        let (g, a) = (i / QB, i % QB);
        for (j, &v) in row.iter().enumerate() {
            qq[g * d + j][a] = v;
        }
    }
    let mut tops: Vec<TopK> = (0..groups * QB).map(|_| TopK::new(k)).collect();
    for t0 in (b0..b1).step_by(TILE) {
        let t1 = (t0 + TILE).min(b1);
        for (g, qg) in qq.chunks_exact(d).enumerate() {
            let tg = &mut tops[g * QB..(g + 1) * QB];
            for b in t0..t1 {
                let acc = block_dists::<M>(qg, &blocks[b * d..(b + 1) * d]);
                for (top, row) in tg.iter_mut().zip(&acc) {
                    for (l, &v) in row.iter().enumerate() {
                        // Skip the padding slots past the last vector: they hold infinity, which
                        // never enters a top-k for the distances but is the best possible inner
                        // product (or NaN, against a zero coordinate).
                        let i = b * L + l;
                        if i < n {
                            top.push(v, i as u32);
                        }
                    }
                }
            }
        }
    }
    tops.truncate(m);
    tops
}

/// The k smallest of several `TopK`s of one query, by distance then index: the order one `TopK` keeps
/// when it sees every vector in index order, so that splitting the base among threads changes nothing.
fn merge_tops(parts: &[&TopK], k: usize, out: &mut [u32]) {
    let mut all: Vec<(f32, u32)> = parts
        .iter()
        .flat_map(|t| t.dist.iter().copied().zip(t.idx.iter().copied()))
        .collect();
    all.sort_unstable_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    for (o, (_, i)) in out.iter_mut().zip(all.iter().take(k)) {
        *o = *i;
    }
}

/// Exact search of `n` vectors held in `blocks`. Each thread takes `chunk` queries at a time; when there
/// are fewer chunks than threads (small batches), each chunk's scan of the base is also divided among
/// threads, by ranges of blocks, and each query's best k from the ranges merged.
pub fn flat_search<M: Metric>(
    blocks: &[[f32; L]],
    d: usize,
    n: usize,
    xq: &[f32],
    k: usize,
    chunk: usize,
    split: bool,
) -> Vec<u32> {
    let nq = xq.len() / d;
    let nb = blocks.len() / d;
    let chunks = nq.div_ceil(chunk);
    let threads = rayon::current_num_threads();
    // ranges per chunk: enough to occupy every thread, each at least one cache tile of the base
    let parts = if split {
        (threads / chunks.max(1)).clamp(1, nb.div_ceil(TILE).max(1))
    } else {
        1
    };
    let mut out = vec![0u32; nq * k];
    if parts == 1 {
        out.par_chunks_mut(chunk * k)
            .zip(xq.par_chunks(chunk * d))
            .for_each(|(o, qc)| {
                let tops = flat_range::<M>(blocks, d, n, qc, k, 0, nb);
                for (o, top) in o.chunks_exact_mut(k).zip(&tops) {
                    o.copy_from_slice(&top.idx);
                }
            });
        return out;
    }
    let step = nb.div_ceil(parts).div_ceil(TILE) * TILE;
    let ranges: Vec<(usize, usize)> = (0..nb)
        .step_by(step)
        .map(|b0| (b0, (b0 + step).min(nb)))
        .collect();
    let qcs: Vec<&[f32]> = xq.chunks(chunk * d).collect();
    let tops: Vec<Vec<TopK>> = (0..qcs.len() * ranges.len())
        .into_par_iter()
        .map(|j| {
            let (c, (b0, b1)) = (j / ranges.len(), ranges[j % ranges.len()]);
            flat_range::<M>(blocks, d, n, qcs[c], k, b0, b1)
        })
        .collect();
    out.par_chunks_mut(chunk * k)
        .enumerate()
        .for_each(|(c, o)| {
            let mine = &tops[c * ranges.len()..(c + 1) * ranges.len()];
            for (i, oq) in o.chunks_exact_mut(k).enumerate() {
                let parts: Vec<&TopK> = mine.iter().map(|r| &r[i]).collect();
                merge_tops(&parts, k, oq);
            }
        });
    out
}

/// An exact index: the base vectors in blocks.
pub struct Flat {
    pub d: usize,
    n: usize,
    metric: Kind,
    blocks: Vec<[f32; L]>,
}

impl Flat {
    /// From `x`, n rows of `d` values.
    pub fn new(x: &[f32], d: usize, metric: Kind) -> Self {
        Flat {
            d,
            n: x.len() / d,
            metric,
            blocks: to_blocks(x.chunks_exact(d), d),
        }
    }

    /// The indices of the k nearest base vectors to each query of `xq` (rows of `d`), flattened, nearest
    /// first. Each thread takes `chunk` queries at a time; with `split`, a batch of fewer chunks than
    /// threads also divides the base among threads (the same results).
    pub fn search(&self, xq: &[f32], k: usize, chunk: usize, split: bool) -> Result<Vec<u32>> {
        if k == 0 || k > self.n || chunk == 0 {
            return Err("need 0 < k <= n and chunk > 0".into());
        }
        Ok(
            with_metric!(self.metric, M => flat_search::<M>(&self.blocks, self.d, self.n, xq, k, chunk, split)),
        )
    }
}
