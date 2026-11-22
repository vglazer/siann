//! siann's CPU paths, in safe Rust: exact and IVF k-nearest neighbors for L1, L2, L-infinity and inner
//! product, the top-k that ranks BLAS products for L2 and inner product, and the scores of returned
//! neighbors. Built into the siann Python package as `siann._cpu`.
//!
//! Vectors are stored in blocks of `L`, dimension by dimension (`blocks[b * d + j][l]` is dimension
//! `j` of vector `b * L + l`). The inner loop computes queries against one block, keeping one
//! running distance per (query, vector) pair: a sum for L1 and L2, a maximum for L-infinity. They are
//! independent, so the compiler can map them onto NEON (Apple's vector instructions) without
//! reordering any floating-point additions. Padding vectors hold infinity and never enter a top-k.
//!
//! - `FlatIndex`: exact search. `QB` queries at a time against each block; each thread scans the
//!   base in tiles of `TILE` blocks, reused for all of its queries while in cache.
//! - `IvfIndex`: inverted lists. Each base vector belongs to its nearest centroid's list, stored in
//!   blocks. A query finds its `nprobe` nearest centroids, then scans their lists. One thread per
//!   query, so that single queries are cheap and batches spread across cores.

use numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::buffer::PyBuffer;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

/// The thread pool for split queries of `threads` threads, made once and kept. Split queries run on
/// a small pool of their own: on rayon's global pool (every core), a query injected into the pool
/// wakes its sleeping threads, which costs more than the split saves (BENCHMARKS.md, "Approximate search: single queries").
fn split_pool(threads: usize) -> PyResult<Arc<rayon::ThreadPool>> {
    static POOLS: OnceLock<Mutex<HashMap<usize, Arc<rayon::ThreadPool>>>> = OnceLock::new();
    let mut pools = POOLS
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .map_err(|_| PyValueError::new_err("split pool lock poisoned"))?;
    if let Some(pool) = pools.get(&threads) {
        return Ok(Arc::clone(pool));
    }
    let pool = Arc::new(
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .thread_name(|i| format!("siann-split-{i}"))
            .build()
            .map_err(|e| PyValueError::new_err(format!("cannot start split pool: {e}")))?,
    );
    pools.insert(threads, Arc::clone(&pool));
    Ok(pool)
}

/// Vectors per block.
const L: usize = 8;
/// Queries per register block in exact search.
const QB: usize = 4;
/// Blocks per cache tile in exact search: 128 blocks of 8 vectors of 128 dimensions is 512 KB.
const TILE: usize = 128;

/// How a metric builds a distance one coordinate at a time. L2 distances are squared.
trait Metric {
    /// Whether the integer path for byte lists applies: true for the distances; not for the inner
    /// product, whose values are negative.
    const INTEGER: bool = true;
    /// The running distance `acc` with the coordinates `a` and `b` added.
    fn combine(acc: f32, a: f32, b: f32) -> f32;
    /// Two running distances over disjoint coordinates, joined.
    fn join(x: f32, y: f32) -> f32;
    /// The distance between two byte vectors, in integers: exact. A plain loop over each vector's
    /// contiguous bytes, which the compiler may reorder (integer sums are associative), and so
    /// vectorizes 16 bytes at a time (`uabd`, then `udot` for L2).
    fn dist_int(q: &[u8], x: &[u8]) -> u32;
}

/// The sum of |a - b|.
struct L1;
/// The sum of (a - b)^2.
struct L2;
/// The largest |a - b|.
struct Linf;
/// The inner product, negated (the sum of -a b), so that, as for the distances, smaller is nearer.
struct Ip;

impl Metric for L1 {
    #[inline(always)]
    fn combine(acc: f32, a: f32, b: f32) -> f32 {
        acc + (a - b).abs()
    }
    #[inline(always)]
    fn join(x: f32, y: f32) -> f32 {
        x + y
    }
    #[inline(always)]
    fn dist_int(q: &[u8], x: &[u8]) -> u32 {
        q.iter()
            .zip(x)
            .map(|(&a, &b)| u32::from(a.abs_diff(b)))
            .sum()
    }
}

impl Metric for L2 {
    #[inline(always)]
    fn combine(acc: f32, a: f32, b: f32) -> f32 {
        let e = a - b;
        acc + e * e
    }
    #[inline(always)]
    fn join(x: f32, y: f32) -> f32 {
        x + y
    }
    #[inline(always)]
    fn dist_int(q: &[u8], x: &[u8]) -> u32 {
        q.iter()
            .zip(x)
            .map(|(&a, &b)| {
                let e = u32::from(a.abs_diff(b));
                e * e
            })
            .sum()
    }
}

// A comparison and select rather than `f32::max`: LLVM treats `max` as a reduction it may reorder,
// and then vectorizes the wrong loop (over dimensions, with shuffles and a 3.7 KB stack frame),
// taking twice L1's time; so does an unsigned-integer max on the bits. The select keeps one running
// maximum per lane, like L1's sums (fcmgt and bit where L1 has one fadd: 1.35 times L1's time on
// 200,000 Gaussian vectors, against 2.1 with `max`). The inputs hold no NaN, so the two agree.
#[inline(always)]
fn larger(x: f32, y: f32) -> f32 {
    if y > x { y } else { x }
}

impl Metric for Linf {
    #[inline(always)]
    fn combine(acc: f32, a: f32, b: f32) -> f32 {
        larger(acc, (a - b).abs())
    }
    #[inline(always)]
    fn join(x: f32, y: f32) -> f32 {
        larger(x, y)
    }
    #[inline(always)]
    fn dist_int(q: &[u8], x: &[u8]) -> u32 {
        u32::from(
            q.iter()
                .zip(x)
                .map(|(&a, &b)| a.abs_diff(b))
                .max()
                .unwrap_or(0),
        )
    }
}

impl Metric for Ip {
    const INTEGER: bool = false;
    #[inline(always)]
    fn combine(acc: f32, a: f32, b: f32) -> f32 {
        acc - a * b
    }
    #[inline(always)]
    fn join(x: f32, y: f32) -> f32 {
        x + y
    }
    fn dist_int(_q: &[u8], _x: &[u8]) -> u32 {
        unreachable!("the integer path is not used for the inner product")
    }
}

/// The metric an index was built with, chosen at run time; each search dispatches on it once.
#[derive(Clone, Copy)]
enum Kind {
    L1,
    L2,
    Linf,
    Ip,
}

/// Runs `$body` with `$m` the metric type for `$kind`.
macro_rules! with_metric {
    ($kind:expr, $m:ident => $body:expr) => {
        match $kind {
            Kind::L1 => {
                type $m = L1;
                $body
            }
            Kind::L2 => {
                type $m = L2;
                $body
            }
            Kind::Linf => {
                type $m = Linf;
                $body
            }
            Kind::Ip => {
                type $m = Ip;
                $body
            }
        }
    };
}

/// How list vectors are stored: as f32; as float16 bits (`u16`), half the memory traffic; or as
/// bytes (`u8`), a quarter, for vectors whose coordinates are integers from 0 to 255 (SIFT's). Either
/// way, distances are computed in f32 from the stored values, widened exactly.
trait Elem: Copy + Send + Sync {
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
fn widen_half(h: u16) -> f32 {
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

/// The distances between `QB` queries (interleaved: `qq[j][a]` is dimension `j` of query `a`) and
/// the `L` vectors of one block.
#[inline]
fn block_dists<M: Metric>(qq: &[[f32; QB]], blk: &[[f32; L]]) -> [[f32; L]; QB] {
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
fn block_dists_one<M: Metric>(q: &[f32], blk: &[[f32; L]]) -> [f32; L] {
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
fn block_dists_one_direct<M: Metric, T: Elem>(q: &[f32], blk: &[[T; L]]) -> [f32; L] {
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
struct TopK {
    dist: Vec<f32>,
    idx: Vec<u32>,
}

impl TopK {
    fn new(k: usize) -> Self {
        TopK {
            dist: vec![f32::INFINITY; k],
            idx: vec![u32::MAX; k],
        }
    }

    #[inline]
    fn push(&mut self, v: f32, i: u32) {
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
fn to_blocks<'a, T: Elem + 'a>(
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
fn flat_search<M: Metric>(
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

fn parse_metric(metric: &str) -> PyResult<Kind> {
    match metric {
        "l1" => Ok(Kind::L1),
        "l2" => Ok(Kind::L2),
        "linf" => Ok(Kind::Linf),
        "ip" => Ok(Kind::Ip),
        _ => Err(PyValueError::new_err(
            "metric must be 'l1', 'l2', 'linf' or 'ip' (inner product)",
        )),
    }
}

/// Each query's nearest `nprobe` lists from `products` (nq, nlist), its products with the centroids, as
/// BLAS computes them for a batch: ranked by |c|^2 - 2 q.c for L2 (the same order as the distance) and by
/// -q.c for inner product, selected as the Rust coarse step selects (in linear time, in no set order).
fn probes_from_products(
    products: &PyReadonlyArray2<'_, f32>,
    nq: usize,
    cnorms: &[f32],
    metric: Kind,
    nprobe: usize,
) -> PyResult<Vec<u32>> {
    let nlist = cnorms.len();
    let (rows, cols) = products.as_array().dim();
    let p = products
        .as_slice()
        .map_err(|_| PyValueError::new_err("products must be C-contiguous"))?;
    if rows != nq || cols != nlist {
        return Err(PyValueError::new_err("products must be (nq, nlist)"));
    }
    let l2 = match metric {
        Kind::L2 => true,
        Kind::Ip => false,
        _ => {
            return Err(PyValueError::new_err(
                "products give the coarse step for L2 and inner product only",
            ));
        }
    };
    let nprobe = nprobe.min(nlist);
    let mut out = vec![0u32; nq * nprobe];
    out.par_chunks_mut(nprobe)
        .zip(p.par_chunks(nlist))
        .for_each(|(o, row)| {
            let mut dist: Vec<(f32, u32)> = row
                .iter()
                .zip(cnorms)
                .enumerate()
                .map(|(c, (&pv, &n))| (if l2 { n - 2.0 * pv } else { -pv }, c as u32))
                .collect();
            if nprobe < nlist {
                dist.select_nth_unstable_by(nprobe - 1, |a, b| a.0.total_cmp(&b.0));
            }
            for (o, &(_, c)) in o.iter_mut().zip(&dist[..nprobe]) {
                *o = c;
            }
        });
    Ok(out)
}

fn read(py: Python<'_>, buf: &Bound<'_, PyAny>, d: usize) -> PyResult<Vec<f32>> {
    let buf = PyBuffer::<f32>::get(buf)?;
    let v = buf.to_vec(py)?;
    if d == 0 || v.len() % d != 0 {
        return Err(PyValueError::new_err(
            "expected a C-contiguous float32 array with d columns",
        ));
    }
    Ok(v)
}

/// Exact search.
#[pyclass]
struct FlatIndex {
    d: usize,
    n: usize,
    metric: Kind,
    blocks: Vec<[f32; L]>,
}

#[pymethods]
impl FlatIndex {
    /// Build from a C-contiguous float32 array of shape (n, d); `metric` is "l1", "l2", "linf" or "ip" (inner product, largest first).
    #[new]
    #[pyo3(signature = (xb, d, metric = "l1"))]
    fn py_new(py: Python<'_>, xb: &Bound<'_, PyAny>, d: usize, metric: &str) -> PyResult<Self> {
        let metric = parse_metric(metric)?;
        let v = read(py, xb, d)?;
        let n = v.len() / d;
        let blocks = py.detach(|| to_blocks(v.chunks_exact(d), d));
        Ok(FlatIndex {
            d,
            n,
            metric,
            blocks,
        })
    }

    /// The indices of the k nearest base vectors to each query, flattened (nq * k), nearest first.
    /// Each thread takes `chunk` queries at a time; with `split`, a batch of fewer chunks than threads
    /// also divides the base among threads (the same results).
    #[pyo3(signature = (xq, k, chunk = 32, split = true))]
    fn search<'py>(
        &self,
        py: Python<'py>,
        xq: &Bound<'_, PyAny>,
        k: usize,
        chunk: usize,
        split: bool,
    ) -> PyResult<Bound<'py, PyArray1<u32>>> {
        if k == 0 || k > self.n || chunk == 0 {
            return Err(PyValueError::new_err("need 0 < k <= n and chunk > 0"));
        }
        let v = read(py, xq, self.d)?;
        let out = py.detach(
            || with_metric!(self.metric, M => flat_search::<M>(&self.blocks, self.d, self.n, &v, k, chunk, split)),
        );
        Ok(out.into_pyarray(py))
    }
}

/// One inverted list: its vectors in blocks, and their ids in the base.
struct List<T> {
    blocks: Vec<[T; L]>,
    ids: Vec<u32>,
    /// The vectors row by row, if `T::ROW_MAJOR` (byte lists, for integer queries); else empty.
    rows: Vec<T>,
}

/// The inverted lists, stored as f32 or as float16.
enum Lists {
    F32(Vec<List<f32>>),
    F16(Vec<List<u16>>),
    U8(Vec<List<u8>>),
}

/// The vectors given for the lists, in their storage type.
enum Stored {
    F32,
    F16(Vec<u16>),
    U8(Vec<u8>),
}

/// Runs `$body` with `$lists` the lists, whichever their storage.
macro_rules! with_lists {
    ($lists:expr, $l:ident => $body:expr) => {
        match $lists {
            Lists::F32($l) => $body,
            Lists::F16($l) => $body,
            Lists::U8($l) => $body,
        }
    };
}

/// Inverted lists over given centroids.
#[pyclass]
struct IvfIndex {
    d: usize,
    metric: Kind,
    nlist: usize,
    centroids: Vec<[f32; L]>,
    /// The centroids' squared norms, for a coarse step from BLAS products.
    cnorms: Vec<f32>,
    lists: Lists,
    /// For a two-level coarse step (`set_coarse_groups`), the centroids clustered into groups.
    groups: Option<Groups>,
}

/// How one query is searched: lists to scan, parallel tasks (one: this thread), and groups for a
/// two-level coarse step (zero: exact coarse step).
#[derive(Clone, Copy)]
struct Probe {
    nprobe: usize,
    tasks: usize,
    gprobe: usize,
}

/// The centroids clustered into groups, for a two-level coarse step: a query is compared with the
/// groups' centres, then only with the centroids of its `gprobe` nearest groups.
struct Groups {
    /// The groups' centres, in blocks.
    centres: Vec<[f32; L]>,
    count: usize,
    /// Each group's centroids, in blocks, and their ids.
    blocks: Vec<Vec<[f32; L]>>,
    ids: Vec<Vec<u32>>,
}

/// Each list's members (ids), in blocks.
fn make_lists<T: Elem>(members: Vec<Vec<u32>>, x: &[T], d: usize) -> Vec<List<T>> {
    members
        .into_par_iter()
        .map(|ids| {
            let row = |i: u32| &x[i as usize * d..(i as usize + 1) * d];
            let blocks = to_blocks(ids.iter().map(|&i| row(i)), d);
            let rows = if T::ROW_MAJOR {
                ids.iter().flat_map(|&i| row(i).iter().copied()).collect()
            } else {
                Vec::new()
            };
            List { blocks, ids, rows }
        })
        .collect()
}

impl IvfIndex {
    /// The distances from one query to every centroid, with their ids, in `tasks` parallel tasks
    /// (one: on this thread).
    fn coarse<M: Metric>(&self, q: &[f32], tasks: usize, gprobe: usize) -> Vec<(f32, u32)> {
        let d = self.d;
        if let (Some(g), true) = (&self.groups, gprobe > 0) {
            // Two levels: the nearest gprobe groups, then their centroids only.
            let mut near: Vec<(f32, u32)> = g
                .centres
                .chunks_exact(d)
                .enumerate()
                .flat_map(|(b, blk)| {
                    block_dists_one::<M>(q, blk)
                        .into_iter()
                        .enumerate()
                        .map(move |(l, v)| (v, (b * L + l) as u32))
                })
                .collect();
            near.truncate(g.count);
            let gprobe = gprobe.min(g.count);
            if gprobe < near.len() {
                near.select_nth_unstable_by(gprobe - 1, |a, b| a.0.total_cmp(&b.0));
            }
            let mut out = Vec::new();
            for &(_, gi) in &near[..gprobe] {
                let (blocks, ids) = (&g.blocks[gi as usize], &g.ids[gi as usize]);
                for (b, blk) in blocks.chunks_exact(d).enumerate() {
                    for (l, v) in block_dists_one::<M>(q, blk).into_iter().enumerate() {
                        if let Some(&c) = ids.get(b * L + l) {
                            out.push((v, c));
                        }
                    }
                }
            }
            return out;
        }
        let block = |(b, blk): (usize, &[[f32; L]])| {
            block_dists_one::<M>(q, blk)
                .into_iter()
                .enumerate()
                .map(move |(l, v)| (v, (b * L + l) as u32))
        };
        let mut coarse: Vec<(f32, u32)> = if tasks > 1 {
            let per = (self.centroids.len() / d).div_ceil(tasks);
            self.centroids
                .par_chunks(d * per)
                .enumerate()
                .flat_map_iter(|(c, chunk)| {
                    chunk
                        .chunks_exact(d)
                        .enumerate()
                        .flat_map(move |(i, blk)| block((c * per + i, blk)))
                })
                .collect()
        } else {
            self.centroids
                .chunks_exact(d)
                .enumerate()
                .flat_map(block)
                .collect()
        };
        coarse.truncate(self.nlist);
        coarse
    }

    /// The vectors of one list, pushed into `top` in order.
    fn scan_one<M: Metric, T: Elem>(
        &self,
        list: &List<T>,
        q: &[f32],
        q8: Option<&[u8]>,
        top: &mut TopK,
        buf: &mut Vec<[f32; L]>,
    ) {
        if let Some(q8) = q8 {
            // integer query, byte list: exact integer distances, vector by vector
            for (x, &id) in list.rows.chunks_exact(self.d).zip(&list.ids) {
                top.push(T::int_dist::<M>(q8, x) as f32, id);
            }
            return;
        }
        for (b, blk) in list.blocks.chunks_exact(self.d).enumerate() {
            let dists = if T::DIRECT {
                block_dists_one_direct::<M, T>(q, blk)
            } else {
                block_dists_one::<M>(q, T::as_f32(blk, buf))
            };
            for (l, v) in dists.into_iter().enumerate() {
                if let Some(&id) = list.ids.get(b * L + l) {
                    top.push(v, id);
                }
            }
        }
    }

    /// One query. With `tasks` above one (run on a split pool of that many threads), the coarse step
    /// and the list scans are spread over that many parallel tasks, each scanning a contiguous run of
    /// the probed lists, and the tasks' best k merged: the same result as one thread, ties included,
    /// as a best k breaks ties by id whatever the order of the scan.
    fn search_one<M: Metric, T: Elem>(
        &self,
        lists: &[List<T>],
        q: &[f32],
        k: usize,
        how: Probe,
        given: Option<&[u32]>,
        out: &mut [u32],
    ) {
        let Probe {
            nprobe,
            tasks,
            gprobe,
        } = how;
        // Byte lists with an integer query take the exact integer path.
        let q8 = if M::INTEGER {
            T::integer_query(q)
        } else {
            None
        };
        // The nearest nprobe centroids (among those the two-level step examined, with gprobe), unless
        // the caller found them (a batch's coarse step through BLAS).
        let found: Vec<u32>;
        let probes: &[u32] = match given {
            Some(p) => p,
            None => {
                let mut coarse = self.coarse::<M>(q, tasks, gprobe);
                let nprobe = nprobe.min(coarse.len());
                if nprobe < coarse.len() {
                    coarse.select_nth_unstable_by(nprobe - 1, |a, b| a.0.total_cmp(&b.0));
                }
                found = coarse[..nprobe].iter().map(|&(_, c)| c).collect();
                &found
            }
        };
        // Scan their lists.
        let mut top = TopK::new(k);
        if tasks > 1 {
            let parts: Vec<TopK> = probes
                .par_chunks(probes.len().div_ceil(tasks).max(1))
                .map(|run| {
                    let mut t = TopK::new(k);
                    let mut buf = Vec::new();
                    for &c in run {
                        self.scan_one::<M, T>(
                            &lists[c as usize],
                            q,
                            q8.as_deref(),
                            &mut t,
                            &mut buf,
                        );
                    }
                    t
                })
                .collect();
            for t in &parts {
                for (&v, &id) in t.dist.iter().zip(&t.idx) {
                    top.push(v, id);
                }
            }
        } else {
            let mut buf = Vec::new();
            for &c in probes {
                self.scan_one::<M, T>(&lists[c as usize], q, q8.as_deref(), &mut top, &mut buf);
            }
        }
        out.copy_from_slice(&top.idx);
    }

    /// One list against the queries that probe it, `QB` at a time: the best k in this list for each
    /// of those queries, as (query, distances and ids).
    fn scan_list<M: Metric, T: Elem>(
        &self,
        list: &List<T>,
        queries: &[u32],
        xq: &[f32],
        k: usize,
    ) -> Vec<(u32, TopK)> {
        let d = self.d;
        let mut out = Vec::with_capacity(queries.len());
        let mut qq = vec![[0.0f32; QB]; d];
        let mut buf = Vec::new();
        for group in queries.chunks(QB) {
            for slot in qq.iter_mut() {
                *slot = [0.0; QB];
            }
            for (a, &qi) in group.iter().enumerate() {
                for (j, &v) in xq[qi as usize * d..(qi as usize + 1) * d]
                    .iter()
                    .enumerate()
                {
                    qq[j][a] = v;
                }
            }
            let mut tops: Vec<TopK> = (0..group.len()).map(|_| TopK::new(k)).collect();
            for (b, blk) in list.blocks.chunks_exact(d).enumerate() {
                let acc = block_dists::<M>(&qq, T::as_f32(blk, &mut buf));
                for (top, row) in tops.iter_mut().zip(&acc) {
                    for (l, &v) in row.iter().enumerate() {
                        if let Some(&id) = list.ids.get(b * L + l) {
                            top.push(v, id);
                        }
                    }
                }
            }
            out.extend(group.iter().copied().zip(tops));
        }
        out
    }

    fn search_list_major_impl<M: Metric, T: Elem>(
        &self,
        lists: &[List<T>],
        xq: &[f32],
        k: usize,
        nprobe: usize,
        given: Option<&[u32]>,
    ) -> Vec<u32> {
        let d = self.d;
        let nq = xq.len() / d;
        // Each query's nearest nprobe centroids, by exact search over the centroids, unless given.
        let (nprobe, probes) = match given {
            Some(p) => (p.len() / nq.max(1), p.to_vec()),
            None => {
                let nprobe = nprobe.min(self.nlist);
                let found = flat_search::<M>(&self.centroids, d, self.nlist, xq, nprobe, 32, false);
                (nprobe, found)
            }
        };
        // For each list, the queries that probe it.
        let mut members: Vec<Vec<u32>> = vec![Vec::new(); self.nlist];
        for (qi, ps) in probes.chunks_exact(nprobe).enumerate() {
            for &c in ps {
                if let Some(m) = members.get_mut(c as usize) {
                    m.push(qi as u32);
                }
            }
        }
        // Scan each list once for all of its queries.
        let partial: Vec<Vec<(u32, TopK)>> = lists
            .par_iter()
            .zip(members.par_iter())
            .map(|(list, queries)| self.scan_list::<M, T>(list, queries, xq, k))
            .collect();
        // Merge: each query's best k over its lists.
        let mut tops: Vec<TopK> = (0..nq).map(|_| TopK::new(k)).collect();
        for (qi, t) in partial.iter().flatten() {
            let top = &mut tops[*qi as usize];
            for (&v, &id) in t.dist.iter().zip(&t.idx) {
                top.push(v, id);
            }
        }
        tops.iter().flat_map(|t| t.idx.iter().copied()).collect()
    }
}

#[pymethods]
impl IvfIndex {
    /// Build from base vectors (n, d) and centroids (nlist, d), both C-contiguous float32. Each base
    /// vector goes to the list of its nearest centroid in `assign` (by default `metric`, except for
    /// inner product, whose lists are assigned by L2: by inner product, centroids pointing towards long
    /// vectors attract much of the data), found by exact search; queries probe by `metric`. The lists
    /// store `xb` itself (`storage="float32"`), or `stored`, the same vectors in a smaller type:
    /// with "float16", rounded to float16 (a C-contiguous uint16 array of their bits, such as NumPy's
    /// `xb.astype(np.float16).view(np.uint16)`), half the memory traffic; with "uint8", as bytes (a
    /// uint8 array), a quarter, which needs every coordinate of `xb` to be an integer from 0 to 255,
    /// so that nothing is lost. Distances are computed in f32 from the stored values. `lists`, if given
    /// (int64, one list per base vector, as `siann.ivf_build` assigns them), replaces the assignment,
    /// so that the CPU and GPU IVFs hold the same lists, near-ties included.
    #[new]
    #[pyo3(signature = (xb, centroids, d, metric = "l1", stored = None, storage = "float32", assign = None, lists = None))]
    #[allow(clippy::too_many_arguments)] // PyO3 takes the keyword arguments as parameters
    fn py_new(
        py: Python<'_>,
        xb: &Bound<'_, PyAny>,
        centroids: &Bound<'_, PyAny>,
        d: usize,
        metric: &str,
        stored: Option<&Bound<'_, PyAny>>,
        storage: &str,
        assign: Option<&str>,
        lists: Option<PyReadonlyArray1<'_, i64>>,
    ) -> PyResult<Self> {
        let kind = parse_metric(metric)?;
        let assign_kind = match assign {
            Some(a) => parse_metric(a)?,
            None if matches!(kind, Kind::Ip) => Kind::L2,
            None => kind,
        };
        let xb = read(py, xb, d)?;
        let shape_err = || PyValueError::new_err("stored must have the shape of xb");
        let copy = match (storage, stored) {
            ("float32", None) => Stored::F32,
            ("float16", Some(buf)) => {
                let h = PyBuffer::<u16>::get(buf)?.to_vec(py)?;
                if h.len() != xb.len() {
                    return Err(shape_err());
                }
                if h.iter().any(|&b| (b >> 10) & 0x1f == 0x1f) {
                    return Err(PyValueError::new_err(
                        "stored holds infinity or NaN: values beyond float16's range (65504)",
                    ));
                }
                Stored::F16(h)
            }
            ("uint8", Some(buf)) => {
                let h = PyBuffer::<u8>::get(buf)?.to_vec(py)?;
                if h.len() != xb.len() {
                    return Err(shape_err());
                }
                if h.iter().zip(&xb).any(|(&b, &v)| f32::from(b) != v) {
                    return Err(PyValueError::new_err(
                        "uint8 lists need xb's coordinates to be integers from 0 to 255, equal to stored",
                    ));
                }
                Stored::U8(h)
            }
            _ => {
                return Err(PyValueError::new_err(
                    "storage is 'float32' (no stored), or 'float16' or 'uint8' with stored",
                ));
            }
        };
        let cents = read(py, centroids, d)?;
        let nlist = cents.len() / d;
        if nlist == 0 {
            return Err(PyValueError::new_err("need at least one centroid"));
        }
        let given: Option<Vec<u32>> = match lists {
            Some(a) => {
                let a = a
                    .as_slice()
                    .map_err(|_| PyValueError::new_err("lists must be contiguous"))?;
                if a.len() != xb.len() / d || a.iter().any(|&l| l < 0 || l as usize >= nlist) {
                    return Err(PyValueError::new_err(
                        "lists must give each base vector a list in range",
                    ));
                }
                Some(a.iter().map(|&l| l as u32).collect())
            }
            None => None,
        };
        Ok(py.detach(|| {
            let centroids = to_blocks(cents.chunks_exact(d), d);
            let assign = given.unwrap_or_else(|| {
                with_metric!(assign_kind, M => flat_search::<M>(&centroids, d, nlist, &xb, 1, 256, false))
            });
            let mut members: Vec<Vec<u32>> = vec![Vec::new(); nlist];
            for (i, &c) in assign.iter().enumerate() {
                members[c as usize].push(i as u32);
            }
            let lists = match &copy {
                Stored::F32 => Lists::F32(make_lists(members, &xb, d)),
                Stored::F16(h) => Lists::F16(make_lists(members, h, d)),
                Stored::U8(h) => Lists::U8(make_lists(members, h, d)),
            };
            let cnorms = (0..nlist)
                .map(|c| {
                    let (b, l) = (c / L, c % L);
                    centroids[b * d..(b + 1) * d].iter().map(|v| v[l] * v[l]).sum()
                })
                .collect();
            IvfIndex {
                d,
                metric: kind,
                nlist,
                centroids,
                cnorms,
                lists,
                groups: None,
            }
        }))
    }

    /// The indices of the k nearest base vectors to each query among the lists of its `nprobe`
    /// nearest centroids, flattened (nq * k), nearest first; u32::MAX where fewer than k were found.
    /// One thread per query, or with `split` above one, one query at a time, its coarse step and
    /// its lists spread over a pool of that many threads, kept for later calls: lower latency for
    /// single queries, the same results.
    /// With `gprobe` above zero (after `set_coarse_groups`), the coarse step is two-level: the
    /// query's `gprobe` nearest groups of centroids, then the nearest `nprobe` of their centroids.
    /// With `threads` above zero, a batch (one query per thread) runs on a pool of that many threads
    /// of its own instead of rayon's global pool (all cores).
    /// With `products` ((nq, nlist) float32, the queries' products with the centroids, as a BLAS
    /// matrix multiply gives them for a batch), the coarse step ranks those instead of computing each
    /// query's distances (L2 and inner product).
    #[pyo3(signature = (xq, k, nprobe, split = 1, gprobe = 0, threads = 0, products = None))]
    #[allow(clippy::too_many_arguments)]
    fn search<'py>(
        &self,
        py: Python<'py>,
        xq: &Bound<'_, PyAny>,
        k: usize,
        nprobe: usize,
        split: usize,
        gprobe: usize,
        threads: usize,
        products: Option<PyReadonlyArray2<'_, f32>>,
    ) -> PyResult<Bound<'py, PyArray1<u32>>> {
        if gprobe > 0 && self.groups.is_none() {
            return Err(PyValueError::new_err(
                "gprobe needs set_coarse_groups first",
            ));
        }
        if k == 0 || nprobe == 0 {
            return Err(PyValueError::new_err("need k > 0 and nprobe > 0"));
        }
        let v = read(py, xq, self.d)?;
        let d = self.d;
        let given = match &products {
            Some(p) => Some(probes_from_products(
                p,
                v.len() / d,
                &self.cnorms,
                self.metric,
                nprobe,
            )?),
            None => None,
        };
        let per = nprobe.min(self.nlist);
        let probe = |i: usize| given.as_ref().map(|g| &g[i * per..(i + 1) * per]);
        let pool = if split > 1 {
            Some(split_pool(split)?)
        } else {
            None
        };
        let batch_pool = if split <= 1 && threads > 0 {
            Some(split_pool(threads)?)
        } else {
            None
        };
        let out = py.detach(|| {
            let mut out = vec![u32::MAX; (v.len() / d) * k];
            with_lists!(&self.lists, lists => with_metric!(self.metric, M => if let Some(pool) = &pool {
                pool.install(|| {
                    for (i, (o, q)) in out.chunks_mut(k).zip(v.chunks(d)).enumerate() {
                        self.search_one::<M, _>(lists, q, k, Probe { nprobe, tasks: split, gprobe }, probe(i), o);
                    }
                });
            } else {
                let batch = |out: &mut [u32]| {
                    out.par_chunks_mut(k)
                        .zip(v.par_chunks(d))
                        .enumerate()
                        .for_each(|(i, (o, q))| self.search_one::<M, _>(lists, q, k, Probe { nprobe, tasks: 1, gprobe }, probe(i), o))
                };
                match &batch_pool {
                    Some(bp) => bp.install(|| batch(&mut out)),
                    None => batch(&mut out),
                }
            }));
            out
        });
        Ok(out.into_pyarray(py))
    }

    /// The same search, list-major: queries are grouped by the lists they probe, and each list is
    /// scanned once for all of them, `QB` queries at a time, so that it is read from memory once per
    /// batch rather than once per query. For large batches. `products`, as for `search`.
    #[pyo3(signature = (xq, k, nprobe, products = None))]
    fn search_list_major<'py>(
        &self,
        py: Python<'py>,
        xq: &Bound<'_, PyAny>,
        k: usize,
        nprobe: usize,
        products: Option<PyReadonlyArray2<'_, f32>>,
    ) -> PyResult<Bound<'py, PyArray1<u32>>> {
        if k == 0 || nprobe == 0 {
            return Err(PyValueError::new_err("need k > 0 and nprobe > 0"));
        }
        let v = read(py, xq, self.d)?;
        let given = match &products {
            Some(p) => Some(probes_from_products(
                p,
                v.len() / self.d,
                &self.cnorms,
                self.metric,
                nprobe,
            )?),
            None => None,
        };
        let out = py.detach(
            || with_lists!(&self.lists, lists => with_metric!(self.metric, M => self.search_list_major_impl::<M, _>(lists, &v, k, nprobe, given.as_deref()))),
        );
        Ok(out.into_pyarray(py))
    }

    /// Clusters the centroids into groups for a two-level coarse step: `centres` (ngroups, d), the
    /// groups' centres (C-contiguous float32), and `assign` (nlist,), each centroid's group.
    fn set_coarse_groups(
        &mut self,
        py: Python<'_>,
        centres: &Bound<'_, PyAny>,
        assign: PyReadonlyArray1<'_, i64>,
    ) -> PyResult<()> {
        let c = read(py, centres, self.d)?;
        let count = c.len() / self.d;
        let a = assign
            .as_slice()
            .map_err(|_| PyValueError::new_err("assign must be contiguous"))?;
        if a.len() != self.nlist || a.iter().any(|&g| g < 0 || g as usize >= count) {
            return Err(PyValueError::new_err(
                "assign must give each centroid a group in range",
            ));
        }
        // Each centroid's coordinates, from the blocked layout.
        let d = self.d;
        let row = |i: usize| -> Vec<f32> {
            (0..d)
                .map(|j| self.centroids[(i / L) * d + j][i % L])
                .collect()
        };
        let mut ids: Vec<Vec<u32>> = vec![Vec::new(); count];
        for (i, &g) in a.iter().enumerate() {
            ids[g as usize].push(i as u32);
        }
        let blocks = ids
            .iter()
            .map(|members| {
                let rows: Vec<Vec<f32>> = members.iter().map(|&i| row(i as usize)).collect();
                to_blocks(rows.iter().map(|r| r.as_slice()), d)
            })
            .collect();
        self.groups = Some(Groups {
            centres: to_blocks(c.chunks_exact(d), d),
            count,
            blocks,
            ids,
        });
        Ok(())
    }

    /// The number of vectors in each list.
    fn list_sizes(&self) -> Vec<usize> {
        with_lists!(&self.lists, lists => lists.iter().map(|l| l.ids.len()).collect())
    }
}

/// One IVF list's products for `push_l2_lists`: products (rows x m), norms (m), ids (m), rows.
type ListProducts<'a> = (&'a [f32], &'a [f32], &'a [i64], &'a [i64]);

/// The best k of each query's L2 distances, kept across blocks of base vectors, for exact L2 search
/// with the products from a BLAS matrix multiply (NumPy's `@`, Accelerate on a Mac): for each block,
/// `push_l2` takes the queries' products with the block's vectors and the vectors' squared norms, and
/// ranks by |x|^2 - 2 q.x (|q|^2 is the same for all of a query's candidates). Reads NumPy's arrays in
/// place, without copying; the queries' rows are split over threads.
#[pyclass]
struct L2TopK {
    tops: Vec<TopK>,
}

#[pymethods]
impl L2TopK {
    #[new]
    fn py_new(nq: usize, k: usize) -> PyResult<Self> {
        if k == 0 {
            return Err(PyValueError::new_err("need k > 0"));
        }
        Ok(L2TopK {
            tops: (0..nq).map(|_| TopK::new(k)).collect(),
        })
    }

    /// One block: `prod` (nq, m), C-contiguous float32, the products of the queries with base vectors
    /// `offset` to `offset + m`; `norms` (m,), their squared norms.
    fn push_l2(
        &mut self,
        py: Python<'_>,
        prod: PyReadonlyArray2<'_, f32>,
        norms: PyReadonlyArray1<'_, f32>,
        offset: usize,
    ) -> PyResult<()> {
        let p = prod
            .as_slice()
            .map_err(|_| PyValueError::new_err("prod must be C-contiguous"))?;
        let nm = norms
            .as_slice()
            .map_err(|_| PyValueError::new_err("norms must be contiguous"))?;
        let m = nm.len();
        if m == 0 || p.len() != self.tops.len() * m {
            return Err(PyValueError::new_err("prod must be (nq, len(norms))"));
        }
        let tops = &mut self.tops;
        py.detach(|| {
            tops.par_iter_mut()
                .zip(p.par_chunks(m))
                .for_each(|(top, row)| {
                    for (j, (&pv, &n)) in row.iter().zip(nm).enumerate() {
                        top.push(n - 2.0 * pv, (offset + j) as u32);
                    }
                })
        });
        Ok(())
    }

    /// A group of IVF lists, each list `i` given as `prods[i]` (len(rows[i]), m_i), the products of
    /// the queries `rows[i]` (distinct within a list) with the list's m_i vectors, whose squared norms
    /// are `norms[i]` and whose ids in the base are `ids[i]`. The lists are ranked in parallel, each
    /// into its own best k per query, which are then merged into the queries' best k.
    fn push_l2_lists(
        &mut self,
        py: Python<'_>,
        prods: Vec<PyReadonlyArray2<'_, f32>>,
        norms: Vec<PyReadonlyArray1<'_, f32>>,
        ids: Vec<PyReadonlyArray1<'_, i64>>,
        rows: Vec<PyReadonlyArray1<'_, i64>>,
    ) -> PyResult<()> {
        let n = prods.len();
        if norms.len() != n || ids.len() != n || rows.len() != n {
            return Err(PyValueError::new_err(
                "prods, norms, ids and rows must have one entry per list",
            ));
        }
        let contiguous = || PyValueError::new_err("arrays must be C-contiguous");
        let mut lists: Vec<ListProducts<'_>> = Vec::with_capacity(n);
        let nq = self.tops.len();
        for i in 0..n {
            let (p, nm) = (
                prods[i].as_slice().map_err(|_| contiguous())?,
                norms[i].as_slice().map_err(|_| contiguous())?,
            );
            let (id, rs) = (
                ids[i].as_slice().map_err(|_| contiguous())?,
                rows[i].as_slice().map_err(|_| contiguous())?,
            );
            if id.len() != nm.len()
                || p.len() != rs.len() * nm.len()
                || rs.iter().any(|&r| r < 0 || r as usize >= nq)
            {
                return Err(PyValueError::new_err(
                    "each prod must be (len(rows), m), ids (m,), rows in range",
                ));
            }
            lists.push((p, nm, id, rs));
        }
        let k = self.tops.first().map_or(1, |t| t.dist.len());
        let tops = &mut self.tops;
        py.detach(|| {
            let parts: Vec<Vec<(usize, TopK)>> = lists
                .par_iter()
                .map(|&(p, nm, id, rs)| {
                    let m = nm.len();
                    if m == 0 {
                        return Vec::new();
                    }
                    rs.iter()
                        .zip(p.chunks_exact(m))
                        .map(|(&r, row)| {
                            let mut t = TopK::new(k);
                            for ((&pv, &nv), &i) in row.iter().zip(nm).zip(id) {
                                t.push(nv - 2.0 * pv, i as u32);
                            }
                            (r as usize, t)
                        })
                        .collect()
                })
                .collect();
            for (r, t) in parts.iter().flatten() {
                for (&v, &i) in t.dist.iter().zip(&t.idx) {
                    tops[*r].push(v, i);
                }
            }
        });
        Ok(())
    }

    /// The indices of each query's k nearest, flattened (nq * k), nearest first.
    fn result<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u32>> {
        let out: Vec<u32> = self
            .tops
            .iter()
            .flat_map(|t| t.idx.iter().copied())
            .collect();
        out.into_pyarray(py)
    }
}

/// The score of vector `x` for query `q`: the Euclidean distance for L2, the sums of absolute differences for
/// L1, the largest for L-infinity, the inner product for "ip".
fn score(kind: Kind, q: &[f32], x: &[f32]) -> f32 {
    let pairs = q.iter().zip(x);
    match kind {
        Kind::L2 => pairs.map(|(a, b)| (a - b) * (a - b)).sum::<f32>().sqrt(),
        Kind::L1 => pairs.map(|(a, b)| (a - b).abs()).sum(),
        Kind::Linf => pairs.fold(0.0f32, |m, (a, b)| larger(m, (a - b).abs())),
        Kind::Ip => pairs.map(|(a, b)| a * b).sum(),
    }
}

/// Each returned neighbor's score, computed from the vectors in f32: for `ids` (nq, k) int64 rows of `x`
/// (n, d) and queries `q` (nq, d), both C-contiguous float32 and read in place, the Euclidean distance
/// for "l2", the sum of absolute differences for "l1", the largest for "linf" and the inner product for
/// "ip" (and cosine, on unit vectors); infinity (minus infinity for "ip") where an id is -1. Large batches
/// are spread over threads.
#[pyfunction]
fn scores<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'_, f32>,
    q: PyReadonlyArray2<'_, f32>,
    ids: PyReadonlyArray2<'_, i64>,
    metric: &str,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let kind = parse_metric(metric)?;
    let (n, d) = x.as_array().dim();
    let (nq, qd) = q.as_array().dim();
    let (iq, k) = ids.as_array().dim();
    let contiguous = || PyValueError::new_err("x, q and ids must be C-contiguous");
    let (xs, qs, is) = (
        x.as_slice().map_err(|_| contiguous())?,
        q.as_slice().map_err(|_| contiguous())?,
        ids.as_slice().map_err(|_| contiguous())?,
    );
    if qd != d || iq != nq || is.iter().any(|&i| i >= n as i64 || i < -1) {
        return Err(PyValueError::new_err(
            "q must be (nq, d) and ids (nq, k), each id -1 or a row of x",
        ));
    }
    let missing = if matches!(kind, Kind::Ip) {
        f32::NEG_INFINITY
    } else {
        f32::INFINITY
    };
    let one = |(o, (row, qv)): (&mut [f32], (&[i64], &[f32]))| {
        for (s, &i) in o.iter_mut().zip(row) {
            *s = if i < 0 {
                missing
            } else {
                score(kind, qv, &xs[i as usize * d..(i as usize + 1) * d])
            };
        }
    };
    let mut out = vec![0.0f32; nq * k];
    py.detach(|| {
        let rows = out
            .chunks_mut(k.max(1))
            .zip(is.chunks(k.max(1)).zip(qs.chunks(d.max(1))));
        if nq * k * d < 1 << 22 {
            rows.for_each(one);
        } else {
            out.par_chunks_mut(k.max(1))
                .zip(is.par_chunks(k.max(1)).zip(qs.par_chunks(d.max(1))))
                .for_each(one);
        }
    });
    PyArray1::from_vec(py, out).reshape([nq, k])
}

/// Ids and scores, both (nq, k).
type Ranked<'py> = (Bound<'py, PyArray2<i64>>, Bound<'py, PyArray2<f32>>);

/// The results of a search, ranked: each row of `ids` (nq, k) int64 rows of `x`, with their scores as
/// `scores` computes them, sorted best first by score (smallest for "l1", "l2", "linf"; largest for "ip"), ties
/// by id, and -1 last. Returns (ids, scores). Searches return this order, so that it always agrees with the
/// scores returned, whatever precision a path ranked in (the GPU's float16 lists).
#[pyfunction]
fn rank<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'_, f32>,
    q: PyReadonlyArray2<'_, f32>,
    ids: PyReadonlyArray2<'_, i64>,
    metric: &str,
) -> PyResult<Ranked<'py>> {
    let kind = parse_metric(metric)?;
    let (nq, k) = ids.as_array().dim();
    let s = scores(py, x, q, ids.clone(), metric)?;
    let mut sv: Vec<f32> = s.readonly().as_slice()?.to_vec();
    let mut iv: Vec<i64> = ids
        .as_slice()
        .map_err(|_| PyValueError::new_err("ids must be C-contiguous"))?
        .to_vec();
    let larger = matches!(kind, Kind::Ip);
    let sort_row = |(ir, sr): (&mut [i64], &mut [f32])| {
        let mut pairs: Vec<(f32, i64)> = sr.iter().copied().zip(ir.iter().copied()).collect();
        pairs.sort_by(|a, b| {
            (a.1 < 0)
                .cmp(&(b.1 < 0))
                .then_with(|| {
                    if larger {
                        b.0.total_cmp(&a.0)
                    } else {
                        a.0.total_cmp(&b.0)
                    }
                })
                .then(a.1.cmp(&b.1))
        });
        for (j, (v, i)) in pairs.into_iter().enumerate() {
            sr[j] = v;
            ir[j] = i;
        }
    };
    py.detach(|| {
        // one thread for small batches: waking the pool costs more than sorting 10 values per row
        if nq * k < 1 << 16 {
            iv.chunks_mut(k.max(1))
                .zip(sv.chunks_mut(k.max(1)))
                .for_each(sort_row);
        } else {
            iv.par_chunks_mut(k.max(1))
                .zip(sv.par_chunks_mut(k.max(1)))
                .for_each(sort_row);
        }
    });
    Ok((
        PyArray1::from_vec(py, iv).reshape([nq, k])?,
        PyArray1::from_vec(py, sv).reshape([nq, k])?,
    ))
}

/// One query's row of the merged results, as `merge_results` fills it: its index, its ids and scores, its count
/// of masked candidates.
type MergeRow<'a> = (usize, ((&'a mut [i64], &'a mut [f32]), &'a mut i64));

/// Ids, scores and how many masked rows each query met, (nq, k), (nq, k) and (nq,).
type Merged<'py> = (
    Bound<'py, PyArray2<i64>>,
    Bound<'py, PyArray2<f32>>,
    Bound<'py, PyArray1<i64>>,
);

/// The search results of an index that has changed, merged: for each query, its candidates among the main
/// vectors `x` (`rows`, (nq, m) int64 rows of x, -1 for none) less those `dead` masks, named by `ids` (each
/// row's id; by default the row), and among the added vectors `xb` (`brows`, rows of xb, named by `bids`);
/// each scored as `scores` scores it, and the best k kept, ranked as `rank` ranks them (ties by id, -1 last).
/// Also returns how many of each query's main candidates were masked, so that a caller can search again the
/// queries left with too few.
#[pyfunction]
#[pyo3(signature = (x, q, rows, k, metric, dead = None, ids = None, xb = None, brows = None, bids = None))]
#[allow(clippy::too_many_arguments)] // PyO3 takes the keyword arguments as parameters
fn merge_results<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'_, f32>,
    q: PyReadonlyArray2<'_, f32>,
    rows: PyReadonlyArray2<'_, i64>,
    k: usize,
    metric: &str,
    dead: Option<PyReadonlyArray1<'_, bool>>,
    ids: Option<PyReadonlyArray1<'_, i64>>,
    xb: Option<PyReadonlyArray2<'_, f32>>,
    brows: Option<PyReadonlyArray2<'_, i64>>,
    bids: Option<PyReadonlyArray1<'_, i64>>,
) -> PyResult<Merged<'py>> {
    let kind = parse_metric(metric)?;
    let contiguous = || PyValueError::new_err("arrays must be C-contiguous");
    let (n, d) = x.as_array().dim();
    let (nq, qd) = q.as_array().dim();
    let (rq, m) = rows.as_array().dim();
    let (xs, qs, rs) = (
        x.as_slice().map_err(|_| contiguous())?,
        q.as_slice().map_err(|_| contiguous())?,
        rows.as_slice().map_err(|_| contiguous())?,
    );
    let dead = match &dead {
        Some(a) => Some(a.as_slice().map_err(|_| contiguous())?),
        None => None,
    };
    let ids = match &ids {
        Some(a) => Some(a.as_slice().map_err(|_| contiguous())?),
        None => None,
    };
    let bad = |what: &str| Err(PyValueError::new_err(format!("merge_results: {what}")));
    if qd != d || rq != nq || rs.iter().any(|&r| r < -1 || r >= n as i64) {
        return bad("q must be (nq, d) and rows (nq, m), each -1 or a row of x");
    }
    if dead.is_some_and(|v| v.len() != n) || ids.is_some_and(|v| v.len() != n) {
        return bad("dead and ids must have one entry per row of x");
    }
    let (bxs, brs, bis, mb) = match (&xb, &brows, &bids) {
        (Some(bx), Some(br), Some(bi)) => {
            let (nb, bd) = bx.as_array().dim();
            let (brq, mb) = br.as_array().dim();
            let (bxs, brs, bis) = (
                bx.as_slice().map_err(|_| contiguous())?,
                br.as_slice().map_err(|_| contiguous())?,
                bi.as_slice().map_err(|_| contiguous())?,
            );
            if bd != d
                || brq != nq
                || bis.len() != nb
                || brs.iter().any(|&r| r < -1 || r >= nb as i64)
            {
                return bad("xb must be (nb, d), brows (nq, mb) rows of xb, bids (nb,)");
            }
            (bxs, brs, bis, mb)
        }
        (None, None, None) => (&[][..], &[][..], &[][..], 0),
        _ => return bad("give xb, brows and bids together"),
    };
    let similarity = matches!(kind, Kind::Ip);
    let worst = if similarity {
        f32::NEG_INFINITY
    } else {
        f32::INFINITY
    };
    let mut out_i = vec![-1i64; nq * k];
    let mut out_s = vec![worst; nq * k];
    let mut hits = vec![0i64; nq];
    let one = |(qi, ((oi, os), h)): MergeRow<'_>| {
        let qv = &qs[qi * d..(qi + 1) * d];
        let mut cand: Vec<(f32, i64)> = Vec::with_capacity(m + mb);
        for &r in &rs[qi * m..(qi + 1) * m] {
            if r < 0 {
                continue;
            }
            let r = r as usize;
            if dead.is_some_and(|v| v[r]) {
                *h += 1;
                continue;
            }
            let id = ids.map_or(r as i64, |v| v[r]);
            cand.push((score(kind, qv, &xs[r * d..(r + 1) * d]), id));
        }
        for &r in &brs[qi * mb..(qi + 1) * mb] {
            if r >= 0 {
                let r = r as usize;
                cand.push((score(kind, qv, &bxs[r * d..(r + 1) * d]), bis[r]));
            }
        }
        cand.sort_by(|a, b| {
            if similarity {
                b.0.total_cmp(&a.0)
            } else {
                a.0.total_cmp(&b.0)
            }
            .then(a.1.cmp(&b.1))
        });
        for (j, (s, id)) in cand.into_iter().take(k).enumerate() {
            oi[j] = id;
            os[j] = s;
        }
    };
    py.detach(|| {
        let rows = out_i
            .chunks_mut(k.max(1))
            .zip(out_s.chunks_mut(k.max(1)))
            .zip(hits.iter_mut())
            .enumerate();
        // one thread for small batches: waking the pool costs more than scoring a few candidates
        if nq * (m + mb) * d < 1 << 20 {
            rows.for_each(one);
        } else {
            out_i
                .par_chunks_mut(k.max(1))
                .zip(out_s.par_chunks_mut(k.max(1)))
                .zip(hits.par_iter_mut())
                .enumerate()
                .for_each(one);
        }
    });
    Ok((
        PyArray1::from_vec(py, out_i).reshape([nq, k])?,
        PyArray1::from_vec(py, out_s).reshape([nq, k])?,
        PyArray1::from_vec(py, hits),
    ))
}

/// The rows of `x` scaled to unit length (for cosine), and how many rows were zero (left as they are).
#[pyfunction]
fn normalize<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'_, f32>,
) -> PyResult<(Bound<'py, PyArray2<f32>>, usize)> {
    let (n, d) = x.as_array().dim();
    let mut v = x
        .as_slice()
        .map_err(|_| PyValueError::new_err("x must be C-contiguous"))?
        .to_vec();
    let zeros = py.detach(|| {
        v.par_chunks_mut(d.max(1))
            .map(|row| {
                let len = row.iter().map(|a| a * a).sum::<f32>().sqrt();
                if len == 0.0 {
                    return 1usize;
                }
                for a in row.iter_mut() {
                    *a /= len;
                }
                0
            })
            .sum()
    });
    Ok((PyArray1::from_vec(py, v).reshape([n, d])?, zeros))
}

/// Each row's squared Euclidean norm (for ranking L2 by |x|^2 - 2 q.x).
#[pyfunction]
fn sq_norms<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'_, f32>,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let (_, d) = x.as_array().dim();
    let xs = x
        .as_slice()
        .map_err(|_| PyValueError::new_err("x must be C-contiguous"))?;
    let out: Vec<f32> = py.detach(|| {
        xs.par_chunks(d.max(1))
            .map(|row| row.iter().map(|a| a * a).sum())
            .collect()
    });
    Ok(PyArray1::from_vec(py, out))
}

/// Whether every value of `x` is an integer from 0 to 255 (so that storing it as bytes loses nothing).
#[pyfunction]
fn is_bytes(py: Python<'_>, x: PyReadonlyArray2<'_, f32>) -> PyResult<bool> {
    let xs = x
        .as_slice()
        .map_err(|_| PyValueError::new_err("x must be C-contiguous"))?;
    Ok(py.detach(|| {
        xs.par_iter()
            .all(|&a| (0.0..=255.0).contains(&a) && a == a.round())
    }))
}

/// The ids sorted by list and the lists' offsets.
type Grouped<'py> = (Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>);

/// The base vectors grouped by list: given each one's list (`assign`, int64, each below `nlist`), the ids sorted
/// by list, stably (a counting sort), and the offsets where each list starts (nlist + 1). Returns
/// (order, offsets).
#[pyfunction]
fn group_by_list<'py>(
    py: Python<'py>,
    assign: PyReadonlyArray1<'_, i64>,
    nlist: usize,
) -> PyResult<Grouped<'py>> {
    let a = assign
        .as_slice()
        .map_err(|_| PyValueError::new_err("assign must be contiguous"))?;
    if a.iter().any(|&c| c < 0 || c as usize >= nlist) {
        return Err(PyValueError::new_err("each list must be below nlist"));
    }
    let mut offsets = vec![0i64; nlist + 1];
    for &c in a {
        offsets[c as usize + 1] += 1;
    }
    for i in 0..nlist {
        offsets[i + 1] += offsets[i];
    }
    let mut next: Vec<i64> = offsets[..nlist].to_vec();
    let mut order = vec![0i64; a.len()];
    for (i, &c) in a.iter().enumerate() {
        order[next[c as usize] as usize] = i as i64;
        next[c as usize] += 1;
    }
    Ok((
        PyArray1::from_vec(py, order),
        PyArray1::from_vec(py, offsets),
    ))
}

/// k-means' update: each cluster's mean (its rows of `x` summed in row order in f32, then divided in f64, as
/// NumPy's `add.reduceat` and division compute it), and how many rows each has. Rows of `c` for empty
/// clusters are left as they are. Returns the counts; `c` (k, d) is updated in place.
#[pyfunction]
fn centroid_means<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'_, f32>,
    assign: PyReadonlyArray1<'_, i64>,
    c: &Bound<'py, PyArray2<f32>>,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let (n, d) = x.as_array().dim();
    let xs = x
        .as_slice()
        .map_err(|_| PyValueError::new_err("x must be C-contiguous"))?;
    let a = assign
        .as_slice()
        .map_err(|_| PyValueError::new_err("assign must be contiguous"))?;
    let k = c.readonly().as_array().dim().0;
    if a.len() != n
        || a.iter().any(|&j| j < 0 || j as usize >= k)
        || c.readonly().as_array().dim().1 != d
    {
        return Err(PyValueError::new_err(
            "assign must give each row of x a cluster of c, and c be (k, d)",
        ));
    }
    let mut sums = vec![0.0f32; k * d];
    let mut counts = vec![0i64; k];
    for (row, &j) in xs.chunks_exact(d.max(1)).zip(a) {
        let j = j as usize;
        counts[j] += 1;
        for (s, &v) in sums[j * d..(j + 1) * d].iter_mut().zip(row) {
            *s += v;
        }
    }
    let mut cw = c.readwrite();
    let cs = cw
        .as_slice_mut()
        .map_err(|_| PyValueError::new_err("c must be C-contiguous"))?;
    for j in 0..k {
        if counts[j] > 0 {
            let m = counts[j] as f64;
            for (o, &s) in cs[j * d..(j + 1) * d]
                .iter_mut()
                .zip(&sums[j * d..(j + 1) * d])
            {
                *o = (s as f64 / m) as f32;
            }
        }
    }
    Ok(PyArray1::from_vec(py, counts))
}

/// Recall for calibration: the share of `found` (nq, k) neighbors, -1 a miss, whose `scores` are at least as
/// good as each query's true k-th score `kth` (nq,), with a relative tolerance of 1e-6 (ties count as found).
#[pyfunction]
fn recall_of(
    found: PyReadonlyArray2<'_, i64>,
    scores: PyReadonlyArray2<'_, f32>,
    kth: PyReadonlyArray1<'_, f32>,
    larger_better: bool,
) -> PyResult<f64> {
    let contiguous = || PyValueError::new_err("arrays must be C-contiguous");
    let (f, s, t) = (
        found.as_slice().map_err(|_| contiguous())?,
        scores.as_slice().map_err(|_| contiguous())?,
        kth.as_slice().map_err(|_| contiguous())?,
    );
    let (nq, k) = found.as_array().dim();
    if s.len() != f.len() || t.len() != nq {
        return Err(PyValueError::new_err(
            "found and scores must be (nq, k), kth (nq,)",
        ));
    }
    let mut ok = 0usize;
    for ((fr, sr), &tq) in f.chunks(k.max(1)).zip(s.chunks(k.max(1))).zip(t) {
        let bound = if larger_better {
            tq - 1e-6 * tq.abs() - 1e-9
        } else {
            tq + 1e-6 * tq.abs() + 1e-9
        };
        ok += fr
            .iter()
            .zip(sr)
            .filter(|&(&id, &v)| {
                id >= 0
                    && if larger_better {
                        v >= bound
                    } else {
                        v <= bound
                    }
            })
            .count();
    }
    Ok(if f.is_empty() {
        0.0
    } else {
        ok as f64 / f.len() as f64
    })
}

/// Each row of `rows` (nq, m) without the query's own id `own` (nq,), its first `k` kept and -1 after: the truth
/// and the results when the calibration sample is the index's own vectors.
#[pyfunction]
fn without_own<'py>(
    py: Python<'py>,
    rows: PyReadonlyArray2<'_, i64>,
    own: PyReadonlyArray1<'_, i64>,
    k: usize,
) -> PyResult<Bound<'py, PyArray2<i64>>> {
    let (nq, m) = rows.as_array().dim();
    let (r, o) = (
        rows.as_slice()
            .map_err(|_| PyValueError::new_err("rows must be C-contiguous"))?,
        own.as_slice()
            .map_err(|_| PyValueError::new_err("own must be contiguous"))?,
    );
    if o.len() != nq {
        return Err(PyValueError::new_err("own must have one id per row"));
    }
    let mut out = vec![-1i64; nq * k];
    for q in 0..nq {
        let kept = r[q * m..(q + 1) * m].iter().filter(|&&v| v != o[q]).take(k);
        for (slot, &v) in out[q * k..(q + 1) * k].iter_mut().zip(kept) {
            *slot = v;
        }
    }
    PyArray1::from_vec(py, out).reshape([nq, k])
}

/// float16 values (their bits, a uint16 buffer) widened to f32 as the lists widen them: for
/// check_kernels.py, which compares every finite float16 with NumPy's conversion.
#[pyfunction]
fn widen_f16(py: Python<'_>, bits: &Bound<'_, PyAny>) -> PyResult<Vec<f32>> {
    let h = PyBuffer::<u16>::get(bits)?.to_vec(py)?;
    Ok(h.into_iter().map(widen_half).collect())
}

#[pymodule]
fn _cpu(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<FlatIndex>()?;
    m.add_class::<L2TopK>()?;
    m.add_function(wrap_pyfunction!(widen_f16, m)?)?;
    m.add_function(wrap_pyfunction!(scores, m)?)?;
    m.add_function(wrap_pyfunction!(rank, m)?)?;
    m.add_function(wrap_pyfunction!(merge_results, m)?)?;
    m.add_function(wrap_pyfunction!(normalize, m)?)?;
    m.add_function(wrap_pyfunction!(sq_norms, m)?)?;
    m.add_function(wrap_pyfunction!(is_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(group_by_list, m)?)?;
    m.add_function(wrap_pyfunction!(centroid_means, m)?)?;
    m.add_function(wrap_pyfunction!(recall_of, m)?)?;
    m.add_function(wrap_pyfunction!(without_own, m)?)?;
    m.add_class::<IvfIndex>()
}
