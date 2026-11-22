//! Scoring, ranking and merging search results, and the arithmetic around building and calibrating an
//! index: normalization, norms, grouping by list, k-means' update, recall.

use rayon::prelude::*;

use super::metric::{Kind, score};
use super::{Mat, Result};

/// Each returned neighbor's score, computed from the vectors in f32: for `ids` (nq, k), rows of `x` or -1,
/// and queries `q` (nq, d), the Euclidean distance for L2, the sum of absolute differences for L1, the
/// largest for L-infinity and the inner product for "ip" (and cosine, on unit vectors); infinity (minus
/// infinity for "ip") where an id is -1. Large batches are spread over threads.
pub fn scores(x: Mat<'_, f32>, q: Mat<'_, f32>, ids: Mat<'_, i64>, kind: Kind) -> Result<Vec<f32>> {
    let (n, d) = (x.rows, x.cols);
    let (nq, k) = (q.rows, ids.cols);
    let (xs, qs, is) = (x.data, q.data, ids.data);
    if q.cols != d || ids.rows != nq || is.iter().any(|&i| i >= n as i64 || i < -1) {
        return Err("q must be (nq, d) and ids (nq, k), each id -1 or a row of x".into());
    }
    let missing = if kind.similarity() {
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
    // one thread below about 4 million operations: scoring in parallel between GPU calls slowed the GPU
    if nq * k * d < 1 << 22 {
        out.chunks_mut(k.max(1))
            .zip(is.chunks(k.max(1)).zip(qs.chunks(d.max(1))))
            .for_each(one);
    } else {
        out.par_chunks_mut(k.max(1))
            .zip(is.par_chunks(k.max(1)).zip(qs.par_chunks(d.max(1))))
            .for_each(one);
    }
    Ok(out)
}

/// The results of a search, ranked: each row of `ids` (nq, k), rows of `x`, with their scores as `scores`
/// computes them, sorted best first by score (smallest for "l1", "l2", "linf"; largest for "ip"), ties by
/// id, and -1 last. Returns (ids, scores). Searches return this order, so that it always agrees with the
/// scores returned, whatever precision a path ranked in (the GPU's float16 lists).
pub fn rank(
    x: Mat<'_, f32>,
    q: Mat<'_, f32>,
    ids: Mat<'_, i64>,
    kind: Kind,
) -> Result<(Vec<i64>, Vec<f32>)> {
    let (nq, k) = (ids.rows, ids.cols);
    let mut sv = scores(x, q, ids, kind)?;
    let mut iv = ids.data.to_vec();
    let larger = kind.similarity();
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
    Ok((iv, sv))
}

/// Added vectors not yet merged into an index: `x` (nb, d), each query's candidates among them `rows`
/// (nq, mb, rows of x or -1), and their ids.
#[derive(Clone, Copy)]
pub struct Buffer<'a> {
    pub x: Mat<'a, f32>,
    pub rows: Mat<'a, i64>,
    pub ids: &'a [i64],
}

/// How an index has changed since its main vectors were laid out: rows removed (`dead`, one per main row),
/// the ids of its rows (by default the row numbers), and added vectors in a buffer.
#[derive(Clone, Copy, Default)]
pub struct Changes<'a> {
    pub dead: Option<&'a [bool]>,
    pub ids: Option<&'a [i64]>,
    pub buffer: Option<Buffer<'a>>,
}

/// One query's row of the merged results, as `merge_results` fills it: its index, its ids and scores, its
/// count of masked candidates.
type MergeRow<'a> = (usize, ((&'a mut [i64], &'a mut [f32]), &'a mut i64));

/// Ids, scores and how many masked rows each query met: nq * k, nq * k and nq.
pub type Merged = (Vec<i64>, Vec<f32>, Vec<i64>);

/// The search results of an index that has changed, merged: for each query, its candidates among the main
/// vectors `x` (`rows`, (nq, m) rows of x, -1 for none) less those `changes.dead` masks, named by
/// `changes.ids`, and among the buffer's; each scored as `scores` scores it, and the best k kept, ranked as
/// `rank` ranks them (ties by id, -1 last). Also returns how many of each query's main candidates were
/// masked, so that a caller can search again the queries left with too few.
pub fn merge_results(
    x: Mat<'_, f32>,
    q: Mat<'_, f32>,
    rows: Mat<'_, i64>,
    k: usize,
    kind: Kind,
    changes: Changes<'_>,
) -> Result<Merged> {
    let (n, d) = (x.rows, x.cols);
    let (nq, m) = (q.rows, rows.cols);
    let (xs, qs, rs) = (x.data, q.data, rows.data);
    let Changes { dead, ids, buffer } = changes;
    let bad = |what: &str| Err(format!("merge_results: {what}"));
    if q.cols != d || rows.rows != nq || rs.iter().any(|&r| r < -1 || r >= n as i64) {
        return bad("q must be (nq, d) and rows (nq, m), each -1 or a row of x");
    }
    if dead.is_some_and(|v| v.len() != n) || ids.is_some_and(|v| v.len() != n) {
        return bad("dead and ids must have one entry per row of x");
    }
    let (bxs, brs, bis, mb) = match buffer {
        Some(b) => {
            if b.x.cols != d
                || b.rows.rows != nq
                || b.ids.len() != b.x.rows
                || b.rows.data.iter().any(|&r| r < -1 || r >= b.x.rows as i64)
            {
                return bad("xb must be (nb, d), brows (nq, mb) rows of xb, bids (nb,)");
            }
            (b.x.data, b.rows.data, b.ids, b.rows.cols)
        }
        None => (&[][..], &[][..], &[][..], 0),
    };
    let similarity = kind.similarity();
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
    // one thread for small batches: waking the pool costs more than scoring a few candidates
    if nq * (m + mb) * d < 1 << 20 {
        out_i
            .chunks_mut(k.max(1))
            .zip(out_s.chunks_mut(k.max(1)))
            .zip(hits.iter_mut())
            .enumerate()
            .for_each(one);
    } else {
        out_i
            .par_chunks_mut(k.max(1))
            .zip(out_s.par_chunks_mut(k.max(1)))
            .zip(hits.par_iter_mut())
            .enumerate()
            .for_each(one);
    }
    Ok((out_i, out_s, hits))
}

/// The rows of `x` (rows of `d`) scaled to unit length (for cosine), and how many rows were zero (left as
/// they are).
pub fn normalize(x: &[f32], d: usize) -> (Vec<f32>, usize) {
    let mut v = x.to_vec();
    let zeros = v
        .par_chunks_mut(d.max(1))
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
        .sum();
    (v, zeros)
}

/// Each row's squared Euclidean norm (for ranking L2 by |x|^2 - 2 q.x).
pub fn sq_norms(x: &[f32], d: usize) -> Vec<f32> {
    x.par_chunks(d.max(1))
        .map(|row| row.iter().map(|a| a * a).sum())
        .collect()
}

/// Whether every value of `x` is an integer from 0 to 255 (so that storing it as bytes loses nothing).
pub fn is_bytes(x: &[f32]) -> bool {
    x.par_iter()
        .all(|&a| (0.0..=255.0).contains(&a) && a == a.round())
}

/// The base vectors grouped by list: given each one's list (`assign`, each below `nlist`), the ids sorted
/// by list, stably (a counting sort), and the offsets where each list starts (nlist + 1). Returns
/// (order, offsets).
pub fn group_by_list(assign: &[i64], nlist: usize) -> Result<(Vec<i64>, Vec<i64>)> {
    if assign.iter().any(|&c| c < 0 || c as usize >= nlist) {
        return Err("each list must be below nlist".into());
    }
    let mut offsets = vec![0i64; nlist + 1];
    for &c in assign {
        offsets[c as usize + 1] += 1;
    }
    for i in 0..nlist {
        offsets[i + 1] += offsets[i];
    }
    let mut next: Vec<i64> = offsets[..nlist].to_vec();
    let mut order = vec![0i64; assign.len()];
    for (i, &c) in assign.iter().enumerate() {
        order[next[c as usize] as usize] = i as i64;
        next[c as usize] += 1;
    }
    Ok((order, offsets))
}

/// k-means' update: each cluster's mean (its rows of `x` summed in row order in f32, then divided in f64, as
/// NumPy's `add.reduceat` and division compute it), and how many rows each has. `c` holds `k` rows of
/// `cd` values, updated in place; rows for empty clusters are left as they are. Returns the counts.
pub fn centroid_means(
    x: Mat<'_, f32>,
    assign: &[i64],
    c: &mut [f32],
    k: usize,
    cd: usize,
) -> Result<Vec<i64>> {
    let (n, d) = (x.rows, x.cols);
    if assign.len() != n || assign.iter().any(|&j| j < 0 || j as usize >= k) || cd != d {
        return Err("assign must give each row of x a cluster of c, and c be (k, d)".into());
    }
    let mut sums = vec![0.0f32; k * d];
    let mut counts = vec![0i64; k];
    for (row, &j) in x.data.chunks_exact(d.max(1)).zip(assign) {
        let j = j as usize;
        counts[j] += 1;
        for (s, &v) in sums[j * d..(j + 1) * d].iter_mut().zip(row) {
            *s += v;
        }
    }
    for j in 0..k {
        if counts[j] > 0 {
            let m = counts[j] as f64;
            for (o, &s) in c[j * d..(j + 1) * d]
                .iter_mut()
                .zip(&sums[j * d..(j + 1) * d])
            {
                *o = (s as f64 / m) as f32;
            }
        }
    }
    Ok(counts)
}

/// Recall for calibration: the share of `found` (nq, k) neighbors, -1 a miss, whose `scores` (nq * k) are at
/// least as good as each query's true k-th score `kth` (nq), with a relative tolerance of 1e-6 (ties count
/// as found).
pub fn recall_of(
    found: Mat<'_, i64>,
    scores: &[f32],
    kth: &[f32],
    larger_better: bool,
) -> Result<f64> {
    let (f, nq, k) = (found.data, found.rows, found.cols);
    if scores.len() != f.len() || kth.len() != nq {
        return Err("found and scores must be (nq, k), kth (nq,)".into());
    }
    let mut ok = 0usize;
    for ((fr, sr), &tq) in f.chunks(k.max(1)).zip(scores.chunks(k.max(1))).zip(kth) {
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

/// Each row of `rows` (nq, m) without the query's own id `own` (nq), its first `k` kept and -1 after: the
/// truth and the results when the calibration sample is the index's own vectors.
pub fn without_own(rows: Mat<'_, i64>, own: &[i64], k: usize) -> Result<Vec<i64>> {
    let (r, nq, m) = (rows.data, rows.rows, rows.cols);
    if own.len() != nq {
        return Err("own must have one id per row".into());
    }
    let mut out = vec![-1i64; nq * k];
    for q in 0..nq {
        let kept = r[q * m..(q + 1) * m]
            .iter()
            .filter(|&&v| v != own[q])
            .take(k);
        for (slot, &v) in out[q * k..(q + 1) * k].iter_mut().zip(kept) {
            *slot = v;
        }
    }
    Ok(out)
}
