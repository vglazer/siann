//! Search of a proximity graph on the CPU: each vector linked to `degree` near ones (a CAGRA graph, built on
//! the GPU by mlxvs or cuVS), searched best first from an entry vector, keeping the `ef` nearest found so far,
//! as HNSW searches its bottom layer. One thread per query; a query's visited vectors are marked in a
//! per-thread array of epochs, so that nothing is cleared between queries.

use rayon::prelude::*;
use std::cell::RefCell;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

use super::Result;
use super::elem::Elem;
use super::metric::{Kind, Metric, with_metric};

/// Lanes of running distances over a row: independent sums the compiler maps onto vector instructions.
const LANES: usize = 16;

/// The vectors as the graph's search reads them: rows of f32, or of bytes (SIFT's integer coordinates).
pub enum Rows {
    F32(Vec<f32>),
    U8(Vec<u8>),
}

/// A distance and a vector, ordered by distance then id (so that ties do not depend on the search's order).
#[derive(Clone, Copy, PartialEq)]
struct Near(f32, u32);

impl Eq for Near {}

impl PartialOrd for Near {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Near {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.total_cmp(&other.0).then(self.1.cmp(&other.1))
    }
}

/// One row's distance to the query, in `LANES` running sums joined at the end.
#[inline]
fn row_dist<M: Metric, T: Elem>(q: &[f32], x: &[T]) -> f32 {
    let mut acc = [0.0f32; LANES];
    let (q16, q_rest) = q.as_chunks::<LANES>();
    let (x16, x_rest) = x.as_chunks::<LANES>();
    for (qs, xs) in q16.iter().zip(x16) {
        for l in 0..LANES {
            acc[l] = M::combine(acc[l], qs[l], T::lane(xs[l]));
        }
    }
    for (l, (&a, &b)) in q_rest.iter().zip(x_rest).enumerate() {
        acc[l] = M::combine(acc[l], a, T::lane(b));
    }
    let mut out = acc[0];
    for &v in &acc[1..] {
        out = M::join(out, v);
    }
    out
}

thread_local! {
    /// Per thread: the epoch of the current query, and each vector's epoch when last visited.
    static VISITED: RefCell<(u32, Vec<u32>)> = const { RefCell::new((0, Vec::new())) };
}

/// A graph over n vectors of d dimensions: `links` holds each vector's `degree` neighbors in turn.
pub struct Graph {
    pub d: usize,
    n: usize,
    degree: usize,
    metric: Kind,
    rows: Rows,
    links: Vec<u32>,
    entry: u32,
}

impl Graph {
    /// From the vectors `x` (n rows of `d`, used to find the entry: the vector nearest their mean), the rows the
    /// search reads (`rows`: x itself, or the same as bytes), and the links (n x `degree`, each below n, or -1
    /// for none).
    pub fn new(
        x: &[f32],
        d: usize,
        metric: Kind,
        rows: Rows,
        links: &[i64],
        degree: usize,
    ) -> Result<Self> {
        let n = x.len() / d;
        if n == 0 || degree == 0 || links.len() != n * degree {
            return Err("links must be (n, degree), n > 0".into());
        }
        if links.iter().any(|&l| l < -1 || l >= n as i64) {
            return Err("each link must be -1 or a row of x".into());
        }
        match &rows {
            Rows::F32(v) if v.len() != x.len() => {
                return Err("rows must have the shape of x".into());
            }
            Rows::U8(v) if v.len() != x.len() => return Err("rows must have the shape of x".into()),
            Rows::U8(v) if v.iter().zip(x).any(|(&b, &f)| f32::from(b) != f) => {
                return Err("byte rows need x's coordinates to be integers from 0 to 255".into());
            }
            _ => {}
        }
        let mut mean = vec![0.0f64; d];
        for row in x.chunks_exact(d) {
            for (m, &v) in mean.iter_mut().zip(row) {
                *m += f64::from(v);
            }
        }
        let mean: Vec<f32> = mean.iter().map(|&m| (m / n as f64) as f32).collect();
        let entry = x
            .par_chunks(d)
            .enumerate()
            .map(|(i, row)| Near(row_dist::<super::metric::L2, f32>(&mean, row), i as u32))
            .min()
            .map_or(0, |b| b.1);
        Ok(Graph {
            d,
            n,
            degree,
            metric,
            rows,
            links: links
                .iter()
                .map(|&l| if l < 0 { u32::MAX } else { l as u32 })
                .collect(),
            entry,
        })
    }

    /// The indices of each query's k nearest found, flattened, nearest first (u32::MAX where fewer were found),
    /// searching with `ef` candidates (at least k): one thread per query.
    pub fn search(&self, xq: &[f32], k: usize, ef: usize) -> Result<Vec<u32>> {
        if k == 0 || ef < k {
            return Err("need k > 0 and ef >= k".into());
        }
        let mut out = vec![u32::MAX; (xq.len() / self.d) * k];
        out.par_chunks_mut(k)
            .zip(xq.par_chunks(self.d))
            .for_each(|(o, q)| {
                with_metric!(self.metric, M => match &self.rows {
                    Rows::F32(r) => self.one::<M, f32>(r, q, k, ef, o),
                    Rows::U8(r) => self.one::<M, u8>(r, q, k, ef, o),
                })
            });
        Ok(out)
    }

    fn one<M: Metric, T: Elem>(&self, rows: &[T], q: &[f32], k: usize, ef: usize, out: &mut [u32]) {
        let d = self.d;
        let q8 = if M::INTEGER {
            T::integer_query(q)
        } else {
            None
        };
        let dist = |i: u32| -> f32 {
            let x = &rows[i as usize * d..(i as usize + 1) * d];
            match &q8 {
                Some(q8) => T::int_dist::<M>(q8, x) as f32,
                None => row_dist::<M, T>(q, x),
            }
        };
        VISITED.with(|v| {
            let (epoch, seen) = &mut *v.borrow_mut();
            if seen.len() < self.n {
                seen.resize(self.n, 0);
            }
            *epoch = epoch.wrapping_add(1);
            if *epoch == 0 {
                seen.fill(0);
                *epoch = 1;
            }
            let epoch = *epoch;
            // candidates to expand, nearest first; the ef nearest found, farthest on top
            let mut todo: BinaryHeap<std::cmp::Reverse<Near>> = BinaryHeap::new();
            let mut best: BinaryHeap<Near> = BinaryHeap::with_capacity(ef + 1);
            let start = Near(dist(self.entry), self.entry);
            seen[self.entry as usize] = epoch;
            todo.push(std::cmp::Reverse(start));
            best.push(start);
            while let Some(std::cmp::Reverse(c)) = todo.pop() {
                if best.len() >= ef && c > *best.peek().unwrap() {
                    break;
                }
                for &nb in &self.links[c.1 as usize * self.degree..(c.1 as usize + 1) * self.degree]
                {
                    if nb == u32::MAX || seen[nb as usize] == epoch {
                        continue;
                    }
                    seen[nb as usize] = epoch;
                    let cand = Near(dist(nb), nb);
                    if best.len() < ef || cand < *best.peek().unwrap() {
                        todo.push(std::cmp::Reverse(cand));
                        best.push(cand);
                        if best.len() > ef {
                            best.pop();
                        }
                    }
                }
            }
            let found = best.into_sorted_vec();
            for (o, c) in out.iter_mut().zip(found.iter().take(k)) {
                *o = c.1;
            }
        });
    }
}
