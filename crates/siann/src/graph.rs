//! Search of a proximity graph on the CPU: each vector linked to `degree` near ones (a CAGRA graph, built on
//! the GPU by mlxvs or cuVS), searched best first from a spread set of entry vectors (as CAGRA starts from
//! many: a graph's clusters need not be linked to each other), keeping the `ef` nearest found so far,
//! as HNSW searches its bottom layer. One thread per query; a query's visited vectors are marked in a
//! per-thread array of epochs, so that nothing is cleared between queries.

use rayon::prelude::*;
use std::cell::RefCell;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

use super::Result;
use super::elem::Elem;
use super::ivf::split_pool;
use super::metric::{Kind, Metric, with_metric};

/// Lanes of running distances over a row: independent sums the compiler maps onto vector instructions.
const LANES: usize = 16;
/// Batches below this many queries run on a pool of their own, a thread per query up to the cores.
const SMALL_BATCH: usize = 64;
/// Entry vectors every search starts from: the one nearest the mean, and the rest spread evenly through the rows.
const ENTRIES: usize = 256;
/// The most links per vector a graph may have (CAGRA's are 32 or 64).
pub const MAX_DEGREE: usize = 128;

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

/// Rows read at once by `rows_dist`: their memory loads in flight together.
const GROUP: usize = 4;
/// Lanes per row in `rows_dist`: GROUP x GROUP_LANES running sums fit in the vector registers.
const GROUP_LANES: usize = 8;

/// Distances from the query to `GROUP` rows, computed in one interleaved pass, so that the rows' loads from
/// memory overlap (a search reads rows scattered through memory, and is limited by fetching them).
#[inline]
fn rows_dist<M: Metric, T: Elem>(q: &[f32], xs: [&[T]; GROUP]) -> [f32; GROUP] {
    let mut acc = [[0.0f32; GROUP_LANES]; GROUP];
    let (q8, q_rest) = q.as_chunks::<GROUP_LANES>();
    let rows = xs.map(|x| x.as_chunks::<GROUP_LANES>());
    for (c, qs) in q8.iter().enumerate() {
        for (g, (r, _)) in rows.iter().enumerate() {
            let xs = &r[c];
            for l in 0..GROUP_LANES {
                acc[g][l] = M::combine(acc[g][l], qs[l], T::lane(xs[l]));
            }
        }
    }
    let mut out = [0.0f32; GROUP];
    for (g, (_, rest)) in rows.iter().enumerate() {
        for (l, (&a, &b)) in q_rest.iter().zip(rest.iter()).enumerate() {
            acc[g][l] = M::combine(acc[g][l], a, T::lane(b));
        }
        let mut v = acc[g][0];
        for &a in &acc[g][1..] {
            v = M::join(v, a);
        }
        out[g] = v;
    }
    out
}

thread_local! {
    /// Per thread: the epoch of the current query, and each vector's epoch when last visited, a byte (a quarter
    /// of the memory a search writes into at scattered places), cleared every 255 queries.
    static VISITED: RefCell<(u8, Vec<u8>)> = const { RefCell::new((0, Vec::new())) };
}

/// A graph over n vectors of d dimensions: `links` holds each vector's `degree` neighbors in turn.
pub struct Graph {
    pub d: usize,
    n: usize,
    degree: usize,
    metric: Kind,
    rows: Rows,
    links: Vec<u32>,
    /// Where every search starts: the vector nearest the mean, and others spread evenly through the rows.
    entries: Vec<u32>,
}

impl Graph {
    /// From the vectors `x` (n rows of `d`, used to find the entries: the vector nearest their mean and others
    /// spread evenly through them), the rows the
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
        if n == 0 || degree == 0 || degree > MAX_DEGREE || links.len() != n * degree {
            return Err(format!(
                "links must be (n, degree), n > 0, degree at most {MAX_DEGREE}"
            ));
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
        let step = n.div_ceil(ENTRIES - 1).max(1);
        let mut entries: Vec<u32> = std::iter::once(entry)
            .chain((0..n).step_by(step).map(|i| i as u32))
            .collect();
        entries.sort_unstable();
        entries.dedup();
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
            entries,
        })
    }

    /// The indices of each query's k nearest found, flattened, nearest first (u32::MAX where fewer were found),
    /// searching with `ef` candidates (at least k): one thread per query.
    pub fn search(&self, xq: &[f32], k: usize, ef: usize) -> Result<Vec<u32>> {
        if k == 0 || ef < k {
            return Err("need k > 0 and ef >= k".into());
        }
        let nq = xq.len() / self.d;
        let mut out = vec![u32::MAX; nq * k];
        let one = |o: &mut [u32], q: &[f32]| {
            with_metric!(self.metric, M => match &self.rows {
                Rows::F32(r) => self.one::<M, f32>(r, q, k, ef, o),
                Rows::U8(r) => self.one::<M, u8>(r, q, k, ef, o),
            })
        };
        let batch = |out: &mut [u32]| {
            out.par_chunks_mut(k)
                .zip(xq.par_chunks(self.d))
                .for_each(|(o, q)| one(o, q))
        };
        if nq == 1 {
            one(&mut out, xq); // on this thread: handing one query to a pool costs a thread's wake-up
        } else if nq < SMALL_BATCH {
            // a pool of its own, as many threads as queries up to the cores: waking rayon's global pool costs
            // more than a small batch
            split_pool(nq.min(rayon::current_num_threads()))?.install(|| batch(&mut out));
        } else {
            batch(&mut out);
        }
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
                None => rows_dist::<M, T>(q, [x; GROUP])[0], // summed as the grouped rows are
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
            let (mut ids, mut dists) = ([0u32; MAX_DEGREE], [0.0f32; MAX_DEGREE]);
            let mut best: BinaryHeap<Near> = BinaryHeap::with_capacity(ef + 1);
            let entry_dists: Vec<f32> = if q8.is_none() {
                let row = |i: u32| &rows[i as usize * d..(i as usize + 1) * d];
                self.entries
                    .chunks(GROUP)
                    .flat_map(|g| {
                        let pick = |j: usize| row(g[j.min(g.len() - 1)]);
                        rows_dist::<M, T>(q, [pick(0), pick(1), pick(2), pick(3)])
                            .into_iter()
                            .take(g.len())
                    })
                    .collect()
            } else {
                self.entries.iter().map(|&e| dist(e)).collect()
            };
            for (&e, &de) in self.entries.iter().zip(&entry_dists) {
                let start = Near(de, e);
                seen[e as usize] = epoch;
                if best.len() < ef || start < *best.peek().unwrap() {
                    todo.push(std::cmp::Reverse(start));
                    best.push(start);
                    if best.len() > ef {
                        best.pop();
                    }
                }
            }
            while let Some(std::cmp::Reverse(c)) = todo.pop() {
                if best.len() >= ef && c > *best.peek().unwrap() {
                    break;
                }
                // the unvisited neighbors, then their distances, GROUP rows at a time
                let mut m = 0;
                for &nb in &self.links[c.1 as usize * self.degree..(c.1 as usize + 1) * self.degree]
                {
                    if nb != u32::MAX && seen[nb as usize] != epoch {
                        seen[nb as usize] = epoch;
                        ids[m] = nb;
                        m += 1;
                    }
                }
                // Touch each neighbor's row, a value per cache line, before computing any distance: ordinary
                // loads, so that the rows' fetches from memory all start now and overlap (a safe stand-in for
                // a prefetch instruction).
                let step = (64 / std::mem::size_of::<T>()).max(1);
                let mut touched = 0.0f32;
                for &nb in &ids[..m] {
                    let row = &rows[nb as usize * d..(nb as usize + 1) * d];
                    for v in row.iter().step_by(step) {
                        touched += T::lane(*v);
                    }
                }
                std::hint::black_box(touched);
                let mut j = 0;
                if q8.is_none() {
                    let row = |i: u32| &rows[i as usize * d..(i as usize + 1) * d];
                    while j + GROUP <= m {
                        let g = rows_dist::<M, T>(
                            q,
                            [
                                row(ids[j]),
                                row(ids[j + 1]),
                                row(ids[j + 2]),
                                row(ids[j + 3]),
                            ],
                        );
                        dists[j..j + GROUP].copy_from_slice(&g);
                        j += GROUP;
                    }
                }
                for k in j..m {
                    dists[k] = dist(ids[k]);
                }
                for (&nb, &dv) in ids[..m].iter().zip(&dists[..m]) {
                    let cand = Near(dv, nb);
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
