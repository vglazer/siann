//! Inverted lists. Each base vector belongs to its nearest centroid's list, stored in blocks. A query finds
//! its `nprobe` nearest centroids, then scans their lists: one thread per query, so that single queries
//! are cheap and batches spread across cores; or, for large batches, list-major, each list scanned once for
//! all the queries that probe it.

use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use super::elem::Elem;
use super::flat::flat_search;
use super::kernels::{
    L, QB, TopK, block_dists, block_dists_one, block_dists_one_direct, to_blocks,
};
use super::metric::{Kind, Metric, with_metric};
use super::{Mat, Result};

/// The thread pool for split queries of `threads` threads, made once and kept. Split queries run on
/// a small pool of their own: on rayon's global pool (every core), a query injected into the pool
/// wakes its sleeping threads, which costs more than the split saves (BENCHMARKS.md, "Approximate search: single queries").
fn split_pool(threads: usize) -> Result<Arc<rayon::ThreadPool>> {
    static POOLS: OnceLock<Mutex<HashMap<usize, Arc<rayon::ThreadPool>>>> = OnceLock::new();
    let mut pools = POOLS
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .map_err(|_| "split pool lock poisoned".to_string())?;
    if let Some(pool) = pools.get(&threads) {
        return Ok(Arc::clone(pool));
    }
    let pool = Arc::new(
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .thread_name(|i| format!("siann-split-{i}"))
            .build()
            .map_err(|e| format!("cannot start split pool: {e}"))?,
    );
    pools.insert(threads, Arc::clone(&pool));
    Ok(pool)
}

/// Each query's nearest `nprobe` lists from `products` (nq, nlist), its products with the centroids, as
/// BLAS computes them for a batch: ranked by |c|^2 - 2 q.c for L2 (the same order as the distance) and by
/// -q.c for inner product, selected as the Rust coarse step selects (in linear time, in no set order).
fn probes_from_products(
    products: Mat<'_, f32>,
    nq: usize,
    cnorms: &[f32],
    metric: Kind,
    nprobe: usize,
) -> Result<Vec<u32>> {
    let nlist = cnorms.len();
    if products.rows != nq || products.cols != nlist {
        return Err("products must be (nq, nlist)".into());
    }
    let l2 = match metric {
        Kind::L2 => true,
        Kind::Ip => false,
        _ => return Err("products give the coarse step for L2 and inner product only".into()),
    };
    let nprobe = nprobe.min(nlist);
    let mut out = vec![0u32; nq * nprobe];
    out.par_chunks_mut(nprobe)
        .zip(products.data.par_chunks(nlist))
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

/// One inverted list: its vectors in blocks, and their ids in the base.
struct List<T> {
    blocks: Vec<[T; L]>,
    ids: Vec<u32>,
    /// The vectors row by row, if `T::ROW_MAJOR` (byte lists, for integer queries); else empty.
    rows: Vec<T>,
}

/// The inverted lists, stored as f32, float16 or bytes.
enum Lists {
    F32(Vec<List<f32>>),
    F16(Vec<List<u16>>),
    U8(Vec<List<u8>>),
}

/// The vectors given for the lists, in their storage type: the base vectors themselves (f32), their
/// float16 bits, or their bytes.
pub enum Stored {
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

/// How one query is searched: lists to scan, parallel tasks (one: this thread), and groups for a
/// two-level coarse step (zero: exact coarse step).
#[derive(Clone, Copy)]
struct Probe {
    nprobe: usize,
    tasks: usize,
    gprobe: usize,
}

/// How a batch is searched query by query (`Ivf::search`): `split` above one, one query at a time, its
/// coarse step and lists spread over a pool of that many threads; `gprobe` above zero, a two-level coarse
/// step; `threads` above zero, a batch on a pool of that many threads of its own instead of all cores.
#[derive(Clone, Copy)]
pub struct Options {
    pub split: usize,
    pub gprobe: usize,
    pub threads: usize,
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

/// Inverted lists over given centroids.
pub struct Ivf {
    pub d: usize,
    metric: Kind,
    nlist: usize,
    centroids: Vec<[f32; L]>,
    /// The centroids' squared norms, for a coarse step from BLAS products.
    cnorms: Vec<f32>,
    lists: Lists,
    /// For a two-level coarse step (`set_coarse_groups`), the centroids clustered into groups.
    groups: Option<Groups>,
}

impl Ivf {
    /// From base vectors `xb` and centroids `cents` (rows of `d`), searched by `metric`, each base vector
    /// in the list of its nearest centroid by `assign` (found by exact search), or in the list `lists`
    /// gives it; the lists store `stored` (the base vectors themselves, or the same in a smaller type).
    pub fn new(
        xb: Vec<f32>,
        cents: &[f32],
        d: usize,
        metric: Kind,
        assign: Kind,
        stored: Stored,
        lists: Option<&[i64]>,
    ) -> Result<Self> {
        match &stored {
            Stored::F32 => {}
            Stored::F16(h) => {
                if h.len() != xb.len() {
                    return Err("stored must have the shape of xb".into());
                }
                if h.iter().any(|&b| (b >> 10) & 0x1f == 0x1f) {
                    return Err(
                        "stored holds infinity or NaN: values beyond float16's range (65504)"
                            .into(),
                    );
                }
            }
            Stored::U8(h) => {
                if h.len() != xb.len() {
                    return Err("stored must have the shape of xb".into());
                }
                if h.iter().zip(&xb).any(|(&b, &v)| f32::from(b) != v) {
                    return Err(
                        "uint8 lists need xb's coordinates to be integers from 0 to 255, equal to stored"
                            .into(),
                    );
                }
            }
        }
        let nlist = cents.len() / d;
        if nlist == 0 {
            return Err("need at least one centroid".into());
        }
        let given: Option<Vec<u32>> = match lists {
            Some(a) => {
                if a.len() != xb.len() / d || a.iter().any(|&l| l < 0 || l as usize >= nlist) {
                    return Err("lists must give each base vector a list in range".into());
                }
                Some(a.iter().map(|&l| l as u32).collect())
            }
            None => None,
        };
        let centroids = to_blocks(cents.chunks_exact(d), d);
        let assignment = given.unwrap_or_else(|| {
            with_metric!(assign, M => flat_search::<M>(&centroids, d, nlist, &xb, 1, 256, false))
        });
        let mut members: Vec<Vec<u32>> = vec![Vec::new(); nlist];
        for (i, &c) in assignment.iter().enumerate() {
            members[c as usize].push(i as u32);
        }
        let lists = match &stored {
            Stored::F32 => Lists::F32(make_lists(members, &xb, d)),
            Stored::F16(h) => Lists::F16(make_lists(members, h, d)),
            Stored::U8(h) => Lists::U8(make_lists(members, h, d)),
        };
        let cnorms = (0..nlist)
            .map(|c| {
                let (b, l) = (c / L, c % L);
                centroids[b * d..(b + 1) * d]
                    .iter()
                    .map(|v| v[l] * v[l])
                    .sum()
            })
            .collect();
        Ok(Ivf {
            d,
            metric,
            nlist,
            centroids,
            cnorms,
            lists,
            groups: None,
        })
    }

    /// The indices of the k nearest base vectors to each query of `xq` among the lists of its `nprobe`
    /// nearest centroids, flattened, nearest first; u32::MAX where fewer than k were found. With
    /// `products`, the queries' products with the centroids (as a BLAS matrix multiply gives them for a
    /// batch), the coarse step ranks those (L2 and inner product).
    pub fn search(
        &self,
        xq: &[f32],
        k: usize,
        nprobe: usize,
        how: Options,
        products: Option<Mat<'_, f32>>,
    ) -> Result<Vec<u32>> {
        let Options {
            split,
            gprobe,
            threads,
        } = how;
        if gprobe > 0 && self.groups.is_none() {
            return Err("gprobe needs set_coarse_groups first".into());
        }
        if k == 0 || nprobe == 0 {
            return Err("need k > 0 and nprobe > 0".into());
        }
        let d = self.d;
        let given = match products {
            Some(p) => Some(probes_from_products(
                p,
                xq.len() / d,
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
        let mut out = vec![u32::MAX; (xq.len() / d) * k];
        with_lists!(&self.lists, lists => with_metric!(self.metric, M => if let Some(pool) = &pool {
            pool.install(|| {
                for (i, (o, q)) in out.chunks_mut(k).zip(xq.chunks(d)).enumerate() {
                    self.search_one::<M, _>(lists, q, k, Probe { nprobe, tasks: split, gprobe }, probe(i), o);
                }
            });
        } else {
            let batch = |out: &mut [u32]| {
                out.par_chunks_mut(k)
                    .zip(xq.par_chunks(d))
                    .enumerate()
                    .for_each(|(i, (o, q))| self.search_one::<M, _>(lists, q, k, Probe { nprobe, tasks: 1, gprobe }, probe(i), o))
            };
            match &batch_pool {
                Some(bp) => bp.install(|| batch(&mut out)),
                None => batch(&mut out),
            }
        }));
        Ok(out)
    }

    /// The same search, list-major: queries are grouped by the lists they probe, and each list is
    /// scanned once for all of them, `QB` queries at a time, so that it is read from memory once per
    /// batch rather than once per query. For large batches. `products`, as for `search`.
    pub fn search_list_major(
        &self,
        xq: &[f32],
        k: usize,
        nprobe: usize,
        products: Option<Mat<'_, f32>>,
    ) -> Result<Vec<u32>> {
        if k == 0 || nprobe == 0 {
            return Err("need k > 0 and nprobe > 0".into());
        }
        let given = match products {
            Some(p) => Some(probes_from_products(
                p,
                xq.len() / self.d,
                &self.cnorms,
                self.metric,
                nprobe,
            )?),
            None => None,
        };
        Ok(
            with_lists!(&self.lists, lists => with_metric!(self.metric, M => self.search_list_major_impl::<M, _>(lists, xq, k, nprobe, given.as_deref()))),
        )
    }

    /// Clusters the centroids into groups for a two-level coarse step: `centres` (rows of `d`), the
    /// groups' centres, and `assign` (one per centroid), each centroid's group.
    pub fn set_coarse_groups(&mut self, centres: &[f32], assign: &[i64]) -> Result<()> {
        let count = centres.len() / self.d;
        if assign.len() != self.nlist || assign.iter().any(|&g| g < 0 || g as usize >= count) {
            return Err("assign must give each centroid a group in range".into());
        }
        // Each centroid's coordinates, from the blocked layout.
        let d = self.d;
        let row = |i: usize| -> Vec<f32> {
            (0..d)
                .map(|j| self.centroids[(i / L) * d + j][i % L])
                .collect()
        };
        let mut ids: Vec<Vec<u32>> = vec![Vec::new(); count];
        for (i, &g) in assign.iter().enumerate() {
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
            centres: to_blocks(centres.chunks_exact(d), d),
            count,
            blocks,
            ids,
        });
        Ok(())
    }

    /// The number of vectors in each list.
    pub fn list_sizes(&self) -> Vec<usize> {
        with_lists!(&self.lists, lists => lists.iter().map(|l| l.ids.len()).collect())
    }

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
