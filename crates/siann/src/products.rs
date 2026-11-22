//! Ranking the products of a BLAS matrix multiply: exact L2 search on the CPU, with the products computed
//! by NumPy's `@` (Accelerate on a Mac) and ranked here.

use rayon::prelude::*;

use super::Result;
use super::kernels::TopK;

/// One IVF list's products for `push_l2_lists`: products (rows x m), norms (m), ids (m), rows.
pub type ListProducts<'a> = (&'a [f32], &'a [f32], &'a [i64], &'a [i64]);

/// The best k of each query's L2 distances, kept across blocks of base vectors: for each block, `push_l2`
/// takes the queries' products with the block's vectors and the vectors' squared norms, and ranks by
/// |x|^2 - 2 q.x (|q|^2 is the same for all of a query's candidates). The queries' rows are split over
/// threads.
pub struct L2TopK {
    tops: Vec<TopK>,
}

impl L2TopK {
    pub fn new(nq: usize, k: usize) -> Result<Self> {
        if k == 0 {
            return Err("need k > 0".into());
        }
        Ok(L2TopK {
            tops: (0..nq).map(|_| TopK::new(k)).collect(),
        })
    }

    /// One block: `p` (nq rows of m), the products of the queries with base vectors `offset` to
    /// `offset + m`; `norms` (m), their squared norms.
    pub fn push_l2(&mut self, p: &[f32], norms: &[f32], offset: usize) -> Result<()> {
        let m = norms.len();
        if m == 0 || p.len() != self.tops.len() * m {
            return Err("prod must be (nq, len(norms))".into());
        }
        self.tops
            .par_iter_mut()
            .zip(p.par_chunks(m))
            .for_each(|(top, row)| {
                for (j, (&pv, &n)) in row.iter().zip(norms).enumerate() {
                    top.push(n - 2.0 * pv, (offset + j) as u32);
                }
            });
        Ok(())
    }

    /// A group of IVF lists, each given as its products with the queries `rows` (distinct within a list),
    /// its vectors' squared norms and their ids in the base. The lists are ranked in parallel, each into its
    /// own best k per query, which are then merged into the queries' best k.
    pub fn push_l2_lists(&mut self, lists: &[ListProducts<'_>]) -> Result<()> {
        let nq = self.tops.len();
        for &(p, nm, id, rs) in lists {
            if id.len() != nm.len()
                || p.len() != rs.len() * nm.len()
                || rs.iter().any(|&r| r < 0 || r as usize >= nq)
            {
                return Err("each prod must be (len(rows), m), ids (m,), rows in range".into());
            }
        }
        let k = self.tops.first().map_or(1, |t| t.dist.len());
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
                self.tops[*r].push(v, i);
            }
        }
        Ok(())
    }

    /// The indices of each query's k nearest, flattened, nearest first.
    pub fn result(&self) -> Vec<u32> {
        self.tops
            .iter()
            .flat_map(|t| t.idx.iter().copied())
            .collect()
    }
}
