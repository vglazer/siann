//! Building a CAGRA graph from a k-nearest-neighbor graph, as CAGRA does (Ootomo et al., 2024; NVIDIA cuVS's
//! `graph_core`): each vector's candidate links are reordered by how many detours they have, the best `degree`
//! kept, and the second half of each list given over to reverse links (links into the vector from others), so
//! that every vector can be reached. The k-nearest-neighbor graph itself (each vector's candidates, nearest
//! first) comes from elsewhere: siann's IVF searching the base vectors.
//!
//! A link u -> v has a detour through w, another of u's candidates, when u -> w and w -> v both rank before
//! u -> v (max(rank(u, w), rank(w, v)) < rank(u, v)): v is then reachable through w, and the direct link adds
//! less. The count depends only on ranks, not distances, so the build is the same for every metric.

use rayon::prelude::*;
use std::cell::RefCell;

use super::Result;

/// Per thread: `stamp[x]` is 1 + the vector whose candidates `rank[x]` currently describes (0: none), so that
/// nothing is cleared between vectors.
struct Marks {
    stamp: Vec<u32>,
    rank: Vec<u16>,
}

thread_local! {
    static MARKS: RefCell<Marks> = const { RefCell::new(Marks { stamp: Vec::new(), rank: Vec::new() }) };
}

/// The graph's links, (n, `degree`) rows: from `knn` (n, `k`) rows, each a vector's candidates nearest first
/// (-1 for none; a vector's own row and repeats are skipped). Each row of the result is the `degree / 2` links
/// with the fewest detours (ties by rank), then reverse links (the vectors that kept this one, those that
/// ranked it highest first) and the vector's next-best links in turn, until `degree` are filled; -1 where a
/// vector has fewer.
pub fn optimize(knn: &[i64], n: usize, k: usize, degree: usize) -> Result<Vec<i64>> {
    if k == 0
        || degree == 0
        || knn.len() != n * k
        || n >= u32::MAX as usize
        || k > u16::MAX as usize
    {
        return Err("need knn of n rows of k > 0 candidates, degree > 0".into());
    }
    if knn.iter().any(|&v| v < -1 || v >= n as i64) {
        return Err("each candidate must be -1 or a row".into());
    }
    // a vector's candidates as rows: its own row and repeats removed, order kept
    let cand: Vec<Vec<u32>> = (0..n)
        .into_par_iter()
        .map(|u| {
            let mut out: Vec<u32> = Vec::with_capacity(k);
            for &v in &knn[u * k..(u + 1) * k] {
                if v >= 0 && v as usize != u && !out.contains(&(v as u32)) {
                    out.push(v as u32);
                }
            }
            out
        })
        .collect();

    // each vector's candidates reordered by detours, the best `degree` kept
    let pruned: Vec<Vec<u32>> = (0..n)
        .into_par_iter()
        .map(|u| {
            MARKS.with(|m| {
                let mut m = m.borrow_mut();
                if m.stamp.len() < n {
                    m.stamp = vec![0; n];
                    m.rank = vec![0; n];
                }
                let list = &cand[u];
                let tag = u as u32 + 1;
                for (r, &x) in list.iter().enumerate() {
                    m.stamp[x as usize] = tag;
                    m.rank[x as usize] = r as u16;
                }
                let mut detours = vec![0u32; list.len()];
                for (i, &w) in list.iter().enumerate() {
                    for (r_wx, &x) in cand[w as usize].iter().enumerate() {
                        if m.stamp[x as usize] == tag {
                            let r_ux = m.rank[x as usize] as usize;
                            if r_ux > i.max(r_wx) {
                                detours[r_ux] += 1;
                            }
                        }
                    }
                }
                let mut order: Vec<usize> = (0..list.len()).collect();
                order.sort_by_key(|&r| (detours[r], r));
                order.truncate(degree);
                order.iter().map(|&r| list[r]).collect()
            })
        })
        .collect();

    // reverse links: for each vector, the ones that kept it, by the rank they gave it, then by row
    let mut count = vec![0usize; n + 1];
    for list in &pruned {
        for &v in list {
            count[v as usize + 1] += 1;
        }
    }
    for i in 0..n {
        count[i + 1] += count[i];
    }
    let mut fill = count.clone();
    let mut back = vec![(0u32, 0u32); count[n]];
    for (u, list) in pruned.iter().enumerate() {
        for (p, &v) in list.iter().enumerate() {
            back[fill[v as usize]] = (p as u32, u as u32);
            fill[v as usize] += 1;
        }
    }

    let protect = degree / 2;
    let mut out = vec![-1i64; n * degree];
    out.par_chunks_mut(degree).enumerate().for_each(|(u, row)| {
        let mut rev: Vec<(u32, u32)> = back[count[u]..count[u + 1]].to_vec();
        rev.sort_unstable();
        let fwd = &pruned[u];
        let mut taken: Vec<u32> = Vec::with_capacity(degree);
        for &v in fwd.iter().take(protect) {
            taken.push(v);
        }
        for &(_, v) in &rev {
            if taken.len() == degree {
                break;
            }
            if !taken.contains(&v) {
                taken.push(v);
            }
        }
        for &v in fwd.iter().skip(protect) {
            if taken.len() == degree {
                break;
            }
            if !taken.contains(&v) {
                taken.push(v);
            }
        }
        for (slot, &v) in row.iter_mut().zip(&taken) {
            *slot = v as i64;
        }
    });
    Ok(out)
}
