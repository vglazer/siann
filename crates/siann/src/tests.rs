//! The engine against brute force, without Python: exact search for every metric, the IVF probing every list,
//! the storages and the integer path, and the bookkeeping functions. The full checks, against NumPy, are
//! `benchmarks/check_kernels.py` and the Python tests.

use super::elem::widen_half;
use super::flat::Flat;
use super::graph::{Graph, Rows};
use super::ivf::{Ivf, Options, Stored};
use super::kernels::TopK;
use super::metric::{Kind, score};
use super::{Mat, ops};

const KINDS: [Kind; 4] = [Kind::L1, Kind::L2, Kind::Linf, Kind::Ip];

/// `n` rows of `d` pseudo-random values in [-1, 1), the same on every run.
fn data(n: usize, d: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    (0..n * d)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 40) as f32 / (1u64 << 23) as f32 - 1.0
        })
        .collect()
}

/// Each query's k nearest rows of `x` by brute force: smallest distance (largest inner product) first,
/// ties by row.
fn brute(x: &[f32], q: &[f32], d: usize, k: usize, kind: Kind) -> Vec<u32> {
    q.chunks_exact(d)
        .flat_map(|qv| {
            let mut all: Vec<(f32, u32)> = x
                .chunks_exact(d)
                .enumerate()
                .map(|(i, xv)| {
                    let s = score(kind, qv, xv);
                    (if kind.similarity() { -s } else { s }, i as u32)
                })
                .collect();
            all.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
            all.into_iter().take(k).map(|(_, i)| i)
        })
        .collect()
}

/// Whether two results name the same neighbors in each query, allowing for neighbors whose scores differ
/// only by rounding (the kernels sum in another order than `score`).
fn same_up_to_rounding(a: &[u32], b: &[u32], x: &[f32], q: &[f32], d: usize, k: usize, kind: Kind) {
    for (qi, (ra, rb)) in a.chunks(k).zip(b.chunks(k)).enumerate() {
        let qv = &q[qi * d..(qi + 1) * d];
        for (&ia, &ib) in ra.iter().zip(rb) {
            let (sa, sb) = (
                score(kind, qv, &x[ia as usize * d..(ia as usize + 1) * d]),
                score(kind, qv, &x[ib as usize * d..(ib as usize + 1) * d]),
            );
            assert!(
                (sa - sb).abs() <= 1e-4 * sa.abs().max(1.0),
                "query {qi}: {ia} against {ib}"
            );
        }
    }
}

fn mat(data: &[f32], rows: usize, cols: usize) -> Mat<'_, f32> {
    Mat { data, rows, cols }
}

#[test]
fn topk_keeps_the_smallest_ties_by_id() {
    let mut t = TopK::new(3);
    for (v, i) in [(5.0, 1), (1.0, 9), (1.0, 2), (3.0, 4), (1.0, 7)] {
        t.push(v, i);
    }
    assert_eq!(t.idx, vec![2, 7, 9]);
    assert_eq!(t.dist, vec![1.0, 1.0, 1.0]);
}

#[test]
fn float16_widens_exactly() {
    assert_eq!(widen_half(0x3c00), 1.0);
    assert_eq!(widen_half(0xc000), -2.0);
    assert_eq!(widen_half(0x7bff), 65504.0);
    assert_eq!(widen_half(0x0001), 2f32.powi(-24)); // the smallest subnormal
    assert_eq!(widen_half(0x8000).to_bits(), (-0.0f32).to_bits());
}

#[test]
fn exact_search_matches_brute_force() {
    let (n, d, k) = (1003, 19, 7);
    let (x, q) = (data(n, d, 1), data(37, d, 2));
    for kind in KINDS {
        let flat = Flat::new(&x, d, kind);
        let whole = flat.search(&q, k, 32, false).unwrap();
        same_up_to_rounding(&whole, &brute(&x, &q, d, k, kind), &x, &q, d, k, kind);
        // dividing the base among threads, or taking queries 4 at a time, changes nothing
        assert_eq!(flat.search(&q, k, 32, true).unwrap(), whole);
        assert_eq!(flat.search(&q, k, 4, true).unwrap(), whole);
    }
    assert!(
        Flat::new(&x, d, Kind::L2)
            .search(&q, n + 1, 32, true)
            .is_err()
    );
}

#[test]
fn ivf_probing_every_list_is_exact() {
    let (n, d, k, nlist) = (2000, 12, 10, 23);
    let (x, q) = (data(n, d, 3), data(41, d, 4));
    let cents = &x[..nlist * d];
    for kind in KINDS {
        let flat = Flat::new(&x, d, kind).search(&q, k, 32, true).unwrap();
        let ivf = Ivf::new(x.clone(), cents, d, kind, kind, Stored::F32, None).unwrap();
        assert_eq!(ivf.list_sizes().iter().sum::<usize>(), n);
        let how = |split| Options {
            split,
            gprobe: 0,
            threads: 0,
        };
        let query = ivf.search(&q, k, nlist, how(1), None).unwrap();
        same_up_to_rounding(&query, &flat, &x, &q, d, k, kind);
        assert_eq!(ivf.search(&q, k, nlist, how(4), None).unwrap(), query);
        let list = ivf.search_list_major(&q, k, nlist, None).unwrap();
        same_up_to_rounding(&list, &flat, &x, &q, d, k, kind);
    }
}

#[test]
fn byte_lists_and_the_integer_path_agree_with_float_lists() {
    let (n, d, k, nlist) = (1500, 16, 10, 17);
    let x: Vec<f32> = data(n, d, 5)
        .iter()
        .map(|v| ((v + 1.0) * 127.0).floor())
        .collect();
    let q: Vec<f32> = x[..30 * d].iter().map(|v| (v + 3.0).min(255.0)).collect();
    let bytes: Vec<u8> = x.iter().map(|&v| v as u8).collect();
    let cents = &x[..nlist * d];
    for kind in [Kind::L1, Kind::L2, Kind::Linf] {
        let how = Options {
            split: 1,
            gprobe: 0,
            threads: 0,
        };
        let float = Ivf::new(x.clone(), cents, d, kind, kind, Stored::F32, None).unwrap();
        let u8s = Ivf::new(
            x.clone(),
            cents,
            d,
            kind,
            kind,
            Stored::U8(bytes.clone()),
            None,
        )
        .unwrap();
        // integer queries take the integer path, exact: the same neighbors, ties included
        assert_eq!(
            u8s.search(&q, k, 5, how, None).unwrap(),
            float.search(&q, k, 5, how, None).unwrap()
        );
    }
    let mut wrong = bytes.clone();
    wrong[0] = wrong[0].wrapping_add(1);
    assert!(
        Ivf::new(
            x.clone(),
            cents,
            d,
            Kind::L2,
            Kind::L2,
            Stored::U8(wrong),
            None
        )
        .is_err()
    );
}

#[test]
fn results_ranked_and_merged() {
    let (n, d) = (50, 4);
    let (x, q) = (data(n, d, 6), data(2, d, 7));
    let m = |data, rows| mat(data, rows, d);
    let ids = [3i64, -1, 7, 1, 7, 2];
    let (ranked, scores) = ops::rank(
        m(&x, n),
        m(&q, 2),
        Mat {
            data: &ids,
            rows: 2,
            cols: 3,
        },
        Kind::L2,
    )
    .unwrap();
    assert_eq!(ranked[2], -1); // -1 last
    assert!(scores[0] <= scores[1] && scores[3] <= scores[4] && scores[4] <= scores[5]);
    let dead: Vec<bool> = (0..n).map(|i| i == 7).collect();
    let changes = ops::Changes {
        dead: Some(&dead),
        ..Default::default()
    };
    let (merged, _, hits) = ops::merge_results(
        m(&x, n),
        m(&q, 2),
        Mat {
            data: &ids,
            rows: 2,
            cols: 3,
        },
        2,
        Kind::L2,
        changes,
    )
    .unwrap();
    assert!(!merged.contains(&7)); // a removed row is never returned
    assert_eq!(hits, vec![1, 1]);
}

#[test]
fn grouping_by_list_is_stable() {
    let (order, offsets) = ops::group_by_list(&[2, 0, 2, 1, 0], 3).unwrap();
    assert_eq!(order, vec![1, 4, 3, 0, 2]);
    assert_eq!(offsets, vec![0, 2, 3, 5]);
    assert!(ops::group_by_list(&[3], 3).is_err());
}

#[test]
fn graph_search_on_a_nearest_neighbor_graph_is_exact() {
    // each vector linked to its 24 true nearest: best-first search from any entry finds the exact neighbors
    let (n, d, k, deg) = (1500, 8, 5, 24);
    let x = data(n, d, 9);
    let q: Vec<f32> = x[..40 * d].to_vec();
    for kind in [Kind::L2, Kind::L1, Kind::Ip] {
        let nn = Flat::new(&x, d, kind)
            .search(&x, deg + 1, 32, true)
            .unwrap();
        let links: Vec<i64> = nn
            .chunks(deg + 1)
            .flat_map(|r| r[1..].iter().map(|&i| i as i64))
            .collect();
        let g = Graph::new(&x, d, kind, Rows::F32(x.clone()), &links, deg).unwrap();
        let found = g.search(&q, k, 64).unwrap();
        same_up_to_rounding(&found, &brute(&x, &q, d, k, kind), &x, &q, d, k, kind);
    }
    assert!(Graph::new(&x, d, Kind::L2, Rows::F32(x.clone()), &[0; 3], 2).is_err());
}

#[test]
fn an_optimized_graph_keeps_its_rows_valid_and_finds_the_neighbors() {
    // a 32-nearest graph pruned to 16 links: no vector linked to itself or twice, every vector reachable (linked
    // from some other), and best-first search on it finds nearly every true neighbor of new queries
    let (n, d, k, kin, deg) = (3000, 8, 10, 32, 16);
    let x = data(n, d, 11);
    let q = data(200, d, 12);
    for kind in [Kind::L2, Kind::L1, Kind::Ip] {
        let nn = Flat::new(&x, d, kind)
            .search(&x, kin + 1, 32, true)
            .unwrap();
        let knn: Vec<i64> = nn.iter().map(|&i| i as i64).collect(); // each row's own first: skipped
        let links = super::graph_build::optimize(&knn, n, kin + 1, deg).unwrap();
        let mut into = vec![0usize; n];
        for (u, row) in links.chunks(deg).enumerate() {
            let valid: Vec<i64> = row.iter().copied().filter(|&v| v >= 0).collect();
            assert!(
                valid.iter().all(|&v| v as usize != u),
                "a vector linked to itself"
            );
            let mut s = valid.clone();
            s.sort_unstable();
            s.dedup();
            assert_eq!(s.len(), valid.len(), "a link repeated");
            for &v in &valid {
                into[v as usize] += 1;
            }
        }
        assert!(
            into.iter().filter(|&&c| c == 0).count() < n / 100,
            "vectors no link reaches"
        );
        let g = Graph::new(&x, d, kind, Rows::F32(x.clone()), &links, deg).unwrap();
        let found = g.search(&q, k, 48).unwrap();
        let truth = brute(&x, &q, d, k, kind);
        let hits: usize = found
            .chunks(k)
            .zip(truth.chunks(k))
            .map(|(f, t)| f.iter().filter(|v| t.contains(v)).count())
            .sum();
        let r = hits as f64 / truth.len() as f64;
        assert!(r > 0.95, "recall {r}");
    }
    assert!(super::graph_build::optimize(&[0, 1, 2], 2, 2, 1).is_err()); // 3 values for 2 x 2
    assert!(super::graph_build::optimize(&[0, 5], 2, 1, 1).is_err()); // 5 is not a row
}
