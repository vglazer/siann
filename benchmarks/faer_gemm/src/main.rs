//! The products siann's exact L2 search computes, a tile of m queries (m x d) against blocks of the base
//! (block x d, transposed), timed through faer, a matrix multiply in safe Rust, on every core; numpy_gemm.py
//! times the same shapes through NumPy's BLAS. Run with `cargo run --release` from this folder.
use faer::linalg::matmul::matmul;
use faer::{Accum, Mat, Par};
use std::time::Instant;

fn main() {
    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    let par = Par::rayon(threads);
    println!(
        "faer, {threads} threads; GFLOP/s by (queries, dimensions): one pass over n base vectors in blocks"
    );
    for &(d, n) in &[(128usize, 1_000_000usize), (1536, 100_000)] {
        for &m in &[1usize, 10, 100, 512] {
            let block = ((1usize << 22) / m).next_power_of_two().min(n) / 2; // about 2^22 products per block
            let q = Mat::<f32>::from_fn(m, d, |i, j| ((i * 7 + j * 13) % 17) as f32 * 0.1);
            let x = Mat::<f32>::from_fn(block, d, |i, j| ((i * 3 + j * 5) % 11) as f32 * 0.1);
            let mut c = Mat::<f32>::zeros(m, block);
            let blocks = n.div_ceil(block);
            let run = |c: &mut Mat<f32>| {
                for _ in 0..blocks {
                    matmul(
                        c.as_mut(),
                        Accum::Replace,
                        q.as_ref(),
                        x.as_ref().transpose(),
                        1.0f32,
                        par,
                    );
                }
            };
            run(&mut c);
            let reps = 3;
            let t = Instant::now();
            for _ in 0..reps {
                run(&mut c);
            }
            let secs = t.elapsed().as_secs_f64() / reps as f64;
            let flops = 2.0 * m as f64 * (blocks * block) as f64 * d as f64;
            println!(
                "  m {m:4}, d {d:5}: {:8.1} GFLOP/s ({:.2} ms per pass, block {block})",
                flops / secs / 1e9,
                secs * 1e3
            );
        }
    }
}
