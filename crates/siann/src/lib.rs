//! siann's CPU search in plain Rust, over slices: exact and IVF k-nearest neighbors for L1, L2, L-infinity
//! and inner product, with no Python types, so that it is tested and could be used without PyO3. The
//! siann Python package binds it as `siann._cpu` (the `siann-py` crate); this crate is not published.
//!
//! - `metric`: the four metrics, as types the kernels are generic over, and as a run-time `Kind`.
//! - `elem`: how list vectors are stored (f32, float16 bits, bytes).
//! - `kernels`: the blocked layout, the distance loops over one block, and `TopK`.
//! - `flat`: exact search.
//! - `graph`: search of a proximity graph (CAGRA's, built on a GPU) on the CPU.
//! - `ivf`: the inverted lists: their coarse step and both scans.
//! - `products`: ranking the products of a BLAS matrix multiply (`L2TopK`).
//! - `ops`: scoring, ranking and merging results, and the arithmetic the Python side would otherwise do
//!   in NumPy.

pub mod elem;
pub mod flat;
pub mod graph;
pub mod graph_build;
pub mod ivf;
pub mod kernels;
pub mod metric;
pub mod ops;
pub mod products;
#[cfg(test)]
mod tests;

/// An error, as a message: the bindings raise it as Python's `ValueError`.
pub type Result<T> = std::result::Result<T, String>;

/// A matrix borrowed from the caller, row-major: `data` holds `rows` rows of `cols` values.
#[derive(Clone, Copy)]
pub struct Mat<'a, T> {
    pub data: &'a [T],
    pub rows: usize,
    pub cols: usize,
}
