//! siann's CPU paths, in safe Rust: exact and IVF k-nearest neighbors for L1, L2, L-infinity and inner
//! product, the top-k that ranks BLAS products for L2 and inner product, and the scores of returned
//! neighbors. Built into the siann Python package as `siann._cpu`.
//!
//! The search itself is in the `siann` crate (`crates/siann`), plain Rust over slices; this crate binds it
//! to Python with PyO3:
//! it reads NumPy arrays (in place where it can), releases the GIL while the engine runs, and raises the
//! engine's errors as `ValueError`.

use numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::buffer::PyBuffer;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use siann::Mat;
use siann::elem::widen_half;
use siann::flat::Flat;
use siann::ivf::{Ivf, Options, Stored};
use siann::metric::Kind;
use siann::ops::{self, Buffer, Changes};
use siann::products::ListProducts;

/// The engine's error, as Python's `ValueError`.
fn value_error(e: String) -> PyErr {
    PyValueError::new_err(e)
}

fn metric(name: &str) -> PyResult<Kind> {
    Kind::parse(name).map_err(value_error)
}

/// A C-contiguous float32 buffer with `d` columns, copied.
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

/// A NumPy matrix, read in place; `what` is the error if it is not C-contiguous.
fn mat<'a, T: numpy::Element>(a: &'a PyReadonlyArray2<'_, T>, what: &str) -> PyResult<Mat<'a, T>> {
    let (rows, cols) = a.as_array().dim();
    let data = a
        .as_slice()
        .map_err(|_| PyValueError::new_err(what.to_string()))?;
    Ok(Mat { data, rows, cols })
}

/// A NumPy vector, read in place; `what` is the error if it is not contiguous.
fn vector<'a, T: numpy::Element>(a: &'a PyReadonlyArray1<'_, T>, what: &str) -> PyResult<&'a [T]> {
    a.as_slice()
        .map_err(|_| PyValueError::new_err(what.to_string()))
}

/// Exact search.
#[pyclass]
struct FlatIndex(Flat);

#[pymethods]
impl FlatIndex {
    /// Build from a C-contiguous float32 array of shape (n, d); `metric` is "l1", "l2", "linf" or "ip" (inner product, largest first).
    #[new]
    #[pyo3(signature = (xb, d, metric = "l1"))]
    fn py_new(py: Python<'_>, xb: &Bound<'_, PyAny>, d: usize, metric: &str) -> PyResult<Self> {
        let kind = self::metric(metric)?;
        let v = read(py, xb, d)?;
        Ok(FlatIndex(py.detach(|| Flat::new(&v, d, kind))))
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
        let v = read(py, xq, self.0.d)?;
        let out = py
            .detach(|| self.0.search(&v, k, chunk, split))
            .map_err(value_error)?;
        Ok(out.into_pyarray(py))
    }
}

/// Inverted lists over given centroids.
#[pyclass]
struct IvfIndex(Ivf);

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
        let kind = self::metric(metric)?;
        let assign_kind = match assign {
            Some(a) => self::metric(a)?,
            None if matches!(kind, Kind::Ip) => Kind::L2,
            None => kind,
        };
        let xb = read(py, xb, d)?;
        let copy = match (storage, stored) {
            ("float32", None) => Stored::F32,
            ("float16", Some(buf)) => Stored::F16(PyBuffer::<u16>::get(buf)?.to_vec(py)?),
            ("uint8", Some(buf)) => Stored::U8(PyBuffer::<u8>::get(buf)?.to_vec(py)?),
            _ => {
                return Err(PyValueError::new_err(
                    "storage is 'float32' (no stored), or 'float16' or 'uint8' with stored",
                ));
            }
        };
        let cents = read(py, centroids, d)?;
        let given = match &lists {
            Some(a) => Some(vector(a, "lists must be contiguous")?),
            None => None,
        };
        let ivf = py
            .detach(|| Ivf::new(xb, &cents, d, kind, assign_kind, copy, given))
            .map_err(value_error)?;
        Ok(IvfIndex(ivf))
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
        let v = read(py, xq, self.0.d)?;
        let p = match &products {
            Some(p) => Some(mat(p, "products must be C-contiguous")?),
            None => None,
        };
        let how = Options {
            split,
            gprobe,
            threads,
        };
        let out = py
            .detach(|| self.0.search(&v, k, nprobe, how, p))
            .map_err(value_error)?;
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
        let v = read(py, xq, self.0.d)?;
        let p = match &products {
            Some(p) => Some(mat(p, "products must be C-contiguous")?),
            None => None,
        };
        let out = py
            .detach(|| self.0.search_list_major(&v, k, nprobe, p))
            .map_err(value_error)?;
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
        let c = read(py, centres, self.0.d)?;
        let a = vector(&assign, "assign must be contiguous")?;
        self.0.set_coarse_groups(&c, a).map_err(value_error)
    }

    /// The number of vectors in each list.
    fn list_sizes(&self) -> Vec<usize> {
        self.0.list_sizes()
    }
}

/// The best k of each query's L2 distances, kept across blocks of base vectors, for exact L2 search
/// with the products from a BLAS matrix multiply (NumPy's `@`, Accelerate on a Mac): for each block,
/// `push_l2` takes the queries' products with the block's vectors and the vectors' squared norms, and
/// ranks by |x|^2 - 2 q.x (|q|^2 is the same for all of a query's candidates). Reads NumPy's arrays in
/// place, without copying; the queries' rows are split over threads.
#[pyclass]
struct L2TopK(siann::products::L2TopK);

#[pymethods]
impl L2TopK {
    #[new]
    fn py_new(nq: usize, k: usize) -> PyResult<Self> {
        Ok(L2TopK(
            siann::products::L2TopK::new(nq, k).map_err(value_error)?,
        ))
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
        let nm = vector(&norms, "norms must be contiguous")?;
        let tops = &mut self.0;
        py.detach(|| tops.push_l2(p, nm, offset))
            .map_err(value_error)
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
        let what = "arrays must be C-contiguous";
        let mut lists: Vec<ListProducts<'_>> = Vec::with_capacity(n);
        for i in 0..n {
            let p = prods[i]
                .as_slice()
                .map_err(|_| PyValueError::new_err(what))?;
            lists.push((
                p,
                vector(&norms[i], what)?,
                vector(&ids[i], what)?,
                vector(&rows[i], what)?,
            ));
        }
        let tops = &mut self.0;
        py.detach(|| tops.push_l2_lists(&lists))
            .map_err(value_error)
    }

    /// The indices of each query's k nearest, flattened (nq * k), nearest first.
    fn result<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u32>> {
        self.0.result().into_pyarray(py)
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
    let kind = self::metric(metric)?;
    let what = "x, q and ids must be C-contiguous";
    let (xm, qm, im) = (mat(&x, what)?, mat(&q, what)?, mat(&ids, what)?);
    let out = py
        .detach(|| ops::scores(xm, qm, im, kind))
        .map_err(value_error)?;
    PyArray1::from_vec(py, out).reshape([im.rows, im.cols])
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
    let kind = self::metric(metric)?;
    let what = "x, q and ids must be C-contiguous";
    let (xm, qm, im) = (mat(&x, what)?, mat(&q, what)?, mat(&ids, what)?);
    let (iv, sv) = py
        .detach(|| ops::rank(xm, qm, im, kind))
        .map_err(value_error)?;
    let shape = [im.rows, im.cols];
    Ok((
        PyArray1::from_vec(py, iv).reshape(shape)?,
        PyArray1::from_vec(py, sv).reshape(shape)?,
    ))
}

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
    let kind = self::metric(metric)?;
    let what = "arrays must be C-contiguous";
    let (xm, qm, rm) = (mat(&x, what)?, mat(&q, what)?, mat(&rows, what)?);
    let buffer = match (&xb, &brows, &bids) {
        (Some(bx), Some(br), Some(bi)) => Some(Buffer {
            x: mat(bx, what)?,
            rows: mat(br, what)?,
            ids: vector(bi, what)?,
        }),
        (None, None, None) => None,
        _ => {
            return Err(PyValueError::new_err(
                "merge_results: give xb, brows and bids together",
            ));
        }
    };
    let changes = Changes {
        dead: match &dead {
            Some(a) => Some(vector(a, what)?),
            None => None,
        },
        ids: match &ids {
            Some(a) => Some(vector(a, what)?),
            None => None,
        },
        buffer,
    };
    let (out_i, out_s, hits) = py
        .detach(|| ops::merge_results(xm, qm, rm, k, kind, changes))
        .map_err(value_error)?;
    let nq = qm.rows;
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
    let xm = mat(&x, "x must be C-contiguous")?;
    let (v, zeros) = py.detach(|| ops::normalize(xm.data, xm.cols));
    Ok((
        PyArray1::from_vec(py, v).reshape([xm.rows, xm.cols])?,
        zeros,
    ))
}

/// Each row's squared Euclidean norm (for ranking L2 by |x|^2 - 2 q.x).
#[pyfunction]
fn sq_norms<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'_, f32>,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let xm = mat(&x, "x must be C-contiguous")?;
    let out = py.detach(|| ops::sq_norms(xm.data, xm.cols));
    Ok(PyArray1::from_vec(py, out))
}

/// Whether every value of `x` is an integer from 0 to 255 (so that storing it as bytes loses nothing).
#[pyfunction]
fn is_bytes(py: Python<'_>, x: PyReadonlyArray2<'_, f32>) -> PyResult<bool> {
    let xm = mat(&x, "x must be C-contiguous")?;
    Ok(py.detach(|| ops::is_bytes(xm.data)))
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
    let a = vector(&assign, "assign must be contiguous")?;
    let (order, offsets) = ops::group_by_list(a, nlist).map_err(value_error)?;
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
    let xm = mat(&x, "x must be C-contiguous")?;
    let a = vector(&assign, "assign must be contiguous")?;
    let (k, cd) = c.readonly().as_array().dim();
    let mut cw = c.readwrite();
    let cs = cw
        .as_slice_mut()
        .map_err(|_| PyValueError::new_err("c must be C-contiguous"))?;
    let counts = ops::centroid_means(xm, a, cs, k, cd).map_err(value_error)?;
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
    let what = "arrays must be C-contiguous";
    let (f, s, t) = (mat(&found, what)?, mat(&scores, what)?, vector(&kth, what)?);
    ops::recall_of(f, s.data, t, larger_better).map_err(value_error)
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
    let r = mat(&rows, "rows must be C-contiguous")?;
    let o = vector(&own, "own must be contiguous")?;
    let out = ops::without_own(r, o, k).map_err(value_error)?;
    PyArray1::from_vec(py, out).reshape([r.rows, k])
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
