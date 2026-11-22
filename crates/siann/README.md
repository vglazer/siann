# siann (Rust engine)

siann's CPU search in plain Rust, over slices: exact search and an IVF (inverted lists over given
centroids, stored as f32, float16 or bytes) for L1, L2, L-infinity and inner product, with the scoring,
ranking and merging of results. It has no Python types; the siann Python package binds it as `siann._cpu`
(`crates/siann-py`) and ships it inside its wheels. This crate is not published: the `siann` name on
crates.io holds a placeholder.

`cargo test` checks it against brute force (`src/tests.rs`); the full checks, against NumPy, are the Python
tests and `benchmarks/check_kernels.py`.
