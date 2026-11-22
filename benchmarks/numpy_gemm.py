"""The products siann's exact L2 search computes, a tile of m queries against blocks of the base, through
NumPy's BLAS (Accelerate on a Mac, OpenBLAS on Linux), on the same shapes as faer_gemm (a matrix multiply in
safe Rust): whether a pure-Rust multiply could replace BLAS in siann's CPU paths.

    uv run --group bench python numpy_gemm.py
    (cd faer_gemm && cargo run --release)
"""
import time

import numpy as np


def main() -> None:
    blas = np.show_config(mode="dicts")["Build Dependencies"]["blas"]["name"]
    print(f"NumPy {np.__version__}, BLAS {blas}; GFLOP/s by (queries, dimensions)")
    for d, n in ((128, 1_000_000), (1536, 100_000)):
        for m in (1, 10, 100, 512):
            block = min(1 << ((1 << 22) // m - 1).bit_length(), n) // 2
            q = (np.arange(m * d, dtype=np.float32).reshape(m, d) % 17) * 0.1
            x = (np.arange(block * d, dtype=np.float32).reshape(block, d) % 11) * 0.1
            blocks = -(-n // block)

            def run():
                for _ in range(blocks):
                    q @ x.T
            run()
            t = time.perf_counter()
            for _ in range(3):
                run()
            secs = (time.perf_counter() - t) / 3
            print(f"  m {m:4}, d {d:5}: {2 * m * blocks * block * d / secs / 1e9:8.1f} GFLOP/s "
                  f"({secs * 1e3:.2f} ms per pass, block {block})")


if __name__ == "__main__":
    main()
