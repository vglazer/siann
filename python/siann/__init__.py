"""siann: fast similarity search on Apple silicon.

The GPU paths run on MLX (``siann.knn_kernels``, ``siann.exact``, ``siann.ivf_gpu``; the same kernels on
PyTorch's MPS backend in ``siann.torch_kernels`` and ``siann.ivf_torch``), the CPU paths in Rust
(``siann._cpu``: ``FlatIndex``, ``IvfIndex``, ``L2TopK``) and through NumPy's BLAS (``siann.blas_l2``).
"""
