"""siann: fast similarity search on Apple silicon.

``siann.Index(xb, metric)`` builds an index over dense base vectors and ``index.search(xq, k)``
returns each query's k nearest, exactly, on the GPU (MLX) or the CPU (BLAS through NumPy and siann's
Rust kernels); ``index.plan`` says what was chosen and why. The kernels themselves are in
``siann.knn_kernels``, ``siann.exact``, ``siann.ivf_gpu`` and ``siann.blas_l2`` (the same kernels on
PyTorch's MPS backend in ``siann.torch_kernels`` and ``siann.ivf_torch``), and the Rust ones in
``siann._cpu``.
"""
from importlib.metadata import version as _version

from ._index import METRICS, Index, Plan, SearchResult

__version__ = _version("siann")
__all__ = ["METRICS", "Index", "Plan", "SearchResult", "__version__"]
