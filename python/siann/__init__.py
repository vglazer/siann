"""siann: fast similarity search on Apple silicon.

``siann.Index(xb, metric)`` builds an index over dense base vectors and ``index.search(xq, k)``
returns each query's k nearest, exactly, on the GPU or the CPU (BLAS through NumPy and siann's
Rust kernels), chosen by batch size from a ``siann.Profile``: siann's default, or one measured on your own
data and hardware by ``siann.tune``; ``index.plan`` says what was chosen and why. The kernels themselves are in
``siann.knn_kernels``, ``siann.exact``, ``siann.ivf_gpu`` and ``siann.blas_l2`` (the same kernels on
PyTorch's MPS backend in ``siann.torch_kernels`` and ``siann.ivf_torch``), and the Rust ones in
``siann._cpu``.
"""
from importlib.metadata import version as _version

from . import _build  # noqa: F401 (checks the processor before the Rust extension loads)
from ._index import METRICS, Index, Plan, SearchResult
from ._index import _gpu_available as gpu_available
from ._profile import DEFAULT as DEFAULT_PROFILE
from ._profile import Profile, tune, use_profile

__version__ = _version("siann")
__all__ = [
           "DEFAULT_PROFILE",
           "METRICS",
           "Index",
           "Plan",
           "Profile",
           "SearchResult",
           "__version__",
           "gpu_available",
           "tune",
           "use_profile",
]
