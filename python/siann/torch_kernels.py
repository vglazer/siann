"""siann's GPU kernels on PyTorch's MPS backend through torch.mps.compile_shader.

The kernel bodies are siann._metal's, the same text as on MLX; the compile-time constants (D, T, K) are
`constexpr` constants here, as MLX's template arguments make them there, so each set compiles once and is
cached, and the run-time sizes go in a small ``sz`` buffer. The small final selection over the T * k
survivors uses torch.topk where knn_kernels uses MLX's argpartition.
"""
import torch

from ._metal import COMBINE, SELECT, TILED

_HEADER = "#include <metal_stdlib>\nusing namespace metal;\n"
_LIBS = {}


def _lib(source: str, **consts: int):
    """The compiled library for this source and these compile-time constants, cached."""
    key = (source, tuple(sorted(consts.items())))
    if key not in _LIBS:
        decl = "".join(f"constant constexpr uint {name} = {value};\n" for name, value in consts.items())
        _LIBS[key] = torch.mps.compile_shader(_HEADER + decl + source)
    return _LIBS[key]


_SZ = {}


def sz(*sizes: int, device="mps") -> torch.Tensor:
    """The run-time sizes a kernel reads from ``sz``, in its order (as 32-bit integers). Each distinct set
    is made on the device once and kept: making a small tensor on MPS costs a copy from the host, about a
    fifth of a small search."""
    key = (sizes, str(device))
    if key not in _SZ:
        _SZ[key] = torch.tensor(sizes, dtype=torch.int32, device=device)
    return _SZ[key]


def select_k(dist: torch.Tensor, k: int, threads: int = 512) -> torch.Tensor:
    """The column indices of the k smallest entries of each row, in increasing order."""
    q, n = dist.shape
    vals = torch.empty(q * threads * k, device=dist.device, dtype=torch.float32)
    idx = torch.empty(q * threads * k, device=dist.device, dtype=torch.int32)
    _lib(SELECT.torch_source(), T=threads, K=k).select_k(dist, sz(n, q, device=dist.device), vals, idx,
                                                         threads=[threads, q], group_size=[256, 1])
    vals, idx = vals.view(q, -1), idx.view(q, -1)
    best = torch.topk(vals, k, dim=1, largest=False, sorted=True).indices
    return torch.gather(idx, 1, best)


def dist(q: torch.Tensor, x: torch.Tensor, metric: str) -> torch.Tensor:
    """The L1 or L-infinity ("linf") distance matrix between the rows of q and of x, tiled."""
    nq, d = q.shape
    n = x.shape[0]
    out = torch.empty(nq, n, device=q.device, dtype=torch.float32)
    groups = ((n + 63) // 64, (nq + 63) // 64)
    _lib(COMBINE[metric] + TILED.torch_source(), D=d).tiled(q, x, sz(n, nq, device=q.device), out,
                                                            threads=[groups[0] * 16, groups[1] * 16],
                                                            group_size=[16, 16])
    return out


def knn_tiled(q: torch.Tensor, x: torch.Tensor, k: int, metric: str, threads: int = 512) -> torch.Tensor:
    """The k nearest rows of x to each row of q in L1 or L-infinity distance: the tiled distance
    matrix, then select_k."""
    return select_k(dist(q, x, metric), k, threads)


def l2_knn(q: torch.Tensor, x: torch.Tensor, norms: torch.Tensor, k: int, threads: int = 512) -> torch.Tensor:
    """The k nearest rows of x to each row of q in L2 distance: a matrix multiply, then select_k.
    ``norms`` holds the squared norms of the rows of x."""
    # addmm fuses the multiply, scaling and addition: 18.1 ms against 31.4 ms for 256 x 1M.
    return select_k(torch.addmm(norms[None, :], q, x.T, alpha=-2), k, threads)
