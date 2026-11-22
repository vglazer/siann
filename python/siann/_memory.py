"""How much GPU memory siann lets itself use. On Apple silicon the GPU's memory is the machine's, so a search
that outgrows it swaps or stalls the system rather than failing. siann therefore splits GPU searches so that
each call's working memory stays within a budget, caps MLX's cache of freed buffers, and keeps an index whose
GPU copy would not fit on the CPU. Splitting a search never changes its results: each query's depends on it
alone. The budget depends on the machine, through the GPU's recommended working set (what Metal reports it
can use without paging); the results do not."""
from __future__ import annotations

import functools

GPU_BUDGET = 4 << 30      # a GPU call's working memory at most; larger calls are no faster (`gpu_chunks.py`)
BUDGET_SHARE = 4          # and at most a quarter of the working set, on smaller machines
FIT_SHARE = 2             # an index whose GPU copy exceeds half the working set stays on the CPU
CACHE_SHARE = 2           # MLX's buffer cache: at most twice the budget


@functools.cache
def working_set(backend: str) -> int | None:
    """The GPU's recommended working set in bytes, as its backend reports it, or None where it does not."""
    try:
        if backend == "torch":
            import torch
            return int(torch.mps.recommended_max_memory())
        import mlx.core as mx
        info = mx.device_info() if hasattr(mx, "device_info") else mx.metal.device_info()
        return int(info["max_recommended_working_set_size"])
    except Exception:  # noqa: BLE001 (no GPU, or a backend without the call: no limit known)
        return None


def budget(backend: str) -> int:
    """Bytes a GPU call may work in: GPU_BUDGET, or a quarter of the working set where that is smaller."""
    ws = working_set(backend)
    return GPU_BUDGET if ws is None else min(GPU_BUDGET, ws // BUDGET_SHARE)


def fits(nbytes: int, backend: str) -> bool:
    """Whether a GPU copy of nbytes leaves room to search it: at most half the working set."""
    ws = working_set(backend)
    return ws is None or nbytes <= ws // FIT_SHARE


@functools.cache
def cap_mlx_cache() -> None:
    """Caps MLX's cache of freed buffers (by default as large as the working set, so one large search can leave
    gigabytes held) at twice the budget, unless the program set a lower cap itself. Once per process."""
    import mlx.core as mx
    cap = CACHE_SHARE * budget("mlx")
    previous = mx.set_cache_limit(cap)
    if previous < cap:
        mx.set_cache_limit(previous)


def gb(nbytes: int) -> str:
    return f"{nbytes / (1 << 30):.1f} GB" if nbytes >= 1 << 30 else f"{nbytes / (1 << 20):,.0f} MB"
