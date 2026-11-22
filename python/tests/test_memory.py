"""GPU memory: an index too large for the GPU stays on the CPU (and says so), searches split to stay within the
budget return exactly what one call returns, and MLX's cache cap leaves a lower cap of the program's own."""
import numpy as np
import pytest

import siann
from siann import _index, _ivf, _memory

gpu_only = pytest.mark.skipif(not siann.gpu_available(), reason="no GPU available")


def clustered(n, d, seed):
    rng = np.random.default_rng(seed)
    c = rng.standard_normal((40, d), dtype=np.float32)
    return c[rng.integers(0, 40, n)] + 0.3 * rng.standard_normal((n, d), dtype=np.float32)


@pytest.fixture
def small_gpu(monkeypatch):
    """A GPU that reports 1 MB to work with: an index of more than 512 KB does not fit."""
    if not siann.gpu_available():
        pytest.skip("no GPU available")
    monkeypatch.setattr(_memory, "working_set", lambda backend: 1 << 20)


def test_too_large_for_the_gpu_stays_on_the_cpu(small_gpu):
    x = clustered(5_000, 32, 1)                                       # 640 KB
    index = siann.Index(x, "l2")
    assert index.route(10_000) == "cpu"
    assert "too large for the GPU" in str(index.plan) and "every search on the CPU" in str(index.plan)
    ref = siann.Index(x, "l2", device="cpu").search(x[:20], 5).indices
    np.testing.assert_array_equal(index.search(x[:20], 5).indices, ref)
    with pytest.raises(RuntimeError, match="too large for the GPU"):
        index.search(x[:20], 5, device="gpu")
    with pytest.raises(RuntimeError, match="too large for the GPU"):
        siann.Index(x, "l2", device="gpu")
    ivf = siann.Index(x, "l2", recall=0.9, kind="ivf")
    assert ivf._build_device() == "cpu" and ivf.route(10_000) == "cpu"


def test_grown_too_large_moves_to_the_cpu(small_gpu, monkeypatch):
    monkeypatch.setattr(_index, "BUFFER_MIN", 500)
    monkeypatch.setattr(_index, "BUFFER_SHARE", 0.0)
    x = clustered(4_000, 32, 2)                                       # 512 KB: fits, just
    index = siann.Index(x[:3_000], "l2")
    assert index.route(10_000) == "gpu" and "memory: " in str(index.plan)
    index.add(x[3_000:])                                              # merged: 4,000 vectors, still fits
    assert index.route(10_000) == "gpu"
    index.add(clustered(500, 32, 3))                                  # 4,500: no longer
    assert index.route(10_000) == "cpu" and "too large for the GPU" in str(index.plan)


def test_a_saved_index_too_large_loads_on_the_cpu(small_gpu, tmp_path):
    x = clustered(5_000, 32, 4)
    siann.Index(x, "l1", device="cpu").save(tmp_path / "big.siann")
    loaded = siann.Index.load(tmp_path / "big.siann")
    assert loaded.route(10_000) == "cpu" and "too large for the GPU" in str(loaded.plan)
    with pytest.raises(RuntimeError, match="too large"):
        siann.Index.load(tmp_path / "big.siann", device="gpu")


@gpu_only
@pytest.mark.parametrize("path", ["gpu-query", "gpu-list"])
def test_ivf_searches_split_into_chunks_match_one_call(path, monkeypatch):
    x = clustered(30_000, 24, 5)
    q = np.ascontiguousarray(clustered(3_000, 24, 6))
    index = siann.Index(x, "l2", recall=0.9, kind="ivf")
    monkeypatch.setattr(_memory, "BUDGET_SHARE", 1)
    monkeypatch.setattr(_memory, "GPU_BUDGET", 1 << 40)
    whole = index._ivf.search(q, 10, index.nprobe, path)
    monkeypatch.setattr(_memory, "GPU_BUDGET", 1)                     # the smallest chunks: 1,024 queries
    assert _ivf.gpu_chunk(10, index.nprobe, index._ivf.nlist, _memory.budget("mlx")) == 1024
    np.testing.assert_array_equal(index._ivf.search(q, 10, index.nprobe, path), whole)


@gpu_only
@pytest.mark.parametrize("metric", siann.METRICS)
def test_exact_tiles_shrunk_for_memory_match(metric, monkeypatch):
    x = clustered(20_000, 16, 7)
    q = clustered(300, 16, 8)
    whole = siann.Index(x, metric).search(q, 10, device="gpu")
    monkeypatch.setattr(_memory, "GPU_BUDGET", 6 * 20_000 * 7)       # 7 queries per tile
    assert _index._exact_tile(20_000, "mlx") == 7
    small = siann.Index(x, metric).search(q, 10, device="gpu")
    np.testing.assert_array_equal(small.indices, whole.indices)
    np.testing.assert_array_equal(small.scores, whole.scores)


def test_budget_follows_the_working_set(monkeypatch):
    monkeypatch.setattr(_memory, "working_set", lambda backend: 64 << 30)
    assert _memory.budget("mlx") == _memory.GPU_BUDGET                # 4 GB: a quarter would be 16
    monkeypatch.setattr(_memory, "working_set", lambda backend: 8 << 30)
    assert _memory.budget("mlx") == 2 << 30                           # a quarter of 8 GB
    monkeypatch.setattr(_memory, "working_set", lambda backend: None)
    assert _memory.budget("mlx") == _memory.GPU_BUDGET and _memory.fits(1 << 50, "mlx")


@gpu_only
def test_the_cache_cap_keeps_a_lower_one_of_the_programs():
    import mlx.core as mx
    previous = mx.set_cache_limit(1 << 20)                            # the program's own, lower
    try:
        _memory.cap_mlx_cache.__wrapped__()
        assert mx.set_cache_limit(1 << 20) == 1 << 20
        mx.set_cache_limit(1 << 50)                                   # a higher one is capped
        _memory.cap_mlx_cache.__wrapped__()
        assert mx.set_cache_limit(1 << 50) == _memory.CACHE_SHARE * _memory.budget("mlx")
    finally:
        mx.set_cache_limit(previous)
