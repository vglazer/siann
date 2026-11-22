"""A graph index alongside the IVF (CAGRA through mlxvs): built and calibrated where mlxvs is installed, routed
to the batch sizes where it is faster, kept through changes, saved and loaded, and refused or skipped, with a
reason, where it cannot be used. Skipped where mlxvs or an Apple GPU is missing (as in CI)."""
import numpy as np
import pytest

import siann
from siann import _graph, _index

has_mlxvs = _graph.mlxvs_version() is not None
needs_mlxvs = pytest.mark.skipif(not has_mlxvs, reason="mlxvs (siann[graph]) with an Apple GPU not available")


def clustered(n, d, seed, integer=False):
    rng = np.random.default_rng(seed)
    c = np.random.default_rng(42).standard_normal((60, d), dtype=np.float32)
    x = c[rng.integers(0, 60, n)] + 0.3 * rng.standard_normal((n, d), dtype=np.float32)
    if integer:
        x = np.clip(np.rint((x + 3) * 40), 0, 255).astype(np.float32)
    return x


def recall(found, truth):
    return np.mean([len(set(a) & set(b)) / len(b) for a, b in zip(found, truth)])


@pytest.fixture(scope="module")
def built():
    if not has_mlxvs:
        pytest.skip("mlxvs not available")
    x, q = clustered(30_000, 24, 1), clustered(1_200, 24, 2)
    index = siann.Index(x, "l2", recall=0.9, kind="graph", calibration_queries=q[:600])
    truth = siann.Index(x, "l2").search(q[600:], 10, device="cpu").indices
    return x, q[600:], index, truth


@needs_mlxvs
def test_the_graph_meets_the_target_and_the_plan_says_so(built):
    _x, q, index, truth = built
    text = str(index.plan)
    assert "graph index: CAGRA through mlxvs" in text and "search width" in text
    assert "neighbors depend on the path" in text
    assert index._ivf_path(1000, None) == "gpu-graph"                # kind="graph": every GPU batch
    assert recall(index.search(q, 10).indices, truth) >= 0.85
    found = index._ivf.search(q, 10, index.nprobe, "gpu-graph")
    assert recall(found, truth) >= 0.85 and (found >= 0).all()


@needs_mlxvs
def test_small_batches_stay_on_the_cpu_ivf(built):
    _, _q, index, _ = built
    assert index.route(1) == "cpu" and index._ivf_path(1, None) == "cpu-split"


@needs_mlxvs
def test_saved_and_loaded_with_its_graph(built, tmp_path):
    _x, q, index, _ = built
    index.save(tmp_path / "g.siann")
    again = siann.Index.load(tmp_path / "g.siann")
    assert again._ivf.graph is not None and again._ivf.graph.itopk == index._ivf.graph.itopk
    np.testing.assert_array_equal(again._ivf.search(q, 10, again.nprobe, "gpu-graph"),
                                  index._ivf.search(q, 10, index.nprobe, "gpu-graph"))


@needs_mlxvs
def test_a_file_with_a_graph_loads_without_mlxvs(built, tmp_path, monkeypatch):
    _, _q, index, _ = built
    index.save(tmp_path / "g.siann")
    monkeypatch.setattr(_graph, "why_not", lambda metric, gpu: "no graph index: mlxvs is not installed")
    again = siann.Index.load(tmp_path / "g.siann")
    assert again._ivf.graph is None and "graph index in the file not loaded" in str(again.plan)
    assert again._ivf_path(10_000, None).startswith("gpu-") and again._ivf_path(10_000, None) != "gpu-graph"


@needs_mlxvs
def test_changes_keep_the_graph(monkeypatch):
    monkeypatch.setattr(_index, "BUFFER_MIN", 500)
    monkeypatch.setattr(_index, "BUFFER_SHARE", 0.0)
    monkeypatch.setattr(_index, "DEAD_MAX", 20)
    x = clustered(12_000, 16, 3, integer=True)
    q = clustered(300, 16, 4, integer=True)
    index = siann.Index(x[:10_000], "l2", recall=0.9, kind="graph", calibration_queries=q)
    index.add(x[10_000:])                                            # a merge: the graph extended
    assert index._ivf.graph._cagra._graph.shape[0] == 12_000
    found = index._ivf.search(q, 10, index.nprobe, "gpu-graph")
    truth = siann.Index(x, "l2").search(q, 10, device="cpu").indices
    assert recall(found, truth) >= 0.85
    gone = index.remove(np.arange(30))                               # past DEAD_MAX: compacted, graph rebuilt
    assert gone == 30 and index._ivf.graph._cagra._graph.shape[0] == 11_970
    assert not np.isin(index.search(q, 10, device="gpu").indices, np.arange(30)).any()


@needs_mlxvs
@pytest.mark.parametrize("metric", ["cosine", "ip"])
def test_cosine_and_inner_product(metric):
    x, q = clustered(20_000, 16, 5), clustered(400, 16, 6)
    index = siann.Index(x, metric, recall=0.9, kind="graph", calibration_queries=q[:200])
    truth = siann.Index(x, metric).search(q[200:], 10, device="cpu").indices
    found = index._ivf.search(np.ascontiguousarray(q[200:] / np.linalg.norm(q[200:], axis=1, keepdims=True))
                              if metric == "cosine" else q[200:], 10, index.nprobe, "gpu-graph")
    assert recall(found, truth) >= 0.8


def test_kind_graph_refused_where_it_cannot_run(monkeypatch):
    x = clustered(3_000, 8, 7)
    with pytest.raises(RuntimeError, match="CAGRA has no L1"):
        siann.Index(x, "l1", recall=0.9, kind="graph", device="gpu" if siann.gpu_available() else "cpu")
    monkeypatch.setattr(_graph, "mlxvs_version", lambda: None)
    if siann.gpu_available():
        with pytest.raises(RuntimeError, match="mlxvs is not installed"):
            siann.Index(x, "l2", recall=0.9, kind="graph")
    index = siann.Index(x, "l2", recall=0.9, kind="ivf")              # kind="ivf" never builds one
    assert index._ivf.graph is None
    with pytest.raises(ValueError, match="needs a recall target"):
        siann.Index(x, "l2", kind="graph")
