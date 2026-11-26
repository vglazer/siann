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
    assert "graph index: CAGRA" in text and "by mlxvs" in text and "search width" in text and "ef " in text
    assert "neighbors depend on the path" in text
    assert all(index._ivf_path(b, None).endswith("graph") for b in (1, 10, 1000))   # kind="graph": every batch
    assert recall(index.search(q, 10).indices, truth) >= 0.85
    found = index._ivf.search(q, 10, index.nprobe, "gpu-graph")
    assert recall(found, truth) >= 0.85 and (found >= 0).all()


@needs_mlxvs
def test_single_queries_search_the_graph_on_the_cpu(built):
    _, q, index, truth = built
    assert index.route(1) == "cpu" and index._ivf_path(1, None) == "cpu-graph"   # kind="graph": the CPU too
    found = np.concatenate([index.search(q[i], 10).indices for i in range(len(q))])
    assert recall(found, truth) >= 0.85
    assert index._ivf.graph.ef >= 10 and "on the CPU, ef" in str(index.plan)


@needs_mlxvs
def test_saved_and_loaded_with_its_graph(built, tmp_path):
    _x, q, index, _ = built
    index.save(tmp_path / "g.siann")
    again = siann.Index.load(tmp_path / "g.siann")
    assert again._ivf.graph is not None and again._ivf.graph.itopk == index._ivf.graph.itopk
    np.testing.assert_array_equal(again._ivf.search(q, 10, again.nprobe, "gpu-graph"),
                                  index._ivf.search(q, 10, index.nprobe, "gpu-graph"))


@needs_mlxvs
def test_a_file_with_a_graph_searches_it_on_the_cpu_without_mlxvs(built, tmp_path, monkeypatch):
    """As on a Linux machine: the graph built on the Mac's GPU, saved, loaded without mlxvs, searched by siann's
    own CPU code at the CPU width calibrated when it was built."""
    _, q, index, truth = built
    index.save(tmp_path / "g.siann")
    monkeypatch.setattr(_graph, "mlxvs_version", lambda: None)
    again = siann.Index.load(tmp_path / "g.siann", device="cpu")
    assert again._ivf.graph is not None and again._ivf.graph.ef == index._ivf.graph.ef
    assert "searched on the CPU only" in str(again.plan) and again._ivf_path(1, None) == "cpu-graph"
    np.testing.assert_array_equal(again.search(q, 10, device="cpu").indices, index._ivf.search(q, 10, 0, "cpu-graph"))
    assert recall(again.search(q, 10).indices, truth) >= 0.85


def test_a_merge_without_mlxvs_rebuilds_the_graph(tmp_path, monkeypatch):
    monkeypatch.setattr(_index, "BUFFER_MIN", 300)
    monkeypatch.setattr(_index, "BUFFER_SHARE", 0.0)
    monkeypatch.setattr(_graph, "mlxvs_version", lambda: None)        # as on a Linux machine
    x, q = clustered(6_000, 16, 8), clustered(200, 16, 9)
    siann.Index(x[:5_000], "l2", recall=0.9, kind="graph", calibration_queries=q).save(tmp_path / "m.siann")
    again = siann.Index.load(tmp_path / "m.siann", device="cpu")
    again.add(x[5_000:])                                              # merged: no mlxvs to extend it, so rebuilt
    g = again._ivf.graph
    assert g is not None and g.builder == "siann" and g.links().shape == (6_000, 32)
    truth = siann.Index(x, "l2").search(q, 10, device="cpu").indices
    assert recall(again.search(q, 10).indices, truth) >= 0.85


@pytest.mark.parametrize("metric", ["l2", "cosine", "ip"])
def test_siann_builds_its_own_graph_without_mlxvs(metric, tmp_path, monkeypatch):
    monkeypatch.setattr(_graph, "mlxvs_version", lambda: None)        # no mlxvs: siann's own builder, anywhere
    x, q = clustered(20_000, 16, 5), clustered(400, 16, 6)
    index = siann.Index(x, metric, recall=0.9, kind="graph", calibration_queries=q[:200])
    g = index._ivf.graph
    assert g is not None and g.builder == "siann" and "by siann" in str(index.plan)
    assert "searched on the CPU" in str(index.plan) or not siann.gpu_available() or all(
        not index._ivf_path(b, None).startswith("gpu-graph") for b in (1, 100, 5000))
    truth = siann.Index(x, metric).search(q[200:], 10, device="cpu").indices
    assert recall(index.search(q[200:], 10).indices, truth) >= 0.85
    index.save(tmp_path / "s.siann")
    again = siann.Index.load(tmp_path / "s.siann", device="cpu")
    assert again._ivf.graph.builder == "siann"
    np.testing.assert_array_equal(again._ivf.graph.links(), g.links())


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
    monkeypatch.setattr(_graph, "mlxvs_version", lambda: None)        # without mlxvs: siann's own graph instead
    assert siann.Index(x, "l2", recall=0.9, kind="graph")._ivf.graph.builder == "siann"
    index = siann.Index(x, "l2", recall=0.9, kind="ivf")              # kind="ivf" never builds one
    assert index._ivf.graph is None
    with pytest.raises(ValueError, match="needs a recall target"):
        siann.Index(x, "l2", kind="graph")
