"""Growing an index from empty or from small: searches of an empty index, the choice between an IVF and exact
search made at merges as the index grows, index.calibrate, removing every vector, and all of it saved and
loaded."""
import numpy as np
import pytest

import siann
from siann import _index

DEVICES = ["cpu"] + (["gpu"] if siann.gpu_available() else [])


def clustered(n, d, seed, centers=None):
    rng = np.random.default_rng(seed)
    c = centers if centers is not None else rng.standard_normal((50, d), dtype=np.float32)
    return c[rng.integers(0, len(c), n)] + 0.3 * rng.standard_normal((n, d), dtype=np.float32)


CENTERS = np.random.default_rng(9).standard_normal((50, 16), dtype=np.float32)


def data(n, seed):
    return clustered(n, 16, seed, CENTERS)


def recall(found, truth):
    return np.mean([len(set(a) & set(b)) / len(b) for a, b in zip(found, truth)])


@pytest.fixture
def small_limits(monkeypatch):
    """Merges after 500 added vectors, so that tests reach them."""
    monkeypatch.setattr(_index, "BUFFER_MIN", 500)
    monkeypatch.setattr(_index, "BUFFER_SHARE", 0.0)
    monkeypatch.setattr(_index, "DEAD_MAX", 8)


@pytest.mark.parametrize("metric", siann.METRICS)
def test_an_empty_index_finds_nothing_then_what_is_added(metric):
    index = siann.Index(np.empty((0, 16), np.float32), metric)
    assert index.n == 0 and index.d == 16 and index.kind == "exact"
    worst = -np.inf if metric in ("cosine", "ip") else np.inf
    for device in DEVICES:
        r = index.search(data(3, 1), 5, device=device)
        assert r.indices.shape == (3, 5) and (r.indices == -1).all() and (r.scores == worst).all()
        assert index.search(data(1, 1)[0], 5, device=device).indices.shape == (1, 5)
    x, q = data(7, 2), data(4, 3)
    index.add(x)
    ref = siann.Index(x, metric)
    for device in DEVICES:
        ours, theirs = index.search(q, 10, device=device), ref.search(q, 10, device=device)
        np.testing.assert_array_equal(ours.indices, theirs.indices)
        np.testing.assert_allclose(ours.scores, theirs.scores, rtol=1e-5, atol=1e-5)
        assert (ours.indices[:, 7:] == -1).all()


def test_refuses_vectors_without_dimensions():
    with pytest.raises(ValueError, match="no dimensions"):
        siann.Index(np.empty((0, 0), np.float32), "l2")


def test_without_a_recall_target_it_stays_exact(small_limits):
    index = siann.Index(np.empty((0, 16), np.float32), "l2", device="cpu")
    index.add(data(2_000, 4))
    assert index.kind == "exact" and index._merges >= 1


@pytest.mark.parametrize("kind", ["auto", "ivf"])
def test_grown_from_empty_the_choice_is_made_at_the_first_merge(kind, small_limits):
    index = siann.Index(np.empty((0, 16), np.float32), "l2", recall=0.9, kind=kind, device="cpu")
    assert "at the first merge" in str(index.plan)
    index.add(data(499, 5))
    assert index.kind == "exact" and index._tested_n == 0             # still in the buffer
    index.add(data(1, 6))
    assert index._tested_n == 500
    text = str(index.plan)
    assert "grown from empty: at the merge at 500 vectors" in text
    assert ("IVF chosen" in text or "kept exact search" in text) and "calibrated: nprobe" in text
    if kind == "ivf":
        assert index.kind == "ivf" and "kind='ivf' given" in text


def test_an_ivf_grown_from_empty_meets_its_target(small_limits):
    index = siann.Index(np.empty((0, 16), np.float32), "l2", recall=0.9, kind="ivf", device="cpu",
                        calibration_queries=data(300, 7))
    x = data(6_000, 8)
    for s in range(0, len(x), 1_000):
        index.add(x[s:s + 1_000])
    assert index.kind == "ivf" and index._buf is None
    q = data(300, 9)
    truth = siann.Index(x, "l2").search(q, 10, device="cpu").indices
    assert recall(index.search(q, 10, device="cpu").indices, truth) >= 0.85
    assert "of the given calibration queries" in str(index.plan)


def test_the_choice_is_made_again_only_once_the_index_has_doubled(monkeypatch, small_limits):
    """An index that keeps exact search tries the IVF again at a merge only once it has doubled since the last
    try, so that a large exact index does not build an IVF at every merge."""
    tried = []
    real = siann.Index._build_ivf

    def counting(self, *a, **kw):
        tried.append(len(self._x))
        lines = real(self, *a, **kw)
        self._ivf = None                                              # as if exact search had stayed faster
        return lines
    monkeypatch.setattr(siann.Index, "_build_ivf", counting)
    index = siann.Index(data(1_000, 10), "l2", recall=0.9, device="cpu")
    assert tried == [1_000]
    for s in range(5):
        index.add(data(500, 11 + s))
    # merges at 1,500, 2,000, 2,500, 3,000, 3,500: tried again at 2,000 (twice 1,000), not until 4,000 after
    assert tried == [1_000, 2_000] and index._tested_n == 2_000
    assert "made again at the merge at 2,000 vectors (last made at 1,000)" in str(index.plan)


def test_kind_exact_is_never_revisited(small_limits):
    index = siann.Index(data(600, 12), "l2", recall=0.9, kind="exact", device="cpu")
    index.add(data(2_000, 13))
    assert index.kind == "exact" and index._tested_n == 0


def test_removing_every_vector_then_growing_again(small_limits):
    x = data(3_000, 14)
    index = siann.Index(x, "l2", recall=0.9, kind="ivf", device="cpu")
    index.remove(np.arange(3_000))
    assert index.n == 0
    assert (index.search(data(2, 15), 5, device="cpu").indices == -1).all()
    assert index.kind == "exact" and "empty: added vectors" in str(index.plan)   # compacted: back to empty
    y = data(1_000, 16)
    new = index.add(y)
    assert index.kind == "ivf" and index.n == 1_000
    q = data(100, 17)
    truth = siann.Index(y, "l2").search(q, 10, device="cpu").indices
    assert recall(index.search(q, 10, device="cpu").indices, new[truth]) >= 0.85


def test_calibrate_an_ivf(small_limits):
    x = data(5_000, 18)
    index = siann.Index(x, "l2", recall=0.9, kind="ivf", device="cpu")
    assert "of its own vectors" in str(index.plan)
    q = data(400, 19) + 0.2
    nprobe = index.calibrate(q)
    assert nprobe == index.nprobe
    assert "given to index.calibrate()" in str(index.plan) and "of its own vectors" not in str(index.plan)
    truth = siann.Index(x, "l2").search(q, 10, device="cpu").indices
    assert recall(index.search(q, 10, device="cpu").indices, truth) >= 0.88
    cal = index._cal_q.copy()
    index.add(data(600, 20))                                          # a merge recalibrates on the same queries
    assert index._cal_ids is None and np.array_equal(index._cal_q, cal)


def test_calibrate_an_exact_index_keeps_the_queries_for_the_choice(small_limits):
    index = siann.Index(np.empty((0, 16), np.float32), "l2", recall=0.9, device="cpu")
    assert index.calibrate(data(200, 21)) is None
    index.add(data(500, 22))
    assert "of the given calibration queries" in str(index.plan)


def test_calibrate_needs_a_recall_target():
    with pytest.raises(ValueError, match="recall target"):
        siann.Index(data(100, 23), "l2").calibrate(data(10, 24))
    with pytest.raises(ValueError, match="calibration_queries"):
        siann.Index(data(100, 23), "l2", recall=0.9, kind="exact").calibrate(np.ones((3, 5), np.float32))


def test_an_empty_index_saved_and_loaded_carries_on(tmp_path, small_limits):
    index = siann.Index(np.empty((0, 16), np.float32), "l2", recall=0.9, kind="ivf", device="cpu",
                        calibration_queries=data(200, 25))
    index.save(tmp_path / "e.siann")
    again = siann.Index.load(tmp_path / "e.siann", device="cpu")
    assert again.n == 0 and again._kind_arg == "ivf" and again._cal_given.shape == (200, 16)
    assert (again.search(data(2, 26), 3).indices == -1).all()
    again.add(data(500, 27))
    assert again.kind == "ivf" and "of the given calibration queries" in str(again.plan)


def test_an_exact_index_saved_and_loaded_keeps_its_last_choice(tmp_path, monkeypatch, small_limits):
    real = siann.Index._build_ivf

    def exact_wins(self, *a, **kw):
        lines = real(self, *a, **kw)
        self._ivf = None
        return lines
    monkeypatch.setattr(siann.Index, "_build_ivf", exact_wins)
    index = siann.Index(data(800, 28), "l2", recall=0.9, device="cpu")
    index.save(tmp_path / "x.siann")
    monkeypatch.setattr(siann.Index, "_build_ivf", real)
    again = siann.Index.load(tmp_path / "x.siann", device="cpu")
    assert again._tested_n == 800 and again._kind_arg == "auto"
    again.add(data(500, 29))                                          # 1,300: under twice 800
    assert again._tested_n == 800
    again.add(data(500, 30))                                          # 1,800: the choice made again
    assert again._tested_n == 1_800
