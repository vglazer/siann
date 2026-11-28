"""Searching from several threads, alongside changes: every search returns what exact search returns on one of
the states the index passed through (never a mix of two), with merges and compactions happening meanwhile."""
import threading
import time

import siann
from siann import _index

import numpy as np
import pytest

DEVICES = ["cpu"] + (["gpu"] if siann.gpu_available() else [])


def clustered(n, d, seed):
    rng = np.random.default_rng(seed)
    c = np.random.default_rng(99).standard_normal((30, d), dtype=np.float32)
    return c[rng.integers(0, 30, n)] + 0.3 * rng.standard_normal((n, d), dtype=np.float32)


@pytest.fixture
def small_limits(monkeypatch):
    monkeypatch.setattr(_index, "BUFFER_MIN", 300)
    monkeypatch.setattr(_index, "BUFFER_SHARE", 0.0)
    monkeypatch.setattr(_index, "DEAD_MAX", 20)


def run(threads):
    errors = []

    def guarded(fn):
        def go():
            try:
                fn()
            except Exception as e:  # noqa: BLE001 (reported by the test)
                errors.append(e)
        return go
    ts = [threading.Thread(target=guarded(t)) for t in threads]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert not errors, errors[:3]


@pytest.mark.parametrize("device", DEVICES)
def test_concurrent_searches_match_one_thread(device):
    x = clustered(20_000, 16, 1)
    q = clustered(200, 16, 2)
    for index in (siann.Index(x, "l2", device=device), siann.Index(x, "l1", recall=0.9, kind="ivf", device=device)):
        ref = {b: index.search(q[:b], 10).indices for b in (1, 37, 200)}
        found = []

        def searcher(seed=0, index=index, found=found):
            rng = np.random.default_rng(seed)
            for _ in range(30):
                b = int(rng.choice([1, 37, 200]))
                found.append((b, index.search(q[:b], 10).indices))
        run([lambda s=s: searcher(s) for s in range(6)])
        assert all(np.array_equal(got, ref[b]) for b, got in found)


@pytest.mark.parametrize("metric", ["l2", "l1"])
def test_searches_during_changes_see_one_state(metric, small_limits):
    """An exact index changed by one thread (adds of 100, a removal every third step: merges at 300 buffered,
    compactions past 20 removed) and searched by four: each result equals exact search on one of its states."""
    x = clustered(3_000, 8, 3)
    q = clustered(20, 8, 4)
    index = siann.Index(x[:1_000], metric, device="cpu")
    rng = np.random.default_rng(5)
    live = {i: x[i] for i in range(1_000)}                            # id -> vector, as the index will hold them
    next_id, steps, states = 1_000, [], []

    def state():                                                      # exact search on these vectors, by id
        ids = np.array(sorted(live), dtype=np.int64)
        found = siann.Index(np.stack([live[i] for i in ids]), metric, device="cpu").search(q, 10)
        return ids[found.indices], found.scores
    states.append(state())
    for s in range(20):                                               # the changes, decided in advance
        batch = x[1_000 + 100 * s:1_100 + 100 * s]
        steps.append(("add", batch))
        live.update({next_id + i: v for i, v in enumerate(batch)})
        next_id += len(batch)
        states.append(state())
        if s % 3 == 2:
            gone = rng.choice(np.array(sorted(live)), size=8, replace=False)
            steps.append(("remove", gone))
            for i in gone:
                live.pop(int(i))
            states.append(state())
    seen = []
    done = threading.Event()

    def changer():
        for kind, arg in steps:
            if kind == "add":
                index.add(arg)
            else:
                index.remove(arg)
            time.sleep(0.003)                                         # let searches run between changes
        done.set()

    def searcher():
        while not done.is_set():
            seen.append(index.search(q, 10))
    run([changer] + [searcher] * 4)
    assert index._merges > 0 and len(seen) > 20
    for r in seen:
        assert any(np.array_equal(r.indices, ids) and np.allclose(r.scores, sc, atol=1e-4) for ids, sc in states)
