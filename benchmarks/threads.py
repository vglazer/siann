"""Searching one siann.Index from several threads at once: queries per second with 1 and 4 threads, each
thread searching in a loop, on SIFT1M on the CPU, exact search and an IVF (recall 0.95, calibrated on 1,000
of the dataset's queries), single queries and batches of 10.

    uv run --group bench python threads.py
"""
import threading
import time

import siann

from datasets import load


def main() -> None:
    xb, xq = load("sift", 2000)
    print("SIFT1M on the CPU, queries per second, by threads searching the same index at once")
    for kw, name in (({}, "exact"), ({"recall": 0.95, "calibration_queries": xq[:1000]}, "ivf")):
        index = siann.Index(xb, "l2", device="cpu", **kw)
        for b in (1, 10):
            def work(n, b=b):
                for i in range(n):
                    s = (i * b) % 1000
                    index.search(xq[s:s + b], 10)
            n = 400 if name == "ivf" else 20
            work(20)
            rates = []
            for threads in (1, 4):
                ts = [threading.Thread(target=work, args=(n,)) for _ in range(threads)]
                t = time.perf_counter()
                for x in ts:
                    x.start()
                for x in ts:
                    x.join()
                rates.append(threads * n * b / (time.perf_counter() - t))
            print(f"  {name:5s} batch {b:2d}: 1 thread {rates[0]:9,.0f}, 4 threads {rates[1]:9,.0f} "
                  f"({rates[1] / rates[0]:.2f}x)", flush=True)


if __name__ == "__main__":
    main()
