"""Where a samply profile's time goes: each sample's innermost frame mapped to its function through the
symbols sidecar (`<profile>.syms.json`, written by `samply record --unstable-presymbolicate`), summed
over all threads, and also by "inclusive" time for the functions given with --inclusive.

    uv run --group bench python samply_top.py profile.json.gz --top 25
"""
import argparse
import bisect
import collections
import gzip
import json


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("profile")
    p.add_argument("--top", type=int, default=25)
    p.add_argument("--inclusive", default="search_one,coarse,scan_one,select_nth,TopK,int_dist,par_chunks",
                   help="substrings of function names to total inclusively")
    a = p.parse_args()
    with gzip.open(a.profile) as f:
        prof = json.load(f)
    with open(a.profile.removesuffix(".gz").removesuffix(".json") + ".json.syms.json", encoding="utf-8") as f:
        syms = json.load(f)
    strings = syms["string_table"]
    tables = {}
    for lib in syms["data"]:
        st = sorted(lib["symbol_table"], key=lambda e: e["rva"])
        tables[lib["debug_name"]] = ([e["rva"] for e in st], st)
    libs = prof["libs"]

    def name(lib_index, addr):
        lib = libs[lib_index]["debugName"] if lib_index is not None else "?"
        if lib not in tables:
            return f"{lib}+{addr:#x}"
        rvas, st = tables[lib]
        i = bisect.bisect_right(rvas, addr) - 1
        if i >= 0 and addr < st[i]["rva"] + st[i]["size"]:
            return strings[st[i]["symbol"]]
        return f"{lib}+{addr:#x}"

    self_time = collections.Counter()
    incl = collections.Counter()
    total = 0
    wanted = [w for w in a.inclusive.split(",") if w]
    names = {}                                          # (thread, frame) -> function name
    for t in prof["threads"]:
        ft, fun, res, stk = t["frameTable"], t["funcTable"], t["resourceTable"], t["stackTable"]

        def frame_name(f, cache=names):
            key = (id(t), f)
            if key not in cache:
                r = fun["resource"][ft["func"][f]]
                lib = res["lib"][r] if r is not None and r >= 0 else None
                cache[key] = name(lib, ft["address"][f])
            return cache[key]
        for s in t["samples"]["stack"]:
            if s is None:
                continue
            total += 1
            self_time[frame_name(stk["frame"][s])] += 1
            seen = set()
            while s is not None:
                n = frame_name(stk["frame"][s])
                for w in wanted:
                    if w in n and w not in seen:
                        seen.add(w)
                        incl[w] += 1
                s = stk["prefix"][s]
    print(f"{total:,} samples over {len(prof['threads'])} threads\n\nself time (innermost frame):")
    for n, c in self_time.most_common(a.top):
        print(f"  {100 * c / total:5.1f}%  {n[:150]}")
    print("\ninclusive (any frame of the stack):")
    for w in wanted:
        print(f"  {100 * incl[w] / total:5.1f}%  *{w}*")


if __name__ == "__main__":
    main()
