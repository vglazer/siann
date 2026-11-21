"""The Metal kernels, written once for both GPU backends.

Each kernel is a body of Metal statements plus how it names its thread positions. For MLX
(`mx.fast.metal_kernel`) the positions are MLX's built-in names; for PyTorch (`torch.mps.compile_shader`)
they are declared in the kernel's signature. The bodies are the same text for both, so the two backends
run the same arithmetic in the same order.

Sizes that only bound loops (how many queries, vectors in a row, probes) are read at run time from a
small `sz` buffer, so a new batch size does not compile a new kernel. Sizes that set the length of an
array in registers or threadgroup memory (k, threads per query) or that a loop is better unrolled for
(the dimension, D) are compile-time constants: MLX template arguments, PyTorch `constexpr` constants.
"""

# How each metric folds one coordinate pair (q, x) into a running distance a, as a Metal header.
# Inner product ("ip") is negated, so that, as for the distances, smaller is nearer.
COMBINE = {
    "l1": "inline float combine(float a, float q, float x) { return a + metal::abs(q - x); }\n",
    "l2": "inline float combine(float a, float q, float x) { float e = q - x; return a + e * e; }\n",
    "linf": "inline float combine(float a, float q, float x) { return metal::max(a, metal::abs(q - x)); }\n",
    "ip": "inline float combine(float a, float q, float x) { return a - q * x; }\n",
}

_KEEP_BEST = """
    float best[K];
    int where[K];
    for (int a = 0; a < int(K); ++a) { best[a] = INFINITY; where[a] = -1; }
"""
_INSERT = """
        if (v < best[K - 1]) {
            int p = int(K) - 1;
            while (p > 0 && best[p - 1] > v) { best[p] = best[p - 1]; where[p] = where[p - 1]; --p; }
            best[p] = v;
            where[p] = int(j);
        }
"""
_WRITE = """
    for (int a = 0; a < int(K); ++a) {
        out_val[(i * T + t) * K + a] = best[a];
        out_idx[(i * T + t) * K + a] = where[a];
    }
"""

# Thread positions, by layout: (MLX prologue, PyTorch signature attributes, PyTorch prologue).
POSITIONS = {
    # a 2-D grid: x the thread within a query (t), y the query (i)
    "grid": ("    uint t = thread_position_in_grid.x;\n    uint i = thread_position_in_grid.y;\n",
             "uint2 pos [[thread_position_in_grid]]",
             "    uint t = pos.x, i = pos.y;\n"),
    # 2-D threadgroups of 16 x 16 threads over a tiled matrix
    "tile": ("    uint tx = thread_position_in_threadgroup.x, ty = thread_position_in_threadgroup.y;\n"
             "    uint gx = threadgroup_position_in_grid.x, gy = threadgroup_position_in_grid.y;\n",
             "uint2 tpos [[thread_position_in_threadgroup]], uint2 gpos [[threadgroup_position_in_grid]]",
             "    uint tx = tpos.x, ty = tpos.y, gx = gpos.x, gy = gpos.y;\n"),
    # 1-D threadgroups: l the group, tid the thread within it
    "group": ("    uint l = threadgroup_position_in_grid.x;\n    uint tid = thread_position_in_threadgroup.x;\n",
              "uint l [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]]",
              ""),
}


class Kernel:
    """A kernel's text for both backends. ``inputs`` and ``outputs`` are (name, Metal element type);
    ``sizes`` are the run-time sizes, read from ``sz`` in that order."""

    def __init__(self, name: str, inputs, outputs, sizes, positions: str, body: str):
        self.name, self.inputs, self.outputs, self.sizes = name, inputs, outputs, sizes
        self.positions, self.body = positions, body

    def _sizes(self) -> str:
        return "".join(f"    const uint {s} = sz[{n}];\n" for n, s in enumerate(self.sizes))

    def mlx_source(self) -> str:
        return POSITIONS[self.positions][0] + self._sizes() + self.body

    def torch_source(self) -> str:
        params = [f"device const {t}* {n}" for n, t in self.inputs] + ["device const uint* sz"]
        params += [f"device {t}* {n}" for n, t in self.outputs] + [POSITIONS[self.positions][1]]
        return (f"kernel void {self.name}({', '.join(params)}) {{\n" + POSITIONS[self.positions][2]
                + self._sizes() + self.body + "}\n")


# The k smallest entries of each of Q rows of N, first stage: each of T threads per row keeps the best K
# of a strided slice in registers.
SELECT = Kernel("select_k", [("dist", "float")], [("out_val", "float"), ("out_idx", "int")], ("N", "Q"), "grid", """
    if (t >= T || i >= Q) return;
""" + _KEEP_BEST + """
    for (uint j = t; j < N; j += T) {
        float v = dist[i * N + j];
""" + _INSERT + """
    }
""" + _WRITE)

# A tiled L1 (or L-infinity) distance matrix, like a matrix multiply: each threadgroup of 16 x 16 threads
# computes a 64 x 64 block of distances (4 x 4 per thread, in registers), staging 16 dimensions of 64
# queries and 64 base vectors at a time in threadgroup memory, so each base vector is read once per 64
# queries. Padding is zero in both, a difference of 0 that changes neither metric.
TILED = Kernel("tiled", [("q", "float"), ("x", "float")], [("out", "float")], ("N", "Q"), "tile", """
    const uint B = 64, S = 16, R = 4;
    threadgroup float qs[B][S + 1];
    threadgroup float xs[B][S + 1];
    uint col0 = gx * B, row0 = gy * B;
    uint lane = ty * 16 + tx;
    float acc[R][R];
    for (uint a = 0; a < R; ++a) for (uint b = 0; b < R; ++b) acc[a][b] = 0.0f;
    for (uint d0 = 0; d0 < D; d0 += S) {
        for (uint e = lane; e < B * S; e += 256) {
            uint r = e / S, c = e % S;
            uint qi = row0 + r, xi = col0 + r, dd = d0 + c;
            qs[r][c] = (qi < Q && dd < D) ? q[qi * D + dd] : 0.0f;
            xs[r][c] = (xi < N && dd < D) ? x[xi * D + dd] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint c = 0; c < S; ++c) {
            float qv[R], xv[R];
            for (uint a = 0; a < R; ++a) { qv[a] = qs[ty * R + a][c]; xv[a] = xs[tx * R + a][c]; }
            for (uint a = 0; a < R; ++a) for (uint b = 0; b < R; ++b) acc[a][b] = combine(acc[a][b], qv[a], xv[b]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint a = 0; a < R; ++a) for (uint b = 0; b < R; ++b) {
        uint qi = row0 + ty * R + a, xi = col0 + tx * R + b;
        if (qi < Q && xi < N) out[qi * N + xi] = acc[a][b];
    }
""")

_SCAN_LOOP = """
    float best[K];
    int where[K];
    for (int a = 0; a < int(K); ++a) { best[a] = INFINITY; where[a] = -1; }
    for (uint p = 0; p < P; ++p) {
        int l = probes[i * P + p];
        for (int j = offsets[l] + int(t); j < offsets[l + 1]; j += int(T)) {
            float v = 0.0f;
            for (uint d = 0; d < D; ++d) { v = combine(v, q[i * D + d], x[uint(j) * D + d]); }
            if (v < best[K - 1]) {
                int s = int(K) - 1;
                while (s > 0 && best[s - 1] > v) { best[s] = best[s - 1]; where[s] = where[s - 1]; --s; }
                best[s] = v;
                where[s] = j;
            }
        }
    }
"""


def _ivf_inputs(x_type: str):
    return [("q", "float"), ("x", x_type), ("offsets", "int"), ("probes", "int")]


def scan(x_type: str) -> Kernel:
    """The query-major IVF scan: T threads per query walk its P probed lists, each keeping its best K."""
    return Kernel("ivf_scan", _ivf_inputs(x_type), [("out_val", "float"), ("out_idx", "int")], ("Q", "P"), "grid",
                  "    if (t >= T || i >= Q) return;\n" + _SCAN_LOOP + _WRITE)


def scan_merged(x_type: str) -> Kernel:
    """The query-major scan with the merge inside the kernel: each query's T threads form one threadgroup
    (T a power of two up to 256); each thread's sorted best K goes to threadgroup memory, and the lists are
    merged pairwise in log2(T) steps (the left first on ties), so only K results per query are written."""
    return Kernel("ivf_scan_merged", _ivf_inputs(x_type), [("out_val", "float"), ("out_idx", "int")], ("Q", "P"),
                  "grid", _SCAN_LOOP + """
    threadgroup float tv[T * K];
    threadgroup int ti[T * K];
    for (int a = 0; a < int(K); ++a) { tv[t * K + a] = best[a]; ti[t * K + a] = where[a]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = T / 2; stride > 0; stride /= 2) {
        if (t < stride) {
            float mv[K];
            int mi[K];
            uint a0 = t * K, b0 = (t + stride) * K;
            int ia = 0, ib = 0;
            for (int c = 0; c < int(K); ++c) {
                float va = tv[a0 + ia], vb = tv[b0 + ib];
                if (va <= vb) { mv[c] = va; mi[c] = ti[a0 + ia]; ++ia; }
                else { mv[c] = vb; mi[c] = ti[b0 + ib]; ++ib; }
            }
            for (int c = 0; c < int(K); ++c) { tv[a0 + c] = mv[c]; ti[a0 + c] = mi[c]; }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (t == 0) {
        for (int a = 0; a < int(K); ++a) { out_val[i * K + a] = tv[a]; out_idx[i * K + a] = ti[a]; }
    }
""")


def list_scan(x_type: str) -> Kernel:
    """The list-major IVF scan: one threadgroup per list, QB of the queries probing it at a time. 256 threads
    = QB queries x LANES lanes; the queries and XB of the list's vectors at a time are staged through
    threadgroup memory S dimensions at a time, so each list vector is read once per QB queries. Each thread
    keeps the best K for its query; the output is per (query, probe slot, lane)."""
    return Kernel("ivf_list_scan",
                  [("q", "float"), ("x", x_type), ("offsets", "int"), ("pair_ids", "int"), ("list_first", "int"),
                   ("list_count", "int")],
                  [("out_val", "float"), ("out_idx", "int")], ("P",), "group", """
    const uint QB = 16, LANES = 16, XB = 64, S = 16, R = XB / LANES;
    threadgroup float qs[QB][S + 1];
    threadgroup float xs[XB][S + 1];
    uint qa = tid / LANES, lane = tid % LANES;
    int x0 = offsets[l], x1 = offsets[l + 1];
    int first = list_first[l], total = list_count[l];
    for (int c0 = 0; c0 < total; c0 += int(QB)) {
    int start = first + c0, count = min(int(QB), total - c0);
    bool active = int(qa) < count;
    int pair = active ? pair_ids[start + int(qa)] : 0;
    float best[K];
    int where[K];
    for (int a = 0; a < int(K); ++a) { best[a] = INFINITY; where[a] = -1; }
    for (int base = x0; base < x1; base += int(XB)) {
        float acc[R];
        for (uint r = 0; r < R; ++r) acc[r] = 0.0f;
        for (uint d0 = 0; d0 < D; d0 += S) {
            for (uint e = tid; e < QB * S; e += 256) {
                uint r = e / S, c = e % S;
                qs[r][c] = (int(r) < count && d0 + c < D) ? q[uint(pair_ids[start + int(r)] / int(P)) * D + d0 + c] : 0.0f;
            }
            for (uint e = tid; e < XB * S; e += 256) {
                uint r = e / S, c = e % S;
                int j = base + int(r);
                xs[r][c] = (j < x1 && d0 + c < D) ? x[uint(j) * D + d0 + c] : 0.0f;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint c = 0; c < S; ++c) {
                float qv = qs[qa][c];
                for (uint r = 0; r < R; ++r) acc[r] = combine(acc[r], qv, xs[lane + r * LANES][c]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        for (uint r = 0; r < R; ++r) {
            int j = base + int(lane + r * LANES);
            float v = acc[r];
            if (active && j < x1 && v < best[K - 1]) {
                int s = int(K) - 1;
                while (s > 0 && best[s - 1] > v) { best[s] = best[s - 1]; where[s] = where[s - 1]; --s; }
                best[s] = v;
                where[s] = j;
            }
        }
    }
    if (active) {
        for (int a = 0; a < int(K); ++a) {
            out_val[(uint(pair) * LANES + lane) * K + a] = best[a];
            out_idx[(uint(pair) * LANES + lane) * K + a] = where[a];
        }
    }
    }
""")
