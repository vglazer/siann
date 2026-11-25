"""Writes benchmarks/colab/triton_probe.ipynb, a notebook to run on Colab or Kaggle with an NVIDIA GPU: it installs
FAISS's and cuVS's GPU packages, writes out triton_probe.py (embedded here, so the two cannot drift apart), checks
the Triton kernel against exact search on the GPU, benchmarks it against torch.cdist, FAISS and cuVS, and saves a
cuVS IVF-Flat index of seeded Gaussian vectors (which siann can regenerate) for download, to check that siann
reads cuVS's file format.

    python make_colab_probe.py
"""
import json
from pathlib import Path

from colab_cells import notebook, numbered

HERE = Path(__file__).parent


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip().splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip().splitlines(keepends=True)}


def main() -> None:
    probe = (HERE / "triton_probe.py").read_text(encoding="utf-8")
    cells = [
        md("""
# siann's Triton probe

Run with a GPU (Colab: Runtime, Change runtime type, T4 GPU; Kaggle: Accelerator, GPU). The notebook checks
siann's exact L1 kernel, written in Triton, against exact search on the GPU, then times it against
`torch.cdist`, FAISS's GPU brute force and cuVS's brute force, with the memory each takes. Last, it builds a cuVS IVF-Flat index of seeded
Gaussian vectors and saves it, to check that siann reads cuVS's files: download `ivf_flat_gaussian.cuvs` from
the file browser at the end. Copy the outputs of the check and benchmark cells back.
"""),
        code("!nvidia-smi --query-gpu=name,memory.total --format=csv"),
        code('''
# FAISS's and cuVS's GPU packages for the CUDA version this machine's PyTorch was built with, cuVS only if it is
# not already installed (Colab comes with it): mixing CUDA 12 and 13 packages breaks NVIDIA's other libraries
import subprocess
import sys

import torch

cuda = torch.version.cuda.split(".")[0]
print("PyTorch", torch.__version__, "for CUDA", torch.version.cuda)


def cuvs_works():
    r = subprocess.run([sys.executable, "-c", "from cuvs.neighbors import brute_force"], check=False,
                       capture_output=True, text=True)
    return r.returncode == 0


# cuVS's Python package can be present without its library (libcuvs_c.so): reinstall both, for this CUDA
wanted = ["nvidia-ml-py", "psutil", f"faiss-gpu-cu{cuda}"] + ([] if cuvs_works() else [f"libcuvs-cu{cuda}", f"cuvs-cu{cuda}"])


def install(pkg):
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg, "--extra-index-url=https://pypi.nvidia.com"],
                       check=False, capture_output=True, text=True)
    print(pkg, "installed" if r.returncode == 0 else "not installed: " + (r.stderr.strip().splitlines() or ["?"])[-1])
    return r.returncode == 0


for pkg in wanted:
    # FAISS publishes no CUDA 13 build; its CUDA 12 one ships its own CUDA runtime, which sits beside CUDA 13
    if not install(pkg) and pkg.startswith("faiss-gpu-cu") and pkg != "faiss-gpu-cu12":
        install("faiss-gpu-cu12")
print("cuVS", "works" if cuvs_works() else "still missing its library: the benchmark skips it")
'''),
        code("%%writefile triton_probe.py\n" + probe),
        code("!python -u triton_probe.py --check"),
        md("""
The benchmark, with the 2D-tile kernel at the configuration the T4's tuning found fastest (32 x 128, sort, 4
warps): about 5-10 minutes (compiling included). Every row shows its number out of the total and the minutes so far.
"""),
        code("!python -u triton_probe.py --bench --config 32,128,sort,4"),
        md("""
Optional, long (30 minutes or more on a T4): the same benchmark after tuning all 24 configurations.
"""),
        code("!python -u triton_probe.py --bench"),
        code('''
import os

import numpy as np

x = np.random.default_rng(0).standard_normal((100_000, 128), dtype=np.float32)   # siann regenerates the same
try:
    import cupy as cp
    import cuvs
    from cuvs.neighbors import ivf_flat
    index = ivf_flat.build(ivf_flat.IndexParams(n_lists=316, metric="sqeuclidean"), cp.asarray(x))
    ivf_flat.save("ivf_flat_gaussian.cuvs", index, include_dataset=True)
    print("cuVS", getattr(cuvs, "__version__", "?"), "saved", os.path.getsize("ivf_flat_gaussian.cuvs"), "bytes")
except Exception as e:  # noqa: BLE001 (report whatever went wrong, for the user to paste back)
    print(type(e).__name__, e)
'''),
    ]
    nb = notebook(numbered(cells, [("Which GPU", 0), ("Install FAISS and cuVS", 5), ("Write triton_probe.py", 0),
                           ("Check the kernels against exact search", 15, "optional"),
                           ("Benchmark at the tuned configuration", 10),
                           ("Benchmark after tuning all 24 configurations", 45, "optional"),
                           ("Save a cuVS IVF-Flat file", 2, "optional")]), "GPU")
    out = HERE / "colab" / "triton_probe.ipynb"
    out.write_text(json.dumps(nb, indent=1) + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
