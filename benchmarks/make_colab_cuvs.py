"""Writes benchmarks/colab/cuvs_probe.ipynb, a notebook to run on Colab or Kaggle with an NVIDIA GPU (a T4 or
newer): it installs cuVS, writes out cuvs_probe.py (embedded here, so the two cannot drift apart), checks which
metrics and data types cuVS's brute force, IVF-Flat and CAGRA accept, and builds a CAGRA graph on SIFT1M,
saving it for siann to read.

    python make_colab_cuvs.py
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
    probe = (HERE / "cuvs_probe.py").read_text(encoding="utf-8")
    cells = [
        md("""
# What cuVS offers siann

Run with an NVIDIA GPU, a T4 or newer (Colab: Runtime, Change runtime type, T4 GPU; Kaggle: Accelerator, GPU T4,
not the P100; Internet on). Every cell runs; nothing needs skipping, and every line prints its number out of the
total and the minutes so far.

1. **Capabilities** (about 2-5 minutes): which metrics cuVS's brute force accepts (L1 and L-infinity in
   particular), and which data types (float32, float16, int8, uint8) and metrics its IVF-Flat and CAGRA accept,
   each checked against NumPy.
2. **CAGRA on SIFT1M** (about 10-20 minutes, most of it downloading SIFT1M, about 500 MB): cuVS builds a graph
   (32 links per vector), measures its own search's recall, and saves the graph in cuVS's format and as a NumPy
   file. Download `cagra_sift.cuvs` and `cagra_sift_links.npy` at the end (Colab: the file browser on the left;
   Kaggle: the Output panel).

In all, about 15-30 minutes with the installs. Copy the outputs of the two probe cells back.
"""),
        code("!nvidia-smi --query-gpu=name,memory.total --format=csv"),
        code('''
# cuVS (and CuPy) for the CUDA version this machine's PyTorch was built with, only where missing: Colab comes with
# cuVS; Kaggle does not. Mixing CUDA 12 and 13 packages breaks NVIDIA's other libraries.
import subprocess
import sys

import torch

cuda = torch.version.cuda.split(".")[0]
print("PyTorch", torch.__version__, "for CUDA", torch.version.cuda)


def works(stmt):
    r = subprocess.run([sys.executable, "-c", stmt], check=False, capture_output=True, text=True)
    return r.returncode == 0


def install(pkg):
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg, "--extra-index-url=https://pypi.nvidia.com"],
                       check=False, capture_output=True, text=True)
    print(pkg, "installed" if r.returncode == 0 else "not installed: " + (r.stderr.strip().splitlines() or ["?"])[-1])


if not works("import cupy"):
    install(f"cupy-cuda{cuda}x")
if not works("import h5py"):
    install("h5py")
# cuVS's Python package can be present without its library (libcuvs_c.so): reinstall both, for this CUDA
if not works("from cuvs.neighbors import brute_force, cagra, ivf_flat"):
    install(f"libcuvs-cu{cuda}")
    install(f"cuvs-cu{cuda}")
print("cuVS", "works" if works("from cuvs.neighbors import brute_force, cagra, ivf_flat") else "still does not work")
'''),
        code("%%writefile cuvs_probe.py\n" + probe),
        md("## 1. Capabilities (about 2-5 minutes)"),
        code("!python -u cuvs_probe.py --capabilities"),
        md("## 2. CAGRA on SIFT1M (about 10-20 minutes)"),
        code('''
import os

out = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."     # Kaggle keeps only /kaggle/working
!python -u cuvs_probe.py --cagra --out {out}
'''),
    ]
    nb = notebook(numbered(cells, [("Which GPU", 0), ("Install cuVS", 5), ("Write cuvs_probe.py", 0),
                           ("Capabilities", 5), ("CAGRA on SIFT1M", 20)]), "GPU")
    out = HERE / "colab" / "cuvs_probe.ipynb"
    out.write_text(json.dumps(nb, indent=1) + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
