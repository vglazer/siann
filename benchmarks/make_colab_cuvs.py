"""Writes two notebooks to run on Colab or Kaggle with an NVIDIA GPU (a T4 or newer), each installing cuVS and
writing out cuvs_probe.py (embedded here, so the notebooks and the script cannot drift apart):
benchmarks/colab/cuvs_capabilities.ipynb checks which metrics and data types cuVS's brute force, IVF-Flat and
CAGRA accept; benchmarks/colab/cuvs_cagra.ipynb builds a CAGRA graph on SIFT1M and saves it for siann to read.

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


def shared():
    probe = (HERE / "cuvs_probe.py").read_text(encoding="utf-8")
    return [
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
    ]


def write(name, intro, cells, plan):
    nb = notebook(numbered([md(intro)] + shared() + cells, [("Which GPU", 0), ("Install cuVS where missing", 1),
                                                             ("Write cuvs_probe.py", 0)] + plan), "GPU")
    out = HERE / "colab" / name
    out.write_text(json.dumps(nb, indent=1) + "\n", encoding="utf-8")
    print(out)


SETUP = """Run on Colab with a T4 GPU (Runtime, Change runtime type, T4 GPU): Colab comes with cuVS, so nothing heavy
is installed (on Kaggle, installing cuVS alone takes several minutes). Every step runs; none needs skipping, each
says when it starts, when it ends and what comes next, and the probe stops itself at its time budget."""


def main() -> None:
    write("cuvs_capabilities.ipynb", f"""
# What cuVS accepts

{SETUP}

Which metrics cuVS's brute force accepts (L1 and L-infinity in particular), and which data types (float32,
float16, int8, uint8) and metrics its IVF-Flat and CAGRA accept, each checked against NumPy on small data. About
3-5 minutes in all. Copy the output of the last step back.
""", [
        code("!python -u cuvs_probe.py --capabilities"),
    ], [("Capabilities", 2)])
    write("cuvs_cagra.ipynb", f"""
# A CAGRA graph built by cuVS, for siann

{SETUP}

cuVS builds a CAGRA graph on SIFT1M (32 links per vector, pruned from 64, as siann's graphs), measures its own
search's recall, and saves the graph in cuVS's format and as a NumPy file. About 5 minutes in all, most of it
downloading SIFT1M (about 500 MB: Colab's network usually takes under a minute). Copy the output of the last step
back, and download
`cagra_sift.cuvs` and `cagra_sift_links.npy` (Colab: the file browser on the left; Kaggle: the Output panel).
""", [
        code('''
import os

out = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."     # Kaggle keeps only /kaggle/working
!python -u cuvs_probe.py --cagra --out {out}
'''),
    ], [("CAGRA on SIFT1M", 4)])


if __name__ == "__main__":
    main()
