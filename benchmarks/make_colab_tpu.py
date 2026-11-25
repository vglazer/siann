"""Writes benchmarks/colab/tpu_probe.ipynb, a notebook to run on Colab with a TPU: it installs TorchTPU (or falls
back to PyTorch/XLA where TorchTPU cannot run), writes out tpu_probe.py (embedded here, so the two cannot drift
apart) and runs it: exact search through PyTorch's built-in operations on the TPU, checked against NumPy.

    python make_colab_tpu.py
"""
import json
from pathlib import Path

HERE = Path(__file__).parent


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip().splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip().splitlines(keepends=True)}


def main() -> None:
    probe = (HERE / "tpu_probe.py").read_text(encoding="utf-8")
    cells = [
        md("""
# siann's TPU probe

Run on a TPU (Colab: Runtime, Change runtime type, TPU). The notebook times exact search through PyTorch's
built-in operations on the TPU (L2 and inner product by a matrix multiply and a top-k, in one stage or two; L1 by
tiles), eagerly and compiled, at the TPU's default matrix-multiply precision and at full float32, and checks
every result against NumPy. Each number prints as it is measured; copy the output of the last cell back.
"""),
        code('''
# which TPU this runtime has, and Python's version (TorchTPU needs 3.11 or later)
import os
import sys

print(sys.version)
for var in ("TPU_ACCELERATOR_TYPE", "TPU_NAME", "COLAB_TPU_ADDR"):
    print(var, os.environ.get(var))
print("accelerator devices:", sorted(f for f in os.listdir("/dev") if f.startswith(("accel", "vfio"))))
'''),
        code('''
# TorchTPU (validated on TPU v5e, v5p, v6e and v7); where it cannot run here, PyTorch/XLA if the runtime has it
import subprocess
import sys


def works(stmt):
    r = subprocess.run([sys.executable, "-c", stmt], check=False, capture_output=True, text=True)
    return r.returncode == 0, (r.stderr.strip().splitlines() or [""])[-1]


TPU = "import torch\\ntry:\\n    import torch_tpu\\nexcept ImportError:\\n    pass\\ntorch.ones(1, device='tpu')"
ok, why = works(TPU)
if not ok:
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "psutil", "torch_tpu",
                        "--extra-index-url", "https://download.pytorch.org/whl/cpu"],
                       check=False, capture_output=True, text=True)
    print("torch_tpu", "installed" if r.returncode == 0 else "not installed: " + (r.stderr.strip().splitlines() or ["?"])[-1])
    ok, why = works(TPU)
print("TorchTPU", "works" if ok else f"does not work here: {why}")
if not ok:
    ok, why = works("import torch_xla; print(torch_xla.device())")
    print("PyTorch/XLA", "works: the probe uses it" if ok else f"does not work either: {why}")
'''),
        code("%%writefile tpu_probe.py\n" + probe),
        code("!python -u tpu_probe.py"),
    ]
    nb = {"cells": cells, "metadata": {"accelerator": "TPU", "kernelspec": {"name": "python3", "display_name": "Python 3"},
                                       "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    out = HERE / "colab" / "tpu_probe.ipynb"
    out.write_text(json.dumps(nb, indent=1) + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
