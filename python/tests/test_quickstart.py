"""The quickstart notebook: up to date with examples/make_quickstart.py, and every cell runs (its
outputs vary with the machine, so only errors fail). Skipped unless nbformat, nbclient and ipykernel
are installed."""
import importlib.util
from pathlib import Path

import pytest

nbformat = pytest.importorskip("nbformat")
nbclient = pytest.importorskip("nbclient")
pytest.importorskip("ipykernel")

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT / "examples" / "quickstart.ipynb"


def test_quickstart_notebook_is_up_to_date():
    """The checked-in notebook has the cells make_quickstart.py builds: rerun it after editing."""
    spec = importlib.util.spec_from_file_location("make_quickstart", ROOT / "examples" / "make_quickstart.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert [c.source for c in nbformat.read(NOTEBOOK, as_version=4).cells] == [c.source for c in module.cells]


# CI turns EncodingWarning into an error; here it comes from jupyter_client starting the kernel
@pytest.mark.filterwarnings("ignore::EncodingWarning")
def test_quickstart_runs():
    nb = nbformat.read(NOTEBOOK, as_version=4)
    nbclient.NotebookClient(nb, timeout=600, kernel_name="python3").execute()


# CI turns EncodingWarning into an error; matplotlib's font cache raises one, not the script
@pytest.mark.filterwarnings("ignore::EncodingWarning:matplotlib")
def test_make_figures_draws_the_readme_figures(tmp_path):
    pytest.importorskip("matplotlib")
    spec = importlib.util.spec_from_file_location("make_figures", ROOT / "examples" / "make_figures.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main(tmp_path)
    assert (tmp_path / "metrics.png").stat().st_size > 0
