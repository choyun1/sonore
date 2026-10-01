"""The Colab starter notebook runs from top to bottom."""

import json
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
NOTEBOOK = ROOT / "docs" / "notebooks" / "start.ipynb"

# The source distribution on PyPI carries the code and tests but not the docs.
pytestmark = pytest.mark.skipif(not NOTEBOOK.exists(), reason="docs are not in the sdist")


def test_starter_notebook_runs(tmp_path, monkeypatch):
    """Executes every code cell in one namespace, as a kernel would, without
    Jupyter: shell and magic lines (the %pip install) are skipped, and the
    sentence the notebook downloads is copied in from docs/speech instead."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nb = json.loads(NOTEBOOK.read_text())
    assert nb["cells"][1]["source"][0].startswith("%pip install"), "the first code cell installs sonore"
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            assert not cell["outputs"] and cell["execution_count"] is None, "clear the notebook's outputs"

    shutil.copy(ROOT / "docs" / "speech" / "bdl_arctic_a0131.flac", tmp_path)
    monkeypatch.chdir(tmp_path)
    namespace = {"display": lambda *objs: None}
    try:
        for i, cell in enumerate(nb["cells"]):
            if cell["cell_type"] != "code":
                continue
            lines = "".join(cell["source"]).splitlines()
            code = "\n".join(line for line in lines if not line.lstrip().startswith(("%", "!")))
            exec(compile(code, f"{NOTEBOOK.name}, cell {i}", "exec"), namespace)
    finally:
        plt.close("all")
