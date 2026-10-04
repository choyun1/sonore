"""The Colab starter notebook runs from top to bottom."""

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

import sonore as so

ROOT = Path(__file__).parent.parent
NOTEBOOK = ROOT / "docs" / "notebooks" / "start.ipynb"
# The files the notebook downloads from the repository, copied in from the checkout instead.
DOWNLOADS = [
    "docs/speech/bdl_arctic_a0131.flac",
    "docs/speech/bdl_arctic_a0131_f0.csv",
    "docs/textures/rain.flac",
]

# The source distribution on PyPI carries the code and tests but not the docs.
pytestmark = pytest.mark.skipif(not NOTEBOOK.exists(), reason="docs are not in the sdist")


def synthetic_hrirs(*args, **kwargs):
    """Stands in for so.load_hrirs, which downloads: on a 1 m sphere, each ear's response is one
    impulse, earlier and stronger on the side the source is on."""
    positions, responses = [], []
    for elevation in range(-40, 91, 10):
        for azimuth in range(0, 360, 15):
            lateral = np.sin(np.radians(azimuth)) * np.cos(np.radians(elevation))  # -1 left, 1 right
            pair = np.zeros((2, 64))
            pair[0, 20 - round(12 * lateral)] = 1 - 0.4 * lateral
            pair[1, 20 + round(12 * lateral)] = 1 + 0.4 * lateral
            positions.append(so.hcc_to_rect(100, elevation, azimuth))
            responses.append(pair)
    return so.HRIRSet(np.array(responses), np.array(positions), 44100)


def test_starter_notebook_runs(tmp_path, monkeypatch):
    """Executes every code cell in one namespace, as a kernel would, without
    Jupyter: shell and magic lines (the %pip install) are skipped, the files
    the notebook downloads are copied in from the checkout, the HRIRs are a
    small synthetic set, and figures are closed as they would be shown."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nb = json.loads(NOTEBOOK.read_text())
    assert nb["cells"][1]["source"][0].startswith("%pip install"), "the first code cell installs sonore"
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            assert not cell["outputs"] and cell["execution_count"] is None, "clear the notebook's outputs"

    for path in DOWNLOADS:
        shutil.copy(ROOT / path, tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(so, "load_hrirs", synthetic_hrirs)
    monkeypatch.setattr(plt, "show", lambda: plt.close("all"))  # matplotlib warns past 20 open figures
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
