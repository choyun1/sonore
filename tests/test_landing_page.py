"""The landing page's code cell (docs/index.html) runs, and shows what a notebook would.

The page runs Python in the browser through Pyodide, which the tests can't
start; they run the page's own runner (docs/try/runner.py) under CPython
instead, after the Colab tutorial's setup cell, as the page does, on the
example code as it stands in the page and on cells copied from the tutorial.
"""

import html
import importlib.util
import json
import re
import shutil
from pathlib import Path

import pytest
from test_notebook import DOWNLOADS, NOTEBOOK, synthetic_hrirs

import sonore as so

ROOT = Path(__file__).parent.parent
PAGE = ROOT / "docs" / "index.html"
RUNNER = ROOT / "docs" / "try" / "runner.py"
WORKER = ROOT / "docs" / "try" / "worker.js"

# The source distribution on PyPI carries the code and tests but not the docs.
pytestmark = pytest.mark.skipif(not PAGE.exists(), reason="docs are not in the sdist")


@pytest.fixture(scope="module")
def runner():
    spec = importlib.util.spec_from_file_location("landing_runner", RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.prepare(NOTEBOOK.read_text())
    return module


def example_code():
    match = re.search(r'<textarea id="code"[^>]*>\n?(.*?)</textarea>', PAGE.read_text(), re.DOTALL)
    assert match, "the page has its code cell"
    return html.unescape(match.group(1))


def kinds(items):
    return [item[0] for item in items]


def tutorial_cells():
    cells = json.loads(NOTEBOOK.read_text())["cells"]
    return ["".join(cell["source"]) for cell in cells if cell["cell_type"] == "code"]


def test_prepare_runs_the_tutorial_setup_cell(runner):
    assert callable(runner.namespace["show"]) and callable(runner.namespace["download"])
    assert runner.namespace["fs"] == 44100


def test_example_plays_a_sound_then_shows_its_overview(runner):
    items = runner.run(example_code())
    assert kinds(items) == ["audio", "image"], items
    label, wav = items[0][1], items[0][2]
    assert label.startswith("Sound(0.500 s, 44100 Hz, 1 ch")
    assert wav[:4] == b"RIFF" and wav[8:12] == b"WAVE"
    assert items[1][1][:8] == b"\x89PNG\r\n\x1a\n"


def test_tutorial_cells_paste_in_as_they_are(runner, tmp_path, monkeypatch):
    """The cells that use each part of the setup: the %pip magic, show, download,
    and the scene player's HTML. Each player comes before its sound's overview."""
    for path in DOWNLOADS:
        shutil.copy(ROOT / path, tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(so, "load_hrirs", synthetic_hrirs)
    cells = tutorial_cells()
    assert runner.run(cells[0]) == []  # %pip install: nothing to do here
    two_sounds = next(cell for cell in cells if cell.startswith("tone = "))
    assert kinds(runner.run(two_sounds)) == ["audio", "image", "audio", "image"]
    sentence = next(cell for cell in cells if "download(" in cell and "sentence = " in cell)
    assert kinds(runner.run(sentence)) == ["audio", "image"] * 3
    for cell in cells:  # what the moving talker needs: pink noise, the HRIRs and the scene player
        if cell.startswith(("pink = ", "hrirs = ", "# @title")):
            assert "error" not in kinds(runner.run(cell))
    moving = runner.run(next(cell for cell in cells if "show_moving(" in cell and "walk_around = " in cell))
    assert kinds(moving)[:3] == ["html", "image", "html"], kinds(moving)


def test_cells_share_one_namespace(runner):
    runner.run("level = 3")
    assert runner.run("level + 1") == [("text", "4")]


def test_print_then_value(runner):
    assert runner.run("print('hello')\n1 + 1") == [("text", "hello\n"), ("text", "2")]


def test_errors_show_only_the_cell(runner):
    ((kind, trace),) = runner.run("import sonore as so\nx = 1\nso.pure_tone(0.1, 8000, f=440)")
    assert kind == "error"
    assert trace.startswith('  File "<cell>", line 3') and "so.pure_tone(0.1, 8000, f=440)" in trace
    assert "runner.py" not in trace and trace.rstrip().splitlines()[-1].startswith("TypeError")
    ((kind, trace),) = runner.run("x = (1,")
    assert kind == "error" and "SyntaxError" in trace


def test_no_player_above_full_scale(runner):
    ((kind, text),) = runner.run("import sonore as so\nso.pure_tone(0.1, 8000, 440)")  # RMS 1, peak 1.41
    assert kind == "text" and "No player: the peak is 1.41" in text


def test_a_list_of_sounds_gets_a_player_each(runner):
    items = runner.run("import sonore as so\n[so.pure_tone(0.1, 8000, f) - 20 * so.dB for f in (440, 880)]")
    assert kinds(items) == ["audio", "audio"]


def test_worker_loads_every_dependency_of_sonore():
    """The worker loads Pyodide's own builds of sonore's dependencies up front, in
    one step it can report; a dependency added to pyproject.toml belongs there too."""
    pyproject = (ROOT / "pyproject.toml").read_text()
    block = re.search(r"^dependencies = \[(.*?)^\]", pyproject, re.DOTALL | re.MULTILINE).group(1)
    names = {name.lower() for name in re.findall(r'^\s*"([A-Za-z0-9_.-]+)', block, re.MULTILINE)}
    assert names, "pyproject.toml lists the dependencies"
    packages = re.search(r"const PACKAGES = \[(.*?)\]", WORKER.read_text()).group(1)
    loaded = set(re.findall(r'"([a-z0-9_-]+)"', packages))
    assert names <= loaded, names - loaded
