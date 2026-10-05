"""The landing page's code cell (docs/index.html) runs, and shows what a notebook would.

The page runs Python in the browser through Pyodide, which the tests can't
start; they run the page's own runner (docs/try/runner.py) under CPython
instead, on the example code as it stands in the page.
"""

import html
import importlib.util
import re
from pathlib import Path

import pytest

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
    return module


def example_code():
    match = re.search(r'<textarea id="code"[^>]*>\n?(.*?)</textarea>', PAGE.read_text(), re.DOTALL)
    assert match, "the page has its code cell"
    return html.unescape(match.group(1))


def kinds(items):
    return [item[0] for item in items]


def test_example_plays_a_sound_then_shows_its_overview(runner):
    items = runner.run(example_code())
    assert kinds(items) == ["audio", "image"], items
    label, wav = items[0][1], items[0][2]
    assert label.startswith("Sound(0.500 s, 44100 Hz, 1 ch")
    assert wav[:4] == b"RIFF" and wav[8:12] == b"WAVE"
    assert items[1][1][:8] == b"\x89PNG\r\n\x1a\n"


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
