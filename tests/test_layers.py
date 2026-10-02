"""The package is layered, and each layer imports only from the layers below it.

Bottom to top: core, signals, analysis, stimuli, texture. ``plotting`` sits
above them all: the objects' ``.plot()`` methods import it inside the method,
and nothing imports it at module level. Inside a layer any import is fine.

A few methods import upward inside the method body, so that the call reads
naturally in a notebook: ``Sound.envelope()`` returns an ``Envelope`` from the
analysis layer. Those are listed in ``UPWARD_INSIDE_FUNCTIONS``, so a new one
has to be added there on purpose. docs/design/layout.md has the diagram, drawn
from the source by tools/draw_layout.py.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parent.parent / "src" / "sonore"

LAYERS = ["core", "signals", "analysis", "stimuli", "texture", "plotting"]

# (importer, imported) pairs allowed to point upward from inside a function.
UPWARD_INSIDE_FUNCTIONS = {
    ("sound", "envelopes"),  # Sound.envelope()
}


def modules() -> dict[str, Path]:
    """Every module below the top-level ``__init__``, by dotted name inside sonore."""
    out = {}
    for path in PACKAGE.rglob("*.py"):
        parts = path.relative_to(PACKAGE).with_suffix("").parts
        if parts[-1] == "__init__":
            parts = parts[:-1]
        if parts:
            out[".".join(parts)] = path
    return out


def layer(name: str) -> str:
    """``analysis.frames`` is in the analysis layer; ``plotting`` is its own."""
    return name.split(".")[0]


def internal_imports(path: Path, known: dict[str, Path]) -> list[tuple[str, bool]]:
    """(imported module, whether the import is at module level) for each sonore import."""
    tree = ast.parse(path.read_text())
    top = {id(node) for node in tree.body}
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.split(".")[0] == "sonore":
            base = node.module.removeprefix("sonore").lstrip(".")
            for alias in node.names:
                sub = f"{base}.{alias.name}" if base else alias.name
                target = sub if sub in known else base
                if target:
                    found.append((target, id(node) in top))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("sonore."):
                    found.append((alias.name.removeprefix("sonore."), id(node) in top))
    return found


def test_every_module_has_a_layer():
    for name in modules():
        assert layer(name) in LAYERS, name


def test_imports_point_down():
    known = modules()
    upward_used = set()
    wrong = []
    for name, path in known.items():
        for target, at_top in internal_imports(path, known):
            if LAYERS.index(layer(target)) <= LAYERS.index(layer(name)):
                continue
            if at_top:
                wrong.append(f"{name} imports {target} at module level")
                continue
            if layer(target) == "plotting":
                continue
            pair = (name.split(".")[-1], target.split(".")[-1])
            if pair in UPWARD_INSIDE_FUNCTIONS:
                upward_used.add(pair)
            else:
                wrong.append(f"{name} imports {target} inside a function")
    assert not wrong, "imports pointing up a layer:\n" + "\n".join(wrong)
    assert upward_used == UPWARD_INSIDE_FUNCTIONS, "stale entries in UPWARD_INSIDE_FUNCTIONS"


def test_layout_diagram_is_current():
    """docs/design/layout.svg is what tools/draw_layout.py draws from the source today."""
    import importlib.util

    root = PACKAGE.parent.parent
    diagram = root / "docs" / "design" / "layout.svg"
    if not diagram.exists():
        pytest.skip("docs are not in the sdist")
    spec = importlib.util.spec_from_file_location("draw_layout", root / "tools" / "draw_layout.py")
    draw_layout = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(draw_layout)
    assert diagram.read_text() == draw_layout.render(), "run python tools/draw_layout.py"
