"""Imports between modules never form a cycle, and core stays at the bottom.

The folders follow meaning (docs/design/layout/sound-first.md), not import order:
``sources.ripples`` imports frames and views, and views import ``core``.
What keeps the package from tangling is enforced module by module instead:

- the module-level imports between sonore's modules form no cycle, so every
  module sits above everything it imports;
- ``core`` imports nothing outside ``core`` at module level;
- an import inside a function (or a ``TYPE_CHECKING`` block) that would close
  a cycle is listed by name in ``BACK_IMPORTS``, so a new one has to be added
  there on purpose. They exist so that calls chain in a notebook:
  ``Sound.envelope()`` returns an ``Envelope`` from views.

``plotting`` is exempt: the objects' ``.plot()`` methods import it inside the
method. docs/design/layout.md has the diagram, drawn from the source by
tools/draw_layout.py.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parent.parent / "src" / "sonore"

FOLDERS = ["core", "sources", "frames", "views", "spatial", "texture", "plotting"]

# (importer, imported) imports inside a function that point back up the
# module-level import graph.
BACK_IMPORTS = {
    ("core.sound", "views.envelopes"),  # Sound.envelope()
    ("core.units", "core.sound"),  # refusing snd * dB with a hint
    ("frames.filterbank", "views.envelopes"),  # Subbands.envelopes()
    ("frames.gabor", "frames.mask"),  # STFT * mask
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


def folder(name: str) -> str:
    """``frames.gabor`` is in frames; ``plotting`` is its own."""
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


def module_graph() -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """Module-level and inside-function imports between sonore's modules."""
    known = modules()
    at_module_level = {name: set() for name in known}
    inside_functions = {name: set() for name in known}
    for name, path in known.items():
        for target, at_top in internal_imports(path, known):
            if target != name:
                (at_module_level if at_top else inside_functions)[name].add(target)
    return at_module_level, inside_functions


def reachable(start: str, graph: dict[str, set[str]]) -> set[str]:
    """Every module ``start`` imports, directly or through others."""
    seen, stack = set(), [start]
    while stack:
        for target in graph.get(stack.pop(), ()):
            if target not in seen:
                seen.add(target)
                stack.append(target)
    return seen


def test_every_module_has_a_folder():
    for name in modules():
        assert folder(name) in FOLDERS, name


def test_no_cycles_at_module_level():
    at_module_level, _ = module_graph()
    cyclic = [name for name in at_module_level if name in reachable(name, at_module_level)]
    assert not cyclic, "module-level import cycles through:\n" + "\n".join(sorted(cyclic))


def test_core_imports_only_core_at_module_level():
    at_module_level, _ = module_graph()
    wrong = [
        f"{name} imports {target}"
        for name, targets in at_module_level.items()
        if folder(name) == "core"
        for target in targets
        if folder(target) != "core"
    ]
    assert not wrong, "core imports from above it:\n" + "\n".join(wrong)


def test_back_imports_are_listed():
    at_module_level, inside_functions = module_graph()
    back = {
        (name, target)
        for name, targets in inside_functions.items()
        for target in targets
        if folder(target) != "plotting" and name in reachable(target, at_module_level)
    }
    assert back <= BACK_IMPORTS, "unlisted imports closing a cycle:\n" + "\n".join(
        f"{name} imports {target}" for name, target in sorted(back - BACK_IMPORTS)
    )
    assert back == BACK_IMPORTS, "stale entries in BACK_IMPORTS"


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
