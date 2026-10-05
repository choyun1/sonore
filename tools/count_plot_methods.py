"""Counts where sonore draws, for docs/design/layout/plotting.md.

For every public class (``so.__all__`` and ``so.texture.__all__``, less the bases ``View`` and
``Frame``) it says whether the class has a ``plot`` method and, if so, whether that method
calls into ``sonore.plotting`` or draws with matplotlib itself. It lists the public ``plot_*`` functions of ``plotting.py`` that no
class method calls, and counts the calls written as ``<name>.plot(`` in the repository whose
receiver is not a matplotlib axis, which are the calls a change to ``.plot`` would touch.
Nothing is changed.

    python tools/count_plot_methods.py

The call count is a text match, so it is an estimate: a receiver named like an axis
(``ax``, ``axes[...]``, ``plt``, ``axis``) is left out, and any other ``.plot(`` is counted.
"""

import ast
import inspect
import re
from pathlib import Path

import sonore as so
import sonore.texture
from sonore.frames.frame import Frame
from sonore.views.view import View

ROOT = Path(__file__).resolve().parent.parent
AXIS_RECEIVER = re.compile(r"\b(ax\w*|axes(\[[^\]]*\])*|plt|axis|a)\.plot\(")
ANY_PLOT_CALL = re.compile(r"[\w\])]\.plot\(")


def kind_of(cls):
    if issubclass(cls, View):
        return "view"
    if issubclass(cls, Frame):
        return "frame"
    return "other"


def how_it_draws(cls):
    method = next((klass.__dict__["plot"] for klass in cls.__mro__ if "plot" in klass.__dict__), None)
    if method is None:
        return "no plot"
    source = inspect.getsource(method)
    if "sonore.plotting" in source or "_plot_" in source:
        return "calls plotting.py"
    return "draws itself"


def public_classes():
    namespaces = [(so, so.__all__), (so.texture, so.texture.__all__)]
    return {
        name: getattr(module, name)
        for module, names in namespaces
        for name in names
        if isinstance(getattr(module, name), type)
        and not issubclass(getattr(module, name), BaseException)
        and getattr(module, name) not in (View, Frame)
    }


def main():
    rows = sorted(public_classes().items(), key=lambda item: (kind_of(item[1]), item[0]))
    print("Public classes")
    for name, cls in rows:
        print(f"  {kind_of(cls):6s} {name:24s} {how_it_draws(cls)}")
    for kind in ("view", "frame", "other"):
        in_kind = [cls for _, cls in rows if kind_of(cls) == kind]
        with_plot = [cls for cls in in_kind if how_it_draws(cls) != "no plot"]
        print(f"  {kind}: {len(with_plot)} of {len(in_kind)} have plot")

    plotting_source = (ROOT / "src/sonore/plotting.py").read_text()
    functions = [
        node.name
        for node in ast.parse(plotting_source).body
        if isinstance(node, ast.FunctionDef) and not node.name.startswith("_")
    ]
    package_text = "\n".join(
        path.read_text() for path in (ROOT / "src/sonore").rglob("*.py") if path.name != "plotting.py"
    )
    print("\nplotting.py functions no class method calls")
    for name in functions:
        if not re.search(rf"\b{name}\b", package_text):
            print(f"  {name}")

    print("\n.plot( calls on sonore objects (estimate, see the docstring)")
    total = 0
    for folder in ("docs", "tests", "src", "tools"):
        count = 0
        for path in (ROOT / folder).rglob("*"):
            if path.suffix not in (".py", ".ipynb", ".md", ".rst") or "__pycache__" in path.parts:
                continue
            if path.name == "count_plot_methods.py":
                continue
            for line in path.read_text(errors="ignore").splitlines():
                count += len(ANY_PLOT_CALL.findall(line)) - len(AXIS_RECEIVER.findall(line))
        print(f"  {folder:6s} {count}")
        total += count
    print(f"  total  {total}")
    readme = (ROOT / "README.md").read_text().count(".plot(")
    print(f"README mentions of .plot(): {readme}")


if __name__ == "__main__":
    main()
