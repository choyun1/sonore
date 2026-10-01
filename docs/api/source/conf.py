"""Sphinx configuration for the sonore API reference.

Build from the repository root (the output is committed and served by GitHub
Pages next to the gallery, at https://choyun1.github.io/sonore/api/)::

    python -m pip install -e ".[docs]"
    python -m sphinx -b html -E -d /tmp/sonore-doctrees docs/api/source docs/api
"""

import inspect
from pathlib import Path

import sonore

project = "sonore"
author = "Adrian Y. Cho"
release = version = sonore.__version__

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.linkcode",
]

# The docstrings are numpy style; parameter types live in the signatures.
napoleon_google_docstring = False
napoleon_numpy_docstring = True
autodoc_typehints = "signature"
autodoc_member_order = "bysource"
# Keep numpy's ArrayLike as one word instead of its long expansion.
autodoc_type_aliases = {"ArrayLike": "ArrayLike"}
add_module_names = False
autodoc_default_options = {"members": True, "undoc-members": True, "show-inheritance": True}

html_theme = "furo"
html_title = f"sonore {version}"
html_copy_source = False
html_show_sphinx = False


SRC = Path(sonore.__file__).parent


def linkcode_resolve(domain, info):
    """Link each name's [source] to its lines on GitHub, as the README tags do."""
    if domain != "py" or not info["module"]:
        return None
    obj = __import__(info["module"], fromlist=["_"])
    for part in info["fullname"].split("."):
        obj = getattr(obj, part, None)
    obj = inspect.unwrap(getattr(obj, "fget", obj) or obj)
    try:
        path = Path(inspect.getsourcefile(obj)).relative_to(SRC)
        lines, start = inspect.getsourcelines(obj)
    except (TypeError, OSError, ValueError):
        return None
    end = start + len(lines) - 1
    return f"https://github.com/choyun1/sonore/blob/main/src/sonore/{path.as_posix()}#L{start}-L{end}"
