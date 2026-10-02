"""Sphinx configuration for the sonore API reference.

The pages workflow builds it on every push to main and serves it next to the
gallery, at https://choyun1.github.io/sonore/api/. To build it locally, from
the repository root (the output is not committed)::

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
# Colours and fonts from the listening gallery (docs/gallery/build.py), so the
# two read as one site.
_SANS = '"Atkinson Hyperlegible", system-ui, -apple-system, "Segoe UI", sans-serif'
_MONO = 'ui-monospace, "SF Mono", Menlo, Consolas, "Liberation Mono", monospace'
html_theme_options = {
    "light_css_variables": {
        "color-brand-primary": "#235B7C",
        "color-brand-content": "#235B7C",
        "color-foreground-primary": "#16202A",
        "color-foreground-secondary": "#566573",
        "color-background-secondary": "#ECEFF1",
        "color-api-name": "#235B7C",
        "color-api-pre-name": "#566573",
        "font-stack": _SANS,
        "font-stack--monospace": _MONO,
    },
    "dark_css_variables": {
        "color-brand-primary": "#7FB6D6",
        "color-brand-content": "#7FB6D6",
        "color-foreground-primary": "#E4E9ED",
        "color-foreground-secondary": "#9AA8B4",
        "color-background-primary": "#121A21",
        "color-background-secondary": "#0C1318",
        "color-api-name": "#7FB6D6",
        "color-api-pre-name": "#9AA8B4",
    },
}
html_static_path = ["_static"]
html_css_files = [
    "https://fonts.googleapis.com/css2?family=Atkinson+Hyperlegible:wght@400;700"
    "&family=Spectral:wght@400;500;600&display=swap",
    "sonore.css",
]
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
