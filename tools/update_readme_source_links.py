"""Keep the README's source links pointing at the right lines.

Each module tag in the README's References section links to its file on
GitHub, and a tag that names a function or class, like
`representations.reassigned_spectrogram`, links to the line of its
definition. A tag names the file's module, or the package that re-exports
it: `texture.TextureStats` links into texture/stats.py. Line anchors move whenever code above them changes, so
tests/test_docs.py checks them and this script rewrites them:

    python tools/update_readme_source_links.py
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BLOB = "https://github.com/choyun1/sonore/blob/main/"
LINK = re.compile(r"\[`([\w.]+)`\]\(" + re.escape(BLOB) + r"([\w/]+\.py)(?:#L(\d+))?\)")


def definition_line(path: Path, dotted: list[str]) -> int:
    """Line of the ``def`` or ``class`` named by ``dotted`` inside ``path``."""
    body = ast.parse(path.read_text()).body
    node = None
    for name in dotted:
        node = next(
            (n for n in body if isinstance(n, ast.FunctionDef | ast.ClassDef) and n.name == name),
            None,
        )
        if node is None:
            raise LookupError(f"{'.'.join(dotted)} is not defined in {path.relative_to(ROOT)}")
        body = node.body
    return node.lineno


def expected_link(text: str, rel_path: str) -> str:
    """The correct link for tag ``text`` pointing into ``rel_path``."""
    module, *symbol = text.split(".")
    path = ROOT / rel_path
    if not path.exists():
        raise LookupError(f"{rel_path} does not exist")
    if module not in (path.stem, path.parent.name):
        raise LookupError(f"tag `{text}` links to {rel_path}, a different module")
    anchor = f"#L{definition_line(path, symbol)}" if symbol else ""
    return f"[`{text}`]({BLOB}{rel_path}{anchor})"


def fixed(readme: str) -> str:
    return LINK.sub(lambda m: expected_link(m[1], m[2]), readme)


if __name__ == "__main__":
    readme_path = ROOT / "README.md"
    old = readme_path.read_text()
    new = fixed(old)
    if new == old:
        print("README source links are up to date.")
    else:
        readme_path.write_text(new)
        print("Updated README source links.")
