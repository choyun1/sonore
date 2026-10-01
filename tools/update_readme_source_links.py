"""Keep the source links in the README and the gallery pointing at the right lines.

Each module tag in the README's References section links to its file on
GitHub, and a tag that names a function or class, like
`representations.reassigned_spectrogram`, links to the line of its
definition. A tag names the file's module, or the package that re-exports
it: `texture.TextureStats` links into texture/stats.py. The module table in
What's in it uses the full dotted path, like `analysis.frames`, which links
to the file without a line. The References lists on the gallery pages use the
same tags, in the page scripts (docs/gallery/*.py) and in the built pages
(docs/gallery/*.html), so the script fixes those too without rebuilding them.
Line anchors move whenever code above them changes, so tests/test_docs.py
checks them and this script rewrites them:

    python tools/update_readme_source_links.py
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BLOB = "https://github.com/choyun1/sonore/blob/main/"
LINK = re.compile(r"\[`([\w.]+)`\]\(" + re.escape(BLOB) + r"([\w/]+\.py)(?:#L(\d+))?\)")
HTML_LINK = re.compile(
    r'<a href="' + re.escape(BLOB) + r'([\w/]+\.py)(?:#L(\d+))?"><code>([\w.]+)</code></a>'
)
GALLERY = ROOT / "docs" / "gallery"


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
    path = ROOT / rel_path
    if not path.exists():
        raise LookupError(f"{rel_path} does not exist")
    if text == rel_path.removeprefix("src/sonore/").removesuffix(".py").replace("/", "."):
        return f"[`{text}`]({BLOB}{rel_path})"
    module, *symbol = text.split(".")
    if module not in (path.stem, path.parent.name):
        raise LookupError(f"tag `{text}` links to {rel_path}, a different module")
    anchor = f"#L{definition_line(path, symbol)}" if symbol else ""
    return f"[`{text}`]({BLOB}{rel_path}{anchor})"


def expected_html_link(text: str, rel_path: str) -> str:
    """The built-page form of :func:`expected_link`."""
    m = LINK.fullmatch(expected_link(text, rel_path))
    anchor = f"#L{m[3]}" if m[3] else ""
    return f'<a href="{BLOB}{m[2]}{anchor}"><code>{text}</code></a>'


def fixed(text: str) -> str:
    text = LINK.sub(lambda m: expected_link(m[1], m[2]), text)
    return HTML_LINK.sub(lambda m: expected_html_link(m[3], m[1]), text)


def linked_files() -> list[Path]:
    """The README and every gallery page script and built page."""
    return [ROOT / "README.md", *sorted(GALLERY.glob("*.py")), *sorted(GALLERY.glob("*.html"))]


if __name__ == "__main__":
    changed = []
    for path in linked_files():
        old = path.read_text()
        new = fixed(old)
        if new != old:
            path.write_text(new)
            changed.append(str(path.relative_to(ROOT)))
    print("Updated source links in " + ", ".join(changed) if changed else "Source links are up to date.")
