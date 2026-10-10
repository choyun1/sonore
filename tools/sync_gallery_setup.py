"""Copy the gallery's shared setup, docs/gallery/common.py, into every page script.

Each page opens with that setup in full, so its code can be pasted into a notebook and run as it
is. Edit common.py, run this from the repository root, and rebuild the pages:

    python tools/sync_gallery_setup.py

tests/test_docs.py checks that every page holds the current copy.
"""

import re
from pathlib import Path

GALLERY = Path(__file__).parent.parent / "docs" / "gallery"
# The setup cell runs from its header to the next cell.
SETUP = re.compile(r"(?ms)^# %% \[setup\]\n.*?(?=^# %%)")


def main() -> None:
    cell = "# %% [setup]\n" + (GALLERY / "common.py").read_text().rstrip("\n") + "\n\n\n"
    for script in sorted(GALLERY.glob("*/*.py")):
        text = script.read_text()
        new, n = SETUP.subn(lambda _: cell, text)
        if n != 1:
            raise SystemExit(f"{script}: expected one setup cell, found {n}")
        if new != text:
            script.write_text(new)
            print(f"updated {script.relative_to(GALLERY)}")


if __name__ == "__main__":
    main()
