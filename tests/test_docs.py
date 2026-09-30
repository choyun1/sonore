"""README examples must be a subset of the listening gallery."""

import re
from pathlib import Path

ROOT = Path(__file__).parent.parent


def test_readme_links_point_to_gallery_entries():
    readme = (ROOT / "README.md").read_text()
    page = (ROOT / "docs" / "gallery" / "index.html").read_text()
    keys = re.findall(r"gallery/#d-([\w]+)\)", readme)
    assert len(keys) >= 10
    ids = set(re.findall(r'<article class="sound" id="d-([\w]+)"', page))
    assert set(keys) <= ids, set(keys) - ids
    for img in re.findall(r"docs/images/(\w+)\.png", readme):
        assert (ROOT / "docs" / "images" / f"{img}.png").exists()
