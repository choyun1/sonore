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


def test_gallery_playhead_regions_do_not_overlap():
    """Each time axis gets its own playhead; overlapping regions mean the
    positions were computed in the wrong coordinates (e.g. subfigure-relative)."""
    import html
    import json

    page = (ROOT / "docs" / "gallery" / "index.html").read_text()
    for key, raw in re.findall(r'<article class="sound" id="d-(\w+)" data-regions="([^"]*)"', page):
        regions = json.loads(html.unescape(raw))
        for r in regions:
            assert 0 <= r["x0"] < r["x1"] <= 1 and 0 <= r["top"] < r["bottom"] <= 1, key
        for i, a in enumerate(regions):
            for b in regions[i + 1 :]:
                overlap_x = min(a["x1"], b["x1"]) - max(a["x0"], b["x0"])
                overlap_y = min(a["bottom"], b["bottom"]) - max(a["top"], b["top"])
                assert overlap_x <= 1e-6 or overlap_y <= 1e-6, f"overlapping playhead regions in d-{key}"


def test_readme_source_links_point_at_definitions():
    """Reference tags link to src/ files and, for named functions or classes,
    to the line that defines them. Fix drifted lines with
    ``python tools/update_readme_source_links.py``."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("links", ROOT / "tools" / "update_readme_source_links.py")
    links = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(links)
    readme = (ROOT / "README.md").read_text()
    found = links.LINK.findall(readme)
    assert len(found) >= 30
    assert readme.count(links.BLOB + "src/") == len(found), (
        "a source link does not match the `module.name` form"
    )
    drifted = "README source links have drifted; run python tools/update_readme_source_links.py"
    assert links.fixed(readme) == readme, drifted
