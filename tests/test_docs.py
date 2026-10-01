"""README examples must be a subset of the listening gallery."""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent

# The source distribution on PyPI carries the code and tests but not the docs.
pytestmark = pytest.mark.skipif(
    not (ROOT / "docs" / "gallery" / "index.html").exists(), reason="docs are not in the sdist"
)


GALLERY = ROOT / "docs" / "gallery"
PAGES = ["index.html", "speech.html", "textures.html", "reverb.html", "irn.html"]


def test_readme_links_point_to_gallery_entries():
    readme = (ROOT / "README.md").read_text()
    links = re.findall(r"gallery/([\w]+\.html)?#d-([\w]+)\)", readme)
    assert len(links) >= 10
    for page, key in links:
        html = (GALLERY / (page or "index.html")).read_text()
        assert f'id="d-{key}"' in html, f"README links to d-{key}, which is not on {page or 'index.html'}"
    for img in re.findall(r"docs/images/(\w+)\.png", readme):
        assert (ROOT / "docs" / "images" / f"{img}.png").exists()


def test_gallery_pages_link_existing_media():
    for page in PAGES:
        html = (GALLERY / page).read_text()
        for src in re.findall(r'src="((?:audio|img)/[^"]+)"', html):
            assert (GALLERY / src).exists(), f"{page} links to missing {src}"


def test_gallery_playhead_regions_do_not_overlap():
    """Each time axis gets its own playhead; overlapping regions mean the
    positions were computed in the wrong coordinates (e.g. subfigure-relative)."""
    import html
    import json

    for page in PAGES:
        text = (GALLERY / page).read_text()
        for key, raw in re.findall(r'<article class="sound" id="d-(\w+)" data-regions="([^"]*)"', text):
            regions = json.loads(html.unescape(raw))
            assert regions, key
            for r in regions:
                assert 0 <= r["x0"] < r["x1"] <= 1 and 0 <= r["top"] < r["bottom"] <= 1, key
            for i, a in enumerate(regions):
                for b in regions[i + 1 :]:
                    overlap_x = min(a["x1"], b["x1"]) - max(a["x0"], b["x0"])
                    overlap_y = min(a["bottom"], b["bottom"]) - max(a["top"], b["top"])
                    assert overlap_x <= 1e-6 or overlap_y <= 1e-6, f"overlapping playhead regions in d-{key}"


def test_readme_source_links_point_at_definitions():
    """Reference tags in the README and the gallery link to src/ files and, for
    named functions or classes, to the line that defines them. Fix drifted lines with
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
    drifted = "source links have drifted; run python tools/update_readme_source_links.py"
    for path in links.linked_files():
        text = path.read_text()
        assert links.fixed(text) == text, f"{path.name}: {drifted}"


def test_gallery_reference_lists_are_in_the_readme():
    """Every work in a gallery page's References is also in the README's, and
    every source tag a gallery page links is the same form the README uses."""
    readme = (ROOT / "README.md").read_text()
    listed = readme[readme.index("## References") : readme.index("### Reference implementations")]
    for page in sorted(GALLERY.glob("*.py")):
        text = page.read_text()
        if "# ## References" not in text:
            continue
        refs = text[text.index("# ## References") :]
        for entry in re.findall(r"(?m)^# - (.+)", refs):
            first, year = entry.split()[0].strip(","), re.search(r"\((\d{4})\)", entry)
            key = rf"(?m)^- {re.escape(first)}\b[^(\n]*" + (rf"\({year[1]}\)" if year else "")
            missing = f"{page.name} cites {entry[:40]}..., missing from README References"
            assert re.search(key, listed), missing
