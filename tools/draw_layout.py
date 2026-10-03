"""Draw docs/design/layout.svg, the import graph of the package, from the source.

    python tools/draw_layout.py

The folders follow meaning, so the picture is drawn by import order and
coloured by folder. Each module sits one row above the highest module it
imports at module level, so grey arrows always point down and core is at the
bottom. Arrows into core from outside it are left out, since nearly every
module imports sound and utils, and so are arrows into view, the base class
of every view. Red dashed arrows are the imports inside a
function that point back up, the ones tests/test_layers.py lists by name.
Imports of plotting are listed beside its box instead of drawn. The import
graph is read by tests/test_layers.py, so the test and the picture agree.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "design" / "layout.svg"
sys.path.insert(0, str(ROOT / "tests"))

from test_layers import FOLDERS, folder, module_graph, reachable  # noqa: E402

FOLDER_COLOURS = {
    "core": "#e6e6e6",
    "sources": "#d9f0dd",
    "frames": "#d5eeee",
    "views": "#d8e8f8",
    "spatial": "#f8dbe3",
    "texture": "#e8dcf8",
}
GREY, RED = "#667", "#d0451b"

WIDTH, LEFT, RIGHT = 1180, 20, 1040  # the boxes use LEFT..RIGHT; plotting sits to the right
BOX_HEIGHT, ROW_HEIGHT, TOP = 34, 74, 20


def rows(at_module_level: dict[str, set[str]]) -> dict[str, int]:
    """Row of each module: one above the highest module it imports at module level."""
    row: dict[str, int] = {}

    def place(name: str) -> int:
        if name not in row:
            row[name] = 1 + max((place(target) for target in at_module_level[name]), default=-1)
        return row[name]

    for name in at_module_level:
        place(name)
    return row


def render() -> str:
    """The diagram as SVG text."""
    at_module_level, inside_functions = module_graph()
    # Package __init__ files (``texture``) re-export a module; draw the module instead.
    known = sorted(name for name in at_module_level if "." in name and folder(name) != "plotting")

    def resolve(target: str) -> set[str]:
        return {target} if target in known else {name for name in at_module_level[target] if name in known}

    down = {name: set().union(*map(resolve, at_module_level[name] - {"plotting"})) for name in known}
    back = {
        (name, target)
        for name in known
        for target in inside_functions[name]
        if target in known and name in reachable(target, at_module_level)
    }
    plotting_users = sorted(
        {name.split(".")[1] for name in known if "plotting" in at_module_level[name] | inside_functions[name]}
    )
    row = rows(down)
    n_rows = 1 + max(row.values())

    # Within a row, modules go in folder order, then by name; boxes share the row's width.
    by_row = [
        sorted((name for name in known if row[name] == level), key=lambda n: (FOLDERS.index(folder(n)), n))
        for level in range(n_rows)
    ]
    x_centre, box_width = {}, {}
    for members in by_row:
        spacing = (RIGHT - LEFT) / max(len(members), 1)
        for index, name in enumerate(members):
            x_centre[name] = LEFT + spacing * (index + 0.5)
            box_width[name] = min(150.0, spacing - 10)

    def box_y(name: str) -> float:
        return TOP + (n_rows - 1 - row[name]) * ROW_HEIGHT

    legend_y = TOP + n_rows * ROW_HEIGHT + 10
    height = legend_y + 60

    def edge(source: str, target: str, colour: str, dashed: bool) -> str:
        sx, sy, tx, ty = x_centre[source], box_y(source), x_centre[target], box_y(target)
        if abs(sy - ty) < 1:  # same row: side to side
            direction = 1 if tx > sx else -1
            x1, x2 = sx + direction * box_width[source] / 2, tx - direction * box_width[target] / 2
            y1 = y2 = sy + BOX_HEIGHT / 2
        elif ty > sy:  # target below
            x1, y1, x2, y2 = sx, sy + BOX_HEIGHT, tx, ty
        else:  # target above
            x1, y1, x2, y2 = sx + 8, sy, tx + 8, ty + BOX_HEIGHT
        style = 'stroke-dasharray="5,4" marker-end="url(#r)"' if dashed else 'marker-end="url(#a)"'
        return (
            f'<line x1="{x1:.0f}" y1="{y1:.0f}" x2="{x2:.0f}" y2="{y2:.0f}" stroke="{colour}" '
            f'stroke-width="1.2" {style} opacity="0.7"/>'
        )

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {height:.0f}" '
        'font-family="Helvetica,Arial,sans-serif" font-size="13">',
        '<defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
        f'orient="auto"><path d="M0,0L10,5L0,10z" fill="{GREY}"/></marker>',
        '<marker id="r" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
        f'orient="auto"><path d="M0,0L10,5L0,10z" fill="{RED}"/></marker></defs>',
        f'<rect width="{WIDTH}" height="{height:.0f}" fill="#ffffff"/>',
    ]
    for name in known:
        for target in sorted(down[name]):
            if (folder(target) == "core" and folder(name) != "core") or target == "views.view":
                continue
            parts.append(edge(name, target, GREY, dashed=False))
    for name, target in sorted(back):
        parts.append(edge(name, target, RED, dashed=True))
    for name in known:
        width = box_width[name]
        x, y = x_centre[name] - width / 2, box_y(name)
        parts.append(
            f'<rect x="{x:.1f}" y="{y:.0f}" width="{width:.1f}" height="{BOX_HEIGHT}" rx="5" '
            f'fill="{FOLDER_COLOURS[folder(name)]}" stroke="#556"/>'
        )
        parts.append(
            f'<text x="{x_centre[name]:.1f}" y="{y + 22:.0f}" text-anchor="middle" font-size="12">'
            f"{name.split('.')[1]}</text>"
        )

    # plotting, beside the rows, with the modules that call it.
    plot_x = (RIGHT + WIDTH) / 2
    parts.append(
        f'<rect x="{plot_x - 57:.0f}" y="{TOP}" width="115" height="{BOX_HEIGHT}" rx="5" fill="#fff" stroke="{RED}"/>'
    )
    parts.append(f'<text x="{plot_x:.0f}" y="{TOP + 22}" text-anchor="middle">plotting</text>')
    lines, line = ["called lazily by"], ""
    for user in plotting_users:
        candidate = f"{line}, {user}" if line else user
        if len(candidate) > 18:
            lines.append(line + ",")
            line = user
        else:
            line = candidate
    lines.append(line + " (.plot)")
    for index, text in enumerate(lines):
        parts.append(
            f'<text x="{plot_x:.0f}" y="{TOP + 56 + 17 * index}" text-anchor="middle" font-size="12" fill="{RED}">{text}</text>'
        )

    # Legend: one swatch per folder, then the two kinds of arrow.
    for index, name in enumerate(FOLDER_COLOURS):
        x = LEFT + 110 * index
        parts.append(
            f'<rect x="{x}" y="{legend_y}" width="18" height="14" rx="3" fill="{FOLDER_COLOURS[name]}" stroke="#556"/>'
        )
        parts.append(f'<text x="{x + 24}" y="{legend_y + 12}" font-size="12" fill="#444">{name}/</text>')
    parts.append(
        f'<text x="{LEFT}" y="{legend_y + 34}" font-size="12" fill="#444">grey arrow: module-level import (A uses B); '
        "arrows into core/ from outside it, and into view (the base of every view), are omitted.</text>"
    )
    parts.append(
        f'<text x="{LEFT}" y="{legend_y + 52}" font-size="12" fill="{RED}">red dashed: import inside a function '
        "that points back up, listed by name in tests/test_layers.py.</text>"
    )
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def main() -> None:
    OUT.write_text(render())
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
