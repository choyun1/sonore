"""Draw docs/design/layout.svg, the import graph of the package, from the source.

    python tools/draw_layout.py

Each layer is a band, bottom to top as in tests/test_layers.py. Inside a band,
a module sits one row above every module of its own layer that it imports at
module level, so arrows inside a band point down. Grey arrows are module-level
imports; arrows into core from the layers above are left out, since nearly
every module imports sound and utils. Red dashed arrows are imports inside a
function or a TYPE_CHECKING block that point up or sideways. Imports of
plotting are listed beside its box instead of drawn.
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PACKAGE = ROOT / "src" / "sonore"
OUT = ROOT / "docs" / "design" / "layout.svg"

LAYERS = ["core", "signals", "analysis", "stimuli", "texture"]
BAND_COLOURS = {
    "texture": "#efe6fb",
    "stimuli": "#fdeedd",
    "analysis": "#e3f0fb",
    "signals": "#e5f5e8",
    "core": "#eeeeee",
}
GREY, RED = "#667", "#d0451b"

BAND_LEFT, BAND_WIDTH = 10, 850
BOX_HEIGHT, ROW_HEIGHT, BAND_PADDING, BAND_GAP = 34, 70, 18, 35
LABEL_WIDTH = 95  # room on the left of each band for its name


def modules() -> dict[str, Path]:
    """Every module file in the layers, as 'layer.module'."""
    found = {}
    for layer in LAYERS:
        for path in sorted((PACKAGE / layer).glob("*.py")):
            if path.stem != "__init__":
                found[f"{layer}.{path.stem}"] = path
    return found


def imports(path: Path, known: set[str]) -> list[tuple[str, bool]]:
    """(imported module, whether at module level) for each import of another sonore module."""
    tree = ast.parse(path.read_text())
    top = {id(node) for node in tree.body}
    layer = path.parent.name
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.level:  # relative: from .frames import X, from ..core.sound import Sound
            base_parts = [layer] if node.level == 1 else []
            base = ".".join(base_parts + ([node.module] if node.module else []))
        elif node.module and node.module.split(".")[0] == "sonore":
            base = node.module.removeprefix("sonore").lstrip(".")
        else:
            continue
        for alias in node.names:
            candidates = [f"{base}.{alias.name}" if base else alias.name, base]
            target = next((name for name in candidates if name in known or name == "plotting"), None)
            if target:
                found.append((target, id(node) in top))
    return found


def rows_within_layers(graph: dict[str, set[str]]) -> dict[str, int]:
    """Row of each module inside its layer: one above the highest same-layer module it imports."""
    row: dict[str, int] = {}

    def place(name: str) -> int:
        if name not in row:
            row[name] = -1  # guards against a cycle; module-level cycles cannot import anyway
            below = [place(other) for other in graph[name] if other.split(".")[0] == name.split(".")[0]]
            row[name] = 1 + max(below, default=-1)
        return row[name]

    for name in graph:
        place(name)
    return row


def render() -> str:
    """The diagram as SVG text."""
    known = modules()
    names = set(known)
    module_level = {name: set() for name in known}
    inside = {name: set() for name in known}
    plotting_users = []
    for name, path in known.items():
        for target, at_top in imports(path, names):
            if target == "plotting":
                plotting_users.append(name.split(".")[1])
            elif target != name:
                (module_level if at_top else inside)[name].add(target)
    row = rows_within_layers(module_level)

    # Bands from the top of the picture down: texture first.
    rows_per_layer = {layer: 1 + max(row[n] for n in known if n.startswith(layer + ".")) for layer in LAYERS}
    band_top, y = {}, 22
    for layer in reversed(LAYERS):
        band_top[layer] = y
        y += rows_per_layer[layer] * ROW_HEIGHT + 2 * BAND_PADDING - (ROW_HEIGHT - BOX_HEIGHT) + BAND_GAP
    legend_y = y + 5
    height = legend_y + 45

    def box_y(name: str) -> float:
        layer = name.split(".")[0]
        top_row = rows_per_layer[layer] - 1 - row[name]
        return band_top[layer] + BAND_PADDING + top_row * ROW_HEIGHT

    # Place each module near the modules it is joined to, so arrows stay short: rows are laid
    # out bottom up from what they import, then top down from what imports them, a few times.
    # Within a row, modules keep at least a box's width apart and stay inside the band.
    usable_left, usable_right = BAND_LEFT + LABEL_WIDTH, BAND_LEFT + BAND_WIDTH - 10
    all_rows = [
        sorted(n for n in known if n.startswith(layer + ".") and row[n] == level)
        for layer in LAYERS
        for level in range(rows_per_layer[layer])
    ]
    neighbours = {name: module_level[name] | inside[name] for name in known}
    for name in known:
        for target in module_level[name] | inside[name]:
            neighbours[target].add(name)
    box_width = {name: 150.0 for name in known}
    for members in all_rows:
        spacing = (usable_right - usable_left) / len(members)
        for name in members:
            box_width[name] = min(150.0, spacing - 12)
    x_centre = {}
    for members in all_rows:
        spacing = (usable_right - usable_left) / len(members)
        for index, name in enumerate(members):
            x_centre[name] = usable_left + spacing * (index + 0.5)

    def spread(members: list[str], wanted: dict[str, float]) -> None:
        """Positions as close to `wanted` as the minimum gap and the band's edges allow."""
        members = sorted(members, key=lambda name: wanted[name])
        gap = max(box_width[name] for name in members) + 12
        left, right = usable_left + gap / 2, usable_right - gap / 2
        positions = [wanted[name] for name in members]
        positions[0] = max(positions[0], left)
        for index in range(1, len(positions)):  # apart, from the left edge
            positions[index] = max(positions[index], positions[index - 1] + gap)
        positions[-1] = min(positions[-1], right)
        for index in range(len(positions) - 2, -1, -1):  # and back inside the right edge
            positions[index] = min(positions[index], positions[index + 1] - gap)
        for name, x in zip(members, positions, strict=True):
            x_centre[name] = x

    for _ in range(4):
        for members in all_rows + all_rows[::-1]:
            wanted = {
                name: sum(x_centre[other] for other in neighbours[name]) / len(neighbours[name])
                if neighbours[name]
                else x_centre[name]
                for name in members
            }
            spread(members, wanted)

    def edge(source: str, target: str, colour: str, dashed: bool) -> str:
        sx, sy, tx, ty = x_centre[source], box_y(source), x_centre[target], box_y(target)
        if abs(sy - ty) < 1:  # same row: side to side
            direction = 1 if tx > sx else -1
            x1, x2 = sx + direction * box_width[source] / 2, tx - direction * box_width[target] / 2
            y1 = y2 = sy + BOX_HEIGHT / 2
        elif ty > sy:  # target below
            x1, y1, x2, y2 = sx, sy + BOX_HEIGHT, tx, ty
        else:  # target above
            x1, y1, x2, y2 = sx, sy, tx, ty + BOX_HEIGHT
        same_layer = source.split(".")[0] == target.split(".")[0]
        if same_layer and abs(row[source] - row[target]) > 1:  # bow around the rows in between
            bend = min(x1, x2) - 0.6 * max(box_width[source], box_width[target])
            dash = 'stroke-dasharray="5,4" ' if dashed else ""
            marker = "r" if dashed else "a"
            return (
                f'<path d="M{x1 - 30:.0f},{y1:.0f} Q{bend:.0f},{(y1 + y2) / 2:.0f} {x2 - 30:.0f},{y2:.0f}" fill="none" '
                f'stroke="{colour}" stroke-width="1.3" {dash}marker-end="url(#{marker})" opacity="0.8"/>'
            )
        if dashed and abs(sy - ty) >= 1:  # beside any grey arrow between the same two boxes
            x1, x2 = x1 + 8, x2 + 8
        style = 'stroke-dasharray="5,4" marker-end="url(#r)"' if dashed else 'marker-end="url(#a)"'
        return (
            f'<line x1="{x1:.0f}" y1="{y1:.0f}" x2="{x2:.0f}" y2="{y2:.0f}" stroke="{colour}" '
            f'stroke-width="1.3" {style} opacity="0.8"/>'
        )

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 {height:.0f}" '
        'font-family="Helvetica,Arial,sans-serif" font-size="14">',
        '<defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
        f'orient="auto"><path d="M0,0L10,5L0,10z" fill="{GREY}"/></marker>',
        '<marker id="r" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
        f'orient="auto"><path d="M0,0L10,5L0,10z" fill="{RED}"/></marker></defs>',
        f'<rect width="1000" height="{height:.0f}" fill="#ffffff"/>',
    ]
    for layer in LAYERS:
        band_height = rows_per_layer[layer] * ROW_HEIGHT + 2 * BAND_PADDING - (ROW_HEIGHT - BOX_HEIGHT)
        parts.append(
            f'<rect x="{BAND_LEFT}" y="{band_top[layer]}" width="{BAND_WIDTH}" height="{band_height}" '
            f'rx="8" fill="{BAND_COLOURS[layer]}"/>'
        )
        parts.append(
            f'<text x="20" y="{band_top[layer] + band_height / 2 + 5:.0f}" font-weight="bold" fill="#444">{layer}/</text>'
        )
    for name in sorted(known):
        for target in sorted(module_level[name]):
            if target.startswith("core.") and not name.startswith("core."):
                continue
            parts.append(edge(name, target, GREY, dashed=False))
    for name in sorted(known):
        for target in sorted(inside[name] - module_level[name]):
            layer_up = LAYERS.index(target.split(".")[0]) > LAYERS.index(name.split(".")[0])
            same_layer = target.split(".")[0] == name.split(".")[0]
            if layer_up or (same_layer and row[target] >= row[name]):
                parts.append(edge(name, target, RED, dashed=True))
    for name in sorted(known):
        width = box_width[name]
        x, y = x_centre[name] - width / 2, box_y(name)
        parts.append(
            f'<rect x="{x:.1f}" y="{y:.0f}" width="{width:.1f}" height="{BOX_HEIGHT}" rx="5" fill="#fff" stroke="#556"/>'
        )
        parts.append(
            f'<text x="{x_centre[name]:.1f}" y="{y + 22:.0f}" text-anchor="middle">{name.split(".")[1]}</text>'
        )

    # plotting, beside the bands, with the modules that call it.
    plot_y = band_top["analysis"]
    parts.append(
        f'<rect x="875" y="{plot_y}" width="115" height="{BOX_HEIGHT}" rx="5" fill="#fff" stroke="{RED}"/>'
    )
    parts.append(f'<text x="932" y="{plot_y + 22}" text-anchor="middle">plotting</text>')
    lines, line = ["called lazily by"], ""
    for user in sorted(set(plotting_users)):
        candidate = f"{line}, {user}" if line else user
        if len(candidate) > 18:
            lines.append(line + ",")
            line = user
        else:
            line = candidate
    lines.append(line + " (.plot)")
    for index, text in enumerate(lines):
        parts.append(
            f'<text x="932" y="{plot_y + 56 + 17 * index}" text-anchor="middle" font-size="12" fill="{RED}">{text}</text>'
        )

    parts.append(
        f'<text x="20" y="{legend_y}" font-size="12" fill="#444">grey arrow: module-level import (A uses B); '
        "arrows into core/ from layers above it are omitted, since nearly every module imports sound and utils.</text>"
    )
    parts.append(
        f'<text x="20" y="{legend_y + 18}" font-size="12" fill="{RED}">red dashed: import inside a function or '
        "TYPE_CHECKING block that points up or sideways; these close the cycles.</text>"
    )
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def main() -> None:
    OUT.write_text(render())
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
