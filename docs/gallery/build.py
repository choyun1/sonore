"""Build the listening gallery: the topic pages, each a runnable script, and an index of them.

    python docs/gallery/build.py                 # -> docs/gallery/ (for GitHub Pages)
    python docs/gallery/build.py --single out.html   # also a self-contained page

Everything is generated from sonore itself with fixed seeds, so the gallery is
reproducible. Audio is FLAC (lossless: lossy codecs would alter the interaural
phase and correlation that the binaural examples are about).
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import html
import io
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

import sonore as so  # noqa: E402

HERE = Path(__file__).parent
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "figure.dpi": 100})


# ------------------------------------------------------------------- figures
def encode_figure(fig, time_axes) -> tuple[bytes, list[dict], tuple[int, int]]:
    """The figure as a 256-colour PNG, and where each time axis sits (for the playhead)."""
    # Constrained layout moves the axes a little on its second pass, so lay the figure out once without
    # drawing anything, then draw it for real: the pixels and the playhead regions below then come from
    # the same, settled layout, and the figure is rendered only once.
    fig.draw_without_rendering()
    fig.canvas.draw()
    regions = []
    for ax in time_axes:
        # ax.bbox is in display units; converting through the *figure* transform gives true figure
        # fractions even for axes inside subfigures (whose get_position() is relative to the subfigure).
        box, (t0, t1) = ax.bbox.transformed(fig.transFigure.inverted()), ax.get_xlim()
        regions.append(
            {"x0": box.x0, "x1": box.x1, "top": 1 - box.y1, "bottom": 1 - box.y0, "t0": t0, "t1": t1}
        )
    # The pixels of that draw are what savefig would write (the figure dpi is 100 in rcParams, and there
    # is no bbox_inches cropping, which keeps figure fractions valid for the playhead).
    image = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())).convert("RGB")
    plt.close(fig)
    quantized = image.quantize(colors=256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    out = io.BytesIO()
    quantized.save(out, "PNG", optimize=True)
    return out.getvalue(), regions, image.size


def flac_bytes(snd: so.Sound) -> bytes:
    import soundfile as sf

    buf = io.BytesIO()
    sf.write(buf, snd.data, int(snd.fs), format="FLAC", subtype="PCM_16")
    return buf.getvalue()


def file_name(key: str, title: str) -> str:
    return f"{key}_{title.lower().replace(' ', '_').replace(',', '')}"


# ------------------------------------------------------------- example pages
# The example pages (speech, textures, moving, vocoder, cepstrum, ...) are runnable scripts in percent format,
# where "# %%" starts a cell:
#
#   # %% [markdown]            prose: "# Title" names the page, "## Heading" starts a section
#   # %% [about]               the description of the example that follows
#   # %% [demo KEY] Title      code that leaves `sound`, `fig` and `playhead` (the axes the playhead follows),
#                              and optionally `scene`, sources to draw from above as the sound plays,
#                              and `live`, an image that changes frame by frame as the sound plays,
#                              and `lissajous`, two notes drawn against each other as the sound plays
#   # %% [figure KEY] Title    code that leaves `fig`: a figure without sound
#   # %%                       any other code; what it prints is shown under it
#
# Every code cell is shown on the page exactly as it ran. Prose may use $TeX$, $$display TeX$$,
# `code`, **bold**, *italic*, [links](url), "- " lists, and {{ expression }}, which is evaluated
# where the cell stands. The scripts run from the repository root.
EXAMPLE_PAGES = [
    "speech",
    "cepstrum",
    "harmonics",
    "formants",
    "aperiodicity",
    "voice",
    "modspectrogram",
    "modtargets",
    "resynthesis",
    "pv",
    "ripples",
    "irn",
    "binaural",
    "classic",
    "textures",
    "moving",
    "reverb",
    "vocoder",
    "timbre",
    "temperament",
    "organ",
]
ROOT = HERE.parent.parent
CELL = re.compile(r"# %%(?: \[(\w+)(?: (\w+))?\])?(?: (.*))?")


@dataclass
class Cell:
    kind: str  # markdown, about, demo, figure or code
    source: str
    key: str = ""
    title: str = ""


def read_cells(path: Path) -> list[Cell]:
    cells = []
    for chunk in re.split(r"(?m)^(?=# %%)", path.read_text())[
        1:
    ]:  # what precedes the first cell is not shown
        head, _, body = chunk.partition("\n")
        m = CELL.fullmatch(head.rstrip())
        if not m:
            raise ValueError(f"{path.name}: bad cell header {head!r}")
        kind = m.group(1) or "code"
        body = body.strip("\n")
        if kind in ("markdown", "about"):
            body = "\n".join(re.sub(r"^# ?", "", line) for line in body.splitlines())
        cells.append(Cell(kind, body, m.group(2) or "", (m.group(3) or "").strip()))
    return cells


# A link target may hold one level of parentheses, as DOIs such as 10.1016/S0167-6393(98)00032-6 do.
URL = r"((?:[^()\s]|\([^()\s]*\))+)"
INLINE = re.compile(r"\$\$(.+?)\$\$|\$(.+?)\$|\[`([^`]+)`\]\(" + URL + r"\)|`([^`]+)`", re.S)


def _marks(s: str) -> str:
    s = re.sub(r"\[([^\]]+)\]\(" + URL + r"\)", r'<a href="\2">\1</a>', s)
    s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s, flags=re.S)
    return re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"<em>\1</em>", s, flags=re.S)


def inline(text: str) -> str:
    out, pos = [], 0
    for m in INLINE.finditer(text):
        out.append(_marks(html.escape(text[pos : m.start()], quote=False)))
        display, tex, link_code, href, code = m.groups()
        if link_code is not None:  # a source tag, [`module.Name`](url)
            out.append(f'<a href="{html.escape(href)}"><code>{html.escape(link_code)}</code></a>')
        elif code is not None:
            out.append(f"<code>{html.escape(code)}</code>")
        else:
            cls = "tex display" if display is not None else "tex"
            out.append(f'<span class="{cls}">{html.escape((display or tex).strip())}</span>')
        pos = m.end()
    out.append(_marks(html.escape(text[pos:], quote=False)))
    return "".join(out)


def markdown(text: str) -> list[tuple[str, str]]:
    """Blocks of prose as (kind, html), kind one of h1, h2, h3 or p (anything else)."""
    blocks = []
    for block in re.split(r"\n\s*\n", text.strip()):
        lines = block.splitlines()
        if m := re.match(r"(#{1,3}) (.*)", block):
            blocks.append((f"h{len(m.group(1))}", inline(m.group(2).strip())))
        elif block.startswith("$$") and block.endswith("$$"):
            blocks.append(("p", f'<div class="tex display">{html.escape(block[2:-2].strip())}</div>'))
        elif lines[0].startswith("- "):
            items = re.split(r"(?m)^- ", block)[1:]
            lis = "".join(f"<li>{inline(' '.join(i.split()))}</li>" for i in items)
            blocks.append(("p", f"<ul>{lis}</ul>"))
        else:
            blocks.append(("p", f"<p>{inline(block)}</p>"))
    return blocks


def substitute(text: str, ns: dict) -> str:
    return re.sub(r"\{\{(.+?)\}\}", lambda m: str(eval(m.group(1), ns)), text)  # noqa: S307


def code_block(source: str) -> str:
    return f'<pre><code class="language-python">{html.escape(source)}</code></pre>'


def example_page(path: Path) -> dict:
    """Run a page script cell by cell. Returns its title, intro and sections, where each
    part is either HTML or an example (a dict with the media to write)."""
    ns: dict = {"__name__": "__gallery__", "__file__": str(path)}
    title, intro, sections, about = "", [], [], ""
    cwd = Path.cwd()
    os.chdir(ROOT)
    try:
        for cell in read_cells(path):
            if cell.kind in ("markdown", "about"):
                blocks = markdown(substitute(cell.source, ns))
                if cell.kind == "about":
                    about = "".join(h for _, h in blocks)
                    continue
                for kind, h in blocks:
                    if kind == "h1":
                        title = html.unescape(re.sub("<[^>]+>", "", h))
                    elif kind == "h2":
                        sections.append({"title": h, "parts": []})
                    else:
                        (sections[-1]["parts"] if sections else intro).append(h)
                continue
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                exec(compile(cell.source, f"{path.name}, cell {cell.key or cell.title}", "exec"), ns)  # noqa: S102
            part = {"key": cell.key, "title": cell.title, "about": about, "code": code_block(cell.source)}
            if cell.kind == "demo":
                snd = ns.pop("sound")
                png, regions, size = encode_figure(ns.pop("fig"), ns.pop("playhead"))
                part.update(sound=snd, audio=flac_bytes(snd), png=png, regions=regions, size=size)
                if "scene" in ns:
                    part["scene"] = scene_json(ns.pop("scene"))
                if "live" in ns:
                    part["live"] = live_json(ns.pop("live"))
                if "lissajous" in ns:
                    part["lissajous"] = lissajous_json(ns.pop("lissajous"))
            elif cell.kind == "figure":
                png, _, size = encode_figure(ns.pop("fig"), [])
                part.update(png=png, size=size)
            else:
                printed = out.getvalue().rstrip()
                code = f'<details class="code"><summary>Code</summary>{part["code"]}</details>'
                part = code + (f'<pre class="out">{html.escape(printed)}</pre>' if printed else "")
                part = f'<div class="cell">{part}</div>'
            (sections[-1]["parts"] if sections else intro).append(part)
            about = ""
    finally:
        os.chdir(cwd)
    return {"title": title, "intro": intro, "sections": sections}


# ---------------------------------------------------------------------- page
CSS = """
:root { --paper: #ECEFF1; --plate: #FFFFFF; --ink: #16202A; --muted: #566573; --accent: #235B7C;
  --rule: #CDD4DA; --head: #235B7C; --code: #F7F9FA; --string: #2F6B3A; --number: #8A4B12;
  --serif: "Spectral", Georgia, "Times New Roman", serif;
  --sans: "Atkinson Hyperlegible", system-ui, -apple-system, "Segoe UI", sans-serif;
  --mono: ui-monospace, "SF Mono", Menlo, Consolas, "Liberation Mono", monospace;
  box-sizing: border-box; padding-top: env(safe-area-inset-top, 0px); padding-bottom: env(safe-area-inset-bottom, 0px); }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { --paper: #121A21; --ink: #E4E9ED;
  --muted: #9AA8B4; --accent: #7FB6D6; --rule: #2C3A46; --code: #0C1318; --string: #9CCB8E; --number: #E0A76B; } }
:root[data-theme="dark"] { --paper: #121A21; --ink: #E4E9ED; --muted: #9AA8B4; --accent: #7FB6D6; --rule: #2C3A46;
  --code: #0C1318; --string: #9CCB8E; --number: #E0A76B; }
*, *::before, *::after { box-sizing: inherit; }
html { scroll-padding-top: env(safe-area-inset-top, 0px); }
body { margin: 0; background: var(--paper); color: var(--ink); font-family: var(--serif); font-size: 1.0625rem; line-height: 1.6; }
main { max-width: 76rem; margin: 0 auto; padding: 3.5rem clamp(1rem, 4vw, 2.5rem) 5rem; }
header { max-width: 44rem; margin-bottom: 3.5rem; }
h1 { font-weight: 500; font-size: clamp(2.4rem, 5vw, 3.6rem); line-height: 1.05; letter-spacing: -0.015em; margin: 0 0 1.25rem; }
header p { margin: 0 0 0.9rem; }
header ul { margin: 0 0 0.9rem; padding-left: 1.25rem; }
header li { margin: 0 0 0.35rem; }
header .how { font-family: var(--sans); font-size: 0.95rem; color: var(--muted); line-height: 1.55; }
nav.pages { display: flex; flex-wrap: wrap; align-items: center; gap: 0.4rem 1.5rem; font-family: var(--sans); font-size: 0.95rem; margin: 0 0 2rem; }
nav.pages a { color: var(--accent); }
nav.pages a[aria-current] { color: var(--ink); font-weight: 700; text-decoration: none; }
nav.pages details { position: relative; }
nav.pages summary { color: var(--accent); cursor: pointer; list-style: none; }
nav.pages summary::-webkit-details-marker { display: none; }
nav.pages summary::after { content: "\\25BE"; margin-left: 0.3rem; font-size: 0.8em; }
nav.pages details.here > summary { color: var(--ink); font-weight: 700; }
nav.pages details ul { position: absolute; z-index: 10; top: calc(100% + 0.4rem); left: 0; min-width: 15rem; margin: 0;
  padding: 0.4rem 0; list-style: none; background: var(--paper); border: 1px solid var(--rule); border-radius: 3px;
  box-shadow: 0 6px 18px rgba(0, 0, 0, 0.12); }
nav.pages details li a { display: block; padding: 0.35rem 0.9rem; white-space: nowrap; }
nav.pages a.ref { margin-left: auto; }
nav.pages a.ref + a.repo { margin-left: 0; }
nav.pages a.repo { margin-left: auto; color: var(--muted); display: inline-flex; }
nav.pages a.repo:hover { color: var(--ink); }
@media (max-width: 34rem) { nav.pages a.repo { margin-left: 0; } }
a { color: var(--accent); }
section { border-top: 1px solid var(--rule); padding-top: 2.25rem; margin-top: 3rem; }
h2 { font-weight: 500; font-size: 1.75rem; line-height: 1.2; margin: 0 0 0.5rem; }
.section-intro { max-width: 44rem; color: var(--muted); margin: 0 0 1rem; }
section > p, section > ul, section > .tex.display { max-width: 44rem; }
section > p, header > p { margin: 0 0 1rem; }
.sound, .still { display: grid; grid-template-columns: minmax(15rem, 19rem) minmax(0, 1fr); gap: 1.25rem 2rem; align-items: start; padding: 1.75rem 0; }
.sound > *, .still > * { min-width: 0; }
article + article, .cell + article, article + .cell { border-top: 1px dotted var(--rule); }
article.sound, article.still { scroll-margin-top: 1.5rem; }
article:target h3 { text-decoration: underline; text-decoration-thickness: 2px; text-underline-offset: 4px; }
h3 { font-weight: 600; font-size: 1.25rem; line-height: 1.25; margin: 0 0 0.5rem; }
.desc { margin: 0 0 1rem; }
.desc p { margin: 0 0 0.75rem; }
.headphones { font-size: 1rem; font-weight: 400; margin-left: 0.35rem; cursor: help; }
audio { width: 100%; max-width: 19rem; display: block; }
.scene { display: block; width: 100%; max-width: 19rem; aspect-ratio: 1; margin-top: 1rem; }
.live { display: block; width: 100%; max-width: 19rem; aspect-ratio: 0.95; margin-top: 1rem; }
.lissajous { display: block; width: 100%; max-width: 16rem; aspect-ratio: 1; margin-top: 1rem; }
audio:focus-visible, .plate:focus-visible { outline: 3px solid var(--accent); outline-offset: 3px; }
.file { font-family: var(--sans); font-size: 0.8rem; color: var(--muted); margin: 0.5rem 0 0; overflow-wrap: anywhere; }
figure { margin: 0; }
.plate { position: relative; background: var(--plate); border-radius: 2px; cursor: crosshair; }
.still .plate { cursor: auto; }
.plate img { display: block; width: 100%; height: auto; }
.heads { position: absolute; inset: 0; pointer-events: none; }
.head { position: absolute; width: 2px; margin-left: -1px; background: var(--head); box-shadow: 0 0 0 1px rgba(255,255,255,0.7); display: none; }
code { font-family: var(--mono); font-size: 0.86em; }
pre { font-family: var(--mono); font-size: 0.8rem; line-height: 1.5; background: var(--code); border: 1px solid var(--rule);
  border-radius: 3px; padding: 0.85rem 1rem; margin: 0; overflow-x: auto; }
pre code { font-size: inherit; }
pre.out { background: transparent; border-style: dashed; margin-top: 0.5rem; }
.cell { padding: 1.25rem 0; }
details.code { grid-column: 1 / -1; }
details.code summary { font-family: var(--sans); font-size: 0.9rem; color: var(--muted); cursor: pointer; margin: 0 0 0.5rem; }
.hljs-keyword, .hljs-built_in, .hljs-literal { color: var(--accent); }
.hljs-string { color: var(--string); }
.hljs-number { color: var(--number); }
.hljs-comment { color: var(--muted); font-style: italic; }
.tex.display { display: block; margin: 1rem 0 1.25rem; overflow-x: auto; overflow-y: hidden; }
footer { margin-top: 4rem; font-family: var(--sans); font-size: 0.85rem; color: var(--muted); max-width: 40rem; }
@media (max-width: 54rem) { .sound, .still { grid-template-columns: minmax(0, 1fr); gap: 1rem; } audio { max-width: none; } }
.side { display: none; }
@media (min-width: 75rem) {
  .layout { display: grid; grid-template-columns: 16rem minmax(0, 1fr); }
  .side { display: block; position: sticky; top: 0; height: 100vh; overflow-y: auto; padding: 2.5rem 1.25rem 2rem 1.75rem;
    border-right: 1px solid var(--rule); font-family: var(--sans); font-size: 0.9rem; line-height: 1.4; }
  html:not(.side-closed) nav.pages > a:first-child, html:not(.side-closed) nav.pages details { display: none; }
  .side-closed .layout { grid-template-columns: 3rem minmax(0, 1fr); }
  .side-closed .side { padding: 2.5rem 0 0; overflow: hidden; }
  .side-closed .side .home, .side-closed .side-body { display: none; }
  .side-closed .side-top { justify-content: center; }
  .layout { transition: grid-template-columns 0.25s ease; }
  .side-opening .side .home, .side-opening .side-body { animation: side-in 0.25s ease both; }
  .side-closing .side .home, .side-closing .side-body { animation: side-in 0.15s ease reverse both; }
  .side-opening .side { overflow: hidden; }
  .side-opening .side-top, .side-opening .side-body { min-width: 13rem; }
}
@keyframes side-in { from { opacity: 0; transform: translateX(-0.75rem); } to { opacity: 1; transform: none; } }
.side .has-sections { display: flex; flex-wrap: wrap; align-items: baseline; }
.side .has-sections > a { flex: 1; }
.sections-toggle { flex: none; font: inherit; font-size: 0.8rem; line-height: 1; color: var(--muted); background: none;
  border: 0; padding: 0.2rem 0.3rem; cursor: pointer; }
.sections-toggle:hover { color: var(--ink); }
.sections-toggle::before { content: "\\25B8"; display: inline-block; transition: transform 0.15s; }
.sections-toggle[aria-expanded="true"]::before { transform: rotate(90deg); }
.side .sections { flex-basis: 100%; display: grid; grid-template-rows: 0fr; visibility: hidden;
  transition: grid-template-rows 0.2s ease, visibility 0.2s; }
.side .sections.open { grid-template-rows: 1fr; visibility: visible; }
.side .sections > div { overflow: hidden; min-height: 0; }
@media (prefers-reduced-motion: reduce) {
  .layout, .side .sections, .sections-toggle::before { transition: none; }
  .side-opening .side .home, .side-opening .side-body, .side-closing .side .home, .side-closing .side-body { animation: none; }
}
.side a { text-decoration: none; }
.side a:hover { text-decoration: underline; }
.side a[aria-current] { color: var(--ink); font-weight: 700; }
.side-top { display: flex; align-items: flex-start; justify-content: space-between; gap: 0.5rem; margin: 0 0 1.25rem; }
.side .home { font-family: var(--serif); font-size: 1.6rem; font-weight: 600; line-height: 1.1; color: var(--ink);
  letter-spacing: -0.01em; padding-bottom: 0.9rem; border-bottom: 2px solid var(--accent); }
.side .home:hover { color: var(--accent); text-decoration: none; }
.side .home[aria-current] { font-weight: 600; }
.side-toggle { flex: none; font: inherit; font-size: 1.1rem; line-height: 1; color: var(--muted); background: none;
  border: 1px solid var(--rule); border-radius: 4px; width: 1.9rem; height: 1.9rem; cursor: pointer; }
.side-toggle:hover { color: var(--ink); border-color: var(--muted); }
.side .group { margin: 1.1rem 0 0; }
.side .group > summary { font-size: 0.75rem; font-weight: 700; letter-spacing: 0.06em; text-transform: uppercase; color: var(--muted);
  cursor: pointer; list-style: none; padding: 0.2rem 0; margin: 0 0 0.3rem; }
.side .group > summary::-webkit-details-marker { display: none; }
.side .group > summary::before { content: "\\25B8"; display: inline-block; width: 1em; transition: transform 0.15s; }
.side .group[open] > summary::before { transform: rotate(90deg); }
.side .group > summary:hover { color: var(--ink); }
.side .group > ul { padding-left: 1em; }
.side ul { list-style: none; margin: 0; padding: 0; }
.side li a { display: block; padding: 0.2rem 0; }
.side ul ul { margin: 0.25rem 0 0.5rem 0.2rem; padding-left: 0.75rem; border-left: 2px solid var(--rule); }
.side ul ul a { color: var(--muted); font-size: 0.85rem; }
.side ul ul a:hover { color: var(--ink); }
"""

JS = """
(() => {
  const items = [...document.querySelectorAll(".sound")].map((el) => {
    const regions = JSON.parse(el.dataset.regions);
    const heads = el.querySelector(".heads");
    const lines = regions.map((r) => {
      const d = document.createElement("div");
      d.className = "head";
      d.style.top = (r.top * 100) + "%";
      d.style.height = ((r.bottom - r.top) * 100) + "%";
      heads.appendChild(d);
      return d;
    });
    const canvas = el.querySelector(".scene");
    const scene = canvas ? JSON.parse(canvas.dataset.scene) : null;
    const liveCanvas = el.querySelector(".live");
    let live = null;
    if (liveCanvas) {
      live = JSON.parse(liveCanvas.dataset.live);
      live.bytes = Uint8Array.from(atob(live.data), (c) => c.charCodeAt(0));
      live.played = false;
    }
    const figureCanvas = el.querySelector(".lissajous");
    const figure = figureCanvas ? JSON.parse(figureCanvas.dataset.lissajous) : null;
    return { el, audio: el.querySelector("audio"), plate: el.querySelector(".plate"), regions, lines, canvas, scene,
      liveCanvas, live, figureCanvas, figure };
  });
  // A top-down view: the listener's head in the middle, nose up (the front), and each source on a
  // circle at its azimuth (clockwise from straight ahead) at the current time.
  function drawScene(item) {
    const c = item.canvas;
    if (!c) return;
    const dpr = window.devicePixelRatio || 1, size = c.clientWidth;
    if (c.width !== Math.round(size * dpr)) { c.width = c.height = Math.round(size * dpr); }
    const g = c.getContext("2d"), css = getComputedStyle(document.documentElement);
    const ink = css.getPropertyValue("--ink").trim(), muted = css.getPropertyValue("--muted").trim();
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, size, size);
    // Sources at 1 m unless the scene gives distances; the farthest one sets the scale.
    const farthest = Math.max(1, ...item.scene.flatMap((src) => src.r || [1]));
    const cx = size / 2, cy = size / 2, R = size * 0.38, unit = R / farthest, t = item.audio.currentTime;
    const head = Math.min(size * 0.07, unit * 0.12);
    g.strokeStyle = muted; g.lineWidth = 1; g.setLineDash([3, 4]);
    g.fillStyle = muted; g.font = "11px system-ui, sans-serif"; g.textAlign = "center";
    (farthest > 1.5 ? [1, Math.round(farthest)] : [1]).forEach((m) => {
      g.beginPath(); g.arc(cx, cy, unit * m, 0, 2 * Math.PI); g.stroke();
      g.fillText(`${m} m`, cx + unit * m * 0.72, cy + unit * m * 0.72 + 14);
    });
    g.setLineDash([]);
    g.strokeStyle = ink; g.lineWidth = 1.5;
    g.beginPath(); g.moveTo(cx - head * 0.45, cy - head * 0.9); g.lineTo(cx, cy - head * 1.45); g.lineTo(cx + head * 0.45, cy - head * 0.9); g.stroke();
    g.beginPath(); g.arc(cx, cy, head, 0, 2 * Math.PI); g.stroke();
    g.beginPath(); g.ellipse(cx - head, cy, head * 0.18, head * 0.4, 0, 0, 2 * Math.PI); g.stroke();
    g.beginPath(); g.ellipse(cx + head, cy, head * 0.18, head * 0.4, 0, 0, 2 * Math.PI); g.stroke();
    const at = (src, values, time) => {
      if (!src.t) return values[0];
      const k = Math.min(values.length - 1, Math.max(0, time / src.t)), i = Math.floor(k), f = k - i;
      return i + 1 < values.length ? values[i] * (1 - f) + values[i + 1] * f : values[i];
    };
    const xy = (az, r) => [cx + unit * r * Math.sin(az * Math.PI / 180), cy - unit * r * Math.cos(az * Math.PI / 180)];
    item.scene.forEach((src) => {
      const distances = src.r || src.az.map(() => 1);
      if (src.t) {  // the path it travels, faintly
        g.strokeStyle = src.color; g.globalAlpha = 0.25; g.lineWidth = 6; g.lineCap = "round"; g.lineJoin = "round";
        g.beginPath();
        src.az.forEach((az, i) => { const [x, y] = xy(az, distances[i]); i ? g.lineTo(x, y) : g.moveTo(x, y); });
        g.stroke();
        g.globalAlpha = 1;
      }
      const [x, y] = xy(at(src, src.az, t), at(src, distances, t));
      g.fillStyle = src.color; g.beginPath(); g.arc(x, y, size * 0.035, 0, 2 * Math.PI); g.fill();
      g.fillStyle = ink; g.fillText(src.label, x, y + size * 0.035 + 13);
    });
  }
  // One frame of a live image: cells on an even grid (the axes are log-spaced), round values
  // marked on each axis, and a colour bar. Before the sound has played, the frame at live.start.
  function drawLive(item) {
    const c = item.liveCanvas;
    if (!c) return;
    const L = item.live, dpr = window.devicePixelRatio || 1, W = c.clientWidth, H = c.clientHeight;
    if (c.width !== Math.round(W * dpr)) { c.width = Math.round(W * dpr); c.height = Math.round(H * dpr); }
    const g = c.getContext("2d"), css = getComputedStyle(document.documentElement);
    const ink = css.getPropertyValue("--ink").trim(), muted = css.getPropertyValue("--muted").trim();
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, W, H);
    const t = L.played || item.audio.currentTime > 0 ? item.audio.currentTime : L.start;
    const n = L.bytes.length / (L.rows * L.cols), f = Math.min(n - 1, Math.max(0, Math.round(t / L.dt)));
    const left = 52, right = 40, top = 22, bottom = 34, w = W - left - right, h = H - top - bottom;
    const cw = w / L.cols, ch = h / L.rows, off = f * L.rows * L.cols;
    for (let r = 0; r < L.rows; r++) {
      for (let k = 0; k < L.cols; k++) {
        const v = L.bytes[off + r * L.cols + k];
        g.fillStyle = v === 255 ? "#bfbfbf" : L.colors[v];
        g.fillRect(left + k * cw, top + h - (r + 1) * ch, Math.ceil(cw), Math.ceil(ch));
      }
    }
    g.fillStyle = ink; g.strokeStyle = muted; g.lineWidth = 1;
    g.font = "11px system-ui, sans-serif"; g.textAlign = "center"; g.textBaseline = "top";
    L.xticks.forEach(([p, s]) => { const x = left + (p + 0.5) * cw; g.fillText(s, x, top + h + 4); });
    g.fillText(L.xlabel, left + w / 2, top + h + 18);
    g.textAlign = "right"; g.textBaseline = "middle";
    L.yticks.forEach(([p, s]) => { const y = top + h - (p + 0.5) * ch; g.fillText(s, left - 4, y); });
    g.save(); g.translate(9, top + h / 2); g.rotate(-Math.PI / 2); g.textAlign = "center";
    g.fillText(L.ylabel, 0, 0); g.restore();
    g.textAlign = "left"; g.textBaseline = "top";
    g.fillText(L.title + " at " + (f * L.dt).toFixed(2) + " s", left, 4);
    const bx = left + w + 8, bw = 8;  // the colour bar
    for (let i = 0; i < 255; i++) { g.fillStyle = L.colors[i]; g.fillRect(bx, top + h - (i + 1) * h / 255, bw, Math.ceil(h / 255)); }
    g.fillStyle = ink; g.textBaseline = "middle";
    g.fillText(L.range[1] + "", bx + bw + 2, top + 4); g.fillText(L.range[0] + "", bx + bw + 2, top + h - 4);
    g.fillText("dB", bx + bw + 2, top + h / 2);
  }
  // A Lissajous figure of the two notes sounding at the playhead, as two sine waves at their
  // fundamentals with phases counted from the notes' start: x the first, y the second, over a
  // short stretch. Before the sound has played, the notes at figure.start. For the first
  // tenth of a second of a note that follows another without a gap, the curve morphs from the
  // last note's figure to the new one, point by point, rather than jumping.
  function drawLissajous(item) {
    const c = item.figureCanvas;
    if (!c) return;
    const F = item.figure, dpr = window.devicePixelRatio || 1, size = c.clientWidth;
    if (c.width !== Math.round(size * dpr)) { c.width = c.height = Math.round(size * dpr); }
    const g = c.getContext("2d"), css = getComputedStyle(document.documentElement);
    const ink = css.getPropertyValue("--ink").trim(), muted = css.getPropertyValue("--muted").trim();
    const accent = css.getPropertyValue("--accent").trim() || ink;
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, size, size);
    const t = F.played || item.audio.currentTime > 0 ? item.audio.currentTime : F.start;
    const index = F.notes.findIndex((n) => t >= n[0] && t < n[1]), note = F.notes[index];
    const cx = size / 2, cy = size / 2 + 8, R = size * 0.4;
    g.strokeStyle = muted; g.lineWidth = 1; g.strokeRect(cx - R, cy - R, 2 * R, 2 * R);
    g.fillStyle = ink; g.font = "11px system-ui, sans-serif"; g.textAlign = "left"; g.textBaseline = "top";
    if (!note) { g.fillText(F.title, 2, 2); return; }
    const before = index > 0 && Math.abs(F.notes[index - 1][1] - note[0]) < 0.02 ? F.notes[index - 1] : null;
    const morph = 0.1, w = before ? Math.min(1, (t - note[0]) / morph) : 1, mix = w * w * (3 - 2 * w);
    const label = note[4], steps = 600;
    g.fillText(label || F.title, 2, 2);
    g.strokeStyle = accent; g.lineWidth = 1.5; g.beginPath();
    const at = (n, s) => { const tau = t - n[0] + s;
      return [Math.sin(2 * Math.PI * n[2] * tau), Math.sin(2 * Math.PI * n[3] * tau)]; };
    for (let i = 0; i <= steps; i++) {
      const s = (i / steps) * F.window, [xn, yn] = at(note, s);
      const [xb, yb] = mix < 1 ? at(before, s) : [xn, yn];
      const x = cx + R * (xb + mix * (xn - xb)), y = cy - R * (yb + mix * (yn - yb));
      i ? g.lineTo(x, y) : g.moveTo(x, y);
    }
    g.stroke();
    g.fillStyle = muted; g.textAlign = "center"; g.textBaseline = "top";
    g.fillText(F.xlabel, cx, cy + R + 4);
    g.save(); g.translate(cx - R - 6, cy); g.rotate(-Math.PI / 2); g.textBaseline = "bottom";
    g.fillText(F.ylabel, 0, 0); g.restore();
  }
  function draw(item) {
    drawScene(item);
    drawLive(item);
    drawLissajous(item);
    const t = item.audio.currentTime;
    item.regions.forEach((r, i) => {
      const line = item.lines[i];
      if (t < r.t0 || t > r.t1) { line.style.display = "none"; return; }
      line.style.left = ((r.x0 + (t - r.t0) / (r.t1 - r.t0) * (r.x1 - r.x0)) * 100) + "%";
      line.style.display = "block";
    });
  }
  let active = null;
  function loop() { if (!active) return; draw(active); if (!active.audio.paused) requestAnimationFrame(loop); }
  items.forEach((item) => {
    drawScene(item);
    if (item.liveCanvas) {
      drawLive(item);
      new ResizeObserver(() => drawLive(item)).observe(item.liveCanvas);
      window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => drawLive(item));
    }
    if (item.figureCanvas) {
      drawLissajous(item);
      new ResizeObserver(() => drawLissajous(item)).observe(item.figureCanvas);
      window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => drawLissajous(item));
    }
    if (item.canvas) {
      new ResizeObserver(() => drawScene(item)).observe(item.canvas);
      window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => drawScene(item));
    }
    item.audio.addEventListener("play", () => {
      items.forEach((o) => { if (o !== item && !o.audio.paused) o.audio.pause(); });
      if (item.live) item.live.played = true;
      if (item.figure) item.figure.played = true;
      active = item; requestAnimationFrame(loop);
    });
    item.audio.addEventListener("seeked", () => draw(item));
    item.audio.addEventListener("ended", () => item.lines.forEach((l) => (l.style.display = "none")));
    item.plate.tabIndex = 0;
    item.plate.setAttribute("role", "button");
    item.plate.setAttribute("aria-label", "Play or pause; click a time axis to play from that point");
    item.plate.addEventListener("click", (e) => {
      const box = item.plate.getBoundingClientRect();
      const fx = (e.clientX - box.left) / box.width, fy = (e.clientY - box.top) / box.height;
      const r = item.regions.find((r) => fx >= r.x0 && fx <= r.x1 && fy >= r.top && fy <= r.bottom);
      if (r) { item.audio.currentTime = Math.max(0, r.t0 + (fx - r.x0) / (r.t1 - r.t0) * (r.t1 - r.t0)); item.audio.play(); }
    });
    item.plate.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") { e.preventDefault(); item.audio.paused ? item.audio.play() : item.audio.pause(); }
    });
  });
})();
"""

# TeX and code highlighting are drawn in the browser; without scripts the TeX source and plain code show.
KATEX = "https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/"
HLJS = "https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/"
EXAMPLE_HEAD = f"""<link rel="stylesheet" href="{KATEX}katex.min.css" crossorigin="anonymous">
<script>function renderTex() {{ document.querySelectorAll(".tex").forEach((el) => katex.render(el.textContent, el,
  {{ displayMode: el.classList.contains("display"), throwOnError: false }})); }}</script>
<script defer src="{KATEX}katex.min.js" crossorigin="anonymous" onload="renderTex()"></script>
<script defer src="{HLJS}highlight.min.js" onload="hljs.highlightAll()"></script>"""

TITLES = {
    "index.html": "Listening gallery",
    "classic.html": "Classic stimuli",
    "irn.html": "Iterated rippled noise",
    "ripples.html": "Spectrotemporal ripples",
    "binaural.html": "Binaural cues",
    "textures.html": "Sound textures",
    "speech.html": "Seeing speech",
    "resynthesis.html": "Analysis and resynthesis",
    "cepstrum.html": "Cepstral analysis",
    "modspectrogram.html": "Modulation spectrogram",
    "modtargets.html": "Hearing a modulation spectrum",
    "pv.html": "Phase vocoder",
    "harmonics.html": "Voices from harmonics",
    "formants.html": "Formant synthesis",
    "aperiodicity.html": "Source, filter and aperiodicity",
    "voice.html": "Changing a voice",
    "vocoder.html": "Hearing through a vocoder",
    "reverb.html": "Synthetic reverberation",
    "moving.html": "Moving talkers",
    "timbre.html": "Timbre",
    "temperament.html": "Tuning and temperament",
    "organ.html": "The pipe organ",
}

# The topic pages in groups, for the index and the menus at the top of every page. Each
# group's scripts live in their own folder of docs/gallery.
# Groups and the pages within them run from simple to elaborate, roughly up sonore's layers:
# stimuli from plain generators (signals) to binaural cues (stimuli) and textures (texture);
# analysis from one frame (the STFT) to views built on it, a phase vocoder that changes the
# sound, and a voice rebuilt from its F0 track and envelope; then whole listening scenes.
TOPICS = [
    (
        "Stimuli",
        "stimuli",
        [
            (
                "classic.html",
                "speech-shaped noise, beats and roughness, binaural beats, tone sequences, band-limited waveforms.",
            ),
            ("irn.html", "a pitch made from noise and a delay."),
            ("ripples.html", "sounds defined by a moving pattern of modulation."),
            ("textures.html", "recordings and their syntheses from statistics."),
        ],
    ),
    (
        "Seeing and changing sound",
        "seeing",
        [
            ("speech.html", "a short course in time-frequency analysis on one spoken sentence."),
            ("resynthesis.html", "a filterbank that reconstructs exactly, and spectrogram masking."),
            ("vocoder.html", "a simulation of cochlear-implant hearing."),
            ("modspectrogram.html", "how fast and how deeply each band's envelope moves, moment by moment."),
            (
                "modtargets.html",
                "sounds made from a modulation spectrum: measured, edited, drawn, or traded between sounds.",
            ),
            ("pv.html", "how it works, and duration, pitch and partials changed independently."),
        ],
    ),
    (
        "Voices",
        "voice",
        [
            ("formants.html", "vowels and consonants written as a source, formants and a few numbers."),
            ("cepstrum.html", "separating a voice's pitch from its timbre."),
            ("harmonics.html", "a voice rebuilt from its pitch track and spectral envelope, and changed."),
            (
                "aperiodicity.html",
                "how much of a voice is noise, frequency by frequency, and WORLD's resynthesis.",
            ),
            ("voice.html", "pitch and formants moved separately, with any pitch track and any envelope."),
        ],
    ),
    (
        "Spatial hearing",
        "spatial",
        [
            ("binaural.html", "differences between the ears: timing, and correlation that changes."),
            ("reverb.html", "rooms built from the statistics of real ones, and rooms that break them."),
            ("moving.html", "three talkers rendered through measured HRIRs, one of them moving."),
        ],
    ),
    (
        "Music",
        "music",
        [
            ("timbre.html", "attack time, brightness and spectral flux, heard and measured."),
            (
                "temperament.html",
                "just intonation, equal temperament and others, heard and drawn as Lissajous figures.",
            ),
            ("organ.html", "organ stops as additive synthesis (a template, sounds to come)."),
        ],
    ),
]

# GitHub's mark (Octicons, MIT licence), so the repository link reads as a link out, not a page.
GITHUB_MARK = (
    '<svg viewBox="0 0 16 16" width="22" height="22" aria-hidden="true"><path fill="currentColor" d="M8 0C3.58 0 '
    "0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23"
    "-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78"
    "-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 "
    "1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29"
    '.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"/></svg>'
)

# One menu open at a time; a click elsewhere or Escape closes it. With a mouse, a menu also
# opens on hover and closes shortly after the pointer leaves; a click on its name then keeps it
# open rather than closing it. Touch screens open menus with a tap.
NAV_JS = """(() => { const menus = [...document.querySelectorAll("nav.pages details")];
  menus.forEach((m) => m.addEventListener("toggle", () => { if (m.open) menus.forEach((o) => { if (o !== m) o.open = false; }); }));
  document.addEventListener("click", (e) => menus.forEach((m) => { if (!m.contains(e.target)) m.open = false; }));
  if (matchMedia("(hover: hover)").matches) menus.forEach((m) => { let t;
    m.addEventListener("mouseenter", () => { clearTimeout(t); m.open = true; });
    m.addEventListener("mouseleave", () => { t = setTimeout(() => { m.open = false; }, 200); });
    m.querySelector("summary").addEventListener("click", (e) => { if (e.detail && m.open) e.preventDefault(); }); });
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") menus.forEach((m) => { if (m.open) { m.open = false; m.querySelector("summary").focus(); } }); });
})();"""


# The sidebar folds to a narrow strip, with a short slide and fade either way, and each of
# its groups slides shut on its own, as does the list of the current page's sections (folded at
# first); all three are remembered in the browser between pages. The group holding the page being read always
# starts open. While the sidebar is folded, the menus at the top of the page come back.
# SIDE_HEAD_JS runs in <head>, so a folded sidebar does not flash open as the page loads.
SIDE_HEAD_JS = (
    'try { if (localStorage.getItem("sonore-side") === "closed") '
    'document.documentElement.classList.add("side-closed"); } catch (e) {}'
)
SIDE_JS = """(() => { const root = document.documentElement, side = document.currentScript.parentElement;
  const toggle = side.querySelector(".side-toggle"), groups = [...side.querySelectorAll("details.group")];
  const save = (key, value) => { try { localStorage.setItem(key, value); } catch (e) {} };
  const show = (closed) => { root.classList.toggle("side-closed", closed); toggle.setAttribute("aria-expanded", String(!closed));
    const label = closed ? "Show the menu" : "Hide the menu"; toggle.setAttribute("aria-label", label); toggle.title = label;
    toggle.innerHTML = closed ? "&raquo;" : "&laquo;"; };
  show(root.classList.contains("side-closed"));
  // Opening widens the strip as the menu slides in; closing fades the menu out, then narrows the strip.
  const still = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const after = (ms, then) => (still ? then() : setTimeout(then, ms));
  toggle.addEventListener("click", () => { const closed = !root.classList.contains("side-closed");
    save("sonore-side", closed ? "closed" : "open");
    if (closed) { root.classList.add("side-closing");
      after(150, () => { root.classList.remove("side-closing"); show(true); }); }
    else { show(false); root.classList.add("side-opening"); after(250, () => root.classList.remove("side-opening")); } });
  // The page's own sections fold under its title, folded unless opened before.
  const sectionsToggle = side.querySelector(".sections-toggle"), sections = side.querySelector(".sections");
  if (sectionsToggle) { const showSections = (open) => { sections.classList.toggle("open", open);
      sectionsToggle.setAttribute("aria-expanded", String(open));
      const label = open ? "Hide this page's sections" : "Show this page's sections";
      sectionsToggle.setAttribute("aria-label", label); sectionsToggle.title = label; };
    let open = false; try { open = localStorage.getItem("sonore-side-sections") === "open"; } catch (e) {}
    sections.style.transition = "none"; showSections(open); sections.offsetHeight; sections.style.transition = "";
    sectionsToggle.addEventListener("click", () => { const now = !sections.classList.contains("open"); showSections(now);
      save("sonore-side-sections", now ? "open" : "closed"); }); }
  let folded = []; try { folded = JSON.parse(localStorage.getItem("sonore-side-groups") || "[]"); } catch (e) {}
  groups.forEach((group) => { const name = group.dataset.group;
    if (folded.includes(name) && !group.classList.contains("here")) group.open = false;
    group.addEventListener("toggle", () => { folded = folded.filter((other) => other !== name);
      if (!group.open) folded.push(name); save("sonore-side-groups", JSON.stringify(folded)); });
    // A group's pages slide open and shut rather than appearing at once.
    const list = group.querySelector("ul"); let running = null;
    group.querySelector("summary").addEventListener("click", (event) => { if (still) return;
      event.preventDefault(); const opening = running ? !running.opening : !group.open;
      if (running) running.cancel(); if (opening) group.open = true;
      const full = list.scrollHeight + "px";
      running = list.animate(opening ? [{ height: "0px", opacity: 0 }, { height: full, opacity: 1 }]
        : [{ height: full, opacity: 1 }, { height: "0px", opacity: 0 }], { duration: 200, easing: "ease" });
      running.opening = opening; list.style.overflow = "hidden";
      running.onfinish = () => { running = null; list.style.overflow = ""; if (!opening) group.open = false; };
      running.oncancel = () => { list.style.overflow = ""; }; }); });
})();"""


def nav(current: str) -> str:
    def link(href: str) -> str:
        here = ' aria-current="page"' if href == current else ""
        return f'<a href="{href}"{here}>{TITLES[href]}</a>'

    parts = [link("index.html")]
    for group, _, pages in TOPICS:
        hrefs = [href for href, _ in pages]
        here = ' class="here"' if current in hrefs else ""
        items = "".join(f"<li>{link(href)}</li>" for href in hrefs)
        parts.append(f"<details{here}><summary>{html.escape(group)}</summary><ul>{items}</ul></details>")
    parts.append('<a class="ref" href="../api/">API reference</a>')
    parts.append(
        '<a class="repo" href="https://github.com/choyun1/sonore" title="sonore on GitHub" '
        f'aria-label="sonore on GitHub">{GITHUB_MARK}</a>'
    )
    return f'<nav class="pages" aria-label="Gallery pages">{"".join(parts)}<script>{NAV_JS}</script></nav>'


def sidebar(current: str, sections_html: str) -> str:
    """The menu down the left of a wide screen: every page by group, and under the page
    being read, its sections. The menu and each group fold away (SIDE_JS). Narrow screens keep only the menus at the top."""

    def link(href: str) -> str:
        here = ' aria-current="page"' if href == current else ""
        item = f'<a href="{href}"{here}>{TITLES[href]}</a>'
        if href == current:
            sections = re.findall(r'<h2 id="([^"]+)">(.*?)</h2>', sections_html)
            if sections:
                item += (
                    '<button class="sections-toggle" type="button" aria-expanded="false" '
                    'aria-label="Show this page\'s sections" title="Show this page\'s sections"></button>'
                    '<div class="sections"><div><ul>'
                    + "".join(f'<li><a href="#{slug}">{title}</a></li>' for slug, title in sections)
                    + "</ul></div></div>"
                )
                return f'<li class="has-sections">{item}</li>'
        return f"<li>{item}</li>"

    here = ' aria-current="page"' if current == "index.html" else ""
    top = (
        f'<div class="side-top"><a class="home" href="index.html"{here}>{TITLES["index.html"]}</a>'
        '<button class="side-toggle" type="button" aria-expanded="true" aria-label="Hide the menu" '
        'title="Hide the menu">&laquo;</button></div>'
    )
    groups = []
    for group, folder, pages in TOPICS:
        is_current_group = current in [href for href, _ in pages]
        css_class = "group here" if is_current_group else "group"
        groups.append(
            f'<details class="{css_class}" data-group="{folder}" open><summary>{html.escape(group)}</summary>'
            f"<ul>{''.join(link(href) for href, _ in pages)}</ul></details>"
        )
    body = f'<div class="side-body">{"".join(groups)}</div>'
    return f'<aside class="side" aria-label="All gallery pages">{top}{body}<script>{SIDE_JS}</script></aside>'


def how(binaural: bool) -> str:
    """The note on playing a page's sounds; the lossless remark only where it matters."""
    lossless = " The audio is lossless, since compression would alter the binaural sounds." if binaural else ""
    return f"""<p class="how">Press play and a line follows the sound across every time axis in its plots. Click any time
  axis to play from that point.{lossless} Start with your volume low.</p>"""


def page(title: str, current: str, header: str, sections_html: str, head: str = "", footer: str = "") -> str:
    fonts = (
        '<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" '
        'href="https://fonts.gstatic.com" crossorigin><link href="https://fonts.googleapis.com/css2?'
        'family=Atkinson+Hyperlegible:wght@400;700&family=Spectral:wght@400;500;600&display=swap" '
        'rel="stylesheet">'
    )
    footer = footer or (
        "Each page is a script in <code>docs/gallery</code> of the "
        '<a href="https://github.com/choyun1/sonore">sonore repository</a>, run by <code>docs/gallery/build.py</code>.'
    )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>{html.escape(title)}</title>
{fonts}
{head}
<script>{SIDE_HEAD_JS}</script>
<style>{CSS}</style>
</head>
<body>
<div class="layout">
{sidebar(current, sections_html)}
<main>
{nav(current)}
<header>
  <h1>{html.escape(title)}</h1>
{header}
</header>
{sections_html}
<footer>{footer}</footer>
</main>
</div>
<script>{JS}</script>
</body>
</html>
"""


def scene_json(scene: list[dict], rate: float = 50.0) -> str:
    """A top-down scene for the page to animate: each source's azimuth [deg, clockwise
    from straight ahead] and, optionally, ``distance`` [m, 1 if not given] over time,
    resampled to ``rate`` Hz to keep the page small."""
    out = []
    for src in scene:
        t = np.asarray(src["t"], float)
        grid = np.arange(0, t[-1] + 0.5 / rate, 1 / rate) if len(t) > 1 else t
        az = np.interp(grid, t, np.asarray(src["azimuth"], float))
        entry = {
            "label": src["label"],
            "color": src["color"],
            "t": round(1 / rate, 6) if len(t) > 1 else 0,
            "az": [round(float(a), 2) for a in az],
        }
        if "distance" in src:
            distance = np.interp(grid, t, np.broadcast_to(np.asarray(src["distance"], float), t.shape))
            entry["r"] = [round(float(d), 3) for d in distance]
        out.append(entry)
    return json.dumps(out, separators=(",", ":"))


LIVE_TICKS = [0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]


def live_json(live: dict, rate: float = 20.0) -> str:
    """An image for the page to show frame by frame as the sound plays.

    ``live`` holds ``t`` (frame times [s]), ``image`` (rows x columns x frames, NaN drawn
    grey), ``x`` and ``y`` (the column and row centres, log-spaced), ``range`` (the values
    at the two ends of the colour map ``cmap``), ``xlabel``, ``ylabel`` and ``title``, and
    optionally ``start``, the time shown before the sound plays. The frames are resampled
    to ``rate`` Hz and quantised to 255 levels to keep the page small."""
    t = np.asarray(live["t"], float)
    image = np.asarray(live["image"], float)
    grid = np.arange(0.0, t[-1] + 0.5 / rate, 1 / rate)
    idx = np.clip(np.searchsorted(t, grid), 0, len(t) - 1)
    frames = np.moveaxis(image[:, :, idx], -1, 0)  # (n, rows, columns)
    lo, hi = live["range"]
    q = np.clip(np.round((frames - lo) / (hi - lo) * 254), 0, 254)
    q = np.where(np.isfinite(frames), q, 255).astype(np.uint8)
    cmap = matplotlib.colormaps[live.get("cmap", "magma")]
    colors = [matplotlib.colors.to_hex(cmap(i / 254)) for i in range(255)]

    def ticks(centres):  # (fractional cell position, label) for round values inside the axis
        lc = np.log(np.asarray(centres, float))
        return [
            [round(float(np.interp(np.log(v), lc, np.arange(len(lc)))), 3), f"{v:g}"]
            for v in LIVE_TICKS
            if centres[0] <= v <= centres[-1]
        ]

    return json.dumps(
        {
            "dt": 1 / rate,
            "rows": frames.shape[1],
            "cols": frames.shape[2],
            "data": base64.b64encode(q.tobytes()).decode(),
            "colors": colors,
            "range": [round(float(lo)), round(float(hi))],
            "xticks": ticks(live["x"]),
            "yticks": ticks(live["y"]),
            "xlabel": live["xlabel"],
            "ylabel": live["ylabel"],
            "title": live["title"],
            "start": float(live.get("start", 0.0)),
        },
        separators=(",", ":"),
    )


def lissajous_json(lissajous: dict) -> str:
    """Two notes for the page to draw against each other as the sound plays.

    ``lissajous`` holds ``notes``, rows of (start [s], end [s], x frequency [Hz], y frequency
    [Hz], label), each pair drawn as sine waves whose phases count from the start; ``window``,
    the stretch drawn [s]; ``xlabel``, ``ylabel`` and ``title``; and optionally ``start``, the
    time shown before the sound plays."""
    notes = [
        [round(float(t0), 4), round(float(t1), 4), round(float(fx), 4), round(float(fy), 4), str(label)]
        for t0, t1, fx, fy, label in lissajous["notes"]
    ]
    return json.dumps(
        {
            "notes": notes,
            "window": float(lissajous.get("window", 0.03)),
            "xlabel": lissajous["xlabel"],
            "ylabel": lissajous["ylabel"],
            "title": lissajous["title"],
            "start": float(lissajous.get("start", 0.0)),
        },
        separators=(",", ":"),
    )


HEADPHONES = "Headphones required for binaural sounds"


def sound_article(
    key,
    title,
    desc_html,
    snd,
    audio_src,
    img_src,
    regions,
    size,
    extra="",
    code="",
    scene=None,
    live=None,
    lissajous=None,
) -> str:
    name, (w, h) = file_name(key, title), size
    scene = (
        f'\n    <canvas class="scene" data-scene="{html.escape(scene)}" role="img" '
        f'aria-label="The sources seen from above, moving as the sound plays"></canvas>'
        if scene
        else ""
    )
    if live:
        scene += (
            f'\n    <canvas class="live" data-live="{html.escape(live)}" role="img" '
            f'aria-label="An image that changes with the sound as it plays"></canvas>'
        )
    if lissajous:
        scene += (
            f'\n    <canvas class="lissajous" data-lissajous="{html.escape(lissajous)}" role="img" '
            f'aria-label="A Lissajous figure of two notes, changing as the sound plays"></canvas>'
        )
    chan = "stereo" if snd.n_channels == 2 else "mono"
    # Every two-channel demo is binaural: its effect is lost over speakers.
    phones = (
        f' <span class="headphones" title="{HEADPHONES}" role="img" aria-label="{HEADPHONES}">🎧</span>'
        if chan == "stereo"
        else ""
    )
    code = f'\n  <details class="code"><summary>Code</summary>{code}</details>' if code else ""
    return f"""
<article class="sound" id="d-{key}" data-regions="{html.escape(json.dumps(regions))}">
  <div class="about">
    <h3>{html.escape(title)}{phones}</h3>{extra}
    {desc_html}
    <audio controls preload="metadata" src="{audio_src}"></audio>
    <p class="file">{html.escape(name)}.flac, {snd.duration:.1f} s, {chan}</p>{scene}
  </div>
  <figure class="plot"><div class="plate">
    <img src="{img_src}" width="{w}" height="{h}" alt="Plots of the {html.escape(title.lower())} sound" loading="lazy">
    <div class="heads" aria-hidden="true"></div>
  </div></figure>{code}
</article>"""


def still_article(key, title, desc_html, img_src, size, code) -> str:
    w, h = size
    return f"""
<article class="still" id="d-{key}">
  <div class="about">
    <h3>{html.escape(title)}</h3>
    <div class="desc">{desc_html}</div>
  </div>
  <figure class="plot"><div class="plate">
    <img src="{img_src}" width="{w}" height="{h}" alt="{html.escape(title)}" loading="lazy">
  </div></figure>
  <details class="code"><summary>Code</summary>{code}</details>
</article>"""


class Site:
    """Writes media next to the pages (relative links) and/or inlines it (one self-contained file per page)."""

    def __init__(self, out_dir: Path | None, single: Path | None):
        self.out_dir, self.single = out_dir, single
        if out_dir:
            (out_dir / "audio").mkdir(parents=True, exist_ok=True)
            (out_dir / "img").mkdir(parents=True, exist_ok=True)

    def media(self, name: str, audio: bytes | None, png: bytes) -> tuple[list, list]:
        """(audio, image) sources for the site and for the single file."""
        rel, inl = [], []
        if self.out_dir:
            if audio is not None:
                (self.out_dir / "audio" / f"{name}.flac").write_bytes(audio)
                rel.append(f"audio/{name}.flac")
            (self.out_dir / "img" / f"{name}.png").write_bytes(png)
            rel.append(f"img/{name}.png")
        if self.single:
            if audio is not None:
                inl.append("data:audio/flac;base64," + base64.b64encode(audio).decode())
            inl.append("data:image/png;base64," + base64.b64encode(png).decode())
        return rel, inl

    def write(self, file: str, make) -> None:
        """``make(variant)`` renders a page; variant 0 links media, 1 inlines it."""
        if self.out_dir:
            (self.out_dir / file).write_text(make(0))
        if self.single:
            stem = self.single.stem if file == "index.html" else f"{self.single.stem}-{Path(file).stem}"
            self.single.with_name(stem + self.single.suffix).write_text(make(1))


def section_html(title_html: str, parts: list[str], intro: str = "") -> str:
    # Runs of spaces and punctuation become one hyphen, so "Speech, babble" is
    # h-speech-babble, the slug a reader writes by hand in a topic link.
    text = html.unescape(re.sub("<[^>]+>", "", title_html)).lower()
    slug = "h-" + re.sub(r"[^0-9a-z]+", "-", text).strip("-")
    out = [f'<section aria-labelledby="{slug}">', f'<h2 id="{slug}">{title_html}</h2>']
    if intro:
        out.append(f'<p class="section-intro">{html.escape(intro)}</p>')
    return "\n".join(out + parts + ["</section>"])


def script_path(name: str) -> str:
    """Where a page's script lives, relative to the repository root."""
    folder = next(folder for _, folder, pages in TOPICS if f"{name}.html" in dict(pages))
    return f"docs/gallery/{folder}/{name}.py"


def build_example_page(site: Site, name: str) -> list[str]:
    """Build one example page; returns the keys of its examples."""
    script = script_path(name)
    content = example_page(ROOT / script)
    keys, rendered = [], {}  # part id -> (linked, inline) HTML
    for part in [p for s in content["sections"] for p in s["parts"]] + content["intro"]:
        if not isinstance(part, dict):
            continue
        key, title = part["key"], part["title"]
        keys.append(key)
        srcs = site.media(file_name(key, title), part.get("audio"), part["png"])
        variants = []
        for src in srcs:
            if not src:
                variants.append("")
            elif "sound" in part:
                desc = f'<div class="desc">{part["about"]}</div>'
                variants.append(
                    sound_article(
                        key,
                        title,
                        desc,
                        part["sound"],
                        src[0],
                        src[1],
                        part["regions"],
                        part["size"],
                        code=part["code"],
                        scene=part.get("scene"),
                        live=part.get("live"),
                        lissajous=part.get("lissajous"),
                    )
                )
            else:
                variants.append(still_article(key, title, part["about"], src[0], part["size"], part["code"]))
        rendered[id(part)] = variants
        print(f"  {name}: {file_name(key, title)}")

    # Every two-channel demo is binaural, as in sound_article.
    binaural = any(
        isinstance(p, dict) and "sound" in p and p["sound"].n_channels == 2
        for p in [p for s in content["sections"] for p in s["parts"]] + content["intro"]
    )

    def make(variant):
        def show(p):
            return rendered[id(p)][variant] if isinstance(p, dict) else p

        header = "\n".join(show(p) for p in content["intro"]) + "\n" + how(binaural)
        sections = "\n".join(
            section_html(s["title"], [show(p) for p in s["parts"]]) for s in content["sections"]
        )
        footer = (
            f"This page is the script <code>{script}</code> in the "
            '<a href="https://github.com/choyun1/sonore">sonore repository</a>, run cell by cell by '
            "<code>docs/gallery/build.py</code>: each block of code is shown exactly as it ran. Run it yourself "
            f"from the repository root with <code>python {script}</code>, or a cell at a time."
        )
        return page(content["title"], f"{name}.html", header, sections, EXAMPLE_HEAD, footer)

    site.write(f"{name}.html", make)
    return keys


# Examples that were on the index before it became a list of pages. 15 (a noise-vocoded
# syllabic tone) was dropped: the vocoder page covers noise vocoding, on speech.
RETIRED = {"15": "vocoder.html"}


def build(out_dir: Path | None, single: Path | None) -> None:
    site = Site(out_dir, single)
    moved = dict(RETIRED)
    for name in EXAMPLE_PAGES:
        moved.update({key: f"{name}.html" for key in build_example_page(site, name)})

    # Links to examples that moved to their own pages still land on them.
    redirect = (
        f"<script>(() => {{ const moved = {json.dumps(moved)}; const k = location.hash.slice(3);"
        ' if (location.hash.startsWith("#d-") && moved[k]) location.replace(moved[k] + location.hash); })();</script>'
    )
    header = """  <p>Sounds made with <a href="https://github.com/choyun1/sonore">sonore</a>, each beside plots of the
  same audio you hear and the code that made it. Every topic has a page of its own.</p>"""
    sections = "\n".join(
        section_html(
            html.escape(group),
            [
                "<ul>"
                + "".join(
                    f'<li><a href="{href}">{TITLES[href]}</a>: {html.escape(text)}</li>'
                    for href, text in pages
                )
                + "</ul>"
            ],
        )
        for group, _, pages in TOPICS
    )
    site.write("index.html", lambda v: page("Listening to sonore", "index.html", header, sections, redirect))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=HERE, help="directory for the Pages site")
    ap.add_argument("--single", type=Path, default=None, help="also write self-contained HTML files")
    ap.add_argument("--no-site", action="store_true")
    args = ap.parse_args()
    build(None if args.no_site else args.out, args.single)
