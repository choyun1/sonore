"""Draw sonore's logo as SVG files in docs/logo/.

The logo is three lines and a patch of noise, an abstract picture of the spectrogram of
"sonore" said by the formant synthesizer (the aside at the end of the Formant synthesis page in
the gallery): vertical strokes for the s, a middle line that rises and falls once for the
n, and a top line that steps down once for the r. It is not a plot of that sound.

The wordmark is set in Spectral SemiBold (SIL Open Font License), the gallery's serif, and
written into the files as outlines, so they need no font. Run from the repository root with the
font file's path:

    python docs/logo/make_logo.py path/to/Spectral-SemiBold.ttf

(the font is at https://fonts.google.com/specimen/Spectral).
"""

import sys
from pathlib import Path

from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen
from fontTools.ttLib import TTFont

HERE = Path(__file__).parent
# the gallery's ink and accent colors, light and dark
ROW = 24  # spacing of the three lines
# the small mark's middle and bottom rows: the middle one lower, so the n and r steps can be
# deeper and survive at 16 px, and the bottom one close under it, since the bottom line has no step
SMALL_ROWS = (32, 52)
# one gray for the s strokes, the same on light and dark backgrounds
NOISE_GRAY = "#7D8891"
COLORS = {"light": ("#16202A", "#235B7C"), "dark": ("#E4E9ED", "#7FB6D6")}


def lines(ink, accent, small=False):
    """The three lines (rows at 0, ROW and 2 ROW, or at 0 and SMALL_ROWS) and the strokes for the
    s, as SVG elements. The strokes run most of the height, since the noise of an s spreads over
    most frequencies, and sit close to where the lines start, as the s runs into the vowel. Each
    stroke is given as its x and how far it is moved down; the middle one sits a little higher and
    the right one a little lower, so they read as noise. The small version is shorter, with
    thinner lines and deeper steps, so the n and the r still show at 16 px."""
    row, low = SMALL_ROWS if small else (ROW, 2 * ROW)
    if small:
        width, (rise, back), (drop, land), stroke, s_stroke, step = 60, (2, 40), (42, 56), 11, 5, 16
        strokes = [(-27, 0), (-20, -4), (-13, 4)]
    else:
        width, (rise, back), (drop, land), stroke, s_stroke, step = 105, (16, 62), (72, 84), 10, 4.5, 10
        strokes = [(-25, 0), (-18, -5), (-11, 5)]
    # step is how far the middle line rises for the n and the top line drops for the r
    reach = (stroke - s_stroke) / 2 - 11  # the strokes stop well inside the top and bottom lines
    top = f"M0,0 H{drop - 6} C{drop},0 {land - 6},{step} {land},{step} H{width}"
    peak = row - step
    middle = (
        f"M0,{row} H{rise} C{rise + 8},{row} {rise + 8},{peak} {rise + 16},{peak} H{back - 16} "
        f"C{back - 8},{peak} {back - 8},{row} {back},{row} H{width}"
    )
    bottom = f"M0,{low} H{width}"
    line = f'fill="none" stroke-width="{stroke}" stroke-linecap="round" stroke-linejoin="round"'
    s_line = f'stroke="{NOISE_GRAY}" stroke-width="{s_stroke}" stroke-linecap="round"'
    s = "".join(f'<path d="M{x},{down - reach:g} V{low + reach + down:g}" {s_line}/>' for x, down in strokes)
    return (
        s + f'<path d="{top}" stroke="{accent}" {line}/><path d="{middle}" stroke="{ink}" {line}/>'
        f'<path d="{bottom}" stroke="{ink}" {line}/>'
    )


def wordmark(font, text, size, x, baseline, fill):
    """`text` as glyph outlines, `size` units per em, starting at `x` on `baseline`."""
    glyphs, cmap = font.getGlyphSet(), font.getBestCmap()
    scale = size / font["head"].unitsPerEm
    paths = []
    for char in text:
        name = cmap[ord(char)]
        pen = SVGPathPen(glyphs)
        glyphs[name].draw(TransformPen(pen, (scale, 0, 0, -scale, x, baseline)))  # font y points up
        paths.append(pen.getCommands())
        x += glyphs[name].width * scale
    return f'<path d="{" ".join(paths)}" fill="{fill}"/>', x


def banner(font, theme):
    """The lines beside the wordmark, whose x-height is centered on the middle row."""
    ink, accent = COLORS[theme]
    size = 98
    baseline = ROW + font["OS/2"].sxHeight / font["head"].unitsPerEm * size / 2
    word, right = wordmark(font, "sonore", size, 140, baseline, ink)
    left, top, bottom = -33, ROW - 54, ROW + 44
    view = f"{left} {top} {right - left + 6:.0f} {bottom - top}"
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{view}" role="img" aria-label="sonore">'
        f"<title>sonore</title>{lines(ink, accent)}{word}</svg>\n"
    )


def mark():
    """The small mark, which follows the browser's light or dark setting (as a favicon does). The
    gallery sidebar inlines it and colors the two classes itself."""
    (ink, accent), (dark_ink, dark_accent) = COLORS["light"], COLORS["dark"]
    dark = f".logo-ink{{stroke:{dark_ink}}}.logo-accent{{stroke:{dark_accent}}}"
    style = (
        f"<style>.logo-ink{{stroke:{ink}}}.logo-accent{{stroke:{accent}}}"
        f"@media (prefers-color-scheme: dark){{{dark}}}</style>"
    )
    body = lines("INK", "ACCENT", small=True)
    body = body.replace('stroke="INK"', 'class="logo-ink"').replace('stroke="ACCENT"', 'class="logo-accent"')
    width = 60  # the small lines' width, as in lines()
    size = width + 44  # square, with room for the s strokes on the left
    view = f"-36 {SMALL_ROWS[1] / 2 - size / 2:g} {size} {size}"  # centered on the lines
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{view}" role="img" aria-label="sonore">'
    return f"{svg}<title>sonore</title>{style}{body}</svg>\n"


if __name__ == "__main__":
    font = TTFont(sys.argv[1])
    for theme in COLORS:
        (HERE / f"sonore-banner-{theme}.svg").write_text(banner(font, theme))
    (HERE / "sonore-mark.svg").write_text(mark())
