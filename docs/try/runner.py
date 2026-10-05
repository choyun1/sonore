"""Runs a code cell for the landing page and returns what it shows.

The page's worker loads this file into Pyodide; tests/test_landing_page.py runs
it under CPython. It keeps one namespace across runs, as a notebook kernel
does, and shows what a notebook would, in the order a notebook shows it:
printed text, `display(...)` and `plt.show()` output where they happen, then
any error, the value of the last line (a Sound, or a list of them, gets an
audio player), and last the figures still open.

`prepare` runs the Colab tutorial's setup cell first, so `show`, `download`
and the rest are defined and a cell copied from the tutorial runs unchanged.
"""

import ast
import contextlib
import io
import json
import linecache
import traceback
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import soundfile  # noqa: E402
from matplotlib.artist import Artist  # noqa: E402

import sonore as so  # noqa: E402

namespace = {"__name__": "__main__"}
items = []  # what the cell running now has shown so far


def wav_bytes(sound):
    """The sound as 16-bit WAV at its true level, as a notebook player plays it."""
    buffer = io.BytesIO()
    soundfile.write(buffer, sound.data, int(sound.fs), format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def sound_item(sound):
    """A player for the sound, or a note when it peaks above full scale (the rule
    Sound._repr_html_ follows in a notebook)."""
    if sound.n_samples == 0 or sound.peak == 0:
        return ("text", repr(sound))
    if sound.peak > 1:
        return (
            "text",
            f"{sound!r}\nNo player: the peak is {sound.peak:.2f}, above full scale (1). "
            "Use snd.normalize(peak=0.9) to listen.",
        )
    return ("audio", repr(sound), wav_bytes(sound))


def figure_item(figure):
    buffer = io.BytesIO()
    with warnings.catch_warnings():  # Pyodide's matplotlib 3.10 warns about its own mathtext code
        warnings.simplefilter("ignore", DeprecationWarning)
        figure.savefig(buffer, format="png", dpi=100)
    return ("image", buffer.getvalue())


def item_for(value):
    """How a notebook would show the value: a player for a Sound, a picture for a
    figure, the markup of anything with an HTML view, else its repr."""
    if isinstance(value, so.Sound):
        return sound_item(value)
    if isinstance(value, plt.Figure):
        return figure_item(value)
    markup = value._repr_html_() if hasattr(value, "_repr_html_") else None
    return ("html", markup) if markup else ("text", repr(value))


def display(*values):
    """The notebook's display(), for the cells copied from the tutorial."""
    items.extend(item_for(value) for value in values)


def show_figures():
    """Stands in for plt.show(): shows the open figures here, then closes them."""
    items.extend(figure_item(plt.figure(number)) for number in plt.get_fignums())
    plt.close("all")


class Printed(io.TextIOBase):
    """Collects printed text as items, so it lands between the players and figures
    it was printed between."""

    def write(self, text):
        if items and items[-1][0] == "text":
            items[-1] = ("text", items[-1][1] + text)
        elif text:
            items.append(("text", text))
        return len(text)


namespace["display"] = display


def without_magics(code):
    """The cell with its notebook magics (`%pip install ...`, `!ls`) blanked out,
    as there is no shell here; blank, so the line numbers stay."""
    return "".join("\n" if line.lstrip().startswith(("%", "!")) else line for line in code.splitlines(True))


def run(code):
    """Runs the cell; returns a list of ("text", str), ("html", markup),
    ("error", traceback), ("audio", label, wav bytes) and ("image", png bytes)."""
    code = without_magics(code)
    items.clear()
    plt.close("all")  # each run shows only the figures it made
    linecache.cache["<cell>"] = (len(code), None, code.splitlines(True), "<cell>")  # so tracebacks quote it
    value = None
    real_show, plt.show = plt.show, show_figures
    try:
        tree = ast.parse(code, "<cell>")
        last = tree.body.pop() if tree.body and isinstance(tree.body[-1], ast.Expr) else None
        with contextlib.redirect_stdout(Printed()), contextlib.redirect_stderr(Printed()):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)  # as in figure_item
                exec(compile(tree, "<cell>", "exec"), namespace)
                if last is not None:
                    value = eval(compile(ast.Expression(last.value), "<cell>", "eval"), namespace)
    except Exception as error:
        # Only the frames in the cell: the runner's own frames would mean nothing to the reader.
        frames = [frame for frame in traceback.extract_tb(error.__traceback__) if frame.filename == "<cell>"]
        trace = "".join(traceback.format_list(frames)) if frames else ""
        items.append(("error", trace + "".join(traceback.format_exception_only(error))))
    finally:
        plt.show = real_show
    sounds = value if isinstance(value, list | tuple) and value else [value]
    if all(isinstance(sound, so.Sound) for sound in sounds):
        items.extend(sound_item(sound) for sound in sounds)
    elif value is not None and not isinstance(value, Artist):  # a figure is shown below
        items.append(item_for(value))
    show_figures()  # figures still open come last, as in a notebook
    return list(items)


def prepare(notebook_json):
    """Runs the tutorial notebook's setup cell, its first code cell after the
    `%pip install`, and returns what it showed."""
    cells = [cell for cell in json.loads(notebook_json)["cells"] if cell["cell_type"] == "code"]
    setup = next(cell for cell in cells if not "".join(cell["source"]).lstrip().startswith("%"))
    return run("".join(setup["source"]))
