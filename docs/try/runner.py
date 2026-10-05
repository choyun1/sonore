"""Runs a code cell for the landing page and returns what it shows.

The page's worker loads this file into Pyodide; tests/test_landing_page.py runs
it under CPython. It keeps one namespace across runs, as a notebook kernel
does, and returns what a notebook would show, in the same order: the printed
text, any error, the value of the last line (a Sound, or a list of them, as WAV
bytes for an audio player; anything else as its repr), then every open figure
as a PNG.
"""

import ast
import contextlib
import io
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


def run(code):
    """Runs the cell; returns a list of ("text", str), ("error", traceback),
    ("audio", label, wav bytes) and ("image", png bytes) items."""
    plt.close("all")  # each run shows only the figures it made
    printed = io.StringIO()
    value, error_text = None, ""
    linecache.cache["<cell>"] = (len(code), None, code.splitlines(True), "<cell>")  # so tracebacks quote it
    try:
        tree = ast.parse(code, "<cell>")
        last = tree.body.pop() if tree.body and isinstance(tree.body[-1], ast.Expr) else None
        # Pyodide's matplotlib 3.10 warns about its own mathtext code on every figure (here and below).
        output = contextlib.redirect_stdout(printed), contextlib.redirect_stderr(printed)
        with output[0], output[1], warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            exec(compile(tree, "<cell>", "exec"), namespace)
            if last is not None:
                value = eval(compile(ast.Expression(last.value), "<cell>", "eval"), namespace)
    except Exception as error:
        # Only the frames in the cell: the runner's own frames would mean nothing to the reader.
        frames = [frame for frame in traceback.extract_tb(error.__traceback__) if frame.filename == "<cell>"]
        trace = "".join(traceback.format_list(frames)) if frames else ""
        error_text = trace + "".join(traceback.format_exception_only(error))
    items = [("text", printed.getvalue())] if printed.getvalue() else []
    if error_text:
        items.append(("error", error_text))
    sounds = value if isinstance(value, list | tuple) and value else [value]
    if all(isinstance(sound, so.Sound) for sound in sounds):
        items.extend(sound_item(sound) for sound in sounds)
    elif value is not None and not isinstance(value, Artist):  # a figure is shown below
        items.append(("text", repr(value)))
    for number in plt.get_fignums():  # figures come last, as in a notebook
        buffer = io.BytesIO()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            plt.figure(number).savefig(buffer, format="png", dpi=100)
        items.append(("image", buffer.getvalue()))
    plt.close("all")
    return items
