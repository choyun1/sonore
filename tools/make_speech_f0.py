"""Make the F0 track for the "seeing speech" sentence (docs/design/frames/frames.md, step 3).

Runs WORLD's Harvest estimator through pyworld once, at
development time, and writes a small CSV beside the recording, so the gallery
needs neither pyworld nor an F0 estimator of its own.

    pip install pyworld            # dev-time only; not a sonore dependency
    python tools/make_speech_f0.py docs/speech/bdl_arctic_a0131.flac

Writes ``<name>_f0.csv``: a ``#`` provenance line, then columns ``time`` [s]
and ``f0`` [Hz] (0 where unvoiced), at Harvest's default 5 ms hop (its frame period) and default search range
(71-800 Hz).
"""

import sys
from pathlib import Path

import numpy as np
import pyworld
import soundfile as sf

HOP_MS = 5.0


def main(path: Path) -> None:
    x, fs = sf.read(path, dtype="float64")
    if x.ndim != 1:
        raise ValueError(f"{path} must be mono")
    f0, t = pyworld.harvest(x, fs, frame_period=HOP_MS)
    out = path.with_name(path.stem + "_f0.csv")
    source = f"# WORLD Harvest via pyworld {pyworld.__version__}, frame period {HOP_MS:g} ms"
    data = np.column_stack([t, f0])
    np.savetxt(out, data, fmt=["%.3f", "%.2f"], delimiter=",", header=f"{source}\ntime,f0", comments="")
    v = f0 > 0
    print(f"{out}: {len(t)} time windows, {v.mean():.0%} voiced, F0 {f0[v].min():.0f}-{f0[v].max():.0f} Hz")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
