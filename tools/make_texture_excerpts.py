"""Cut the texture excerpts in docs/textures/ from the original recordings.

    python tools/make_texture_excerpts.py DIR

DIR holds the original downloads under the file names listed in EXCERPTS
(Freesound's default names). Each excerpt is the middle ``duration`` seconds
of its file, mixed to mono, resampled to 44.1 kHz, peak-normalized to 0.9,
and written as 16-bit FLAC. Sources, licenses and the reasons behind each
choice are in docs/textures/SOURCES.md.
"""

import sys
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

FS = 44100
OUT = Path(__file__).parent.parent / "docs" / "textures"

# name: (original file, excerpt duration in seconds)
EXCERPTS = {
    "rain": ("234317__nick121087__rain-ambience.wav", 7.0),
    "stream": ("370870__cognito-perceptu__water-in-creek-burbling.wav", 7.0),
    "crickets": ("175020__sengjinn__ambience-night-field-cricket-01.wav", 7.0),
    "applause": ("462362__breviceps__small-applause.wav", 3.5),
    "fire": ("204348__sauron974__crackling-fire.wav", 20.0),
    "mud": ("bubbling_mud-yell-FountainPaintPot.mp3", 20.0),
    "wind_rain": ("213872__sonicwars__wind-gusts-and-rain.wav", 7.0),
}


def excerpt(path: Path, duration: float) -> tuple[np.ndarray, float]:
    x, fs = sf.read(path)
    x = x.mean(axis=1) if x.ndim > 1 else x
    start = len(x) / fs / 2 - duration / 2
    seg = x[int(start * fs) : int(start * fs) + int(duration * fs)]
    if fs != FS:
        g = gcd(FS, int(fs))
        seg = resample_poly(seg, FS // g, int(fs) // g)
    return seg * (0.9 / np.abs(seg).max()), start


def main(src: Path) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, (fname, duration) in EXCERPTS.items():
        seg, start = excerpt(src / fname, duration)
        sf.write(OUT / f"{name}.flac", seg, FS, subtype="PCM_16")
        print(f"{name:10s} {start:6.2f}-{start + duration:6.2f} s of {fname}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
