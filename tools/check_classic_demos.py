"""Numerical checks of the classic demonstration reproductions against their booklets.

The gallery's Classic demonstrations pages reproduce Bregman & Ahad (1996) and Houtsma, Rossing
& Wagenaars (1987) from the numbers in their booklets. This script reads the committed audio
files with soundfile and measures them with NumPy alone, sharing no code with sonore, so a page
that drifts from the booklet numbers it states shows up here. The audio is normalized for
playback, so levels are checked relative to each other, never absolutely. Each line prints what
was measured and the booklet value it is compared with.

    python tools/check_classic_demos.py
"""

from pathlib import Path

import numpy as np
import soundfile as sf

AUDIO = Path(__file__).resolve().parent.parent / "docs" / "gallery" / "audio"
failures = []


def load(key):
    (path,) = AUDIO.glob(f"{key}_*.flac")
    data, fs = sf.read(path, always_2d=True)
    return data, fs


def report(name, measured, expected, tolerance):
    ok = np.all(np.abs(np.asarray(measured) - np.asarray(expected)) <= tolerance)
    if not ok:
        failures.append(name)
    print(f"{'ok  ' if ok else 'FAIL'} {name:<64s} measured {np.round(measured, 2)}, booklet {expected}")


def peak_frequency(x, fs):
    """Frequency of the largest spectral peak of `x`, refined by parabolic interpolation."""
    n = 1 << int(np.ceil(np.log2(len(x) * 8)))
    spectrum = np.abs(np.fft.rfft(x * np.hanning(len(x)), n))
    k = int(np.argmax(spectrum))
    a, b, c = np.log(spectrum[k - 1 : k + 2] + 1e-300)
    return (k + 0.5 * (a - c) / (a - 2 * b + c)) * fs / n


def onsets(x, fs, threshold_db=-30, gap=5e-3):
    """Times where the envelope first rises above `threshold_db` re its peak after a silence."""
    loud = np.abs(x) > np.abs(x).max() * 10 ** (threshold_db / 20)
    idx = np.flatnonzero(loud)
    starts = idx[np.concatenate([[True], np.diff(idx) > gap * fs])]
    return starts / fs


def rms_db(x):
    return 10 * np.log10(np.mean(x**2))


# Bregman & Ahad, Demonstration 1: highs 2500, 2000, 1600 Hz and lows 350, 430, 550 Hz,
# alternating H1 L1 H2 L2 H3 L3, 100 ms a tone in the fast cycle (which starts at 8.2 s).
data, fs = load("sg1")
x = data[:, 0]
slots = [x[round((8.2 + 0.1 * k + 0.02) * fs) : round((8.2 + 0.1 * k + 0.08) * fs)] for k in range(6)]
report(
    "sg1 six-tone cycle, tone frequencies [Hz]",
    [peak_frequency(s, fs) for s in slots],
    [2500, 350, 2000, 430, 1600, 550],
    1.0,
)
report(
    "sg1 lows re highs [dB]",
    np.mean([rms_db(s) for s in slots[1::2]]) - np.mean([rms_db(s) for s in slots[::2]]),
    6.0,
    0.3,
)

# Bregman & Ahad, Demonstration 3: a 1400 / 500 Hz gallop whose unit shrinks from 287 to 88 ms.
data, fs = load("sg3")
x = data[:, 0]
# The tones abut, so the units are rebuilt from the booklet: 80 units (20 gallops) of geometrically
# shrinking length, then 24 more at 88 ms; every tone's middle should hold its frequency.
lengths = np.concatenate([np.geomspace(0.287, 0.088, 80), np.full(24, 0.088)])
starts = np.concatenate([[0], np.cumsum(lengths[:-1])])
measured = [
    peak_frequency(x[round((t + 0.3 * d) * fs) : round((t + 0.7 * d) * fs)], fs)
    for i, (t, d) in enumerate(zip(starts, lengths, strict=True))
    if i % 4 != 3
]
expected = [(1400, 500, 1400)[i % 4] for i in range(104) if i % 4 != 3]
report(
    "sg3 largest error of any tone's frequency [Hz]", np.max(np.abs(np.subtract(measured, expected))), 0, 2.0
)
gap = x[round((starts[3] + 0.3 * lengths[3]) * fs) : round((starts[3] + 0.7 * lengths[3]) * fs)]
report("sg3 first silent unit, peak re the sound's peak", np.abs(gap).max() / np.abs(x).max(), 0.0, 1e-3)

# Houtsma, Rossing & Wagenaars, Demonstration 19: A stays at 2000 Hz while B moves from 1000 Hz
# to 4000 Hz and back, one semitone per ABA- triplet (here 400 ms long, 50 ms tones).
data, fs = load("sg4")
x = data[:, 0]
a_tones = [
    peak_frequency(x[round((0.4 * k + 0.01) * fs) : round((0.4 * k + 0.04) * fs)], fs) for k in range(49)
]
b_tones = [
    peak_frequency(x[round((0.4 * k + 0.11) * fs) : round((0.4 * k + 0.14) * fs)], fs) for k in range(49)
]
report("sg4 A tone, lowest and highest over the sweep [Hz]", [min(a_tones), max(a_tones)], [2000, 2000], 2.0)
report(
    "sg4 B tone, first, highest and last [Hz]",
    [b_tones[0], max(b_tones), b_tones[-1]],
    [1000, 4000, 1000],
    4.0,
)

# Bregman & Ahad, Demonstration 4: runs of 2, 4, 8, 16 and 32 gallops of 125 ms units, the
# 700 Hz low 6 dB above the 2000 Hz high, 4 s of silence between runs.
data, fs = load("sg5")
x = data[:, 0]
times = onsets(x, fs)
runs = np.split(times, np.flatnonzero(np.diff(times) > 1.0) + 1)
report("sg5 tones per run (3 a gallop)", [len(r) for r in runs], [6, 12, 24, 48, 96], 0)
report("sg5 onset spacing within a gallop [ms]", 1000 * np.median(np.diff(runs[-1])), 125, 1.0)
high = x[round((runs[0][0] + 0.03) * fs) : round((runs[0][0] + 0.09) * fs)]
low = x[round((runs[0][1] + 0.03) * fs) : round((runs[0][1] + 0.09) * fs)]
report("sg5 low re high [dB]", rms_db(low) - rms_db(high), 6.0, 0.3)

# Bregman & Ahad, Demonstration 41 (stand-in): the two parts start in the middle, the same in both
# ears, and end in opposite ears.
data, fs = load("sg20")
second = round(fs)
start, end = data[: second * 2], data[-second * 2 :]
report(
    "sg20 left/right correlation, first 2 s and last 2 s",
    [np.corrcoef(start.T)[0, 1], np.corrcoef(end.T)[0, 1]],
    [1.0, 0.0],
    0.05,
)

print()
print(f"{len(failures)} failed" + (": " + ", ".join(failures) if failures else ""))
raise SystemExit(bool(failures))
