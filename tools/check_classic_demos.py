"""Numerical checks of the classic demonstration reproductions against their booklets.

The gallery's Classic demonstrations pages reproduce Bregman & Ahad (1996) and Houtsma, Rossing
& Wagenaars (1987) from the numbers in their booklets (and, on the Fusion page, comodulation
masking release, whose numbers are the page's own). This script reads the committed audio
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


# Bregman & Ahad, Demonstration 14: low (500-1000 Hz) and high (2000-4000 Hz) bands of noise at
# equal spectrum level, the first burst high and the second low.
data, fs = load("sg9")
x = data[:, 0]
first, second = onsets(x, fs)[:2]
densities = []
for start, lo, hi in ((first, 2000, 4000), (second, 500, 1000)):
    burst = x[round(start * fs) : round((start + 0.05) * fs)]
    power = np.abs(np.fft.rfft(burst)) ** 2
    freqs = np.fft.rfftfreq(len(burst), 1 / fs)
    densities.append(power[(freqs > lo) & (freqs < hi)].sum() / (hi - lo))
report("sg9 spectrum level, low band re high band [dB]", 10 * np.log10(densities[1] / densities[0]), 0.0, 0.5)


def band_envelope(x, fs, centre, half_width):
    """The envelope of `x` within `half_width` Hz of `centre`: an FFT band-pass, then the
    magnitude of the analytic signal."""
    spectrum = np.fft.fft(x)
    freqs = np.fft.fftfreq(len(x), 1 / fs)
    keep = (freqs > centre - half_width) & (freqs < centre + half_width)  # positive side only
    return 2 * np.abs(np.fft.ifft(np.where(keep, spectrum, 0)))


def peak_in_band(x, fs, lo, hi):
    """Frequency of the largest spectral peak of `x` between `lo` and `hi` Hz."""
    n = 1 << int(np.ceil(np.log2(len(x) * 8)))
    spectrum = np.abs(np.fft.rfft(x * np.hanning(len(x)), n))
    freqs = np.fft.rfftfreq(n, 1 / fs)
    inside = np.flatnonzero((freqs >= lo) & (freqs <= hi))
    k = inside[np.argmax(spectrum[inside])]
    a, b, c = np.log(spectrum[k - 1 : k + 2] + 1e-300)
    return (k + 0.5 * (a - c) / (a - 2 * b + c)) * fs / n


# Bregman & Ahad, Demonstration 18: eight harmonics, the third (here 600 Hz on a 200 Hz
# fundamental) held still while the frame drops 1 % a step for 15 steps, then rises back.
data, fs = load("fu2")
x = data[:, 0]
frame = np.concatenate([1 - np.arange(16) / 100, (1 - np.arange(16) / 100)[::-1]])
starts = 1.25 * np.concatenate([2 + np.arange(16), 20 + np.arange(16)])
segments = [x[round((t + 0.02) * fs) : round((t + 0.23) * fs)] for t in starts]
report(
    "fu2 target, lowest and highest over both series [Hz]",
    [
        min(peak_in_band(s, fs, 570, 630) for s in segments),
        max(peak_in_band(s, fs, 570, 630) for s in segments),
    ],
    [600, 600],
    0.5,
)
errors = [peak_in_band(s, fs, 300, 450) - 400 * f for s, f in zip(segments, frame, strict=True)]
report("fu2 frame's second harmonic, largest error [Hz]", np.max(np.abs(errors)), 0, 0.5)


def peak_level(x, fs, lo, hi):
    """Height in dB of the largest spectral peak of `x` between `lo` and `hi` Hz."""
    spectrum = np.abs(np.fft.rfft(x * np.hanning(len(x)), 1 << 16))
    freqs = np.fft.rfftfreq(1 << 16, 1 / fs)
    return 20 * np.log10(spectrum[(freqs >= lo) & (freqs <= hi)].max())


in_tune = segments[0]
report(
    "fu2 in tune, target and harmonics 1, 2, 8 re harmonic 4 [dB]",
    [
        peak_level(in_tune, fs, *band) - peak_level(in_tune, fs, 750, 850)
        for band in ((550, 650), (150, 250), (350, 450), (1550, 1650))
    ],
    [0, 0, 0, 0],
    0.3,
)

# Bregman & Ahad, Demonstration 20: all four partials at equal intensity. In the first sound both
# sets follow the fast pattern (400 to 660 Hz), so the single partial stays below 700 Hz and the
# other three above 1100 Hz.
data, fs = load("fu4")
x = data[round(0.2 * fs) : round(2.3 * fs), 0]
power = np.abs(np.fft.rfft(x)) ** 2
freqs = np.fft.rfftfreq(len(x), 1 / fs)
single = power[(freqs > 300) & (freqs < 750)].sum()
three = power[(freqs > 1100) & (freqs < 3500)].sum()
report("fu4 single partial re the mean of the other three [dB]", 10 * np.log10(3 * single / three), 0.0, 0.3)

# Bregman & Ahad, Demonstration 21: with a 125 ms rise, the second and third tones of a cluster
# peak at 0.88 and 0.76 of the first (the second pair starts at 4 s; its first cluster is MHLM).
data, fs = load("fu7")
x = data[round(4.0 * fs) : round(5.0 * fs), 0]
first = band_envelope(x, fs, 800, 40)[: round(0.3 * fs)].max()
report(
    "fu7 peaks of the second and third tones re the first",
    [band_envelope(x, fs, 900, 40).max() / first, band_envelope(x, fs, 700, 40).max() / first],
    [0.88, 0.76],
    0.03,
)

# Bregman & Ahad, Demonstration 22: each masker (1000 Hz) has flankers at 250, 500, 2000 and
# 4000 Hz, 12 dB down; in the last part the flankers start late. The masker times come from
# the page's seed, as the booklet leaves them open.
data, fs = load("fu8")
x = data[:, 0]
rng = np.random.default_rng(22)
gaps = [(0.5 * k + 0.08 + rng.uniform(0, 0.3), rng.uniform(0.03, 0.06)) for k in range(10)]
levels = []
for onset, length in gaps:
    segment = x[round((12.0 + onset + 0.008) * fs) : round((12.0 + onset + length - 0.008) * fs)]
    n = 1 << 15
    spectrum = np.abs(np.fft.rfft(segment * np.hanning(len(segment)), n))
    at = [spectrum[round(f * n / fs) - 3 : round(f * n / fs) + 4].max() for f in (250, 500, 1000, 2000, 4000)]
    levels.append(20 * np.log10(np.delete(at, 2) / at[2]))
report(
    "fu8 flankers re masker, lowest and highest over the third part [dB]",
    [np.min(levels), np.max(levels)],
    [-12, -12],
    0.5,
)
delays = []
for onset, length in gaps:
    region = x[round((18.0 + onset - 0.02) * fs) : round((18.0 + onset + length + 0.06) * fs)]
    starts = [
        np.argmax(band_envelope(region, fs, f, 100) > 0.5 * band_envelope(region, fs, f, 100).max())
        for f in (1000, 4000)
    ]
    delays.append(1000 * (starts[1] - starts[0]) / fs)
report("fu8 flanker delay in the last part, median [ms] (the page's own value)", np.median(delays), 30, 2.0)

# Comodulation masking release (every number ours): the flanking bands share the on-target
# band's envelope in the second part and have their own in the third.
data, fs = load("fu9")
x = data[:, 0]
correlations = []
for part in (1, 2):
    segment = x[round((0.5 + 9.0 * part + 0.5) * fs) : round((0.5 + 9.0 * part + 7.5) * fs)]
    correlations.append(
        np.corrcoef(band_envelope(segment, fs, 500, 40), band_envelope(segment, fs, 2000, 40))[0, 1]
    )
report(
    "fu9 envelope correlation, 500 and 2000 Hz bands, parts 2 and 3 (page's own)",
    correlations,
    [1.0, 0.0],
    0.15,
)

print()
print(f"{len(failures)} failed" + (": " + ", ".join(failures) if failures else ""))
raise SystemExit(bool(failures))
