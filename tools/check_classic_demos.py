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
    n = max(1 << 16, 1 << int(np.ceil(np.log2(len(x)))))
    spectrum = np.abs(np.fft.rfft(x * np.hanning(len(x)), n))
    freqs = np.fft.rfftfreq(n, 1 / fs)
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

# Bregman & Ahad, Demonstration 25: B 650 and C 300 Hz always, A 1800 Hz in the first example; in
# each 0.8 s cycle A sounds at 0 s and B and C at 0.4 s, equal in amplitude, after four cycles of
# the standard.
data, fs = load("cc1")
x = data[:, 0]
cycle = x[round((0.8 * 6) * fs) : round((0.8 * 7) * fs)]
a, chord = cycle[round(0.03 * fs) : round(0.17 * fs)], cycle[round(0.43 * fs) : round(0.57 * fs)]
report(
    "cc1 A, B and C [Hz]",
    [peak_in_band(a, fs, 1000, 3000), peak_in_band(chord, fs, 500, 800), peak_in_band(chord, fs, 200, 400)],
    [1800, 650, 300],
    1.0,
)
report(
    "cc1 C re B, and A re B [dB]",
    [
        peak_level(chord, fs, 200, 400) - peak_level(chord, fs, 500, 800),
        peak_level(a, fs, 1000, 3000) - peak_level(chord, fs, 500, 800),
    ],
    [0, 0],
    0.3,
)

# Bregman & Ahad, Demonstration 26, third example: C starts and stops 60 ms after B.
data, fs = load("cc2c")
x = data[round(0.8 * 6 * fs) : round(0.8 * 7 * fs), 0]
starts = [
    np.argmax(band_envelope(x, fs, f, 60) > 0.5 * band_envelope(x, fs, f, 60).max()) for f in (650, 300)
]
report("cc2c C's onset after B's [ms]", 1000 * (starts[1] - starts[0]) / fs, 60, 2.0)

# Bregman & Ahad, Demonstration 28: 900 ms of a 1000 Hz tone and 100 ms of white noise each
# second, two seconds at each noise amplitude; at the last, the noise is 20 times the tone.
data, fs = load("cc4")
x = data[:, 0]
tone_rms = np.sqrt(np.mean(x[round(0.2 * fs) : round(0.8 * fs)] ** 2))
noise_rms = [
    np.sqrt(np.mean(x[round((k + 0.92) * fs) : round((k + 0.98) * fs)] ** 2)) for k in range(1, 14, 2)
]
report(
    "cc4 noise re tone (RMS), each level",
    np.array(noise_rms) / tone_rms,
    20 * np.array([0, 0.025, 0.05, 0.1, 0.2, 0.5, 1]),
    0.1 * 20 * np.array([0.025, 0.025, 0.05, 0.1, 0.2, 0.5, 1]),
)

# Houtsma, Rossing & Wagenaars, Demonstration 11: a 2000 Hz tone (125 ms) alternating with a
# noise band (125 ms), the tone starting at the noise's level and dropping 1 dB every four tones.
data, fs = load("cc5")
x = data[:, 0]
tone_db = [
    rms_db(
        np.concatenate(
            [x[round((s + 0.015) * fs) : round((s + 0.11) * fs)] for s in step + 0.25 * np.arange(4)]
        )
    )
    for step in range(16)
]
noise_db = rms_db(
    np.concatenate(
        [x[round((s + 0.015) * fs) : round((s + 0.11) * fs)] for s in 0.125 + 0.25 * np.arange(64)]
    )
)
report("cc5 first tone re noise [dB]", tone_db[0] - noise_db, 0.0, 0.5)
report(
    "cc5 tone steps re the first, largest error [dB]",
    np.max(np.abs(np.subtract(tone_db, tone_db[0]) + np.arange(16))),
    0,
    0.05,
)

# Bregman & Ahad, Demonstration 30 with noise: the 1500-3000 Hz band at an RMS 10.4 times the
# tone's (the page's reading of "3 dB above the tone's spectrum level"), the first cut 400 ms wide
# around the top at 0.75 s.
data, fs = load("cc7b")
x = data[:, 0]
ratio = np.sqrt(
    np.mean(x[round(0.6 * fs) : round(0.9 * fs)] ** 2) / np.mean(x[round(1.2 * fs) : round(1.7 * fs)] ** 2)
)
report("cc7b noise re tone (RMS) (page's own reading)", ratio, 10.4, 0.5)

# Bregman & Ahad, Demonstration 31: noise with twice the RMS of the loudest sixth of a second of
# speech kept. The sound is the gapped sentence, 0.75 s of silence, then the filled one.
data, fs = load("cc8")
x = data[:, 0]
length = 2.525
kept = [x[round(s * fs) : round((s + 1 / 6) * fs)] for s in np.arange(0, length, 1 / 3)]
gaps = [
    x[round((length + 0.75 + s) * fs) : round((length + 0.75 + s + 1 / 6) * fs)]
    for s in np.arange(1 / 6, length - 1 / 6, 1 / 3)
]
loudest = max(np.sqrt(np.mean(k**2)) for k in kept)
report(
    "cc8 noise re loudest speech (RMS), median over gaps",
    np.median([np.sqrt(np.mean(g**2)) for g in gaps]) / loudest,
    2.0,
    0.15,
)

# Bregman & Ahad, Demonstration 33: B holds harmonics 1, 3, 5, 6 and 9 of 300 Hz at 0, -20, -28,
# -33 and -38 dB (alone from 0.4 to 0.8 s).
data, fs = load("cc10")
b = data[round(0.45 * fs) : round(0.75 * fs), 0]
report(
    "cc10 harmonics 3, 5, 6, 9 re 1 [dB]",
    [peak_level(b, fs, 300 * n - 50, 300 * n + 50) - peak_level(b, fs, 250, 350) for n in (3, 5, 6, 9)],
    [-20, -28, -33, -38],
    0.3,
)

# Bregman & Ahad, Demonstration 37: the third harmonic (375 Hz) of "i" at the same level when it
# sounds on its own between vowels in the last part; absent from "en".
data, fs = load("cc14")
x = data[:, 0]
in_vowel = peak_level(x[round(0.35 * fs) : round(0.65 * fs)], fs, 340, 410)
between = peak_level(x[round(9.75 * fs) : round(10.05 * fs)], fs, 340, 410)
in_en = peak_level(x[round(4.35 * fs) : round(4.65 * fs)], fs, 340, 410)
report(
    "cc14 third harmonic between vowels, and in en (floor -60), re in i [dB]",
    [between - in_vowel, max(in_en - in_vowel, -60)],
    [0, -60],
    0.3,
)

# Houtsma, Rossing & Wagenaars, Demonstration 13: bursts of 1, 2, 4, ... 128 periods; the 300 Hz
# series comes first, a burst every 0.6 s.
data, fs = load("pk1")
x = data[:, 0]
lengths = []
for k in range(8):
    burst = np.flatnonzero(np.abs(x[round(0.6 * k * fs) : round((0.6 * k + 0.5) * fs)]) > 1e-4)
    lengths.append((burst[-1] - burst[0] + 1) / fs)
report(
    "pk1 300 Hz bursts, length in periods",
    np.array(lengths) * 300,
    2 ** np.arange(8),
    0.1 * 2 ** np.arange(8) + 0.1,
)

# Demonstration 15: 500 Hz alternating with 985 to 1035 Hz in 5 Hz steps, a pair every 2.7 s.
data, fs = load("pk2")
x = data[:, 0]
uppers = [peak_frequency(x[round((2.7 * k + 1.3) * fs) : round((2.7 * k + 2.1) * fs)], fs) for k in range(11)]
report("pk2 comparison tones [Hz]", uppers, np.arange(985, 1036, 5), 0.1)

# Demonstration 20: ten equal harmonics of 200 Hz, and noise below 300 Hz 10 dB below them all.
data, fs = load("pk3")
x = data[:, 0]
first = x[round(0.05 * fs) : round(0.75 * fs)]
report(
    "pk3 harmonics 2 to 10 re harmonic 1, largest error [dB]",
    max(
        abs(peak_level(first, fs, 200 * n - 50, 200 * n + 50) - peak_level(first, fs, 350, 450))
        for n in range(3, 11)
    ),
    0,
    0.3,
)
gap = x[round(0.85 * fs) : round(1.15 * fs)]  # noise alone, between the first two tones
tone_part = first - np.real(
    np.fft.ifft(np.fft.fft(first) * (np.abs(np.fft.fftfreq(len(first), 1 / fs)) < 320))
)
report("pk3 noise re the full ten-harmonic tone (RMS) [dB]", rms_db(gap) - rms_db(tone_part), -10, 0.5)

# Demonstration 21: three partials 200 Hz apart, starting at 800 Hz and moving up 20 Hz a step.
data, fs = load("pk4")
x = data[:, 0]
steps = [x[round((0.75 * k + 0.05) * fs) : round((0.75 * k + 0.45) * fs)] for k in (0, 5, 10)]
report(
    "pk4 partials at steps 0, 5 and 10 [Hz]",
    [
        peak_in_band(seg, fs, f - 30, f + 30)
        for seg, k in zip(steps, (0, 5, 10), strict=True)
        for f in 20 * k + np.array([800, 1000, 1200])
    ],
    [800, 1000, 1200, 900, 1100, 1300, 1000, 1200, 1400],
    0.5,
)

# Demonstration 22: the complex tones (harmonics 4 to 6) at the pure tones' RMS.
data, fs = load("pk6")
x = data[:, 0]
pure, high = x[round(0.05 * fs) : round(0.35 * fs)], x[round(0.55 * fs) : round(0.85 * fs)]
report("pk6 first note, complex re pure (RMS) [dB] (page's own)", rms_db(high) - rms_db(pure), 0, 0.2)

# Demonstration 26: in the first note the clicks of a pair are 15 ms apart, in the last 15/32 ms.
data, fs = load("pk8")
x = data[:, 0]


def pair_delay(segment, longest):
    """The lag of the largest autocorrelation peak between 0.3 ms and `longest` s."""
    lags = np.correlate(segment, segment, "full")[len(segment) - 1 :]
    lo, hi = round(0.3e-3 * fs), round(longest * fs)
    k = lo + int(np.argmax(lags[lo:hi]))
    a, b, c = lags[k - 1 : k + 2]
    return (k + 0.5 * (a - c) / (a - 2 * b + c)) / fs


report(
    "pk8 click delay, first and last note [ms]",
    [1000 * pair_delay(x[: round(0.25 * fs)], 0.02), 1000 * pair_delay(x[-round(0.25 * fs) :], 0.003)],
    [15, 15 / 32],
    0.02,
)

# Demonstration 32: 1000 Hz with 2004, 1502 and 1334.67 Hz in turn (2.7 s each, 0.5 s pauses).
data, fs = load("pk9b")
x = data[:, 0]
report(
    "pk9b upper tones [Hz]",
    [
        peak_in_band(x[round((3.2 * k + 0.5) * fs) : round((3.2 * k + 2.2) * fs)], fs, 1100, 2500)
        for k in range(3)
    ],
    [2004, 1502, 1334.67],
    0.05,
)

# Demonstration 34: 1000 and 1200 Hz with nothing at 2f1 - f2 = 800 Hz or f2 - f1 = 200 Hz in the
# sound (the first 2 s, before the 804 Hz probe), then the probe 20 dB below the two tones.
data, fs = load("pk10")
x = data[:, 0]
before, after = x[round(0.3 * fs) : round(1.8 * fs)], x[round(2.5 * fs) : round(5.7 * fs)]
report(
    "pk10 energy at 800 and 200 Hz re 1000 Hz, before the probe [dB] (below -90 counts as -90)",
    [max(peak_level(before, fs, f - 5, f + 5) - peak_level(before, fs, 995, 1005), -90) for f in (800, 200)],
    [-90, -90],
    0,
)
report(
    "pk10 probe re the 1000 Hz tone [dB]",
    peak_level(after, fs, 800, 810) - peak_level(after, fs, 995, 1005),
    -20,
    0.3,
)


# Levels and masking. The page states its levels in dB re a pure tone at RMS 1; each series of
# sounds shares one playback gain, so levels can be compared across the files of a series.
def segment(x, fs, start, stop):
    return x[round(start * fs) : round(stop * fs)]


def band_power_db(x, fs, lo, hi):
    """Power in dB of the part of `x` between `lo` and `hi` Hz (a rectangular window)."""
    spectrum = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(len(x), 1 / fs)
    return 10 * np.log10(2 * np.sum(np.abs(spectrum[(freqs >= lo) & (freqs < hi)]) ** 2) / len(x) ** 2)


# Demonstration 4: noise bursts of 0.4 s every 0.6 s in steps of 6, 3 and 1 dB.
for key, step, count in (("lm1", 6, 10), ("lm1b", 3, 15), ("lm1c", 1, 20)):
    x = load(key)[0][:, 0]
    levels = [rms_db(segment(x, fs, 0.6 * k + 0.05, 0.6 * k + 0.35)) for k in range(count)]
    report(f"{key} steps between bursts [dB]", np.diff(levels).mean(), -step, 0.1)
    report(f"{key} largest step error [dB]", np.max(np.abs(np.diff(levels) + step)), 0, 0.3)

# Demonstration 5: each band at the spectrum level of the white noise before it, so its level
# re the white noise is 10 log10 of its share of the bandwidth.
data, fs = load("lm2")
nyquist = fs / 2
bands = {
    "lm2": [(0, c) for c in (10000, 4000, 2000, 1000, 500)],
    "lm2b": [(c, nyquist) for c in (500, 1000, 2000, 4000, 10000)],
    "lm2c": [(c * 2 ** (-1 / 6), c * 2 ** (1 / 6)) for c in (500, 1000, 2000, 4000, 8000)],
}
for key, key_bands in bands.items():
    x = load(key)[0][:, 0]
    white = rms_db(segment(x, fs, 0.1, 1.4))
    report(
        f"{key} bands re the white noise [dB]",
        [rms_db(segment(x, fs, 1.8 * k + 0.1, 1.8 * k + 1.4)) - white for k in range(1, 6)],
        np.round([10 * np.log10((hi - lo) / nyquist) for lo, hi in key_bands], 2),
        0.3,
    )
x = load("lm2d")[0][:, 0]
white, pink = segment(x, fs, 0.1, 1.9), segment(x, fs, 2.6, 4.4)
report("lm2d pink re white noise [dB]", rms_db(pink) - rms_db(white), 0, 0.3)
report(
    "lm2d pink, 4000-8000 Hz re 500-1000 Hz [dB] (white: 10 log10 8 = 9.03)",
    band_power_db(pink, fs, 4000, 8000) - band_power_db(pink, fs, 500, 1000),
    0,
    0.5,
)

# Demonstration 2: the 2000 Hz tone in 10 steps of 5 dB, then in noise of spectrum level 50 dB
# below the first step, over bands of 9980, 1000, 250 and 10 Hz.
x = load("lm3")[0][:, 0]
steps = [rms_db(segment(x, fs, 0.25 + 0.5 * k + 0.05, 0.25 + 0.5 * k + 0.25)) for k in range(10)]
report("lm3 tone steps [dB]", np.diff(steps), [-5.0] * 9, 0.05)
tone_alone = x
for key, width in (("lm3b", 9980), ("lm3c", 1000), ("lm3d", 250), ("lm3e", 10)):
    # The tone is the same in every file, so the difference is the noise alone, over all 5.25 s.
    masker = load(key)[0][:, 0] - tone_alone
    report(
        f"{key} noise re the first tone step [dB]",
        rms_db(masker) - steps[0],
        round(-50 + 10 * np.log10(width), 2),
        0.1,
    )

# Demonstration 3: reference band 930-1075 Hz and test bands k x 145 Hz wide around 1000 Hz
# (geometric center), all at one RMS.
x = load("lm4")[0][:, 0]
levels = [
    rms_db(segment(x, fs, 0.2 + 1.8 * k + offset, 0.2 + 1.8 * k + offset + 0.5))
    for k in range(8)
    for offset in (0, 0.7)
]
# Each band is at RMS 1 before its 10 ms ramps; what the ramps take depends on the noise near its
# ends, a few hundredths of a dB for a band this narrow.
report("lm4 largest level difference across bursts [dB]", np.ptp(levels), 0, 0.25)
edges = []
for k in range(8):
    test = segment(x, fs, 0.2 + 1.8 * k + 0.7, 0.2 + 1.8 * k + 1.2)
    power = np.abs(np.fft.rfft(test, 1 << 16)) ** 2
    freqs = np.fft.rfftfreq(1 << 16, 1 / fs)
    cumulative = np.cumsum(power) / power.sum()
    edges.append(np.interp([0.005, 0.995], cumulative, freqs))
edges = np.array(edges)
report(
    "lm4 test bandwidths / 145 Hz (99% of the power)",
    (edges[:, 1] - edges[:, 0]) / 145,
    list(range(1, 9)),
    0.15,
)
report("lm4 test bands' geometric centers [Hz]", np.sqrt(edges[:, 0] * edges[:, 1]), [1000] * 8, 15)

# Demonstration 8: one-second bursts at 0, -16, -20, ... -40 dB in a masker at -45 dB.
x = load("lm5")[0][:, 0]
masker = rms_db(segment(x, fs, 0.1, 0.45))
bursts = [rms_db(segment(x, fs, 0.5 + 1.5 * k + 0.1, 0.5 + 1.5 * k + 0.9)) for k in range(8)]
burst_levels = 10 * np.log10(10 ** (np.array(bursts) / 10) - 10 ** (masker / 10))
report(
    "lm5 bursts re the first [dB]",
    burst_levels - burst_levels[0],
    [0, -16, -20, -24, -28, -32, -36, -40],
    0.3,
)
report("lm5 masker re the first burst [dB]", masker - burst_levels[0], -45, 0.3)
for key, burst in (("lm5d", 0.03), ("lm5e", 0.01), ("lm5g", 0.001)):
    # Each burst is at its RMS before its linear ramps, which take away 2/3 of their length.
    ramp = min(0.5e-3, burst / 4)
    x = load(key)[0][:, 0]
    background = np.mean(segment(x, fs, 0.1, 0.45) ** 2) * (burst + 0.02) * fs
    energy = np.sum(segment(x, fs, 0.49, 0.5 + burst + 0.01) ** 2) - background
    report(
        f"{key} first burst's energy re 1 s at its level, less its ramps [dB]",
        10 * np.log10(energy / fs / (burst - 4 * ramp / 3)) - burst_levels[0],
        0,
        0.3 if burst > 0.005 else 1.0,
    )

# Demonstration 10: the 10 ms blip in steps of 4 dB, and the noise 100 or 20 ms after (backward)
# or before (forward) it.
x = load("lm6")[0][:, 0]
blips = [rms_db(segment(x, fs, 0.2 + 0.6 * k + 0.002, 0.2 + 0.6 * k + 0.008)) for k in range(10)]
report("lm6 blip steps [dB]", np.diff(blips), [-4.0] * 9, 0.1)
for key, gap, before in (
    ("lm6b", 0.1, False),
    ("lm6c", 0.02, False),
    ("lm6e", 0.1, True),
    ("lm6f", 0.02, True),
):
    x = load(key)[0][:, 0]
    first = onsets(segment(x, fs, 0.1, 0.75), fs, gap=2e-3) + 0.1
    measured = first[1] - first[0] - (0.25 if before else 0.01)
    report(f"{key} gap between tone and noise [ms]", 1000 * measured, 1000 * gap, 0.5)

# Demonstration 9: test tone pulses at 0, then -15 to -55 dB re the masker, low masking high and
# high masking low.
for key, masker_freq, test_freq in (("lm7", 1200, 2000), ("lm7b", 2000, 1200)):
    x = load(key)[0][:, 0]
    levels = []
    for k in range(4):
        pulse = segment(x, fs, 0.6 * k + 0.32, 0.6 * k + 0.48)
        hi = peak_level(pulse, fs, test_freq - 20, test_freq + 20)
        levels.append(hi - peak_level(pulse, fs, masker_freq - 20, masker_freq + 20))
    report(f"{key} test re masker, first four steps [dB]", levels, [0, -15, -20, -25], 0.3)

# Demonstration 14: noise 20-900 Hz 20 dB above the 1000 Hz tone. The second pair's sound less
# the fourth pair's (the same tone, no noise) is the noise alone; both have 20 ms linear ramps.
x = load("lm8")[0][:, 0]
masker = segment(x, fs, 1.8, 2.5) - segment(x, fs, 5.4, 6.1)
probe = segment(x, fs, 5.5, 6.0)
report(
    "lm8 noise re the tone, less their ramps [dB]",
    rms_db(masker)
    - 10 * np.log10((0.7 - 0.04 * 2 / 3) / 0.7)
    - rms_db(probe)
    + 10 * np.log10((0.5 - 0.04 * 2 / 3) / 0.5),
    20,
    0.05,
)

# Demonstration 12: pairs of tones 30 dB apart.
x = load("lm9")[0][:, 0]
pairs = [
    rms_db(segment(x, fs, 1.5 + 2 * k + 0.8 + 0.05, 1.5 + 2 * k + 1.25))
    - rms_db(segment(x, fs, 1.5 + 2 * k + 0.05, 1.5 + 2 * k + 0.45))
    for k in range(5)
]
report("lm9 loud re soft tone of each pair [dB]", pairs, [30.0] * 5, 0.05)

# Demonstration 33: a symmetric compressor adds odd harmonics only, a rectifier even ones; the
# pointer tone sits at the level of the harmonic it points to.
for key, odd in (("lm10", True), ("lm10b", False)):
    x = load(key)[0][:, 0]
    distorted = segment(x, fs, 0.1, 0.9)
    fundamental = peak_level(distorted, fs, 430, 450)
    second, third = (peak_level(distorted, fs, f - 10, f + 10) - fundamental for f in (880, 1320))
    if odd:
        report(
            f"{key} second harmonic re third [dB] (below -60 counts as -60)", max(second - third, -60), -60, 0
        )
    else:
        report(f"{key} second harmonic re fundamental [dB], above -20", min(second, -20), -20, 0)
    target = 1320 if odd else 880
    pointer = segment(x, fs, 1.4, 2.2)
    report(
        f"{key} pointer re the harmonic it points to [dB]",
        peak_level(pointer, fs, target - 10, target + 10)
        - peak_level(distorted, fs, target - 10, target + 10),
        0,
        0.3,
    )
x = load("lm10c")[0][:, 0]
distorted = segment(x, fs, 0.1, 0.9)
report(
    "lm10c 400 Hz re the 700 Hz tone [dB], above -40",
    min(peak_level(distorted, fs, 390, 410) - peak_level(distorted, fs, 690, 710), -40),
    -40,
    0,
)
report(
    "lm10c 300 Hz (f2 - f1, even order) re the 700 Hz tone [dB] (below -60 counts as -60)",
    max(peak_level(distorted, fs, 290, 310) - peak_level(distorted, fs, 690, 710), -60),
    -60,
    0,
)

print()
print(f"{len(failures)} failed" + (": " + ", ".join(failures) if failures else ""))
raise SystemExit(bool(failures))
