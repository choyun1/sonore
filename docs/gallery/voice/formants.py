"""Formant synthesis: speech made from a source, resonances and a handful of numbers.

This script is the gallery page https://choyun1.github.io/sonore/gallery/formants.html:
docs/gallery/build.py runs it cell by cell from the repository root and shows each
cell's code beside what it made. Run it yourself from the repository root,

    python docs/gallery/voice/formants.py

or a cell at a time ("# %%" starts a cell in VS Code, Spyder and Jupytext).
"""

# %% [markdown]
# # Formant synthesis
#
# A voice is a source filtered by the vocal tract. The source is the buzz of the vocal folds,
# which sets the pitch, or the hiss of air through a narrow gap. The filter is the vocal tract's
# resonances, the formants, which set the vowel. The two are made in different places and can
# change independently; the other pages on voices build on this picture. In a formant
# synthesizer every part of it is a number: the source's pitch and level, and the frequency and
# bandwidth of each formant. `so.klatt_synthesize` follows Klatt (1980). Voicing is a set of
# harmonics of $F_0$ with the spectrum of Klatt's glottal pulse, aspiration and frication are
# noise, and formants are second-order resonators
#
# $$y[n] = A\,x[n] + B\,y[n-1] + C\,y[n-2],$$
#
# with $C = -e^{-2\pi B_k T}$, $B = 2e^{-\pi B_k T}\cos(2\pi F_k T)$ and $A = 1 - B - C$ for a
# formant at $F_k$ Hz, $B_k$ Hz wide, sampled every $T$ seconds. Every parameter is a number or a
# track of `(times, values)`, so the sounds below are written down rather than recorded, and any
# cue can be changed while the others stay put.
#
# - [A vowel, piece by piece](#h-a-vowel-piece-by-piece): the source, then one formant at a time.
# - [Source times filter](#h-source-times-filter): the same vowel as a product of spectra.
# - [Six vowels](#h-six-vowels): Peterson and Barney's male and female averages, and a
#   continuum between two vowels.
# - [Consonants from transitions](#h-consonants-from-transitions): /ba/, /da/ and /ga/ differ
#   only in where the formants start.
# - [Noise](#h-noise): frication, aspiration and breathy voice.
# - [Cascade and parallel](#h-cascade-and-parallel): why the parallel formants alternate in sign.
# - [Voice quality](#h-voice-quality): the same vowel from a tense, a modal and a lax glottal
#   pulse.
# - [Finding the formants again](#h-finding-the-formants-again): linear prediction reads the
#   formants back out of a sound, and drives the synthesizer from a recording.
# - [What this page leaves out](#h-what-this-page-leaves-out): the rest of voice quality, rules,
#   and drawing the tracks.

# %% [markdown]
# ## Code the examples share
#
# All the sounds are at 16 kHz. Each one is drawn as a waveform
# and a wideband spectrogram (Hann 6 ms), which shows formants as dark bands, with the formant
# frequencies the synthesizer was given drawn over it.

# %% [setup]
import os
import urllib.request

import matplotlib.pyplot as plt
import numpy as np

import sonore as so

plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "figure.dpi": 100})


def fetch(path):
    """A file from the sonore repository, by its path there: the local copy when this runs from
    the repository root, otherwise downloaded from GitHub to the same relative path."""
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        urllib.request.urlretrieve("https://raw.githubusercontent.com/choyun1/sonore/main/" + path, path)
    return path


def finish(snd):
    """How every sound in the gallery is played: 5 ms ramps, RMS 0.1, peak at most 0.95."""
    snd = snd.ramp(5e-3).normalize(rms=0.1)
    return snd.normalize(peak=0.95) if snd.peak > 0.95 else snd


# %%
FS = 16000


def onoff(dur, level=60, on=0.02, off=0.05):
    """An amplitude track: up over `on` seconds, down over the last `off` seconds."""
    return ([0, on, dur - off, dur], [0, level, level, 0])


def draw(container, snd, title, formants=(), long_titles=True):
    """Waveform and wideband spectrogram in a figure or subfigure, with formant tracks ((times,
    values) or numbers) drawn over it. Returns the two panels, for the playhead."""
    ax0, ax1 = container.subplots(2, 1, sharex=True, height_ratios=[0.45, 1])
    snd.plot(ax0, color="k", lw=0.4)
    ax0.set(title=title if not long_titles else f"Waveform: {title}", xlabel="")
    frame = so.GaborFrame(0.006, 0.001, n_fft=512)  # zero-padded for smooth bands
    so.STFT(snd, frame=frame).plot(ax1, db_range=60, colorbar=False, fmax=5000)
    if long_titles:
        ax1.set_title("Spectrogram (Hann 6 ms), with the formant frequencies given to the synthesizer")
    else:
        ax1.set_title("Spectrogram (Hann 6 ms), formants given dashed")
    for f in formants:
        t, v = ([0], [f]) if np.isscalar(f) else f  # a track is held after its last point
        ax1.plot([*t, snd.duration], np.array([*v, v[-1]]) / 1000, color="c", lw=1, ls="--")  # in kHz
    for ax in (ax0, ax1):
        ax.set_xlim(0, snd.duration)
    return [ax0, ax1]


def show(snd, title, formants=()):
    """One sound: waveform and wideband spectrogram, with its formants."""
    fig = plt.figure(figsize=(10, 4.2), layout="constrained")
    return fig, draw(fig, snd, title, formants)


def show_pair(sounds, title, formants):
    """Two sounds side by side, one column each; `formants` maps each sound's label to its tracks."""
    fig = plt.figure(figsize=(10, 4.4), layout="constrained")
    columns = fig.subfigures(1, 2)
    playhead = {
        label: draw(column, snd, f"{label}: {title}", formants[label], long_titles=False)
        for column, (label, snd) in zip(columns, sounds.items(), strict=True)
    }
    return fig, playhead


# %% [markdown]
# ## A vowel, piece by piece
#
# The vowel of "hod", with Peterson and Barney's (1952) average male formants: 730, 1090 and
# 2440 Hz. The pitch falls from 130 to 100 Hz, as at the end of a statement. Here the pieces are
# put together by hand with `so.resonator`, the same resonator `so.klatt_synthesize` uses.

# %%
DUR = 1.0
F0 = ([0, DUR], [130, 100])
HOD = [(730, 60), (1090, 90), (2440, 150), (3500, 200), (4500, 250)]

# Klatt's glottal source: harmonics with the spectrum of an impulse through a low-pass
# resonator at 0 Hz, 100 Hz wide (about -12 dB per octave above 50 Hz).
glottal = so.Sound(np.r_[1.0, np.zeros(2**14 - 1)], FS)
rgp = np.abs(np.fft.rfft(so.resonator(glottal, 0, 100).data[:, 0]))
freqs = np.fft.rfftfreq(2**14, 1 / FS)
source = so.harmonic_complex(DUR, FS, F0, amplitudes=lambda t, f: np.interp(f, freqs, rgp))
# Radiation from the lips: a first difference, +6 dB per octave.
source = so.Sound(np.diff(source.data[:, 0], prepend=0.0), FS)

# %% [about]
# The source on its own: the glottal pulse train after radiation, a buzz with no vowel in it.
# Its harmonics fall smoothly with frequency, with no peaks.

# %% [demo fv1] The source alone
sound = finish(source)
fig, playhead = show(sound, "the glottal source after radiation")

# %% [about]
# Through the first formant only, at 730 Hz. The buzz becomes a dull, muffled vowel: all that is
# left is energy around F1.

# %% [demo fv2] The first formant
one = so.resonator(source, *HOD[0])
sound = finish(one)
fig, playhead = show(sound, "the source through F1", [730])

# %% [about]
# Through the first two formants. F1 and F2 are what tells most vowels apart, and with both, the
# sound is recognizably the vowel of "hod", if thin.

# %% [demo fv3] Two formants
two = so.resonator(one, *HOD[1])
sound = finish(two)
fig, playhead = show(sound, "the source through F1 and F2", [730, 1090])

# %% [about]
# All five formants in cascade (one after the other). F3 to F5 add the brightness of a real
# voice; their levels follow from their frequencies, as they do in a vocal tract.

# %% [demo fv4] Five formants
vowel = source
for f, bw in HOD:
    vowel = so.resonator(vowel, f, bw)
sound = finish(vowel)
fig, playhead = show(sound, "the source through F1 to F5", [f for f, _ in HOD])

# %% [about]
# The same vowel from `so.klatt_synthesize`, with the voicing switched on and off over 20 and
# 50 ms. Apart from those ramps it is the sound above: the synthesizer does the same steps.

# %% [demo fv5] The same vowel from so.klatt_synthesize
hod = dict(F1=730, F2=1090, F3=2440)
sound = finish(so.klatt_synthesize(DUR, FS, F0=F0, AV=onoff(DUR), **hod))
fig, playhead = show(sound, "so.klatt_synthesize", [730, 1090, 2440, 3500, 4500])

# %% [markdown]
# ## Source times filter
#
# Because the source and the formants are separate, the spectrum of the vowel is the product of
# their spectra: in dB, a sum. Here the pitch is held at 100 Hz, so the harmonics sit at
# multiples of 100 Hz; each one's level is the source's level at that frequency plus the
# formants' gain there.

# %% [figure fs1] The vowel spectrum as source plus filter
vowel = so.klatt_synthesize(1.0, FS, F0=100, **hod)
spec = np.abs(np.fft.rfft(vowel.data[-FS // 2 :, 0]))  # the last 0.5 s, 2 Hz per bin
k = np.arange(1, 50)
measured = 20 * np.log10(spec[k * 50])  # harmonic k at 100 k Hz is bin 50 k
f = np.linspace(50, 5000, 2000)
z = np.exp(-2j * np.pi * f / FS)


def gain(fk, bk, f):
    c, b = -np.exp(-2 * np.pi * bk / FS), 2 * np.exp(-np.pi * bk / FS) * np.cos(2 * np.pi * fk / FS)
    zf = np.exp(-2j * np.pi * f / FS)
    return np.abs((1 - b - c) / (1 - b * zf - c * zf**2))


src_db = 20 * np.log10(gain(0, 100, f) * np.abs(1 - z))
filt_db = 20 * np.log10(np.prod([gain(fk, bk, f) for fk, bk in HOD], axis=0))
total = src_db + filt_db
harmonics = measured - measured[0] + np.interp(100, f, total)  # the output's level is arbitrary

fig, ax = plt.subplots(figsize=(10, 3.4), layout="constrained")
ax.plot(f, src_db, color="0.6", label="source (glottal pulse and radiation)")
ax.plot(f, filt_db, color="c", label="formants (the filter)")
ax.plot(f, total, color="k", lw=0.8, label="source + formants")
ax.plot(k * 100, harmonics, "o", ms=3, color="C3", label="harmonics of the synthesized vowel")
ax.set(xlabel="Frequency (Hz)", ylabel="Level (dB)", xlim=(0, 5000))
ax.set_title("The vowel of 'hod' at 100 Hz: every harmonic lands on source + filter")
ax.legend(loc="upper right", fontsize=8)

# %% [markdown]
# ## Six vowels
#
# Peterson and Barney (1952) measured the formants of ten American English vowels, spoken in words
# of the form h-vowel-d by 33 men, 28 women and 15 children, and published the averages of each
# group. Their male and female averages for six of the vowels are below, each set synthesized with
# everything else the same within the set: the pitch, the higher formants and the bandwidths.
#
# Every female formant is higher than the male one, since female vocal tracts are on average
# shorter, but not by one common factor: the cell prints the ratios. The female pitch is higher
# too, falling from 230 to 200 Hz where the male one falls from 125 to 105 Hz, and F4 and F5,
# which Peterson and Barney did not measure, are raised from 3500 and 4500 Hz to 4100 and 4900 Hz
# to stay above the female F3. These are averages over many talkers. The two recorded talkers
# the other pages use are one man and one woman, and how their formants differ is measured on
# [Two talkers](talkers.html#h-spectral-envelopes).

# %%
VOWELS = {  # male averages, F1 F2 F3 (Hz)
    "heed": (270, 2290, 3010),
    "head": (530, 1840, 2480),
    "had": (660, 1720, 2410),
    "hod": (730, 1090, 2440),
    "hawed": (570, 840, 2410),
    "who'd": (300, 870, 2240),
}
VOWELS_FEMALE = {  # female averages
    "heed": (310, 2790, 3310),
    "head": (610, 2330, 2990),
    "had": (860, 2050, 2850),
    "hod": (850, 1220, 2810),
    "hawed": (590, 920, 2710),
    "who'd": (370, 950, 2670),
}
VOWEL_DUR = 0.42
GAP = 0.15
# Each set: its vowels, and what else it is made with (a falling pitch, and F4 and F5).
SETS = {
    "Male formants and pitch": (VOWELS, dict(F0=([0, VOWEL_DUR], [125, 105]), F4=3500, F5=4500)),
    "Female formants and pitch": (VOWELS_FEMALE, dict(F0=([0, VOWEL_DUR], [230, 200]), F4=4100, F5=4900)),
}


def say(steps, dur, **rest):
    """Vowels in turn, from a list of parameter sets (F1, F2, F3), each `dur` long with a gap
    after it, and their F1 to F3 as tracks to draw, one dashed segment per vowel."""
    parts = []
    for p in steps:
        parts += [so.klatt_synthesize(dur, FS, p, AV=onoff(dur), **rest), so.silence(GAP, FS)]
    starts = np.arange(len(steps)) * (dur + GAP)
    times = np.ravel([[start, start + dur, start + dur] for start in starts])
    tracks = [(times, np.ravel([[p[name], p[name], np.nan] for p in steps])) for name in ("F1", "F2", "F3")]
    return finish(so.concat(parts[:-1])), tracks


def formant_dicts(vowels):
    return [dict(F1=f1, F2=f2, F3=f3) for f1, f2, f3 in vowels.values()]


ratios = {word: np.divide(VOWELS_FEMALE[word], VOWELS[word]) for word in VOWELS}
for word, ratio in ratios.items():
    print(f"{word:6} female / male, F1 F2 F3: " + ", ".join(f"{r:.2f}" for r in ratio))
lowest, highest = min(r.min() for r in ratios.values()), max(r.max() for r in ratios.values())

# %% [markdown]
# The ratios run from {{ f"{lowest:.2f}" }} to {{ f"{highest:.2f}" }}, so the female vowels are
# not the male ones moved up as a whole: each vowel, and each formant, moves by its own amount.

# %% [about]
# The six vowels in turn, from "heed" to "who'd", with each set's formants and pitch. The dashed
# lines are each vowel's F1, F2 and F3. The female harmonics are farther apart, so each formant
# peak is drawn by fewer of them, as [Two talkers](talkers.html#h-harmonics-sample-the-envelope)
# shows for one vowel. [Timbre](timbre.html#d-tb6) sings three of the female vowels on one note.

# %% [demo fw1] Six vowels
sounds, formant_tracks = {}, {}
for label, (vowels, rest) in SETS.items():
    sounds[label], formant_tracks[label] = say(formant_dicts(vowels), VOWEL_DUR, **rest)
fig, playhead = show_pair(sounds, "six vowels", formant_tracks)

# %% [about]
# Both sets by their first two formants, joined vowel by vowel, with the axes reversed as is
# usual, so that the chart reads like the mouth seen from the left: vowels with the tongue high
# at the top, vowels with the tongue forward at the left. Every female vowel has a higher F1 and
# F2 than the male one, and the female vowels spread wider, most of all in F2 for the front
# vowels.

# %% [figure fw2] The vowels by their first two formants
fig, ax = plt.subplots(figsize=(5, 4), layout="constrained")
for word in VOWELS:
    (f1_male, f2_male, _), (f1_female, f2_female, _) = VOWELS[word], VOWELS_FEMALE[word]
    ax.plot([f2_male, f2_female], [f1_male, f1_female], color="0.7", lw=0.8)
    ax.annotate(word, (f2_female, f1_female), textcoords="offset points", xytext=(6, -12))
for (label, (vowels, _)), color in zip(SETS.items(), ["C0", "C1"], strict=True):
    f1, f2, _ = np.transpose(list(vowels.values()))
    ax.plot(f2, f1, "o", color=color, label=label.split()[0].lower())
ax.set(xlabel="F2 (Hz)", ylabel="F1 (Hz)", xlim=(3000, 600), ylim=(950, 200))
ax.legend(loc="lower left", fontsize=8)
ax.set_title("Peterson and Barney's averages, male and female")

# %% [about]
# A continuum from "heed" to "hod" in seven equal steps of F1, F2 and F3, made with
# `so.klatt_continuum`, from the male averages at 110 Hz and from the female ones at 215 Hz.
# Somewhere in the middle the vowel stops being one and becomes the other; experiments on vowel
# categories use continua like these.

# %% [demo fw3] From heed to hod in seven steps
sounds, formant_tracks = {}, {}
for (label, (vowels, rest)), f0 in zip(SETS.items(), [110, 215], strict=True):
    steps = so.klatt_continuum(*formant_dicts({word: vowels[word] for word in ("heed", "hod")}), 7)
    sounds[label], formant_tracks[label] = say(steps, 0.36, F0=f0, F4=rest["F4"], F5=rest["F5"])
fig, playhead = show_pair(sounds, "heed to hod", formant_tracks)

# %% [markdown]
# ## Consonants from transitions
#
# A stop consonant before a vowel is mostly heard in the first few tens of milliseconds of the
# vowel: the formants move from where the closure left them to the vowel's values. Here F1 rises
# from 250 Hz in each, and only the starting points of F2 and F3 differ. A short burst of
# frication marks the release: flat for /b/, high for /d/, near F2 for /g/.

# %%
DUR_CV = 0.4
ONSETS = {"ba": (900, 2100), "da": (1700, 2700), "ga": (1500, 2000)}
BURSTS = {"ba": dict(AB=55), "da": dict(A4=55, A5=52), "ga": dict(A2=58)}


def syllable(name):
    f2, f3 = ONSETS[name]
    tracks = dict(F1=([0, 0.04], [250, 700]), F2=([0, 0.04], [f2, 1220]), F3=([0, 0.04], [f3, 2500]))
    snd = so.klatt_synthesize(
        DUR_CV,
        FS,
        F0=([0, DUR_CV], [125, 100]),
        AV=onoff(DUR_CV, on=0.005),
        AF=([0, 0.001, 0.008, 0.01], [0, 66, 60, 0]),
        rng=0,
        **tracks,
        **BURSTS[name],
    )
    return finish(snd), [tracks["F1"], tracks["F2"], tracks["F3"]]


# %% [about]
# /ba/: F2 and F3 rise into the vowel, from low starting points (the lips close the front of the
# tract).

# %% [demo fc1] The syllable ba
sound, tracks = syllable("ba")
fig, playhead = show(sound, "/ba/", tracks)

# %% [about]
# /da/: F2 and F3 fall into the vowel from high starting points.

# %% [demo fc2] The syllable da
sound, tracks = syllable("da")
fig, playhead = show(sound, "/da/", tracks)

# %% [about]
# /ga/: F2 and F3 start close together and part, F2 falling and F3 rising.

# %% [demo fc3] The syllable ga
sound, tracks = syllable("ga")
fig, playhead = show(sound, "/ga/", tracks)

# %% [markdown]
# ## Noise
#
# Klatt's synthesizer has two noise sources. Aspiration goes through the same formants as
# voicing (the cascade), as the breath noise of /h/ does. Frication goes through a separate,
# parallel set of formants, each with its own level, because a fricative's noise is made in a
# constriction near the front of the mouth and excites mainly the resonances in front of it.
# While voicing is on, both noises are modulated at $F_0$, 50% deep, as breath through vibrating
# vocal folds is.

# %% [about]
# /sa/: frication through the parallel formants 5 and 6 (4500 and 5500 Hz), then the vowel. As
# in speech, there is no silence between them: the voicing starts while the frication is still
# fading, and the first two formants glide from where the tongue left them (F1 low, F2 near
# 1700 Hz) to the vowel's values over the first 60 ms of voicing.

# %% [demo fn1] The syllable sa
dur = 0.515
tracks = dict(F1=([0.08, 0.14], [300, 730]), F2=([0.08, 0.14], [1700, 1090]))
sound = finish(
    so.klatt_synthesize(
        dur,
        FS,
        F0=([0, dur], [125, 100]),
        AV=([0, 0.08, 0.085, 0.105, dur - 0.05, dur], [0, 0, 48, 60, 60, 0]),
        AF=([0, 0.015, 0.085, 0.1, 0.115], [0, 74, 74, 64, 0]),
        A5=60,
        A6=62,
        F6=5500,
        B6=500,
        F3=2440,
        rng=0,
        **tracks,
    )
)
fig, playhead = show(sound, "/sa/: frication, then voicing", [tracks["F1"], tracks["F2"], 4500, 5500])

# %% [about]
# /ha/: aspiration alone through the vowel's formants, then voicing. The formants of the vowel
# are already there in the breath, and the breath carries on into the first 50 ms or so of the
# voicing, fading as it goes, so the vowel starts breathy.

# %% [demo fn2] The syllable ha
sound = finish(
    so.klatt_synthesize(
        dur,
        FS,
        F0=([0, dur], [125, 100]),
        AV=([0, 0.08, 0.085, 0.105, dur - 0.05, dur], [0, 0, 48, 60, 60, 0]),
        AH=([0, 0.015, 0.095, 0.125, 0.165], [0, 62, 62, 56, 0]),
        rng=0,
        **hod,
    )
)
fig, playhead = show(sound, "/ha/: aspiration, then voicing", [730, 1090, 2440])

# %% [about]
# A breathy "hod": voicing with aspiration 6 dB below it throughout, and a wider first formant.
# The noise fills in between the harmonics, most at high frequencies where the harmonics are
# weak. [Source and aperiodicity](aperiodicity.html#d-ap3) makes a breathy vowel whose share of
# noise at each frequency is known, and measures it.

# %% [demo fn3] A breathy vowel
sound = finish(
    so.klatt_synthesize(DUR, FS, F0=F0, AV=onoff(DUR), AH=onoff(DUR, level=54), B1=120, rng=0, **hod)
)
fig, playhead = show(sound, "breathy: aspiration 6 dB below voicing", [730, 1090, 2440])

# %% [markdown]
# ## Cascade and parallel
#
# In the cascade, formants run one after another and their levels follow from their
# frequencies. In the parallel branch each formant gets the source separately and is added at a
# level of its own. Added with the same sign, neighboring formants partly cancel between their
# peaks, because their phases differ by about half a cycle there. Klatt adds them with
# alternating signs, and then the sum follows the cascade closely, given the right levels.

# %% [figure fp1] Five formants in cascade, and in parallel with each sign rule
f = np.linspace(50, 5000, 4000)
z = np.exp(-2j * np.pi * f / FS)


def response(fk, bk):
    c, b = -np.exp(-2 * np.pi * bk / FS), 2 * np.exp(-np.pi * bk / FS) * np.cos(2 * np.pi * fk / FS)
    return (1 - b - c) / (1 - b * z - c * z**2)


hs = [response(fk, bk) for fk, bk in HOD]
cascade = np.prod(hs, axis=0)
peaks = [np.argmin(np.abs(f - fk)) for fk, _ in HOD]
levels = [np.abs(cascade[i]) / np.abs(h[i]) for h, i in zip(hs, peaks, strict=True)]
same = sum(g * h for g, h in zip(levels, hs, strict=True))
alternating = sum((-1) ** k * g * h for k, (g, h) in enumerate(zip(levels, hs, strict=True)))

fig, ax = plt.subplots(figsize=(10, 3.4), layout="constrained")
db = lambda h: 20 * np.log10(np.abs(h))  # noqa: E731
ax.plot(f, db(cascade), color="k", lw=2.5, alpha=0.35, label="cascade")
ax.plot(f, db(same), color="C3", lw=1, label="parallel, same signs")
ax.plot(f, db(alternating), color="C0", lw=1, ls="--", label="parallel, alternating signs")
ax.set(xlabel="Frequency (Hz)", ylabel="Gain (dB)", xlim=(0, 5000), ylim=(-40, 50))
ax.set_title("The vowel of 'hod': parallel formants, each set to the cascade's level at its peak")
ax.legend(loc="upper right", fontsize=8)

# %%
inside = (f > 100) & (f < 4000)
print("Largest difference from the cascade, 100-4000 Hz:")
print(f"  same signs        {np.abs(db(same) - db(cascade))[inside].max():5.1f} dB")
print(f"  alternating signs {np.abs(db(alternating) - db(cascade))[inside].max():5.1f} dB")

# %% [markdown]
# ## Voice quality
#
# Everything above used Klatt's 1980 source, whose spectrum is fixed. The glottal pulse itself
# changes with how the vocal folds are held: pressed together they close abruptly, which makes
# strong high harmonics; held loosely they close gradually, leaving little but the fundamental.
# The Liljencrants-Fant (LF) model (Fant, Liljencrants & Lin, 1985) describes one period of the
# glottal flow's derivative $E(t)$: an exponentially growing sinusoid while the glottis opens,
#
# $$E(t) = E_0\, e^{\alpha t} \sin(\pi t / t_p), \qquad 0 \le t \le t_e,$$
#
# down to its sharp negative peak $-E_e$ at closure $t_e$, then an exponential return to zero.
# Fant (1995) set the whole shape with one number,
#
# $$R_d = \frac{U_0}{0.11\, E_e T_0},$$
#
# the peak flow $U_0$ against the strength of the closure, in a period $T_0$: about 0.3 for a
# tense, pressed voice to 2.7 for a lax, breathy one, and close to 0.7 for typical male
# voices. `so.glottal_source` makes LF pulses from their harmonics, which have an exact
# formula, so nothing aliases, and `so.klatt_synthesize` uses them with `SS=3` (the source
# switch of Klatt & Klatt's KLSYN88) and `RD`. The `RD` control is sonore's: KLSYN88 shapes its
# LF pulse with the open quotient, the speed quotient and a spectral tilt instead.

# %% [about]
# The same peak excitation $E_e$ in all three. The tense pulse opens for a shorter part of the
# period and passes less air; the lax one stays open longer, and its closure is a gentle slope
# rather than a sharp corner. The sharp corner is what makes high harmonics.

# %% [figure fq1] One period of the LF pulse at three values of Rd
x = np.linspace(0, 1, 1000)
RDS = {0.5: "tense, Rd 0.5", 1.0: "modal, Rd 1", 2.5: "lax, Rd 2.5"}
fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 3.2), layout="constrained")
for (rd, label), color in zip(RDS.items(), ["C3", "k", "C0"], strict=True):
    ax0.plot(x, so.lf_pulse(x, rd, flow=True), color=color, label=label)
    ax1.plot(x, so.lf_pulse(x, rd), color=color, label=label)
ax0.set(xlabel="Time (fraction of a period)", ylabel="Flow (units of $E_e T_0$)")
ax0.set_title("Glottal flow")
ax1.set(xlabel="Time (fraction of a period)", ylabel="$E(t)$ (units of $E_e$)")
ax1.set_title("Its derivative: the excitation, with radiation from the lips")
ax0.legend(fontsize=8)

# %% [about]
# The harmonics' levels depend on the harmonic number only, not on $F_0$. The difference
# between the first two harmonics, H1-H2, is the usual measure of this in recordings; Fant's
# (1995) fit to it, $-7.6 + 11.1\,R_d$ dB, agrees with these spectra to within half a decibel
# up to $R_d$ 1.4. Klatt's 1980 source has the H1-H2 of a modal voice, but around 3 kHz its harmonics
# are stronger than even the tense pulse's.

# %% [figure fq2] The spectra of the three pulses, and of the 1980 source
k = np.arange(1, 41)
klatt_1980 = gain(0, 100, k * 100) * np.abs(1 - np.exp(-2j * np.pi * k * 100 / FS))
fig, ax = plt.subplots(figsize=(10, 3.4), layout="constrained")
ax.plot(
    k * 100,
    20 * np.log10(klatt_1980 / klatt_1980[0]),
    "s-",
    ms=3,
    lw=0.8,
    color="0.6",
    label="Klatt's 1980 source",
)
for (rd, label), color in zip(RDS.items(), ["C3", "k", "C0"], strict=True):
    levels = 20 * np.log10(np.abs(so.lf_harmonics(k, rd)))
    levels -= levels[0]
    ax.plot(k * 100, levels, "o-", ms=3, lw=0.8, color=color, label=f"{label}: H1-H2 {-levels[1]:.1f} dB")
ax.set(xlabel="Frequency at F0 = 100 Hz (Hz)", ylabel="Level re first harmonic (dB)", xlim=(0, 4100))
ax.set_title("Harmonics of the excitation (the flow derivative), relative to the first")
ax.legend(fontsize=8)

# %% [about]
# The vowel of "hod" three times, with everything the same except the pulse: tense (Rd 0.5),
# modal (Rd 1) and lax (Rd 2.5), with the male averages and pitch and with the female ones. The
# vowel stays the same; the voice goes from pressed and bright to soft and muffled. Since the
# pulse sets the level of each harmonic by its number, the same Rd at the higher pitch spreads the
# same levels over a wider range of frequencies: relative to its first harmonic, the female voice
# falls off more slowly with frequency.

# %% [demo fq3] One vowel, three voices
PITCHES = {"Male formants and pitch": F0, "Female formants and pitch": ([0, DUR], [230, 200])}
sounds = {}
for label, (vowels, rest) in SETS.items():
    formants = dict(zip(("F1", "F2", "F3"), vowels["hod"], strict=True), F4=rest["F4"], F5=rest["F5"])
    parts = []
    for rd in RDS:
        snd = so.klatt_synthesize(DUR, FS, F0=PITCHES[label], AV=onoff(DUR), SS=3, RD=rd, **formants)
        parts += [snd, so.silence(0.2, FS)]
    sounds[label] = finish(so.concat(parts[:-1]))
fig, playhead = show_pair(sounds, "Rd 0.5, 1, 2.5", {label: SETS[label][0]["hod"] for label in SETS})

# %% [about]
# Klatt's 1980 source, then an LF pulse at the default Rd of 0.7, close to typical male voices.
# The LF voice is darker: at 3 kHz its harmonics are about 11 dB weaker, relative to the first.

# %% [demo fq4] The 1980 source, then LF at Rd 0.7
parts = [
    so.klatt_synthesize(DUR, FS, F0=F0, AV=onoff(DUR), **hod),
    so.silence(0.2, FS),
    so.klatt_synthesize(DUR, FS, F0=F0, AV=onoff(DUR), SS=3, **hod),
]
sound = finish(so.concat(parts))
fig, playhead = show(sound, "SS=1 (the default), then SS=3 with RD=0.7", [730, 1090, 2440])

# %% [about]
# A voice relaxing at the end of a phrase: Rd rises from 0.6 to 2.4 while the pitch falls. The
# LF model is the periodic pulse only, and a lax pulse alone sounds soft rather than breathy;
# Klatt & Klatt (1990) found aspiration noise the most important cue to breathiness, so the
# aspiration (`AH`) rises with Rd. How noise mixes with the harmonics in a breathy voice is on
# [Source and aperiodicity](aperiodicity.html#d-ap3).

# %% [demo fq5] Relaxing into breathy voice
dur = 1.6
sound = finish(
    so.klatt_synthesize(
        dur,
        FS,
        F0=([0, dur], [130, 90]),
        AV=onoff(dur),
        SS=3,
        RD=([0.3, 1.3], [0.6, 2.4]),
        AH=([0.3, 1.3, dur - 0.05, dur], [0, 52, 52, 0]),
        rng=0,
        **hod,
    )
)
fig, playhead = show(sound, "Rd from 0.6 to 2.4, with aspiration rising", [730, 1090, 2440])

# %% [markdown]
# ## Finding the formants again
#
# Everything above runs one way: formants in, sound out. *Linear prediction* (LPC) runs the
# other way. It predicts each sample from the $p$ before it,
#
# $$x[n] \approx -\left(a_1 x[n-1] + \dots + a_p x[n-p]\right),$$
#
# with the weights chosen to make the error as small as possible over a short time window. The
# predictor is a filter: the window is modeled as white noise through $1/A(z)$, with
# $A(z) = 1 + a_1 z^{-1} + \dots + a_p z^{-p}$. That is an all-pole filter like the cascade of
# resonators above, and each pair of roots of $A(z)$, at radius $r$ and angle $\theta$, is a
# resonance at $\theta f_s / 2\pi$ Hz, $-\ln r \, f_s / \pi$ Hz wide: Klatt's resonator read
# backward (its poles sit at radius $e^{-\pi B_k T}$ and angle $2\pi F_k T$). `so.LPC` fits the
# predictor to every time window, and `so.formant_track` follows Praat's recipe for formants
# (resample to twice a ceiling, 5000 Hz for a male voice and 5500 Hz for a female one, fit ten
# poles, keep every root) and then links the roots of neighboring windows into F1, F2 and F3.
#
# On a synthesized vowel the answer is known, so the estimate can be checked. The numbers below
# come from the page's own sounds; docs/design/views/lpc.md measures them more widely.

# %%
from scipy.signal import lfilter


def preemphasize(snd):
    """y[n] = x[n] - 0.97 x[n-1]: lifts the highs so the poles go to the formants, not the
    falling slope of the source."""
    return so.Sound(lfilter([1, -0.97], 1, snd.data, axis=0), snd.fs)


def peaks(freqs, level_db):
    """Frequencies of the local maxima of an envelope, refined by a parabola, 90 Hz to 5 kHz."""
    middle = level_db[1:-1]
    index = np.flatnonzero((middle > level_db[:-2]) & (middle >= level_db[2:])) + 1
    before, at, after = level_db[index - 1], level_db[index], level_db[index + 1]
    found = freqs[index] + 0.5 * (before - after) / (before - 2 * at + after) * (freqs[1] - freqs[0])
    return found[(found > 90) & (found < 5000)]


# %% [about]
# The vowel of "hod" at 100 Hz from `so.klatt_synthesize`, one time window in its middle. The
# dots are its harmonics; the dashed lines are the formants it was given. Four envelopes are
# drawn over them, each moved up or down to sit on the harmonics: LPC (order 18), CheapTrick, the
# cepstrum liftered at half a period, and thirteen MFCCs, all from the same pre-emphasized
# sound. LPC follows the peaks closely. Its roots are not all formants: with 18 poles at 16 kHz
# some fit the slope of the source or the harmonics instead, which is why `so.formant_track`
# fits ten poles below a 5000 Hz ceiling and then picks among them. The triangles are its F1 to
# F3, within a few hertz of the given formants (the cell prints them). The cepstral envelope
# keeps some ripple from the harmonics, so it has extra peaks below F1; thirteen MFCCs smooth F1
# and F2 into one hump. MFCCs are not meant to find formants; the figure shows what each
# envelope keeps.

# %% [figure ft1] One vowel, four envelopes, and the formants LPC finds
raw = so.klatt_synthesize(0.5, FS, F0=100, **hod)
vowel = preemphasize(raw)
time = 0.25
freqs = np.arange(0, 5001, 2.0)
envelopes = {}
lpc = so.LPC(vowel, win_dur=0.040, hop_dur=0.005)
column = int(np.argmin(np.abs(lpc.t - time)))
envelopes["LPC"] = lpc.envelope(freqs)[0, :, column]
envelopes["CheapTrick"] = so.cheaptrick(vowel, ([time], [100.0]))([time], freqs)[0, :, 0]
stft = so.STFT(vowel, win_dur=0.040, hop_dur=0.005)
cepstrum = so.Cepstrum(stft).lifter(0.5 / 100)
cep_freqs = np.arange(cepstrum.n_fft // 2 + 1) * FS / cepstrum.n_fft
envelopes["cepstrum"] = np.interp(freqs, cep_freqs, cepstrum.envelope()[0, :, column] ** 2)
mfcc = so.MFCC(vowel, triangles="area", hop_dur=0.005)
envelopes["13 MFCCs"] = mfcc.envelope(freqs)[0, :, int(np.argmin(np.abs(mfcc.t - time)))]

spectrum = np.abs(np.fft.rfft(vowel.data[FS // 4 - 320 : FS // 4 + 320, 0] * np.hanning(640), 8192)) ** 2
spectrum_freqs = np.fft.rfftfreq(8192, 1 / FS)
harmonic_freqs = np.arange(1, 50) * 100.0
harmonic_db = 10 * np.log10(np.interp(harmonic_freqs, spectrum_freqs, spectrum))

fig, ax = plt.subplots(figsize=(10, 3.6), layout="constrained")
ax.plot(harmonic_freqs, harmonic_db, "o", ms=3, color="k", label="harmonics")
for (name, envelope), color in zip(envelopes.items(), ["C3", "C0", "C2", "C4"], strict=True):
    level_db = 10 * np.log10(envelope)
    shift = np.mean(harmonic_db - np.interp(harmonic_freqs, freqs, level_db))
    ax.plot(freqs, level_db + shift, color=color, lw=1.2 if name == "LPC" else 0.9, label=name)
    found = peaks(freqs, level_db)[:3]
    print(f"{name:10} first three peaks: " + ", ".join(f"{f:.0f}" for f in found) + " Hz")
roots, _ = lpc.candidates()
print("Order-18 LPC roots: " + ", ".join(f"{f:.0f}" for f in roots[0, :, column] if f < 5000) + " Hz")
track = so.formant_track(raw)  # pre-emphasizes inside, as Praat does
lpc_formants = track.frequencies[0][:, int(np.argmin(np.abs(track.t - time)))]
print("so.formant_track: " + ", ".join(f"{f:.0f}" for f in lpc_formants) + " Hz (given 730, 1090, 2440)")
for formant, _ in HOD[:3]:
    ax.axvline(formant, color="0.5", ls="--", lw=0.8)
ax.plot(
    lpc_formants,
    np.interp(lpc_formants, freqs, 10 * np.log10(envelopes["LPC"])) + 3,
    "v",
    color="C3",
    ms=6,
    label="so.formant_track",
)
ax.set(xlabel="Frequency (Hz)", ylabel="Level (dB)", xlim=(0, 5000))
ax.set_ylim(harmonic_db.max() - 70, harmonic_db.max() + 8)
ax.set_title("The vowel of 'hod' at 100 Hz: four envelopes, and the formants given (dashed)")
ax.legend(loc="upper right", fontsize=8, ncols=3)

# %% [about]
# The twelve vowels of [Six vowels](#h-six-vowels), each run through `so.formant_track` (ceiling
# 5000 Hz for the male set, 5500 Hz for the female one) and averaged over its middle. Each arrow
# goes from the formants given to the formants found. The male vowels come back within a few
# percent; the female ones less closely, because with the pitch twice as high the harmonics
# sample the envelope half as often, and LPC is pulled toward the harmonics nearest a formant.
# The cell prints the largest errors.

# %% [figure ft2] Given and recovered formants, male and female
fig, ax = plt.subplots(figsize=(5.4, 4.2), layout="constrained")
for (label, (vowels, rest)), color, ceiling in zip(SETS.items(), ["C0", "C1"], [5000, 5500], strict=True):
    errors = []
    for f1, f2, f3 in vowels.values():
        snd = so.klatt_synthesize(
            0.5, FS, F0=rest["F0"][1][0], F1=f1, F2=f2, F3=f3, F4=rest["F4"], F5=rest["F5"]
        )
        track = so.formant_track(snd, ceiling=ceiling)
        middle = (track.t > 0.1) & (track.t < 0.4)
        found = np.median(track.frequencies[0][:, middle], axis=1)
        errors.append(np.abs(found - (f1, f2, f3)) / (f1, f2, f3))
        ax.annotate("", (found[1], found[0]), (f2, f1), arrowprops=dict(arrowstyle="->", color=color, lw=0.8))
        ax.plot(f2, f1, "o", color=color, ms=4)
    errors = np.array(errors)
    print(
        f"{label.split()[0]}: largest error {100 * errors.max():.1f}%, median {100 * np.median(errors):.1f}%"
    )
    ax.plot([], [], "o-", color=color, ms=4, label=label.split()[0].lower())
ax.set(xlabel="F2 (Hz)", ylabel="F1 (Hz)", xlim=(3000, 600), ylim=(950, 200))
ax.legend(loc="lower left", fontsize=8)
ax.set_title("Given (dots) and found by so.formant_track (arrow tips)")


# %%
def draw_tracks(axes, track, voiced=None):
    """F1 to F3 from a FormantTrack over a spectrogram panel (in kHz), voiced time windows only."""
    keep = np.ones(len(track.t), bool) if voiced is None else voiced
    for number in range(3):
        axes.plot(track.t, np.where(keep, track.frequencies[0, number], np.nan) / 1000, color="w", lw=1.4)
        axes.plot(track.t, np.where(keep, track.frequencies[0, number], np.nan) / 1000, color="C3", lw=0.8)


# %% [about]
# The vowel of "hod" with the pitch gliding from 100 to 300 Hz. The dashed lines are the formants
# it was given, the red ones what `so.formant_track` finds. The error grows with the pitch: the
# cell prints its median over each quarter of the glide, about 10 Hz at the bottom and 30 Hz at
# the top, with single windows off by up to 50 Hz. This is the main limit of LPC for formants:
# it fits the harmonics, and with few of them under a formant it is pulled toward the harmonic
# nearest it.

# %% [demo ft3] The estimate as the pitch rises
dur = 2.0
glide = so.klatt_synthesize(dur, FS, F0=([0, dur], [100, 300]), AV=onoff(dur), **hod)
track = so.formant_track(glide)
pitch_at = 100 + 200 * track.t / dur
for low in (100, 150, 200, 250):
    inside = (pitch_at >= low) & (pitch_at < low + 50) & (track.t > 0.05) & (track.t < dur - 0.05)
    error = np.median(np.abs(track.frequencies[0][:, inside] - np.array([[730], [1090], [2440]])), axis=1)
    print(
        f"F0 {low}-{low + 50} Hz: median error of F1, F2, F3 " + ", ".join(f"{e:.0f}" for e in error) + " Hz"
    )
sound = finish(glide)
fig, playhead = show(sound, "hod, F0 gliding from 100 to 300 Hz", [730, 1090, 2440])
draw_tracks(playhead[1], track)
playhead[1].set_title("Spectrogram (Hann 6 ms): formants given (dashed) and found by so.formant_track (red)")

# %% [markdown]
# ### A recording
#
# The same analysis on the two talkers the other pages use, saying "Author of the danger trail,
# Philip Steels, etc." (CMU ARCTIC; sources in docs/speech/SOURCES.md). Formants are only
# meaningful where the voice is voiced, so `so.f0_track` decides where to draw them. There is no
# answer to check against here, only Praat: the tracks agree with Praat's own formants to a few
# hertz in the median (the tests check a stretch of each sentence).

# %%
SPEAKERS = {"Male talker": ("bdl", 5000), "Female talker": ("slt", 5500)}
talkers, formant_tracks, voicing, pitch = {}, {}, {}, {}
for label, (speaker, ceiling) in SPEAKERS.items():
    talkers[label] = so.load(fetch(f"docs/speech/{speaker}_arctic_a0131.flac"))
    formant_tracks[label] = so.formant_track(talkers[label], ceiling=ceiling)
    pitch[label] = so.f0_track(talkers[label])
    on_grid = np.interp(formant_tracks[label].t, pitch[label].t, pitch[label].voiced[0].astype(float))
    voicing[label] = on_grid > 0.5

# %% [about]
# Both sentences, with F1 to F3 drawn where the voice is voiced. The female talker's formants
# are higher, and her tracks are less steady: with her harmonics farther apart there is less
# envelope for LPC to fit, as in the glide above.

# %% [demo ft4] Formant tracks of two talkers
fig = plt.figure(figsize=(10, 3.4), layout="constrained")
playhead = {}
for column, (label, snd) in zip(fig.subfigures(1, 2), talkers.items(), strict=True):
    ax = column.subplots()
    so.STFT(snd, win_dur=0.006, hop_dur=0.001).plot(ax, db_range=60, colorbar=False, fmax=5000)
    draw_tracks(ax, formant_tracks[label], voicing[label])
    ax.set(xlim=(0, snd.duration), title=f"{label}: F1 to F3 from so.formant_track")
    playhead[label] = [ax]
sounds = {label: finish(snd) for label, snd in talkers.items()}

# %% [markdown]
# ### Copy synthesis
#
# With formant tracks and a pitch track, the synthesizer can be driven from a recording rather
# than by hand. Here `so.klatt_synthesize` gets, every 5 ms, the measured F1 to F3 (with a
# five-window median to take out single-window jumps), the measured $F_0$ while voiced, a
# voicing level that follows the recording's level, and, where the voice is unvoiced,
# aspiration noise at that level through the same tracked formants (none where the recording
# is more than 35 dB below its loudest). F4 and F5, the bandwidths
# and the source are the synthesizer's defaults. This is a crude copy: it keeps the formants,
# the pitch and the level, and drops everything else, including the shape of the fricatives
# and bursts, which go through the formants as breath noise.

# %%
from scipy.ndimage import median_filter


def copy_synthesis(snd, track, f0_track, voiced):
    """Klatt parameters from measured tracks: F1-F3 (median of 5 windows), F0 where voiced, AV
    following the level in dB where voiced, AH at that level where unvoiced and not silent."""
    times = track.t
    inside = (times >= 0) & (times <= snd.duration)
    times, voiced = times[inside], voiced[inside]
    formants = median_filter(np.nan_to_num(track.frequencies[0][:, inside], nan=1000.0), size=(1, 5))
    samples, window = snd.data[:, 0], round(0.025 * snd.fs)
    starts = np.clip(np.round(times * snd.fs).astype(int) - window // 2, 0, len(samples) - window)
    level = np.array([np.sqrt(np.mean(samples[s : s + window] ** 2)) for s in starts]) + 1e-12
    level_db = 20 * np.log10(level / level.max())
    f0 = np.interp(times, f0_track.t, np.where(f0_track.voiced[0], f0_track.f0[0], 0.0))
    return so.klatt_synthesize(
        snd.duration,
        snd.fs,
        F0=(times, np.where(voiced, f0, 0.0)),
        AV=(times, np.where(voiced, np.maximum(60 + level_db, 0), 0)),
        AH=(times, np.where(voiced | (level_db < -35), 0, 55 + level_db)),
        F1=(times, formants[0]),
        F2=(times, formants[1]),
        F3=(times, formants[2]),
        rng=0,
    )


copies = {
    label: copy_synthesis(snd, formant_tracks[label], pitch[label], voicing[label])
    for label, snd in talkers.items()
}

# %% [about]
# The male talker: the recording, then the copy. The formant movements and the melody carry
# over; the voice quality, the consonants and the higher formants do not. The copy is also
# darker above 1 kHz, most likely because the synthesizer's default source falls off faster
# than this talker's voice. How intelligible the
# copy is has not been measured.

# %% [demo ft5] The male talker and his copy
fig = plt.figure(figsize=(10, 3.4), layout="constrained")
playhead = {}
pair = {"Recording": talkers["Male talker"], "Copy": copies["Male talker"]}
for column, (label, snd) in zip(fig.subfigures(1, 2), pair.items(), strict=True):
    ax = column.subplots()
    so.STFT(snd, win_dur=0.006, hop_dur=0.001).plot(ax, db_range=60, colorbar=False, fmax=5000)
    ax.set(xlim=(0, snd.duration), title=f"Male talker: {label.lower()}")
    playhead[label] = [ax]
sounds = {label: finish(snd) for label, snd in pair.items()}

# %% [about]
# The female talker and her copy, made the same way.

# %% [demo ft6] The female talker and her copy
fig = plt.figure(figsize=(10, 3.4), layout="constrained")
playhead = {}
pair = {"Recording": talkers["Female talker"], "Copy": copies["Female talker"]}
for column, (label, snd) in zip(fig.subfigures(1, 2), pair.items(), strict=True):
    ax = column.subplots()
    so.STFT(snd, win_dur=0.006, hop_dur=0.001).plot(ax, db_range=60, colorbar=False, fmax=5000)
    ax.set(xlim=(0, snd.duration), title=f"Female talker: {label.lower()}")
    playhead[label] = [ax]
sounds = {label: finish(snd) for label, snd in pair.items()}

# %% [markdown]
# ## What this page leaves out
#
# - **The rest of voice quality.** KLSYN88 (Klatt & Klatt, 1990) has more voice controls than
#   the LF pulse: open quotient, a spectral tilt filter, flutter (slow random jitter of $F_0$)
#   and double pulsing, and its own polynomial pulse (KLGLOTT88). sonore has the LF pulse only.
# - **A better copy.** [Copy synthesis](#d-ft5) above keeps three formants, the pitch
#   and the level. A fuller copy would also fit F4, the bandwidths, the voicing source and the
#   parallel branch for the consonants. [Rebuilding and changing a
#   voice](voice.html) rebuilds a recording from its measured pitch and whole envelope instead,
#   which keeps far more of it.
# - **Rules.** Text-to-speech systems drive formant synthesizers from phonetic rules, which
#   generate the tracks from a transcription. Here every track is set by hand.
# - **Drawing the tracks.** Every track here is written in code. [sonore-sketch](https://choyun1.github.io/sonore-sketch/),
#   a separate browser app built on sonore, draws formant, $F_0$, voicing and bandwidth tracks
#   with a mouse and plays what the synthesizer makes of them.

# %% [markdown]
# ## References
#
# - Klatt (1980). Software for a cascade/parallel formant synthesizer. *J. Acoust. Soc. Am.*
#   67(3), 971–995. [doi:10.1121/1.383940](https://doi.org/10.1121/1.383940).
#   [`klatt.klatt_synthesize`](https://github.com/choyun1/sonore/blob/main/src/sonore/sources/klatt.py#L103)
#   [`processing.resonator`](https://github.com/choyun1/sonore/blob/main/src/sonore/core/processing.py#L232)
# - Fant (1995). The LF-model revisited. Transformations and frequency domain analysis.
#   *STL-QPSR* 36(2–3), 119–156. The Rd parameter.
#   [`waveforms.glottal_source`](https://github.com/choyun1/sonore/blob/main/src/sonore/sources/waveforms.py#L863)
# - Fant, Liljencrants & Lin (1985). A four-parameter model of glottal flow. *STL-QPSR* 26(4),
#   1–13. [`waveforms.lf_harmonics`](https://github.com/choyun1/sonore/blob/main/src/sonore/sources/waveforms.py#L784)
# - Klatt & Klatt (1990). Analysis, synthesis, and perception of voice quality variations among
#   female and male talkers. *J. Acoust. Soc. Am.* 87(2), 820–857.
#   [doi:10.1121/1.398894](https://doi.org/10.1121/1.398894). The `SS` source switch.
# - Makhoul (1975). Linear prediction: A tutorial review. *Proc. IEEE* 63(4), 561–580.
#   [doi:10.1109/PROC.1975.9792](https://doi.org/10.1109/PROC.1975.9792).
#   [`lpc.LPC`](https://github.com/choyun1/sonore/blob/main/src/sonore/views/lpc.py#L69)
# - Boersma & Weenink. Praat: doing phonetics by computer (computer program).
#   [praat.org](https://www.praat.org). Its formant recipe, which `so.formant_track` follows.
#   [`lpc.formant_track`](https://github.com/choyun1/sonore/blob/main/src/sonore/views/lpc.py#L365)
# - Peterson & Barney (1952). Control methods used in a study of the vowels. *J. Acoust. Soc. Am.*
#   24(2), 175–184. [ASA](https://pubs.aip.org/asa/jasa/article/24/2/175/722376/Control-Methods-Used-in-a-Study-of-the-Vowels).
#   The vowels' formant frequencies.
