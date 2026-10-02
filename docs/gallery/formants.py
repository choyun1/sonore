"""Formant synthesis: speech made from a source, resonances and a handful of numbers.

This script is the gallery page https://choyun1.github.io/sonore/gallery/formants.html:
docs/gallery/build.py runs it cell by cell from the repository root and shows each
cell's code beside what it made. Run it yourself from the repository root,

    python docs/gallery/formants.py

or a cell at a time ("# %%" starts a cell in VS Code, Spyder and Jupytext).
"""

# %% [markdown]
# # Formant synthesis
#
# A voice is a source filtered by the vocal tract. In a formant synthesizer every part of that
# sentence is a number: the source's pitch and level, and the frequency and bandwidth of each
# resonance (formant). `so.klatt_synthesize` follows Klatt (1980). Voicing is a set of harmonics
# of $F_0$ with the spectrum of Klatt's glottal pulse, aspiration and frication are noise, and
# formants are second-order resonators
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
# - [Six vowels](#h-six-vowels): Peterson and Barney's averages, and a continuum between two.
# - [Consonants from transitions](#h-consonants-from-transitions): /ba/, /da/ and /ga/ differ
#   only in where the formants start.
# - [Noise](#h-noise): frication, aspiration and breathy voice.
# - [Cascade and parallel](#h-cascade-and-parallel): why the parallel formants alternate in sign.
# - [What this page leaves out](#h-what-this-page-leaves-out): voice quality, and copying a
#   recording.

# %% [markdown]
# ## Code the examples share
#
# All the sounds are at 16 kHz, a little longer than a syllable. Each one is drawn as a waveform
# and a wideband spectrogram (Hann 6 ms), which shows formants as dark bands, with the formant
# frequencies the synthesizer was given drawn over it.

# %%
import matplotlib.pyplot as plt
import numpy as np

import sonore as so

plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "figure.dpi": 100})
FS = 16000


def finish(snd):
    """How every sound in the gallery is played: 5 ms ramps, RMS 0.1, peak at most 0.95."""
    snd = snd.ramp(5e-3).normalize(rms=0.1)
    return snd.normalize(peak=0.95) if snd.peak > 0.95 else snd


def onoff(dur, level=60, on=0.02, off=0.05):
    """An amplitude track: up over `on` seconds, down over the last `off` seconds."""
    return ([0, on, dur - off, dur], [0, level, level, 0])


def show(snd, title, formants=()):
    """Waveform and wideband spectrogram, with formant tracks ((times, values) or numbers) drawn over it."""
    fig = plt.figure(figsize=(10, 4.2), layout="constrained")
    ax0, ax1 = fig.subplots(2, 1, sharex=True, height_ratios=[0.45, 1])
    snd.plot(ax0, color="k", lw=0.4)
    ax0.set(title=f"Waveform: {title}", xlabel="")
    frame = so.GaborFrame(0.006, 0.001, n_fft=512)  # zero-padded for smooth bands
    so.STFT(snd, frame=frame).plot(ax1, db_range=60, colorbar=False, fmax=5000)
    ax1.set_title("Spectrogram (Hann 6 ms), with the formant frequencies given to the synthesizer")
    for f in formants:
        t, v = ([0], [f]) if np.isscalar(f) else f  # a track is held after its last point
        ax1.plot([*t, snd.duration], np.array([*v, v[-1]]) / 1000, color="c", lw=1, ls="--")  # in kHz
    for ax in (ax0, ax1):
        ax.set_xlim(0, snd.duration)
    return fig, [ax0, ax1]


# %% [markdown]
# ## A vowel, piece by piece
#
# The vowel of "hod", with Peterson and Barney's (1952) average formants for men: 730, 1090 and
# 2440 Hz. The pitch falls from 130 to 100 Hz, as at the end of a statement. Here the pieces are
# put together by hand with `so.resonator`, the same resonator `so.klatt_synthesize` uses.

# %%
DUR = 0.6
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
# sound is recognisably the vowel of "hod", if thin.

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
# Peterson and Barney (1952) measured the formants of ten American English vowels, spoken in
# words of the form h-vowel-d by men, women and children. Their averages for men, for six of
# the vowels, synthesized with everything else the same: the pitch, the higher formants and the
# bandwidths.

# %%
VOWELS = {
    "heed": (270, 2290, 3010),
    "head": (530, 1840, 2480),
    "had": (660, 1720, 2410),
    "hod": (730, 1090, 2440),
    "hawed": (570, 840, 2410),
    "who'd": (300, 870, 2240),
}
gap = so.silence(0.15, FS)
parts, starts = [], []
t = 0.0
for f1, f2, f3 in VOWELS.values():
    v = so.klatt_synthesize(0.35, FS, F0=([0, 0.35], [125, 105]), AV=onoff(0.35), F1=f1, F2=f2, F3=f3)
    parts += [v, gap]
    starts.append(t)
    t += 0.5

# %% [about]
# The six vowels in turn, from "heed" to "who'd". The dashed lines are each vowel's F1, F2 and F3.

# %% [demo fw1] Six vowels
sound = finish(so.concat(parts))
tracks = [
    (np.repeat(starts, 2) + np.tile([0, 0.35], 6), np.repeat([v[i] for v in VOWELS.values()], 2))
    for i in range(3)
]
fig, playhead = show(sound, "heed, head, had, hod, hawed, who'd")
for tt, ff in tracks:
    for j in range(6):
        playhead[1].plot(tt[2 * j : 2 * j + 2], ff[2 * j : 2 * j + 2] / 1000, color="c", lw=1, ls="--")

# %% [figure fw2] The vowels by their first two formants
fig, ax = plt.subplots(figsize=(5, 4), layout="constrained")
for word, (f1, f2, _) in VOWELS.items():
    ax.plot(f2, f1, "o", color="C0")
    ax.annotate(word, (f2, f1), textcoords="offset points", xytext=(6, 4))
ax.set(xlabel="F2 (Hz)", ylabel="F1 (Hz)", xlim=(2500, 600), ylim=(800, 200))
ax.set_title("F1 against F2, axes reversed as is usual:\nthe tongue's height and backness")

# %% [about]
# A continuum from "heed" to "hod" in seven equal steps of F1, F2 and F3, made with
# `so.klatt_continuum`. Somewhere in the middle the vowel stops being one and becomes the
# other; experiments on vowel categories use continua like this one.

# %% [demo fw3] From heed to hod in seven steps
steps = so.klatt_continuum(dict(F1=270, F2=2290, F3=3010), dict(F1=730, F2=1090, F3=2440), 7)
parts = []
for p in steps:
    parts += [so.klatt_synthesize(0.3, FS, p, F0=110, AV=onoff(0.3)), so.silence(0.15, FS)]
sound = finish(so.concat(parts))
fig, playhead = show(sound, "seven steps from heed to hod")
for i, p in enumerate(steps):
    for name in ("F1", "F2", "F3"):
        playhead[1].plot([0.45 * i, 0.45 * i + 0.3], [p[name] / 1000] * 2, color="c", lw=1, ls="--")

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
# /sa/: 150 ms of frication through the parallel formants 5 and 6 (4500 and 5500 Hz), then the
# vowel.

# %% [demo fn1] The syllable sa
dur = 0.6
tracks = dict(F1=([0.19, 0.25], [300, 730]), F2=([0.19, 0.25], [1500, 1090]))
sound = finish(
    so.klatt_synthesize(
        dur,
        FS,
        F0=([0, dur], [125, 100]),
        AV=([0, 0.17, 0.19, dur - 0.05, dur], [0, 0, 60, 60, 0]),
        AF=([0, 0.03, 0.15, 0.19], [0, 74, 74, 0]),
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
# are already there in the breath.

# %% [demo fn2] The syllable ha
sound = finish(
    so.klatt_synthesize(
        dur,
        FS,
        F0=([0, dur], [125, 100]),
        AV=([0, 0.17, 0.2, dur - 0.05, dur], [0, 0, 60, 60, 0]),
        AH=([0, 0.03, 0.17, 0.21], [0, 62, 62, 0]),
        rng=0,
        **hod,
    )
)
fig, playhead = show(sound, "/ha/: aspiration, then voicing", [730, 1090, 2440])

# %% [about]
# A breathy "hod": voicing with aspiration 6 dB below it throughout, and a wider first formant.
# The noise fills in between the harmonics, most at high frequencies where the harmonics are
# weak.

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
# level of its own. Added with the same sign, neighbouring formants partly cancel between their
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
# ## What this page leaves out
#
# - **Voice quality.** The glottal source here is Klatt's 1980 one, with a fixed spectral slope.
#   His later KLSYN88 (Klatt & Klatt, 1990) adds a glottal pulse model with controls for voice
#   quality, to match the differences between female and male voices and breathy ones.
# - **Copying a recording.** The parameters here were written by hand. Copy synthesis fits
#   formant tracks to a recording, which needs a formant tracker that sonore does not have; the
#   [Voices from harmonics](harmonics.html) page rebuilds a recording from its measured pitch and
#   envelope instead.
# - **Rules.** Text-to-speech systems drive formant synthesizers from phonetic rules, which
#   generate the tracks from a transcription. Here every track is set by hand.

# %% [markdown]
# ## References
#
# - Klatt (1980). Software for a cascade/parallel formant synthesizer. *J. Acoust. Soc. Am.*
#   67(3), 971–995. [doi:10.1121/1.383940](https://doi.org/10.1121/1.383940).
#   [`klatt.klatt_synthesize`](https://github.com/choyun1/sonore/blob/main/src/sonore/stimuli/klatt.py#L101)
#   [`processing.resonator`](https://github.com/choyun1/sonore/blob/main/src/sonore/signals/processing.py#L188)
# - Klatt & Klatt (1990). Analysis, synthesis, and perception of voice quality variations among
#   female and male talkers. *J. Acoust. Soc. Am.* 87.
#   [doi:10.1121/1.398894](https://doi.org/10.1121/1.398894).
# - Peterson & Barney (1952). Control methods used in a study of the vowels. *J. Acoust. Soc. Am.*
#   24(2), 175–184. [ASA](https://pubs.aip.org/asa/jasa/article/24/2/175/722376/Control-Methods-Used-in-a-Study-of-the-Vowels).
#   The vowels' formant frequencies.
