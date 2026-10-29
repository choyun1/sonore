"""Pitch of complex tones: sonore's reproductions of the classic pitch demonstrations.

This script is the gallery page https://choyun1.github.io/sonore/gallery/complexpitch.html:
docs/gallery/build.py runs it cell by cell from the repository root and shows each
cell's code beside what it made. Run it yourself from the repository root,

    python docs/gallery/demos/complexpitch.py

or a cell at a time ("# %%" starts a cell in VS Code, Spyder and Jupytext).
"""

# %% [markdown]
# # Pitch of complex tones: reproducing Houtsma, Rossing & Wagenaars
#
# Every sound on this page is sonore's reproduction of a demonstration from the Acoustical
# Society of America's *Auditory Demonstrations* (Houtsma, Rossing & Wagenaars, 1987), the
# compact disc made at the Institute for Perception Research (IPO) in Eindhoven. The
# demonstrations, their design and what they show are their authors' work; their booklet is
# published by the Eindhoven University of Technology
# ([TU/e](https://research.tue.nl/en/publications/auditory-demonstrations)).
#
# **What they show.** A tone made of many harmonics of one fundamental is heard with a single
# pitch, the pitch of the fundamental, even though no one harmonic carries it alone. Take the
# fundamental away and the pitch stays: it is heard from the pattern the other harmonics make,
# a *virtual pitch*. Shift every harmonic by the same amount and the pitch moves by less than any
# of them, as if hearing looked for the nearest fundamental that fits. A sound added to a delayed
# copy of itself has a pitch at one over the delay, and two tones heard together can make beats
# and tones that are not in the sound at all. The examples below build each case from the
# booklet's numbers.
#
# This page is a test of sonore: can its generators rebuild each stimulus from the description
# in the booklet? Each example names the demonstration it reproduces, rebuilds it from the
# booklet's frequencies, levels and timing, and says where it departs from them: where the
# booklet leaves a value open, or where a long track is shortened. What listeners hear is
# reported as the booklet reports it; this page has not tested any of it. Related pages:
# [Pitch class and height](chroma.html) has the Shepard and Risset tones (Demonstration 27), and
# [Iterated rippled noise](irn.html) carries repetition pitch further.
#
# - [How long a pitch takes](#h-how-long-a-pitch-takes): tone bursts from one period to 128
#   (Demonstration 13), and matching an octave (Demonstration 15).
# - [Pitch without a fundamental](#h-pitch-without-a-fundamental): harmonics removed from the
#   bottom, partials shifted off the harmonic series, and two ways to hear a two-tone change
#   (Demonstrations 20, 21 and 25).
# - [Melodies of virtual pitch](#h-melodies-of-virtual-pitch): a tune that noise cannot mask where
#   its pitch would be, and the same tune on random harmonics (Demonstrations 22 and 23).
# - [Pitch from a repetition](#h-pitch-from-a-repetition): scales played by the delay between two
#   clicks or two copies of a noise (Demonstration 26).
# - [Beats and combination tones](#h-beats-and-combination-tones): beats between tones close to a
#   simple ratio, and tones the ear adds (Demonstrations 32 and 34).

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


# %% [markdown]
# ## Code the examples share
#
# Tones are placed at their onset times, as on the [Streams](streams.html) page. Like every
# sonore generator, `so.pure_tone` and `so.harmonic_complex` scale the sound they make to RMS 1,
# so a pure tone here has RMS 1 and `partials` gives each partial of a complex tone RMS 1 too;
# the booklet's level relations are set from there, and `finish` keeps them, scaling only the
# whole sound. The sample rate is 22.05 kHz.

# %%
FS = 22050


def envelope(n, rise, decay):
    """Linear rise and decay ramps, in seconds, over `n` samples."""
    i = np.arange(n)
    return np.clip(np.minimum(i / (rise * FS), (n - 1 - i) / (decay * FS)), 0, 1)


def ramped(snd, ramp=20e-3):
    """`snd` with linear ramps at both ends."""
    return so.Sound(envelope(snd.n_samples, ramp, ramp) * snd.data[:, 0], FS)


def tone(freq, dur, ramp=20e-3):
    """A sine tone at RMS 1 (before its ramps), `dur` seconds long including linear ramps."""
    return ramped(so.pure_tone(dur, FS, freq, phase=-np.pi / 2), ramp)


def partials(freqs, dur, ramp=20e-3):
    """Pure tones at `freqs` added, each at RMS 1, with linear ramps."""
    return ramped(so.Sound(sum(so.pure_tone(dur, FS, f).data[:, 0] for f in freqs), FS), ramp)


def harmonics(f0, numbers, dur, ramp=20e-3):
    """Harmonics `numbers` of `f0` in sine phase, each at RMS 1, with linear ramps. One call to
    `so.harmonic_complex`, which makes them at RMS 1 together, scaled back up."""
    snd = so.harmonic_complex(dur, FS, f0, harmonics=numbers, phases="sine", ramp=0)
    return ramped(snd * np.sqrt(len(numbers)), ramp)


def place(events, dur=None):
    """Sounds added at their onset times: `events` is a list of (onset in seconds, Sound)."""
    end = max(onset + snd.duration for onset, snd in events)
    out = np.zeros((round((dur or end) * FS) + 1, 1))
    for onset, snd in events:
        i = round(onset * FS)
        out[i : i + snd.n_samples] += snd.data[: len(out) - i]
    return so.Sound(out, FS)


def noise(dur, band, rms_db, seed):
    """Gaussian noise in `band` (Hz), at an RMS `rms_db` dB re a pure tone's (RMS 1)."""
    rng = np.random.default_rng(seed)
    return so.gaussian_noise(dur, FS, band=band, rng=rng).gain_db(rms_db)


def show(snd, fmin=50, fmax=5000, title="Spectrogram (Hann 30 ms)", log=True, win=30e-3):
    """A spectrogram, on a log frequency axis unless `log` is False. Returns the figure and the
    panels the playhead follows."""
    fig, ax = plt.subplots(figsize=(10, 3.2), layout="constrained")
    stft = so.STFT(snd, win_dur=win, hop_dur=5e-3)
    keep = (stft.f >= fmin) & (stft.f <= fmax)
    top = stft.db[:, keep].max()
    ax.pcolormesh(stft.t, stft.f[keep], stft.db[0][keep], vmin=top - 60, vmax=top, cmap="magma")
    ax.set(yscale="log" if log else "linear", ylim=(fmin, fmax), ylabel="Frequency [Hz]", title=title)
    ax.set(xlabel="Time [s]", xlim=(0, snd.duration))
    return fig, [ax]


# %% [markdown]
# ## How long a pitch takes
#
# A pitch needs some time to form, and the pitch heard is not always the one a frequency ratio
# predicts.

# %% [about]
# **Listen for** the moment a click turns into a tone. Each of the three series plays bursts of
# one tone lasting 1, 2, 4 and on up to 128 periods: first at 300 Hz, then 1000 Hz, then 3000 Hz.
# The booklet asks how many periods it takes to establish a sense of pitch, and describes the
# shortest bursts as clicks that take on a pitch as they lengthen.
#
# Houtsma, Rossing & Wagenaars, Demonstration 13, after Doughty & Garner (1948). Bursts of 1, 2,
# 4, 8, 16, 32, 64 and 128 periods of 300, 1000 and 3000 Hz, as in the booklet, each starting and
# ending at a zero crossing of a sine with no ramps. Ours: the bursts start 0.6 s apart, with
# 1 s between the series, which are played once where the disc plays them twice.

# %% [demo pk1] Tone bursts from one period to 128
events, start = [], 0.0
for freq in (300, 1000, 3000):
    for k, periods in enumerate(2 ** np.arange(8)):
        burst = so.pure_tone(periods / freq, FS, freq, phase=-np.pi / 2)
        events.append((start + 0.6 * k, burst))
    start += 0.6 * 8 + 1.0
sound = finish(place(events))
fig, playhead = show(sound, fmin=100, fmax=8000, win=10e-3, title="Spectrogram (Hann 10 ms)")

# %% [about]
# **Listen for** the pair that sounds like a true octave. A 500 Hz tone alternates with a
# second tone that steps up from 985 to 1035 Hz, 5 Hz at a time, so that the exact octave,
# 1000 Hz, is the fourth pair. The booklet expects most listeners to choose a tone somewhere
# around 1010 Hz, an octave a little wider than 2 to 1.
#
# Houtsma, Rossing & Wagenaars, Demonstration 15, after Ward (1954). A 500 Hz tone of 1 s
# alternating with a 1 s tone from 985 to 1035 Hz in 5 Hz steps, as in the booklet. Ours: equal
# levels, 20 ms ramps, 0.2 s between the tones of a pair and 0.5 s between pairs, and one pass
# where the disc plays two.

# %% [demo pk2] Matching an octave
events = []
for k, upper in enumerate(np.arange(985, 1036, 5)):
    events += [(2.7 * k, tone(500, 1.0)), (2.7 * k + 1.2, tone(upper, 1.0))]
sound = finish(place(events))
fig, playhead = show(sound, fmin=300, fmax=1500)

# %% [markdown]
# ## Pitch without a fundamental
#
# The pitch of a harmonic tone does not need its fundamental: hearing takes it from the spacing
# and pattern of the harmonics that remain.

# %% [about]
# **Listen for** whether the pitch changes as the tone loses its lowest harmonics, one at a time.
# The timbre gets thinner and brighter at each step; the booklet asks whether the pitch changes
# too. A soft low rumble plays throughout.
#
# Houtsma, Rossing & Wagenaars, Demonstration 20. Ten harmonics of 200 Hz at equal amplitude, first
# all ten, then without the fundamental, then without the lowest two, and so on, as in the booklet,
# which adds noise below 300 Hz at -10 dB to mask any 200 Hz difference tone that distortion in the
# playback might make. Ours: the steps go on until harmonics 7 to 10 are left; the tones are 0.8 s
# long, 1.2 s apart, in sine phase with 20 ms ramps; and the noise's -10 dB is taken relative to the
# full ten-harmonic tone.

# %% [demo pk3] Harmonics removed from the bottom
tones = [(1.2 * k, harmonics(200, np.arange(lowest, 11), 0.8)) for k, lowest in enumerate(range(1, 8))]
complete_rms_db = 10 * np.log10(10)  # ten harmonics, each at RMS 1
rumble = noise(8.4, (20, 300), complete_rms_db - 10, seed=20)
sound = finish(so.Sound(place(tones, dur=8.4).data[: rumble.n_samples] + rumble.data, FS))
fig, playhead = show(sound, fmin=50, fmax=3000)

# %% [about]
# **Listen for** a low pitch that rises a little at each step. Three partials 200 Hz apart, 800,
# 1000 and 1200 Hz, all move up 20 Hz at a time until they are 1000, 1200 and 1400 Hz. Each
# partial rises by 200 Hz in all, yet the booklet expects the low pitch heard to rise only from
# 200 Hz to about 241 Hz, a nearly common factor of the three, and notes that a second or even
# third rising pitch may be heard as well. A difference tone made by distortion would stay at
# 200 Hz, the spacing.
#
# Houtsma, Rossing & Wagenaars, Demonstration 21, after Schouten, Ritsma & Cardozo (1962). Three
# partials, equal in level, 0.5 s long, shifted up in ten 20 Hz steps, as in the booklet. Ours:
# 20 ms ramps, a pause of 0.25 s between tones, and one pass where the disc plays two.

# %% [demo pk4] Three partials shifted off the harmonic series
events = [(0.75 * k, partials(np.array([800, 1000, 1200]) + 20 * k, 0.5)) for k in range(11)]
sound = finish(place(events))
fig, playhead = show(sound, fmin=100, fmax=2000)

# %% [about]
# **Listen for** the pitch of each three-partial tone against the harmonic tone after it. First
# 800, 1000 and 1200 Hz, then a harmonic tone on 200 Hz; then 850, 1050 and 1250 Hz, then a
# harmonic tone on 210 Hz. The booklet reports the low pitches of the two three-partial tones well
# matched by the two harmonic tones, although the second set is not harmonic and its spacing is
# still 200 Hz.
#
# Houtsma, Rossing & Wagenaars, Demonstration 21, its second part. The partials and fundamentals
# are the booklet's. Ours: the matching tones hold harmonics 1 to 10 at equal amplitude, every
# tone is 0.5 s long, and the pauses are 0.25 s within a pair and 0.75 s between pairs.

# %% [demo pk4b] Matching the shifted pitch
pairs = [((800, 1000, 1200), 200), ((850, 1050, 1250), 210)]
events = []
for k, (freqs, f0) in enumerate(pairs):
    events += [(2.0 * k, partials(freqs, 0.5)), (2.0 * k + 0.75, harmonics(f0, np.arange(1, 11), 0.5))]
sound = finish(place(events))
fig, playhead = show(sound, fmin=100, fmax=3000)

# %% [about]
# **Listen for** which way the pitch goes from the first tone of each pair to the second: up or
# down? In each pair, 800 and 1000 Hz are followed by 750 and 1000 Hz. The booklet explains that
# listening to the separate partials, you hear one go down, from 800 to 750 Hz, while listening
# to the whole, you hear the missing fundamental go up a major third, from 200 to 250 Hz. A soft
# noise plays behind them.
#
# Houtsma, Rossing & Wagenaars, Demonstration 25, after Smoorenburg (1970). Two-tone complexes of
# 800 and 1000 Hz, then 750 and 1000 Hz, the pair played four times against a background of
# noise, as in the booklet. Ours: tones 0.5 s long with 20 ms ramps, 0.1 s apart, a pair every
# 1.5 s, and the noise, from 100 to 5000 Hz at 20 dB below one tone.

# %% [demo pk5] Up or down
events = []
for k in range(4):
    events += [(1.5 * k, partials((800, 1000), 0.5)), (1.5 * k + 0.6, partials((750, 1000), 0.5))]
pairs = place(events, dur=6.0)
hiss = noise(6.0, (100, 5000), -20, seed=25)
sound = finish(so.Sound(pairs.data[: hiss.n_samples] + hiss.data, FS))
fig, playhead = show(sound, fmin=100, fmax=3000)

# %% [markdown]
# ## Melodies of virtual pitch
#
# A tune played with tones that have no energy at their own pitch still carries its melody. Noise
# that covers the frequencies of the pitches does not mask it; noise that covers the harmonics
# does.

# %% [about]
# **Listen for** the tune of the Westminster chime twice over, in pairs: each note first as a pure
# tone, then as a tone of three high harmonics with the same pitch. The two versions carry the
# same melody, though the second has no energy at any of the notes' frequencies.
#
# Houtsma, Rossing & Wagenaars, Demonstration 22. Each note is a pure tone, then a complex of its
# 4th, 5th and 6th harmonics, as in the booklet. The notes are ours: the first two changes of the
# Westminster Quarters, G#4 F#4 E4 B3 and E4 G#4 F#4 B3, in equal temperament (247 to 415 Hz), each
# tone 0.4 s long with 20 ms ramps, 0.1 s apart within a pair, a pair every 1.1 s. The complex tones
# have the same RMS as the pure tones.

# %% [demo pk6] A tune in pure tones and in high harmonics
NOTES = {"B3": 246.94, "E4": 329.63, "F#4": 369.99, "G#4": 415.30}
TUNE = [NOTES[n] for n in ("G#4", "F#4", "E4", "B3", "E4", "G#4", "F#4", "B3")]


def westminster(masker=None):
    """Each note as a pure tone, then as harmonics 4 to 6 at the same total RMS; with `masker`
    noise added under the whole tune."""
    events = []
    for k, f0 in enumerate(TUNE):
        high = harmonics(f0, [4, 5, 6], 0.4) * np.sqrt(1 / 3)
        events += [(1.1 * k, tone(f0, 0.4)), (1.1 * k + 0.5, high)]
    tune = place(events, dur=1.1 * len(TUNE))
    if masker is not None:
        tune = so.Sound(tune.data[: masker.n_samples] + masker.data, FS)
    return tune


sound = finish(westminster())
fig, playhead = show(sound, fmin=100, fmax=5000)

# %% [about]
# **Listen for** the same pairs under a low noise, which covers the pure tones. The booklet
# reports that the pitches of the complex tones are still heard, so the melody is still there,
# although the noise covers the frequencies where those pitches lie.
#
# Houtsma, Rossing & Wagenaars, Demonstration 22, its second part: low-pass noise added. The
# booklet does not give the noise; here it runs from 20 to 600 Hz, below the lowest harmonic of
# the complex tones (988 Hz), at 15 dB above a pure tone's RMS, and is ours.

# %% [demo pk6b] The tune under low noise
low_noise = noise(1.1 * len(TUNE), (20, 600), 15, seed=22)
sound = finish(westminster(low_noise))
fig, playhead = show(sound, fmin=100, fmax=5000)

# %% [about]
# **Listen for** the same pairs under a high noise, which covers the harmonics of the complex
# tones. The booklet reports that the pure-tone melody is still heard.
#
# Houtsma, Rossing & Wagenaars, Demonstration 22, its third part: high-pass noise added. Ours:
# the noise runs from 800 to 5000 Hz, above the highest pure tone, at 20 dB above a pure tone's
# RMS.

# %% [demo pk6c] The tune under high noise
high_noise = noise(1.1 * len(TUNE), (800, 5000), 20, seed=23)
sound = finish(westminster(high_noise))
fig, playhead = show(sound, fmin=100, fmax=5000)

# %% [about]
# **Listen for** the same tune, now with each note made of three successive harmonics chosen at
# random from the second to the sixth. The timbre changes from note to note; the booklet reports
# that the melody stays clearly recognizable, which shows that it is not followed by tracking any
# one harmonic from note to note.
#
# Houtsma, Rossing & Wagenaars, Demonstration 23, after Houtsma (1984). Three successive harmonics
# of each note, with harmonic numbers between 2 and 6, as in the booklet. Ours: the tune and
# timing (0.4 s notes, one every 0.5 s), the random draws (seed 231), equal amplitudes and 20 ms
# ramps.
#
# The three figures share one frequency axis. Above each note are its harmonic numbers, and the
# dashed line is its fundamental, which is not in the sound.


# %% [demo pk7] The tune on random low harmonics
def random_harmonics(lowest, highest, seed):
    rng = np.random.default_rng(seed)
    events, firsts = [], []
    for k, f0 in enumerate(TUNE):
        first = rng.integers(lowest, highest - 1)  # three successive harmonics within the range
        events.append((0.5 * k, harmonics(f0, [first, first + 1, first + 2], 0.4)))
        firsts.append(first)
    return place(events), firsts


def show_harmonics(snd, firsts):
    """The spectrogram on a linear axis, the same for all three parts, with the harmonic numbers
    of each note written above it and its missing fundamental dashed."""
    fig, panels = show(snd, fmin=0, fmax=5400, log=False)
    for k, (f0, first) in enumerate(zip(TUNE, firsts, strict=True)):
        panels[0].plot([0.5 * k, 0.5 * k + 0.4], [f0, f0], color="white", linestyle="--", linewidth=1)
        panels[0].text(
            0.5 * k + 0.2,
            f0 * (first + 2) + 150,
            f"{first}-{first + 2}",
            color="white",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    return fig, panels


sound, firsts = random_harmonics(2, 6, seed=231)
sound = finish(sound)
fig, playhead = show_harmonics(sound, firsts)

# %% [about]
# **Listen for** the tune again, with harmonics drawn from the fifth to the ninth.
#
# Houtsma, Rossing & Wagenaars, Demonstration 23, its second part: harmonic numbers between 5 and
# 9. Ours as before (seed 232).

# %% [demo pk7b] The tune on random middle harmonics
sound, firsts = random_harmonics(5, 9, seed=232)
sound = finish(sound)
fig, playhead = show_harmonics(sound, firsts)

# %% [about]
# **Listen for** the tune once more, with harmonics drawn from the eighth to the twelfth: the
# highest and most crowded together of the three.
#
# Houtsma, Rossing & Wagenaars, Demonstration 23, its third part: harmonic numbers between 8 and
# 12. Ours as before (seed 233).

# %% [demo pk7c] The tune on random high harmonics
sound, firsts = random_harmonics(8, 12, seed=233)
sound = finish(sound)
fig, playhead = show_harmonics(sound, firsts)

# %% [markdown]
# ## Pitch from a repetition
#
# A sound added to a copy of itself delayed by a few milliseconds has a pitch at one over the
# delay, whatever the sound. The booklet traces this to Huygens, who heard a pitch in the sound
# of a fountain reflected from the steps of a staircase.

# %% [about]
# **Listen for** a scale rising five octaves, played by pairs of clicks. Within each pair the
# second click repeats the first after a delay, and the delay shrinks from note to note; the
# booklet gives the pitch heard as one over the delay.
#
# Houtsma, Rossing & Wagenaars, Demonstration 26, its first part. A diatonic scale of five octaves
# played with pairs of identical pulses whose delay goes from 15 ms (a pitch of 67 Hz) down. The
# booklet gives the last delay as 0.48 ms; this scale, in equal temperament, ends at 15/32 =
# 0.47 ms. Ours: the scale's tuning; the pairs repeat regularly, 25 a second, which is too slow to
# give a pitch of its own; each note lasts 0.25 s; and the clicks are band-limited, so that delays
# between samples are exact.

# %% [demo pk8] A scale of click pairs
STEPS = np.array([0, 2, 4, 5, 7, 9, 11])  # the major scale, in semitones
NOTE = 0.25


def scale_delays(octaves, longest=15e-3):
    semitones = np.concatenate([STEPS + 12 * k for k in range(octaves)] + [[12 * octaves]])
    return longest / 2 ** (semitones / 12)


def click_pair(delay, half_width=64):
    """Two band-limited clicks `delay` seconds apart: windowed sincs, so the delay need not be a
    whole number of samples."""
    n = np.arange(-half_width, round(delay * FS) + half_width + 1)

    def windowed_sinc(x):
        return np.sinc(x) * np.where(np.abs(x) <= half_width, np.cos(np.pi * x / (2 * half_width)) ** 2, 0)

    return so.Sound(windowed_sinc(n) + windowed_sinc(n - delay * FS), FS)


def click_scale(delays, intervals, seed):
    rng = np.random.default_rng(seed)
    notes = []
    for delay in delays:
        notes.append(so.event_train(click_pair(delay), NOTE, intervals, rng=rng))
    return so.concat(notes)


sound = finish(click_scale(scale_delays(5), 1 / 25, seed=261))
fig, playhead = show(sound, fmin=50, fmax=5000)

# %% [about]
# **Listen for** a scale rising four octaves, from click pairs that now come at random times. The
# booklet reports that the pitch at one over the delay is heard as before, though the pairs no
# longer come regularly.
#
# Houtsma, Rossing & Wagenaars, Demonstration 26, its second part. A diatonic scale of four
# octaves; the pairs start at the times of a Poisson process, and the delay goes from 15 ms to
# 15/16 = 0.94 ms (the booklet gives 0.95 ms). Ours: 100 pairs a second on average, each note
# 0.25 s.

# %% [demo pk8b] A scale of click pairs at random times
import scipy.stats

sound = finish(click_scale(scale_delays(4), scipy.stats.expon(scale=1 / 100), seed=262))
fig, playhead = show(sound, fmin=50, fmax=5000)

# %% [about]
# **Listen for** the same four-octave scale played by bursts of hiss, each added to a copy of
# itself delayed by the same amounts.
#
# Houtsma, Rossing & Wagenaars, Demonstration 26, its third part: bursts of white noise echoed
# 15 ms to 0.94 ms later. Ours: 0.25 s bursts with 10 ms ramps, and the delay made exact by a
# phase shift in the frequency domain.


# %% [demo pk8c] A scale of echoed noise
def echoed_noise(delay, rng):
    burst = so.gaussian_noise(NOTE, FS, rng=rng).data[:, 0]
    n = len(burst) + round(0.02 * FS)  # room for the echo
    spectrum = np.fft.rfft(burst, n)
    freqs = np.fft.rfftfreq(n, 1 / FS)
    echo = np.fft.irfft(spectrum * np.exp(-2j * np.pi * freqs * delay), n)
    both = np.pad(burst, (0, n - len(burst))) + echo
    return ramped(so.Sound(both[: len(burst)], FS), 10e-3)


rng = np.random.default_rng(263)
sound = finish(so.concat([echoed_noise(delay, rng) for delay in scale_delays(4)]))
fig, playhead = show(sound, fmin=50, fmax=8000)

# %% [markdown]
# ## Beats and combination tones
#
# Two tones heard together can make beats, both when their frequencies are close and when they
# are close to a simple ratio; and the ear itself adds tones that are not in the sound.

# %% [about]
# **Listen for** a steady tone, a second steady tone, then the two together, rising and falling
# in loudness four times a second: beats.
#
# Houtsma, Rossing & Wagenaars, Demonstration 32, its first part. Pure tones of 1000 and 1004 Hz,
# first separately and then together, as in the booklet, which gives beats at 4 Hz. Ours: equal
# levels, 2 s each alone and 4 s together, 20 ms ramps and pauses of 0.5 s; one pass where the
# disc plays two.

# %% [demo pk9] Primary beats
pause = so.silence(0.5, FS)
sound = finish(so.concat([tone(1000, 2.0), pause, tone(1004, 2.0), pause, partials((1000, 1004), 4.0)]))
fig, playhead = show(sound, fmin=500, fmax=2500)

# %% [about]
# **Listen for** beats again, four a second, in three pairs of tones that are far apart: a little
# more than an octave, a fifth and a fourth. The booklet invites you to compare how strong the
# beats are with those of the first part, where the two tones were 4 Hz apart.
#
# Houtsma, Rossing & Wagenaars, Demonstration 32, its second part. A 1000 Hz tone with 2004,
# 1502 and 1334.67 Hz in turn, each mistuned from 2/1, 3/2 and 4/3 so that the second-order beats
# come 4 times a second, as in the booklet. Ours: equal levels, 2.7 s each, 20 ms ramps, pauses
# of 0.5 s.

# %% [demo pk9b] Beats of mistuned consonances
mistuned = [partials((1000, f), 2.7) for f in (2004, 1502, 1334.67)]
sound = finish(so.concat([mistuned[0], pause, mistuned[1], pause, mistuned[2]]))
fig, playhead = show(sound, fmin=500, fmax=3000)

# %% [about]
# **Listen for** a third, softer tone joining two steady ones, and beating slowly, four times a
# second. The two tones are 1000 and 1200 Hz; the third is 804 Hz. Nothing in the sound is at
# 800 Hz, but the booklet reports a combination tone made in the ear at 2 x 1000 - 1200 = 800 Hz,
# which the 804 Hz probe beats with. Combination tones are louder at higher levels; the booklet
# notes that at about 80 dB they are heard over a wide range.
#
# Houtsma, Rossing & Wagenaars, Demonstration 34, after Goldstein (1967). Tones of 1000 and 1200 Hz,
# then an 804 Hz probe added, as in the booklet. Its text calls 800 Hz the quadratic difference
# tone; it is the cubic one, 2f1 - f2 (the quadratic one, f2 - f1, is at 200 Hz). Ours: equal
# levels for the two tones, the probe 20 dB below them, 2 s of the two tones before the probe
# joins for 4 s, and 20 ms ramps.

# %% [demo pk10] A probe beating with a tone that is not there
two = partials((1000, 1200), 6.0)
probe = place([(2.0, tone(804, 4.0).gain_db(-20))], dur=6.0)
sound = finish(so.Sound(two.data + probe.data[: two.n_samples], FS))
fig, playhead = show(sound, fmin=500, fmax=2500)

# %% [about]
# **Listen for** a low tone moving down while the upper of the two tones rises, from 1200 to 1600
# Hz and back. The booklet describes the cubic difference tone, 2f1 - f2, moving opposite to the
# upper tone, soon joined by the quadratic one, f2 - f1, which rises with it; the two should
# cross when the upper tone is at 1500 Hz, where both are at 500 Hz. Neither is in the sound.
#
# Houtsma, Rossing & Wagenaars, Demonstration 34, its second part. A steady 1000 Hz tone with a
# second tone gliding from 1200 to 1600 Hz and back, as in the booklet, which says only that it
# moves slowly. Ours: 8 s up and 8 s down in a straight line, equal levels, 20 ms ramps.

# %% [demo pk10b] Combination tones moving
times = np.array([0.0, 8.0, 16.0])
glide = so.harmonic_complex(16.0, FS, (times, np.array([1200, 1600, 1200])), harmonics=[1], ramp=0)
steady = so.pure_tone(16.0, FS, 1000)
sound = finish(ramped(so.Sound(steady.data[:, 0] + glide.data[:, 0], FS)))
fig, playhead = show(sound, fmin=800, fmax=1800, log=False, win=0.1, title="Spectrogram (Hann 100 ms)")

# %% [markdown]
# ## References
#
# - Houtsma, Rossing & Wagenaars (1987). *Auditory Demonstrations* (compact disc and booklet).
#   Institute for Perception Research (IPO), Eindhoven, and the Acoustical Society of America.
#   [Booklet](https://research.tue.nl/en/publications/auditory-demonstrations).
# - Doughty & Garner (1948). Pitch characteristics of short tones. II. Pitch as a function of
#   tonal duration. *J. Exp. Psychol.* 38(4), 478–494. [doi:10.1037/h0057850](https://doi.org/10.1037/h0057850).
# - Goldstein (1967). Auditory nonlinearity. *JASA* 41(3), 676–699.
#   [doi:10.1121/1.1910396](https://doi.org/10.1121/1.1910396).
# - Houtsma (1984). Pitch salience of various complex sounds. *Music Perception* 1(3), 296–307.
#   [doi:10.2307/40285262](https://doi.org/10.2307/40285262).
# - Schouten, Ritsma & Cardozo (1962). Pitch of the residue. *JASA* 34(9B), 1418–1424.
#   [doi:10.1121/1.1918360](https://doi.org/10.1121/1.1918360).
# - Smoorenburg (1970). Pitch perception of two-frequency stimuli. *JASA* 48(4B), 924–942.
#   [doi:10.1121/1.1912232](https://doi.org/10.1121/1.1912232).
# - Ward (1954). Subjective musical pitch. *JASA* 26(3), 369–380.
#   [doi:10.1121/1.1907344](https://doi.org/10.1121/1.1907344).
#
# The papers were checked for their publication details only; the parameters on this page come
# from the booklet, and those marked as ours are choices made for this page.
