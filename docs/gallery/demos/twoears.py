"""Two ears: sonore's reproductions of the classic demonstrations of hearing with two ears.

This script is the gallery page https://choyun1.github.io/sonore/gallery/twoears.html:
docs/gallery/build.py runs it cell by cell from the repository root and shows each
cell's code beside what it made. Run it yourself from the repository root,

    python docs/gallery/demos/twoears.py

or a cell at a time ("# %%" starts a cell in VS Code, Spyder and Jupytext).
"""

# %% [markdown]
# # Two ears: reproducing Houtsma, Rossing & Wagenaars and Bregman & Ahad
#
# Every sound on this page is sonore's reproduction of a published demonstration. Most are from
# the Acoustical Society of America's *Auditory Demonstrations* (Houtsma, Rossing & Wagenaars,
# 1987), the compact disc made at the Institute for Perception Research (IPO) in Eindhoven, whose
# booklet is published by the Eindhoven University of Technology
# ([TU/e](https://research.tue.nl/en/publications/auditory-demonstrations)); the last is from
# Albert Bregman and Pierre Ahad's *Demonstrations of Auditory Scene Analysis* (1996), whose
# recordings and booklet Bregman's laboratory publishes
# ([McGill](https://webpages.mcgill.ca/staff/Group2/abregm1/web/downloadsdl.htm)). The
# demonstrations, their design and what they show are their authors' work, and the original
# recordings are the ones to listen to.
#
# **What they show.** With two ears, the brain compares what each ear receives. A sound from one
# side reaches the nearer ear a fraction of a millisecond earlier, and louder, so the difference
# in time and the difference in level between the ears tell where it is. The same comparison
# helps hear a tone in noise when the tone and the noise differ between the ears, and decides
# whether what the two ears hear is one sound or two. Each ear here receives its own signal, so
# **listen over headphones**; over loudspeakers both ears hear both channels and most of these
# effects fade.
#
# This page is a test of sonore: can its generators rebuild each stimulus from the description
# in the booklets? Each example names the demonstration it reproduces, rebuilds it from the
# booklet's frequencies, levels and timing, and says where it departs from them: where the
# booklet leaves a value open, such as the level of a masker, or where its numbers do not agree
# with each other. What listeners hear is reported as the booklets report it; this page has not
# tested any of it. Related pages: [Binaural cues](binaural.html) has Demonstration 36, binaural
# beats ([b4](binaural.html#d-b4)), and sounds moved by timing alone
# ([Timing alone](binaural.html#h-timing-alone)); [Rooms](reverb.html) has echoes and a reversed
# room ([18](reverb.html#d-18)), related to Demonstration 35.
#
# - [Time and level](#h-time-and-level): tones and clicks moved from side to side by a
#   difference in phase, time or level between the ears (Demonstration 37).
# - [Masking level differences](#h-masking-level-differences): a tone in noise, easier to hear
#   when it differs between the ears and the noise does not (Demonstration 38).
# - [An octave illusion](#h-an-octave-illusion): two tones an octave apart, swapping ears
#   (Demonstration 39).
# - [Correlation and fusion](#h-correlation-and-fusion): noise heard as one sound in the middle
#   when the ears get the same noise, and as two when they get different noises (Bregman & Ahad,
#   Demonstration 40).

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
# Every sound here has two channels, left ear first. Levels are in dB re a pure tone at RMS 1,
# which is how `so.pure_tone` makes it; `so.gaussian_noise` also makes its noise at RMS 1. The
# examples of one demonstration are scaled together by `finish_together`, so that a step at the
# same level in two of them plays at the same level. Two kinds of figure: one panel per ear's
# spectrogram, or the waveforms with the interaural cues `so.interaural_cues` measures from them,
# in short windows: the time difference (ITD, positive when the right ear leads), the level
# difference (ILD, positive when the right ear is louder) and the correlation between the ears.
# The sample rate is 44.1 kHz, so that delays between the ears are fine-grained.

# %%
FS = 44100


def envelope(n, ramp):
    """Linear ramps of `ramp` seconds at both ends of `n` samples."""
    i = np.arange(n)
    return np.clip(np.minimum(i, n - 1 - i) / max(ramp * FS, 1), 0, 1)


def sine(freq, dur, phase=0.0, ramp=10e-3):
    """A sine at RMS 1, `dur` seconds long including linear ramps, starting at `phase` radians."""
    t = np.arange(round(dur * FS)) / FS
    return envelope(len(t), ramp) * np.sqrt(2) * np.sin(2 * np.pi * freq * t + phase)


def ears(left, right):
    """A two-channel Sound from the left and right ears' samples."""
    return so.Sound(np.column_stack([left, right]), FS)


def place(events, dur=None):
    """Two-channel sounds added at their onset times: `events` is a list of (onset, Sound)."""
    end = max(onset + snd.duration for onset, snd in events)
    out = np.zeros((round((dur or end) * FS), 2))
    for onset, snd in events:
        i = round(onset * FS)
        out[i : i + snd.n_samples] += snd.data[: len(out) - i]
    return so.Sound(out, FS)


def finish_together(sounds):
    """Sounds scaled by one gain, as `finish` would scale the loudest of them, so their levels
    keep their relation from one sound to the next."""
    loudest = max(sounds, key=lambda snd: snd.rms)
    gain = 0.1 / loudest.rms
    gain = min(gain, 0.95 / max(snd.peak for snd in sounds))
    return [(snd * gain).ramp(5e-3) for snd in sounds]


def show_ears(snd, fmin=100, fmax=5000, win=30e-3):
    """A spectrogram of each ear on a log frequency axis. Returns the figure and the panels the
    playhead follows."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 5.4), sharex=True, layout="constrained")
    stft = so.STFT(snd, win_dur=win, hop_dur=5e-3)
    step = stft.f[1] - stft.f[0]
    keep = (stft.f >= fmin - step) & (stft.f <= fmax + step)
    top = stft.db[:, keep].max()
    for ch, ax in enumerate(axes):
        ax.pcolormesh(stft.t, stft.f[keep], stft.db[ch][keep], vmin=top - 70, vmax=top, cmap="magma")
        so.plotting.log_frequency_axis(ax)
        ear = ("left", "right")[ch]
        ax.set(
            ylim=(fmin, fmax),
            ylabel="Frequency [Hz]",
            title=f"Spectrogram (Hann {win * 1000:g} ms), {ear} ear",
        )
    axes[1].set(xlabel="Time [s]", xlim=(0, snd.duration))
    return fig, list(axes)


def show_cues(snd, win=10e-3, max_itd=1e-3, show="both"):
    """The two ears' waveforms, then the cues measured in windows of `win` seconds: the ITD and
    ILD and the correlation between the ears (`show` "both"), the ILD and the correlation ("ild"),
    the correlation alone ("corr"), or the ITD alone ("itd", for clicks, whose windows hold too
    little for a correlation). Returns the figure and the panels the playhead follows."""
    rows = 2 if show in ("corr", "itd") else 3
    fig, axes = plt.subplots(rows, 1, figsize=(10, 2.2 * rows), sharex=True, layout="constrained")
    snd.plot(axes[0], lw=0.4)
    axes[0].set_title("Waveform, left and right")
    cues = so.interaural_cues(snd, win_dur=win, max_itd=max_itd)
    if show == "both":
        cues.plot(axes[1])
        axes[1].set_xlabel("")
    elif show == "itd":  # dots, since only the windows holding a click have a value
        axes[1].plot(cues.t, cues.itd * 1e6, ".", color="tab:purple")
        axes[1].set(ylabel="ITD [µs] (+ = right leads)", title="Interaural time difference")
        axes[1].set(xlabel="Time [s]", xlim=(0, snd.duration))
        axes[1].grid(ls=":")
        return fig, list(axes)
    elif show == "ild":
        axes[1].plot(cues.t, cues.ild, color="tab:green")
        axes[1].set(ylabel="ILD [dB] (+ = right louder)", title="Interaural level difference")
        axes[1].grid(ls=":")
    ax = axes[-1]
    ax.plot(cues.t, cues.corr0, color="tab:orange", lw=1, label="correlation at zero lag")
    ax.plot(cues.t, cues.iac, color="k", alpha=0.6, lw=1, label="coherence (peak over lags)")
    ax.axhline(0, color="k", alpha=0.25, lw=0.8)
    ax.set(ylim=(-1.05, 1.05), ylabel="Interaural corr.", xlabel="Time [s]", xlim=(0, snd.duration))
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(ls=":")
    return fig, list(axes)


# %% [markdown]
# ## Time and level
#
# The booklet credits Lord Rayleigh (1907) with seeing that low sounds are placed mainly by the
# difference in their time of arrival at the two ears, and high sounds mainly by the difference
# in their level, since the head casts a sound shadow only for wavelengths shorter than itself.

# %% [about]
# **Listen for** a 500 Hz tone jumping from one side of your head to the other: in every burst
# the two ears get the same tone, but its phase differs between them by 45 degrees, the left ear
# ahead in one burst and the right in the next.
#
# Houtsma, Rossing & Wagenaars, Demonstration 37, its first part: a 500 Hz tone with interaural
# phases alternating between +45 and -45 degrees, as in the booklet. At 500 Hz, 45 degrees is a
# time difference of 0.25 ms. The booklet's figures for this time difference, 14 ms at 500 Hz and
# 3.6 ms at 2000 Hz, do not follow from its own formula, which gives 0.25 and 0.0625 ms; ours
# follows the phases. Ours also: bursts of 0.4 s with 20 ms ramps, one every 0.6 s, eight in all.


# %%
def phase_bursts(freq, phase_deg=45, n=8):
    """Bursts of a tone whose phase leads in the left ear by `phase_deg` degrees, then in the
    right, alternately."""
    events = []
    for k in range(n):
        half = np.deg2rad(phase_deg) / 2 * (1 if k % 2 == 0 else -1)
        burst = ears(sine(freq, 0.4, half, ramp=20e-3), sine(freq, 0.4, -half, ramp=20e-3))
        events.append((0.1 + 0.6 * k, burst))
    return place(events, dur=0.1 + 0.6 * n)


phase_sounds = finish_together([phase_bursts(500), phase_bursts(2000)])

# %% [demo te1] A 500 Hz tone with its phase ahead in one ear
sound = phase_sounds[0]
fig, playhead = show_cues(sound, max_itd=1e-3)

# %% [about]
# **Listen for** the same at 2000 Hz. The booklet says this tone does not move: at this
# frequency the ear no longer follows the phase of the waveform.
#
# Houtsma, Rossing & Wagenaars, Demonstration 37: a 2000 Hz tone with interaural phases of +45
# and -45 degrees, a time difference of 0.0625 ms. Ours as before. The measured ITD is searched
# within a quarter of the tone's period, where the cross-correlation has one peak.

# %% [demo te1b] A 2000 Hz tone with its phase ahead in one ear
sound = phase_sounds[1]
fig, playhead = show_cues(sound, max_itd=0.125e-3)

# %% [about]
# **Listen for** a click moving back and forth between your ears as the time between the ears
# swings from the left ear first to the right ear first and back.
#
# Houtsma, Rossing & Wagenaars, Demonstration 37, its second part: a 100 µs pulse whose
# interaural time difference cycles. The booklet gives a range of 5 ms to -5 ms; that is several
# times the largest time difference a head gives (under 1 ms), and two clicks that far apart
# tend to be heard as a click and its echo, so ours cycles between 0.5 ms and -0.5 ms, as if the
# booklet's figure lost a decimal point. Ours also: a click every 0.1 s, the time difference
# moving in straight lines from the left ear 0.5 ms first to the right ear 0.5 ms first and back,
# every 2 s, for 6 s. Each click's delay is fractional, by `so.apply_itd_ild`.

# %% [demo te2] A click moving between the ears
click = so.Sound(np.r_[np.ones(round(100e-6 * FS)), np.zeros(round(5e-3 * FS))], FS)
onsets = 0.1 + 0.1 * np.arange(59)
swing = 4 * np.abs(((onsets - 0.1) / 2) % 1 - 0.5) - 1  # +1 to -1 and back every 2 s
# A positive ITD in apply_itd_ild delays the left ear, so the left ear first is a negative one.
clicks = [(t, so.apply_itd_ild(click, itd=-0.5e-3 * s)) for t, s in zip(onsets, swing, strict=True)]
sound = finish(place(clicks, dur=6.1))
fig, playhead = show_cues(sound, win=0.1, show="itd")

# %% [about]
# **Listen for** a 250 Hz tone moving from the left ear to the right and back as the level
# difference between the ears swings. The booklet notes that a real head casts little shadow at
# so low a frequency, so a level difference this large hardly occurs for a real source; through
# headphones it still moves the tone.
#
# Houtsma, Rossing & Wagenaars, Demonstration 37, its third part: a 250 Hz tone whose interaural
# level difference changes from 32 dB to -32 dB in 1.25 s, as in the booklet. Ours: the level
# difference moving in straight lines (in dB) from 32 dB in favor of the left ear to 32 dB in
# favor of the right and back, four sweeps in 5 s, split equally between the ears.


# %%
def level_sweep(freq, dur=5.0, sweep=1.25, ild_db=32):
    """A tone whose level difference swings from `ild_db` dB louder in the left ear to `ild_db`
    dB louder in the right and back, one sweep every `sweep` seconds."""
    tone = sine(freq, dur, ramp=20e-3)
    t = np.arange(len(tone)) / FS
    # +1 (left louder) to -1 (right louder) over one sweep, then back.
    side = 4 * np.abs((t / (2 * sweep)) % 1 - 0.5) - 1
    ild = ild_db * side
    return ears(tone * 10 ** (ild / 40), tone * 10 ** (-ild / 40))


level_sounds = finish_together([level_sweep(250), level_sweep(4000)])

# %% [demo te3] A 250 Hz tone moved by level
sound = level_sounds[0]
fig, playhead = show_cues(sound, win=20e-3, show="ild")

# %% [about]
# **Listen for** the same at 4000 Hz. In both, the booklet says, the tone moves from side to
# side.
#
# Houtsma, Rossing & Wagenaars, Demonstration 37: a 4000 Hz tone with the same level sweep. Ours
# as before.

# %% [demo te3b] A 4000 Hz tone moved by level
sound = level_sounds[1]
fig, playhead = show_cues(sound, win=20e-3, show="ild")

# %% [markdown]
# ## Masking level differences
#
# A tone in noise is easier to hear when the tone and the noise differ between the ears, for
# instance when the noise is the same in both ears and the tone is inverted in one of them. The
# booklet credits Licklider (1948) with finding this for speech and Hirsh (1948) for tones.
# Each example is a staircase: a 500 Hz tone in ten steps that get quieter, against a noise that
# stays the same. Count the steps you hear in each.

# %% [about]
# **Listen for** the staircase in the left ear alone, with no noise: every step is audible here,
# for comparison.
#
# Houtsma, Rossing & Wagenaars, Demonstration 38: a 500 Hz signal of 100 ms in the left ear, in
# ten steps, the first at -10 dB and each of the other nine 3 dB below the one before, as in the
# booklet. The booklet does not give the noise or its level; ours: noise from 20 Hz to 10 kHz, the
# same throughout, at a spectrum level 26 dB below the first step (an estimate meant to leave
# about half the steps audible when signal and noise are the same in both ears). Ours also: 10 ms
# ramps within the 100 ms, one step every 0.5 s, and one pass where the disc plays two.

# %%
mld_rng = np.random.default_rng(38)
MLD_BAND = (20, 10000)
NOISE_SPECTRUM_DB = -36  # dB per Hz re a tone at RMS 1, 26 dB below the first step


def staircase():
    """The 500 Hz signal in 10 steps: -10 dB, then 3 dB lower each time."""
    signal = np.zeros(round(5.25 * FS))
    for k in range(10):
        i = round((0.25 + 0.5 * k) * FS)
        burst = sine(500, 0.1) * 10 ** ((-10 - 3 * k) / 20)
        signal[i : i + len(burst)] += burst
    return signal


band_level = NOISE_SPECTRUM_DB + 10 * np.log10(MLD_BAND[1] - MLD_BAND[0])
masker = so.gaussian_noise(5.25, FS, band=MLD_BAND, rng=mld_rng).gain_db(band_level).data[:, 0]
signal, silent = staircase(), np.zeros(round(5.25 * FS))
mld_sounds = finish_together(
    [
        ears(signal, silent),  # signal left
        ears(signal + masker, silent),  # signal and noise left
        ears(signal + masker, masker),  # signal left, noise both
        ears(signal + masker, signal + masker),  # signal and noise both
        ears(signal + masker, -signal + masker),  # noise both, signal inverted on the right
    ]
)

# %% [demo te4] The staircase in the left ear
sound = mld_sounds[0]
fig, playhead = show_ears(sound, fmin=100, fmax=5000)

# %% [about]
# **Listen for** the staircase with noise added in the same ear. Count the steps: the noise
# hides the quieter ones.
#
# Houtsma, Rossing & Wagenaars, Demonstration 38, its second example: signal and noise in the
# left ear. Ours as before.

# %% [demo te4b] Signal and noise in the left ear
sound = mld_sounds[1]
fig, playhead = show_ears(sound, fmin=100, fmax=5000)

# %% [about]
# **Listen for** the same with the noise also in the right ear. The booklet says the noise now
# seems to come from a different place in the head than the tone, and the tone is much easier
# to hear: count again.
#
# Houtsma, Rossing & Wagenaars, Demonstration 38, its third example: the signal in the left ear,
# the same noise in both. Ours as before.

# %% [demo te4c] Noise in both ears, signal in the left
sound = mld_sounds[2]
fig, playhead = show_ears(sound, fmin=100, fmax=5000)

# %% [about]
# **Listen for** the signal added to the right ear too, so that both ears get the same mixture.
# The booklet says tone and noise again seem to come from the same place, and the tone is again
# hard to hear.
#
# Houtsma, Rossing & Wagenaars, Demonstration 38, its fourth example: signal and noise the same
# in both ears. Ours as before.

# %% [demo te4d] Signal and noise in both ears
sound = mld_sounds[3]
fig, playhead = show_ears(sound, fmin=100, fmax=5000)

# %% [about]
# **Listen for** the same, but with the signal turned upside down (its phase inverted) in the
# right ear. The two ears' spectrograms look the same as before; the difference is in the sign
# of the tone. The booklet says signal and noise now seem to be in different places and the
# signal is easy to hear.
#
# Houtsma, Rossing & Wagenaars, Demonstration 38, its fifth example: the same noise in both ears,
# the signal in both but inverted in the right. Ours as before. The booklet notes that the
# Harvard tapes repeat the demonstration at 2000 Hz, where the effect is smaller; ours does not.

# %% [demo te4e] Signal inverted in the right ear
sound = mld_sounds[4]
fig, playhead = show_ears(sound, fmin=100, fmax=5000)

# %% [markdown]
# ## An octave illusion
#
# Diana Deutsch (1974) found that two tones an octave apart, played to the two ears and swapped
# back and forth between them, are heard in a way that matches neither ear's signal.

# %% [about]
# **Listen for** what you hear in each ear. Each ear gets a tone that alternates between 400 and
# 800 Hz, the two ears always opposite. The booklet reports that almost all listeners hear one
# low tone in one ear alternating with one high tone in the other, and that reversing the
# headphones does not swap them. Try that, then look at the figure.
#
# Houtsma, Rossing & Wagenaars, Demonstration 39, after Deutsch (1974): tones of 400 and 800 Hz
# alternating in the two ears in opposite phase, as in the booklet. The booklet does not give the
# tone durations; ours: 250 ms each, as in Deutsch's paper, with no gaps and 10 ms ramps, 40
# tones in 10 s.

# %% [demo te5] Two tones an octave apart, swapping ears
low, high = sine(400, 0.25, ramp=10e-3), sine(800, 0.25, ramp=10e-3)
tones = [(0.25 * k, ears(low, high) if k % 2 == 0 else ears(high, low)) for k in range(40)]
sound = finish(place(tones))
fig, playhead = show_ears(sound, fmin=100, fmax=2000)

# %% [markdown]
# ## Correlation and fusion
#
# Bregman & Ahad's booklet suggests that one way the brain decides whether the two ears hear one
# sound or two is to compare the changes in them: if the two are highly correlated they are
# combined into one sound with one position, and if not, into two.

# %% [about]
# **Listen for** a noise in the middle of your head, then two noises, one at each ear. The first
# is the same noise in both ears; the second has a different noise in each, with the same
# spectrum and level. The booklet adds that a softer noise may still be heard in the middle.
#
# Bregman & Ahad, Demonstration 40, its first part: white noise bursts of 2 s with 10 ms rise and
# decay, identical in the two channels ("mono"), then independent ("stereo"), as in the booklet.
# Ours: each once, 0.5 s apart, where the disc plays the pair twice.

# %%
fusion_rng = np.random.default_rng(40)


def white(dur):
    """White noise at RMS 1 with 10 ms ramps."""
    return so.gaussian_noise(dur, FS, rng=fusion_rng).data[:, 0] * envelope(round(dur * FS), 10e-3)


one = white(2.0)
mono = ears(one, one)
stereo = ears(white(2.0), white(2.0))


def joined(same):
    """Noise in the left ear for 4 s, joined after 2 s by the right ear's noise: the same noise
    as the left's if `same`, otherwise an independent one."""
    left = so.gaussian_noise(4.0, FS, rng=fusion_rng).data[:, 0]
    right = left.copy() if same else so.gaussian_noise(4.0, FS, rng=fusion_rng).data[:, 0].copy()
    right[: round(2.0 * FS)] = 0
    right[round(2.0 * FS) :] *= envelope(round(2.0 * FS), 10e-3)
    return ears(left * envelope(len(left), 10e-3), right)


fusion_sounds = finish_together([place([(0, mono), (2.5, stereo)]), joined(True), joined(False)])

# %% [demo te6] The same noise in both ears, then different noises
sound = fusion_sounds[0]
fig, playhead = show_cues(sound, win=20e-3, show="corr")

# %% [about]
# **Listen for** a noise in the left ear, joined after 2 s by the same noise in the right ear.
# The booklet says that as soon as the right ear's noise starts, the two fuse into one noise in
# the middle, and the noise on the left soon disappears.
#
# Bregman & Ahad, Demonstration 40, its second part: the left channel on for 4 s and joined after
# 2 s by the right, both with 10 ms rise and decay, here identical ("mono"), as in the booklet.
# Ours: once, where the disc alternates this with the next example twice.

# %% [demo te6b] Left noise joined by the same noise on the right
sound = fusion_sounds[1]
fig, playhead = show_cues(sound, win=20e-3, show="corr")

# %% [about]
# **Listen for** the same, but the right ear's noise is a different noise. The booklet says the
# two fuse less, and the noise on the left goes on being heard to the end, beside a separate
# noise on the right: the noise that started first in the left ear has captured what follows in
# that ear.
#
# Bregman & Ahad, Demonstration 40, its second part, with independent noises ("stereo"). Ours as
# before.

# %% [demo te6c] Left noise joined by a different noise on the right
sound = fusion_sounds[2]
fig, playhead = show_cues(sound, win=20e-3, show="corr")

# %% [markdown]
# ## References
#
# - Houtsma, Rossing & Wagenaars (1987). *Auditory Demonstrations* (compact disc and booklet).
#   Institute for Perception Research (IPO), Eindhoven, and the Acoustical Society of America.
#   [Booklet](https://research.tue.nl/en/publications/auditory-demonstrations).
# - Bregman & Ahad (1996). *Demonstrations of Auditory Scene Analysis: The Perceptual
#   Organization of Sound* (compact disc and booklet). Auditory Perception Laboratory, McGill
#   University; MIT Press. [Recordings and booklet](https://webpages.mcgill.ca/staff/Group2/abregm1/web/downloadsdl.htm).
# - Deutsch (1974). An auditory illusion. *Nature* 251(5473), 307–309.
#   [doi:10.1038/251307a0](https://doi.org/10.1038/251307a0).
# - Hirsh (1948). The influence of interaural phase on interaural summation and inhibition. *JASA*
#   20(4), 536–544. [doi:10.1121/1.1906407](https://doi.org/10.1121/1.1906407).
# - Licklider (1948). The influence of interaural phase relations upon the masking of speech by
#   white noise. *JASA* 20(2), 150–159.
#   [doi:10.1121/1.1906358](https://doi.org/10.1121/1.1906358).
# - Rayleigh (1907). On our perception of sound direction. *Philosophical Magazine* 13(74),
#   214–232. [doi:10.1080/14786440709463595](https://doi.org/10.1080/14786440709463595).
#
# The papers were checked for their publication details only; the parameters on this page come
# from the booklets, and those marked as ours are choices made for this page.
