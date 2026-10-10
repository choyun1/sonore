"""Binaural cues: what differs between the two ears, and what you hear when it changes.

This script is the gallery page https://choyun1.github.io/sonore/gallery/binaural.html:
docs/gallery/build.py runs it cell by cell from the repository root and shows each
cell's code beside what it made. Run it yourself from the repository root,

    python docs/gallery/spatial/binaural.py

or a cell at a time ("# %%" starts a cell in VS Code, Spyder and Jupytext).
"""

# %% [markdown]
# # Binaural cues
#
# A sound off to one side reaches the nearer ear sooner and louder: an interaural time
# difference (ITD) and an interaural level difference (ILD). How alike the two ears' signals are,
# their interaural correlation, decides whether the image is a compact point or a diffuse cloud.
# Each example on this page changes one of these and holds the others still. They need
# headphones: over speakers the two ears' signals mix in the room and the effects disappear.
#
# - [Timing alone](#h-timing-alone): a sound moved sideways by an ITD with no level difference.
# - [Correlation that changes](#h-correlation-that-changes): binaural beats, phasewarp and Oscor,
#   sounds whose interaural phase or correlation changes over time while each ear alone hears
#   something steady.
#
# For real sources at real positions, with every cue together, see [Moving talkers](moving.html),
# where three talkers are rendered through measured head-related impulse responses and one of them
# moves.

# %% [markdown]
# ## How the cues are measured
#
# `so.interaural_cues` cuts the two ears' signals into short windows (10 ms here) and, in each,
# finds the ITD as the lag (within ±1 ms) of the peak of the interaural cross-correlation, the ILD
# as the level difference in dB, and the correlation both at zero lag and at its peak over lags
# (the coherence). It can also do this per frequency band, given a filterbank; here it uses the
# whole signal. Each figure shows the waveforms, the ITD and ILD, and the two correlations, all
# measured from the sound you hear.

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
FS = 44100


def show(snd):
    """Waveforms, ITD and ILD, and interaural correlation, all on one time axis.
    Returns the figure and the panels the playhead follows."""
    fig, axes = plt.subplots(3, 1, figsize=(10, 6.6), sharex=True, layout="constrained")
    snd.plot(axes[0], lw=0.4)
    axes[0].set_title("Waveform, left and right")
    cues = so.interaural_cues(snd, win_dur=10e-3)
    cues.plot(axes[1])
    axes[1].set_xlabel("")
    axes[2].plot(cues.t, cues.corr0, color="tab:orange", lw=1, label="correlation at zero lag")
    axes[2].plot(cues.t, cues.iac, color="k", alpha=0.6, lw=1, label="coherence (peak over lags)")
    axes[2].axhline(0, color="k", alpha=0.25, lw=0.8)
    axes[2].set(ylim=(-1.05, 1.05), ylabel="Interaural corr.", xlabel="Time [s]", xlim=(0, snd.duration))
    axes[2].legend(loc="lower right", fontsize=8)
    axes[2].grid(ls=":")
    return fig, list(axes)

# %% [markdown]
# ## Timing alone

# %% [about]
# Noise with a 500 µs interaural time difference, leading in the left ear and then in the right.
# The level difference is zero except in the first and last few milliseconds, where one ear has
# started or stopped and the other not yet; the sideways shift comes from timing. The zero-lag
# correlation is near 0, since broadband noise is uncorrelated with itself 500 µs later, while the
# coherence is 1: the two ears hold the same noise, 500 µs apart.

# %% [demo 10] Timing alone
noise = so.gaussian_noise(0.8, FS, rng=0).ramp(20e-3)
noise2 = so.gaussian_noise(0.8, FS, rng=1).ramp(20e-3)
sound = finish(
    so.concat(
        [
            so.apply_itd_ild(noise, itd=-500e-6),
            so.silence(0.3, FS),
            so.apply_itd_ild(noise2, itd=500e-6),
        ]
    )
)
fig, playhead = show(sound)

# %% [markdown]
# ## Correlation that changes
#
# Three sounds whose two ears, heard alone, are steady: a tone, or plain noise. What changes over
# time is how the two ears relate. In the first two, the interaural phase turns round and round;
# in the third, the interaural phase stays put and the correlation falls and rises because the
# right ear's noise is mixed with an independent one. Each panel of zero-lag correlation follows a
# cosine or a sine at the rate of the change.

# %% [about]
# 440 Hz in the left ear, 444 Hz in the right: two steady tones, so nothing in either ear beats.
# Their interaural phase difference turns through a full cycle four times a second, and the
# binaural system, which follows the phase of low-frequency tones, hears that: the image moves or
# wobbles inside the head. The zero-lag correlation, measured in 10 ms windows, follows
# $\cos(2\pi \cdot 4\,t)$ (dotted). The ITD sweeps across the ±1 ms the measurement searches and
# jumps back when the next cycle of the tone comes closer, while the coherence (the peak over lags)
# stays close to 1. Binaural beats are heard only at low frequencies, and only for small
# differences (Licklider, Webster & Hedlun, 1950). Mixed into the same sound, the same two tones
# beat in loudness instead: [Synthetic sounds](classic.html#d-b1) plays 440 and 444 Hz added
# together, and [440 and 480 Hz](classic.html#d-b2), whose 40 Hz beat is heard as roughness.

# %% [demo b4] Binaural beats, 4 Hz
sound = finish(so.Sound.from_channels(so.pure_tone(3, FS, 440), so.pure_tone(3, FS, 444), fs=FS))
fig, playhead = show(sound)
t = so.interaural_cues(sound, win_dur=10e-3).t
playhead[2].plot(t, np.cos(2 * np.pi * 4 * t), color="k", ls=":", lw=1, label="cos(2π·4t)")
playhead[2].legend(loc="lower right", fontsize=8)

# %% [markdown]
# Siveke et al. (2008) asked how fast the binaural system can follow such changes, using two noises
# in which each ear alone hears plain noise. Phasewarp, which they introduced, is a binaural beat
# for every component of a noise at once: the right ear hears the left ear's noise shifted up in
# frequency by the modulation rate (sonore makes the shift with a single-sideband modulator), so
# every component's interaural phase rotates at that rate, as the tones' phase did above. Oscor,
# an older stimulus they also used, makes the right ear's noise a moving mixture of the left ear's
# noise and an independent one: the correlation swings between +1 and −1, but the interaural
# phase is only ever 0 or 180°, so there is no ITD that moves.

# %% [about]
# Phasewarp at 2 Hz. The interaural phase of every component rotates through 360° twice a
# second, so the zero-lag correlation follows a 2 Hz cosine. Siveke et al. describe it as a noise
# that rotates around the head.

# %% [demo 08] Phasewarp
sound = finish(so.phasewarp(4, FS, f_mod=2, rng=0))
fig, playhead = show(sound)

# %% [about]
# Oscor at 3 Hz. The interaural correlation follows a 3 Hz sine between +1 and −1, and the image
# alternates between focused and diffuse rather than moving. Each ear alone is plain noise; only
# the correlation panel shows what changes.

# %% [demo 07] Oscor
sound = finish(so.oscor(4, FS, f_mod=3, rng=0))
fig, playhead = show(sound)

# %% [markdown]
# ## References
#
# - Licklider, Webster & Hedlun (1950). On the frequency limits of binaural beats. *J. Acoust.
#   Soc. Am.* 22(4), 468–473. [doi:10.1121/1.1906629](https://doi.org/10.1121/1.1906629).
# - Siveke, Ewert, Grothe & Wiegrebe (2008). Psychophysical and physiological evidence for fast
#   binaural processing. *J. Neurosci.* 28(9), 2043–2052.
#   [doi:10.1523/JNEUROSCI.4488-07.2008](https://doi.org/10.1523/JNEUROSCI.4488-07.2008).
#   [`binaural.oscor`](https://github.com/choyun1/sonore/blob/main/src/sonore/spatial/binaural.py#L182)
#   [`binaural.phasewarp`](https://github.com/choyun1/sonore/blob/main/src/sonore/spatial/binaural.py#L191)
