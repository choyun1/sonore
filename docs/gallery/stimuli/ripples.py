"""Spectrotemporal ripples: sounds defined by their modulation content.

This script is the gallery page https://choyun1.github.io/sonore/gallery/ripples.html:
docs/gallery/build.py runs it cell by cell from the repository root and shows each
cell's code beside what it made. Run it yourself from the repository root,

    python docs/gallery/stimuli/ripples.py

or a cell at a time ("# %%" starts a cell in VS Code, Spyder and Jupytext).
"""

# %% [markdown]
# # Spectrotemporal ripples
#
# A ripple is a sinusoidal pattern drawn on the time-frequency plane: its level rises and falls
# along frequency, and the whole pattern drifts over time. Ripples are to hearing what gratings
# are to vision. They were used to map the spectrotemporal receptive fields of auditory neurons
# (Kowalski et al., 1996) and to measure how sensitive listeners are to modulation (Chi et al.,
# 1999).
#
# - [Moving ripples](#h-moving-ripples): a ripple drifting down, drifting up, and two at once,
#   each with its measured modulation spectrum.
# - [Carriers](#h-carriers): the same pattern on log-spaced tones, harmonics and noise.
# - [Dynamic moving ripple](#h-dynamic-moving-ripple): a ripple whose rate and density wander at
#   random.
# - [Ripples and rippled noise](#h-ripples-and-rippled-noise): how these relate to the static
#   ripple of [iterated rippled noise](irn.html).

# %% [markdown]
# ## How a ripple is made
#
# `so.Ripple(rate, density, depth)` is the envelope
#
# $$E(t, x) = 1 + m \sin\bigl(2\pi(\omega t + \Omega x) + \varphi\bigr),$$
#
# where $x$ is frequency in octaves above the lowest, $\omega$ the rate in Hz, $\Omega$ the density
# in cycles per octave and $m$ the depth. A positive rate drifts down in frequency, a negative one
# up. Ripples add: `so.Ripple(...) + so.Ripple(...)` is a sum of patterns. `so.ripple_sound`
# imposes the envelope on a carrier spanning 250 Hz to 8 kHz, by default 20 log-spaced tones per
# octave with random phases.
#
# Each figure shows the pattern as specified, the envelopes measured back from the sound in 24
# bands per octave (a cochleagram), the waveform, and the modulation spectrum measured from the
# sound: the two-dimensional Fourier transform of the cochleagram, with rate across and density
# up (Singh & Theunissen, 2003). A single ripple puts a single peak in it.

# %%
import matplotlib.pyplot as plt

import sonore as so

plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "figure.dpi": 100})
FS = 44100


def finish(snd):
    """How every sound in the gallery is played: 5 ms ramps, RMS 0.1, peak at most 0.95."""
    snd = snd.ramp(5e-3).normalize(rms=0.1)
    return snd.normalize(peak=0.95) if snd.peak > 0.95 else snd


def show(snd, pattern, dmr=False):
    """The pattern as specified, then the sound's cochleagram, waveform and modulation spectrum.
    Returns the figure and the panels the playhead follows."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.2), layout="constrained")
    pattern.plot(duration=snd.duration, f_lo=250, f_hi=8000, ax=axes[0, 0], colorbar=False)
    axes[0, 0].set_title("Pattern as specified (envelope, dB)")
    bank = so.cosine_filterbank(f_lo=250, f_hi=8000, spacing=1 / 24, scale="octave")
    bank.analyze(snd).envelopes(lowpass=200, fs=1000).plot(axes[0, 1], db_range=30, colorbar=False)
    axes[0, 1].set_title("Measured envelopes of the sound (cochleagram)")
    snd.plot(axes[1, 0], lw=0.4)
    axes[1, 0].set_title("Waveform (the pattern barely shows here)")
    ms = so.ModulationSpectrum.octave(snd, f_lo=250, f_hi=8000, scale="db" if dmr else "linear")
    ms.plot(axes[1, 1], db_range=30, wt_max=50 if dmr else 20, wf_max=4, colorbar=False)
    axes[1, 1].set_title("Measured modulation spectrum")
    return fig, [axes[0, 0], axes[0, 1], axes[1, 0]]


# %% [markdown]
# ## Moving ripples

# %% [about]
# Rate 4 Hz, density 1 cycle/octave, on log-spaced tones. Listen for a continuous downward
# sweep, four times a second. The diagonal bands in the cochleagram are what you are hearing;
# the waveform alone shows almost none of it.

# %% [demo 01] Downward ripple
down = so.Ripple(4, 1)
sound = finish(so.ripple_sound(down, 3, FS, rng=0))
fig, playhead = show(sound, down)

# %% [about]
# The same ripple at −4 Hz, so the sweep rises. Its modulation-spectrum peak moves to negative
# rate.

# %% [demo 02] Upward ripple
up = so.Ripple(-4, 1)
sound = finish(so.ripple_sound(up, 3, FS, rng=0))
fig, playhead = show(sound, up)

# %% [about]
# 4 Hz at 1 cycle/octave plus −12 Hz at 2.5 cycles/octave: two motions in opposite directions,
# superimposed. The modulation spectrum separates them into two clean peaks.

# %% [demo 03] Two ripples at once
both = so.Ripple(4, 1, depth=0.45) + so.Ripple(-12, 2.5, depth=0.45)
sound = finish(so.ripple_sound(both, 3, FS, rng=0))
fig, playhead = show(sound, both)

# %% [markdown]
# ## Carriers
#
# The pattern is an envelope; what carries it changes how it sounds, not where it sits in the
# modulation spectrum.

# %% [about]
# The first ripple on harmonics of 60 Hz. The motion is the same, now over a low pitch; resolved
# low harmonics show as horizontal lines in the cochleagram.

# %% [demo 04] Same ripple, harmonic carrier
sound = finish(so.ripple_sound(down, 3, FS, carrier="harmonic", f0=60, rng=0))
fig, playhead = show(sound, down)

# %% [about]
# The same pattern on narrowband noise, which brings its own random fluctuations, so the sweep
# sounds rougher.

# %% [demo 05] Same ripple, noise carrier
sound = finish(so.ripple_sound(down, 3, FS, carrier="noise", rng=0))
fig, playhead = show(sound, down)

# %% [markdown]
# ## Dynamic moving ripple
#
# Escabí & Schreiner (2002) let a ripple's rate and density wander slowly and at random, so that
# one long sound covers many combinations of the two: a stimulus for mapping receptive fields
# without choosing the ripples in advance.

# %% [about]
# Rate and density wander randomly (rates within ±40 Hz at the lowest frequency). Where the
# density passes through zero, every band pulses together, and only there does the waveform show
# the modulation.

# %% [demo 06] Dynamic moving ripple
dmr = so.DynamicRipple(rate_range=(-40, 40), seed=3)
sound = finish(so.ripple_sound(dmr, 4, FS, rng=0))
fig, playhead = show(sound, dmr, dmr=True)

# %% [markdown]
# ## Ripples and rippled noise
#
# [Iterated rippled noise](irn.html) has a rippled spectrum too, but a different kind. Its ripple
# does not move, so all of it sits at zero rate; and it is periodic on a *linear* frequency axis,
# one peak every $1/d$ hertz, rather than on the logarithmic axis used here. An 8 ms delay gives a
# density of 8 cycles/kHz, which is where its modulation spectrum peaks. Equally spaced peaks
# along linear frequency are harmonics, which is why rippled noise has a pitch and these ripples,
# sinusoidal along log frequency, do not.

# %% [markdown]
# ## References
#
# - Chi, Gao, Guyton, Ru & Shamma (1999). Spectro-temporal modulation transfer functions and
#   speech intelligibility. *J. Acoust. Soc. Am.* 106(5), 2719–2732.
#   [JASA](https://pubs.aip.org/asa/jasa/article/106/5/2719/550617).
#   [`ripples.Ripple`](https://github.com/choyun1/sonore/blob/main/src/sonore/sources/ripples.py#L86)
#   [`modulation.ModulationSpectrum`](https://github.com/choyun1/sonore/blob/main/src/sonore/views/modulation.py#L333)
# - Escabí & Schreiner (2002). Nonlinear spectrotemporal sound analysis by neurons in the auditory
#   midbrain. *J. Neurosci.* 22(10), 4114–4131.
#   [doi:10.1523/JNEUROSCI.22-10-04114.2002](https://doi.org/10.1523/JNEUROSCI.22-10-04114.2002).
#   [`ripples.DynamicRipple`](https://github.com/choyun1/sonore/blob/main/src/sonore/sources/ripples.py#L163)
# - Kowalski, Depireux & Shamma (1996). Analysis of dynamic spectra in ferret primary auditory
#   cortex. I. Characteristics of single-unit responses to moving ripple spectra. *J.
#   Neurophysiol.* 76(5), 3503–3523.
#   [doi:10.1152/jn.1996.76.5.3503](https://doi.org/10.1152/jn.1996.76.5.3503).
#   [`ripples.Ripple`](https://github.com/choyun1/sonore/blob/main/src/sonore/sources/ripples.py#L86)
# - Singh & Theunissen (2003). Modulation spectra of natural sounds and ethological theories of
#   auditory processing. *J. Acoust. Soc. Am.* 114(6), 3394–3411.
#   [doi:10.1121/1.1624067](https://doi.org/10.1121/1.1624067).
#   [`modulation.ModulationSpectrum`](https://github.com/choyun1/sonore/blob/main/src/sonore/views/modulation.py#L333)
