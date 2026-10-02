"""Iterated rippled noise: a pitch made from noise and a delay.

This script is the gallery page https://choyun1.github.io/sonore/gallery/irn.html:
docs/gallery/build.py runs it cell by cell from the repository root and shows each
cell's code beside what it made. Run it yourself from the repository root,

    python docs/gallery/irn.py

or a cell at a time ("# %%" starts a cell in VS Code, Spyder and Jupytext).
"""

# %% [markdown]
# # Iterated rippled noise
#
# Add noise to a copy of itself delayed by $d$, and do it again to the result, many times over.
# What comes out is still noise, with no tones in it, yet it has a pitch at $1/d$ (Yost, 1996).
# It is a classic stimulus for pitch because its strength can be turned up step by step.
#
# - [How it is made](#h-how-it-is-made): the delay-and-add network, and what it does to the
#   spectrum and the autocorrelation.
# - [Pitch from delay](#h-pitch-from-delay): an 8 ms delay, sixteen times over, and its pitch at
#   125 Hz.
# - [Fewer iterations](#h-fewer-iterations): one and four iterations, from a faint pitch to a clear
#   one.
# - [Subtracting instead](#h-subtracting-instead): a negative gain, whose pitch is ambiguous.
# - [Rippled noise and moving ripples](#h-rippled-noise-and-moving-ripples): how its rippled
#   spectrum relates to the [spectrotemporal ripples](ripples.html).

# %% [markdown]
# ## How it is made
#
# `so.iterated_ripple_noise` starts from Gaussian noise $x$ and repeats $y \leftarrow y + g\,y(t -
# d)$ for the given number of iterations $n$ (the "add-same" network; `network="add-original"`
# adds the delayed copy to the original noise instead). In frequency the delay-and-add is a comb
# filter, so the whole network is
#
# $$H(f) = \left(1 + g\,e^{-i 2\pi f d}\right)^n,$$
#
# whose peaks, with $g = 1$, sit at multiples of $1/d$ and grow sharper with each iteration. In
# time, the noise becomes partly correlated with itself $d$ later: its autocorrelation has a peak
# at lag $d$, and its height, which grows with $n$, goes with how strong the pitch is. sonore
# computes the network exactly in the frequency domain, so the delay need not be a whole number
# of samples, and drops a warm-up stretch so that the noise is stationary from the start.

# %%
import matplotlib.pyplot as plt
import numpy as np

import sonore as so

plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "figure.dpi": 100})
FS = 44100


def finish(snd):
    """How every sound in the gallery is played: 5 ms ramps, RMS 0.1, peak at most 0.95."""
    snd = snd.ramp(5e-3).normalize(rms=0.1)
    return snd.normalize(peak=0.95) if snd.peak > 0.95 else snd


def show(snd):
    """so.overview: waveform, spectrum, spectrogram (50 ms windows) and modulation spectrum.
    Returns the figure and the panels the playhead follows."""
    fig = so.overview(snd, win_dur=50e-3, figsize=(10, 6.2), fmax=4000)
    return fig, [ax for ax in fig.axes if ax.get_title() in ("Waveform", "Spectrogram")]


# %% [markdown]
# ## Pitch from delay

# %% [about]
# Noise added to an 8 ms delayed copy of itself, sixteen times over. A 125 Hz pitch rises out
# of the hiss; the spectrum ripples at multiples of 125 Hz and the modulation spectrum peaks at
# 8 cycles/kHz, the delay in milliseconds.

# %% [demo 09] Iterated rippled noise
sound = finish(so.iterated_ripple_noise(2, FS, delay=8e-3, iterations=16, rng=0))
fig, playhead = show(sound)

# %% [markdown]
# ## Fewer iterations
#
# The same delay and the same noise, with fewer iterations. The ripple in the spectrum is
# shallower and the autocorrelation peak at 8 ms lower, and the pitch is weaker.

# %% [about]
# One iteration: noise plus a single delayed copy. The spectrum is a gentle cosine ripple, and
# the pitch is faint, a coloring of the hiss rather than a note.

# %% [demo i1] One iteration
sound = finish(so.iterated_ripple_noise(2, FS, delay=8e-3, iterations=1, rng=0))
fig, playhead = show(sound)

# %% [about]
# Four iterations: the peaks at multiples of 125 Hz are sharper, and the pitch is clear.

# %% [demo i4] Four iterations
sound = finish(so.iterated_ripple_noise(2, FS, delay=8e-3, iterations=4, rng=0))
fig, playhead = show(sound)

# %% [about]
# The normalized autocorrelation of each sound over its first 30 ms. The peak at 8 ms grows
# with the number of iterations (0.47, 0.75 and 0.88), and so do the peaks at its multiples.

# %% [figure i0] Autocorrelation and iterations
fig, axes = plt.subplots(3, 1, figsize=(10, 4.8), sharex=True, layout="constrained")
for ax, n, color in zip(axes, (1, 4, 16), ("tab:blue", "tab:orange", "tab:green"), strict=True):
    x = so.iterated_ripple_noise(2, FS, delay=8e-3, iterations=n, rng=0).data[:, 0]
    X = np.fft.rfft(x, 2 * len(x))
    r = np.fft.irfft(np.abs(X) ** 2)[: int(0.030 * FS)]
    ax.plot(1e3 * np.arange(len(r)) / FS, r / r[0], color=color, lw=1)
    ax.set(ylim=(-0.3, 1.05), ylabel="Autocorr.", title=f"{n} iteration{'s' * (n > 1)}")
    ax.grid(ls=":")
axes[-1].set(xlabel="Lag [ms]", xlim=(0, 30))

# %% [markdown]
# ## Subtracting instead

# %% [about]
# Sixteen iterations with gain $g = -1$: the delayed copy is subtracted. The spectral peaks
# move to odd multiples of 62.5 Hz, halfway between the ones above, and the autocorrelation at
# 8 ms turns negative, with the first positive peak at 16 ms. The pitch is ambiguous: neither
# simply 125 Hz nor an octave below (Yost, 1996).

# %% [demo in] Negative gain
sound = finish(so.iterated_ripple_noise(2, FS, delay=8e-3, gain=-1, iterations=16, rng=0))
fig, playhead = show(sound)

# %% [markdown]
# ## Rippled noise and moving ripples
#
# The rippled spectrum here is one kind of ripple; the [spectrotemporal ripples](ripples.html) are
# another. This one does not move, so all of it sits at zero rate in a modulation spectrum, and it
# is periodic along *linear* frequency, one peak every $1/d$ hertz: an 8 ms delay gives 8
# cycles/kHz, where the modulation spectrum of the first example peaks. Spectrotemporal ripples
# are sinusoidal along *log* frequency and drift over time. Equally spaced peaks along linear
# frequency are harmonics of $1/d$, which is why rippled noise has a pitch and those ripples do
# not.

# %% [markdown]
# ## References
#
# - Yost (1996). Pitch of iterated rippled noise. *J. Acoust. Soc. Am.* 100(1), 511–518.
#   [JASA (PDF)](https://pubs.aip.org/asa/jasa/article-pdf/100/1/511/11401642/511_1_online.pdf).
#   [`generators.iterated_ripple_noise`](https://github.com/choyun1/sonore/blob/main/src/sonore/signals/generators.py#L496)
