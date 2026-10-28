"""Analysis and resynthesis: taking a sound apart, changing the parts, and putting it back.

This script is the gallery page https://choyun1.github.io/sonore/gallery/resynthesis.html:
docs/gallery/build.py runs it cell by cell from the repository root and shows each
cell's code beside what it made. Run it yourself from the repository root,

    python docs/gallery/seeing/resynthesis.py

or a cell at a time ("# %%" starts a cell in VS Code, Spyder and Jupytext).
"""

# %% [markdown]
# # Analysis and resynthesis
#
# Many of sonore's analyses can be run backwards: split a sound into bands or time-frequency
# cells, change what you find there, and synthesize a sound from the result. This page goes from
# doing nothing to removing the noise from a mixture.
#
# - [Perfect reconstruction](#h-perfect-reconstruction): a filterbank whose bands sum back to the
#   original exactly.
# - [Nothing is lost](#h-nothing-is-lost): every short-time Fourier transform and filterbank drawn
#   on [Seeing speech](speech.html) gives the sentence back, and why overlap-add works.
# - [Keeping fewer numbers](#h-keeping-fewer-numbers): a filterbank that keeps each band at its own
#   rate, an exact constant-Q transform with about two numbers per sample.
# - [Masking the spectrogram](#h-masking-the-spectrogram): switching off the cells where noise
#   dominates, the ideal binary mask.
#
# Three other pages go further: the [Phase vocoder](pv.html) changes a sound's duration, pitch or
# partials, [Hearing through a vocoder](vocoder.html) keeps only band envelopes, and [The source-filter
# model](sourcefilter.html) separates a voice's spectral envelope from its source.

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
from sonore import dB

FS = 44100

# %% [markdown]
# ## Perfect reconstruction
#
# `so.cosine_filterbank` splits a sound with half-cosine filters equally spaced on the ERB scale, plus a
# lowpass and a highpass filter at the edges. Their squared responses sum to 1 at every
# frequency, so analysis followed by synthesis (each band filtered again, then summed) gives back
# the original, up to floating-point rounding.


# %%
def show_bands(original):
    """The six bands of the sweep beside the original, the reconstruction and their difference.
    Returns the figure and the panels the playhead follows."""
    fig = plt.figure(figsize=(10, 6.2), layout="constrained")
    left, right = fig.subfigures(1, 2, width_ratios=[1.35, 1])
    sb = so.cosine_filterbank(6, 100, 6000).analyze(original)
    band_axes = left.subplots(len(sb), 1, sharex=True)
    sb.plot(band_axes)
    band_axes[0].set_title(
        "so.cosine_filterbank(6, 100, 6000).analyze(sweep)", family="monospace", fontsize=8
    )
    r = right.subplots(3, 1, sharex=True)
    original.plot(r[0], color="k", lw=0.4)
    r[0].set(title="Original sweep, 100 Hz to 6 kHz", xlabel="")
    recon = sb.to_sound()  # recomputed: the played sound has been level-normalized
    recon.plot(r[1], color="tab:blue", lw=0.4)
    r[1].set(title="Reconstruction: sb.to_sound()", xlabel="")
    err = recon - original
    r[2].plot(err.t, 1e15 * err.data[:, 0], color="tab:red", lw=0.5)
    r[2].set(title=f"Difference (max |error| = {err.peak:.1e})", ylabel="× 1e-15", xlabel="Time [s]")
    r[2].grid(ls=":")
    return fig, list(band_axes) + list(r)


# %% [about]
# An exponential sweep split into 6 ERB-spaced bands plus the lowpass and highpass edge filters,
# then summed back. What you hear is the reconstruction; it differs from the original only at
# the level of floating-point rounding.

# %% [demo 26] Perfect reconstruction
sweep = so.exponential_chirp(2.0, FS, 100, 6000).ramp(20e-3)
sound = finish(so.cosine_filterbank(6, 100, 6000).analyze(sweep).to_sound())
fig, playhead = show_bands(sweep)

# %% [markdown]
# ## Nothing is lost
#
# The cosine filterbank is one *frame*: a set of analysis functions whose coefficients determine
# the signal, so that synthesis recovers it. The short-time Fourier transform (STFT) is another.
# For the STFT with window $w$ and hop $H$, overlap-adding the windowed inverse transforms of the
# coefficients gives back the signal multiplied by
#
# $$s(t) = K \sum_q \bigl|w(t - qH)\bigr|^2,$$
#
# with $K$ the FFT length, so synthesis divides by $s(t)$, which works wherever the windows leave
# no gap. A filterbank's analogue is $s(f) = \sum_k |H_k(f)|^2$, divided out in frequency; for the
# cosine filterbank above it is 1. The time-varying windows of a pitch-adaptive frame work the
# same way, each with its own $w_q$.
#
# [Seeing speech](speech.html) draws one sentence through many analyses, with pictures that differ a
# great deal. The cells below load that sentence, read by each of the [two
# talkers](whatisavoice.html#h-the-two-talkers), and take each recording through each of those
# analyses and back.

# %%
# The two talkers, by the CMU ARCTIC speaker names. Sources: docs/speech/SOURCES.md.
SPEAKERS = {"Male talker": "bdl", "Female talker": "slt"}
talkers = {
    label: finish(so.load(fetch(f"docs/speech/{speaker}_arctic_a0131.flac")))
    for label, speaker in SPEAKERS.items()
}
tracks = {label: so.f0_track(snd) for label, snd in talkers.items()}

# %%
frames = {
    "Hann 2 ms": so.GaborFrame(0.002, 0.0005, n_fft=2048),
    "Hann 5 ms": so.GaborFrame(0.005, 0.001, n_fft=1024),
    "Hann 33.3 ms": so.GaborFrame(0.0333, 0.001, n_fft=1024),
    "Hann 100 ms": so.GaborFrame(0.1, 0.0005, n_fft=2048),
    "Morlet, 6 cycles": so.morlet_filterbank(54, 70, 7000, cycles=6),
    "gammatone": so.gammatone_filterbank(60, 70, 7000),
}
for label, snd in talkers.items():
    track = tracks[label]
    f0 = np.where(track.voiced[0], track.f0[0], 0)  # 0 where unvoiced
    frames["pitch-adaptive, 3 periods"] = so.TVGaborFrame.pitch_adaptive(
        track.t, f0, t_end=snd.duration + 0.02, periods=3
    )
    for name, frame in frames.items():
        back = frame.synthesize(frame.analyze(snd))
        error = np.max(np.abs(back.data - snd.data)) / snd.peak
        print(f"{label}, {name}: largest error {error:.1e} of the peak")

# %% [markdown]
# Every frame gives each recording back to within floating-point rounding. However differently
# the pictures on Seeing speech look, no analysis there loses anything: the tradeoff between
# resolution in time and in frequency appears only once the phase is discarded and the magnitudes
# are drawn. Two pictures on that page are not frames, the TANDEM-style power spectrum and the
# reassigned spectrogram: they keep only magnitudes, and have no synthesis.

# %% [markdown]
# ## Keeping fewer numbers
#
# Exact is not the same as economical. A filterbank keeps one number per band for every sample of
# the sound, so a bank of 85 filters turns each second of sound into 85 seconds' worth of numbers.
# A band only a few hertz wide cannot change faster than its width allows, so most of those
# numbers are redundant. A *decimated* filterbank keeps each band at its own rate: band $k$, whose
# filter is zero outside a stretch of $L_k$ frequency bins, is held by $M_k \ge L_k$ complex
# coefficients, taken $N / M_k$ samples apart on a sound of $N$ samples. Because no two of its bins
# fold onto the same coefficient, nothing is aliased and synthesis is still exact. These are the
# nonstationary Gabor frames of Balazs et al. (2011), here in frequency; the pitch-adaptive frame
# above is the same idea in time.
#
# On the semitone bank of [Pitch class and height](chroma.html#h-a-constant-q-filter-bank), 83
# cosines one semitone apart from 55 Hz to 7040 Hz plus the two edge filters, this is an exact
# constant-Q transform. The low bands are a few hertz wide and need a coefficient only every
# hundred milliseconds or so; the high bands are hundreds of hertz wide and need one every
# millisecond or two.

# %%
semitones = so.cosine_filterbank(f_lo=55, f_hi=7040, scale="cents", spacing=100)
constant_q = so.DecimatedFilterbank(semitones)
decimated = {label: constant_q.analyze(snd) for label, snd in talkers.items()}
for label, snd in talkers.items():
    coefs = decimated[label]
    numbers = 2 * sum(len(band) for band in coefs) / coefs.n_padded  # real numbers per sample
    error = np.max(np.abs(coefs.to_sound().data - snd.data)) / snd.peak
    print(
        f"{label}: {semitones.n_filters} numbers per sample undecimated, {numbers:.2f} decimated; "
        f"largest error {error:.1e} of the peak"
    )
hops = [1e3 * (coefs.times[k][1] - coefs.times[k][0]) for k in (1, len(coefs) - 2)]
print(
    f"time between coefficients: {hops[0]:.0f} ms at {semitones.cfs[1]:.0f} Hz, "
    f"{hops[1]:.1f} ms at {semitones.cfs[-2]:.0f} Hz"
)


# %%
def show_decimated(coefs_by_talker, title):
    """Each talker's decimated coefficients. Returns the figure and the panels the playhead follows."""
    fig = plt.figure(figsize=(10, 3.6), layout="constrained")
    panels = {}
    columns = fig.subfigures(1, 2, wspace=0.03)
    for column, (label, coefs) in zip(columns, coefs_by_talker.items(), strict=True):
        ax = column.subplots()
        coefs.plot(ax, colorbar=False, db_range=60)
        ax.set_title(f"{label}: {title}")
        panels[label] = [ax]
    return fig, panels


# %% [about]
# Each band drawn with its own cells: long ones at the bottom, where a band has few coefficients,
# short ones at the top. The harmonics are resolved where the bands are narrower than their spacing,
# and the glottal pulses where the cells are shorter than the period. What you hear is the sentence
# synthesized back from these coefficients, the same to within rounding.

# %% [demo dq1] A decimated constant-Q transform
sounds = {label: finish(coefs.to_sound()) for label, coefs in decimated.items()}
fig, playhead = show_decimated(decimated, "all coefficients")

# %% [markdown]
# The coefficients can be changed and played like any others. Keeping only the largest few, and
# setting the rest to zero, is the simplest test of how much of the sound they concentrate in
# few numbers. Each bandpass coefficient stands for its band's negative frequencies too, so it
# counts twice in the energy.

# %%
KEEP = 0.02  # the share of coefficients kept
weights = [1.0] + [2.0] * (len(semitones.cfs) - 2) + [1.0]  # the edge filters count once
sparse, snrs = {}, {}
for label, coefs in decimated.items():
    energy = np.concatenate([w * np.abs(band[:, 0]) ** 2 for w, band in zip(weights, coefs, strict=True)])
    cutoff = np.quantile(energy, 1 - KEEP)
    kept_bands = [
        np.where(w * np.abs(band) ** 2 >= cutoff, band, 0) for w, band in zip(weights, coefs, strict=True)
    ]
    sparse[label] = so.DecimatedSubbands(kept_bands, coefs.fs, constant_q, coefs.n_padded, pad=coefs.pad)
    snd = talkers[label]
    difference = sparse[label].to_sound() - snd
    snrs[label] = 10 * np.log10(np.sum(snd.data**2) / np.sum(difference.data**2))
    kept = energy[energy >= cutoff].sum() / energy.sum()
    print(
        f"{label}: the largest {KEEP:.0%} keep {kept:.1%} of the energy; "
        f"the error is {snrs[label]:.1f} dB below the sentence"
    )

# %% [about]
# Only the largest {{ f"{KEEP:.0%}" }} of each talker's coefficients, the rest set to zero, synthesized
# back. The difference from the sentence is {{ f"{snrs['Male talker']:.0f}" }} dB below it for the
# male talker and {{ f"{snrs['Female talker']:.0f}" }} dB for the female talker.

# %% [demo dq2] The largest coefficients only
sounds = {label: finish(coefs.to_sound()) for label, coefs in sparse.items()}
fig, playhead = show_decimated(sparse, f"largest {KEEP:.0%}")

# %% [markdown]
# ## Masking the spectrogram
#
# Since the STFT is invertible, a spectrogram with some of its cells switched off is still a
# sound: synthesis gives the sound whose STFT is closest, in the least-squares sense, to the
# masked coefficients. Wang (2005) proposed the *ideal binary mask* as the goal of separating speech from
# noise: keep every cell where the target is louder than the noise, discard the rest. It is
# ideal because computing it needs the target and the noise separately, which a listener never
# has; what a listener hears through it shows how much a perfect separation would leave.


# %%
def gliding_target(dur=2.0):
    """A harmonic complex gliding 150 -> 250 Hz with a 3 Hz level fluctuation: a stand-in for speech."""
    t = np.arange(int(dur * FS)) / FS
    phase = 2 * np.pi * np.cumsum(150 + 50 * t) / FS
    target = so.Sound(sum(np.cos(k * phase) / k for k in range(1, 30)), FS).normalize()
    return (target * (0.6 + 0.4 * np.sin(2 * np.pi * 3 * t))).ramp(20e-3)


def show_mask(snd, target, mask):
    """Waveform, the mask, and spectrograms of this sound and of the target alone.
    Returns the figure and the panels the playhead follows."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.2), sharex=True, layout="constrained")
    snd.plot(axes[0, 0], lw=0.4)
    axes[0, 0].set_title("Waveform")
    mask.plot(axes[0, 1])
    axes[0, 1].set(title="Ideal binary mask (target > noise)", ylim=(0, 5))
    so.STFT(snd, win_dur=25e-3).plot(axes[1, 0], fmax=5000, colorbar=False)
    axes[1, 0].set_title("Spectrogram of this sound")
    so.STFT(target, win_dur=25e-3).plot(axes[1, 1], fmax=5000, colorbar=False)
    axes[1, 1].set_title("The target alone (not played)")
    return fig, [axes[0, 0], axes[1, 0]]


target = gliding_target()
masker = so.gaussian_noise(target.duration, FS, tilt=-3, rng=0)
mixture = target + (masker + 5 * dB)
S_t, S_m, S_x = (so.STFT(x, win_dur=25e-3) for x in (target, masker, mixture))
mask = so.ideal_binary_mask(S_t, S_m, lc_db=0)
snr = 20 * np.log10(target.rms / (masker + 5 * dB).rms)
print(f"signal-to-noise ratio: {snr:.1f} dB")

# %% [about]
# A gliding harmonic target (a stand-in for a voice) in pink noise, at a signal-to-noise ratio of
# {{ f"{snr:.1f}" }} dB.

# %% [demo 24] Target in noise
sound = finish(mixture)
fig, playhead = show_mask(sound, target, mask)

# %% [about]
# The same mixture with every time-frequency cell where the noise dominates switched off. The
# mask is computed from the separate target and noise, which is what makes it ideal; the
# resynthesis is the least-squares one described above.

# %% [demo 25] Ideal binary mask
sound = finish((S_x * mask).to_sound())
fig, playhead = show_mask(sound, target, mask)

# %% [markdown]
# ## References
#
# - Balazs, Dörfler, Jaillet, Holighaus & Velasco (2011). Theory, implementation and applications
#   of nonstationary Gabor frames. *J. Comput. Appl. Math.* 236, 1481–1496.
#   [doi:10.1016/j.cam.2011.09.011](https://doi.org/10.1016/j.cam.2011.09.011).
#   [`filterbank.DecimatedFilterbank`](https://github.com/choyun1/sonore/blob/main/src/sonore/frames/filterbank.py#L888)
# - Wang (2005). On ideal binary mask as the computational goal of auditory scene analysis. In
#   *Speech Separation by Humans and Machines*, 181–197. Springer.
#   [doi:10.1007/0-387-22794-6_12](https://doi.org/10.1007/0-387-22794-6_12).
#   [`mask.ideal_binary_mask`](https://github.com/choyun1/sonore/blob/main/src/sonore/views/mask.py#L145)
