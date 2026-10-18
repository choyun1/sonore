"""Modulation spectrogram: how a sound's modulation spectrum changes over time.

This script is the gallery page https://choyun1.github.io/sonore/gallery/modspectrogram.html:
docs/gallery/build.py runs it cell by cell from the repository root and shows each
cell's code beside what it made. Run it yourself from the repository root,

    python docs/gallery/seeing/modspectrogram.py

or a cell at a time ("# %%" starts a cell in VS Code, Spyder and Jupytext).
"""

# %% [markdown]
# # Modulation spectrogram
#
# A spectrogram shows how a sound's power spectrum changes over time. A *modulation
# spectrogram* does the same for its modulation spectrum: how fast, and how deeply, the
# envelope of each frequency band rises and falls, moment by moment (Greenberg & Kingsbury,
# 1997; Kingsbury et al., 1998). It has three axes, time, acoustic frequency and modulation
# rate, so it is a cube, and a page can only show cuts through it. The main one here is
# modulation rate against time, pooled over the acoustic bands. The other is acoustic frequency
# against modulation rate at one moment (Atlas & Shamma, 2003): beside the player it follows the
# sound as it plays, and under the speech it is drawn at three moments.
#
# This is a different modulation spectrum from the two-dimensional one of rates and spectral
# densities, measured over a whole sound, that [Spectrotemporal ripples](ripples.html) defines
# and [Hearing a modulation spectrum](modtargets.html) turns back into sound.
#
# - [How it is computed](#h-how-it-is-computed): band envelopes, then a bank of windowed
#   complex exponentials on each envelope.
# - [A modulation rate that glides](#h-a-modulation-rate-that-glides): a pattern a spectrogram
#   cannot show.
# - [Modulation that glides in frequency](#h-modulation-that-glides-in-frequency): the same
#   picture turned on its side, and rate and frequency gliding together.
# - [Speech, babble and noise](#h-speech-babble-and-noise): the syllable rhythm of each talker,
#   blurred by more voices and gone in noise with the same spectrum.
# - [Three textures](#h-three-textures): crickets, applause and rain, each with its own rates.
# - [Three cuts through the cube](#h-three-cuts-through-the-cube): one moment, one rate and the
#   pooled view, side by side.
# - [What this page leaves out](#h-what-this-page-leaves-out): power, live analysis, inversion
#   and the front end.

# %% [markdown]
# ## How it is computed
#
# The sound first goes through 16 bands from 100 Hz to 7 kHz, equally spaced on the ERB scale
# (`so.cosine_filterbank`, as on [Analysis and resynthesis](resynthesis.html#h-perfect-reconstruction)),
# and each band's envelope is taken at 1000 Hz (a cochleagram). Each envelope then goes through a
# bank of modulation filters, one per rate up to 64 Hz. A filter is a complex exponential at its
# rate under a Hann window, so it measures the envelope's Fourier component at that rate over the
# window, the way one bin of an STFT does for the sound itself. The same window, without the
# exponential, measures the envelope's local mean, and the ratio of the two is the *modulation
# depth*:
#
# $$m_{b,k}(t) = \frac{2\,|y_{b,k}(t)|}{\mu_{b,k}(t)}$$
#
# A band whose envelope is $1 + m \sin(2\pi f_m t)$ reads $m$ at the rate $f_m$, so 100%
# modulation is 0 dB and 50% is $-6$ dB. Depth says how modulated a band is, whatever its
# level, which is what makes a quiet band's rhythm as visible as a loud one's. The pooled view
# adds the bands' modulation power and divides by their summed squared means, so there the louder
# bands count for more.
#
# The window length is a choice made for the sound at hand, as it is for an STFT. Here every rate
# shares one window, so the bank is the STFT of each envelope: rates sit on a grid of 1/T Hz for a
# window of T seconds, starting at two cycles per window. A 1 s window resolves rates 1 Hz apart
# from 2 Hz up but blurs each change over about half a second; a 0.25 s window places changes four
# times more precisely in time, with rates 4 Hz apart from 8 Hz up. The examples show both, like a
# narrowband and a wideband spectrogram. Cells drawn gray are ones the analysis cannot trust:
# where the window runs past either end of the sound, half a window at each end, the same at
# every rate. A rate faster than a band is wide can't be trusted either, since a band's envelope
# can't change faster than that, and is drawn gray too; with 16 bands even the lowest is wider
# than 64 Hz, so on this page that never happens.

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
# The two talkers, by the CMU ARCTIC speaker names. Sources: docs/speech/SOURCES.md.
SPEAKERS = {"Male talker": "bdl", "Female talker": "slt"}
talkers = {
    label: finish(so.load(fetch(f"docs/speech/{speaker}_arctic_a0131.flac")))
    for label, speaker in SPEAKERS.items()
}
tracks = {label: so.f0_track(snd) for label, snd in talkers.items()}

fs = 16000
rng = np.random.default_rng(0)


def analyze(snd):
    """The cochleagram, and modulation spectrograms with a 1 s and a 0.25 s window."""
    env = so.cosine_filterbank(n_bands=16, f_lo=100, f_hi=7000).analyze(snd).envelopes(fs=1000)
    return env, so.ModulationSpectrogram(env, window=1.0), so.ModulationSpectrogram(env, window=0.25)


def depth_db(msg):
    """Depth in dB for every band, rate and time window, NaN where the cell is not valid."""
    db = 20 * np.log10(msg.depth[0])
    db[~msg.valid] = np.nan
    return db


def rate_ticks(axis):
    """Label a logarithmic rate axis at whole octaves (2, 4, ..., 64 Hz) and nowhere else."""
    from matplotlib.ticker import FixedLocator, NullFormatter, ScalarFormatter

    axis.set_major_locator(FixedLocator([2, 4, 8, 16, 32, 64]))
    axis.set_major_formatter(ScalarFormatter())
    axis.set_minor_formatter(NullFormatter())


def draw(column, snd, marks):
    """The cochleagram and both modulation spectrograms, one above the other in ``column`` (a
    figure or subfigure), with dashed lines at ``marks`` [s]. Returns the analyses and axes."""
    env, msg, sharp = analyze(snd)
    axes = column.subplots(3, 1, sharex=True)
    env.plot(ax=axes[0])
    axes[0].set(title="Cochleagram", xlabel="")
    msg.plot(ax=axes[1], colorbar=False)
    axes[1].set(title="Modulation spectrogram, 1 s window (pooled over bands)", xlabel="")
    sharp.plot(ax=axes[2], colorbar=False)
    axes[2].set_title("Modulation spectrogram, 0.25 s window (pooled over bands)")
    # One color scale for both windows, so their depths can be compared
    top = max(np.nanmax(20 * np.log10(view.pooled_depth()[0])) for view in (msg, sharp))
    for ax in axes[1:]:
        ax.collections[0].set_clim(top - 30, top)
    column.colorbar(axes[1].collections[0], ax=axes[1:], label="Depth [dB]", shrink=0.9)
    for ax in axes[1:]:
        rate_ticks(ax.yaxis)
    for ax in axes:
        for m in marks:
            ax.axvline(m, color="c", lw=1, ls="--")
    return msg, list(axes)


def show(snd, marks=(), start=None):
    """``draw`` for one sound, and the band x rate depth image (1 s window) that follows the sound
    beside the player."""
    fig = plt.figure(figsize=(9, 8.4), layout="constrained")
    msg, axes = draw(fig, snd, marks)
    db = depth_db(msg)
    top = np.nanmax(db)
    # Only the time windows that lie wholly inside the sound: before the first one and after the
    # last, the image holds still. And only rates about half an octave apart, so the image's
    # even grid of cells reads as a logarithmic rate axis, as in the figures.
    measured = msg.valid.any(axis=(0, 1))
    columns = np.unique([np.argmin(np.abs(np.log(msg.fm / (2 * 2 ** (k / 2))))) for k in range(11)])
    live = {
        "t": msg.t[measured],
        "image": db[:, columns][:, :, measured],
        "x": msg.fm[columns],
        "y": msg.f,
        "range": (top - 30, top),
        "xlabel": "Modulation rate [Hz]",
        "ylabel": "Frequency [Hz]",
        "title": "Depth",
        "start": snd.duration / 2 if start is None else start,
    }
    return fig, axes, live


def show_pair(sounds, marks, moments):
    """One column per talker: ``draw``, and under it the band x rate depth image (1 s window) at
    each of that talker's ``moments`` [s], named by their keys. Returns the figure and, for
    each talker, the panels the playhead follows."""
    fig = plt.figure(figsize=(12, 11), layout="constrained")
    columns = fig.subfigures(1, 2)
    playhead = {}
    for column, (label, snd) in zip(columns, sounds.items(), strict=True):
        upper, lower = column.subfigures(2, 1, height_ratios=[3, 1])
        msg, axes = draw(upper, snd, marks[label])
        upper.suptitle(label, fontsize=11)
        db = depth_db(msg)
        top = np.nanmax(db)
        cuts = lower.subplots(1, len(moments[label]), sharey=True)
        for ax, (name, moment) in zip(cuts, moments[label].items(), strict=True):
            i = int(np.argmin(np.abs(msg.t - moment)))
            image = ax.pcolormesh(
                msg.fm, msg.f, db[:, :, i], cmap="magma", vmin=top - 30, vmax=top, shading="nearest"
            )
            image.cmap.set_bad("0.75")
            ax.set(xscale="log", yscale="log", title=f"{name}, {msg.t[i]:.1f} s", xlabel="Rate [Hz]")
            rate_ticks(ax.xaxis)
        cuts[0].set_ylabel("Frequency [Hz]")
        lower.colorbar(image, ax=cuts, label="Depth [dB]", shrink=0.9)
        playhead[label] = axes
    return fig, playhead


# %% [markdown]
# How much depth does a sound without any rhythm have? 20 s of white noise, through the same
# analysis. The envelope of a narrow band of noise fluctuates at random, so even steady noise has
# some depth at every rate. On the left, the depth of one band (the line is the median over
# bands and moments, the shading the middle 68% of moments) and of all bands pooled, for both
# windows. On the right, the 16 Hz filter of each window.

# %% [figure n1] The depth of steady noise
noise = so.Sound(rng.standard_normal(20 * fs), fs).normalize(rms=0.1)
_, noise_msg, noise_sharp = analyze(noise)
fig, (left, right) = plt.subplots(1, 2, figsize=(9, 3.4), layout="constrained", width_ratios=[3, 2])
rates = np.linspace(4, 28, 2001)
for msg, label, color in ((noise_msg, "1 s window", "C0"), (noise_sharp, "0.25 s window", "C1")):
    one_band = np.moveaxis(depth_db(msg), 1, 0).reshape(len(msg.fm), -1)  # rate x (band, moment)
    low, middle, high = np.nanpercentile(one_band, [16, 50, 84], axis=1)
    pooled = np.nanmedian(20 * np.log10(msg.pooled_depth()[0]), axis=1)
    left.fill_between(msg.fm, low, high, color=color, alpha=0.25, lw=0)
    left.plot(msg.fm, middle, color=color, label=f"{label}, one band")
    left.plot(msg.fm, pooled, color=color, ls="--", label=f"{label}, pooled")
    response = msg.bank.response(rates)[:, list(msg.fm).index(16.0)]
    right.plot(rates, response, color=color, label=label)
left.set(xscale="log", xlabel="Modulation rate [Hz]", ylabel="Depth [dB]", ylim=(-40, -5))
rate_ticks(left.xaxis)
left.legend(fontsize=8, ncols=2, loc="upper center")
right.set(xlabel="Modulation rate [Hz]", ylabel="Gain", title="The filter at 16 Hz")
right.legend(fontsize=8)

noise_db = depth_db(noise_msg)
band_median = np.nanmedian(noise_db, axis=(0, 2))
band_scatter = np.nanmedian(np.nanstd(noise_db, axis=2), axis=0)
pooled_scatter = np.nanstd(20 * np.log10(noise_msg.pooled_depth()[0]), axis=1)
sharper_by = np.nanmedian(depth_db(noise_sharp)) - np.nanmedian(noise_db)
response = noise_msg.bank.response(rates)[:, list(noise_msg.fm).index(16.0)]
half_power = rates[response >= response.max() / np.sqrt(2)]

# %% [markdown]
# Steady noise reads about {{ f"{band_median[0]:.0f}" }} dB at 2 Hz and
# {{ f"{band_median[-1]:.0f}" }} dB at 64 Hz: nearly the same depth at every rate. In one band the
# depth scatters by about {{ f"{np.median(band_scatter):.1f}" }} dB (standard deviation) from
# moment to moment; pooled over the bands it scatters by about
# {{ f"{np.median(pooled_scatter):.1f}" }} dB. That is why the pooled plots are smooth and the
# band x rate images grainy. Every filter of the 1 s window is
# {{ f"{half_power[-1] - half_power[0]:.1f}" }} Hz wide at half power, whatever its rate, as every
# bin of an STFT is equally wide. The filters of the 0.25 s window are four times wider and take
# in more of the envelope's random fluctuation, so on steady noise they read about
# {{ f"{sharper_by:.0f}" }} dB deeper; that is why the 0.25 s panels look brighter throughout.
# The modulation filters of a model of modulation detection (Dau et al., 1997) instead widen
# with their rate, with a Q of 2 above 10 Hz; sonore's bank can do that too (see
# [What this page leaves out](#h-what-this-page-leaves-out)).
#
# Beside each player the band x rate image follows the sound: press play, or click a time axis.
# Before that it shows a moment chosen for each example. In the first and last half second, where
# the window would run past the sound, it holds the nearest moment that was measured.

# %% [markdown]
# ## A modulation rate that glides
#
# Noise whose amplitude is modulated at a rate that rises smoothly from 2 to 32 Hz over 8 s,
# at 50% depth throughout. The modulation is applied to the whole noise, so every band carries
# it.

# %% [about]
# You hear a slow throb speed up into a flutter and then a rough buzz. The cochleagram is a
# blur of noise, but both modulation spectrograms show the rate climbing. The 1 s window draws
# it as a thin line from 2 Hz up, its rows 1 Hz apart. The 0.25 s window starts at 8 Hz, its
# slowest rate, and its rows are 4 Hz apart, so the line is wider and climbs in steps. Next to
# the player, every band lights up at the same rate at once: the modulation is
# common to all frequencies.

# %% [demo g1] A gliding modulation rate
dur, r0, r1 = 8.0, 2.0, 32.0
t = np.arange(int(dur * fs)) / fs
phase = 2 * np.pi * r0 * dur / np.log(r1 / r0) * ((r1 / r0) ** (t / dur) - 1)
glide = (1 + 0.5 * np.sin(phase)) * rng.standard_normal(len(t))
sound = finish(so.Sound(glide, fs))
fig, playhead, live = show(sound)

# %% [markdown]
# ## Modulation that glides in frequency
#
# The gliding rate lights a vertical strip in the band x rate image: one rate, every band. Its
# counterpart on the other axis is a horizontal strip, every rate in one band, and here it glides
# up in frequency. The sound is a train of 5 ms noise bursts at random times, 12 a second on
# average. Random timing modulates the envelope at every rate at once. The bursts are confined to
# a band a third of an octave wide whose center rises from 200 Hz to 5 kHz over 8 s, and a steady
# noise 30 dB below them fills the other bands. `in_band` does the confining with a mask on an
# STFT (64 ms windows), whose Gaussian gain on a log-frequency axis follows the band's center.

# %%
glide_rng = np.random.default_rng(1)  # the glides' own draws, so the examples below stay as they were


def log_glide(start, end, duration, t):
    """A value gliding from ``start`` to ``end`` at a constant rate in octaves, at times ``t`` [s]."""
    return start * (end / start) ** (t / duration)


def in_band(snd, center, width=1 / 3, background_db=-30):
    """``snd`` kept only in a band ``width`` octaves wide (at half height) around ``center(t)``
    [Hz], plus steady noise ``background_db`` below it."""
    stft = so.STFT(snd, win_dur=0.064, hop_dur=0.008)
    octaves = np.log2(np.maximum(stft.f, 1.0)[:, None] / center(stft.t)[None, :])
    gains = np.exp(-0.5 * (octaves / (width / 2.355)) ** 2)  # 2.355 SDs make the width at half height
    band = (stft * so.Mask(gains[None], stft)).to_sound()
    background = so.Sound(glide_rng.standard_normal(len(snd)), snd.fs)
    return band + background.normalize(rms=band.rms * 10 ** (background_db / 20))


# %% [about]
# You hear a crackle rise in pitch. In the cochleagram each burst is a short mark, and the marks
# climb. Next to the player a horizontal strip climbs with them: one or two bands deeply modulated
# at every rate, the rest at the depth of steady noise. The pooled views are bright at every rate
# throughout, since some band always carries the bursts.

# %% [demo g2] Bursts in a band that glides up
dur = 8.0
t = np.arange(int(dur * fs)) / fs
burst = np.hanning(int(0.005 * fs))
train = np.zeros(len(t))
for start in np.cumsum(glide_rng.exponential(1 / 12, size=200)):
    i = int(start * fs)
    if i + len(burst) <= len(t):
        train[i : i + len(burst)] += burst
bursts = so.Sound(train * glide_rng.standard_normal(len(t)), fs)
sound = finish(in_band(bursts, lambda times: log_glide(200, 5000, dur, times)))
fig, playhead, live = show(sound)

# %% [markdown]
# Now both at once: noise fully modulated at one rate, confined to a band, with the rate and the
# band's center each gliding up or down. The four combinations follow one another, 4 s each: rate
# up and frequency up, rate up and frequency down, rate down and frequency up, and both down. The
# rate glides between 2 and 32 Hz, the center between 500 Hz and 4 kHz.

# %% [about]
# Next to the player a single spot moves diagonally: up and to the right, down and to the right,
# up and to the left, then down and to the left. In the pooled views the rate draws a rising and
# a falling line in each half. A band also lights up and goes dark as the noise passes through
# it, which is itself a slow modulation, so the bands the spot has just crossed glow at the
# slowest rates. The dashed lines mark the joins.

# %% [demo g3] Rate and frequency gliding together
dur = 4.0
t = np.arange(int(dur * fs)) / fs
parts = []
for r0, r1 in ((2, 32), (32, 2)):  # rate up, then down
    for f0, f1 in ((500, 4000), (4000, 500)):  # center up, then down
        phase = 2 * np.pi * r0 * dur / np.log(r1 / r0) * ((r1 / r0) ** (t / dur) - 1)
        modulated = so.Sound((1 + np.sin(phase)) * glide_rng.standard_normal(len(t)), fs)
        part = in_band(modulated, lambda times, f0=f0, f1=f1: log_glide(f0, f1, dur, times))
        parts.append(part.normalize(rms=0.1))
sound = finish(so.concat(parts))
fig, playhead, live = show(sound, marks=[4.0, 8.0, 12.0], start=2.0)

# %% [markdown]
# ## Speech, babble and noise
#
# For each talker ([Two talkers](talkers.html)): the sentence from the CMU ARCTIC corpus
# (Kominek & Black, 2004), then 4 s of babble, then 4 s of noise with the sentence's long-term
# spectrum. The collection has one sentence by the female talker, so each talker's babble is that
# talker's own sentence six times over, each copy repeated end to end and started at a random
# point. Speech is modulated most strongly at a few hertz, the rate of its syllables (Greenberg &
# Kingsbury, 1997). That rhythm is what babble
# partly hides and speech-shaped noise ([Tones and noises](classic.html#d-k1)) lacks entirely.
# [Hearing a modulation spectrum](modtargets.html#d-mt1) measures the male talker's sentence as a
# whole, and hears what is left when its modulation keeps its rates but loses its timing
# ([mt2](modtargets.html#d-mt2)).

# %%
n = int(4 * fs)
mixes, segments = {}, {}
for label, snd in talkers.items():
    x = snd.data[:, 0]
    repeated = np.tile(x, n // len(x) + 2)
    babble = np.zeros(n)
    for start in rng.integers(len(x), size=6):
        babble += repeated[start : start + n]
    ssn = so.long_term_spectrum([snd]).to_sound(4, fs, rng=0)
    parts = [snd, so.Sound(babble, fs).normalize(rms=0.1), ssn.normalize(rms=0.1)]
    mixes[label] = finish(so.concat(parts))
    ends = np.cumsum([p.duration for p in parts])
    segments[label] = {"Speech": (0, ends[0]), "Babble": (ends[0], ends[1]), "Noise": (ends[1], ends[2])}

# The pooled depth (1 s window), median over each part's time windows
shown = [2, 4, 8, 16, 32, 64]
pooled_medians = {}
print("depth [dB] at", ", ".join(f"{rate} Hz" for rate in shown))
for label, mix in mixes.items():
    _, msg, _ = analyze(mix)
    pooled = 20 * np.log10(msg.pooled_depth()[0])
    columns = [list(msg.fm).index(rate) for rate in shown]
    for name, (begin, end) in segments[label].items():
        during = (msg.t >= begin) & (msg.t < end)
        pooled_medians[label, name] = np.nanmedian(pooled[columns][:, during], axis=1)
        print(f"{label:13} {name:6}", "  ".join(f"{v:6.1f}" for v in pooled_medians[label, name]))
speech_gap = np.abs(pooled_medians["Male talker", "Speech"] - pooled_medians["Female talker", "Speech"])
lowest_f0 = {label: track.f0[0][track.voiced[0]].min() for label, track in tracks.items()}
print("lowest F0:", ", ".join(f"{label} {f0:.0f} Hz" for label, f0 in lowest_f0.items()))

# %% [about]
# The dashed lines mark where the babble and the noise begin; under the modulation
# spectrograms is the band x rate image at one moment of each part. Under the sentence the pooled
# depth is high from 2 to 8 Hz. In the babble it is several dB lower there and more even across
# rates, since the six copies' syllables fill each other's gaps. In the speech-shaped noise it
# falls further and is nearly even across rates, as steady noise's is (measured above); above
# 16 Hz the noise is more deeply modulated than the speech. The bright
# patch just before the first dashed line is the end of the sentence running into the babble,
# which windows that span both read as modulation.
#
# The two talkers look much alike. During the sentence their pooled depths differ by
# {{ f"{speech_gap.max():.0f}" }} dB or less at every rate printed. The rhythm is set by the
# syllables, and the two talkers read the same syllables at nearly the same pace. Pitch does not
# show at all: the fastest rate here, 64 Hz, is below either talker's lowest F0 (printed above),
# so the beating of harmonics within a band, at the rate of F0, is beyond the top of the plots.
# The difference between the voices is in a spectrogram
# ([Harmonics sample the envelope](sourcefilter.html#h-harmonics-sample-the-envelope)), not here.

# %% [demo s1] Speech, babble and noise
sounds = mixes
marks = {label: [parts["Babble"][0], parts["Noise"][0]] for label, parts in segments.items()}
moments = {label: {name: (a + b) / 2 for name, (a, b) in parts.items()} for label, parts in segments.items()}
fig, playhead = show_pair(sounds, marks, moments)

# %% [markdown]
# ## Three textures
#
# Crickets, applause and rain, 3.5 s of each, from the recordings on the
# [Sound textures](textures.html) page ([crickets](textures.html#d-t02a),
# [applause](textures.html#d-t03a), [rain](textures.html#d-t00a)). A texture is a sound whose
# statistics stay put, so its modulation spectrogram should hold still while each texture lasts
# and change at the joins.

# %%
textures = [so.load(fetch(f"docs/textures/{n}.flac")).resample(fs) for n in ("crickets", "applause", "rain")]
three = so.concat([s[:3.5].normalize(rms=0.1) for s in textures])
_, msg, _ = analyze(finish(three))
pooled = 20 * np.log10(msg.pooled_depth()[0])
deepest = {}
for name, begin in (("crickets", 0.0), ("applause", 3.5), ("rain", 7.0)):
    during = (msg.t >= begin + 0.5) & (msg.t < begin + 3.0)  # away from the joins
    median = np.nanmedian(pooled[:, during], axis=1)
    deepest[name] = msg.fm[np.nanargmax(median)]
    print(
        f"{name:9} deepest at {msg.fm[np.nanargmax(median)]:4.3g} Hz ({np.nanmax(median):5.1f} dB),",
        f"median over 2 to 64 Hz {np.nanmedian(median[msg.fm >= 2]):5.1f} dB",
    )

# %% [about]
# The printout gives each texture's pooled depth, median over its middle 2.5 s. The crickets
# chirp in fast pulses, and their modulation sits at the top of the rate axis (deepest at
# {{ f"{deepest['crickets']:.0f}" }} Hz) and hardly anywhere else. Applause is a crowd of claps
# and adds a band of slow modulation, deepest at {{ f"{deepest['applause']:.0f}" }} Hz. Rain has
# many small drops at no particular rate: no rate stands out, and its depth is nearly the same at
# every rate, as steady noise's is. The 1 s window separates the rates more cleanly, since
# nothing here changes quickly; the 0.25 s window marks the joins more sharply. The same crickets are on
# [Hearing a modulation spectrum](modtargets.html#d-mt11), with a twin that keeps each band's
# modulation ([crickets, twin band by band](modtargets.html#d-mt13)).

# %% [demo x1] Crickets, applause and rain
sound = finish(three)
fig, playhead, live = show(sound, marks=[3.5, 7.0], start=1.75)

# %% [markdown]
# ## Three cuts through the cube
#
# `msg.slices(t, rate)` draws three cuts through the cube at once, on one color scale: rate
# against time pooled over bands, frequency against time at one rate, and frequency against
# rate at one moment. Here they are for each talker's speech, babble and noise from above, at
# 4 Hz and at a moment in the sentence.

# %% [about]
# The male talker's sentence, babble and noise from above, cut at 4 Hz and at 1.3 s. The middle
# panel shows which bands carry the 4 Hz rhythm and when: during the sentence, nearly all of
# them, in step with the syllables, since a syllable's onset raises the level across the
# spectrum; in the babble, still many, but no longer in step with each other; in the noise,
# almost none. On the right, the
# moment at 1.3 s: depth is high from 2 Hz to about 10 Hz in most bands and falls off above
# that.

# %% [figure s2] Three cuts, male talker
_, msg, _ = analyze(mixes["Male talker"])
fig = msg.slices(1.3, rate=4.0)
rate_ticks(fig.axes[0].yaxis)
rate_ticks(fig.axes[2].xaxis)

# %% [about]
# The same cuts for the female talker. The sentence looks much as it does for the male talker.
# In the babble, a group of bands around 1 kHz keeps a strong 4 Hz rhythm for most of its 4 s:
# six voices hide the rhythm unevenly, more in some bands and moments than in others.

# %% [figure s3] Three cuts, female talker
_, msg, _ = analyze(mixes["Female talker"])
fig = msg.slices(1.3, rate=4.0)
rate_ticks(fig.axes[0].yaxis)
rate_ticks(fig.axes[2].xaxis)

# %% [markdown]
# ## What this page leaves out
#
# - **Windows that scale with the rate.** `so.ModulationSpectrogram(env)` without `window=` gives
#   each rate a window of three of its own cycles (`cycles=3`), rates half an octave apart from
#   0.5 Hz: 6 s at 0.5 Hz, 47 ms at 64 Hz. Slow rates are then resolved finely and fast ones placed
#   sharply in time, like a wavelet scalogram, but the gray edges are as wide as half of each
#   rate's window, so on a short sound the trusted cells form a U.
# - **Power.** Each cell also has a modulation power, $|2y|^2$, which says how much of the sound
#   a modulation is rather than how modulated a band is (`msg.power`, and `msg.average()` over
#   time).
# - **Live analysis.** With `align="causal"` every window ends at the time it reports, as a live
#   analysis would see the sound, so each number comes half a window later than in the centered
#   analysis used here.
# - **Inversion.** The phase of each filter output is dropped and the local mean divided out, so
#   the modulation spectrogram can't be turned back into sound. To change a sound's modulation,
#   filter its envelopes and resynthesize, as on the
#   [Analysis and resynthesis](resynthesis.html) page.
# - **The front end.** Any filterbank's envelopes will do, a gammatone bank's for instance, and
#   the envelopes can be compressed first; this page keeps one front end throughout.

# %% [markdown]
# ## References
#
# - Atlas & Shamma (2003). Joint acoustic and modulation frequency. *EURASIP J. Appl. Signal
#   Processing* 2003(7).
#   [doi:10.1155/S1110865703305013](https://doi.org/10.1155/S1110865703305013).
#   [`modspectrogram.ModulationSpectrogram.at`](https://github.com/choyun1/sonore/blob/main/src/sonore/views/modspectrogram.py#L171)
# - Dau, Kollmeier & Kohlrausch (1997). Modeling auditory processing of amplitude modulation. I.
#   Detection and masking with narrow-band carriers. *JASA* 102(5), 2892–2905.
#   [PubMed](https://pubmed.ncbi.nlm.nih.gov/9373976/). Modulation filters a few cycles long.
#   [`modulation.HannModulationFilterbank`](https://github.com/choyun1/sonore/blob/main/src/sonore/views/modulation.py#L681)
# - Greenberg & Kingsbury (1997). The modulation spectrogram: in pursuit of an invariant
#   representation of speech. *Proc. ICASSP 1997*, vol. 3, 1647–1650.
#   [Semantic Scholar](https://www.semanticscholar.org/paper/71c0095d37084b6055a1abc8d4edcde3ef9f130b).
#   [`modspectrogram.ModulationSpectrogram`](https://github.com/choyun1/sonore/blob/main/src/sonore/views/modspectrogram.py#L35)
# - Kingsbury, Morgan & Greenberg (1998). Robust speech recognition using the modulation
#   spectrogram. *Speech Communication* 25(1–3), 117–132.
#   [doi:10.1016/S0167-6393(98)00032-6](https://doi.org/10.1016/S0167-6393(98)00032-6).
# - Kominek & Black (2004). The CMU Arctic speech databases. *Proc. 5th ISCA Speech Synthesis
#   Workshop (SSW5)*, 223–224. [ISCA Archive](https://www.isca-archive.org/ssw_2004/kominek04b_ssw.html).
#   The sentence, read by speakers bdl and slt.
