"""Sound quilts: recordings cut into segments and strung together in a new order.

This script is the gallery page https://choyun1.github.io/sonore/gallery/quilts.html:
docs/gallery/build.py runs it cell by cell from the repository root and shows each
cell's code beside what it made. Run it yourself from the repository root,

    python docs/gallery/stimuli/quilts.py

or a cell at a time ("# %%" starts a cell in VS Code, Spyder and Jupytext).
"""

# %% [markdown]
# # Sound quilts
#
# A *sound quilt* (Overath et al., 2015) cuts a recording into segments of one length and strings
# them together in a new order, keeping the recording's structure inside each segment and
# scrambling everything longer than a segment.
#
# - [How a quilt is made](#h-how-a-quilt-is-made): the rule that picks the order, and the
#   joining.
# - [Quilting speech](#h-quilting-speech): two readers quilted at segment lengths from 30 to
#   960 ms, and how much the joins change the cochleagram.
# - [Quilting a texture](#h-quilting-a-texture): fire and a stream quilted the same way, and
#   what of the texture a quilt keeps.

# %% [markdown]
# ## Code the examples share
#
# Every example below is the code shown with it, run after these cells: the level the gallery plays
# sounds at, and how a recording and its quilts are played together.

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
def finish_together(sounds):
    """How a recording and its quilts are played: 5 ms ramps, and one gain for all of them, the
    largest that keeps every RMS at most 0.1 and every peak at most 0.95. A quilt keeps the level
    of its recording, and one gain keeps that visible; scaling each sound on its own would let a
    single loud event in one of them turn that one down."""
    ramped = {label: snd.ramp(5e-3) for label, snd in sounds.items()}
    gain = min(min(0.1 / snd.rms, 0.95 / snd.peak) for snd in ramped.values())
    return {label: so.Sound(snd.data * gain, snd.fs) for label, snd in ramped.items()}


# %% [markdown]
# ## How a quilt is made
#
# Each next segment is the one whose join with the current segment changes the cochleagram by about
# as much as the recording itself changed at that point, and a join the recording already had is
# never used. Overath et al. quilted German speech for listeners who knew no German, at segment
# lengths from 30 to 960 ms, and found a region of the superior temporal sulcus whose response
# grew with segment length up to about 500 ms. `so.quilt` follows the paper's methods;
# `so.splice` is its joining step, which moves each segment by up to 15 ms so that its waveform
# lines up with what came before, then cross-fades over 30 ms. A quilt keeps the level of its
# recording, so each recording and its quilts below are played at one gain.

# %% [markdown]
# ## Quilting speech
#
# The passages here are about 30 s of English read speech from LibriSpeech (Panayotov et al.,
# 2015). For a listener who knows English, words in the long-segment quilts carry meaning, which
# the paper avoided by using a language its listeners did not know. As in the paper, pauses are
# removed first. Without that, short-segment quilts string silences together: one silent segment
# changes little into another, and little is what the passage itself changes across a silence (a
# first build of this page gave a 6 s quilt of 30 ms segments with 4 s of silence). The cut uses
# the thresholds of McDermott's toolbox, and `so.splice` does the cutting: the kept 30 ms windows,
# joined with 10 ms cross-fades and no shifts, give back the passage exactly except at the cuts.
# The paper also band-limited its sources and faded out each quilt; that is not done here.

# %%
# A male and a female LibriSpeech reader (docs/speech/SOURCES.md), at 16 kHz.
READERS = {"Male reader": "3000_15664", "Female reader": "1462_170142"}


def without_pauses(snd, window=0.03, below_db=35, min_windows=8):
    """``snd`` with every run of at least ``min_windows`` windows more than ``below_db`` below the
    loudest window cut out, the rest joined by ``so.splice`` with 10 ms cross-fades."""
    n = int(round(window * snd.fs))
    frames = snd.data[: len(snd) // n * n, 0].reshape(-1, n)
    level = 10 * np.log10(np.mean(frames**2, axis=1) + 1e-20)  # dB per window
    quiet = np.concatenate([[False], level < level.max() - below_db, [False]])
    edges = np.flatnonzero(np.diff(quiet.astype(int)))  # starts and ends of quiet runs
    keep = np.ones(len(frames), dtype=bool)
    for start, end in zip(edges[::2], edges[1::2], strict=True):
        if end - start >= min_windows:
            keep[start:end] = False
    return so.splice(snd, np.flatnonzero(keep) * n / snd.fs, n / snd.fs, crossfade=0.01, max_shift=0)


passages = {
    label: without_pauses(so.load(fetch(f"docs/speech/librispeech_{reader}.flac")))
    for label, reader in READERS.items()
}
for label, passage in passages.items():
    print(f"{label}: {passage.duration:.1f} s without pauses")
SEGMENTS = [0.03, 0.12, 0.48, 0.96]  # s
QUILT_DUR = 6.0  # s
quilt_bank = so.cosine_filterbank(30, 20, 8000)  # the paper's bank, up to Nyquist at 16 kHz


def quilt_demo(passage):
    """The passage's first 6 s and a 6 s quilt at each segment length, with their cochleagrams."""
    sounds = {"Original": so.Sound(passage.data[: int(QUILT_DUR * passage.fs)], passage.fs)}
    for seg in SEGMENTS:
        sounds[f"{seg * 1000:.0f} ms"] = so.quilt(passage, seg, QUILT_DUR, rng=1)
    sounds = finish_together(sounds)
    fig, axes = plt.subplots(len(sounds), 1, figsize=(10, 5.6), sharex=True, layout="constrained")
    for ax, (label, snd) in zip(axes, sounds.items(), strict=True):
        quilt_bank.analyze(snd).envelopes().plot(ax, colorbar=False)
        title = label if label == "Original" else f"Quilt, {label} segments"
        ax.set(title=title, xlabel="", xlim=(0, QUILT_DUR))
    axes[-1].set_xlabel("Time [s]")
    return fig, sounds, {label: [ax] for label, ax in zip(sounds, axes, strict=True)}


# %% [about]
# The first 6 s of the passage without its pauses, then quilts of the whole passage made of 30,
# 120, 480 and 960 ms segments. A 30 ms segment is shorter than most speech sounds, a 120 ms one
# about a syllable, and a 960 ms one a few words. At 30 ms the rule tends to chain segments with
# similar spectra, and the cochleagram shows long, steady stretches that the passage does not
# have.

# %% [demo sq1] Quilts of a male reader
fig, sounds, playhead = quilt_demo(passages["Male reader"])

# %% [about]
# The same for the female reader.

# %% [demo sq2] Quilts of a female reader
fig, sounds, playhead = quilt_demo(passages["Female reader"])

# %% [markdown]
# The rule that picks the order is what keeps the joins from standing out. The change across a
# join is measured as the paper does: the mean squared difference between the last 30 ms before
# it and the first 30 ms after it, in the cochleagram compressed by the power 0.3. Every join of a
# quilt falls at a whole number of segments, since the alignment moves where a segment is read
# from, not where it is written, so the changes can be measured on the quilt itself.

# %% [about]
# The changes across 120 ms joins: between neighboring segments of the male reader's passage, at
# the joins of 30 s of quilts made from it, and at the joins of 30 s of the same segments put in a
# random order with `so.splice`. The quilts' joins change about as much as the passage's own
# boundaries (medians in the legend); the random joins change between two and three times as
# much.

# %% [figure sq3] Change across the joins


def join_changes(snd, segment, border=0.03):
    """The cochleagram change across every boundary at a multiple of ``segment`` [s]."""
    n, nb = int(round(segment * snd.fs)), int(round(border * snd.fs))
    env = quilt_bank.analyze(snd).envelopes().data[:, :, 0] ** 0.3
    joins = np.arange(n, len(env) - nb + 1, n)
    return np.array([np.mean((env[j - nb : j] - env[j : j + nb]) ** 2) for j in joins])


passage, seg = passages["Male reader"], 0.12
n_segments = len(passage) // int(seg * passage.fs)
rng = np.random.default_rng(0)
quilts = [so.quilt(passage, seg, QUILT_DUR, rng=k) for k in range(5)]
shuffled = [so.splice(passage, rng.permutation(n_segments)[:50] * seg, seg) for _ in range(5)]
changes = {
    "Passage": join_changes(passage, seg),
    "Quilts": np.concatenate([join_changes(q, seg) for q in quilts]),
    "Random order": np.concatenate([join_changes(s, seg) for s in shuffled]),
}
fig, ax = plt.subplots(figsize=(7, 3.4), layout="constrained")
bins = np.linspace(0, np.percentile(np.concatenate(list(changes.values())), 99), 40)
for label, values in changes.items():
    legend = f"{label} (median {np.median(values):.4f})"
    ax.hist(values, bins, density=True, histtype="step", lw=1.8, label=legend)
ax.set(xlabel="Change across a 120 ms join (mean squared, compressed cochleagram)", ylabel="Density")
ax.legend()

# %% [markdown]
# ## Quilting a texture
#
# A texture, such as fire or a stream, has little structure longer than a few hundred
# milliseconds, so a quilt of one should sound like more of the same texture even with short
# segments, as McDermott's notes to the quilting toolbox observe. The texture model of
# [Sound textures](textures.html) (McDermott & Simoncelli, 2011) summarizes a texture by a few
# time-averaged statistics of its cochlear envelopes, and the last figure uses three of them to
# check this.

# %%
texture_bank = so.cosine_filterbank(30, 20, 10000)


def texture_quilt_demo(recording, duration, segments=(0.03, 0.12, 0.96)):
    """The recording's first ``duration`` seconds and a quilt of the whole recording at each
    segment length, played at one gain, with their cochleagrams."""
    sounds = {"Original": so.Sound(recording.data[: int(duration * recording.fs)], recording.fs)}
    for seg in segments:
        sounds[f"{seg * 1000:.0f} ms"] = so.quilt(recording, seg, duration, rng=1)
    sounds = finish_together(sounds)
    fig, axes = plt.subplots(len(sounds), 1, figsize=(10, 4.6), sharex=True, layout="constrained")
    for ax, (label, snd) in zip(axes, sounds.items(), strict=True):
        texture_bank.analyze(snd).envelopes().plot(ax, colorbar=False)
        title = label if label == "Original" else f"Quilt, {label} segments"
        ax.set(title=title, xlabel="", xlim=(0, duration))
    axes[-1].set_xlabel("Time [s]")
    return fig, sounds, {label: [ax] for label, ax in zip(sounds, axes, strict=True)}


# %% [about]
# The first 5 s of a 20 s recording of fire, then 5 s quilts of the whole recording made of 30,
# 120 and 960 ms segments. The recording has one crackle, at 10 s, that peaks about five times
# higher than anything in its first 5 s. Each quilt here draws that crackle or another large pop
# from the rest of the recording, and the loudest of them sets the one gain all four are played
# at, so the quilts' quieter stretches play at the same level as the original's.

# %% [demo sq4] Quilts of a fire
fig, sounds, playhead = texture_quilt_demo(so.load(fetch("docs/textures/fire.flac")), 5.0)

# %% [about]
# The same for 7 s of a stream (the recording on [Sound textures](textures.html)), with 4 s
# quilts. The recording is short: a quilt of 960 ms segments uses 5 of its 7 segments, so the
# rule has few to choose from, and a quilt of 30 ms segments uses 134 of 233. With 30 ms
# cross-fades on 30 ms segments every sample lies in a cross-fade, and the paper's raised-cosine
# ramps keep only 3/4 of the power of two unrelated stretches of noise, so its 30 ms quilts of a
# stream play about 1 dB quieter. `so.quilt` scales each cross-fade by the correlation it
# measures between the two segments (`fade="adaptive"`, the default), which keeps the level.

# %% [demo sq6] Quilts of a stream
fig, sounds, playhead = texture_quilt_demo(so.load(fetch("docs/textures/stream.flac")), 4.0)

# %% [about]
# Three of the statistics the texture model of [Sound textures](textures.html) measures, for the
# whole recording (black) and for each quilt heard above (colors), with the level set aside.
# Left, the spectrum: the mean envelope of each cochlear band. Middle, how strongly the envelopes
# fluctuate at each rate from 0.5 to 200 Hz, averaged over the bands. Right, how peaky each band's
# envelope is (its kurtosis): high where the sound comes in brief pops, like fire's crackle above
# about 150 Hz, and about 3 for steady noise. The quilts follow the recording closely in
# all three, which is what "more of the same texture" means here. They part most at the slowest
# fluctuations, below a few hertz, where fire's quilts have less power than the recording, and in
# fire's peakiness, which depends on which pops a quilt happens to draw.

# %% [figure sq5] What a texture quilt keeps
from sonore.texture import TextureModel, TextureStats

model = TextureModel()
band_cfs, rate_cfs = model.filterbank.cfs, model.mod_bank.cfs
fig, axes = plt.subplots(2, 3, figsize=(10, 6), layout="constrained")
for row, (name, duration) in enumerate((("fire", 5.0), ("stream", 4.0))):
    recording = so.load(fetch(f"docs/textures/{name}.flac"))
    stats = {"Recording": TextureStats.measure(recording)}
    for seg in (0.03, 0.12, 0.96):  # the quilts heard above: same seed
        stats[f"{seg * 1000:.0f} ms quilt"] = TextureStats.measure(so.quilt(recording, seg, duration, rng=1))
    for label, st in stats.items():
        style = {"color": "k", "lw": 2.2} if label == "Recording" else {"lw": 1.2}
        audible = st.channel_mask()  # bands within 30 dB of the loudest
        axes[row, 0].plot(band_cfs, 20 * np.log10(st.env_mean), label=label, **style)
        axes[row, 1].plot(rate_cfs, np.mean(st.mod_power[audible], axis=0), **style)
        axes[row, 2].plot(band_cfs, st.env_kurt, **style)
    axes[row, 0].set(xscale="log", ylabel=f"{name.capitalize()}\nMean envelope [dB]")
    axes[row, 1].set(xscale="log", yscale="log", ylabel="Modulation power (relative)")
    axes[row, 2].set(xscale="log", yscale="log", ylabel="Envelope kurtosis")
axes[0, 0].set_title("Spectrum")
axes[0, 1].set_title("How fast the envelopes fluctuate")
axes[0, 2].set_title("How peaky the envelopes are")
for ax in axes[1]:
    ax.set_xlabel("Modulation rate [Hz]" if ax is axes[1, 1] else "Band center frequency [Hz]")
axes[0, 0].legend()

# %% [markdown]
# ## References
#
# - McDermott & Simoncelli (2011). Sound texture perception via statistics of the auditory
#   periphery: evidence from sound synthesis. *Neuron* 71(5), 926–940.
#   [doi:10.1016/j.neuron.2011.06.032](https://doi.org/10.1016/j.neuron.2011.06.032).
#   [`texture`](https://github.com/choyun1/sonore/blob/main/src/sonore/texture/stats.py)
# - Overath, McDermott, Zarate & Poeppel (2015). The cortical analysis of speech-specific temporal
#   structure revealed by responses to sound quilts. *Nature Neuroscience* 18, 903–911.
#   [doi:10.1038/nn.4021](https://doi.org/10.1038/nn.4021).
#   [`quilting.quilt`](https://github.com/choyun1/sonore/blob/main/src/sonore/views/quilting.py#L26)
#   [`processing.splice`](https://github.com/choyun1/sonore/blob/main/src/sonore/core/processing.py#L118)
# - Panayotov, Chen, Povey & Khudanpur (2015). LibriSpeech: an ASR corpus based on public domain
#   audio books. *Proc. ICASSP 2015*, 5206–5210.
#   [doi:10.1109/ICASSP.2015.7178964](https://doi.org/10.1109/ICASSP.2015.7178964). The two
#   quilted passages (CC BY 4.0; sources in docs/speech/SOURCES.md).
