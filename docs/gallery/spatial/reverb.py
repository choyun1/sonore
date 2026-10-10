"""Rooms: synthetic reverberation from the statistics of real rooms, and how distance changes
the balance of direct and reverberant sound.

This script is the gallery page https://choyun1.github.io/sonore/gallery/reverb.html:
docs/gallery/build.py runs it cell by cell from the repository root and shows each
cell's code beside what it made. Run it yourself from the repository root,

    python docs/gallery/spatial/reverb.py

or a cell at a time ("# %%" starts a cell in VS Code, Spyder and Jupytext).
"""

# %% [markdown]
# # Rooms
#
# A room's reverberation is its impulse response (IR): what a microphone records after a single
# click. Convolve any dry sound with it and the sound is in that room. Traer & McDermott (2016)
# measured the IRs of 271 spaces that people found themselves in during daily life and found
# them tightly constrained, and `so.synth_ir` builds new ones from their statistics. Distance is
# taught here too: moving a source away weakens its direct sound, but not the room's
# reverberation.
#
# - [How a room is synthesized](#h-how-a-room-is-synthesized): decaying noise in cochlear bands,
#   and the parameters that shape it.
# - [Natural rooms](#h-natural-rooms): a starter pistol, then the sentence read by the male and
#   the female talker, in a synthetic room of the kind Traer & McDermott's listeners could not
#   tell from real ones.
# - [Farther away](#h-farther-away): the inverse square law and the direct-to-reverberant ratio,
#   with both talkers four times as far from the listener.
# - [Rooms that break the rules](#h-rooms-that-break-the-rules): the paper's atypical rooms,
#   each heard with the starter pistol and then with the sentence.

# %% [markdown]
# ## How a room is synthesized
#
# In real rooms the reverberant tail is close to Gaussian noise whose level, in each frequency
# band, falls exponentially: a straight line in dB. How fast it falls is the band's RT60, the
# time it takes to drop 60 dB. In the rooms Traer & McDermott measured, decay was slowest
# between about 200 Hz and 2 kHz and faster at low and high frequencies. `so.synth_ir` splits
# Gaussian noise into ERB-spaced bands, multiplies each band by its decay, and sums the bands
# back. Its parameters:
#
# - `rt60`, the broadband reverberation time in seconds, the median over bands. The time each
#   band takes to decay, and its level at the start, follow the paper's regressions on the 271
#   rooms, scaled to this value.
# - `drr_db`, the direct-to-reverberant ratio: the energy of the direct sound, a unit impulse at
#   time zero, over that of the tail. Without it, only the tail is returned.
# - `n_channels`, independent tails per channel, for example 2 for a decorrelated binaural tail.
# - `decay_shape`, `rt60_profile` and `drr_profile`, which leave the regularities of real rooms
#   behind: they make the paper's atypical rooms, heard below.
#
# `so.band_rt60s` gives the RT60s of each band for a broadband RT60, and `so.measure_rt60`
# measures them from an IR by fitting a line to each band's energy decay curve.
#
# Every room below starts from the same natural room, with an RT60 of 1 s and a
# direct-to-reverberant ratio of −3 dB, and the same seed, so the rooms share their noise.
# Unlike the paper's, they are not further equated for how much they distort the sound, so some
# differences in loudness and length remain. They are presented diotically (the same in both
# ears), as in the paper.

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
from scipy.signal import butter, sosfilt

SPEAKERS = {"Male talker": "bdl", "Female talker": "slt"}
talkers = {
    label: finish(so.load(fetch(f"docs/speech/{speaker}_arctic_a0131.flac")))
    for label, speaker in SPEAKERS.items()
}

FS = 44100
# the recordings are at 16 kHz; the rooms are built at 44.1 kHz
sentences = {label: snd.resample(FS) for label, snd in talkers.items()}


def starter_pistol():
    """A sharp broadband crack: a shock-like pulse (0.15 ms exponential), highpassed.
    Its spectrum is smooth; a short burst of noise would have random deep notches."""
    t = np.arange(int(0.03 * FS)) / FS
    x = sosfilt(butter(2, 300, "highpass", fs=FS, output="sos"), np.exp(-t / 0.15e-3))
    return so.Sound(x, FS).pad(before=0.05, after=0.05)


def room(**kw):
    """A synthetic room: RT60 1 s, direct-to-reverberant ratio -3 dB, a fixed seed."""
    return so.synth_ir(1.0, FS, drr_db=-3, rng=5, **kw)


def drr_db(ir):
    """The direct-to-reverberant ratio of an IR [dB]: its first sample's energy over the rest's."""
    return 10 * np.log10(ir.data[0, 0] ** 2 / np.sum(ir.data[1:, 0] ** 2))


def level_db(ir, t, window=0.05):
    """The level of an IR's tail [dB] in a 50 ms window starting at time t, relative to its
    first 50 ms."""
    tail = ir.data[1:, 0] ** 2
    n, k = int(window * ir.fs), int(t * ir.fs)
    return 10 * np.log10(tail[k : k + n].mean() / tail[:n].mean())


def share_above(snd, f=4000):
    """The fraction of a sound's energy above f [Hz]."""
    power = np.abs(np.fft.rfft(snd.data[:, 0])) ** 2
    return power[np.fft.rfftfreq(len(snd), 1 / snd.fs) > f].sum() / power.sum()


def show_sound(snd, ax_wave, ax_coch, title=""):
    """Waveform and cochleagram (60 dB range) of a sound."""
    snd.plot(ax_wave, color="k", lw=0.4)
    ax_wave.set(title=title or "Waveform", xlabel="", ylabel="")
    # no extra lowpass: resampling to 1 kHz is already band-limited, and a
    # lowpass would ring visibly around the sharp onset of the shot
    cochleagram = so.cosine_filterbank(30, 50, 8000).analyze(snd).envelopes(fs=1000)
    cochleagram.plot(ax_coch, colorbar=False, db_range=60)
    ax_coch.set_title("Cochleagram (60 dB range)")
    for ax in (ax_wave, ax_coch):
        ax.set_xlim(0, snd.duration)


def show_room(ax_decay, ax_rt60, ir, kw=None):
    """The IR's decay in four bands, and its RT60s by band against those of natural rooms."""
    tail = so.Sound(ir.data[1:], ir.fs)
    env = so.cosine_filterbank(30, 50, 8000).analyze(tail).envelopes(lowpass=30, fs=1000)
    for f, color in zip(
        (125, 500, 2000, 6000), ("tab:blue", "tab:green", "tab:orange", "tab:red"), strict=True
    ):
        k = int(np.argmin(np.abs(env.cfs - f)))
        db = env.db[:, k, 0]
        ax_decay.plot(env.t, db - db.max(), color=color, lw=0.8, label=f"{env.cfs[k]:.0f} Hz")
    ax_decay.set(
        ylim=(-70, 3), xlabel="Time [s]", ylabel="Band envelope [dB]", title="The IR's decay in four bands"
    )
    # a time-reversed decay rises to the upper right, so its legend goes on the left
    reversed_decay = (kw or {}).get("decay_shape") == "time_reversed"
    ax_decay.legend(fontsize=8, loc="upper left" if reversed_decay else "upper right")
    ax_decay.grid(ls=":")
    # measured RT60 profile vs the ecological one
    cfs, rt = so.measure_rt60(tail)
    ax_rt60.semilogx(
        cfs,
        so.band_rt60s(1.0, cfs, "ecological"),
        color="k",
        lw=1.2,
        ls="--",
        label="natural rooms (RT60 = 1 s)",
    )
    profile = (kw or {}).get("rt60_profile")
    if profile:
        # requested profile, computed over synth_ir's own bands (which run to 16 kHz)
        synth_cfs = so.cosine_filterbank(32, 20, min(16000, FS / 2)).cfs
        requested = np.interp(cfs, synth_cfs, so.band_rt60s(1.0, synth_cfs, profile))
        ax_rt60.semilogx(cfs, requested, color="tab:purple", lw=1, ls=":", label=f"requested ({profile})")
    if not kw or "decay_shape" not in kw:
        ax_rt60.semilogx(cfs, rt, color="tab:purple", lw=1.2, marker=".", label="this IR (measured)")
    else:
        ax_rt60.text(
            0.03,
            0.06,
            "decay isn't exponential, so an RT60\nisn't meaningful for this IR",
            transform=ax_rt60.transAxes,
            fontsize=8,
        )
    ax_rt60.set(xlabel="Frequency [Hz]", ylabel="RT60 [s]", title="Decay time by frequency", ylim=(0, 1.8))
    ax_rt60.legend(fontsize=8, loc="upper right")
    ax_rt60.grid(ls=":", which="both")


def show(snd, ir, kw=None):
    """One sound: waveform and cochleagram, then the IR's decay by band and its RT60s.
    Returns the figure and the panels the playhead follows."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.2), layout="constrained")
    show_sound(snd, axes[0, 0], axes[0, 1])
    if ir is None:
        for ax in axes[1]:
            ax.axis("off")
        axes[1, 0].text(0.0, 0.5, "No room: the dry source.", transform=axes[1, 0].transAxes, fontsize=11)
    else:
        show_room(axes[1, 0], axes[1, 1], ir, kw)
    return fig, [axes[0, 0], axes[0, 1]]


def show_pair(sounds, ir, kw=None, panels=show_room):
    """One column per talker (waveform and cochleagram), and below them the room both are in,
    drawn by `panels` in two axes. Returns the figure and, for each talker, the panels the
    playhead follows."""
    fig = plt.figure(figsize=(10, 8.4), layout="constrained")
    top, bottom = fig.subfigures(2, 1, height_ratios=[1.25, 1])
    playhead = {}
    for column, (label, snd) in zip(top.subfigures(1, 2), sounds.items(), strict=True):
        ax_wave, ax_coch = column.subplots(2, 1, sharex=True, height_ratios=[0.45, 1])
        show_sound(snd, ax_wave, ax_coch, label)
        playhead[label] = [ax_wave, ax_coch]
    panels(*bottom.subplots(1, 2), ir, kw)
    return fig, playhead


pistol = starter_pistol()

# %% [markdown]
# The RT60s the natural room below asks for in a few of `so.synth_ir`'s bands, and the band
# that rings longest:

# %%
bands = so.cosine_filterbank(32, 20, 16000).cfs
natural_rt60s = so.band_rt60s(1.0, bands)
for f in (125, 250, 500, 1000, 2000, 4000, 8000):
    k = int(np.argmin(np.abs(bands - f)))
    print(f"{bands[k]:6.0f} Hz: RT60 {natural_rt60s[k]:.2f} s")
print(f"longest: {natural_rt60s.max():.2f} s at {bands[natural_rt60s.argmax()]:.0f} Hz")

# %% [markdown]
# ## Natural rooms

# %% [about]
# The source for the room examples: a sharp, broadband crack lasting a fraction of a
# millisecond. Its cochleagram shows the filterbank's own response to a click: the filters are
# zero-phase, so they ring symmetrically before and after it (visible at this 60 dB range),
# whereas a real cochlea rings only afterwards.

# %% [demo 16] Starter pistol, dry
sound = finish(pistol)
fig, playhead = show(sound, None)

# %% [about]
# A synthetic room with RT60 = 1 s whose decay follows the statistics of the 271 real rooms:
# exponential, with mid frequencies ringing longest. In Traer & McDermott's experiments,
# listeners could not tell IRs synthesized this way from real ones.

# %% [demo 17] In a natural room
ir = ir_natural = room()
sound = finish(pistol.convolve(ir))
fig, playhead = show(sound, ir)

# %% [about]
# The sentence from [Seeing speech](speech.html), read by each of the [Two talkers](talkers.html),
# in the same room. The reverberation fills the gaps between words and smears each syllable into
# the next. The impulse response belongs to the room, not to the talker, so the panels at the
# bottom are the same for both voices: the room does the same thing to each.

# %% [demo r1] A sentence in the same room
sounds = {label: finish(snd.convolve(ir)) for label, snd in sentences.items()}
fig, playhead = show_pair(sounds, ir)

# %% [markdown]
# ## Farther away
#
# The direct sound from a point source falls with distance by the inverse square law: its
# energy as one over the distance squared, its level by 20 log10 of the distance ratio, which is
# 6 dB for every doubling (`so.distance_gain_db`). The reverberant tail is built up from
# reflections off every wall, and on this page it is held at the same level wherever the talker
# stands, as in the simplest model of a room. So moving a talker away lowers the
# direct-to-reverberant ratio by exactly as much as it lowers the direct sound. The cell below
# prints the direct sound's change at two and four times the distance, then builds the room
# with the talker four times farther and measures the ratio in both IRs. For a talker walking up
# to the listener, without and with a room, see [Coming closer](moving.html#h-coming-closer)
# on the Moving talkers page.

# %%
for distance_ratio in [2, 4]:
    print(f"{distance_ratio} times farther: {so.distance_gain_db(distance_ratio):.1f} dB")
far_gain = so.distance_gain_db(4)
ir_far = so.synth_ir(1.0, FS, drr_db=-3 + far_gain, rng=5)
print(f"direct-to-reverberant ratio, near: {drr_db(ir_natural):.1f} dB")
print(f"direct-to-reverberant ratio, four times farther: {drr_db(ir_far):.1f} dB")


def show_distance(ax_level, ax_drr, ir, kw=None):
    """The near and far IRs' energy in 10 ms windows, with the tails at the same level; and the
    direct-to-reverberant ratio against distance."""
    window = int(0.01 * FS)
    tail = ir_natural.data[1:, 0] ** 2
    n = len(tail) // window
    windows = 10 * np.log10(tail[: n * window].reshape(n, window).sum(axis=1))
    ax_level.plot(
        (np.arange(n) + 0.5) * window / FS, windows, color="0.4", lw=0.8, label="tail, near and far"
    )
    for this_ir, color, label in ((ir_natural, "k", "near"), (ir, "tab:purple", "four times farther")):
        # scaled so that the tail has the near IR's energy: only the direct sound changes
        scale = np.sqrt(np.sum(tail) / np.sum(this_ir.data[1:, 0] ** 2))
        direct = 20 * np.log10(scale * this_ir.data[0, 0])
        ax_level.plot([0, 0], [-90, direct], color=color, lw=2)
        ax_level.plot(0, direct, "o", color=color, ms=6, label=f"direct sound, {label}")
    ax_level.set(
        xlim=(-0.03, 1.25),
        ylim=(-70, 5),
        xlabel="Time [s]",
        ylabel="Energy per 10 ms [dB]",
        title="Direct sound and tail",
    )
    ax_level.legend(fontsize=8, loc="upper right")
    ax_level.grid(ls=":")
    ratios = np.geomspace(0.5, 8, 100)
    ax_drr.semilogx(ratios, drr_db(ir_natural) + so.distance_gain_db(ratios), color="0.4", lw=1.2)
    ax_drr.plot(1, drr_db(ir_natural), "o", color="k", label="near (measured)")
    ax_drr.plot(4, drr_db(ir), "o", color="tab:purple", label="four times farther (measured)")
    ax_drr.set(
        xlabel="Distance, relative to the first demo",
        ylabel="Direct-to-reverberant ratio [dB]",
        title="Inverse square law, tail held constant",
    )
    ax_drr.set_xticks([0.5, 1, 2, 4, 8], ["0.5", "1", "2", "4", "8"])
    ax_drr.set_xticks([], minor=True)
    ax_drr.legend(fontsize=8, loc="upper right")
    ax_drr.grid(ls=":")


# %% [about]
# The same sentence, read by each talker, in the same room, with the direct sound
# {{ f"{-far_gain:.0f}" }} dB weaker: the direct-to-reverberant ratio drops from
# {{ f"{drr_db(ir_natural):.0f}" }} to {{ f"{drr_db(ir_far):.0f}" }} dB. Every sound on this
# page is played at the same RMS, so what changes is not the loudness but how much of the
# sentence is room. Compare [the sentence nearby](#d-r1), and the walk in a room on the Moving
# talkers page ([Walking in, in a room](moving.html#d-m6)), where the same balance shifts the
# other way as the talker comes close.

# %% [demo r2] The same sentence, four times farther
sounds = {label: finish(snd.convolve(ir_far)) for label, snd in sentences.items()}
fig, playhead = show_pair(sounds, ir_far, panels=show_distance)

# %% [markdown]
# ## Rooms that break the rules
#
# Synthetic rooms that break the regularities of real ones sound wrong (Traer & McDermott,
# 2016). Each room below changes one thing about the natural room above, and is heard twice:
# first with the starter pistol, which lays the room bare, then with the sentence, to compare
# with [the sentence in the natural room](#d-r1). Where the decay is no longer exponential, an
# RT60 no longer describes it, so the last panel shows only what natural rooms would do. The
# three profiles that change how decay time depends on frequency do not all keep the natural
# room's median RT60; the cell prints the median of each over `so.synth_ir`'s bands.

# %%
for profile in ("ecological", "inverted", "exaggerated", "reduced"):
    print(f"{profile:12s} median RT60 {np.median(so.band_rt60s(1.0, bands, profile)):.2f} s")

# %% [about]
# The same decay run backwards: the reverberation swells up to the shot instead of dying away
# after it. In the paper, listeners readily heard this decay, and the two linear ones after it,
# as synthetic, with impulses, speech and noise alike.

# %% [demo 18] Time-reversed decay
kw = {"decay_shape": "time_reversed"}
ir = room(**kw)
sound = finish(pistol.convolve(ir))
fig, playhead = show(sound, ir, kw)

# %% [about]
# The sentence in the time-reversed room. Each sound's reverberation now grows after it rather
# than fading, cresting {{ f"{len(ir) / FS:.1f}" }} s later, so every syllable is followed by a
# swell of itself and the swells pile up across the sentence.

# %% [demo r4] The sentence, time-reversed decay
sounds = {label: finish(snd.convolve(ir)) for label, snd in sentences.items()}
fig, playhead = show_pair(sounds, ir, kw)

# %% [about]
# Starts at the natural level but falls linearly (in amplitude) rather than exponentially, with
# the same energy per band; it has to end early to do so.

# %% [demo 19] Linear decay, matched start
kw = {"decay_shape": "linear_matched_start"}
ir = room(**kw)
sound = finish(pistol.convolve(ir))
fig, playhead = show(sound, ir, kw)

# %% [about]
# The sentence in the room whose decay falls linearly from the natural starting level. Its IR
# is {{ f"{len(ir) / FS:.2f}" }} s long, where the natural one is
# {{ f"{len(ir_natural) / FS:.2f}" }} s, so less of each word spills into the gaps after it.

# %% [demo r5] The sentence, linear decay, matched start
sounds = {label: finish(snd.convolve(ir)) for label, snd in sentences.items()}
fig, playhead = show_pair(sounds, ir, kw)

# %% [about]
# Linear decay that reaches zero where the natural decay is 60 dB down, again with the same
# energy per band. On a dB scale the decay bows outward instead of falling in a straight line.

# %% [demo 20] Linear decay, matched end
kw = {"decay_shape": "linear_matched_end"}
ir = room(**kw)
sound = finish(pistol.convolve(ir))
fig, playhead = show(sound, ir, kw)

# %% [about]
# The sentence in the room whose linear decay ends where the natural one is 60 dB down. Half a
# second in, its tail is {{ f"{-level_db(ir, 0.5):.0f}" }} dB below its first 50 ms, the natural
# one's {{ f"{-level_db(ir_natural, 0.5):.0f}" }} dB, so each word lingers at nearly full
# strength before it falls away.

# %% [demo r6] The sentence, linear decay, matched end
sounds = {label: finish(snd.convolve(ir)) for label, snd in sentences.items()}
fig, playhead = show_pair(sounds, ir, kw)

# %% [about]
# Exponential, but low and high frequencies ring longest and the middle dies fastest, the
# reverse of real rooms. Listeners in the paper readily heard it as synthetic, and several
# reported noticing a high-frequency hiss.

# %% [demo 21] Inverted frequency dependence
kw = {"rt60_profile": "inverted"}
ir = room(**kw)
sound = finish(pistol.convolve(ir))
fig, playhead = show(sound, ir, kw)

# %% [about]
# The sentence in the room where low and high frequencies ring longest. Above 4 kHz lies
# {{ f"{100 * share_above(sentences['Male talker']):.1f}" }}% of the male talker's energy and
# {{ f"{100 * share_above(sentences['Female talker']):.1f}" }}% of the female talker's, against
# {{ f"{100 * share_above(pistol):.0f}" }}% of the shot's, so the long high-frequency tail has
# much less to ring with here.

# %% [demo r7] The sentence, inverted frequency dependence
sounds = {label: finish(snd.convolve(ir)) for label, snd in sentences.items()}
fig, playhead = show_pair(sounds, ir, kw)

# %% [about]
# The profile of a room twice as reverberant, scaled down: more sharply peaked than real rooms
# of this size. The subtlest variant: in the paper, it and the reduced profile below were
# detected as synthetic with impulses but not with speech or noise.

# %% [demo 22] Exaggerated frequency dependence
kw = {"rt60_profile": "exaggerated"}
ir = room(**kw)
sound = finish(pistol.convolve(ir))
fig, playhead = show(sound, ir, kw)

# %% [about]
# The sentence in the room with the sharply peaked profile, a room the paper's listeners did not
# tell from a natural one when they heard it with speech.

# %% [demo r8] The sentence, exaggerated frequency dependence
sounds = {label: finish(snd.convolve(ir)) for label, snd in sentences.items()}
fig, playhead = show_pair(sounds, ir, kw)

# %% [about]
# The profile of a room half as reverberant, scaled up: flatter than real rooms of this size.
# Also subtle.

# %% [demo 23] Reduced frequency dependence
kw = {"rt60_profile": "reduced"}
ir = room(**kw)
sound = finish(pistol.convolve(ir))
fig, playhead = show(sound, ir, kw)

# %% [about]
# The sentence in the room with the flatter profile, which the paper's listeners, too, did not
# tell from a natural one with speech.

# %% [demo r9] The sentence, reduced frequency dependence
sounds = {label: finish(snd.convolve(ir)) for label, snd in sentences.items()}
fig, playhead = show_pair(sounds, ir, kw)

# %% [markdown]
# ## References
#
# - Kominek & Black (2004). The CMU Arctic speech databases. *Proc. 5th ISCA Speech Synthesis
#   Workshop*, 223–224. [ISCA Archive](https://www.isca-archive.org/ssw_2004/kominek04b_ssw.html).
#   The sentence, by speakers bdl and slt.
# - Traer & McDermott (2016). Statistics of natural reverberation enable perceptual separation of
#   sound and space. *Proc. Natl. Acad. Sci. USA* 113(48), E7856–E7865.
#   [doi:10.1073/pnas.1612524113](https://doi.org/10.1073/pnas.1612524113).
#   [`reverb.synth_ir`](https://github.com/choyun1/sonore/blob/main/src/sonore/spatial/reverb.py#L81)
