"""Synthetic reverberation: room impulse responses from the statistics of real rooms.

This script is the gallery page https://choyun1.github.io/sonore/gallery/reverb.html:
docs/gallery/build.py runs it cell by cell from the repository root and shows each
cell's code beside what it made. Run it yourself from the repository root,

    python docs/gallery/spatial/reverb.py

or a cell at a time ("# %%" starts a cell in VS Code, Spyder and Jupytext).
"""

# %% [markdown]
# # Synthetic reverberation
#
# A room's reverberation is its impulse response (IR): what a microphone records after a single
# click. Convolve any dry sound with it and the sound is in that room. Traer & McDermott (2016)
# measured the IRs of 271 real rooms and found them tightly constrained, and `so.synth_ir` builds
# new ones from their statistics.
#
# - [How a room is synthesized](#h-how-a-room-is-synthesized): decaying noise in cochlear bands,
#   and the parameters that shape it.
# - [Natural rooms](#h-natural-rooms): a starter pistol and a sentence, read by a male and by a
#   female talker, in a synthetic room built like those Traer & McDermott's listeners could not tell
#   from real ones.
# - [Farther away](#h-farther-away): the same sentence four times as far from the listener,
#   where less of what arrives is direct sound.
# - [Rooms that break the rules](#h-rooms-that-break-the-rules): the paper's atypical rooms,
#   which their listeners heard as wrong, each heard with the starter pistol and then with the
#   sentence.

# %% [markdown]
# ## How a room is synthesized
#
# In real rooms the reverberant tail is close to Gaussian noise whose level, in each frequency
# band, falls exponentially: a straight line in dB. How fast it falls is the band's RT60, the
# time it takes to drop 60 dB. In real rooms mid frequencies ring longest, and both low and high
# frequencies die away faster. `so.synth_ir` splits Gaussian
# noise into ERB-spaced bands, multiplies each band by its decay, and sums the bands back. Its
# parameters:
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
# Every room below has the same median RT60 (1 s) and direct-to-reverberant ratio (−3 dB), and
# the same seed, so they share their noise. Unlike the paper, they are not further equated for
# distortion, so some differences in loudness and length remain. They are presented diotically,
# as in the paper.

# %%
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, sosfilt

import sonore as so

plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "figure.dpi": 100})
FS = 44100


def finish(snd):
    """How every sound in the gallery is played: 5 ms ramps, RMS 0.1, peak at most 0.95."""
    snd = snd.ramp(5e-3).normalize(rms=0.1)
    return snd.normalize(peak=0.95) if snd.peak > 0.95 else snd


def starter_pistol():
    """A sharp broadband crack: a shock-like pulse (0.15 ms exponential), highpassed.
    Its spectrum is smooth; a short burst of noise would have random deep notches."""
    t = np.arange(int(0.03 * FS)) / FS
    x = sosfilt(butter(2, 300, "highpass", fs=FS, output="sos"), np.exp(-t / 0.15e-3))
    return so.Sound(x, FS).pad(before=0.05, after=0.05)


def room(**kw):
    """A synthetic room: RT60 1 s, direct-to-reverberant ratio -3 dB, a fixed seed."""
    return so.synth_ir(1.0, FS, drr_db=-3, rng=5, **kw)


def show(snd, ir, kw=None):
    """Waveform and cochleagram of the sound, then the IR's decay by band and its RT60s.
    Returns the figure and the panels the playhead follows."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.2), layout="constrained")
    snd.plot(axes[0, 0], lw=0.4)
    axes[0, 0].set_title("Waveform")
    # no extra lowpass: resampling to 1 kHz is already band-limited, and a
    # lowpass would ring visibly around the sharp onset of the shot
    cochleagram = so.cosine_filterbank(30, 50, 8000).analyze(snd).envelopes(fs=1000)
    cochleagram.plot(axes[0, 1], colorbar=False, db_range=60)
    axes[0, 1].set_title("Cochleagram (60 dB range)")
    time_axes = [axes[0, 0], axes[0, 1]]
    if ir is None:
        for ax in axes[1]:
            ax.axis("off")
        axes[1, 0].text(0.0, 0.5, "No room: the dry source.", transform=axes[1, 0].transAxes, fontsize=11)
        return fig, time_axes
    # band decays of the impulse response itself (dB), a few bands
    tail = so.Sound(ir.data[1:], ir.fs)
    env = so.cosine_filterbank(30, 50, 8000).analyze(tail).envelopes(lowpass=30, fs=1000)
    for f, color in zip(
        (125, 500, 2000, 6000), ("tab:blue", "tab:green", "tab:orange", "tab:red"), strict=True
    ):
        k = int(np.argmin(np.abs(env.cfs - f)))
        db = env.db[:, k, 0]
        axes[1, 0].plot(env.t, db - db.max(), color=color, lw=0.8, label=f"{env.cfs[k]:.0f} Hz")
    axes[1, 0].set(
        ylim=(-70, 3), xlabel="Time [s]", ylabel="Band envelope [dB]", title="The IR's decay in four bands"
    )
    axes[1, 0].legend(fontsize=8, loc="upper right")
    axes[1, 0].grid(ls=":")
    # measured RT60 profile vs the ecological one
    cfs, rt = so.measure_rt60(tail)
    eco = so.band_rt60s(1.0, cfs, "ecological")
    axes[1, 1].semilogx(cfs, eco, color="k", lw=1.2, ls="--", label="natural rooms (RT60 = 1 s)")
    profile = (kw or {}).get("rt60_profile")
    if profile:
        # requested profile, computed over synth_ir's own bands (which run to 16 kHz)
        synth_cfs = so.cosine_filterbank(32, 20, min(16000, 0.95 * FS / 2)).cfs
        requested = np.interp(cfs, synth_cfs, so.band_rt60s(1.0, synth_cfs, profile))
        axes[1, 1].semilogx(cfs, requested, color="tab:purple", lw=1, ls=":", label=f"requested ({profile})")
    if not kw or "decay_shape" not in kw:
        axes[1, 1].semilogx(cfs, rt, color="tab:purple", lw=1.2, marker=".", label="this IR (measured)")
    else:
        axes[1, 1].text(
            0.03,
            0.06,
            "decay isn't exponential, so an RT60\nisn't meaningful for this IR",
            transform=axes[1, 1].transAxes,
            fontsize=8,
        )
    axes[1, 1].set(
        xlabel="Frequency [Hz]", ylabel="RT60 [s]", title="Decay time by frequency", ylim=(0, None)
    )
    axes[1, 1].legend(fontsize=8, loc="upper right")
    axes[1, 1].grid(ls=":", which="both")
    return fig, time_axes


pistol = starter_pistol()

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
# A synthetic room with RT60 = 1 s whose decay follows the statistics of 271 real rooms:
# exponential, with mid frequencies ringing longest. In Traer & McDermott's experiments, listeners
# could not tell IRs synthesized this way from real ones.

# %% [demo 17] In a natural room
ir = room()
sound = finish(pistol.convolve(ir))
fig, playhead = show(sound, ir)

# %% [about]
# The sentence from [Seeing speech](speech.html), resampled to 44.1 kHz, in the same room. The
# reverberation fills the gaps between words and smears each syllable into the next.

# %% [demo r1] A sentence in the same room
sentence = so.load("docs/speech/bdl_arctic_a0131.flac").resample(FS)
sound = finish(sentence.convolve(ir))
fig, playhead = show(sound, ir)

# %% [about]
# The same sentence read by a female talker (slt), in the same room. The impulse response is the
# room's, not the talker's, so it does to the female voice what it did to the male one.

# %% [demo r3] A higher voice in the same room
sentence_female = so.load("docs/speech/slt_arctic_a0131.flac").resample(FS)
sound = finish(sentence_female.convolve(ir))
fig, playhead = show(sound, ir)

# %% [markdown]
# ## Farther away
#
# The direct sound from a point source falls 6 dB for every doubling of distance (the inverse
# square law; `so.distance_gain_db`). The reverberant tail, built up from reflections off every
# wall, stays at about the same level anywhere in the room. So moving a talker away lowers the
# direct-to-reverberant ratio by as much as it lowers the direct sound, and that ratio is one of
# the cues to how far away a source is. Here the talker is four times farther than above. The
# cell below prints the direct sound's level change at two and four times the distance: the
# second number is the 12 dB the next demo takes off the direct sound.

# %%
for distance_ratio in [2, 4]:
    print(f"{distance_ratio} times farther: {so.distance_gain_db(distance_ratio):.1f} dB")

# %% [about]
# The same sentence and the same room, with the direct sound 12 dB weaker: the
# direct-to-reverberant ratio drops from −3 to −15 dB. Every sound on this page is played at the
# same RMS, so what changes is not the loudness but how much of the sentence is room.

# %% [demo r2] The same sentence, four times farther
ir_far = so.synth_ir(1.0, FS, drr_db=-3 + so.distance_gain_db(4), rng=5)
sound = finish(sentence.convolve(ir_far))
fig, playhead = show(sound, ir_far)

# %% [markdown]
# ## Rooms that break the rules
#
# Synthetic rooms that break the regularities of real ones sound wrong (Traer & McDermott,
# 2016). Each room below changes one thing about the natural room above, and is heard twice:
# first with the starter pistol, which lays the room bare, then with the sentence, to compare
# with [the sentence in the natural room](#d-r1). Where the decay is no longer exponential, an
# RT60 no longer describes it, so the last panel shows only what natural rooms would do.

# %% [about]
# The same decay run backwards: the reverberation swells up to the shot instead of dying away
# after it.

# %% [demo 18] Time-reversed decay
kw = {"decay_shape": "time_reversed"}
ir = room(**kw)
sound = finish(pistol.convolve(ir))
fig, playhead = show(sound, ir, kw)

# %% [about]
# The sentence in the time-reversed room. Each sound's reverberation now grows after it rather
# than fading, cresting about a second later, so every syllable is followed by a swell of itself
# and the swells pile up across the sentence.

# %% [demo r4] The sentence, time-reversed decay
sound = finish(sentence.convolve(ir))
fig, playhead = show(sound, ir, kw)

# %% [about]
# Starts at the natural level but falls linearly (in amplitude) rather than exponentially, with
# the same energy per band; it has to end early to do so.

# %% [demo 19] Linear decay, matched start
kw = {"decay_shape": "linear_matched_start"}
ir = room(**kw)
sound = finish(pistol.convolve(ir))
fig, playhead = show(sound, ir, kw)

# %% [about]
# The sentence in the room whose decay falls linearly from the natural starting level. Its tail
# is over within a quarter of a second, where the natural one lasts more than a second, so less
# of each word spills into the gaps after it.

# %% [demo r5] The sentence, linear decay, matched start
sound = finish(sentence.convolve(ir))
fig, playhead = show(sound, ir, kw)

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
# second in, its tail is about 11 dB below where it started, the natural one about 27 dB, so each
# word lingers at nearly full strength before it falls away.

# %% [demo r6] The sentence, linear decay, matched end
sound = finish(sentence.convolve(ir))
fig, playhead = show(sound, ir, kw)

# %% [about]
# Exponential, but low and high frequencies ring longest and the middle dies fastest, the
# reverse of real rooms. In the paper listeners often heard a separate high-frequency hiss
# rather than a room.

# %% [demo 21] Inverted frequency dependence
kw = {"rt60_profile": "inverted"}
ir = room(**kw)
sound = finish(pistol.convolve(ir))
fig, playhead = show(sound, ir, kw)

# %% [about]
# The sentence in the room where low and high frequencies ring longest. Under 1% of the
# sentence's energy is above 4 kHz, against about 20% of the shot's, so the long high-frequency
# tail has much less to ring with here.

# %% [demo r7] The sentence, inverted frequency dependence
sound = finish(sentence.convolve(ir))
fig, playhead = show(sound, ir, kw)

# %% [about]
# The profile of a room twice as reverberant, scaled down: more sharply peaked than real rooms
# of this size. The subtlest variant; in the paper it was detected with impulses but not with
# speech.

# %% [demo 22] Exaggerated frequency dependence
kw = {"rt60_profile": "exaggerated"}
ir = room(**kw)
sound = finish(pistol.convolve(ir))
fig, playhead = show(sound, ir, kw)

# %% [about]
# The sentence in the room with the sharply peaked profile: the room the paper's listeners
# could not tell apart from a natural one when they heard it with speech.

# %% [demo r8] The sentence, exaggerated frequency dependence
sound = finish(sentence.convolve(ir))
fig, playhead = show(sound, ir, kw)

# %% [about]
# The profile of a room half as reverberant, scaled up: flatter than real rooms of this size.
# Also subtle.

# %% [demo 23] Reduced frequency dependence
kw = {"rt60_profile": "reduced"}
ir = room(**kw)
sound = finish(pistol.convolve(ir))
fig, playhead = show(sound, ir, kw)

# %% [about]
# The sentence in the room with the flatter profile.

# %% [demo r9] The sentence, reduced frequency dependence
sound = finish(sentence.convolve(ir))
fig, playhead = show(sound, ir, kw)

# %% [markdown]
# ## References
#
# - Kominek & Black (2004). The CMU Arctic speech databases. *Proc. 5th ISCA Speech Synthesis
#   Workshop*, 223–224. [ISCA Archive](https://www.isca-archive.org/ssw_2004/kominek04b_ssw.html).
#   The sentence, by speakers bdl and slt.
# - Traer & McDermott (2016). Statistics of natural reverberation enable perceptual separation of
#   sound and space. *Proc. Natl. Acad. Sci. USA* 113(48), E7856–E7865.
#   [doi:10.1073/pnas.1612524113](https://doi.org/10.1073/pnas.1612524113).
#   [`reverb.synth_ir`](https://github.com/choyun1/sonore/blob/main/src/sonore/spatial/reverb.py#L77)
