"""Rebuilding and changing a voice: a pitch track and a spectral envelope, put back together as
sound, and changed one at a time.

This script is the gallery page https://choyun1.github.io/sonore/gallery/voice.html:
docs/gallery/build.py runs it cell by cell from the repository root and shows each
cell's code beside what it made. Run it yourself from the repository root,

    python docs/gallery/voice/voice.py

or a cell at a time ("# %%" starts a cell in VS Code, Spyder and Jupytext).
"""

# %% [markdown]
# # Rebuilding and changing a voice
#
# A voiced sound is a set of harmonics, multiples of a pitch $F_0$ that moves over time, whose
# levels follow a smooth spectral envelope, the formants (see [The source-filter
# model](sourcefilter.html)). Measure the two, a pitch track and an envelope, and a voice
# can be put back together from them. Change one before putting it back, and the pitch moves
# without the formants, or the formants without the pitch. This page does both, on the sentence
# read by the [two talkers](whatisavoice.html#h-the-two-talkers), shown side by side throughout.
#
# The first part starts from the simple rebuild of [Putting it back,
# simply](sourcefilter.html#h-putting-it-back-simply), harmonics from `so.harmonic_complex` shaped
# by a liftered envelope:
#
# - [Envelopes on harmonics and on noise](#h-envelopes-on-harmonics-and-on-noise): a channel
#   vocoder on a harmonic carrier, and a whisper.
#
# The second part changes the voice with a vocoder built from WORLD's analyses (Morise, Yokomori &
# Ozawa, 2016), which add a third measurement, the aperiodicity (see [A richer source-filter
# model](richer.html)).
# Two changes are basic: `so.scale_f0(track, ratio)` multiplies every F0 by a ratio and leaves the
# envelope alone; `so.warp_frequency(envelope, ratio)` reads the envelope at $f / \text{ratio}$, so
# a peak at 1000 Hz moves to $1000 \times \text{ratio}$ Hz, and leaves the pitch alone.
#
# - [Resynthesis](#h-resynthesis): nothing changed, and how each change is
#   measured.
# - [Pitch and formants, one at a time](#h-pitch-and-formants-one-at-a-time): each changed alone,
#   and both together.
# - [The phase vocoder moves both](#h-the-phase-vocoder-moves-both): the same pitch change by time
#   stretch and resampling, for contrast.
# - [Pitch range](#h-pitch-range): a monotone, a doubled range and a flipped contour.
# - [The aperiodicity](#h-the-aperiodicity): kept in place or moved with the formants.
# - [Toward another talker](#h-toward-another-talker): each talker moved toward the other.
# - [Any pitch track, any envelope](#h-any-pitch-track-any-envelope): three trackers and four
#   envelopes mixed, scored on resynthesis.
# - [What this page leaves out](#h-what-this-page-leaves-out).
#
# The page measures what each change did; it does not say how the results sound.

# %% [markdown]
# ## The sentence, and code the examples share
#
# The two recordings and their `so.f0_track` pitch tracks, as on [What is a voice](whatisavoice.html#d-tk1).
# Every rebuild and change on this page follows those tracks, except where the last section swaps
# in other trackers.

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
fs = talkers["Male talker"].fs


def contour_of(track):
    """The times and F0 values of a track (an F0Track or a (times, values) pair), NaN where
    unvoiced, for drawing."""
    times, values = (track.t, track.f0[0]) if hasattr(track, "f0") else track
    return times, np.where(np.asarray(values) > 0, values, np.nan)


def show_pair(sounds, title, contours=None, fmax=5000):
    """One column per talker: the waveform, and a narrowband spectrogram (Hann 33 ms, which
    resolves the harmonics of both voices) with an F0 contour over it if one is given. The title
    is one string, or one per talker. Returns the figure and, for each talker, the panels the
    playhead follows."""
    fig = plt.figure(figsize=(10, 4.4), layout="constrained")
    columns = fig.subfigures(1, 2)
    playhead = {}
    for column, (label, snd) in zip(columns, sounds.items(), strict=True):
        top, bottom = column.subplots(2, 1, sharex=True, height_ratios=[0.4, 1])
        snd.plot(top, color="k", lw=0.4)
        top.set(title=f"{label}: {title[label] if isinstance(title, dict) else title}", xlabel="", ylabel="")
        so.STFT(snd, win_dur=0.0333, hop_dur=0.002).plot(bottom, db_range=70, colorbar=False, fmax=fmax)
        bottom.set_title("Spectrogram (Hann 33 ms)")
        if contours is not None:
            times, values = contour_of(contours[label])
            bottom.plot(times, values / 1000, color="c", lw=1.2)  # the spectrogram's axis is in kHz
        for ax in (top, bottom):
            ax.set_xlim(0, snd.duration)
        playhead[label] = [top, bottom]
    return fig, playhead


medians = {label: np.median(track.f0[0][track.voiced[0]]) for label, track in tracks.items()}
for label, median in medians.items():
    print(f"{label}: median voiced F0 {median:.0f} Hz")

# %% [about]
# The sentence, read by each talker, with `so.f0_track`'s contour drawn on the spectrogram.

# %% [demo vc1] The sentence
sounds = talkers
fig, playhead = show_pair(sounds, "the sentence", tracks)

# %% [markdown]
# ## Envelopes on harmonics and on noise
#
# The simplest rebuild, harmonics on the pitch track shaped by a liftered cepstral envelope with
# envelope-shaped noise where the track says unvoiced, is on [Putting it back,
# simply](sourcefilter.html#h-putting-it-back-simply). This section keeps that envelope, which the
# last section uses again, and hears two relatives of the rebuild: a channel vocoder and a
# whisper.


# %%
def cepstral_envelope(snd, track):
    """The cepstrum of 40 ms Hann time windows every 5 ms, liftered below half the pitch period
    of each window (the track's voiced F0, interpolated on a log scale across unvoiced gaps)."""
    cepstrum = so.Cepstrum(so.STFT(snd, win_dur=0.040, hop_dur=0.005))
    is_voiced = track.voiced[0]
    periods = 1 / np.exp(np.interp(cepstrum.t, track.t[is_voiced], np.log(track.f0[0][is_voiced])))
    return cepstrum.lifter(0.5 * periods).envelope_view()


def shaped_noise(snd, envelope, seed):
    """White noise with the envelope as its STFT magnitude, on the envelope's time windows."""
    noise = so.STFT(so.gaussian_noise(snd.duration, fs, rng=seed), win_dur=0.040, hop_dur=0.005)
    noise.data = np.sqrt(envelope.data) * np.exp(1j * np.angle(noise.data))
    return noise.to_sound().data[: len(snd), 0]


cepstral = {label: cepstral_envelope(snd, tracks[label]) for label, snd in talkers.items()}
breaths = {label: shaped_noise(snd, cepstral[label], seed=2) for label, snd in talkers.items()}

# %% [markdown]
# A channel vocoder ([Hearing through a vocoder](vocoder.html)) does the same job with band
# envelopes in place of a cepstral one. On noise it puts back the envelopes and not the pitch; on
# a harmonic carrier that follows the pitch track it puts the pitch back too. The carrier below is
# `so.harmonic_complex` with `unvoiced="noise"`, which fills the unvoiced stretches with white
# noise of the harmonics' power, so the consonants have something to modulate.

# %% [about]
# The sentence through a 16-band noise vocoder (Shannon et al., 1995): the band envelopes are
# there, the harmonics are not.

# %% [demo he2] Sixteen noise bands
sounds = {label: finish(so.channel_vocode(snd, 16, 80, 7600, rng=1)) for label, snd in talkers.items()}
fig, playhead = show_pair(sounds, "16-band noise vocoder")

# %% [about]
# The same 16 bands on a harmonic carrier that follows each talker's `so.f0_track`, with noise
# where it says unvoiced. The harmonics within a band share one envelope, so the formants are
# blurrier than with the cepstral envelope.

# %% [demo he3] Sixteen bands on harmonics
sounds = {
    label: finish(
        so.channel_vocode(
            snd,
            16,
            80,
            7600,
            carrier=so.harmonic_complex(snd.duration, fs, tracks[label], unvoiced="noise", rng=1),
        )
    )
    for label, snd in talkers.items()
}
fig, playhead = show_pair(sounds, "16 bands, harmonic carrier", tracks)

# %% [about]
# A whisper: no harmonics at all, the shaped noise everywhere. This is close to the [noise
# vocoder](#d-he2), with the cepstral envelope in place of 16 band envelopes.

# %% [demo hm4] Whispered
sounds = {label: finish(so.Sound(breath, fs)) for label, breath in breaths.items()}
fig, playhead = show_pair(sounds, "shaped noise only")

# %% [markdown]
# ## Resynthesis
#
# WORLD measures the envelope with CheapTrick (Morise, 2015), which [Against the lifter, on a
# recording](richer.html#h-against-the-lifter-on-a-recording) compares with the liftered cepstrum, and the
# aperiodicity with D4C (Morise, 2016): how much of the voice is noise, frequency by frequency, so
# that every time window can mix harmonics and noise rather than being one or the other as in
# [Putting it back, simply](sourcefilter.html#h-putting-it-back-simply).
# `so.source_filter_synthesize` puts the three back together: harmonics on the pitch track carrying
# the periodic part of the envelope, plus white noise shaped to the rest. The envelope and the
# aperiodicity are measured here on the `so.f0_track` tracks.
#
# Each change below is checked by measurement on the output. The pitch: `so.f0_track` on the
# output, its median over voiced time windows against the same on the original. The formants:
# CheapTrick's envelope of the output, averaged in dB over the original's voiced time windows,
# against the original's average read at $f / r$ for a range of ratios $r$. The ratio whose curve
# fits best over 100 to 5000 Hz, with the overall level left free, is the **fitted warp**.

# %%
envelopes = {label: so.pitch_adaptive_envelope(snd, tracks[label]) for label, snd in talkers.items()}
aperiodicities = {label: so.band_aperiodicity(snd, tracks[label]) for label, snd in talkers.items()}
frequencies = envelopes["Male talker"].f
scored = (frequencies >= 100) & (frequencies <= 5000)
RATIOS = np.arange(0.60, 2.001, 0.005)


def median_f0(snd):
    """The median F0 that so.f0_track measures over the voiced time windows of a sound."""
    track = so.f0_track(snd)
    return np.median(track.f0[0][track.voiced[0]])


def mean_voiced_db(view, label):
    """An envelope (or aperiodicity) averaged in dB over the talker's voiced time windows."""
    track = tracks[label]
    return np.mean(10 * np.log10(view(track.t[track.voiced[0]], frequencies)[0]), axis=1)


def misfit(changed_db, reference_db, ratio, tilt=False):
    """How far changed_db is from reference_db read at f / ratio (RMS dB, 100-5000 Hz), with the
    level left free and, if tilt is True, a straight line in dB over log frequency too. Also
    returns that level and tilt (dB per octave)."""
    difference = changed_db[scored] - np.interp(frequencies[scored] / ratio, frequencies, reference_db)
    basis = np.column_stack([np.ones(scored.sum()), np.log2(frequencies[scored])][: 2 if tilt else 1])
    fit = np.linalg.lstsq(basis, difference, rcond=None)[0]
    return np.sqrt(np.mean((difference - basis @ fit) ** 2)), (*fit, 0.0)[:2]


def fitted_warp(changed_db, reference_db, tilt=False):
    """The ratio r for which changed_db best matches reference_db read at f / r, and the tilt
    fitted with it (0 unless tilt is True)."""
    misfits = [misfit(changed_db, reference_db, ratio, tilt)[0] for ratio in RATIOS]
    best = RATIOS[np.argmin(misfits)]
    return best, misfit(changed_db, reference_db, best, tilt)[1][1]


original_db = {label: mean_voiced_db(envelope, label) for label, envelope in envelopes.items()}


def report(changed):
    """Print, for each changed sentence and each talker, the measured pitch ratio and the fitted
    warp. changed maps a name to the pair of sentences and the F0 contours they were made on."""
    print(f"{'':44}{'pitch ratio':>12}{'fitted warp':>12}")
    for name, (sounds, changed_tracks) in changed.items():
        for label, snd in sounds.items():
            changed_db = mean_voiced_db(so.pitch_adaptive_envelope(snd, changed_tracks[label]), label)
            warp = fitted_warp(changed_db, original_db[label])[0]
            print(f"{name + ', ' + label.lower():44}{median_f0(snd) / medians[label]:12.3f}{warp:12.3f}")


changed = {}  # each change's sentences, measured after the demos that make them


# %% [about]
# The resynthesis with nothing changed: the pitch track, CheapTrick's envelope and D4C's
# aperiodicity put back together. Every change below is a change of this.

# %% [demo vc2] Resynthesis, nothing changed
sounds = {
    label: finish(so.source_filter_synthesize(tracks[label], envelopes[label], aperiodicities[label], rng=1))
    for label in talkers
}
fig, playhead = show_pair(sounds, "so.source_filter_synthesize", tracks)
changed["nothing changed"] = (sounds, tracks)

# %% [markdown]
# ## Pitch and formants, one at a time
#
# The pitch up by half (a ratio of 1.5, a musical fifth), the formants up by 20% (a ratio of
# 1.2), and both, for each talker.

# %% [about]
# The pitch times 1.5, the envelope as it was. The harmonics are half as far apart again; the dark
# bands of the formants stay at the same frequencies.

# %% [demo vc3] Higher pitch
higher = {label: so.scale_f0(track, 1.5) for label, track in tracks.items()}
sounds = {
    label: finish(so.source_filter_synthesize(higher[label], envelopes[label], aperiodicities[label], rng=1))
    for label in talkers
}
fig, playhead = show_pair(sounds, "F0 × 1.5, formants kept", higher)
changed["F0 × 1.5"] = (sounds, higher)

# %% [about]
# The formants times 1.2, the pitch as it was. The harmonics are where they were; the formants sit
# 20% higher in frequency.

# %% [demo vc4] Higher formants
sounds = {
    label: finish(
        so.source_filter_synthesize(
            tracks[label], so.warp_frequency(envelopes[label], 1.2), aperiodicities[label], rng=1
        )
    )
    for label in talkers
}
fig, playhead = show_pair(sounds, "formants × 1.2, F0 kept", tracks)
changed["formants × 1.2"] = (sounds, tracks)

# %% [about]
# Both: the pitch times 1.5 and the formants times 1.2.

# %% [demo vc5] Higher pitch and formants
sounds = {
    label: finish(
        so.source_filter_synthesize(
            higher[label], so.warp_frequency(envelopes[label], 1.2), aperiodicities[label], rng=1
        )
    )
    for label in talkers
}
fig, playhead = show_pair(sounds, "F0 × 1.5, formants × 1.2", higher)
changed["both"] = (sounds, higher)

# %% [markdown]
# Measured on each output: the pitch ratio, and the fitted warp of its envelope against the
# original's.

# %%
report(changed)

# %% [markdown]
# Each change shows up in its own measurement and not in the other's. What the warp does to one
# time window's envelope, drawn on a linear frequency axis where a ratio stretches it from 0 Hz:
# every feature moves up by the same proportion, so the higher formants move farther in hertz.

# %% [figure vc6] One envelope, warped
fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), layout="constrained", sharey=False)
for ax, (label, envelope) in zip(axes, envelopes.items(), strict=True):
    track = tracks[label]
    window = np.argmax(envelope.data[0].sum(axis=0) * track.voiced[0])  # the strongest voiced time window
    for ratio, color in [(1.0, "k"), (1.2, "C3")]:
        level = 10 * np.log10(
            so.warp_frequency(envelope, ratio)(track.t[window : window + 1], frequencies)[0, :, 0]
        )
        ax.plot(frequencies, level, color=color, label=f"warp ratio {ratio:g}")
    top = 10 * np.log10(envelope.data[0, :, window].max()) + 5
    ax.vlines(
        1.5 * track.f0[0][window] * np.arange(1, 40),
        top - 70,
        top,
        color="C0",
        lw=0.5,
        alpha=0.5,
        label="harmonics of F0 × 1.5",
    )
    ax.set(xlabel="Frequency (Hz)", ylabel="Level (dB)", xlim=(0, 6000), ylim=(top - 70, top))
    ax.set_title(f"{label}: CheapTrick at {track.t[window]:.2f} s, and read at f / 1.2")
    ax.legend(loc="upper right", fontsize=8)

# %% [markdown]
# ## The phase vocoder moves both
#
# The [Phase vocoder](pv.html) page changes pitch the classic way: stretch the sound in time, then
# resample it back to its length. Resampling scales every frequency in the sound, so the envelope
# moves with the harmonics: [The two talkers trade pitches](pv.html#d-p3) there moves pitch and formants
# together, where [Higher pitch](#d-vc3) here moves the pitch alone. The measurement shows it.

# %% [about]
# Each sentence a fifth higher by `so.pitch_shift`: time stretch and resampling.

# %% [demo vc7] Phase vocoder, a fifth up
sounds = {label: finish(so.pitch_shift(snd, 12 * np.log2(1.5))) for label, snd in talkers.items()}
fig, playhead = show_pair(sounds, "so.pitch_shift, a fifth up")
changed["so.pitch_shift, a fifth up"] = (sounds, higher)

# %% [markdown]
# Measured as above, beside [Higher pitch](#d-vc3):

# %%
report({name: changed[name] for name in ("F0 × 1.5", "so.pitch_shift, a fifth up")})

# %% [markdown]
# Both raise the pitch by about 1.5. The phase vocoder's fitted warp follows its pitch ratio, so
# the formants went up a fifth too; the vocoder's stays near 1.

# %% [markdown]
# ## Pitch range
#
# `so.scale_f0` has a second argument, `range`, that spreads the contour around its median on a
# log scale: $\text{median} \times \text{ratio} \times (F_0 / \text{median})^{\text{range}}$. A
# range of 0 is a monotone at the median, and 2 doubles every interval from it. A range of $-1$,
# which `so.scale_f0` does not take, flips the contour: $\text{median}^2 / F_0$, so where the talker
# rises, the flipped contour falls by the same musical interval. The formants and the aperiodicity
# are left as they were.

# %%
ranges = {
    "range 1 (the sentence)": tracks,
    "range 0": {label: so.scale_f0(track, 1.0, range=0.0) for label, track in tracks.items()},
    "range 2": {label: so.scale_f0(track, 1.0, range=2.0) for label, track in tracks.items()},
    "flipped": {
        label: (
            track.t,
            np.where(track.voiced[0], medians[label] ** 2 / np.where(track.voiced[0], track.f0[0], 1), 0),
        )
        for label, track in tracks.items()
    },
}


def on_contours(contours):
    """Each talker's resynthesis on another contour, envelope and aperiodicity kept."""
    return {
        label: finish(
            so.source_filter_synthesize(contours[label], envelopes[label], aperiodicities[label], rng=1)
        )
        for label in talkers
    }


# %% [figure vc8] Three pitch ranges and a flipped contour
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), layout="constrained", sharey=True)
for ax, (label, snd) in zip(axes, talkers.items(), strict=True):
    for (name, contours), color, style in zip(
        ranges.items(), ["k", "C0", "C3", "C2"], ["-", "-", "-", "--"], strict=True
    ):
        ax.plot(*contour_of(contours[label]), color=color, ls=style, lw=1, label=name)
    ax.set(
        xlabel="Time (s)", xlim=(0, snd.duration), yscale="log", title=f"{label}: so.f0_track, range changed"
    )
    ax.set_yticks([50, 70, 100, 140, 200, 280, 400], ["50", "70", "100", "140", "200", "280", "400"])
    ax.set_yticks([], minor=True)
axes[0].set_ylabel("F0 (Hz), log scale")
fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncols=4, fontsize=8)

# %% [about]
# A monotone: every voiced time window at the talker's median F0. The words are all there,
# without the rises and falls that mark the stressed syllables.

# %% [demo vc9] Monotone
sounds = on_contours(ranges["range 0"])
fig, playhead = show_pair(sounds, "range 0, a monotone", ranges["range 0"])

# %% [about]
# The range doubled: every rise and fall around the median twice as large, in musical intervals.

# %% [demo vc10] Twice the range
sounds = on_contours(ranges["range 2"])
fig, playhead = show_pair(sounds, "range 2, intervals doubled", ranges["range 2"])

# %% [about]
# The contour flipped around the median on a log scale.

# %% [demo hm3] The contour flipped
sounds = on_contours(ranges["flipped"])
fig, playhead = show_pair(sounds, "contour flipped around the median", ranges["flipped"])

# %% [markdown]
# ## The aperiodicity
#
# When the formants move, the aperiodicity can stay where it is or move with them, and
# `so.warp_frequency` warps it the same way as an envelope. Which is right depends on where the
# noise comes from. Noise made at the source (breath at the glottis) passes through the formants
# like the harmonics do, so its share at each frequency belongs to the source and should stay. A
# share that follows the formants should move with them. Both are on offer;
# `so.warp_frequency(aperiodicity, ratio)` is the second.
#
# At 16 kHz, D4C's aperiodicity has one bend, at 3 kHz (see
# [What D4C reports](richer.html#h-what-d4c-reports)), and a warp by 1.2 moves the bend to
# 3.6 kHz. Averaged in dB over voiced time windows, kept and warped:

# %% [figure vc11] The aperiodicity, kept and warped
warped_aperiodicities = {label: so.warp_frequency(view, 1.2) for label, view in aperiodicities.items()}
fig, axes = plt.subplots(1, 2, figsize=(10, 3.2), layout="constrained", sharey=True)
for ax, label in zip(axes, talkers, strict=True):
    kept_db = mean_voiced_db(aperiodicities[label], label)
    warped_db = mean_voiced_db(warped_aperiodicities[label], label)
    ax.plot(frequencies, kept_db, color="k", label="kept")
    ax.plot(frequencies, warped_db, color="C3", label="warped by 1.2")
    ax.set(xlabel="Frequency (Hz)", xlim=(0, 8000), title=f"{label}: D4C, mean over voiced time windows")
    ax.legend(loc="lower right", fontsize=8)
axes[0].set_ylabel("Share of noise (dB)")

# %% [about]
# The formants times 1.2 with the aperiodicity warped too. The other formant changes on this page
# keep it; compare [Higher formants](#d-vc4).

# %% [demo vc12] Higher formants, aperiodicity warped
sounds = {
    label: finish(
        so.source_filter_synthesize(
            tracks[label], so.warp_frequency(envelopes[label], 1.2), warped_aperiodicities[label], rng=1
        )
    )
    for label in talkers
}
fig, playhead = show_pair(sounds, "formants and aperiodicity × 1.2", tracks)

# %% [markdown]
# ## Toward another talker
#
# The two talkers differ in pitch and in their spectral envelopes, which [Two
# talkers](whatisavoice.html#h-size-the-spectral-envelopes) measures: averaged over voiced time windows, the
# female talker's envelope is close to the male talker's read at $f / r$ for one ratio $r$, once a
# tilt (a straight line in dB over log frequency) is left free as well. The cell fits that ratio
# and tilt as that page does, and takes the pitch ratio as the ratio of the median F0s. The male
# talker is moved by the ratios and the tilt; the female talker by their inverses, the pitch
# divided by the pitch ratio, the envelope read at $f \times r$ and the tilt reversed. Each talker
# is moved toward the other in three steps: the pitch, then the formants, then the tilt.

# %%
toward = {"Male talker": "Female talker", "Female talker": "Male talker"}
pitch_ratio = medians["Female talker"] / medians["Male talker"]
warp, tilt = fitted_warp(original_db["Female talker"], original_db["Male talker"], tilt=True)
print(f"pitch ratio {pitch_ratio:.3f}, fitted warp {warp:.3f}, tilt {tilt:+.1f} dB per octave")
pitch_ratios = {"Male talker": pitch_ratio, "Female talker": 1 / pitch_ratio}
warps = {"Male talker": warp, "Female talker": 1 / warp}
tilts = {"Male talker": tilt, "Female talker": -tilt}


def tilted(view, ratio, tilt):
    """An envelope warped by ratio, and tilted by tilt dB per octave (0 dB at 1 kHz)."""
    warped = so.warp_frequency(view, ratio)

    def envelope(t, f):
        gain = 10 ** (tilt * np.log2(np.maximum(f, 50.0) / 1000) / 10)
        return warped(t, f) * gain[None, :, None]

    return envelope


moved = {label: so.scale_f0(tracks[label], pitch_ratios[label]) for label in talkers}

# %% [about]
# Each talker at the other talker's median pitch, the formants as they were. Compare the
# originals in [The sentence](#d-vc1).

# %% [demo vc14] Each at the other pitch
sounds = on_contours(moved)
fig, playhead = show_pair(sounds, {label: f"F0 × {pitch_ratios[label]:.2f}" for label in talkers}, moved)
changed["pitch"] = (sounds, moved)

# %% [about]
# The same, with the formants moved by the fitted ratio.

# %% [demo vc15] Each at the other pitch and formants
sounds = {
    label: finish(
        so.source_filter_synthesize(
            moved[label], so.warp_frequency(envelopes[label], warps[label]), aperiodicities[label], rng=1
        )
    )
    for label in talkers
}
titles = {label: f"F0 × {pitch_ratios[label]:.2f}, formants × {warps[label]:.2f}" for label in talkers}
fig, playhead = show_pair(sounds, titles, moved)
changed["pitch and formants"] = (sounds, moved)

# %% [about]
# The same, with the fitted tilt as well.

# %% [demo vc19] Each at the other pitch, formants and tilt
sounds = {
    label: finish(
        so.source_filter_synthesize(
            moved[label], tilted(envelopes[label], warps[label], tilts[label]), aperiodicities[label], rng=1
        )
    )
    for label in talkers
}
titles = {
    label: f"F0 × {pitch_ratios[label]:.2f}, formants × {warps[label]:.2f}, tilt {tilts[label]:+.1f} dB/oct"
    for label in talkers
}
fig, playhead = show_pair(sounds, titles, moved)
changed["pitch, formants and tilt"] = (sounds, moved)

# %% [markdown]
# Measured on each step: the pitch ratio and the fitted warp against the talker's own original,
# as above. Then how far each step's average envelope is from the other talker's: the RMS of the
# difference in dB, level removed, over 100 to 5000 Hz and over 100 to 4000 Hz, starting from the
# originals.

# %%
steps = ["pitch", "pitch and formants", "pitch, formants and tilt"]
report({name: changed[name] for name in steps})


def level_free_rms(a_db, b_db, top):
    """RMS dB between two average envelopes from 100 Hz to top, with the mean difference removed."""
    band = (frequencies >= 100) & (frequencies <= top)
    difference = a_db[band] - b_db[band]
    return np.sqrt(np.mean((difference - difference.mean()) ** 2))


for top in (5000, 4000):
    print(f"\nfrom the other talker's average envelope, 100 to {top} Hz:")
    for label, other in toward.items():
        averages = [original_db[label]] + [
            mean_voiced_db(
                so.pitch_adaptive_envelope(changed[name][0][label], changed[name][1][label]), label
            )
            for name in steps
        ]
        distances = [level_free_rms(average, original_db[other], top) for average in averages]
        print(f"{label:14}" + ", ".join(f"{d:.1f}" for d in distances) + " dB")
around_5k = (frequencies >= 4500) & (frequencies <= 5500)
print("\naverage envelope, dB re its mean over 100 to 5000 Hz:")
for label, average in original_db.items():
    relative = average - average[scored].mean()
    lowest = np.argmin(np.where(around_5k, relative, np.inf))
    at = {f: relative[np.argmin(np.abs(frequencies - f))] for f in (4000, 6000)}
    print(
        f"{label:14}{at[4000]:+.1f} at 4 kHz, {relative[lowest]:+.1f} at {frequencies[lowest]:.0f} Hz "
        f"(lowest from 4.5 to 5.5 kHz), {at[6000]:+.1f} at 6 kHz"
    )

# %% [markdown]
# The pitch step leaves the average envelopes where they were, as it should. The formant and tilt
# steps bring the male talker's average envelope closer to the female talker's. Moving the female
# talker by the inverse brings it closer to the male talker's below 4 kHz, but farther over the
# band up to 5 kHz: the male talker's average has a deep valley near 5 kHz that the female
# talker's lacks. Reading the male envelope at $f / r$ moves that valley above 5 kHz, out of the
# band the ratio was fitted on; reading the female envelope at $f \times r$ cannot make it. One
# ratio and one tilt fit the averages in one direction better than in the other.
#
# Two ratios and a tilt move the averages; they leave the shape of each formant, the voice
# quality, the aperiodicity, the timing and the accent as they were.

# %% [markdown]
# ## Any pitch track, any envelope
#
# `so.source_filter_synthesize` reads any envelope at the aperiodicity's time windows and
# frequencies, and any pitch track at its own times, so the pitch can come from one analysis and the
# envelope from another. Three pitch tracks, those of [Measuring the source:
# pitch](sourcefilter.html#h-measuring-the-source-pitch): `so.f0_track`, Harvest (Morise, 2017)
# stored with each recording, and the cepstral peak in each 40 ms window (Noll, 1967). Four
# envelopes: CheapTrick's, the liftered cepstrum of [the first
# part](#h-envelopes-on-harmonics-and-on-noise), and the envelope kept by 13 MFCCs (26 mel bands,
# Davis & Mermelstein, 1980; see [MFCCs: a cepstrum on the mel
# scale](sourcefilter.html#h-mfccs-a-cepstrum-on-the-mel-scale)) with each band's triangle of height
# 1 (the default) or of area 1. Triangles of height 1 sum more power the wider they are, so that
# envelope rises with frequency; area 1 removes most of the rise.
#
# A recording has no true envelope to score against, so the score is a distance between the
# original's and the resynthesis's 40-band log mel spectrograms: the RMS of their difference in
# dB per time window with the level removed, median over voiced time windows. It smooths the way
# the MFCC envelope does, so it favors that envelope; read it as a check, not a ranking.
# `tools/compare_voice_methods.py` also scores synthetic vowels against their true envelopes.

# %%
pitch_tracks, mixed_envelopes, mixed_aperiodicities = {}, {}, {}
for label, snd in talkers.items():
    harvest = tuple(
        np.loadtxt(fetch(f"docs/speech/{SPEAKERS[label]}_arctic_a0131_f0.csv"), delimiter=",", skiprows=2).T
    )
    cepstral_times, cepstral_f0, _ = so.Cepstrum(so.STFT(snd, win_dur=0.040, hop_dur=0.005)).f0(
        f_lo=75, f_hi=400
    )
    pitch_tracks[label] = {
        "so.f0_track": tracks[label],
        "Harvest": harvest,
        "cepstral": (cepstral_times, cepstral_f0[0]),
    }
    mixed_envelopes[label] = {
        "CheapTrick": envelopes[label],
        "cepstral": cepstral[label],
        "MFCC": so.MFCC(snd).envelope_view(),
        "MFCC, area": so.MFCC(snd, triangles="area").envelope_view(),
    }
    # the cepstral track's time windows start before 0 s, so the cepstral pitch is resynthesized
    # with the aperiodicity measured on so.f0_track's time windows
    mixed_aperiodicities[label] = {
        "so.f0_track": aperiodicities[label],
        "Harvest": so.band_aperiodicity(snd, harvest),
        "cepstral": aperiodicities[label],
    }


def log_mel(snd):
    """The 40-band log mel spectrogram (25 ms Hamming, 10 ms hop) used for the score."""
    return so.MFCC(snd, n_mels=40).mel_db[0]


def distance(snd, label):
    """Level-free RMS dB between the log mel spectrograms of snd and the talker's recording,
    median over the recording's voiced time windows."""
    original = log_mel(talkers[label])
    track = tracks[label]
    mel_voiced = np.interp(so.MFCC(talkers[label]).t, track.t, track.voiced[0].astype(float)) > 0.5
    resynthesized = log_mel(snd)
    count = min(original.shape[1], resynthesized.shape[1])
    difference = (resynthesized[:, :count] - original[:, :count])[:, mel_voiced[:count]]
    return np.median(np.sqrt(np.mean((difference - difference.mean(axis=0)) ** 2, axis=0)))


for label in talkers:
    print(f"{label}\n{'  pitch from':<14}" + "".join(f"{name:>13}" for name in mixed_envelopes[label]))
    for track_name, track in pitch_tracks[label].items():
        aperiodicity = mixed_aperiodicities[label][track_name]
        scores = [
            distance(so.source_filter_synthesize(track, view, aperiodicity, rng=1), label)
            for view in mixed_envelopes[label].values()
        ]
        print(f"  {track_name:<12}" + "".join(f"{score:>10.2f} dB" for score in scores))

# %% [markdown]
# CheapTrick's envelope scores best for both talkers, whichever track drives it. The liftered
# cepstrum scores worse for the female talker than for the male talker: with the harmonics farther
# apart, the lifter has to cut lower (see [One time window](sourcefilter.html#h-one-time-window)).
#
# The cepstral track's time windows start before 0 s, so its resynthesis uses the aperiodicity
# measured on `so.f0_track`'s time windows; `so.source_filter_synthesize` reads the cepstral track at its
# own times.

# %% [about]
# The pitch from the cepstrum and the envelope from the MFCCs (area 1); the only WORLD analysis is
# D4C's aperiodicity.

# %% [demo vc17] Cepstral pitch, MFCC envelope
sounds = {
    label: finish(
        so.source_filter_synthesize(
            pitch_tracks[label]["cepstral"],
            mixed_envelopes[label]["MFCC, area"],
            mixed_aperiodicities[label]["cepstral"],
            rng=1,
        )
    )
    for label in talkers
}
cepstral_tracks = {label: pitch_tracks[label]["cepstral"] for label in talkers}
fig, playhead = show_pair(sounds, "cepstral F0, MFCC envelope", cepstral_tracks)

# %% [about]
# The same mixture with both changes: the cepstral pitch times 1.5 and the MFCC envelope times
# 1.2.

# %% [demo vc18] Cepstral pitch and MFCC envelope, both changed
changed_tracks = {label: so.scale_f0(track, 1.5) for label, track in cepstral_tracks.items()}
sounds = {
    label: finish(
        so.source_filter_synthesize(
            changed_tracks[label],
            so.warp_frequency(mixed_envelopes[label]["MFCC, area"], 1.2),
            mixed_aperiodicities[label]["cepstral"],
            rng=1,
        )
    )
    for label in talkers
}
fig, playhead = show_pair(sounds, "cepstral F0 × 1.5, MFCC envelope × 1.2", changed_tracks)
changed["cepstral F0 × 1.5, MFCC × 1.2"] = (sounds, higher)

# %%
report({"cepstral F0 × 1.5, MFCC × 1.2": changed["cepstral F0 × 1.5, MFCC × 1.2"]})

# %% [markdown]
# `so.source_filter_synthesize` builds its harmonics with `so.harmonic_complex`, as in [Putting it
# back, simply](sourcefilter.html#h-putting-it-back-simply): it reads the envelope at each
# harmonic's frequency and time, on the aperiodicity's grid (15.6 Hz apart at 16 kHz), so an
# envelope with deep, narrow valleys comes through nearly as it is.

# %% [markdown]
# ## What this page leaves out
#
# - **Listening.** The measurements say where the pitch and formants went, not how the results
#   sound, nor who they sound like.
# - **The glottal pulse.** The harmonics' phases are fixed numbers (cosine phase in
#   `so.source_filter_synthesize`). A voice's phases come from the shape of each glottal pulse
#   and the vocal tract's phase response (see [The source-filter model](sourcefilter.html#h-voice-quality)).
# - **One talker into another.** A pitch ratio, a formant ratio and a tilt leave the voice
#   quality, the aperiodicity, the timing and the accent as they were.
# - **Ratios that change across frequency.** Formants do not all move by the same ratio between
#   talkers. `so.warp_frequency` also takes a ratio that changes over time, or any map from
#   output frequency to source frequency, but this page uses one ratio throughout.
# - **The consonants.** Where the track says unvoiced, the resynthesis is noise shaped by the
#   envelope, so a formant warp moves the noise of the fricatives too.

# %% [markdown]
# ## References
#
# - Davis & Mermelstein (1980). Comparison of parametric representations for monosyllabic word
#   recognition in continuously spoken sentences. *IEEE Trans. Acoust., Speech, Signal Process.*
#   28(4), 357–366. [`mfcc.MFCC`](https://github.com/choyun1/sonore/blob/main/src/sonore/views/mfcc.py#L106)
# - Kominek & Black (2004). The CMU Arctic speech databases. *Proc. 5th ISCA Speech Synthesis
#   Workshop*, 223–224. [ISCA Archive](https://www.isca-archive.org/ssw_2004/kominek04b_ssw.html).
#   The sentence, by speakers bdl and slt.
# - Morise (2015). CheapTrick, a spectral envelope estimator for high-quality speech synthesis.
#   *Speech Communication* 67, 1–7.
#   [doi:10.1016/j.specom.2014.09.003](https://doi.org/10.1016/j.specom.2014.09.003).
#   [`spectral_envelope.pitch_adaptive_envelope`](https://github.com/choyun1/sonore/blob/main/src/sonore/views/spectral_envelope.py#L249)
# - Morise (2016). D4C, a band-aperiodicity estimator for high-quality speech synthesis. *Speech
#   Communication* 84, 57–65.
#   [doi:10.1016/j.specom.2016.09.001](https://doi.org/10.1016/j.specom.2016.09.001).
#   [`aperiodicity.band_aperiodicity`](https://github.com/choyun1/sonore/blob/main/src/sonore/views/aperiodicity.py#L132)
# - Morise (2017). Harvest: a high-performance fundamental frequency estimator from speech
#   signals. *Proc. Interspeech 2017*, 2321–2325.
#   [doi:10.21437/Interspeech.2017-68](https://doi.org/10.21437/Interspeech.2017-68). The stored
#   F0 tracks.
# - Morise, Yokomori & Ozawa (2016). WORLD: a vocoder-based high-quality speech synthesis system for
#   real-time applications. *IEICE Trans. Inf. & Syst.* E99-D(7), 1877–1884.
#   [doi:10.1587/transinf.2015EDP7457](https://doi.org/10.1587/transinf.2015EDP7457).
# - Noll (1967). Cepstrum pitch determination. *J. Acoust. Soc. Am.* 41(2), 293–309.
#   [PubMed](https://pubmed.ncbi.nlm.nih.gov/6040805/).
#   [`cepstrum.Cepstrum.f0`](https://github.com/choyun1/sonore/blob/main/src/sonore/views/cepstrum.py#L185)
# - Shannon, Zeng, Kamath, Wygonski & Ekelid (1995). Speech recognition with primarily temporal
#   cues. *Science* 270(5234), 303–304.
#   [doi:10.1126/science.270.5234.303](https://doi.org/10.1126/science.270.5234.303).
#   [`envelopes.channel_vocode`](https://github.com/choyun1/sonore/blob/main/src/sonore/views/envelopes.py#L497)
