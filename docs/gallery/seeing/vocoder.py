"""Hearing through a vocoder: simulating cochlear-implant hearing with a noise vocoder.

This script is the gallery page https://choyun1.github.io/sonore/gallery/vocoder.html:
docs/gallery/build.py runs it cell by cell from the repository root and shows each
cell's code beside what it made. Run it yourself from the repository root,

    python docs/gallery/seeing/vocoder.py

or a cell at a time ("# %%" starts a cell in VS Code, Spyder and Jupytext).
"""

# %% [markdown]
# # Hearing through a vocoder
#
# A cochlear implant replaces the hair cells with a row of electrodes in the cochlea. Its
# processor splits sound into a few frequency bands, keeps only each band's slowly varying
# envelope, and uses it to set the current on one electrode (Wilson et al., 1991). The fine
# structure inside each band, which carries most of what a normal ear hears as pitch, is thrown
# away. A noise vocoder does the same to a sound and plays the result to a normal ear: each
# band's envelope modulates a band of noise in its place (Shannon et al., 1995). It is the
# standard way to let a listener with normal hearing hear roughly what an implant passes on.
#
# Everything on this page is one function, `so.channel_vocode`, applied to the sentence read by
# the [two talkers](talkers.html) and to a short melody:
#
# - [How many bands](#h-how-many-bands): from a coarse picture of the spectrum to a fine one.
# - [Noise or tones](#h-noise-or-tones): what carries the envelopes.
# - [Pitch from the envelope](#h-pitch-from-the-envelope): the weak temporal pitch cue an implant
#   leaves, for a low voice and a higher one.
# - [A melody](#h-a-melody): music, which cannot do without pitch.
# - [What this simulation leaves out](#h-what-this-simulation-leaves-out): current spread,
#   insertion depth, and the rest.

# %% [markdown]
# ## Motivation
#
# The vocoder lets you vary one thing at a time that an implant user cannot: how many bands,
# how fast the envelopes may change, and what carries them. That is how its classic results were
# found:
#
# - **Few bands are enough for speech in quiet.** Shannon et al. (1995) found high recognition of
#   the words in simple sentences through only three or four bands of noise, with no fine
#   structure at all.
# - **Implant users get only a handful.** Implants have 12 to 22 electrodes, but current spreads
#   between neighbors. Friesen et al. (2001) found that implant users did no better with more
#   than about seven or eight channels, while listeners hearing a vocoder kept improving up to the
#   twenty channels tested, especially in noise.
# - **Pitch is what suffers most.** With the fine structure gone, the only pitch cues left are
#   which bands are loud and how fast the envelopes fluctuate. Music, intonation, and telling one
#   voice from another all depend on pitch, and are hard for implant users.

# %% [markdown]
# ## The sentence, and code the examples share
#
# Both talkers read the same sentence, recorded at 16 kHz. The vocoder splits each recording
# with half-cosine filters equally spaced on the ERB scale, neighbors overlapping by half, that
# together cover 80 Hz to 7.6 kHz; what lies below 80 Hz and between 7.6 kHz and the 8 kHz
# Nyquist frequency is dropped. It takes each band's Hilbert envelope, lowpasses it at 50 Hz
# unless stated (a fourth-order Butterworth filter run forward and backward), and multiplies it
# into the same band of fresh Gaussian noise. The output has the input's RMS. Every sentence
# figure has one column per talker, the male talker on the left.

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
# The sentence, read by each talker, at its native 16 kHz. Sources: docs/speech/SOURCES.md.
SPEAKERS = {"Male talker": "bdl", "Female talker": "slt"}
talkers = {
    label: finish(so.load(fetch(f"docs/speech/{speaker}_arctic_a0131.flac")))
    for label, speaker in SPEAKERS.items()
}
tracks = {label: so.f0_track(snd) for label, snd in talkers.items()}
fs = talkers["Male talker"].fs
F_LO, F_HI = 80, 7600

# Every spectrogram: a wideband STFT (Hann 5 ms), dB re its own maximum over 60 dB, 0 to 8 kHz.
wide = so.GaborFrame(0.005, 0.001, n_fft=1024)
FMAX, DB = 8000, 60


def bands(snd, n_bands, env_lowpass=50.0):
    """The envelopes the vocoder keeps: ERB bands, Hilbert envelopes, lowpassed."""
    fb = so.cosine_filterbank(n_bands, F_LO, F_HI)
    return fb.analyze(snd).envelopes(lowpass=env_lowpass, fs=1000)


def draw(axes, snd, envelopes=None, title=""):
    """Waveform, spectrogram and (if given) the band envelopes, down one column of axes."""
    snd.plot(axes[0], color="k", lw=0.4)
    axes[0].set_title(title)
    wide.analyze(snd).plot(axes[1], db_range=DB, colorbar=False, fmax=FMAX)
    axes[1].set_title("Spectrogram (Hann 5 ms)")
    if envelopes is not None and envelopes.data.shape[1] == 3:  # one band (and the two dropped edges)
        axes[2].plot(envelopes.t, envelopes.data[:, 1, 0], color="k", lw=0.6)
        axes[2].set(title="The one envelope the vocoder keeps", ylabel="Envelope")
    elif envelopes is not None:
        envelopes.plot(axes[2], db_range=DB, colorbar=False, fscale="linear", fmax=FMAX)
        axes[2].set_title(f"The {envelopes.data.shape[1] - 2} band envelopes the vocoder keeps")
    for ax in axes:
        ax.set_xlim(0, snd.duration)
        ax.set_xlabel("")
    axes[-1].set_xlabel("Time [s]")


def show(snd, envelopes=None, title=""):
    """One sound: waveform, spectrogram and (if given) band envelopes on one time axis.
    Returns the figure and the panels the playhead follows."""
    n = 3 if envelopes is not None else 2
    fig = plt.figure(figsize=(10, 2.0 + 2.2 * (n - 1)), layout="constrained")
    axes = fig.subplots(n, 1, sharex=True, height_ratios=[0.55] + [1] * (n - 1))
    draw(axes, snd, envelopes, f"Waveform{': ' + title if title else ''}")
    return fig, list(axes)


def show_pair(sounds, n_bands=None, env_lowpass=50.0, title=""):
    """One column per talker, as `show`; the band envelopes are drawn from each talker's
    original recording. Returns the figure and, for each talker, the panels the playhead follows."""
    n = 2 if n_bands is None else 3
    fig = plt.figure(figsize=(10, 1.6 + 2.0 * (n - 1)), layout="constrained")
    columns = fig.subfigures(1, 2)
    playhead = {}
    for column, (label, snd) in zip(columns, sounds.items(), strict=True):
        axes = column.subplots(n, 1, sharex=True, height_ratios=[0.55] + [1] * (n - 1))
        envelopes = None if n_bands is None else bands(talkers[label], n_bands, env_lowpass)
        draw(axes, snd, envelopes, f"{label}{': ' + title if title else ''}")
        playhead[label] = list(axes)
    return fig, playhead


def vocode_both(n_bands, **options):
    """Both talkers through the same vocoder."""
    return {
        label: finish(so.channel_vocode(snd, n_bands, F_LO, F_HI, **options))
        for label, snd in talkers.items()
    }


# %% [markdown]
# ## How many bands
#
# Each band gives the listener one envelope: when there is energy in that region of the spectrum,
# but not what it is made of. Fewer bands mean a coarser picture of the spectrum, so formants
# that fall in the same band can no longer be told apart.

# %% [about]
# The two readings, for comparison.

# %% [demo ci0] The original sentence
sounds = talkers
fig, playhead = show_pair(sounds, title="original")

# %% [about]
# One band: the envelope of the whole sentence on a single noise. The rhythm and syllables are
# there; the words are not.

# %% [demo ci1] One band
sounds = vocode_both(1, rng=1)
fig, playhead = show_pair(sounds, 1, title="one band")

# %% [about]
# Four bands, the most that Shannon et al. (1995) used; with three or four, their listeners
# recognized most of the words in simple sentences. The band envelopes are four broad blocks,
# switching on and off with the syllables.

# %% [demo ci4] Four bands
sounds = vocode_both(4, rng=4)
fig, playhead = show_pair(sounds, 4, title="four bands")

# %% [about]
# Eight bands, about as many as the implant users of Friesen et al. (2001) could make use of. The
# formant movements are now visible as energy moving from band to band.

# %% [demo ci8] Eight bands
sounds = vocode_both(8, rng=8)
fig, playhead = show_pair(sounds, 8, title="eight bands")

# %% [about]
# Sixteen bands. The words are clearer still, but both voices are still whispers: no band of
# noise has a pitch, so the intonation is gone. What is left to tell the two talkers apart is
# mostly where their energy sits across the bands, which follows their
# [spectral envelopes](talkers.html#h-spectral-envelopes), and how fast each reads.

# %% [demo ci16] Sixteen bands
sounds = vocode_both(16, rng=16)
fig, playhead = show_pair(sounds, 16, title="sixteen bands")

# %% [markdown]
# ## Noise or tones
#
# The carrier is the vocoder's stand-in for the electrode, and noise is only one choice. A tone
# vocoder puts a sinusoid at each band's center instead. The envelopes are the same, but a tone
# has no envelope fluctuations of its own to blur them, so fast modulations survive better (the
# same blur shows in [a drawn modulation put on noise](modtargets.html#d-mt4)), and
# the result sounds more like a buzzy voice than a whisper. With few bands, the tones are heard
# as separate pitches. Speech through either is about equally intelligible (Dorman et al., 1997).
#
# A carrier can also be another sound, whose fine structure then carries the envelopes.
# [Hearing a modulation spectrum](modtargets.html#d-mt5) does the reverse of this page: it keeps
# the sentence's fine structure and gives it new envelopes.

# %% [about]
# The sentence through eight tones, at the same eight band centers as the noise vocoder above.

# %% [demo ct8] Eight tones
sounds = vocode_both(8, carrier="tone")
fig, playhead = show_pair(sounds, 8, title="eight tones")

# %% [markdown]
# ## Pitch from the envelope
#
# The vocoder's envelope lowpass decides how fast an envelope may change. At 50 Hz it keeps the
# syllables and the formant movements but not the voice's periodicity. Raise it above the
# fundamental and each band's envelope pulses once per glottal period: a *temporal* pitch cue,
# the kind implant users rely on. These pulses are the vertical striations of a wideband
# spectrogram ([Seeing speech](speech.html#d-27)).
#
# Whether a talker's pulses get through depends on where the pitch sits against the cutoff. The
# gain of the envelope lowpass (a fourth-order Butterworth run forward and backward, so
# $1 / (1 + (f/f_c)^8)$ in amplitude) at the median and at the 95th percentile of each talker's
# pitch track (see [Two talkers](talkers.html#h-pitch)):


# %%
def lowpass_gain_db(f, cutoff):
    """Amplitude gain [dB] of the envelope lowpass at f [Hz]."""
    return -20 * np.log10(1 + (f / cutoff) ** 8)


pitch = {}
for label, track in tracks.items():
    voiced = track.f0[0][track.voiced[0]]
    pitch[label] = {"median": np.median(voiced), "95th percentile": np.percentile(voiced, 95)}
    for name, f0 in pitch[label].items():
        gains = ", ".join(f"{lowpass_gain_db(f0, cutoff):6.1f} dB at {cutoff} Hz" for cutoff in (50, 300))
        print(f"{label}, {name:>15}: {f0:3.0f} Hz, lowpass gain {gains}")

# %% [markdown]
# With 50 Hz envelopes, the pulses of both voices are cut by at least
# {{ f"{np.floor(-max(lowpass_gain_db(p['median'], 50) for p in pitch.values())):.0f}" }} dB at the
# median pitch, so the noise-vocoded sentences above carry no voice pitch. With 300 Hz envelopes,
# the pulses of both pass almost untouched: the female talker's, faster and closer to the cutoff,
# lose only {{ f"{-lowpass_gain_db(pitch['Female talker']['95th percentile'], 300):.1f}" }} dB near
# the top of its range. A voice pitched above the cutoff would lose them.

# %% [about]
# One band's envelope over the first voiced stretch of each reading, lowpassed at 50 Hz and at
# 300 Hz. Only the 300 Hz envelope follows the glottal pulses, which come faster in the female
# talker's reading.

# %% [figure cp0] One envelope, two cutoffs
fb = so.cosine_filterbank(8, F_LO, F_HI)
band = int(np.argmin(np.abs(fb.cfs - 700)))  # the band nearest 700 Hz
STRETCH = 0.15  # seconds shown
fig, axes = plt.subplots(1, 2, figsize=(10, 3.0), layout="constrained")
for ax, (label, snd) in zip(axes, talkers.items(), strict=True):
    track = tracks[label]
    # the start of the first voiced stretch at least STRETCH long
    voiced = np.r_[False, track.voiced[0], False].astype(int)
    starts, ends = np.flatnonzero(np.diff(voiced) == 1), np.flatnonzero(np.diff(voiced) == -1)
    start = next(
        track.t[a] for a, b in zip(starts, ends, strict=True) if track.t[b - 1] - track.t[a] >= STRETCH
    )
    sub = fb.analyze(snd)
    for cutoff, color in ((50, "tab:blue"), (300, "tab:red")):
        env = sub.envelopes(lowpass=cutoff)
        ax.plot(env.t, env.data[:, band, 0], color=color, lw=1, label=f"lowpass {cutoff} Hz")
    ax.set(
        xlim=(start, start + STRETCH),
        xlabel="Time [s]",
        ylabel="Envelope",
        title=f"{label}: the band centered at {fb.cfs[band]:.0f} Hz",
    )
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(ls=":")

# %% [about]
# Eight bands with envelopes lowpassed at 300 Hz instead of 50 Hz. Listen for the intonation of
# each talker, faint but back. The wideband spectrograms show the pulses as vertical striations
# again, closer together for the female talker; zoom in to see single ones.

# %% [demo cp8] Eight bands, 300 Hz envelopes
sounds = vocode_both(8, env_lowpass=300, rng=8)
fig, playhead = show_pair(sounds, 8, env_lowpass=300, title="eight bands, 300 Hz envelopes")

# %% [markdown]
# ## A melody
#
# Speech can be understood without pitch; a melody cannot. The opening of "Twinkle, Twinkle,
# Little Star" in harmonic complex tones, low in the bass clef so that the fundamentals are in
# reach of a temporal cue.

# %%
notes = [131, 131, 196, 196, 220, 220, 196]  # C3 C3 G3 G3 A3 A3 G3 [Hz]
durs = [0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.8]
tones = [
    so.harmonic_complex(d, fs, f, np.arange(1, 21)).ramp(20e-3) for f, d in zip(notes, durs, strict=True)
]
melody = finish(so.concat(tones))

# %% [about]
# The melody as played.

# %% [demo cm0] The melody
fig, playhead = show(melody, title="C3 C3 G3 G3 A3 A3 G3")
sound = melody

# %% [about]
# Through eight noise bands with 50 Hz envelopes. Each note changes which bands are loud only a
# little, and the tune is close to gone.

# %% [demo cm8] Eight bands, 50 Hz envelopes
vocoded = finish(so.channel_vocode(melody, 8, F_LO, F_HI, rng=8))
fig, playhead = show(vocoded, bands(melody, 8), "eight bands, 50 Hz envelopes")
sound = vocoded

# %% [about]
# The same with 300 Hz envelopes: every band now pulses at the note's fundamental, and the tune
# comes back as a rough, buzzy pitch.

# %% [demo cm8p] Eight bands, 300 Hz envelopes
vocoded = finish(so.channel_vocode(melody, 8, F_LO, F_HI, env_lowpass=300, rng=8))
fig, playhead = show(vocoded, bands(melody, 8, env_lowpass=300), "eight bands, 300 Hz envelopes")
sound = vocoded

# %% [markdown]
# ## What this simulation leaves out
#
# A vocoder shows what is lost when only band envelopes are kept. It is not a model of any one
# implant user, and several things that matter for real implants are not on this page:
#
# - **Current spread.** Neighboring electrodes excite overlapping nerve fibers, so the useful
#   number of channels is smaller than the number of electrodes. Vocoders model it with wider,
#   shallower synthesis filters than analysis filters; here both are the same.
# - **Insertion depth.** An electrode array does not reach the apex of the cochlea, so each band
#   is delivered to a place tuned higher than the band itself, a frequency shift. Here every
#   carrier band sits where its envelope came from.
# - **Dynamic range and pulses.** Electric hearing has a much narrower dynamic range, so the
#   processor compresses the envelopes, and it delivers them as pulse trains at a fixed rate.

# %% [markdown]
# ## References
#
# - Dorman, Loizou & Rainey (1997). Speech intelligibility as a function of the number of channels
#   of stimulation for signal processors using sine-wave and noise-band outputs. *J. Acoust. Soc.
#   Am.* 102(4), 2403–2411. [doi:10.1121/1.420354](https://doi.org/10.1121/1.420354).
#   [`envelopes.channel_vocode`](https://github.com/choyun1/sonore/blob/main/src/sonore/views/envelopes.py#L492)
# - Friesen, Shannon, Baskent & Wang (2001). Speech recognition in noise as a function of the number
#   of spectral channels: comparison of acoustic hearing and cochlear implants. *J. Acoust. Soc.
#   Am.* 110(2), 1150–1163. [PubMed](https://pubmed.ncbi.nlm.nih.gov/11519582/).
#   [`envelopes.channel_vocode`](https://github.com/choyun1/sonore/blob/main/src/sonore/views/envelopes.py#L492)
# - Kominek & Black (2004). The CMU Arctic speech databases. *Proc. 5th ISCA Speech Synthesis
#   Workshop*, 223–224. [ISCA Archive](https://www.isca-archive.org/ssw_2004/kominek04b_ssw.html).
#   The sentence, by speakers bdl and slt.
# - Shannon, Zeng, Kamath, Wygonski & Ekelid (1995). Speech recognition with primarily temporal
#   cues. *Science* 270(5234), 303–304.
#   [doi:10.1126/science.270.5234.303](https://doi.org/10.1126/science.270.5234.303).
#   [`envelopes.channel_vocode`](https://github.com/choyun1/sonore/blob/main/src/sonore/views/envelopes.py#L492)
# - Wilson, Finley, Lawson, Wolford, Eddington & Rabinowitz (1991). Better speech recognition with
#   cochlear implants. *Nature* 352, 236–238. [PubMed](https://pubmed.ncbi.nlm.nih.gov/1857418/).
#   [`envelopes.channel_vocode`](https://github.com/choyun1/sonore/blob/main/src/sonore/views/envelopes.py#L492)
