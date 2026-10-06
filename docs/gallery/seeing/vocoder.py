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
# Everything on this page is one function, `so.noise_vocode`, applied to the sentence from
# [Seeing speech](speech.html) and to a short melody:
#
# - [How many bands](#h-how-many-bands): from a coarse picture of the spectrum to a fine one.
# - [Noise or tones](#h-noise-or-tones): what carries the envelopes.
# - [Pitch from the envelope](#h-pitch-from-the-envelope): the weak temporal pitch cue an implant
#   leaves.
# - [A higher voice](#h-a-higher-voice): the sentence read by a female talker, through the same
#   vocoders.
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
# - **Few bands are enough for speech in quiet.** Shannon et al. (1995) found that listeners
#   understood most of the words in sentences through four noise bands, with no fine structure
#   at all.
# - **Implant users get only a handful.** Implants have 12 to 22 electrodes, but current spreads
#   between neighbors. Friesen et al. (2001) found that most implant users did no better with
#   more than about eight channels, while listeners hearing a vocoder kept improving up to about
#   twenty, especially in noise.
# - **Pitch is what suffers most.** With the fine structure gone, the only pitch cues left are
#   which bands are loud and how fast the envelopes fluctuate. Music, intonation, and telling one
#   voice from another all depend on pitch, and are hard for implant users.

# %% [markdown]
# ## The sentence, and code the examples share
#
# The vocoder below splits the sound with half-cosine filters equally spaced on the ERB scale
# between 80 Hz and 7.6 kHz (just under the 8 kHz Nyquist frequency of this 16 kHz recording),
# neighbors overlapping by half, takes each band's Hilbert
# envelope, lowpasses it at 50 Hz unless stated, and multiplies it into the same band of fresh
# Gaussian noise. The output has the input's RMS.

# %%
import matplotlib.pyplot as plt
import numpy as np

import sonore as so

plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "figure.dpi": 100})


def finish(snd):
    """How every sound in the gallery is played: 5 ms ramps, RMS 0.1, peak at most 0.95."""
    snd = snd.ramp(5e-3).normalize(rms=0.1)
    return snd.normalize(peak=0.95) if snd.peak > 0.95 else snd


# The sentence at its native 16 kHz. Sources: docs/speech/SOURCES.md.
sentence = finish(so.load("docs/speech/bdl_arctic_a0131.flac"))
fs = sentence.fs
F_LO, F_HI = 80, 7600

# Every spectrogram: a wideband STFT (Hann 5 ms), dB re its own maximum over 60 dB, 0 to 8 kHz.
wide = so.GaborFrame(0.005, 0.001, n_fft=1024)
FMAX, DB = 8000, 60


def bands(snd, n_bands, env_lowpass=50.0):
    """The envelopes the vocoder keeps: ERB bands, Hilbert envelopes, lowpassed."""
    fb = so.cosine_filterbank(n_bands, F_LO, F_HI)
    return fb.analyze(snd).envelopes(lowpass=env_lowpass, fs=1000)


def show(snd, envelopes=None, title=""):
    """Waveform, spectrogram and (if given) the band envelopes, on one time axis.
    Returns the figure and the panels the playhead follows."""
    n = 3 if envelopes is not None else 2
    fig = plt.figure(figsize=(10, 2.0 + 2.2 * (n - 1)), layout="constrained")
    axes = fig.subplots(n, 1, sharex=True, height_ratios=[0.55] + [1] * (n - 1))
    snd.plot(axes[0], color="k", lw=0.4)
    axes[0].set_title(f"Waveform{': ' + title if title else ''}")
    wide.analyze(snd).plot(axes[1], db_range=DB, colorbar=False, fmax=FMAX)
    axes[1].set_title("Spectrogram (Hann 5 ms)")
    if envelopes is not None:
        envelopes.plot(axes[2], db_range=DB, colorbar=False, fscale="linear", fmax=FMAX)
        axes[2].set_title(f"The {envelopes.data.shape[1] - 2} band envelopes the vocoder keeps")
    for ax in axes:
        ax.set_xlim(0, snd.duration)
        ax.set_xlabel("")
    axes[-1].set_xlabel("Time [s]")
    return fig, list(axes)


# %% [markdown]
# ## How many bands
#
# Each band gives the listener one envelope: when there is energy in that region of the spectrum,
# but not what it is made of. Fewer bands mean a coarser picture of the spectrum, so formants
# that fall in the same band can no longer be told apart.

# %% [about]
# The original, for comparison.

# %% [demo ci0] The original sentence
fig, playhead = show(sentence)
sound = sentence

# %% [about]
# One band: the envelope of the whole sentence on a single noise. The rhythm and syllables are
# there; the words are not.

# %% [demo ci1] One band
vocoded = finish(so.noise_vocode(sentence, 1, F_LO, F_HI, rng=1))
fig, playhead = show(vocoded, bands(sentence, 1), "one band")
sound = vocoded

# %% [about]
# Four bands, about where Shannon et al. (1995) found listeners began to understand most words in
# quiet. The spectrogram shows four broad blocks, switching on and off with the syllables.

# %% [demo ci4] Four bands
vocoded = finish(so.noise_vocode(sentence, 4, F_LO, F_HI, rng=4))
fig, playhead = show(vocoded, bands(sentence, 4), "four bands")
sound = vocoded

# %% [about]
# Eight bands, about as many as most implant users can use. The formant movements are now
# visible as energy moving from band to band.

# %% [demo ci8] Eight bands
vocoded = finish(so.noise_vocode(sentence, 8, F_LO, F_HI, rng=8))
fig, playhead = show(vocoded, bands(sentence, 8), "eight bands")
sound = vocoded

# %% [about]
# Sixteen bands. The words are easy, but the voice is still a whisper: no band of noise has a
# pitch, so the talker's intonation is gone.

# %% [demo ci16] Sixteen bands
vocoded = finish(so.noise_vocode(sentence, 16, F_LO, F_HI, rng=16))
fig, playhead = show(vocoded, bands(sentence, 16), "sixteen bands")
sound = vocoded

# %% [markdown]
# ## Noise or tones
#
# The carrier is the vocoder's stand-in for the electrode, and noise is only one choice. A tone
# vocoder puts a sinusoid at each band's center instead. The envelopes are the same, but a tone
# has no envelope fluctuations of its own to blur them, so fast modulations survive better, and
# the result sounds more like a buzzy voice than a whisper. With few bands, the tones are heard
# as separate pitches. Speech through either is about equally intelligible (Dorman et al., 1997).

# %% [about]
# The sentence through eight tones, at the same eight band centers as the noise vocoder above.

# %% [demo ct8] Eight tones
vocoded = finish(so.noise_vocode(sentence, 8, F_LO, F_HI, carrier="tone"))
fig, playhead = show(vocoded, bands(sentence, 8), "eight tones")
sound = vocoded

# %% [markdown]
# ## Pitch from the envelope
#
# The vocoder's envelope lowpass decides how fast an envelope may change. At 50 Hz it keeps the
# syllables and the formant movements but not the voice's periodicity. Raise it above the
# fundamental, here about 90 to 190 Hz, and each band's envelope pulses once per glottal
# period: a *temporal* pitch cue, the kind implant users rely on. It is weak and works only for
# low pitches, a few hundred hertz at most.

# %% [about]
# One band's envelope over a stretch of voiced speech, lowpassed at 50 Hz and at 300 Hz. Only
# the 300 Hz envelope follows the glottal pulses.

# %% [figure cp0] One envelope, two cutoffs
fb = so.cosine_filterbank(8, F_LO, F_HI)
sub = fb.analyze(sentence)
band = int(np.argmin(np.abs(fb.cfs - 700)))  # the band nearest 700 Hz
fig, ax = plt.subplots(figsize=(10, 3.0), layout="constrained")
for cutoff, color in ((50, "tab:blue"), (300, "tab:red")):
    env = sub.envelopes(lowpass=cutoff)
    ax.plot(env.t, env.data[:, band, 0], color=color, lw=1, label=f"lowpass {cutoff} Hz")
ax.set(
    xlim=(0.1, 0.3),
    xlabel="Time [s]",
    ylabel="Envelope",
    title=f"The band centered at {fb.cfs[band]:.0f} Hz",
)
ax.legend(loc="upper right", fontsize=8)
ax.grid(ls=":")

# %% [about]
# Eight bands with envelopes lowpassed at 300 Hz instead of 50 Hz. Listen for the intonation,
# faint but back. The wideband spectrogram shows the pulses as vertical striations again.

# %% [demo cp8] Eight bands, 300 Hz envelopes
vocoded = finish(so.noise_vocode(sentence, 8, F_LO, F_HI, env_lowpass=300, rng=8))
fig, playhead = show(vocoded, bands(sentence, 8, env_lowpass=300), "eight bands, 300 Hz envelopes")
sound = vocoded

# %% [markdown]
# ## A higher voice
#
# The same sentence read by a female talker (slt). The female fundamental, about 150 to 230 Hz here
# by `so.f0_track`, is higher than the male talker's, so a temporal cue has to follow faster pulses.

# %%
sentence_female = finish(so.load("docs/speech/slt_arctic_a0131.flac"))
track_female = so.f0_track(sentence_female)
print(f"female F0: {track_female}")

# %% [about]
# The female talker's sentence as recorded.

# %% [demo cf0] Female talker
fig, playhead = show(sentence_female)
sound = sentence_female

# %% [about]
# Through eight noise bands with 50 Hz envelopes. As with the male talker's sentence, the words come through
# and the intonation does not; with the pitch gone, the main difference left between the two
# vocoded voices is where their formants sit.

# %% [demo cf8] Female talker, eight bands
vocoded = finish(so.noise_vocode(sentence_female, 8, F_LO, F_HI, rng=8))
fig, playhead = show(vocoded, bands(sentence_female, 8), "eight bands")
sound = vocoded

# %% [about]
# Eight bands with 300 Hz envelopes. The female fundamental is still below the cutoff, so the
# envelopes pulse with it, as the male talker's do. Listen for the intonation, and compare it with
# the male talker's above.

# %% [demo cf8p] Female talker, eight bands, 300 Hz envelopes
vocoded = finish(so.noise_vocode(sentence_female, 8, F_LO, F_HI, env_lowpass=300, rng=8))
fig, playhead = show(vocoded, bands(sentence_female, 8, env_lowpass=300), "eight bands, 300 Hz envelopes")
sound = vocoded

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
vocoded = finish(so.noise_vocode(melody, 8, F_LO, F_HI, rng=8))
fig, playhead = show(vocoded, bands(melody, 8), "eight bands, 50 Hz envelopes")
sound = vocoded

# %% [about]
# The same with 300 Hz envelopes: every band now pulses at the note's fundamental, and the tune
# comes back as a rough, buzzy pitch.

# %% [demo cm8p] Eight bands, 300 Hz envelopes
vocoded = finish(so.noise_vocode(melody, 8, F_LO, F_HI, env_lowpass=300, rng=8))
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
#   [`envelopes.noise_vocode`](https://github.com/choyun1/sonore/blob/main/src/sonore/views/envelopes.py#L440)
# - Friesen, Shannon, Baskent & Wang (2001). Speech recognition in noise as a function of the number
#   of spectral channels: comparison of acoustic hearing and cochlear implants. *J. Acoust. Soc.
#   Am.* 110(2), 1150–1163. [PubMed](https://pubmed.ncbi.nlm.nih.gov/11519582/).
#   [`envelopes.noise_vocode`](https://github.com/choyun1/sonore/blob/main/src/sonore/views/envelopes.py#L440)
# - Kominek & Black (2004). The CMU Arctic speech databases. *Proc. 5th ISCA Speech Synthesis
#   Workshop*, 223–224. [ISCA Archive](https://www.isca-archive.org/ssw_2004/kominek04b_ssw.html).
#   The sentence, by speakers bdl and slt.
# - Shannon, Zeng, Kamath, Wygonski & Ekelid (1995). Speech recognition with primarily temporal
#   cues. *Science* 270(5234), 303–304.
#   [doi:10.1126/science.270.5234.303](https://doi.org/10.1126/science.270.5234.303).
#   [`envelopes.noise_vocode`](https://github.com/choyun1/sonore/blob/main/src/sonore/views/envelopes.py#L440)
# - Wilson, Finley, Lawson, Wolford, Eddington & Rabinowitz (1991). Better speech recognition with
#   cochlear implants. *Nature* 352, 236–238. [PubMed](https://pubmed.ncbi.nlm.nih.gov/1857418/).
#   [`envelopes.noise_vocode`](https://github.com/choyun1/sonore/blob/main/src/sonore/views/envelopes.py#L440)
