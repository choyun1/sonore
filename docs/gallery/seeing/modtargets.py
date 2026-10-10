"""Hearing a modulation spectrum: sounds made from a modulation spectrum, measured, edited or drawn.

This script is the gallery page https://choyun1.github.io/sonore/gallery/modtargets.html:
docs/gallery/build.py runs it cell by cell from the repository root and shows each
cell's code beside what it made. Run it yourself from the repository root,

    python docs/gallery/seeing/modtargets.py

or a cell at a time ("# %%" starts a cell in VS Code, Spyder and Jupytext).
"""

# %% [markdown]
# # Hearing a modulation spectrum
#
# A modulation spectrum says how much of a sound's envelope pattern moves at each rate (Hz) and
# each density (cycles per octave). [Spectrotemporal ripples](ripples.html#h-how-a-ripple-is-made)
# defines it and goes from a pattern to its spectrum. This page goes the other way: from a
# spectrum, measured and edited or drawn from scratch, to a sound that has it.
#
# A magnitude is not enough to make a sound. The spectrum has dropped the phase of the
# modulations, which says when each event happens, and the fine structure under each band's
# envelope. `ModulationSpectrum.to_sound(carrier=...)` takes both from a *carrier*: a sound lends
# its own, while `"tones"` and `"noise"` draw a random modulation phase and put the envelopes on
# steady tones or on noise.
#
# The sentence is read by the [two talkers](talkers.html) every speech page uses, and every demo
# on it plays both.
#
# - [Same spectrum, different sound](#h-same-spectrum-different-sound): the sentence, and a sound
#   with its modulation spectrum and a random modulation phase.
# - [A drawn spectrum](#h-a-drawn-spectrum): one patch of modulation, heard on three carriers.
# - [Random spectrograms](#h-random-spectrograms): noise shaped by a spectrogram drawn at
#   random with the coarse correlations of natural sounds.
# - [An edited sentence](#h-an-edited-sentence): the sentence with every modulation faster than
#   4 Hz removed, and how close the sound comes to that.
# - [Timing from one sound, magnitudes from another](#h-timing-from-one-sound-magnitudes-from-another):
#   the sentence and rain trade halves.
# - [Twins band by band](#h-twins-band-by-band): crickets and a fire with their timing
#   randomized, keeping each band's own modulation.
# - [What this page leaves out](#h-what-this-page-leaves-out): drawing with a mouse, dB targets,
#   and the other modulation spectra.

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
FS = 44100
SPEAKERS = {"Male talker": "bdl", "Female talker": "slt"}
talkers = {
    label: finish(so.load(fetch(f"docs/speech/{speaker}_arctic_a0131.flac")))
    for label, speaker in SPEAKERS.items()
}


def measure(snd, f_lo):
    """The modulation spectrum on the page's grid: 12 bands per octave from f_lo to 8 kHz."""
    return so.ModulationSpectrum.octave(snd, f_lo=f_lo, f_hi=8000)


bank = so.cosine_filterbank(
    f_lo=125, f_hi=8000, spacing=1 / 12, scale="octave"
)  # what measure(snd, 125) uses


def cochleagram(ax, snd, f_lo):
    """The sound's envelopes in 24 bands per octave."""
    bank = so.cosine_filterbank(f_lo=f_lo, f_hi=8000, spacing=1 / 24, scale="octave")
    bank.analyze(snd).envelopes(lowpass=200, fs=1000).plot(ax, db_range=30, colorbar=False)
    ax.set_title("Envelopes (cochleagram)")


def spectra(axes, snd, target, f_lo):
    """The target spectrum, if there is one, and the spectrum measured from the sound."""
    if target is not None:
        target.plot(axes[0], db_range=30, wt_max=20, wf_max=4, colorbar=False)
        axes[0].set_title("Target modulation spectrum")
    measure(snd, f_lo).plot(axes[-1], db_range=30, wt_max=20, wf_max=4, colorbar=False)
    axes[-1].set_title("Measured from the sound")


def show(snd, target=None, f_lo=250):
    """The sound's cochleagram, the target spectrum if there is one, and the spectrum measured
    from the sound. Returns the figure and the panel the playhead follows."""
    n_panels = 2 if target is None else 3
    fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 3.4), layout="constrained")
    cochleagram(axes[0], snd, f_lo)
    spectra(axes[1:], snd, target, f_lo)
    return fig, [axes[0]]


def show_pair(sounds, targets=None, f_lo=125):
    """One column per talker: the cochleagram above, and below it the talker's target spectrum
    (if there is one) beside the spectrum measured from the sound. Returns the figure and, for
    each talker, the panel the playhead follows."""
    fig = plt.figure(figsize=(10, 6.4), layout="constrained")
    columns = fig.subfigures(1, 2)
    playhead = {}
    for column, (label, snd) in zip(columns, sounds.items(), strict=True):
        target = None if targets is None else targets[label]
        layout = [["top"], ["measured"]] if target is None else [["top"] * 2, ["target", "measured"]]
        top, *bottom = column.subplot_mosaic(layout).values()
        cochleagram(top, snd, f_lo)
        top.set_title(f"{label}: envelopes (cochleagram)")
        spectra(bottom, snd, target, f_lo)
        playhead[label] = [top]
    return fig, playhead


# %% [markdown]
# ## Same spectrum, different sound
#
# Each talker's modulation spectrum, measured in 12 bands per octave from 125 Hz to 8 kHz, then
# heard with the sentence's own modulation phase thrown away. `to_sound` with `carrier="tones"`
# draws a random phase, builds the envelopes from it and the stored magnitudes, and puts each one
# on a steady tone at its band's center. The random phase leaves the zero-rate column alone,
# since that column (its phase as well as its magnitudes) is the sentence's long-term spectrum:
# which bands are loud.

# %% [about]
# The sentence, read by each talker. Its modulation spectrum is brightest at low rates and
# densities, as for most natural sounds (Singh & Theunissen, 2003); how that modulation changes
# over the sentence is on [Modulation spectrogram](modspectrogram.html#d-s1). The
# [speech-shaped noise](classic.html#d-k1) on Synthetic sounds has the long-term spectrum of speech
# but none of this: its modulation spectrum is a single line at zero rate. In the cochleagrams
# the lowest harmonics show as separate lines, farther apart for the female talker
# ([Two talkers](talkers.html#h-harmonics-sample-the-envelope)); at the slow rates shown here the
# two modulation spectra are much alike.

# %% [demo mt1] A sentence
spectra_of = {label: measure(snd, f_lo=125) for label, snd in talkers.items()}
sounds = talkers
fig, playhead = show_pair(sounds)

# %% [about]
# The same magnitudes with a random modulation phase. The syllables, the pauses and the onsets
# are gone from the envelopes, which change at the sentence's rates and densities but never at
# its moments. A random phase also asks for envelopes below zero, which no envelope can be;
# `to_sound` clips them at zero, with a warning, and that is why the measured spectrum is
# smoother than the target. The long-term spectrum is kept only roughly. A two-dimensional
# modulation spectrum does not say which bands carry which modulation, so the random phase
# spreads the sentence's modulation into bands that were quiet, and clipping turns it into level
# there.

# %% [demo mt2] Its modulation spectrum, random modulation phase
sounds = {
    label: finish(spectra_of[label].to_sound(carrier="tones", fs=snd.fs, rng=0))
    for label, snd in talkers.items()
}
fig, playhead = show_pair(sounds, targets=spectra_of)

# %% [markdown]
# The same draw as envelopes, against the sentence's own envelopes: how many values were
# clipped, how closely the two arrays correlate, and how often the summed envelope is more than
# 30 dB below its peak, as it is in the pauses.


# %%
def summed_quiet(envelopes):
    """Share of the time the envelope summed over bands is more than 30 dB below its peak."""
    summed = envelopes.sum(axis=1)
    return np.mean(summed < summed.max() * 10 ** (-30 / 20))


for label, snd in talkers.items():
    own = bank.analyze(snd).envelopes(fs=1000).data[:, 1:-1, 0]  # (time, band), edge bands dropped
    twin = spectra_of[label].to_envelopes(rng=0).data[:, 1:-1, 0]  # the draw mt2 heard; clipped values are 0
    clipped, correlation = np.mean(twin == 0), np.corrcoef(own.ravel(), twin.ravel())[0, 1]
    print(
        f"{label}: {clipped:.0%} clipped, correlation {correlation:.2f}, "
        f"quiet {summed_quiet(own):.1%} of the time in the sentence, {summed_quiet(twin):.1%} in the twin"
    )

# %% [markdown]
# So a modulation spectrum fixes how a sound changes, not when. The two envelope arrays barely
# correlate, and the twin has no pauses. Sounds made this way, with a natural sound's modulation
# spectrum and a random modulation phase, have been used to ask what auditory neurons respond
# to (Hsu et al., 2004).

# %% [markdown]
# ## A drawn spectrum
#
# A target can be drawn from scratch as a few patches on the rate × density plane.
# `so.ModulationBlob(rate, density)` is a Gaussian bump at that rate [Hz] and density
# [cycles/octave], with a standard deviation of half an octave in rate and a quarter of a cycle
# per octave in density by default. The signs follow the
# [ripples](ripples.html#h-how-a-ripple-is-made): a positive rate and density is a downward
# sweep. A ripple is a blob of no width.
#
# A drawing sets a shape, not a depth, so `from_blobs` takes the depth separately: `rms_depth`,
# the rms of the envelopes about their mean, relative to the mean (0.2 by default). Envelopes
# cannot go below zero, and a random phase reaches only a limited depth before they would: a
# random-phase field is close to Gaussian, whose peaks lie several standard deviations from its
# mean, while a single ripple's peak is only $\sqrt{2}$ of them away. A drawn target that would
# need clipping is refused, with the largest depth that fits.

# %% [about]
# Downward sweeps around 4 Hz and half a cycle per octave, on steady tones at the band centers.
# Unlike a ripple, the sweeps come at irregular moments and with varying slopes, since the blob
# spreads over a range of rates and densities and the phase is random. The faint patch at
# negative rates along the bottom is the tail of the blob's mirror image at (−4 Hz, −0.5
# cycle/octave), which every real pattern has.

# %% [demo mt3] A drawn blob, on tones
target = so.ModulationSpectrum.from_blobs(so.ModulationBlob(4, 0.5), duration=3.0)
on_tones = target.to_sound(carrier="tones", fs=FS, rng=0)
sound = finish(on_tones)
fig, playhead = show(sound, target=target)

# %% [about]
# The same target on noise: each envelope multiplies a band of noise instead of a tone. The
# sweeps are still there, under a hiss whose own random fluctuations fill the whole modulation
# spectrum.

# %% [demo mt4] The same blob, on noise
on_noise = target.to_sound(carrier="noise", fs=FS, rng=0)
sound = finish(on_noise)
fig, playhead = show(sound, target=target)

# %% [about]
# The same target on each talker's sentence, drawn for as long as that sentence and a little
# shallower (`rms_depth=0.15`): the default depth would push some of the female talker's
# envelope values below zero with this phase, and is refused. A sound as carrier lends its own
# modulation phase as well as its fine structure, so the sweeps now come when the sentence's
# syllables did, in each talker's voice: the sentence keeps its fine structure and gets new
# envelopes. A [noise vocoder](vocoder.html#d-ci8) makes the opposite
# trade, keeping the envelopes and replacing the fine structure with noise.

# %% [demo mt5] The same blob, on the sentence
short_targets = {
    label: so.ModulationSpectrum.from_blobs(so.ModulationBlob(4, 0.5), duration=snd.duration, rms_depth=0.15)
    for label, snd in talkers.items()
}
on_sentence = {label: short_targets[label].to_sound(carrier=snd) for label, snd in talkers.items()}
sounds = {label: finish(snd) for label, snd in on_sentence.items()}
fig, playhead = show_pair(sounds, targets=short_targets, f_lo=250)

# %% [markdown]
# How much of each sound's measured modulation power lies where the target has its power (within
# 20 dB of its peak), leaving out the zero-rate column, which holds the long-term spectrum:


# %%
def share_in_target(snd, target):
    """Share of the measured modulation power, off the zero-rate column, inside the target."""
    measured = measure(snd, f_lo=250)
    measured_power = 10 ** (measured.level / 10)
    in_target = target.level > target.level.max() - 20
    moving = measured.w_t != 0
    return measured_power[in_target & moving].sum() / measured_power[:, moving].sum()


carriers = [("tones", on_tones, target), ("noise", on_noise, target)]
carriers += [(label.lower(), on_sentence[label], short_targets[label]) for label in talkers]
for name, snd, drawn in carriers:
    print(f"{name:>13}: {share_in_target(snd, drawn):.0%}")

# %% [markdown]
# The carrier matters as much as the target. A steady tone at each band's center adds almost no
# modulation of its own, so the target survives. Noise and speech fluctuate inside every band,
# and those fluctuations land in the measured spectrum on top of the drawn one; a tone vocoder
# keeps fast envelope modulations better than a noise vocoder for the same reason
# ([Noise or tones](vocoder.html#h-noise-or-tones)). That is why `"tones"` is the default
# carrier.

# %% [markdown]
# ## Random spectrograms
#
# To study how listeners pick out a sound they have never heard before, McDermott, Wrobleski &
# Oxenham (2011) needed sounds that were novel yet shared the coarse statistics of natural
# sources. They drew spectrograms, one level in dB per ERB band and 20 ms window, from a
# Gaussian whose correlations fall off exponentially in time and in frequency, as those of
# spoken words and animal calls roughly do, and imposed them on noise. That is a drawn
# modulation spectrum too: a random phase on a spectrum makes a Gaussian field, and exponential
# correlations make a spectrum peaked at zero rate and density, a single blob at the origin,
# drawn in dB. `so.gaussian_spectrogram` draws such a spectrogram as `Envelopes`, and `to_sound`
# puts it on the fine structure of a noise. The defaults are the paper's: 39 bands from 20 Hz to
# 4 kHz and correlation lengths of about 8.8 ERB and 154 ms. The paper gives no spread for the
# levels; the default standard deviation of 14.1 dB is carried over from an earlier
# implementation, not from the paper. Each figure shows the spectrogram that was drawn above the
# envelopes measured on the sound.
#
# Drawing in dB is also what lets these go deep: any level in dB is a positive envelope, so
# nothing is clipped. Blobs can be drawn the same way, with
# `from_blobs(..., scale="db", sd_db=...)` in place of `rms_depth`.


# %%
def random_spectrogram_sound(seed, **kwargs):
    """A drawn spectrogram, and the sound made by putting it on a noise's fine structure."""
    env = so.gaussian_spectrogram(1.0, FS, rng=seed, **kwargs)
    sound = finish(env.to_sound(so.gaussian_noise(1.0, FS, rng=100 + seed)))
    return env, sound


def show_spectrograms(env, sound):
    fig, axes = plt.subplots(2, 1, figsize=(10, 5.2), sharex=True, layout="constrained")
    env.plot(axes[0])
    axes[0].set_title("Drawn spectrogram")
    env.filterbank.analyze(sound).envelopes().plot(axes[1])
    axes[1].set_title("Envelopes of the sound")
    return fig, list(axes)


# %% [about]
# The paper's settings. Each draw is a new sound; this one is seed 1.

# %% [demo rs1] A random spectrogram
env, sound = random_spectrogram_sound(1)
fig, playhead = show_spectrograms(env, sound)

# %% [about]
# Longer correlations, 500 ms in time and 20 ERB in frequency: slower, broader shapes.

# %% [demo rs2] Longer correlations
env, sound = random_spectrogram_sound(2, time_correlation=0.5, band_correlation_erb=20)
fig, playhead = show_spectrograms(env, sound)

# %% [about]
# Shorter correlations, 40 ms and 3 ERB: closer to noise with a lumpy spectrum.

# %% [demo rs3] Shorter correlations
env, sound = random_spectrogram_sound(3, time_correlation=0.04, band_correlation_erb=3)
fig, playhead = show_spectrograms(env, sound)

# %% [markdown]
# ## An edited sentence
#
# A measured spectrum can be edited and heard. `with_gain(g)` multiplies every cell by
# `g(rate, density)`; `lambda r, d: abs(r) <= 4` keeps only the modulations at 4 Hz and below and
# removes everything faster. Elliott & Theunissen (2009) filtered the modulation spectrum of
# speech's spectrogram in a similar way to find which modulations intelligibility needs. With the
# sentence itself as carrier, its own modulation phase and fine structure go back under the
# edited magnitudes, so wherever the gain is 1 the timing is kept.

# %% [about]
# The edit, put straight back on each sentence. The measured spectrum keeps some power above
# 4 Hz, less than the sentence had but more than the edit asks for: the sentence's fine
# structure brings fast modulation of its own back into every band.

# %% [demo mt6] Modulations above 4 Hz removed
slow = {
    label: spectrum.with_gain(lambda rate, density: np.abs(rate) <= 4)
    for label, spectrum in spectra_of.items()
}
edited = {label: slow[label].to_sound(carrier=snd) for label, snd in talkers.items()}
sounds = {label: finish(snd) for label, snd in edited.items()}
fig, playhead = show_pair(sounds, targets=slow)

# %% [about]
# The same edit with 20 rounds of a search in the manner of Griffin & Lim (1984): analyze the
# sound, keep its fine structure and modulation phase, impose the edited magnitudes again, and
# resynthesize. Each round brings the sound's own spectrum closer to the edit. This is the
# slowest cell on the page: each round is one analysis and one synthesis.

# %% [demo mt7] The same edit, after 20 iterations
searched = {label: slow[label].to_sound(carrier=snd, iterations=20) for label, snd in talkers.items()}
sounds = {label: finish(snd) for label, snd in searched.items()}
fig, playhead = show_pair(sounds, targets=slow)

# %% [about]
# The same edit on steady tones, with a random modulation phase: the slow modulations survive,
# but the sentence's timing does not.

# %% [demo mt8] The same edit, on tones
on_tones_edit = {
    label: slow[label].to_sound(carrier="tones", fs=snd.fs, rng=0) for label, snd in talkers.items()
}
sounds = {label: finish(snd) for label, snd in on_tones_edit.items()}
fig, playhead = show_pair(sounds, targets=slow)

# %% [markdown]
# The modulation power left between 6 and 40 Hz, relative to each talker's sentence:


# %%
def fast_share(snd):
    """Share of the measured modulation power at 6-40 Hz."""
    measured = measure(snd, f_lo=125)
    power = 10 ** (measured.level / 10)
    fast = (np.abs(measured.w_t) >= 6) & (np.abs(measured.w_t) <= 40)
    return power[:, fast].sum() / power.sum()


for label, snd in talkers.items():
    print(label)
    for name, version in [("edit", edited), ("20 iterations", searched), ("on tones", on_tones_edit)]:
        print(f"  {name:>13}: {10 * np.log10(fast_share(version[label]) / fast_share(snd)):.1f} dB")

# %% [markdown]
# None of them reaches the edit exactly, which removed everything there. A sound's envelopes are
# not free: they come from filtering one waveform, so most arrays of magnitudes belong to no
# sound at all, and the search finds a sound whose spectrum is close, never equal.

# %% [markdown]
# ## Timing from one sound, magnitudes from another
#
# A sound as carrier lends its modulation phase, so one sound's magnitudes can be heard with
# another sound's timing. `to_envelopes(carrier=...)` gives the envelopes alone, which can then go
# on either sound's fine structure. Here each talker's sentence and as much rain trade halves.

# %%
rain_long = so.load(fetch("docs/textures/rain.flac")).mono()
rains = {
    label: so.Sound(rain_long.resample(snd.fs).data[: snd.n_samples], snd.fs)
    for label, snd in talkers.items()
}
rain_spectra = {label: measure(rain, f_lo=125) for label, rain in rains.items()}

# %% [about]
# Rain's modulation magnitudes and fine structure, with each sentence's modulation phase. The
# rain now swells and fades with the syllables.

# %% [demo mt9] Rain magnitudes, sentence timing
sounds = {}
for label, snd in talkers.items():
    envelopes = rain_spectra[label].to_envelopes(carrier=snd)
    sounds[label] = finish((envelopes * bank.analyze(rains[label]).tfs()).to_sound())
fig, playhead = show_pair(sounds, targets=rain_spectra)

# %% [about]
# The other way round: each sentence's magnitudes and fine structure, with rain's modulation
# phase. The voice is still there, but its events come at rain's moments.

# %% [demo mt10] Sentence magnitudes, rain timing
sounds = {}
for label, snd in talkers.items():
    envelopes = spectra_of[label].to_envelopes(carrier=rains[label])
    sounds[label] = finish((envelopes * bank.analyze(snd).tfs()).to_sound())
fig, playhead = show_pair(sounds, targets=spectra_of)

# %% [markdown]
# Listen for which words can still be made out in each: the first keeps the sentence's timing
# and the second its modulation magnitudes.

# %% [markdown]
# ## Twins band by band
#
# A two-dimensional modulation spectrum says how much modulation there is at each rate and
# density, but not which bands carry it. A random phase therefore spreads modulation into bands
# that were steady or quiet, as in the sentence's twin above. A narrower description keeps it:
# each band's own modulation spectrum, close to the per-band modulation power that McDermott &
# Simoncelli (2011) use among their texture statistics. A twin of that keeps each band's
# envelope magnitudes over rate and randomizes only its phase in time, band by band. Both kinds
# of twin below keep each band's own fine structure, so only the envelopes differ.


# %%
def band_twin(snd, rng=0):
    """Each band keeps its envelope's magnitude spectrum over time (and so its level); the phase is
    random, band by band. On the sound's own fine structure."""
    subbands = bank.analyze(snd)
    envelopes = subbands.envelopes(fs=1000).data[:, :, 0]  # (time, band)
    magnitude = np.abs(np.fft.rfft(envelopes, axis=0))
    phase = np.random.default_rng(rng).uniform(0, 2 * np.pi, magnitude.shape)
    phase[0] = 0  # each band's mean stays
    rebuilt = np.fft.irfft(magnitude * np.exp(1j * phase), n=len(envelopes), axis=0)
    twin_envelopes = so.Envelopes(np.maximum(rebuilt, 0), 1000, bank)
    return (twin_envelopes * subbands.tfs()).to_sound()


def plane_twin(snd, rng=0):
    """The same modulation spectrum on the plane, a random phase, on the sound's own fine structure."""
    plane = measure(snd, f_lo=125)
    return (plane.to_envelopes(rng=rng) * bank.analyze(snd).tfs()).to_sound()


def texture(name, seconds=3.5):
    snd = so.load(fetch(f"docs/textures/{name}.flac")).mono()
    return so.Sound(snd.data[: int(seconds * snd.fs)], snd.fs)


# %% [about]
# 3.5 s of crickets, from the [sound textures](textures.html) page, whose modulation over time
# is on [Modulation spectrogram](modspectrogram.html#d-x1).

# %% [demo mt11] Crickets
crickets = texture("crickets")
sound = finish(crickets)
fig, playhead = show(sound, f_lo=125)

# %% [about]
# The crickets' twin on the plane. The long-term spectrum is kept, so the chirps stay at the
# top, but some of their modulation spreads into the bands below, which were nearly silent.

# %% [demo mt12] Crickets, twin on the plane
sound = finish(plane_twin(crickets))
fig, playhead = show(sound, f_lo=125)

# %% [about]
# The crickets' twin band by band. Each band keeps its own modulation, and the chirps stay at
# the top.

# %% [demo mt13] Crickets, twin band by band
sound = finish(band_twin(crickets))
fig, playhead = show(sound, f_lo=125)

# %% [about]
# 3.5 s of a fire: a low roar with sharp crackles.

# %% [demo mt14] Fire
fire = texture("fire")
sound = finish(fire)
fig, playhead = show(sound, f_lo=125)

# %% [about]
# The fire's twin band by band. Every band keeps its level and its modulation spectrum, but the
# crackles are gone: a crackle is a brief event in many bands at once, and that alignment lives
# in the phase.

# %% [demo mt15] Fire, twin band by band
sound = finish(band_twin(fire))
fig, playhead = show(sound, f_lo=125)

# %% [markdown]
# So a band-by-band twin keeps a steady texture such as crickets, but not the sparse events of
# a fire. McDermott & Simoncelli's statistics also include each band's envelope skew and
# kurtosis, which measure sparseness, and correlations between bands, which measure how events
# line up across them; [What the statistics do](textures.html#h-what-the-statistics-do) hears a
# fire synthesized with some of them left out.

# %% [markdown]
# ## What this page leaves out
#
# - **Drawing with a mouse.** Targets here are written in code, as blobs or as edits.
#   [sonore-sketch](https://choyun1.github.io/sonore-sketch/), a separate browser app built on sonore, draws
#   them with a mouse: blobs on the plane, or cuts on a sound's measured spectrum.
# - **dB targets.** A spectrum measured with `scale="db"` (of the log envelope, as Elliott &
#   Theunissen, 2009, define it), or blobs drawn with `scale="db"`, also goes back to sound and
#   never needs clipping, but the linear spectrum of the result is not the one drawn. Apart from
#   the random spectrograms above, none is heard here.
# - **The other modulation spectra.** A [modulation spectrogram](modspectrogram.html) changes
#   over time and has no route back to sound, and a [sound texture](textures.html)'s per-band
#   modulation power is set through its statistics, not here.

# %% [markdown]
# ## References
#
# - Elliott & Theunissen (2009). The modulation transfer function for speech intelligibility.
#   *PLoS Comput. Biol.* 5(3), e1000302.
#   [doi:10.1371/journal.pcbi.1000302](https://doi.org/10.1371/journal.pcbi.1000302).
#   [`modulation.ModulationSpectrum.with_gain`](https://github.com/choyun1/sonore/blob/main/src/sonore/views/modulation.py#L531)
# - Griffin & Lim (1984). Signal estimation from modified short-time Fourier transform. *IEEE
#   Trans. Acoust. Speech Signal Process.* 32(2), 236–243.
#   [doi:10.1109/TASSP.1984.1164317](https://doi.org/10.1109/TASSP.1984.1164317).
#   [`modulation.ModulationSpectrum.to_sound`](https://github.com/choyun1/sonore/blob/main/src/sonore/views/modulation.py#L633)
# - Hsu, Woolley, Fremouw & Theunissen (2004). Modulation power and phase spectrum of natural
#   sounds enhance neural encoding performed by single auditory neurons. *J. Neurosci.* 24(41),
#   9201–9211.
#   [doi:10.1523/JNEUROSCI.2449-04.2004](https://doi.org/10.1523/JNEUROSCI.2449-04.2004).
#   [`modulation.ModulationSpectrum.to_sound`](https://github.com/choyun1/sonore/blob/main/src/sonore/views/modulation.py#L633)
# - McDermott & Simoncelli (2011). Sound texture perception via statistics of the auditory
#   periphery. *Neuron* 71(5), 926–940.
#   [doi:10.1016/j.neuron.2011.06.032](https://doi.org/10.1016/j.neuron.2011.06.032).
#   [`texture`](https://github.com/choyun1/sonore/blob/main/src/sonore/texture/stats.py)
# - McDermott, Wrobleski & Oxenham (2011). Recovering sound sources from embedded repetition.
#   *PNAS* 108(3), 1188–1193. [doi:10.1073/pnas.1004765108](https://doi.org/10.1073/pnas.1004765108).
#   [`gaussian_spectrogram.gaussian_spectrogram`](https://github.com/choyun1/sonore/blob/main/src/sonore/sources/gaussian_spectrogram.py#L42)
# - Singh & Theunissen (2003). Modulation spectra of natural sounds and ethological theories of
#   auditory processing. *J. Acoust. Soc. Am.* 114(6), 3394–3411.
#   [doi:10.1121/1.1624067](https://doi.org/10.1121/1.1624067).
#   [`modulation.ModulationSpectrum`](https://github.com/choyun1/sonore/blob/main/src/sonore/views/modulation.py#L333)
