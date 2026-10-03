"""Drawing a modulation spectrum: sounds made to order from a target modulation spectrum.

This script is the gallery page https://choyun1.github.io/sonore/gallery/modtargets.html:
docs/gallery/build.py runs it cell by cell from the repository root and shows each
cell's code beside what it made. Run it yourself from the repository root,

    python docs/gallery/seeing/modtargets.py

or a cell at a time ("# %%" starts a cell in VS Code, Spyder and Jupytext).
"""

# %% [markdown]
# # Drawing a modulation spectrum
#
# A modulation spectrum is the two-dimensional Fourier transform of a sound's envelopes, band by
# band over time: how much of the pattern moves at each rate (Hz, across) and at each density
# (cycles per octave, up), keeping only the magnitude (Singh & Theunissen, 2003). The
# [ripples page](ripples.html) goes from a pattern to its spectrum. This page goes the other way:
# from a spectrum, measured and edited or drawn from scratch, to a sound that has it.
#
# A magnitude is not enough to make a sound. The spectrum has dropped the phase of the
# modulations, which says when each event happens, and the fine structure under each band's
# envelope. `ModulationSpectrum.to_sound(carrier=...)` takes both from a *carrier*, the way an
# envelope is put back on a carrier: a sound lends its own, while `"tones"` and `"noise"` draw a
# random modulation phase and put the envelopes on steady tones or on noise.
#
# - [Same spectrum, different sound](#h-same-spectrum-different-sound): a sentence and a sound
#   with the sentence's modulation spectrum and random modulation phase.
# - [A drawn spectrum](#h-a-drawn-spectrum): one patch of modulation, heard on three carriers.
# - [An edited sentence](#h-an-edited-sentence): a sentence with every modulation faster than
#   4 Hz removed, and how close the sound comes to that.
# - [What this page leaves out](#h-what-this-page-leaves-out): drawing with a mouse, dB targets,
#   and the other modulation spectra.

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


def measure(snd, f_lo):
    """The modulation spectrum on the page's grid: 12 bands per octave from f_lo to 8 kHz."""
    return so.ModulationSpectrum.octave(snd, f_lo=f_lo, f_hi=8000)


def show(snd, target=None, f_lo=250):
    """The sound's envelopes (a cochleagram), the target spectrum if there is one, and the
    spectrum measured from the sound. Returns the figure and the panel the playhead follows."""
    n_panels = 2 if target is None else 3
    fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 3.4), layout="constrained")
    bank = so.cosine_filterbank(f_lo=f_lo, f_hi=8000, spacing=1 / 24, scale="octave")
    bank.analyze(snd).envelopes(lowpass=200, fs=1000).plot(axes[0], db_range=30, colorbar=False)
    axes[0].set_title("Envelopes (cochleagram)")
    if target is not None:
        target.plot(axes[1], db_range=30, wt_max=20, wf_max=4, colorbar=False)
        axes[1].set_title("Target modulation spectrum")
    measure(snd, f_lo).plot(axes[-1], db_range=30, wt_max=20, wf_max=4, colorbar=False)
    axes[-1].set_title("Measured from the sound")
    return fig, [axes[0]]


# %% [markdown]
# ## Same spectrum, different sound
#
# A sentence's modulation spectrum, measured in 12 bands per octave from 125 Hz to 8 kHz, then
# heard with the sentence's own modulation phase thrown away. `to_sound` with `carrier="tones"`
# draws a random phase, builds the envelopes from it and the stored magnitudes, and puts each one
# on a steady tone at its band's centre. The random phase leaves the zero-rate column alone,
# since that column (its phase as well as its magnitudes) is the sentence's long-term spectrum:
# which bands are loud.

# %% [about]
# The sentence, a male talker reading one of the CMU ARCTIC prompts. Its modulation spectrum
# is brightest at low rates and densities, as for most natural sounds (Singh & Theunissen,
# 2003).

# %% [demo mt1] A sentence
sentence = so.load("docs/speech/bdl_arctic_a0131.flac")
spectrum = measure(sentence, f_lo=125)
sound = finish(sentence)
fig, playhead = show(sound, f_lo=125)

# %% [about]
# The same magnitudes with a random modulation phase. The syllables, the pauses and the onsets
# are gone from the envelopes, which change at the sentence's rates and densities but never at
# its moments. A random phase also asks for envelopes below zero, which no
# envelope can be: about a third of the values (34% here) are clipped at zero, with a warning,
# and that is why the measured spectrum is smoother than the target. The colour is kept only
# roughly. A two-dimensional modulation spectrum does not say which bands carry which
# modulation, so the random phase spreads the sentence's modulation into bands that were quiet,
# and clipping turns it into level there.

# %% [demo mt2] Its modulation spectrum, random modulation phase
twin = spectrum.to_sound(carrier="tones", fs=sentence.fs, rng=0)
sound = finish(twin)
fig, playhead = show(sound, target=spectrum, f_lo=125)

# %% [markdown]
# So a modulation spectrum fixes how a sound changes, not when. In the design note's check of
# the same sentence (`docs/design/views/modulation-targets.md`, C1) the two envelope arrays
# correlate at −0.08, and the pooled envelope, more than 30 dB below its peak 9.6% of the time
# in the sentence, never is in the twin. Sounds made this way, with a natural sound's modulation
# spectrum and a random modulation phase, have been used to ask what auditory neurons respond
# to (Hsu et al., 2004).

# %% [markdown]
# ## A drawn spectrum
#
# A target can be drawn from scratch as a few patches on the rate × density plane.
# `so.ModulationBlob(rate, density)` is a Gaussian bump at that rate [Hz] and density
# [cycles/octave], half an octave wide in rate and a quarter of a cycle per octave in density by
# default. The signs follow the [ripples](ripples.html): a positive rate and density is a
# downward sweep. A ripple is a blob of no width.
#
# A drawing sets a shape, not a depth, so `from_blobs` takes the depth separately: `rms_depth`,
# the rms of the envelopes about their mean, relative to the mean (0.2 by default). Envelopes
# cannot go below zero, and a random phase reaches only a limited depth before they would: about
# 0.28 for the blob below (C2 of the design note), against 0.71 for one full ripple. A drawn
# target that would need clipping is refused, with the largest depth that fits.

# %% [about]
# Downward sweeps around 4 Hz and half a cycle per octave, on steady tones at the band centres.
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
# The same target on the sentence. A sound as carrier lends its own modulation phase as well as
# its fine structure, so the sweeps now come when the sentence's syllables did, in its voice. The
# sentence is 2.5 s long, so this target is drawn for 2.5 s.

# %% [demo mt5] The same blob, on the sentence
short_target = so.ModulationSpectrum.from_blobs(so.ModulationBlob(4, 0.5), duration=2.5)
on_sentence = short_target.to_sound(carrier=sentence)
sound = finish(on_sentence)
fig, playhead = show(sound, target=short_target)

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


for name, snd, drawn in [
    ("tones", on_tones, target),
    ("noise", on_noise, target),
    ("sentence", on_sentence, short_target),
]:
    print(f"{name:>8}: {share_in_target(snd, drawn):.0%}")

# %% [markdown]
# The carrier matters as much as the target. A steady tone at each band's centre adds almost no
# modulation of its own, so the target survives. Noise and speech fluctuate inside every band,
# and those fluctuations land in the measured spectrum on top of the drawn one. That is why
# `"tones"` is the default carrier (C3 of the design note).

# %% [markdown]
# ## An edited sentence
#
# A measured spectrum can be edited and heard. `with_gain(g)` multiplies every cell by
# `g(rate, density)`; `lambda r, d: abs(r) <= 4` keeps only the modulations at 4 Hz and below,
# roughly the syllable rate and slower, and removes everything faster. Elliott & Theunissen
# (2009) filtered the modulation spectrum of speech's spectrogram in a similar way to find which
# modulations intelligibility needs. With the sentence itself as carrier, its own modulation phase and fine
# structure go back under the edited magnitudes, so wherever the gain is 1 the timing is kept.

# %% [about]
# The edit, put straight back on the sentence. The measured spectrum keeps some power above
# 4 Hz, less than the sentence had but more than the edit asks for: the sentence's fine
# structure brings fast modulation of its own back into every band.

# %% [demo mt6] Modulations above 4 Hz removed
slow = spectrum.with_gain(lambda rate, density: np.abs(rate) <= 4)
edited = slow.to_sound(carrier=sentence)
sound = finish(edited)
fig, playhead = show(sound, target=slow, f_lo=125)

# %% [about]
# The same edit with 20 rounds of a search in the manner of Griffin & Lim (1984): analyse the
# sound, keep its fine structure and modulation phase, impose the edited magnitudes again, and
# resynthesize. Each round brings the sound's own spectrum closer to the edit.

# %% [demo mt7] The same edit, after 20 iterations
searched = slow.to_sound(carrier=sentence, iterations=20)
sound = finish(searched)
fig, playhead = show(sound, target=slow, f_lo=125)

# %% [about]
# The same edit on steady tones, with a random modulation phase: the slow modulations survive,
# but the sentence's timing does not.

# %% [demo mt8] The same edit, on tones
on_tones_edit = slow.to_sound(carrier="tones", fs=sentence.fs, rng=0)
sound = finish(on_tones_edit)
fig, playhead = show(sound, target=slow, f_lo=125)

# %% [markdown]
# The modulation power left between 6 and 40 Hz, relative to the sentence's:


# %%
def fast_share(snd):
    """Share of the measured modulation power at 6-40 Hz."""
    measured = measure(snd, f_lo=125)
    power = 10 ** (measured.level / 10)
    fast = (np.abs(measured.w_t) >= 6) & (np.abs(measured.w_t) <= 40)
    return power[:, fast].sum() / power.sum()


for name, snd in [("edit", edited), ("20 iterations", searched), ("on tones", on_tones_edit)]:
    print(f"{name:>13}: {10 * np.log10(fast_share(snd) / fast_share(sentence)):.1f} dB")

# %% [markdown]
# None of them reaches the edit exactly, which removed everything there. A sound's envelopes are
# not free: they come from filtering one waveform, so most arrays of magnitudes belong to no
# sound at all, and the search finds a sound whose spectrum is close, never equal. The design
# note measures the same comparison in C6 and C7.

# %% [markdown]
# ## What this page leaves out
#
# - **Drawing with a mouse.** Targets here are written in code, as blobs or as edits; a program
#   for painting them is left to a separate project.
# - **dB targets.** A spectrum measured with `scale="db"` (of the log envelope, as Elliott &
#   Theunissen, 2009, define it) also goes back to sound, and never needs clipping, but the
#   linear spectrum of the result is not the one drawn.
# - **The other modulation spectra.** A [modulation spectrogram](modspectrogram.html) changes
#   over time and has no route back to sound, and a [sound texture](textures.html)'s per-band
#   modulation power is set through its statistics, not here.

# %% [markdown]
# ## References
#
# - Elliott & Theunissen (2009). The modulation transfer function for speech intelligibility.
#   *PLoS Comput. Biol.* 5(3), e1000302.
#   [doi:10.1371/journal.pcbi.1000302](https://doi.org/10.1371/journal.pcbi.1000302).
#   [`modulation.ModulationSpectrum.with_gain`](https://github.com/choyun1/sonore/blob/main/src/sonore/views/modulation.py#L503)
# - Griffin & Lim (1984). Signal estimation from modified short-time Fourier transform. *IEEE
#   Trans. Acoust. Speech Signal Process.* 32(2), 236–243.
#   [doi:10.1109/TASSP.1984.1164317](https://doi.org/10.1109/TASSP.1984.1164317).
#   [`modulation.ModulationSpectrum.to_sound`](https://github.com/choyun1/sonore/blob/main/src/sonore/views/modulation.py#L605)
# - Hsu, Woolley, Fremouw & Theunissen (2004). Modulation power and phase spectrum of natural
#   sounds enhance neural encoding performed by single auditory neurons. *J. Neurosci.* 24(41),
#   9201–9211.
#   [doi:10.1523/JNEUROSCI.2449-04.2004](https://doi.org/10.1523/JNEUROSCI.2449-04.2004).
#   [`modulation.ModulationSpectrum.to_sound`](https://github.com/choyun1/sonore/blob/main/src/sonore/views/modulation.py#L605)
# - Singh & Theunissen (2003). Modulation spectra of natural sounds and ethological theories of
#   auditory processing. *J. Acoust. Soc. Am.* 114(6), 3394–3411.
#   [doi:10.1121/1.1624067](https://doi.org/10.1121/1.1624067).
#   [`modulation.ModulationSpectrum`](https://github.com/choyun1/sonore/blob/main/src/sonore/views/modulation.py#L333)
