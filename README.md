<h1><a href="https://choyun1.github.io/sonore/gallery/"><picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/choyun1/sonore/main/docs/logo/sonore-banner-mark-dark.svg"><img alt="sonore gallery" src="https://raw.githubusercontent.com/choyun1/sonore/main/docs/logo/sonore-banner-mark-light.svg" height="56"></picture></a><picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/choyun1/sonore/main/docs/logo/sonore-banner-name-dark.svg"><img alt="sonore" src="https://raw.githubusercontent.com/choyun1/sonore/main/docs/logo/sonore-banner-name-light.svg" height="56"></picture></h1>

[![tests](https://github.com/choyun1/sonore/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/choyun1/sonore/actions/workflows/tests.yml) [![PyPI](https://img.shields.io/pypi/v/sonore)](https://pypi.org/project/sonore/) [![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://pypi.org/project/sonore/) [![License: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23086165.svg)](https://doi.org/10.5281/zenodo.23086165)

**Signals and stimuli for auditory research, built for Jupyter.**

sonore is a small Python library for making, manipulating, and analyzing sounds
the way hearing scientists think about them. Its analysis and synthesis tools
cover tones, harmonic complexes, shaped and correlated noises, ERB-spaced
subbands, invertible spectrograms, a phase vocoder, interaural cues, HRIR
spatialization of moving sources, synthetic room reverberation, and sound
texture synthesis. Levels are written as levels (`snd + 6*dB`), times as
seconds (`snd[0.1:0.5]`), and any sound at the end of a notebook cell plays.

It brings the sounds and representations of hearing research together in
one coherent system, held to the following standard: every
frame inverts exactly, the mathematics in its design documents is
checked by standalone scripts and the code by its tests, and every
example in the gallery can be heard beside the code that made it. It is built
to learn from and to build on.

The name comes from Pierre Schaeffer's *objet sonore*, the "sound object": a
sound taken as a thing in its own right and studied for how it is heard rather
than for what produced it. The `Sound` object at the center of this library is
meant in the same spirit.
The logo is drawn from the spectrogram of
[sonore saying its own name](https://choyun1.github.io/sonore/gallery/sourcefilter.html#d-fl1)
with a formant synthesizer.

**[▶ Listen to the gallery](https://choyun1.github.io/sonore/gallery/)**: every sound in this README and more, each
playable next to its plots, with a playhead that follows the sound.

**[▶ Try it in your browser](https://choyun1.github.io/sonore/)**: one editable cell that runs sonore
on the page, through Pyodide, with nothing to install.

**[▶ Draw sounds with sonore-sketch](https://choyun1.github.io/sonore-sketch/)**: draw speech, paint a
spectrogram or draw a modulation spectrum with a mouse, or erase parts of a
recording's spectrogram or modulation spectrum, and hear what sonore makes
of it, in your browser.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/choyun1/sonore/blob/main/docs/notebooks/start.ipynb)
A short tutorial you can run in the browser, with nothing to install: sounds, classic and
complex stimuli, sound textures, voices, and spatial hearing with reverberation and movement.

sonore is a library under active development and verification. Until
version 1.0, its API may change between releases without warning, and it has
not yet been fully validated for research use. Please check anything you rely on
against an independent implementation.

<!-- Contents: one entry per ## section, in order (tests/test_docs.py checks it). -->
<table><tr><td>

<small><b>Contents</b><br>
1&ensp;Using it<br>
&emsp;1.1&ensp;[What it's for](#what-its-for)<br>
&emsp;1.2&ensp;[Install](#install)<br>
&emsp;1.3&ensp;[Gallery](#gallery)<br>
2&ensp;The library<br>
&emsp;2.1&ensp;[Conventions](#conventions)<br>
&emsp;2.2&ensp;[What's in it](#whats-in-it)<br>
&emsp;2.3&ensp;[Related projects](#related-projects)<br>
3&ensp;Background<br>
&emsp;3.1&ensp;[Roadmap](#roadmap)<br>
&emsp;3.2&ensp;[References](#references)<br>
4&ensp;The project<br>
&emsp;4.1&ensp;[Development](#development)<br>
&emsp;4.2&ensp;[How sonore was developed](#how-sonore-was-developed)<br>
&emsp;4.3&ensp;[License and citation](#license-and-citation)</small>

</td></tr></table>

## What it's for

- **Psychophysical stimuli.** Pure tones, harmonic complexes with any phase
  scheme (cosine, sine, alternating, random, Schroeder±), band-limited square,
  sawtooth and pulse trains, all on a fixed F0 or following any F0 contour
  (an `F0Track`, or window times and values) without aliasing, chirps, band-limited and spectrally tilted noise,
  iterated rippled noise. Everything is reproducible from a seed.
- **Binaural and spatial hearing.** Exact fractional ITDs, ILDs, interaurally
  correlated noise, Oscor and Phasewarp, windowed ITD/ILD/coherence analysis
  (broadband or per band), and rendering of static or moving sources through
  measured HRIRs (PKU-IOA, downloaded on first use, or any SOFA file), with
  each ear's delay sliding continuously so a source can change distance
  (level, travel time, Doppler) in a room.
- **Synthetic speech.** A Klatt-style cascade/parallel formant synthesizer
  (Klatt, 1980) driven by named parameter tracks (F0, formant frequencies
  and bandwidths, voicing, aspiration and frication levels), for vowels,
  consonant continua and breathy voice with every acoustic cue set exactly.
- **Speech in noise.** Speech-shaped noise from a long-term average spectrum,
  mixing at a target SNR, ideal binary and ratio masks with exact resynthesis.
- **Cochlear-implant and envelope/TFS studies.** A perfect-reconstruction ERB
  filterbank, Hilbert envelopes and fine structure, and a channel vocoder.
- **Spectrotemporal modulation.** Moving ripples, sums of ripples, and
  dynamic moving ripples, specified as patterns in time and log-frequency and
  rendered on tone, harmonic, noise, or low-noise carriers (or any sound's
  fine structure), plus a modulation spectrum in cycles/octave to verify them.
- **Time and pitch manipulation.** A phase vocoder (Gordon & Strawn, 1985) with
  phase locking: time-stretch without changing pitch, pitch-shift without
  changing duration, and oscillator-bank resynthesis with arbitrary frequency
  remapping (e.g. shifting a harmonic complex to make it inharmonic).
- **Rooms.** Synthetic impulse responses with frequency-dependent decay from the
  statistics of real rooms (Traer & McDermott, 2016), with a controllable DRR
  and decorrelated binaural tails, plus the paper's "unnatural" variants
  (time-reversed and linear decays; inverted, exaggerated and reduced frequency
  dependence) and a per-band RT60 measurement. A room's RT60 is one number or
  a curve over frequency, such as one drawn by hand.
- **Sound textures.** The texture model of McDermott & Simoncelli (2011):
  measure a recording's envelope, modulation and correlation statistics, and
  synthesize new samples that share them. A clean-room implementation with
  analytic gradients; every deviation from the MATLAB toolbox is documented
  (`so.texture.DIFFERENCES_FROM_TOOLBOX`).
- **Teaching and demos.** One-call overview plots (waveform, spectrum,
  spectrogram, modulation spectrum) next to an audio player.

sonore is not an experiment runner, does not calibrate to dB SPL, and has no
models of the ear or of perception; see "Related projects" below for those.

## Install

```bash
pip install "sonore[notebook]"   # extras: sofa (HRIR files), play (sounddevice), dev (tests)
```

or, for the development version:

```bash
git clone https://github.com/choyun1/sonore
cd sonore
pip install -e ".[notebook]"
```

Requires Python ≥ 3.11, numpy, scipy ≥ 1.12, matplotlib, and soundfile.

A first cell in a notebook:

```python
import sonore as so
from sonore import dB

fs = 44100
tone = so.pure_tone(0.5, fs, 1000).ramp(10e-3) - 20 * dB
noise = so.gaussian_noise(0.5, fs, rng=0).ramp(10e-3) - 20 * dB
tone_in_noise = tone + (noise + 5 * dB)  # the tone at -5 dB SNR
so.overview(tone_in_noise)
tone_in_noise  # a Sound at the end of a cell plays
```

## Gallery

The [listening gallery](https://choyun1.github.io/sonore/gallery/) has every
sound beside plots of the same audio, with a playhead that follows it, and the
code for each example. Three of the kinds of plot sonore draws:

**Spectrograms.** One sentence, read by a male and a female talker, through a
wideband (5 ms) and a narrowband (33 ms) Gabor frame: the first resolves the
glottal pulses, the second the harmonics. [▶ listen](https://choyun1.github.io/sonore/gallery/speech.html#d-27)

![Wideband and narrowband spectrograms of a sentence, read by two talkers](https://raw.githubusercontent.com/choyun1/sonore/main/docs/gallery/img/27_two_classic_spectrograms.png)

**Cepstrum.** The cepstrogram of the same sentence for each talker, with
`so.Cepstrum`'s F0 beside WORLD's Harvest. [▶ listen](https://choyun1.github.io/sonore/gallery/sourcefilter.html#d-c2)

![Cepstrograms and cepstral pitch of a sentence, read by two talkers](https://raw.githubusercontent.com/choyun1/sonore/main/docs/gallery/img/c2_the_cepstrogram_and_its_pitch.png)

**Modulation spectra.** Three ripple patterns as specified (top), the
synthesized sounds' subband envelopes (middle), and their measured
`so.ModulationSpectrum` (bottom), which peaks at each specified rate and density.
[▶ single](https://choyun1.github.io/sonore/gallery/ripples.html#d-01) [▶ sum of two](https://choyun1.github.io/sonore/gallery/ripples.html#d-03) [▶ dynamic](https://choyun1.github.io/sonore/gallery/ripples.html#d-06)

![Ripple patterns, envelopes, and modulation spectra](https://raw.githubusercontent.com/choyun1/sonore/main/docs/images/ripples.png)

More in the gallery:

**Synthetic sounds**

- [Tones and noises](https://choyun1.github.io/sonore/gallery/classic.html): speech-shaped noise, beats, tone sequences, aliasing. [▶ speech-shaped noise](https://choyun1.github.io/sonore/gallery/classic.html#d-k1)
- [Iterated rippled noise](https://choyun1.github.io/sonore/gallery/irn.html): a pitch made from noise and a delay. [▶ sixteen iterations](https://choyun1.github.io/sonore/gallery/irn.html#d-09)
- [Spectrotemporal ripples](https://choyun1.github.io/sonore/gallery/ripples.html): moving ripples on different carriers. [▶ dynamic ripple](https://choyun1.github.io/sonore/gallery/ripples.html#d-06)

**Classic demonstrations** (reproductions, credited to their authors)

- [Streams: reproducing Bregman & Ahad](https://choyun1.github.io/sonore/gallery/streams.html): sonore's attempt to rebuild Bregman & Ahad's (1996) demonstrations of streaming from their booklet. [▶ a six-tone cycle](https://choyun1.github.io/sonore/gallery/streams.html#d-sg1)

**Seeing and changing sound**

- [Seeing speech](https://choyun1.github.io/sonore/gallery/speech.html): time-frequency analysis on one sentence. [▶ reassignment](https://choyun1.github.io/sonore/gallery/speech.html#d-30)
- [Analysis and resynthesis](https://choyun1.github.io/sonore/gallery/resynthesis.html): perfect reconstruction, a decimated constant-Q transform with about two numbers per sample, and the ideal binary mask. [▶ binary mask](https://choyun1.github.io/sonore/gallery/resynthesis.html#d-25)
- [Phase vocoder](https://choyun1.github.io/sonore/gallery/pv.html): duration, pitch and partials changed independently. [▶ up a fifth](https://choyun1.github.io/sonore/gallery/pv.html#d-13)
- [Hearing through a vocoder](https://choyun1.github.io/sonore/gallery/vocoder.html): cochlear-implant simulation. [▶ sixteen bands](https://choyun1.github.io/sonore/gallery/vocoder.html#d-ci16)
- [Modulation spectrogram](https://choyun1.github.io/sonore/gallery/modspectrogram.html): how each band's envelope moves, moment by moment. [▶ speech, babble, noise](https://choyun1.github.io/sonore/gallery/modspectrogram.html#d-s1)
- [Hearing a modulation spectrum](https://choyun1.github.io/sonore/gallery/modtargets.html): sounds made from a measured, edited or drawn modulation spectrum. [▶ rain with a sentence's timing](https://choyun1.github.io/sonore/gallery/modtargets.html#d-mt9)

**Textures**

- [Sound textures](https://choyun1.github.io/sonore/gallery/textures.html): syntheses from statistics (McDermott & Simoncelli, 2011). [▶ stream](https://choyun1.github.io/sonore/gallery/textures.html#d-t01b)
- [Events and pops](https://choyun1.github.io/sonore/gallery/events.html): one event copied into a texture, and the power law of a fire's pops. [▶ 30 events per second](https://choyun1.github.io/sonore/gallery/events.html#d-e04)
- [Sound quilts](https://choyun1.github.io/sonore/gallery/quilts.html): recordings cut into segments and reordered (Overath et al., 2015). [▶ fire](https://choyun1.github.io/sonore/gallery/quilts.html#d-sq4)

**Voices**

- [What is a voice](https://choyun1.github.io/sonore/gallery/whatisavoice.html): what a voice is made of, how voices differ, and the male and female talker every speech page uses. [▶ both](https://choyun1.github.io/sonore/gallery/whatisavoice.html#d-tk1)
- [The source-filter model](https://choyun1.github.io/sonore/gallery/sourcefilter.html): a voice built from a source and a filter, each measured back (pitch, formants, envelope), and put back together. [▶ ga](https://choyun1.github.io/sonore/gallery/sourcefilter.html#d-fc3) [▶ formant copy](https://choyun1.github.io/sonore/gallery/sourcefilter.html#d-ft5) [▶ buzz on the pitch track](https://choyun1.github.io/sonore/gallery/sourcefilter.html#d-hp2) [▶ envelope only](https://choyun1.github.io/sonore/gallery/sourcefilter.html#d-c3)
- [A richer source-filter model](https://choyun1.github.io/sonore/gallery/richer.html): an envelope measured to the pitch period, a source that is partly noise, and the phases. [▶ whispered](https://choyun1.github.io/sonore/gallery/richer.html#d-ap13)
- [Rebuilding and changing a voice](https://choyun1.github.io/sonore/gallery/voice.html): pitch and formants moved separately. [▶ one talker toward another](https://choyun1.github.io/sonore/gallery/voice.html#d-vc15)

**Spatial hearing** (🎧 headphones)

- [Binaural cues](https://choyun1.github.io/sonore/gallery/binaural.html): timing and correlation between the ears, and height. [▶ over the head](https://choyun1.github.io/sonore/gallery/binaural.html#d-u2)
- [Rooms](https://choyun1.github.io/sonore/gallery/reverb.html): distance, and rooms built from the statistics of real ones. [▶ time-reversed decay](https://choyun1.github.io/sonore/gallery/reverb.html#d-18)
- [Moving talkers](https://choyun1.github.io/sonore/gallery/moving.html): talkers rendered through measured HRIRs. [▶ pass-by](https://choyun1.github.io/sonore/gallery/moving.html#d-m7)
- [A "cocktail party"](https://choyun1.github.io/sonore/gallery/cocktail.html): two to six talkers walking around the listener. [▶ six talkers](https://choyun1.github.io/sonore/gallery/cocktail.html#d-cp5)

**Music**

- [Timbre](https://choyun1.github.io/sonore/gallery/timbre.html): attack, brightness and spectral flux, measured.
- [Tuning and temperament](https://choyun1.github.io/sonore/gallery/temperament.html): just and tempered intervals, and Twinkle, Twinkle in four tunings. [▶ just intonation](https://choyun1.github.io/sonore/gallery/temperament.html#d-tt3)
- [Pitch class and height](https://choyun1.github.io/sonore/gallery/chroma.html): a constant-Q filter bank, chroma, Shepard tones and the Risset glide. [▶ a scale that keeps rising](https://choyun1.github.io/sonore/gallery/chroma.html#d-pc3)
- [The pipe organ](https://choyun1.github.io/sonore/gallery/organ.html): stops as additive synthesis (a template).

## Conventions

- **Sounds.** `Sound` = immutable `(n_samples, n_channels)` float array + `fs`. Operations return new Sounds.
- **Arithmetic.** `a + b` mixes, `a * b` multiplies sample-wise, `2 * a` scales, mono broadcasts to stereo.
- **Bands and envelopes.** A filterbank's output (`Subbands`) is a collection of Sounds, and so is its fine structure (`.tfs()`). Envelopes are *not* sounds: a `TemporalEnvelope` is its own type, non-negative, often at a low sampling rate, and applied to sounds by multiplication. It holds one envelope, or one per band of the filterbank it keeps; the banded case is what the field calls a **cochleagram**. A `SpectralEnvelope` is the other meaning of the word: power over frequency with the harmonics smoothed away, on a grid of time windows, whatever estimator made it. The Hilbert decomposition is literal: `sb == sb.envelopes() * sb.tfs()`.
- **Levels.** `a + 6*dB`, `a - 3*dB`. Adding a bare number is an error, so it can't be mistaken for a DC offset. dB is always `20*log10(amplitude)`.
- **Time.** `snd[0.1:0.5]` slices by seconds; `snd.data` for samples.
- **Randomness.** Every stochastic function takes `rng=` (a seed or `np.random.Generator`).
- **Binaural.** Positive ITD = right ear leads; positive ILD = right ear louder.
- **Space.** Meters, head-centered, x = right, y = front, z = up. `hcc` = (distance cm, elevation °, azimuth ° clockwise from front).
- **Threads.** The large FFTs (filterbanks, Hilbert envelopes, resampling) use every available core. Results don't depend on it; when running several jobs in parallel, `so.set_fft_workers(1)` (also as a `with` block) keeps them from competing.
- **Plots.** Every plotting function takes an optional `ax` and returns it; global matplotlib settings are never touched.

## What's in it

The folders follow meaning. `core` holds `Sound` and the processing that
needs no analysis; `sources` makes sounds from parameters; `frames` are
analyses with an exact inverse, and `views` are one-way analyses together
with the routes back to sound they have (the channel vocoder, the phase
vocoder; a voice's views go back through `sources.source_filter`); `spatial` and `texture` are topics built on
those. Imports between modules never form a cycle, and `core` imports
nothing above it at module level. `plotting` is called from every object's
`.plot()`. docs/design/layout.md has the diagram.
Most names are also at the top level as `so.name`; the texture ones are
under `so.texture` and `sonore.texture.synth`.

| Module | Contents |
|---|---|
| [`core.sound`](https://github.com/choyun1/sonore/blob/main/src/sonore/core/sound.py) | `Sound`, `load` |
| [`core.units`](https://github.com/choyun1/sonore/blob/main/src/sonore/core/units.py) | `dB`, `Decibels` |
| [`core.utils`](https://github.com/choyun1/sonore/blob/main/src/sonore/core/utils.py) | `rms`, `amp_to_db`, `power_to_db`, `db_to_amp`, `db_to_power`, `freq_to_erb`, `erb_to_freq`, `freq_to_mel`, `mel_to_freq` (HTK or Slaney mel) |
| [`core.fft`](https://github.com/choyun1/sonore/blob/main/src/sonore/core/fft.py) | `set_fft_workers`, `fft_workers` (threads for the large FFTs; default every available core; results identical for any setting) |
| [`core.processing`](https://github.com/choyun1/sonore/blob/main/src/sonore/core/processing.py) | `pad`, `truncate`, `concat`, `splice` (segments of a sound in a new order, joined with cross-fades aligned by cross-correlation), `mix`, `normalize`, `match_fs`, `match_channels`, `relative_db`, `bandpass`, `butter_filter`, `amplitude_modulate`, `resonator` and `antiresonator` (Klatt's formant and antiformant; frequency and bandwidth may glide, with no clicks) |
| [`sources.waveforms`](https://github.com/choyun1/sonore/blob/main/src/sonore/sources/waveforms.py) | `silence`, `pure_tone`, `harmonic_complex`, `schroeder_complex`, `square_wave`, `sawtooth_wave`, `pulse_train`, `linear_chirp`, `exponential_chirp`, `gaussian_noise`, `correlated_noise`, `iterated_ripple_noise`, `glottal_source` (Liljencrants-Fant glottal pulses on a fixed F0 or a contour, shape set by Fant's Rd, which may change over time; no aliasing), `lf_harmonics` (the pulse's Fourier coefficients in closed form), `lf_pulse` (one period, to draw) |
| [`sources.klatt`](https://github.com/choyun1/sonore/blob/main/src/sonore/sources/klatt.py) | `klatt_synthesize` (a Klatt-style cascade/parallel formant synthesizer: harmonic voicing with Klatt's glottal spectrum or LF pulses (`SS`, `RD`), aspiration and frication noise modulated at F0, nasal pole and zero, formants 1-5 in cascade and 1-6 in parallel, radiation; every parameter a number or a `(times, values)` track), `klatt_continuum` (evenly spaced parameter sets between two endpoints), `KLATT_DEFAULTS` |
| [`sources.source_filter`](https://github.com/choyun1/sonore/blob/main/src/sonore/sources/source_filter.py) | `source_filter_synthesize` (a voice from an F0 track, a spectral envelope and an aperiodicity: harmonics carrying the periodic part of the envelope plus white noise shaped to the rest; reads any F0 track and any envelope; checked in `tools/check_voice_recipe_claims.py`) |
| [`sources.ripples`](https://github.com/choyun1/sonore/blob/main/src/sonore/sources/ripples.py) | `Ripple`, `RippleSum`, `DynamicRipple`, `ripple_sound`; patterns can also be any function `f(t, x)` of time and octaves, and `pattern.render(filterbank, dur, fs)` gives their `TemporalEnvelope` on that filterbank |
| [`sources.gaussian_spectrogram`](https://github.com/choyun1/sonore/blob/main/src/sonore/sources/gaussian_spectrogram.py) | `gaussian_spectrogram` (a random spectrogram whose levels correlate exponentially in time and frequency, after McDermott, Wrobleski & Oxenham, 2011; returns a `TemporalEnvelope`, heard through `to_sound`) |
| [`sources.events`](https://github.com/choyun1/sonore/blob/main/src/sonore/sources/events.py) | `event_train` (copies of one event at random times, levels and pitches, each drawn from a distribution, with a rate that can change over time; from single events to a texture) |
| [`frames.frame`](https://github.com/choyun1/sonore/blob/main/src/sonore/frames/frame.py) | `Frame` (invertible analyses: `analyze`, `synthesize` as least squares, `frame_bounds`, `energy`, `adjoint`) |
| [`frames.filterbank`](https://github.com/choyun1/sonore/blob/main/src/sonore/frames/filterbank.py) | `Filterbank` (one class for every undecimated bank: a frequency scale, centers on it and a filter shape; canonical dual, tightness measured), `cosine_filterbank` (perfect-reconstruction cosine banks on the ERB, octave, mel or linear scale, any centers), `gammatone_filterbank` (exact gammatone responses, causal or zero-phase; `envelope_peak_delay` gives each filter's latency), `morlet_filterbank` (Morlet wavelets); all add edge filters by default so synthesis is exact on the whole band, and `edges=False` gives the bare bank for cochleagrams. `subbands`, `Subbands` (a collection of Sounds: `.envelopes()`, `.tfs()`, `.to_sound()`). `DecimatedFilterbank` (any bank with each band kept at its own rate, exact; on the semitone cosine bank, a decimated constant-Q transform), `DecimatedSubbands` (its complex coefficients, one sequence per band, with `.times`, `.db`, `.plot()`, `.to_sound()`) |
| [`frames.gabor`](https://github.com/choyun1/sonore/blob/main/src/sonore/frames/gabor.py) | `GaborFrame` (the STFT as a frame; any window, zero-padded FFTs), `TVGaborFrame` (a Gabor frame whose window changes over time, from an explicit schedule, `from_function`, or `pitch_adaptive` from an F0 track; exact inverse; coefficients are a `TVSTFT`), `STFT` (a `GaborFrame` analysis: exact inverse, fast Griffin-Lim), `TVSTFT` (a `TVGaborFrame` analysis) |
| [`views.mask`](https://github.com/choyun1/sonore/blob/main/src/sonore/views/mask.py) | `Mask` (gains for the coefficients of an `STFT`, `TVSTFT` or `Subbands`; multiply and go back with `.to_sound()`), `ideal_binary_mask`, `ideal_ratio_mask` |
| [`views.view`](https://github.com/choyun1/sonore/blob/main/src/sonore/views/view.py) | `View` (the base of every view: a `discards` sentence saying what it drops, and a `synthesize` and a `to_sound` that raise `NotInvertibleError` with that reason and the route back to sound, if any; views with a canonical route back override `to_sound`) |
| [`views.spectra`](https://github.com/choyun1/sonore/blob/main/src/sonore/views/spectra.py) | `Spectrum` (a power spectral density in dB re 1 per Hz; `.to_sound` with a noise, a sound's phase or the minimum phase as carrier), `long_term_spectrum`, `tandem_power` (TANDEM-STRAIGHT-style pitch-adaptive power, after Kawahara et al., 2011; magnitude only, a `TFPower`), `reassigned_spectrogram` (Kodera et al., 1978; Auger & Flandrin, 1995: spectrogram cells moved to their reassigned time and frequency, binned for display; not invertible) |
| [`views.temporal_envelope`](https://github.com/choyun1/sonore/blob/main/src/sonore/views/temporal_envelope.py) | `TemporalEnvelope` (one envelope, or one per band of a filterbank it keeps, i.e. a cochleagram; `env * snd` modulates, band by band through the filterbank when it has one; `.plot()`, `.modulation_spectrum()`, `env * subbands`, `.to_sound(carrier)`); `channel_vocode` (the channel vocoder of cochlear-implant simulations, after Shannon et al., 1995: band envelopes, lowpassed at any cutoff, on a carrier of noise, tones at the band centers, or any sound) |
| [`views.modulation`](https://github.com/choyun1/sonore/blob/main/src/sonore/views/modulation.py) | `ConstantQModulationFilterbank`, `OctaveModulationFilterbank` (circular, analytic output optional), `HannModulationFilterbank` (Hann-windowed complex kernels of a whole number of cycles, defined in time: constant Q or one fixed window, centered or causal), `ModulationSpectrum` (linear-frequency from an STFT, or `.octave()` in cycles/octave; one made from envelopes can be edited with `.with_gain()` and heard with `.to_sound(carrier=...)`, the carrier supplying the modulation phase and fine structure; `.from_blobs()` draws a target from `ModulationBlob`s at a chosen rms depth, or in dB at a chosen spread, which can go deeper) |
| [`views.modspectrogram`](https://github.com/choyun1/sonore/blob/main/src/sonore/views/modspectrogram.py) | `ModulationSpectrogram` (a modulation spectrum per time window: power, local mean and depth for every acoustic band and modulation rate, from any filterbank's `TemporalEnvelope`, with a `valid` mask; `.plot()` as rate against time, one band, or band against time at one rate; `.at(t)`, `.slices(t)`, `.animate()`; not invertible) |
| [`views.cepstrum`](https://github.com/choyun1/sonore/blob/main/src/sonore/views/cepstrum.py) | `Cepstrum` (the real cepstrum of an `STFT` or `TVSTFT`: rectangular liftering with a fixed or per-time-window cutoff, the cepstral envelope, resynthesis with the original phase, exact when unliftered, or the minimum phase, and classic cepstral F0 after Noll, 1967) |
| [`views.mfcc`](https://github.com/choyun1/sonore/blob/main/src/sonore/views/mfcc.py) | `MFCC` (mel-frequency cepstral coefficients of a `Sound`, with the usual speech settings, or of any `STFT` or `TVSTFT`: HTK or Slaney mel, height- or area-normalized triangles straight in mel or in Hz, the mel spectrogram, deltas, the smoothed envelope the coefficients keep, `.plot()`; reproduces Kaldi's and librosa's numbers to rounding error; not invertible) |
| [`views.lpc`](https://github.com/choyun1/sonore/blob/main/src/sonore/views/lpc.py) | `LPC` (linear prediction of each time window of a `Sound` or an `STFT`, by the autocorrelation method and `levinson`: the all-pole envelope, as a `SpectralEnvelope` too, and every resonance from the roots; `.plot()`; not invertible), `formant_track` → `FormantTrack` (F1 to F3 every 5 ms: Praat's recipe for candidates, within a few hertz of Praat's own, and a Viterbi tracker; `.track(n)` gives the `(times, values)` that `klatt_synthesize` takes; `.plot()`) |
| [`views.f0`](https://github.com/choyun1/sonore/blob/main/src/sonore/views/f0.py) | `f0_track` → `F0Track` (F0 every 5 ms with a voiced/unvoiced decision: candidates from YIN's difference function, refinement by the instantaneous frequency of the harmonics up to 1.5 kHz, a periodicity score, a Viterbi pass; `.plot()`; checked against laryngograph F0), `scale_f0` (a pitch change on any F0 contour, with a range factor) |
| [`views.timbre`](https://github.com/choyun1/sonore/blob/main/src/sonore/views/timbre.py) | `log_attack_time` and `attack_segment` (the attack by Peeters et al.'s weakest-effort method), `spectral_centroid` (magnitude spectrum by default, as the paper) and `spectral_flux` → `DescriptorTrack` (one value per 5.8 ms, with `.median` and `.iqr`); timbre descriptors written from Peeters et al. (2011), checked in `tools/check_timbre_claims.py` |
| [`views.chroma`](https://github.com/choyun1/sonore/blob/main/src/sonore/views/chroma.py) | `chroma` → `Chroma` (the power in each of the 12 pitch classes per 0.05 s time window, from a cosine filterbank with one band per semitone, A1 to A8, the octaves added together); `Chroma.to_sound(fs, octaves=...)` plays each pitch class as a Shepard tone, with `octaves` saying how the power spreads over octaves; checked in `tools/check_chroma_claims.py` |
| [`views.spectral_envelope`](https://github.com/choyun1/sonore/blob/main/src/sonore/views/spectral_envelope.py) | `pitch_adaptive_envelope` → `SpectralEnvelope` (CheapTrick, written from Morise's 2015 paper; within 0.01 dB of WORLD's at the paper's `q1`), `warp_frequency` (formants moved along frequency on any envelope or aperiodicity, by a ratio, a ratio over time or any frequency map), `SpectralEnvelope` (any envelope as power on a grid of time windows and frequencies, labelled by the `method` that made it: CheapTrick, or `Cepstrum.envelope_view()`, `MFCC.envelope_view()` and `LPC.envelope_view()`), `TimeFrequencyGrid` (the Protocol for anything on such a grid that the synthesizers read) |
| [`views.aperiodicity`](https://github.com/choyun1/sonore/blob/main/src/sonore/views/aperiodicity.py) | `band_aperiodicity` → `Aperiodicity` (D4C, written from Morise's 2016 paper, with its measured 8 dB bias corrected), `harmonic_aperiodicity` (the share of noise, by fitting the harmonics) |
| [`views.phasevocoder`](https://github.com/choyun1/sonore/blob/main/src/sonore/views/phasevocoder.py) | `time_stretch`, `pitch_shift` (identity phase locking), `pv_analyze` → `PVAnalysis` (instantaneous frequency; oscillator-bank `to_sound` with `time_scale` and `freq_map`) |
| [`views.quilting`](https://github.com/choyun1/sonore/blob/main/src/sonore/views/quilting.py) | `quilt` (the sound quilts of Overath et al., 2015: segments of a sound in a new order, each chosen so that its join changes the cochleagram as much as the sound itself did there; written from the paper's methods) |
| [`views.envelope_peaks`](https://github.com/choyun1/sonore/blob/main/src/sonore/views/envelope_peaks.py) | `envelope_peaks` → `EnvelopePeaks` (how big a sound's pops are: the largest envelope value in each 5 ms bin above 300 Hz and 2 kHz, and the power-law exponent of their upper tail by the Hill estimator, with its uncertainty; `.plot()`) |
| [`spatial.binaural`](https://github.com/choyun1/sonore/blob/main/src/sonore/spatial/binaural.py) | `apply_itd_ild`, `simple_bir`, `interaural_cues`, `oscor`, `phasewarp` |
| [`spatial.spatialization`](https://github.com/choyun1/sonore/blob/main/src/sonore/spatial/spatialization.py) | `HRIRSet` (PKU-IOA, SOFA; onset-aligned interpolation), `spatialize`, `move_sound` (paths as functions of time, continuous ear delays, room tail), `hcc_trajectory` and other trajectories, coordinate conversions, `distance_gain_db` |
| [`spatial.hrir_data`](https://github.com/choyun1/sonore/blob/main/src/sonore/spatial/hrir_data.py) | `load_hrirs`: public HRIR databases (PKU-IOA) downloaded on first use, checksum-verified and cached |
| [`spatial.reverb`](https://github.com/choyun1/sonore/blob/main/src/sonore/spatial/reverb.py) | `synth_ir` (natural rooms, or the paper's atypical `decay_shape` / `rt60_profile` / `drr_profile` variants; `rt60` is a number or a curve `(freqs, rt60s)`), `band_rt60s`, `measure_rt60` |
| [`texture.stats`](https://github.com/choyun1/sonore/blob/main/src/sonore/texture/stats.py) | `TextureModel`, `TextureStats` (`.measure`, `.snr`, `.replace` for hybrids, `.save`/`.load`) |
| [`texture.synth`](https://github.com/choyun1/sonore/blob/main/src/sonore/texture/synth.py) | `synthesize` (full loop), `impose_channel`; gradients in [`texture.grad`](https://github.com/choyun1/sonore/blob/main/src/sonore/texture/grad.py) |
| [`plotting`](https://github.com/choyun1/sonore/blob/main/src/sonore/plotting.py) | `overview` and the `plot_*` functions behind each object's `.plot()`; `plot_tf_db` draws any time-frequency level on non-uniform time windows; modulation spectrograms draw invalid cells gray and animate with their sound; cochleagrams take `align="peak"` (draw causal gammatone bands without their latency) and `fscale="linear"` (to match spectrograms) |

## Related projects

Where to go for what sonore leaves out:

- [slab](https://github.com/DrMarc/slab): calibrated levels in dB SPL,
  playback, trial sequences and adaptive staircases. Its sound making
  overlaps with sonore's, and the two share the same sample layout
  (samples × channels), so a sound passes between them in one line:

  ```python
  s = slab.Sound(snd.data, samplerate=snd.fs)  # sonore to slab; then set s.level in dB SPL
  snd = so.Sound(s.data, s.samplerate)  # slab to sonore
  ```

  slab reads samples as pascals, so a sonore sound at RMS 1 shows as 94 dB SPL
  until you set its level.
- [PsychoPy](https://www.psychopy.org/): running experiments.
- [Auditory Modeling Toolbox](https://amtoolbox.org/) (MATLAB/Octave) and
  [torch_amt](https://github.com/StefanoGiacomelli/torch_amt) (PyTorch):
  models of the auditory system that predict what a listener hears.
- [brian2hears](https://brian2hears.readthedocs.io/): auditory periphery and
  spiking models.
- [MoSQITo](https://github.com/Eomys/MoSQITo): loudness, sharpness, roughness
  and other sound quality metrics.
- [Parselmouth](https://github.com/YannickJadoul/Parselmouth) (Praat in
  Python) and [pyworld](https://github.com/JeremyCCHsu/Python-Wrapper-for-World-Vocoder)
  (WORLD): speech analysis and synthesis.
- [librosa](https://librosa.org/): music and audio analysis.
- [pyroomacoustics](https://github.com/LCAV/pyroomacoustics): geometric room simulation.
- [pyfar](https://pyfar.org/) / [sofar](https://github.com/pyfar/sofar): acoustics and SOFA files.
- [sonore-sketch](https://choyun1.github.io/sonore-sketch/) ([source](https://github.com/choyun1/sonore-sketch)):
  five tabs, each one of sonore's routes back to sound with a surface to
  draw on. Draw speech (formant tracks for the Klatt synthesizer), Paint
  spectrogram, Erase spectrogram (of a recording), Draw modulation (blobs on
  a modulation spectrum) and Erase modulation (of a recording). It runs
  sonore in the browser, with nothing to install.

## Roadmap

Only work still to do is listed here. Finished items are in
[docs/roadmap-done.md](https://github.com/choyun1/sonore/blob/main/docs/roadmap-done.md), and changes by release in
[CHANGELOG.md](https://github.com/choyun1/sonore/blob/main/CHANGELOG.md).

- Hand-written intro pages for the API reference, one per folder: the last
  step of the code audit ([issue #48](https://github.com/choyun1/sonore/issues/48)).
- The rest of KLSYN88's voice-quality controls (Klatt & Klatt, 1990) for the formant synthesizer:
  open quotient, spectral tilt, flutter, double pulsing, and its KLGLOTT88 source.
- Peak-based sinusoidal modeling (McAulay & Quatieri, 1986) alongside the channel oscillator bank.
- Routes back to sound from a sharpened time-frequency picture: a synchrosqueezed STFT (the
  coefficients moved in frequency only, so it stays exactly invertible) and partials tracked on
  the reassigned spectrogram, played by an oscillator bank (as in Fitz & Haken's Loris).
- On-demand download of other public HRIR databases.
- A block-by-block (streaming) modulation spectrogram, as the reference for a
  live version on a phone: the modulation spectrum of everyday sounds as they happen.
- Quilts longer than their source, which would reuse segments. The paper's rule picks
  the best match every time, so a reused segment would start a loop that repeats; it
  would need a random pick among the best few and a minimum gap before a segment
  returns. That would be sonore's own variant, to be checked by ear for repeats.


## References

The works sonore implements, follows or cites are listed in
[docs/references.md](docs/references.md), each with a link to the work, tags linking
to the code that implements it, and the gallery pages that use it. The same page lists
the [reference implementations](docs/references.md#reference-implementations) sonore was
checked against or consulted.

## Development

```bash
pip install -e ".[dev]"
pytest                             # ~40 s; one test file per module
ruff check . && ruff format .
python docs/gallery/build.py       # regenerate the listening gallery (a few minutes)
```

## How sonore was developed

sonore began as sigtools, the code I (Adrian Cho) wrote in graduate school to
make psychoacoustic stimuli ([migrating from sigtools](https://github.com/choyun1/sonore/blob/main/docs/migrating-from-sigtools.md)). The 0.2 redesign and everything since were
developed together with Claude, Anthropic's AI assistant, in chat sessions
during 2026.

**What Claude did.** Wrote most of the code, tests, documentation, and
gallery since 0.2, delivered as patches; drafted design documents; ran
numerical checks and profiling; and looked up and checked citations.

**What I did.** Decided what sonore is for and what goes in it, including
its API conventions, the texture work and its milestones, and the roadmap and
architecture. I chose and documented the texture recordings and set the
working rules: implement from the papers, verify every claim numerically,
document every deviation and data choice, and write a design document before
large features. I reviewed and applied each patch. The design principles that came out of
this are summarized in [docs/design/philosophy.md](https://github.com/choyun1/sonore/blob/main/docs/design/philosophy.md).

**How it is verified.** I have not read every line by hand. A code audit went
through every module of `src/sonore`, one sitting per module
([issue #48](https://github.com/choyun1/sonore/issues/48)): Claude wrote a
reading guide for each (every claim the code makes and the test behind it, the
lines no test runs, the results of deliberately breaking the code), I read the
guide and parts of the code, and each sitting fixed the bugs it found and
closed the gaps in the tests. Besides the audit, I rely on the following:

- The test suite, with one file per module.
- Finite-difference and dense-matrix checks of the mathematics.
- Cross-checks against independent implementations (see "Reference
  implementations").
- Written records of every decision (`DIFFERENCES_FROM_TOOLBOX`,
  `docs/textures/SOURCES.md`, `docs/design/`).
- The listening gallery, since these are sounds and should be heard.

I am responsible for sonore's correctness. If something is wrong, please
open an issue.

## License and citation

MIT; see [LICENSE](https://github.com/choyun1/sonore/blob/main/LICENSE). `pitch_adaptive_envelope`
(CheapTrick) and `band_aperiodicity` (D4C) are written from Morise's papers, not ported from WORLD's
code, and sonore's F0 tracker is its own, not WORLD's. If sonore is useful in your research, please cite
it using [CITATION.cff](https://github.com/choyun1/sonore/blob/main/CITATION.cff); the
*Cite this repository* button in the GitHub sidebar gives the same citation in APA and BibTeX.

Every release is archived on Zenodo. The all-versions DOI
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23086165.svg)](https://doi.org/10.5281/zenodo.23086165)
always points to the latest version. Each version also has its own DOI; cite the version you used.

- 0.9.0: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23745468.svg)](https://doi.org/10.5281/zenodo.23745468)
- 0.8.0: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23463707.svg)](https://doi.org/10.5281/zenodo.23463707)
- 0.7.0: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23358082.svg)](https://doi.org/10.5281/zenodo.23358082)
- 0.6.0: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23289306.svg)](https://doi.org/10.5281/zenodo.23289306)
- 0.5.0: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23148920.svg)](https://doi.org/10.5281/zenodo.23148920)
- 0.4.0: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23114390.svg)](https://doi.org/10.5281/zenodo.23114390)
- 0.3.1: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23086166.svg)](https://doi.org/10.5281/zenodo.23086166)
