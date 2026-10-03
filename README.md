# sonore

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23086165.svg)](https://doi.org/10.5281/zenodo.23086165)

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

**[▶ Listen to the gallery](https://choyun1.github.io/sonore/gallery/)**: every sound in this README and more, each
playable next to its plots, with a playhead that follows the sound.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/choyun1/sonore/blob/main/docs/notebooks/start.ipynb)
A first tour you can run in the browser, with nothing to install: stimuli, spectrograms, the
cepstrum, a phase vocoder, ripples and binaural cues.

<!-- Contents: one entry per ## section, in order (tests/test_docs.py checks it). -->
<table><tr><td>

<small><b>Contents</b><br>
1&ensp;Using it<br>
&emsp;1.1&ensp;[What it's for](#what-its-for)<br>
&emsp;1.2&ensp;[Install](#install)<br>
&emsp;1.3&ensp;[A short tour](#a-short-tour)<br>
&emsp;1.4&ensp;[Gallery](#gallery)<br>
2&ensp;The library<br>
&emsp;2.1&ensp;[Conventions](#conventions)<br>
&emsp;2.2&ensp;[What's in it](#whats-in-it)<br>
&emsp;2.3&ensp;[Related projects](#related-projects)<br>
3&ensp;Background<br>
&emsp;3.1&ensp;[Roadmap](#roadmap)<br>
&emsp;3.2&ensp;[References](#references)<br>
&emsp;3.3&ensp;[Migrating from sigtools](#migrating-from-sigtools)<br>
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
  dependence) and a per-band RT60 measurement.
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

Requires Python ≥ 3.10, numpy, scipy ≥ 1.12, matplotlib, and soundfile.

To try it without installing anything, open the [starter
notebook](https://colab.research.google.com/github/choyun1/sonore/blob/main/docs/notebooks/start.ipynb)
on Google Colab: its first cell installs sonore from PyPI.

## A short tour

```python
import sonore as so
from sonore import dB

fs = 44100

# Stimuli: every generator returns a Sound with RMS = 1
tone = so.pure_tone(0.5, fs, 1000).ramp(10e-3)
complex_ = so.harmonic_complex(0.5, fs, f0=200, harmonics=range(1, 21), phases="schroeder+")
noise = so.gaussian_noise(0.5, fs, band=(100, 8000), tilt=-3, rng=0)  # pink, band-limited

# Levels are dB units: + changes level, + with a Sound mixes
target_in_noise = tone + (noise + 5 * dB)  # tone at -5 dB SNR
quieter = complex_ - 12 * dB

# Time is in seconds
middle = target_in_noise[0.1:0.4]

# Binaural: positive ITD/ILD = toward the right
lateral = so.apply_itd_ild(noise, itd=300e-6, ild=6)
cues = so.interaural_cues(lateral, win_dur=20e-3)
cues.plot()

# Time and pitch (phase vocoder)
longer = so.time_stretch(complex_, 1.5)  # same pitch, 50% longer
up_a_fifth = so.pitch_shift(complex_, 7)  # same duration, +7 semitones

# Analysis
so.overview(target_in_noise)  # waveform, spectrum, spectrogram, modulation spectrum
target_in_noise  # in a notebook: an audio player
```

Every example in the [gallery](#gallery) is a runnable script like this one, with its code
shown beside the sound it makes.

## Gallery

The [listening gallery](https://choyun1.github.io/sonore/gallery/) has every
sound beside plots of the same audio, with a playhead that follows it, and the
code for each example. Three of the kinds of plot sonore draws:

**Spectrograms.** One sentence through a wideband (5 ms) and a narrowband
(33 ms) Gabor frame: the first resolves the glottal pulses, the second the
harmonics. [▶ listen](https://choyun1.github.io/sonore/gallery/speech.html#d-27)

![Wideband and narrowband spectrograms of a sentence](https://raw.githubusercontent.com/choyun1/sonore/main/docs/gallery/img/27_two_classic_spectrograms.png)

**Cepstrum.** The cepstrogram of the same sentence, with `so.Cepstrum`'s F0
beside WORLD's Harvest. [▶ listen](https://choyun1.github.io/sonore/gallery/cepstrum.html#d-c2)

![Cepstrogram and cepstral pitch of a sentence](https://raw.githubusercontent.com/choyun1/sonore/main/docs/gallery/img/c2_the_cepstrogram_and_its_pitch.png)

**Modulation spectra.** Three ripple patterns as specified (top), the
synthesized sounds' subband envelopes (middle), and their measured
`so.ModulationSpectrum` (bottom), which peaks at each specified rate and density.
[▶ single](https://choyun1.github.io/sonore/gallery/ripples.html#d-01) [▶ sum of two](https://choyun1.github.io/sonore/gallery/ripples.html#d-03) [▶ dynamic](https://choyun1.github.io/sonore/gallery/ripples.html#d-06)

![Ripple patterns, envelopes, and modulation spectra](https://raw.githubusercontent.com/choyun1/sonore/main/docs/images/ripples.png)

More in the gallery:

**Stimuli**

- [Classic stimuli](https://choyun1.github.io/sonore/gallery/classic.html): [▶ speech-shaped noise](https://choyun1.github.io/sonore/gallery/classic.html#d-k1), beats and roughness, and [▶ binaural beats](https://choyun1.github.io/sonore/gallery/classic.html#d-b4) [🎧](https://choyun1.github.io/sonore/gallery/classic.html#d-b4 "Headphones required for binaural sounds").
- [Iterated rippled noise](https://choyun1.github.io/sonore/gallery/irn.html): a pitch made from noise and a delay (Yost, 1996), from [▶ one iteration](https://choyun1.github.io/sonore/gallery/irn.html#d-i1) to [▶ sixteen](https://choyun1.github.io/sonore/gallery/irn.html#d-09).
- [Spectrotemporal ripples](https://choyun1.github.io/sonore/gallery/ripples.html): moving ripples on different carriers, and a [▶ dynamic moving ripple](https://choyun1.github.io/sonore/gallery/ripples.html#d-06).
- [Sound textures](https://choyun1.github.io/sonore/gallery/textures.html): recordings and their syntheses from statistics (McDermott & Simoncelli, 2011), such as a [▶ stream](https://choyun1.github.io/sonore/gallery/textures.html#d-t01b).

**Seeing and changing sound**

- [Seeing speech](https://choyun1.github.io/sonore/gallery/speech.html): a short course in time-frequency analysis on one sentence, from [▶ window length](https://choyun1.github.io/sonore/gallery/speech.html#d-w1) to [▶ reassignment](https://choyun1.github.io/sonore/gallery/speech.html#d-30).
- [Analysis and resynthesis](https://choyun1.github.io/sonore/gallery/resynthesis.html): a filterbank's [▶ perfect reconstruction](https://choyun1.github.io/sonore/gallery/resynthesis.html#d-26), and the [▶ ideal binary mask](https://choyun1.github.io/sonore/gallery/resynthesis.html#d-25).
- [Hearing through a vocoder](https://choyun1.github.io/sonore/gallery/vocoder.html): cochlear-implant simulation, from [▶ one band](https://choyun1.github.io/sonore/gallery/vocoder.html#d-ci1) to [▶ sixteen](https://choyun1.github.io/sonore/gallery/vocoder.html#d-ci16).
- [Modulation spectrogram](https://choyun1.github.io/sonore/gallery/modspectrogram.html): how fast and how deeply each band's envelope moves, moment by moment, from a [▶ gliding modulation rate](https://choyun1.github.io/sonore/gallery/modspectrogram.html#d-g1) to [▶ speech, babble and noise](https://choyun1.github.io/sonore/gallery/modspectrogram.html#d-s1).
- [Drawing a modulation spectrum](https://choyun1.github.io/sonore/gallery/modtargets.html): sounds made from a modulation spectrum, measured and edited or drawn, from [▶ a sentence's spectrum with random modulation phase](https://choyun1.github.io/sonore/gallery/modtargets.html#d-mt2) to [▶ a drawn patch on three carriers](https://choyun1.github.io/sonore/gallery/modtargets.html#d-mt3).
- [Phase vocoder](https://choyun1.github.io/sonore/gallery/pv.html): how it works, and duration, pitch and partials changed independently, such as [▶ up a fifth](https://choyun1.github.io/sonore/gallery/pv.html#d-13).

**Voices**

- [Formant synthesis](https://choyun1.github.io/sonore/gallery/formants.html): Klatt's synthesizer taken apart, from [▶ the source alone](https://choyun1.github.io/sonore/gallery/formants.html#d-fv1) through [▶ five formants](https://choyun1.github.io/sonore/gallery/formants.html#d-fv4), then [▶ six vowels](https://choyun1.github.io/sonore/gallery/formants.html#d-fw1) and [▶ /da/](https://choyun1.github.io/sonore/gallery/formants.html#d-fc2) from its formant transitions, and [▶ one vowel in three voices](https://choyun1.github.io/sonore/gallery/formants.html#d-fq3) from LF glottal pulses.
- [Cepstral analysis](https://choyun1.github.io/sonore/gallery/cepstrum.html): separating a voice's pitch from its timbre, [▶ envelope only](https://choyun1.github.io/sonore/gallery/cepstrum.html#d-c3) and [▶ harmonics only](https://choyun1.github.io/sonore/gallery/cepstrum.html#d-c4), and the same sentence as [▶ MFCCs](https://choyun1.github.io/sonore/gallery/cepstrum.html#d-m2).
- [Voices from harmonics](https://choyun1.github.io/sonore/gallery/harmonics.html): a voice rebuilt from a pitch track and a spectral envelope, from [▶ a buzz on the pitch track](https://choyun1.github.io/sonore/gallery/harmonics.html#d-hp2) to the [▶ full resynthesis](https://choyun1.github.io/sonore/gallery/harmonics.html#d-hu2), then [▶ up a fifth](https://choyun1.github.io/sonore/gallery/harmonics.html#d-hm2) with the formants kept.
- [Source, filter and aperiodicity](https://choyun1.github.io/sonore/gallery/aperiodicity.html): how much of a voice is noise, frequency by frequency, from [▶ a breathy vowel](https://choyun1.github.io/sonore/gallery/aperiodicity.html#d-ap3) whose aperiodicity is known to the sentence [▶ rebuilt by WORLD's synthesis](https://choyun1.github.io/sonore/gallery/aperiodicity.html#d-ap10) and [▶ whispered](https://choyun1.github.io/sonore/gallery/aperiodicity.html#d-ap13).
- [Changing a voice](https://choyun1.github.io/sonore/gallery/voice.html): pitch and formants moved separately, from [▶ a higher pitch](https://choyun1.github.io/sonore/gallery/voice.html#d-vc3) and [▶ higher formants](https://choyun1.github.io/sonore/gallery/voice.html#d-vc4) to [▶ one talker toward another](https://choyun1.github.io/sonore/gallery/voice.html#d-vc15), with the pitch and the envelope from different analyses.

**Spatial hearing**

- [Binaural cues](https://choyun1.github.io/sonore/gallery/binaural.html) [🎧](https://choyun1.github.io/sonore/gallery/binaural.html "Headphones required for binaural sounds"): [▶ timing alone](https://choyun1.github.io/sonore/gallery/binaural.html#d-10), and correlation that changes, such as [▶ Oscor](https://choyun1.github.io/sonore/gallery/binaural.html#d-07).
- [Synthetic reverberation](https://choyun1.github.io/sonore/gallery/reverb.html): rooms built from the statistics of real ones (Traer & McDermott, 2016), from a [▶ natural room](https://choyun1.github.io/sonore/gallery/reverb.html#d-17) to ones that break the rules, such as a [▶ time-reversed decay](https://choyun1.github.io/sonore/gallery/reverb.html#d-18).
- [Moving talkers](https://choyun1.github.io/sonore/gallery/moving.html) [🎧](https://choyun1.github.io/sonore/gallery/moving.html "Headphones required for binaural sounds"): three talkers rendered through measured HRIRs, [▶ one of them moving](https://choyun1.github.io/sonore/gallery/moving.html#d-m1), a talker [▶ walking in, in a room](https://choyun1.github.io/sonore/gallery/moving.html#d-m6), and a [▶ pass-by](https://choyun1.github.io/sonore/gallery/moving.html#d-m7).

## Conventions

- **Sounds.** `Sound` = immutable `(n_samples, n_channels)` float array + `fs`. Operations return new Sounds.
- **Arithmetic.** `a + b` mixes, `a * b` multiplies sample-wise, `2 * a` scales, mono broadcasts to stereo.
- **Bands and envelopes.** A filterbank's output (`Subbands`) is a collection of Sounds, and so is its fine structure (`.tfs()`). Envelopes are *not* sounds: `Envelope` and `Envelopes` are their own types, non-negative, often at a low sampling rate, and applied to sounds by multiplication. `Envelopes` (one envelope per band) is what the field calls a **cochleagram**. The Hilbert decomposition is literal: `sb == sb.envelopes() * sb.tfs()`.
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
with the routes back to sound they have (the channel vocoder, WORLD's
synthesis, the phase vocoder); `spatial` and `texture` are topics built on
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
| [`core.processing`](https://github.com/choyun1/sonore/blob/main/src/sonore/core/processing.py) | `pad`, `truncate`, `concat`, `mix`, `normalize`, `match_fs`, `match_channels`, `relative_db`, `bandpass`, `butter_filter`, `amplitude_modulate`, `resonator` and `antiresonator` (Klatt's formant and antiformant; frequency and bandwidth may glide, with no clicks) |
| [`sources.waveforms`](https://github.com/choyun1/sonore/blob/main/src/sonore/sources/waveforms.py) | `silence`, `pure_tone`, `harmonic_complex`, `schroeder_complex`, `square_wave`, `sawtooth_wave`, `pulse_train`, `linear_chirp`, `exponential_chirp`, `gaussian_noise`, `correlated_noise`, `iterated_ripple_noise`, `glottal_source` (Liljencrants-Fant glottal pulses on a fixed F0 or a contour, shape set by Fant's Rd, which may change over time; no aliasing), `lf_harmonics` (the pulse's Fourier coefficients in closed form), `lf_pulse` (one period, to draw) |
| [`sources.klatt`](https://github.com/choyun1/sonore/blob/main/src/sonore/sources/klatt.py) | `klatt_synthesize` (a Klatt-style cascade/parallel formant synthesizer: harmonic voicing with Klatt's glottal spectrum or LF pulses (`SS`, `RD`), aspiration and frication noise modulated at F0, nasal pole and zero, formants 1-5 in cascade and 1-6 in parallel, radiation; every parameter a number or a `(times, values)` track), `klatt_continuum` (evenly spaced parameter sets between two endpoints), `KLATT_DEFAULTS` |
| [`sources.ripples`](https://github.com/choyun1/sonore/blob/main/src/sonore/sources/ripples.py) | `Ripple`, `RippleSum`, `DynamicRipple`, `ripple_sound`; patterns can also be any function `f(t, x)` of time and octaves, and `pattern.render(filterbank, dur, fs)` gives their `Envelopes` |
| [`frames.frame`](https://github.com/choyun1/sonore/blob/main/src/sonore/frames/frame.py) | `Frame` (invertible analyses: `analyze`, `synthesize` as least squares, `frame_bounds`, `energy`, `adjoint`) |
| [`frames.filterbank`](https://github.com/choyun1/sonore/blob/main/src/sonore/frames/filterbank.py) | `Filterbank` (one class for every undecimated bank: a frequency scale, centers on it and a filter shape; canonical dual, tightness measured), `cosine_filterbank` (perfect-reconstruction cosine banks on the ERB, octave, mel or linear scale, any centers), `gammatone_filterbank` (exact gammatone responses, causal or zero-phase; `envelope_peak_delay` gives each filter's latency), `morlet_filterbank` (Morlet wavelets); all add edge filters by default so synthesis is exact on the whole band, and `edges=False` gives the bare bank for cochleagrams. `subbands`, `Subbands` (a collection of Sounds: `.envelopes()`, `.tfs()`, `.to_sound()`) |
| [`frames.gabor`](https://github.com/choyun1/sonore/blob/main/src/sonore/frames/gabor.py) | `GaborFrame` (the STFT as a frame; any window, zero-padded FFTs), `TVGaborFrame` (a Gabor frame whose window changes over time, from an explicit schedule, `from_function`, or `pitch_adaptive` from an F0 track; exact inverse; coefficients are a `TVSTFT`), `STFT` (a `GaborFrame` analysis: exact inverse, fast Griffin-Lim), `TVSTFT` (a `TVGaborFrame` analysis) |
| [`views.mask`](https://github.com/choyun1/sonore/blob/main/src/sonore/views/mask.py) | `Mask` (gains for the coefficients of an `STFT`, `TVSTFT` or `Subbands`; multiply and go back with `.to_sound()`), `ideal_binary_mask`, `ideal_ratio_mask` |
| [`views.view`](https://github.com/choyun1/sonore/blob/main/src/sonore/views/view.py) | `View` (the base of every view: a `discards` sentence saying what it drops, and a `synthesize` and a `to_sound` that raise `NotInvertibleError` with that reason and the route back to sound, if any; views with a canonical route back override `to_sound`) |
| [`views.spectrum`](https://github.com/choyun1/sonore/blob/main/src/sonore/views/spectrum.py) | `Spectrum` (`.to_sound` with a noise, a sound's phase or the minimum phase as carrier), `long_term_spectrum`, `tandem_power` (TANDEM-STRAIGHT-style pitch-adaptive power, after Kawahara et al., 2011; magnitude only, a `TFPower`) |
| [`views.reassigned`](https://github.com/choyun1/sonore/blob/main/src/sonore/views/reassigned.py) | `reassigned_spectrogram` (Kodera et al., 1978; Auger & Flandrin, 1995: spectrogram cells moved to their reassigned time and frequency, binned for display; not invertible) |
| [`views.envelopes`](https://github.com/choyun1/sonore/blob/main/src/sonore/views/envelopes.py) | `Envelope` (one envelope; `env * snd` modulates), `Envelopes` (one per band, i.e. a cochleagram; `.plot()`, `.modulation_spectrum()`, `env * subbands`); `noise_vocode` (the channel vocoder of cochlear-implant simulations, after Shannon et al., 1995: band envelopes, lowpassed at any cutoff, on a carrier of noise, tones at the band centres, or any sound) |
| [`views.modulation`](https://github.com/choyun1/sonore/blob/main/src/sonore/views/modulation.py) | `ConstantQModulationFilterbank`, `OctaveModulationFilterbank` (circular, analytic output optional), `HannModulationFilterbank` (Hann-windowed complex kernels of a whole number of cycles, defined in time: constant Q or one fixed window, centred or causal), `ModulationSpectrum` (linear-frequency from an STFT, or `.octave()` in cycles/octave; one made from envelopes can be edited with `.with_gain()` and heard with `.to_sound(carrier=...)`, the carrier supplying the modulation phase and fine structure; `.from_blobs()` draws a target from `ModulationBlob`s at a chosen rms depth) |
| [`views.modspectrogram`](https://github.com/choyun1/sonore/blob/main/src/sonore/views/modspectrogram.py) | `ModulationSpectrogram` (a modulation spectrum per time window: power, local mean and depth for every acoustic band and modulation rate, from any `Envelopes`, with a `valid` mask; `.plot()` as rate against time, one band, or band against time at one rate; `.at(t)`, `.slices(t)`, `.animate()`; not invertible) |
| [`views.cepstrum`](https://github.com/choyun1/sonore/blob/main/src/sonore/views/cepstrum.py) | `Cepstrum` (the real cepstrum of an `STFT` or `TVSTFT`: rectangular liftering with a fixed or per-time-window cutoff, the cepstral envelope, resynthesis with the original phase, exact when unliftered, or the minimum phase, and classic cepstral F0 after Noll, 1967) |
| [`views.mfcc`](https://github.com/choyun1/sonore/blob/main/src/sonore/views/mfcc.py) | `MFCC` (mel-frequency cepstral coefficients of a `Sound`, with the usual speech settings, or of any `STFT` or `TVSTFT`: HTK or Slaney mel, height- or area-normalised triangles straight in mel or in Hz, the mel spectrogram, deltas, the smoothed envelope the coefficients keep, `.plot()`; reproduces Kaldi's and librosa's numbers to rounding error; not invertible) |
| [`views.f0`](https://github.com/choyun1/sonore/blob/main/src/sonore/views/f0.py) | `f0_track` → `F0Track` (F0 every 5 ms with a voiced/unvoiced decision: candidates from YIN's difference function, refinement by the instantaneous frequency of six harmonics, a periodicity score, a Viterbi pass; `.plot()`; checked against laryngograph F0), `scale_f0` (a pitch change on any F0 contour, with a range factor) |
| [`views.spectral_envelope`](https://github.com/choyun1/sonore/blob/main/src/sonore/views/spectral_envelope.py) | `cheaptrick` → `SpectralEnvelope` (WORLD's CheapTrick, ported exactly: it matches WORLD to floating-point precision), `warp_frequency` (formants moved along frequency on any envelope or aperiodicity, by a ratio, a ratio over time or any frequency map), `GridEnvelope` (any envelope as power on a grid, as `Cepstrum.envelope_view()` and `MFCC.envelope_view()` give) |
| [`views.aperiodicity`](https://github.com/choyun1/sonore/blob/main/src/sonore/views/aperiodicity.py) | `d4c` → `Aperiodicity` (WORLD's D4C, ported exactly), `harmonic_aperiodicity` (the share of noise, by fitting the harmonics) |
| [`views.world`](https://github.com/choyun1/sonore/blob/main/src/sonore/views/world.py) | `world_synthesize` (WORLD's synthesis, sample for sample, from an F0 track, envelope and aperiodicity; WORLD's own noise stream by default, or fresh noise from an `rng`; reads any F0 track and any envelope), `DIFFERENCES_FROM_WORLD` |
| [`views.phasevocoder`](https://github.com/choyun1/sonore/blob/main/src/sonore/views/phasevocoder.py) | `time_stretch`, `pitch_shift` (identity phase locking), `pv_analyze` → `PVAnalysis` (instantaneous frequency; oscillator-bank `to_sound` with `time_scale` and `freq_map`) |
| [`spatial.binaural`](https://github.com/choyun1/sonore/blob/main/src/sonore/spatial/binaural.py) | `apply_itd_ild`, `simple_bir`, `interaural_cues`, `oscor`, `phasewarp` |
| [`spatial.spatialization`](https://github.com/choyun1/sonore/blob/main/src/sonore/spatial/spatialization.py) | `HRIRSet` (PKU-IOA, SOFA; onset-aligned interpolation), `spatialize`, `move_sound` (paths as functions of time, continuous ear delays, room tail), `hcc_trajectory` and other trajectories, coordinate conversions, `distance_gain_db` |
| [`spatial.hrir_data`](https://github.com/choyun1/sonore/blob/main/src/sonore/spatial/hrir_data.py) | `load_hrirs`: public HRIR databases (PKU-IOA) downloaded on first use, checksum-verified and cached |
| [`spatial.reverb`](https://github.com/choyun1/sonore/blob/main/src/sonore/spatial/reverb.py) | `synth_ir` (natural rooms, or the paper's atypical `decay_shape` / `rt60_profile` / `drr_profile` variants), `band_rt60s`, `measure_rt60` |
| [`texture.stats`](https://github.com/choyun1/sonore/blob/main/src/sonore/texture/stats.py) | `TextureModel`, `TextureStats` (`.measure`, `.snr`, `.replace` for hybrids, `.save`/`.load`) |
| [`texture.synth`](https://github.com/choyun1/sonore/blob/main/src/sonore/texture/synth.py) | `synthesize` (full loop), `impose_channel`; gradients in [`texture.grad`](https://github.com/choyun1/sonore/blob/main/src/sonore/texture/grad.py) |
| [`plotting`](https://github.com/choyun1/sonore/blob/main/src/sonore/plotting.py) | `overview` and the `plot_*` functions behind each object's `.plot()`; `plot_tf_db` draws any time-frequency level on non-uniform time windows; modulation spectrograms draw invalid cells grey and animate with their sound; cochleagrams take `align="peak"` (draw causal gammatone bands without their latency) and `fscale="linear"` (to match spectrograms) |

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

## Roadmap

What is planned comes first; finished work is listed at the end.

**Next**

- **Cocktail party scenes.** Gallery scenes on the Moving talkers page
  with three to six talkers walking on slowly bending paths, 10 to 30 s
  long, with speech from LibriSpeech dev-clean (CC BY 4.0).

**Texture synthesis**

- **Modulation convergence.** Rebalance the objective so modulation power
  converges: it reaches 30 dB SNR when imposed without the correlation
  classes, but 18-23 dB in full synthesis.
- **All channels at once.** Impose the channels jointly; the per-channel
  objective is overhead-bound (about 2 s per iteration for 5 s of sound).
- **Validation.** Run the MATLAB toolbox on the same original recordings
  and compare with its published examples.

**Other**

- The rest of KLSYN88's voice-quality controls (Klatt & Klatt, 1990) for the formant synthesizer:
  open quotient, spectral tilt, flutter, double pulsing, and its KLGLOTT88 source.
- Free-form modulation patterns: specify a modulation spectrum and synthesize it.
- A decimated, invertible constant-Q transform (nonstationary Gabor frames in frequency).
- Peak-based sinusoidal modeling (McAulay & Quatieri, 1986) alongside the channel oscillator bank.
- On-demand download of other public HRIR databases.
- A separate package for Bayesian inference of sound sources, built on
  sonore (JAX plus a probabilistic-programming layer), using sonore's
  sources, frames and texture statistics as its differentiable forward
  model.
- A block-by-block (streaming) modulation spectrogram, as the reference for a
  live version on a phone: the modulation spectrum of everyday sounds as they happen.

<details>
<summary><b>Done</b>, oldest first</summary>

- **Frames.** A `Frame` contract for invertible time-frequency
  analyses: `analyze`, `synthesize` (canonical dual, least-squares for
  modified coefficients), `frame_bounds()` and `adjoint`. The STFT
  (`GaborFrame`), the cosine, gammatone and Morlet filterbanks, and a
  time-varying Gabor frame with pitch-adaptive windows are all frames; tests
  enforce `synthesize(analyze(x)) == x` and the reported bounds. Reassigned
  spectrograms and a TANDEM-STRAIGHT-style power spectrum are drawn beside
  them in the [Seeing speech](https://choyun1.github.io/sonore/gallery/speech.html) page.
- **Package layout.** One subpackage per layer (`core`, `signals`,
  `analysis`, `stimuli`, `texture`), with imports pointing down a layer,
  enforced by `tests/test_layers.py`; see `docs/design/layout.md`.
- **CI and releases.** Tests and lint on Python 3.10 and 3.14 for every push
  and pull request, a check that the PyPI files build and pass their tests,
  and a trusted-publishing release workflow (`docs/releasing.md`).
- **HRIRs on demand.** `so.load_hrirs()` downloads the PKU-IOA database
  (Qu et al., 2009) on first use, checks each file's checksum and caches it,
  correcting the left-right mirroring of its SOFA copy; see
  `docs/design/spatial/hrir-data.md`.
- **Cepstrum.** `Cepstrum` on any STFT: liftering, resynthesis with the
  original or minimum phase, and classic cepstral F0; see
  `docs/design/views/cepstrum.md`.
- **The MSM archive.** The experiment code behind Cho & Kidd (2022), written
  with sigtools 0.1, stays a separate archive at
  [choyun1/MSM](https://github.com/choyun1/MSM) rather than being folded in;
  the Moving talkers page carries its stimuli forward.
- **JAX trial, decided against for now.** A JAX port of the texture channel
  objective matched the NumPy gradient to about 1e-15 but ran no faster
  (about 2 ms per call either way, plus compile time), and float32 would
  break bit-for-bit output. The core stays NumPy. An optional `sonore[jax]`
  extra is worth revisiting only if inference work needs gradients through
  the whole model.
- **PyPI, Zenodo and Colab.** sonore is on [PyPI](https://pypi.org/project/sonore/) from
  0.3.0, and each GitHub release is archived on Zenodo with a DOI (from
  0.3.1). A starter notebook runs in Colab with nothing to install.
- **Modulation spectrogram.** `ModulationSpectrogram`: how strongly each
  band's envelope is modulated at each rate, in every time window, with linked
  slices and an animation; see `docs/design/views/modulation-spectrogram.md` and
  the [Modulation spectrogram](https://choyun1.github.io/sonore/gallery/modspectrogram.html) gallery page.
- **Gallery pages.** Seventeen pages, listed under [Gallery](#gallery), each
  a runnable script shown with its code. The Cepstral analysis page is
  cross-checked against SciPy, MATLAB's `rceps` and Praat by
  `tools/crosscheck_cepstrum.py`; the Moving talkers page follows Cho & Kidd
  (2022), with interaural cues and a top-down view that follows playback.
- **Faster filterbanks.** FFT lengths padded to fast sizes and the large
  FFTs spread over all cores (`so.set_fft_workers`); subbands and envelopes
  are 3 to 4 times faster, and texture synthesis is unchanged bit for bit.
- **F0 tracking.** `so.f0_track`: YIN-style candidates refined by
  instantaneous frequency (after WORLD's StoneMask), a periodicity score and
  a Viterbi voicing decision. Against laryngograph reference F0 (the FDA
  database, Bagshaw et al., 1993) it gets the voicing of 5.6% (male) and
  1.5% (female) of time windows wrong, where WORLD's Harvest gets about 21%; see
  `docs/design/views/f0.md`.
- **Harmonic complexes on an F0 contour.** `so.harmonic_complex` takes an
  F0 contour as well as a number: the phase is the contour's exact running
  integral, unvoiced gaps are bridged and switched off with 5 ms ramps (or
  filled with noise), and harmonics fade out below `f_max` so a rising
  pitch never aliases. The square, sawtooth, pulse train and Schroeder
  complexes follow contours too. With `so.noise_vocode(snd, 16,
  carrier=...)` it puts a sound's band envelopes on harmonics that follow
  its own F0 track. The harmonic half of the pulse-plus-noise synthesis
  that WORLD's vocoder (below) completes; see `docs/design/sources/harmonic-source.md` and the
  [Voices from harmonics](https://choyun1.github.io/sonore/gallery/harmonics.html) gallery page.
- **Klatt-style formant synthesizer.** `so.klatt_synthesize` after Klatt
  (1980): harmonic voicing with Klatt's glottal spectrum, aspiration and
  frication noise, formants in cascade and in parallel (alternating signs,
  which match the cascade between peaks to 0.15 dB where equal signs miss
  by 15 dB), radiation, and parameters as tracks interpolated to every
  sample. `so.resonator` and `so.antiresonator` are its formants. A vowel's
  harmonics equal source x formants x radiation to 1e-6 dB; see
  `docs/design/sources/klatt.md` and the
  [Formant synthesis](https://choyun1.github.io/sonore/gallery/formants.html) gallery page.
- **WORLD vocoder.** `so.cheaptrick` (spectral envelope; Morise, 2015),
  `so.d4c` (aperiodicity; Morise, 2016) and `so.world_synthesize` reproduce
  WORLD (Morise et al., 2016, the successor of STRAIGHT, Kawahara et al.,
  1999) in NumPy, including its own noise generator: on the gallery
  sentence they match pyworld to 4e-9 dB, 7e-12 dB and 1e-13 of the peak,
  and the tests compare against stored WORLD output, so pyworld is not a
  dependency. Options that depart from WORLD are listed in
  `so.DIFFERENCES_FROM_WORLD`. `so.harmonic_aperiodicity` measures the
  share of noise directly, beside D4C. See `docs/design/views/world.md` and the
  [Source, filter and aperiodicity](https://choyun1.github.io/sonore/gallery/aperiodicity.html) gallery page.
- **LF glottal source.** `so.glottal_source` makes Liljencrants-Fant pulses
  (Fant, Liljencrants & Lin, 1985) from their exact harmonics, whose
  coefficients have a closed form that depends only on the harmonic number,
  so the source does not alias and each period takes its own length on a
  moving F0. One control, Fant's (1995) Rd, runs from tense to lax voice
  and may change over time. `so.klatt_synthesize` takes it with `SS = 3`
  and `RD`, as in KLSYN88 (Klatt & Klatt, 1990); the default source is
  unchanged. See `docs/design/sources/glottal-source.md`.
- **Moving-sound renderer.** `so.move_sound` takes a path as a function of
  time (`so.hcc_trajectory`, with any coordinate a number, a contour or a
  function, such as the azimuth swing of Cho & Kidd, 2022), a
  `(times, points)` pair, or evenly spread points. Each ear reads the sound
  through its own delay, the HRIR onset, which slides from sample to sample
  instead of being cross-faded between fixed delays, which comb-filters
  when distance changes; Doppler comes out of the same read. The PKU-IOA
  responses already hold travel time and 1/r level, and beyond the
  measured distances distance acts through both alone. An optional room
  tail keeps its level while the direct sound falls. See
  `docs/design/spatial/moving-sound.md`.
- **Moving sounds in the gallery.** The
  [Moving talkers](https://choyun1.github.io/sonore/gallery/moving.html)
  page has a talker walking in from 3 m, dry and in a room, and a buzz
  passing at 15 m/s whose measured pitch follows the Doppler shift.
- **MFCCs.** `so.MFCC` on a sound or any STFT: mel band powers, their log
  and a DCT, deltas, the mel spectrogram and the smoothed envelope the
  coefficients keep. Tests compare it with Kaldi's and librosa's stored output; see
  `docs/design/views/mfcc.md`. The
  [Cepstral analysis](https://choyun1.github.io/sonore/gallery/cepstrum.html#h-mfccs-a-cepstrum-on-the-mel-scale)
  page shows how much a vowel's MFCCs move with its pitch.
- **Voice changes, any method.** `so.scale_f0` changes the pitch and
  `so.warp_frequency` moves the formants, on any F0 contour (`f0_track`,
  Harvest, `Cepstrum.f0`) and any envelope (CheapTrick, the cepstrum,
  MFCCs), and both synthesizers take any envelope.
  `tools/compare_voice_methods.py` compares the trackers and envelopes at
  resynthesis and voice change; see `docs/design/views/voice-change.md` and the
  [Changing a voice](https://choyun1.github.io/sonore/gallery/voice.html) gallery page.
- **API reference and test layout.** An API reference built from the
  docstrings in CI, and a test folder that mirrors `src/sonore`.
- **Faster gallery build.** Each figure is drawn once rather than twice;
  the images are byte for byte the same.
- **More moving talkers and rooms.** Straight paths across the plane and a
  path no real source could take on the Moving talkers page, and the
  gallery sentence in each rule-breaking room on the
  [Synthetic reverberation](https://choyun1.github.io/sonore/gallery/reverb.html) page.
- **Gallery in four groups.** Stimuli; Seeing and changing sound; Voices;
  Spatial hearing, one script folder per group, with page URLs unchanged.
- **Frames and views.** `analysis` split into frames (invertible) and views
  (one-way), and a `View` base class whose `synthesize` raises
  `NotInvertibleError`, saying what the view discards and naming the route
  back to sound where one exists; see `docs/design/layout/reorganization.md`.
- **Sound first.** Folders follow meaning (`core`, `sources`, `frames`,
  `views`, `spatial`, `texture`), with import order kept module by module.
  A view goes back to sound through `to_sound` where a canonical route
  exists, taking what the view discarded (`Spectrum.to_sound` a carrier,
  `PVAnalysis.to_sound` a time scale and a frequency map), and refuses
  otherwise; see `docs/design/layout/sound-first.md`. Released as 0.4.0
  ([10.5281/zenodo.23114390](https://doi.org/10.5281/zenodo.23114390)).
</details>

## References

Each entry is the citation and a link to the work: the DOI where one is confirmed, otherwise
the publisher or another stable page. After it come tags naming the module(s) in
[What's in it](#whats-in-it) that implement or follow the work, linked to the source: a tag such
as `representations.reassigned_spectrogram` goes to that definition, a bare module name to the
whole file. Last, set apart by a ·, are the gallery pages (▶) and roadmap items that cite it.
Works with no tag are not implemented yet.

- Atlas & Shamma (2003). Joint acoustic and modulation frequency. *EURASIP J. Appl. Signal Processing* 2003(7). [doi:10.1155/S1110865703305013](https://doi.org/10.1155/S1110865703305013). [`modspectrogram.ModulationSpectrogram.at`](https://github.com/choyun1/sonore/blob/main/src/sonore/views/modspectrogram.py#L164) · [▶ Modulation spectrogram](https://choyun1.github.io/sonore/gallery/modspectrogram.html)
- Auger & Flandrin (1995). Improving the readability of time-frequency and time-scale representations by the reassignment method. *IEEE Trans. Signal Processing* 43(5). [doi:10.1109/78.382394](https://doi.org/10.1109/78.382394). [`reassigned.reassigned_spectrogram`](https://github.com/choyun1/sonore/blob/main/src/sonore/views/reassigned.py#L57) · [▶ Seeing speech](https://choyun1.github.io/sonore/gallery/speech.html)
- Bagshaw, Hiller & Jack (1993). Enhanced pitch tracking and the processing of F0 contours for computer aided intonation teaching. *Proc. Eurospeech 1993*. [CSTR, FDA database](https://www.cstr.ed.ac.uk/research/projects/fda/). Its laryngograph F0 is the reference in `tools/check_f0_fda.py` (data not in the repository). [`f0.f0_track`](https://github.com/choyun1/sonore/blob/main/src/sonore/views/f0.py#L80)
- Balazs, Dörfler, Jaillet, Holighaus & Velasco (2011). Theory, implementation and applications of nonstationary Gabor frames. *J. Comput. Appl. Math.* 236, 1481–1496. [doi:10.1016/j.cam.2011.09.011](https://doi.org/10.1016/j.cam.2011.09.011). [`gabor.TVGaborFrame`](https://github.com/choyun1/sonore/blob/main/src/sonore/frames/gabor.py#L212)
- Boersma & Weenink. Praat: doing phonetics by computer (computer program). [praat.org](https://www.praat.org). Cross-checks `Cepstrum` (see Reference implementations). · [▶ Cepstral analysis](https://choyun1.github.io/sonore/gallery/cepstrum.html)
- Bogert, Healy & Tukey (1963). The quefrency alanysis of time series for echoes: cepstrum, pseudo-autocovariance, cross-cepstrum and saphe cracking. In M. Rosenblatt (ed.), *Time Series Analysis*, Wiley. [Semantic Scholar](https://www.semanticscholar.org/paper/15bb1365026071ae3423d64ed2d18c554cafd6f6). [`cepstrum.Cepstrum`](https://github.com/choyun1/sonore/blob/main/src/sonore/views/cepstrum.py#L35) · [▶ Cepstral analysis](https://choyun1.github.io/sonore/gallery/cepstrum.html)
- Bohannon & Andrews (2011). Normal walking speed: a descriptive meta-analysis. *Physiotherapy* 97(3), 182–189. [doi:10.1016/j.physio.2010.12.004](https://doi.org/10.1016/j.physio.2010.12.004). The walking speeds of the cocktail-party talkers. · [▶ Moving talkers](https://choyun1.github.io/sonore/gallery/moving.html)
- Brandtsegg, Saue & Lazzarini (2018). Live convolution with time-varying filters. *Applied Sciences* 8(1), 103. [MDPI](https://www.mdpi.com/2076-3417/8/1/103). Reviews the ways of filtering with a changing filter; `move_sound` is one of them. [`spatialization.move_sound`](https://github.com/choyun1/sonore/blob/main/src/sonore/spatial/spatialization.py#L520) · [▶ Moving talkers](https://choyun1.github.io/sonore/gallery/moving.html) · [Roadmap](#roadmap)
- Brungart, Chang, Simpson & Wang (2006). Isolating the energetic component of speech-on-speech masking with ideal time-frequency segregation. *JASA* 120(6), 4007–4018. [doi:10.1121/1.2363929](https://doi.org/10.1121/1.2363929). The local criterion `lc_db`. [`mask.ideal_binary_mask`](https://github.com/choyun1/sonore/blob/main/src/sonore/views/mask.py#L119)
- Byrne et al. (1994). An international comparison of long-term average speech spectra. *JASA* 96(4), 2108–2120. [doi:10.1121/1.410152](https://doi.org/10.1121/1.410152). [`spectrum.long_term_spectrum`](https://github.com/choyun1/sonore/blob/main/src/sonore/views/spectrum.py#L118) · [▶ Classic stimuli](https://choyun1.github.io/sonore/gallery/classic.html)
- Chi, Gao, Guyton, Ru & Shamma (1999). Spectro-temporal modulation transfer functions and speech intelligibility. *JASA* 106. [JASA](https://pubs.aip.org/asa/jasa/article/106/5/2719/550617). [`ripples.Ripple`](https://github.com/choyun1/sonore/blob/main/src/sonore/sources/ripples.py#L84) [`modulation.ModulationSpectrum`](https://github.com/choyun1/sonore/blob/main/src/sonore/views/modulation.py#L333) · [▶ Spectrotemporal ripples](https://choyun1.github.io/sonore/gallery/ripples.html)
- Carlile & Leung (2016). The perception of auditory motion. *Trends in Hearing* 20. [doi:10.1177/2331216516644254](https://doi.org/10.1177/2331216516644254). Reviews which cues listeners use to judge motion; level and interaural differences outweigh Doppler. · [▶ Moving talkers](https://choyun1.github.io/sonore/gallery/moving.html)
- Cho & Kidd (2022). Auditory motion as a cue for source segregation and selection in a "cocktail party" listening environment. *JASA* 152(3), 1684–1694. [doi:10.1121/10.0013990](https://doi.org/10.1121/10.0013990). Its experiment code is archived at [choyun1/MSM](https://github.com/choyun1/MSM). [`spatialization.move_sound`](https://github.com/choyun1/sonore/blob/main/src/sonore/spatial/spatialization.py#L520) [`binaural.interaural_cues`](https://github.com/choyun1/sonore/blob/main/src/sonore/spatial/binaural.py#L105) · [▶ Moving talkers](https://choyun1.github.io/sonore/gallery/moving.html) · [Roadmap](#roadmap)
- Christensen (2003). *An Introduction to Frames and Riesz Bases*. Birkhäuser. [doi:10.1007/978-0-8176-8224-8](https://doi.org/10.1007/978-0-8176-8224-8). [`frame.Frame`](https://github.com/choyun1/sonore/blob/main/src/sonore/frames/frame.py#L49)
- Cuevas-Rodríguez, Picinali, González-Toledo et al. (2019). 3D Tune-In Toolkit: an open-source library for real-time binaural spatialisation. *PLOS ONE* 14(3), e0211899. [doi:10.1371/journal.pone.0211899](https://doi.org/10.1371/journal.pone.0211899). Removes the interaural delay before interpolating HRIRs, as sonore's onset alignment does. [`spatialization.HRIRSet`](https://github.com/choyun1/sonore/blob/main/src/sonore/spatial/spatialization.py#L175) · [▶ Moving talkers](https://choyun1.github.io/sonore/gallery/moving.html)
- Davis & Mermelstein (1980). Comparison of parametric representations for monosyllabic word recognition in continuously spoken sentences. *IEEE Trans. Acoust., Speech, Signal Process.* 28(4), 357–366. [`mfcc.MFCC`](https://github.com/choyun1/sonore/blob/main/src/sonore/views/mfcc.py#L99) · [▶ Cepstral analysis](https://choyun1.github.io/sonore/gallery/cepstrum.html) [▶ Changing a voice](https://choyun1.github.io/sonore/gallery/voice.html)
- Daubechies, Grossmann & Meyer (1986). Painless nonorthogonal expansions. *J. Math. Phys.* 27(5). [doi:10.1063/1.527388](https://doi.org/10.1063/1.527388). [`gabor.GaborFrame`](https://github.com/choyun1/sonore/blob/main/src/sonore/frames/gabor.py#L56)
- Dau, Kollmeier & Kohlrausch (1997). Modeling auditory processing of amplitude modulation. I. Detection and masking with narrow-band carriers. *JASA* 102(5), 2892–2905. [PubMed](https://pubmed.ncbi.nlm.nih.gov/9373976/). [`modulation.HannModulationFilterbank`](https://github.com/choyun1/sonore/blob/main/src/sonore/views/modulation.py#L158) · [▶ Modulation spectrogram](https://choyun1.github.io/sonore/gallery/modspectrogram.html)
- de Cheveigné & Kawahara (2002). YIN, a fundamental frequency estimator for speech and music. *JASA* 111(4). [doi:10.1121/1.1458024](https://doi.org/10.1121/1.1458024). [`f0.f0_track`](https://github.com/choyun1/sonore/blob/main/src/sonore/views/f0.py#L80) · [▶ Voices from harmonics](https://choyun1.github.io/sonore/gallery/harmonics.html)
- Dolson (1986). The phase vocoder: A tutorial. *Computer Music Journal* 10(4). [Semantic Scholar](https://www.semanticscholar.org/paper/31d9e1cc5d87c2b84cde2d4527b15b644544380e). [`phasevocoder`](https://github.com/choyun1/sonore/blob/main/src/sonore/views/phasevocoder.py) · [▶ Phase vocoder](https://choyun1.github.io/sonore/gallery/pv.html)
- Dorman, Loizou & Rainey (1997). Speech intelligibility as a function of the number of channels of stimulation for signal processors using sine-wave and noise-band outputs. *JASA* 102(4), 2403–2411. [doi:10.1121/1.420354](https://doi.org/10.1121/1.420354). [`envelopes.noise_vocode`](https://github.com/choyun1/sonore/blob/main/src/sonore/views/envelopes.py#L380) · [▶ Hearing through a vocoder](https://choyun1.github.io/sonore/gallery/vocoder.html)
- Elliott & Theunissen (2009). The modulation transfer function for speech intelligibility. *PLoS Comput. Biol.* 5(3), e1000302. [doi:10.1371/journal.pcbi.1000302](https://doi.org/10.1371/journal.pcbi.1000302). [`modulation.ModulationSpectrum.with_gain`](https://github.com/choyun1/sonore/blob/main/src/sonore/views/modulation.py#L498) · [▶ Drawing a modulation spectrum](https://choyun1.github.io/sonore/gallery/modtargets.html)
- Escabí & Schreiner (2002). Nonlinear spectrotemporal sound analysis by neurons in the auditory midbrain. *J. Neurosci.* 22. [doi:10.1523/JNEUROSCI.22-10-04114.2002](https://doi.org/10.1523/JNEUROSCI.22-10-04114.2002). [`ripples.DynamicRipple`](https://github.com/choyun1/sonore/blob/main/src/sonore/sources/ripples.py#L161) · [▶ Spectrotemporal ripples](https://choyun1.github.io/sonore/gallery/ripples.html)
- Fant (1995). The LF-model revisited. Transformations and frequency domain analysis. *STL-QPSR* 36(2–3), 119–156. The Rd parameter. [`waveforms.glottal_source`](https://github.com/choyun1/sonore/blob/main/src/sonore/sources/waveforms.py#L821) [`klatt.klatt_synthesize`](https://github.com/choyun1/sonore/blob/main/src/sonore/sources/klatt.py#L103) · [▶ Formant synthesis](https://choyun1.github.io/sonore/gallery/formants.html) · [Roadmap](#roadmap)
- Fant, Liljencrants & Lin (1985). A four-parameter model of glottal flow. *STL-QPSR* 26(4), 1–13. [`waveforms.lf_harmonics`](https://github.com/choyun1/sonore/blob/main/src/sonore/sources/waveforms.py#L745) [`waveforms.lf_pulse`](https://github.com/choyun1/sonore/blob/main/src/sonore/sources/waveforms.py#L776) · [▶ Formant synthesis](https://choyun1.github.io/sonore/gallery/formants.html) · [Roadmap](#roadmap)
- Flanagan & Golden (1966). Phase vocoder. *Bell System Technical Journal* 45. [doi:10.1002/j.1538-7305.1966.tb01706.x](https://doi.org/10.1002/j.1538-7305.1966.tb01706.x). [`phasevocoder`](https://github.com/choyun1/sonore/blob/main/src/sonore/views/phasevocoder.py) · [▶ Phase vocoder](https://choyun1.github.io/sonore/gallery/pv.html)
- Friesen, Shannon, Baskent & Wang (2001). Speech recognition in noise as a function of the number of spectral channels: comparison of acoustic hearing and cochlear implants. *JASA* 110(2), 1150–1163. [PubMed](https://pubmed.ncbi.nlm.nih.gov/11519582/). [`envelopes.noise_vocode`](https://github.com/choyun1/sonore/blob/main/src/sonore/views/envelopes.py#L380) · [▶ Hearing through a vocoder](https://choyun1.github.io/sonore/gallery/vocoder.html)
- Gabor (1946). Theory of communication. Part 1: The analysis of information. *J. IEE* 93(26). [doi:10.1049/ji-3-2.1946.0074](https://doi.org/10.1049/ji-3-2.1946.0074). [`gabor.GaborFrame`](https://github.com/choyun1/sonore/blob/main/src/sonore/frames/gabor.py#L56) · [▶ Seeing speech](https://choyun1.github.io/sonore/gallery/speech.html)
- Gamper (2013). Head-related transfer function interpolation in azimuth, elevation, and distance. *JASA* 134(6), EL547. [doi:10.1121/1.4828983](https://doi.org/10.1121/1.4828983). The HRIR interpolation used in Cho & Kidd (2022); sonore interpolates onset-aligned responses instead. [`spatialization.HRIRSet.at`](https://github.com/choyun1/sonore/blob/main/src/sonore/spatial/spatialization.py#L422)
- Glasberg & Moore (1990). Derivation of auditory filter shapes from notched-noise data. *Hearing Research* 47. [doi:10.1016/0378-5955(90)90170-T](https://doi.org/10.1016/0378-5955(90)90170-T). [`utils.freq_to_erb`](https://github.com/choyun1/sonore/blob/main/src/sonore/core/utils.py#L66) · [▶ Seeing speech](https://choyun1.github.io/sonore/gallery/speech.html) [▶ Classic stimuli](https://choyun1.github.io/sonore/gallery/classic.html)
- Gordon & Strawn (1985). An introduction to the phase vocoder. In J. Strawn (ed.), *Digital Audio Signal Processing: An Anthology*. Also Stanford CCRMA report STAN-M-55. [CCRMA](https://ccrma.stanford.edu/papers/introduction-phase-vocoder). [`phasevocoder`](https://github.com/choyun1/sonore/blob/main/src/sonore/views/phasevocoder.py) · [▶ Phase vocoder](https://choyun1.github.io/sonore/gallery/pv.html)
- Greenberg & Kingsbury (1997). The modulation spectrogram: in pursuit of an invariant representation of speech. *Proc. ICASSP 1997*, vol. 3, 1647–1650. [Semantic Scholar](https://www.semanticscholar.org/paper/71c0095d37084b6055a1abc8d4edcde3ef9f130b). [`modspectrogram.ModulationSpectrogram`](https://github.com/choyun1/sonore/blob/main/src/sonore/views/modspectrogram.py#L29) · [▶ Modulation spectrogram](https://choyun1.github.io/sonore/gallery/modspectrogram.html)
- Griffin & Lim (1984). Signal estimation from modified short-time Fourier transform. *IEEE TASSP* 32. [doi:10.1109/TASSP.1984.1164317](https://doi.org/10.1109/TASSP.1984.1164317). [`gabor.STFT.griffin_lim`](https://github.com/choyun1/sonore/blob/main/src/sonore/frames/gabor.py#L544) · [▶ Drawing a modulation spectrum](https://choyun1.github.io/sonore/gallery/modtargets.html)
- Hillenbrand, Getty, Clark & Wheeler (1995). Acoustic characteristics of American English vowels. *JASA* 97(5), 3099–3111. [doi:10.1121/1.411872](https://doi.org/10.1121/1.411872). Male and female formants, used for the synthetic vowels in `tools/compare_voice_methods.py`. · [▶ Source, filter and aperiodicity](https://choyun1.github.io/sonore/gallery/aperiodicity.html) [▶ Changing a voice](https://choyun1.github.io/sonore/gallery/voice.html)
- Hsu, Woolley, Fremouw & Theunissen (2004). Modulation power and phase spectrum of natural sounds enhance neural encoding performed by single auditory neurons. *J. Neurosci.* 24(41). [doi:10.1523/JNEUROSCI.2449-04.2004](https://doi.org/10.1523/JNEUROSCI.2449-04.2004). [`modulation.ModulationSpectrum.to_sound`](https://github.com/choyun1/sonore/blob/main/src/sonore/views/modulation.py#L596) · [▶ Drawing a modulation spectrum](https://choyun1.github.io/sonore/gallery/modtargets.html)
- Kawahara, Masuda-Katsuse & de Cheveigné (1999). Restructuring speech representations using a pitch-adaptive time-frequency smoothing and an instantaneous-frequency-based F0 extraction. *Speech Communication* 27. [doi:10.1016/S0167-6393(98)00085-5](https://doi.org/10.1016/S0167-6393(98)00085-5). [`gabor.TVGaborFrame.pitch_adaptive`](https://github.com/choyun1/sonore/blob/main/src/sonore/frames/gabor.py#L285) · [Roadmap](#roadmap)
- Kawahara et al. (2011). Technical foundations of TANDEM-STRAIGHT, a speech analysis, modification and synthesis framework. *Sādhanā* 36(5). [doi:10.1007/s12046-011-0043-3](https://doi.org/10.1007/s12046-011-0043-3). [`spectrum.tandem_power`](https://github.com/choyun1/sonore/blob/main/src/sonore/views/spectrum.py#L172) · [▶ Seeing speech](https://choyun1.github.io/sonore/gallery/speech.html)
- Kingsbury, Morgan & Greenberg (1998). Robust speech recognition using the modulation spectrogram. *Speech Communication* 25(1–3), 117–132. [doi:10.1016/S0167-6393(98)00032-6](https://doi.org/10.1016/S0167-6393(98)00032-6). [`modspectrogram.ModulationSpectrogram`](https://github.com/choyun1/sonore/blob/main/src/sonore/views/modspectrogram.py#L29) · [▶ Modulation spectrogram](https://choyun1.github.io/sonore/gallery/modspectrogram.html)
- Klatt (1980). Software for a cascade/parallel formant synthesizer. *JASA* 67(3). [doi:10.1121/1.383940](https://doi.org/10.1121/1.383940). [`klatt.klatt_synthesize`](https://github.com/choyun1/sonore/blob/main/src/sonore/sources/klatt.py#L103) [`processing.resonator`](https://github.com/choyun1/sonore/blob/main/src/sonore/core/processing.py#L230) · [▶ Formant synthesis](https://choyun1.github.io/sonore/gallery/formants.html) · [Roadmap](#roadmap)
- Klatt & Klatt (1990). Analysis, synthesis, and perception of voice quality variations among female and male talkers. *JASA* 87(2), 820–857. [doi:10.1121/1.398894](https://doi.org/10.1121/1.398894). The `SS` source switch. [`klatt.klatt_synthesize`](https://github.com/choyun1/sonore/blob/main/src/sonore/sources/klatt.py#L103) · [▶ Formant synthesis](https://choyun1.github.io/sonore/gallery/formants.html) · [Roadmap](#roadmap)
- Kodera, Gendrin & de Villedary (1978). Analysis of time-varying signals with small BT values. *IEEE Trans. ASSP* 26(1). [doi:10.1109/TASSP.1978.1163047](https://doi.org/10.1109/TASSP.1978.1163047). [`reassigned.reassigned_spectrogram`](https://github.com/choyun1/sonore/blob/main/src/sonore/views/reassigned.py#L57) · [▶ Seeing speech](https://choyun1.github.io/sonore/gallery/speech.html)
- Kominek & Black (2004). The CMU Arctic speech databases. *Proc. 5th ISCA Speech Synthesis Workshop (SSW5)*, 223–224. [ISCA Archive](https://www.isca-archive.org/ssw_2004/kominek04b_ssw.html). The gallery's speech (speakers bdl and rms; see `docs/speech/SOURCES.md`). · [▶ Seeing speech](https://choyun1.github.io/sonore/gallery/speech.html) [▶ Cepstral analysis](https://choyun1.github.io/sonore/gallery/cepstrum.html) [▶ Hearing through a vocoder](https://choyun1.github.io/sonore/gallery/vocoder.html) [▶ Moving talkers](https://choyun1.github.io/sonore/gallery/moving.html) [▶ Synthetic reverberation](https://choyun1.github.io/sonore/gallery/reverb.html) [▶ Phase vocoder](https://choyun1.github.io/sonore/gallery/pv.html) [▶ Classic stimuli](https://choyun1.github.io/sonore/gallery/classic.html) [▶ Modulation spectrogram](https://choyun1.github.io/sonore/gallery/modspectrogram.html) [▶ Voices from harmonics](https://choyun1.github.io/sonore/gallery/harmonics.html) [▶ Changing a voice](https://choyun1.github.io/sonore/gallery/voice.html)
- Kowalski, Depireux & Shamma (1996). Analysis of dynamic spectra in ferret primary auditory cortex. I. *J. Neurophysiol.* 76. [doi:10.1152/jn.1996.76.5.3503](https://doi.org/10.1152/jn.1996.76.5.3503). [`ripples.Ripple`](https://github.com/choyun1/sonore/blob/main/src/sonore/sources/ripples.py#L84) · [▶ Spectrotemporal ripples](https://choyun1.github.io/sonore/gallery/ripples.html)
- Laroche & Dolson (1999). Improved phase vocoder time-scale modification of audio. *IEEE Trans. Speech Audio Process.* 7(3). [IEEE Xplore](https://ieeexplore.ieee.org/document/759041/). [`phasevocoder.time_stretch`](https://github.com/choyun1/sonore/blob/main/src/sonore/views/phasevocoder.py#L175) · [▶ Phase vocoder](https://choyun1.github.io/sonore/gallery/pv.html)
- Licklider, Webster & Hedlun (1950). On the frequency limits of binaural beats. *JASA* 22(4), 468–473. [doi:10.1121/1.1906629](https://doi.org/10.1121/1.1906629). · [▶ Classic stimuli](https://choyun1.github.io/sonore/gallery/classic.html)
- McAulay & Quatieri (1986). Speech analysis/synthesis based on a sinusoidal representation. *IEEE TASSP* 34. [Internet Archive](https://archive.org/details/SpeechAnalysisSynthesisBasedOnASinusoidalRepresentation). · [Roadmap](#roadmap)
- McDermott & Simoncelli (2011). Sound texture perception via statistics of the auditory periphery. *Neuron* 71(5), 926–940. [doi:10.1016/j.neuron.2011.06.032](https://doi.org/10.1016/j.neuron.2011.06.032). [`texture`](https://github.com/choyun1/sonore/blob/main/src/sonore/texture/stats.py) [`filterbank.Cosine`](https://github.com/choyun1/sonore/blob/main/src/sonore/frames/filterbank.py#L193) [`modulation`](https://github.com/choyun1/sonore/blob/main/src/sonore/views/modulation.py) · [▶ Sound textures](https://choyun1.github.io/sonore/gallery/textures.html)
- Monson, Hunter & Story (2012). Horizontal directivity of low- and high-frequency energy in speech and singing. *JASA* 132(1), 433–441. [doi:10.1121/1.4725963](https://doi.org/10.1121/1.4725963). Talker directivity, which the cocktail-party scenes leave out. · [▶ Moving talkers](https://choyun1.github.io/sonore/gallery/moving.html)
- Morise (2015). CheapTrick, a spectral envelope estimator for high-quality speech synthesis. *Speech Communication* 67. [doi:10.1016/j.specom.2014.09.003](https://doi.org/10.1016/j.specom.2014.09.003). [`spectral_envelope.cheaptrick`](https://github.com/choyun1/sonore/blob/main/src/sonore/views/spectral_envelope.py#L217) [`gabor.TVGaborFrame.pitch_adaptive`](https://github.com/choyun1/sonore/blob/main/src/sonore/frames/gabor.py#L285) · [▶ Cepstral analysis](https://choyun1.github.io/sonore/gallery/cepstrum.html) [▶ Voices from harmonics](https://choyun1.github.io/sonore/gallery/harmonics.html) [▶ Source, filter and aperiodicity](https://choyun1.github.io/sonore/gallery/aperiodicity.html) [▶ Changing a voice](https://choyun1.github.io/sonore/gallery/voice.html) · [Roadmap](#roadmap)
- Morise (2016). D4C, a band-aperiodicity estimator for high-quality speech synthesis. *Speech Communication* 84, 57–65. [doi:10.1016/j.specom.2016.09.001](https://doi.org/10.1016/j.specom.2016.09.001). [`aperiodicity.d4c`](https://github.com/choyun1/sonore/blob/main/src/sonore/views/aperiodicity.py#L119) · [▶ Source, filter and aperiodicity](https://choyun1.github.io/sonore/gallery/aperiodicity.html) [▶ Changing a voice](https://choyun1.github.io/sonore/gallery/voice.html) · [Roadmap](#roadmap)
- Morise (2017). Harvest: a high-performance fundamental frequency estimator from speech signals. *Proc. Interspeech 2017*. [doi:10.21437/Interspeech.2017-68](https://doi.org/10.21437/Interspeech.2017-68). Compared with `f0_track` in `tools/check_f0_fda.py`. · [▶ Voices from harmonics](https://choyun1.github.io/sonore/gallery/harmonics.html) [▶ Source, filter and aperiodicity](https://choyun1.github.io/sonore/gallery/aperiodicity.html) [▶ Changing a voice](https://choyun1.github.io/sonore/gallery/voice.html)
- Morise, Yokomori & Ozawa (2016). WORLD: A vocoder-based high-quality speech synthesis system for real-time applications. *IEICE Trans. Inf. & Syst.* E99-D(7). [doi:10.1587/transinf.2015EDP7457](https://doi.org/10.1587/transinf.2015EDP7457). Its synthesis is reproduced by `world_synthesize`, and its StoneMask refinement followed by `f0_track`. [`world.world_synthesize`](https://github.com/choyun1/sonore/blob/main/src/sonore/views/world.py#L233) [`f0.f0_track`](https://github.com/choyun1/sonore/blob/main/src/sonore/views/f0.py#L80) · [▶ Cepstral analysis](https://choyun1.github.io/sonore/gallery/cepstrum.html) [▶ Voices from harmonics](https://choyun1.github.io/sonore/gallery/harmonics.html) [▶ Source, filter and aperiodicity](https://choyun1.github.io/sonore/gallery/aperiodicity.html) [▶ Changing a voice](https://choyun1.github.io/sonore/gallery/voice.html) · [Roadmap](#roadmap)
- Noll (1967). Cepstrum pitch determination. *JASA* 41(2). [PubMed](https://pubmed.ncbi.nlm.nih.gov/6040805/). [`cepstrum.Cepstrum.f0`](https://github.com/choyun1/sonore/blob/main/src/sonore/views/cepstrum.py#L179) · [▶ Cepstral analysis](https://choyun1.github.io/sonore/gallery/cepstrum.html) [▶ Voices from harmonics](https://choyun1.github.io/sonore/gallery/harmonics.html) [▶ Changing a voice](https://choyun1.github.io/sonore/gallery/voice.html)
- O'Shaughnessy (2000). *Speech Communications: Human and Machine*, 2nd ed. IEEE Press. Eq. 4.2, p. 128: the mel formula `2595 log10(1 + f/700)`, given without an earlier source. [`utils.freq_to_mel`](https://github.com/choyun1/sonore/blob/main/src/sonore/core/utils.py#L89)
- Oppenheim & Schafer (2010). *Discrete-Time Signal Processing*, 3rd ed., ch. 13. Pearson. [Pearson](https://www.pearson.com/en-us/subject-catalog/p/Oppenheim-Discrete-Time-Signal-Processing-3rd-Edition/P200000003226). [`cepstrum.Cepstrum.to_stft`](https://github.com/choyun1/sonore/blob/main/src/sonore/views/cepstrum.py#L150)
- Panayotov, Chen, Povey & Khudanpur (2015). LibriSpeech: an ASR corpus based on public domain audio books. *Proc. IEEE ICASSP 2015*, 5206–5210. [doi:10.1109/ICASSP.2015.7178964](https://doi.org/10.1109/ICASSP.2015.7178964). The cocktail-party passages (CC BY 4.0; see `docs/speech/SOURCES.md`). · [▶ Moving talkers](https://choyun1.github.io/sonore/gallery/moving.html)
- Patterson, Robinson, Holdsworth, McKeown, Zhang & Allerhand (1992). Complex sounds and auditory images. In *Auditory Physiology and Perception* (Proc. 9th International Symposium on Hearing), Pergamon, 429–446. [doi:10.1016/B978-0-08-041847-6.50054-X](https://doi.org/10.1016/B978-0-08-041847-6.50054-X). [`filterbank.Gammatone`](https://github.com/choyun1/sonore/blob/main/src/sonore/frames/filterbank.py#L276)
- Perraudin, Balazs & Søndergaard (2013). A fast Griffin-Lim algorithm. *IEEE WASPAA*. [doi:10.1109/WASPAA.2013.6701851](https://doi.org/10.1109/WASPAA.2013.6701851). [`gabor.STFT.griffin_lim`](https://github.com/choyun1/sonore/blob/main/src/sonore/frames/gabor.py#L544)
- Peterson & Barney (1952). Control methods used in a study of the vowels. *JASA* 24(2), 175–184. [ASA](https://pubs.aip.org/asa/jasa/article/24/2/175/722376/Control-Methods-Used-in-a-Study-of-the-Vowels). Average formant frequencies of ten vowels. · [▶ Formant synthesis](https://choyun1.github.io/sonore/gallery/formants.html) [▶ Cepstral analysis](https://choyun1.github.io/sonore/gallery/cepstrum.html)
- Plomp & Levelt (1965). Tonal consonance and critical bandwidth. *JASA* 38(4), 548–560. [doi:10.1121/1.1909741](https://doi.org/10.1121/1.1909741). · [▶ Classic stimuli](https://choyun1.github.io/sonore/gallery/classic.html)
- Portilla & Simoncelli (2000). A parametric texture model based on joint statistics of complex wavelet coefficients. *Int. J. Computer Vision* 40(1), 49–71. Its radial filters are the cosine filters on a log2 scale. [`filterbank.Cosine`](https://github.com/choyun1/sonore/blob/main/src/sonore/frames/filterbank.py#L193)
- Qu et al. (2009). Distance-dependent head-related transfer functions measured with high spatial resolution using a spark gap. *IEEE TASLP* 17. [PKU Scholar](http://scholar.pku.edu.cn/qutianshu/publications/distance-dependent-head-related-transfer-functions-measured-high-spatial). [`spatialization.HRIRSet.from_pku_ioa`](https://github.com/choyun1/sonore/blob/main/src/sonore/spatial/spatialization.py#L212) [`hrir_data.load_hrirs`](https://github.com/choyun1/sonore/blob/main/src/sonore/spatial/hrir_data.py#L122) · [▶ Moving talkers](https://choyun1.github.io/sonore/gallery/moving.html)
- Schroeder (1970). Synthesis of low-peak-factor signals and binary sequences with low autocorrelation. *IEEE Trans. Inf. Theory* 16. [doi:10.1109/TIT.1970.1054411](https://doi.org/10.1109/TIT.1970.1054411). [`waveforms.schroeder_complex`](https://github.com/choyun1/sonore/blob/main/src/sonore/sources/waveforms.py#L374) · [▶ Voices from harmonics](https://choyun1.github.io/sonore/gallery/harmonics.html)
- Shannon et al. (1995). Speech recognition with primarily temporal cues. *Science* 270. [doi:10.1126/science.270.5234.303](https://doi.org/10.1126/science.270.5234.303). [`envelopes.noise_vocode`](https://github.com/choyun1/sonore/blob/main/src/sonore/views/envelopes.py#L380) · [▶ Hearing through a vocoder](https://choyun1.github.io/sonore/gallery/vocoder.html) [▶ Voices from harmonics](https://choyun1.github.io/sonore/gallery/harmonics.html)
- Simoncelli & Freeman (1995). The steerable pyramid: a flexible architecture for multi-scale derivative computation. *Proc. 2nd IEEE Int. Conf. Image Processing*, vol. III, 444–447. Squared responses summing to one, its "flat system response". [`filterbank.Cosine`](https://github.com/choyun1/sonore/blob/main/src/sonore/frames/filterbank.py#L193)
- Singh & Theunissen (2003). Modulation spectra of natural sounds and ethological theories of auditory processing. *JASA* 114(6). [doi:10.1121/1.1624067](https://doi.org/10.1121/1.1624067). [`modulation.ModulationSpectrum`](https://github.com/choyun1/sonore/blob/main/src/sonore/views/modulation.py#L333) [`envelopes.Envelopes.modulation_spectrum`](https://github.com/choyun1/sonore/blob/main/src/sonore/views/envelopes.py#L347) · [▶ Spectrotemporal ripples](https://choyun1.github.io/sonore/gallery/ripples.html) [▶ Drawing a modulation spectrum](https://choyun1.github.io/sonore/gallery/modtargets.html)
- Siveke et al. (2008). Psychophysical and physiological evidence for fast binaural processing. *J. Neurosci.* 28. [J. Neurosci.](https://www.jneurosci.org/content/28/9/2043). [`binaural.oscor`](https://github.com/choyun1/sonore/blob/main/src/sonore/spatial/binaural.py#L180) [`binaural.phasewarp`](https://github.com/choyun1/sonore/blob/main/src/sonore/spatial/binaural.py#L189) · [▶ Binaural cues](https://choyun1.github.io/sonore/gallery/binaural.html)
- Slaney (1993). An efficient implementation of the Patterson-Holdsworth auditory filter bank. Apple Computer Technical Report #35. Gives `b = 1.019 ERB` as Patterson's recommendation for the fourth-order gammatone. [`filterbank.Gammatone`](https://github.com/choyun1/sonore/blob/main/src/sonore/frames/filterbank.py#L276)
- Srinivasan, Roman & Wang (2006). Binary and ratio time-frequency masks for robust speech recognition. *Speech Communication* 48(11), 1486–1501. [doi:10.1016/j.specom.2006.09.003](https://doi.org/10.1016/j.specom.2006.09.003). [`mask.ideal_ratio_mask`](https://github.com/choyun1/sonore/blob/main/src/sonore/views/mask.py#L127)
- Traer & McDermott (2016). Statistics of natural reverberation enable perceptual separation of sound and space. *PNAS* 113. [doi:10.1073/pnas.1612524113](https://doi.org/10.1073/pnas.1612524113). [`reverb.synth_ir`](https://github.com/choyun1/sonore/blob/main/src/sonore/spatial/reverb.py#L77) · [▶ Synthetic reverberation](https://choyun1.github.io/sonore/gallery/reverb.html)
- Wang (2005). On ideal binary mask as the computational goal of auditory scene analysis. In *Speech Separation by Humans and Machines*. [doi:10.1007/0-387-22794-6_12](https://doi.org/10.1007/0-387-22794-6_12). [`mask.ideal_binary_mask`](https://github.com/choyun1/sonore/blob/main/src/sonore/views/mask.py#L119) · [▶ Analysis and resynthesis](https://choyun1.github.io/sonore/gallery/resynthesis.html)
- Wang, Narayanan & Wang (2014). On training targets for supervised speech separation. *IEEE/ACM Trans. Audio, Speech, Lang. Process.* 22(12), 1849–1858. The ratio mask with `beta = 0.5`. [`mask.ideal_ratio_mask`](https://github.com/choyun1/sonore/blob/main/src/sonore/views/mask.py#L127)
- Wilson, Finley, Lawson, Wolford, Eddington & Rabinowitz (1991). Better speech recognition with cochlear implants. *Nature* 352, 236–238. [PubMed](https://pubmed.ncbi.nlm.nih.gov/1857418/). [`envelopes.noise_vocode`](https://github.com/choyun1/sonore/blob/main/src/sonore/views/envelopes.py#L380) · [▶ Hearing through a vocoder](https://choyun1.github.io/sonore/gallery/vocoder.html)
- Yost (1996). Pitch of iterated rippled noise. *JASA* 100. [JASA (PDF)](https://pubs.aip.org/asa/jasa/article-pdf/100/1/511/11401642/511_1_online.pdf). [`waveforms.iterated_ripple_noise`](https://github.com/choyun1/sonore/blob/main/src/sonore/sources/waveforms.py#L567) · [▶ Iterated rippled noise](https://choyun1.github.io/sonore/gallery/irn.html)

### Reference implementations

Implementations by a paper's authors or widely used ports, with how sonore
relates to each. "Cross-checked" means a script in `tools/` compares the two
numerically; "consulted" means the code was read for behavior but not copied.

- [Sound Texture Synthesis Toolbox v1.7](https://mcdermottlab.mit.edu/downloads.html) (MATLAB), McDermott lab: the
  authors' implementation of McDermott & Simoncelli (2011). Consulted; sonore
  is a clean-room implementation from the paper, and every deliberate
  difference is listed in `so.texture.DIFFERENCES_FROM_TOOLBOX`. [`texture`](https://github.com/choyun1/sonore/blob/main/src/sonore/texture/stats.py) [`filterbank.Cosine`](https://github.com/choyun1/sonore/blob/main/src/sonore/frames/filterbank.py#L193) [`modulation`](https://github.com/choyun1/sonore/blob/main/src/sonore/views/modulation.py)
- [wil-j-wil/texture_stats](https://github.com/wil-j-wil/texture_stats) (Python, MIT): a port of the toolbox's
  statistics. Cross-checked by `tools/crosscheck_texture_stats.py`. [`texture.TextureStats`](https://github.com/choyun1/sonore/blob/main/src/sonore/texture/stats.py#L196)
- [mcdermottLab/pycochleagram](https://github.com/mcdermottLab/pycochleagram) (Python): the lab's port of the
  toolbox's cochleagram code, including the cosine filterbank. Not yet
  cross-checked. [`filterbank.Cosine`](https://github.com/choyun1/sonore/blob/main/src/sonore/frames/filterbank.py#L193)
- [LTFAT](https://ltfat.github.io/) (MATLAB/Octave, GPLv3): `frsynabs` with `'fgriflim'` is the fast Griffin-Lim from
  the group of Perraudin, Balazs & Søndergaard (2013);
  [`librosa.griffinlim`](https://librosa.org/doc/latest/generated/librosa.griffinlim.html) is a widely used Python
  version. Neither is cross-checked yet. [`gabor.STFT.griffin_lim`](https://github.com/choyun1/sonore/blob/main/src/sonore/frames/gabor.py#L544)
- [SciPy `ShortTimeFFT`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.ShortTimeFFT.html)
  (BSD-3): wrapped by `GaborFrame`. Its frame operator, bounds and least-squares
  inverse are cross-checked against dense matrices in the tests and in
  `tools/check_frames_step1_claims.py` (docs/design/frames/frames.md, step 1). [`gabor.GaborFrame`](https://github.com/choyun1/sonore/blob/main/src/sonore/frames/gabor.py#L56) [`gabor.STFT`](https://github.com/choyun1/sonore/blob/main/src/sonore/frames/gabor.py#L465)
- Gammatone filterbanks in Slaney's Auditory Toolbox and MATLAB's `gammatoneFilterBank` are time-domain IIR
  approximations; `gammatone_filterbank` uses the exact frequency response instead (derivation in
  docs/design/frames/frames.md, step 2). Consulted for conventions only. [`filterbank.Gammatone`](https://github.com/choyun1/sonore/blob/main/src/sonore/frames/filterbank.py#L276)
- [SciPy `minimum_phase`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.minimum_phase.html)
  (homomorphic method), the real-cepstrum definition MATLAB's `rceps` documents, and Praat's
  PowerCepstrogram through [parselmouth](https://github.com/YannickJadoul/Parselmouth) (GPLv3):
  cross-checked by `tools/crosscheck_cepstrum.py`, Praat at development time only. [`cepstrum.Cepstrum`](https://github.com/choyun1/sonore/blob/main/src/sonore/views/cepstrum.py#L35)
- Kaldi's `compute-mfcc-feats`, through [kaldi-native-fbank](https://github.com/csukuangfj/kaldi-native-fbank)
  (Apache-2.0), a C++ re-implementation of Kaldi's feature code: its MFCCs and log mel energies are stored by
  `tools/make_kaldi_fixtures.py`, and the tests compare `MFCC` with them to float32 precision (no DC removal,
  pre-emphasis or energy, which sonore leaves to the sound). The primary reference, standing in for HTK,
  whose download site was unreachable. [`mfcc.MFCC`](https://github.com/choyun1/sonore/blob/main/src/sonore/views/mfcc.py#L99)
- [librosa](https://librosa.org/) (ISC) `feature.mfcc`, `feature.melspectrogram` and `feature.delta`: their
  output for three settings is stored by `tools/make_mfcc_fixtures.py`, and the tests compare `MFCC` with it
  (mel power to 3e-7, coefficients to 1e-8, both relative to the largest value). `tools/crosscheck_mfcc.py` also
  reproduces [python_speech_features](https://github.com/jameslyons/python_speech_features) 0.6 exactly. Both
  are development-time only. [`mfcc.MFCC`](https://github.com/choyun1/sonore/blob/main/src/sonore/views/mfcc.py#L99)
- [LTFAT](https://ltfat.github.io/) (GPLv3) and [nsgt](https://github.com/grrrr/nsgt) (Artistic License 2.0):
  frame theory in code, for dev-time cross-checks only because of their licenses. Not yet cross-checked. [`frame`](https://github.com/choyun1/sonore/blob/main/src/sonore/frames/frame.py)

## Migrating from sigtools

sonore was previously `sigtools`, renamed to avoid a clash with an unrelated
PyPI package of that name. Version 0.2 also redesigned the API:

| sigtools 0.1 | sonore |
|---|---|
| `from sigtools.sounds import *` etc. | `import sonore as so` |
| `PureTone(dur, fs, f)`, `GaussianNoise(...)`, ... | `so.pure_tone(dur, fs, f)`, `so.gaussian_noise(...)`, ... |
| `GaussianNoise(dur, fs, lo, hi, tilt)` | `so.gaussian_noise(dur, fs, band=(lo, hi), tilt=...)`; tilt is now dB/octave |
| `SchroederPhase(dur, fs, f0, n)` | `so.schroeder_complex(dur, fs, f0, n)` |
| `SoundLoader(path)`, `Silence(dur, fs)` | `so.load(path)`, `so.silence(dur, fs)` |
| `snd + 6` (dB gain) | `snd + 6*dB` |
| `snd.make_binaural()`, `snd.extract_envelope()` | `snd.to_stereo()`, `snd.envelope()` (now returns an `Envelope`, not a Sound) |
| `ramp_edges(snd, d)` | `snd.ramp(d)` |
| `butter_bandpass_filter(snd, lo, hi)` | `so.bandpass(snd, lo, hi)` (no longer RMS-normalizes) |
| `equalize_fs`, `zeropad_sounds`, `center_sounds`, `truncate_sounds` | `so.match_fs`, `so.match_lengths(align="start"/"center")`, `so.match_lengths(mode="truncate")` |
| `normalize_rms`, `zero_mean`, `concat_sounds`, `compare_relative_db` | `so.normalize`, `snd.zero_mean()`, `so.concat`, `so.relative_db` |
| `sum(zeropad_sounds([a, b]))` | `so.mix([a, b])` |
| `MagnitudeSpectrum(s).to_Noise(dur, fs)` | `so.long_term_spectrum(s).to_sound(dur, fs)` |
| `STFT(snd, win)`, `S.to_Sound()`, `method="GLA"` | `so.STFT(snd, win)`, `S.to_sound()`, `S.griffin_lim()` |
| `IBM = S_t > S_m + lc`; `IBM * S_mix` | `so.ideal_binary_mask(S_t, S_m, lc_db=lc)`; `S_mix * mask` |
| `Subbands(snd, n)`, `.extract_envelopes()`, `.to_Sound()` | `so.cosine_filterbank(n).analyze(snd)`, `.envelopes()`, `.to_sound()` |
| `InterauralCues(snd, win)` | `so.interaural_cues(snd, win)` |
| `SimpleBIR(fs, itd, ild)` | `so.simple_bir(fs, itd, ild)` or `so.apply_itd_ild(snd, itd, ild)` |
| `SynthIR(drr, rt60, dB_thresh, fs)` | `so.synth_ir(rt60, fs, drr_db=..., decay_db=-dB_thresh)` |
| `move_sound(traj, snd)` | `so.move_sound(snd, traj, hrirs)` with `so.load_hrirs()` (downloads PKU-IOA), `so.HRIRSet.from_pku_ioa(dir)` or `.from_sofa(path)` |
| `display_STFT(x, S)`, `AudioControl(snd).display()` | `so.overview(x)`; put `snd` at the end of a cell |

Results computed with 0.1 can differ, because these 0.1 bugs were fixed:
spectrum and STFT "dB" were half the true value; the bandpass filter filtered
stereo across channels; `SimpleBIR` was a sample short and got louder with
larger ITDs; `SynthIR`'s DRR had no effect and its resynthesis filters were
shifted in frequency; tone frequencies were off by a factor of `(n-1)/n`;
`move_sound` summed ~100 unwindowed overlapping convolutions per sample; and IAC
was never computed. The ILD in `apply_itd_ild` is now split ±ILD/2 across the
ears (0.1 applied it to the right ear only).

## Development

```bash
pip install -e ".[dev]"
pytest                             # ~40 s; one test file per module
ruff check . && ruff format .
python docs/gallery/build.py       # regenerate the listening gallery (a few minutes)
```

## How sonore was developed

sonore began as sigtools, the code I (Adrian Cho) wrote in graduate school to
make psychoacoustic stimuli. The 0.2 redesign and everything since were
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

**How it is verified.** I have not read every line by hand. What I rely on
instead is the following:

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

MIT; see [LICENSE](https://github.com/choyun1/sonore/blob/main/LICENSE). If sonore is useful in your research, please cite
it using [CITATION.cff](https://github.com/choyun1/sonore/blob/main/CITATION.cff); the
*Cite this repository* button in the GitHub sidebar gives the same citation in APA and BibTeX.

Every release is archived on Zenodo. [10.5281/zenodo.23086165](https://doi.org/10.5281/zenodo.23086165)
always points to the latest version; each version also has its own DOI, listed on that page
(0.4.0 is [10.5281/zenodo.23114390](https://doi.org/10.5281/zenodo.23114390), 0.3.1 is
[10.5281/zenodo.23086166](https://doi.org/10.5281/zenodo.23086166)). Cite the version you used.
