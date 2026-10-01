# sonore

**Signals and stimuli for auditory research, built for Jupyter.**

**[▶ Listen to the gallery](https://choyun1.github.io/sonore/gallery/)**: every sound in this README and more, each
playable next to its plots, with a playhead that follows the sound.

sonore is a small Python library for making, manipulating, and analyzing sounds
the way hearing scientists think about them. Its analysis and synthesis tools
cover tones, harmonic complexes, shaped and correlated noises, ERB-spaced
subbands, invertible spectrograms, a phase vocoder, interaural cues, HRIR
spatialization of moving sources, synthetic room reverberation, and sound
texture synthesis. Levels are written as levels (`snd + 6*dB`), times as
seconds (`snd[0.1:0.5]`), and any sound at the end of a notebook cell plays.

The name comes from Pierre Schaeffer's *objet sonore*, the "sound object": a
sound taken as a thing in its own right and studied for how it is heard rather
than for what produced it. The `Sound` object at the center of this library is
meant in the same spirit.

![Overview of an iterated rippled noise](https://raw.githubusercontent.com/choyun1/sonore/main/docs/images/overview_irn.png)

## What it's for

- **Psychophysical stimuli.** Pure tones, harmonic complexes with any phase
  scheme (cosine, sine, alternating, random, Schroeder±), band-limited square,
  sawtooth and pulse trains, chirps, band-limited and spectrally tilted noise,
  iterated rippled noise. Everything is reproducible from a seed.
- **Binaural and spatial hearing.** Exact fractional ITDs, ILDs, interaurally
  correlated noise, Oscor and Phasewarp, windowed ITD/ILD/coherence analysis
  (broadband or per band), and rendering of static or moving sources through
  measured HRIRs (PKU-IOA, downloaded on first use, or any SOFA file).
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

sonore is not an experiment runner and does not calibrate to dB SPL; see
"Related projects" below for those.

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

The [recipes notebook](https://github.com/choyun1/sonore/blob/main/notebooks/recipes.ipynb) has worked examples of
everything below, including speech-shaped noise, moving sources with reverb,
and the classic binaural stimuli.

## Gallery

These are a selection from the [listening gallery](https://choyun1.github.io/sonore/gallery/); the ▶ links play
each sound there. The `so.overview` of iterated rippled noise at the top is
[▶ iterated rippled noise](https://choyun1.github.io/sonore/gallery/#d-09).

**Ideal binary mask.** A gliding harmonic target at −5 dB SNR, the IBM computed
from the separate STFTs, and the masked mixture resynthesized.
[▶ mixture](https://choyun1.github.io/sonore/gallery/#d-24) [▶ masked](https://choyun1.github.io/sonore/gallery/#d-25)

![Ideal binary mask](https://raw.githubusercontent.com/choyun1/sonore/main/docs/images/ibm.png)

```python
S_t, S_m, S_x = (so.STFT(s, 25e-3) for s in (target, masker, target + masker))
separated = (S_x * so.ideal_binary_mask(S_t, S_m, lc_db=0)).to_sound()
```

**Oscor and Phasewarp** (Siveke et al., 2008). The zero-lag interaural
correlation swings between +1 and −1 at the modulation rate (3 and 2 Hz here).
Headphones needed. [▶ Oscor](https://choyun1.github.io/sonore/gallery/#d-07) [▶ Phasewarp](https://choyun1.github.io/sonore/gallery/#d-08)

![Binaural cues of Oscor and Phasewarp](https://raw.githubusercontent.com/choyun1/sonore/main/docs/images/binaural_cues.png)

**ERB filterbank with perfect reconstruction.** An exponential sweep split
into 6 ERB-spaced bands, plus the lowpass and highpass edge filters that make
the bank power-complementary, then summed back together. The reconstruction
error is at the level of floating-point rounding. [▶ reconstruction](https://choyun1.github.io/sonore/gallery/#d-26)

![Filterbank decomposition and reconstruction](https://raw.githubusercontent.com/choyun1/sonore/main/docs/images/filterbank.png)

```python
sb = so.subbands(sweep, n_bands=6, f_lo=100, f_hi=6000)  # 8 filters: 6 bands + 2 edges
sb.plot()  # stacked waveforms, shared scale
reconstructed = sb.synthesize()  # == sweep to ~1e-15
```

**Spectrotemporal ripples.** A pattern is an object you can add, plot, and
inspect before any sound exists. Top: the patterns as specified. Middle: the
synthesized sounds' subband envelopes. Bottom: their measured modulation
spectra. The single and summed ripples land exactly at their specified
(rate, density); the dynamic ripple fills its range.
[▶ single](https://choyun1.github.io/sonore/gallery/#d-01) [▶ sum of two](https://choyun1.github.io/sonore/gallery/#d-03) [▶ dynamic](https://choyun1.github.io/sonore/gallery/#d-06)

![Ripples](https://raw.githubusercontent.com/choyun1/sonore/main/docs/images/ripples.png)

```python
pattern = so.Ripple(4, 1, depth=0.45) + so.Ripple(-12, 2.5, depth=0.45)  # Hz, cycles/octave
pattern.plot()  # look before you listen
snd = so.ripple_sound(pattern, 1.0, fs, carrier="tones")  # or "harmonic", "noise", "low-noise", a Sound
so.ModulationSpectrum.octave(snd).plot()  # peaks at (4, 1) and (-12, 2.5)
```

The first second of the same three sounds as waveforms. The broadband waveform (top) hardly shows
the pattern, because a ripple spanning a cycle or more across frequency averages
out when all bands are summed. It lives across bands: in narrow bands (bottom,
every fifth quarter-octave band), the single ripple's 4 Hz modulation arrives
later in each lower band, which is the downward drift. The dynamic ripple shows
up in the broadband waveform only where its density passes near zero and every
band is modulated in phase.

![Ripple waveforms](https://raw.githubusercontent.com/choyun1/sonore/main/docs/images/ripple_waveforms.png)

```python
sb = so.OctaveFilterbank.per_octave(8, 250, 8000).analyze(snd)  # quarter-octave bands
sb.plot(bands=range(3, len(sb) - 1, 5))  # every fifth band
```

Positive rates drift downward in frequency (Chi et al., 1999 convention).
Carriers are scaled to equal energy per octave, so changing the carrier
changes the fine structure but not the long-term spectrum.

**Noise vocoding.** Eight ERB-spaced bands; envelopes survive, harmonic fine
structure doesn't. `so.noise_vocode` is shorthand for the typed pipeline
`(speech.envelopes(lowpass=50) * noise.tfs()).synthesize()`, where `speech`
and `noise` are the two sounds' `Subbands` on the same filterbank.
[▶ vocoded](https://choyun1.github.io/sonore/gallery/#d-15)
The [Hearing through a vocoder](https://choyun1.github.io/sonore/gallery/vocoder.html) page uses it to
simulate cochlear-implant hearing: band count, noise or tone carriers, and envelope pitch.

![Noise vocoder](https://raw.githubusercontent.com/choyun1/sonore/main/docs/images/vocoder.png)

**Phase vocoder.** A 220 Hz complex with 5 Hz vibrato, stretched to twice the
duration (the vibrato slows too, since every temporal feature is stretched),
shifted up a fifth (vibrato rate unchanged), and resynthesized through an
oscillator bank with every partial moved up 70 Hz, which makes it inharmonic
(290, 510, 730 Hz, ...: still 220 Hz apart, but no longer harmonics of anything
nearby). A shift of exactly half the spacing would not: +110 Hz gives odd
harmonics of 110 Hz, a clarinet-like tone an octave down.
[▶ original](https://choyun1.github.io/sonore/gallery/#d-11) [▶ stretched](https://choyun1.github.io/sonore/gallery/#d-12) [▶ up a fifth](https://choyun1.github.io/sonore/gallery/#d-13) [▶ +70 Hz](https://choyun1.github.io/sonore/gallery/#d-14) [▶ +110 Hz](https://choyun1.github.io/sonore/gallery/#d-14b)

![Phase vocoder](https://raw.githubusercontent.com/choyun1/sonore/main/docs/images/phase_vocoder.png)

```python
analysis = so.pv_analyze(snd)
inharmonic = analysis.resynthesize(freq_map=lambda f: f + 70)
```

**Seeing speech.** One sentence from CMU ARCTIC through several analyses on
shared axes: wideband and narrowband spectrograms, Morlet and gammatone
cochleagrams (the causal gammatone drawn with and without its latency), a
pitch-adaptive frame whose windows are 3 periods long, a TANDEM-STRAIGHT-style
power spectrum (Kawahara et al., 2011), and reassigned spectrograms (Auger &
Flandrin, 1995). Every frame among them resynthesizes the sentence to about
1e-15. [▶ classic](https://choyun1.github.io/sonore/gallery/speech.html#d-27) [▶ constant-Q](https://choyun1.github.io/sonore/gallery/speech.html#d-28) [▶ pitch](https://choyun1.github.io/sonore/gallery/speech.html#d-29) [▶ reassignment](https://choyun1.github.io/sonore/gallery/speech.html#d-30)

```python
snd = so.load("docs/speech/bdl_arctic_a0131.flac")
f0_times, f0 = np.loadtxt("docs/speech/bdl_arctic_a0131_f0.csv", delimiter=",", skiprows=2).T
wide = so.GaborFrame(5e-3, 1e-3, n_fft=1024)  # Hann 5 ms, 1 ms hop
wide.analyze(snd).plot(fmax=5000, db_range=60)
adaptive = so.TVGaborFrame.pitch_adaptive(f0_times, f0, t_end=snd.duration + 0.02)  # 3 periods
adaptive.analyze(snd).plot(fmax=5000)
so.tandem_power(snd, f0_times, f0).plot(fmax=5000)
t_edges, f_edges = np.arange(0, snd.duration, 5e-3), np.arange(0, 5020, 20)  # bins of about a pixel
so.reassigned_spectrogram(snd, wide).binned(t_edges, f_edges).plot(fmax=5000)
```

**Cepstrum.** The real cepstrum of each STFT frame separates the smooth
spectral envelope (low quefrencies) from the harmonics (a peak at one period).
Cepstral F0 on the sentence above agrees with WORLD's Harvest within 5% on
95% of the frames both call voiced; it is a baseline, not an F0 tracker.

```python
cep = so.Cepstrum(so.STFT(snd, win_dur=0.040, hop_dur=0.005))
t, f0, peak = cep.f0(f_lo=75, f_hi=400)          # windows must hold 3 periods of f_lo
envelope = cep.lifter(0.5 / 120).envelope()      # quefrencies below half a period
residual = cep.lifter(0.5 / 120, keep="high").to_sound()
```

**Sound textures** (McDermott & Simoncelli, 2011). Measure a recording's
statistics and synthesize a new sample from noise:

```python
from sonore.texture import TextureStats
from sonore.texture.synth import synthesize

stats = TextureStats.measure(so.load("applause.flac"))   # 1515 statistics
new, report = synthesize(stats, duration=5, max_iter=30, progress=True)
```

Synthesis is slow: about 2 s per iteration for 5 s of sound on one core, so
30 iterations take about a minute. The gallery's syntheses are precomputed by
`tools/make_texture_synths.py`.
[▶ applause](https://choyun1.github.io/sonore/gallery/textures.html#d-t03a) [▶ synthesized](https://choyun1.github.io/sonore/gallery/textures.html#d-t03b)
[▶ marginals only](https://choyun1.github.io/sonore/gallery/textures.html#d-u00) [▶ all textures](https://choyun1.github.io/sonore/gallery/textures.html#d-t00a)

A creek recording and its synthesis: a new waveform with the same statistics
(dashed black: the original's). [▶ stream](https://choyun1.github.io/sonore/gallery/textures.html#d-t01a) [▶ synthesized](https://choyun1.github.io/sonore/gallery/textures.html#d-t01b)

![Stream texture, original and synthesized](https://raw.githubusercontent.com/choyun1/sonore/main/docs/images/texture_stream.png)

Figures are regenerated by `python docs/make_figures.py`, which takes every
sound from the gallery's demo list so the two can't drift apart.

## Conventions

- **Sounds.** `Sound` = immutable `(n_samples, n_channels)` float array + `fs`. Operations return new Sounds.
- **Arithmetic.** `a + b` mixes, `a * b` multiplies sample-wise, `2 * a` scales, mono broadcasts to stereo.
- **Bands and envelopes.** A filterbank's output (`Subbands`) is a collection of Sounds, and so is its fine structure (`.tfs()`). Envelopes are *not* sounds: `Envelope` and `Envelopes` are their own types, non-negative, often at a low sampling rate, and applied to sounds by multiplication. `Envelopes` (one envelope per band) is what the field calls a **cochleagram**. The Hilbert decomposition is literal: `sb == sb.envelopes() * sb.tfs()`.
- **Levels.** `a + 6*dB`, `a - 3*dB`. Adding a bare number is an error, so it can't be mistaken for a DC offset. dB is always `20*log10(amplitude)`.
- **Time.** `snd[0.1:0.5]` slices by seconds; `snd.data` for samples.
- **Randomness.** Every stochastic function takes `rng=` (a seed or `np.random.Generator`).
- **Binaural.** Positive ITD = right ear leads; positive ILD = right ear louder.
- **Space.** Meters, head-centered, x = right, y = front, z = up. `hcc` = (distance cm, elevation °, azimuth ° clockwise from front).
- **Plots.** Every plotting function takes an optional `ax` and returns it; global matplotlib settings are never touched.

## What's in it

The modules are grouped in layers, and each imports only from the layers
listed before it here (core first); `plotting` is called from every
object's `.plot()`. docs/design/layout.md has the diagram. Most names are
also at the top level as `so.name`; the texture ones are under
`so.texture` and `sonore.texture.synth`.

| Module | Contents |
|---|---|
| [`core.sound`](https://github.com/choyun1/sonore/blob/main/src/sonore/core/sound.py) | `Sound`, `load` |
| [`core.units`](https://github.com/choyun1/sonore/blob/main/src/sonore/core/units.py) | `dB`, `Decibels` |
| [`signals.generators`](https://github.com/choyun1/sonore/blob/main/src/sonore/signals/generators.py) | `silence`, `pure_tone`, `harmonic_complex`, `schroeder_complex`, `square_wave`, `sawtooth_wave`, `pulse_train`, `linear_chirp`, `exponential_chirp`, `gaussian_noise`, `correlated_noise`, `iterated_ripple_noise` |
| [`signals.processing`](https://github.com/choyun1/sonore/blob/main/src/sonore/signals/processing.py) | `pad`, `truncate`, `concat`, `mix`, `normalize`, `match_fs`, `match_channels`, `relative_db`, `bandpass`, `butter_filter`, `amplitude_modulate` |
| [`analysis.frames`](https://github.com/choyun1/sonore/blob/main/src/sonore/analysis/frames.py) | `Frame` (invertible analyses: `analyze`, `synthesize` as least squares, `frame_bounds`, `energy`, `adjoint`), `Filterbank` (frequency-domain filters, any shape; canonical dual), `GaborFrame` (the STFT as a frame; any window, zero-padded FFTs), `TVGaborFrame` (a Gabor frame whose window changes over time, from an explicit schedule, `from_function`, or `pitch_adaptive` from an F0 track; exact inverse; coefficients are a `TVSTFT`) |
| [`analysis.filterbank`](https://github.com/choyun1/sonore/blob/main/src/sonore/analysis/filterbank.py) | `ERBFilterbank`, `OctaveFilterbank` (perfect-reconstruction cosine banks sharing `CosineFilterbank`, a tight `Filterbank`), `GammatoneFilterbank` (exact 4th-order gammatone responses, causal or zero-phase; `envelope_peak_delay` gives each filter's latency), `MorletFilterbank` (log-spaced Morlet wavelets); both add edge filters by default so synthesis is exact on the whole band, and `edges=False` gives the bare bank for cochleagrams. `subbands`, `Subbands` (a collection of Sounds: `.envelopes()`, `.tfs()`, `.synthesize()`), `noise_vocode` |
| [`analysis.representations`](https://github.com/choyun1/sonore/blob/main/src/sonore/analysis/representations.py) | `Spectrum`, `long_term_spectrum`, `STFT` (a `GaborFrame` analysis: exact inverse, fast Griffin-Lim), `TVSTFT` (a `TVGaborFrame` analysis), `tandem_power` (TANDEM-STRAIGHT-style pitch-adaptive power, after Kawahara et al., 2011; magnitude only, a `TFPower`), `reassigned_spectrogram` (Kodera et al., 1978; Auger & Flandrin, 1995: spectrogram cells moved to their reassigned time and frequency, binned for display; not invertible), `Mask`, `ideal_binary_mask`, `ideal_ratio_mask`, `ModulationSpectrum` (linear-frequency from an STFT, or `.octave()` in cycles/octave) |
| [`analysis.cepstrum`](https://github.com/choyun1/sonore/blob/main/src/sonore/analysis/cepstrum.py) | `Cepstrum` (the real cepstrum of an `STFT` or `TVSTFT`: rectangular liftering with a fixed or per-frame cutoff, the cepstral envelope, resynthesis with the original phase, exact when unliftered, or the minimum phase, and classic cepstral F0 after Noll, 1967) |
| [`analysis.envelopes`](https://github.com/choyun1/sonore/blob/main/src/sonore/analysis/envelopes.py) | `Envelope` (one envelope; `env * snd` modulates), `Envelopes` (one per band, i.e. a cochleagram; `.plot()`, `.modulation_spectrum()`, `env * subbands`) |
| [`analysis.modulation`](https://github.com/choyun1/sonore/blob/main/src/sonore/analysis/modulation.py) | `ConstantQModulationFilterbank`, `OctaveModulationFilterbank` (circular, analytic output optional) |
| [`stimuli.ripples`](https://github.com/choyun1/sonore/blob/main/src/sonore/stimuli/ripples.py) | `Ripple`, `RippleSum`, `DynamicRipple`, `ripple_sound`; patterns can also be any function `f(t, x)` of time and octaves, and `pattern.render(filterbank, dur, fs)` gives their `Envelopes` |
| [`stimuli.phasevocoder`](https://github.com/choyun1/sonore/blob/main/src/sonore/stimuli/phasevocoder.py) | `time_stretch`, `pitch_shift` (identity phase locking), `pv_analyze` → `PVAnalysis` (instantaneous frequency; oscillator-bank `resynthesize` with `time_scale` and `freq_map`) |
| [`stimuli.binaural`](https://github.com/choyun1/sonore/blob/main/src/sonore/stimuli/binaural.py) | `apply_itd_ild`, `simple_bir`, `interaural_cues`, `oscor`, `phasewarp` |
| [`stimuli.spatialization`](https://github.com/choyun1/sonore/blob/main/src/sonore/stimuli/spatialization.py) | `HRIRSet` (PKU-IOA, SOFA; onset-aligned interpolation), `spatialize`, `move_sound`, trajectories, coordinate conversions, `distance_gain_db` |
| [`stimuli.hrir_data`](https://github.com/choyun1/sonore/blob/main/src/sonore/stimuli/hrir_data.py) | `load_hrirs`: public HRIR databases (PKU-IOA) downloaded on first use, checksum-verified and cached |
| [`stimuli.reverb`](https://github.com/choyun1/sonore/blob/main/src/sonore/stimuli/reverb.py) | `synth_ir` (natural rooms, or the paper's atypical `decay_shape` / `rt60_profile` / `drr_profile` variants), `band_rt60s`, `measure_rt60` |
| [`texture.stats`](https://github.com/choyun1/sonore/blob/main/src/sonore/texture/stats.py) | `TextureModel`, `TextureStats` (`.measure`, `.snr`, `.replace` for hybrids, `.save`/`.load`) |
| [`texture.synth`](https://github.com/choyun1/sonore/blob/main/src/sonore/texture/synth.py) | `synthesize` (full loop), `impose_channel`; gradients in [`texture.grad`](https://github.com/choyun1/sonore/blob/main/src/sonore/texture/grad.py) |
| [`plotting`](https://github.com/choyun1/sonore/blob/main/src/sonore/plotting.py) | `overview` and the `plot_*` functions behind each object's `.plot()`; `plot_tf_db` draws any time-frequency level on non-uniform frames; cochleagrams take `align="peak"` (draw causal gammatone bands without their latency) and `fscale="linear"` (to match spectrograms) |

## Related projects

- [slab](https://github.com/DrMarc/slab): sound manipulation and psychoacoustic
  experiments, with calibration and experiment tools. The closest overlap; use
  it if you need calibrated levels or trial sequencing.
- [PsychoPy](https://www.psychopy.org/): running experiments.
- [brian2hears](https://brian2hears.readthedocs.io/): auditory periphery models.
- [librosa](https://librosa.org/): music and audio analysis.
- [pyroomacoustics](https://github.com/LCAV/pyroomacoustics): geometric room simulation.
- [pyfar](https://pyfar.org/) / [sofar](https://github.com/pyfar/sofar): acoustics and SOFA files.

## Roadmap

**Done**

- **Frames.** A `Frame` contract for invertible time-frequency
  analyses: `analyze`, `synthesize` (canonical dual, least-squares for
  modified coefficients), `frame_bounds()` and `adjoint`. The STFT
  (`GaborFrame`), the cosine, gammatone and Morlet filterbanks, and a
  time-varying Gabor frame with pitch-adaptive windows are all frames; tests
  enforce `synthesize(analyze(x)) == x` and the reported bounds. Reassigned
  spectrograms and a TANDEM-STRAIGHT-style power spectrum are drawn beside
  them in the [Seeing speech](https://choyun1.github.io/sonore/gallery/speech.html) page.
- **Cepstrum.** `Cepstrum` on any STFT: liftering, resynthesis with the
  original or minimum phase, and classic cepstral F0; see
  `docs/design/cepstrum.md`.
- **Package layout.** One subpackage per layer (`core`, `signals`,
  `analysis`, `stimuli`, `texture`), with imports pointing down a layer,
  enforced by `tests/test_layers.py`; see `docs/design/layout.md`.

**Next, in order**

1. **First PyPI release.** CI and a trusted-publishing workflow are in place
   (`docs/releasing.md`). An on-demand loader for the PKU-IOA HRTF SOFA files
   is in progress.
2. **JAX spike.** Port the texture channel objective to JAX, check it
   against the NumPy reference with the existing tests, and measure it
   against today's ~2 s per iteration. On the evidence, decide on an optional
   `sonore[jax]` backend for the heavy, optimization-shaped parts (texture
   synthesis now; the differentiable forward models that source inference
   needs later). The core stays NumPy.
3. **Speech synthesis.** Source-filter vowels (glottal source, formant
   resonators, radiation), then the Klatt synthesizer (Klatt, 1980; KLSYN88,
   Klatt & Klatt, 1990).

**Texture synthesis**

- Rebalance the objective so modulation power converges (it reaches 30 dB
  SNR when imposed without the correlation classes, but 18-23 dB in full
  synthesis); try joint imposition of all channels.
- Impose several channels at once; the per-channel objective is
  overhead-bound (about 2 s per iteration for 5 s of sound).
- Validate against the MATLAB toolbox's published examples by running both
  on the same original recordings.

**Architecture**

- Model subpackages (`sonore.texture`, later `sonore.speech`) sit on top of
  the layers below them and are never imported by them. Heavy dependencies
  go in optional extras.
- Split a component into its own distribution only when it needs a heavy
  dependency, a different release cadence, or a separate audience.
- Bayesian inference of sound sources will be a separate package built on
  sonore (JAX plus a probabilistic-programming layer), using sonore's
  generators, frames and texture statistics as its differentiable forward
  model.

**Other**

- Free-form modulation patterns: specify a modulation spectrum and synthesize it.
- A decimated, invertible constant-Q transform (nonstationary Gabor frames in frequency).
- A high-quality speech analysis/resynthesis model with robust F0 tracking
  (STRAIGHT, Kawahara et al., 1999, or its open successor WORLD, Morise et al., 2016).
- Peak-based sinusoidal modeling (McAulay & Quatieri, 1986) alongside the channel oscillator bank.
- Faster `move_sound` via batched frequency-domain filtering.
- Sources that change distance: `move_sound` with level change, travel-time
  delay and Doppler shift, and room reverberation, so that approaching and
  receding trajectories sound convincing.
- On-demand download of other public HRIR databases.
- Revisit the moving-sound renderer (linear trajectories sound unconvincing): level with distance,
  travel-time delay, Doppler, room reverberation, or the earlier unwindowed overlapping convolutions.

## References

Tags name the module(s) in [What's in it](#whats-in-it) that implement or follow each work,
and link to the source: a tag such as `representations.reassigned_spectrogram` goes to that
definition, a bare module name to the whole file. Untagged works are cited in the [Roadmap](#roadmap). Links go to the DOI where one is
confirmed, otherwise to the publisher or another stable page.

- Auger & Flandrin (1995). Improving the readability of time-frequency and time-scale representations by the reassignment method. *IEEE Trans. Signal Processing* 43(5). [doi:10.1109/78.382394](https://doi.org/10.1109/78.382394). [`representations.reassigned_spectrogram`](https://github.com/choyun1/sonore/blob/main/src/sonore/analysis/representations.py#L356)
- Balazs, Dörfler, Jaillet, Holighaus & Velasco (2011). Theory, implementation and applications of nonstationary Gabor frames. *J. Comput. Appl. Math.* 236(6). [doi:10.1016/j.cam.2011.09.011](https://doi.org/10.1016/j.cam.2011.09.011). [`frames.TVGaborFrame`](https://github.com/choyun1/sonore/blob/main/src/sonore/analysis/frames.py#L421)
- Chi, Gao, Guyton, Ru & Shamma (1999). Spectro-temporal modulation transfer functions and speech intelligibility. *JASA* 106. [JASA](https://pubs.aip.org/asa/jasa/article/106/5/2719/550617). [`ripples.Ripple`](https://github.com/choyun1/sonore/blob/main/src/sonore/stimuli/ripples.py#L84) [`representations.ModulationSpectrum`](https://github.com/choyun1/sonore/blob/main/src/sonore/analysis/representations.py#L457)
- Christensen (2003). *An Introduction to Frames and Riesz Bases*. Birkhäuser. [doi:10.1007/978-0-8176-8224-8](https://doi.org/10.1007/978-0-8176-8224-8). [`frames.Frame`](https://github.com/choyun1/sonore/blob/main/src/sonore/analysis/frames.py#L76)
- Daubechies, Grossmann & Meyer (1986). Painless nonorthogonal expansions. *J. Math. Phys.* 27(5). [doi:10.1063/1.527388](https://doi.org/10.1063/1.527388). [`frames.GaborFrame`](https://github.com/choyun1/sonore/blob/main/src/sonore/analysis/frames.py#L259)
- Dolson (1986). The phase vocoder: A tutorial. *Computer Music Journal* 10(4). [Semantic Scholar](https://www.semanticscholar.org/paper/31d9e1cc5d87c2b84cde2d4527b15b644544380e). [`phasevocoder`](https://github.com/choyun1/sonore/blob/main/src/sonore/stimuli/phasevocoder.py)
- Escabí & Schreiner (2002). Nonlinear spectrotemporal sound analysis by neurons in the auditory midbrain. *J. Neurosci.* 22. [doi:10.1523/JNEUROSCI.22-10-04114.2002](https://doi.org/10.1523/JNEUROSCI.22-10-04114.2002). [`ripples.DynamicRipple`](https://github.com/choyun1/sonore/blob/main/src/sonore/stimuli/ripples.py#L159)
- Flanagan & Golden (1966). Phase vocoder. *Bell System Technical Journal* 45. [doi:10.1002/j.1538-7305.1966.tb01706.x](https://doi.org/10.1002/j.1538-7305.1966.tb01706.x). [`phasevocoder`](https://github.com/choyun1/sonore/blob/main/src/sonore/stimuli/phasevocoder.py)
- Glasberg & Moore (1990). Derivation of auditory filter shapes from notched-noise data. *Hearing Research* 47. [doi:10.1016/0378-5955(90)90170-T](https://doi.org/10.1016/0378-5955(90)90170-T). [`filterbank.ERBFilterbank`](https://github.com/choyun1/sonore/blob/main/src/sonore/analysis/filterbank.py#L100)
- Gordon & Strawn (1985). An introduction to the phase vocoder. In J. Strawn (ed.), *Digital Audio Signal Processing: An Anthology*. Also Stanford CCRMA report STAN-M-55. [CCRMA](https://ccrma.stanford.edu/papers/introduction-phase-vocoder). [`phasevocoder`](https://github.com/choyun1/sonore/blob/main/src/sonore/stimuli/phasevocoder.py)
- Griffin & Lim (1984). Signal estimation from modified short-time Fourier transform. *IEEE TASSP* 32. [doi:10.1109/TASSP.1984.1164317](https://doi.org/10.1109/TASSP.1984.1164317). [`representations.STFT.griffin_lim`](https://github.com/choyun1/sonore/blob/main/src/sonore/analysis/representations.py#L176)
- Kawahara, Masuda-Katsuse & de Cheveigné (1999). Restructuring speech representations using a pitch-adaptive time-frequency smoothing and an instantaneous-frequency-based F0 extraction. *Speech Communication* 27. [doi:10.1016/S0167-6393(98)00085-5](https://doi.org/10.1016/S0167-6393(98)00085-5).
- Kawahara et al. (2011). Technical foundations of TANDEM-STRAIGHT, a speech analysis, modification and synthesis framework. *Sādhanā* 36(5). [doi:10.1007/s12046-011-0043-3](https://doi.org/10.1007/s12046-011-0043-3). [`representations.tandem_power`](https://github.com/choyun1/sonore/blob/main/src/sonore/analysis/representations.py#L287)
- Klatt (1980). Software for a cascade/parallel formant synthesizer. *JASA* 67(3). [doi:10.1121/1.383940](https://doi.org/10.1121/1.383940).
- Klatt & Klatt (1990). Analysis, synthesis, and perception of voice quality variations among female and male talkers. *JASA* 87. [doi:10.1121/1.398894](https://doi.org/10.1121/1.398894).
- Kodera, Gendrin & de Villedary (1978). Analysis of time-varying signals with small BT values. *IEEE Trans. ASSP* 26(1). [doi:10.1109/TASSP.1978.1163047](https://doi.org/10.1109/TASSP.1978.1163047). [`representations.reassigned_spectrogram`](https://github.com/choyun1/sonore/blob/main/src/sonore/analysis/representations.py#L356)
- Kominek & Black (2004). The CMU Arctic speech databases. *Proc. 5th ISCA Speech Synthesis Workshop (SSW5)*, 223–224. [ISCA Archive](https://www.isca-archive.org/ssw_2004/kominek04b_ssw.html). The gallery's spoken sentence.
- Kowalski, Depireux & Shamma (1996). Analysis of dynamic spectra in ferret primary auditory cortex. I. *J. Neurophysiol.* 76. [doi:10.1152/jn.1996.76.5.3503](https://doi.org/10.1152/jn.1996.76.5.3503). [`ripples.Ripple`](https://github.com/choyun1/sonore/blob/main/src/sonore/stimuli/ripples.py#L84)
- Laroche & Dolson (1999). Improved phase vocoder time-scale modification of audio. *IEEE Trans. Speech Audio Process.* 7(3). [IEEE Xplore](https://ieeexplore.ieee.org/document/759041/). [`phasevocoder.time_stretch`](https://github.com/choyun1/sonore/blob/main/src/sonore/stimuli/phasevocoder.py#L157)
- McAulay & Quatieri (1986). Speech analysis/synthesis based on a sinusoidal representation. *IEEE TASSP* 34. [Internet Archive](https://archive.org/details/SpeechAnalysisSynthesisBasedOnASinusoidalRepresentation).
- McDermott & Simoncelli (2011). Sound texture perception via statistics of the auditory periphery. *Neuron* 71. [doi:10.1016/j.neuron.2011.06.032](https://doi.org/10.1016/j.neuron.2011.06.032). [`texture`](https://github.com/choyun1/sonore/blob/main/src/sonore/texture/stats.py) [`filterbank.CosineFilterbank`](https://github.com/choyun1/sonore/blob/main/src/sonore/analysis/filterbank.py#L44) [`modulation`](https://github.com/choyun1/sonore/blob/main/src/sonore/analysis/modulation.py)
- Morise (2015). CheapTrick, a spectral envelope estimator for high-quality speech synthesis. *Speech Communication* 67. [doi:10.1016/j.specom.2014.09.003](https://doi.org/10.1016/j.specom.2014.09.003). [`frames.TVGaborFrame.pitch_adaptive`](https://github.com/choyun1/sonore/blob/main/src/sonore/analysis/frames.py#L494)
- Morise, Yokomori & Ozawa (2016). WORLD: A vocoder-based high-quality speech synthesis system for real-time applications. *IEICE Trans. Inf. & Syst.* E99-D(7). [doi:10.1587/transinf.2015EDP7457](https://doi.org/10.1587/transinf.2015EDP7457).
- Patterson, Robinson, Holdsworth, McKeown, Zhang & Allerhand (1992). Complex sounds and auditory images. In *Auditory Physiology and Perception* (Proc. 9th International Symposium on Hearing). [doi:10.1016/B978-0-08-041847-6.50054-X](https://doi.org/10.1016/B978-0-08-041847-6.50054-X). [`filterbank.GammatoneFilterbank`](https://github.com/choyun1/sonore/blob/main/src/sonore/analysis/filterbank.py#L260)
- Noll (1967). Cepstrum pitch determination. *JASA* 41(2). [PubMed](https://pubmed.ncbi.nlm.nih.gov/6040805/). [`cepstrum.Cepstrum.f0`](https://github.com/choyun1/sonore/blob/main/src/sonore/analysis/cepstrum.py#L142)
- Perraudin, Balazs & Søndergaard (2013). A fast Griffin-Lim algorithm. *IEEE WASPAA*. [doi:10.1109/WASPAA.2013.6701851](https://doi.org/10.1109/WASPAA.2013.6701851). [`representations.STFT.griffin_lim`](https://github.com/choyun1/sonore/blob/main/src/sonore/analysis/representations.py#L176)
- Qu et al. (2009). Distance-dependent head-related transfer functions measured with high spatial resolution using a spark gap. *IEEE TASLP* 17. [PKU Scholar](http://scholar.pku.edu.cn/qutianshu/publications/distance-dependent-head-related-transfer-functions-measured-high-spatial). [`spatialization.HRIRSet.from_pku_ioa`](https://github.com/choyun1/sonore/blob/main/src/sonore/stimuli/spatialization.py#L130) [`hrir_data.load_hrirs`](https://github.com/choyun1/sonore/blob/main/src/sonore/stimuli/hrir_data.py#L122)
- Schroeder (1970). Synthesis of low-peak-factor signals and binary sequences with low autocorrelation. *IEEE Trans. Inf. Theory* 16. [doi:10.1109/TIT.1970.1054411](https://doi.org/10.1109/TIT.1970.1054411). [`generators.schroeder_complex`](https://github.com/choyun1/sonore/blob/main/src/sonore/signals/generators.py#L108)
- Shannon et al. (1995). Speech recognition with primarily temporal cues. *Science* 270. [doi:10.1126/science.270.5234.303](https://doi.org/10.1126/science.270.5234.303). [`filterbank.noise_vocode`](https://github.com/choyun1/sonore/blob/main/src/sonore/analysis/filterbank.py#L467)
- Singh & Theunissen (2003). Modulation spectra of natural sounds and ethological theories of auditory processing. *JASA* 114(6). [doi:10.1121/1.1624067](https://doi.org/10.1121/1.1624067). [`representations.ModulationSpectrum`](https://github.com/choyun1/sonore/blob/main/src/sonore/analysis/representations.py#L457) [`envelopes.Envelopes.modulation_spectrum`](https://github.com/choyun1/sonore/blob/main/src/sonore/analysis/envelopes.py#L342)
- Siveke et al. (2008). Psychophysical and physiological evidence for fast binaural processing. *J. Neurosci.* 28. [J. Neurosci.](https://www.jneurosci.org/content/28/9/2043). [`binaural.oscor`](https://github.com/choyun1/sonore/blob/main/src/sonore/stimuli/binaural.py#L170) [`binaural.phasewarp`](https://github.com/choyun1/sonore/blob/main/src/sonore/stimuli/binaural.py#L179)
- Traer & McDermott (2016). Statistics of natural reverberation enable perceptual separation of sound and space. *PNAS* 113. [doi:10.1073/pnas.1612524113](https://doi.org/10.1073/pnas.1612524113). [`reverb.synth_ir`](https://github.com/choyun1/sonore/blob/main/src/sonore/stimuli/reverb.py#L77)
- Wang (2005). On ideal binary mask as the computational goal of auditory scene analysis. In *Speech Separation by Humans and Machines*. [doi:10.1007/0-387-22794-6_12](https://doi.org/10.1007/0-387-22794-6_12). [`representations.ideal_binary_mask`](https://github.com/choyun1/sonore/blob/main/src/sonore/analysis/representations.py#L443)
- Yost (1996). Pitch of iterated rippled noise. *JASA* 100. [JASA (PDF)](https://pubs.aip.org/asa/jasa/article-pdf/100/1/511/11401642/511_1_online.pdf). [`generators.iterated_ripple_noise`](https://github.com/choyun1/sonore/blob/main/src/sonore/signals/generators.py#L258)

### Reference implementations

Implementations by a paper's authors or widely used ports, with how sonore
relates to each. "Cross-checked" means a script in `tools/` compares the two
numerically; "consulted" means the code was read for behavior but not copied.

- [Sound Texture Synthesis Toolbox v1.7](https://mcdermottlab.mit.edu/downloads.html) (MATLAB), McDermott lab: the
  authors' implementation of McDermott & Simoncelli (2011). Consulted; sonore
  is a clean-room implementation from the paper, and every deliberate
  difference is listed in `so.texture.DIFFERENCES_FROM_TOOLBOX`. [`texture`](https://github.com/choyun1/sonore/blob/main/src/sonore/texture/stats.py) [`filterbank.CosineFilterbank`](https://github.com/choyun1/sonore/blob/main/src/sonore/analysis/filterbank.py#L44) [`modulation`](https://github.com/choyun1/sonore/blob/main/src/sonore/analysis/modulation.py)
- [wil-j-wil/texture_stats](https://github.com/wil-j-wil/texture_stats) (Python, MIT): a port of the toolbox's
  statistics. Cross-checked by `tools/crosscheck_texture_stats.py`. [`texture.TextureStats`](https://github.com/choyun1/sonore/blob/main/src/sonore/texture/stats.py#L181)
- [mcdermottLab/pycochleagram](https://github.com/mcdermottLab/pycochleagram) (Python): the lab's port of the
  toolbox's cochleagram code, including the cosine filterbank. Not yet
  cross-checked. [`filterbank.CosineFilterbank`](https://github.com/choyun1/sonore/blob/main/src/sonore/analysis/filterbank.py#L44)
- [LTFAT](https://ltfat.github.io/) (MATLAB/Octave, GPLv3): `frsynabs` with `'fgriflim'` is the fast Griffin-Lim from
  the group of Perraudin, Balazs & Søndergaard (2013);
  [`librosa.griffinlim`](https://librosa.org/doc/latest/generated/librosa.griffinlim.html) is a widely used Python
  version. Neither is cross-checked yet. [`representations.STFT.griffin_lim`](https://github.com/choyun1/sonore/blob/main/src/sonore/analysis/representations.py#L176)
- [SciPy `ShortTimeFFT`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.ShortTimeFFT.html)
  (BSD-3): wrapped by `GaborFrame`. Its frame operator, bounds and least-squares
  inverse are cross-checked against dense matrices in the tests and in
  `tools/check_frames_step1_claims.py` (docs/design/frames.md, step 1). [`frames.GaborFrame`](https://github.com/choyun1/sonore/blob/main/src/sonore/analysis/frames.py#L259) [`representations.STFT`](https://github.com/choyun1/sonore/blob/main/src/sonore/analysis/representations.py#L98)
- Gammatone filterbanks in Slaney's Auditory Toolbox and MATLAB's `gammatoneFilterBank` are time-domain IIR
  approximations; `GammatoneFilterbank` uses the exact frequency response instead (derivation in
  docs/design/frames.md, step 2). Consulted for conventions only. [`filterbank.GammatoneFilterbank`](https://github.com/choyun1/sonore/blob/main/src/sonore/analysis/filterbank.py#L260)
- [LTFAT](https://ltfat.github.io/) (GPLv3) and [nsgt](https://github.com/grrrr/nsgt) (Artistic License 2.0):
  frame theory in code, for dev-time cross-checks only because of their licenses. Not yet cross-checked. [`frames`](https://github.com/choyun1/sonore/blob/main/src/sonore/analysis/frames.py)

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
| `equalize_fs`, `zeropad_sounds`, `center_sounds`, `truncate_sounds` | `so.match_fs`, `so.pad(align="start"/"center")`, `so.truncate` |
| `normalize_rms`, `zero_mean`, `concat_sounds`, `compare_relative_db` | `so.normalize`, `snd.zero_mean()`, `so.concat`, `so.relative_db` |
| `sum(zeropad_sounds([a, b]))` | `so.mix([a, b])` |
| `MagnitudeSpectrum(s).to_Noise(dur, fs)` | `so.long_term_spectrum(s).to_noise(dur, fs)` |
| `STFT(snd, win)`, `S.to_Sound()`, `method="GLA"` | `so.STFT(snd, win)`, `S.to_sound()`, `S.griffin_lim()` |
| `IBM = S_t > S_m + lc`; `IBM * S_mix` | `so.ideal_binary_mask(S_t, S_m, lc_db=lc)`; `S_mix * mask` |
| `Subbands(snd, n)`, `.extract_envelopes()`, `.to_Sound()` | `so.subbands(snd, n)`, `.envelopes()`, `.synthesize()` |
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
it using [CITATION.cff](https://github.com/choyun1/sonore/blob/main/CITATION.cff).
