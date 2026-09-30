# sonore

**Signals and stimuli for auditory research, built for Jupyter.**

sonore is a small Python library for making, manipulating, and analyzing sounds
the way hearing scientists think about them: tones and complexes, shaped and
correlated noises, ERB-spaced subbands, invertible spectrograms, a phase
vocoder, interaural cues, HRIR spatialization of moving sources, and synthetic
room reverberation.
Levels are written as levels (`snd + 6*dB`), times as seconds (`snd[0.1:0.5]`),
and any sound at the end of a notebook cell plays.

The name comes from Pierre Schaeffer's *objet sonore*, the "sound object": a
sound taken as a thing in its own right and studied for how it is heard rather
than for what produced it. Schaeffer called that mode of listening
*acousmatic*, after the *akousmatikoi*, Pythagoras's students who listened to
his teaching from behind a veil. The `Sound` object at the center of this
library is meant in the same spirit.

![Overview of an iterated rippled noise](https://raw.githubusercontent.com/choyun1/sonore/main/docs/images/overview_irn.png)

## What it's for

- **Psychophysical stimuli.** Pure tones, harmonic complexes with any phase
  scheme (cosine, sine, alternating, random, Schroeder±), band-limited square,
  sawtooth and pulse trains, chirps, band-limited and spectrally tilted noise,
  iterated rippled noise. Everything is reproducible from a seed.
- **Binaural and spatial hearing.** Exact fractional ITDs, ILDs, interaurally
  correlated noise, Oscor and Phasewarp, windowed ITD/ILD/coherence analysis
  (broadband or per band), and rendering of static or moving sources through
  measured HRIRs (PKU-IOA or any SOFA file).
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
  and decorrelated binaural tails.
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

**Ideal binary mask.** A gliding harmonic target at −5 dB SNR, the IBM computed
from the separate STFTs, and the masked mixture resynthesized.

![Ideal binary mask](https://raw.githubusercontent.com/choyun1/sonore/main/docs/images/ibm.png)

```python
S_t, S_m, S_x = (so.STFT(s, 25e-3) for s in (target, masker, target + masker))
separated = (S_x * so.ideal_binary_mask(S_t, S_m, lc_db=0)).to_sound()
```

**Oscor and Phasewarp** (Siveke et al., 2008). The zero-lag interaural
correlation follows sin and cos of the modulation rate, respectively.

![Binaural cues of Oscor and Phasewarp](https://raw.githubusercontent.com/choyun1/sonore/main/docs/images/binaural_cues.png)

**ERB filterbank with perfect reconstruction.** An exponential sweep split
into 6 ERB-spaced bands, plus the lowpass and highpass edge filters that make
the bank power-complementary, then summed back together. The reconstruction
error is at the level of floating-point rounding.

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

![Ripples](https://raw.githubusercontent.com/choyun1/sonore/main/docs/images/ripples.png)

```python
pattern = so.Ripple(4, 1, depth=0.45) + so.Ripple(-12, 2.5, depth=0.45)  # Hz, cycles/octave
pattern.plot()  # look before you listen
snd = so.ripple_sound(pattern, 1.0, fs, carrier="tones")  # or "harmonic", "noise", "low-noise", a Sound
so.ModulationSpectrum.octave(snd).plot()  # peaks at (4, 1) and (-12, 2.5)
```

Positive rates drift downward in frequency (Chi et al., 1999 convention).
Carriers are scaled to equal energy per octave, so changing the carrier
changes the fine structure but not the long-term spectrum.

**Noise vocoding.** Eight ERB-spaced bands; envelopes survive, harmonic fine
structure doesn't.

![Noise vocoder](https://raw.githubusercontent.com/choyun1/sonore/main/docs/images/vocoder.png)

**Phase vocoder.** A 220 Hz complex with 5 Hz vibrato, stretched to twice the
duration (the vibrato slows too, since every temporal feature is stretched),
shifted up a fifth (vibrato rate unchanged), and resynthesized through an
oscillator bank with every partial moved up 110 Hz, which makes it inharmonic.

![Phase vocoder](https://raw.githubusercontent.com/choyun1/sonore/main/docs/images/phase_vocoder.png)

```python
analysis = so.pv_analyze(snd)
inharmonic = analysis.resynthesize(freq_map=lambda f: f + 110)
```

Figures are regenerated by `python docs/make_figures.py`.

## Conventions

| | |
|---|---|
| Sounds | `Sound` = immutable `(n_samples, n_channels)` float array + `fs`. Operations return new Sounds. |
| Arithmetic | `a + b` mixes, `a * b` multiplies sample-wise, `2 * a` scales, mono broadcasts to stereo. |
| Levels | `a + 6*dB`, `a - 3*dB`. Adding a bare number is an error, so it can't be mistaken for a DC offset. dB is always `20*log10(amplitude)`. |
| Time | `snd[0.1:0.5]` slices by seconds; `snd.data` for samples. |
| Randomness | Every stochastic function takes `rng=` (a seed or `np.random.Generator`). |
| Binaural | Positive ITD = right ear leads; positive ILD = right ear louder. |
| Space | Meters, head-centered, x = right, y = front, z = up. `hcc` = (distance cm, elevation °, azimuth ° clockwise from front). |
| Plots | Every plotting function takes an optional `ax` and returns it; global matplotlib settings are never touched. |

## What's in it

| Module | Contents |
|---|---|
| `sound` | `Sound`, `load` |
| `units` | `dB`, `Decibels` |
| `generators` | `silence`, `pure_tone`, `harmonic_complex`, `schroeder_complex`, `square_wave`, `sawtooth_wave`, `pulse_train`, `linear_chirp`, `exponential_chirp`, `gaussian_noise`, `correlated_noise`, `iterated_ripple_noise` |
| `processing` | `pad`, `truncate`, `concat`, `mix`, `normalize`, `match_fs`, `match_channels`, `relative_db`, `bandpass`, `butter_filter`, `amplitude_modulate` |
| `representations` | `Spectrum`, `long_term_spectrum`, `STFT` (exact inverse, fast Griffin-Lim), `Mask`, `ideal_binary_mask`, `ideal_ratio_mask`, `ModulationSpectrum` (linear-frequency from an STFT, or `.octave()` in cycles/octave) |
| `filterbank` | `ERBFilterbank`, `OctaveFilterbank` (both perfect-reconstruction cosine banks), `subbands`, `Subbands` (envelopes, TFS, synthesis), `noise_vocode` |
| `ripples` | `Ripple`, `RippleSum`, `DynamicRipple`, `ripple_sound`; patterns can also be any function `f(t, x)` of time and octaves |
| `phasevocoder` | `time_stretch`, `pitch_shift` (identity phase locking), `pv_analyze` → `PVAnalysis` (instantaneous frequency; oscillator-bank `resynthesize` with `time_scale` and `freq_map`) |
| `binaural` | `apply_itd_ild`, `simple_bir`, `interaural_cues`, `oscor`, `phasewarp` |
| `spatialization` | `HRIRSet` (PKU-IOA, SOFA; onset-aligned interpolation), `spatialize`, `move_sound`, trajectories, coordinate conversions, `distance_gain_db` |
| `reverb` | `synth_ir` |
| `plotting` | `overview` and the `plot_*` functions behind each object's `.plot()` |

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

- Free-form modulation patterns: specify a modulation spectrum and synthesize it.
- Sound texture synthesis from auditory statistics (McDermott & Simoncelli, 2011).
- A high-quality speech analysis/resynthesis model with robust F0 tracking
  (STRAIGHT, Kawahara et al., 1999, or its open successor WORLD, Morise et al., 2016).
- Peak-based sinusoidal modeling (McAulay & Quatieri, 1986) alongside the channel oscillator bank.
- Faster `move_sound` via batched frequency-domain filtering.
- On-demand download of public HRIR databases.
- Differentiable (JAX) versions of the core renderers.

## References

- Chi, Gao, Guyton, Ru & Shamma (1999). Spectro-temporal modulation transfer functions and speech intelligibility. *JASA* 106.
- Dolson (1986). The phase vocoder: A tutorial. *Computer Music Journal* 10(4).
- Escabí & Schreiner (2002). Nonlinear spectrotemporal sound analysis by neurons in the auditory midbrain. *J. Neurosci.* 22.
- Flanagan & Golden (1966). Phase vocoder. *Bell System Technical Journal* 45.
- Glasberg & Moore (1990). Derivation of auditory filter shapes from notched-noise data. *Hearing Research* 47.
- Gordon & Strawn (1985). An introduction to the phase vocoder. In J. Strawn (ed.), *Digital Audio Signal Processing: An Anthology*. Also Stanford CCRMA report STAN-M-55.
- Griffin & Lim (1984). Signal estimation from modified short-time Fourier transform. *IEEE TASSP* 32.
- Kowalski, Depireux & Shamma (1996). Analysis of dynamic spectra in ferret primary auditory cortex. I. *J. Neurophysiol.* 76.
- Laroche & Dolson (1999). Improved phase vocoder time-scale modification of audio. *IEEE Trans. Speech Audio Process.* 7(3).
- McDermott & Simoncelli (2011). Sound texture perception via statistics of the auditory periphery. *Neuron* 71.
- Perraudin, Balazs & Søndergaard (2013). A fast Griffin-Lim algorithm. *IEEE WASPAA*.
- Qu et al. (2009). Distance-dependent head-related transfer functions measured with high spatial resolution using a spark gap. *IEEE TASLP* 17.
- Schroeder (1970). Synthesis of low-peak-factor signals and binary sequences with low autocorrelation. *IEEE Trans. Inf. Theory* 16.
- Shannon et al. (1995). Speech recognition with primarily temporal cues. *Science* 270.
- Siveke et al. (2008). Psychophysical and physiological evidence for fast binaural processing. *J. Neurosci.* 28.
- Traer & McDermott (2016). Statistics of natural reverberation enable perceptual separation of sound and space. *PNAS* 113.
- Wang (2005). On ideal binary mask as the computational goal of auditory scene analysis. In *Speech Separation by Humans and Machines*.
- Yost (1996). Pitch of iterated rippled noise. *JASA* 100.

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
| `snd.make_binaural()`, `snd.extract_envelope()` | `snd.to_stereo()`, `snd.envelope()` |
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
| `move_sound(traj, snd)` | `so.move_sound(snd, traj, hrirs)` with `so.HRIRSet.from_pku_ioa(dir)` or `.from_sofa(path)` |
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
pytest
ruff check . && ruff format .
```

## License and citation

MIT; see [LICENSE](https://github.com/choyun1/sonore/blob/main/LICENSE). If sonore is useful in your research, please cite
it using [CITATION.cff](https://github.com/choyun1/sonore/blob/main/CITATION.cff).
