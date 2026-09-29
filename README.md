# sigtools

Signals and stimuli for auditory research, designed for Jupyter notebooks:
stimulus generators, an ERB filterbank with perfect reconstruction, invertible
STFTs and T-F masks, modulation spectra, binaural cue analysis, HRIR
spatialization of moving sources, and synthetic room impulse responses.

```bash
git clone https://github.com/choyun1/sigtools && cd sigtools
pip install -e ".[notebook,sofa]"      # add "play" for sounddevice playback, "dev" for tests
```

```python
import sigtools as st

x = st.harmonic_complex(1.0, 44100, f0=200, harmonics=range(1, 20), phases="schroeder+")
x = x.ramp(10e-3)
x  # in a notebook: an audio player
st.overview(x)  # waveform, spectrum, spectrogram, modulation spectrum

b = st.apply_itd_ild(st.gaussian_noise(1, 44100, rng=0), itd=300e-6, ild=6)
st.interaural_cues(b).plot()
```

See `notebooks/recipes.ipynb` for worked examples: speech-shaped noise,
vocoding, ideal binary masks, moving sources with reverb, Oscor/Phasewarp,
IRN, and more.

## Conventions

- A `Sound` is an immutable `(n_samples, n_channels)` array plus `fs`. Every
  operation returns a new `Sound`.
- `a + b` mixes, `a * b` multiplies sample-wise, `2 * a` scales. Mono
  broadcasts against stereo. Level changes in dB use `a.gain_db(6)`.
- `snd[0.1:0.5]` slices by **seconds**; use `snd.data` for samples.
- Generators return RMS = 1. Everything random takes `rng=` (seed or
  `np.random.Generator`).
- dB means `20*log10(amplitude)` everywhere.
- Binaural: positive ITD = right ear leads; positive ILD = right ear louder.
- Space: meters, x = right, y = front, z = up. `hcc` = (dist cm, elev °,
  azimuth ° clockwise from front).

## Migrating from 0.1

| 0.1 | 0.2 |
|---|---|
| `from sigtools.sounds import *` etc. | `import sigtools as st` |
| `PureTone(dur, fs, f)`, `GaussianNoise(...)`, ... | `st.pure_tone(dur, fs, f)`, `st.gaussian_noise(...)`, ... |
| `GaussianNoise(dur, fs, lo, hi, tilt)` | `st.gaussian_noise(dur, fs, band=(lo, hi), tilt=...)`; tilt is now dB/octave |
| `SchroederPhase(dur, fs, f0, n)` | `st.schroeder_complex(dur, fs, f0, n)` |
| `SoundLoader(path)` | `st.load(path)` |
| `Silence(dur, fs)` | `st.silence(dur, fs)` |
| `snd + 6` (dB gain) | `snd.gain_db(6)` |
| `snd.make_binaural()` | `snd.to_stereo()` |
| `snd.extract_envelope()` | `snd.envelope()` |
| `ramp_edges(snd, d)` | `snd.ramp(d)` |
| `butter_bandpass_filter(snd, lo, hi)` | `st.bandpass(snd, lo, hi)` (no longer RMS-normalizes) |
| `equalize_fs`, `zeropad_sounds`, `center_sounds`, `truncate_sounds` | `st.match_fs`, `st.pad(align="start"/"center")`, `st.truncate` |
| `normalize_rms`, `zero_mean`, `concat_sounds`, `compare_relative_db` | `st.normalize`, `snd.zero_mean()`, `st.concat`, `st.relative_db` |
| `sum(zeropad_sounds([a, b]))` | `st.mix([a, b])` |
| `MagnitudeSpectrum(snd)` | `st.Spectrum.from_sound(snd)` or `st.long_term_spectrum(snds)` |
| `MagnitudeSpectrum(s).to_Noise(dur, fs)` | `st.long_term_spectrum(s).to_noise(dur, fs)` |
| `STFT(snd, win)` / `S.to_Sound()` / `method="GLA"` | `st.STFT(snd, win)` / `S.to_sound()` / `S.griffin_lim()` |
| `IBM = S_t > S_m + lc` ; `IBM * S_mix` | `st.ideal_binary_mask(S_t, S_m, lc_db=lc)` ; `S_mix * mask` |
| `ModulationSpectrum(S)` | `st.ModulationSpectrum(S)` |
| `Subbands(snd, n)` / `.extract_envelopes()` / `.to_Sound()` | `st.subbands(snd, n)` / `.envelopes()` / `.synthesize()` |
| `carrier_subbands / carrier_env` (TFS) | `.tfs()` |
| `InterauralCues(snd, win)` | `st.interaural_cues(snd, win)` (adds coherence and zero-lag correlation; NaN in silence) |
| `SimpleBIR(fs, itd, ild)` | `st.simple_bir(fs, itd, ild)` or `st.apply_itd_ild(snd, itd, ild)` |
| `SynthIR(drr, rt60, dB_thresh, fs)` | `st.synth_ir(rt60, fs, drr_db=..., decay_db=-dB_thresh)` |
| `move_sound(traj, snd)` | `st.move_sound(snd, traj, hrirs)`, with `hrirs = st.HRIRSet.from_pku_ioa(dir)` or `.from_sofa(path)` |
| `make_linear_trajectory`, `make_hcc_circular_trajectory` | `st.linear_trajectory`, `st.circular_trajectory` |
| `display_STFT(x, S)` | `st.overview(x)` |
| `AudioControl(snd).display()` | put `snd` at the end of a cell |

## Behavior changes worth knowing

These were bugs in 0.1, so results computed with 0.1 may differ.

- Spectrum/STFT "dB" was `10*log10(amplitude)` (half the true dB); now `20*log10`.
- `butter_bandpass_filter` filtered stereo *across channels*; now per channel, zero-phase.
- `SimpleBIR` was one sample short and its gain grew with the ITD.
- `SynthIR`'s `synth_DRR` had no effect (it was normalized away), and the
  resynthesis filterbank was misaligned in frequency. `synth_ir` has a real
  direct path and DRR.
- Tone generators used `linspace(0, dur, n)`, so frequencies were off by a
  factor of `(n-1)/n` and consecutive segments didn't join in phase.
- Square/sawtooth/pulse trains are band-limited by default (`bandlimited=False`
  for the naive, aliased versions).
- `move_sound` summed ~100 overlapping unwindowed HRIR convolutions per sample;
  it now cross-fades between neighbouring positions.
- `InterauralCues` never computed IAC.

## Development

```bash
pip install -e ".[dev]"
pytest
ruff check . && ruff format .
```

An unrelated package called `sigtools` exists on PyPI, so install from this
repo rather than with `pip install sigtools`.
