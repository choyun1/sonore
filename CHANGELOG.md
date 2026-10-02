# Changelog

Notable changes to sonore. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Until 1.0, a minor version (0.x.0) may change import paths or behaviour; a patch
version (0.x.y) only fixes bugs.

## [Unreleased]

### Added
- `so.scale_f0` and `so.warp_frequency`: a pitch change (voiced F0 times a
  ratio, optionally spread around the median) and a formant shift (an
  envelope read at `f / ratio`; a number, a ratio over time, or any
  frequency map), on any F0 contour and any envelope or aperiodicity,
  whatever measured them. A ratio of 1 returns the input itself.
  `so.GridEnvelope` holds any envelope as power on a grid of times and
  frequencies; `Cepstrum.envelope_view()` and `MFCC.envelope_view()`
  return one. See `docs/design/voice-change.md` and the gallery page
  "Changing a voice" (`docs/gallery/seeing/voice.py`).
- `so.MFCC`: mel-frequency cepstral coefficients (Davis & Mermelstein,
  1980) of a sound, with the usual speech settings (25 ms Hamming window,
  10 ms hop, 26 HTK mel bands, 13 coefficients), or of any `STFT` or
  `TVSTFT`. Options for the Slaney mel scale, area-normalised triangles, a
  floor and HTK's lifter; triangles straight in mel (HTK, Kaldi; the
  default) or in Hz (librosa); `.mel_power` (the mel spectrogram), `.db`,
  `.deltas()` (computed as librosa's), `.envelope(f)` and `.plot()`. Tests
  compare it with Kaldi's MFCCs (through kaldi-native-fbank) and with
  librosa 0.11's mel power, MFCCs and deltas, all stored. See
  `docs/design/mfcc.md`.
- `so.glottal_source`: a voiced source of Liljencrants-Fant (LF) glottal
  pulses (Fant, Liljencrants & Lin, 1985) on a fixed F0 or an F0 contour,
  built from the pulse's exact harmonics so it does not alias. Its shape is
  Fant's (1995) `rd`, from tense (0.3) to lax (2.7), default 0.7; a number
  or a `(times, values)` track. `so.lf_harmonics` gives the pulse's complex
  Fourier coefficients in closed form (from `rd`, or from `ra`, `rg`, `rk`),
  `so.lf_pulse` one period of the flow derivative or flow. See
  `docs/design/glottal-source.md`.
- `so.klatt_synthesize`: KLSYN88's source switch `SS` (1, the default, is
  the existing source, unchanged; 3 is the LF source) and `RD`, the LF
  shape, which may be a track.
- `so.harmonic_complex`: an amplitude function may take the harmonic
  number as a third argument and return complex gains, whose angle shifts
  that harmonic's phase. Existing outputs are unchanged.
- `so.klatt_synthesize`: a Klatt-style cascade/parallel formant synthesizer
  (Klatt, 1980). Voicing is `harmonic_complex` on F0 with Klatt's glottal
  spectrum; aspiration and frication are white noise, modulated at F0 while
  voiced; the nasal pole and zero and formants 1-5 run in cascade, formants
  1-6 in parallel with alternating signs, and the voiced source gets the
  radiation difference. Parameters use Klatt's names (`F0`, `AV`, `AH`,
  `AF`, `AB`, `F1`-`F6`, `B1`-`B6`, `A1`-`A6`, `FNP`, `BNP`, `FNZ`, `BNZ`),
  each a number or a `(times, values)` track interpolated to every sample;
  defaults are in `so.KLATT_DEFAULTS`. `so.klatt_continuum` makes evenly
  spaced parameter sets between two endpoints. See `docs/design/klatt.md`.
- `so.resonator` and `so.antiresonator`: Klatt's second-order formant and
  its exact inverse, with unit gain at 0 Hz and a frequency and bandwidth
  that may follow a track; the filter state is carried through every change,
  so gliding formants make no clicks.
- `so.harmonic_complex` takes an F0 contour for `f0` as well as a number:
  an `F0Track`, or any `(times, values)` pair with 0 where unvoiced (one
  row per channel). Every harmonic follows its multiple of the contour
  with no phase jumps, unvoiced gaps are bridged and gated with Hann ramps
  (`ramp`, or `unvoiced="noise"` to fill them with noise), and harmonics
  fade out below `f_max` (default `0.45 * fs`), so nothing aliases.
  `amplitudes` may also be a function of time and frequency (a spectral
  envelope), `harmonics` is now optional, and a phase array of the wrong
  length raises a clear error. `square_wave`, `sawtooth_wave`,
  `pulse_train` and `schroeder_complex` take contours too. See
  `docs/design/harmonic-source.md`.
- `so.f0_track` and `F0Track`: an F0 tracker with a voiced/unvoiced
  decision. Candidates from YIN's difference function, refinement by the
  instantaneous frequency of six harmonics (after WORLD's StoneMask), a
  periodicity score, and a Viterbi pass. Within 0.12% on synthetic glides
  and vibrato, works without the fundamental, and gets the voicing of about
  6% (male) and 1.5% (female) of frames wrong against laryngograph reference
  F0. Its `t` and `f0` feed `TVGaborFrame.pitch_adaptive` and
  `Cepstrum.lifter`; `.plot()` draws the track and, optionally, every
  candidate. See `docs/design/f0.md`.
- `so.set_fft_workers` and `so.fft_workers`: the large FFTs (filterbank
  analysis and synthesis, Hilbert envelopes, FFT resampling in texture
  synthesis, modulation filtering in `ModulationSpectrogram`) now use every
  available core by default. Results are bit-identical for any number of
  threads. Use `so.set_fft_workers(1)`, or the same in a `with` block, when
  running several jobs in parallel.
- An API reference generated from the docstrings (source in `docs/api/source`),
  built by a new pages workflow and served next to the gallery, with a `docs`
  extra for building it locally. A test keeps every public
  name documented; the names still waiting are listed in
  `tests/undocumented.txt`.

### Changed (development)
- Tests are in folders that mirror `src/sonore` (`tests/core/`,
  `tests/analysis/`, ...). The dB tests moved to `tests/core/test_units.py` and
  the HRIR download tests to `tests/stimuli/test_hrir_data.py`.

### Changed
- `so.world_synthesize` takes any envelope read as `envelope(t, f)` (a
  `GridEnvelope`, a warped envelope, a function), reading it at the
  aperiodicity's time windows and frequencies, and reads an F0 contour on
  other time windows onto them (an F0 track whose times differ from the
  aperiodicity's was refused before). A `SpectralEnvelope` and F0 track on
  that grid are used as before, so WORLD's output is unchanged.
- `so.harmonic_complex` takes a spectral envelope (`SpectralEnvelope`,
  `GridEnvelope`) as `amplitudes`, read point by point as amplitude; the
  new `SpectralEnvelope.amplitude(t, f)` does the reading.
- `so.f0_track` keeps up to eight candidates per time window (was four),
  and its subharmonic rule now covers every whole multiple of a candidate's
  frequency, not only the octave. Steady synthetic vowels at 250 to 400 Hz
  were tracked at F0/3 or F0/5 (`docs/design/female-voices.md`); now they
  are tracked exactly. On the FDA laryngograph database the voicing error
  is 5.6% (male) and 1.5% (female), as before within 0.1 points, and
  tracking takes about twice as long. `F0Track.candidates` has eight
  columns.
- The number goes first in a level: `dB*6` and `(6*dB)*2` now raise a
  `TypeError` that suggests `6*dB` and `2*(6*dB)`. `6*dB` is unchanged.
- A level is no longer a plain number: `float(6*dB)` raises and suggests
  `(6*dB).value`, so a `Decibels` can't slip silently into a function that
  expects a float. `snd.gain_db(6*dB)` still works.
- The band-limited `square_wave`, `sawtooth_wave` and `pulse_train` are
  now computed by `harmonic_complex`; their outputs differ from before by
  less than 1e-10. Fixed-F0 `harmonic_complex` and `schroeder_complex`
  output is unchanged bit for bit.
- `Filterbank.analyze` with the default `pad="auto"` rounds the padding up
  so that the FFT length has no prime factor above 11 (typically 0.2% longer,
  at most a few percent). Subbands and envelopes are 3 to 4 times faster for
  most lengths, and differ from before by about 1e-4 of their peak: the
  wrap-around of ringing below the -60 dB padding threshold, which a longer
  padding reduces. Texture statistics and synthesis are unchanged (they use
  `pad=0`).

### Fixed
- `harmonic_complex` (and `schroeder_complex`, `square_wave`,
  `sawtooth_wave`, `pulse_train`, `glottal_source`) on a fixed F0 ignored
  `f_max`, so a sound asked to stop at 5 kHz had harmonics up to Nyquist.
  Now a fixed F0 honors `f_max` as a contour does: harmonics below it by
  default, with the same cos² fade from 0.9 `f_max`. Without `f_max`, a
  fixed F0 is unchanged bit for bit.
- `move_sound` returned a near-silent sound, without an error, for a source
  coming toward the head faster than sound. It now refuses such a path
  (and, with PKU-IOA, one at 300 m/s, where the measured onsets change
  faster than the travel time). Slower paths are unchanged to 1e-10.

## [0.3.1] - 2026-10-01

The first release archived on Zenodo, which gives it a DOI. No change to the
library's code.

### Added
- A starter notebook, `docs/notebooks/start.ipynb`, with an "Open in Colab"
  badge in the README: a first tour that installs sonore from PyPI and runs
  in the browser. `tests/test_notebook.py` runs its code cells.
- `CITATION.cff` gives the version, release date, abstract and PyPI link, which
  Zenodo reads when it archives a GitHub release.

## [0.3.0] - 2026-10-01

First release on PyPI.

### Added
- `so.Cepstrum`: the real cepstrum of an `STFT` or `TVSTFT`, with liftering
  (fixed or per-frame cutoff), the cepstral envelope, resynthesis with the
  original or minimum phase, classic cepstral F0, and `plot`. Design and
  numerical checks in `docs/design/cepstrum.md` and
  `tools/check_cepstrum_claims.py`.
- `so.load_hrirs()` downloads the PKU-IOA HRIR database on first use (1 m by
  default, any of its 8 distances on request), checks each file's SHA-256 and
  caches it in `$SONORE_DATA_DIR` or the user cache directory. The SOFA copy it
  downloads is mirrored left to right relative to the original `.dat` files;
  the loader corrects it.
- `HRIRSet.concat` merges HRIR sets, e.g. one per distance.
- Continuous integration: tests and lint on Python 3.10 and 3.14 for every push
  and pull request, plus a check that the PyPI files build and pass their tests.
- Release workflow publishing to TestPyPI and PyPI with trusted publishing
  (see `docs/releasing.md`).
- Gallery pages of their own for seeing speech, a short course in time-frequency
  analysis, and for sound textures. Each is a runnable script
  (`docs/gallery/speech.py`, `docs/gallery/textures.py`) shown with every example's code.

### Changed
- `HRIRSet.from_pku_ioa` also finds `.dat` files in subfolders, so the
  database's own `dist*/elev*/` layout loads directly.
- The source distribution now holds the code, tests and tools only; the docs,
  gallery and audio samples stay in the repository.

## [0.2.0] - 2026-09-29

Renamed to sonore, with the version kept in one place (`src/sonore/__init__.py`).
Not published to PyPI.

[Unreleased]: https://github.com/choyun1/sonore/compare/v0.3.1...main
[0.3.1]: https://github.com/choyun1/sonore/releases/tag/v0.3.1
[0.3.0]: https://github.com/choyun1/sonore/releases/tag/v0.3.0
[0.2.0]: https://github.com/choyun1/sonore/commit/213e21a
