# Changelog

Notable changes to sonore. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Until 1.0, a minor version (0.x.0) may change import paths or behaviour; a patch
version (0.x.y) only fixes bugs.

## [Unreleased]

### Added
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

### Changed
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
