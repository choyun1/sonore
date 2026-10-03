# Changelog

Notable changes to sonore. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Until 1.0, a minor version (0.x.0) may change import paths or behaviour; a patch
version (0.x.y) only fixes bugs.

## [Unreleased]

### Added
- `ModulationSpectrum.to_sound(carrier=...)`: a sound whose envelopes have
  the spectrum, the carrier supplying what it lacks, the modulation phase
  and the fine structure. A `Sound` lends its own (so the spectrum of `x`
  with `carrier=x` gives `x`'s envelopes back); `"tones"` (steady tones at
  the band centres, the default) and `"noise"` draw a random modulation
  phase. `ModulationSpectrum.with_gain(g)` edits a spectrum, for example
  `g = lambda rate, density: abs(rate) <= 4`, and `to_envelopes` gives the
  rebuilt envelopes. Spectra made from envelopes keep how they were made
  for this. Design and measurements in
  `docs/design/views/modulation-targets.md`.

### Changed
- `so.normalize(sounds, peak=...)` scales each sound to its own peak, as
  `Sound.normalize(peak=...)` does; before, the list version took only `rms`.
- Clearer errors from the list helpers: an empty list, an unknown `mode` in
  `match_fs`, and a silent reference in `relative_db` raise a `ValueError`
  saying so (they raised `KeyError` or `ZeroDivisionError`).
- `so.pad` and `so.truncate` are replaced by
  `so.match_lengths(sounds, mode="pad" | "truncate", align=...)`, next to
  `so.match_fs` and `so.match_channels`; `align` now also chooses which
  part a truncated sound keeps. `Sound.pad` is unchanged and is now the only
  `pad`.
- `Sound.fs` is read-only, like the samples.
- In a notebook, a Sound plays at its true level instead of being
  normalized by the player; a sound peaking above 1 shows a note suggesting
  `normalize(peak=...)` instead of a player.
- Clearer errors: `2 - snd` says why it is ambiguous (`0 - snd` still
  inverts), an array of the wrong length gets the "lengths differ" message,
  `snd.right` on a mono sound and `normalize` of silence raise a sonore
  message instead of NumPy's `IndexError` or a division by zero.
- One `so.Filterbank` class holds every undecimated bank: a frequency scale
  (`"erb"`, `"octave"`, `"mel"`, `"linear"` or your own `Scale`), the
  centers as positions on it, and a filter type (`Cosine`, `Gammatone`,
  `Morlet`, or your own `FilterType`). `so.cosine_filterbank`,
  `so.gammatone_filterbank` and `so.morlet_filterbank` build the common
  ones and replace `ERBFilterbank`, `OctaveFilterbank`, `CosineFilterbank`,
  `GammatoneFilterbank`, `MorletFilterbank` and `BandpassFilterbank`
  (`OctaveFilterbank.per_octave(12, lo, hi)` is now
  `cosine_filterbank(f_lo=lo, f_hi=hi, spacing=1/12, scale="octave")`).
  Cosine banks keep their output bit for bit. New: any increasing
  `centers=`, wider cosines (`width=`), and every factory on any scale
  (docs/design/frames/filterbanks.md).
- Tightness is measured, not declared: the `tight` attribute is gone, and
  `bank.is_tight(n_samples, fs)` and synthesis decide from `s` on the
  grid the coefficients live on (to 1e-12), so a bank is never treated as
  tight when it is not.
- `f_lo` and `f_hi` mean the same for every filter type: the outer knots, with
  the bandpass centers strictly inside. For gammatone and Morlet banks this
  moves the centers slightly for the same arguments (they used to start at
  `f_lo`).
- `envelope_peak_delay` is measured from each filter's impulse response,
  so every bank has it (0 for zero-phase filters).
- The default top band edge of `so.noise_vocode`, `ModulationSpectrum.octave`,
  `so.synth_ir` and `so.measure_rt60` is now the Nyquist
  frequency, not 95% of it (an arbitrary margin with no recorded reason). Output
  changes only when the requested `f_hi` is above 95% of Nyquist.
- `power_to_db` treats a negative power as zero and returns the floor,
  instead of the level of its absolute value (audit sitting 1c).
- `so.subbands(snd, ...)` is removed; write
  `so.cosine_filterbank(...).analyze(snd)`, which shows the bank (audit
  sitting 8).
- `Subbands` can be multiplied by a number or by one gain per band (an
  equalizer); gains that change over time still go through a `Mask`.
- A `Scale` checks that its two conversions invert each other where it is
  defined, so a mismatched pair raises instead of misplacing every filter.
- `Subbands.synthesize()` is now `Subbands.to_sound()`, like every other
  set of coefficients.
- `Mask` is a View (`sonore.views.mask`, was `sonore.frames.mask`): it holds
  gains, not a sound, so `to_sound()` refuses and names the route back
  (`(coefs * mask).to_sound()`). Masks now work on `TVSTFT` and `Subbands`
  as well as `STFT`; for subbands the power is the squared Hilbert envelope
  of each band. The docstrings cite the sources of the ideal binary and
  ratio masks.

### Documentation
- Gammatone: the factor 1.019 is credited to Slaney (1993), who gives it
  as Patterson's recommendation (checked against the report). The
  envelope-peak formula is described as approximate, which it is.
- Cosine filters credit the steerable pyramid as well as McDermott &
  Simoncelli (2011): Simoncelli & Freeman (1995) for squared responses
  summing to one, Portilla & Simoncelli (2000) for the cosines on a log2
  scale. The README references gain these two and the mask sources, with
  pages checked against the papers.
- `GaborFrame` and `TVGaborFrame` docstrings put their Parameters last, so
  the reference no longer folds the text after them into the list.
- `frames`: the module docstring no longer promises a JAX port, and the
  STFT's levels are described as "dB, `20 log10 |X|`" (audit sitting 7).
- `Sound`: time slices behave like Python index slicing (negative times
  count from the end, times past the end are clipped); `from_channels`
  zero-pads shorter channels at the end; `resample` names SciPy's default
  anti-aliasing filter. Docstrings for the remaining undocumented `Sound`
  members.

### Changed (development)
- `docs/make_figures.py` and the eight README figures only it drew are
  removed: it no longer ran once the gallery examples moved into the page
  scripts, and two of the README's three figures now come from the
  gallery. `docs/images/ripples.png` stays, as the README still shows it.

## [0.4.0] - 2026-10-03

The package is reorganized by meaning: core, sources (sounds made from
parameters), frames (invertible analyses), views (one-way analyses) and the
topics spatial and texture. Every `so.name` is unchanged, but deep import
paths move (see Changed). Views refuse to synthesize, with the reason, and
go back to sound through `to_sound` where a canonical route exists. Also
new since 0.3.1: MFCCs, F0 tracking, WORLD's analysis and synthesis, Klatt
and LF voice synthesis, voice changes and a moving-sound renderer.

### Added
- `so.View` and `so.NotInvertibleError`: every view (the spectra,
  envelopes, modulation spectra, the cepstrum, MFCCs, F0 tracks, spectral
  envelopes and aperiodicity, interaural cues, texture statistics and the
  phase vocoder's analysis) is now a `View`, with a `discards` sentence
  saying what it drops. Its `synthesize` raises `NotInvertibleError` (a
  `NotImplementedError`) with that sentence and the route back to sound
  where one exists, such as `Cepstrum.to_sound` or `so.world_synthesize`.
  Before, `synthesize` on a view failed with a plain `AttributeError`. See
  "Only frames synthesize" in `docs/design/layout/reorganization.md`.
- `to_sound` is the route from a view back to a sound, and takes what the
  view discarded (`docs/design/layout/sound-first.md`). `Spectrum.to_sound(duration,
  fs, carrier=...)` takes the phase from a carrier: `"noise"` (the default,
  as `to_noise` was), a Sound's own phase, or `"minimum"` for the
  minimum-phase impulse response. On a view with no canonical route, such
  as `ModulationSpectrum`, `to_sound` raises `NotInvertibleError`.
- `STFT` and `TVSTFT` have `n_fft` and `shortest_window` properties.
- `so.power_to_db(x, ref=1.0, floor_db=-300.0)` and `so.db_to_power(db)`,
  the power counterparts of `amp_to_db` and `db_to_amp`.
- `so.cheaptrick`, `so.d4c` and `so.world_synthesize`: WORLD's spectral
  envelope (Morise, 2015), aperiodicity (Morise, 2016) and synthesis,
  ported exactly to NumPy with no pyworld dependency, returning
  `SpectralEnvelope` and `Aperiodicity` views. A stored pyworld fixture
  holds them to WORLD's numbers; `so.DIFFERENCES_FROM_WORLD` lists every way
  sonore departs from WORLD. `so.harmonic_aperiodicity` is a second
  measure, not WORLD's: the share of noise left after fitting the
  harmonics. See `docs/design/views/world.md` and the gallery page
  "Aperiodicity".
- `so.ModulationSpectrogram` and `so.HannModulationFilterbank`: modulation
  power, local mean and depth for every time window, acoustic band and
  modulation rate, from any `Envelopes`, with a `valid` mask, plots and an
  animation. See `docs/design/views/modulation-spectrogram.md` and the gallery
  page "Modulation spectrogram".
- `so.move_sound` is a new renderer: each ear reads the sound through its
  own smoothly changing delay, so a source changing distance glides in
  pitch (Doppler) instead of comb-filtering; HRIRs are interpolated between
  measured distances and travel time and 1/r carry it beyond them; an
  optional `room` tail with `drr_db`. A path can be a function of time;
  `so.hcc_trajectory` builds one in head-centred coordinates, and
  `so.SPEED_OF_SOUND` is the default 343 m/s. See
  `docs/design/spatial/moving-sound.md` and the gallery page "Moving talkers".
- `so.scale_f0` and `so.warp_frequency`: a pitch change (voiced F0 times a
  ratio, optionally spread around the median) and a formant shift (an
  envelope read at `f / ratio`; a number, a ratio over time, or any
  frequency map), on any F0 contour and any envelope or aperiodicity,
  whatever measured them. A ratio of 1 returns the input itself.
  `so.GridEnvelope` holds any envelope as power on a grid of times and
  frequencies; `Cepstrum.envelope_view()` and `MFCC.envelope_view()`
  return one. See `docs/design/views/voice-change.md` and the gallery page
  "Changing a voice" (`docs/gallery/voice/voice.py`).
- `so.MFCC`: mel-frequency cepstral coefficients (Davis & Mermelstein,
  1980) of a sound, with the usual speech settings (25 ms Hamming window,
  10 ms hop, 26 HTK mel bands, 13 coefficients), or of any `STFT` or
  `TVSTFT`. Options for the Slaney mel scale, area-normalised triangles, a
  floor and HTK's lifter; triangles straight in mel (HTK, Kaldi; the
  default) or in Hz (librosa); `.mel_power` (the mel spectrogram), `.db`,
  `.deltas()` (computed as librosa's), `.envelope(f)` and `.plot()`. Tests
  compare it with Kaldi's MFCCs (through kaldi-native-fbank) and with
  librosa 0.11's mel power, MFCCs and deltas, all stored. See
  `docs/design/views/mfcc.md`.
- `so.glottal_source`: a voiced source of Liljencrants-Fant (LF) glottal
  pulses (Fant, Liljencrants & Lin, 1985) on a fixed F0 or an F0 contour,
  built from the pulse's exact harmonics so it does not alias. Its shape is
  Fant's (1995) `rd`, from tense (0.3) to lax (2.7), default 0.7; a number
  or a `(times, values)` track. `so.lf_harmonics` gives the pulse's complex
  Fourier coefficients in closed form (from `rd`, or from `ra`, `rg`, `rk`),
  `so.lf_pulse` one period of the flow derivative or flow. See
  `docs/design/sources/glottal-source.md`.
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
  spaced parameter sets between two endpoints. See `docs/design/sources/klatt.md`.
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
  `docs/design/sources/harmonic-source.md`.
- `so.f0_track` and `F0Track`: an F0 tracker with a voiced/unvoiced
  decision. Candidates from YIN's difference function, refinement by the
  instantaneous frequency of six harmonics (after WORLD's StoneMask), a
  periodicity score, and a Viterbi pass. Within 0.12% on synthetic glides
  and vibrato, works without the fundamental, and gets the voicing of about
  6% (male) and 1.5% (female) of frames wrong against laryngograph reference
  F0. Its `t` and `f0` feed `TVGaborFrame.pitch_adaptive` and
  `Cepstrum.lifter`; `.plot()` draws the track and, optionally, every
  candidate. See `docs/design/views/f0.md`.
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

### Changed
- `Spectrum.to_noise(duration, fs)` is `Spectrum.to_sound(duration, fs)`,
  and `PVAnalysis.resynthesize` is `PVAnalysis.to_sound`, with the same
  arguments and output. The old names are gone.
- `so.long_term_spectrum(sounds, win_dur=0.1)` takes its Welch segment
  length in seconds, replacing `nperseg=4096` samples, with no
  compatibility argument. The default now gives 10 Hz spacing at any
  sample rate, so its output changes slightly, and with it speech-shaped
  noise made from it.
- In `sonore.texture.synth`, `ChannelObjective.ctx` and
  `impose_channel(ctx=...)` are renamed to `context`.
- The package is reorganized by meaning
  (`docs/design/layout/reorganization.md`, `docs/design/layout/sound-first.md`,
  `docs/design/layout.md`). Every `so.name` is unchanged; only deep import
  paths move, with no compatibility modules at the old paths:
  - `sonore.analysis`, `sonore.signals` and `sonore.stimuli` are gone.
  - `sonore.core` gains `processing` (`pad`, `mix`, `normalize`, the
    filters, `resonator`, `antiresonator`, ...); `freq_to_mel` and
    `mel_to_freq` are in `sonore.core.utils` and exported as
    `so.freq_to_mel` and `so.mel_to_freq`.
  - Sounds made from parameters are in `sonore.sources`: `waveforms` (the
    old `generators` and `glottal`), `klatt` and `ripples`.
  - Invertible analyses are in `sonore.frames`: `frame` (`Frame`),
    `filterbank` (`Filterbank` and the banks, `subbands`, `Subbands`),
    `gabor` (`GaborFrame`, `TVGaborFrame`, `STFT`, `TVSTFT`) and `mask`
    (`Mask`, `ideal_binary_mask`, `ideal_ratio_mask`).
  - One-way analyses are in `sonore.views`: `spectrum` (`Spectrum`,
    `long_term_spectrum`, `TFPower`, `tandem_power`), `reassigned`,
    `envelopes` (now also `noise_vocode`), `modulation` (now also
    `ModulationSpectrum`), `modspectrogram`, `cepstrum`, `mfcc`, `f0` (now
    also `scale_f0`), `spectral_envelope` (`cheaptrick`, `SpectralEnvelope`,
    `GridEnvelope`, `warp_frequency`), `aperiodicity` (`d4c`,
    `Aperiodicity`, `harmonic_aperiodicity`), `world` (WORLD's synthesis and
    the helpers it shares with the analysis: `world_randn`,
    `world_fft_size`, `DIFFERENCES_FROM_WORLD`) and `phasevocoder`.
  - `binaural`, `spatialization`, `hrir_data` and `reverb` are in
    `sonore.spatial`.
- `world_synthesize` reads its inputs by what they provide (an F0 track's
  `.t` and `.f0`, an envelope as `env(t, f)`, an aperiodicity's grid)
  instead of checking their types; its output is unchanged.
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
  were tracked at F0/3 or F0/5 (`docs/design/views/female-voices.md`); now they
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
- Gabor analysis works when the hop is longer than the window (windows
  with gaps between them). Before, `GaborFrame` refused to analyze such a
  layout; synthesis still refuses, since it is not a frame
  (`docs/design/frames/frames.md`, D4).
- `Subbands` and `Envelopes` copy the array they are given, so the
  caller's array stays writeable; before, `Subbands` made it read-only.
  Their shape errors now say the shape they received.
- Two numbers in docstrings were wrong. `DynamicRipple` puts about 7% of
  its modulation power outside `rate_range`, not 40%, and `fast_padding`'s
  median growth for odd lengths is 1.1%, not 0.3%. Every number a
  docstring quotes is now measured by `tools/measure_docstring_numbers.py`.
- `sonore.plotting.plot_f0_track` is listed in `plotting.__all__`, and the
  per-band ITD image is rasterized like the other images (this changes SVG
  and PDF output only).
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

### Changed (development)
- `docs/design/philosophy.md`: names that come from a cited paper or a
  reference implementation (Klatt, LF, the texture statistics C, C1 and
  C2, WORLD) are never renamed.
- Tests are in folders that mirror `src/sonore` (`tests/core/`,
  `tests/sources/`, `tests/frames/`, `tests/views/`, ...). The dB tests moved to `tests/core/test_units.py` and
  the HRIR download tests to `tests/spatial/test_hrir_data.py`.

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
  numerical checks in `docs/design/views/cepstrum.md` and
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

[Unreleased]: https://github.com/choyun1/sonore/compare/v0.4.0...main
[0.4.0]: https://github.com/choyun1/sonore/releases/tag/v0.4.0
[0.3.1]: https://github.com/choyun1/sonore/releases/tag/v0.3.1
[0.3.0]: https://github.com/choyun1/sonore/releases/tag/v0.3.0
[0.2.0]: https://github.com/choyun1/sonore/commit/213e21a
