# Changelog

Notable changes to sonore. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Until 1.0, a minor version (0.x.0) may change import paths or behavior; a patch
version (0.x.y) only fixes bugs.

## [Unreleased]

### Added

- Streams: views computed one block of sound at a time, equal to the batch
  analysis of the whole sound bit for bit (docs/design/streaming.md). A view
  that streams has `start(fs)`, `step(state, block)` and `finish(state)`,
  with an immutable state; `stream(view, blocks)` runs them lazily over any
  iterable of `Sound` blocks, and `blocks(sound, size)` cuts a sound into
  them. `GaborFrame` streams: each step returns an `STFT` of the time windows
  that block completed, and `STFT.concatenate` joins the parts.
  `SlidingWindowStream` gives the three methods to any view whose time
  windows each depend on a fixed stretch of samples; `Streamable` is the
  protocol `stream` takes.
- `STFT.is_whole`: false for a part of a stream, which `to_sound`,
  `griffin_lim` and the frame's `synthesize` and `adjoint` refuse.
- `Cepstrum(..., reference=)` and `MFCC(..., reference=)`: a fixed floor,
  `floor_db` below that magnitude in every time window, so each time window
  depends on its own samples only. The default, `None`, is the floor below
  each channel's peak as before. With it, cepstra and MFCCs of a stream's
  STFT parts join (`Cepstrum.concatenate`, `MFCC.concatenate`) into exactly
  those of the whole sound.
- `Filterbank.analyze(sound, domain="time")`: a bank of causal gammatones
  without edge filters run as exact recursive (IIR) filters, the sampled
  gammatone's own z-transform. The bands are within 2e-5 of the
  frequency-domain ones, and their envelopes are the filters' complex
  magnitude, with no Hilbert transform. The bank streams in this domain
  (`start`, `step`, `finish`; `Subbands.concatenate` joins the parts). These
  subbands are not resynthesized.
- `TemporalEnvelope.pooled(win_dur, hop_dur)` and `EnvelopePooling`, its
  stream: the envelope's power in Hann time windows (25 ms every 10 ms by
  default), as an amplitude at `1 / hop_dur`, a cochleagram that streams.
  `TemporalEnvelope.concatenate` joins its parts.
- `Subbands.t` and `first_sample` on subbands and envelopes, the sample a
  part of a stream starts at.
- `ModulationSpectrogramStream`: the causal modulation spectrogram
  (`align="causal"`) a block of envelopes at a time, each step a
  `ModulationSpectrogram` of the time windows that block completed;
  `ModulationSpectrogram.concatenate` joins them into the batch result to
  within 1e-12 of its peak (the stream carries sliding sums restarted
  every 1024 samples, the batch path convolves by FFT).
- `f0_track(..., delay=)`: `None` (the default) is the whole-track
  Viterbi pass as before; a number of seconds decides each time window that
  long after it. `F0TrackStream` gives exactly that result block by block
  (50 ms by default), each step an `F0Track` of the time windows it
  decided; `F0Track.concatenate` joins them.

### Changed

- `f0_track` computes each candidate from its own time window's samples
  alone: the refinement and the score used a window length taken from the
  group of candidates computed together, so a candidate could change with
  its neighbors. Rows are now padded to a length set by the candidate.
  Candidates and F0s move by rounding only (under 4e-13 Hz on
  `bdl_arctic_a0131`, no voicing change); on that recording the track
  also ran faster, 1.2 s against 2.1 s (timed once).

## [0.9.0] - 2026-10-28

A release from the code audit's last sittings (14 to 23), with two
additions: a decimated filterbank (a constant-Q transform with an exact
inverse) and RT60 curves for synthetic rooms. The fix that changes the most
sounds is the HRIR gain: renders at a sampling rate other than the HRIR set's
are louder, by 2.7 dB at 48 kHz for PKU-IOA. Some inputs that used to run
without a word are now refused in plain words.

### Added

- `DecimatedFilterbank`: any `Filterbank` with each band kept at its own
  rate, the nonstationary Gabor frame in frequency, with an exact inverse. On
  the semitone cosine bank it is a decimated constant-Q transform that keeps
  about 2 real numbers per sample instead of 85. Its coefficients,
  `DecimatedSubbands`, are complex, one sequence per band, and draw each band
  with its own time cells (docs/design/frames/decimated-filterbanks.md).
- `synth_ir` and `band_rt60s` also take a room's RT60 as a curve,
  `rt60=(freqs, rt60s)` in Hz and seconds, such as one drawn by hand. Each
  band reads the curve at its center frequency, in log frequency and log RT60,
  held flat beyond its ends. A single number gives the same samples as before.

### Changed

- `SpectralEnvelope` and `Aperiodicity` refuse NaN and infinite values, as
  `TemporalEnvelope` does (a NaN made `source_filter_synthesize` silent for
  that time window without a word).
- `Aperiodicity.db` is floored like `SpectralEnvelope.db`, so a share of
  zero reads as a very low, finite level instead of -inf.
- `harmonic_aperiodicity` refuses a window too short for its fit (fewer than
  1.25 samples per unknown, about 2.5 periods) and names the shortest that
  works; such a window used to report all noise.
- Docstrings: what the harmonic residual's amplitude ramps follow (a tremolo
  or an onset, not a vibrato), and the measured distances of
  `pitch_adaptive_envelope` and `band_aperiodicity` from WORLD (audit
  sitting 14).
- `source_filter_synthesize` fades the harmonics at a voicing edge over
  `harmonic_complex`'s 5 ms gate. Their power was 0 in unvoiced time windows,
  and read in dB between windows that cut them within half a millisecond of
  the last voiced window; it is now held at the nearest voiced window's. On
  the checker's two sentences the unvoiced distance from the originals drops
  from 4.01 to 3.97 dB (bdl) and 4.15 to 4.13 dB (slt). The gallery pages "The
  source-filter model", "A richer source-filter model" and "Rebuilding and
  changing a voice" are rebuilt with it. Its docstring now says the noise is
  shaped on the grid's hop, not a 5 ms one (audit sitting 14b).
- A `Mask` applies only to coefficients on its own grid: the same sampling
  rate and frame (or filterbank and padding), not only the same shape; a
  16 kHz mask used to apply without a word to a same-shaped 32 kHz STFT. The
  ideal masks check their target and masker the same way. `Mask` refuses NaN
  and infinite gains, `ideal_ratio_mask` refuses `beta <= 0`, and `Mask.t`
  says it leaves out the subbands' padding (audit sitting 7b).
- `Cepstrum` and `MFCC` refuse a `floor_db` of 0 or above, which floored
  every magnitude at or above the loudest and returned flat coefficients.
  The `MFCC` docstring says its FFT length is the smallest power of two at
  least as long as the window, as Kaldi rounds (audit sitting 15).
- A sound shorter than half the analysis window is refused by `STFT` with a
  plain message (it failed inside SciPy), and so by every view built on it;
  `so.formant_track` checks the same before filtering, so an empty sound no
  longer fails in the pre-emphasis. `so.levinson` refuses a negative order
  and fewer lags than the order needs, and its docstring gives the sign of
  its reflection coefficients. The `so.formant_track` docstring says its
  window and hop are rounded to whole samples at twice the ceiling, and
  `FormantTrack`'s repr prints "none" for a formant missing everywhere
  (audit sitting 15b).
- HRIRs resampled to a sound's rate keep their gain. `HRIRSet.at`,
  `so.spatialize` and `so.move_sound` resampled a set's IRs as signals, which
  scaled the filter's gain by the ratio of rates: PKU-IOA's 65,536 Hz set
  rendered 2.7 dB too quiet at 48 kHz, 3.4 dB at 44.1 kHz and 12.2 dB at
  16 kHz. The gallery page "Moving talkers" plays normalized sounds, but its
  two "Level at each ear, re the talker" figures are 12.2 dB too low until
  the page is rebuilt.
- `so.spatialize` keeps the start of a sound placed nearer than a
  single-distance set's distance; it read the sound early and dropped what
  fell before time 0, so it now agrees with a still `so.move_sound`.
- `HRIRSet` says plainly when its directions at a distance do not surround
  the head (they lie in one plane or on one side; this used to fail inside
  Qhull or as a singular matrix) and when a source is at the head's center.
  `HRIRSet.from_sofa` adds a nonzero `Data.Delay` to the IRs. Docstrings:
  `align` applies to `at` and `so.spatialize` while `so.move_sound` always
  aligns, the length `at` returns, `so.hcc_trajectory` interpolating azimuth
  as a number of degrees, and the refusal of a source nearer than the
  nearest of several distances. `tools/check_move_sound_speed.py` draws its
  directions above the hole below -40 degrees again (audit sitting 18).
- `DecimatedFilterbank` cuts Morlet filters at `floor_db` as its docstring
  says. A Morlet's tails underflow to exactly zero far from its center, which
  the code took for a support, so most Morlet filters were left uncut: still
  exact, but 16.2 real numbers per sample instead of 7.2 for a 54-band bank.
  Filter types now say whether they are zero outside a band by construction
  (`FilterType.compact_support`, true for cosines). `DecimatedSubbands.db` is
  documented as the envelope for bandpass bands only (for the lowpass and
  highpass it is the coefficient's magnitude), the class docstring says what
  an edge band's coefficients are, and the repr counts real numbers per
  padded sample (audit sitting 8b).
- `synth_ir`, `band_rt60s` and `interaural_cues` refuse what they cannot
  use, in plain words: an RT60 curve with a NaN or infinite frequency (it
  gave a flat 1 s room without a word), `rt60` of 0, a negative or a non-finite
  number, `decay_db` of 0 or less, a sound shorter than one analysis window,
  and a `max_itd` as long as the window. Where only one ear is silent,
  `interaural_cues` reports no ITD (NaN) instead of 0. `synth_ir`'s onset
  levels are read straight from the regressions' intercepts: the fits share
  one slope against RT60 across bands, so the room's RT60 never changed them
  (IRs change by rounding only, under 4e-16 of the peak). Docstrings say how
  closely the band RT60s' median follows `rt60` (within about 6%), credit the
  listening results to the paper, give where the fit coefficients came from,
  how flat `simple_bir` is, what windows and times `interaural_cues` uses, and
  that oscor and phasewarp are normalized and phasewarp moves the right ear's
  spectrum up by `f_mod` (audit sitting 19).
- Texture statistics: `TextureModel.prepare` refuses a sound shorter than one
  envelope sample (2.5 ms at the defaults) instead of failing inside NumPy, and
  `TextureStats.snr` refuses statistics measured with a different model.
  Docstrings say that `prepare` averages the channels and that `measure` takes
  any length but needs about 2 s for one cycle of the slowest modulation band.
  New tests cover the channel average, resampling, the window's ramps, the
  subband variance, `duration` and `snr` (audit sitting 20).
- `ModulationFilterbank.transfer` gives what `filter` multiplies a spectrum
  by, and the texture gradients use it instead of their own copy of the
  analytic filters; synthesis output is unchanged bit for bit.
  `ChannelContext.build` refuses a window whose length is not the envelope's
  and says it normalizes the window. New tests cover an odd number of envelope
  samples and the window's normalization (audit sitting 21).
- Texture synthesis refuses a sound too short for the slowest octave
  modulation band that C1 or C2 uses (under about 0.32 s at the defaults),
  where it used to run with a NaN objective and return a poor sound with only
  a warning. The module docstring describes the whole synthesis and gives the
  conjugate-gradient budget exactly; `pop_rate` is listed in `__all__`. New
  tests: a sound that already has the target statistics comes back from an
  iteration unchanged, a louder subband target makes that subband louder, the
  budget per iteration, the best iterate returned, the clip at 0, the mean
  kept without `env_mean`, and `pop_start` rebuilt from its definition
  (audit sitting 22).
- `plot_interaural_cues` draws a sound with one ear silent, where it used to
  fail with "Axis limits cannot be NaN or Inf"; those windows are left out
  and the axis limits come from the others. Plots of a filterbank built with
  `edges=False` draw every band and label none of them as an edge filter
  (`plot_envelopes` and `plot_decimated_subbands` dropped the first and last
  bands; `plot_subbands` labeled them `<` and `>`); the per-band ITD image
  labels every band by its center. `plot_waveform`, `plot_spectrum`,
  `plot_stft` and `plot_modulation_spectrum` have docstrings, and
  `plot_decimated_subbands`, `plot_lpc` and `plot_formant_track` are listed in
  `__all__`. New tests read back what each plot draws: the image values, cell
  edges, lines, color limits and cursors (audit sitting 23).
- Gallery: "What is a voice" gives a formula for every concept and measure it
  discusses: the source-filter product and the level of each harmonic, Klatt's
  resonator and the radiation at the lips, semitones and the F0 spread, local
  jitter and shimmer as Praat defines them, Yumoto's HNR and its relation to a
  noise share, CPP, H1-H2 and H1-A3, Rd, the spectral tilt, the quarter-wave
  tube behind formant scaling, and the envelope warp fit. The warp fit with
  only the level free now searches up to a ratio of 2: its best ratio, 1.65,
  lay past the old grid's edge at 1.6, so the page printed 1.595. The formant
  ratio is now read both ways, a male tract 20% longer or a female tract 17%
  shorter; the page called 1.20 a 20% shorter tract.

## [0.8.0] - 2026-10-19

A breaking release: the envelope classes are merged into `TemporalEnvelope`
and `SpectralEnvelope`, WORLD's two analyses are rewritten from their papers
under new names, and WORLD's synthesis is replaced by
`source_filter_synthesize`. The gallery is rebuilt on this version.

### Added

- `so.source_filter_synthesize(f0, envelope, aperiodicity=None)`: a voice
  rebuilt from an F0 track, a spectral envelope and an aperiodicity, as
  harmonics (`so.harmonic_complex` on the track) carrying the envelope's
  periodic part plus white noise shaped to its noise part by an STFT.
  Without an aperiodicity voicing is all or nothing: harmonics where the
  track is voiced, noise where it is not, the classic source-filter
  resynthesis. It replaces `so.world_synthesize` (see Removed) and reads
  any F0 track and any envelope, as that did. On the bdl
  and slt sentences it is as close to the originals as WORLD's synthesis
  was (log mel distance 1.80 and 1.87 dB voiced, against 1.83 and 2.06);
  its harmonics start in cosine phase, so each period is peakier than a
  voice's. Checked in `tools/check_voice_recipe_claims.py`.
- `sonore.texture.synth.pop_start`: a starting sound for `synthesize` that
  already has pops, Gaussian noise times a sparse train of 4 ms bumps at
  power-law sizes, at the recording's own pop rate (`pop_rate`) and
  exponent. Started from it instead of from noise, the synthesis of a
  recording with heavy pops (a crackling fire) keeps the pops' width and
  their alignment across channels and matches the statistics better (our
  fire: 30 to 40 dB average SNR). `synthesize` is unchanged; it is passed
  as `init=`. The start is sonore's own, where the paper starts from noise;
  design and measurements in docs/design/texture/pop-start.md and
  `tools/check_pop_start_claims.py`.
- `so.chroma`: the power in each of the 12 pitch classes per time window,
  from the cosine filterbank on the cents scale with one band per semitone
  (A1 to A8 by default, A4 at `reference`), the octaves of each pitch class
  added together. A one-way view (`Chroma`) that drops octave height.
  `Chroma.to_sound(fs, octaves=...)` plays each pitch class as a Shepard
  tone, which gives the same chroma back; `octaves` is the spread over
  octaves the view dropped (a bell with no height by default, a note such
  as `"C4"`, or a function of frequency). `Chroma.plot` and
  `plot_chroma`. Design and checks in docs/design/views/chroma.md and
  `tools/check_chroma_claims.py`.
- Gallery: "Pitch class and height" in the Music group, after Tuning and
  temperament: the semitone filter bank as a constant-Q filter bank, one
  chord in two voicings with one chroma, Shepard tones (a scale that keeps
  rising, two tones a tritone apart), the Risset glide, and a tune's chroma
  played back as Shepard tones with no height and near two chosen octaves.
- `so.envelope_peaks`: how big a sound's pops are. The largest envelope
  value in each 5 ms bin of the sound high-passed at 300 Hz and at 2 kHz,
  and the power-law exponent of their upper tail by the Hill estimator,
  with its uncertainty; it warns when too few values are fitted. Below about
  2 a sound is made of pops; Gaussian noise gives 20 to 40. A one-way view,
  for checking whether a texture synthesis keeps the pops of its original
  (docs/design/views/envelope-peaks.md). Texture synthesis is unchanged.
- Gallery: "How big the pops are" on the Sound textures page, the envelope
  peaks of the fire recording and its synthesis, and pops from
  `so.event_train` at power-law exponents 1, 2 and 4.
- `plot_envelopes(db_max=...)` (and `TemporalEnvelope.plot`): the top of the
  color scale in dB, so several cochleagrams can be drawn on one scale. The
  Sound quilts page draws each quilt on its original's scale.
- Gallery: "Background plus pops" on the Sound textures page, a recipe of
  sonore's own (not from a paper, judged only by ear) for pop-like textures
  such as fire: synthesize the recording with its pops
  limited away, then add its own pops back with `so.event_train` at the
  exponent and rate it measures. The section opens with the math of heavy
  tails. `tools/make_texture_synths.py` stores the fire's background
  synthesis (`--backgrounds-only` makes only those).

### Changed

- **One class for each meaning of "envelope"** (`docs/design/views/spectral-envelope-classes.md`).
  `so.TemporalEnvelope` replaces `Envelope` and `Envelopes`: one envelope, or
  one per band of the filterbank it holds. `snd.envelope()` and
  `sb.envelopes()` return it; everything either class did still works under
  the one name (`env * snd`, `env * subbands`, `env[i]`, `.lowpass`,
  `.resample`, `.modulation_spectrum()`, `.plot()`, `.to_sound(carrier)`).
  New: a filterbank's envelopes multiply a Sound (split with the filterbank,
  band by band, summed back), two banded envelopes multiply or divide, and a
  single envelope goes on a `"noise"` carrier (white noise's fine structure).
  `len(env)` now counts samples, as for a Sound; `env.n_bands` counts bands.
  Operations that need the filterbank (`env[i]`, iteration, `cfs`,
  `without_edges`, `modulation_spectrum`, the `"tone"` carrier) raise
  `TypeError` on a single envelope. The module is `views/temporal_envelope.py`.
- `so.SpectralEnvelope` is now the one class for a spectral envelope on a grid
  of time windows and frequencies, whatever estimator made it: it absorbs
  `GridEnvelope`, and `Cepstrum.envelope_view()`, `MFCC.envelope_view()` and
  `LPC.envelope_view()` return it. `env.method` names the estimator
  (`"CheapTrick"`, `"cepstrum"`, `"MFCC"`, `"LPC"`, or None for one built by
  hand) and `env.settings` holds its parameters (`q1`, `n_mfcc`, `order`, ...);
  `env.q1` is gone. The constructor is `SpectralEnvelope(data, t, f, *, fs=None,
  method=None, settings=None)`: the third argument is the frequency axis, not
  the sampling rate. `n_fft` is `2 (len(f) - 1)` when `fs` is given and `f` is
  that FFT's grid, and None otherwise. `Aperiodicity` shares its base and its
  grid reading; `Aperiodicity.bands(edges, envelope=)` now reads an envelope
  given on another grid at its own times and frequencies instead of assuming
  the grids agree.
- `so.TimeFrequencyGrid`, a `typing.Protocol` for anything held on a grid
  (`t`, `f`, `data`, read as `view(t, f)`), types what
  `source_filter_synthesize`, `warp_frequency`, `harmonic_complex(amplitudes=)`
  and `Aperiodicity.bands(envelope=)` accept, beside a plain function.
- Gallery: "Rebuilding and changing a voice" is condensed to the fourth page of
  the Voices arc. Its opening section "Envelopes on harmonics and on noise" is
  gone: the 16-band noise vocoder was the Vocoder page's own example, and the
  16 bands on a harmonic carrier that follows the pitch track move to that
  page as a new section, "A harmonic carrier that follows the pitch". The
  whisper stays, at the end of "The aperiodicity". The page now introduces
  itself by the three measurements `so.source_filter_synthesize` reads
  rather than as a WORLD vocoder. Old links forward.
- Gallery: "Two talkers" becomes "What is a voice" (`whatisavoice.html`), the
  first page of the Voices group: what a voice is made of (one source-filter
  diagram), the two talkers every speech page uses, how voices differ (pitch,
  cycle-to-cycle irregularity, noise, phonation type, vocal effort, size), each
  as source or filter and with how it is measured, and a small table of the
  indicators sonore computes on the two recordings (F0, H1-H2, CPP, noise share
  by band, formant scaling). Jitter, shimmer and HNR are defined, not computed;
  GRBAS and CAPE-V are named. "One vowel, three voices" moved here from The
  source-filter model. Old links forward.
- Gallery: "Source and aperiodicity" becomes "A richer source-filter model"
  (`richer.html`), ordered by the three shortfalls the source-filter page ends
  on. A new first section builds the envelope measured to the pitch period:
  why a fixed window ripples, STRAIGHT and TANDEM-STRAIGHT in prose and
  equations, and CheapTrick's three steps written out and checked against
  `so.pitch_adaptive_envelope`, with the effect of `q1`. The lifter against
  CheapTrick moved here from the source-filter page, and the phase examples
  from the voice page. A comparison with the simple rebuild finds that
  CheapTrick's envelope is what brings the unvoiced stretches back to the
  recordings' level. A paragraph places learned vocoders. Old links forward.
- Gallery: the Formant synthesis, Finding formants, Pitch and Cepstrum pages
  are one page, "The source-filter model": a voice built from a source and a
  filter by rule, each measured back from a recording (pitch, formants, the
  envelope), and put back together. Repeated examples were cut (seven steps
  from heed to hod, cascade against parallel formants, two of the voice
  quality changes, two of the four sawtooth tracks, two LPC checks, the MFCC
  pair of spectra, silence in the gaps); old links forward to the new
  sections. "Long-term spectra" and "Harmonics sample the envelope" moved
  there from Two talkers; the long-term spectra now compare each recording
  with its rebuilt copy. The rebuilt voice uses `so.source_filter_synthesize`
  with the liftered cepstral envelope, and the page says where it falls
  short: the noise comes out 5 and 9 dB too loud against the voiced stretches.
- WORLD's two analyses are renamed for what they do: `so.cheaptrick` is now
  `so.pitch_adaptive_envelope` and `so.d4c` is `so.band_aperiodicity`. The
  old names are gone. sonore has these two parts of WORLD and not its F0
  trackers or its synthesis, so the old names made it read as all of WORLD.
- `so.pitch_adaptive_envelope` (CheapTrick) and `so.band_aperiodicity` (D4C)
  are rewritten from their papers (Morise, 2015 and 2016) instead of ported
  from WORLD's code. Their output changes:
  - `pitch_adaptive_envelope`'s `q1` defaults to the paper's -0.09 instead
    of WORLD's -0.15. At the same `q1` its envelopes are within 0.01 dB of
    WORLD's (median over voiced time windows of two sentences); the first and
    last time windows and 0 Hz differ more. It takes `unvoiced_f0=500.0`.
  - `band_aperiodicity` follows the paper. On harmonics plus white noise its
    values track the true noise share with a slope of 1; the paper's values
    read about 8 dB too periodic (as the paper finds, about 6 dB there), and
    `correction_db=8.0` adds that back. WORLD's code reads about half as
    steeply. The `threshold` voicing test is gone: every time window with
    F0 > 0 is analyzed, and unvoiced ones are all noise (a share of exactly 1).
    The sampling rate must be at least 12 kHz, so that one 6 kHz band fits.
  - Both accept any sampling rate, draw no random numbers, and read zeros
    beyond the ends of the sound.
  Measured in `tools/check_paper_vocoder_claims.py`.
- `so.f0_track` for low voices (60 to 90 Hz, inside the default range):
  the refinement uses every harmonic up to 1.5 kHz (at least six, at most
  24) instead of the first six, which below 100 Hz hold a few percent of a
  vowel's power and in noise put the refined F0 a percent off, enough for
  the true period's score to collapse (a steady 60 Hz /a/ at 20 dB SNR was
  wrong on 74% of time windows); and the score compares two periods (at
  least 16.7 ms) instead of three, so that a low voice whose F0 moves
  still lines up with its copy one period later (a clean 70 Hz vowel with
  a ±3% vibrato was wrong on 62% of time windows) and a short-period
  candidate is judged across a whole glottal cycle. Against the
  laryngograph reference the male speaker's voicing error falls from 5.6%
  to 4.0% and the female's stays at 1.6%; precision moves by half a point
  (97.6% and 95.3% within 5% where both voice). On the gallery sentence
  the agreement with Harvest is 98.6% within 5% (was 99.7%) with 83% of
  its voiced time windows voiced (was 79%). From the second-model re-audit
  of sitting 12; measurements in docs/design/views/f0.md.
- No private class appears in a public signature any more (no change to
  any output). `TVGaborFrame.layout` is now private, since it returned a
  private type that only sonore's own code reads. `gaussian_noise`'s
  `spectrum` accepts any `sonore.sources.waveforms.MeasuredSpectrum`
  (anything with `level_in_range(freqs)`), and `Spectrum.level_in_range`
  is now public; the type used to name a private Protocol with a private
  method, which no user class could satisfy. `klatt_synthesize`'s types
  now say that `F0` may be an F0 contour object such as an `F0Track`,
  which it already accepted.
- `ripple_sound` and `render` type their `pattern` argument with a public
  Protocol, `sonore.sources.ripples.EnvelopePattern` (anything with an
  `envelope(t, x)` method), instead of the private base class `_Pattern`.
  Any class of your own with that method already worked; now its type says
  so. `docs/design/philosophy.md` states when sonore uses a class's
  constructor and when a factory function.
- `DynamicRipple`'s rate and density patterns now depend only on `seed`.
  Before, the same seed gave a different pattern for each duration (seed 3's
  rate at 0.5 s was 121, 169 and 226 Hz for sounds of 1, 2 and 4 s); now a
  shorter sound gets the start of a longer one's pattern. Each trajectory has
  its own generator, and the filtered noise is mapped to uniform with its
  exact spread, so the values are uniform over the range across draws. Every
  `DynamicRipple` pattern changes, and the gallery's dynamic moving ripple is
  rebuilt. The docstring also gives the top octave's share of modulation
  power outside `rate_range` (about 12%; the 7% is over all 5 octaves),
  measured by `tools/measure_docstring_numbers.py`, and new tests cover the
  ripple checks, carriers and plot that had none.
- Texture synthesis, from an independent re-audit of `sonore.texture`
  (no change to any output): the full-loop test now checks the result's
  subband levels and long-term spectrum against the target's, and new
  tests pin `measure(window=...)`, the Hilbert envelope and C2's phase
  convention; `synthesize` says its length is a whole number of envelope
  samples and that `init` sets the length; the growing per-channel
  conjugate-gradient budget is documented; `DIFFERENCES_FROM_TOOLBOX`
  lists the toolbox's 7 s cap and non-circular envelope resampling.

- `so.note_to_freq` accepts only a whole-number octave (digits with an
  optional minus): names such as `"A 4"`, `"A+4"` or `"A4.0"` now raise
  ValueError. Its docstring lists the spellings it reads (letters in either
  case, `#`/`♯` and `b`/`♭`, outer spaces ignored) and says it is equal
  temperament with A4 at 440 Hz (ISO 16) by default, as do `so.cents_scale`
  and `so.chroma`.
- `so.event_train` refuses a timing distribution that gives only zero
  intervals, which used to loop forever while using more and more memory.
  Its docstring says onsets fall on the nearest sample and that the expected
  power grows with the rate only for an event with no steady (DC) part.
- `so.quilt` needs a sound with at least one segment more than the quilt
  uses. A quilt of every segment could be left with only a forbidden
  segment for its last place, so it failed on some seeds and not others;
  it is now refused for every seed.
- `so.splice` and `so.quilt` docstrings say that a silent side of a join is
  joined as with equal gain, that `splice` reads the first segment where
  asked, and that a quilt never starts on the sound's last segment.
- `so.splice` searches every shift up to `max_shift` at every join, reading
  past either end of the sound as silence. Before, a shift had to keep its
  read inside the sound, which left a segment at the sound's very start or
  end one allowed shift, the full `max_shift`, and no alignment; in a quilt
  that was the sound's first segment placed anywhere but first, and its
  last segment ending the quilt. Quilts that drew either segment change.
  Contiguous segments of noise now rebuild the sound with the default
  `max_shift` too. Found by a re-audit of sitting 3b.
- `so.cents_scale` returns the same scale for the same reference, so
  `cents_scale(440) == cents_scale()` and both equal the named `"cents"`
  scale; before, two calls gave scales that compared unequal.
- `so.klatt_synthesize` no longer opens a voiced sound with a click. The
  radiation difference started from 0, so the voice's first sample was the
  whole glottal value, 21 times any later one; it also made the voice
  quieter than its dB (3.7 dB for a 0.1 s sound, 1.0 dB for 0.5 s). Every
  Klatt sound changes slightly. The docstring says that `AB` needs `AF` on
  and where in each period the noise is at full level.
- `so.envelope_peaks` docstrings say the exponent describes pops only when
  they fill more bins than the fraction fitted (more than about 10 per
  second with the defaults); sparser pops read below 1 whatever their
  sizes, and a smaller `fraction` fits them. They give Gaussian noise's
  exponent at 8, 16 and 44.1 kHz instead of "20 to 40". The design note
  adds the measurements and marks the survey's tapping as probably sparse.
- `so.time_stretch` docstring says how far the achieved stretch can be from
  the requested factor (at most 1.1% for factors 1 to 4 at 16 kHz with the
  default window, measured by tools/measure_docstring_numbers.py), that the
  end is then cut or silent, and that a longer `win_dur` narrows the gap.
- `so.gaussian_spectrogram` docstring: the example passes `carrier=` (it
  raised a TypeError since `to_sound` options became keyword-only) and
  normalizes the sound, and a paragraph gives the sound's level (RMS 29 dB
  above 1 at the median, 21 to 43 dB for 90% of draws, measured by
  tools/measure_docstring_numbers.py). Output unchanged.

- Gallery regrouped: Synthetic sounds (was Stimuli), Seeing and changing
  sound (Seeing speech, Analysis and resynthesis, Phase vocoder and the
  envelope and modulation pages, in one group), Textures (Sound textures and
  Sound quilts, now after the modulation pages), Voices (now after
  Textures), Spatial hearing, Music. The page "Synthetic sounds" is now
  "Tones and noises". Page URLs are unchanged; the scripts of Seeing speech, Sound
  textures and Sound quilts moved to docs/gallery/seeing and
  docs/gallery/textures.
- Gallery: "Events and pops", a page of its own in the Textures group, takes
  the event-train and fire-pop sections ("From single events to a texture",
  "How big the pops are") off Sound textures. Old links to them on Sound
  textures forward to the new page.
- Gallery: "Finding formants", a page of its own in Voices after Formant
  synthesis, takes linear prediction, the two talkers' formant tracks and
  the copy synthesis off Formant synthesis. Old links forward to it.
- Gallery: 'A "cocktail party"', a page of its own in Spatial hearing after
  Moving talkers, takes the five walking-talker scenes off Moving talkers.
  Old links forward to it.
- `ModulationSpectrum.to_sound` is now a recipe: the envelopes of
  `to_envelopes`, put on the carrier by `TemporalEnvelope.to_sound`, so the tones
  are re-filtered by the filterbank's synthesis like every other carrier.
  `carrier` must be named (no default) and is `"tone"` (was `"tones"`; the
  tones now start at zero phase, as in `TemporalEnvelope.to_sound`) or a Sound.
  `carrier="noise"` is gone: it meant a noise's fine structure here and
  bands of noise in `TemporalEnvelope.to_sound`; pass a noise as the Sound for the
  first. Measured in `tools/check_modulation_carriers.py`.
- `ModulationSpectrogram.average()` averages only the time windows where a
  cell is `valid`, and is NaN for a band and rate with none (a rate faster
  than the band is wide, or a window longer than the sound). It used to
  count the windows that run past the ends, which read 3-4 dB high at slow
  rates on a sentence.
- `views/modulation.py` opens with `ModulationSpectrum` (Singh & Theunissen,
  2003); the modulation filterbanks follow.

### Removed

- `so.Envelope`, `so.Envelopes` and `so.GridEnvelope` (see Changed:
  `TemporalEnvelope` and `SpectralEnvelope`), `SpectralEnvelope.to_world` and
  `Aperiodicity.to_world` (write `np.ascontiguousarray(view.data[0].T)` for
  pyworld's layout), `SpectralEnvelope.q1` (now `settings["q1"]`), and the
  old `SpectralEnvelope(data, t, fs, q1)` constructor.
- `so.world_synthesize` (WORLD's synthesis, ported sample for sample), with
  `sonore.views.world`, `so.DIFFERENCES_FROM_WORLD` and WORLD's noise
  generator. The new `so.source_filter_synthesize` (see Added) takes its
  place; for WORLD's own output use pyworld. sonore no longer holds any of WORLD's code, so the
  package license is MIT alone.

### Fixed

- `so.pitch_adaptive_envelope` no longer returns NaN for a whole time window when part of the spectrum
  is empty, as above the last harmonic of a synthesized vowel: the smoothing's roundoff could leave
  a power just below zero there, and its logarithm was NaN.
- `so.f0_track` with `f_hi` raised for singing: a steady high voice has a
  dip at every multiple of its period, more than the eight candidates, and
  the period itself could be left out, so a sung 600 to 1150 Hz vowel came
  out at F0/2, F0/3 or unvoiced (9 of 18 test vowels and F0s). As in YIN,
  the shortest-lag dip within 0.1 of the deepest is now always offered.
  The laryngograph numbers are unchanged; on the sentences in `docs/speech`
  the F0 changes on 0.1% of time windows, by at most 1.4%.
- `so.f0_track` of an empty sound gives an empty track instead of raising
  `IndexError`, and `so.scale_f0` with `range` other than 1 multiplies by
  `ratio` the candidates of a channel with no voiced time window (they were
  left unchanged).
- The `f0_track` and `F0Track` docstrings: where each stage comes from
  (YIN, StoneMask, RAPT and Praat; the multiple rule is sonore's own), the
  candidates' shape (8 per time window, not 4), the measured effect of the
  multiple rule, that the F0 can come out up to 10% outside `f_lo`-`f_hi`,
  and that lowering `f_lo` below 60 Hz is not reliable. From audit sitting
  12, with tests for every surviving deliberate break that changes the
  output.

## [0.7.0] - 2026-10-14

Sound quilts (`so.quilt`, `so.splice`) and event trains (`so.event_train`),
from single events to a texture. Options are now keyword-only, sonore ships
type annotations and needs Python 3.11. `ModulationSpectrum` has public hooks
for its search, `HRIRSet.at` refuses directions in the hole below PKU-IOA's
lowest ring, and the gallery adds a Sound quilts page and an "Up and down"
section on Binaural cues.

Breaking changes since 0.6.0 (each is described under Changed):

- Options are keyword-only. A call passes the arguments that say what is made
  or analyzed by position and every option by name, as NumPy, SciPy and
  librosa do: `so.harmonic_complex(1.0, fs, 200, harmonics=range(1, 11))`,
  `so.STFT(snd, win_dur=0.025)`, `snd.normalize(rms=0.05)`. Calls that passed
  an option by position raise a `TypeError`.
- sonore needs Python 3.11 or newer.

### Added

- sonore ships `py.typed`, so type checkers read its annotations, and CI
  checks them with mypy. String options are typed as the values they accept
  (`Literal`), so an editor can offer them and a checker can catch a typo.
- `so.event_train`: copies of one event (a chirp, say) scattered in time,
  from single events to a dense texture. The time between onsets, each
  copy's level and each copy's pitch are drawn from distributions given as
  numbers or SciPy distributions, and a rate track speeds the onsets up or
  slows them down over time (docs/design/sources/event-texture.md). Each
  transposed copy is the event read with the windowed sinc `move_sound`
  uses, so a pitch spread renders in under a second rather than seconds
  (docs/design/sources/event-train-speed.md).
- `so.quilt`: the sound quilts of Overath et al. (2015), segments of a
  sound in a new order, each chosen so that its join changes the
  cochleagram as much as the sound itself did there; and `so.splice`, which
  joins segments of a sound with cross-fades aligned by cross-correlation
  and, by default, scaled by the correlation at each join so that the
  level stays the same (`fade="equal_gain"` gives the paper's ramps;
  docs/design/views/quilting.md).
- `ModulationSpectrum` has public hooks for the search `to_sound(iterations=n)`
  takes: `filterbank`, `modulation_phase(sound)`, `to_envelopes(phase=...,
  quiet=True)` and `iterate(sound)`, one step of the search, which
  `to_sound` now calls. Every existing call gives the same samples. Code
  that called the private `_modulation_phase` or `_rebuild` (sonore-sketch's
  two modulation tabs) should move to these names.
- `ModulationSpectrum.to_sound(carrier=sound, rng=seed)` draws the modulation
  phase from `rng` and takes only the fine structure from the Sound, so a
  drawn target (`from_blobs`) can be heard on a harmonic complex, a
  recording or a vowel. This call used to raise a `TypeError`.
- Gallery: an "Up and down" section on the Binaural cues page, noise at four
  heights in front and noise passing over the head, through PKU-IOA's HRIRs
  at 1 m, with the measured ITD and ILD in the median plane next to those at
  the side (docs/design/spatial/elevation.md, `tools/check_elevation_claims.py`).

### Changed

- Every public function and method takes its options as keywords only: the
  arguments without a default (and, where they name what is made, the first
  defaulted one, such as `n_bands, f_lo, f_hi` of a filterbank or a
  `channel_vocode`, `ax` of a `.plot`, or the voice quality `rd`) stay
  positional; everything after them goes after a `*`.
- A ripple pattern's `.plot` and `so.plot_ripple_pattern` take `ax` second,
  like every other plot; `duration`, `f_lo` and `f_hi` are keywords.
- The minimum Python is 3.11 (3.10 reaches its end of life this month), and CI
  tests 3.11 and 3.14.
- CI checks formatting with `ruff format --check` on the package, tests and
  tools.
- Gallery: Sound quilts is a page of its own under Stimuli, with quilts
  played at their recording's gain; the event-train demos on Sound textures
  get the texture page's plots; the sidebar folds its groups per page and
  carries the logo, API reference and GitHub links. The README's references
  move to `docs/references.md` and its gallery list is shorter. Links to other
  sites open in a new tab.
- `docs/philosophy.md` states the keyword-only and typing rules.

### Fixed

- `HRIRSet.at`, and so `spatialize` and `move_sound`, raise a `ValueError` for
  a direction in a gap in the measured directions, such as below PKU-IOA's
  lowest ring at −40°. The convex hull closed that hole with triangles whose
  corners are far apart, so a direction below −40° was blended, without a
  warning, from directions up to 180° apart in azimuth (straight below was
  half 105° and half 285°). Every direction the measurements cover gives the
  same HRIR as before (docs/design/spatial/elevation.md).

## [0.6.0] - 2026-10-10

Linear prediction and formant tracking: `so.LPC` and `so.formant_track`, with
copy synthesis of the gallery's two talkers through the Klatt synthesizer.
Also new: `so.gaussian_spectrogram`, a `.plot()` on every view, envelopes heard
on any carrier through `to_sound`, a faster `move_sound` with HRIR onsets fitted
across distances, and the fixes from audit sittings 9 and 10. The gallery is
rewritten around a male and a female talker, every page is also a notebook, a
landing page runs sonore in the browser, and sonore has a logo.

Breaking changes since 0.5.0 (each is described under Changed):

- `sonore.views.spectrum` and `sonore.views.reassigned` are one module,
  `sonore.views.spectra`; `sonore.views.descriptors` is `sonore.views.timbre`.
  The `so.` names are unchanged.
- `Spectrum.from_sound` levels are a power spectral density in dB re 1 per Hz
  (they were dB of the unnormalized FFT magnitude).
- `noise_vocode` is `channel_vocode`: its carrier can be noise, tones or any Sound,
  and the noise vocoder of Shannon et al. (1995) is its default
  `carrier="noise"`. The old name is gone.
- `channel_vocode` with the noise carrier sounds different (see Changed).
- `Envelopes * Envelopes` and `Envelopes * Envelope` are gone: nothing used
  them, a `Mask` on `Subbands` gives each band its own gain over time, and
  `envelope * sound` before the analysis gives every band the same one.
  `Envelopes` multiply `Subbands` and numbers.

### Added

- `so.LPC`: linear prediction of each time window of a `Sound` or an `STFT`
  by the autocorrelation method (`so.levinson`), with the all-pole envelope
  (`envelope`, `envelope_view`) and every resonance from the roots
  (`candidates`). `so.formant_track` → `FormantTrack`: F1 to F3 every 5 ms
  from Praat's recipe for candidates and a Viterbi tracker; `track(n)` gives
  the `(times, values)` that `klatt_synthesize` takes. The tests compare it
  with Praat's formants, stored by `tools/make_lpc_fixtures.py`. See
  `docs/design/views/lpc.md`. The gallery's "Formant synthesis" page has a
  section on them, "Finding the formants again", with copy synthesis of the
  two gallery talkers.
- `so.gaussian_spectrogram`: random spectrograms whose levels correlate
  exponentially in time and frequency (McDermott, Wrobleski & Oxenham, 2011),
  returned as `Envelopes` and heard through `to_sound`
  (docs/design/sources/gaussian-spectrogram.md).
  The gallery's "Hearing a modulation spectrum" page has a section on them,
  and its "Classic stimuli" page is now "Synthetic sounds".
- `ModulationSpectrum.from_blobs(..., scale="db", sd_db=6)` draws blobs as the
  modulation spectrum of envelopes in dB. Nothing has to be clipped, so the
  modulation can go deeper than a linear draw's limit of about 0.28. The
  default stays linear.
- Every view has `.plot()` (docs/design/layout/plotting.md). A view with no
  single picture refuses with `NotImplementedError` and its `no_plot`
  sentence, which names what to plot instead: `ReassignedSpectrogram`
  (plot `.binned(...)`), `PVAnalysis` (plot the STFT) and `TextureStats`
  (plot the arrays you need).
- `so.plotting.plot_descriptor_track`.
- `Envelopes.to_sound(carrier)` and `Envelope.to_sound(carrier)`: the
  envelopes on a carrier's fine structure (`"noise"`, `"tone"` or a Sound
  for `Envelopes`; a Sound for `Envelope`). `Envelopes.to_sound` is
  `Envelope.to_sound` in every band followed by the filterbank's synthesis.
  `channel_vocode` is now a recipe over `Envelopes.to_sound`.
- A landing page at choyun1.github.io/sonore with one editable code cell
  (the tutorial's tone in noise) that installs sonore from PyPI and runs it in
  the browser through Pyodide (`docs/index.html`, `docs/try/runner.py`,
  `docs/design/landing-page.md`).
- A logo: three lines and the strokes of an s, drawn from the spectrogram of
  "sonore" said by the Klatt synthesizer, its tracks taken from a recording of
  the word with `so.f0_track` and `so.formant_track` (`docs/logo/`). The README
  opens with the banner and the gallery uses the mark; the Formant synthesis
  page ends with the word as an example.
- Every gallery page is also a notebook in `docs/gallery/notebooks`, linked
  from the page and openable in Colab.
- `tools/crosscheck_gammatone_hohmann.py` compares `so.gammatone_filterbank`
  with a Hohmann (2002) style IIR gammatone bank.
- The "Tuning and temperament" gallery page lets you hear the meantone wolf fifth.

### Changed

- `Spectrum.from_sound` gives the one-sided power spectral density,
  `|X|^2 / (fs N)` with every bin but 0 Hz and Nyquist counted twice, in dB
  re 1 per Hz: the scale of `long_term_spectrum`, so the two agree for a
  steady noise. They were about 80 dB apart for one second of noise at
  16 kHz. `Spectrum.to_sound` reads levels on this scale.
- A sound made from a `Spectrum` (`Spectrum.to_sound`, or
  `gaussian_noise(spectrum=...)`) is silent outside the spectrum's
  frequencies; it held the last level up to the new Nyquist, so a spectrum
  measured at 16 kHz gave a 44.1 kHz noise energy above 8 kHz.
  `Spectrum.level_at` still holds the end values.
- `long_term_spectrum` zero-pads a sound shorter than `win_dur` onto the same
  frequencies (it raised a numpy broadcasting error when mixed with longer
  sounds) and refuses an empty list.
- `channel_vocode` (and `Envelopes.to_sound("noise")`) puts each envelope on a
  band of noise, which keeps its own random envelope fluctuations, as in the
  classic noise vocoder and as the vocoder gallery page describes. It used
  the noise's fine structure, which flattened those fluctuations (the median
  envelope standard deviation over mean of a steady harmonic complex's
  output bands was 0.32; it is now 0.51). The vocoder and harmonics gallery
  pages' noise-vocoded sounds are rebuilt. A carrier Sound at another sample
  rate is refused with "sample rates differ"; the docstring describes the
  vocoder as after Shannon et al. (1995), with Hilbert envelopes.
- Envelopes refuse non-finite values, so dividing an `Envelope` by zero
  raises instead of holding `inf` and `nan`.
- `DescriptorTrack.plot()` draws through `plotting.py` and, given no `ax`,
  opens a new figure as every other `.plot()` does, instead of drawing on the
  current axes.
- Modules: `views/spectrum.py` and `views/reassigned.py` merged into
  `views/spectra.py`; `views/descriptors.py` renamed `views/timbre.py`.
- Docs, docstrings, comments and messages use American spelling ("behavior",
  "center", "analyze"); no public name changes.
- `so.move_sound` renders about four times faster (3 s at 44.1 kHz in 0.3 s
  instead of 1.3 s): the windowed sinc that reads the sound between samples
  is tabulated (outputs change by -117 dB for white noise), and each
  lookup point is tested only against the triangles near it (same weights).
- `HRIRSet(fit_onsets=True)`, the default: with three or more measured
  distances, each IR's two ears are delayed together so their mean onset
  follows a fit across distances (travel time plus `a + b/r` per
  direction), keeping each ITD as measured. PKU-IOA's horizontal ring at
  1.3 m arrives about 110 µs early, which made a source moving through it
  at 15 m/s 10 cents too sharp between 1.3 and 1.6 m and 10 cents too flat
  between 1.0 and 1.3 m; it now follows the travel time to within about
  2 cents there. `spatialize`, `HRIRSet.at` and `move_sound` all use the
  fitted onsets; `fit_onsets=False` keeps the measured ones. A source
  coming at the head at 300 m/s is now rendered instead of refused.
- Gallery: every page opens with the same setup cell, `docs/gallery/common.py`
  in full. The Voices, Phase vocoder, Envelopes and modulation and Spatial
  pages play the male and the female talker side by side instead of ending in
  a "higher voice" section; a new Two talkers page measures how they differ,
  Pitch tracking is its own page, and Voices from harmonics is merged into
  Rebuilding and changing a voice. Reverberation is now Rooms and binaural
  beats move to Binaural cues. Numbers in the prose are computed on the pages,
  claims the pages could not back are removed or cited, and old links forward
  to their new place.
- README: tests, PyPI, Python and license badges; the sigtools migration guide
  moves to `docs/migrating-from-sigtools.md` and the finished roadmap to
  `docs/roadmap-done.md`; links to sonore-sketch.
- Choices among argument kinds and operator overloads are written as `match`
  statements; behavior is unchanged.

### Fixed

- `attack_segment` and `log_attack_time` zero-pad before the Hilbert
  transform. A sound that ended loud wrapped round to its start, so its
  attack started at 0 s: 0.5 s of silence and a 10 ms rise measured as
  531 ms. It now measures 494-531 ms; tones that fade out are unchanged.
- `Envelopes.modulation_spectrum` dropped the first and last bands of a
  filterbank built with `edges=False`, which has no edge bands, so two real
  bands were lost. It now drops edge bands only where they exist.
- Envelopes at a rate whose ratio to the target is not a small fraction are
  resampled through the FFT instead of a huge polyphase filter: a 1 s band
  at 314.16 Hz brought to 44.1 kHz took 3.6 s and now takes about 10 ms.
  Ratios that need factors above 1000 also used linear interpolation when
  the fraction was inexact; they are now band-limited.

## [0.5.0] - 2026-10-05

Sound from a modulation spectrum: `ModulationSpectrum.to_sound(carrier=...)`,
spectra edited with `with_gain` or drawn as Gaussian blobs with
`ModulationSpectrum.from_blobs`, and an optional Griffin & Lim style search.
The filterbanks become one `so.Filterbank` class on any `so.FrequencyScale`
(ERB, octave, cents, mel, linear), with `so.cents_scale` and helpers for
cents and note names. Also new: timbre descriptors after Peeters et al.
(2011) with a Timbre gallery page, and the fixes from audit sittings 1-5, 7
and 8.

Breaking changes since 0.4.0 (each is described under Changed):

- `so.ERBFilterbank`, `so.OctaveFilterbank`, `so.CosineFilterbank`,
  `so.GammatoneFilterbank` and `so.MorletFilterbank` are gone; use
  `so.cosine_filterbank`, `so.gammatone_filterbank`, `so.morlet_filterbank`
  or `so.Filterbank`. A bank's `tight` attribute is now `bank.is_tight(n_samples, fs)`.
- `so.subbands(snd, ...)` is gone (write `so.cosine_filterbank(...).analyze(snd)`),
  and `Subbands.synthesize()` is `Subbands.to_sound()`.
- `so.pad` and `so.truncate` are replaced by `so.match_lengths(sounds, mode=...)`.
- `Mask` moves from `sonore.frames.mask` to `sonore.views.mask` and is a View
  (`so.Mask` is unchanged).
- `so.gaussian_noise` loses `tilt_ref`; `Sound.fs` is read-only.
- The LF source (`so.lf_harmonics`, `so.lf_pulse`, `so.glottal_source`,
  Klatt's `RD`) refuses Rd outside 0.3-2.7.
- Output changes for the same arguments: gammatone and Morlet centers move
  slightly (`f_lo` and `f_hi` are now the outer knots); the default top band
  edge is the Nyquist frequency rather than 95% of it; `so.correlated_noise`
  hits the requested correlation exactly; a random-phase
  `ModulationSpectrum.to_sound` keeps the long-term spectrum.

### Added
- Timbre descriptors, written from Peeters et al. (2011): `so.log_attack_time`
  and `so.attack_segment` (the weakest-effort attack on an energy envelope
  low-passed at 20 Hz, forward and backward), `so.spectral_centroid` and
  `so.spectral_flux` (spectra 100 ms apart by default), the last two returning
  a `so.DescriptorTrack` with `.median` and `.iqr`. The centroid is of the power
  spectrum by default (`scale="magnitude"` for the other), changed from
  magnitude before release: the magnitude centroid of a dull tone is dominated
  by the window's sidelobes and changes with the sample rate (n^-3 at E-flat 4:
  810 Hz at 44.1 kHz, 592 Hz at 22.05 kHz, against 414 Hz from the partials).
  Design and measurements in `docs/design/music/timbre-page.md`; the Timbre
  gallery page uses all three.
- `so.ratio_to_cents`, `so.cents_to_ratio` and `so.note_to_freq(note, a4=440.0)`
  (equal-tempered frequency of a name like "C#3" or "Bb2"), beside `so.cents_scale`,
  which now uses the same cents formula. Design in `docs/design/music/music-pages.md`.
- `ModulationSpectrum.to_sound(carrier=...)`: a sound whose envelopes have
  the spectrum, the carrier supplying what it lacks, the modulation phase
  and the fine structure. A `Sound` lends its own (so the spectrum of `x`
  with `carrier=x` gives `x`'s envelopes back); `"tones"` (steady tones at
  the band centers, the default) and `"noise"` draw a random modulation
  phase. `ModulationSpectrum.with_gain(g)` edits a spectrum, for example
  `g = lambda rate, density: abs(rate) <= 4`, and `to_envelopes` gives the
  rebuilt envelopes. Spectra made from envelopes keep how they were made
  for this. Design and measurements in
  `docs/design/views/modulation-targets.md`.
- `ModulationSpectrum.from_blobs(blobs, duration, rms_depth=0.2)` draws a
  target in code as a sum of `so.ModulationBlob(rate, density, ...)`
  Gaussian patches, on the grid `ModulationSpectrum.octave` measures. Its
  `to_envelopes` and `to_sound` refuse a draw whose envelopes would go below
  zero and name the largest depth that fits it.
- `ModulationSpectrum.to_sound(..., iterations=n)` searches for a sound
  whose own envelopes come closer to the spectrum, Griffin & Lim's iteration
  through the filterbank: on a sentence with every rate above 4 Hz removed,
  20 iterations from its own fine structure leave 16.5 dB less power at
  6-40 Hz, against 3.5 dB in one step.
- Gallery page "Hearing a modulation spectrum" (Seeing and changing
  sound): a sentence and its random-phase twin, a drawn blob on tones,
  noise and the sentence, and the sentence with every rate above 4 Hz
  removed, with and without iterations. Renamed "Hearing a modulation
  spectrum" when it gained the sentence and rain trading magnitudes and
  timing, and crickets and a fire with twins on the plane and band by band.
- `so.cents_scale(reference=440.0)`: cents from a reference frequency,
  `1200 log2(f / reference)`, as a frequency scale; `scale="cents"` is cents
  from A4 at 440 Hz. A filterbank on it with `spacing=100` has one band per
  equal-tempered semitone (the octave bank with `spacing=1/12`).

### Changed
- `so.lf_harmonics`, `so.lf_pulse`, `so.glottal_source` and Klatt's `RD`
  refuse Rd outside Fant's (1995) range 0.3-2.7, the range his predictions
  were fitted to (above it the pulse no longer peaks at -1 at the
  excitation, and near Rd 7 its flow goes negative). Given directly, `rk`
  must be below 1 (at 1 the pulse blew up silently). An F0 contour that is
  never voiced gives silence from `so.glottal_source`, as from
  `so.harmonic_complex`, instead of an error.
- `so.gaussian_noise` loses `tilt_ref`, which had no effect: it scaled the
  whole spectrum, and the output is scaled to RMS 1 anyway.
- `so.correlated_noise` gives exactly the requested interaural correlation and
  RMS 1 in each channel: the two noises are made orthogonal and equal in power
  before mixing (Hartmann & Cho, 2011). Before, each draw missed `corr` by
  chance (SD about 0.2 for 9 spectral components).
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
  (`"erb"`, `"octave"`, `"cents"`, `"mel"`, `"linear"` or your own
  `so.FrequencyScale`, which lives in `core/utils.py` beside the ERB and mel
  converters), the
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
- A `FrequencyScale` checks that its two conversions invert each other where it is
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
- The Colab notebook (`docs/notebooks/start.ipynb`) is now a short tutorial
  in six steps: sounds, canonical stimuli, glides, ripples, iterated rippled
  noise and noise vocoding, sound textures, voices (three F0 trackers,
  WORLD-style pitch and formant changes, a phase vocoder, a Klatt vowel),
  and spatial hearing (interaural cues, HRTFs, reverberation, a talker
  walking past). Every sound plays with its overview below it. It runs on
  sonore 0.4.0 from PyPI.
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

### Fixed
- `ModulationSpectrum.to_envelopes` and `to_sound` with a random modulation
  phase (`carrier="tones"` or `"noise"`) scrambled the long-term spectrum:
  which band is loud is set by the phase of the zero-rate column, and the
  draw randomized it too. It now keeps that column's own phase. The
  "Hearing a modulation spectrum" gallery page's random-phase demos are
  rebuilt.

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
  `so.hcc_trajectory` builds one in head-centered coordinates, and
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
  `TVSTFT`. Options for the Slaney mel scale, area-normalized triangles, a
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

[Unreleased]: https://github.com/choyun1/sonore/compare/v0.9.0...main
[0.9.0]: https://github.com/choyun1/sonore/releases/tag/v0.9.0
[0.8.0]: https://github.com/choyun1/sonore/releases/tag/v0.8.0
[0.7.0]: https://github.com/choyun1/sonore/releases/tag/v0.7.0
[0.6.0]: https://github.com/choyun1/sonore/releases/tag/v0.6.0
[0.5.0]: https://github.com/choyun1/sonore/releases/tag/v0.5.0
[0.4.0]: https://github.com/choyun1/sonore/releases/tag/v0.4.0
[0.3.1]: https://github.com/choyun1/sonore/releases/tag/v0.3.1
[0.3.0]: https://github.com/choyun1/sonore/releases/tag/v0.3.0
[0.2.0]: https://github.com/choyun1/sonore/commit/213e21a
