# Finished roadmap items

Items moved off the README roadmap once done, oldest first. Changes by
release are in [CHANGELOG.md](../CHANGELOG.md).

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
- **Gallery pages.** Seventeen pages, listed under [Gallery](../README.md#gallery), each
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
  [Rebuilding and changing a voice](https://choyun1.github.io/sonore/gallery/voice.html) gallery page.
- **Klatt-style formant synthesizer.** `so.klatt_synthesize` after Klatt
  (1980): harmonic voicing with Klatt's glottal spectrum, aspiration and
  frication noise, formants in cascade and in parallel (alternating signs,
  which match the cascade between peaks to 0.15 dB where equal signs miss
  by 15 dB), radiation, and parameters as tracks interpolated to every
  sample. `so.resonator` and `so.antiresonator` are its formants. A vowel's
  harmonics equal source x formants x radiation to 1e-6 dB; see
  `docs/design/sources/klatt.md` and the
  [Formant synthesis](https://choyun1.github.io/sonore/gallery/sourcefilter.html#h-synthesis-by-rule) gallery page.
- **WORLD vocoder.** `so.cheaptrick` (spectral envelope; Morise, 2015),
  `so.d4c` (aperiodicity; Morise, 2016) and `so.world_synthesize` reproduce
  WORLD (Morise et al., 2016, the successor of STRAIGHT, Kawahara et al.,
  1999) in NumPy, including its own noise generator: on the gallery
  sentence they match pyworld to 4e-9 dB, 7e-12 dB and 1e-13 of the peak,
  and the tests compare against stored WORLD output, so pyworld is not a
  dependency. Options that depart from WORLD are listed in
  `so.DIFFERENCES_FROM_WORLD`. `so.harmonic_aperiodicity` measures the
  share of noise directly, beside D4C. See `docs/design/views/world.md` and the
  [A richer source-filter model](https://choyun1.github.io/sonore/gallery/richer.html) gallery page.
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
  [Spectral envelope](https://choyun1.github.io/sonore/gallery/sourcefilter.html#h-mfccs-a-cepstrum-on-the-mel-scale)
  page shows how much a vowel's MFCCs move with its pitch.
- **Voice changes, any method.** `so.scale_f0` changes the pitch and
  `so.warp_frequency` moves the formants, on any F0 contour (`f0_track`,
  Harvest, `Cepstrum.f0`) and any envelope (CheapTrick, the cepstrum,
  MFCCs), and both synthesizers take any envelope.
  `tools/compare_voice_methods.py` compares the trackers and envelopes at
  resynthesis and voice change; see `docs/design/views/voice-change.md` and the
  [Rebuilding and changing a voice](https://choyun1.github.io/sonore/gallery/voice.html) gallery page.
- **API reference and test layout.** An API reference built from the
  docstrings in CI, and a test folder that mirrors `src/sonore`.
- **Faster gallery build.** Each figure is drawn once rather than twice;
  the images are byte for byte the same.
- **More moving talkers and rooms.** Straight paths across the plane and a
  path no real source could take on the Moving talkers page, and the
  gallery sentence in each rule-breaking room on the
  [Rooms](https://choyun1.github.io/sonore/gallery/reverb.html) page.
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
- **Cocktail party scenes.** Two to six talkers walking and talking in a
  room on the [Moving talkers](https://choyun1.github.io/sonore/gallery/moving.html)
  page, 20 to 30 s long, with speech from LibriSpeech dev-clean (CC BY 4.0).
- **Sound from a modulation spectrum.** `ModulationSpectrum.to_sound(carrier=...)`,
  spectra edited with `with_gain` or drawn as blobs, and an optional
  Griffin & Lim style search; see the
  [Hearing a modulation spectrum](https://choyun1.github.io/sonore/gallery/modtargets.html) page.
  Released as 0.5.0 ([10.5281/zenodo.23148920](https://doi.org/10.5281/zenodo.23148920)).
- **Texture synthesis checked against the MATLAB toolbox.** The toolbox
  (v1.7, run in GNU Octave) and sonore each synthesized 5 s from the same
  four recordings (fire, stream and applause excerpts from
  `docs/textures`, and the paper's Fire1) with the paper's settings, 60
  iterations. Scored with sonore's statistics (which favors sonore, since it
  optimizes those definitions), sonore matched or beat the
  toolbox in every class except modulation power, where the two came within
  about 2 dB of each other
  (27 to 30 dB on stream, applause and Fire1). The roadmap's earlier
  "modulation convergence" item (18 to 23 dB) came from 30-iteration runs,
  not from a fault in the objective, which is the paper's and the
  toolbox's unweighted sum of squared errors. Weighting each class by the
  size of its target lifts modulation power further but did not make fire
  sound better, so synthesis output is unchanged. A one-off comparison;
  the scripts are not part of the repository.
- **Envelope peaks (`so.envelope_peaks`).** How big a sound's pops are: the
  largest envelope value in each 5 ms bin above 300 Hz and 2 kHz, and the
  power-law exponent of their upper tail by the Hill estimator, with its
  uncertainty. On the 168 recordings of McDermott & Simoncelli (2011), 14
  have an exponent below 2 (tapping, fire, castanets, a typewriter), and
  the exponent does not drift with the excerpt's length as kurtosis does.
  The measurement is checked by `tools/check_envelope_peaks_claims.py`;
  the survey of the recordings is not part of the repository
  (`docs/design/views/envelope-peaks.md`).
- **Sparse pops: background plus pops.** Some texture syntheses lose their
  original's pops, or, like our fire's, make too many of them medium-sized
  over too loud a background. A longer synthesis or more iterations did not
  help by ear. Synthesizing the recording with its pops limited away and
  adding its own pops back with `event_train`, at the exponent and rate
  measured with `so.envelope_peaks`, sounded better to one listener on our
  fire, two of the paper's fires and a clapper close to the microphone; not
  on rhythmic or impact sounds, which the paper lists among the model's
  failures. The recipe is sonore's own and untested beyond that listening,
  so it is shown as a recipe on the Sound textures gallery page, not built
  into the library; `synthesize` is unchanged
  (`docs/design/views/envelope-peaks.md`, D5).
- **Chroma, Shepard tones and the constant-Q filter bank.** `so.chroma`, the
  power in each of the 12 pitch classes per time window, from the cosine
  filterbank on the cents scale with one band per semitone (an exact,
  undecimated constant-Q filter bank, Q 17.3). `Chroma.to_sound` plays each
  pitch class as a Shepard tone, with `octaves=` for the spread over octaves
  the view drops. Shepard tones and the Risset glide are recipes for
  `harmonic_complex`, not new functions, on the "Pitch class and height"
  gallery page (`docs/design/views/chroma.md`).
- **Texture channels imposed one at a time, kept.** The roadmap asked
  whether imposing all envelope channels at once, in one optimization,
  would be faster than the loop over channels. Measured, it is not: each
  channel's objective spends about a quarter of its time on fixed per-call
  costs, so batching the channels could save at most about 12% of a
  synthesis, and the joint optimization needed two to three times as many
  objective evaluations per iteration. It came within about 1 dB of the
  loop on stream and applause and was about 4 dB worse on fire, mostly in
  the correlations between channels. `synthesize` is unchanged. A one-off
  comparison; the scripts are not part of the repository.
- **Voices gallery, four pages.** The seven Voices pages became four, in the
  order `docs/design/layout/voices-gallery.md` sets out: What is a voice (what
  makes one, how voices differ and how each difference is measured, the two
  talkers, a table of the indicators sonore computes), The source-filter model
  (formant synthesis, pitch tracking, finding formants and the spectral
  envelope on one page, opening on why the problem is solvable at all), A
  richer source-filter model (an envelope measured to the pitch period, a
  source that is partly noise, the phases, and the voice rebuilt by
  `so.source_filter_synthesize`) and Rebuilding and changing a voice
  (condensed). Every example that moved kept its key, so old links forward.
  Done 2026-10-18.
- **A decimated, invertible constant-Q transform.** `so.DecimatedFilterbank`
  keeps each band of any `Filterbank` at its own rate (nonstationary Gabor
  frames in frequency, Balazs et al., 2011), with an exact inverse; on the
  semitone cosine bank it is a constant-Q transform with about two real
  numbers per sample instead of 85. Design in
  `docs/design/frames/decimated-filterbanks.md`; the gallery section "Keeping
  fewer numbers" on Analysis and resynthesis. Done 2026-10-28.
- **Code audit.** Every module of `src/sonore` and its tests, one sitting
  per module from the bottom of the import graph up, each with a reading
  guide written by Claude (every claim and the test behind it, untested
  lines, the results of deliberately breaking the code), read by Cho with
  parts of the code, and a pull request closing the gaps it found. Sittings 1 to 23 cover every module; among the bugs they
  fixed are HRIRs losing gain when resampled (2.7 dB at 48 kHz, 12.2 dB at
  16 kHz), Klatt voices opening with a click, `DynamicRipple` patterns that
  depended on the sound's duration, and F0 halving for high voices. Progress
  and each sitting's findings are in
  [issue #48](https://github.com/choyun1/sonore/issues/48); the reference's
  intro pages, its last step, are still on the roadmap. Done 2026-10-28.
- **Streaming views.** Views computed one block of sound at a time, whose
  joined parts equal the batch analysis of the whole sound: the STFT,
  cepstrum and MFCCs, a time-domain gammatone bank and pooled envelopes
  (a cochleagram), the causal modulation spectrogram (by sliding sums,
  within 1e-12 of batch), and the F0 track with a decision delay
  (`f0_track(..., delay=)`, `F0TrackStream`). Design in
  `docs/design/streaming.md`, numbers in `tools/check_streaming_claims.py`;
  pull requests #275, #277, #279 and #281. Done 2026-10-29.
