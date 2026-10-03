# Reorganizing the package and the gallery

A proposal for how `src/sonore` and the listening gallery should be grouped
now that sonore has grown from five texture-era layers to 30 modules and 17
gallery pages. Cho asked for it on 2026-10-02: "i don't like that there's
multiple vocoder.py - under stimuli and analysis. we should also rationalize
the gallery- right now a ton falls under seeing sound category, but maybe
most of the voice stuff should be in one category, another category for more
premitive stimuli, one for spatial hearing, etc."

Status: accepted. On 2026-10-02 Cho accepted every decision with its
recommended option: D1 (a), D2 (a), D3 (a), D4 (a), D5, D6 (a), D7 (a), D8
to D11, D12 (a), D13 and D14. D2, D3 and D7 were rewritten before acceptance
after Cho's point that "voice is not a signal different in kind": there is
no `voice` subpackage, synthesizers go to `signals` and analyses of a voice
to `views`. Every count below is printed by `python
tools/count_reorganization_references.py`, run on 2026-10-02 at main
`b08063c` (before the move; the script runs only on that tree); numbers
that are not are labelled estimates.

Migration: step 3, the gallery (PR #82), and step 4, the source move
(PR #83), are merged. D13's `View` base class and `NotInvertibleError`
follow in their own PR (`src/sonore/views/view.py`), then release 0.4.0.

## What is wrong today

**The word "vocoder" names four things in three places.**

| Where | What it is |
|---|---|
| `analysis/vocoder.py` (705 lines) | WORLD's analysis: `cheaptrick`, `d4c`, `SpectralEnvelope`, `Aperiodicity`, plus sonore's own `harmonic_aperiodicity` |
| `stimuli/vocoder.py` (236 lines) | WORLD's synthesis: `world_synthesize` only |
| `stimuli/phasevocoder.py` | the phase vocoder: `pv_analyze`, `time_stretch`, `pitch_shift` |
| `analysis/filterbank.py`, `noise_vocode` | the channel vocoder of the cochlear-implant page |

Each is a procedure (take a sound apart, change it, put a sound together)
rather than one kind of object, so they need no shared home: each part goes
where its kind goes (D2).

**Synthesizers are spread over two layers.** `glottal_source` is in
`signals`, but the Klatt synthesizer and WORLD's synthesis are in `stimuli`,
though all three make a sound from parameters. The LF source is already a
`harmonic_complex` with closed-form coefficients (`signals/glottal.py:277`),
like `square_wave`, `sawtooth_wave` and `pulse_train`.

**Analyses of a voice sit apart from the other analyses.** The F0 tracker,
CheapTrick and D4C describe any periodic or harmonic sound, and the pitch
and formant changes act on any F0 contour and any envelope, yet they read as
a separate topic (`analysis/vocoder.py`, `analysis/voice.py`).

**Some modules sit in a layer they don't need.** The channel vocoder is a
stimulus but lives in `analysis/filterbank.py`. The phase vocoder is in
`stimuli` but imports only `core.sound`.

**Inside `analysis`, the files don't follow the concepts.** The `Filterbank`
base class is in `frames.py` while the banks built on it are in
`filterbank.py`; the STFT's coefficient types (`STFT`, `TVSTFT`) sit in
`representations.py` beside one-way views (`Spectrum`,
`ReassignedSpectrogram`, `ModulationSpectrum`). Yet philosophy.md already
draws the line the files should follow: every frame inverts exactly, and
views may discard information (Cho, 2026-10-02: "i like the idea of analysis
dividing into frames and non-frames/views, that should be exhaustive").

**The gallery's middle group is half of it.** Of 17 pages, 9 are under
"Seeing and changing sounds", 5 under "Stimuli" and 3 under "Listeners in
the world". The middle group holds both the time-frequency course and five
voice pages, and the binaural page sits under "Stimuli" while the other
spatial pages sit under "Listeners".

## What the layer rule does, and what it should keep doing

`tests/test_layers.py` holds a strict order, core < signals < analysis <
stimuli < texture, and fails when a module imports from a layer above its
own. Cho chose that in PR #12 so the direction of dependencies shows in the
file tree. The proposal keeps that guarantee and changes only its shape:
a trunk that everything builds on, with `analysis` divided into `frames` and
`views` (D12), and three branches by topic that build on the trunk and never
on each other.

```
              spatial        stimuli        texture      (branches: do not import each other)
                   \            |            /
                    --------- views ---------            (one-way analyses)
                                |
                             frames                      (invertible analyses)
                                |
                             signals                     (sounds from parameters: generators,
                                |                         processing, Klatt, WORLD's synthesis)
                              core
```

The measured imports allow it. Resolved name by name to where each name
would go, every import between subpackages points down the trunk or from a
branch to the trunk (signals to core 13; frames to core 9; views to frames
10, signals 1, core 12; spatial to views 1, frames 2, signals 4, core 7;
stimuli to views 1, frames 2, signals 2, core 5; texture to views 2, frames
2, core 5). No branch imports another. Two point up, both inside a method so
that calls chain in a notebook: `Sound.envelope()`, which the test already
allows by name, and `Subbands.envelopes()`, which would be added beside it
(frames to views). Three more would point up and are removed instead (D2):
WORLD's synthesis checks the types `SpectralEnvelope` and `Aperiodicity`,
and its F0 reader checks `F0Track`; in `signals` it accepts them by what
they provide, as `harmonic_complex` and `klatt_synthesize` already accept
an `F0Track` without importing it. Not counted, because today they are
calls inside one module: CheapTrick and D4C would import WORLD's shared
helpers (noise stream, FFT size, MATLAB rounding) from `signals/world.py`,
which points down.

## Proposed layout

| Subpackage | Modules | What it is | Imports from |
|---|---|---|---|
| `core` | `sound`, `units`, `utils`, `fft` | unchanged, plus the mel scale (D14) | nothing in sonore |
| `signals` | `generators`, `processing`, `klatt`, `world` | sounds from parameters: waveforms (the LF source among them), filters, and two synthesizers | core |
| `frames` | `frame`, `filterbank`, `gabor`, `mask` | invertible analyses, their coefficients, and changes to coefficients with exact least-squares resynthesis | core, signals |
| `views` | `spectrum`, `reassigned`, `envelopes`, `modulation`, `modspectrogram`, `cepstrum`, `mfcc`, `f0`, `spectral_envelope`, `aperiodicity` | one-way analyses: each says what it drops | core, signals, frames |
| `spatial` | `binaural`, `spatialization`, `hrir_data`, `reverb` | two ears, heads and rooms | the trunk |
| `stimuli` | `ripples`, `channel_vocoder`, `phasevocoder` | sounds made by shaping or changing other sounds through an analysis | the trunk |
| `texture` | `stats`, `grad`, `synth` | unchanged | the trunk |
| `plotting.py` | | unchanged | imported only inside `.plot()` |

Every move, with the number of files that name the module's current path
outside the module itself (each would need an edit):

| Now | Proposed | Files naming it |
|---|---|---|
| `analysis/frames.py` | `frames/frame.py`, `frames/filterbank.py` (`Filterbank`), `frames/gabor.py` (`GaborFrame`, `TVGaborFrame`) | 17 |
| `analysis/filterbank.py` | `frames/filterbank.py`; `noise_vocode` to `stimuli/channel_vocoder.py` | 22 |
| `analysis/representations.py` | `frames/gabor.py` (`STFT`, `TVSTFT`), `frames/mask.py`, `views/spectrum.py`, `views/reassigned.py`, `views/modulation.py` (`ModulationSpectrum`) | 17 |
| `analysis/cepstrum.py` | `views/cepstrum.py` | 12 |
| `analysis/mfcc.py` | `views/mfcc.py`; `freq_to_mel`, `mel_to_freq` to `core/utils.py` (D14) | 10 |
| `analysis/envelopes.py` | `views/envelopes.py` | 11 |
| `analysis/modulation.py` | `views/modulation.py` | 11 |
| `analysis/modspectrogram.py` | `views/modspectrogram.py` | 7 |
| `analysis/f0.py` | `views/f0.py` | 13 |
| `analysis/vocoder.py` | `views/spectral_envelope.py` (CheapTrick), `views/aperiodicity.py` (D4C, harmonic residual), `signals/world.py` (WORLD's shared helpers) | 14 |
| `stimuli/vocoder.py` | `signals/world.py` | 12 |
| `analysis/voice.py` | `views/spectral_envelope.py` (`GridEnvelope`, `warp_frequency`), `views/f0.py` (`scale_f0`) | 8 |
| `signals/glottal.py` | `signals/generators.py` | 7 |
| `stimuli/klatt.py` | `signals/klatt.py` | 6 |
| `stimuli/binaural.py` | `spatial/binaural.py` | 8 |
| `stimuli/spatialization.py` | `spatial/spatialization.py` | 8 |
| `stimuli/hrir_data.py` | `spatial/hrir_data.py` | 6 |
| `stimuli/reverb.py` | `spatial/reverb.py` | 5 |

78 distinct files are touched in all (moved modules included). By kind they
are the moved modules' importers in `src`, the API reference pages
(`docs/api/source/*.rst`, one `automodule` line per module), the README
module table and References tags, gallery scripts that import a module by
its path, the gallery HTML pages whose References tags link to a source
file, design docs, tools and tests. The tests mirror `src/`, so all 10 files
in `tests/analysis` move, `test_glottal.py` joins `test_generators.py`, and
5 more move (klatt and the four spatial ones).

Estimates from today's line ranges (labelled, not measured as built):
`signals/generators.py` about 870 lines (589 plus glottal.py's 287, less
its imports); `signals/world.py` about 460 (`world_synthesize`, 236, plus
WORLD's shared helpers, about 220); `views/f0.py` about 420;
`views/spectral_envelope.py` about 360; `views/aperiodicity.py` about 290;
`frames/gabor.py` about 570; `frames/filterbank.py` about 640;
`frames/frame.py` about 110. The longest module today is `frames.py`, 677.

What does not change: every `so.name`. The top-level `__init__` re-exports
the public names from every subpackage, and it would go on doing so, so code
written as `so.world_synthesize(...)` or `so.GaborFrame(...)` runs
unchanged. Only deep imports such as `from sonore.analysis.frames import
GaborFrame` change.

## Decisions

- **D1. Layers or topics.** Accepted (a), 2026-10-02: a trunk (core,
  signals, frames, views) and sibling branches by topic (spatial, stimuli,
  texture) that import only from the trunk. The rule changes from "below me
  in one list" to "below me in the trunk, or inside my own subpackage".
  Rejected: keeping the five layers and renaming only the WORLD modules (18
  files touched, but the synthesizers and voice analyses stay spread over
  three layers), and a flat package (Cho split it in PR #12 to see the
  dependencies).
- **D2. The four vocoders.** Accepted (a), 2026-10-02. Each is a procedure, so its parts go where
  their kinds go, and no file is called `vocoder.py`. (a) Recommended:
  - **WORLD's synthesis** to `signals/world.py`, with the helpers it shares
    with the analysis (WORLD's noise stream `world_randn`, `world_fft_size`,
    MATLAB rounding, `DIFFERENCES_FROM_WORLD`). It makes a sound from an F0
    track, an envelope and an aperiodicity, as Klatt makes one from formant
    tracks. To sit in `signals` it accepts its inputs by what they provide
    (`.t` and `.f0`; an envelope read as `envelope(t, f)`; an aperiodicity's
    grid `.t`, `.f`, `.data`) instead of checking their types: the one code
    change in the source PR, checked by WORLD's stored pyworld comparison
    (`tests/data/world_reference.npz`), which must still pass unchanged.
  - **WORLD's analysis** to `views`: CheapTrick and `SpectralEnvelope` to
    `views/spectral_envelope.py`, D4C, `Aperiodicity` and sonore's own
    `harmonic_aperiodicity` to `views/aperiodicity.py`, importing the shared
    helpers from `signals/world.py`. Only one file is named after WORLD.
  - **The channel vocoder** to `stimuli/channel_vocoder.py` (37 lines today
    inside `filterbank.py`: subbands, their envelopes, a carrier).
  - **The phase vocoder** keeps `stimuli/phasevocoder.py` (D4).

  (b) WORLD's synthesis to `stimuli/world_synthesis.py` instead: a pure
  move with no code change, but the synthesizer then sits away from Klatt
  for a reason of imports only. (c) WORLD whole in `views/world.py`: the port
  stays in one file, but a synthesizer sits among the views, against D13.
- **D3. No `voice` subpackage.** Accepted (a), 2026-10-02. A voice is made and measured with the same
  tools as any other sound (Cho, 2026-10-02: "voice is not a signal
  different in kind"), so `analysis/voice.py` splits by kind:
  `GridEnvelope` and `warp_frequency` (which moves any envelope along
  frequency) go to `views/spectral_envelope.py` with `SpectralEnvelope` and
  the interpolation helpers they share; `scale_f0` goes to `views/f0.py`
  beside `F0Track`; the private `_contour_on_grid`, used by WORLD's
  synthesis, goes to `signals/world.py`. (a) Recommended. (b) A `voice`
  branch (this proposal's earlier draft): voice code in one place, but it
  calls voice a different kind of sound, and the tracker and envelope would
  sit outside `views` though they are views.
- **D4. The phase vocoder stays in `stimuli`.** Accepted (a), 2026-10-02. It is neither a frame nor a
  view: its analysis is an STFT with an instantaneous frequency per bin, and
  what it is for is changing a sound (duration, pitch, partials), with a
  resynthesis that is not exact after a change. (a) Leave it in
  `stimuli/phasevocoder.py`, beside the channel vocoder and ripples, sounds
  made by changing or shaping other sounds (recommended, and it moves
  nothing). (b) Later, rebuild `PVAnalysis` on `GaborFrame` (today it uses
  SciPy's `ShortTimeFFT` directly); its analysis would then be a view, close
  kin to the reassigned spectrogram, and `time_stretch` and `pitch_shift`
  procedures on it. Not now: a change to working code with no new behaviour.
  *Amended 2026-10-03 (Cho, PR #86):* `PVAnalysis` is a `View` after all, so that
  every analysis is a frame or a view; it stays in `stimuli`, and
  `resynthesize` is its named, approximate route back to sound.
- **D5. Where the cepstrum and MFCCs live.** Accepted, 2026-10-02. In `views`, with the other
  one-way analyses. Package and gallery need not mirror each other: the
  gallery groups by what one listens to (its Voices group holds the cepstrum
  page, D8), the package by what a thing is.
- **D6. Old import paths.** Accepted (a), 2026-10-02. (a) No compatibility modules; the release
  notes of the next version (0.4.0) list every moved path (recommended, and
  what PR #12 did). Of the moved modules, 10 shipped in 0.3.0
  (`analysis/cepstrum`, `envelopes`, `filterbank`, `frames`, `modulation`,
  `representations`, and `stimuli/binaural`, `hrir_data`, `reverb`,
  `spatialization`), so only those deep paths can be in anyone's code; the
  other 8 were added after the release. (b) Thin modules at the old paths
  that re-export and warn, removed one version later.
- **D7. The LF source and Klatt in `signals`.** Accepted (a), 2026-10-02. `signals` becomes
  everything that makes a sound from parameters.
  - **The LF source** into `signals/generators.py` beside `pulse_train`
    (Cho, 2026-10-02: "is it really inadvisable to put lf in generators
    also?"). It stays a function, as every generator is: `glottal_source`
    carries the model in its docstring (the LF equations, Fant's Rd, why it
    is built from harmonics so it does not alias), `lf_harmonics` and
    `lf_pulse` sit beside it as public helpers, and the shape solving
    (`_r_parameters`, `_LFShape`, two root solves) as private functions
    below them, as `harmonic_complex` sits with `F0Contour` and its helpers.
  - **Klatt** to `signals/klatt.py`: it takes parameters, not a sound, so it
    is not processing; it is a third kind beside generators and processing,
    a synthesizer built from both (a voiced source from `harmonic_complex`
    or `glottal_source`, noise, `resonator` and `antiresonator`). It already
    imports only core and signals.

  (a) As above (recommended). (b) Keep the LF model in its own file
  (`signals/glottal.py`, or `signals/lf.py`), with only `glottal_source` in
  generators: about 200 of its 287 lines are the model rather than sound
  generation. (c) Klatt into `generators.py` too: one file of everything
  that makes sound, at about 1160 lines (estimate).

- **D8. Gallery groups.** (a) Four, in this order (recommended):

  | Group (folder) | Pages, simple to elaborate |
  |---|---|
  | Stimuli (`stimuli`) | Classic stimuli, Iterated rippled noise, Spectrotemporal ripples, Sound textures |
  | Seeing and changing sound (`seeing`) | Seeing speech, Analysis and resynthesis, Hearing through a vocoder, Modulation spectrogram, Phase vocoder |
  | Voices (`voice`) | Formant synthesis, Cepstral analysis, Voices from harmonics, Source, filter and aperiodicity, Changing a voice |
  | Spatial hearing (`spatial`) | Binaural cues, Synthetic reverberation, Moving talkers |

  That is 4, 5, 5 and 3 pages instead of 5, 9 and 3. The voice pages run
  from a vowel written as numbers (Klatt), to a recorded voice taken apart
  (cepstrum), rebuilt from pitch and envelope (harmonics), and with its
  noise measured and resynthesized (WORLD), and finally with its pitch and
  formants changed; the formant page comes first because it introduces
  source and filter, which the other four assume.
  The cochlear-implant page joins "Seeing and changing sound" after
  "Analysis and resynthesis", since it is the subband analysis of that page
  with the fine structure replaced. "Listeners in the world" goes; it
  suggested listener models, which are not planned.
  (b) Five groups, with "Sound textures" as its own group: room for the
  texture convergence work on the roadmap, but one page alone for now.
  (c) The vocoder page under Voices, since its examples are a sentence and a
  melody: it then sits apart from the subband analysis it is built on.
  "Primitive stimuli" (Cho's phrase) would not fit sound textures, which is
  why the first group keeps the name "Stimuli".
- **D9. Gallery folders and URLs.** Each group's scripts live in a folder of
  `docs/gallery` named after it, as now (`script_path()` in build.py); the
  folders become `stimuli`, `seeing`, `voice` and `spatial`, and
  `listeners` goes. Nine scripts change folder. The HTML pages stay flat
  at `docs/gallery/<name>.html` (all 18, the index included), so no page URL
  changes and no link in the README, the design docs or elsewhere breaks.
  Apart from each page naming its own script, only 8 other files name a
  moved script's path (`tools/check_moving_trajectories.py`, CHANGELOG.md
  and the design docs views/cepstrum.md, views/mfcc.md, sources/glottal-source.md, sources/klatt.md,
  views/voice-change.md and views/world.md).
- **D10. README.** The "More in the gallery" list (17 entries today, in no
  particular order) is regrouped under the four group names, in the order of
  TOPICS, as short bold lead-ins rather than headings, so the Contents table
  and its test do not change. The three gallery images stay. The "What's in
  it" module table is regrouped by subpackage in the trunk-then-branches
  order, and its paragraph describes the trunk-and-branches rule in place of
  the single list.
- **D11. docs/design/layout.md** is rewritten for the new rule, and
  `tools/draw_layout.py` draws the trunk as stacked bands with the branches
  side by side above it, so the diagram shows that branches do not depend on
  each other.
- **D12. Frames and views.** `analysis` divides into two subpackages, and
  every class and function in it lands in exactly one. The test for
  `frames`: it is a `Frame`, the coefficients of one (they synthesize back
  exactly), or a change to coefficients whose resynthesis is the frame's
  least-squares inverse. Everything else that takes a sound apart is a view,
  and its docstring says what it drops (philosophy.md, "Views may discard
  information").

  | Subpackage | Module | Names |
  |---|---|---|
  | `frames` | `frame` | `Frame` |
  | | `filterbank` | `Filterbank`, `CosineFilterbank`, `ERBFilterbank`, `OctaveFilterbank`, `BandpassFilterbank`, `GammatoneFilterbank`, `MorletFilterbank`, `subbands`, `Subbands` |
  | | `gabor` | `GaborFrame`, `TVGaborFrame`, `STFT`, `TVSTFT` |
  | | `mask` | `Mask`, `ideal_binary_mask`, `ideal_ratio_mask` |
  | `views` | `spectrum` | `Spectrum`, `long_term_spectrum`, `TFPower`, `tandem_power` |
  | | `reassigned` | `ReassignedSpectrogram`, `reassigned_spectrogram` |
  | | `envelopes` | `Envelope`, `Envelopes` |
  | | `modulation` | `ModulationFilterbank`, `ConstantQModulationFilterbank`, `OctaveModulationFilterbank`, `HannModulationFilterbank`, `ModulationSpectrum` |
  | | `modspectrogram` | `ModulationSpectrogram` |
  | | `cepstrum` | `Cepstrum` |
  | | `mfcc` | `MFCC`, `mel_filterbank`, `symmetric_hamming`, `delta_features` (the mel scale goes to core, D14) |
  | | `f0` | `F0Track`, `f0_track`, `scale_f0` |
  | | `spectral_envelope` | `GridEnvelope`, `SpectralEnvelope`, `cheaptrick`, `warp_frequency` |
  | | `aperiodicity` | `Aperiodicity`, `d4c`, `harmonic_aperiodicity` |

  The cases that need a word:
  - **A bare gammatone or Morlet bank** (`edges=False`) is still a
    `Filterbank` and stays in `frames`, though numerically it may not be a
    frame: `frame_bounds` reports it and `synthesize` refuses when the lower
    bound is 0. The class is a frame; a particular bank may be a poor one.
  - **The modulation filterbanks** carry the word "filterbank" but are not
    `Filterbank`s: they filter envelopes and have no synthesis. They are
    views. Renaming them is not proposed, but the docstring should say so.
  - **Envelopes** discard the fine structure, so they are a view, though
    `Subbands.envelopes()` returns them: that is the one new upward import
    (frames to views, inside the method), allowed by name in the layer test.
  - **`STFT.griffin_lim`** rebuilds a sound from magnitudes only; it is a
    method on frame coefficients and stays with `STFT`.
  - **Views of voices** (`F0Track`, `SpectralEnvelope`, `Aperiodicity`)
    are views like any other (D3), so the division is exhaustive for every
    analysis in sonore.
  - **`scale_f0` and `warp_frequency`** change a view and return one, so
    they sit with the views they change, as masks sit with frames.
  - **The phase vocoder** is neither (D4).

  (a) `frames` and `views` as two subpackages of the trunk, replacing
  `analysis` (recommended: the trunk reads core < signals < frames < views,
  and the layer test then also fails if a frame starts depending on a view).
  (b) Nested, `analysis/frames/` and `analysis/views/`: keeps the word
  "analysis" in every path, at the cost of one more level
  (`sonore.analysis.frames.gabor`). (c) Keep `analysis` as one subpackage
  and only regroup its files along the same line: no new rule for the test
  to hold.
- **D13. Only frames synthesize.** `Frame.synthesize` is already abstract
  in the frame base class (`analysis/frames.py:86`), so every frame has it.
  Views get a base class of their own, `View`, with a class attribute
  `discards` (one sentence, the mathematical reason the view cannot be
  inverted) and a `synthesize` that raises `NotInvertibleError`, a subclass
  of `NotImplementedError`, with that sentence as its message (Cho,
  2026-10-02: views should refuse "with the mathematical reason when you try
  to use synthesize"). It raises rather than returns `NotImplemented`:
  Python reserves that return value for operators such as `*`, where it
  means "try the other operand", and a method returning it would hand the
  caller a sentinel instead of an error. A test checks that every `View`
  subclass has a non-empty `discards` and that its `synthesize` raises.
  The message also names the route back to sound where one exists (see
  "What a user sees" below). Proposed sentences (to be checked against each
  docstring when built):

  | View | Discards |
  |---|---|
  | `Spectrum` | the phase, and all timing: it is a time average |
  | `TFPower` | the phase of every coefficient |
  | `ReassignedSpectrogram` | the phase, and moves energy to new positions many-to-one, so different signals give the same picture |
  | `Envelope`, `Envelopes` | the fine structure: only the Hilbert magnitude of each band is kept |
  | `ModulationSpectrum` | the phase of the modulations and the fine structure under them |
  | `ModulationSpectrogram` | the same, per time window |
  | `Cepstrum` | the phase: the real cepstrum is the transform of the log magnitude |
  | `MFCC` | the phase, the detail inside each mel band, and every coefficient past the last kept |
  | `GridEnvelope`, `SpectralEnvelope` | the harmonics, the phase, and everything finer than the envelope's smoothing |
  | `Aperiodicity` | everything but the share of noise in each band |
  | `F0Track` | everything but the pitch and the voicing |

  Views that already make a sound keep doing so under names that say what
  they assume, never `synthesize`: `Cepstrum.to_sound` borrows the phase of
  the STFT the cepstrum was computed from (exact only for an unliftered
  cepstrum with `phase="original"`, as its docstring says), `Spectrum.to_noise`
  draws a new noise with that spectrum, and `Envelopes * Subbands` imposes
  envelopes on a carrier. Each docstring says it is not an inverse.
  The modulation filterbanks are tools that make views, not views
  themselves, so they get no `synthesize` at all. (a) As above
  (recommended). (b) No `View` base class; views simply have no
  `synthesize`, and calling it fails with Python's plain `AttributeError`,
  which gives no reason.

- **D14. Frequency scales in one place.** `core/utils.py` holds the ERB
  scale (`freq_to_erb`, `erb_to_freq`, `erb_bandwidth`, exported as
  `so.freq_to_erb` and `so.erb_to_freq`) beside `amp_to_db` and
  `db_to_amp`, while the mel scale (`freq_to_mel`, `mel_to_freq`, with
  their HTK and Slaney forms) sits in `analysis/mfcc.py` and is not
  exported at the top level. Both pairs are plain unit conversions with no
  dependency on anything in sonore. (a) Move `freq_to_mel` and
  `mel_to_freq` (and their two Slaney constants) to `core/utils.py` and
  export them as `so.freq_to_mel` and `so.mel_to_freq`, matching the ERB
  pair (recommended, Cho 2026-10-02: "should that be made a part of
  utilities?"). `mel_filterbank`, `symmetric_hamming` and `delta_features`
  stay in `mfcc`, since they build MFCC features rather than convert units.
  (b) A new `core/scales.py` for the frequency scales (ERB, mel) and
  `units` for decibels: tidier once a third scale (Bark) arrives, but today
  it would be a module of five functions. The octave scale stays as it is,
  `log2` inside `OctaveFilterbank`, since it needs no function of its own.
  This rides the source PR (step 4). Four files name the mel functions
  today: `mfcc.py`, its test, and the Cepstral analysis page's script and
  HTML.

## What a user sees (D13)

Cho asked (2026-10-02) whether hearing scientists would be confused that no
view has `synthesize`. Precedent says they already expect it: LTFAT (whose
frame theory `Filterbank` follows, Balazs et al., 2011) separates frame
analysis and synthesis from everything else, and librosa keeps its
approximate inversions (`mel_to_audio`, Griffin-Lim) in a separate `inverse`
module under names that say what they assume. What would confuse is a
`synthesize` that quietly approximates. This is a newcomer's session on the
Seeing speech sentence. Lines marked "today" are printed by
`python tools/check_reorganization_walkthrough.py`; the error messages
marked "proposed" are D13's as proposed, and the same script now prints them
as built.

```python
import sonore as so
sentence = so.load("docs/speech/bdl_arctic_a0131.flac")   # 2.525 s at 16 kHz

# A frame goes both ways.
frame = so.GaborFrame(win_dur=0.025, hop_dur=0.005)
coefs = frame.analyze(sentence)          # an STFT
frame.synthesize(coefs)                  # today: the sentence, max error 2.2e-16
so.subbands(sentence, n_bands=16).synthesize()   # today: max error 6.4e-16

# A view goes one way.
mfcc = so.MFCC(sentence)
mfcc.synthesize()
# today:    AttributeError: 'MFCC' object has no attribute 'synthesize'
# proposed: NotInvertibleError: MFCC discards the phase, the detail inside each
#           mel band, and every coefficient past the last kept, so no sound has
#           these MFCCs alone. For an approximate voice, read the envelope with
#           mfcc.envelope_view() and synthesize it with so.world_synthesize.

so.subbands(sentence, n_bands=16).envelopes().synthesize()
# today:    AttributeError: 'Envelopes' object has no attribute 'synthesize'
# proposed: NotInvertibleError: Envelopes discard the fine structure: only the
#           Hilbert magnitude of each band is kept. To hear them, impose them on
#           a carrier's subbands (envelopes * subbands), as the vocoder does.

# The honest routes back to sound keep their own names.
so.Cepstrum(coefs).to_sound()            # today: exact (4.4e-16), because it
                                         # borrows the STFT's phase
so.world_synthesize(f0_track, mfcc.envelope_view(), aperiodicity)
                                         # today: 2.530 s of a voice rebuilt
                                         # from 13 MFCCs, a pitch track and D4C
```

So the message is the useful part of D13: each view's `discards` sentence
says what was lost, and a second sentence names the route back to sound
that exists, if one does. A view with no such route says so.

## Order

The reverb speech examples (PR #78) and the Changing a voice page (PR #77)
merged before this proposal was last measured, so they are counted above.
A folder move and an edit to the same script in two open PRs conflict, so:

1. Cho decides D1 to D14.
2. Any gallery PR still open then merges first.
3. **Gallery PR** (small, what readers see): `git mv` the nine scripts,
   rewrite TOPICS and the comment above it, patch the menus and sidebar of
   all 18 HTML pages in place (a full rebuild does not run in the cloud
   container), regroup the README gallery list (D10). Checks: the test
   suite, and every page's menu showing the four groups.
4. **Source PR** (mechanical, one PR so the tree is never half-moved):
   `git mv` each module, split `analysis/frames.py`, `filterbank.py`,
   `representations.py`, `voice.py` and `vocoder.py` (D12, D3, D2), merge
   `glottal.py` into `generators.py` (D7), move the mel scale (D14), let
   WORLD's synthesis accept its inputs by what they provide (D2, the only
   change to code rather than its place), update imports in `src`, tests,
   tools and gallery scripts, the
   API reference pages, layout.md and its diagram, the layer test, the
   README module table and References tags
   (`tools/update_readme_source_links.py` rewrites line anchors but not file
   paths, so the paths are edited first). D13's `View` base class and error
   come in a separate PR after this one, since they add behaviour and the
   move should add none. Checks: the full test suite, the
   texture bit-for-bit hash against the previous commit, WORLD's stored
   pyworld comparison and the Klatt and LF tests unchanged, every gallery
   script run to its last cell without writing media, and a grep for each
   old dotted path returning nothing outside the release notes and design
   history.
5. Any gallery work started after step 3 uses the new folders; any source
   work started after step 4 uses the new subpackages.
6. Release 0.4.0 with the moved paths in its notes (D6).

Steps 3 and 4 are independent; the gallery goes first because it is smaller
and is what Cho asked about most directly. Design docs written before the
move keep their old paths in their history sections; their "As built" and
file references are updated in step 4.

## Out of scope

- Splitting `plotting.py` (609 lines). Possible later, independent of this.
  (`representations.py` is split by D12.)
- Renaming public functions or classes. Everything keeps its `so.` name.
- Gallery page titles. "Seeing speech" keeps its title in the Seeing group:
  it is a course in time-frequency analysis that happens to use a sentence.
