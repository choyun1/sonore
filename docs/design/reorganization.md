# Reorganizing the package and the gallery

A proposal for how `src/sonore` and the listening gallery should be grouped
now that sonore has grown from five texture-era layers to 30 modules and 17
gallery pages. Cho asked for it on 2026-10-02: "i don't like that there's
multiple vocoder.py - under stimuli and analysis. we should also rationalize
the gallery- right now a ton falls under seeing sound category, but maybe
most of the voice stuff should be in one category, another category for more
premitive stimuli, one for spatial hearing, etc."

Status: partly accepted. On 2026-10-02 Cho accepted D1 (a), D8 to D11,
D12 (a) and D13 to D14; D2 to D7 are open. Nothing has been moved. Every count below is printed by
`python tools/count_reorganization_references.py`, run on 2026-10-02 at
main `b08063c`; numbers that are not are labelled estimates.

## What is wrong today

**The word "vocoder" names four things in three places.**

| Where | What it is |
|---|---|
| `analysis/vocoder.py` (705 lines) | WORLD's analysis: `cheaptrick`, `d4c`, `SpectralEnvelope`, `Aperiodicity`, plus sonore's own `harmonic_aperiodicity` |
| `stimuli/vocoder.py` (236 lines) | WORLD's synthesis: `world_synthesize` only |
| `stimuli/phasevocoder.py` | the phase vocoder: `pv_analyze`, `time_stretch`, `pitch_shift` |
| `analysis/filterbank.py`, `noise_vocode` | the channel vocoder of the cochlear-implant page |

WORLD is split across two layers only because the layer names say what a
module is *for* (analysis, stimuli) as well as what it may import. Nothing
in the layer rule forces the split: `world_synthesize` imports only
`analysis.vocoder`, `analysis.voice` and `core.sound`, all of which an
`analysis` module may import.

**Voice code is spread over three layers.** The LF glottal pulse is in
`signals`, the F0 tracker, WORLD's analysis and the pitch and formant changes
are in `analysis`, and the Klatt synthesizer and WORLD's synthesis are in
`stimuli`. A reader looking for "voice" has to know the layer rule first.

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
`views` (D12), and branches by topic that build on the trunk and never on
each other.

```
            voice    spatial    stimuli    texture       (branches: do not import each other)
               \        |          |          /
                ----------- views ------------           (one-way analyses)
                            |
                          frames                          (invertible analyses)
                            |
                         signals
                            |
                          core
```

The measured imports allow it. Resolved name by name to where each name
would go, every import between subpackages points down the trunk or from a
branch to the trunk (views to frames 10, signals 1, core 10; frames to core
9; voice to signals 11, core 8; spatial to views 1, frames 2, signals 4,
core 7; stimuli to views 1, frames 2, signals 2, core 5; texture to views 2,
frames 2, core 5). No branch imports another. Two point up, both inside a
method so that calls chain in a notebook: `Sound.envelope()`, which the test
already allows by name, and `Subbands.envelopes()`, which would be added
beside it (frames to views). Not counted, because it is inside today's
`analysis/vocoder.py`: WORLD's `SpectralEnvelope` and `Aperiodicity`
subclass the helper `_FrequencyView`, which D3 moves to `views`, so voice
would also import from views.

## Proposed layout

| Subpackage | Modules | What it is | Imports from |
|---|---|---|---|
| `core` | `sound`, `units`, `utils`, `fft` | unchanged | nothing in sonore |
| `signals` | `generators`, `processing` | making and editing sounds | core |
| `frames` | `frame`, `filterbank`, `gabor`, `mask` | invertible analyses, their coefficients, and changes to coefficients with exact least-squares resynthesis | core, signals |
| `views` | `spectrum`, `reassigned`, `envelopes`, `modulation`, `modspectrogram`, `cepstrum`, `mfcc`, `spectral_envelope` | one-way analyses: each says what it drops | core, signals, frames |
| `voice` | `glottal`, `klatt`, `f0`, `world`, `aperiodicity`, `change` | voices: made from parameters, measured, rebuilt and changed | the trunk |
| `spatial` | `binaural`, `spatialization`, `hrir_data`, `reverb` | two ears, heads and rooms | the trunk |
| `stimuli` | `ripples`, `channel_vocoder`, `phasevocoder` | sounds made by shaping or changing other sounds | the trunk |
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
| `analysis/mfcc.py` | `views/mfcc.py` | 10 |
| `analysis/envelopes.py` | `views/envelopes.py` | 11 |
| `analysis/modulation.py` | `views/modulation.py` | 11 |
| `analysis/modspectrogram.py` | `views/modspectrogram.py` | 7 |
| `analysis/f0.py` | `voice/f0.py` | 13 |
| `analysis/vocoder.py` | `voice/world.py`, `voice/aperiodicity.py`; helpers to `views/spectral_envelope.py` | 14 |
| `stimuli/vocoder.py` | `voice/world.py` | 12 |
| `analysis/voice.py` | `views/spectral_envelope.py` (`GridEnvelope`), `voice/change.py` | 8 |
| `signals/glottal.py` | `voice/glottal.py` | 7 |
| `stimuli/klatt.py` | `voice/klatt.py` | 6 |
| `stimuli/binaural.py` | `spatial/binaural.py` | 8 |
| `stimuli/spatialization.py` | `spatial/spatialization.py` | 8 |
| `stimuli/hrir_data.py` | `spatial/hrir_data.py` | 6 |
| `stimuli/reverb.py` | `spatial/reverb.py` | 5 |

78 distinct files are touched in all (moved modules included), 41 without
the frames/views split. By kind they are the moved modules' importers in
`src`, the API reference pages (`docs/api/source/*.rst`, one `automodule`
line per module), the README module table and References tags, gallery
scripts that import a module by its path, the gallery HTML pages whose
References tags link to a source file, design docs, tools and tests. The
tests mirror `src/`, so all 10 files in `tests/analysis` move, and 6 more
(glottal, klatt and the four spatial ones).

Estimates from today's line ranges (labelled, not measured as built):
`frames/gabor.py` about 570 lines (`GaborFrame` and `TVGaborFrame`, 405,
plus `STFT` and `TVSTFT`, 167), `frames/filterbank.py` about 640
(`Filterbank`, 164, plus the banks and `Subbands`, 473), `frames/frame.py`
about 110.

What does not change: every `so.name`. The top-level `__init__` re-exports
the public names from every subpackage, and it would go on doing so, so code
written as `so.world_synthesize(...)` or `so.GaborFrame(...)` runs
unchanged. Only deep imports such as `from sonore.analysis.frames import
GaborFrame` change.

## Decisions

- **D1. Layers or topics.** (a) A trunk (core, signals, frames, views) and
  sibling branches by topic (voice, spatial, stimuli, texture) that import
  only from the trunk, as above (recommended: the subpackage names then say
  what a module is about, the gallery groups can match them (D8), and
  `tests/test_layers.py` still fails on any import that points the wrong
  way; the rule changes from "below me in one list" to "below me in the
  trunk, or inside my own subpackage"). (b) The same branches, but `analysis`
  kept whole (D12 declined): 41 files instead of 78. (c) Keep the five layers
  and fix only the vocoder names (D2), moving 2 modules and touching 18
  files: the smallest change, but the voice code stays spread over three
  layers. (d) Go back to a flat package: rejected, since Cho split it in
  PR #12 to see the dependencies.
- **D2. The four vocoders.** (a) Name each by its method: WORLD's analysis
  and synthesis together in `voice/world.py`, the phase vocoder keeps
  `phasevocoder.py`, and the channel vocoder gets `stimuli/channel_vocoder.py`
  (recommended: no file is called plain `vocoder.py` any more, and each
  name says which vocoder it is). `world.py` would hold WORLD's ports only;
  `harmonic_aperiodicity`, sonore's own measure (world.md says anything
  different from WORLD gets its own name), goes to `voice/aperiodicity.py`.
  Estimate: `world.py` would be about 840 lines (705 + 236 less the 102 of
  the harmonic residual), longer than any module today (`frames.py` is 677).
  (b) Keep WORLD as two modules, `voice/world_analysis.py` and
  `voice/world_synthesis.py`: shorter files, but the pair reads as one
  thing split for no reason, which is the complaint. (c) Leave
  `noise_vocode` in `filterbank.py`: it is 37 lines on top of
  `Subbands`, but then the channel vocoder is the one stimulus among the
  frames.
- **D3. Split `analysis/voice.py`.** It holds two things. `GridEnvelope`,
  an envelope on a time-frequency grid, is what `Cepstrum.envelope_view` and
  `MFCC.envelope_view` return; with the interpolation helpers it shares with
  WORLD's `SpectralEnvelope` (`_FrequencyView`, `_positions`, `_pointwise`,
  now in `analysis/vocoder.py`) it moves to `views/spectral_envelope.py`.
  `scale_f0` and `warp_frequency` need `F0Track` and go to `voice/change.py`.
  (Recommended, and needed for D1(a): otherwise `cepstrum` and `mfcc` in the
  trunk would import from `voice`.) The alternative is to move `cepstrum` and
  `mfcc` into `voice` as well, see D5.
- **D4. The phase vocoder stays in `stimuli`.** It is neither a frame nor a
  view: its analysis is an STFT with instantaneous frequencies, but what it
  is for is changing a sound (duration, pitch, partials), and its
  resynthesis after a change is not exact. (a) Leave it in
  `stimuli/phasevocoder.py`, beside the channel vocoder and ripples, sounds
  made by changing or shaping other sounds (recommended, and it moves
  nothing). (b) Rebuild `PVAnalysis` on `GaborFrame` (today it uses SciPy's
  `ShortTimeFFT` directly), after which its analysis would be a frame's
  coefficients: a change to working code with no new behaviour, so not now.
  An earlier draft moved it to `analysis`; the exhaustive split leaves no
  place for it there.
- **D5. Where the cepstrum and MFCCs live.** (a) In `views` (recommended:
  both are views of any sound built on the STFT, and the MFCC is used well
  beyond voices). (b) In `voice`, since their gallery page is in the Voices
  group (D8). Package and gallery need not mirror each other exactly: the
  gallery groups by what one listens to, the package by what depends on
  what.
- **D6. Old import paths.** (a) No compatibility modules; the release
  notes of the next version (0.4.0) list every moved path (recommended, and
  what PR #12 did). Of the moved modules, 10 shipped in 0.3.0
  (`analysis/cepstrum`, `envelopes`, `filterbank`, `frames`, `modulation`,
  `representations`, and `stimuli/binaural`, `hrir_data`, `reverb`,
  `spatialization`), so only those deep paths can be in anyone's code; the
  other 8 were added after the release. (b) Thin modules at the old paths
  that re-export and warn, removed one version later.
- **D7. Signals keeps the LF pulse?** (a) No: `glottal.py` moves to `voice`
  (recommended: Rd, open quotient and the LF shape are voice parameters, and
  its only user in sonore is the Klatt synthesizer). (b) Yes, as a generator beside
  `harmonic_complex`. `resonator` and `antiresonator` stay in
  `signals/processing.py` either way: they are general filters that Klatt
  happens to use.
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
  and the design docs cepstrum.md, mfcc.md, glottal-source.md, klatt.md,
  voice-change.md and world.md).
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
  | | `mfcc` | `MFCC`, `freq_to_mel`, `mel_to_freq`, `mel_filterbank`, `symmetric_hamming`, `delta_features` |
  | | `spectral_envelope` | `GridEnvelope` |

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
    are views too, but live in the `voice` branch with the rest of WORLD and
    the tracker (D1); the frames/views division is exhaustive for the trunk.
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
  | `GridEnvelope` | the harmonics and everything finer than the envelope's smoothing |

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
marked "proposed" are D13's and do not exist yet.

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
   `representations.py`, `voice.py` and `vocoder.py` (D12, D3, D2), update imports in `src`, tests, tools and gallery scripts, the
   API reference pages, layout.md and its diagram, the layer test, the
   README module table and References tags
   (`tools/update_readme_source_links.py` rewrites line anchors but not file
   paths, so the paths are edited first). D13's `View` base class and error
   come in a separate PR after this one, since they add behaviour and the
   move should add none. Checks: the full test suite, the
   texture bit-for-bit hash against the previous commit, every gallery
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
