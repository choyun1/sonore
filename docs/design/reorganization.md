# Reorganizing the package and the gallery

A proposal for how `src/sonore` and the listening gallery should be grouped
now that sonore has grown from five texture-era layers to 30 modules and 16
gallery pages. Cho asked for it on 2026-10-02: "i don't like that there's
multiple vocoder.py - under stimuli and analysis. we should also rationalize
the gallery- right now a ton falls under seeing sound category, but maybe
most of the voice stuff should be in one category, another category for more
premitive stimuli, one for spatial hearing, etc."

Status: proposed. Nothing has been moved. Every count below is printed by
`python tools/count_reorganization_references.py`, run on 2026-10-02 at
main `b3f6a10`; numbers that are not are labelled estimates.

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

**Some modules sit in a layer they don't need.** The phase vocoder is in
`stimuli` but imports only `core.sound`; it is an analysis and resynthesis,
like the STFT it is built on. The channel vocoder is a stimulus but lives in
`analysis/filterbank.py`.

**The gallery's middle group is half of it.** Of 16 pages, 8 are under
"Seeing and changing sounds", 5 under "Stimuli" and 3 under "Listeners in
the world". The middle group holds both the time-frequency course and four
voice pages, and the binaural page sits under "Stimuli" while the other
spatial pages sit under "Listeners".

## What the layer rule does, and what it should keep doing

`tests/test_layers.py` holds a strict order, core < signals < analysis <
stimuli < texture, and fails when a module imports from a layer above its
own. Cho chose that in PR #12 so the direction of dependencies shows in the
file tree. The proposal keeps that guarantee and changes only its shape:
a trunk of three layers that everything builds on, and branches by topic that
build on the trunk and never on each other.

```
                 voice   spatial   stimuli   texture      (branches: do not import each other)
                    \       |         |        /
                     -------- analysis --------
                                 |
                              signals
                                 |
                               core
```

The measured imports allow it: under the proposed layout, every import
between subpackages points from a branch to the trunk or down the trunk
(voice to analysis 1, signals 3, core 6; spatial to analysis 3, signals 4,
core 6; stimuli to analysis 2, signals 1, core 2; texture to analysis 3,
core 5). No branch imports another. The one import that points up is the one
the test already allows by name, `Sound.envelope()`. The tool also reports
two imports from `analysis` to `voice`; both come from `analysis/voice.py`,
which D3 splits so that the part left in `analysis` imports nothing from
`voice` (see D3).

## Proposed layout

| Subpackage | Modules | What it is | Imports from |
|---|---|---|---|
| `core` | `sound`, `units`, `utils`, `fft` | unchanged | nothing in sonore |
| `signals` | `generators`, `processing` | making and editing sounds | core |
| `analysis` | `frames`, `filterbank`, `representations`, `spectral_envelope`, `cepstrum`, `mfcc`, `envelopes`, `modulation`, `modspectrogram`, `phasevocoder` | taking sounds apart and putting them back | core, signals |
| `voice` | `glottal`, `klatt`, `f0`, `world`, `aperiodicity`, `change` | voices: made from parameters, measured, rebuilt and changed | core, signals, analysis |
| `spatial` | `binaural`, `spatialization`, `hrir_data`, `reverb` | two ears, heads and rooms | core, signals, analysis |
| `stimuli` | `ripples`, `channel_vocoder` | stimuli built from the analysis tools | core, signals, analysis |
| `texture` | `stats`, `grad`, `synth` | unchanged | core, analysis |
| `plotting.py` | | unchanged | imported only inside `.plot()` |

Every move, with the number of files that name the module's current path
outside the module itself (each would need an edit):

| Now | Proposed | Files naming it |
|---|---|---|
| `analysis/f0.py` | `voice/f0.py` | 13 |
| `analysis/vocoder.py` | `voice/world.py` and `voice/aperiodicity.py` | 12 |
| `stimuli/vocoder.py` | `voice/world.py` | 10 |
| `analysis/voice.py` | `analysis/spectral_envelope.py` and `voice/change.py` | 8 |
| `signals/glottal.py` | `voice/glottal.py` | 7 |
| `stimuli/klatt.py` | `voice/klatt.py` | 6 |
| `stimuli/binaural.py` | `spatial/binaural.py` | 8 |
| `stimuli/spatialization.py` | `spatial/spatialization.py` | 8 |
| `stimuli/hrir_data.py` | `spatial/hrir_data.py` | 6 |
| `stimuli/reverb.py` | `spatial/reverb.py` | 5 |
| `stimuli/phasevocoder.py` | `analysis/phasevocoder.py` | 6 |
| `noise_vocode` in `analysis/filterbank.py` | `stimuli/channel_vocoder.py` | 6 |

41 distinct files are touched in all. By kind they are the moved modules'
importers in `src`, the API reference pages (`docs/api/source/*.rst`, one
`automodule` line per module), the README module table and References tags,
gallery scripts that import a module by its path, the gallery HTML pages
whose References tags link to a source file, design docs, tools and tests.
Ten test files move with their modules, since `tests/` mirrors `src/`.

What does not change: every `so.name`. The top-level `__init__` re-exports
the public names from every subpackage, and it would go on doing so, so code
written as `so.world_synthesize(...)` or `so.klatt_synthesize(...)` runs
unchanged. Only deep imports such as `from sonore.stimuli.reverb import
synth_ir` change.

## Decisions

- **D1. Layers or topics.** (a) A trunk (core, signals, analysis) and
  sibling branches by topic (voice, spatial, stimuli, texture) that import
  only from the trunk, as above (recommended: the subpackage names then say
  what a module is about, the gallery groups can match them (D8), and
  `tests/test_layers.py` still fails on any import that points the wrong
  way; the rule changes from "below me in one list" to "below me in the
  trunk, or inside my own subpackage"). (b) Keep the five layers and fix only
  the vocoder names (D2), moving 2 modules and touching 16 files instead of
  41: the smallest change, but the voice code stays spread over three
  layers. (c) Go back to a flat package: rejected, since Cho split it in
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
  `Subbands`, but then the channel vocoder is the one stimulus in
  `analysis`.
- **D3. Split `analysis/voice.py`.** It holds two things. `GridEnvelope`,
  an envelope on a time-frequency grid, is what `Cepstrum.envelope_view` and
  `MFCC.envelope_view` return; with the interpolation helpers it shares with
  WORLD's `SpectralEnvelope` (`_FrequencyView`, `_positions`, `_pointwise`,
  now in `analysis/vocoder.py`) it moves down to
  `analysis/spectral_envelope.py`. `scale_f0` and `warp_frequency` need
  `F0Track` and go to `voice/change.py`. (Recommended, and needed for D1(a):
  otherwise `cepstrum` and `mfcc` in the trunk would import from `voice`.)
  The alternative is to move `cepstrum` and `mfcc` into `voice` as well, see
  D5.
- **D4. The phase vocoder moves to `analysis`.** It imports only
  `core.sound`, and on the gallery it is a way of changing a sound through
  its STFT, beside analysis and resynthesis. (a) `analysis/phasevocoder.py`
  (recommended). (b) Leave it in `stimuli`, which then holds ripples, the
  channel vocoder and the phase vocoder.
- **D5. Where the cepstrum and MFCCs live.** (a) In `analysis` (recommended:
  both are views of any sound built on the STFT, and the MFCC is used well
  beyond voices). (b) In `voice`, since their gallery page is in the Voices
  group (D8). Package and gallery need not mirror each other exactly: the
  gallery groups by what one listens to, the package by what depends on
  what.
- **D6. Old import paths.** (a) No compatibility modules; the release
  notes of the next version (0.4.0) list every moved path (recommended, and
  what PR #12 did). Of the moved modules, 5 shipped in 0.3.0
  (`stimuli/binaural`, `hrir_data`, `phasevocoder`, `reverb`,
  `spatialization`), so only those deep paths can be in anyone's code; the
  other 6 were added after the release. (b) Thin modules at the old paths
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
  | Voices (`voice`) | Formant synthesis, Cepstral analysis, Voices from harmonics, Source, filter and aperiodicity, and "Changing a voice" when it is written (voice-change.md D11) |
  | Spatial hearing (`spatial`) | Binaural cues, Synthetic reverberation, Moving talkers |

  That is 4, 5, 4 and 3 pages instead of 5, 8 and 3. The voice pages run
  from a vowel written as numbers (Klatt), to a recorded voice taken apart
  (cepstrum), rebuilt from pitch and envelope (harmonics), and with its
  noise measured and resynthesized (WORLD); the formant page comes first
  because it introduces source and filter, which the other three assume.
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
  `listeners` goes. Eight scripts change folder. The HTML pages stay flat
  at `docs/gallery/<name>.html` (all 17, the index included), so no page URL
  changes and no link in the README, the design docs or elsewhere breaks.
  Apart from each page naming its own script, only 6 other files name a
  moved script's path (`tools/check_moving_trajectories.py` and the design
  docs cepstrum.md, mfcc.md, glottal-source.md, klatt.md and world.md).
- **D10. README.** The "More in the gallery" list (16 entries today, in no
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

## Order

Other threads are changing gallery pages now (speech examples on the reverb
page, sections on the cepstrum and voice-change pages). A folder move and an
edit to the same script in two open PRs conflict, so:

1. Cho decides D1 to D10.
2. The open gallery PRs merge first.
3. **Gallery PR** (small, what readers see): `git mv` the eight scripts,
   rewrite TOPICS and the comment above it, patch the menus and sidebar of
   all 17 HTML pages in place (a full rebuild does not run in the cloud
   container), regroup the README gallery list (D10). Checks: the test
   suite, and every page's menu showing the four groups.
4. **Source PR** (mechanical, one PR so the tree is never half-moved):
   `git mv` each module, split `analysis/voice.py` and `analysis/vocoder.py`
   (D3, D2), update imports in `src`, tests, tools and gallery scripts, the
   API reference pages, layout.md and its diagram, the layer test, the
   README module table and References tags
   (`tools/update_readme_source_links.py` rewrites line anchors but not file
   paths, so the paths are edited first). Checks: the full test suite, the
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

- Splitting large mixed modules: `representations.py` (550 lines: spectra,
  STFTs, masks, reassignment, modulation spectra) and `plotting.py` (609
  lines). Possible later, independent of this.
- Renaming public functions or classes. Everything keeps its `so.` name.
- Gallery page titles. "Seeing speech" keeps its title in the Seeing group:
  it is a course in time-frequency analysis that happens to use a sentence.
