# Sound first: folders by meaning, and the way back to sound

A proposal, following the reorganization (`docs/design/reorganization.md`,
merged as PRs #82–#84) and the audit fixes (#86–#90). On 2026-10-03, before
releasing 0.4.0, Cho asked to rethink the split between `signals`,
`stimuli` and `generators`: "i don't like generators vs signals vs stimuli
having slightly confusing and arbitrary definitions. so Sound is at the
bedrock."

Status: accepted. On 2026-10-03 Cho accepted D1, D2, D3, D5, D6 and D7 as
recommended (option a) and decided D4 (one `sources/waveforms.py`). Cho deferred 0.4.0 until
this is resolved, and asked for `philosophy.md` to say what was decided in
discussion (its "Sound first" section and the `to_sound` principle). The table of every
module in three hierarchies (folder, import rank, meaning) is in the
project notes (`notes/layout/module-hierarchies.md`). Counts below come
from `grep` on main `67b3ba5`; they are labelled where they are estimates.

## What Cho has decided in discussion

1. **Meaning leads.** Folders follow what modules mean, and user intuition
   follows from that. Import order is still enforced, but between modules
   rather than between folders.
2. **Sound first, so analysis first.** `Sound` is the bedrock. The objects
   a user holds are sounds and analyses of them. Synthesis that goes back
   from an analysis belongs on that analysis.
3. **Synthesis stays first-class.** Making a sound from a description, for
   example from an arbitrary modulation spectrum, is a capability worth
   having, even when it is underdetermined.
4. **`synthesize` means the exact inverse, and `to_sound` means a route
   back.** Only frames have a working `synthesize`. A view has `to_sound`
   when, and only when, a canonical route exists, and the route's arguments
   are what the view discarded. Cho rejected `match` as a name: it is a
   soft keyword (Python 3.10 pattern matching).
5. **No `to_sound` without a canonical method.** For a modulation spectrum
   there is no standard way back. McDermott & Simoncelli's synthesis is one
   method among many, and the space of carriers is wider than for any other
   view. Such views refuse, with the reason, until a canonical route is
   found. Experiments come first.

## Why the current split reads as arbitrary

`signals` holds what builds a Sound from `core` alone. `stimuli` holds what
needs a frame or a view to build one. The split follows import rank, not
meaning:
- `pure_tone` (signals) and `ripple_sound` (stimuli) are both stimuli.
- `pad` (signals) and `noise_vocode` (stimuli) both take a sound and return
  one.
- "Generators" is only a file name.
- Every module handles signals, so `signals` names nothing in particular.

## The rule for the way back

Every analysis drops something, or keeps everything:

| Kind | Example | Goes back with | Meaning |
|---|---|---|---|
| Frame | `GaborFrame` | `synthesize(coefs)` | exact (least squares for changed coefficients) |
| View with a canonical route | `Spectrum` | `to_sound(..., carrier=...)` | a sound with this view, made by supplying what the view dropped |
| View without one | `ModulationSpectrum` | none; `to_sound` raises `NotInvertibleError` | the message says what was dropped and that there is no canonical route yet |

`to_sound`'s arguments are what the view discarded. So the `discards`
sentence each view already carries predicts its signature:

| View | Discards | `to_sound` takes | Today |
|---|---|---|---|
| `Spectrum` | the phase, and with it all timing | `duration`, `fs`, `carrier="noise"` (random phase), a Sound (its phase, with this magnitude), or `"minimum"` (an impulse response) | `to_noise(duration, fs)` |
| `Cepstrum` | the phase | `phase="original" \| "minimum"` | `to_sound(phase=...)`, already this rule |
| `Envelopes` | the fine structure | `carrier=` noise, tones or a Sound | `envelopes * subbands`; `noise_vocode` wraps it |
| `MFCC` | the phase, the detail in each mel band, and the source | an F0 track and an aperiodicity | `world_synthesize(f0, mfcc.envelope_view(), aperiodicity)` |
| `SpectralEnvelope`, `GridEnvelope` | the harmonics and the phase | an F0 track and an aperiodicity | `world_synthesize` |
| `Aperiodicity` | everything but the noise share | an F0 track and an envelope | `world_synthesize` |
| `F0Track` | everything but pitch and voicing | an envelope and an aperiodicity | `world_synthesize` |
| `PVAnalysis` | the phase between bins | `time_scale`, `freq_map` | `resynthesize(time_scale, freq_map)` |
| `TextureStats` | everything but the statistics | `duration`, `rng`, iteration settings | `texture.synth.synthesize(stats, ...)` (see D5) |
| `ModulationSpectrum`, `ModulationSpectrogram`, `ReassignedSpectrogram`, `TFPower`, `InterauralCues` | (as their `discards` say) | refuses: no canonical route | refuses through `synthesize` |

The three WORLD views go back only together. The natural form is one route
that takes all three, which is what `world_synthesize` is (D3).

Parameter descriptions that are not analyses of any sound, such as a
`Ripple` pattern, follow the same rule: `Ripple(...).to_sound(duration,
fs, carrier=...)` takes what the pattern does not specify, which is the
carrier. `ripple_sound(pattern, duration, fs, carrier=...)` already has
exactly these arguments.

Synthesis from parameters alone (tones, noises, chirps, harmonic complexes,
the glottal source, Klatt) has no object to attach to. It stays as plain
constructors, in one folder (D1).

## Proposed layout

| Folder | Holds | Modules |
|---|---|---|
| `core` | `Sound`, its operations, units, scales, numeric helpers | `sound`, `units`, `utils`, `fft`, plus today's `signals/processing.py` (D2) |
| `sources` | sounds from parameters | `waveforms`, `klatt`, `ripples` (D1, D4) |
| `frames` | Sound ⇄ coefficients | unchanged |
| `views` | Sound → a one-way summary, with its route back where one exists | unchanged, plus `world` (D3) and `phasevocoder` (D6) |
| `spatial` | topic: two ears, heads, rooms | unchanged |
| `texture` | topic: texture statistics and synthesis | unchanged |
| `plotting` | display | unchanged |

`signals` and `stimuli` disappear. `noise_vocode` goes with `Envelopes`
(D6). Every `so.name` stays the same, except the renames in D5.

**Import order.** The layer test changes from a rank per folder to two
rules: no cycles between modules, and `core` imports nothing above it at
module level (inside-function imports stay allowed by name, as now).
`layout.svg` draws modules, colored by folder. With WORLD in `views`, views
no longer reach into a synthesis module for WORLD's internals. `sources/ripples`
imports `frames` and `views`, `sources/klatt` imports `core`'s resonators.
Both directions are fine, because each module still sits above everything
it imports.

## Decisions

- **D1. The folder of sounds from parameters.** (a) `sources` (recommended;
  Cho chose `sources/waveforms` in D4):
  each module is a kind of sound source. The cost is that the word also
  means a source position in `spatial` (`move_sound`'s "source"). (b)
  `synthesis`: clear, but synthesis also lives on frames and views, so the
  folder would not hold all of it. (c) `generators`: today's file name,
  which Cho found vague.
- **D2. Where processing goes.** Today's `signals/processing.py` holds
  multi-sound operations (`mix`, `pad`, `concat`, `normalize`, `match_fs`,
  `match_channels`, `relative_db`) and filters (`butter_filter`,
  `bandpass`, `resonator`, `antiresonator`, `amplitude_modulate`). They
  operate on sounds and belong to none. (a) Move the file to `core`
  unchanged, as `core/processing.py` (recommended): `Sound` and its
  operations sit together, and `Sound` already has methods for one-sound
  edits (`pad`, `ramp`, `gain_db`, `resample`, `convolve`). (b) Split it
  into `core/editing.py` and `core/filters.py`: tidier names, one more
  file. (c) Keep a top-level `processing` folder: more folders for one
  module.
- **D3. WORLD.** `signals/world.py` holds WORLD's synthesis and its shared
  internals, which `views/spectral_envelope` and `views/aperiodicity`
  import (10 names, 9 of them private; counted with `ast` on main). (a) Move it to `views/world.py`
  (recommended): WORLD's analysis, synthesis and internals then sit
  together, as one vocoder, and `world_synthesize` is the route back from
  its three views, as the rule asks. (b) A bundle class (for example
  `WorldAnalysis(f0, envelope, aperiodicity).to_sound()`) with
  `world_synthesize` kept as the function form: more uniform, but a new
  type for something a function already does. It can be added later
  without moving anything.
- **D4. The file of waveforms.** Decided (Cho, 2026-10-03): one file,
  `sources/waveforms.py`, holding everything in today's `generators.py`
  (silence, tones, harmonic and Schroeder complexes, square, sawtooth,
  pulse trains, chirps, the noises, and the LF glottal source with
  `lf_harmonics` and `lf_pulse`). "Generators" said nothing inside a
  folder called `sources`. A split by kind was considered and dropped:
  `glottal_source` is built on `harmonic_complex` (it calls it and shares
  its harmonic-count helper), so it would need a private import between
  files, and the reorganization's D7 merged it for that reason.
- **D5. Renames that follow the rule.** No compatibility names, as for
  the rest of 0.4.0. (a) Rename now (recommended):
  `Spectrum.to_noise(duration, fs)` becomes `Spectrum.to_sound(duration,
  fs, carrier="noise")`, with `carrier` also accepting a Sound and
  `"minimum"`. `PVAnalysis.resynthesize` becomes `PVAnalysis.to_sound`.
  `View.to_sound` raises `NotInvertibleError` by default, with the
  `discards` sentence and "no canonical route yet". (b) Rename now, and
  also add `Envelopes.to_sound(carrier=...)`, `Ripple.to_sound(...)` and
  `TextureStats.to_sound(...)`. These are new methods over existing code:
  more to review in one release. (c) Only the folder move now; renames
  later, in a minor release of their own. About 20 lines outside the
  changelog name `to_noise` or `resynthesize(` (estimate from `grep`,
  including gallery scripts).
- **D6. Modules that go back from a view.** `stimuli/channel_vocoder.py`
  (`noise_vocode`) is envelopes imposed on a carrier, and
  `stimuli/phasevocoder.py` is `PVAnalysis` with its routes back
  (`time_stretch`, `pitch_shift`). (a) Move both to `views` (recommended):
  `views/envelopes.py` gains `noise_vocode` (its one function), and
  `views/phasevocoder.py` keeps its name. (b) Keep `channel_vocoder.py` as
  its own file in `views`.
- **D7. Ripples.** A ripple pattern describes a sound; it analyses none.
  (a) `sources/ripples.py` (recommended), with `ripple_sound` unchanged
  and, if D5 (b), `Ripple.to_sound`. (b) `views`, beside `Envelopes`,
  since `render` returns one: but a pattern is not a view of any sound.

## What this changes for 0.4.0

0.4.0 already moves deep import paths. Done before it ships, deep paths
break once.

| Old path | New path |
|---|---|
| `sonore.signals.generators` | `sonore.sources.waveforms` (D4) |
| `sonore.signals.klatt` | `sonore.sources.klatt` |
| `sonore.signals.processing` | `sonore.core.processing` (D2) |
| `sonore.signals.world` | `sonore.views.world` (D3) |
| `sonore.stimuli.ripples` | `sonore.sources.ripples` (D7) |
| `sonore.stimuli.channel_vocoder` | `sonore.views.envelopes` (D6) |
| `sonore.stimuli.phasevocoder` | `sonore.views.phasevocoder` (D6) |

Files naming each old path (from `grep`, including design history, which
keeps old paths on purpose): generators 21, world 16, processing 8,
ripples 8, channel vocoder 8, klatt 6, phase vocoder 6.

Checks, as for the source move: the full test suite, the texture and WORLD
bit-for-bit hashes against main, every gallery script to its last cell,
Sphinx warnings unchanged, and a grep for each old path returning nothing
outside the changelog and design history. The gallery groups do not
change: "Stimuli" there names a use, and stays.

## Order

1. Cho decides D1–D7. Done, 2026-10-03.
2. One PR: the move, the layer test, the diagram, the docs.
3. If D5 (a) or (b): the renames, in their own PR.
4. PR #85's release notes gain the moved paths; then 0.4.0.

## Out of scope

- Gradient search as a route back (for the modulation spectrum and other
  views without a canonical route). Cho: an open question for experiments
  first.
- `Sound.tone(...)`-style constructors on the class. Discussed; not
  proposed now.
