# Package layout

The source is a trunk and three branches, one subpackage each, so the
direction of dependencies shows in the file tree (docs/design/reorganization.md
has the reasons). The trunk, bottom to top:

| Subpackage | Modules | What it is | Imports from |
|---|---|---|---|
| `core` | `sound`, `units`, `utils`, `fft` | `Sound`, decibels, numeric helpers and frequency scales, FFT sizes and threads | nothing in sonore |
| `signals` | `generators`, `processing`, `klatt`, `world` | sounds from parameters: waveforms (the LF source among them), filters, and two synthesizers | core |
| `frames` | `frame`, `filterbank`, `gabor`, `mask` | invertible analyses, their coefficients, and changes to coefficients with exact least-squares resynthesis | core, signals |
| `views` | `spectrum`, `reassigned`, `envelopes`, `modulation`, `modspectrogram`, `cepstrum`, `mfcc`, `f0`, `spectral_envelope`, `aperiodicity` | one-way analyses, each saying what it drops | core, signals, frames |

The branches, which import from the trunk and never from each other:

| Subpackage | Modules | What it is |
|---|---|---|
| `spatial` | `binaural`, `spatialization`, `hrir_data`, `reverb` | two ears, heads and rooms |
| `stimuli` | `ripples`, `channel_vocoder`, `phasevocoder` | sounds made by shaping or changing other sounds through an analysis |
| `texture` | `stats`, `grad`, `synth` | sound texture statistics and synthesis |

`plotting.py` holds the `plot_*` functions and `overview`, and is imported only
inside `.plot()` methods.

Inside a subpackage any import is fine (`filterbank` and `gabor` build on
`frame`). `tests/test_layers.py` fails when a module imports from higher up
the trunk, or from another branch.

![Import graph](layout.svg)

The diagram is drawn from the source by `python tools/draw_layout.py`, and
`tests/test_layers.py` fails when it is out of date. Inside a band, a module
sits one row above the modules of its own subpackage that it imports.

Grey arrows are module-level imports. The red dashed ones are imports inside a
function or a `TYPE_CHECKING` block that point up or sideways. They exist so
that calls chain in a notebook: `Frame.analyze()` returns `Subbands` or an
`STFT`, `STFT * mask` takes a `Mask`, and every `.plot()` reaches into
`plotting`. Inside a subpackage they can close a cycle (`gabor` and `mask`,
`sound` and `units`), which is harmless because none of them runs at import.
Two cross from the trunk upward, `Sound.envelope()` returning an `Envelope`
and `Subbands.envelopes()` returning `Envelopes`; both are listed by name in
the test, so a new upward import has to be added there on purpose.

WORLD's synthesis sits in `signals` though its inputs come from views: it
reads an F0 track as `.t` and `.f0`, an envelope as `env(t, f)` and an
aperiodicity as its grid, rather than checking their types, as
`harmonic_complex` and `klatt_synthesize` already did.

`import sonore as so` re-exports the public names from every subpackage, so
code written against `so.` doesn't need to know where anything lives. The
texture names stay under `so.texture`; synthesis is `sonore.texture.synth`.
