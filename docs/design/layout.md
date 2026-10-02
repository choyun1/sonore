# Package layout

The source is split into layers, one subpackage each, so the direction of
dependencies shows in the file tree. Bottom to top:

| Subpackage | Modules | What it is | Imports from |
|---|---|---|---|
| `core` | `sound`, `units`, `utils`, `fft` | `Sound`, decibels, numeric helpers, FFT sizes and threads | nothing in sonore |
| `signals` | `generators`, `glottal`, `processing` | making and editing sounds | core |
| `analysis` | `frames`, `filterbank`, `representations`, `cepstrum`, `f0`, `vocoder`, `envelopes`, `modulation`, `modspectrogram` | taking sounds apart | core, signals |
| `stimuli` | `ripples`, `binaural`, `spatialization`, `hrir_data`, `reverb`, `phasevocoder`, `klatt`, `vocoder` | stimuli built from the analysis tools | core, signals, analysis |
| `texture` | `stats`, `grad`, `synth` | sound texture statistics and synthesis | everything below |
| `plotting.py` | | the `plot_*` functions and `overview` | imported only inside `.plot()` methods |

Inside a layer any import is fine (`filterbank` and `representations` build on
`frames`). `tests/test_layers.py` fails when a module imports from a layer above
its own.

![Import graph](layout.svg)

The diagram is drawn from the source by `python tools/draw_layout.py`, and
`tests/test_layers.py` fails when it is out of date. Inside a band, a module
sits one row above the modules of its own layer that it imports.

Grey arrows are module-level imports. The red dashed ones are imports inside a
function or a `TYPE_CHECKING` block that point up or sideways. They exist so
that calls chain in a notebook: `Frame.analyze()` returns `Subbands` or an
`STFT`, `Envelopes.modulation_spectrum()` returns a `ModulationSpectrum`, and
every `.plot()` reaches into `plotting`. Inside a layer they can close a
cycle (`frames` and `filterbank`, `sound` and `units`), which is harmless
because none of them runs at import.
The one that crosses layers, `Sound.envelope()` returning an `Envelope`, is
listed by name in the test, so a new upward import has to be added there on
purpose.

`import sonore as so` re-exports the public names from every layer, so code
written against `so.` doesn't need to know where anything lives. The texture
names stay under `so.texture`; synthesis is `sonore.texture.synth`.
