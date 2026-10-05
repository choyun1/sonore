# Plotting: where the picture of an object comes from

Status: proposal, 2026-10-05 (audit sitting 9). Nothing here is decided yet.

Cho asked, while reading `Spectrum.plot` in the views sitting, why a class
carries a plotting method when `plotting.py` exists, and whether every class
should have one. This doc sets out what sonore does now, what a user sees
under each choice, and what each choice means for classes added later.

## What there is now

Counted by `python tools/count_plot_methods.py` on this branch:

- **Views:** 15 of 18 public views have `.plot`. The three without are
  `ReassignedSpectrogram`, `PVAnalysis` and `TextureStats`.
- **Frames** (the operators `GaborFrame`, `TVGaborFrame`, `Filterbank`):
  none of the 3 has `.plot`. Their coefficients do: `STFT`, `TVSTFT` and
  `Subbands`.
- **Other public classes:** 7 of 17 have `.plot` (`Sound`, `STFT`,
  `TVSTFT`, `Subbands` and the three ripples). The ones without are tools
  or data rather than analyses of a sound: `FrequencyScale`, `HRIRSet`,
  `TextureModel`, the modulation filterbanks, `ModulationBlob`,
  `Decibels`, `set_fft_workers`.
- **Where the drawing is:** every `.plot` but one is a few lines that import
  a `plot_*` function from `plotting.py` and call it, as `layout.md` asks.
  The exception is `DescriptorTrack.plot` (from #130), which draws with
  matplotlib itself.
- **Calls:** about 102 calls of the form `x.plot(` on sonore objects (78 in
  the gallery and notebook, 16 in tests, 5 in `src`, 3 in `tools`; a text
  match, so an estimate) and 6 mentions in the README.

So the rule in practice is "a `.plot` wherever one picture is the obvious
one, and nothing elsewhere", but it is written down nowhere, and a missing
method gives an `AttributeError` that says nothing about why.

## What a user sees

| | (a) A method on every class | (b) One function, `so.plot(x)` |
|---|---|---|
| Call | `spectrum.plot()` | `so.plot(spectrum)` |
| No picture | `NotImplementedError` with one sentence: why, and what to plot instead | `TypeError` with the same sentence |
| Options | `help(track.plot)` and tab completion show that class's own options (`candidates=` on an `F0Track`) | every class's options behind `**kwargs`, documented in one long docstring |
| Overlays | `a.plot(ax=ax); b.plot(ax=ax)` | `so.plot(a, ax=ax); so.plot(b, ax=ax)` |
| Several objects in one call | no | could grow `so.plot(a, b)` |

Either way the drawing code stays in `plotting.py`. The difference for a
user is mostly where the name is found and how options are discovered.

## What it means for new classes

- **(a)** A new view inherits a `plot` from `View` that refuses, so the
  method is never missing. A test, like the one that makes every view say
  what it `discards`, makes each view either draw or give a sentence on why
  it cannot, so a refusal is chosen on purpose rather than forgotten. The
  drawing goes into `plotting.py` as a `plot_*` function and the method is
  a three-line call into it.
- **(b)** A new class has no picture until a case is added to the `match`
  in `plotting.py`, which grows with every class and imports every class
  (inside the function, so no import cycle). The same kind of test could
  enforce a case or a stated reason.

Precedents: pandas and xarray put `.plot()` on their data objects;
matplotlib and `librosa.display` are functions.

## Decisions

- **D1. Where the entry point is.** (a) A method on every class
  (recommended): it matches how `synthesize` and `to_sound` already refuse
  with a reason, a class's options show up where its users look, and no
  existing call changes. (b) `so.plot(x)` dispatching on type with `match`:
  one name to learn and room for multi-object plots, at the cost of
  rewriting about 102 calls and one long docstring. (c) Both, `so.plot(x)`
  calling `x.plot()`: two names for one thing, so not recommended.
- **D2. Which classes promise a `.plot`.** (a) Every view, plus every
  analysis result that is not a view: `Sound`, `STFT`, `TVSTFT`, `Subbands`
  (recommended). They already have one, so the promise is a test, not new
  code, and no new base class is needed. Tools and data (`FrequencyScale`,
  `HRIRSet`, the modulation filterbanks...) promise nothing. (b) Every
  public class: a refusing method on things nobody would think to plot.
- **D3. How a view with no picture refuses.** `View.plot` raises
  `NotImplementedError` with a class attribute `no_plot`, one sentence
  naming what to plot instead, as `discards` names what was dropped. The
  test fails for a view that neither overrides `plot` nor sets `no_plot`.
  The three today would read:
  - `ReassignedSpectrogram`: its points have no grid until you choose one;
    plot `.binned(t_edges, f_edges)`.
  - `PVAnalysis`: its magnitudes are an STFT's; plot
    `so.GaborFrame(...).analyze(sound)`.
  - `TextureStats`: many statistics of different kinds, with no single
    picture; plot the parts you need from its arrays.

  Alternative: give `PVAnalysis` a magnitude image. It would repeat
  `STFT.plot`, so not recommended.
- **D4. `DescriptorTrack`.** Its drawing moves into `plotting.py` as
  `plot_descriptor_track`, under any choice above.
- **D5. The `plot_*` functions.** They stay public: they draw things with no
  class (`overview`, `plot_lissajous`, `plot_tf_db` for any array), and
  they are what the methods call.

## Out of scope

Plots of the frames themselves (a `Filterbank`'s frequency responses, a
`GaborFrame`'s window). They would be useful, but they are new pictures, not
part of this rule; a later doc can add them.
