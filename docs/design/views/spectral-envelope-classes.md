# One SpectralEnvelope class

Status: proposed, 2026-10-18. Cho asked what the relationship is between
`Envelope`, `Envelopes` and `SpectralEnvelope`, whether there should be a
`SpectralEnvelopes`, and whether an envelope isn't spectral anyway, and
suggested unifying the concepts. He accepted three points, written up here
before any code: one `SpectralEnvelope` class that absorbs `GridEnvelope`
and that every estimator returns; `Envelope` and `Envelopes` kept, with the
two meanings of "envelope" named in the docs; and `SpectralEnvelope.to_world`
dropped. He also asked whether all of these are just time-frequency
representations under one `TFRepresentation` base that `STFT` and the rest
inherit from. This note works out the details and lists what is left to
decide.

## Two meanings of "envelope"

Both kinds are smooth, non-negative magnitudes on a time by frequency grid.
A temporal envelope (`Envelope`, `Envelopes` in `views/envelopes.py`) is
smoothed along time within one band: the Hilbert envelope of a band drops
the fine structure, and `Envelopes` holds one per filterbank band, which is
a cochleagram. A spectral envelope (`views/spectral_envelope.py`) is smoothed
across frequency within one time window: it drops the harmonics, that is the
source. So a pitch-adaptive envelope is not a time-varying `Envelopes`: both
have time and frequency axes, but they smooth along different axes and keep
different things. No `SpectralEnvelopes` is needed, because one
`SpectralEnvelope` already holds every time window and channel; the
singular-plural pair `Envelope`/`Envelopes` is about one band against a bank
of bands, which has no counterpart on the spectral side.

The module docstrings of both files will say "temporal envelope" and
"spectral envelope" explicitly, and `docs/design/philosophy.md` gets one line
stating the two meanings as above.

## What there is today

Both spectral classes hold `data` of shape `(n_channels, len(f), len(t))`
as power, with `t` and `f` arrays, and both have `db`, `amplitude(t, f,
channel=)`, `plot(ax, channel=, db_range=, fmax=)`, `__call__(t, f)` (linear
in time and in log power over frequency, held beyond the grid), and the
`discards` and `back_to_sound` texts, which differ only in the class name.
The `plot` method is the same shared helper for both.

`SpectralEnvelope(data, t, fs, q1)` is returned only by
`pitch_adaptive_envelope` (CheapTrick). It derives from `_FrequencyView`,
which it shares with `Aperiodicity`. Its grid is the uniform FFT grid:
`n_fft = 2 (n_freqs - 1)` is stored and `f` is a property, `n_fft // 2 + 1`
bins from 0 to `fs / 2`. It has `fs`, `n_fft`, `q1`, and `to_world(channel)`
(the one channel transposed and C-contiguous, for `pyworld.synthesize`).
Its `db` takes the logarithm with no floor, and its constructor checks
nothing.

`GridEnvelope(data, t, f)` is returned by `Cepstrum.envelope_view()`,
`MFCC.envelope_view(f=None)` and `LPC.envelope_view(f=None)`, and is used
inside `source_filter_synthesize` for the harmonic and noise powers. Its `f`
is any array (MFCC and LPC take an arbitrary `f`; by default all three use
the FFT grid of their own `fs` and `n_fft`). It has no `fs`, `n_fft` or
method settings, so `source_filter_synthesize` asks for `fs` when given one
without an aperiodicity. It promotes 2-d data to one channel, checks the
shape and that the power is not negative, and floors the power at the
smallest positive float before a logarithm.

On the same CheapTrick data (16 kHz, 513 bins, 180 time windows), a
`GridEnvelope` built from `env.data, env.t, env.f` reads the same values as
the `SpectralEnvelope` itself, within 2.3e-14 dB for `__call__` (including
points beyond the grid) and exactly for `amplitude`. So `GridEnvelope`'s
reading code can serve both.

References, from `grep -rn "GridEnvelope\|SpectralEnvelope\|to_world"` over
`src`, `tests`, `tools` and `docs` (`.py`, `.md`, `.rst`; not `.html`):
97 lines. `GridEnvelope` appears 46 times in 16 files, `SpectralEnvelope`
50 times in 16 files, `to_world` 8 times in 5 files. The gallery `.py` and
`.md` files name neither class (they call `envelope_view` and
`pitch_adaptive_envelope`, 17 lines). Of the `to_world` hits, three are
calls: `tests/views/test_spectral_envelope.py` line 44, and
`tools/check_voice_recipe_claims.py` line 188, which calls it on both the
envelope and the aperiodicity. The others are the definition, its docstring,
a test name, one note in `world.md`, and `to_world_grid`, a different name
in `voice-change.md`.

## The unified class

`SpectralEnvelope(data, t, f, *, fs=None, method=None, settings=None)`:
power on any grid of times and frequencies, with `GridEnvelope`'s checks,
2-d promotion, log floor and reading code. `fs` is the sampling rate of the
analyzed sound, or None for an envelope made by hand. `method` names the
estimator ("CheapTrick", "cepstrum", "MFCC", "LPC"), as `Aperiodicity`
already does with its `method`. `settings` holds the estimator's parameters
(for CheapTrick `q1`, `f0_floor`, `unvoiced_f0`). `db`, `amplitude`,
`plot` and `__call__` all carry over unchanged in behavior. `__repr__` shows
the method and settings. Every estimator passes its own `fs`, so
`source_filter_synthesize` needs its `fs` argument only for a hand-made
envelope. `warp_frequency` copies the object, so a warped envelope keeps its
method and settings and only `data` changes, as now.

`GridEnvelope` is removed with no alias, for 0.8.0, and `to_world` goes.

## A TFRepresentation base?

A shared base class would cross the line sonore is built on. The `View`
docstring (`views/view.py`) and `docs/design/philosophy.md` split every
analysis into a frame's coefficients (`STFT`, `TVSTFT`, `Subbands`), which
`synthesize` exactly, and views, which keep only part of a sound and say in
`discards` what they dropped. A base above both would hold an STFT and a
spectral envelope as the same kind of thing, which they are not.

The objects also share less than their shape suggests. The second axis is
uniform FFT bins for an STFT, filterbank bands with center frequencies
(`cfs`) for `Subbands` and `Envelopes`, quefrency (`q`) for `Cepstrum`, and
coefficient index for `MFCC`. Some views have no time-frequency grid at
all: `Envelope` and `F0Track` are over time only, and `Spectrum` is over
frequency only. What is common is data on time by something, not behavior;
units, reading and plotting differ per class, and each class already has
its own `plot` into `sonore.plotting`.

Where a function does take "anything held on a time by frequency grid",
philosophy.md's rule is a public `typing.Protocol`, as with `F0Contour`,
`Distribution` and `EnvelopePattern`, so that a user's class works without
inheriting anything. Today such arguments are typed `Any` or by the bare
callable alias `EnvelopeFunction`, and read `.t`, `.f`, `.data` (and
`.amplitude`) by `hasattr` or `getattr`. Counted from the code, four public
callables, with five parameters, would take that Protocol:
`source_filter_synthesize` (`envelope` and `aperiodicity`, both `Any`),
`warp_frequency` (`view`), `harmonic_complex` (`amplitudes`, which looks for
an `amplitude` method), and `Aperiodicity.bands` (`envelope`, typed
`SpectralEnvelope`). `SpectralEnvelope` and `Aperiodicity` would both
satisfy it without a shared public base.

## Decisions

D1. Settings as a dict or as attributes. Either `env.settings["q1"]` with a
`settings` dict, or one optional attribute per known parameter (`env.q1`,
None for other methods). Recommend the dict, with `method` as its own
attribute: it covers every estimator without growing the class. Outside
the class, `.q1` is read in one place, `tools/check_voice_change_claims.py`
line 58, which builds a `SpectralEnvelope` by hand and would pass the
settings through instead.

D2. `n_fft` on an arbitrary grid. Recommend `n_fft` as a property that gives
`2 (len(f) - 1)` when `fs` is set and `f` equals the FFT grid
`arange(len(f)) fs / n_fft` (to floating-point tolerance), and None
otherwise. The FFT-size test keeps working, and nobody gets a wrong `n_fft`
for an MFCC or LPC envelope read on its own frequencies. The alternative is
to drop `n_fft` from the envelope.

D3. What `Aperiodicity` shares. `Aperiodicity` stays its own class, on the
FFT grid with `fs`, `n_fft` and `method`. Recommend a small private base
holding `data`, `t`, `f` and the grid reading (`__call__` on any `f`),
which both classes use; `_FrequencyView` and its bin-position reading go.
`warp_frequency`'s grid test then checks that base. The alternative,
folding `_FrequencyView` into `Aperiodicity` alone, duplicates the reading
code.

D4. `to_world` on `Aperiodicity`. It is defined on the shared base, so
`Aperiodicity` has it too, and the tool calls it. Recommend dropping it
there as well and writing `np.ascontiguousarray(x.data[0].T)` in the test
and the tool; keeping it on one class only would be inconsistent.

D5. `Aperiodicity.bands(envelope=)` weights by `envelope.data` and assumes
the envelope is on the aperiodicity's grid. Once any estimator's envelope
can be passed, recommend reading it at the aperiodicity's `t` and `f` when
the grids differ, as `source_filter_synthesize` already does, rather than
silently misaligning.

D6. The `discards` text. `GridEnvelope` and `SpectralEnvelope` say the
same thing; recommend keeping that one text. An MFCC envelope also drops
detail finer than the mel bands and an LPC one models only peaks, which
the estimators' own docstrings already say, so the class need not.

D7. A `TFRepresentation` base class. Recommend none, for the reasons
above. Instead, name a public Protocol for "power or ratio held on a grid"
(`t`, `f`, `data`, and `__call__(t, f)`) as part of this change, and type
the five parameters above with it, or with it or `EnvelopeFunction` where a
plain function is also accepted. The Protocol's name is Cho's choice;
`TimeFrequencyGrid` is one option. A method like `amplitude` stays outside
it, since only `harmonic_complex` reads it and falls back to calling.

D8. The names of the temporal pair. `Envelope`/`Envelopes` read as if a
`SpectralEnvelope` should have a plural too. The plural is literal: an
`Envelopes` is one `Envelope` per band, and `env[i]` gives one back. A
`SpectralEnvelope` holds all its time windows in one array and is not built
from one-window objects, so it has no singular to be plural over. Recommend
keeping the names and stating the rule in `philosophy.md` with the two
meanings: a plural class name means a bank of the singular class, one per
band. The alternative, renaming the temporal pair (for example
`TemporalEnvelope` and `Cochleagram`) so both kinds read alike, touches
every gallery page that draws a cochleagram for a gain in naming only.

## Work, once agreed

One change: the class and its readers in `spectral_envelope.py`,
`aperiodicity.py`, `cepstrum.py`, `mfcc.py`, `lpc.py`, `source_filter.py`,
`waveforms.py` and `__init__.py`; the tests and tools that build a
`GridEnvelope` or call `to_world`; the docs naming `GridEnvelope`; the two
module docstrings and the line in `philosophy.md`; and a CHANGELOG entry
for the breaking removal. The size is an estimate from the counts above:
roughly 100 lines touched, most of them renames.
