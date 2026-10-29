# Streaming: views computed one block at a time

The roadmap item "a block-by-block (streaming) modulation spectrogram",
widened to the views that can work the same way: the STFT, the cepstrum,
MFCCs, gammatone envelopes (a cochleagram), the modulation spectrogram and
the F0 track. A streaming view takes sound a block at a time as it arrives,
keeps what it needs from earlier blocks, and returns the time windows each
block completes.

Status: Cho accepted D1–D11 on 2026-10-29, D2 with the mixin and D7 with option (b). D12, added afterwards, was accepted the same day. The claims are checked by
`tools/check_streaming_claims.py` (C1–C12). It writes every streaming version
out from its definition in NumPy and SciPy, as the step functions D1
proposes, and imports sonore only for the batch results it compares them
with. Every number below is what it printed on 2026-10-28 in the cloud
container, for 3 s of white noise at 44.1 kHz, unless stated otherwise.
Timings are from that container's CPU. None were measured in a browser.

## Why

Two reasons, from the thread "Explore what to do with sonore" (2026-10-28):

- **Live use.** A page or an app that shows a running cepstrum, F0 track or
  modulation spectrum of the room needs analysis that keeps up with the
  microphone. That means a little work for each block, not one call on a
  finished sound.
- **A reference for sonore-infer.** Its online design (`docs/design/online.md`
  §9 in sonore-infer) has a step 2, "live input ... with the cochleagram
  computed one time window at a time as samples arrive". sonore-infer writes
  its own PyTorch cochleagram and tests it against sonore. A streamed sonore
  cochleagram that equals its batch version is what that step can be tested
  against.

Streaming is not a speedup. A whole file is analyzed fastest in one batch
call, which is unchanged. Streaming buys *latency*: each time window comes out
as soon as the samples it needs have arrived.

## The idea in plain words

### A step function, and a scan

Every streaming view is two pure functions:

```python
state = view.start(fs)                        # nothing seen yet
state, windows = view.step(state, block)      # one block in, new time windows out
```

`state` is a value, like a `Sound`: an immutable record of what the view
still needs from the past (for the STFT, the samples of the time window that
is not finished yet). `step` never changes it in place. It returns a new
state alongside the output. Nothing is hidden in an object that changes
under you.

Running `step` over a sequence of blocks, carrying the state from each call
to the next, is a *fold* that keeps every intermediate output, which
functional programming calls a *scan* (Scala's `scanLeft`, Haskell's
`mapAccumL`, JAX's `lax.scan`):

```python
def scan(view, blocks, fs):
    state = view.start(fs)
    for block in blocks:
        state, windows = view.step(state, block)
        yield windows
    yield view.finish(state)                  # the time windows that need the end
```

The contract is then one line: **concatenating what the scan yields gives
the batch view, for any way of cutting the sound into blocks.** The tests
check exactly that, with blocks of random length down to one sample.

"Cutting" means the blocks are consecutive slices of the sound, with no gaps
and no overlap. Their lengths are free and need not be equal: 441 samples
(10 ms at 44.1 kHz) is the example here, and a browser's audio callback
delivers 128. Blocks are not windowed. The view does its own windowing,
and the state carries any time window a block leaves unfinished. So the
latency is set by the view's window, not by the block length.

### Streams: where the blocks come from

`scan` above is a Python generator: it produces its outputs lazily, one at a
time, as whoever reads them asks. That is the `Stream` of the 2012 Scala
course (a sequence whose elements are computed only when read). Generators
compose like a Unix pipe, so stages chain:

```python
blocks = so.blocks(snd, 441)                  # Sounds of 441 samples; any iterable of them
for cochleagram in so.stream(bank, blocks):   # each item: the new time windows
    draw(cochleagram)
```

A modulation spectrogram is a stage reading another stage's output:
`so.stream(modulation, so.stream(bank, blocks))`. No stage knows where its
blocks come from: a file read in pieces, a list, or a microphone.

The reactive version (the second course) is the same thing turned around.
A microphone *pushes* blocks, and a callback runs for each one. Because
`step` is a plain function, a callback can call it directly and keep the
state in one variable:

```python
def on_audio(buffer):                         # called by the browser for every block
    global state
    state, windows = bank.step(state, so.Sound(buffer, fs))
    draw(windows)
```

Pull (a generator reads blocks) or push (a callback receives them) is only a
matter of who calls `step`. The step itself is the same in both cases.

### What it buys

- **One rule to test, the same for every view.** Streamed equals batch is
  checked by one generic test helper, given any view's `start`, `step` and
  `finish`.
- **No hidden state.** A state is a value you can print, compare, save,
  and run twice from. sonore-infer can branch one stream into several scene
  hypotheses without copying an object graph. A live page can rewind.
- **One copy of each formula.** The batch view and its step call the same
  per-window code (D2), so a fix to one is a fix to both.
- **Composition.** Pipelines are written as nested calls, and a new view
  that has a step slots into them.

What it costs: a new state is built at every step instead of updating one in
place. C7 measures this for the largest state in view.

## Which views can stream, and how

| View | What a time window depends on | Streams exactly? | Latency |
|---|---|---|---|
| STFT (`GaborFrame`) | its own samples | yes, bit for bit (C1) | half a window: 10 ms at the default 20 ms |
| Cepstrum, MFCC | one STFT time window, and a floor set by the whole sound's peak | with a fixed floor (D12), bit for bit (C2) | the STFT's |
| Gammatone subbands and envelopes | today: the whole sound (FFT, Hilbert) | not as they are; a time-domain gammatone does, bit for bit (C3) | the filter's own delay, nothing added |
| Envelopes at a lower rate | today: zero-phase lowpass, edge-extended resampling | not as they are; any FIR rate change does | half the FIR |
| Modulation spectrogram | the envelope over each kernel's length | with `align="causal"`, yes (its C8) | half a kernel: 3.0 s at 0.5 Hz, 23 ms at 64 Hz |
| F0 track | candidates: a stretch around the time window (cutting 0.1 s before it changes nothing, C9); tracking: the whole track | after two changes (C8, C9) | the candidates' look-ahead (not measured; about 50 ms from reading the code, an estimate) plus the decision delay |

## Claims

- **C1. The STFT streams bit for bit.** A step that keeps the samples of the
  next unfinished time window and computes each window as it completes,
  windowing and phasing it as SciPy's `ShortTimeFFT` does, gives the same
  numbers as `so.STFT`. The worst difference over blocks of random length
  (one sample each, then up to 64, 441 and 4410 samples) is 0. A 20 ms window can be
  computed 10 ms after its center. The end of the sound needs a `finish` call
  for the last time windows, which SciPy computes with zeros past the end.
- **C2. The cepstrum and MFCCs follow, except for their floor.** Each is
  computed from one STFT time window, apart from one global number: before the
  logarithm, both floor the magnitude `floor_db` (default -200 dB) below the
  largest value in each channel over the whole sound. On noise the floor
  never binds, and the two halves of the time windows give exactly what all of
  them give (worst difference 0). With 0.5 s of digital silence in the second
  half and the loudest stretch in the first, the floors differ, and so do
  the results: worst difference 0.60 in the cepstrum and 6.8 in the MFCCs
  (found 2026-10-29, after D1–D11 were accepted; D12).
- **C3. The sampled gammatone is an exact IIR filter, and it streams bit for
  bit.** The complex gammatone `k**(n-1) p**k`, with
  `p = exp((-2 pi b + 2 pi i cf) / fs)`, is the analytic impulse response
  `t**(n-1) exp(-2 pi b t) exp(2 pi i cf t)` sampled at `t = k / fs`. Its
  z-transform is rational: `p z^-1 A(p z^-1) / (1 - p z^-1)**n`, where `A` is
  the Eulerian polynomial (for `n = 4`, `A(x) = 1 + 4x + x**2`). This is the
  sampled impulse response exactly, not an approximation of it. Run by
  `scipy.signal.lfilter` with its state carried between blocks, it equals one
  call on the whole sound bit for bit (worst difference 0, 32 bands, random
  blocks up to 4410 samples).
- **C4. Its real part is today's gammatone band, to within 2e-5.** For a
  32-band ERB bank from 80 Hz to 8 kHz with no edge filters, the rms
  difference between the IIR's real part and `gammatone_filterbank(...).analyze`
  is 1.6e-7 of the band's rms in the median band and 1.7e-5 in the worst (the
  highest, 7.2 kHz), measured after the first 0.1 s. Today's bank is the
  continuous filter's transform sampled on the DFT grid. The IIR is the
  impulse response sampled in time. The likely reason they differ at all is
  the aliasing of the filter's tails, which grows toward high frequencies. That
  is an explanation, not something the checker tests.
- **C5. Its magnitude is the envelope, without a Hilbert transform, to
  within 0.7%.** The magnitude of the complex output against the Hilbert
  envelope of today's band: rms difference 0.11% in the median band, 0.70%
  in the worst. The likely reason (not tested) is that the complex
  gammatone has a little response at negative frequencies, which the Hilbert
  transform removes.
- **C6. Both run well inside real time in this container.** One 10 ms block
  (441 samples) takes 0.04 to 0.11 ms through the streamed STFT (20 ms Hann,
  5 ms hop) and 0.34 ms through the 32-band complex gammatone, over two runs
  (timings in a shared container vary). That is at most 1.1% and 3.4% of the
  block's duration. A browser running Pyodide will be slower by
  an amount not measured here. A live page needs that check first.
- **C7. Building a new state at every step is cheap.** The largest state in
  view is the modulation spectrogram's 0.5 Hz band, which keeps 6 s of
  envelope: 24 bands at 1 kHz, 1.15 MB. Copying it into a new array every
  10 ms block takes 0.05 to 0.08 ms over two runs (under 1% of the block).
  Writing the block into a ring buffer in place takes 0.003 to 0.004 ms. So the pure step costs nothing that
  matters, and a ring buffer stays available if a larger state ever needs it.
- **C8. A live F0 track can decide 50 ms late without changing the result.**
  Today's tracker runs a Viterbi pass over the whole track. A live tracker
  has to commit to each time window after a fixed delay. Decoding with that
  delay means running the pass up to `delay` past the time window, tracing back
  from the cheapest end state, and keeping only that time window's choice. On
  16 recordings in `docs/speech` (the first 4 s of each, 12,505 time windows),
  the share of time windows that come out unlike the whole-track pass is
  5.75% when deciding at once, 0.15% after 25 ms, and 0 after 50, 100 and
  200 ms. A zero on 16 recordings shows that differences are rare, not that
  they cannot happen.
- **C9. The F0 candidates are local, except for one detail of today's
  code.** Computed from the sound cut 0.1 s before a time window, the
  candidates are identical to the whole sound's on 99.64% of time windows of
  `bdl_arctic_a0131`. The rest differ because today's code processes 512
  time windows at a time, and the refinement takes one window length from the
  lowest candidate in each group. With every time window computed on its own,
  100% are identical. On the 16 recordings, computing each time window on its
  own changes no voicing decision and moves no F0 by more than 1%. The
  candidates themselves do move on some time windows: 27 of 505 on
  `bdl_arctic_a0131`.
- **C10. Making each block a `Sound` costs almost nothing.** Wrapping a
  441-sample block in a `Sound` (which copies the samples and checks their
  shape) takes 0.0013 ms, about 0.01% of the block's 10 ms.
- **C11. Computing the modulation spectrogram's kept time windows directly
  is far slower than batch's FFT.** Batch convolves each modulation band by
  FFT over the whole envelope and keeps every hop-th sample. Computing only
  the kept time windows, each as a dot product with the kernel, agrees with it
  to 1.5e-15 of the peak. For 10 s of 24 envelopes at 1 kHz, a 10 ms hop and
  the default 15 kernels, it takes 17.4 s against 0.24 s, about 70 times
  slower (a straightforward NumPy version; a better-written one was not tried).
- **C12. Sliding sums make the streamed modulation spectrogram cheaper, and
  stay within 1e-12 of batch.** A Hann window times a complex exponential
  is a sum of three complex exponentials, so each kernel's `y` and mean
  are five sliding sums `S_n(a) = sum_j e[n - L + 1 + j] a**j` with
  `|a| = 1`, each moved on one sample at a time with
  `S_n = conj(a) (S_(n-1) - e[n - L]) + e[n] a**(L - 1)`. They are summed
  again directly every 1024 samples so that rounding cannot build up. On
  10 min of 4 envelopes at 1 kHz, cut into random blocks, the worst
  difference from the batch FFT is 3.2e-14 of the peak (2.1e-14 in the last
  minute, so it does not grow). A 10 ms block takes 0.81 ms with 24 bands
  and 2.2 ms with 64 bands; computing each time window directly (the code
  first written for D7) took about 2 ms and 6 ms on the same container, a
  measurement made while writing it and not kept in the checker.

## Decisions

**D1. A streaming view is a pure step function with an immutable state. (accepted 2026-10-29)**
Each view that streams provides `start(fs) -> state`,
`step(state, block) -> (state, output)` and `finish(state) -> output`. States
are frozen dataclasses holding arrays. A public `typing.Protocol`,
`Streamable`, names the three methods, because `so.stream` takes "anything
that streams". `core/stream.py` holds the protocol, `so.stream`
(the generator above) and `so.blocks` (cut a `Sound` into blocks, which are
`Sound`s, D11).
*Alternatives.* (A) A stateful object per view, `s = so.STFTStream(frame, fs);
s.push(block)`. It is familiar, and it is what most audio libraries do. Its
best form is a thin class over the pure step, and that can be written
in a few lines on top of D1 if it is ever wanted. On its own, its state is hidden
and changes in place, so it can't be compared, saved or branched without
copying the object, and it breaks "a sound is a value" (philosophy.md).
(B) Generators only, with the state inside the generator. They are pleasant
to read, but the state is inaccessible, so a push-driven callback can't
use them and sonore-infer can't branch them.
*Recommended: D1 as stated.* C7 shows that building new states costs under 1% of
real time.

**D2. The step lives with the batch view, and both use one copy of each
formula. (accepted 2026-10-29, with the mixin below)** `GaborFrame.start/step/finish` sit in `frames/gabor.py`, next to
`analyze`. The modulation spectrogram's sit in `views/modspectrogram.py`,
and so on. There is no streaming folder, because folders follow meaning
(philosophy.md). The batch path and the step call the same function for one
time window (or one block of filtering). The batch path is not rewritten as
"the scan over one block", because that could change today's outputs.
Every batch result stays bit for bit as it is, the texture output included.

The buffering is written once. Streaming views come in three kinds:
- *Sliding-window views* (the STFT, pooled envelopes, FIR rate changes, the
  modulation kernels) all step the same way: append the block, compute every
  time window now complete, keep the rest. A mixin, `SlidingWindowStream` in
  `core/stream.py`, writes `start`, `step` and `finish` once. A view
  supplies `window_layout(fs)` (window length, hop, start of time window 0)
  and `time_windows(samples, fs, first, count)` (the view for those time
  windows, from the same code its batch path runs). The zeros before sample
  0, the zeros after the end and the time-window bookkeeping, where an
  off-by-one would make a stream disagree with batch, then exist in one
  place.
- *Maps* (the cepstrum, MFCCs) are computed from each STFT a stream yields,
  so they need no state of their own.
- *Recursive views* (the IIR gammatone's filter state, the F0 track's
  Viterbi pass) write their own `start`, `step` and `finish`.

*Alternative.* A decorator that adds the three methods (Cho asked,
2026-10-29). It works at runtime, but Python's type system cannot say "this
class plus three methods", so mypy (run in CI) would reject every call to
`step` and every use as a `Streamable`, and a reader would find no `step`
in the view's source. A decorator form mypy accepts would still need the
class to name the methods itself (`start, step, finish = ...`), which is
clumsier than inheriting.

**D3. A step returns ordinary views of the new time windows. (accepted 2026-10-29)** Each output
is an `STFT`, `Cepstrum`, `MFCC`, `TemporalEnvelope` or
`ModulationSpectrogram` holding only the time windows that block completed,
so `.plot()`, `.db` and everything else work on it. This needs one addition
per view: the index of its first time window, so that `t` is right. A
`concatenate` class method then joins the parts, and it is what the
"streamed equals batch" test calls.
*Alternative.* Return bare arrays and times. That is faster for a live
display and simpler, but every stage of a pipeline would need to know
array layouts, and a user couldn't plot a part. *Recommended: views.* A
live page can still read `.data`.

**D4. The contract is equality, tested bit for bit where measured. (accepted 2026-10-29)**
C1–C3 are bit for bit, so the tests assert `np.array_equal`, not a
tolerance. Each test cuts the sound at random, including into one-sample
blocks and empty blocks. Any later view that cannot be bit for bit (one
whose batch path groups FFTs differently, for example) says so in its
docstring, with the measured worst difference and a tolerance in the test.

**D5. A time-domain gammatone, batch and streamed. (accepted 2026-10-29)** For banks of
`Gammatone(phase="causal")` filters, a second analysis path runs each band
as the exact IIR filter of C3 and keeps the complex output. Its real part is
the band (within 2e-5 of today's, C4) and its magnitude is the envelope
(within 0.7% of the Hilbert envelope, C5). It is chosen with
`bank.analyze(snd, domain="time")`. The default stays `"frequency"`, today's
path, unchanged. Banks of other filter types refuse the time domain with a
message, because cosine and Morlet filters are zero-phase and have no causal
form. Edge filters (the lowpass and highpass) are refused too.
*Alternatives.* (A) A separate class, `IIRGammatoneBank`. That would be a
second bank with the same filters, contradicting "one general Filterbank".
(B) Hohmann's (2002) cascade of n one-pole complex filters. It is the usual
real-time gammatone, but its impulse response is
`C(k+n-1, n-1) p**k`, not `k**(n-1) p**k`, so it is a slightly different
filter. sonore already cross-checks it in
`tools/crosscheck_gammatone_hohmann.py` (it is the filter of Cho's
earlier gammatone notebook). (C) A truncated FIR. It is exact
only up to its truncation, and much slower at low frequencies.
*Recommended: the exact IIR, as an option of `analyze`.*

**D6. Lower envelope rates by FIR filters, so they stream too. (accepted 2026-10-29)** Today
`TemporalEnvelope.lowpass` is zero-phase and `resample` extends the ends
along a line. Neither has a causal form. Any FIR rate change (a polyphase
FIR decimator, or Hann-weighted pooling of the squared envelope, which is
what sonore-infer's cochleagram does: 25 ms time windows every 10 ms)
depends on a finite stretch of samples, like an STFT time window, so it
streams the same way (C1). Proposed: `TemporalEnvelope.pooled(win_dur,
hop_dur)`, Hann-weighted power in each time window, and a causal option for
`resample`, with zeros rather than edge extension at the ends. Today's
`lowpass` and `resample` stay as they are.

**D7. The modulation spectrogram streams as already designed. (accepted 2026-10-29, option (b))** Its
accepted D6 (2026-10-01) set the rule: keep the last `L_k - 1` envelope
samples per band, and match `align="causal"` exactly. Its C8 measured a
64-sample block version against the offline causal result (worst
difference 2e-16). Here it becomes `ModulationSpectrogram`'s step, fed by D5
and D6. Four adaptations:
- *It is a sliding-window view* (D2's mixin): its window is the longest
  kernel (6 s at 0.5 Hz), and shorter kernels take the tail of the buffer.
- *Its blocks are `TemporalEnvelope`s*, the output of D5 and D6, not
  `Sound`s. D11 applies to views that read sound.
- *Only `align="causal"` streams.* The centered result is the same numbers
  shifted in time and needs half a kernel of the future.
- *The arithmetic.* Batch convolves by FFT; a stream computes each time window
  directly. They agree to 1.5e-15 but not bit for bit (C11). Options:
  (a) batch also computes each kept time window directly, which gives one formula and bitwise
  equality, but is about 70 times slower (C11); (b) batch keeps its FFT, the
  step computes directly, and the test allows 1e-12, stated in the docstring
  as D4 allows. Cho first chose (a) on condition that the timing was close;
  C11 shows it is not, and Cho then chose (b). *Accepted: (b).* The step
  was first written with the direct sum; Cho then asked (2026-10-29) for
  the sliding sums of C12, about three times faster, which keep (b)'s
  1e-12 tolerance.

**D8. The F0 track gets a decision delay, and each time window's candidates
are computed on their own. (accepted 2026-10-29)** Two changes, both measured on 16 recordings:
(a) the refinement's window length depends only on the candidate itself, not
on the group of 512 time windows it is computed with (C9: no voicing or F0
change of over 1% on the 16 recordings, though the candidates moved on 27
of 505 time windows of the one recording compared). This changes `F0Track.candidates` slightly, so it is a
change to batch output, which is why it's a decision. (b) `f0_track(...,
delay=None)`: `None` is today's whole-track pass, and a number of seconds
is the fixed-delay decoding of C8, which the step reproduces exactly.
*Alternative for (b).* Stream with a delay and let the result differ from
batch now and then. That breaks the contract, and C8 shows it would differ
on 0% of 12,505 time windows at 50 ms anyway, so nothing is gained.
*Recommended: both changes, and a default `delay` of 0.05 s for the step.*

**D9. Order of work. (accepted 2026-10-29)** One PR per stage, each with its tests and docs:
1. `core/stream.py` (D1), the STFT, cepstrum and MFCC steps (C1, C2).
2. The time-domain gammatone and pooled envelopes (D5, D6): the cochleagram
   that sonore-infer's step 2 needs.
3. The modulation spectrogram step (D7).
4. The F0 track (D8).

A live microphone page is a separate design after these. It needs a Pyodide
timing check first (C6 has container numbers only).

**D10. Tests and checker. (accepted 2026-10-29)** A generic test helper takes any `Streamable`
and a batch function, and asserts equality over random cuts, one-sample
blocks, empty blocks, and a sound shorter than one time window. This checker
stays as the record of the measurements above. Target: under 1 s added to
the suite per stage.

**D11. Blocks are Sounds. (accepted 2026-10-29)** (Cho asked 2026-10-29.) `step` takes a `Sound`,
the same type every batch analysis takes, so a block can be listened to,
plotted or analyzed on its own like any other sound. Its `fs` is checked
against the state's, and a mismatch raises. `so.blocks` yields `Sound`s, and a
microphone source wraps each buffer in one. The cost is negligible (C10).
One caveat goes in the docstrings: analyzing a block on its own
(`so.STFT(block)`) is not the streamed result. A lone block has no samples
before or after it, so its first and last time windows see zeros where the
stream sees the neighboring blocks.
*Alternatives.* (A) Bare arrays only. This is slightly faster and is what
audio callbacks deliver, but it adds a second input type that carries no
`fs`. (B) Accept both, wrapping arrays with the state's `fs`. This is
convenient in a callback, but it is two ways to do one thing, and the wrap
is one call anyway.
*Recommended: Sounds only.* That keeps "a sound is a value" and one input
type, and inputs match D3's views-out.

**D12. A fixed floor for streamed cepstra and MFCCs. (accepted 2026-10-29)** (Found after D1–D11
were accepted; C2.) A stream can't know the whole sound's peak in advance.
Proposed: `Cepstrum` and `MFCC` take `reference=None`. `None` is today's
floor, `floor_db` below each channel's peak, so batch output is unchanged. A number
puts the floor at `floor_db` below that amplitude, the same for every time
window, which makes each time window depend on its own samples only. Streams
require it, and the stream equals batch called with the same `reference`.
*Alternatives.* (A) A running peak: the floor would rise as louder sound
arrives, so early time windows would disagree with batch. (B) No relative
floor in streams, only the smallest positive number: on silence that gives
values near log(1e-308), which swamp a plot or a distance.
*Accepted: `reference`.*

## API sketch

```python
import numpy as np
import sonore as so

snd = so.Sound.load("talker.wav")
frame = so.GaborFrame(20e-3)

# streamed equals batch, for any blocks
parts = list(so.stream(frame, so.blocks(snd, 441)))
streamed = so.STFT.concatenate(parts)
assert np.array_equal(streamed.data, so.STFT(snd, frame=frame).data)

# a cochleagram as samples arrive (D5, D6)
bank = so.gammatone_filterbank(n_bands=64, edges=False)
pooling = so.EnvelopePooling(25e-3, 10e-3, bank)
envelopes = (subbands.envelopes() for subbands in so.stream(bank, so.blocks(snd, 441)))
for cochleagram in so.stream(pooling, envelopes):   # the time windows each block completed
    ...

# pushed from a callback instead (a live page)
state = bank.start(snd.fs)
def on_audio(buffer):                         # the browser hands over a bare array
    global state
    state, subbands = bank.step(state, so.Sound(buffer, snd.fs))
```

Names are proposals. `blocks` and `stream` are functions (they make no class
of their own), and states are built by `start`, so no factory is involved.

## Out of scope

- A live microphone page and its Pyodide timing (D9).
- Microphone and sound-card input in Python. The proposed work runs on
  existing `Sound`s cut into blocks. A later adapter, for example over
  python-sounddevice as an optional extra, would wrap each input buffer in a
  `Sound` (D11) and call `step`. Cho expects it will be needed (2026-10-29).
- Streaming synthesis (Klatt, the source-filter synthesizer): a separate
  direction ("real-time synthesis").
- Streaming the time-varying STFT (`TVGaborFrame`) and WORLD-style views: their
  window lengths follow an F0 track, so they would follow D8.
- Textures and quilting: whole-sound statistics by design.
- Listener models.

## References

- de Cheveigné, A. & Kawahara, H. (2002). YIN, a fundamental frequency
  estimator for speech and music. *J. Acoust. Soc. Am.* 111(4), 1917–1930.
  As cited in `views/f0.md`; not checked again here.
- Hohmann, V. (2002). Frequency analysis and synthesis using a gammatone
  filterbank. *Acta Acustica united with Acustica* 88(3), 433–442. As cited
  by `tools/crosscheck_gammatone_hohmann.py`; not checked again here.
- Patterson, R. D. et al. (1992). Complex sounds and auditory images. As
  cited in sonore's `Gammatone` docstring; not checked again here.
- The Eulerian-polynomial form of `sum_k k**m x**k` is standard; C3 checks
  the filter numerically rather than relying on a citation.
