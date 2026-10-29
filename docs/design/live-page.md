# A live page: sonore's streams on the microphone, in the browser

Status: proposed 2026-10-29, for Cho to accept or change. No page code
until then.

The streaming views (docs/design/streaming.md) take sound one block at a
time and give exactly what the batch analysis gives. This note proposes a
page that runs them on the visitor's microphone, in Pyodide, the way the
landing page's code cell already runs sonore. It answers the timing check
that streaming.md's D9 asked for first.

## What the page is for

You sing or speak into the laptop and see sonore's own analyses of it as it
happens: a cochleagram scrolling past, with the F0 track drawn over it. It
shows that the streams are real streams, and it is the "live analyzer" item
of the directions list. It is a demonstration, not a measuring instrument:
the browser resamples the microphone and may apply its own processing
(L4).

## Timing

`tools/time_streams.py` runs each stream over `bdl_arctic_a0131` (2.5 s at
16 kHz) and prints the best of three runs, in ms of computing per 10 ms of
sound. Under 10 keeps up with the microphone. The same script runs in
headless Chromium under Pyodide 314 through `tools/pyodide_timing/`. Both
columns come from this project's cloud machine, which is shared and gives
numbers that vary by about 20% from run to run; a laptop will differ, so
the page's last step (L7) is to time it on one.

| Stream | CPython, 10 ms blocks | CPython, 50 ms | Pyodide, 10 ms | Pyodide, 50 ms |
|---|---|---|---|---|
| STFT, 25 ms windows every 10 ms | 0.2 | 0.1 | 0.4 | 0.2 |
| cochleagram, 32 gammatones pooled to 10 ms | 1.3 | 0.6 | 3.3 | 1.0 |
| cochleagram, 64 gammatones | 1.8 | 1.1 | 4.2 | 1.6 |
| modulation spectrogram, 32 bands | 2.1 | 1.4 | 4.0 | 2.0 |
| F0 track, decided 50 ms late | 6.2 | 3.8 | 11.8 | 6.2 |
| cochleagram (32) and F0 together | 9.2 | 4.5 | 14.0 | 8.0 |

What it says:

- Pyodide is about two to three times slower than CPython here.
- Larger blocks are much cheaper per second of sound. Most of a 10 ms step
  is fixed cost (Python and NumPy calls per step), not arithmetic.
- In Pyodide every view except the F0 track keeps up with 10 ms blocks; the
  F0 track does not (11.8 ms per 10 ms). With 50 ms blocks the cochleagram
  and F0 together take 8.0 ms per 10 ms, which keeps up, with little to
  spare on this machine (an earlier run of the same pair gave 6.8).

## Decisions

**L1. Where the page lives.** `docs/live/index.html`, next to `docs/try/`
(the landing page's worker), linked from the gallery menu under "Seeing and
changing sound" and from the landing page. Gallery pages are built by
`docs/gallery/build.py` with rendered sounds and figures; this one is
hand-written HTML and JavaScript, like the landing page, because nothing
on it can be rendered ahead of time.

**L2. Audio path.** An `AudioContext` at 16 kHz (the browser resamples the
microphone), an `AudioWorklet` that copies each 128-sample buffer to a
Pyodide worker, and the worker gathers 50 ms blocks (800 samples), steps
the streams and posts each part back to the page, which draws it on a
canvas. Python never runs on the audio thread, so a slow step cannot glitch
the audio; it only delays the picture. Latency is the 50 ms block plus the
F0 track's 50 ms decision delay, about 0.1 s.
*Alternatives.* 10 ms blocks: lower latency, but the F0 track falls
behind (table). 48 kHz: the browser's usual rate, but every stream costs
about three times more.

**L3. What it shows.** A scrolling cochleagram (32 gammatone bands over
`gammatone_filterbank`'s default range, pooled to 10 ms) over the last 5 s, with the F0 track drawn on
it where it is voiced, and the current F0 as a number. One "Start
microphone" button; nothing is recorded or sent anywhere, and the page
says so. The modulation spectrogram is left for later: with the other two
it would not keep up in Pyodide on this machine.

**L4. What the browser does to the sound.** The page asks for the
microphone with `echoCancellation`, `noiseSuppression` and
`autoGainControl` off, so the analysis sees the signal rather than the
browser's speech cleanup; browsers may still apply some. The page says
that this is a demonstration of the streams and not a calibrated
measurement.

**L5. Falling behind.** The worker counts the samples the worklet sent
against those it has analyzed. If the backlog passes 1 s (the page cannot
keep up on that machine), it drops the backlog, starts the streams again
from their start state, and shows a line saying it fell behind and
restarted. That is the browser counterpart of `DroppedSamplesError` in
`so.live_blocks`: a gap is never analyzed as if it were continuous sound,
but a page restarts instead of stopping.

**L6. Which sonore it runs.** The landing page's worker installs sonore from
PyPI, and 0.9.0 has no streams. The page goes live with the next release
(0.10.0), and is developed against a wheel of the working tree, as the
timing harness does.
*Alternative.* Ship the wheel with the Pages site. That would make the page
work before a release but adds a second way of installing sonore to keep
working.

**L7. Testing.** A Playwright test (local, like the timing harness, not in
CI, which has no Pyodide download) opens the page in Chromium with a fake
microphone playing a WAV file (`--use-fake-device-for-media-stream
--use-file-for-fake-audio-capture`). It checks that the page draws and
that the F0s it shows equal `f0_track(..., delay=0.05)` on the samples the
worker received, which the worker keeps for the test. Then Cho opens it on
a laptop with a real microphone and reads off the timing line the page
shows (computing time per 10 ms of sound).

## Out of scope

- The modulation spectrogram on the page (L3), and any choice of view by
  the visitor.
- A phone layout beyond what fits by itself: phones are slower, and the
  timing check here is for a laptop.
- Recording or saving the microphone's sound.
