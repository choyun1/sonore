# Making event_train fast with a pitch spread

Status: Cho chose B1 on 2026-10-14; built (see "As built" at the end).

`so.event_train` (docs/design/sources/event-texture.md) is quick without a
pitch spread and slow with one. That matters for a sonore-sketch tab with
sliders for the rate, level spread and pitch spread, where every slider move
re-renders. This note measures the problem on main and lays out ways to fix
it, with what each changes in the output. Every number comes from
`tools/check_event_train_speed.py` (cloud container, one run each, and a second
run agreed within about 10%; times are rough sizes, and Pyodide in a browser was 1.3 to 1.7 times slower than native
in the other sonore-sketch tabs *[estimate, not measured for this code]*).

## The problem

The event is the gallery's chirp (30 ms, 2 to 6 kHz), Poisson timing, levels
spread by 3 dB. Seconds per render on main, with the number of distinct
whole-cent shifts in brackets:

| duration | rate | SD 0 cents | SD 10 | SD 100 | SD 300 |
|---|---|---|---|---|---|
| 5 s | 30/s | 0.03 (1) | 0.54 (36) | 1.8 (114) | 2.1 (128) |
| 5 s | 300/s | 0.02 (1) | 0.97 (58) | 7.1 (419) | 15 (826) |
| 5 s | 1000/s | 0.02 (1) | 1.2 (71) | 9.2 (528) | 24 (1277) |
| 60 s | 1000/s | 0.55 (1) | 1.4 (82) | 11 (693) | 33 (1873) |

The cost follows the number of distinct shifts, not the duration: a 5 s
render at 300/s with a 100-cent spread is nearly as slow as a minute. So a
slider tab cannot fix it by rendering less.

Where the time goes: each distinct shift is one `Sound.resample`, which hands
SciPy's `resample_poly` the rate ratio as a fraction with denominator up to
10000. A shift of 37 cents becomes 8749/8564, for which SciPy designs a
Kaiser-windowed filter of 174 981 taps. In the 60 s, 1000/s, 100-cent case,
`firwin` (the filter design) took 10.1 of 11.1 s, 6.5 s of it computing the
Kaiser window. Filtering the 1323-sample event itself is cheap.

## Options

**A. Leave sonore alone; the tab rounds the pitch draws.** `pitch_cents`
accepts anything with `rvs`, so the tab can pass a wrapper that rounds the
normal draws to a 5- or 10-cent grid. Onsets and levels come out the same
(the draws are the same; only the pitches are rounded); each event's pitch
moves by at most half the step. Sonore's output is untouched.

| case | main | 5-cent grid | 10-cent grid |
|---|---|---|---|
| 5 s, 300/s, SD 100 | 6.3 s | 1.4 s | 0.8 s |
| 5 s, 1000/s, SD 100 | 7.9 s | 1.8 s | 1.0 s |
| 60 s, 1000/s, SD 300 | 33 s | 7.6 s | 4.1 s |

Still about a second natively for a short render, so one to two seconds in
the browser *[estimate]*, and the waveform is a different sound from what
`event_train` would make with the same seed (RMS difference 30% at 5 cents,
60% at 10 cents: the copies drift out of phase). The long-term spectrum in
the 2 to 6 kHz band differs by 1 to 1.8 dB in the 5 s renders and 0.3 to
0.5 dB in the 60 s ones, which is the scatter between two short noise-like renders
rather than a change of shape *[inferred from it shrinking with the longer
render, not tested further]*.

**B. One shared interpolation kernel instead of `resample_poly`
(recommended).** Each copy is the event read every 2^(cents/1200) samples
with the 32-sample Kaiser-windowed sinc that `move_sound` already uses to
read between samples. Going up in pitch, the kernel is stretched by the
ratio so its cutoff falls to fs / (2 × ratio) and nothing above it folds
back. Without that stretch, a noise burst shifted up 400 cents differs from
main's copy by 42% RMS (aliasing); with it, by 2%.

- Speed: 1.0 ms per copy against 16.6 ms on main. Whole renders:
  5 s at 300/s, SD 100: 6.3 s to 0.40 s; 5 s at 1000/s: 7.9 s to 0.53 s;
  60 s at 1000/s, SD 300: 33 s to 2.6 s.
- Output, for the chirp: RMS difference 6.4 × 10⁻⁴ of main's (−64 dB);
  long-term spectrum in the 2 to 6 kHz band within 0.015 dB. Per copy,
  4.5 to 6.8 × 10⁻⁴ from −400 to +400 cents.
- Output, for a broadband event (a 30 ms noise burst): 2% RMS difference
  (−34 dB), almost all of it above 20 kHz, where the two filters roll off
  differently; below 18 kHz the difference is about −60 dB re the copy's power.
- Unchanged, bit for bit: every render without a pitch spread (no copies are
  made; checked), and events given as a function of the pitch ratio (they
  make their own copies).
- Changed: every render with a pitch spread, and a render where every event
  shares one non-zero shift (one copy is made the same way). Two gallery
  sounds use a pitch spread (the 150-cent ones on the Sound textures page);
  they would change by about −64 dB.
- One copy of the formula: the reader would move out of
  `spatial/spatialization.py` into `core/` with a stretch argument, and
  `move_sound` would call it unstretched. Its output must stay bit for bit
  the same, which the build would test.

  B can be built two ways. **B1:** it simply replaces resampling in
  `event_train` (the default output changes, by the amounts above). **B2:**
  a keyword such as `resampling="exact"` (today's) or `"fast"`, with today's
  as the default so nothing changes unless asked; the tab passes `"fast"`.
  B1 keeps the API as it is; B2 keeps every existing output.

**C. A coarser fraction (rejected).** Limiting the denominator to 1000
keeps `resample_poly` but shortens its filter: 1.8 ms per copy, with pitch
off by up to 0.73 cents, more than the cent rounding already allows.

**D. Caching the filter designs (rejected).** Bit-identical, but each design
is one per distinct shift and about 175 000 taps (1.4 MB), so covering ±400
cents would hold about 1 GB *[arithmetic]*, and the first render is as slow
as now.

## The choice

1. **A**: no change to sonore; the tab rounds pitches; about 1 s natively.
2. **B1** (recommended): fast for everyone, pitch-spread outputs change by
   about −64 dB for the chirp.
3. **B2**: fast only when asked; every existing output stays identical.

Cho chose B1 on 2026-10-14.

What this note does not claim: no listening test was run, so "inaudible"
is not claimed for B, only the measured differences above; and the times
are one run on one machine.

## As built

`_read_between_samples` and its table moved from `spatial/spatialization.py`
to `core/utils.py` and gained a `stretch` argument; `move_sound` calls it
unstretched, and its output is bit for bit the same as before (checked on
150 000 random positions in white noise, an evenly spaced read, and reads
running off both ends). The stretched read reuses the same table: the sinc
widened by `stretch` at a distance d is the table's tap for the whole-sample
offset −floor(d / stretch), read at the fraction of d / stretch past it.
`event_train` reads a `Sound` event every `ratio` samples with `stretch` at
the ratio going up and 1 going down; the copy has ceil(n / ratio) samples,
the same count `Sound.resample` gave.

The checker now measures the built version against the resampling it
replaced, written out in the checker. Per copy: 1.0 ms against 16.8 ms.
Whole renders, before and now: 5 s at 300/s with a 100-cent spread, 7.0 s
to 0.40 s; 5 s at 1000/s with a 300-cent spread, 25 s to 1.4 s; 60 s at
1000/s with a 100-cent spread, 11.7 s to 1.0 s. The differences from the
resampled output are the ones measured above for option B (6.4 × 10⁻⁴ RMS
for the chirp trains, under 0.015 dB in the band spectrum). Renders without
a pitch spread are unchanged.

A new test compares copies with `Sound.resample` at −400, −7, 100 and 400
cents: within 0.1% RMS for the chirp and 3% for the noise burst.

The two gallery sounds with a pitch spread (Sound textures page) were not
re-rendered: they would change by about −64 dB, and the gallery's audio is
rendered on Cho's machine.
