# A start with pops for texture synthesis

Texture synthesis (McDermott & Simoncelli, 2011) starts from Gaussian noise
and adjusts it until its statistics match a recording's. On recordings with
heavy pops, a crackling fire above all, the statistics end up matched but
the pops do not: they come out wider, more numerous and less aligned across
channels than the recording's, and the background between them pumps. The
gallery handles the fire with a recipe of its own, background plus pops
(`docs/design/views/envelope-peaks.md`, D5), which takes the pops out,
synthesizes the rest and puts the recording's biggest pops back with
`so.event_train`.

Cho asked (2026-10-15) whether that recipe was necessary, or whether the
plain synthesis could be made to handle such recordings with a simpler
change. The answer, measured on our fire and two of the paper's fires, is
that the synthesis itself needs no change: started from noise that already
has sparse, sharp bumps at power-law sizes, the same iterations keep the
bumps, shape them, and match the statistics better than from plain noise.
This document records the measurements, sets out the start as a function,
and lists the decisions. The start is sonore's own; the paper and its
toolbox start from noise.

Status: decisions D1 to D6 proposed, 2026-10-15; built on the thread's
branch for Cho to review alongside this document.

## How the claims are verified

- **[check]**: printed by `tools/check_pop_start_claims.py`, which runs
  `synthesize` from both starts on the recordings named and measures the
  results with NumPy and SciPy only (statistics SNR by `TextureStats.snr`;
  pop exponent, gap, crest, pop width, rate and channel coherence by the
  checker's own code). Our fire, stream and applause come from
  `docs/textures/`; the paper's Fire1, "Fire in chimney" and Fire2 are read
  from a directory given on the command line (their licenses are unknown,
  so they are not in the repository). About 20 minutes on four cores.
- **[listen]**: Cho, by ear, on the thread's listening page, 2026-10-15.
- **[estimate]**: a judgment, labeled as such, not measured.

Numbers below are from the thread's scratch runs (30 iterations, seed 1,
the start's seed 5, the recording's own length: 20 s for our fire, 5.7 s
for Fire1, 7 s for the chimney). The checker reproduces them from the
built function; its printout is the record.

## What goes wrong from a noise start [check]

Measured on the plain synthesis of our fire (20 s) against the recording:

| | Original | From noise |
|---|---|---|
| Average statistics SNR (dB) | | 30.0 |
| Pop exponent above 300 Hz | 1.15 | 0.85 |
| Pops per second more than 10 dB / 20 dB above the median envelope | 9.4 / 2.3 | 15.1 / 3.1 |
| Pop width at half maximum (ms, median) | 1.66 | 2.8 |
| 5th percentile of the envelope below its median (dB) | 4.7 | 9.5 |
| Channels peaking together at the 20 biggest pops | 81% | 56% |

The pop exponent alone would say the synthesis has pops as heavy as the
recording. What it misses: the pops are wider, there are more of them, the
background between them dips twice as deep, and a pop in the synthesis is
often a pop in only half the channels, where in the recording it is in
four fifths of them. Two things rule out the usual suspects:

- The model's own representation keeps the pops. Rebuilding the recording
  from its 400 Hz compressed envelopes and fine structure keeps exponent,
  gap and crest within measurement error (our fire: exponent 1.24 for
  1.15, crest 37.1 for 36.7 dB). The loss happens in the iterations.
- The statistics are unstable for these recordings, but that is not the
  cause. Two halves of our fire agree with each other to only 10 dB
  (Fire1 13 dB, chimney 17 dB), because a few pops dominate the variance
  and kurtosis. A synthesis at 30 or 40 dB is as matched as the statistics
  can say, from either start.

Longer syntheses (20 s), more iterations (60), and less envelope
compression (exponent 0.6 or 1.0 instead of 0.3) do not help. Compression
makes everything worse: average SNR 18 and 8 dB on our fire.

## The start [check]

Gaussian noise multiplied by a gain that is 1 everywhere except at sparse
bumps: each bump is 4 ms long, decays with a 1 ms time constant, and has a
height of `1 + p` with `p` Pareto-distributed with the recording's pop
exponent, capped at 40 dB; where bumps overlap the larger wins. Onsets are
uniform at random, at the recording's pop rate.

| | Original | From noise | From a pop start |
|---|---|---|---|
| Our fire: SNR (dB) | | 30.0 | 40.3 |
| exponent | 1.15 | 0.85 | 0.98 |
| pop width (ms) | 1.66 | 2.8 | 1.65 |
| 5th percentile below median (dB) | 4.7 | 9.5 | 4.9 |
| channels peaking together | 81% | 56% | 93% |
| Fire1: SNR (dB) | | 38.9 | 41.9 |
| exponent | 1.83 | 2.43 | 1.59 |
| Chimney: SNR (dB) | | 40.5 | 42.2 |
| exponent | 1.33 | 2.02 | 0.99 |

(Those runs used 40 bumps a second; see "Rate" below.) Every statistic
class improves; the modulation power, the hardest, goes from 11.5 to
12.8 dB on our fire and from 21 to 22.5 dB on Fire1. With the function as
built (the recording's own rate, below), our fire reaches 42.6 dB, with
exponent 1.14, gap 43.0 dB and crest 38.5 dB against the recording's 1.15,
43.3 and 36.7, pop width 2.0 ms (1.66 in the recording, 2.8 from noise)
and 80% of channels peaking together (81%; 55% from noise); Fire1 41.2 dB
and the chimney 41.2 dB. The gallery's 5 s synthesis goes from 21 to
37 dB. The checker's printout of 2026-10-15 is the record.

**Exponent.** The start's exponent barely matters: on Fire1 a start drawn
at a fixed exponent of 1.5 ends at 1.51 and one at 3.0 ends at 1.86 (the
recording is 1.83), with the same SNR. So the statistics do carry the tail
once the start has sharp events to work with; what a noise start never
grows is the events themselves. The start still takes the recording's
exponent, because it costs nothing and keeps the biggest bumps in
proportion [estimate].

**Rate.** The start's rate carries through to the result, which the
statistics do not pin. On our fire (9.4 pops a second more than 10 dB
above the median envelope): 40 bumps a second give 11.3, 20 give 8.3, 10
give 7.2, with the SNR rising to 43.1 dB at 10. Cho heard the 40-a-second
result as having too-frequent crackles [listen]. So the rate is measured
from the recording: the number of 5 ms bins whose envelope maximum (the
bins of `so.envelope_peaks`, high-passed at 300 Hz) first rises more than
10 dB above the median maximum, per second (`pop_rate`). On our fire that
is 5.2 a second, Fire1 6.1, the chimney 12.3, Fire2 9.6, stream 3.3,
applause 3.1.

**Where it does not belong.** On stream, applause and the paper's Fire2
(exponents 3.4, 4.5, 5.0) the pop start matches the statistics as well as
noise (within 1.5 dB at 40 bumps a second; within 0.5 dB at their own
rates, 3.3 and 3.1 a second) but leaves a few pops the statistics do not
remove:
Fire2's crest factor is 13 dB in the recording, 20 dB from noise and 24 dB
from a pop start. Cho heard applause as a bit better and Fire2 as a bit
worse [listen]. So the start is for recordings whose pops measure heavy,
an exponent below about 2 from `so.envelope_peaks` [estimate, from six
recordings].

**What it does not fix.** The chimney's background is louder than the
recording's from either start (5th percentile 6.2 to 7.5 dB below the
median, against 5.4 in the recording and 8.1 from noise), and Cho heard the
background-plus-pops recipe as a little better there, because its
background is synthesized separately and turned down 3 dB [listen]. The
recipe stays for that case and for the rhythmic sounds, which no start
helps.

## Design

```python
start = so.texture.synth.pop_start(recording, duration)   # rate=None, exponent=None, fs=None, rng=None
sound, report = synthesize(target, duration=duration, init=start)
```

```python
def pop_rate(sound: Sound, *, cutoff_hz: float = 300.0) -> float: ...

def pop_start(
    recording: Sound,
    duration: float,
    *,
    rate: float | None = None,       # pops a second; the recording's own by default
    exponent: float | None = None,   # the recording's, from envelope_peaks at 300 Hz, by default
    fs: float | None = None,         # the recording's rate by default
    rng: RNG = None,
) -> Sound: ...
```

- Both live in `sonore/texture/synth.py` beside `synthesize`, which is
  unchanged: the start goes in through the existing `init` argument, and
  with no `init` every synthesis stays bit for bit what it was.
- `pop_start` says in its docstring that the start is sonore's own, where
  the paper starts from noise, and what it does on recordings without heavy
  pops.
- The gallery's `tools/make_texture_synths.py` grows a `POP_STARTS` list
  (our fire), written as `<name>__popstart.flac`, so the Sound textures
  page can play the fire from a pop start next to the synthesis from noise.

## Decisions

**D1. Where it lives. Proposed: two functions in `texture/synth.py`.** The
alternative, a `start="pops"` option on `synthesize`, would need the
recording, which `synthesize` does not have (it takes statistics). A
function that returns a `Sound` for the existing `init` keeps `synthesize`
untouched and lets a caller make any other start the same way.

**D2. Name. Proposed: `pop_start` and `pop_rate`.** A start with pops, and
the rate it measures. "Spiky start" was the thread's working name.

**D3. Defaults. Proposed:** rate and exponent measured from the recording,
4 ms bumps with a 1 ms decay, capped at 40 dB, onsets uniform at random.
The rate measurement is the crossing count above, which reuses the bins of
`so.envelope_peaks`; a more careful count (peaks on the continuous envelope)
gives larger numbers (9.4 against 5.2 on our fire) and the checker reports
how the result's rate compares with the recording's by the same measure.

**D4. Not the default start. Proposed: `synthesize` keeps starting from
noise.** On recordings without heavy pops the pop start leaves pops behind
(Fire2 above). The docstring says when to use it.

**D5. Gallery. Proposed:** on the Sound textures page, after the fire's
synthesis from noise, the fire synthesized from a pop start, with a short
paragraph that says the start is sonore's own, what it measured, and that
one listener compared them. On the Events and pops page, the sentence "a
synthesis as long as the original does no better" gains "but one started
with pops does", with a link, and the recipe's paragraph says when to reach
for it instead (a background that needs turning down; rhythmic sounds, for
which neither works). The listening-page numbers are not repeated there
beyond the statistics SNR the page already reports.

**D6. Not now:** tuning the bump (its shape, per-channel starts), making the
rate follow the continuous-envelope count, or imposing the rate as a
statistic. The rate following the start rather than the statistics is
stated as a limitation in the docstring and on the page.

## References

- McDermott, J. H., & Simoncelli, E. P. (2011). Sound texture perception
  via statistics of the auditory periphery: Evidence from sound synthesis.
  *Neuron*, 71(5), 926–940.
  [doi:10.1016/j.neuron.2011.06.032](https://doi.org/10.1016/j.neuron.2011.06.032)
- Hill, B. M. (1975). A simple general approach to inference about the
  tail of a distribution. *Annals of Statistics*, 3(5), 1163–1174.
  [doi:10.1214/aos/1176343247](https://doi.org/10.1214/aos/1176343247)
