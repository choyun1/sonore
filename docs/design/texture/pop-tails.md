# How big the pops are: a tail exponent for textures

Some textures are made of pops: fire crackling, a typewriter, a clapper
standing close, tapping. Their loudest events are far louder than their
typical ones, and the sizes of the events follow a power law. The texture
model of McDermott & Simoncelli (2011) describes the spread of envelope
values by their variance, skew and kurtosis. For a power law with a small
exponent those moments are infinite: the value measured on a recording
keeps growing with the recording's length, so it describes the excerpt
more than the sound. This document proposes measuring the exponent of the
tail itself, as a one-way view that sonore reports, and checks whether
synthesis loses it. It does not propose imposing it, so synthesis output
does not change.

The idea is Cho's (2026-10-14): pops look like events with no stable
moments, and a power law might describe them where moments cannot. The
measurements below come from that thread. No code yet.

Status: draft, decisions D1 to D6 open for Cho.

## How the claims are verified

- **[check]**: printed by `tools/check_pop_tail_claims.py`. It uses NumPy
  and SciPy only, shares no code with sonore, and runs in about 10 s.
- **[survey]**: measured on the 168 recordings of McDermott & Simoncelli
  (2011), which Cho supplied. Their licences are unknown, so neither the
  recordings nor the survey scripts are in the repository; the scripts and
  their output are kept with the project's notes.
- **[ear]**: Cho's listening, 2026-10-14.
- **[math]**: follows in one line from the formulas shown.

## Background [math]

If event amplitudes A have a Pareto tail, P(A > a) ~ a^-alpha, then the
n-th moment of A exists only for n < alpha. The variance needs alpha > 2
and the kurtosis alpha > 4. Below that, a sample's kurtosis does not
converge: it grows with the number of events. The level in dB of a Pareto
amplitude is exponentially distributed, with scale 20 / (alpha ln 10), so
`so.event_train(..., level_db=scipy.stats.expon(scale=20 / (alpha * np.log(10))))`
already makes power-law pops (T1 [check]: slopes 0.994, 1.485, 2.495 for
alpha 1, 1.5, 2.5).

With alpha below 2 there is no smooth background to separate from the
pops: by the generalized central limit theorem, the sum of many such
events tends to a stable distribution, not to Gaussian noise. Above 2, the
small events add up toward Gaussian noise and the large ones stand out.

Sample kurtosis of Pareto samples (T3 [check], medians of 200 trials):

| alpha | 1000 samples | 10 000 | 100 000 |
|---|---|---|---|
| 1.5 | 228 | 1531 | 17 669 |
| 3 | 51 | 151 | 331 |
| 6 | 18 | 26 | 32 |

At alpha 6 the kurtosis is finite but still settling slowly.

## The measurement

1. High-pass the sound (at 300 Hz and, separately, at 2 kHz; pops are
   broadband, and the higher cutoff keeps low rumble out).
2. Take the Hilbert envelope and smooth it with a 500 Hz low-pass.
3. Pick the envelope's peaks at least 5 ms apart: the event sizes.
4. Fit the tail exponent by the Hill estimator on the largest 5% of the
   peaks (at least 10).

Accuracy:

- **The estimator alone** (T2 [check]): on 1000 Pareto samples, about the
  number of peaks in 7 s of a texture, the median is right and the spread
  across trials is 14-15% of alpha.
- **The whole measurement** (T4 [check]): on 7 s of shot noise, 150 events
  per second with Pareto amplitudes capped 40 dB above the smallest, the
  medians over 8 seeds are 1.49 for alpha 1, 1.33 for alpha 1.5 and 2.46
  for alpha 2.5. Overlapping events and the cap blur the range below about
  1.5, so 1 and 1.5 cannot be told apart reliably; 1.5 and 2.5 can.
- **Gaussian noise** (T5 [check]): about 36 (33 to 41 over 8 seeds).

## What the survey found [survey]

On the first 7 s of each of the 168 recordings, taking the smaller exponent
of the two cutoffs:

- **14 have an exponent below 2**: tapping (about 0.5), fire in a chimney
  (1.2), a bongo loop, a clapper close to the microphone, Fire1, writing
  with a pen, castanets, a typewriter. Rustling paper, hammering, a snare
  drum, gravel and gargling follow, between 2 and 2.5. 78 are below 4.
  Machines and steady noises sit at 12 to 23 (sander, blender, power
  saws).
- **The exponent mostly agrees with kurtosis on the ranking.** Its rank
  correlation with the model's median envelope kurtosis is -0.67. What it
  adds is a number that does not drift with the excerpt's length.
- **It does not predict convergence.** sonore's synthesis (30 iterations,
  5 s) converged to an average SNR of 20 dB or more on all 14 heaviest,
  with a median of 37 dB, the same as for the others.
- **Synthesis keeps most of the tail, but not all.** The exponent of each
  synthesis tracks its original's (rank correlation 0.81), but for the
  heavy-tailed sounds it comes out about 25% higher (lighter-tailed) in the
  median. Three lose much more than the measurement's spread:

  | Recording | original | synthesis |
  |---|---|---|
  | clapper, close to the microphone | 1.4 | 3.6 |
  | typewriter | 1.8 | 3.9 |
  | writing with a pen | 1.6 | 2.8 |

- **By ear** [ear]: the clapper's synthesis sounds like a water stream, the
  typewriter's like paper crumpling, the pen's pretty good. The two that
  lost the most tail are the two that lost their identity. Three sounds is
  a small sample, and clapping and typing also have a recognizable event
  shape and near-regular timing that the model does not describe either.
- **Our own fire recording** (`docs/textures/fire.flac`): a noise floor
  below the 80th percentile of peaks, then a tail with an exponent near 1
  over the top 5%, about 30 dB wide. Its envelope kurtosis grows from about
  220 over 1 s pieces to about 1900 over the whole 20 s. It is the
  recording that both sonore and the MATLAB toolbox synthesize worst
  (`docs/roadmap-done.md`).

## Proposed design

A one-way view, measured from any sound:

```python
tail = so.pop_tail(sound)              # cutoffs_hz=(300.0, 2000.0), fraction=0.05
tail.alpha                             # (2,) one exponent per cutoff
tail.standard_error                    # (2,) alpha / sqrt(number of peaks in the tail)
tail.peak_sizes                        # the peaks, per cutoff, for plotting
tail.plot(ax)                          # survival of peak sizes on log-log axes, with the fitted line
```

```python
def pop_tail(
    sound: Sound,
    *,
    cutoffs_hz: Sequence[float] = (300.0, 2000.0),
    fraction: float = 0.05,
    min_spacing: float = 0.005,
) -> PopTail: ...
```

`PopTail` is a `View`. It discards everything but the sizes of the
envelope's peaks, and has no `to_sound`: `event_train` with power-law
levels is one way to hear an exponent, but not a canonical inverse, so it
refuses and names that route.

Checking a synthesis is then two calls, `so.pop_tail(original)` and
`so.pop_tail(synthesis)`. Nothing in `TextureStats` or `synthesize`
changes, and texture synthesis stays bit-for-bit identical.

## Decisions

**D1. Where it lives.** (a) A view in `views`, for any sound, as above;
(b) a new class in `TextureStats`, measured on the model's own envelopes
but left out of the default classes, so synthesis would not impose it.
Recommend (a): the measurement is not tied to the texture model, works on
the broadband envelope where pops are clearest, and changes no saved
`TextureStats` files. (b) would measure on envelopes at 400 Hz, where a
pop shorter than 2.5 ms is smeared.

**D2. Name.** `so.pop_tail` and `PopTail`. Alternatives: `so.event_sizes`,
`so.tail_exponent`. Recommend `pop_tail`: it says what is measured in
plain words.

**D3. Defaults.** Cutoffs 300 Hz and 2 kHz, the largest 5% of peaks,
peaks at least 5 ms apart: the survey's settings. Recommend these, with
the caveats of T4 in the docstring.

**D4. A gallery section.** On the Sound textures page: our fire recording
and its synthesis with their tail plots side by side, and `event_train`
with power-law levels at a few exponents to hear what the number means.
The paper's recordings cannot be used (licences unknown). Recommend yes,
after the code.

**D5. Imposing the tail.** Out of scope here. Two routes would be: a tail
term in the synthesis objective, or texture synthesis of the background
plus `event_train` pops. Both change synthesis output and need their own
document. Recommend deciding after D4, by ear.

**D6. Roadmap.** The README's "Sparse pops" item points here; once built,
the measurement moves to `docs/roadmap-done.md` and the item keeps only
the open question of imposing the tail.

## References

- McDermott, J. H., & Simoncelli, E. P. (2011). Sound texture perception
  via statistics of the auditory periphery: evidence from sound synthesis.
  *Neuron* 71(5), 926-940.
- Hill, B. M. (1975). A simple general approach to inference about the
  tail of a distribution. *Annals of Statistics* 3(5), 1163-1174.
