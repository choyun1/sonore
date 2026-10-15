# Envelope peaks: how big the pops are

Some textures are made of pops: fire crackling, a typewriter, a clapper
standing close, tapping. Their loudest moments are far louder than their
typical ones, and the sizes of the loud moments follow a power law. The
texture model of McDermott & Simoncelli (2011) describes the spread of
envelope values by their variance, skew and kurtosis. For a power law with a
small exponent those moments are infinite: the value measured on a
recording keeps growing with the recording's length, so it describes the
excerpt more than the sound. This document proposes a view that measures
the power law itself, `so.envelope_peaks`, and uses it to check whether
texture synthesis keeps the pops. Synthesis does not impose it, so
synthesis output does not change.

The idea is Cho's (2026-10-14): pops look like events with no stable
moments, and a power law might describe them where moments cannot. Cho also
simplified the measurement to the histogram of the envelope's maximum in
each 5 ms bin.

Status: D1 to D4 settled by Cho, 2026-10-14. D1: a `View` class under
`views`. D2: `so.envelope_peaks` returning `EnvelopePeaks`, after Cho
turned down `pop_tail` ("tail" also means a reverberant decay). D3 and D4
as recommended. D6 as recommended. D5 settled by ear, 2026-10-14 (below).

## How the claims are verified

- **[check]**: printed by `tools/check_envelope_peaks_claims.py`. It uses
  NumPy and SciPy only, shares no code with sonore, and runs in about 10 s.
- **[survey]**: measured on the 168 recordings of McDermott & Simoncelli
  (2011), which Cho supplied. Their licences are unknown, so neither the
  recordings nor the survey scripts are in the repository; the scripts and
  their output are kept with the project's notes. The survey of all 168
  used local envelope peaks at least 5 ms apart instead of the maximum per
  5 ms bin; on eight recordings checked both ways the exponents agree
  within about 0.2.
- **[ear]**: Cho's listening, 2026-10-14.
- **[math]**: follows in one line from the formulas shown.

## Background [math]

If amplitudes A have a Pareto tail, P(A > x) ~ x^-alpha, then the n-th
moment of A exists only for n < alpha. The variance needs alpha > 2 and the
kurtosis alpha > 4. Below that, a sample's kurtosis does not converge: it
grows with the sample. The level in dB of a Pareto amplitude is
exponentially distributed, with scale 20 / (alpha ln 10), so
`so.event_train(..., level_db=scipy.stats.expon(scale=20 / (alpha * np.log(10))))`
already makes power-law pops (T1 [check]: slopes 0.994, 1.485, 2.495 for
alpha 1, 1.5, 2.5).

With alpha below 2 there is no smooth background to separate from the
pops: by the generalized central limit theorem, the sum of many such events
tends to a stable distribution, not to Gaussian noise. Above 2, the small
events add up toward Gaussian noise and the large ones stand out.

Sample kurtosis of Pareto samples (T3 [check], medians of 200 trials):

| alpha | 1000 samples | 10 000 | 100 000 |
|---|---|---|---|
| 1.5 | 219 | 1749 | 17 878 |
| 3 | 49 | 148 | 326 |
| 6 | 18 | 26 | 32 |

At alpha 6 the kurtosis is finite but still settling slowly.

## The measurement

1. High-pass the sound, at 300 Hz and, separately, at 2 kHz. Pops are
   broadband, and the higher cutoff keeps low rumble out.
2. Take the Hilbert envelope and smooth it with a 500 Hz low-pass.
3. Take the envelope's maximum in each 5 ms bin. These maxima are the
   histogram the view holds.
4. Fit the exponent of their upper tail by the Hill estimator (Hill, 1975)
   on the largest 5% (at least 10 values), and give its uncertainty as
   alpha / sqrt(k), where k is the number of values fitted.

The maximum matters: the mean in each bin smooths short pops away, and its
exponents come out higher (typewriter 2.0 by maximum, 3.0 by mean; pen on
paper 1.8 and 3.3 [survey]).

## How far to trust the number

- **The estimator alone** (T2 [check], 400 trials of Pareto samples): the
  median is right, and the spread matches alpha / sqrt(k): 36% at k = 10
  (1 s of bins), 25% at k = 20 (2 s), 14% at k = 50 (5 s), 13% at k = 70
  (7 s).
- **The whole measurement** (T4 [check]): on 7 s of shot noise, 150 events
  per second with Pareto amplitudes capped 40 dB above the smallest, the
  medians over 8 seeds are 1.28 for alpha 1, 1.61 for alpha 1.5 and 2.36
  for alpha 2.5. Overlapping events and the cap blur the range below about
  1.5: 1 and 1.5 cannot be told apart reliably, 1.5 and 2.5 can.
- **Gaussian noise** (T5 [check]): about 34 (32 to 38 over 8 seeds) at
  44.1 kHz. It grows with the sample rate, which puts more samples in each
  bin: about 24 at 16 kHz (measured with `so.envelope_peaks` on one 7 s
  sample).
- **Real recordings** [survey], 2 kHz cutoff:

  | Recording | 7 s, top 5% | top 10% | top 20% | 2 s pieces | 1 s pieces |
  |---|---|---|---|---|---|
  | clapper, close to the microphone | 1.7 | 1.4 | 1.8 | 1.7-2.1 | 1.2-2.4 |
  | typewriter | 2.1 | 2.1 | 1.7 | 1.6-2.1 | 1.4-2.5 |
  | writing with a pen | 1.8 | 2.1 | 2.0 | 1.4-3.9 | 1.2-4.7 |
  | fire in a chimney | 1.3 | 1.3 | 1.3 | 1.3-1.7 | 1.0-2.7 |
  | stream | 5.7 | 4.6 | 4.2 | 4.8-7.6 | 4.2-8.9 |
  | sander | 12.5 | 12.5 | 9.6 | 11.2-12.6 | 9.7-14.9 |

  Over 7 s the fraction fitted (5 to 20%) moves the pop sounds' exponents
  by up to about 0.4; the top 2% is unreliable. A 2 s piece can miss the
  big events altogether (pen: 3.9). Coarse classes hold at 2 s: heavy
  (below 2), moderate (2 to 6), light (above 8). Telling 1.4 from 1.8
  needs 5 to 7 s.

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
  adds is a number that does not drift with the excerpt's length, and an
  uncertainty that says how long the excerpt needs to be.
- **It does not predict convergence.** sonore's synthesis (30 iterations,
  5 s) reached an average SNR of 20 dB or more on all 14 heaviest, with a
  median of 37 dB, the same as for the others.
- **Synthesis keeps most of the pops, but not all.** The exponent of each
  synthesis tracks its original's (rank correlation 0.81), but for the
  heavy-tailed sounds it comes out about 25% higher in the median. Three
  lose much more than the measurement's spread:

  | Recording | original | synthesis |
  |---|---|---|
  | clapper, close to the microphone | 1.4 | 3.6 |
  | typewriter | 1.8 | 3.9 |
  | writing with a pen | 1.6 | 2.8 |

- **By ear** [ear]: the clapper's synthesis sounds like a water stream, the
  typewriter's like paper crumpling, the pen's pretty good. The two that
  lost the most were the two that lost their identity. Three sounds is a
  small sample, and clapping and typing also have a recognizable event
  shape and near-regular timing that the model does not describe either.
- **The pops are lost across channels, not within them.** Measured on the
  texture model's own envelopes, channel by channel (compressed, at
  400 Hz; exponents scaled back by the compression), the syntheses keep
  their tails: clapper 2.0 to 2.6, typewriter 3.5 to 4.1, pen 3.8 to 3.4,
  fire in a chimney 1.8 to 1.8 (medians over channels). Matching kurtosis
  band by band works. What goes is the alignment: in a real clap every
  band peaks at the same instant, while the synthesis seems to make its
  big moments in each band at different times, so they never add up to
  one broadband pop. The envelope correlations the model imposes describe
  average co-modulation, not whether the extremes line up. That reading
  fits these numbers but is not tested directly.
- **Our own fire recording** (`docs/textures/fire.flac`): a noise floor
  below the 80th percentile of peaks, then a tail with an exponent near 1
  over the top 5%, about 30 dB wide. Its envelope kurtosis grows from about
  220 over 1 s pieces to about 1900 over the whole 20 s. It is the
  recording that both sonore and the MATLAB toolbox synthesize worst
  (`docs/roadmap-done.md`).

## Design

A one-way view, measured from any sound:

```python
peaks = so.envelope_peaks(sound)    # cutoffs_hz=(300.0, 2000.0), bin_seconds=0.005, fraction=0.05
peaks.exponent                      # (2,) one exponent per cutoff
peaks.uncertainty                   # (2,) exponent / sqrt(number of values fitted)
peaks.maxima                        # (n_bins, 2) the envelope's maximum in each bin, per cutoff
peaks.plot(ax)                      # survival of the maxima on log-log axes, with the fitted line
```

```python
def envelope_peaks(
    sound: Sound,
    *,
    cutoffs_hz: Sequence[float] = (300.0, 2000.0),
    bin_seconds: float = 0.005,
    fraction: float = 0.05,
) -> EnvelopePeaks: ...
```

- `EnvelopePeaks` is a `View` in `views/envelope_peaks.py`. Its `discards`
  sentence: everything but the largest envelope value in each bin above
  each cutoff, including when the bins occurred and the spectrum. Its
  `to_sound` refuses: `event_train` with power-law levels is a way to hear
  an exponent, not an inverse, and the refusal names it.
- It warns when fewer than 50 values are fitted (under 5 s at the default
  fraction), with the uncertainty in the message.
- Checking a synthesis is two calls, `so.envelope_peaks(original)` and
  `so.envelope_peaks(synthesis)`. Nothing in `TextureStats` or
  `synthesize` changes, and texture synthesis stays bit-for-bit identical.

## Decisions

**D1. Where it lives. Settled: a view.** A `View` class under `views`, for
any sound. The alternative, a class inside `TextureStats` measured per
channel on the model's envelopes, would have measured where synthesis
already does well (see "The pops are lost across channels").

**D2. Name. Settled: `so.envelope_peaks`, returning `EnvelopePeaks`.**

**D3. Defaults. Settled:** cutoffs 300 Hz and 2 kHz, the maximum in each
5 ms bin, the largest 5% fitted, with the uncertainty and the warning
above.

**D4. A gallery section. Settled: yes, after the code.** On the Sound
textures page: our fire recording and its synthesis with their survival
plots side by side, and `event_train` with power-law levels at a few
exponents to hear what the number means. The paper's recordings cannot be
used (licences unknown).

**D5. Imposing it. Settled by Cho, 2026-10-14, by ear: no change to
synthesis; a recipe on the gallery page instead.** Texture synthesis of the
fire with its pops limited away, plus the recording's own pops added back
with `event_train` at the measured exponent and rate, sounded better than
the plain synthesis on our fire, two of the paper's fires and the close
clapper, to Cho's ear. The recipe is our own combination of standard parts,
not from a paper, and that listening is its only test, so the gallery
says so and claims no more. Making the plain synthesis longer (20 s) or iterating longer did
not help. It does not help rhythmic or impact sounds (tapping, a bongo
loop, a typewriter, castanets), which the paper itself lists among the
model's failures (rhythm). Imposing the alignment of extremes across
channels stays an idea, not a plan. Later (2026-10-15): a synthesis started
from noise that already has pops keeps them, with no change to the
synthesis; see `texture/pop-start.md`. The recipe stays for a background
that needs turning down.

**D6. Roadmap. Settled:** the README's "Sparse pops" item points
here now; once built, the view moves to `docs/roadmap-done.md`. With D5
settled, the item is done too.

## References

- Hill, B. M. (1975). A simple general approach to inference about the
  tail of a distribution. *Annals of Statistics* 3(5), 1163-1174.
- McDermott, J. H., & Simoncelli, E. P. (2011). Sound texture perception
  via statistics of the auditory periphery: evidence from sound synthesis.
  *Neuron* 71(5), 926-940.
