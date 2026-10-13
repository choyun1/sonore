# From single events to a texture

The idea is one source that takes a sound from a single event, through
sparse events, to a dense texture. One event, a short chirp say, is copied
many times. Three distributions say how the copies differ: the time between
onsets, the level of each copy, and its pitch. The main control is the
event rate, the inverse of the mean interval. At one event a second, each
chirp stands alone (an estimate, from the overlaps in E3, not a listening
test). At a thousand a second, the events overlap thirty deep and the sum
is close to Gaussian noise with the chirp's spectrum (E2, E4).

This makes the usual definition of a sound texture concrete. McDermott &
Simoncelli (2011) describe textures as sounds made of many similar events
superimposed, whose statistics stay the same over time, and sonore's
Sound textures page opens with the same definition. With identical events
at Poisson times, the process is stationary at every rate and its
long-term spectrum does not change with the rate (E1). What changes is how
many events overlap (E3), and so how far the sound's higher-order
statistics, such as envelope variance, sit from those of Gaussian noise
(E2, E4). The level and pitch distributions then move the sound in other
directions: spread in level keeps it sparse for longer (E6), spread in
pitch widens its spectrum, and regular timing puts the rate into the
envelope (E5, E7). In the field the Poisson case is called shot noise
(Rice, 1944); in music the whole family is granular synthesis.

The idea started from bird chirps and from something to play for Cho's
cats. The sonore version is about the idea only: a stimulus and a gallery
demonstration. This document sets out the model, where it fits in sonore,
and the decisions for Cho. No code yet.

Status: all decisions settled by Cho, 2026-10-13. D1 and D7 were accepted
on the first draft. Cho then asked for timing, level and pitch each to be
a distribution the caller gives, and for a name that does not tie the
source to Poisson timing; D2 to D6 were rewritten for that, and D2, D3, D5
and D6 accepted as recommended. For D4 Cho chose variable rates now (the
rate track in the design and E8). Ready to build.

## How the claims are verified

- **[check]**: printed by `tools/check_event_texture_claims.py`. It uses
  NumPy and SciPy only, shares no code with sonore, and runs in about 45 s.
  The event is a 30 ms sweep from 2 to 6 kHz (exponential in frequency)
  under a Hann window, at 44.1 kHz. Each rate is 4 seeds of 60 s.
- **[math]**: follows in one line from the formulas shown.
- **[estimate]**: a judgment, labeled as such, not measured.

## The model

Onset times t_k are drawn by adding up intervals drawn independently from
the timing distribution (a renewal process). Each event k also gets an
amplitude a_k, from a level in dB drawn from the level distribution, and a
pitch ratio r_k, from a transposition in cents drawn from the pitch
distribution. The sound is

    x(t) = sum over k of a_k h_k(t - t_k)

where h_k is the event h transposed by r_k.

**A rate that changes over time.** A rate track `rate_scale(t)` (a number,
or `(times, values)` like sonore's other tracks) multiplies the rate the
interval distribution gives. It works by warping time. Onsets are first
drawn on a warped clock, where the intervals keep their distribution, and
each is then mapped back to real time through the inverse of
`Lambda(t) = integral from 0 to t of rate_scale`. Where the track is 2,
events come twice as often; where it is 1, nothing changes. For
exponential intervals this gives a Poisson process whose rate at each
moment is the base rate times the track (E8). For regular intervals it
gives a train whose spacing follows the track smoothly, as a pulse train
follows an F0 contour.

**Where convolution comes in.** If every event has the same pitch, h_k = h
for all k, and x is a train of impulses of heights a_k at the onsets,
convolved with h. That is convolution in the ordinary mathematical sense:
`(p * h)(t) = integral p(s) h(t - s) ds`, and convolving h with an impulse
of height a at time t_k gives exactly a copy of h, scaled by a, starting at
t_k. `Sound.convolve` computes this full linear convolution (with
`scipy.signal.fftconvolve`). Once pitches differ, each copy is a different
waveform, so x is no longer one convolution: the copies are added one by
one (overlap-add). The design uses the single convolution when it applies,
because it is faster, and the sum otherwise. Both compute the formula
above.

**Exponential intervals are the Poisson case.** Then Campbell's theorem
gives the statistics: the n-th cumulant of x is
`rate * E[a^n] * integral(h^n)` (with fixed pitch). Two consequences:

- The power spectrum is `rate * E[a^2] * |H(f)|^2`. Scaled to equal power,
  every rate has the same expected spectrum (E1). With pitch spread, the
  spectrum becomes the average of the transposed events' spectra *[math]*,
  so pitch spread does change it.
- The excess kurtosis is `E[a^4] / E[a^2]^2 * integral(h^4) /
  (rate * integral(h^2)^2)`. It falls as 1 / rate, toward Gaussian noise's
  0 (E2), and spread in level multiplies it (E6).

## Claims

- **E1 [check].** Identical events at Poisson times, scaled to equal
  expected power: the Welch spectrum over 2.5-5 kHz is within 0.5 dB of
  `|H(f)|^2` at every rate from 1 to 1000 per second (mean differences from
  -0.30 dB at 1/s to 0.00 dB at 1000/s).
- **E2 [check].** Excess kurtosis of the waveform, measured and predicted:
  106 and 97 at 1/s (about 240 events in all, so the estimate is noisy),
  9.8 and 9.7 at 10/s, 1.00 and 0.97 at 100/s, 0.11 and 0.10 at 1000/s.
- **E3 [math].** For Poisson timing, an event of length T overlaps another
  with probability `1 - exp(-2 rate T)`, and on average `rate T` events
  sound at once. For the 30 ms chirp: 6% and 0.03 at 1/s, 45% and 0.3 at
  10/s, 84% and 0.9 at 30/s, all of them and 3 at 100/s, all and 30 at
  1000/s. Events start to overlap more often than not near
  `rate = 1 / (3 T)`, so a longer event gets there at a lower rate.
- **E4 [check].** The envelope variance over the squared mean in one band
  (2.5-5 kHz, Hilbert envelope) is the texture model's envelope variance
  statistic (`env_var`, Eq. 2 of McDermott & Simoncelli, 2011). It is 62
  at 1/s, 5.6 at 10/s, 0.52 at 100/s, 0.32 at 300/s and 0.29 at 1000/s.
  Gaussian noise gives 0.27, the Rayleigh value `4 / pi - 1`.
- **E5 [check].** Regular intervals instead, at the same 100 per second,
  give a spectrum of lines spaced 100 Hz apart: the peak in 2.5-5 kHz
  stands 112 dB above the median, against 3.8 dB for Poisson timing.
- **E6 [check].** Levels normal in dB with standard deviation sd multiply
  the excess kurtosis by `exp(4 s^2)`, with `s = sd ln(10) / 20` *[math]*.
  At 100/s: 1.62 measured against 1.57 predicted for 3 dB (a factor of
  1.6), 6.2 against 6.6 for 6 dB (a factor of 6.7). So a 6 dB spread needs
  about 6.7 times the rate to come as close to Gaussian by this measure.
- **E7 [check].** Gamma-distributed intervals with mean 10 ms go from
  Poisson (shape 1) toward regular as the shape grows. The fine structure's
  lines of E5 vanish with any of this jitter: in 2.5-5 kHz the spectrum's
  peak over its median stays between 3.8 and 4.0 dB for shapes 1 to 64.
  The regularity shows in the envelope instead. The band envelope's
  spectrum peaks at 100 Hz by 0.9 dB over its median for shape 1, 2.4 dB
  for 4, 11.7 dB for 16, 22 dB for 64, and 94.5 dB for exactly regular
  intervals. The timing distribution is a second axis, from texture toward
  a periodic sound such as a pulse train; a helicopter's blade passes are
  nearer the regular end (an inference from how rotors work, not measured
  here).

- **E8 [check].** A rate rising exponentially from 1 to 1000 per second
  over 30 s, drawn by the time warp with exponential intervals, 400 draws,
  counted in 1 s windows (expected counts from 1.1 to 893). The mean count
  over the integrated rate is 0.94 to 1.05 across the windows, and the
  variance over the mean (1 for Poisson counts) is 0.81 to 1.19. The
  widest departures are in the first windows, where about one event is
  expected and 400 draws leave sampling errors of about 5% on the mean and
  7% on the ratio.

## Design

    from scipy import stats

    chorus = so.event_train(chirp, 4.0, stats.expon(scale=1 / 30), level_db=stats.norm(0, 3),
                            pitch_cents=stats.norm(0, 100), rng=1)
    sweep = so.event_train(chirp, 30.0, stats.expon(scale=1.0), rate_scale=([0, 30], [1, 1000]), rng=2)

`event_train(event, duration, intervals, *, rate_scale=1.0, level_db=0.0,
pitch_cents=0.0, lead=0.0, rng=None)` returns a `Sound` at the event's
sampling rate. (In the sweep, the track rises linearly between its two
points, as all of sonore's tracks do.)

1. **Distributions.** `intervals` (seconds), `level_db` (dB, relative to
   the event as given) and `pitch_cents` (cents) each take either a
   number, which every event shares, or a SciPy frozen distribution
   (anything with `.rvs(size=, random_state=)`). `stats.expon(scale=1/30)`
   is Poisson timing at 30 per second, `1/30` is regular timing at the same
   rate, and `stats.gamma(16, scale=1/(30*16))` lies between (E7). SciPy is
   already a dependency of sonore.
2. **Onsets.** Intervals are drawn and summed until they pass
   `duration + lead`; the first onset is at `-lead` plus one interval, and
   each is mapped through the rate track's warp (above) and rounded to the
   nearest sample (at most 11 µs off at 44.1 kHz,
   far below the chirp's 30 ms *[math]*). With `lead` set to the event's
   length, an event already under way at t = 0 is heard. For Poisson
   timing that makes the density the same at the start as anywhere else;
   for other renewal processes the start is only approximately stationary
   *[estimate]*. The sound is `duration` long; events running past the
   end are cut.
3. **Levels.** Each event's amplitude is `10^(level_db / 20)` times the
   event as given.
4. **Pitch.** Each event is transposed by resampling: played back
   `2^(cents / 1200)` times faster, so it gets shorter as it gets higher,
   as a tape played fast does (D5). With every pitch the same, the
   weighted impulse train is convolved with the one event; otherwise the
   transposed copies are added one by one.
5. **The overall level** is left to the caller: the sum's power grows with
   the rate (`rate * E[a^2]`, E1), and the gallery brings every sound to
   the same RMS, as it does all its sounds. (This was the first draft's
   D3; Cho's per-event level distribution is the different thing in step
   3.)

Size: one function of about 60 lines and a test file *[estimate]*. Cost
with pitch spread: one resampling per event, so a minute at 1000/s is
60 000 short resamplings; if that is slow, pitches can be rounded to the
nearest cent and each distinct transposition made once *[estimate]*.

Tests: onset counts and intervals against the given distribution over many
draws; counts under a rate track against its integral (E8);
reproducibility from `rng`; a constant interval giving exactly regular
onsets; E1, E2 and E6 on the built function within stated tolerances; and
the single-convolution and one-by-one paths agreeing when every pitch is
the same.

Texture code is not touched, so texture synthesis stays bit-for-bit
identical.

## Decisions

- **D1. Where it lives.** Its own file, `sources/events.py`, like
  `gaussian_spectrogram`. *Accepted by Cho 2026-10-13.*
- **D2. Shape of the API.** (a) One function with the three distributions
  as arguments, as above. (b) Two steps: a function that draws onsets,
  levels and pitches as arrays, and a second that renders them, so a
  caller can edit the draws or supply measured onset times in between.
  *Recommended: (a) now. (b) is easy to split out later, because the
  rendering step is internal to (a) anyway.* Accepted by Cho 2026-10-13.
- **D3. How a distribution is given.** (a) A number or a SciPy frozen
  distribution, as above. (b) Also any callable `(rng, n) -> array`, for
  distributions SciPy does not have (an empirical one, say). (c) sonore's
  own short list of names and parameters. *Recommended: (a), adding (b)
  only when a case needs it.* Accepted by Cho 2026-10-13.
- **D4. Timing.** (a) A distribution of intervals between onsets, as above
  (a renewal process: Poisson, regular and gamma all fit). (b) Also
  processes whose rate changes over time (a rate that swells and fades),
  which intervals alone cannot express. *Recommended: (a) now; (b) later if
  wanted, as a rate track like sonore's other tracks.* Decided by Cho
  2026-10-13: both now. The rate track is `rate_scale`, by time warping
  (the model and E8). The level and pitch distributions stay fixed over
  time; tracks for them would be a later proposal.
- **D5. How pitch is changed.** The first draft's D5 (vary each event?) is
  answered by Cho: level and pitch both get distributions. What remains is
  the method. (a) Resampling: pitch and length change together, exactly,
  with no new artifacts. (b) A phase-vocoder shift that keeps the length:
  sonore has one, but on a 30 ms event it smears the onset
  *[estimate]*. (c) An event given as a function of the pitch ratio,
  `event(ratio) -> Sound`, so a synthetic chirp is rebuilt at the new
  pitch with its length kept, exactly. *Recommended: (a) by default, and
  accept (c) as well, since a chirp made by sonore can be rebuilt exactly
  and a recording cannot.* Accepted by Cho 2026-10-13.
- **D6. Name.** `event_train` (like `pulse_train`, which is its regular
  special case with an impulse for the event), `event_sequence`, or
  `shot_noise` (the textbook name, but it usually means Poisson timing).
  *Recommended: `event_train`.* Accepted by Cho 2026-10-13.
- **D7. Gallery.** A section on the Sound textures page, "From single
  events to a texture": one event, then a ladder of rates (1, 3, 10, 30,
  100, 300 and 1000 per second) as audio with spectrograms, and the
  measured `env_var` in one band against rate as a one-way view (E4).
  Then the other two axes in a few sounds: level spread at one rate (E6),
  pitch spread, timing from Poisson to regular (E7), and one sound whose
  rate sweeps from sparse to dense (E8). A live slider fits
  sonore-sketch later. *Accepted by Cho 2026-10-13 (the section); the
  extra sounds for the new axes are proposed here.*

## What this document does not claim

- **No realism.** The chirp is a sweep under a window, not a model of
  birdsong. The chorus is a statistical construction, not a recording or a
  rendering of birds.
- **No perception.** sonore has no listener models, and nothing here was
  listened to under controlled conditions. The numbers say how much events
  overlap and how far a statistic is from noise's; they do not say at what
  rate a listener hears a texture. The 1/s and 1000/s descriptions in the
  first paragraph are estimates.
- **Nothing about animals.** No claim that cats or any other animal
  respond to these sounds, or that they are suitable for animal
  enrichment, and nothing about how loudspeakers reproduce them.
- **Little reading.** Campbell's theorem and the shot-noise results are
  checked numerically above (E1, E2, E6), not taken from a source read for
  this document. Rice (1944) and Roads (2001) are cited for the names only.

## References

- McDermott, J. H., & Simoncelli, E. P. (2011). Sound texture perception
  via statistics of the auditory periphery: Evidence from sound
  synthesis. *Neuron*, 71(5), 926–940. doi:10.1016/j.neuron.2011.06.032
- Rice, S. O. (1944). Mathematical analysis of random noise. *Bell System
  Technical Journal*, 23(3), 282–332.
- Roads, C. (2001). *Microsound*. MIT Press.
