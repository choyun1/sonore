# From single events to a texture

The idea is one control that takes a sound from a single event, through
sparse events, to a dense texture. The event's shape stays fixed: one short
chirp, say. Only the statistics of when it happens change. At one event a
second, each chirp is heard on its own (an estimate, from the overlaps in
E3, not a listening test). At a thousand a second, the events overlap
thirty deep and the sum is close to Gaussian noise with the chirp's
spectrum (E2, E4).

This makes the usual definition of a sound texture concrete. McDermott &
Simoncelli (2011) describe textures as sounds made of many similar events
superimposed, whose statistics stay the same over time, and sonore's
Sound textures page opens with the same definition. The process below is
stationary at every rate, and its long-term spectrum does not change with
the rate (E1). What changes is how many events overlap (E3), and so how
far the sound's higher-order statistics, such as envelope variance, sit
from those of Gaussian noise (E2, E4). In the field this process is called
shot noise (Rice, 1944); in music it is granular synthesis with a single
grain.

The idea started from bird chirps and from something to play for Cho's
cats. The sonore version is about the idea only: a stimulus and a
gallery demonstration. This document sets out the model, where it fits in
sonore, and the decisions for Cho. No code yet.

Status: proposed, waiting on Cho.

## How the claims are verified

- **[check]**: printed by `tools/check_event_texture_claims.py`. It uses
  NumPy and SciPy only, shares no code with sonore, and runs in about 30 s.
  The event is a 30 ms sweep from 2 to 6 kHz (exponential in frequency)
  under a Hann window, at 44.1 kHz. Each rate is 4 seeds of 60 s.
- **[math]**: follows in one line from the formulas shown.
- **[estimate]**: a judgment, labeled as such, not measured.

## The model

Onset times t_k are drawn from a Poisson process with rate `rate` events
per second. The sound is

    x(t) = sum over k of h(t - t_k)

for an event h. Written another way, x is a train of unit impulses at the
onsets, convolved with h. `Sound.convolve` already does the second step, so
the only new piece is the impulse train.

Campbell's theorem gives the statistics. The n-th cumulant of x is
`rate * integral(h^n)`. Two consequences carry the whole design:

- The power spectrum is `rate * |H(f)|^2`, where H is the event's
  spectrum. Scaled to equal power, every rate has the same expected
  spectrum.
- The excess kurtosis is `integral(h^4) / (rate * integral(h^2)^2)`. It
  falls as 1 / rate, toward Gaussian noise's 0.

So the slider changes nothing a long-term spectrum can see. It moves the
sound only along the statistics that the texture model measures beyond the
spectrum.

## Claims

- **E1 [check].** At equal expected power, the Welch spectrum over
  2.5-5 kHz is within 0.5 dB of `|H(f)|^2` at every rate from 1 to 1000
  per second (mean differences from -0.30 dB at 1/s to 0.00 dB at
  1000/s). The rate changes the spectrum only through the scaling.
- **E2 [check].** Excess kurtosis of the waveform, measured and predicted:
  106 and 97 at 1/s (about 240 events in all, so the estimate is noisy),
  9.8 and 9.7 at 10/s, 1.00 and 0.97 at 100/s, 0.11 and 0.10 at 1000/s.
- **E3 [math].** An event of length T overlaps another with probability
  `1 - exp(-2 rate T)`, and on average `rate T` events sound at once. For
  the 30 ms chirp: 6% and 0.03 at 1/s, 45% and 0.3 at 10/s, 84% and 0.9
  at 30/s, all of them and 3 at 100/s, all and 30 at 1000/s. The
  crossover, where events start to overlap more often than not, sits near
  `rate = 1 / (3 T)`, so a longer event reaches it at a lower rate.
- **E4 [check].** The envelope variance over the squared mean in one band
  (2.5-5 kHz, Hilbert envelope) is the texture model's envelope variance
  statistic (`env_var`, Eq. 2 of McDermott & Simoncelli, 2011). It is 62
  at 1/s, 5.6 at 10/s, 0.52 at 100/s, 0.32 at 300/s and 0.29 at 1000/s.
  Gaussian noise gives 0.27, the Rayleigh value `4 / pi - 1`. So by 1000/s
  this statistic is within 5% of noise's.
- **E5 [check].** Regular onsets instead of Poisson ones, at the same 100
  per second, give a spectrum of lines spaced 100 Hz apart: the peak in
  2.5-5 kHz stands 112 dB above the median, against 3.8 dB for the
  Poisson chorus. A regular train is a pulse train convolved with the
  event, which sonore already has (`so.pulse_train(..., bandlimited=False)`).
  It is a different sound: periodic, not a texture. A helicopter's blade
  passes are closer to this case than to the Poisson one (an inference
  from how rotors work, not measured here).

## Proposed design

    impulses = so.poisson_impulses(2.0, fs, rate=30, rng=1)
    chorus = impulses.convolve(chirp)

1. `poisson_impulses(duration, fs, rate, rng=None)` draws the number of
   events from a Poisson distribution with mean `rate * (duration + lead)`,
   places them uniformly, rounds each to the nearest sample, and returns a
   `Sound` of unit impulses (two landing on one sample add).
2. A `lead` argument (seconds, default 0) also draws onsets before t = 0,
   so that an event already under way at the start is heard. With `lead`
   set to the event's length, the density at the start of the sound is the
   same as anywhere else; the caller then trims the convolved sound to
   `duration`. A built-in trim would need the event, so it belongs to the
   one-call wrapper of D2 (b).
3. `rate=0` returns silence. A single event is not a rate; it is the event
   itself, placed with `Sound.pad`, and the gallery shows it that way.

Rounding to samples moves an onset by at most 11 µs at 44.1 kHz, far
below the chirp's 30 ms *[math]*. Size: one function of about 20 lines and
a test file *[estimate]*.

Tests: the count's mean and variance over many draws (Poisson: both equal
`rate * duration`); reproducibility from `rng`; and E1 and E2 on the
convolved sound within stated tolerances, so the claims above are also
tested against the built function.

Texture code is not touched, so texture synthesis stays bit-for-bit
identical.

## Decisions

- **D1. Where it lives.** (a) A new source in `sources`, its own file
  (`sources/events.py`), like `gaussian_spectrogram`. (b) Inside
  `sonore.texture`. The texture branch measures statistics and imposes
  them on noise by gradient descent. This process draws a sound from a
  generative model and matches nothing, so it would share no code with the
  branch. It only meets the texture model as something to measure (E4).
  *Recommended: (a).*
- **D2. Shape of the API.** (a) Only the impulse sampler; the chorus is
  `.convolve(event)`, one line the reader can see is shot noise. (b) Also
  a one-call `event_train(event, duration, rate, rng)` that wraps (a),
  draws the lead-in and trims. (c) Only (b). *Recommended: (a), adding
  (b) only if the same line turns up repeatedly.*
- **D3. Level across rates.** (a) Unit impulses, and the caller sets the
  level; the gallery brings every sound to the same RMS anyway. (b) Scale
  by `1 / sqrt(rate)` inside, so that expected power is constant and the
  slider changes only how the power is spread in time (E1). (b) hides a
  choice inside a sampler. *Recommended: (a), with (b)'s scaling shown
  in the docstring example.*
- **D4. Timing process.** (a) Poisson only. (b) Also a regularity
  control, from Poisson to periodic (a gamma or jittered process). E5
  shows that the periodic end is a different sound, already covered by
  `pulse_train`. *Recommended: (a) now; (b) is a second slider worth a
  separate proposal if wanted.*
- **D5. Variation between events.** The brief fixes the event's shape. A
  natural next step is random level or pitch per event, which moves the
  texture's statistics in other directions. *Recommended: fixed shape
  now; per-event variation left out.*
- **D6. Name.** `poisson_impulses` (says what it returns), `shot_noise`
  (the textbook name for the result, but it returns impulses under (a)),
  or `event_onsets` (returning times, not a `Sound`). *Recommended:
  `poisson_impulses`, with "shot noise" in the docstring.*
- **D7. Gallery.** (a) A section on the Sound textures page, "From single
  events to a texture", since that page already defines a texture as many
  similar events. One event, then a ladder of rates (1, 3, 10, 30, 100,
  300 and 1000 per second) as audio with spectrograms, and the measured
  `env_var` in one band against rate as a one-way view of where each
  sound sits (E4). (b) A section on the Synthetic sounds page. (c) Its own
  page in the Stimuli group. *Recommended: (a).* A live slider fits
  sonore-sketch later, once the function exists, rather than the static
  gallery.

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
  checked numerically above (E1, E2), not taken from a source read for this
  document. Rice (1944) and Roads (2001) are cited for the names only.

## References

- McDermott, J. H., & Simoncelli, E. P. (2011). Sound texture perception
  via statistics of the auditory periphery: Evidence from sound
  synthesis. *Neuron*, 71(5), 926–940. doi:10.1016/j.neuron.2011.06.032
- Rice, S. O. (1944). Mathematical analysis of random noise. *Bell System
  Technical Journal*, 23(3), 282–332.
- Roads, C. (2001). *Microsound*. MIT Press.
