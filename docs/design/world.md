# WORLD-style speech analysis and synthesis

The design of a source-filter vocoder in the manner of WORLD (Morise,
Yokomori & Ozawa, 2016): a recording is taken apart into an F0 track, a
smooth spectral envelope and an aperiodicity, each of which can be changed,
and put back together as a sound. It is roadmap item "Next 1". It builds on
`so.f0_track` (`f0.md`), the pitch-adaptive frame
(`TVGaborFrame.pitch_adaptive`, `frames.md`), `so.Cepstrum`
(`cepstrum.md`), `harmonic_complex` on an F0 contour
(`harmonic-source.md`) and the Klatt synthesizer (`klatt.md`).

Status: proposal. No library code until Cho answers D1–D8.

## Why

With the Klatt synthesizer, sonore can build speech from numbers set by
hand. The WORLD-style vocoder is the other direction: it measures those
numbers from a recording, so a reader can change one thing about a real
voice (its pitch, its formants, how breathy it is) and hear the rest stay
the same. Most of the pieces exist:

- the F0 track (`so.f0_track`), which WORLD gets from Harvest or DIO;
- the analysis window: `TVGaborFrame.pitch_adaptive` with three periods is
  CheapTrick's Hann window (`frames.md`, D15), on an exact frame;
- the cepstrum and its lifter, which CheapTrick uses on a smoothed
  spectrum (`cepstrum.md` C5 showed the plain lifter sits about 4 dB below
  the harmonic peaks and named CheapTrick as the fix);
- the voiced excitation: `harmonic_complex` on a contour, which `klatt.md`
  C4 showed is an impulse train through a filter, without the whole-sample
  period error.

Missing: the envelope estimator, an aperiodicity estimator, and the
synthesis that combines a periodic and an aperiodic part by frequency.

## How the claims are verified

As in the other design documents, each claim is numbered and tagged:

- **[source]**: read from WORLD's C++ source (github.com/mmorise/World,
  commit d625e76 of 2025-02-21, files `cheaptrick.cpp`, `d4c.cpp`,
  `synthesis.cpp`, `common.cpp`; modified BSD). The papers' full texts
  could not be read from the container (see References), so algorithm
  details come from the code, not from the papers.
- **[check]**: a number printed by `tools/check_world_claims.py`. The
  script uses only NumPy and SciPy, shares no code with sonore, and holds
  a prototype of the two estimators below. It runs in about half a minute.
  The numbers come from NumPy 2.4.6 and SciPy 1.17.1.
- **[crosscheck]**: a number printed by `tools/crosscheck_world_vocoder.py`,
  which runs WORLD itself (pyworld 0.3.5, development only) on the
  checker's test vowels and on the gallery sentence.
- **[proof]**: a short argument given here.

The test vowels are built so that their aperiodicity is known exactly:
harmonics of a steady 120 Hz F0 or a 5.5 Hz ±3% vibrato, with Klatt's
glottal low-pass, five /a/ formants and radiation, plus noise whose power
at every frequency is a set share A(f) of the total, 16 kHz.

## What WORLD does [source]

**CheapTrick (envelope).** At each frame, the sound under a Hann window
three F0 periods long, scaled to unit energy, minus its weighted mean;
the power spectrum (1024 points at 16 kHz); the power below F0 folded
back onto itself about F0/2; a moving average over 2F0/3 on the power;
then the log, a lifter sinc(F0 q) (smoothing over F0 on the log axis)
times a "recovery" lifter (1 − 2q₁) + 2q₁ cos(2π F0 q) with q₁ = −0.15,
and back. Unvoiced frames use F0 = 500 Hz.

**D4C (aperiodicity).** At each frame, a "static group delay" from two
Blackman-windowed spectra (four periods long) a quarter period either
side of the frame centre, divided by a smoothed power spectrum, smoothed
over F0/2, minus a version of itself smoothed over F0. Around
every multiple of 3 kHz up to min(15 kHz, fs/2 − 3 kHz), a Nuttall window
over a 3 kHz span of that group delay is transformed, its power sorted,
and the aperiodicity is the share of power outside the largest bins, in
dB. Then a correction: + (F0 − 100)/50 dB, capped at 0. The 0 Hz value is
fixed at −60 dB and the Nyquist value at 0 dB, and the curve between is
linear in dB. A separate test ("LoveTrain") leaves a frame fully aperiodic
when less than 85% of its power between 100 Hz and 7.9 kHz lies below
4 kHz. Values are stored as amplitudes (dB/20).

**Synthesis.** Pulse times are where the running phase of the F0 track
crosses multiples of 2π, with the fraction of a sample kept as a linear
phase shift. At each pulse, the envelope and aperiodicity are interpolated
linearly between frames; the periodic part is the minimum-phase response
of S(1 − A²) (S the envelope, A the stored amplitude ratio, so A² is a
power share), scaled by the square root of the pulse interval; the
aperiodic part is white noise as long as the pulse interval, through the
minimum-phase response of S·A². The responses are overlap-added. Unvoiced
stretches carry pulses every 1/500 s with noise only.

So WORLD's aperiodicity has a plain meaning in synthesis: **the share of
the power at each frequency that is noise rather than harmonics.** The
D4C abstract defines it the same way (a power ratio between the signal
and its aperiodic component, given per frequency band).

## Claims

**C1. CheapTrick's envelope follows the harmonic peaks' shape at a fixed
level, and the prototype is WORLD's.** [check, crosscheck] Noise-free test
vowel, harmonics below 4 kHz, 40 frame positions within one period. The
envelope sits a constant distance below each harmonic's peak in the
windowed spectrum (−3.1, −2.9, −2.7 dB at F0 100, 200, 300 Hz) and
matches the true envelope's shape to 0.41, 0.64 and 0.92 dB RMS once that
offset is removed; the plain cepstral lifter of `cepstrum.md` C5 sits
4.6–4.8 dB below, with 0.8–1.0 dB RMS shape error. The checker's
prototype matches pyworld's CheapTrick within 0.27 dB on these vowels and
within 0.09 dB on every voiced frame of the gallery sentence (wherever
WORLD's value is within 80 dB of the frame's peak; below that WORLD's
added safety noise sets the value).

**C2. Its point is that it does not flicker within a period.** [check]
Same vowels: across 40 positions within one period, CheapTrick's value at
any harmonic below 4 kHz changes by at most 0.04, 0.04 and 0.07 dB; the
plain lifter's changes by up to 5.2, 2.9 and 2.3 dB. A three-period
window still sees the period's structure; the smoothing over 2F0/3 is
what removes it. This is what makes the envelope usable frame by frame
for synthesis.

**C3. The recovery lifter lifts the peaks.** [check] At 200 Hz, the
harmonic nearest F1 sits 3.5 dB below its peak with q₁ = 0 and 2.1 dB
below with WORLD's q₁ = −0.15. It is a fixed, fitted correction, kept as
WORLD has it.

**C4. Fitting the harmonics and measuring what is left recovers a known
aperiodicity.** [check] The proposed estimator (D3): at each frame, a
weighted least-squares fit of phase-locked harmonics k·Φ(t), Φ the running
phase of the F0 track, each with a linear amplitude change, over a Hann
window four periods long; the aperiodicity of a band is the residual's
power over the signal's power there. The fit also absorbs some noise near
every harmonic, which is known exactly from the fit's projection and
divided out, cell by cell (cells two harmonic spacings wide). On the test
vowels, in four bands (0–1, 1–2, 2–4, 4–7 kHz), with A flat at −20 dB,
flat at −6 dB, or rising from −30 dB at 0 Hz to −5 dB at 8 kHz, steady and
with vibrato: every band within 1.4 dB of the truth, mostly slightly low.
Without noise it reads −247 dB (steady) and −36 dB (vibrato, the floor
left by the formants changing each harmonic's amplitude faster than a
linear term follows).

**C5. It needs F0 to about 0.1%, which `so.f0_track` provides on clean
vowels.** [check] Vibrato vowel, A rising, the track made too high by a
constant factor: 0.1% moves the 2–4 and 4–7 kHz bands by 0.6 and 0.1 dB,
0.3% by 2.9 and 4.0 dB, 1% by 18 dB. A wrong F0 drifts the high harmonics
out of phase with the fit within the window, and that reads as noise. The
tracker's refined F0 is within 0.04% (median) and 0.12% (worst frame) on
a ±6% vibrato (`f0.md`, C2), so this is enough on clean vowels; on
voices with jitter, the cycle-to-cycle irregularity will read as
aperiodicity, which is arguably right (a resynthesis from a smooth track
can only carry jitter as noise) but makes the value depend on the
tracker. A per-frame correction (rescale F0 by the factor that leaves the
least residual) was tried and not kept: the noise the many free
parameters fit moves the residual more than a 0.3% F0 change does, and it
picked factors up to 0.5% off with the exact track.

**C6. D4C at 16 kHz measures one number per frame, and does not recover
these aperiodicities.** [source, crosscheck] At 16 kHz, D4C has one
measured band (3 kHz): every voiced frame's curve is exactly the two
straight lines (in dB) from −60 dB at 0 Hz through the 3 kHz value to
0 dB at 8 kHz (largest departure 1e-14 dB). On the test vowels, as band
power shares weighted by CheapTrick's envelope:

| Test vowel | Truth, 4 bands [dB] | D4C [dB] |
|---|---|---|
| A flat −20 dB, steady | −21.2, −20.5, −19.9, −19.8 | −52.0, −47.7, −31.4, −19.1 |
| A flat −6 dB, steady | −6.9, −6.4, −5.9, −5.9 | −48.2, −42.0, −18.0, −7.9 |
| A rising, steady | −29.0, −27.1, −22.1, −16.2 | −52.0, −47.7, −31.2, −19.0 |
| no noise, steady | (−∞) | up to −23.9 |

The vibrato versions give the same picture (worst band errors −30.9,
−41.4, −23.0 dB; −22.5 dB without noise). D4C is, on the other hand,
insensitive to F0 error: with the track 0.3%, 1% and 3% high, its worst
band error stays at −23.0, −23.0 and −22.9 dB, where the harmonic residual
is off by 4.0, 18.8 and 25.5 dB. On the gallery sentence, 7.9% of the
frames Harvest calls voiced are left fully aperiodic by the LoveTrain test.
D4C was tuned to make resynthesized speech sound natural (its paper
reports listening tests), not to report the share of noise; the −60 dB
anchor says "low frequencies are periodic", which is usually true of
speech and is what makes the low bands wrong here.

**C7. Aperiodicity is a property of the source.** [proof] At a frequency
f, the periodic and aperiodic parts both pass through the same vocal-tract
filter H, so the share is A = |H|²N / (|H|²N + |H|²P) = N / (N + P): the
filter cancels. The envelope carries the filter (and the source's overall
tilt); the aperiodicity says only how the source divides its power
between harmonics and noise at each frequency. This is why the two can be
changed independently. (In a wide band the cancellation is approximate
when A varies across the band; per frequency it is exact.)

**C8. The periodic part can be built from harmonics instead of pulses.**
[proof] For a constant F0 of a whole number of samples, a pulse train
through a filter h has, over one period, the DFT H(k F0): it is the sum of
harmonics k F0 with complex gains H(k F0) (sampling in frequency is
aliasing in time). So WORLD's pulses through minimum-phase responses are
the same sound as harmonics with amplitude |H_min(k F0)| and phase
arg H_min(k F0), the identity `klatt.md` C4 checked numerically for
Klatt's source. The harmonic form also covers moving F0 with no pulse
rounding, which WORLD handles with its fractional shift.

**C9. A Klatt breathy vowel has an aperiodicity that rises with
frequency.** [proof] In Klatt's synthesizer the voiced source falls about
12 dB/octave (RGP, `klatt.md` C4) and the radiation difference adds
6 dB/octave, while the aspiration noise is flat after radiation (C5
there); both go through the same cascade. So with AV and AH fixed, the
noise share rises about 6 dB per octave until noise dominates, and the
true aperiodicity of a sonore Klatt vowel can be written down from its
parameters. This is the bridge for the gallery page below.

## Proposed design

Layers follow `layout.md`.

**analysis/vocoder.py** (name open, D7):

- `SpectralEnvelope`: CheapTrick on the coefficients of a pitch-adaptive
  frame (D1, D2). Data are power, shape `(n_channels, n_freqs,
  n_frames)`, with `.t`, `.f`, `.db`, `.plot()`. It is callable,
  `env(t, f)`, interpolating linearly in time and in dB over frequency,
  so it can be handed straight to `harmonic_complex(amplitudes=...)`.
  (`Envelope` is already the Hilbert envelope class.)
- `Aperiodicity`: the harmonic-residual estimator (D3), stored as the
  noise share in dB on the envelope's frequency grid, from per-frame cell
  values interpolated in dB (D4). Also callable, `ap(t, f)`, and with
  `.bands(edges)` to average over bands for display and tests.
- A function that builds both from a sound and an F0 track (D7).

**signals**: one generalization of `harmonic_complex`: `amplitudes(t, f)`
may return complex values, whose angle is added to each harmonic's phase
(D6). A minimum-phase response is then just an amplitude function.

**stimuli/vocoder.py**: the synthesis (D5): periodic part from
`harmonic_complex` on the F0 track with complex amplitudes
√(S(1 − A)) · e^{i·φ_min}; aperiodic part from white noise analysed with
a Gabor frame, its coefficients multiplied by √(S·A) interpolated to the
frame's times and frequencies, and synthesized (the least-squares sound
with those coefficients, `philosophy.md`); unvoiced frames noise only
(A = 1). Output RMS 1 as the generators.

### Views, and what this one drops

By `philosophy.md` ("Views may discard information"), the triple
(F0 track, envelope, aperiodicity) is a view, not a frame. It drops:

- **phase**: synthesis uses minimum phase, so the waveform within a
  period is not the original's (the glottal pulse shape's phase, the
  group delay of the vocal tract beyond its minimum-phase part);
- **the noise waveform**: only its power share by frequency is kept;
- **spectral detail finer than F0**: the envelope is smoothed over F0, so
  two harmonics' levels between envelope points are interpolated, not
  kept;
- **temporal detail within about three periods**, and cycle-to-cycle
  jitter and shimmer, which move into the aperiodicity (C5).

So a sound is recoverable only approximately, and not by search (as
Griffin–Lim or texture synthesis are) but by a model: harmonics plus
noise through a smooth filter. What is checked instead is consistency:
analysing the synthesized sound should return the envelope and
aperiodicity it was made from, within stated dB, and the library tests
will check that. The envelope itself is computed from an exact frame's
coefficients, so everything up to the smoothing is invertible.

### Aperiodicity is explained on a gallery page

**Requirement (Cho, 2026-10-02): the aperiodicity measure is to be
explained in a gallery page.** It is the least natural of the three
parameters for someone thinking in first-order source-filter terms, where
the source is either a buzz or a hiss (a voicing switch) and everything
else is the filter. Aperiodicity is a third thing: per frequency, how
the source divides its power between harmonics and noise. It is
measured on the output but is a property of the source (C7); it usually
rises with frequency, so a voice can be clearly periodic at 500 Hz and
mostly noise at 5 kHz; and in WORLD its estimator (D4C) is the most
opaque step (a sorted spectrum of a smoothed group delay, with fitted
corrections). The page will build it up from things a reader already
has:

1. Klatt's breathy /a/ (AV and AH set): its true aperiodicity drawn from
   the parameters, rising about 6 dB per octave (C9), with the voicing
   switch of the first-order model shown as the special cases A = 0 and
   A = 1.
2. One frame's spectrum: the harmonic peaks, the noise between them, and
   the residual after the harmonics are fitted and removed (C4). The
   share is read off as residual over total.
3. The estimate on the Klatt vowel against the truth, then on the
   gallery sentence as a time–frequency map beside its spectrogram.
4. Listening: the sentence resynthesized with its measured aperiodicity,
   with A = 0 everywhere (buzzy), and A = 1 (whispered), same envelope
   and F0 throughout.
5. A note on what else is called aperiodicity: D4C's value and why it
   differs (C6), and that jitter reads as noise (C5).

## Order

1. `SpectralEnvelope` with tests against the checker's numbers and a dense
   oracle; the crosscheck against pyworld stays in `tools/`.
2. `Aperiodicity` with the test vowels as tests.
3. Complex amplitudes in `harmonic_complex` (existing outputs unchanged
   bit for bit).
4. The synthesis, with the analysis–synthesis consistency tests.
5. The gallery page (with the aperiodicity explanation above), README row,
   recipe and roadmap Done entry.

Steps 1 and 3 are independent of each other and of step 2.

## Decisions

Each option is stated in its best form before the recommendation.

**D1. Envelope estimator.**
- *CheapTrick* (recommended): temporally stable within a period (C2),
  WORLD's choice, a few lines on top of sonore's frame and cepstrum, and
  matched to WORLD's own output (C1).
- *The plain cepstral lifter* sonore already has: no new code and one
  idea fewer for the reader; but 4.6 dB low and flickering by up to 5 dB
  within a period (C1, C2), so a resynthesis would be modulated at F0.
- *True envelope* (iterated cepstrum that rises to the peaks; Röbel &
  Rodet, cited from memory): passes through the harmonic peaks rather
  than a fixed distance below, without CheapTrick's fitted q₁; but
  iterative, with a convergence threshold to choose, and not what WORLD
  users compare against. A reasonable later option.
- *LPC*: an all-pole model whose poles are formants, which a reader can
  name; but biased toward harmonic peaks at high F0 and wrong for nasal
  zeros. Better as its own view later than as the vocoder's envelope.

**D2. Where the envelope's spectra come from.**
- *The coefficients of `TVGaborFrame.pitch_adaptive`* (recommended): the
  window is CheapTrick's (three-period Hann), the frame is exact, and the
  envelope becomes a view of coefficients like `Cepstrum(coefs)`. Frame
  times then follow the frame (a hop of a quarter window), not a fixed
  5 ms; the envelope interpolates to any time. Each frame's F0 is given
  by its window length, but the class takes the track too, so no
  "periods" convention is hidden in it.
- *WORLD's own windowing at the track's times*: bit-for-bit closer to
  pyworld (the frame rounds the window to a whole number of samples,
  WORLD to an even number, so the lengths can differ by one sample) and fixed 5 ms
  frames everyone knows; but a second windowing code path beside the
  frames, which is what sonore avoided for the cepstrum.

**D3. Aperiodicity estimator.**
- *Harmonic residual* (recommended): its measurement is its definition
  (fit the best periodic sound, measure what is left), so it is the one a
  gallery page can explain; within 1.4 dB on known cases (C4); reuses
  the F0 track's phase, as `harmonic_complex` does. Its weakness is real:
  it needs F0 to 0.1% (C5), so it reports jitter and tracker error as
  noise.
- *A port of D4C*: robust to F0 error (C6), WORLD's standard, tuned by
  listening tests for natural synthesis, and the number WORLD users quote.
  But at 16 kHz it measures one band, pins 0 Hz at −60 dB and Nyquist at
  0 dB, and misreads every test vowel by 23–41 dB in some band (C6);
  `d4c.cpp` is about 400 lines of fitted heuristics to explain. pyworld already gives D4C for
  comparison, so the crosscheck script keeps it in view without a port.
- *Peak-to-valley ratio* (STRAIGHT's idea; Kawahara et al., 1999): simple
  to show on a spectrum; but with a three-period Hann window a perfectly
  periodic sound already has valleys only about 12 dB below its peaks
  (`frames.md`, C16), so it needs a calibration table to mean anything.

**D4. How aperiodicity is stored.**
- *Noise share in dB on the envelope's frequency grid* (recommended):
  what the synthesis needs at every frequency; from per-frame cells two
  harmonics wide, interpolated in dB; `.bands()` averages for display.
- *Coarse fixed bands* (D4C's 3 kHz): compact and smooth; but at 16 kHz
  that is one number, and the bands do not scale with F0.
- *ERB bands*, like sonore's filterbanks: auditory resolution, which
  suits a psychoacoustics library; but the estimate's natural resolution
  is the harmonic spacing, which is coarser than an ERB at high F0 and
  finer at low; could be a `.bands()` choice rather than the storage.

**D5. Synthesis.**
- *Harmonics plus frame-filtered noise* (recommended): the periodic part
  is `harmonic_complex` on the track (exact running phase, no pulse
  rounding, the taper below Nyquist, the voicing ramps), equal to WORLD's
  pulses for a steady F0 (C8); the noise part is the least-squares sound
  of filtered coefficients on an exact frame, so its meaning is the one
  `philosophy.md` already gives.
- *WORLD's pulse-by-pulse overlap-add*: the reference algorithm,
  efficient, and what a reader of the WORLD paper expects
  (`synthesis.cpp`, about 400 lines of C++). But it adds pulse placement, fractional
  shifts, DC removal and per-pulse noise segments that sonore does not
  otherwise need, and its F0 handling would differ from
  `harmonic_complex`'s.

**D6. Phases of the harmonics.**
- *`amplitudes(t, f)` may return complex values* (recommended): one
  argument carries a filter's full response, so a minimum-phase envelope
  (or a Klatt cascade) is simply passed in; real-valued outputs keep
  today's results bit for bit, dispatching on the dtype.
- *A separate `phase_response(t, f)` argument*: more explicit; but two
  callables that must describe the same filter.
- *Keep cosine phase*: no change; but the waveform within each period
  then differs from a vocal tract's (it is maximally peaky); that it
  sounds harsher at low F0 is expected but has not been listened to.

**D7. API shape and names.**
- *Two view classes plus one synthesis function* (recommended):
  `so.SpectralEnvelope`, `so.Aperiodicity` (callable, so they plug into
  `harmonic_complex`), a convenience `so.vocoder_analyze(sound, f0=None)`
  returning both with the track (running `f0_track` when none is given),
  and `so.vocoder_synthesize(f0, envelope, aperiodicity, fs)`. A small
  frozen dataclass holding the three, with `.synthesize()`, makes "change
  one, keep the rest" a `dataclasses.replace`.
- *WORLD's names* (`cheaptrick`, `d4c`, `synthesize`): familiar to anyone
  who has used pyworld; but the aperiodicity is not D4C (D3), and sonore
  names things by what they are (`klatt.md` D5 kept Klatt's names because
  they are the literature's parameter names, not algorithm names).
- *Plain arrays*, as pyworld returns: nothing new to learn; but the axes
  (times, frequencies) and the dB-or-power convention travel separately
  and get mixed up.

**D8. Where the aperiodicity explanation goes.** The explanation itself
is a requirement (above). As a new gallery page, "Source, filter and
aperiodicity" (recommended), with the vocoder's other demos (pitch
change, formant shift, breathiness), or as a section of the Formant
synthesis page, which already has the breathy vowel (C9) but no analysis.

## References

Verified by lookup on 2026-10-02: Morise (2015) by its Crossref record;
Morise (2016) by its ScienceDirect abstract (quoted for the definition of
aperiodicity); Morise, Yokomori & Ozawa (2016) by its J-STAGE abstract
page. The full texts could not be read (the J-STAGE PDF is closed to
automated fetching and Semantic Scholar refused the request), so the
algorithm details above are from WORLD's source code, cited by commit.
Kawahara et al. (1999) was verified for `frames.md`. Röbel & Rodet (2005)
is cited from memory.

- Kawahara, H., Masuda-Katsuse, I. & de Cheveigné, A. (1999). Restructuring
  speech representations using a pitch-adaptive time-frequency smoothing
  and an instantaneous-frequency-based F0 extraction. *Speech
  Communication* 27, 187–207.
- Morise, M. (2015). CheapTrick, a spectral envelope estimator for
  high-quality speech synthesis. *Speech Communication* 67, 1–7.
  doi:10.1016/j.specom.2014.09.003.
- Morise, M. (2016). D4C, a band-aperiodicity estimator for high-quality
  speech synthesis. *Speech Communication* 84, 57–65.
  doi:10.1016/j.specom.2016.09.001.
- Morise, M., Yokomori, F. & Ozawa, K. (2016). WORLD: a vocoder-based
  high-quality speech synthesis system for real-time applications. *IEICE
  Trans. Inf. & Syst.* E99-D(7), 1877–1884. doi:10.1587/transinf.2015EDP7457.
- Röbel, A. & Rodet, X. (2005). Efficient spectral envelope estimation and
  its application to pitch shifting and envelope preservation. *Proc.
  DAFx 2005*.
- WORLD source code, github.com/mmorise/World, commit d625e76
  (2025-02-21), modified BSD licence.
