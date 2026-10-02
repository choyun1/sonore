# A Klatt-style formant synthesizer

What sonore would need to synthesize speech from formant parameters in the
manner of Klatt (1980): voiced and noise sources, formant resonators in
cascade and parallel, radiation, and parameter tracks. It sets out what
sonore already has, what is missing, how it relates to the WORLD-style
synthesis in roadmap item 2, and an order for doing it.

Status: D1–D6 accepted by Cho 2026-10-02, all as recommended. Built as
`so.resonator`, `so.antiresonator` and `so.klatt_synthesize` (see "As
built" below), with the gallery page `docs/gallery/seeing/formants.py` (Formant synthesis).

## Why

Cho's observation: once the time-varying harmonic source
(`harmonic-source.md`) and the F0 tracker (`f0.md`) exist, sonore is close
to a Klatt-like synthesizer, and mostly lacks the voiceless parts. That is
right, and the gap is smaller than it looks: the voiced source is the
harmonic source with Klatt's glottal spectrum as its amplitudes (C4), and the
rest is one small time-varying filter (C1, C3), a noise source (C5), and a
table of parameter tracks.

A formant synthesizer suits sonore's purpose. Every control is something a
reader can name and see on a spectrogram (F1, a bandwidth, the frication
level), so it teaches the source-filter model directly; and it is the classic
way to make speech stimuli whose acoustics are set exactly, such as vowel and
/ba/–/da/–/ga/ continua. Neural vocoders sound better but have no such
controls; they are out of scope (see "Newer parametric models" below).

## How the claims are verified

As in the other design documents, each claim is numbered and tagged:

- **[paper]**: read from Klatt (1980), quoted or paraphrased.
- **[check]**: a number printed by `tools/check_klatt_claims.py`. The script
  uses only NumPy and SciPy, shares no code with sonore, and holds a small
  prototype of the pieces described here. It runs in about a second. The
  numbers come from NumPy 2.4.6 and SciPy 1.17.1.

## Klatt (1980) in brief [paper]

- 39 control parameters, about 20 of them varied in a typical utterance,
  updated every 5 ms; 10 kHz sampling.
- Each formant is a second-order digital resonator
  y[n] = A x[n] + B y[n−1] + C y[n−2], with C = −exp(−2π BW T),
  B = 2 exp(−π BW T) cos(2π F T) and A = 1 − B − C. An antiresonator (for
  the nasal zero) inverts it: A′ = 1/A, B′ = −B/A, C′ = −C/A, applied to the
  input.
- The voicing source is an impulse train at F0 shaped by low-pass
  resonators: RGP (F = 0, BW = 100 Hz), described as falling about
  12 dB/octave above 50 Hz, an antiresonator RGZ, and RGS for
  quasi-sinusoidal voicing (AVS).
- The noise source is a pseudo-random Gaussian generator, low-pass filtered
  by y[n] = x[n] + y[n−1]. While voicing is on, the noise is amplitude
  modulated by a square wave at F0, 50% deep.
- Aspiration (AH) goes through the cascade branch with voicing; frication
  (AF) goes through a parallel branch of six resonators with their own
  amplitudes (A2–A6) and a bypass (AB). Parallel formant outputs are
  combined with alternating signs, and a first difference removes low
  frequencies from the higher formants' input.
- Radiation is a first difference, p[n] = u[n] − u[n−1].
- The paper notes that sudden large parameter changes can cause clicks, and
  discusses quantizing the pitch period to whole samples.
- Parameters for copy synthesis were set by matching a natural utterance's
  spectrograms and spectra; the synthesizer does no analysis itself.

KLSYN88 (Klatt & Klatt, 1990) adds a better glottal source (KLGLOTT88,
with open quotient and spectral tilt controls), flutter and other voice
quality controls; these are from that paper's description and were not
re-read for this document.

## What sonore has, and what is missing

| Klatt piece | sonore today | Missing |
|---|---|---|
| Voiced source, F0 track | `so.harmonic_complex` on an F0 contour (landed as PR #45 in place of the proposed `harmonic_source`), `so.f0_track` | Klatt's glottal spectrum as amplitudes: a one-line callable (C4) |
| Glottal pulse shape, voice quality | nothing | KLGLOTT88 or LF; later (D4) |
| Aspiration and frication noise | `so.gaussian_noise` (fixed level, optional tilt) | Noise with an amplitude track, F0-synchronous modulation when voiced (C5) |
| Formant resonators | Fixed filters only (`butter_filter`, `bandpass`); the cepstrum checker builds a fixed all-pole /a/ by hand | A resonator and antiresonator whose F and BW follow tracks (C1, C3) |
| Cascade and parallel branches | nothing | Assembly; alternating signs in the parallel branch (C2) |
| Radiation | nothing named | A first difference (C5, C6) |
| Parameter tracks | `F0Track` (proposed) | A table of named tracks at 5 ms time windows |
| Voicing gate | in `harmonic_complex` (5 ms Hann ramps) | Reused for AV |

## Claims

**C1. Klatt's resonator is exactly the filter it claims to be.** [check] At
16 kHz, F = 500 Hz, BW = 60 Hz has gain 1 at 0 Hz, its peak 0.88 Hz from F
and its −3 dB width 0.18 Hz from BW; F = 3500, BW = 200 gives 1, 0.38 Hz
and 0.07 Hz. The formulas carry over unchanged from 10 kHz to any sampling
rate, so sonore need not fix one.

**C2. The parallel branch matches the cascade only with alternating signs.**
[check] Five /a/ formants, each parallel resonator's gain set to the
cascade's gain at its peak: summed with the same sign, the parallel response
differs from the cascade by up to 15.5 dB between 100 and 4000 Hz (spurious
zeros between formants); with alternating signs, by 0.15 dB. This is why
Klatt alternates them, and why the parallel branch is reserved for
frication, where formant levels are set one by one.

**C3. Updating every 5 ms is clean if the filter state is carried.**
[check] A /da/-like transition (F1 300 → 700 Hz and F2 1700 → 1200 Hz over
40 ms) on a 120 Hz voiced source, first 60 ms: with coefficients held for
each 5 ms time window and the state carried across updates, the power above 5 kHz
is −95 dB re total; interpolating coefficients every sample gives −108 dB.
Restarting the filter at each update instead gives −29 dB: clicks every
5 ms. So Klatt's update, held for each time window, is enough for listening, and per-sample
interpolation is a cheap refinement (D2).

**C4. Klatt's voiced source is the harmonic source with RGP's spectrum.**
[check] RGP falls 11.6 dB/octave between 400 and 3200 Hz. At 125 Hz and
16 kHz (128 samples per period), an impulse train through RGP and the
harmonic source with amplitudes |RGP(k F0)| have the same harmonic levels
to 9e-12 dB (harmonics 1–49); only the phases differ. The harmonic source
is better where the period is not a whole number of samples: at 230 Hz and
16 kHz, whole-sample impulses make each period 69 or 70 samples, an error
of up to 0.81%, which the paper itself flags; the accumulated phase of the
harmonic source has none. So sonore needs no impulse-train source.

**C5. The noise path is flat after radiation.** [check] Klatt's integrator
(here slightly leaky, 0.99, to stay bounded) makes white noise fall 5.7 dB
from the 1–2 kHz octave to the 2–4 kHz octave; the radiation difference
brings it back to 0.02 dB. While voiced, 50% square-wave modulation at F0
puts a line at F0 in the noise's power envelope 43 dB above its
neighbours: the noise carries the pitch, as breathy voice does.

**C6. Source and filter separate exactly.** [check] A 100 Hz harmonic
source with RGP amplitudes, through five cascade /a/ formants and the
radiation difference: harmonics 1–39 equal source × formants × radiation to
1e-11 dB. A reader can change the source and the filter independently and
predict the result from the product, which is the lesson the gallery page
would teach.

## Proposed design

Layers follow `layout.md`:

- **signals**: `resonator(sound, f, bw)` and `antiresonator(...)`, Klatt's
  difference equations with F and BW given as a number, a per-sample array,
  or a 5 ms track (held or interpolated, D2). Plain arrays in and out. A
  noise source with an amplitude track and optional F0-synchronous
  modulation, beside `harmonic_complex`.
- **stimuli**: `klatt_synthesize(params, fs)` assembling the cascade
  (voicing and aspiration), the parallel branch (frication, alternating
  signs, first difference for the upper formants, bypass) and radiation,
  from a parameter table: a mapping of Klatt's names (F0, AV, AH, AF, F1–F6,
  B1–B6, A2–A6, AB, FNP, FNZ…) to tracks at 5 ms time windows, with defaults for
  everything not given (D3). Helpers for the classic cases: a steady vowel
  from (F1, F2, F3), and a two-endpoint continuum.
- **gallery**: a "Formant synthesis" page: one vowel built up step by step
  (source, each formant, radiation, with spectra), the vowel triangle, a
  /ba/–/da/–/ga/ continuum, a fricative, breathy voice; each with its
  parameter tracks drawn over its spectrogram.

### Relation to WORLD-style synthesis (roadmap item 2)

Both are pulse-plus-noise source-filter synthesizers and share three
pieces: the harmonic source, the voicing gate and the noise source. They
differ in the filter. WORLD's is a smooth spectral envelope measured from a
recording (CheapTrick) and an aperiodicity measured with it; Klatt's is a
handful of formant numbers set by hand or by rule. So Klatt synthesis needs
no analysis at all, and can be built and checked first. Copy synthesis of a
recording with Klatt (fitting formant tracks to it) would need a formant
tracker, which sonore does not have; it is not part of this proposal (D6).

### Newer parametric models

Cho asked about newer versions of the idea. The ones that keep named,
interpretable controls:

- **KLSYN88** (Klatt & Klatt, 1990): Klatt's own update; a glottal flow
  model with open quotient and spectral tilt for voice quality.
- **LF model** (Fant, Liljencrants & Lin, 1985): the most widely used
  parametric glottal flow derivative; four parameters per period. Either
  this or KLGLOTT88 could replace RGP (D4). Proposed in
  `glottal-source.md`, which also records what of the citation could be
  checked.
- **KlattGrid** (Weenink, 2009): Praat's reimplementation, with each
  parameter a tier of time–value points rather than fixed 5 ms time windows. That
  form fits sonore, whose tracks are already functions of time (D3).
- **DDSP** (Engel, Hantrakul, Gu & Roberts, ICLR 2020): a harmonic
  oscillator plus filtered noise, the same sound model as the harmonic
  source plus a noise source, with a neural network choosing the controls.
  Its novelty is the learned control, not the synthesizer, so sonore can
  offer the synthesizer and leave the estimation out; keeping the code in
  plain array operations would keep that door open.

Sample-by-sample neural vocoders have no such controls and stay out.

## Order

1. Land the harmonic source (PR #45, done: `harmonic_complex` on a contour).
2. `resonator` and `antiresonator` in signals, with tests against C1 and C3.
3. The noise source with amplitude tracks and F0-synchronous modulation.
4. `klatt_synthesize` with the parameter table, cascade and parallel.
5. The gallery page.
6. Later: KLGLOTT88 or LF voice quality (D4), then the WORLD envelope and
   aperiodicity work, which reuses steps 1 and 3.

Steps 2–5 do not depend on the F0 tracker (Klatt's F0 is a parameter set
by hand), only on the harmonic source.

## Decisions

All six were accepted as recommended (Cho, 2026-10-02).

- **D1.** Do it, as a part of roadmap item 2 placed before the WORLD
  envelope work (recommended), or after it?
- **D2.** Hold coefficients for 5 ms time windows as Klatt did, or interpolate
  per sample (recommended: per sample, C3 shows it costs nothing in sound
  and avoids a fixed hop in the API)?
- **D3.** Parameter tracks as Klatt-style 5 ms tables, or as time–value
  points like KlattGrid (recommended: time–value points, sampled to the
  sample rate, which also accept a 5 ms table)?
- **D4.** Glottal source: RGP-shaped harmonics first, with KLGLOTT88 or LF
  later (recommended), or a pulse shape from the start?
- **D5.** Keep Klatt's parameter names (F1, B1, AV, AF…) as the public
  names (recommended: they are what the literature and Praat use), or
  descriptive names?
- **D6.** Leave formant tracking (copy synthesis) out for now
  (recommended)?

## As built

What changed from the proposal above while writing the code:

- `resonator` and `antiresonator` live in `signals.processing` and take a
  `Sound`, like the other filters there. A track is a `(times, values)`
  pair, interpolated to every sample (D2, D3); constant values run through
  `scipy.signal.lfilter`, changing ones through a per-sample loop (about
  0.06 s per second of sound at 44.1 kHz).
- The noise source is not a public function: it is three lines inside
  `klatt_synthesize` (white Gaussian noise, the F0-synchronous modulation),
  and can move to `signals` when the WORLD work needs it.
- Radiation is applied to the voiced source before the cascade, and the
  noises are left white, instead of integrating the noise and differencing
  everything at the end. With fixed formants the two are the same filter
  (the difference commutes with them, C6); this way no integrator can
  drift.
- Levels: at 60 dB every source has RMS 1 as it leaves the lips, before
  the formants, so equal values mean equal source levels (aspiration and
  voicing at 60 dB come out within 3 dB of each other through an /a/).
  Each parallel formant is scaled to unit gain at its own frequency, so
  `A1`–`A6` are peak levels. The output is normalized to RMS 1, like the
  generators.
- A default formant at or above Nyquist is dropped (F5 below 9 kHz); one
  given explicitly raises an error.
- Klatt's quasi-sinusoidal voicing (AVS, RGS) and RGZ are left out (D4).

`tests/stimuli/test_klatt.py` checks the vowel against source × formants ×
radiation (C6, to 1e-6 dB), the parallel levels, the source calibration,
and the continuum; `tests/signals/test_processing.py` checks C1 and C3 on the
library resonator.

## Listening examples

`/mnt/project-files/notes/klatt/make_examples.py` (project files, not the
repository) uses the checker's prototype to write six short 16 kHz files
beside itself: /a/, /i/, /u/, a breathy /a/ (aspiration 12 dB below
voicing), /da/ (burst, then F1 rising and F2 falling over 60 ms) and /sa/
(150 ms of frication through high parallel formants, then /a/). Parameters
were set by hand and are rough. They are not claims.

## References

- Klatt, D. H. (1980). Software for a cascade/parallel formant synthesizer.
  *JASA* 67(3), 971–995. doi:10.1121/1.383940.
- Klatt, D. H., & Klatt, L. C. (1990). Analysis, synthesis, and perception
  of voice quality variations among female and male talkers. *JASA* 87.
  doi:10.1121/1.398894.
- Fant, G., Liljencrants, J., & Lin, Q. (1985). A four-parameter model of
  glottal flow. *STL-QPSR*.
- Weenink, D. (2009). The KlattGrid speech synthesizer. *Interspeech 2009*.
- Engel, J., Hantrakul, L., Gu, C., & Roberts, A. (2020). DDSP:
  Differentiable digital signal processing. *ICLR 2020*.
