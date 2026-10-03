# WORLD-style speech analysis and synthesis

The design of a source-filter vocoder in the manner of WORLD (Morise,
Yokomori & Ozawa, 2016): a recording is taken apart into an F0 track, a
smooth spectral envelope and an aperiodicity, each of which can be changed,
and put back together as a sound. It is roadmap item "Next 1". It builds on
`so.f0_track` (`f0.md`), the pitch-adaptive frame
(`TVGaborFrame.pitch_adaptive`, `frames.md`), `so.Cepstrum`
(`cepstrum.md`) and the Klatt synthesizer (`klatt.md`).

Status: accepted. Cho accepted every recommendation on 2026-10-02. The
first draft proposed several departures from WORLD to fit sonore's own
pieces; Cho asked how to square that with reproducibility and decided:
reproduce WORLD, with no pyworld dependency ("Reproducing WORLD" below).
Steps 1–5 of "Order" are built (`sonore.views.spectral_envelope`, `sonore.views.aperiodicity`
and `sonore.views.world` since the reorganization), and the gallery page explains aperiodicity
(`docs/gallery/voice/aperiodicity.py`).

## Why

With the Klatt synthesizer, sonore can build speech from numbers set by
hand. The WORLD-style vocoder is the other direction: it measures those
numbers from a recording, so a reader can change one thing about a real
voice (its pitch, its formants, how breathy it is) and hear the rest stay
the same. WORLD is the vocoder most speech researchers know and quote, so
sonore's version is only useful for comparison if it gives WORLD's
numbers. Several pieces it needs exist already:

- the F0 track (`so.f0_track`), which WORLD gets from Harvest or DIO;
- the cepstrum and its lifter, which CheapTrick uses on a smoothed
  spectrum (`cepstrum.md` C5 showed the plain lifter sits about 4 dB below
  the harmonic peaks and named CheapTrick as the fix);
- the pitch-adaptive frame, whose three-period window is CheapTrick's
  Hann window (`frames.md`, D15), up to rounding of its length.

Missing: the envelope estimator, an aperiodicity estimator, and the
synthesis that combines a periodic and an aperiodic part by frequency.

## How the claims are verified

As in the other design documents, each claim is numbered and tagged:

- **[paper]**: read from the three papers (Morise, 2015; Morise, 2016;
  Morise, Yokomori & Ozawa, 2016), whose PDFs Cho supplied on
  2026-10-02.
- **[source]**: read from WORLD's C++ source (github.com/mmorise/World,
  commit d625e76 of 2025-02-21, files `cheaptrick.cpp`, `d4c.cpp`,
  `synthesis.cpp`, `common.cpp`, `matlabfunctions.cpp`; modified BSD).
  Where the code and the papers differ, both are given ("Papers and code"
  below).
- **[check]**: a number printed by `tools/check_world_claims.py`. The
  script uses only NumPy and SciPy and shares no code with sonore. It
  holds ports of CheapTrick, D4C and WORLD's synthesis, written from the
  source, and a prototype of the harmonic-residual aperiodicity. It runs
  in about half a minute. The numbers come from NumPy 2.4.6 and SciPy
  1.17.1.
- **[crosscheck]**: a number printed by `tools/crosscheck_world_vocoder.py`,
  which runs WORLD itself (pyworld 0.3.5, development only) on the
  checker's test vowels and on the gallery sentence, and compares the
  ports with it.
- **[proof]**: a short argument given here.

The test vowels are built so that their aperiodicity is known exactly:
harmonics of a steady 120 Hz F0 or a 5.5 Hz ±3% vibrato, with Klatt's
glottal low-pass, five /a/ formants and radiation, plus noise whose power
at every frequency is a set share A(f) of the total, 16 kHz.

## Reproducing WORLD

Cho's rule (2026-10-02): wherever sonore says WORLD, it gives WORLD's
numbers, and sonore does not depend on pyworld. In detail:

1. **One pinned reference.** WORLD's C++ code at commit d625e76 is the
   reference, run through pyworld 0.3.5. The ports in the checker were
   written from that commit and match pyworld to floating-point precision
   (C10).
2. **The code, not the paper, sets the defaults.** Where the papers and
   the code differ, the default is the code's value, because that is what
   every WORLD user gets. The paper's value can be an option, off by
   default (CheapTrick's q̃₁, D1).
3. **A different measure gets a different name.** A step named after
   WORLD (`cheaptrick`, `d4c`, the synthesis) is a faithful port. The
   harmonic-residual aperiodicity measures something else, so it is
   offered under its own name beside D4C, never in its place (D3).
4. **Every difference is written down.** As the texture code keeps
   `so.texture.DIFFERENCES_FROM_TOOLBOX`, the vocoder module keeps
   `DIFFERENCES_FROM_WORLD`: each option that departs from WORLD, and
   each thing sonore leaves out. It starts short: F0 estimation is not
   ported (the track is an argument, D7); the paper's q̃₁ is an option;
   fresh noise is an option (D8). With every option at its default, the
   list says the output is WORLD's.
5. **Tests compare against stored WORLD output.** A script in `tools/`
   (pyworld, development only, like the crosscheck) writes WORLD's
   envelope, aperiodicity and synthesized sound for a short test vowel
   and a half-second of the gallery sentence to a small file under
   `tests/`. The library tests compare against that file, so they need no
   pyworld. The tolerances are the ones C10 measures, with room for other
   platforms' floating point (to be set when the tests are written, from
   CI's numbers).
6. **Synthesis is held to WORLD, then to consistency.** The synthesis
   port must give WORLD's samples (C10). Whether analysing a synthesized
   sound returns what it was made from is then a property of WORLD
   itself, which C11 measures and the documentation states, rather than a
   standard sonore sets on its own.

Two costs come with exactness. WORLD cuts each time window itself, with its
own rounding of the window length, so the envelope cannot be computed
from `TVGaborFrame.pitch_adaptive` coefficients and stay exact (D2). And
WORLD draws its noise from its own generator, a sum of twelve xorshift
draws, which must be reproduced draw for draw: in pure Python that is
most of the D4C port's time (4.3 s for the 2.5 s gallery sentence,
against 0.14 s in pyworld). Because WORLD restarts the generator on every
call, the stream is the same each time, so the library can compute it
once and keep it (about 0.8 million values, 6.5 MB, for that sentence).
It does, many lanes of the generator at a time, so the library's D4C
takes about 0.8 s on that sentence, including the first computation of
the stream.

## What WORLD's code does [source]

**CheapTrick (envelope).** At each time window, the sound under a Hann window
three F0 periods long, scaled to unit energy, minus its weighted mean;
the power spectrum (1024 points at 16 kHz); the power below F0 folded
back onto itself about F0/2; a moving average over 2F0/3 on the power;
then the log, a lifter sinc(F0 q) (smoothing over F0 on the log axis)
times a "recovery" lifter (1 − 2q₁) + 2q₁ cos(2π F0 q) with q₁ = −0.15,
and back. Unvoiced time windows use F0 = 500 Hz.

**D4C (aperiodicity).** At each time window, a "static group delay" from two
Blackman-windowed spectra (four periods long) a quarter period either
side of the window centre, divided by a smoothed power spectrum, smoothed
over F0/2, minus a version of itself smoothed over F0. Around
every multiple of 3 kHz up to min(15 kHz, fs/2 − 3 kHz), a Nuttall window
over a 3 kHz span of that group delay is transformed, its power sorted,
and the aperiodicity is the share of power outside the largest bins, in
dB. Then a correction: + (F0 − 100)/50 dB, capped at 0. The 0 Hz value is
fixed at −60 dB and the Nyquist value at 0 dB, and the curve between is
linear in dB. A separate test ("LoveTrain") leaves a time window fully aperiodic
when less than 85% of its power between 100 Hz and 7.9 kHz lies below
4 kHz. A tiny noise (10⁻⁶ of WORLD's randn) is added to each windowed
segment to keep the divisions finite. Values are stored as amplitudes
(dB/20).

**Synthesis.** Pulse times are where the running phase of the F0 track
crosses multiples of 2π, with the fraction of a sample kept as a linear
phase shift. At each pulse, the envelope and aperiodicity are interpolated
linearly between time windows; the periodic part is the minimum-phase response
of S(1 − A²) (S the envelope, A the stored amplitude ratio, so A² is a
power share), scaled by the square root of the pulse interval, with its
DC removed; the aperiodic part is WORLD's randn noise as long as the
pulse interval, through the minimum-phase response of S·A². The responses
are overlap-added. Unvoiced stretches carry pulses every 1/500 s with
noise only. The noise generator is restarted at every call, so the same
inputs always give the same sound.

## Papers and code [paper, source]

The papers were read after the first draft of this document. Where they
and the code differ:

- **CheapTrick's recovery lifter.** The paper gives q̃₀ = 1.18 and
  q̃₁ = −0.09, "obtained" by "an exploratory evaluation"; the code uses
  q̃₁ = −0.15 (and q̃₀ = 1 − 2q̃₁ = 1.3). Both make the lifter 1 at zero
  quefrency. On the test vowels the paper's value fits the envelope's
  shape better (C3). The paper also explains the 2F0/3 smoothing as
  ensuring the power spectrum "has no zeros" before the log, with
  neighbouring harmonics' influence below 30 dB. The folding below F0 and
  the weighted-mean removal are in the code only.
- **D4C's bands and anchors.** The paper evaluates at 48 kHz with five
  centre frequencies (3, 6, 9, 12, 15 kHz) and a 6 kHz window. The
  −60 dB at 0 Hz is described as added for interpolation in the
  subjective evaluation, "on the basis of our past research (Kawahara and
  Morise, 2012)". The paper concludes that "only one estimated
  aperiodicity (3 kHz) is enough to synthesize natural speech", which is
  what the code does at 16 kHz.
- **D4C's tuning.** The Blackman window of 4T0 and "the other parameters"
  were chosen from "an exploratory experiment including an unofficial
  listening test with limited speech and few subjects", and the paper
  says they "were not optimized". It reports a bias of "around 6 dB" for
  SNRs above 5 dB, and errors within 3 dB for F0 errors of ±10%. The
  + (F0 − 100)/50 dB correction and the LoveTrain voicing test are in the
  code, not the paper.
- **D4C's own tests** use harmonics plus white or pink noise and look only
  at 3 kHz and above; none measures the bands below 3 kHz that C6 finds
  far off.
- **The WORLD paper predates D4C.** It describes PLATINUM, which extracts
  an excitation signal (the windowed waveform divided by the envelope's
  minimum-phase spectrum) instead of an aperiodicity, and says WORLD then
  "cannot manipulate the aperiodic parameter as well as" STRAIGHT. Current
  WORLD uses D4C. The paper also says that "an approximation using the
  minimum phase is inappropriate for low-pitch speech", since phase
  differences are easier to hear at low F0, and names phase modelling as
  future work.

So WORLD's aperiodicity has a plain meaning in synthesis: **the share of
the power at each frequency that is noise rather than harmonics.** The
D4C abstract defines it the same way (a power ratio between the signal
and its aperiodic component, given per frequency band).

## Claims

**C1. CheapTrick's envelope follows the harmonic peaks' shape at a fixed
level, and the port is WORLD's.** [check, crosscheck] Noise-free test
vowel, harmonics below 4 kHz, 40 window positions within one period. The
envelope sits a constant distance below each harmonic's peak in the
windowed spectrum (−3.1, −2.9, −2.7 dB at F0 100, 200, 300 Hz) and
matches the true envelope's shape to 0.41, 0.64 and 0.92 dB RMS once that
offset is removed; the plain cepstral lifter of `cepstrum.md` C5 sits
4.6–4.8 dB below, with 0.9–1.1 dB RMS shape error. The port matches
pyworld's CheapTrick within 0.0007 dB on these vowels and within
0.00003 dB on every voiced time window of the gallery sentence (wherever
WORLD's value is within 80 dB of the time window's peak; below that the
envelope is set by the floor that keeps the log finite).

**C2. Its point is that it does not flicker within a period.** [check]
Same vowels: across 40 positions within one period, CheapTrick's value at
any harmonic below 4 kHz changes by at most 0.04, 0.04 and 0.07 dB; the
plain lifter's changes by up to 5.3, 3.1 and 2.2 dB. A three-period
window still sees the period's structure; the smoothing over 2F0/3 is
what removes it. This is what makes the envelope usable at every time window
for synthesis.

**C3. The paper's recovery lifter fits the shape better than the
code's.** [check] Noise-free test vowels, harmonics below 4 kHz, RMS
shape error with the offset removed, for q̃₁ = 0 (no recovery), −0.09
(the paper) and −0.15 (the code): 0.24, 0.24, 0.41 dB at F0 100 Hz;
0.42, 0.36, 0.64 dB at 200 Hz; 0.64, 0.52, 0.92 dB at 300 Hz. At 200 Hz
the harmonic nearest F1, relative to the mean offset, is 0.63 dB low
without recovery, 0.19 dB high with the paper's value and 0.74 dB high
with the code's: the code's value overshoots the sharpest peak. C1's
numbers use the code's value. One vowel is not a tuning study; it is
enough to offer the paper's value as an option (D1).

**C4. Fitting the harmonics and measuring what is left recovers a known
aperiodicity.** [check] The harmonic-residual measure (D3): at each
time window, a weighted least-squares fit of phase-locked harmonics k·Φ(t), Φ
the running phase of the F0 track, each with a linear amplitude change,
over a Hann window four periods long; the aperiodicity of a band is the
residual's power over the signal's power there. The fit also absorbs some
noise near every harmonic, which is known exactly from the fit's
projection and divided out, cell by cell (cells two harmonic spacings
wide). On the test vowels, in four bands (0–1, 1–2, 2–4, 4–7 kHz), with A
flat at −20 dB, flat at −6 dB, or rising from −30 dB at 0 Hz to −5 dB at
8 kHz, steady and with vibrato: every band within 1.4 dB of the truth,
mostly slightly low. Without noise it reads −247 dB (steady) and −36 dB
(vibrato, the floor left by the formants changing each harmonic's
amplitude faster than a linear term follows).

**C5. It needs F0 to about 0.1%, which `so.f0_track` provides on clean
vowels.** [check] Vibrato vowel, A rising, the track made too high by a
constant factor: 0.1% moves the 2–4 and 4–7 kHz bands by 0.6 and 0.1 dB,
0.3% by 2.9 and 4.0 dB, 1% by 18 dB. A wrong F0 drifts the high harmonics
out of phase with the fit within the window, and that reads as noise. The
tracker's refined F0 is within 0.04% (median) and 0.12% (worst time window) on
a ±6% vibrato (`f0.md`, C2), so this is enough on clean vowels; on
voices with jitter, the cycle-to-cycle irregularity will read as
aperiodicity, which is arguably right (a resynthesis from a smooth track
can only carry jitter as noise) but makes the value depend on the
tracker. A per-time-window correction (rescale F0 by the factor that leaves the
least residual) was tried and not kept: the noise the many free
parameters fit moves the residual more than a 0.3% F0 change does, and it
picked factors up to 0.5% off with the exact track.

**C6. D4C at 16 kHz measures one number per time window, and does not recover
these aperiodicities.** [source, crosscheck] At 16 kHz, D4C has one
measured band (3 kHz): every voiced time window's curve is exactly the two
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
time windows Harvest calls voiced are left fully aperiodic by the LoveTrain test.
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

**C8. The periodic part could be built from harmonics instead of
pulses.** [proof] For a constant F0 of a whole number of samples, a pulse
train through a filter h has, over one period, the DFT H(k F0): it is the
sum of harmonics k F0 with complex gains H(k F0) (sampling in frequency
is aliasing in time). So WORLD's pulses through minimum-phase responses
are the same sound as harmonics with amplitude |H_min(k F0)| and phase
arg H_min(k F0), the identity `klatt.md` C4 checked numerically for
Klatt's source. With a moving F0 the two differ (WORLD rounds pulses to
samples and shifts by the fraction), which is why the harmonic route is
not WORLD's synthesis (D5).

**C9. A Klatt breathy vowel has an aperiodicity that rises with
frequency.** [proof] In Klatt's synthesizer the voiced source falls about
12 dB/octave (RGP, `klatt.md` C4) and the radiation difference adds
6 dB/octave, while the aspiration noise is flat after radiation (C5
there); both go through the same cascade. So with AV and AH fixed, the
noise share rises about 6 dB per octave until noise dominates, and the
true aperiodicity of a sonore Klatt vowel can be written down from its
parameters. This is the bridge for the gallery page below.

**C10. CheapTrick, D4C and WORLD's synthesis can be reproduced in
NumPy.** [check, crosscheck] Given the same F0 track and inputs, the
checker's ports against pyworld 0.3.5: CheapTrick within 0.00003 dB on
the gallery sentence (C1); D4C within 7e-12 dB on every time window of the
sentence and within 2e-10 dB on the steady and vibrato vowels; the
synthesized sound within 1.4e-15, 3.0e-15 and 2.7e-15 of its largest
sample, which is rounding. The D4C and synthesis ports needed WORLD's own
noise generator, drawn in WORLD's order, and two details of the code not
in any paper: the DC removal overwrites the first half of each periodic
response, and the safety noise is added to the windowed segment before
its weighted mean is removed. The checker's CheapTrick port leaves out
the safety noise CheapTrick also draws from that generator, which is
where its 0.00003 dB comes from; the library's port includes it and gets
within 4e-9 dB on the whole sentence.

**C11. WORLD's own round trip is not tight on real speech.** [crosscheck]
The gallery sentence analysed with its stored Harvest track, synthesized,
and the result analysed again with the same track: on voiced time windows,
wherever the envelope is within 40 dB of the time window's peak, the second
envelope differs from the first by 1.2 dB (median) and 7.4 dB (95th
percentile); D4C at 3 kHz differs by 1.6 and 5.2 dB. So the consistency
of analysis and synthesis is something to report about WORLD, not a
tolerance sonore can tighten without departing from it.

## Proposed design

Layers follow `layout.md`.

**analysis/vocoder.py**:

- `so.cheaptrick(sound, f0, *, q1=-0.15, f0_floor=71.0)`: the port,
  returning a `SpectralEnvelope`. Data are WORLD's power spectra, stored
  as sonore stores spectra, shape `(n_channels, n_freqs, n_windows)`, with
  `.t`, `.f`, `.db`, `.plot()`; `.to_world()` gives one channel in
  WORLD's `(n_windows, n_freqs)` layout. It is callable, `env(t, f)`,
  interpolating linearly in time and in dB over frequency. (`Envelope` is
  already the Hilbert envelope class.)
- `so.d4c(sound, f0, *, threshold=0.85, f0_floor=71.0)`: the port,
  returning an `Aperiodicity` in WORLD's storage (amplitude ratio per
  frequency). Callable, `ap(t, f)`, with `.share`, `.db`,
  `.bands(edges, envelope)` to average the power share over bands for
  display and tests, and `.method`, the measure that made it.
- `so.harmonic_aperiodicity(sound, f0)`: the harmonic-residual measure
  (C4), in the same storage, so either can be passed to the synthesis;
  its documentation says it is not D4C and when they differ (C5, C6). It
  is the slow one: a dense least-squares fit per time window, about 13 s for
  the 2.5 s gallery sentence.
- `world_randn(n)` and `world_fft_size(fs)`: WORLD's noise stream and
  CheapTrick's FFT size, public so that tests and readers can check them.
  The stream is computed many lanes at a time (the generator is linear
  over GF(2), so a jump ahead is one 128 × 128 binary matrix) and kept;
  the gallery sentence's analysis takes about 0.7 s per step.
- `so.DIFFERENCES_FROM_WORLD`, as above.

**stimuli/vocoder.py**: `so.world_synthesize(f0, envelope, aperiodicity,
*, rng=None)`, the port of WORLD's synthesis (D5). The hop (WORLD's frame period) is
the F0 track's spacing, whose times must start at 0. With `rng=None` it
uses WORLD's generator and gives WORLD's samples (D8).

The F0 track (an `F0Track` or a `(times, f0)` pair) is an argument
everywhere (D7). The windows, the 1024-point spectra at 16 kHz and the
time windows are WORLD's; the envelope is a view of the sound computed the way
WORLD computes it, not a view of a sonore frame (D2).

**Tests.** `tools/make_world_fixtures.py` (pyworld, development only)
stores WORLD's envelope (with both q̃₁), aperiodicity and synthesis for a
breathy vowel with vibrato and 0.3 s of the gallery sentence, every 8th
frequency bin, in `tests/data/world_reference.npz` (289 kB, sounds included, since the sdist carries no docs).
`tests/views/test_world.py` holds sonore to within 1e-6 dB (envelope),
1e-8 dB (aperiodicity) and 1e-9 of the peak (synthesis); the measured
differences are about 4e-9 dB, 7e-12 dB and 1e-13.

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
  jitter and shimmer;
- with D4C, **most of the aperiodicity's shape**: at 16 kHz one number per
  time window (C6).

So a sound is recoverable only approximately, by a model (harmonics plus
noise through a smooth filter), not by search as Griffin–Lim or texture
synthesis are. What the tests check is that sonore's steps are WORLD's
(C10); how closely WORLD's own round trip returns its inputs is measured
and stated (C11).

### Aperiodicity is explained on a gallery page

**Requirement (Cho, 2026-10-02): the aperiodicity measure is to be
explained in a gallery page.** It is the least familiar of the three
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
2. One time window's spectrum: the harmonic peaks, the noise between them, and
   the residual after the harmonics are fitted and removed (C4). The
   share is read off as residual over total. This is the definition made
   visible.
3. What WORLD reports instead: D4C on the same vowel, its one band at
   16 kHz and its anchors (C6), and why it was built that way (tuned for
   natural-sounding synthesis, robust to F0 error).
4. Both measures on the gallery sentence as time–frequency maps beside
   its spectrogram.
5. Listening: the sentence through WORLD's synthesis with D4C's
   aperiodicity, with the harmonic residual's, with A = 0 everywhere
   (buzzy) and A = 1 (whispered), same envelope and F0 throughout.

## Order

1. A tools script that stores WORLD's output for the tests (pyworld,
   development only).
2. `so.cheaptrick` and `SpectralEnvelope`, tested against the stored
   output and the checker's numbers.
3. `so.d4c` and `Aperiodicity`, with WORLD's generator, tested the same
   way.
4. `so.world_synthesize`, tested sample by sample against the stored
   sound.
5. `so.harmonic_aperiodicity`, with the test vowels as tests.
6. The gallery page (with the aperiodicity explanation above), README
   row, recipe and roadmap Done entry.

Steps 2–4 are the WORLD ports and come first; step 5 is independent of
them except for the storage.

## Decisions

Each option is stated in its best form before the recommendation. All
recommendations follow the reproducibility rule above.

**D1. Envelope estimator, and CheapTrick's q̃₁.**
- *CheapTrick, ported exactly, with the code's q̃₁ = −0.15 as default and
  the paper's −0.09 as an option* (recommended): WORLD's numbers by
  default (C10); stable within a period (C2); the paper's value, which
  fits the test vowel's shape better (C3), one argument away and listed in
  `DIFFERENCES_FROM_WORLD`.
- *CheapTrick with the paper's −0.09 as default*: published, and the
  better shape in C3; but every comparison with a WORLD user's numbers
  would then need the argument set, which is the drift Cho wants to
  avoid.
- *True envelope* (iterated cepstrum that rises to the peaks; Röbel &
  Rodet, cited from memory) or *LPC*: each is a reasonable view later,
  under its own name; neither is WORLD's.

**D2. Where the envelope's spectra come from.**
- *WORLD's own windowing at the track's times* (recommended): the only
  way to give WORLD's numbers (C10), since WORLD rounds the window length
  differently from the frame and works on a fixed 5 ms hop. The cost is
  a second windowing path beside sonore's frames; the gallery can still
  show the link to `TVGaborFrame.pitch_adaptive` (the same three-period
  Hann window) in words and a figure.
- *The coefficients of `TVGaborFrame.pitch_adaptive`*: the envelope
  becomes a view of an exact frame, like `Cepstrum(coefs)`; but window
  lengths can differ by a sample and the frame's times are not WORLD's,
  so it would not be CheapTrick's output and would have to carry another
  name. Not needed now.

**D3. Aperiodicity.**
- *D4C ported exactly as `so.d4c`, and the harmonic residual under its own
  name* (recommended): D4C is what WORLD's synthesis was tuned with and
  what WORLD users quote, and is robust to F0 error (C6); the harmonic
  residual is the measure whose method is its definition (C4), so it is
  the one the gallery page can explain, and the one that reports the
  share of noise on the test vowels. Both store the same quantity, so
  either feeds the synthesis.
- *Only D4C*: least code; but sonore could then not show what
  aperiodicity means, only what WORLD reports, and C6 shows the two far
  apart below 3 kHz.
- *Only the harmonic residual*: the first draft's recommendation; but
  under the rule it could not be called WORLD's, and it needs F0 to
  0.1% (C5), so a vocoder built on it would not be WORLD's vocoder.
- *Peak-to-valley ratio* (STRAIGHT's idea; Kawahara et al., 1999): simple
  to show on a spectrum; but with a three-period Hann window a perfectly
  periodic sound already has valleys only about 12 dB below its peaks
  (`frames.md`, C16), so it needs a calibration table to mean anything.

**D4. How aperiodicity and envelope are stored.**
- *WORLD's arrays* (recommended): power spectra and amplitude ratios on
  WORLD's frequency grid and window times, so a value can be compared
  with pyworld's directly; the classes add `.t`, `.f`, `.db` and
  `.bands()` on top.
- *Noise share in dB, or ERB bands*: closer to how sonore displays things,
  but a conversion every comparison would need. `.db` and `.bands()` give
  these views without changing what is stored.

**D5. Synthesis.**
- *WORLD's pulse-by-pulse overlap-add, ported exactly* (recommended): the
  reference algorithm, sample for sample (C10), including pulse
  placement, the fractional shift, DC removal and per-pulse noise.
- *Harmonics plus frame-filtered noise* (the first draft's
  recommendation): exact running phase from `harmonic_complex`, noise on
  an exact frame; equal to WORLD only for a steady F0 of whole samples
  (C8), so under the rule it is a different synthesis and would need its
  own name. Deferred; it would also need complex amplitudes in
  `harmonic_complex`, which nothing else needs now.

**D6. Names.**
- *WORLD's names for the ports* (recommended): `so.cheaptrick`,
  `so.d4c`, `so.world_synthesize`, which tell a reader exactly which
  algorithm and which numbers to expect; sonore's own names for what is
  sonore's (`so.harmonic_aperiodicity`, `SpectralEnvelope`,
  `Aperiodicity`). `klatt.md` D5 kept Klatt's names for the same reason.
- *Descriptive names throughout* (`spectral_envelope`, `aperiodicity`):
  sonore's usual style; but they hide which algorithm runs, and a second
  estimator would need a `method=` switch.

**D7. The F0 track.**
- *Required argument* (recommended): WORLD's numbers depend on the track,
  and sonore's `so.f0_track` is not Harvest, so taking the track
  explicitly keeps the comparison honest; the gallery uses the stored
  Harvest track for the sentence, as `f0.md` does.
- *Default to `so.f0_track` when none is given*: one call from a sound to
  a resynthesis; but the result would silently differ from WORLD's, and
  would have to be listed as a difference.

**D8. The noise.**
- *`rng=None` means WORLD's generator, restarted on every call; a NumPy
  Generator gives fresh noise* (recommended): the default output is
  WORLD's, deterministic and identical on every call; a reader who wants
  independent noise tokens passes an `rng`, and that is listed in
  `DIFFERENCES_FROM_WORLD`. `philosophy.md` says randomness comes from an
  explicit `rng`; WORLD's stream is not hidden state but a fixed
  sequence, so the default is a stated constant rather than hidden
  randomness. D4C's safety noise always uses WORLD's stream (it is part
  of the measure).
- *Require an `rng`, with a named constant for WORLD's stream*
  (`rng=so.WORLD_NOISE`): the rule kept to the letter; but the common
  call (reproduce WORLD) needs an extra argument.

**D9. Where the aperiodicity explanation goes.** The explanation itself
is a requirement (above). As a new gallery page, "Source, filter and
aperiodicity" (recommended), with the vocoder's other demos (pitch
change, formant shift, breathiness), or as a section of the Formant
synthesis page, which already has the breathy vowel (C9) but no analysis.

## References

Morise (2015), Morise (2016) and Morise, Yokomori & Ozawa (2016) were read
in full from PDFs Cho supplied on 2026-10-02; their citation details
match the Crossref, ScienceDirect and J-STAGE records checked earlier
that day. Kawahara et al. (1999) was verified for `frames.md`. Kawahara &
Morise (2012) is cited as the D4C paper cites it, not read. Röbel & Rodet
(2005) is cited from memory.

- Kawahara, H., Masuda-Katsuse, I. & de Cheveigné, A. (1999). Restructuring
  speech representations using a pitch-adaptive time-frequency smoothing
  and an instantaneous-frequency-based F0 extraction. *Speech
  Communication* 27, 187–207.
- Kawahara, H. & Morise, M. (2012). Simplified aperiodicity representation
  for high-quality speech manipulation systems. *Proc. ICSP 2012*,
  579–584. As cited by Morise (2016).
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
