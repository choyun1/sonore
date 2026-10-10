# Linear prediction (LPC)

The design of linear prediction in sonore: an all-pole model of each time
window, its smooth spectral envelope, the formant frequencies read from the
roots of its polynomial, and a formant tracker that links them into F1, F2
and F3 over time. Cho asked on 2026-10-10 for an LPC demo, perhaps
on the Formant synthesis page. sonore has no LPC today: nothing in `src/`
matches `lpc`, `levinson` or `linear predict`, and the only mentions are
`views/world.md` and `views/voice-change.md`, which name LPC as a possible
envelope "later, under its own name".

Status: accepted 2026-10-10, with decisions D1–D9 as recommended below.
The first draft left the formant tracker out of scope; Cho asked that it be
part of this work and that LPC be compared with the other envelope methods
(C11–C14), and the draft was revised before Cho accepted it. Implemented in
`src/sonore/views/lpc.py`, tested in `tests/views/test_lpc.py` against the
formulas, synthesized vowels and Praat's formants
(`tests/data/praat_formants_reference.npz`, from
`tools/make_lpc_fixtures.py`). One difference from the checker's tracker:
when a time window has fewer candidates than formants, the library fills
the lowest formants, where the checker tried every placement; with Praat's
recipe it happens in none of the voiced time windows of the gallery
sentences (one unvoiced window each), and the two give identical tracks on
both.

## Why

The Formant synthesis page builds vowels from formants that are written
down. LPC runs the other way: from a vowel back to its formants. On a
synthesized vowel the answer is known, so the page can show how close the
estimate comes and where it fails (high voices, C4 and C5). It is also the
oldest of the envelope estimators and the one that explains the source–filter
picture most directly: the predictor *is* the filter, and what it cannot
predict, the residual, is the source. Next to the cepstral envelope and
CheapTrick on the Spectral envelope page, it is the third way of drawing the
same curve, with a different bias (C9), and the only one of the four that
gives formant frequencies directly rather than as peaks to be found (C14).

The tracker is the point of the exercise. With formant tracks, the Formant
synthesis page can run backward on a recording: measure a talker's F1–F3 and
F0, hand them to `so.klatt_synthesize`, and listen to the copy next to the
original. The page lists that, "copying a recording", among the things it
leaves out, for want of a formant tracker.

## How the claims are verified

As in the other design documents, each claim is numbered and tagged:

- **[proof]**: a short argument given here.
- **[check]**: a number printed by `tools/check_lpc_claims.py`. The script
  uses only NumPy, SciPy and soundfile, synthesizes its vowels with its own
  impulse train and Klatt resonators, writes the autocorrelation, the
  Levinson–Durbin recursion and the root picking out from their formulas,
  and shares no code with sonore. It also builds an utterance with moving
  formants from its own time-varying resonators, and a formant tracker. It
  runs in about 3 s. The numbers below
  come from NumPy 2.5.3 and SciPy 1.18.1.
- **[crosscheck]**: a number printed by `tools/crosscheck_lpc.py`, which
  compares the same formula-level LPC with Praat's `To Formant (burg)`
  through parselmouth (a development-time dependency only), and reads
  formants off sonore's cepstral, CheapTrick and MFCC envelopes.
- **[source]**: a published result (see References).

The synthesized vowels are Peterson and Barney's six male and six female
averages from the Formant synthesis page, with F4 and F5 as there, Klatt's
default bandwidths (60, 90, 150, 200, 250 Hz), an impulse train at 120 Hz
(male) or 220 Hz (female) through Klatt's glottal low-pass and a first
difference. The F0 is rounded to a whole-sample period, so the window can be
moved over exactly one period: every recovery number is taken at 20 window
positions spread over one period, because the estimate depends on where the
window falls against the pulses. Unless a claim says otherwise the analysis
is the textbook one: 25 ms symmetric Hamming window, pre-emphasis
y[n] = x[n] − 0.97 x[n−1], order 18.

## Setting

Linear prediction models each sample as a weighted sum of the previous p,

    x[n] ≈ −(a1 x[n−1] + … + ap x[n−p]),

and chooses a1..ap to minimize the squared error over a windowed segment.
With the segment zero outside the window (the *autocorrelation method*), the
minimizing coefficients solve the Toeplitz normal equations

    sum_j a_j r[|i − j|] = −r[i],   i = 1..p,

where r[k] is the autocorrelation of the windowed segment. Writing
A(z) = 1 + a1 z^−1 + … + ap z^−p, the model of the segment is the all-pole
filter 1 / A(z) driven by white noise of power E, the prediction error that
is left: its power spectrum is

    P_model(f) = E / |A(e^{j 2π f / fs})|².

Each pair of complex roots z = |z| e^{±jθ} of A(z) is a resonance at
θ fs / 2π Hz with bandwidth −ln|z| fs / π Hz, the same relation Klatt's
resonator uses in reverse (its pole radius is e^{−π B / fs}).

## Claims

**C1. The Levinson–Durbin recursion solves the normal equations exactly.**
[source, check] Levinson (1947) and Durbin (1960) solve a Toeplitz system in
O(p²) by raising the order one step at a time, producing the reflection
coefficients k1..kp and the error power after each order,
E_i = E_{i−1} (1 − k_i²). The checker's recursion agrees with
`scipy.linalg.solve_toeplitz` to 4e-15 in the coefficients, and its final
error power equals r[0] + a·r[1:] to a relative 1e-15.

**C2. The autocorrelation method always gives a stable filter.** [source,
proof, check] A windowed segment's autocorrelation is positive definite
unless the segment is zero, so every |k_i| < 1, and that is equivalent to
every root of A(z) lying inside the unit circle (Makhoul, 1975). Over all 907
voiced time windows (5 ms hop) of both gallery sentences, the largest
|k_i| is 0.991 and the largest root magnitude 0.997: inside, though close to
the circle. The covariance method, which does not window, has no such
guarantee (Makhoul, 1975); it is not measured here.

**C3. The autocorrelation can come from an existing STFT, exactly when the
FFT is at least p samples longer than the window.** [proof, check] The
inverse FFT of |X[k]|² on n_fft bins is the circular autocorrelation:
lag k picks up r[k] + r[n_fft − k]. A segment of length L has r[m] = 0 for
m ≥ L, so lags 0..p are exact when n_fft − p ≥ L. On a 400-sample (25 ms)
window of male "hod": with n_fft = 400 the lags are off by up to 1.1e-3 of
r[0] and F1–F3 move by up to 4.5 Hz; with n_fft = 418 = L + p or 1024 the
error is 4e-16 of r[0] and F1–F3 agree to 6e-12 Hz. sonore's STFTs default
to n_fft equal to the window length, so this matters (D1).

**C4. On synthesized vowels the formants come back within a few percent for
the male set, and with errors two to four times larger for the female set.**
[check] Error of the root nearest each given formant, over the 6 vowels × 20
window positions of each set:

| | F1 median | F2 median | F3 median | largest, as a fraction | bandwidth median |
|---|---|---|---|---|---|
| male, 120 Hz | 8.5 Hz | 7.3 Hz | 3.8 Hz | 4.1% (F1) | 13 Hz |
| female, 220 Hz | 25 Hz | 35 Hz | 19 Hz | 11% (F1) | 39 Hz |

The worst male vowel is "hawed" (largest error 16 Hz), the worst female one
"head" (46 Hz). Bandwidths are much less reliable than frequencies: the
given ones are 60 to 150 Hz, so a 39 Hz median error is a large fraction.
These are errors against the formants *given to the synthesizer*; for a
recording there is no such truth (C7).

**C5. The error depends on the pitch, and not smoothly.** [source, check]
With higher F0 the harmonics sample the envelope more sparsely, and the
all-pole fit is pulled toward the harmonics near a formant rather than the
formant itself (Atal & Hanauer, 1971, discuss this for high-pitched voices;
cited from memory). Largest |F1 error| over the window positions, male
formants:

| F0 | 100 | 150 | 200 | 250 | 300 Hz |
|---|---|---|---|---|---|
| "hod" (F1 730 Hz) | 16 | 12 | 41 | 16 | 91 Hz |
| "heed" (F1 270 Hz) | 16 | 26 | 17 | 14 | 37 Hz |

It is not monotonic: what matters is how close a harmonic falls to the
formant, so the same vowel can come out better at 250 Hz than at 200 Hz.
This is the effect the demo should let the reader hear and see.

**C6. Order 18 at 16 kHz (fs/1000 + 2) is near the best for these vowels,
and the error rises on both sides.** [source, check] The rule of thumb is two
poles per kilohertz of bandwidth (one resonance per kilohertz for an adult
male vocal tract) plus two for the glottal and radiation tilt (Markel &
Gray, 1976; cited from memory). Median over the twelve vowels of the median
|F1–F3 error|:

| order | 8 | 12 | 14 | 16 | 18 | 20 | 24 | 30 |
|---|---|---|---|---|---|---|---|---|
| error | 268 | 14 | 11 | 9.5 | 8.9 | 12 | 17 | 18 Hz |

Below 12 formants merge; above 20 the extra poles start fitting harmonics.

**C7. On the recordings, the textbook recipe disagrees with Praat, mostly
because of how roots are picked.** [crosscheck] At order 18 on the 16 kHz
sound, keeping roots narrower than 400 Hz and calling the k-th one Fk, the
median difference from Praat's F1, F2 and F3 is 7, 35 and 74 Hz for the male
talker and 27, 53 and 65 Hz for the female one, and the tracks jump by more
than 20% between neighboring time windows far more often than Praat's (F3:
24% against 0.9% of voiced neighbors, male; 42% against 5.5%, female). C13
shows that the cause is the bandwidth threshold: a root that widens past
400 Hz for one time window drops out and every label above it shifts by
one.

**C8. The window's shape and length matter little.** [check] Median over
vowels of the median |F1–F3 error|, and the worst vowel's median:
Hamming 25 ms 8.9 / 42 Hz, Hamming 40 ms 8.6 / 42 Hz, Hann 25 ms 7.4 /
43 Hz, Hann 40 ms 7.8 / 42 Hz. Pre-emphasis matters more: without it the
median is 16 Hz against 8.9 Hz with it, since without it two of the poles go
to the glottal tilt instead of the formants.

**C9. The LPC envelope sits closer to the harmonic peaks than a cepstral
envelope.** [source, check] Minimizing the prediction error is equivalent to
minimizing the mean of P / P_model − ln(P / P_model) over frequency, which
penalizes P above the model more than below it, so the model follows the
peaks and bridges the valleys; at the minimum, mean(P / P_model) = 1
exactly (Makhoul, 1975). The checker finds that mean equal to 1 to machine
precision. On male "hod" at 120 Hz (40 ms Hamming), the harmonics below
4 kHz stand on average 5.3 dB above the LPC envelope and 13.5 dB above the
cepstral envelope liftered at half a period, which is the log-domain
average and so runs through the middle of the harmonic comb. CheapTrick is
built to sit on the peaks by another route; C14 compares all four as ways
of finding formants.

**C10. Praat's recipe for formants recovers the synthesized vowels as well
as the textbook one.** [check] Praat resamples the sound to twice a formant
ceiling (Praat's advice: 5000 Hz for a male voice, 5500 Hz for a female
one), fits 2 × 5 poles, and keeps *every* root between 50 Hz and 50 Hz below
the ceiling, whatever its bandwidth. Done with the autocorrelation method
on a 25 ms Hamming window, the root nearest each given formant is off by a
median of 6.0, 7.4 and 9.9 Hz (F1–F3, male, at most 21 Hz) and 29, 28 and
15 Hz (female, at most 44 Hz): the same accuracy as C4.

**C11. On synthesized speech with moving formants, labeling the k-th root
as Fk is already right, so a tracker has nothing to fix.** [check] The test
utterance is the six vowels in a row, each held 200 ms and joined by 60 ms
glides, the pitch falling from 130 to 100 Hz (male) or 240 to 200 Hz
(female), made by time-varying resonators so the true formants are known
every sample. Every 5 ms, with Praat's recipe:

| | median F1–F3 error | gross errors (missing or > 10% off) |
|---|---|---|
| male, clean | 8.3 Hz | 0% |
| male, white noise 30 dB down | 12 Hz | 7.5% |
| female, clean | 20 Hz | 0.6% |
| female, white noise 30 dB down | 22 Hz | 8.9% |

The tracker (D5) gives exactly the same numbers in all four cases. The
errors in noise are roots in the wrong place, not roots with the wrong
label: looked at by hand in the male utterance (not printed by the
checker), the weak F3 of "who'd" comes out 230 to 420 Hz high, and no
labeling can move it. So a synthesized utterance cannot show what a tracker is for;
the recordings can (C13).

**C12. Burg's method gives the same formants as the autocorrelation
method.** [check] Root nearest each given formant, order 18, 25 ms Hamming,
pre-emphasis, over the static vowels: autocorrelation 5.5 Hz median (at most
16 Hz) male, 31 Hz (46 Hz) female; Burg 4.7 Hz (14 Hz) male, 32 Hz (45 Hz)
female. The method is not what separates sonore's recipe from Praat's.

**C13. With Praat's recipe the LPC formants of the recordings match Praat's
to a few hertz, and the tracker makes the female talker's tracks smoother
than Praat's.** [crosscheck] Median difference from Praat's F1, F2, F3, and
the fraction of voiced neighbors where a track jumps by more than 20%:

| | median diff. F1, F2, F3 | jumps F1, F2, F3 |
|---|---|---|
| male, Praat | | 7.0%, 4.2%, 0.9% |
| male, Praat's recipe, by count | 1.1, 2.9, 7.3 Hz | 6.3%, 4.4%, 1.6% |
| male, Praat's recipe, tracker | 1.1, 2.9, 7.3 Hz | 6.5%, 4.4%, 1.4% |
| female, Praat | | 16%, 9.4%, 5.5% |
| female, Praat's recipe, by count | 1.9, 4.1, 4.6 Hz | 16%, 10%, 6.4% |
| female, Praat's recipe, tracker | 2.0, 4.4, 5.2 Hz | 12%, 7.2%, 3.2% |

So the autocorrelation method on a Hamming window reproduces Praat's Burg
method on its Gaussian window to within a few hertz once the roots are
picked the same way (C12 says the method hardly matters). For the male
talker the tracker changes almost nothing; for the female talker it halves
the F3 jumps without moving the median agreement by more than 0.6 Hz.
Jumps are a symptom, not a measure of error: some are real (a consonant
release), and with no hand-labeled formants for these sentences the
tracker's gain on recordings is shown as smoothness only. The tracker
applied to the textbook recipe (16 kHz, bandwidth under 400 Hz) cuts the
F3 jumps from 24% to 14% (male) and from 42% to 30% (female), but not to
Praat's level: picking all roots matters more than tracking.

**C14. Of the four envelopes, only LPC gives the formants directly; reading
peaks off the others is less accurate, and off the cepstral and MFCC
envelopes it needs to know where to look.** [crosscheck] F1–F3 of the
synthesized vowels at 40 time windows each, taken as the k-th peak of each
envelope ("by count") and, as a best case no real use could reach, as the
peak nearest each true formant; the sound pre-emphasized for every method:

| | male, by count | male, nearest | female, by count | female, nearest |
|---|---|---|---|---|
| LPC roots (Praat's recipe) | 7.5 Hz, 0% | 7.5 Hz, 0% | 25 Hz, 5.6% | 25 Hz, 5.6% |
| CheapTrick peaks | 27 Hz, 11% | 27 Hz, 11% | 71 Hz, 39% | 67 Hz, 28% |
| cepstral envelope peaks | 690 Hz, 100% | 27 Hz, 18% | 520 Hz, 83% | 80 Hz, 26% |
| MFCC envelope peaks | 76 Hz, 66% | 93 Hz, 50% | 290 Hz, 76% | 190 Hz, 48% |

(median F1–F3 error, then the fraction of gross errors: missing or more than
10% off; a missing formant counts as gross but stays out of the median,
which is why a median by count can be lower than the nearest-peak one.)
The cepstral envelope liftered at half a period keeps a ripple
that makes extra peaks, so its k-th peak is rarely Fk, though a peak lies
near each formant. Thirteen MFCCs smooth too much to separate close
formants (F1 and F2 of "hawed" are 270 Hz apart). CheapTrick is built to
follow the harmonic peaks and does so, but is still three to four times
less accurate than the roots. No method was tuned for this, and MFCCs were
never meant to locate formants; the table says what each envelope keeps,
not which is better at its own job.

## Views, and what this one drops

By `philosophy.md` LPC is a view, not a frame. It keeps, per time window, p
coefficients and one error power. It drops:

- **the residual**: whatever the predictor cannot predict, which in speech is
  mostly the excitation (the pulses, their timing and shape, and the noise).
  The residual plus the coefficients would give the sound back exactly
  (filtering by A(z) and then by 1 / A(z)), but the view keeps only the
  coefficients;
- **the phase**, beyond the minimum phase that 1 / A(z) implies;
- **zeros**: an all-pole model represents antiresonances (nasals, the side
  branches of laterals) only by spending poles on them;
- **detail finer than p / 2 resonances** across the band.

## Decisions

Each recommendation follows the strongest form of each alternative.

**D1. One new view, `so.LPC`, in `src/sonore/views/lpc.py`, dispatching on
its input as `so.MFCC` does.** A `Sound` is analyzed with the speech recipe
(25 ms symmetric Hamming window, the window `MFCC` already uses, a 10 ms hop,
and an FFT length rounded up to a power of two at least `order` samples
longer than the window, so the autocorrelation is exact by C3). An `STFT` or
`TVSTFT` is used as it is, and the autocorrelation of each time window comes
from its power by one inverse FFT; if its FFT is shorter than the longest
window plus `order`, `LPC` raises and says which `n_fft` would work (C3: the
error is small, 4.5 Hz in the example, but a view called "the
autocorrelation method" should compute it). `LPC.candidates()` returns every
root in the upper half plane, with its frequency and bandwidth, sorted and
NaN-padded; filtering by bandwidth is left to the caller, since C7 and C13
show a fixed threshold is what breaks labeling. Alternatives:

- **A function `so.lpc(segment, order) -> ndarray`**, as in librosa. The
  simplest, but it loses the time grid, the envelope and the plot that
  every other sonore view keeps.
- **Windowing the sound itself rather than taking an STFT**, so no FFT is
  needed. Equally exact, but a second copy of the windowing that `STFT`
  already does; the FFT route keeps one copy.

**D2. The autocorrelation method with Levinson–Durbin.** Stable by
construction (C2), exact from an STFT (C3), and the textbook method. Burg's
method gives the same formants (C12) and is what Praat uses, but it works on
samples, not on an STFT, so it would be a second path for no measured gain.
The covariance method gives no stability guarantee. Neither is offered now.

**D3. `so.LPC`'s `order` defaults to round(fs / 1000) + 2.** 18 at 16 kHz,
which C6 finds near the best for these vowels. The rule is cited from memory
(Markel & Gray, 1976) and is an estimate for other voices and rates.
Alternative: make `order` required. Honest, but the rule is what every
course gives first, and the docstring can say it is a rule of thumb.

**D4. A function `so.formant_track(sound, ceiling=5000, n_formants=5, ...)`
returning a `FormantTrack` view, with Praat's recipe.** It mirrors
`so.f0_track` and `F0Track`: a function that analyzes a sound and a view
that keeps the result. It resamples the sound to twice `ceiling`
(`Sound.resample`), pre-emphasizes it, runs `so.LPC` on it with order
2 × `n_formants` (25 ms window, 5 ms hop), keeps every root between 50 Hz
and `ceiling` − 50 Hz (C10), and tracks F1–F3 (D5). `FormantTrack` holds
`t`, `frequencies` and `bandwidths` of shape `(n_channels, 3, n_windows)`,
and the `candidates` it chose from, as `F0Track` keeps its candidates. It
does not decide voicing: unvoiced time windows get formants too, and the
caller masks them with an F0 track's voicing, as the gallery would. Why
Praat's recipe: it matches Praat to a few hertz (C13), so sonore's tracks
can be checked against the tool phoneticians use, and it removes most
labeling errors before any tracking (C7 against C13). The ceiling is the
user's choice, as in Praat; the docstring says 5000 Hz for a male voice and
5500 Hz for a female one (Praat's advice, not measured here beyond these two
talkers). Alternatives:

- **A `formants()` method on `so.LPC`** with the 16 kHz textbook recipe.
  One name fewer, but C7 and C13 show the textbook recipe labels worse,
  and resampling a sound is a change to the sound, which a view should not
  make (D6).
- **Labeling by count with no tracker**, which is what Praat's
  `To Formant` returns. Simpler, and C11 finds it already right on
  synthesized speech, but on the female recording the tracker halves the
  F3 jumps (C13).

**D5. The tracker is a Viterbi search over assignments of candidates to
F1–F3.** A state is an increasing choice of three candidates in one time
window (a formant may be missing only when there are fewer than three).
Its cost is, per formant, |ln(f / nominal)| with nominal frequencies 500,
1500 and 2500 Hz (a uniform 17.5 cm tube), plus bandwidth / f, plus 2 for
every candidate skipped below the highest one chosen; moving between time
windows costs 5 |ln(f_now / f_before)| per formant. This is the shape of
Talkin's (1987) dynamic-programming tracker as cited from memory, not a port
of it. The weights were set by trying a handful of values on the two
gallery sentences and the synthesized utterance (C11, C13); they are an
estimate, not an optimum, and they are parameters. Two pieces were needed
to keep it from making things worse, and are kept: the skip cost (without
it the tracker skipped F2's root and called F3 "F2") and allowing a missing
formant only when candidates run out (without it, missing slots dodged the
transition cost). Alternative: no nominal frequencies, only continuity.
Fewer parameters, but nothing then anchors which candidate is F1 at the
start of each voiced stretch.

**D6. Pre-emphasis is not an argument of `so.LPC`, as in `MFCC`; it is
part of `so.formant_track`'s recipe.** `MFCC`'s D3 (accepted 2026-10-02)
decided that pre-emphasis is a change to the sound, applied before a view;
`so.LPC` follows it, and its docstring gives the `scipy.signal.lfilter([1,
-0.97], 1, ...)` line and says it matters (C8: 8.9 against 16 Hz).
`so.formant_track` is a named recipe from a sound, like `so.f0_track`, and
resampling and pre-emphasis are steps of that recipe, stated in its
docstring. Alternative: no pre-emphasis inside `formant_track` either, so
the caller must remember it; consistent, but every call would need the same
line, and forgetting it doubles the error.

**D7. `envelope(f)` and `envelope_view()` give E / |A|² on the analyzed
sound's power scale, as a `GridEnvelope`.** Then LPC plugs into everything
that reads an envelope: `warp_frequency`, `world_synthesize`,
`harmonic_complex`, and the Spectral envelope page's comparisons. By C9 its
level is that of the time window's power spectrum, the same scale as
`Cepstrum.envelope_view()`. When the sound was pre-emphasized the envelope
is the pre-emphasized one; the view does not undo a change it did not make.

**D8. The gallery demo goes on Formant synthesis, in a new section
"Finding the formants again" before "What this page leaves out", with a
link from Spectral envelope.** Proposed items:

1. *figure*: one time window of the synthesized male "hod": its harmonics,
   the filter the synthesizer was given, and the LPC, CheapTrick, cepstral
   and MFCC envelopes, with the LPC formants marked against the given ones
   and C14's table printed beside it;
2. *figure*: the vowel chart (F2 against F1) with the given male and female
   averages and arrows to what LPC recovers, showing the female errors
   (C4);
3. *demo*: "hod" with the pitch gliding from 100 to 300 Hz, its wideband
   spectrogram with the LPC candidates as dots over the given formants, so
   the estimate can be seen breaking as harmonics cross F1 (C5);
4. *demo*: the two talkers' sentence with F1–F3 from `so.formant_track`
   over the spectrogram, voiced parts only, by count and tracked;
5. *demo*: copy synthesis. Each talker's tracks and `so.f0_track`'s F0 and
   voicing drive `so.klatt_synthesize`, played next to the recording. How
   it sounds has not been tried; what it leaves out (bandwidths fixed,
   fricatives and bursts unmodeled unless their noise is added) would be
   said beside it.

The "Copying a recording" bullet in "What this page leaves out" goes, and
the "formant tracker that sonore does not have" there and on Spectral
envelope is updated. On Spectral envelope, one paragraph and, if the
gallery-rewrite thread agrees, an LPC curve on its one-window figure.
Alternative: a page of its own; not recommended, since the point is
recovering the formants this page writes down, and then using them.

**D9. No LPC resynthesis in this step.** An LPC vocoder (the residual, or
pulses and noise, through 1 / A(z)) is the classic next demo, and the
residual would make an exact analysis–synthesis pair. It is a separate
addition (a `residual()` method and a time-varying all-pole filter);
proposed later if wanted. Copy synthesis (D8) resynthesizes through the
formant synthesizer instead.

## API sketch

    emphasized = so.Sound(scipy.signal.lfilter([1, -0.97], 1, snd.data, axis=0), snd.fs)
    lpc = so.LPC(emphasized)                # order 18 at 16 kHz, 25 ms, 10 ms hop
    lpc = so.LPC(stft, order=16)            # any STFT with n_fft >= window + order
    lpc.data          # (n_channels, order + 1, n_windows), data[:, 0] == 1
    lpc.error_power   # (n_channels, n_windows)
    lpc.reflection    # (n_channels, order, n_windows)
    lpc.t             # window times [s]
    lpc.envelope(f)   # power at frequencies f, (n_channels, len(f), n_windows)
    lpc.envelope_view()                     # GridEnvelope on the STFT's bins
    freqs, bandwidths = lpc.candidates()    # every root, NaN-padded
    lpc.plot(ax)      # the envelope as a spectrogram, candidates as dots

    track = so.formant_track(snd, ceiling=5500)   # Praat's recipe plus the tracker
    track.t, track.frequencies, track.bandwidths  # (n_channels, 3, n_windows)
    track.candidates                              # what the tracker chose from
    track.plot(ax)                                # over a spectrogram
    so.klatt_synthesize(dur, fs, F1=(track.t, track.frequencies[0, 0]), ...)

A time window of digital silence (r[0] = 0) gets A(z) = 1, zero error
power and no candidates.

## Tests (target: under 1 s added; an estimate)

- Levinson against `scipy.linalg.solve_toeplitz` on random windows (C1).
- The autocorrelation from an STFT equals the direct sum when n_fft ≥ L + p,
  and an STFT with a shorter FFT raises (C3).
- Every root inside the unit circle on the gallery sentences (C2).
- A synthesized male vowel's F1–F3 within the checker's numbers (C4, C10).
- `formant_track` on the male sentence within a few hertz of Praat's
  formants, stored as a small fixture so the tests need no parselmouth
  (C13).
- The tracker on the synthesized utterance agrees with labeling by count
  (C11), and on a candidate list with a planted spurious root it skips it.
- mean(P / P_model) = 1 for one window (C9), and the envelope reads through
  `warp_frequency`.
- Silent windows give finite output.

## Patch plan

1. This document, `tools/check_lpc_claims.py` and `tools/crosscheck_lpc.py`
   (this PR).
2. After Cho's decisions: `so.LPC`, `so.formant_track`, `FormantTrack`,
   tests, API docs, README row, CHANGELOG; `crosscheck_lpc.py` switched to
   call them.
3. The gallery section (D8), on top of the gallery rewrite.

## Out of scope

- Burg and covariance methods (D2), line spectral frequencies, and LPC
  cepstra.
- Antiresonances (pole-zero models) and formant tracking through nasals.
- Voicing decisions inside `formant_track` (D4).
- LPC resynthesis (D9).

## References

None of these was verified by lookup for this document; they are cited from
memory. Every number above comes from the checker or the crosscheck. Praat's
advice on ceilings and its formant recipe are from Praat's manual as
remembered; C13 checks the recipe against Praat itself.

- Atal, B. S. & Hanauer, S. L. (1971). Speech analysis and synthesis by
  linear prediction of the speech wave. *J. Acoust. Soc. Am.* 50(2B),
  637–655.
- Durbin, J. (1960). The fitting of time-series models. *Rev. Int. Stat.
  Inst.* 28(3), 233–244.
- Levinson, N. (1947). The Wiener RMS (root mean square) error criterion in
  filter design and prediction. *J. Math. Phys.* 25, 261–278.
- Makhoul, J. (1975). Linear prediction: A tutorial review. *Proc. IEEE*
  63(4), 561–580. Stability, the matching condition, and the spectral
  matching interpretation.
- Markel, J. D. & Gray, A. H. (1976). *Linear Prediction of Speech*.
  Springer. The order rule of thumb.
- Peterson, G. E. & Barney, H. L. (1952). Control methods used in a study of
  the vowels. *J. Acoust. Soc. Am.* 24(2), 175–184. The vowels.
- Talkin, D. (1987). Speech formant trajectory estimation using dynamic
  programming with modulated transition costs. *J. Acoust. Soc. Am.* 82(S1),
  S55 (a meeting abstract; volume and page not checked). The shape of the
  tracker (D5); only the idea is used.
