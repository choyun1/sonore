# Linear prediction (LPC)

The design of a linear prediction view in sonore: an all-pole model of each
time window, its smooth spectral envelope, and formant frequencies read from
the roots of its polynomial. Cho asked on 2026-10-10 for an LPC demo, perhaps
on the Formant synthesis page. sonore has no LPC today: nothing in `src/`
matches `lpc`, `levinson` or `linear predict`, and the only mentions are
`views/world.md` and `views/voice-change.md`, which name LPC as a possible
envelope "later, under its own name".

Status: draft, waiting for Cho's decisions D1–D8. No library code yet.

## Why

The Formant synthesis page builds vowels from formants that are written
down. LPC runs the other way: from a vowel back to its formants. On a
synthesized vowel the answer is known, so the page can show how close the
estimate comes and where it fails (high voices, C4 and C5). It is also the
oldest of the envelope estimators and the one that explains the source–filter
picture most directly: the predictor *is* the filter, and what it cannot
predict, the residual, is the source. Next to the cepstral envelope and
CheapTrick on the Spectral envelope page, it is the third way of drawing the
same curve, with a different bias (C9).

## How the claims are verified

As in the other design documents, each claim is numbered and tagged:

- **[proof]**: a short argument given here.
- **[check]**: a number printed by `tools/check_lpc_claims.py`. The script
  uses only NumPy, SciPy and soundfile, synthesizes its vowels with its own
  impulse train and Klatt resonators, writes the autocorrelation, the
  Levinson–Durbin recursion and the root picking out from their formulas,
  and shares no code with sonore. It runs in about 2 s. The numbers below
  come from NumPy 2.5.3 and SciPy 1.18.1.
- **[crosscheck]**: a number printed by `tools/crosscheck_lpc.py`, which
  compares the same formula-level LPC with Praat's `To Formant (burg)`
  through parselmouth (a development-time dependency only).
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

**C7. On the recordings, the estimates agree with Praat's for most of the
male talker's voiced frames and for about half to two thirds of the female
talker's.** [crosscheck] Praat's Burg formants (ceiling 5000 Hz male,
5500 Hz female, 5 formants, 25 ms) against the k-th narrow root (bandwidth
under 400 Hz, above 90 Hz) at order 18, at the voiced times both report
three formants:

| | F1 median diff. | F2 | F3 | within 10%: F1, F2, F3 |
|---|---|---|---|---|
| male (424 times) | 7 Hz | 38 Hz | 81 Hz | 92%, 81%, 63% |
| female (445 times) | 27 Hz | 72 Hz | 149 Hz | 67%, 59%, 53% |

Neither side is the truth, and the two methods differ in more than the
algorithm (Praat resamples to twice the ceiling and uses a Gaussian
window). The disagreement comes mostly from *picking*, not from the model:
calling the k-th narrow root "Fk" breaks whenever one root's bandwidth
crosses the threshold or a spurious narrow root appears. The crosscheck
also prints the F2 agreement at other orders: the male median difference
is 11 Hz at order 16, 38 Hz at 18 and 31 Hz at 20, but 1193 Hz at order 14,
and the female one 693, 104, 72 and 101 Hz at orders 14 to 20. A jump of
a thousand hertz is a label off by one formant, not a poor fit (inferred
from its size; which root went missing was not checked). So the view
should return all candidate roots and leave labeling to a tracker (D4).

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
built to sit on the peaks by another route; comparing it here is left to
the gallery.

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

**D1. One new type, `so.LPC`, in `src/sonore/views/lpc.py`, dispatching on
its input as `so.MFCC` does.** A `Sound` is analyzed with the speech recipe
(25 ms symmetric Hamming window, the window `MFCC` already uses, a 10 ms hop,
and an FFT length rounded up to a power of two at least `order` samples
longer than the window, so the autocorrelation is exact by C3). An `STFT` or
`TVSTFT` is used as it is, and the autocorrelation of each time window comes
from its power by one inverse FFT; if its FFT is shorter than the longest
window plus `order`, `LPC` raises and says which `n_fft` would work (C3: the
error is small, 4.5 Hz in the example, but a view called "the
autocorrelation method" should compute it). Alternatives:

- **A function `so.lpc(segment, order) -> ndarray`**, as in librosa. The
  simplest, and it would still be offered as the inner step, but it loses
  the time grid, the envelope and the plot that every other sonore view
  keeps. Could be added later beside the class if wanted.
- **Windowing the sound itself rather than taking an STFT**, so no FFT is
  needed. Equally exact, but a second copy of the windowing that `STFT`
  already does; the FFT route keeps one copy (dedup-over-line-count).

**D2. The autocorrelation method with Levinson–Durbin.** Stable by
construction (C2), exact from an STFT (C3), and the textbook method.
Alternatives: **Burg's method** (Praat's default for formants, librosa's
`lpc`) also guarantees stability and avoids the window's effect on the
estimate, which helps with short windows; it does not come from an STFT and
would be a second, sample-domain path. **The covariance method** fits
without windowing and can be more accurate on a few pitch periods, but
gives no stability guarantee. Either could be added later as
`method="burg"`; not now.

**D3. `order` defaults to round(fs / 1000) + 2.** 18 at 16 kHz, which C6
finds near the best for these vowels. The rule is cited from memory (Markel
& Gray, 1976) and is an estimate for other voices and rates; at 44.1 kHz it
gives 46, fitting the whole band to 22 kHz, so for formants a user would
analyze a resampled sound. Alternative: make `order` required. Honest, but
the rule is what every course gives first, and the docstring can say it is a
rule of thumb.

**D4. `formants()` returns every candidate root, not labeled F1, F2,
F3.** Frequencies and bandwidths of the roots in the upper half plane,
narrower than `max_bandwidth` (400 Hz) and above `min_freq` (90 Hz),
sorted by frequency, NaN-padded to the same count per time window: shape
`(n_channels, n_candidates, n_windows)` each. The thresholds are common
choices, not measured optima (estimates). C7 shows that labeling by count
is where estimates go wrong, and fixing that needs continuity across time:
a formant tracker, which is out of scope here. The gallery draws candidates
as dots. Alternative: return F1–F3 by count, as many tools do; simpler to
use, but it would build C7's failure into the API.

**D5. Pre-emphasis is not an argument, as in `MFCC`.** `MFCC`'s D3
(accepted 2026-10-02) decided that pre-emphasis is a change to the sound,
applied to the sound before the view; LPC follows the same rule, and its
docstring gives the same `scipy.signal.lfilter([1, -0.97], 1, ...)` line.
Here it matters more than for MFCCs (C8: 8.9 against 16 Hz), so the
docstring says so plainly, and the gallery always shows the line.
Alternatives: a `preemphasis=0.97` argument (convenient and the usual
default for LPC, but it would contradict MFCC's accepted decision); or a
small public `so.preemphasize(snd, 0.97)` used by both docstrings (a new
name for a one-line filter; worth it only if more views need it).

**D6. `envelope(f)` and `envelope_view()` give E / |A|² on the analyzed
sound's power scale, as a `GridEnvelope`.** Then LPC plugs into everything
that reads an envelope: `warp_frequency`, `world_synthesize`,
`harmonic_complex`, and the Spectral envelope page's comparisons. By C9 its
level is that of the time window's power spectrum, the same scale as
`Cepstrum.envelope_view()`. When the sound was pre-emphasized the envelope
is the pre-emphasized one; the view does not undo a change it did not make
(D5).

**D7. The gallery demo goes on Formant synthesis, in a new section
"Finding the formants again" before "What this page leaves out", with a
link from Spectral envelope.** Proposed items:

1. *figure*: one time window of the synthesized male "hod": its harmonics,
   the filter the synthesizer was given, the LPC envelope, and the
   recovered formants marked against the given ones;
2. *figure*: the vowel chart (F2 against F1) with the given male and female
   averages and arrows to what LPC recovers, showing the female errors
   (C4);
3. *demo*: "hod" with the pitch gliding from 100 to 300 Hz, its wideband
   spectrogram with the LPC candidates as dots over the given formants, so
   the estimate can be seen breaking as harmonics cross F1 (C5);
4. *demo*: the two talkers' sentence with LPC candidates as dots over the
   spectrogram.

The "Copying a recording" bullet in "What this page leaves out" changes to
say that LPC finds formant candidates per time window, and that linking them
into tracks (a tracker) is what is still missing. On Spectral envelope, one
paragraph and, if the gallery-rewrite thread agrees, an LPC curve on its
one-window figure, linking to the new section. Alternative: a page of its
own; not recommended, since the point of LPC here is recovering the
formants this page writes down.

**D8. No resynthesis in this step.** An LPC vocoder (residual or
pulse-and-noise excitation through 1 / A(z)) is the classic next demo, and
the residual would make an exact analysis–synthesis pair. It is a separate
addition (a `residual()` method and a time-varying all-pole filter);
proposed later if wanted.

## API sketch

    snd = so.load(...)
    emphasized = so.Sound(scipy.signal.lfilter([1, -0.97], 1, snd.data, axis=0), snd.fs)
    lpc = so.LPC(emphasized)                # order 18 at 16 kHz, 25 ms, 10 ms hop
    lpc = so.LPC(stft, order=16)            # any STFT with n_fft >= window + order
    lpc.data          # (n_channels, order + 1, n_windows), data[:, 0] == 1
    lpc.error_power   # (n_channels, n_windows)
    lpc.reflection    # (n_channels, order, n_windows)
    lpc.t             # window times [s]
    lpc.envelope(f)   # power at frequencies f, (n_channels, len(f), n_windows)
    lpc.envelope_view()                     # GridEnvelope on the STFT's bins
    freqs, bandwidths = lpc.formants(max_bandwidth=400, min_freq=90)
    lpc.plot(ax)      # the envelope as a spectrogram, candidates as dots

A time window of digital silence (r[0] = 0) gets A(z) = 1 and zero error
power, and no candidates.

## Tests (target: under 1 s added; an estimate)

- Levinson against `scipy.linalg.solve_toeplitz` on random windows (C1).
- The autocorrelation from an STFT equals the direct sum when n_fft ≥ L + p,
  and an STFT with a shorter FFT raises (C3).
- Every root inside the unit circle on the gallery sentences (C2).
- A synthesized male vowel's F1–F3 within the checker's numbers (C4).
- mean(P / P_model) = 1 for one window (C9), and the envelope reads through
  `warp_frequency`.
- Silent windows give finite output.

## Patch plan

1. This document, `tools/check_lpc_claims.py` and `tools/crosscheck_lpc.py`
   (this PR).
2. After Cho's decisions: `so.LPC`, tests, API docs, README row, CHANGELOG;
   `crosscheck_lpc.py` switched to call `so.LPC`.
3. The gallery section (D7), on top of the gallery rewrite.

## Out of scope

- A formant tracker (continuity over time, labeling F1–F3), and copy
  synthesis driven by it.
- Burg and covariance methods (D2), line spectral frequencies, and LPC
  cepstra.
- Resynthesis (D8).

## References

None of these was verified by lookup for this document; they are cited from
memory. Every number above comes from the checker or the crosscheck.

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
