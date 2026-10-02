# Cepstrum

The design of `sonore.views.cepstrum`: a `Cepstrum` representation built
on sonore's STFTs, with liftering, resynthesis with the original or minimum
phase, and classic cepstral F0. This was item 3 of the README roadmap (now
Done). It is also the first step toward a WORLD-style analysis and synthesis: the
liftering and minimum phase are what a CheapTrick-style envelope and a
pulse-based vocoder need, and cepstral F0 is a baseline that a real F0
tracker must beat.

Status: accepted 2026-10-01, with decisions D1–D7 as recommended below.
Implemented in `src/sonore/views/cepstrum.py`, tested in
`tests/views/test_cepstrum.py`.

## How the claims are verified

As in `frames.md`, each claim is numbered and tagged:

- **[proof]**: a short argument given here.
- **[check]**: a number printed by `tools/check_cepstrum_claims.py`. The
  script uses only NumPy, SciPy and soundfile (to read the gallery sentence),
  writes every transform out from its formula, and shares no code with
  sonore. It runs in under a second. The numbers below come from NumPy 2.4.6
  and SciPy 1.17.1.
- **[source]**: a published result (see References).

## Setting

One time window of a channel is a windowed segment x of length L, zero-padded to
N = n_fft and transformed: X[k], k = 0..N−1. Its **real cepstrum** is

    c[n] = IDFT_N( ln |X[k]| ),   n = 0..N−1,

with quefrency n / fs seconds. The natural logarithm (not dB) is used so
that exp undoes it exactly and the minimum-phase construction (C3) applies
without rescaling. sonore's STFTs already hold X for every time window, so a
cepstrum is one inverse FFT per time window on top of an existing analysis.

## Claims

**C1. The real cepstrum of a real time window is real and even, so half of it is
stored.** [proof, check] ln|X[k]| is real and, because x is real, even in k
(|X[N−k]| = |X[k]|). The IDFT of a real even sequence is real and even. So
c is determined by n = 0..N/2, and `irfft` of the one-sided log magnitude
gives it directly. The checker finds an imaginary part 2e-17 of the real
part, an asymmetry of 2e-17, and a difference of 2e-16 between `irfft` of
the half spectrum and the full IDFT.

**C2. Without liftering, cepstrum and back is exact; scaling the sound only
moves c[0].** [proof, check] exp(DFT(c)) = |X| by construction, and
putting the original phase back recovers X, so the frame's own synthesis
recovers the sound exactly (it inherits the frame's exactness). Scaling x by
a adds ln a to every ln|X[k]|, and the IDFT of a constant is a constant at
n = 0 only. Checker: round-trip error 1.3e-15 in magnitude and 9.5e-16 with
phase; scaling by 3.7 moves c[0] by ln 3.7 to within 2e-16 and every other
quefrency by at most 1e-16.

**C3. Folding the real cepstrum gives the minimum-phase spectrum with the
same magnitude.** [source, check] For a minimum-phase sequence the complex
cepstrum is causal, and it equals the real cepstrum folded: c[0] kept,
2c[n] for 0 < n < N/2, c[N/2] kept, zero for n > N/2. Then exp(DFT(folded))
is the minimum-phase spectrum (Oppenheim & Schafer, 2010, ch. 13). On the
DFT grid the cepstrum is time-aliased, so this is exact only as N grows.
Checker, a 25-tap minimum-phase FIR: error 3.6e-6 of the peak at N = 64,
and 4e-15 to 6e-15 from N = 256 up. For a mixed-phase FIR the folded result
keeps the magnitude (error 9.5e-12) and moves energy to the front: the
first 5 taps hold 46% of the energy, against 0.09% in the original.
Minimum phase is what source-filter and WORLD-style synthesis use for the
filter of each pulse.

**C4. Cepstral F0 needs a window of at least three periods of the lowest
F0.** [source, check] A periodic excitation puts ripples of period F0 in
ln|X|, so the cepstrum peaks at quefrency 1/F0 (Noll, 1967). The ripple is
resolved only if the window holds several periods. The checker passes a
band-limited pulse train through a four-formant /a/ and picks the largest
peak in 1/400 to 1/60 s, refined by a parabola, at 7 F0s (80–300 Hz) and 11
window positions. With a Hann window 1.5 or 2 periods long, 39 and 32 of
77 time windows are off by an octave or more. At 3 periods none are, and the
worst error is 0.48%; at 4 periods, 0.32%. A fixed 40 ms Hann window (3.2
periods at 80 Hz) has a worst error of 0.24% over 80–300 Hz; a fixed 20 ms
one fails at the low F0s (worst error 376%).

**C5. A low-quefrency lifter gives the envelope's shape but sits about 4 dB
below the harmonic peaks.** [check] Same synthetic /a/, a 3-period Hann
window, all quefrencies below half a period kept. At the harmonics below
4 kHz, the liftered envelope matches the true filter's shape to 0.35 dB RMS
(F0 100 Hz) and 0.70 dB RMS (200 Hz), once a constant offset is removed. But
it runs 4.4 dB (100 Hz) and 4.2 dB (200 Hz) below the harmonic peaks on
average, and 5.0 and 4.8 dB at the harmonic nearest F1. This is
expected: the log spectrum dips between harmonics, and the lifter averages
peaks with dips. It is why CheapTrick smooths the power spectrum over one
F0 before taking the cepstrum and corrects the lifter (Morise, 2015). That
is a later step; this one provides the lifter.

**C6. On the gallery sentence, cepstral F0 agrees with Harvest about as well
as plain autocorrelation does, and its peak height is a usable voicing
cue.** [check] `bdl_arctic_a0131`, 40 ms Hann windows at Harvest's 5 ms
times, search 75–400 Hz. On the 432 time windows Harvest calls voiced, 78.7%
agree within 5%; 7 are about double Harvest's F0 and none half. The median
cepstral peak is 0.17 on Harvest-voiced time windows and 0.076 on unvoiced ones.
Calling a time window voiced when its peak exceeds 0.1 keeps 73% of
Harvest-voiced time windows, admits 11% of unvoiced ones, and on the time windows both
call voiced, 95% agree within 5%. (The autocorrelation check in
`docs/speech/SOURCES.md` found 96% on the time windows where it was confident.)
So cepstral F0 is a fair baseline, not a tracker: it has no continuity, no
candidate scoring and a crude voicing rule.

**C7. ln 0 needs a floor, and real recordings stay well above a −200 dB
one.** [check] A time window of digital silence has ln|X| = −inf everywhere (the
checker confirms the log is not finite), and leading or trailing zeros in a
file produce such time windows. On the sentence, the lowest bin of any 40 ms time window
is 143 dB below that time window's maximum, so a floor 200 dB below the channel's
maximum changes no bin of real speech.

## Decisions

**D1. A new module and type. (accepted 2026-10-01)** `sonore/analysis/cepstrum.py` with a
`Cepstrum` class, exported as `so.Cepstrum`, built from an existing
`STFT` or `TVSTFT`: `so.Cepstrum(coefs)`. It keeps the coefficients it was
built from (for their phase, frame and times) and stores `data` of shape
`(n_channels, n_fft // 2 + 1, n_windows)` (C1), with `q` the quefrencies in
seconds and `t` the window times. Recommended over methods on `STFT`
(`stft.cepstrum()`): one way in, and `representations.py` (541 lines) does
not grow. The module sits in the analysis layer and imports
`representations`, not the other way round.

**D2. Real cepstrum only. (accepted 2026-10-01)** The complex cepstrum needs phase unwrapping,
which is fragile on real sounds, and nothing on the roadmap needs it: the
minimum phase comes from the real cepstrum (C3), and the original phase is
taken from the STFT. Natural log of the magnitude, as in the Setting.

**D3. The floor. (accepted 2026-10-01)** Magnitudes are floored at 200 dB below the channel's
maximum over all time windows before the log (C7). Relative, so scaling a sound
only moves c[0] (C2); per channel rather than per time window, so a silent time window
gets a flat log spectrum at the floor instead of amplified noise; −200 dB to
match the floor sonore already uses for STFT displays. An all-zero channel
gets a cepstrum of zeros apart from c[0].

**D4. Liftering. (accepted 2026-10-01)** `cep.lifter(cutoff, keep="low")` returns a new
`Cepstrum` with a rectangular lifter: `"low"` keeps quefrencies below
`cutoff` (and their mirror images), `"high"` keeps the rest. `cutoff` [s]
is a scalar or one value per time window, so a pitch-adaptive cutoff (half a
period, as in C5) is one line from an F0 track. Smooth lifters, and
CheapTrick's corrected one, are left for the envelope step.

**D5. Back to spectra and sound. (accepted 2026-10-01)** `cep.to_stft(phase="original")` returns
coefficients of the same type and frame as the source. `"original"` puts the
source's phase back, exact when nothing was liftered (C2). `"minimum"` uses
the fold (C3). `cep.to_sound(phase=...)` is that, synthesized by the
source's frame, so for modified coefficients it is the least-squares signal,
as everywhere in sonore. `cep.envelope()` returns exp(DFT(c)) as a
magnitude array for plotting and for the later envelope work. With
`"minimum"`, the response of each time window starts at its phase
reference, the window's middle sample. That is the right convention for
pulse-based synthesis later, but resynthesizing speech this way is not a
goal of this step.

**D6. Cepstral F0. (accepted 2026-10-01)** `cep.f0(f_lo=75, f_hi=400, threshold=0.1)` returns
`(t, f0, peak)`, with f0 = 0 where the peak is below `threshold`: the
largest peak in 1/f_hi to 1/f_lo, refined by a parabola, as in the
checker. It raises if any time window is shorter than three periods of
`f_lo` (C4), with a message saying how long the window must be. It is
documented as the classic method (Noll, 1967), with C6's numbers, and not
as a tracker. A `so.cepstral_f0(sound, ...)` shortcut that makes its own
40 ms STFT is not recommended: it would hide C4's window rule.

**D7. Display and docs. (accepted 2026-10-01)** `cep.plot()` draws quefrency in ms against time,
like `STFT.plot`, via `sonore.plotting`. The README gets a module-table row,
a short recipe (envelope by liftering, F0 of the gallery sentence), and the
roadmap item moves to Done. No gallery section in this step.

## API sketch

```python
snd = so.load("docs/speech/bdl_arctic_a0131.flac")
coefs = so.STFT(snd, win_dur=0.040, hop_dur=0.005)
cep = so.Cepstrum(coefs)               # data (1, n_fft//2 + 1, n_windows)
t, f0, peak = cep.f0(f_lo=75, f_hi=400)

env = cep.lifter(0.5 / 120).envelope()  # low quefrencies: spectral envelope
fine = cep.lifter(0.5 / 120, keep="high")
y = fine.to_sound(phase="original")     # excitation-like residual
z = cep.lifter(2e-3).to_sound(phase="minimum")
```

## Tests (target: under 1 s added)

- C1, C2: `Cepstrum(STFT)` and `Cepstrum(TVSTFT)` round trip to 1e-12 with
  the original phase; scaling moves only c[0].
- C3: on a short minimum-phase FIR padded into one time window, `"minimum"`
  reproduces its spectrum; on a mixed-phase one, the magnitude only.
- C4, D6: a synthetic vowel at 100 and 200 Hz with a 3-period window gives
  F0 within 1%; a window under three periods of `f_lo` raises.
- D3: a sound with leading digital silence gives finite cepstra.
- D4: a per-time-window cutoff of the wrong length raises.

## Patch plan

1. This document and `tools/check_cepstrum_claims.py`.
2. `Cepstrum`, `lifter`, `envelope`, `to_stft`, `to_sound`, tests (D1–D5).
3. `f0` and `plot`, README rows, recipe and roadmap (D6, D7).

## Out of scope

- The complex cepstrum and phase unwrapping (D2).
- A real F0 tracker: candidates, voicing, continuity, refinement.
- CheapTrick-style smoothing and lifter correction (C5), aperiodicity, and
  pulse-plus-noise synthesis.
- Mel-cepstra and MFCCs.

## Reference implementations

`tools/crosscheck_cepstrum.py` compares `so.Cepstrum` with independent
implementations (SciPy 1.17.1; Praat 6.1.38 through `parselmouth` 0.4.7, a
development-time dependency only):

- **MATLAB `rceps`**: its documented definition,
  `real(ifft(log(abs(fft(x)))))`, written out in NumPy for one 40 ms time window,
  matches to 4e-16.
- **SciPy `scipy.signal.minimum_phase(method="homomorphic", half=False)`**,
  the same fold: on a 17-tap mixed-phase FIR at n_fft 4096, it matches to
  4e-8 of the peak. SciPy adds 1e-7 of the smallest magnitude before the
  log, which accounts for the difference.
- **Praat's PowerCepstrogram** (Gaussian window, power spectrum in dB, sound
  resampled to 10 kHz), peak searched in 75–400 Hz on the gallery sentence.
  On the 432 time windows Harvest calls voiced, sonore's and Praat's peaks agree
  within 5% on 81%; on the 318 of those whose sonore peak exceeds 0.1, on
  98%. Against Harvest, Praat's peak agrees on 75% and sonore's on 78%.

The gallery page `docs/gallery/voice/cepstrum.py` (cepstrum.html) demonstrates the
class on the sentence and lists the same comparisons.

## References

Noll (1967) was verified by lookup (PubMed 6040805: *JASA* 41(2), 293–309).
The others are cited from memory and not yet verified; Morise (2015) was
verified for `frames.md`.

- Bogert, B. P., Healy, M. J. R. & Tukey, J. W. (1963). The quefrency
  alanysis of time series for echoes: cepstrum, pseudo-autocovariance,
  cross-cepstrum and saphe cracking. In M. Rosenblatt (Ed.), *Proc. Symp.
  Time Series Analysis*, 209–243. Wiley.
- Noll, A. M. (1967). Cepstrum pitch determination. *J. Acoust. Soc. Am.*
  41(2), 293–309.
- Oppenheim, A. V. & Schafer, R. W. (2010). *Discrete-Time Signal
  Processing*, 3rd ed., ch. 13 (cepstrum analysis and homomorphic
  deconvolution). Pearson.
- Morise, M. (2015). CheapTrick, a spectral envelope estimator for
  high-quality speech synthesis. *Speech Communication* 67, 1–7. Already
  cited in `frames.md`.
- Peterson, G. E. & Barney, H. L. (1952). Control methods used in a study
  of the vowels. *J. Acoust. Soc. Am.* 24(2), 175–184. Source of the
  checker's /a/ formants (rounded male averages).
