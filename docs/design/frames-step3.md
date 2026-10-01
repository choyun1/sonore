# Frames, step 3: seeing speech

Status: draft, 2026-10-01. Every decision below is **recommended**, awaiting
Cho's review. No code has been written for step 3.

Steps 1 and 2 (docs/design/frames.md, docs/design/frames-step2.md) built the
frames: the cosine, gammatone and Morlet banks, the STFT, and the
time-varying Gabor frame. Step 3 uses them. It adds one gallery section,
"Seeing speech", that runs one spoken sentence through several analyses and
shows the magnitudes side by side, plus the few small pieces of library code
the section needs:

1. **Pitch-adaptive schedules** for `TVGaborFrame`, including a
   TANDEM-STRAIGHT-style pair.
2. **A reassigned spectrogram,** the one nonlinear analysis, for comparison.
3. **Display-time delay compensation** for causal gammatone plots.
4. **A shared way to draw** every magnitude on the same axes.

## Why this section

Cho's motivation, from the step 2 handoff: an invertible frame loses
nothing, so every panel in the section holds the whole sentence, and the
time-frequency tradeoff appears only once phase is discarded and the
magnitudes are drawn. Which magnitude shows the speech best depends on what
is being looked for. Speech is highly constrained (a periodic source shaped by
a slowly moving filter), so analyses adapted to it, and eventually a
Bayesian model of it, can show the harmonic fine structure and the timing
together.

The section should make three points that a reader can see and hear:

- **The tradeoff is real for fixed windows.** A wideband spectrogram shows
  glottal pulses and formants; a narrowband one shows harmonics; neither
  shows both (C15, C16).
- **Adapting to the signal moves the tradeoff.** Constant-Q analyses
  (Morlet, gammatone) resolve low harmonics and high-frequency timing at
  once; a pitch-adaptive window gives the same harmonic resolution at every
  F0 (C16); a TANDEM-style pair removes the period-rate flicker that a
  pitch-length window leaves (C17).
- **Nonlinear sharpening is a different thing.** The reassigned spectrogram
  moves energy to where it "belongs" (C18). It looks sharper than any frame,
  but it is not invertible, and it sharpens whichever structure its window
  already resolves.

## How the claims are verified

As before, claims are numbered (continuing at C15) and tagged [proof],
[check] or [source]. [check] numbers come from
`tools/check_frames_step3_claims.py`, which uses only NumPy and SciPy,
writes every window, filter and transform out from its formula, and shares no
code with sonore. It runs in about 2 s. The numbers below are from SciPy
1.17.1 and NumPy 2.4.6, at fs = 16 kHz.

The test signal for C16 and C17 is a band-limited pulse train: equal-amplitude
cosine harmonics of F0 up to 7.6 kHz. That is the crudest model of voiced
speech (a periodic source with no vocal tract), which is the point: it
isolates what the window does to harmonics and pulses.

## Claims

**C15. Classic sonograph bandwidths as Hann windows.** [proof, check] The
equivalent noise bandwidth of a periodic Hann window of duration T is
1.5/T Hz.

- Proof: ENBW = N Σw² / (Σw)² bins; for Hann, Σw = N/2 and Σw² = 3N/8.
- Check: 1.5 bins exactly, at N = 80 and 528.
- Consequence: the classic wideband and narrowband analyzing filters of
  300 Hz and 45 Hz (Koenig, Dunn & Lacy, 1946; see References) correspond
  to Hann windows of **5.0 ms** and **33.3 ms**.
- Caveat: the paper quotes the narrow filter as about 45 Hz wide at the
  3 dB points, while ENBW is a different width measure; and the spectrograph
  replayed speech sped up, so whether these are the effective bandwidths at
  speech rate was not checked against the page. The two window lengths are
  conventional either way; nothing below depends on the match being exact.

**C16. Fixed windows depend on F0; pitch-adaptive windows do not.**
[proof, check] For a periodic signal, the spectrogram with a window k
periods long, plotted against t/T0 and f/F0, is the same at every F0.

- Proof: scaling time by T0 maps the signal and the window together, so
  every quantity measured in periods and harmonics is unchanged. (The
  check confirms this survives sampling.)
- Two measures, for each window: the **harmonic dip**, the peak-to-dip
  ratio between two harmonics near 1 kHz, in the power spectrum averaged
  over window positions; and the **pulse depth**, the max-to-min ratio over
  one period of the power summed over 1–3 kHz, which is how strongly each
  glottal pulse shows as a vertical stripe. Larger means more visible.

| Window | F0 | Harmonic dip | Pulse depth |
|---|---|---|---|
| Hann 33.3 ms (narrowband) | 100 Hz | 17.6 dB | 0.08 dB |
| | 130 Hz | 31.5 dB | 0.02 dB |
| | 200 Hz | 50.7 dB | 0.00 dB |
| Hann 5 ms (wideband) | 100 Hz | 0 dB | 77.0 dB |
| | 130 Hz | 0 dB | 74.7 dB |
| | 200 Hz | 0 dB | 71.0 dB |
| Hann 1.5 periods | 100 or 200 Hz | 0.5 dB | 9.0 dB |
| Hann 2 periods | 100 or 200 Hz | 3.0 dB | 3.0 dB |
| Hann 2.5 periods | 100 or 200 Hz | 6.9 dB | 0.74 dB |
| Hann 3 periods | 100 or 200 Hz | 12.4 dB | 0.00 dB |
| Hann 3.5 periods | 100 or 200 Hz | 21.1 dB | 0.07 dB |

- The fixed narrowband window resolves harmonics three times better (in
  dB) at 200 Hz than at 100 Hz, so a sentence whose F0 falls by an octave
  changes appearance for reasons that have nothing to do with the voice. A
  pitch-adaptive window does not.
- No single window shows both: the harmonic dip and the pulse depth trade
  off directly. At 2 periods both are 3 dB.
- At 3 or more whole periods the pulse depth is essentially 0, because the
  squared Hann window then sums to a constant over shifts by T0 (its Fourier
  series has no component at 1/T0). At non-integer lengths (2.5, 3.5) a small
  flicker remains.
- 4 whole periods gives a 67 dB dip, but that is the integer coincidence
  of Hann's spectral zeros landing on the neighboring harmonics; it is
  fragile once F0 moves within a window, so it is not quoted in the table.

**C17. The TANDEM pair removes the period-rate flicker.** [proof, check]
For a periodic signal, the power P(t, f) through a window centered at t is
T0-periodic in t. The sum P(t, f) + P(t + T0/2, f) keeps only the even
Fourier components of that periodic function, so the component at the
fundamental rate, the largest one, cancels exactly.

- Proof: shifting by half a period multiplies the m-th Fourier component by
  (−1)^m.
- Check: F0 = 125 Hz (T0 = 128 samples, so T0/2 is exact). Fluctuation is
  (max − min)/mean over t, per frequency bin, 300–4000 Hz:

| Window | Single: worst | Single: median | Pair: worst | Pair: median |
|---|---|---|---|---|
| Hann 2 T0 | 2.00 | 0.67 | 0.0078 | 1.6e-5 |
| Hann 2.5 T0 | 2.05 | 0.19 | 0.052 | 0.0073 |
| Hann 3 T0 | 1.93 | 0.092 | 0.078 | 0.0033 |
| Blackman 2.5 T0 (TANDEM-STRAIGHT's window) | 2.01 | 0.63 | 0.0070 | 0.0014 |

- The worst case for a single window is between harmonics, where the power
  dips; there the pair reduces the flicker 25–250-fold. This is what lets a
  short (about 2-period) window show the spectral envelope steadily. It is
  the principle of TANDEM-STRAIGHT [source]: Kawahara et al. (2011)
  describe a Blackman window 2.5 T0 long and the average
  ½[P(t − T0/4) + P(t + T0/4)]. A Blackman window of 2.5 T0 behaves much
  like a Hann of 2 T0 here, since Blackman's effective length is shorter.
  Only this cancellation is claimed; the published method's other steps are
  not.

**C18. Reassignment, with sonore's phase convention.** [proof, check] For
an STFT whose phase is referenced to the window's center (SciPy's
convention, which `GaborFrame` and `TVGaborFrame` both follow), with frame
center t and bin frequency f,

- t̂ = t + Re(X_tw · conj X) / |X|²,
- f̂ = f − Im(X_dw · conj X) / (2π |X|²),

where X uses the window w, X_tw the time-weighted window τ·w(τ), and X_dw
the derivative w′(τ), with τ in seconds from the window center.

- Proof: for a tone at f0, X_dw = i2π(f − f0)·X, since differentiating the
  window multiplies its transform by i2πν. For an impulse at t0,
  X_tw = (t0 − t)·X. Solving each for f0 and t0 gives the formulas. This is
  the reassignment method of Kodera et al. and Auger & Flandrin (see
  References); only the signs depend on the STFT convention, which is why
  they are checked here.
- Check, at a 32 ms window (512 samples), with Hann and with a Gaussian
  (σ = 4 ms), both sampled from their continuous formulas:

| Test | Hann | Gaussian |
|---|---|---|
| Tone 0.37 bin off the grid: max \|f̂ − f0\|, bins within 20 dB | 0.019 Hz | 0.089 Hz |
| The same with the opposite sign of the correction | 102 Hz | 164 Hz |
| Impulse 3.1 ms from the center: max \|t̂ − t0\|, bins within 40 dB | 0 | 0 |
| Chirp, 3000 Hz/s: max \|f̂ − IF(t̂)\|, bins within 20 dB | 5e-5 Hz | 1e-3 Hz |

- For comparison, the plain spectrogram spreads the same chirp over 125 Hz
  of bins within 20 dB. Reassignment puts every one of them on the chirp's
  instantaneous-frequency line.
- Not claimed: what reassignment does to two components inside one window
  (it averages them), or to noise (it scatters points). Both are visible in
  speech and are the reason for D17's threshold.

**C19. Delay compensation for causal gammatone plots.** [proof, check]
Step 2 (C10) gave two delays for the 4th-order gammatone: the group delay at
CF, n/(2πb), and the envelope peak, (n − 1)/(2πb). The group delay is the
centroid of the envelope t^(n−1) e^(−2πbt) (the ratio of two gamma
integrals, n/(2πb)); the peak is its mode.

- Check, a click through 23 causal gammatones from 100 Hz to 5 kHz
  (quarter-octave spacing), using the exact response:

| Display | Spread of the envelope peaks across channels |
|---|---|
| Uncompensated | 12.2 ms |
| Each channel shifted by its group delay | 4.2 ms |
| Each channel shifted by its envelope peak | 0.17 ms |

- Centroid check at 1 kHz: centroid / (n/(2πb)) = 1.000.
- The remaining 0.17 ms is the mirror term of C10 at low CFs (the measured
  peak is at most 0.11 ms from the formula).
- Consequence: to draw a click as a vertical line, shift each row by its
  envelope peak, not its group delay.

**C20. Pitch-adaptive schedules are well-conditioned frames if F0 is
bridged.** [check] Schedule: Hann windows k periods long, hop = window /
overlap, on a sentence-like contour (voiced 0.1–0.9 s gliding 180 → 90 Hz,
and 1.2–1.9 s gliding 140 → 100 Hz). The frame operator is diagonal (step 2,
C12), so A/B is the min/max of s(t) over the interior.

| Unvoiced handling | k | Overlap | Frames | n_fft | A/B |
|---|---|---|---|---|---|
| Fixed 20 ms window in gaps | 3 | 4 | 356 | 531 | 0.71 |
| Fixed 10 ms window in gaps | 3 | 4 | 454 | 531 | 0.34 |
| Fixed 20 ms window in gaps | 3 | 3 | 267 | 531 | 0.63 |
| Fixed 20 ms window in gaps | 4 | 4 | 292 | 708 | 0.51 |
| F0 bridged across gaps | 3 | 4 | 339 | 531 | 0.97 |
| F0 bridged across gaps | 4 | 4 | 255 | 708 | 0.98 |
| Constant F0 (reference) | 3 | 4 | | | 1.00 |

- "Bridged" carries F0 through unvoiced stretches, log-linearly between
  the voiced neighbors and held constant before the first and after the last
  voiced frame, so the window length never jumps.
- Synthesis is exact either way (step 2's dual divides by s(t)); A/B only
  bounds how much edited coefficients can be amplified. The jumps are what
  cost conditioning, not the adaptation itself.

## Decisions

All recommended, awaiting review. The ones that most need Cho's answer are
marked **(open)**.

**D12. The sentence. (open)** One sentence, about 2–3 s, a male voice
(F0 around 100–130 Hz), recorded cleanly at 16 kHz or more, redistributable
in the repo.

- Why male: the wideband window must be shorter than a period to show
  pulses, and 5 ms is shorter than T0 only for F0 below 200 Hz. A low voice
  also makes the narrowband panel's resolution limit visible (C16: 17.6 dB at
  100 Hz). A falling F0 through the sentence shows C16's point best.
- Recommended: one utterance from **CMU ARCTIC** (Kominek & Black, 2004),
  speaker `bdl` (US male), 16 kHz. Its licence is a permissive CMU notice:
  use, copy and modify for any purpose, provided the copyright notice,
  conditions and disclaimer are kept, modifications are marked, and the
  authors' names stay. That text was read from a copy redistributed in
  another project, not from festvox.org (which could not be fetched here),
  so it must be confirmed against the COPYING file in the `bdl` download
  before committing audio. `bdl` also has a simultaneous EGG channel
  (secondary sources; see D13).
- Alternatives: LibriSpeech (CC BY 4.0, verified), VCTK (commonly given as
  CC BY 4.0, not verified), the Open Speech Repository's Harvard sentences
  (free for "any reasonable application" with credit, verified); or Cho
  records one sentence himself and dedicates it CC0, the simplest licence.
- It goes in `docs/speech/` with a `SOURCES.md` like `docs/textures/`:
  source, licence, the exact processing (mono, kept at its native rate,
  peak-normalized, 16-bit FLAC).
- The gallery keeps it at its native rate. Upsampling to the gallery's
  44.1 kHz would add an empty band to every panel.

**D13. Where F0 comes from. (open)** Pitch-adaptive and TANDEM panels need
an F0 track; sonore has no F0 estimator.

- Recommended: compute the track **once, offline,** and store it beside
  the sentence as a small text file (time, F0, voiced) with its provenance in
  `SOURCES.md`. The gallery then needs no new dependency, and an F0
  estimator in sonore stays a separate, later decision (README roadmap:
  "robust F0 tracking"). The script that made it goes in `tools/`.
- Source of the track, in order of preference: (a) the **EGG channel** of
  the ARCTIC `bdl` recording, if the download confirms it, by picking the
  glottal closures in the differentiated EGG; this measures the vocal folds
  directly rather than estimating from the sound. (b) Otherwise WORLD's
  Harvest estimator through `pyworld` (MIT licence, from its LICENSE file;
  WORLD itself is modified BSD), run once at dev time.
- Not recommended now: writing a sonore F0 estimator (YIN-style) inside
  step 3. It is a real feature with its own design questions.

**D14. The fixed windows.** Narrowband Hann 33.3 ms and wideband Hann 5 ms
(C15: the classic 45 Hz and 300 Hz analyzing bandwidths), hop 1 ms for both
so the time axes match, n_fft = 1024 (zero-padded) so the narrowband
harmonics are drawn smoothly.

**D15. The pitch-adaptive schedule.** A classmethod
`TVGaborFrame.pitch_adaptive(f0_times, f0, periods=3, overlap=4,
t_end=...)`: Hann windows `periods` F0-periods long, hop = window /
`overlap`, built on `from_function`. Unvoiced stretches are bridged (C20).

- Why 3 periods: it is the shortest whole-period length at which harmonics
  separate clearly (12 dB dip, C16) and the period-rate flicker is gone
  (pulse depth 0). It is also the length WORLD's CheapTrick uses (a Hann
  window of 3 T0, Morise, 2015). It is a constructor argument.
- `f0` uses 0 or NaN for unvoiced frames, the common convention of F0
  trackers.
- The gallery shows it next to the narrowband panel: the two have similar
  window lengths at F0 ≈ 100 Hz, but only the adaptive one keeps the same
  harmonic contrast as F0 rises.

**D16. The TANDEM-style panel.** Two `pitch_adaptive` frames with the same
`n_fft`, their centers offset by −T0/4 and +T0/4, and their powers averaged
at each pair's mid-point (C17). A function, say
`tandem_power(sound, f0_times, f0, periods=2.5, window="blackman")`,
returning a magnitude-only result drawn like the other panels (D18).

- It is magnitude only. Its synthesis would be a union of two frames,
  deferred by step 2's D11, and nothing in step 3 needs it.
- Default Blackman, 2.5 periods: the published TANDEM-STRAIGHT window. It
  shows the spectral envelope with harmonics barely resolved, and the pair
  cancels the flicker almost entirely (C17: worst 0.007, median 0.0014).
- This is a TANDEM-STRAIGHT-*style* power spectrum, not TANDEM-STRAIGHT
  or STRAIGHT: no spectral smoothing, no F0-adaptive envelope, no
  aperiodicity. The gallery text says so.

**D17. The reassigned spectrogram.** `reassigned_spectrogram(sound, frame,
threshold_db=-60)`, where `frame` is a `GaborFrame`. sonore's own
implementation, from three STFTs (w, τw, w′) with C18's formulas. It returns
points (t̂, f̂, power) and draws them by summing power into the display grid.

- Bins more than `threshold_db` below the maximum are dropped: their
  reassigned positions are mostly noise.
- The window derivative is computed for Hann and Gaussian windows from
  their formulas; other windows are refused rather than differentiated
  numerically.
- **(open)** Which windows the gallery reassigns. Recommended: both the
  5 ms and the 33.3 ms windows, drawn beside their plain spectrograms, so a
  reader sees that reassignment sharpens pulses with a short window and
  harmonics with a long one, but does not escape the choice.
- librosa has a reassigned spectrogram (ISC licence). It is a possible
  dev-time cross-check; status: not yet checked.

**D18. A shared display. (open on the frequency axis)** Every panel in the
section is drawn on the same time axis and the same frequency axis, each in
dB re its own maximum, over the same 60 dB range.

- **No resampling.** Each representation is drawn with its own native cells
  (`pcolormesh` with cell edges at mid-points between frame centers or
  between band centers). Interpolating onto one grid would invent detail in
  some panels and blur it in others; shared axes are enough for the eye to
  compare.
- The reassigned spectrogram is the exception: it is a cloud of points, so
  it is binned onto a grid of about one screen pixel (1 ms by 10 Hz).
- Filterbank magnitudes are the existing envelopes, decimated to 1 kHz as
  the other gallery cochleagrams are.
- `TVSTFT` gets a `plot` method (its frame centers are non-uniform), and
  the existing `STFT.plot` and `Envelopes.plot` get the axis options needed
  to match.
- **(open)** The frequency axis. Recommended: **linear, 0–5 kHz,** the
  convention for reading speech, on which harmonics are evenly spaced. The
  alternative is an ERB-number axis, natural for the gammatone and Morlet
  panels but compressing the formant region. The low harmonics that the
  constant-Q panels resolve still occupy the bottom fifth of a linear axis.

**D19. Gammatone delay compensation is display-only.** `plot_envelopes`
(and `Envelopes.plot`) gets `align=None | "peak"`. With `"peak"`, each row is
drawn shifted earlier by its filter's envelope-peak latency (C19), and the
data are untouched.

- The shift is applied through `pcolormesh`'s per-row time coordinates, so
  it is exact (no rounding to the 1 ms display rate).
- The latency comes from the bank: a `GammatoneFilterbank` property giving
  (n − 1)/(2πb) per filter (0 for the edge filters, which are zero-phase).
  Banks without it raise on `align="peak"`.
- Why the envelope peak and not the group delay: C19. A click then appears
  as a vertical line to within 0.2 ms.
- The gallery shows the causal bank both ways, because the uncompensated
  12 ms sweep is what a gammatone is (step 2's D6) and worth seeing once.
- Not claimed: that the Auditory Image Model offers this option. What
  could be checked is that AIM aligns channels later, by strobed temporal
  integration (Patterson et al., 1992), and that its documentation discusses
  filterbank latency without a compensation option. The alignment here is
  sonore's own display choice.

**D20. Gallery layout.** One section, "Seeing speech", with the motivation
above as its introduction and four articles, each playing the same sentence:

1. **Two classic spectrograms:** waveform, wideband (5 ms), narrowband
   (33.3 ms).
2. **Constant-Q and the cochlea:** Morlet (6 cycles), gammatone
   uncompensated, gammatone aligned (D19).
3. **Following the pitch:** narrowband with the F0 track drawn over it,
   pitch-adaptive (Hann, 3 periods), TANDEM-style (Blackman, 2.5 periods).
4. **Reassignment:** the two fixed windows, plain and reassigned (D17).

The introduction states, from numbers computed at build time, that every
panel except the TANDEM and reassigned ones is a frame whose synthesis
returns the sentence to floating-point precision.

**D21. Build cost.** Measured on a 3 s, 16 kHz synthetic voiced signal in
this container: all the analyses together take about 2 s (Morlet and
gammatone at 40 bands and their 1 kHz envelopes are most of it; the
pitch-adaptive analysis is 0.03 s). A four-panel figure with about 10⁶ cells
takes about 3.4 s to render and save. Four articles should therefore add
roughly 15–20 s to the gallery build (about 4 min today). README figures are
not affected unless a step 3 figure is added to the README, which is not
proposed.

**D22. What becomes public API.** Recommended public, each with a README
entry and tests: `TVGaborFrame.pitch_adaptive`, `tandem_power`,
`reassigned_spectrogram`, `TVSTFT.plot`, the `align` option, and the
gammatone latency property. Figure layout stays in `docs/gallery/build.py`.
The sentence and its F0 track live under `docs/`, not in the package.

## Tests (target: under 1 s added)

- `pitch_adaptive`: window lengths equal `periods / F0` at voiced centers;
  bridging leaves no jump; a constant F0 gives A/B = 1 (C20); synthesis is
  exact.
- `tandem_power`: on a pulse train, the period-rate fluctuation is below
  C17's numbers.
- `reassigned_spectrogram`: off-bin tone, impulse and chirp to C18's
  tolerances; the opposite sign fails (guards against a sign slip).
- `align="peak"`: a click through the causal bank peaks within 0.2 ms
  across rows (C19).
- `tests/test_docs.py` already checks README ▶ links; new README entries
  get the same check.
- Invariants: no existing code path changes, and texture synthesis is
  checked bit-for-bit against a worktree of the previous commit.

## Patch plan (after this document is agreed)

1. This document and `tools/check_frames_step3_claims.py`.
2. The sentence, its F0 track and `docs/speech/SOURCES.md` (D12, D13).
3. `TVGaborFrame.pitch_adaptive` and `tandem_power` (D15, D16).
4. `reassigned_spectrogram` (D17).
5. Display: `TVSTFT.plot`, the shared axis options, `align` (D18, D19).
6. The gallery section (D20), with README entries.

## Out of scope for step 3

- An F0 estimator in sonore (D13).
- STRAIGHT or WORLD spectral envelopes, aperiodicity, and resynthesis from
  them.
- Union synthesis (step 2's D11), so no synthesis from the TANDEM pair.
- Synchrosqueezing and other nonlinear sharpening besides reassignment.
- Unions and the CG dual, as in step 2.

## References

Each entry was checked by lookup on 2026-10-01, or is marked otherwise.
Several publisher sites (IEEE Xplore, pubs.aip.org, doi.org) refused
automated access from this environment, so "verified" below means the DOI
and title were seen together on an index or reference page, as noted.

- Auger, F. & Flandrin, P. (1995). Improving the readability of
  time-frequency and time-scale representations by the reassignment method.
  *IEEE Trans. Signal Processing* 43(5), 1068–1089.
  doi:10.1109/78.382394. Verified (ADS record and reference lists). C18.
- Boersma, P. (1993). Accurate short-term analysis of the fundamental
  frequency and the harmonics-to-noise ratio of a sampled sound. *Proc.
  Institute of Phonetic Sciences* 17, 97–110. No DOI. Verified only via a
  secondary listing. Mentioned for D13's later F0 decision.
- de Cheveigné, A. & Kawahara, H. (2002). YIN, a fundamental frequency
  estimator for speech and music. *JASA* 111(4), 1917–1930.
  doi:10.1121/1.1458024. Verified. D13.
- Fulop, S. A. & Fitz, K. (2006). Algorithms for computing the
  time-corrected instantaneous frequency (reassigned) spectrogram, with
  applications. *JASA* 119(1), 360–371. Title, volume and pages verified;
  DOI not verified. The standard reference for reassigned spectrograms of
  speech. D17.
- Kawahara, H., Morise, M., Takahashi, T., Nisimura, R., Irino, T. &
  Banno, H. (2008). TANDEM-STRAIGHT: A temporally stable power spectral
  representation for periodic signals and applications to interference-free
  spectrum, F0, and aperiodicity estimation. *Proc. ICASSP 2008*,
  3933–3936. DOI probably 10.1109/ICASSP.2008.4518514, not verified
  (title and DOI not seen together). C17, D16.
- Kawahara, H. et al. (2011). Technical foundations of TANDEM-STRAIGHT, a
  speech analysis, modification and synthesis framework. *Sādhanā* 36(5),
  713–727. doi:10.1007/s12046-011-0043-3 (DOI from search results; the
  text was read). Source of the Blackman 2.5 T0 window and the ±T0/4 pair.
  Full author list not checked. C17, D16.
- Kodera, K., Gendrin, R. & de Villedary, C. (1978). Analysis of
  time-varying signals with small BT values. *IEEE Trans. ASSP* 26(1),
  64–76. doi:10.1109/TASSP.1978.1163047. Verified. The origin of
  reassignment (their 1976 paper is not verified). C18.
- Koenig, W., Dunn, H. K. & Lacy, L. Y. (1946). The sound spectrograph.
  *JASA* 18(1), 19–49. doi:10.1121/1.1916342. Verified; the text was read
  through a summarizing tool, which reported the 45 Hz (3 dB) narrow and
  300 Hz wide filters. See C15's caveat. C15, D14.
- Kominek, J. & Black, A. W. (2004). The CMU Arctic speech databases.
  *Proc. 5th ISCA Speech Synthesis Workshop (SSW5)*, 223–224.
  https://www.isca-archive.org/ssw_2004/kominek04b_ssw.html. Verified.
  D12. The EGG channel for `bdl`, `slt` and `jmk` is from secondary sources
  (pyroomacoustics documentation; an Interspeech 2020 paper), not verified
  against the corpus itself.
- Morise, M. (2015). CheapTrick, a spectral envelope estimator for
  high-quality speech synthesis. *Speech Communication* 67, 1–7.
  doi:10.1016/j.specom.2014.09.003. Verified. Hann window of 3 T0. D15.
- Morise, M., Yokomori, F. & Ozawa, K. (2016). WORLD. See
  docs/design/frames-step2.md. D13.
- Patterson, R. D. et al. (1992). Complex sounds and auditory images. See
  docs/design/frames-step2.md. D19.
- Patterson, R. D., Allerhand, M. H. & Giguère, C. (1995). Time-domain
  modeling of peripheral auditory processing: A modular architecture and a
  software platform. *JASA* 98(4), 1890–1894. doi:10.1121/1.414456.
  Verified from a reference list; the paper itself was not read, so nothing
  about AIM's delay handling is claimed from it. D19.

## Reference implementations

- **WORLD** (github.com/mmorise/World; modified BSD, verified from its
  README). Harvest is the fallback F0 source (D13); CheapTrick fixes D15's
  default window length. Status: consulted; not run against sonore.
- **pyworld** (github.com/JeremyCCHsu/Python-Wrapper-for-World-Vocoder):
  MIT, verified from the repository's LICENSE file; the PyPI metadata was
  not checked. Would be a dev-time dependency of one `tools/` script only.
  Status: not yet used.
- **librosa** `reassigned_spectrogram` (ISC licence, verified): a possible
  dev-time cross-check for D17. Status: not yet checked.
- **TANDEM-STRAIGHT:** its own code was not looked for. The pair in D16 is
  implemented from the published description. Status: not checked against
  it.
