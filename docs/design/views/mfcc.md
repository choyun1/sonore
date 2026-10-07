# Mel-frequency cepstral coefficients (MFCCs)

The design of an MFCC view in sonore: the standard speech feature that
summarizes the spectral envelope of each time window in about thirteen
numbers. Cho asked on 2026-10-02 whether sonore has MFCCs. It does not:
there is no mel scale, mel filterbank, DCT feature or delta feature in
`src/` (searched for `mfcc`, `mel`, `dct`, `htk` and `slaney`; the only
mention is `views/cepstrum.md`, which put "Mel-cepstra and MFCCs" out of scope,
and a sentence on the cepstrum gallery page).

Status: accepted 2026-10-02. Cho accepted D1–D9 as recommended and chose
HTK as the primary reference (D10). HTK's site returned 502 errors that
day, so Cho chose Kaldi in its place (D11) and then dropped HTK.
Implemented in `src/sonore/views/mfcc.py`, tested in
`tests/views/test_mfcc.py` against Kaldi's and librosa's stored output
(C10, C9). The claims are checked by
`tools/check_mfcc_claims.py`; `tools/crosscheck_mfcc.py` compares the
recipes with librosa and python_speech_features and with sonore's
CheapTrick envelope. Cho asked on 2026-10-02 that the tests match
librosa: `tools/make_mfcc_fixtures.py` stores librosa's output in
`tests/data/librosa_mfcc_reference.npz`, and C9 and D9 say how the tests
use it.

## Why

MFCCs are the most widely used description of a speech spectrum: speech
recognition used them for decades, speaker identification and many
machine-learning front ends still do, and a reader who meets "MFCC" in a
paper should be able to compute one in sonore, see what it keeps, and
compare it with the envelopes sonore already has. The pieces they relate
to exist:

- **`so.Cepstrum`** (`views/cepstrum.md`): the real cepstrum, the inverse DFT of
  the log magnitude on linear frequency. An MFCC is the same idea on a
  warped, coarsely sampled frequency axis (C1).
- **`ERBFilterbank`** (`frames/frames.md`): sonore's auditory frequency axis. The
  mel scale is a close relative (C2).
- **`so.cheaptrick`** (`views/world.md`): an F0-adaptive spectral envelope. MFCCs
  are a fixed-resolution envelope summary and depend on F0 where
  CheapTrick largely does not (C5).

## How the claims are verified

As in the other design documents, each claim is numbered and tagged:

- **[proof]**: a short argument given here.
- **[check]**: a number printed by `tools/check_mfcc_claims.py`. The script
  uses only NumPy, SciPy and soundfile (to read the gallery sentence),
  writes every scale, filter and transform out from its formula, and
  shares no code with sonore. It runs in under half a second. The numbers
  below come from NumPy 2.4.6 and SciPy 1.17.1.
- **[cross]**: a number printed by `tools/crosscheck_mfcc.py`, which uses
  librosa 0.11.0 and python_speech_features 0.6 (development-time only,
  never dependencies) and `so.cheaptrick`.
- **[source]**: a published result or a library's documented behavior
  (see References; which ones were checked is said there).

The sentence is `docs/speech/bdl_arctic_a0131.flac` (16 kHz). "The speech
recipe" below means 25 ms Hamming time windows every 10 ms, a 512-point
FFT, power spectrum, 26 HTK-mel triangles from 0 to 8 kHz, natural log,
orthonormal DCT-II, 13 coefficients.

## What an MFCC is, and its variants

For one time window with power spectrum P[k]:

1. **Mel filterbank.** M triangular weights W[m, k], with feet and peaks
   equally spaced on a mel scale between f_lo and f_hi. Band power
   E[m] = Σ_k W[m, k] P[k].
2. **Log.** L[m] = ln E[m] (some libraries use 10 log10).
3. **DCT.** c[n] = Σ_m L[m] · α_n cos(π n (m + ½) / M), n = 0..N−1, with
   the orthonormal α_n. Keep the first N (13 is the usual number, c0 to
   c12).
4. Optionally: pre-emphasis of the sound before step 1, a "lifter" that
   rescales c[n], c0 replaced by the log energy of the time window, and
   deltas (time slopes of each coefficient).

The recipe comes from Davis & Mermelstein (1980). The libraries that
people actually compare against differ in almost every step [source,
cross]:

| Step | HTK (Young et al.) | python_speech_features 0.6 | librosa 0.11 (`feature.mfcc`) |
|---|---|---|---|
| Time windows | 25 ms Hamming, 10 ms hop | 25 ms **rectangular**, 10 ms, zero-padded end | 2048-point Hann every 512 samples, centered with zero padding |
| Pre-emphasis | 0.97 | 0.97 | none |
| Spectrum | magnitude by default (`USEPOWER=F`) | power / n_fft | power |
| Mel scale | 2595 log10(1 + f/700) | same | Slaney: linear below 1 kHz, log above |
| Triangles | height 1, straight in mel | height 1, **feet rounded to FFT bins** | area 1 (`norm="slaney"`), exact frequencies |
| Bands | 26 (typical config) | 26 | 128 |
| Log | natural | natural | 10 log10, then clipped **80 dB below the loudest cell of the whole sound** |
| Coefficients | 12 + c0 or energy | 13, c0 replaced by log energy | 20, c0 kept |
| Lifter | 22 | 22 | none |
| Deltas | regression over ±2, edges repeated | same (`delta(feat, 2)`) | Savitzky–Golay, width 9 |

The HTK column is from memory of the HTK Book and was not checked; the
other two are checked to rounding error by the crosscheck (C8). Kaldi and
torchaudio's `compliance.kaldi` follow HTK closely (from memory, not
checked).

## Claims

**C1. An MFCC is a real cepstrum of the log mel spectrum.** [proof,
check] Mirror the M log band powers to 2M values (L, then L reversed).
That sequence is real and even about a half sample, as a real log
spectrum is even about 0 and Nyquist, and its DFT, shifted by half a
sample, is exactly the (unnormalized) DCT-II of L. So the DCT is the
cepstrum's inverse DFT for a log spectrum sampled on M mel bands, and
low-order coefficients are the slow ripples of that log spectrum, as in
`so.Cepstrum` (`views/cepstrum.md` C5). Checker: the two agree to 3.6e-15, with
an imaginary part of 8.8e-16. With the orthonormal scaling the DCT matrix
is orthogonal (error 4.4e-16), so sums of squared coefficient differences
equal sums of squared log-band differences (used in C4 and C5).

**C2. The HTK mel scale is ERB-like with a higher break frequency; the
two mel scales differ by up to about 260 Hz in band centers.** [proof,
check] HTK's mel is 2595 log10(1 + f / 700) and the ERB number is
21.4 log10(1 + f / 228.8): both are linear well below their break
frequency and logarithmic well above it, so the mel scale stays linear up
to a higher frequency. Normalized to 0..1 over 0–8 kHz, they differ by at
most 0.12; HTK's and Slaney's mel differ by at most 0.048, and with 26
bands from 0 to 8 kHz their band centers differ by up to 262 Hz. A
height-1 triangle has an equivalent rectangular width of half its base;
for 26 HTK bands over 0–8 kHz this is 1.76 ERB at 226 Hz, 1.41 at 525 Hz,
1.22 at 921 Hz, 1.06 at 1.9 kHz and 0.97 at 3.8 kHz. So the speech recipe
is coarser than the auditory filters at low frequencies and about equal
above 2 kHz.

**C3. Height-1 triangles sum to one; the choice between height-1 and
area-1 triangles only adds a fixed vector to the MFCCs.** [proof, check]
Between the first and last band centers, each frequency lies on the
falling side of one triangle and the rising side of the next, and the two
weights add to 1 (checker: exactly 0 error). Area normalization multiplies
band m by a constant 2 / (f_{m+2} − f_m); after the log that is a constant
added to L[m], and after the DCT a constant vector added to every time
window's coefficients. Checker on the sentence: the shift's standard
deviation over time windows is 1.4e-13 (it is constant), and its size is
up to 27.7. So deltas and differences between sounds are identical under
either choice; the coefficients themselves are not.

**C4. MFCCs are a one-way view: many spectra give the same coefficients.**
[proof, check] The mel matrix maps 257 power-spectrum bins to 26 band
powers (rank 26), and 13 of 26 DCT coefficients are kept. In the loudest
time window of the sentence, the checker changes every bin's power by a
factor between 0.1 and 1.9 in a direction no mel band sees: the MFCCs
change by at most 1.1e-15, while the median bin changes by 2.2 dB and 43%
of the bins by more than 3 dB. Keeping 13 of the 26 coefficients loses a
further 1.8 dB rms of the log mel spectrum in the median time window
(2.5 dB at the 90th percentile). A time window keeps 13 numbers where its
power spectrum had 257 (5%), and no phase.

**C5. The MFCCs of a vowel depend on its F0 by about a third as much as
on the vowel, because the low mel bands are narrower than the harmonic
spacing; CheapTrick's envelope cuts that to about an eighth.** [check,
cross] Two synthetic vowels (/a/ and /i/, four-formant all-pole filters
with Peterson & Barney's male formants, flat harmonic source) at F0 100,
150, 200, 250 and 300 Hz, one 25 ms Hamming time window each. Distance:
the rms over the 26 bands of the difference between the 13-coefficient
smoothed log mel spectra, c0 (level) excluded, in dB (C1 makes this a
coefficient distance). /a/ and /i/ at the same F0 are 15.4–16.0 dB apart.
The same vowel at two F0s is 5.6 dB apart in the median and up to 9.2 dB.
The lowest HTK band centers are 68 Hz apart and stay closer than 200 Hz
up to 1.4 kHz, so below that each band catches zero, one or two harmonics
depending on F0. Taking the band powers from `so.cheaptrick`'s envelope
instead of the power spectrum (crosscheck), the same-vowel distances fall
to 1.9 dB median and 3.3 dB at most, with /a/ and /i/ still 16 dB apart.
This is the didactic point of the comparison proposed for the gallery
(D8). It is measured on synthetic vowels with one window position each;
how it carries over to recordings is not measured here.

**C6. Deltas are a band-pass modulation filter peaking near 14 Hz at a
10 ms hop.** [proof, check] The regression formula
d_t = Σ_{n=−2..2} n c_{t+n} / Σ n² is exactly the least-squares slope over
five time windows (checker: 3.3e-16). As a filter on the coefficient
track, its gain matches a true derivative at slow rates (0.998 of it at
1 Hz), peaks at 13.8 Hz with 8.7 times the 1 Hz gain, is above half power
from 6.9 to 21.0 Hz, and is zero at 50 Hz. So deltas pick out the
syllable-to-phoneme modulation range that `so.ModulationSpectrogram`
displays (`views/modulation-spectrogram.md`), at a fixed rate set by the hop.

**C7. dB MFCCs are natural-log MFCCs times 10 / ln 10, unless the log is
floored relative to the loudest cell.** [proof, check, cross]
10 log10 x = (10 / ln 10) ln x, and the DCT is linear (checker: ratio
4.343 to 6e-12). librosa's `power_to_db` default raises every mel cell to
at least 80 dB below the loudest cell of the whole sound: on the sentence
this touches 3.7% of the cells (leading and trailing silence) and changes
those time windows' coefficients by up to 162, 19% of the coefficients'
range. So a coefficient depends on the loudest moment of the whole
recording, not only on its own time window.

**C8. The recipes are reproduced to rounding error.** [cross] With the
checker's formulas and each library's settings: `librosa.feature.mfcc`
with its defaults to 1.3e-9 of the largest coefficient;
`librosa.filters.mel` to 5e-8 (Slaney) and 3e-8 (HTK, `norm=None`), the
float32 rounding of librosa's weights; `python_speech_features.mfcc` with
its defaults exactly (0). python_speech_features rounds the triangle feet
to FFT bins; with exact triangles instead, c1..c12 on the sentence change
by 13% rms (weights differ by up to 0.25).

**C9. sonore's STFT has librosa's time windows, so the whole pipeline
can be tested against librosa's output, deltas included.** [cross] A
`GaborFrame` with a Hann window of librosa's `win_length`, its hop and its
`n_fft` gives the same power spectra as `librosa.stft` (centered, zero
padding) with a difference of exactly 0, for all three stored settings.
sonore's grid has one more time window, centered one hop before the first
sample, and one or two more at the end; dropping those leaves librosa's
grid. On that grid the checker's formulas reproduce the stored mel power
to at most 5.1e-8 of its maximum (librosa's float32 filter weights) and the
stored MFCCs to at most 1.0e-9 of the largest coefficient, on the 0.8 s
excerpt the fixture holds. `librosa.feature.delta` is
SciPy's `savgol_filter` with `deriv=order`, `polyorder=order` and
`mode="interp"`; called that way it reproduces the stored deltas exactly
(0) for widths 5 and 9 and orders 1 and 2. Its order-1 width-5 deltas
equal HTK's regression over ±2 (C6) in the interior (3.6e-14) but not in
the first and last two time windows (up to 5.0 dB units on the excerpt), where librosa uses the
slope of a line fitted to the first or last five, and HTK repeats the
edge values. Its order 2 is the second derivative of a local quadratic,
not the delta of the delta.

**C10. With triangles straight in mel, `so.MFCC` reproduces Kaldi's
MFCCs to float32 precision; straight-in-Hz triangles change them by
0.3%.** [cross] Kaldi's MFCCs (`compute-mfcc-feats`, through
kaldi-native-fbank 1.22.3, a C++ re-implementation of Kaldi's feature
code) on the fixture's excerpt at 16-bit integer scale, dither off.
Kaldi's time windows start at multiples of the hop (`snip-edges`) where
sonore's are centered on them, so the sound is trimmed by 40 samples (half
the 400-sample window, modulo the 160-sample hop) and Kaldi's first time
window, which needs the trimmed samples, is left out. With no DC removal,
no pre-emphasis, a Hamming window, 26 bins from 0 Hz, no lifter and c0
kept, the MFCCs agree to 1.4e-7 of the largest coefficient and the log mel
energies to 3.1e-5 (in natural-log units of values near 25). With
Kaldi's "povey" window (a symmetric Hann window to the power 0.85), 23
bins from 20 Hz and lifter 22, they agree to 2.4e-6. Kaldi, like HTK,
draws each triangle as straight lines in mel, not in Hz; with straight-in-Hz
triangles (librosa's) the coefficients c1–c12 differ by 0.28% rms, at
most 0.04. Kaldi's own defaults add DC removal, pre-emphasis inside each
time window and the raw log energy in place of c0; against those,
sonore's c1–c12 differ by 43% rms. That is the cost of D3 (pre-emphasis
belongs to the sound), and it is measured here, not tested.

## Views, and what this one drops

By `philosophy.md` ("Views may discard information"), MFCCs are a view
built on a frame (the STFT), not a frame. They drop:

- **phase**, entirely;
- **harmonic and fine spectral detail**: everything inside a mel band,
  and, with 13 coefficients, the faster ripples of the log mel spectrum
  (C4). This is the point of the feature: it is meant to keep the
  envelope and not the pitch, though it does so imperfectly (C5);
- **the level**, which goes into c0 alone (or is replaced by the log
  energy);
- **anything between time windows** beyond what the 10 ms hop samples.

A sound is recoverable only approximately and by a model, not by search
through an exact frame: the usual route (librosa's `mfcc_to_audio`)
inverts the DCT, takes a non-negative least-squares power spectrum under
the mel matrix, and runs Griffin–Lim, giving a whispery, buzz-free
version. sonore would say this in the class docstring and not provide a
`to_sound` in this step (D6).

## Decisions

Each recommendation follows the strongest form of each alternative.

**D1. One new type, `so.MFCC`, that dispatches on its input. (accepted 2026-10-02)** A new
module `sonore/analysis/mfcc.py` with `MFCC(source, ...)`:

- `source` a `Sound`: builds the speech recipe's analysis itself (a
  `GaborFrame` with a 25 ms Hamming window, 10 ms hop and an FFT length
  rounded up to a power of two), so `so.MFCC(snd)` gives the familiar
  numbers in one line;
- `source` an `STFT` or `TVSTFT`: uses its power, so any window, hop or
  pitch-adaptive analysis can be summarized, as `so.Cepstrum(coefs)`
  does. Dispatch by `match` on type, as `noise_vocode(carrier=...)`
  already does.

It keeps `mel_power` (shape `(n_channels, n_mels, n_windows)`), `data`
(the coefficients, `(n_channels, n_mfcc, n_windows)`), `t`, and the band
edges, so the log mel spectrogram, the other feature in common use, needs
no second type.

Alternatives, at their best:

- **A mel option on `Cepstrum`** (`so.Cepstrum(coefs, bands="mel")`). By
  C1 this is the same operation, and one class for "cepstrum" is
  elegant. But every `Cepstrum` method assumes a full linear-frequency
  log spectrum: `lifter` cuts in seconds of quefrency (meaningless on mel
  bands), `to_stft` puts the original or minimum phase back on 257 bins
  (impossible from 26 bands), and `f0` reads the pitch peak (removed by
  the mel smoothing). Half the class would have to raise. Not
  recommended, but the docstrings say the relation in both directions.
- **Two types, `MelSpectrogram` and `MFCC`** (librosa's split). Clean, and
  the log mel spectrogram is a common input in its own right. But the
  second type is one DCT on top of the first; keeping `mel_power` on
  `MFCC` gives the same thing with one name. Could be split later if a
  use appears.
- **A function `so.mfcc(snd) -> ndarray`**, as in every library. Simplest,
  but loses the times, the band edges and the plot, which every other
  sonore view keeps.
- **A tight mel filterbank** (`MelFilterbank`, a `CosineFilterbank` on the
  mel scale, two lines of code) and MFCCs from its subband energies. It
  would be sonore's own exact-inverse analysis, but its time resolution
  differs per band and it reproduces no published MFCC. Worth adding on
  its own if wanted for displays; not the MFCC route.

**D2. Defaults follow the speech recipe; parameters reproduce librosa
and python_speech_features. (accepted 2026-10-02)** Defaults: 25 ms Hamming, 10 ms hop, power,
`n_mels=26`, `n_mfcc=13`, `f_lo=0`, `f_hi=fs/2`, `mel_scale="htk"`,
`triangles="height"` (or `"area"`), natural log, orthonormal DCT-II, c0
kept, no lifter, no pre-emphasis. Each library's output is reached by
stating its settings, and tests check the mel, log and DCT stages against
the checker's formulas on the same power spectra, as `views/world.md` reproduces
WORLD. The alternative, librosa's defaults, is what many Python users
will compare with; but 128 bands and 2048-point windows at 16 kHz (128 ms)
are a music-analysis choice, and its log floor makes coefficients depend
on the whole sound (C7). A recipe for "the librosa numbers" goes in the
docstring.

**D3. Pre-emphasis is not an argument. (accepted 2026-10-02)** It is a fixed first-order
filter on the sound, y[n] = x[n] − 0.97 x[n−1], so it is applied to the
sound (one `scipy.signal.lfilter` line in the docstring's recipe; sonore
has no filter method on `Sound` today), and the class analyzes what it is
given. Alternative: a `preemphasis=0.97` argument as in
python_speech_features. Convenient, but it hides a change to the sound
inside a view, and librosa does without it.

**D4. Natural log with a floor relative to each channel's loudest band
power, −200 dB by default, as in `Cepstrum` (D3 there). (accepted 2026-10-02)** Scaling the
sound then moves only c0, and silent time windows get flat, finite
coefficients. `.db` gives the dB version (C7). `floor_db=-80` reproduces
librosa's clipping when that is wanted. Alternative: librosa's 80 dB clip
as the default; not recommended because of C7.

**D5. Triangles on exact frequencies, not rounded to FFT bins; straight
in mel by default. (accepted 2026-10-02; the default axis amended by D11)** That is
the definition and librosa's construction. python_speech_features' bin
rounding changes c1..c12 by 13% rms (C8); it is listed as a difference in
the docstring and reproducible in the crosscheck, not offered as an
option.

**D6. c0, lifter, deltas, and no `to_sound`. (accepted 2026-10-02)**

- c0 is kept as the DCT of the log band powers (with orthonormal scaling,
  √M times their mean). Replacing it with the log energy of the time
  window is one line for a user and is not built in.
- `lifter=0` by default; `lifter=22` applies HTK's
  1 + (L/2) sin(π n / L). It is a fixed gain per coefficient, so it
  changes Euclidean distances and nothing else.
- `mfcc.deltas(order=1, width=5)` computes deltas exactly as librosa
  does (Savitzky–Golay, `mode="interp"` at the edges), so they match it
  (C9). With width 5 and order 1 this is HTK's regression over ±2 (C6)
  except in the first and last two time windows; `width=9` gives
  librosa's default. Alternative: HTK's regression with edges repeated,
  as python_speech_features does. It differs from librosa only at the
  edges, and matching librosa, which Cho asked for, decides it.
- No `to_sound` (see "Views"). `mfcc.envelope(f)` returns the smoothed
  power implied by the kept coefficients (inverse DCT, then linear in mel
  between band centers) at frequencies `f`, for plotting next to an
  envelope; its docstring says it is a display, not a spectrum estimate
  (C4).

**D7. Display. (accepted 2026-10-02)** `mfcc.plot()` draws coefficient index against time;
`mfcc.plot(kind="mel")` draws the log mel spectrogram with band centers
labeled in Hz. Both via `sonore.plotting`. README: a module-table row,
a short recipe, Davis & Mermelstein (1980) in the references.

**D8. Gallery. (accepted 2026-10-02; built 2026-10-02, all three parts)** A section "MFCCs: a
cepstrum on the mel scale" on the cepstral analysis page
(`docs/gallery/voice/cepstrum.py`, which already mentions mel-cepstra):

1. The synthetic /a/ at F0 100, 200 and 300 Hz: the 13-MFCC envelope and
   the CheapTrick envelope over the true filter response, showing the
   MFCC envelope moving with F0 and CheapTrick's not (C5).
2. The gallery sentence: the log mel spectrogram and the MFCCs, beside
   CheapTrick's envelope on the same mel bands.
3. Optionally, C4's two spectra that differ by several dB in 43% of their
   bins and give identical MFCCs.

As built (figures m1 to m3): part 1 compares the 13 MFCCs of the power
spectrum with the 13 MFCCs of CheapTrick's envelope summed into the same
bands, which is what C5 measures, rather than the raw CheapTrick envelope;
the page prints C5's distances (median 5.6 dB, largest 9.2, against 1.9 and
3.4 from CheapTrick; /a/ to /i/ 15.4 to 18.4 dB). Part 2 adds the band
powers the 13 coefficients keep (`envelope` at the band centers). Part 3
uses a different random direction in the null space from the checker's, so
30% of the bins change by more than 3 dB (median 1.9 dB) instead of 43%;
the MFCCs still agree to 9e-16.

**D9. Tests match librosa. (accepted 2026-10-02, at Cho's request)** The tests compare
`so.MFCC` with librosa's own output, stored by
`tools/make_mfcc_fixtures.py` in `tests/data/librosa_mfcc_reference.npz`
(140 KiB) for three settings on 0.8 s of the gallery sentence (0.5 to
1.3 s, stored in the file as 16-bit samples so the tests run from the
sdist, which has no docs folder): librosa's
defaults, the speech recipe's sizes with HTK mel and height-1 triangles,
and the same sizes with Slaney mel and 40 bands. It holds the mel power
and the MFCCs, and for the HTK setting the deltas of width 5 and 9,
orders 1 and 2. The tests run `so.MFCC` on a Hann STFT with librosa's
sizes, drop the extra edge time windows (C9), and compare: mel power to
3e-7 of its maximum (float32 weights are off by up to about 6e-8 of
themselves), `mfcc.db` with `floor_db=-80` to 1e-8 of the largest
coefficient, deltas computed from the stored MFCCs to 1e-12, and the
method's deltas on sonore's grid away from the ends to 1e-8 of the
largest coefficient. This follows the WORLD tests, which
compare against stored pyworld output (`tools/make_world_fixtures.py`),
so the test suite needs no librosa. Alternative: call librosa in the
tests, skipped when it is not installed (`pytest.importorskip`). That
always tests the installed version, but CI would have to install
librosa (and its numba dependency) to run the tests at all, and a test
that is usually skipped protects nothing. The fixture records which
librosa version it came from, and regenerating it is one command.

**D10. HTK was to be the primary reference; dropped. (Cho's decisions,
2026-10-02)** HTK's HCopy fixed the MFCC recipe that speech recognition
uses, so Cho first chose it as the primary reference. HTK needs
registration and may not be redistributed, so it would have been run by
hand and its output stored. Its site returned 502 errors to Cho through
the afternoon (a web fetch from here loaded it once, at 18:47), and Cho
dropped it: the tests compare with Kaldi, which follows HTK closely
(D11, C10), and with librosa (D9, C9). How HCopy itself differs from
`so.MFCC` is not checked; the HTK column of the variants table is from
memory.

**D11. Kaldi stands in for HTK; the defaults follow it. (Cho's decision,
2026-10-02)** HTK's site returned 502 errors, so Cho chose Kaldi, the
other speech-recognition standard, which follows HTK closely. Kaldi itself
is a large C++ build; kaldi-native-fbank (k2-fsa) re-implements its
feature code, installs with pip, and is what `tools/make_kaldi_fixtures.py`
runs, storing `tests/data/kaldi_mfcc_reference.npz` (21 KiB). It is a
re-implementation, not Kaldi, which the README says. Following the primary
reference changed two defaults, both small (C10):

- triangles are straight in mel (`triangle_axis="mel"`, as HTK and
  Kaldi), with `triangle_axis="hz"` for librosa. Alternative: keep
  straight-in-Hz triangles as the default and add the mel option. That
  would make the default reproduce librosa and not the reference Cho
  chose.
- a `Sound` is analyzed with the symmetric Hamming window HTK and Kaldi
  use (`sonore.views.mfcc.symmetric_hamming`), not SciPy's periodic one.

The tests compare with Kaldi in two settings (plain, and Kaldi's window,
bins, low frequency and lifter), to 1e-6 and 1e-5 of the largest value,
which leaves about seven and four times C10's float32 differences. Kaldi's
DC removal, in-window pre-emphasis and energy in place of c0 are not
reproduced (D3, D6). A `kaldi_front_end` option that would add the three
was offered; Cho declined it on 2026-10-02, so they stay a documented,
measured difference.

## API sketch

```python
snd = so.load("docs/speech/bdl_arctic_a0131.flac")
mfcc = so.MFCC(snd)  # speech recipe: 13 x n_windows
mfcc.data.shape  # (1, 13, n_windows)
mfcc.mel_power  # (1, 26, n_windows), the mel spectrogram
d1 = mfcc.deltas()  # same shape, computed as librosa.feature.delta

coefs = so.STFT(snd, win_dur=0.040, hop_dur=0.005)
so.MFCC(coefs, n_mels=40, mel_scale="slaney", triangles="area")

f0 = so.f0_track(snd)
env = so.cheaptrick(snd, f0)
mfcc.envelope(env.f)  # for plotting against env
```

## Tests (target: under 1 s added)

- Against librosa's stored output (D9): mel power, MFCCs and deltas for
  three settings.
- Against Kaldi's stored output (D11, C10): MFCCs and log mel energies.
- The mel weights, log and DCT against the checker's formulas for both
  scales and both triangle normalizations (C3, C8).
- `MFCC(snd)` equals the formula pipeline on the same power spectra.
- Scaling a sound moves only c0 (D4); a silent channel is finite.
- Area against height triangles: a constant shift, identical deltas (C3).
- Deltas equal the least-squares slope in the interior (C6).

## Patch plan

1. This document, `tools/check_mfcc_claims.py`, `tools/crosscheck_mfcc.py`,
   `tools/make_mfcc_fixtures.py` and the librosa fixture. Done.
2. `MFCC`, `deltas`, `envelope`, `plot`, tests against librosa, README row,
   CHANGELOG. Done.
3. Kaldi through kaldi-native-fbank: fixture, tests, triangle axis and
   window defaults (D11). Done.
4. The gallery section (D8), built separately.

## Out of scope

- Other cepstral features: PLP, gammatone cepstra (GFCC), mel-generalized
  cepstra used in speech synthesis.
- Resynthesis from MFCCs (D6).
- Speaker or speech recognition, and any listener model (`philosophy.md`;
  the project does not touch listener models for now).

## References

None of these was verified by lookup for this document; they are cited
from memory, apart from the library behavior, which the crosscheck
measures (C8).

- Davis, S. B. & Mermelstein, P. (1980). Comparison of parametric
  representations for monosyllabic word recognition in continuously
  spoken sentences. *IEEE Trans. Acoust., Speech, Signal Process.* 28(4),
  357–366. The MFCC recipe.
- Stevens, S. S., Volkmann, J. & Newman, E. B. (1937). A scale for the
  measurement of the psychological magnitude pitch. *J. Acoust. Soc. Am.*
  8(3), 185–190. The mel scale.
- O'Shaughnessy, D. (2000). *Speech Communications: Human and Machine*,
  2nd ed. IEEE Press. Eq. 4.2, p. 128, prints 2595 log10(1 + f/700) with no
  earlier source; the 1987 first edition is the one usually cited (not read).
- Slaney, M. (1998). *Auditory Toolbox*, version 2. Technical Report
  1998-010, Interval Research Corporation. The linear-then-log mel and
  area-normalized triangles used by librosa.
- Young, S. et al. (2006). *The HTK Book* (version 3.4). Cambridge
  University Engineering Department. HTK's defaults in the table.
- Furui, S. (1986). Speaker-independent isolated word recognition using
  dynamic features of speech spectrum. *IEEE Trans. Acoust., Speech,
  Signal Process.* 34(1), 52–59. Delta features.
- Glasberg, B. R. & Moore, B. C. J. (1990). Derivation of auditory filter
  shapes from notched-noise data. *Hearing Research* 47, 103–138. The ERB
  scale (C2); already cited in the README.
- Morise, M. (2015). CheapTrick, a spectral envelope estimator for
  high-quality speech synthesis. *Speech Communication* 67, 1–7. Already
  cited in `views/world.md`.
- Peterson, G. E. & Barney, H. L. (1952). Control methods used in a study
  of the vowels. *J. Acoust. Soc. Am.* 24(2), 175–184. The checker's
  formants.
- McFee, B. et al. librosa, version 0.11.0; Lyons, J.
  python_speech_features, version 0.6. Measured in C8.
