# Decimated filterbanks: an invertible constant-Q transform

The roadmap item "a decimated, invertible constant-Q transform
(nonstationary Gabor frames in frequency)", and where it goes in the
package and the gallery.

Status: accepted 2026-10-28. Cho accepted D1–D9 as recommended, with D1's
class placed in `frames/filterbank.py` rather than a file of its own. The claims are checked by
`tools/check_constant_q_claims.py` (C1–C9). It writes the decimated
transform out from its definition in NumPy and imports sonore only for the
filter responses and for the timing comparison, so it shares no code with
an implementation that does not exist yet. Every number below is what it
printed on 2026-10-28 (rerun after a fix: the first version rounded a few
signed bin indices down, moving a few response values by one bin; no
conclusion changed) for 3 s of white noise at 44.1 kHz, unless stated.

## Why

`cosine_filterbank(scale="cents", spacing=100)` is already an exact
constant-Q filterbank: 83 semitone bands from 55 Hz to 7040 Hz plus the
lowpass and highpass, tight, with an exact inverse. The chroma design
(`views/chroma.md`, D5) accepted it as sonore's constant-Q transform and
left the decimated version on the roadmap. Its cost is that it is
undecimated: every band keeps one real number per sample, so 3 s of sound
becomes 85 times as many numbers.

A band whose filter spans only a few hertz does not need a sample every
1/44100 s. The nonstationary Gabor frames of Balazs et al. (2011) and the
constant-Q transform built from them by Velasco et al. (2011) keep each
band at its own rate, low bands slowly and high bands quickly, and stay
exactly invertible. The filterbanks note (`frames/filterbanks.md`) named
this as the direction and kept `hop` out of `Filterbank` so it could be
added later (its D12). This note proposes how.

## What the mathematics says

Take a signal of `N` samples, its DFT `X`, and a filter `g_k` that is zero
outside an arc of `L_k` bins (its support). Choose `M_k >= L_k`, place
`X g_k` on its support into `M_k` bins by bin index modulo `M_k`, and take
the inverse FFT of length `M_k`. Because no two bins of the support land
in the same place, nothing is aliased, and the FFT gives the folded values
back exactly. Synthesis unfolds them, multiplies by the dual filter
`conj(g_k) / s` and sums, where `s = sum_k |g_k|^2` is the same frame
operator `Filterbank` already measures. This is the "painless" case of
Balazs et al. (2011), here in frequency; `TVGaborFrame` is the same case
in time. Scaling each band's coefficients by `sqrt(M_k / N)` keeps the
frame bounds of the undecimated bank, so a tight bank stays tight.

Band `k`'s coefficients are `N / M_k` samples apart. When that is a whole
number, they are the undecimated complex band sampled every `N / M_k`
samples (C6). The decimated transform is a filterbank with a hop per band,
not a new kind of analysis.

## Claims

- **C1. The filters are today's.** The transform takes the responses of
  any `Filterbank` unchanged. On the two-sided DFT grid the semitone bank
  (83 bands) has `max |s - 1|` = 4.4e-16.
- **C2. Decimated to each filter's support, it is still exact and still
  tight.** The reconstruction error is 8.5e-16 of the peak, the largest
  imaginary part of the result is 2.9e-15, the coefficient energy equals
  the signal energy to rounding (the checker prints 0), and synthesis by
  the dual and by the adjoint differ by 2.9e-16 of the peak.
- **C3. A real signal needs only the lowpass, the positive halves of the
  bands and the highpass.** Each band's negative-frequency half only adds
  the complex conjugate. Rebuilding from the edges plus twice the real part
  of the positive bands has an error of 7.5e-16.
- **C4. About 2 real numbers per sample, against 85.** The undecimated
  semitone bank keeps 85 N real numbers. Decimated, it keeps 2.02 N: 0.62 N
  for the 83 bands, 1.40 N for the highpass (the flat edge from 7 kHz to
  22 kHz is a wide band), and 0.005 N for the lowpass. Extending the bank
  to 20 kHz (101 bands) gives 2.05 N. With 48 bands per octave instead of
  12 the undecimated bank keeps 337 N and the decimated one 2.01 N: every
  frequency lies under two cosines whatever the spacing, so the count
  hardly depends on it.
- **C5. Hops and FFT lengths.** The 58 Hz band's coefficients are
  150.0 ms apart, the 6.6 kHz band's 1.30 ms. Rounding each `M_k` up to a
  fast FFT length adds 0.28% to the count and leaves the error at 6.4e-16.
  Giving every bandpass filter the same hop (one matrix, like a
  spectrogram) needs 4.29 N.
- **C6. A decimated band is a sampled band.** For the 587 Hz band with
  `M_k` chosen so the hop is a whole 630 samples, the coefficients equal
  the undecimated positive-frequency band sampled every 630 samples, times
  `sqrt(N / M_k)`, to 5.1e-16.
- **C7. Filters that never reach zero are cut at a level, and the cut bank
  is exact.** The positive half of a gammatone response is nonzero at
  every frequency, so its support is the whole half-spectrum. Setting the
  response to zero where it is more than a given level below its peak
  gives these counts, with a reconstruction error of at most 1.1e-15
  every time:

  | Bank | cut at −40 dB | −60 dB | −100 dB | undecimated |
  |---|---|---|---|---|
  | gammatone, 60 bands, 70 Hz–7 kHz | 5.26 N | 8.54 N | 18.3 N | 62 N |
  | Morlet, 54 bands, 70 Hz–7 kHz, 6 cycles | 5.06 N | 5.88 N | 7.17 N | 56 N |

  The cut does not change a cosine bank: at −100 dB no response bin of
  the semitone bank changes.
- **C8. Edited coefficients go back as the least-squares signal.** With
  half the coefficients set to zero, analysis after synthesis is a
  projection: applying it twice differs from once by 6.7e-16.
- **C9. It is not slower.** Analysis plus synthesis of the 3 s takes
  0.11 s with the checker's plain loop over bands, against 0.52 s for
  today's undecimated `Filterbank` with `pad=0` (one run on the cloud
  container; timings vary by machine and run).

## The proposed design

### A frame that holds a filterbank

```python
bank = so.cosine_filterbank(f_lo=55, f_hi=7040, scale="cents", spacing=100)
cq = so.DecimatedFilterbank(bank)      # a Frame
coefs = cq.analyze(snd)                # DecimatedSubbands
coefs.to_sound()                       # exact
coefs.plot()                           # each band drawn with its own time cells
```

`DecimatedFilterbank` is a new `Frame` in `frames/filterbank.py`, next to
the bank it decimates, as `TVGaborFrame` sits next to `GaborFrame` in
`frames/gabor.py`.
It holds a `Filterbank` and takes the filters from it (`rfft_response`),
so the responses, the scales and the measured tightness have one copy.
`Filterbank` itself does not change: its `analyze` still returns real
`Subbands`, and texture synthesis, which uses it, stays bit-identical
without needing a check.

Its coefficients, `DecimatedSubbands`, are complex, one array per filter,
each at its own rate. They keep the lowpass, the positive half of every
bandpass filter and the highpass (C3). Their energy counts each bandpass
coefficient twice, for its mirror, as `GaborFrame` counts its stored bins,
so that it equals the energy of the whole two-sided transform and the
least-squares meaning of synthesis (C8) holds in that norm. A bandpass filter's value at 0 Hz and at
Nyquist belongs to both of its halves, so each half carries it divided by
`sqrt(2)`; the two halves together then have the bank's response there,
and `s` is exactly the bank's.

Philosophy.md says subbands stay real and that complex-coefficient
synthesis waits for a use. This is that use: a band cannot be kept at a
rate set by its bandwidth unless it is complex.

## Decisions

- **D1. A new frame, `DecimatedFilterbank(bank)`, holding a
  `Filterbank`, with coefficients `DecimatedSubbands`, in
  `frames/filterbank.py`** (Cho, 2026-10-28: no separate file; `gabor.py`
  already keeps the fixed and the time-varying Gabor frames together). The alternatives were a `hop` field
  on `Filterbank` (then one class's `analyze` would return two types, real
  or complex depending on a field), the paper's name
  `NonstationaryGaborFrame` (accurate, but `TVGaborFrame` is a
  nonstationary Gabor frame too, so the name would not say which), and
  `ConstantQTransform` (too narrow: the same class decimates an ERB or a
  gammatone bank). No new factory: the constant-Q transform is the
  semitone cosine bank decimated, as chroma's D5 made the undecimated one
  a cosine bank rather than a new function. Accepted.
- **D2. Complex coefficients, the positive half only, scaled by
  `sqrt(M_k / N)`.** A tight bank then gives a Parseval frame with the
  same bound (C2), and a real signal is held in about 2 N real numbers
  (C3, C4). Accepted.
- **D3. Each band keeps its support, rounded up to a fast FFT length.**
  It costs 0.28% more numbers (C5) and makes every inverse FFT fast. Hops
  are then not whole samples in general; each band reports its own times,
  `m * N / M_k / fs`. One hop for every band would give a matrix but
  twice the numbers (4.29 N), and that matrix is close to what
  `Subbands` already offers. Whole-sample hops for every band (each `M_k`
  a divisor of the padded length) are possible but not measured here.
  Accepted: fast lengths.
- **D4. Padding as `Filterbank` does it.** `pad="auto"` pads by at least
  the filters' ringing, the padding travels with the coefficients and is
  removed on the way back, and `pad=0` makes it circular. Not measured
  here; it is the same padding as today's. Accepted.
- **D5. Filters that never reach zero are cut at `floor_db`, default
  −100 dB.** Analysis and synthesis both use the cut filters, so the frame
  is exact for them (C7); a cut filter differs from the true one by less
  than that level below its peak. At −100 dB a gammatone bank keeps 18.3 N
  numbers instead of 62 N and a Morlet bank 7.17 N instead of 56 N, and a
  cosine bank is not changed at all. The cut also makes a causal gammatone
  very slightly noncausal, at about that level (an estimate, not
  measured). Accepted.
- **D6. First version: the `Frame` contract and little else.**
  `analyze`, `synthesize`, `adjoint`, `energy`, measured `frame_bounds`
  and `is_tight`; `DecimatedSubbands.to_sound`, `cfs`, the times of each
  band, `plot`, and multiplication by one gain per band, as `Subbands`
  has. Masks on this ragged grid, and envelopes, come later, each when a
  page or an experiment needs it. Accepted.
- **D7. Out of scope.** Writing `GaborFrame` and `TVGaborFrame` as special
  cases of the same code (one copy of the frame algebra, but no use needs
  it yet); windows that vary over time and frequency at once (not painless
  in general, so no exact inverse to promise); the sliced, real-time
  constant-Q transform of Holighaus et al. (2013); rebuilding chroma or
  the texture model on it. Accepted.
- **D8. Tests.** Exact reconstruction and measured tightness for cosine
  banks on every scale, with uniform and random centers, and for gammatone
  and Morlet banks; C6 as an oracle (a decimated band is the undecimated
  band sampled); the real-signal storage (C3); and the extremes: one band,
  a sound shorter than the lowest band's hop, a band whose support is a
  single bin. Accepted.
- **D9. Gallery: a section on "Analysis and resynthesis", with links
  from two pages.** See below. Accepted.

## Where it goes in the gallery

The decimated transform is a filterbank that reconstructs exactly, with
fewer numbers. That is the topic of
[Analysis and resynthesis](https://choyun1.github.io/sonore/gallery/resynthesis.html),
which would own it, in one new section:

- **"Keeping fewer numbers"**, after "Nothing is lost" and before
  "Masking the spectrogram". The page's sentence analyzed by the semitone
  bank both ways, with the counts printed as the build measures them (85
  numbers per sample against about 2); the decimated coefficients drawn
  cell by cell, long cells low and short cells high; the reconstruction
  error printed; and one edit to listen to, such as keeping only the
  largest tenth of the coefficients, so the page shows that a decimated
  representation can be changed and played like the others.

Two pages would link to it:

- [Seeing speech](https://choyun1.github.io/sonore/gallery/speech.html),
  "The same plane, tiled four ways". The Morlet panel draws boxes of a
  decimated wavelet tiling, but sonore's Morlet bank is undecimated, so no
  sonore object has those cells today. With this transform the panel could
  draw the cells of the decimated semitone transform from its supports and
  hops, so the figure shows an actual tiling. A sentence and a link may be
  enough; redrawing the panel is an option for Cho.
- [Pitch class and height](https://choyun1.github.io/sonore/gallery/chroma.html),
  "A constant-Q filter bank". The section states the undecimated cost; one
  sentence would link to the decimated version.

The README's module table gains the two classes, and the roadmap line
moves to `docs/roadmap-done.md` when the work is done.

## Citations

| Source | Status on 2026-10-28 |
|---|---|
| Balazs, Dörfler, Jaillet, Holighaus & Velasco (2011), "Theory, implementation and applications of nonstationary Gabor frames", J. Comput. Appl. Math. 236, 1481–1496. doi:10.1016/j.cam.2011.09.011 | Checked against the PDF in `frames/filterbanks.md` |
| Velasco, Holighaus, Dörfler & Grill (2011), "Constructing an invertible constant-Q transform with nonstationary Gabor frames", Proc. DAFx-11, 93–99 | As listed in `views/chroma.md`; not checked again here |
| Holighaus, Dörfler, Velasco & Grill (2013), "A framework for invertible, real-time constant-Q transforms", IEEE Trans. Audio, Speech, and Language Processing 21(4), 775–785 | As listed in `views/chroma.md`; not checked again here |
