# Frames, step 2: new filter shapes, time-varying Gabor, adjoints

Status: accepted, 2026-09-30. D5, D6, D7 and D11 were agreed with Cho in
discussion; D7's transition width, D8, D9 and D10 were accepted as
recommended on 2026-09-30. Implemented in patches 2-4 (see the patch plan).

Step 1 (docs/design/frames.md) fixed what `analyze`, `synthesize` and
`frame_bounds` mean and retrofitted the cosine banks and the STFT. Step 2
adds instances of that contract:

1. **Two new filterbank shapes:** gammatone and Morlet, as `Filterbank`
   subclasses. Neither is tight, so both use the canonical dual (D1).
2. **Gabor frames with time-varying windows,** in the painless case.
3. **`Frame.adjoint`,** the analysis operator's adjoint for every frame.

Unions of frames, originally planned for step 2, are deferred (D11). Step 3
is unchanged: the "seeing speech" gallery section.

The promise that governs every choice below: **analysis followed by
synthesis is exact,** for every frame sonore ships, on the whole signal
space, not just a band of it.

## How the claims are verified

As in step 1, claims are numbered (continuing at C8) and tagged [proof],
[check] or [source]. [check] numbers come from
`tools/check_frames_step2_claims.py`, which uses only NumPy and SciPy,
writes every filter response from its formula, and builds operators as dense
matrices on small signals. It shares no code with sonore and runs in about
3 s. The numbers below are from SciPy 1.17.1.

## Claims

**C8. The analytic subband is the Hilbert transform of the real one.**
[proof, check] For a real signal x and a conjugate-symmetric response H (real
impulse response), `hilbert` of the real subband ifft(H·X) equals filtering
with the analytic filter H(f)(1 + sgn f), where DC and Nyquist get weight 1
(SciPy's convention).

- Proof: `hilbert` multiplies the FFT of its (real) input by exactly that
  step, and multiplication by H commutes with it.
- Check: largest difference relative to the largest coefficient, for a
  zero-phase and a causal gammatone at 1 kHz, fs = 16 kHz: 1.5e-15 at
  N = 4001 and 8.2e-16 at N = 4000. The even case needs H(fs/2) real (C14);
  without that, the difference was 1.6e-10.
- Consequence: storing complex analytic subbands adds no information.
  `Subbands._analytic()` already returns them exactly, on the padded
  circular grid (D5).

**C9. The analytic frame on real signals.** [proof, check] The analytic
analysis T_a x = ifft(H_k (1 + sgn f) X), restricted to real x with the real
inner product, has frame operator Re(T_a^H T_a) with eigenvalues 2·s(f) on
bins strictly between DC and Nyquist and s(f) at DC and Nyquist, where
s = Σ_k |H_k|².

- Check: at N = 64 with three Morlet filters, the eigenvalues match to
  1.1e-15 relative.
- This is what complex-coefficient synthesis would divide by, if it is ever
  added (D5). It is the same weight pattern as C3.

**C10. Gammatone formulas.** [proof, check] The 4th-order gammatone impulse
response t³ e^(−2πbt) cos(2πf_c t), t ≥ 0, with b = 1.019·ERB(f_c), has the
closed-form Fourier transform

H(f) = (3!/(2π)⁴)/2 · [(b + i(f − f_c))⁻⁴ + (b + i(f + f_c))⁻⁴],

which is conjugate-symmetric. sonore normalizes it to unit gain at f_c.

- (a) Check: against the FFT of the impulse response sampled at 64 kHz,
  below 8 kHz: 1.2e-10 relative.
- (b) The group delay at f_c is n/(2πb) for order n, up to the small
  contribution of the mirror term. Check (n = 4): 17.59 ms against the
  formula's 17.60 ms at 100 Hz, and 1.369 ms against 1.369 ms at 4 kHz.
- (c) The envelope t³e^(−2πbt) peaks at (n − 1)/(2πb): 13.2 ms at 100 Hz and
  1.03 ms at 4 kHz. This is the latency the textbook impulse-response figure
  shows, and it is distinct from the group delay in (b).

**C11. Edge coverage.** [check] Bare gammatone and Morlet banks are badly
conditioned or not frames at all, because s(f) falls off below the lowest
filter and above the highest.

At fs = 16 kHz, N = 16000:

| Bank | Bare A/B | Exact-fill edges: A/B, ringing | Raised-cosine edges, 1 spacing: A/B, ringing | 2 spacings: A/B, ringing | Bank's own ringing |
|---|---|---|---|---|---|
| Gammatone, 1 per ERB, 50–7000 Hz | 0.0068 | 0.75, 500 ms | 0.53, 107 ms | 0.18, 66 ms | 44 ms |
| Morlet, 6 cycles, 4 per octave, 50–7000 Hz | 0 | 0.74, 419 ms | 0.58, 234 ms | 0.30, 149 ms | 71 ms |

- The Morlet bank's A is exactly 0, because the DC correction makes every
  Morlet response vanish at f = 0. Without edge filters it is not a frame.
- *Exact fill* sets |L|² = max(0, s_floor − s_bank(f)) outside
  [cf_lo, cf_hi], where s_floor is the minimum of s_bank between the
  lowest and highest center frequencies. A = s_floor exactly and B is
  unchanged, but the kinked responses ring for 400–500 ms.
- *Raised cosine* gives the edge filters magnitude √s_floor with a
  raised-cosine transition k filter spacings wide (measured on the bank's own
  scale). The lowpass transition ends at cf_lo; the highpass transition
  starts at cf_hi. Both are zero-phase. They ring far less, at some cost in
  A/B.
- "Ringing" is the same −60 dB measure as `Filterbank.ringing`. It sets the
  `pad="auto"` length, so it costs memory and time, not accuracy.

**C12. Time-varying Gabor, painless case.** [proof, check] Windows w_q of
lengths L_q at positions a_q, each zero-padded to one FFT length M with
L_q ≤ M for every q, give a diagonal frame operator
s(t) = M Σ_q |w_q(t − a_q)|² (full complex FFT; C3's weights handle the
one-sided case exactly as in step 1).

- Proof: each window's DFT rows are orthogonal with norm² M on its support,
  so each frame contributes M·diag(|w_q|²). This is the painless condition
  of Daubechies, Grossmann & Meyer (1986), generalized to one window per
  position as in Balazs et al. (2011).
- Check: at N = 64 with lengths 8–16 and M = 16, the diagonal matches the
  formula to 5.1e-16 and the off-diagonal is 5.1e-17, both relative to
  max s.
- Negative check: once one window is longer than M (20 > 16), the
  off-diagonal part is 0.56% of the largest entry. The operator is no longer
  diagonal, which is why D9 requires M ≥ the longest window.

**C13. The STFT adjoint from SciPy.** [check] For the C3-weighted inner
product, T* = M · `istft` of a `ShortTimeFFT` built with `dual_win = win`,
where M = mfft.

- Check: ⟨Tx, c⟩_w / ⟨x, istft(c)⟩ = 32, 48 and 33 for
  (win, hop, mfft) = (32, 10, 32), (32, 8, 48) and (33, 10, 33). That is
  mfft each time, including zero-padded and odd FFTs, with Hann^1.5 at a
  non-tight hop.

**C14. The Nyquist bin of complex responses.** [proof, check] On an even
grid, `irfft` keeps only the real part of the Nyquist bin, so the filter a
complex response actually applies there is Re H(fs/2). A dual that divides by
|H(fs/2)|² is then not exact. Using Re H(fs/2) consistently, in analysis, in
s and in the dual, restores exactness.

- Check: for causal gammatones up to 7.5 kHz at fs = 16 kHz, |Im H(fs/2)| is
  up to 98% of |H(fs/2)|. The round-trip error is 3.9e-5 naive, and 4.2e-16
  with the rule.
- The cosine banks are unaffected, because their responses are already real.
- This was found by the checker while drafting. It would have been a silent
  bug at the 1e-5 level, larger for banks that reach closer to Nyquist.

## Decisions

**D5. Subbands stay real.** (Agreed.) Morlet and gammatone produce real
`Subbands` like the cosine banks. By C8 this loses nothing: envelopes, `tfs`
and phase all come from `_analytic()` exactly as now. `Envelopes`, `tfs`,
`Envelopes * Subbands` and the texture code are untouched.

- Complex-coefficient synthesis (synthesizing from edited analytic
  coefficients) is deferred until an experiment needs it. It would be one
  method using the C9 weights, proved and checked like C3, not a new type.

**D6. Gammatone phase is causal by default.** (Agreed.) The gammatone *is*
its causal impulse response (C10). The bank takes `phase="causal"` (the
default) or `phase="zero"`. The zero-phase variant keeps |H| and aligns
onsets across bands.

- Exactness holds either way: the dual filters with conj(H)/s, and conj(H)
  is the time-reversed filter, so the delay is undone.
- Padding stays symmetric. Causal analysis rings forward, but the dual rings
  backward, so both ends need room. `ringing` already measures a causal
  response correctly, since it reads the first half of the impulse response.
- A display-time delay compensation for plots, as the Auditory Image Model
  offers, is a step 3 plotting option, not a filter property.

**D7. Edge filters by default.** (Agreed: option (a).) Every non-tight bank
sonore ships adds a zero-phase lowpass and highpass so that it is a
well-conditioned frame on the whole band, as the cosine banks do with their
two extra filters. `n_filters = n_bands + 2`, and `cfs` gets the edge
filters' nominal centers at DC and Nyquist, matching the cosine banks.

- The edge design (accepted as recommended, 2026-09-30): **raised cosine,
  one filter spacing wide** (C11): A/B ≈ 0.5–0.6, with ringing roughly 2–3 times
  the bank's own. Exact fill gives better A/B but pads by about half a
  second. Two spacings halves the ringing again but gives A/B of 0.2–0.3.
  The transition width can be a constructor argument with that default.
- s_floor is computed once per bank from its response on a fine frequency
  grid between the lowest and highest center frequencies, so the edge
  filters depend only on the bank, not on fs or N.
- `edges=False` gives the bare bank for cochleagram use. By D4, `analyze`
  still works, and `synthesize` raises if the bounds say it is not a frame.

**D8. The Nyquist rule.** (Accepted as recommended, 2026-09-30; a
correctness fix.)
`Filterbank.rfft_response` takes the real part of the Nyquist bin on even
grids, so analysis, `frame_power` and `synthesize` all see the filter that is
actually applied (C14). This is a base-class change. It is a no-op for real
responses, so the cosine banks and texture synthesis stay bit-for-bit
unchanged; this will be checked against a worktree of the previous commit.

**D9. Time-varying Gabor.** (Accepted as recommended, 2026-09-30.)

- **Specification:** an explicit schedule, `TVGaborFrame(times, win_durs,
  n_fft=None, window="hann")`, in seconds like `GaborFrame`, rounded to
  samples per fs. `n_fft` defaults to the longest window and may not be
  shorter, which is the painless condition (C12). A
  `TVGaborFrame.from_function(win_dur_of_t, overlap=4, t_end=...)` helper
  steps through time with hop = window/overlap.
- **Coefficients:** a new coefficient type rather than `STFT`. `STFT`'s
  `freqs`, `times`, `repr` and `griffin_lim` all go through SciPy's
  `ShortTimeFFT`, which cannot represent varying windows, and retrofitting
  that risks the step 1 invariants. The new type has data shape
  `(ch, F, frames)` like `STFT` and non-uniform `times`.
- **Implementation:** sonore's own, since SciPy cannot do this. The dual is
  conj-window over s(t) (C12, generalizing C5).
- **Cross-check:** a constant schedule must reproduce `GaborFrame`.
- **Deferred to step 3:** pitch-adaptive schedules (turning an F0 track
  into a schedule), and a TANDEM-STRAIGHT-style pair of schedules offset by
  half a period. Both use this class; neither needs new frame theory.

**D10. `Frame.adjoint(coefs) -> Sound`.** (Accepted as recommended,
2026-09-30.) The adjoint for the
coefficient inner product that `energy` uses (D2):

- filterbanks: filter with conj(H), with no division by s;
- `GaborFrame`: mfft · `istft` with `dual_win = win` (C13), from a second
  cached `ShortTimeFFT`;
- `TVGaborFrame`: conj-window overlap-add without division by s(t).

It is checked against the transpose of the dense oracle. It is needed for
any later union synthesis (D11), and it is the analysis operator's gradient,
which optimization-based synthesis such as `texture_grad` can use.

**D11. Unions deferred.** (Agreed.) Union synthesis, the adjoints summed and
divided once by the summed s, matters only for editing several
representations jointly and resynthesizing, and step 3 does not do that. D10
makes it nearly free later. STRAIGHT and WORLD combine frames only as power
spectra after phase is discarded, and resynthesize parametrically, so they
don't need it either. A shared time-frequency grid for plotting several
magnitudes side by side belongs with the step 3 plotting work.

## API sketch

```python
fb = so.GammatoneFilterbank(n_bands=40, f_lo=50, f_hi=7000)  # 1.019 ERB, order 4
fb = so.GammatoneFilterbank(..., phase="zero", edges=False)
fb = so.MorletFilterbank(n_bands=28, f_lo=50, f_hi=7000, cycles=6)  # log spacing

sb = fb.analyze(snd)        # Subbands, real (D5)
sb.envelopes()              # unchanged
sb.synthesize()             # canonical dual, exact (D1, D7, D8)
fb.frame_bounds(len(snd), snd.fs)

tv = so.TVGaborFrame(times=[...], win_durs=[...])
c = tv.analyze(snd)         # new coefficient type, (ch, F, frames)
tv.synthesize(c)            # exact

fb.adjoint(sb)              # D10, every frame
```

Scale attributes follow the cosine banks, so `modulation_spectrum` works:
gammatone has `spacing` in ERB-number with `unit = "ERB"`, and Morlet has
`spacing` in octaves with `unit = "oct"`.

## Changes to existing code

- `Filterbank.rfft_response`: the Nyquist rule (D8). A no-op for real
  responses.
- `Frame`: a new abstract `adjoint`, implemented for `Filterbank` (one
  method covers all banks), `GaborFrame` and `TVGaborFrame`.
- Edge filters (D7): a shared mechanism for non-tight banks, not a change
  to the cosine banks, which keep their own tight edge filters.
- Exports: `so.GammatoneFilterbank`, `so.MorletFilterbank`,
  `so.TVGaborFrame`.

## Tests (target: under 2 s added)

- Every new frame goes through the existing dense-oracle tests: the oracle
  reproduces the fast analysis, bounds are the extreme eigenvalues, and
  synthesis equals `canonical_lstsq`, mono and stereo.
- Gammatone: response against C10's closed form; group delay and envelope
  peak within 1% of the formulas; `phase="zero"` equals |H|.
- Edges: A ≥ s_floor minus a small tolerance; `edges=False` Morlet raises on
  `synthesize` (D4).
- Nyquist: an even-length round trip through a bank reaching near Nyquist is
  exact to 1e-12.
- Time-varying Gabor: a constant schedule matches `GaborFrame`; a window
  longer than `n_fft` raises at construction.
- `adjoint`: equals the transpose of the dense oracle, with the C3 weights,
  for every frame.
- Invariants: the cosine banks and texture synthesis are bit-for-bit
  unchanged against a worktree of `0ba9597`.

## Patch plan (stacked on 0ba9597, apply in order)

1. This document and `tools/check_frames_step2_claims.py`.
2. `Frame.adjoint` for the existing frames, and the Nyquist rule.
3. `GammatoneFilterbank`, `MorletFilterbank`, and the shared edge filters.
4. `TVGaborFrame` and its coefficient type.

Each patch carries its own tests. README entries come with patches 3 and 4.

## Implementation notes

Details settled while implementing, recorded so the code and this document
agree.

- **Patch 2.** `Filterbank.rfft_response` applies the Nyquist rule only to
  complex responses, returning real ones untouched, so the cosine banks are
  bit-for-bit unchanged. The tests exercise the rule with a test-only
  Gaussian bank with a fractional delay, which needs no step 2 bank.
- **Patch 3, the shared mechanism.** `BandpassFilterbank` holds the edge
  filters; subclasses give the scale and `band_response`. The `n_bands`
  centers run from `f_lo` to `f_hi` inclusive. `s_floor` is the minimum on a
  grid of 64 points per spacing between the outer centers. The raised-cosine
  transition is linear in Hz between its endpoints, which are set on the
  bank's scale, exactly as in the checker. `edge_width` (default 1) is the
  width in spacings.
- **Edge centers.** D7 said the edge filters' nominal centers would be at
  DC and Nyquist, but a bank does not know fs. `cfs` instead puts them where
  the transitions start, `edge_width` spacings outside the band (clamped at
  0 Hz). That is one spacing beyond the outer bandpass, the same place the
  cosine banks put theirs.
- **C11 reproduced.** The tests rebuild the two C11 banks and find A/B 0.530
  and 0.585, with edges ringing 107 and 234 ms. The gammatone figure differs
  from the table's 0.532 because sonore's ERB-number scale,
  9.265 ln(1 + f/228.8), differs from the checker's 21.4 log10(1 + 0.00437 f)
  by 0.3%. Two further differences from the table are expected, not errors.
  First, `Filterbank.ringing` gives 73 ms for the causal gammatone bank,
  against the table's 44 ms for its zero-phase magnitude. Second, the bare
  causal bank's A/B on an even grid is lower than the table's, because
  under the Nyquist rule the bank applies Re H(fs/2), which is smaller than
  |H(fs/2)|.
- **Edge test.** The test plan's "A ≥ s_floor" holds only for exact fill.
  With raised-cosine edges, A falls below s_floor in the transitions, by the
  amount C11 reports. The tests check C11's A/B and ringing instead.
- **`Envelopes.without_edges`** returns the envelopes unchanged for a bank
  built with `edges=False`, which has no edge bands to zero.

- **Patch 4, `TVGaborFrame`.** `times` are window centers, rounded to
  samples, and must be strictly increasing. Each windowed segment is placed
  in the FFT buffer with the window's middle sample at index 0, which is
  SciPy's phase convention, so a constant schedule over SciPy's frames
  reproduces `GaborFrame`'s coefficients exactly (equal to 0.0 in the
  checks). `n_fft` is in samples, like `GaborFrame`'s, while window lengths
  depend on fs. A window longer than `n_fft` is therefore refused when the
  frame is first used at a rate, not at construction as the test plan says.
  The coefficient type is `TVSTFT` (`data`, `f`, `t`, `magnitude`, `db`,
  masking by multiplication, `to_sound`). `from_function(win_dur_of_t,
  t_end, overlap=4, t_start=0)` steps from `t_start` while the center is at
  most `t_end`.

## Out of scope for step 2

- Union synthesis (D11) and the CG canonical dual (D1(b)).
- Complex-coefficient synthesis (D5).
- Pitch-adaptive schedules, STRAIGHT/WORLD-style envelopes, reassignment,
  and the "seeing speech" gallery (step 3).
- Wrapping `nsgt`.

## References

- Balazs et al. (2011) and Daubechies, Grossmann & Meyer (1986): see
  docs/design/frames.md.
- Glasberg, B. R. & Moore, B. C. J. (1990). Derivation of auditory filter
  shapes from notched-noise data. *Hearing Research* 47. The ERB formula,
  already cited in the README.
- Patterson, R. D., Robinson, K., Holdsworth, J., McKeown, D., Zhang, C. &
  Allerhand, M. (1992). Complex sounds and auditory images. In *Auditory
  Physiology and Perception* (Proc. 9th International Symposium on
  Hearing), Y. Cazals et al., eds. Citation verified by lookup: the paper
  defines the gammatone auditory filterbank used in the Auditory Image Model.
  Page numbers not checked.
- Slaney, M. (1993). An efficient implementation of the Patterson–Holdsworth
  auditory filter bank. Apple Computer Technical Report 35. Existence
  verified by lookup. The b = 1.019·ERB convention is commonly attributed to
  this report and to the Patterson et al. APU reports, but that attribution
  has not been checked against the text.
- Morise, M., Yokomori, F. & Ozawa, K. (2016). WORLD: a vocoder-based
  high-quality speech synthesis system for real-time applications. *IEICE
  Trans. Inf. & Syst.* E99-D(7), 1877–1884. Verified by lookup. It reports
  better quality and speed than the systems it compares against, in the
  authors' own evaluation.
- Kawahara, H., Masuda-Katsuse, I. & de Cheveigné, A. (1999). Restructuring
  speech representations using a pitch-adaptive time-frequency smoothing and
  an instantaneous-frequency-based F0 extraction. *Speech Communication* 27,
  187–207. Verified by lookup. STRAIGHT; the complementary-window power
  averaging is described in the corresponding patent (US 6,115,684).

## Reference implementations

- **WORLD** (github.com/mmorise/World; modified BSD, and the README states
  its algorithms are unpatented). The reference for pitch-adaptive analysis
  in step 3. Its permissive license means it is not limited to dev-time
  checks. Its Python wrapper `pyworld` has not had its license checked.
  Status: not yet checked against sonore.
- **LTFAT** and **nsgt**: as in step 1, for dev-time cross-checks only,
  because of their licenses. LTFAT's `filterbank` and nonstationary Gabor
  code are the natural cross-checks for patches 3 and 4. Status: not yet
  checked.
- **SciPy `ShortTimeFFT`:** C13's adjoint was checked against version 1.17.1.
- **Gammatone implementations** (Slaney's Auditory Toolbox, the Ma
  gammatone code, MATLAB's `gammatoneFilterBank`) are time-domain IIR
  approximations. sonore uses the exact frequency response (C10). Status:
  consulted for conventions only; no numeric cross-check planned, since they
  approximate the same function.
